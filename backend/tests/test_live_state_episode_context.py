"""Offline production owner -> Live State -> voice/brief guard integration.

Canonical episode APIs and binders execute unchanged. A scoped SQL fixture and
payload persistence stub replace PostgreSQL only; no provider or device is used.
"""
import asyncio
from datetime import timedelta
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock
from uuid import uuid4

from app.knowledge.models import SourceError
from app.live_state.canonical import context_dependency
from app.live_state.episode_context import EPISODE_NAMESPACES, service_episode_owner
from app.live_state.models import SessionScope
from app.live_state.perception_context import compose_perception_brief
from app.live_state.perception_runtime import CanonicalPerceptionBinder
from app.live_state.projection import select_voice_context
from app.live_state.runtime import create_session_live_state
from app.reminders.duties import ResponsibilityRepository
from app.reminders.episode_checks import EpisodeCheckOwner
from app.reminders.episode_models import AdmitEpisodeCheck, ChangeEpisodeCheck, EpisodeCheckDraft, EpisodeCheckSpec
from app.runtime.events import DomainEventBus
from app.runtime.feature_config import MemoryAuthority
from app.tasks.context_events import ContextRecord
from tests.test_episode_check_owner import Store, NOW


class IntegrationStore(Store):
    def __init__(self):
        super().__init__()
        self.copies, self.edges = {}, []
        self.sessions = set()
        self.materialized = 0

    def event(self, owner, session, **kwargs):
        source = super().event(owner, session, **kwargs)
        self.sessions.add((owner, session))
        return source

    def record(self, identity):
        event = self.events[identity]
        return ContextRecord(**{key:value for key,value in event.items() if key != "current"},
            sequence=list(self.events).index(identity) + 1, origin_sequence=list(self.events).index(identity) + 1,
            source_namespace="episode-offline", source_event_id=str(identity), supersedes_event_id=None,
            content_digest="a" * 64, is_current=event["current"])

    def available(self, owner, identity, seen=None):
        seen = set() if seen is None else seen
        if identity in seen:
            return False
        seen = seen | {identity}
        for event in self.events.values():
            if event["payload_id"] == identity:
                return event["owner_id"] == owner and event["availability"] == "available" and event["content"] is not None
        copy = self.copies.get(identity)
        if copy is None or copy["owner"] != owner or copy["content"] is None:
            return False
        for edge in self.edges:
            if edge[1] == identity:
                event = self.events[edge[2]]
                if not event["current"] or event["revision"] != edge[3]:
                    return False
        return all(self.available(owner, parent, seen) for parent in copy["parents"])

    async def payload(self, connection, owner, content, *, source_payload_ids=()):
        assert connection is self and self.active
        assert all(self.available(owner, parent) for parent in source_payload_ids)
        identity = uuid4()
        self.copies[identity] = dict(owner=owner, content=content, parents=set(source_payload_ids))
        self.materialized += 1
        return identity

    async def fetchrow(self, query, *args):
        q = " ".join(query.lower().split())
        if q.startswith("select responsibility_id,episode_id from jarvis_responsibility_occurrences"):
            self.sql(query)
            row = self.occurrences.get(args[1])
            return dict(row) if row and row["owner_id"] == args[0] else None
        return await super().fetchrow(query, *args)

    async def fetchval(self, query, *args):
        q = " ".join(query.lower().split())
        if q.startswith("select lock_jarvis_copy_sources"):
            self.sql(query)
            return bool(args[1]) and all(self.available(args[0], root) for root in args[1])
        if q.startswith("select true from jarvis_sessions"):
            self.sql(query)
            return (args[0], args[1]) in self.sessions
        if q.startswith("select content from jarvis_context_payloads"):
            self.sql(query)
            row = self.copies.get(args[1])
            return row["content"] if row and row["owner"] == args[0] else None
        return await super().fetchval(query, *args)

    async def executemany(self, query, args):
        q = self.sql(query)
        if "insert into jarvis_context_event_dependencies" in q:
            for row in args:
                assert self.events[row[2]]["owner_id"] == row[0]
                assert self.copies[row[1]]["owner"] == row[0]
                self.edges.append(tuple(row))
            return
        if "insert into jarvis_context_payload_dependencies" in q:
            for owner, parent, child in args:
                assert self.available(owner, parent)
                assert self.copies[child]["owner"] == owner
                self.copies[child]["parents"].add(parent)
            return
        raise AssertionError(q)

    def bind_io(self, canonical):
        canonical.tasks.payload = AsyncMock(side_effect=self.payload)

        async def records(connection, owner, identities):
            assert connection is self and self.active
            return {identity:self.record(identity) for identity in identities
                    if identity in self.events and self.events[identity]["owner_id"] == owner}

        canonical.context._records = AsyncMock(side_effect=records)
        canonical.context._record = AsyncMock(side_effect=lambda c, owner, identity:self.record(identity))


class EpisodeContextIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.db = IntegrationStore()
        self.now = NOW + timedelta(seconds=1)
        self.original_scope = SessionScope(owner_id="episode-context", session_id=uuid4())
        self.scope = SessionScope(owner_id=self.original_scope.owner_id, session_id=uuid4())
        self.db.sessions.add((self.scope.owner_id, self.scope.session_id))
        self.repository = ResponsibilityRepository(self.db)
        self.owner = EpisodeCheckOwner(self.repository, enabled=True, clock=lambda:self.now)
        await self.owner.initialize()
        self.services = SimpleNamespace(memory_authority=MemoryAuthority.LEGACY, _closed=False,
            background_task_context_service=SimpleNamespace(repository=SimpleNamespace(pool=self.db,
                snapshot=AsyncMock(return_value={"recent":[], "pinned":[]}))),
            background_task_service=None, db_pool=self.db, domain_event_bus=DomainEventBus(),
            settings=SimpleNamespace(RESPONSIVE_VOICE_ENABLED=True), responsibility_issue=None,
            attention_service=SimpleNamespace(_closed=False, responsibilities=SimpleNamespace(episode_checks=self.owner)))
        self.lives = []
        self.admitted, self.source = await self.admit()

    async def admit(self, *, cues=()):
        source = self.db.event(self.scope.owner_id, self.original_scope.session_id, at=self.now)
        result = await self.owner.admit(self.original_scope, source, AdmitEpisodeCheck(
            spec=EpisodeCheckSpec(title="Explicit episode"), check=EpisodeCheckDraft(question="What does the control indicate?",
                activation_cues=cues, observation_window_expires_at=NOW + timedelta(minutes=10))))
        return result, source

    def live(self, *, perception=False):
        live = create_session_live_state(self.services, scope=self.scope, enabled=True, clock=lambda:self.now,
            interval_seconds=0.01, timeout_seconds=0.5, perception_brief_enabled=perception)
        self.db.bind_io(live.binder)
        # Focus qualification is separate; keep its unrelated SQL out of this fixture.
        live.perception_runtime.focus_inputs.read = AsyncMock(return_value=None)
        self.lives.append(live)
        return live

    async def asyncTearDown(self):
        for live in self.lives:
            await live.close()

    async def until(self, predicate, *, detail=lambda: "condition"):
        try:
            async with asyncio.timeout(2):
                while not predicate():
                    await asyncio.sleep(0.002)
        except TimeoutError:
            self.fail("Timed out: " + str(detail()))

    async def ready(self, *, perception=False):
        live = self.live(perception=perception)
        await live.start()
        await self.until(lambda: live.service.snapshot().readiness == "ready"
            and any(entry.kind == "situational_check" for entry in live.service.snapshot().entries),
            detail=lambda: (live.episode_context.worker.last_error, live.episode_context.omissions))
        return live

    def command(self, result, operation, **kwargs):
        check = result.check
        return ChangeEpisodeCheck(responsibility_id=check.canonical_owner_ref.source_id,
            expected_revision=check.canonical_owner_ref.revision, occurrence_id=check.id,
            expected_check_revision=check.revision, episode_id=check.episode_or_object_ref.source_id,
            expected_episode_revision=check.episode_or_object_ref.revision, operation=operation, **kwargs)

    async def change(self, operation, **kwargs):
        source = self.db.event(self.scope.owner_id, self.scope.session_id, at=self.now)
        return await self.owner.change(self.scope, source, self.command(self.admitted, operation, **kwargs))

    async def voice_binding(self, live):
        selection = select_voice_context(live.service.snapshot(), now=self.now)
        self.assertIn("Unresolved episode check", selection.text)
        return await live.bind(selection)

    def brief(self, live):
        snapshot = live.service.snapshot()
        checks = live.episode_context.checks(snapshot)
        dependencies = {ref for check in checks for ref in check.dependency_versions}
        return compose_perception_brief(snapshot, now=self.now, scope_epoch=snapshot.cursor.epoch,
            checks=checks, dependency_eligible=lambda source:source in dependencies).brief

    async def test_factory_uses_actual_owner_handle_without_io(self):
        calls = self.db.acquisitions
        live = self.live()
        self.assertIs(live.episode_context.owner, self.owner)
        self.assertEqual(self.db.acquisitions, calls)
        self.assertIs(service_episode_owner(self.services), self.owner)

    async def test_flag_or_uninitialized_owner_does_not_report_available(self):
        for owner in (None, self.repository.episode_checks(enabled=True), SimpleNamespace(capability=SimpleNamespace(status="available"))):
            with self.subTest(owner=type(owner).__name__):
                self.services.attention_service.responsibilities.episode_checks = owner
                live = self.live()
                self.assertIsNone(live.episode_context.owner)
                await live.start()
                await self.until(lambda: live.service.snapshot().readiness == "ready")
                self.assertEqual(live.service.snapshot().capabilities.durable_responsibilities, "unavailable")

    async def test_explicit_disable_is_distinct_from_missing_owner(self):
        self.services.attention_service.responsibilities.episode_checks = None
        self.services.responsibility_issue = "responsibilities_disabled"
        live = self.live()
        await live.start()
        await self.until(lambda: live.service.snapshot().readiness == "ready")
        self.assertEqual(live.service.snapshot().capabilities.durable_responsibilities, "disabled")

    async def test_cross_session_durable_projection_keeps_original_evidence_and_no_reminder(self):
        live = await self.ready()
        snapshot = live.service.snapshot()
        entry, = (entry for entry in snapshot.entries if entry.kind == "situational_check")
        self.assertEqual(snapshot.capabilities.durable_responsibilities, "available")
        self.assertEqual(entry.scope, self.scope)
        self.assertEqual(entry.check_id, self.admitted.check.id)
        self.assertEqual({ref.source_namespace for ref in entry.sources}, EPISODE_NAMESPACES)
        self.assertEqual(entry.observed_at, NOW + timedelta(seconds=1))
        root, = entry.guard_binding.payload_root_ids
        edge = next(edge for edge in self.db.edges if edge[1] == root)
        self.assertEqual(edge[2], self.source.request_event_id)
        self.assertEqual(self.db.events[edge[2]]["session_id"], self.original_scope.session_id)
        self.assertFalse(self.db.reminders)

    async def test_provider_write_revalidates_owner_not_only_payload_survival(self):
        live = await self.ready()
        binding = await self.voice_binding(live)
        async with live.guard(binding) as connection:
            self.assertTrue(connection.is_in_transaction())
        await self.change("cancel")
        self.assertTrue(self.db.available(self.scope.owner_id, binding.payload_root_ids[0]))
        writer = AsyncMock()
        with self.assertRaises(SourceError):
            async with live.guard(binding):
                await writer()
        writer.assert_not_awaited()

    async def test_episode_rebind_and_check_revision_are_checked_without_polling(self):
        live = await self.ready()
        binding = await self.voice_binding(live)
        changed = await self.change("revise", check=EpisodeCheckDraft(question="Inspect the other control"), rebind_episode=True)
        self.assertEqual(changed.check.episode_or_object_ref.revision, 2)
        with self.assertRaises(SourceError):
            async with live.guard(binding):
                self.fail("old episode accepted")

    async def test_expiry_is_not_completion_and_is_fenced_at_write(self):
        live = await self.ready()
        binding = await self.voice_binding(live)
        self.now = NOW + timedelta(minutes=11)
        with self.assertRaises(SourceError):
            async with live.guard(binding):
                self.fail("expired observation admitted")
        occurrence = self.db.occurrences[self.admitted.check.id]
        self.assertEqual(occurrence["status"], "open")
        self.assertIsNone(occurrence["check_resolution"])

    async def test_source_deletion_and_correction_invalidate_derived_context(self):
        live = await self.ready()
        binding = await self.voice_binding(live)
        self.db.events[self.source.request_event_id]["current"] = False
        self.assertFalse(self.db.available(self.scope.owner_id, binding.payload_root_ids[0]))
        with self.assertRaises(SourceError):
            async with live.guard(binding):
                self.fail("corrected source accepted")
        self.db.events[self.source.request_event_id]["current"] = True
        self.db.events[self.source.request_event_id].update(availability="deleted", content=None)
        with self.assertRaises(SourceError):
            async with live.guard(binding):
                self.fail("deleted source accepted")

    async def test_foreign_context_event_is_not_whitelisted_by_episode_proof(self):
        live = await self.ready()
        record = self.db.record(self.source.request_event_id)
        with self.assertRaises(SourceError):
            await live.binder.bind_record(record, "Do not import foreign conversation text")
        entry = next(entry for entry in live.service.snapshot().entries if entry.kind == "situational_check")
        async with live.binder._transaction() as connection:
            with self.assertRaises(SourceError) as caught:
                await live.binder._validate_dependencies(connection, (*entry.sources, context_dependency(record)))
        self.assertEqual(caught.exception.code, "live_state_event_changed")

    async def test_missing_rule_or_episode_dependency_cannot_authorize_a_check(self):
        live = await self.ready()
        entry = next(entry for entry in live.service.snapshot().entries if entry.kind == "situational_check")
        for namespace in ("responsibility", "episode"):
            with self.subTest(namespace=namespace):
                async with live.binder._transaction() as connection:
                    with self.assertRaises(SourceError):
                        await live.binder._validate_dependencies(connection,
                            tuple(ref for ref in entry.sources if ref.source_namespace != namespace))

    async def test_owner_loss_retracts_projection_without_losing_durable_check(self):
        live = await self.ready()
        binding = await self.voice_binding(live)
        self.db.schema = False
        await self.owner.initialize()
        with self.assertRaises(SourceError):
            async with live.guard(binding):
                self.fail("unavailable owner admitted")
        await self.until(lambda: live.service.snapshot().capabilities.durable_responsibilities == "unavailable"
            and not any(entry.kind == "situational_check" for entry in live.service.snapshot().entries))
        self.assertEqual(live.service.snapshot().readiness, "ready")
        self.assertEqual(self.db.occurrences[self.admitted.check.id]["status"], "open")

    async def test_owner_replacement_is_immediate_guard_failure(self):
        live = await self.ready()
        binding = await self.voice_binding(live)
        other = EpisodeCheckOwner(self.repository, enabled=True, clock=lambda:self.now)
        await other.initialize()
        self.services.attention_service.responsibilities.episode_checks = other
        with self.assertRaises(SourceError):
            async with live.guard(binding):
                self.fail("retired owner instance admitted")

    async def test_cue_activation_is_not_guessed_or_resolved(self):
        cue, _ = await self.admit(cues=("When leaving the office",))
        live = await self.ready()
        await self.until(lambda: len(live.episode_context._cache) == 2)
        self.assertNotIn(cue.check.id, {check.id for check in live.episode_context.checks(live.service.snapshot())})
        self.assertIn("episode_activation_activity_unavailable:" + str(cue.check.id), live.episode_context.omissions)
        self.assertIsNone(self.db.occurrences[cue.check.id]["check_resolution"])

    async def test_read_budget_bounds_projection_without_removing_owner_rows(self):
        for _ in range(9):
            await self.admit()
        live = await self.ready()
        await self.until(lambda: len(live.episode_context._cache) == 8)
        self.assertEqual(len(self.db.occurrences), 10)
        self.assertLessEqual(len(live.episode_context.checks(live.service.snapshot())), 8)

    async def test_canonical_brief_contains_check_and_episode_and_links_old_sources(self):
        live = await self.ready()
        brief = self.brief(live)
        self.assertEqual(len(brief.questions), 1)
        self.assertEqual(brief.questions[0].check_ref.source_id, self.admitted.check.id)
        binder = CanonicalPerceptionBinder(live.binder)
        binding = await binder.bind(brief)
        self.assertEqual(self.db.copies[binding.payload_root_ids[0]]["content"]["brief"], brief.model_dump(mode="json"))
        self.assertTrue(any(edge[1] == binding.payload_root_ids[0] and edge[2] == self.source.request_event_id
                            for edge in self.db.edges))

    async def test_effect_guard_rechecks_cancellation_after_brief_binding(self):
        live = await self.ready()
        brief = self.brief(live)
        binder = CanonicalPerceptionBinder(live.binder)
        binding = await binder.bind(brief)
        async with binder.guard(brief, binding):
            pass
        await self.change("cancel")
        effect = AsyncMock()
        with self.assertRaises(SourceError):
            async with binder.guard(brief, binding):
                await effect()
        effect.assert_not_awaited()

    async def test_session_background_composition_uses_real_owner_checks(self):
        live = await self.ready(perception=True)
        await self.until(lambda: live.perception_brief_bridge._binding is not None
            and bool(live.perception_brief_bridge.composition.brief.questions),
            detail=lambda: live.perception_runtime.worker.last_error)
        brief = live.perception_brief_bridge.composition.brief
        self.assertEqual(brief.durable_responsibility.status, "available")
        self.assertEqual(brief.questions[0].check_ref.source_id, self.admitted.check.id)
        self.assertTrue(brief.scene_change_cues)
        self.assertIsNone(self.db.occurrences[self.admitted.check.id]["check_resolution"])

    async def test_minimal_brief_still_binds_without_episode_owner(self):
        self.services.attention_service.responsibilities.episode_checks = None
        live = self.live()
        await live.start()
        await self.until(lambda: live.service.snapshot().readiness == "ready")
        brief = self.brief(live)
        binding = await CanonicalPerceptionBinder(live.binder).bind(brief)
        self.assertEqual(binding.dependencies, ())
        self.assertEqual(brief.questions, ())


if __name__ == "__main__":
    unittest.main()
