"""Offline canonical brief binding and non-Hindsight session lifecycle tests."""
import asyncio
from contextlib import asynccontextmanager
from datetime import timedelta
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock
from uuid import uuid4

from app.knowledge.models import SourceError
from app.live_state.canonical import CanonicalContextBinder, context_dependency, memory_dependency
from app.live_state.models import GuardedPayloadBinding, LiveSnapshot, ObservationEntry, SessionScope, SourceDependency, StateCursor
from app.live_state.perception_context import compose_perception_brief
from app.live_state.perception_runtime import CanonicalPerceptionBinder
from app.live_state.runtime import create_session_live_state
from app.perception.contracts import FastPerceptionFrame, FastPerceptionRequest
from app.runtime.events import DomainEventBus
from app.runtime.feature_config import MemoryAuthority
from test_live_state_canonical_adapters import NOW, memory, record


class _CanonicalIO:
    def __init__(self, scope):
        self.scope = scope
        self.records, self.memories, self.roots = {}, {}, set()
        self.transactions = 0
        self.connection = SimpleNamespace(is_in_transaction=lambda: True,
            fetchval=AsyncMock(side_effect=self.fetchval), fetch=AsyncMock(return_value=[]),
            execute=AsyncMock(), executemany=AsyncMock())
        self.sources = SimpleNamespace(pool=None, transaction=self.transaction,
            _area=AsyncMock(), _read_many=AsyncMock(side_effect=self.read_many))

    @asynccontextmanager
    async def transaction(self, owner):
        assert owner == self.scope.owner_id
        self.transactions += 1
        yield self.connection

    async def fetchval(self, query, *args):
        if "lock_jarvis_copy_sources" in query:
            return set(args[1]).issubset(self.roots)
        return True

    async def read_many(self, connection, owner, area, requests):
        assert (owner, area) == (self.scope.owner_id, self.scope.area_id)
        return [self.memories[identity] for identity, revision in requests]

    def patch_io(self, canonical):
        canonical.sources.transaction = self.transaction
        canonical.sources.read_guard_transaction = self.transaction
        canonical.sources.derivation_transaction = self.transaction
        canonical.sources._read_many = self.sources._read_many
        canonical.sources._area = self.sources._area
        canonical.tasks.lock_owner = AsyncMock()
        canonical.tasks.lock_owner_shared = AsyncMock()

        async def payload(*args, **kwargs):
            root = uuid4()
            self.roots.add(root)
            return root

        canonical.tasks.payload = AsyncMock(side_effect=payload)
        canonical.context._records = AsyncMock(side_effect=lambda conn, owner, ids: {
            identity: self.records[identity] for identity in ids})
        canonical.context._record = AsyncMock(side_effect=lambda conn, owner, identity: self.records[identity])


class CanonicalPerceptionBindingTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.scope = SessionScope(owner_id="brief-binding", session_id=uuid4(), area_id=uuid4())
        self.now = NOW
        self.io = _CanonicalIO(self.scope)
        self.canonical = CanonicalContextBinder(self.io.sources, self.scope, clock=lambda: self.now)
        self.io.patch_io(self.canonical)
        self.binder = CanonicalPerceptionBinder(self.canonical)
        self.snapshot = LiveSnapshot(scope=self.scope, cursor=StateCursor(epoch=uuid4(), revision=1),
                                     as_of=NOW, readiness="ready")

    def brief(self, dependencies=()):
        return compose_perception_brief(self.snapshot, now=NOW, scope_epoch=self.snapshot.cursor.epoch,
            context_dependencies=dependencies, dependency_eligible=lambda source: True).brief

    async def test_minimal_brief_materializes_exact_document_without_memory_source(self):
        brief = self.brief()
        binding = await self.binder.bind(brief)
        content = self.canonical.tasks.payload.call_args.args[2]
        self.assertEqual(content, {"format": "jarvis.perception_brief.v1", "brief": brief.model_dump(mode="json")})
        self.assertEqual(binding.dependencies, ())
        self.assertEqual(self.canonical.tasks.payload.call_args.kwargs["source_payload_ids"], ())
        self.io.sources._read_many.assert_not_awaited()

    async def test_context_binding_links_original_and_exact_event_revision(self):
        original = record(self.scope)
        self.io.records[original.id] = original
        self.io.roots.add(original.payload_id)
        binding = await self.binder.bind(self.brief((context_dependency(original),)))
        self.assertEqual(binding.dependencies, (context_dependency(original),))
        self.assertEqual(self.canonical.tasks.payload.call_args.kwargs["source_payload_ids"], (original.payload_id,))
        row = self.io.connection.executemany.call_args.args[1][0]
        self.assertEqual(row[2:], (original.id, original.revision, original.payload_id, "current"))

    async def test_memory_binding_reuses_existing_provenance_and_generation(self):
        original = memory(self.scope, revision=3, generation=8)
        self.io.memories[original.source_id] = original
        self.io.roots.add(original.payload_id)
        binding = await self.binder.bind(self.brief((memory_dependency(original),)))
        self.assertEqual(binding.dependencies[0].generation, 8)
        self.assertEqual(self.canonical.tasks.payload.call_args.kwargs["source_payload_ids"], (original.payload_id,))
        self.assertTrue(any("jarvis_context_memory_sources" in call.args[0]
                            for call in self.io.connection.executemany.call_args_list))

    async def test_unsupported_owner_namespaces_fail_before_io(self):
        for namespace in ("responsibility", "visual_focus", "situational_check", "responsibility_check", "episode"):
            with self.subTest(namespace=namespace):
                dependency = SourceDependency(source_namespace=namespace, source_id=uuid4(), revision=1)
                with self.assertRaises(SourceError) as raised:
                    await self.binder.bind(self.brief((dependency,)))
                self.assertEqual(raised.exception.code, "perception_brief_dependency_unsupported")
        self.assertEqual(self.io.transactions, 0)
        self.canonical.tasks.payload.assert_not_awaited()

    async def test_changed_generation_cannot_bind_stale_memory(self):
        original = memory(self.scope)
        self.io.memories[original.source_id] = original.model_copy(update={"generation": 2})
        self.io.roots.add(original.payload_id)
        with self.assertRaises(SourceError):
            await self.binder.bind(self.brief((memory_dependency(original),)))
        self.canonical.tasks.payload.assert_not_awaited()

    async def test_foreign_context_session_is_not_bindable(self):
        original = record(self.scope).model_copy(update={"session_id": uuid4()})
        self.io.records[original.id] = original
        self.io.roots.add(original.payload_id)
        with self.assertRaises(SourceError):
            await self.binder.bind(self.brief((context_dependency(original),)))
        self.canonical.tasks.payload.assert_not_awaited()

    async def test_cached_root_is_rechecked_without_new_payload(self):
        brief = self.brief()
        first = await self.binder.bind(brief)
        self.assertEqual(await self.binder.bind(brief), first)
        self.canonical.tasks.payload.assert_awaited_once()
        self.io.roots.remove(first.payload_root_ids[0])
        with self.assertRaises(SourceError):
            await self.binder.bind(brief)
        self.canonical.tasks.payload.assert_awaited_once()

    async def test_expired_brief_never_contacts_storage(self):
        brief = self.brief()
        self.now = brief.expires_at
        with self.assertRaises(SourceError):
            await self.binder.bind(brief)
        self.assertEqual(self.io.transactions, 0)


class SessionPerceptionRuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.scope = SessionScope(owner_id="context-only", session_id=uuid4())
        self.now = NOW
        self.bus = DomainEventBus()
        self.context = SimpleNamespace(pool=object(), snapshot=AsyncMock(return_value={"recent": [], "pinned": []}))
        self.services = SimpleNamespace(memory_authority=MemoryAuthority.LEGACY,
            knowledge_runtime=None, db_pool=self.context.pool, _closed=False,
            background_task_context_service=SimpleNamespace(repository=self.context),
            background_task_service=None, domain_event_bus=self.bus,
            working_memory_service=SimpleNamespace(get_session_working_memory=AsyncMock(side_effect=AssertionError("legacy read"))))
        self.io = _CanonicalIO(self.scope)
        self.live = create_session_live_state(self.services, scope=self.scope, enabled=True,
            perception_brief_enabled=True, clock=lambda: self.now, interval_seconds=0.01, timeout_seconds=0.5)
        self.io.patch_io(self.live.binder)
        self.live.sources.current_snapshot = AsyncMock(side_effect=AssertionError("Hindsight/memory read"))

    async def asyncTearDown(self):
        await self.live.close()

    async def until(self, predicate):
        async with asyncio.timeout(1):
            while not predicate():
                await asyncio.sleep(0.002)

    async def ready(self):
        await self.live.start()
        await self.until(lambda: self.live.perception_runtime.diagnostics()["status"] == "ready")

    def request(self):
        return FastPerceptionRequest(owner_id=self.scope.owner_id, session_id=self.scope.session_id,
            window_id=uuid4(), frames=(FastPerceptionFrame(uuid4(), 1, self.now, "glasses", "image/jpeg", b"offline"),),
            previous_working_state={}, existing_entity_ids=frozenset(), recent_audio_transcripts=(),
            behavior_instructions="", active_reminders=(), active_focus=None, recent_interrupts=(), requested_at=self.now)

    async def test_non_hindsight_background_minimal_brief_has_canonical_binding(self):
        await self.ready()
        await self.until(lambda: self.live.service.snapshot().readiness == "ready"
            and self.live.perception_brief_bridge._binding is not None
            and "still loading" not in " ".join(
                self.live.perception_brief_bridge.composition.brief.scene_change_cues))
        request = self.live.perception_brief_bridge.attach(self.request())
        self.assertIsNotNone(request.perception_brief_binding)
        self.assertEqual(request.perception_brief.questions, ())
        self.assertTrue(request.perception_brief.scene_change_cues)
        self.assertEqual(request.perception_brief.durable_responsibility.status, "unavailable")
        self.assertEqual(self.live.perception_runtime.diagnostics()["selected_memory_binding"], "disabled")
        self.assertIn("visual_focus_binding_unavailable", self.live.perception_runtime.diagnostics()["owner_binding_gaps"])
        self.live.sources.current_snapshot.assert_not_awaited()
        self.services.working_memory_service.get_session_working_memory.assert_not_awaited()
        payload = self.live.binder.tasks.payload
        bound_documents = [call.args[2]["brief"] for call in payload.await_args_list]
        ready_document = request.perception_brief.model_dump(mode="json")
        self.assertEqual(bound_documents.count(ready_document), 1)
        versions = [(document["id"], document["version"]) for document in bound_documents]
        self.assertEqual(len(versions), len(set(versions)))
        # Warming and ready have deliberately distinct semantics. Once ready,
        # repeated refreshes must reuse the same bound payload, not re-copy it.
        before_payloads = payload.await_count
        before_refreshes = self.live.perception_runtime.worker.refresh_count
        await self.until(lambda: self.live.perception_runtime.worker.refresh_count >= before_refreshes + 3)
        self.assertEqual(payload.await_count, before_payloads)

    async def test_guarded_observation_prior_preserves_original_time_and_uncertainty(self):
        await self.ready()
        original = record(self.scope).model_copy(update={"kind": "observation", "trust": "derived"})
        self.io.records[original.id] = original
        self.io.roots.add(original.payload_id)
        dependency = context_dependency(original)
        entry = ObservationEntry(entry_id="fixture-observation", scope=self.scope,
            received_at=NOW, observed_at=NOW - timedelta(seconds=1), expires_at=NOW + timedelta(seconds=10),
            evidence_kind="observed", summary="The display appears dark.", confidence=0.6,
            unknowns=("The off marking is unreadable.",), sources=(dependency,),
            guard_binding=GuardedPayloadBinding(scope=self.scope, payload_root_ids=(original.payload_id,), dependencies=(dependency,)))
        self.assertTrue(self.live.producer.publish(entry).queued)
        await self.live.service.wait_idle()
        await self.until(lambda: bool(self.live.perception_brief_bridge.composition.brief.prior_evidence))
        prior = self.live.perception_brief_bridge.composition.brief.prior_evidence[0]
        self.assertEqual(prior.reference.captured_at, entry.observed_at)
        self.assertEqual(prior.uncertainty, entry.unknowns[0])
        self.assertEqual(prior.reference.source, dependency)
        self.assertEqual(self.live.perception_brief_bridge.composition.brief.questions, ())

    async def test_close_cancels_background_binding_and_unsubscribes(self):
        entered, release = asyncio.Event(), asyncio.Event()

        async def slow(brief):
            entered.set()
            await release.wait()

        self.live.perception_runtime.binder.bind = slow
        await self.live.start()
        await asyncio.wait_for(entered.wait(), timeout=1)
        await asyncio.wait_for(self.live.close(), timeout=0.5)
        self.assertTrue(self.live.perception_runtime.worker._task.done())
        self.assertFalse(self.bus._handlers)
        self.assertEqual(self.live.perception_runtime.diagnostics()["status"], "closed")
        request = self.request()
        self.assertIs(self.live.perception_brief_bridge.attach(request), request)
        self.assertFalse(release.is_set())

    async def test_disabled_brief_has_no_worker_or_binding(self):
        other = create_session_live_state(self.services, scope=self.scope, enabled=True,
            clock=lambda: self.now, interval_seconds=0.01)
        self.io.patch_io(other.binder)
        try:
            await other.start()
            await self.until(lambda: other.service.snapshot().readiness == "ready")
            self.assertIsNone(other.perception_runtime.worker._task)
            self.assertEqual(other.perception_runtime.diagnostics()["status"], "disabled")
            other.binder.tasks.payload.assert_not_awaited()
        finally:
            await other.close()

    async def test_context_only_requires_storage_and_does_not_claim_an_area(self):
        missing = SimpleNamespace(memory_authority=MemoryAuthority.LEGACY, domain_event_bus=self.bus)
        with self.assertRaises(SourceError):
            create_session_live_state(missing, scope=self.scope, enabled=True)
        with self.assertRaises(SourceError):
            create_session_live_state(self.services, scope=self.scope.model_copy(update={"area_id": uuid4()}), enabled=True)
