"""Real canonical capture/pipeline, bounded fake storage and model, no DB calls."""
from contextlib import asynccontextmanager
from copy import deepcopy
from dataclasses import replace
from datetime import UTC, datetime
import json
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock
from uuid import uuid4

from app.knowledge.models import SourceError
from app.knowledge.observation_inputs import request_document
from app.knowledge.observations import ObservationPublication
from app.live_state.models import GuardedPayloadBinding, LiveSnapshot, SessionScope, SourceDependency, StateCursor
from app.live_state.perception import ObservationalQuestion
from app.perception.contracts import FastPerceptionFrame, FastPerceptionRequest
from app.perception.fast_analyzer import FastPerceptionAnalyzer
from app.perception.live_brief import LivePerceptionBriefBridge, perception_brief_document
from app.runtime.concurrency import PriorityAsyncLimiter
from app.tasks.models import digest
from app.visual_memory.pipeline import VisualAnalysisPipeline
from test_fast_perception_pipeline import _BehaviorRepository, _Harness, _SessionRepository


class _Storage:
    """Only persistence calls are faked; capture/guards/serialization are real.

    Recursive availability emulates existing DB copy-source locks for the tests.
    This is not a SQL implementation or evidence of a database integration run.
    """

    def __init__(self, scope):
        self.scope = scope
        self.rows, self.parents = {}, {}
        self.observations = {}
        self.events = []
        self.depth = 0
        self.reject_bulk_lock = False
        self.connection = SimpleNamespace(is_in_transaction=lambda: self.depth > 0,
            fetch=AsyncMock(side_effect=self.fetch), fetchval=AsyncMock(side_effect=self.fetchval),
            executemany=AsyncMock(side_effect=self.executemany), execute=AsyncMock(side_effect=self.execute))
        self.sources = SimpleNamespace(pool=None)
        self.tasks = SimpleNamespace(payload=AsyncMock(side_effect=self.payload))
        self.context = SimpleNamespace(tasks=self.tasks,
            publication=SimpleNamespace(sources=self.sources, area_id=scope.area_id, transaction=self.transaction),
            voice_history_in_transaction=AsyncMock(return_value={"records": [], "total": 0}))
        self.publisher = ObservationPublication(self.context, owner_id=scope.owner_id)

    @asynccontextmanager
    async def transaction(self, tasks, owner):
        assert tasks is self.tasks and owner == self.scope.owner_id
        self.depth += 1
        try:
            yield self.connection
        finally:
            self.depth -= 1

    def original(self, content, *, owner=None):
        identity = uuid4()
        self.rows[identity] = {"owner": owner or self.scope.owner_id, "availability": "available", "content": deepcopy(content)}
        return identity

    def available(self, identity, seen=frozenset()):
        row = self.rows.get(identity)
        return (identity not in seen and row is not None and row["owner"] == self.scope.owner_id
                and row["availability"] == "available"
                and all(self.available(parent, seen | {identity}) for parent in self.parents.get(identity, ())))

    async def fetch(self, query, *args):
        assert self.depth and args[0] == self.scope.owner_id
        if "select id,availability from jarvis_context_payloads" in query:
            return [{"id": identity, "availability": self.rows[identity]["availability"]}
                    for identity in args[1] if identity in self.rows and self.rows[identity]["owner"] == args[0]]
        if "jarvis_knowledge_source_versions" in query:
            # The supplied roots are canonical bound payloads, not indexed memory
            # originals. Their removable parents remain guarded transitively.
            return []
        if "select id,primary_visual_frame_id from perception_observations" in query:
            return []
        raise AssertionError("Unexpected storage fetch: " + query)

    async def fetchval(self, query, *args):
        assert self.depth and args[0] == self.scope.owner_id
        if "select payload_id from jarvis_observation_sources" in query:
            return self.observations.get(args[2])
        if "lock_jarvis_copy_sources(" in query:
            self.events.append(("lock_roots", tuple(args[1])))
            return not self.reject_bulk_lock and all(self.available(identity) for identity in args[1])
        if "lock_jarvis_copy_source(" in query:
            self.events.append(("guard_input", args[1]))
            return "available" if self.available(args[1]) else "unavailable"
        if "select content from jarvis_context_payloads" in query:
            return deepcopy(self.rows[args[1]]["content"])
        raise AssertionError("Unexpected storage fetchval: " + query)

    async def payload(self, connection, owner, content):
        assert self.depth and connection is self.connection and owner == self.scope.owner_id
        identity = self.original(content)
        self.events.append(("payload", identity))
        return identity

    async def executemany(self, query, values):
        assert self.depth
        if "insert into jarvis_context_payload_dependencies" not in query:
            raise AssertionError("Unexpected dependency write")
        for owner, parent, dependent in values:
            assert owner == self.scope.owner_id
            self.parents.setdefault(dependent, set()).add(parent)
            self.events.append(("edge", parent, dependent))

    async def execute(self, query, *args):
        assert self.depth and args[0] == self.scope.owner_id
        if "update jarvis_context_payloads" not in query:
            raise AssertionError("Unexpected storage mutation")
        identity = args[1]
        if not any(identity in parents for parents in self.parents.values()):
            self.rows[identity].update(availability="deleted", content=None)


class _CanonicalSessionRepository(_SessionRepository):
    async def get_recent_messages(self, owner_id, session_id, *, limit):
        messages = await super().get_recent_messages(owner_id, session_id, limit=limit)
        return [SimpleNamespace(**(vars(message) | {"provider_item_id": f"fixture-message-{index}"}))
                for index, message in enumerate(messages)]


class BriefCaptureTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.now = datetime.now(UTC)
        self.scope = SessionScope(owner_id="owner", session_id=uuid4(), area_id=uuid4())
        self.storage = _Storage(self.scope)
        self.source = SourceDependency(source_namespace="visual_focus", source_id=uuid4(), revision=7)
        self.assignment_current = True

        @asynccontextmanager
        async def guard(brief, binding, *, effect_payload_ids=()):
            async with self.storage.transaction(self.storage.tasks, self.scope.owner_id) as connection:
                if not self.assignment_current or not all(self.storage.available(root)
                        for root in (*binding.payload_root_ids, *effect_payload_ids)):
                    raise SourceError("perception_focus_changed", "The fixture focus was replaced.")
                yield connection

        self.bridge = LivePerceptionBriefBridge(scope=self.scope, scope_epoch=uuid4(), enabled=True,
            clock=lambda: self.now, result_guard=guard)
        self.bound_roots = ()

        async def bind(brief):
            # Explicit test-only binder. No production binding authority is added.
            self.bound_roots = tuple(self.storage.original({"brief": perception_brief_document(brief)}) for _ in range(2))
            return GuardedPayloadBinding(scope=self.scope, payload_root_ids=self.bound_roots, dependencies=brief.dependencies)

        self.bind = AsyncMock(side_effect=bind)
        await self.bridge.refresh(LiveSnapshot(scope=self.scope, cursor=StateCursor(epoch=uuid4(), revision=1),
            as_of=self.now, readiness="ready"), bind=self.bind, dependency_eligible=lambda source: source == self.source,
            explicit_questions=(ObservationalQuestion(id=uuid4(), question="Is the display readable?", focus_ref=self.source),))
        self.request = self.bridge.attach(FastPerceptionRequest(owner_id=self.scope.owner_id,
            session_id=self.scope.session_id, window_id=uuid4(),
            frames=(FastPerceptionFrame(uuid4(), 1, self.now, "glasses", "image/jpeg", b"offline"),),
            previous_working_state={}, existing_entity_ids=frozenset(), recent_audio_transcripts=(),
            behavior_instructions="Observe only", active_reminders=(), active_focus=None,
            recent_interrupts=(), requested_at=self.now))
        self.context = SimpleNamespace(owner_id=self.scope.owner_id, session_id=self.scope.session_id,
            recent_visual_interrupts=(), recent_audio_transcripts=())

    async def test_prepare_captures_final_document_locks_all_roots_links_then_seals(self):
        prepared, snapshot = await self.storage.publisher.prepare_inputs(self.request, self.context)
        stored = self.storage.rows[snapshot.payload_id]["content"]
        self.assertEqual(stored["request"], request_document(prepared))
        self.assertEqual(stored["request"]["perception_brief"], perception_brief_document(self.request.perception_brief))
        self.assertEqual(stored["request"]["perception_brief"]["questions"][0]["focus_ref"]["revision"], 7)
        self.assertEqual(prepared.perception_brief_input_digest, snapshot.request_digest)
        self.assertEqual(snapshot.request_digest, digest(request_document(prepared)))
        self.assertNotIn("perception_brief_input_digest", stored["request"])
        self.assertEqual(self.storage.parents[snapshot.payload_id], set(self.bound_roots))
        lock_index = next(i for i, event in enumerate(self.storage.events) if event[0] == "lock_roots")
        payload_index = next(i for i, event in enumerate(self.storage.events) if event[0] == "payload")
        self.assertLess(lock_index, payload_index)
        self.assertEqual(set(self.storage.events[lock_index][1]), set(self.bound_roots))
        await self.storage.publisher.validate_input(prepared, snapshot)

    async def test_default_request_has_no_brief_fields_or_extra_parent_edges(self):
        plain = replace(self.request, perception_brief=None, perception_brief_binding=None)
        self.assertNotIn("perception_brief", request_document(plain))
        prepared, snapshot = await self.storage.publisher.prepare_inputs(plain, self.context)
        self.assertIsNone(prepared.perception_brief_input_digest)
        self.assertNotIn(snapshot.payload_id, self.storage.parents)
        self.assertFalse(any(event[0] == "lock_roots" for event in self.storage.events))
        await self.storage.publisher.validate_input(prepared, snapshot)

    async def test_missing_deleted_and_foreign_roots_fail_without_materialization(self):
        original = deepcopy(self.storage.rows[self.bound_roots[0]])
        for failure in ("missing", "deleted", "foreign_owner"):
            with self.subTest(failure=failure):
                self.storage.rows[self.bound_roots[0]] = deepcopy(original)
                if failure == "missing":
                    del self.storage.rows[self.bound_roots[0]]
                elif failure == "deleted":
                    self.storage.rows[self.bound_roots[0]]["availability"] = "deleted"
                else:
                    self.storage.rows[self.bound_roots[0]]["owner"] = "another-owner"
                with self.assertRaises(SourceError) as raised:
                    await self.storage.publisher.prepare_inputs(self.request, self.context)
                self.assertEqual(raised.exception.code, "observation_brief_context_unavailable")
                self.storage.tasks.payload.assert_not_awaited()

    async def test_root_changed_between_read_and_lock_fails_before_copy(self):
        self.storage.reject_bulk_lock = True
        with self.assertRaises(SourceError) as raised:
            await self.storage.publisher.prepare_inputs(self.request, self.context)
        self.assertEqual(raised.exception.code, "observation_context_changed")
        self.storage.tasks.payload.assert_not_awaited()

    async def test_parent_revocation_after_capture_blocks_existing_input_guard(self):
        prepared, snapshot = await self.storage.publisher.prepare_inputs(self.request, self.context)
        self.storage.rows[self.bound_roots[0]]["availability"] = "deleted"
        with self.assertRaises(SourceError) as raised:
            await self.storage.publisher.validate_input(prepared, snapshot)
        self.assertEqual(raised.exception.code, "observation_input_removed")

    async def test_exact_brief_and_binding_changes_invalidate_capture_fingerprint(self):
        prepared, snapshot = await self.storage.publisher.prepare_inputs(self.request, self.context)
        changes = (
            {"perception_brief": prepared.perception_brief.model_copy(update={"version": prepared.perception_brief.version + 1})},
            {"perception_brief_binding": prepared.perception_brief_binding.model_copy(update={"payload_root_ids": (uuid4(),)})},
        )
        for change in changes:
            with self.subTest(field=next(iter(change))):
                with self.assertRaises(SourceError) as raised:
                    await self.storage.publisher.validate_input(replace(prepared, **change), snapshot)
                self.assertEqual(raised.exception.code, "observation_input_changed")

    async def test_removed_context_is_still_pruned_before_brief_is_sealed(self):
        request = replace(self.request, previous_working_state={"summary": "unbound operational copy"})
        prepared, snapshot = await self.storage.publisher.prepare_inputs(request, self.context)
        self.assertEqual(prepared.previous_working_state, {})
        self.assertTrue(prepared.input_context_gaps)
        self.assertEqual(prepared.perception_brief, request.perception_brief)
        self.assertEqual(prepared.perception_brief_input_digest, digest(request_document(prepared)))
        self.assertEqual(self.storage.parents[snapshot.payload_id], set(self.bound_roots))

    async def pipeline(self, *, resolver=True, selected=True, registered=False,
                       revoke_before_model=False, revoke_during_model=False, obsolete_during_model=False):
        harness = _Harness(static=True)
        await harness.pipeline.close()
        # The harness does not open services or databases; replace its session
        # identity to exercise the real constructor's authenticated lookup seam.
        harness.session_id = self.scope.session_id
        captures, stored_inputs, lookups = [], [], []

        async def generate(*, prompt, images):
            captures.append((prompt, images))
            if revoke_during_model:
                self.storage.rows[self.bound_roots[0]]["availability"] = "deleted"
            if obsolete_during_model:
                self.assignment_current = False
            return json.dumps({"summary": "The display remains unreadable.", "starts_new_event": False, "confidence": 0.6})

        model = SimpleNamespace(provider="offline", model="fixture", generate_temporal_json=AsyncMock(side_effect=generate))

        async def insert(*, observation, frame_links, analysis_input=None):
            if analysis_input is not None:
                async with self.storage.publisher.transaction(observation.owner_id) as connection:
                    # Existing canonical guard at the result-publication boundary.
                    await self.storage.publisher.inputs.check(connection, analysis_input,
                        owner_id=observation.owner_id, session_id=observation.session_id,
                        frame_ids=tuple(link.visual_frame_id for link in frame_links))
                stored_inputs.append(analysis_input)
                original = self.storage.original({"observation_id": str(observation.id)})
                self.storage.parents[original] = {analysis_input.payload_id}
                self.storage.observations[observation.id] = original
            return await harness.perception_repository.insert_observation(observation=observation, frame_links=frame_links)

        async def apply_observation(observation, *, effect_guard=None):
            if effect_guard is None:
                return await harness.working_memory_service.apply_observation(observation)
            async with effect_guard():
                return await harness.working_memory_service.apply_observation(observation)

        def bridge_for(owner_id, session_id):
            lookups.append((owner_id, session_id))
            return self.bridge if (owner_id, session_id) == (self.scope.owner_id, self.scope.session_id) else None

        pipeline = VisualAnalysisPipeline(settings=harness.pipeline._settings,
            repository=harness.visual_repository, session_repository=_CanonicalSessionRepository(harness.session_id),
            behavior_repository=_BehaviorRepository(), interrupt_service=harness.interrupt_service,
            model_limiter=PriorityAsyncLimiter(1), fast_analyzer=FastPerceptionAnalyzer(model),
            perception_repository=SimpleNamespace(insert_observation=insert, observation_publication=None),
            sensory_frame_window=harness.frame_window, domain_event_bus=harness.event_bus,
            working_memory_service=SimpleNamespace(apply_observation=apply_observation,
                get_session_working_memory=harness.working_memory_service.get_session_working_memory),
            live_brief_bridge_for=bridge_for if resolver else None)
        if selected:
            pipeline.set_observation_publication(self.storage.publisher)
        if registered:
            pipeline.register_live_brief(self.bridge)
        if revoke_before_model:
            original_prepare = self.storage.publisher.prepare_inputs

            async def prepare(request, context):
                result = await original_prepare(request, context)
                self.storage.rows[self.bound_roots[0]]["availability"] = "deleted"
                return result

            self.storage.publisher.prepare_inputs = prepare
        frame = harness.frame(sequence=1, captured_at=self.now)
        harness.put(frame, b"offline")
        try:
            await pipeline.enqueue(frame=frame, image_bytes=b"offline")
            await pipeline.drain()
        finally:
            await pipeline.close()
        return harness, model, captures, stored_inputs, lookups

    async def test_actual_pipeline_attaches_before_real_capture_and_runs_one_analyzer(self):
        harness, model, captures, inputs, lookups = await self.pipeline()
        self.assertEqual(harness.visual_repository.failed, [])
        self.assertEqual(lookups, [(self.scope.owner_id, self.scope.session_id)])
        model.generate_temporal_json.assert_awaited_once()
        self.assertIn('"perception_assignment"', captures[0][0])
        self.assertIn(str(self.bridge.composition.brief.id), captures[0][0])
        self.assertEqual(len(harness.perception_repository.inserted), 1)
        self.assertEqual(len(harness.working_memory_service.applied), 1)
        self.assertEqual(self.storage.parents[inputs[0].payload_id], set(self.bound_roots))
        self.assertEqual(harness.interrupt_service.calls, [])
        self.bind.assert_awaited_once()

    async def test_registered_session_reaches_real_capture_without_injected_resolver(self):
        harness, model, captures, inputs, lookups = await self.pipeline(resolver=False, registered=True)
        self.assertEqual(harness.visual_repository.failed, [])
        self.assertEqual(lookups, [])
        model.generate_temporal_json.assert_awaited_once()
        self.assertIn(str(self.bridge.composition.brief.id), captures[0][0])
        self.assertEqual(self.storage.parents[inputs[0].payload_id], set(self.bound_roots))
        self.assertEqual(len(harness.perception_repository.inserted), 1)
        self.assertEqual(len(harness.working_memory_service.applied), 1)
        self.bind.assert_awaited_once()

    async def test_replaced_assignment_retains_history_without_current_effects(self):
        harness, model, _, inputs, _ = await self.pipeline(registered=True, obsolete_during_model=True)
        model.generate_temporal_json.assert_awaited_once()
        self.assertEqual(harness.visual_repository.failed, [])
        self.assertEqual(len(harness.perception_repository.inserted), 1)
        self.assertEqual(len(inputs), 1)
        self.assertEqual(harness.working_memory_service.applied, [])
        self.assertEqual(harness.interrupt_service.calls, [])

    async def test_registered_session_takes_precedence_over_fixture_resolver(self):
        harness, model, _, inputs, lookups = await self.pipeline(registered=True)
        self.assertEqual(harness.visual_repository.failed, [])
        self.assertEqual(lookups, [])
        model.generate_temporal_json.assert_awaited_once()
        self.assertEqual(self.storage.parents[inputs[0].payload_id], set(self.bound_roots))

    async def test_pipeline_without_resolver_preserves_existing_capture_path(self):
        harness, model, captures, inputs, lookups = await self.pipeline(resolver=False)
        self.assertEqual(harness.visual_repository.failed, [])
        self.assertEqual(lookups, [])
        model.generate_temporal_json.assert_awaited_once()
        self.assertNotIn("perception_assignment", captures[0][0])
        self.assertNotIn("perception_brief", self.storage.rows[inputs[0].payload_id]["content"]["request"])

    async def test_attached_brief_requires_selected_canonical_capture(self):
        harness, model, _, inputs, _ = await self.pipeline(selected=False)
        model.generate_temporal_json.assert_not_awaited()
        self.assertEqual(len(harness.visual_repository.failed), 1)
        self.assertEqual(inputs, [])

    async def test_revocation_after_capture_blocks_provider_and_discards_unused_input(self):
        harness, model, _, inputs, _ = await self.pipeline(revoke_before_model=True)
        model.generate_temporal_json.assert_not_awaited()
        self.assertEqual(len(harness.visual_repository.failed), 1)
        self.assertEqual(inputs, [])
        copies = [identity for event, identity, *rest in self.storage.events if event == "payload"]
        self.assertEqual(len(copies), 1)
        self.assertEqual(self.storage.rows[copies[0]]["availability"], "deleted")

    async def test_revocation_during_analysis_blocks_observation_publication(self):
        harness, model, _, inputs, _ = await self.pipeline(revoke_during_model=True)
        model.generate_temporal_json.assert_awaited_once()
        self.assertEqual(len(harness.visual_repository.failed), 1)
        self.assertEqual(harness.perception_repository.inserted, [])
        self.assertEqual(harness.working_memory_service.applied, [])
        self.assertEqual(inputs, [])
        self.assertEqual(harness.interrupt_service.calls, [])


if __name__ == "__main__":
    unittest.main()
