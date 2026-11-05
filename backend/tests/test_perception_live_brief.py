"""Offline real composer -> request -> prompt -> analyzer/pipeline coverage."""
import asyncio
from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import UTC, datetime, timedelta
import json
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from app.knowledge.observation_inputs import ObservationInputSnapshot, request_document
from app.knowledge.models import SourceError
from app.live_state.models import GuardedPayloadBinding, LiveCapabilities, LiveSnapshot, SessionScope, SourceDependency, StateCursor
from app.live_state.perception import CheckLifecycle, PerceptionActivity, SituationalCheck
from app.perception.contracts import FastPerceptionFrame, FastPerceptionRequest
from app.perception.fast_analyzer import FastPerceptionAnalyzer
from app.perception.live_brief import (
    LivePerceptionBriefBridge, live_brief_input_fields, live_brief_parent_payload_ids,
    perception_brief_document, render_live_perception_brief, seal_live_brief_input,
)
from app.perception.prompts import build_fast_perception_prompt
from app.runtime.concurrency import PriorityAsyncLimiter
from app.tasks.models import digest
from app.visual_memory.pipeline import VisualAnalysisPipeline
from test_fast_perception_pipeline import _BehaviorRepository, _Harness
from test_perception_brief_capture import _CanonicalSessionRepository, _Storage


class LiveBriefTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.now = datetime.now(UTC)
        self.scope = SessionScope(owner_id="owner", session_id=uuid4(), area_id=uuid4())
        self.epoch = uuid4()
        self.snapshot = LiveSnapshot(scope=self.scope, cursor=StateCursor(epoch=uuid4(), revision=12),
            as_of=self.now, readiness="ready", capabilities=LiveCapabilities(durable_responsibilities="available"))
        self.episode = SourceDependency(source_namespace="episode", source_id=uuid4(), revision=4)
        self.focus = SourceDependency(source_namespace="visual_focus", source_id=uuid4(), revision=7)
        self.check = SituationalCheck(id=self.focus.source_id, revision=self.focus.revision,
            scope=self.scope, episode_or_object_ref=self.episode, origin="explicit_request",
            canonical_owner_ref=self.focus, question="Is the display readable?",
            lifecycle=CheckLifecycle.TRACKING, relevance_now=True)
        self.allowed = {self.episode, self.focus}
        self.bridge = LivePerceptionBriefBridge(scope=self.scope, scope_epoch=self.epoch,
            enabled=True, clock=lambda: self.now)
        self.bind = AsyncMock(side_effect=lambda brief: GuardedPayloadBinding(scope=brief.scope,
            payload_root_ids=(uuid4(),), dependencies=brief.dependencies))
        self.model = SimpleNamespace(provider="offline", model="bounded-fixture",
            generate_temporal_json=AsyncMock(return_value=json.dumps({"summary": "Display is unreadable from this view.",
                "starts_new_event": False, "confidence": 0.6})))

    async def refresh(self, **changes):
        return await self.bridge.refresh(self.snapshot, **(dict(bind=self.bind,
            checks=(self.check,), dependency_eligible=lambda source: source in self.allowed,
            activity=PerceptionActivity(label="inspecting", confidence=0.7, uncertainty="Display text is unclear.")) | changes))

    def request(self, **changes):
        return FastPerceptionRequest(**(dict(owner_id=self.scope.owner_id, session_id=self.scope.session_id,
            window_id=uuid4(), frames=(FastPerceptionFrame(uuid4(), 1, self.now, "glasses", "image/jpeg", b"offline"),),
            previous_working_state={}, existing_entity_ids=frozenset(), recent_audio_transcripts=(),
            behavior_instructions="Observe only.", active_reminders=(), active_focus=None,
            recent_interrupts=(), requested_at=self.now) | changes))

    def capture(self, request):
        # Lightweight receipt fixture; test_perception_brief_capture exercises
        # the real capture/publication and pipeline paths with bounded storage.
        document = request_document(request)
        snapshot = ObservationInputSnapshot(request.owner_id, request.session_id, uuid4(), digest(document))
        return seal_live_brief_input(request, snapshot), document

    async def test_default_off_is_identity_and_does_not_bind(self):
        bridge = LivePerceptionBriefBridge(scope=self.scope, scope_epoch=self.epoch)
        request = self.request()
        self.assertIsNone(await bridge.refresh(self.snapshot, bind=self.bind))
        self.assertIs(bridge.attach(request), request)
        self.bind.assert_not_awaited()
        self.assertEqual(live_brief_input_fields(request), {})
        self.assertNotIn("perception_assignment", build_fast_perception_prompt(request))

    async def test_actual_analyzer_receives_exact_assignment_with_one_model_call(self):
        composition = await self.refresh()
        request, document = self.capture(self.bridge.attach(self.request()))
        output = await FastPerceptionAnalyzer(self.model).analyze(request)
        self.model.generate_temporal_json.assert_awaited_once()
        prompt = self.model.generate_temporal_json.await_args.kwargs["prompt"]
        context = json.loads(prompt.split("Context JSON:\n```json\n", 1)[1].split("\n```", 1)[0])
        assignment = context["perception_assignment"]
        self.assertEqual(assignment["id"], str(composition.brief.id))
        self.assertEqual(assignment["version"], composition.brief.version)
        self.assertEqual(assignment["scope_epoch"], str(self.epoch))
        self.assertEqual(assignment["questions"][0]["focus_ref"], self.focus.model_dump(mode="json"))
        self.assertEqual(assignment["questions"][0]["check_ref"]["revision"], 7)
        self.assertEqual(assignment["questions"][0]["episode_or_object_ref"]["revision"], 4)
        self.assertIn("meaningful scene or activity transition", assignment["scene_change_cues"][0])
        self.assertIn("Display text is unclear.", prompt)
        self.assertIn("owners alone decide", prompt)
        self.assertEqual(context["frames"][0]["frame_id"], "frame:0")
        self.assertEqual(context["active_reminders"], [])
        self.assertIsNone(context["active_focus"])
        self.assertEqual(output.summary, "Display is unreadable from this view.")
        self.assertEqual(document["perception_brief"], perception_brief_document(composition.brief))
        self.assertTrue(live_brief_parent_payload_ids(request))

    async def test_capture_receipt_cannot_silently_omit_brief(self):
        await self.refresh()
        request = self.bridge.attach(self.request())
        old_document = request_document(request)
        for key in live_brief_input_fields(request):
            old_document.pop(key)
        old_snapshot = ObservationInputSnapshot(request.owner_id, request.session_id, uuid4(), digest(old_document))
        with self.assertRaisesRegex(ValueError, "not_in_canonical_input"):
            seal_live_brief_input(request, old_snapshot)
        with self.assertRaisesRegex(ValueError, "not_in_canonical_input"):
            await FastPerceptionAnalyzer(self.model).analyze(request)
        self.model.generate_temporal_json.assert_not_awaited()

    async def test_post_capture_input_mutation_is_rejected_before_model(self):
        await self.refresh()
        request, _ = self.capture(self.bridge.attach(self.request()))
        mutated = replace(request, behavior_instructions="Changed after capture")
        with self.assertRaisesRegex(ValueError, "not_in_canonical_input"):
            await FastPerceptionAnalyzer(self.model).analyze(mutated)
        self.model.generate_temporal_json.assert_not_awaited()

    async def test_frame_path_neither_recomposes_nor_rebinds(self):
        await self.refresh()
        with patch("app.perception.live_brief.compose_perception_brief", side_effect=AssertionError("frame recomposition")):
            first = self.bridge.attach(self.request())
            second = self.bridge.attach(self.request())
        self.assertIs(first.perception_brief, second.perception_brief)
        self.bind.assert_awaited_once()

    async def test_revoked_dependency_is_omitted_without_inventing_resolution(self):
        await self.refresh()
        self.allowed.remove(self.focus)
        request = self.request()
        self.assertIs(self.bridge.attach(request), request)
        self.assertEqual(self.bridge.last_omission, "assignment_dependency_changed")
        self.assertEqual(self.check.lifecycle, CheckLifecycle.TRACKING)
        self.assertIsNone(self.check.resolution)

    async def test_cross_session_and_inexact_binding_are_rejected(self):
        await self.refresh()
        with self.assertRaisesRegex(ValueError, "request_scope_mismatch"):
            self.bridge.attach(self.request(session_id=uuid4()))
        bad = AsyncMock(return_value=GuardedPayloadBinding(scope=self.scope, payload_root_ids=(uuid4(),)))
        with self.assertRaisesRegex(ValueError, "binding_mismatch"):
            await self.refresh(bind=bad)
        self.assertIsNone(self.bridge.attach(self.request()).perception_brief)

    async def test_assignment_cannot_be_applied_to_an_earlier_frame(self):
        await self.refresh()
        request = self.request()
        earlier = replace(request, frames=(replace(request.primary_frame, captured_at=self.now - timedelta(seconds=1)),))
        self.assertIs(self.bridge.attach(earlier), earlier)
        self.assertEqual(self.bridge.last_omission, "assignment_frame_time_mismatch")

    async def test_queue_delay_past_expiry_blocks_model_without_refreshing_age(self):
        composition = await self.refresh()
        request, _ = self.capture(self.bridge.attach(self.request()))
        with patch("app.perception.live_brief._utcnow", return_value=composition.brief.expires_at):
            with self.assertRaisesRegex(ValueError, "expired_or_frame_time_mismatch"):
                await FastPerceptionAnalyzer(self.model).analyze(request)
        self.model.generate_temporal_json.assert_not_awaited()
        self.now = composition.brief.expires_at
        plain = self.request()
        self.assertIs(self.bridge.attach(plain), plain)

    async def test_new_owner_revision_changes_brief_without_resolving_check(self):
        first = await self.refresh()
        self.focus = self.focus.model_copy(update={"revision": 8})
        self.check = self.check.model_copy(update={"revision": 8, "canonical_owner_ref": self.focus})
        self.allowed = {self.episode, self.focus}
        second = await self.refresh()
        self.assertEqual(second.brief.id, first.brief.id)
        self.assertGreater(second.brief.version, first.brief.version)
        self.assertEqual(second.brief.questions[0].check_ref.revision, 8)
        self.assertEqual(self.check.lifecycle, CheckLifecycle.TRACKING)

    async def test_late_bind_cannot_replace_newer_assignment(self):
        waiting, release = asyncio.Event(), asyncio.Event()

        async def slow(brief):
            waiting.set()
            await release.wait()
            return GuardedPayloadBinding(scope=brief.scope, payload_root_ids=(uuid4(),), dependencies=brief.dependencies)

        old = asyncio.create_task(self.refresh(bind=slow))
        await waiting.wait()
        newest = await self.refresh(checks=())
        release.set()
        await old
        self.assertEqual(self.bridge.attach(self.request()).perception_brief, newest.brief)
        self.assertEqual(newest.brief.questions, ())

    async def test_closed_bridge_ignores_late_binding_and_never_owns_model_work(self):
        async def closing(brief):
            self.bridge.close()
            return GuardedPayloadBinding(scope=brief.scope, payload_root_ids=(uuid4(),), dependencies=brief.dependencies)
        await self.refresh(bind=closing)
        request = self.request()
        self.assertIs(self.bridge.attach(request), request)
        with self.assertRaises(RuntimeError):
            await self.refresh()
        self.model.generate_temporal_json.assert_not_awaited()

    async def test_brief_does_not_add_prospective_subject_authority(self):
        await self.refresh()
        request, _ = self.capture(self.bridge.attach(self.request()))
        self.model.generate_temporal_json.return_value = json.dumps({"summary": "Display visible", "starts_new_event": False,
            "confidence": 0.9, "prospective_matches": [{"subject_type": "visual_focus", "subject_id": str(self.focus.source_id),
            "confidence": 0.9, "observed_condition": "Readable", "supporting_frame_ids": ["frame:0"], "suggested_message": "Done"}]})
        output = await FastPerceptionAnalyzer(self.model).analyze(request)
        self.assertEqual(output.prospective_matches, [])
        self.assertEqual(self.check.lifecycle, CheckLifecycle.TRACKING)

    async def test_real_pipeline_keeps_one_analyzer_and_stores_observations(self):
        harness = _Harness(static=True)
        await harness.pipeline.close()
        self.scope = self.scope.model_copy(update={"session_id": harness.session_id})
        self.snapshot = self.snapshot.model_copy(update={"scope": self.scope})
        self.check = self.check.model_copy(update={"scope": self.scope})
        storage = _Storage(self.scope)
        guarded_effects = []

        @asynccontextmanager
        async def guard(brief, binding, *, effect_payload_ids=()):
            async with storage.transaction(storage.tasks, self.scope.owner_id) as connection:
                if not all(storage.available(root)
                           for root in (*binding.payload_root_ids, *effect_payload_ids)):
                    raise SourceError("perception_source_unavailable", "The fixture source was withdrawn.")
                self.assertEqual(brief, self.bridge.composition.brief)
                self.assertTrue(effect_payload_ids)
                guarded_effects.append(effect_payload_ids)
                yield connection

        self.bridge = LivePerceptionBriefBridge(scope=self.scope, scope_epoch=self.epoch,
            enabled=True, clock=lambda: self.now, result_guard=guard)

        def bind(brief):
            root = storage.original({"brief": perception_brief_document(brief)})
            return GuardedPayloadBinding(scope=brief.scope, payload_root_ids=(root,), dependencies=brief.dependencies)

        self.bind.side_effect = bind
        await self.refresh()
        captured = []

        async def insert(*, observation, frame_links, analysis_input=None):
            self.assertIsNotNone(analysis_input)
            async with storage.publisher.transaction(observation.owner_id) as connection:
                await storage.publisher.inputs.check(connection, analysis_input,
                    owner_id=observation.owner_id, session_id=observation.session_id,
                    frame_ids=tuple(link.visual_frame_id for link in frame_links))
            captured.append(storage.rows[analysis_input.payload_id]["content"]["request"])
            # Match the publication contract in this bounded storage fixture:
            # an observation original retains its captured input as a parent.
            original = storage.original({"observation_id": str(observation.id),
                "frame_links": [{"frame_id": str(link.visual_frame_id)} for link in frame_links]})
            storage.parents[original] = {analysis_input.payload_id}
            storage.observations[observation.id] = original
            return await harness.perception_repository.insert_observation(observation=observation, frame_links=frame_links)

        async def apply_observation(observation, *, effect_guard=None):
            self.assertIsNotNone(effect_guard)
            async with effect_guard():
                return await harness.working_memory_service.apply_observation(observation)

        def bridge_for(owner_id, session_id):
            self.assertEqual((owner_id, session_id), (self.scope.owner_id, self.scope.session_id))
            return self.bridge

        # Use the production attachment/capture/validation path. Only canonical
        # storage, the final observation repository, and the model are fakes.
        harness.pipeline = VisualAnalysisPipeline(settings=harness.pipeline._settings,
            repository=harness.visual_repository, session_repository=_CanonicalSessionRepository(harness.session_id),
            behavior_repository=_BehaviorRepository(), interrupt_service=harness.interrupt_service,
            model_limiter=PriorityAsyncLimiter(1), fast_analyzer=FastPerceptionAnalyzer(self.model),
            perception_repository=SimpleNamespace(insert_observation=insert, observation_publication=None),
            sensory_frame_window=harness.frame_window, domain_event_bus=harness.event_bus,
            working_memory_service=SimpleNamespace(apply_observation=apply_observation,
                get_session_working_memory=harness.working_memory_service.get_session_working_memory),
            live_brief_bridge_for=bridge_for)
        harness.pipeline.set_observation_publication(storage.publisher)
        try:
            for index in (1, 2):
                frame = harness.frame(sequence=index, captured_at=self.now)
                harness.put(frame, b"offline")
                await harness.pipeline.enqueue(frame=frame, image_bytes=b"offline")
                await harness.pipeline.drain()
            self.assertEqual(harness.visual_repository.failed, [])
            self.assertEqual(len(harness.perception_repository.inserted), 2)
            self.assertEqual(len(harness.working_memory_service.applied), 2)
            self.assertEqual(len(guarded_effects), 4)
            self.assertEqual({root for roots in guarded_effects for root in roots},
                             set(storage.observations.values()))
            self.assertEqual(self.model.generate_temporal_json.await_count, 2)
            self.bind.assert_awaited_once()
            self.assertEqual(captured[0]["perception_brief"], captured[1]["perception_brief"])
            self.assertEqual(harness.interrupt_service.calls, [])
            self.assertEqual(self.check.lifecycle, CheckLifecycle.TRACKING)
        finally:
            await harness.pipeline.close()


if __name__ == "__main__":
    unittest.main()
