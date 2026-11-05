"""Real camera lease/session entry -> pipeline -> guarded evidence, offline SQL.

Provider output and storage are fixtures. The actual camera window, capture
binding, consumer, owner guards and canonical evidence ingress execute here.
"""
import asyncio
from dataclasses import replace
from datetime import timedelta
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

from app.knowledge.observation_inputs import ObservationInputSnapshot, request_document
from app.perception.evidence_session import SessionPerceptionEvidence
from app.perception.frame_window import RecentVisualFrameWindow, SensoryVisualFrame
from app.perception.live_brief import seal_live_brief_input
from app.runtime.concurrency import PriorityAsyncLimiter
from app.tasks.models import digest
from app.visual_memory.pipeline import VisualAnalysisPipeline
import tests.test_perception_evidence_consumer as fixtures


class PerceptionEvidenceSessionPipelineTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.f = fixtures.PerceptionEvidenceConsumerTests(
            "test_actual_owner_guard_and_ingress_preserve_exact_provenance_without_settlement")
        await self.f.asyncSetUp()
        f = self.f
        self.window = RecentVisualFrameWindow(max_frames_per_session=2, ttl_seconds=60, now=lambda:f.now)
        self.lease = self.window.begin_stream(f.scope.owner_id, f.scope.session_id, f.stream)
        self.sensory = []
        for frame in f.request.frames:
            image = SensoryVisualFrame(frame_id=frame.frame_id, owner_id=f.scope.owner_id,
                session_id=f.scope.session_id, audio_session_id=None, visual_session_id=f.stream,
                frame_sequence=frame.frame_sequence, source=frame.source, captured_at=frame.captured_at,
                server_received_at=f.now, content_type=frame.content_type, width=1, height=1,
                storage_bucket="fixture", storage_path="offline", raw_image_expires_at=f.now + timedelta(hours=1),
                image_bytes=frame.image_bytes)
            self.sensory.append(image)
            self.assertTrue(self.window.put(image, stream_lease=self.lease))
        self.base_request = replace(f.request, perception_brief=None, perception_brief_binding=None,
                                    perception_brief_input_digest=None)
        self.stages, self.diagnostics, self.analyzed = [], [], []
        self.during_analysis = None
        self.original_consumer = f.consumer

        def attach(request):
            self.stages.append("brief")
            return replace(request, perception_brief=f.brief, perception_brief_binding=f.binding)

        async def prepare(request, context):
            self.stages.append("prepare")
            self.assertIsNotNone(request.perception_brief)
            self.assertIsNone(request.perception_brief_input_digest)
            document = request_document(request)
            snapshot = ObservationInputSnapshot(request.owner_id, request.session_id, f.input_root, digest(document))
            f.db.copies[f.input_root]["content"] = {
                "format":"jarvis.visual_input.v1", "request_digest":snapshot.request_digest, "request":document}
            return seal_live_brief_input(request, snapshot), snapshot

        async def analyze(request):
            self.stages.append("analyze")
            self.analyzed.append(request)
            if self.during_analysis is not None:
                await self.during_analysis()
            return f.output

        f.publication.prepare_inputs = prepare
        self.repository = SimpleNamespace(mark_analysis_pending=AsyncMock(), mark_analysis_skipped=AsyncMock(),
            mark_analysis_failed=AsyncMock(), apply_analysis_result=AsyncMock())
        self.working = SimpleNamespace(apply_observation=AsyncMock())
        self.legacy_lookup = Mock(return_value=f.consumer)
        self.pipeline = VisualAnalysisPipeline(
            settings=SimpleNamespace(VISUAL_FAST_PERCEPTION_TIMEOUT_SECONDS=1, VISUAL_INTERRUPT_CONFIDENCE_THRESHOLD=.9),
            repository=self.repository, session_repository=object(), behavior_repository=object(),
            interrupt_service=SimpleNamespace(maybe_interrupt=AsyncMock()), model_limiter=PriorityAsyncLimiter(1),
            fast_analyzer=SimpleNamespace(provider="offline", model="fixture", analyze=AsyncMock(side_effect=analyze)),
            perception_repository=SimpleNamespace(insert_observation=AsyncMock()), sensory_frame_window=self.window,
            working_memory_service=self.working,
            live_brief_bridge_for=lambda owner, session:SimpleNamespace(attach=attach,
                effect_guard_for=lambda request, root:f.guard),
            perception_evidence_consumer_for=self.legacy_lookup)
        self.pipeline.set_observation_publication(f.publication)
        self.pipeline._context_builder.build = AsyncMock(return_value=object())
        self.pipeline._build_fast_request = lambda **kwargs:self.base_request
        self.frame = SimpleNamespace(owner_id=f.scope.owner_id, session_id=f.scope.session_id,
            id=f.request.primary_frame.frame_id, frame_sequence=f.request.primary_frame.frame_sequence)
        self.entry = self.new_entry()

    def new_entry(self):
        entry = SessionPerceptionEvidence(canonical=self.f.canonical, publication=self.f.publication,
            window=self.window, clock=lambda:self.f.now)
        capture = entry.capture
        def captured(request):
            self.stages.append("capture")
            self.assertIsNotNone(request.perception_brief)
            self.assertIsNone(request.perception_brief_input_digest)
            return capture(request)
        entry.capture = Mock(side_effect=captured)
        entry.activate = Mock(wraps=entry.activate)
        entry.close = Mock(wraps=entry.close)
        entry.consumer.consume = AsyncMock(wraps=entry.consumer.consume)
        return entry

    async def asyncTearDown(self):
        await self.pipeline.close()

    async def run_analysis(self):
        async def diagnostic(payload):
            self.diagnostics.append(payload)
        with patch("app.visual_memory.pipeline.observation_delivery_source", AsyncMock(return_value=self.f.original_root)), \
             patch("app.visual_memory.pipeline._observation_from_fast_output", return_value=self.f.observation):
            await self.pipeline.enqueue(frame=self.frame, image_bytes=b"offline", send_diagnostic=diagnostic)
            await self.pipeline.drain()
        self.repository.mark_analysis_failed.assert_not_awaited()
        self.working.apply_observation.assert_awaited_once()
        self.repository.apply_analysis_result.assert_awaited_once()
        self.pipeline._interrupt_service.maybe_interrupt.assert_not_awaited()

    def evidence_diagnostic(self):
        return next(item for item in self.diagnostics if item["type"] == "visual.perception_evidence")

    async def test_registration_captures_before_sealing_and_records_original_connection(self):
        self.pipeline.register_perception_evidence_session(self.entry)
        await self.run_analysis()
        self.assertEqual(self.stages, ["brief", "capture", "prepare", "analyze"])
        request, = self.analyzed
        self.assertEqual(request.camera_visual_session_id, self.lease.visual_session_id)
        self.assertEqual(request.camera_connection_id, self.lease.connection_id)
        self.assertEqual(request.perception_brief_input_digest, digest(request_document(request)))
        self.assertEqual(self.evidence_diagnostic()["status"], "recorded")
        row, = self.f.db.context_rows.values()
        content = self.f.db.copies[row["payload_id"]]["content"]
        self.assertEqual(content["camera_connection_id"], str(self.lease.connection_id))
        self.assertEqual(content["evidence"]["camera_stream_epoch"], str(self.lease.visual_session_id))
        self.entry.consumer.consume.assert_awaited_once()
        self.legacy_lookup.assert_not_called()

    async def test_publication_and_frame_window_properties_are_read_only(self):
        self.assertIs(self.pipeline.perception_publication, self.f.publication)
        self.assertIs(self.pipeline.perception_frame_window, self.window)
        with self.assertRaises(AttributeError):
            self.pipeline.perception_publication = object()
        with self.assertRaises(AttributeError):
            self.pipeline.perception_frame_window = object()
        self.pipeline.register_perception_evidence_session(self.entry)
        with self.assertRaises(RuntimeError):
            self.pipeline.set_observation_publication(object())

    async def test_duplicate_registration_and_stale_unregister_are_identity_protected(self):
        self.pipeline.register_perception_evidence_session(self.entry)
        self.pipeline.register_perception_evidence_session(self.entry)
        self.entry.activate.assert_called_once()
        replacement = self.new_entry()
        with self.assertRaises(RuntimeError):
            self.pipeline.register_perception_evidence_session(replacement)
        replacement.activate.assert_not_called()
        self.pipeline.unregister_perception_evidence_session(replacement)
        self.entry.close.assert_not_called()
        self.pipeline.unregister_perception_evidence_session(self.entry)
        self.entry.close.assert_called_once()
        self.pipeline.register_perception_evidence_session(replacement)
        self.pipeline.unregister_perception_evidence_session(self.entry)
        replacement.close.assert_not_called()
        await self.run_analysis()
        replacement.consumer.consume.assert_awaited_once()
        self.entry.consumer.consume.assert_not_awaited()
        self.assertEqual(self.evidence_diagnostic()["status"], "recorded")

    async def test_pending_result_cannot_be_redirected_to_replacement_session(self):
        self.pipeline.register_perception_evidence_session(self.entry)
        replacement = self.new_entry()
        async def replace_owner():
            self.pipeline.unregister_perception_evidence_session(self.entry)
            self.pipeline.register_perception_evidence_session(replacement)
        self.during_analysis = replace_owner
        await self.run_analysis()
        self.entry.consumer.consume.assert_awaited_once()
        replacement.consumer.consume.assert_not_awaited()
        replacement.capture.assert_not_called()
        self.assertEqual(self.evidence_diagnostic()["status"], "not_applied")
        self.assertFalse(self.f.db.context_rows)
        self.legacy_lookup.assert_not_called()

    async def test_new_frames_and_eviction_preserve_same_connection_evidence(self):
        self.pipeline.register_perception_evidence_session(self.entry)
        async def new_frames():
            for index in (3, 4):
                frame = replace(self.sensory[-1], frame_id=uuid4(), frame_sequence=index,
                    captured_at=self.f.now + timedelta(seconds=index), image_bytes=b"new scene")
                self.assertTrue(self.window.put(frame, stream_lease=self.lease))
            self.assertFalse(any(self.window.contains(frame) for frame in self.sensory))
        self.during_analysis = new_frames
        await self.run_analysis()
        self.assertEqual(self.evidence_diagnostic()["status"], "recorded")

    async def test_reused_visual_session_id_with_new_connection_rejects_old_analysis(self):
        self.pipeline.register_perception_evidence_session(self.entry)
        async def reconnect():
            new = self.window.begin_stream(self.f.scope.owner_id, self.f.scope.session_id, self.f.stream)
            self.assertNotEqual(new.connection_id, self.lease.connection_id)
            self.assertEqual(new.visual_session_id, self.lease.visual_session_id)
        self.during_analysis = reconnect
        await self.run_analysis()
        self.assertEqual(self.evidence_diagnostic()["code"], "perception_evidence_camera_stream_unavailable")
        self.assertFalse(self.f.db.context_rows)

    async def test_stopped_stream_and_closed_entry_reject_pending_evidence(self):
        self.pipeline.register_perception_evidence_session(self.entry)
        async def stop_stream():
            self.window.end_stream(self.lease)
            self.entry.close()
        self.during_analysis = stop_stream
        await self.run_analysis()
        self.assertEqual(self.evidence_diagnostic()["status"], "not_applied")
        self.assertFalse(self.f.db.context_rows)

    async def test_camera_callback_failure_cannot_fall_back_to_legacy_uuid_predicate(self):
        self.pipeline.register_perception_evidence_session(self.entry)
        self.entry.consumer.camera_is_current = Mock(return_value=False)
        legacy = Mock(return_value=True)
        self.entry.consumer.stream_is_current = legacy
        await self.run_analysis()
        self.entry.consumer.camera_is_current.assert_called()
        legacy.assert_not_called()
        self.assertEqual(self.evidence_diagnostic()["status"], "not_applied")

    async def test_captured_connection_requires_connection_aware_callback(self):
        self.pipeline.register_perception_evidence_session(self.entry)
        self.entry.consumer.camera_is_current = None
        self.entry.consumer.stream_is_current = Mock(return_value=True)
        await self.run_analysis()
        self.entry.consumer.stream_is_current.assert_not_called()
        self.assertEqual(self.evidence_diagnostic()["status"], "not_applied")

    async def test_empty_registry_preserves_legacy_callback_and_pins_its_original_consumer(self):
        self.original_consumer.consume = AsyncMock(wraps=self.original_consumer.consume)
        replacement = self.new_entry()
        async def register_during_analysis():
            self.pipeline.register_perception_evidence_session(replacement)
            self.legacy_lookup.return_value = replacement.consumer
        self.during_analysis = register_during_analysis
        await self.run_analysis()
        self.legacy_lookup.assert_called_once_with(self.f.scope.owner_id, self.f.scope.session_id)
        self.original_consumer.consume.assert_awaited_once()
        replacement.consumer.consume.assert_not_awaited()
        self.assertIsNone(self.analyzed[0].camera_connection_id)
        row, = self.f.db.context_rows.values()
        self.assertNotIn("camera_connection_id", self.f.db.copies[row["payload_id"]]["content"])

    async def test_pipeline_close_fences_registered_entry_and_refuses_reactivation(self):
        self.pipeline.register_perception_evidence_session(self.entry)
        await self.pipeline.close()
        self.entry.close.assert_called_once()
        self.assertFalse(self.entry.camera.is_current(self.f.request, self.f.stream))
        with self.assertRaises(RuntimeError):
            self.pipeline.register_perception_evidence_session(self.entry)
        with self.assertRaises(RuntimeError):
            self.entry.activate()
