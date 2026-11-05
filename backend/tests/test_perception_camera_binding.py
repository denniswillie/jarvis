"""Actual camera lease capture and session ownership, without model/DB/device I/O."""
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

from app.knowledge.observation_inputs import request_document
from app.live_state.models import SessionScope
from app.perception.contracts import FastPerceptionFrame, FastPerceptionRequest
from app.perception.evidence_session import CameraAssignmentBinding
from app.perception.frame_window import RecentVisualFrameWindow, SensoryVisualFrame
from app.realtime.live_state_bridge import SessionLiveStateBridge
from app.tasks.models import digest


class CameraAssignmentBindingTests(unittest.TestCase):
    def setUp(self):
        self.now = datetime.now(UTC)
        self.scope = SessionScope(owner_id="camera-evidence", session_id=uuid4())
        self.window = RecentVisualFrameWindow(max_frames_per_session=2, ttl_seconds=10,
            monotonic=lambda: 1.0, now=lambda: self.now)
        self.lease = self.window.begin_stream(self.scope.owner_id, self.scope.session_id, uuid4())
        self.active = True
        self.binding = CameraAssignmentBinding(self.window, self.scope, is_live=lambda: self.active)
        self.frame = self.put(1)

    def put(self, sequence):
        frame = SensoryVisualFrame(frame_id=uuid4(), owner_id=self.scope.owner_id,
            session_id=self.scope.session_id, audio_session_id=None,
            visual_session_id=self.lease.visual_session_id, frame_sequence=sequence, source="glasses",
            captured_at=self.now, server_received_at=self.now, content_type="image/jpeg",
            width=640, height=480, storage_bucket="fixture", storage_path="fixture",
            raw_image_expires_at=self.now + timedelta(seconds=30), image_bytes=b"fixture")
        self.assertTrue(self.window.put(frame, stream_lease=self.lease))
        return frame

    def request(self):
        return FastPerceptionRequest(owner_id=self.scope.owner_id, session_id=self.scope.session_id,
            window_id=uuid4(), frames=(FastPerceptionFrame(self.frame.frame_id, self.frame.frame_sequence,
                self.frame.captured_at, self.frame.source, self.frame.content_type, self.frame.image_bytes),),
            previous_working_state={}, existing_entity_ids=frozenset(), recent_audio_transcripts=(),
            behavior_instructions="", active_reminders=(), active_focus=None,
            recent_interrupts=(), requested_at=self.now)

    def test_capture_uses_real_socket_identity_and_canonical_input(self):
        original = self.request()
        captured = self.binding.capture(original)
        self.assertEqual(captured.camera_connection_id, self.lease.connection_id)
        self.assertEqual(captured.camera_visual_session_id, self.lease.visual_session_id)
        self.assertIsNone(original.camera_connection_id)
        self.assertTrue(self.binding.is_current(captured, self.lease.visual_session_id))
        self.assertEqual(request_document(captured)["camera_stream"], {
            "visual_session_id": str(self.lease.visual_session_id),
            "connection_id": str(self.lease.connection_id)})
        self.assertNotEqual(digest(request_document(original)), digest(request_document(captured)))

    def test_reconnect_same_visual_id_invalidates_old_analysis(self):
        captured = self.binding.capture(self.request())
        self.lease = self.window.begin_stream(self.scope.owner_id, self.scope.session_id,
                                             self.lease.visual_session_id)
        self.assertNotEqual(self.lease.connection_id, captured.camera_connection_id)
        self.assertFalse(self.binding.is_current(captured, self.lease.visual_session_id))
        self.assertIs(self.binding.capture(captured), captured)

    def test_new_frames_and_retention_eviction_do_not_invalidate_same_connection(self):
        captured = self.binding.capture(self.request())
        self.put(2)
        self.put(3)
        self.assertFalse(self.window.contains(self.frame))
        self.assertTrue(self.binding.is_current(captured, self.lease.visual_session_id))

    def test_stopping_stream_or_session_rejects_pending_evidence(self):
        captured = self.binding.capture(self.request())
        self.active = False
        self.assertFalse(self.binding.is_current(captured, self.lease.visual_session_id))
        self.active = True
        self.window.end_stream(self.lease)
        self.assertFalse(self.binding.is_current(captured, self.lease.visual_session_id))
        self.assertIsNone(self.binding.capture(self.request()).camera_connection_id)

    def test_unowned_frame_or_modified_pixels_cannot_receive_camera_proof(self):
        request = self.request()
        for frame in (replace(request.primary_frame, frame_id=uuid4()),
                      replace(request.primary_frame, image_bytes=b"modified"),
                      replace(request.primary_frame, captured_at=self.now - timedelta(seconds=1))):
            with self.subTest(frame=frame):
                changed = replace(request, frames=(frame,))
                self.assertIs(self.binding.capture(changed), changed)
                self.assertIsNone(changed.camera_connection_id)

    def test_foreign_scope_and_stream_do_not_pass(self):
        captured = self.binding.capture(self.request())
        self.assertFalse(self.binding.is_current(captured, uuid4()))
        for changes in ({"owner_id": "other"}, {"session_id": uuid4()}):
            with self.subTest(changes=changes):
                request = replace(self.request(), **changes)
                self.assertIs(self.binding.capture(request), request)
                self.assertFalse(self.binding.is_current(replace(captured, **changes), self.lease.visual_session_id))

    def test_paired_uuid_contract_and_fingerprint_reject_relabeling(self):
        original = self.request()
        for changes in ({"camera_connection_id": uuid4()},
                        {"camera_visual_session_id": uuid4()},
                        {"camera_connection_id": "invented", "camera_visual_session_id": uuid4()}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(original, **changes)
        captured = self.binding.capture(original)
        relabeled = replace(captured, camera_connection_id=uuid4())
        self.assertNotEqual(digest(request_document(captured)), digest(request_document(relabeled)))
        self.assertFalse(self.binding.is_current(relabeled, self.lease.visual_session_id))

    def test_default_request_has_no_extra_canonical_fields(self):
        request = self.request()
        self.assertNotIn("camera_stream", request_document(request))
        self.assertFalse(self.binding.is_current(request, self.lease.visual_session_id))


class EvidenceAttachmentLifecycleTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.scope = SessionScope(owner_id="evidence-attachment", session_id=uuid4())
        self.brief = SimpleNamespace(scope=self.scope)
        self.live = SimpleNamespace(perception_brief_bridge=self.brief, close=AsyncMock())
        self.session = SimpleNamespace(close=AsyncMock())
        self.route = SimpleNamespace(request=SimpleNamespace(route=SimpleNamespace(scope=self.scope)))
        self.pipeline = SimpleNamespace(register_live_brief=Mock(), unregister_live_brief=Mock(),
            register_perception_evidence_session=Mock(), unregister_perception_evidence_session=Mock())
        self.evidence = SimpleNamespace(activate=Mock(), close=Mock())

    def attachment(self):
        with patch("app.realtime.live_state_bridge.create_session_perception_evidence", return_value=self.evidence):
            return SessionLiveStateBridge(self.session, self.live, self.route, visual_pipeline=self.pipeline)

    async def test_registration_and_close_follow_the_admitted_session(self):
        attachment = self.attachment()
        self.evidence.activate.assert_not_called()
        attachment.registered()
        self.pipeline.register_perception_evidence_session.assert_called_once_with(self.evidence)
        self.evidence.activate.assert_called_once()
        attachment.registered()
        self.evidence.activate.assert_called_once()
        await attachment.close()
        self.evidence.close.assert_called_once()
        self.pipeline.unregister_perception_evidence_session.assert_called_once_with(self.evidence)
        self.pipeline.unregister_live_brief.assert_called_once_with(self.brief)
        self.live.close.assert_awaited_once()

    async def test_partial_registration_and_pre_admission_close_never_leave_owner_active(self):
        self.pipeline.register_perception_evidence_session.side_effect = RuntimeError("owner occupied")
        attachment = self.attachment()
        with self.assertRaisesRegex(RuntimeError, "owner occupied"):
            attachment.registered()
        self.evidence.activate.assert_not_called()
        await attachment.close()
        self.evidence.close.assert_called_once()
        self.pipeline.unregister_live_brief.assert_called_once_with(self.brief)
        self.pipeline.unregister_perception_evidence_session.assert_not_called()
        with self.assertRaises(RuntimeError):
            attachment.registered()
