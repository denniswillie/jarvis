import asyncio
import json
from datetime import UTC, datetime
import unittest
from uuid import uuid4

from app.images import ImageReference
from app.tasks.context_capture import SessionContextCapture
from test_second_brain_capture import CaptureRepository
from test_realtime_visual_delivery import VisualDeliveryTests
from test_gemini_image_delivery import GeminiImageDeliveryTests


class CanonicalImageCaptureTests(unittest.IsolatedAsyncioTestCase):
    def make_capture(self, provider="openai", *, owner="fixture", session_id=None):
        repository = CaptureRepository()
        capture = SessionContextCapture(
            repository, owner_id=owner, session_id=session_id or uuid4(),
            provider=provider, audio_session_id="fixture-audio",
        )
        self.addAsyncCleanup(capture.close)
        return capture, repository

    async def test_transport_evidence_is_metadata_only_and_ack_cannot_regress(self):
        capture, repository = self.make_capture()
        reference = ImageReference(
            frame_id=uuid4(), owner_id=capture.owner_id, session_id=uuid4(),
            source="historical", captured_at=datetime(2026, 9, 12, tzinfo=UTC),
            content_type="image/jpeg", width=48, height=32,
        )
        capture.image_delivery(reference, "provider_acknowledged", "image-1")
        capture.image_delivery(reference, "socket_written", "image-1")
        capture.image_delivery(reference, "write_outcome_unknown", "image-1")
        await capture.flush()
        row = repository.of_kind("observation")[0]
        self.assertEqual(row.content["delivery_state"], "provider_acknowledged")
        self.assertEqual(row.content["model_understanding"], "unknown")
        self.assertEqual(row.content["images"], [reference.to_metadata()])
        event = repository.events[0]
        self.assertEqual(event.source_time, reference.captured_at)
        self.assertEqual(row.content["source_time_basis"], "image_capture")
        for forbidden in ("image_bytes", "storage_path", "base64", "signed_url"):
            self.assertNotIn(forbidden, json.dumps(row.content))
        self.assertEqual(len(repository.of_kind("observation")), 1)

    async def test_gemini_does_not_accept_a_fabricated_provider_ack(self):
        capture, repository = self.make_capture("gemini")
        capture.image_delivery(None, "provider_acknowledged", "image-1")
        await capture.flush()
        self.assertEqual(capture.issue, "context_image_evidence_invalid")
        self.assertEqual(repository.of_kind("observation"), [])

    async def test_unavailable_reference_is_an_explicit_gap(self):
        capture, repository = self.make_capture()
        capture.image_delivery(None, "socket_written", "image-1")
        await capture.flush()
        row = repository.of_kind("observation")[0]
        self.assertEqual(row.content["images"], [])
        self.assertEqual(row.content["evidence_gap"], "image_reference_unavailable")
        self.assertFalse(row.content["provider_acknowledged"])

    async def test_gpt_actual_image_writer_captures_source_before_an_early_ack(self):
        harness = VisualDeliveryTests()
        await harness.asyncSetUp()
        self.addAsyncCleanup(harness.asyncTearDown)
        capture, repository = self.make_capture(owner=harness.key.owner_id, session_id=harness.key.session_id)
        harness.coordinator._on_image_delivery = capture.image_delivery
        frame = harness.offer(1)
        await harness.cycle()
        await capture.flush()
        observation = repository.of_kind("observation")[0].content
        self.assertEqual(observation["images"][0]["frame_id"], str(frame.frame_id))
        self.assertEqual(observation["images"][0]["stream_connection_id"], str(harness.stream_lease.connection_id))
        self.assertEqual(observation["delivery_state"], "provider_acknowledged")
        self.assertEqual(len(harness.creates()), 1)
        self.assertFalse(any(e["type"] == "response.create" for e in harness.provider.events))

    async def test_gpt_unknown_write_and_late_ack_keep_exact_frame_in_distinct_receipts(self):
        harness = VisualDeliveryTests()
        await harness.asyncSetUp()
        self.addAsyncCleanup(harness.asyncTearDown)
        capture, repository = self.make_capture(owner=harness.key.owner_id, session_id=harness.key.session_id)
        harness.coordinator._on_image_delivery = capture.image_delivery
        harness.provider.after_write = asyncio.Event()
        frame = harness.offer(1)
        await harness.cycle()
        await capture.flush()
        before = repository.of_kind("observation")[0]
        self.assertEqual(before.content["delivery_state"], "write_outcome_unknown")
        harness.provider.after_write = None
        harness.provider.ack(harness.creates()[0])
        await capture.flush()
        observations = repository.of_kind("observation")
        self.assertEqual(len(observations), 2)
        after = observations[1]
        self.assertNotEqual(before.source_event_id, after.source_event_id)
        self.assertEqual(before.revision, 1)
        self.assertEqual(after.revision, 1)
        self.assertEqual(before.content["delivery_state"], "write_outcome_unknown")
        self.assertEqual(before.content["images"], after.content["images"])
        self.assertEqual(after.content["delivery_state"], "provider_acknowledged")
        self.assertEqual(after.content["images"][0]["frame_id"], str(frame.frame_id))

    async def test_gemini_automatic_image_write_records_no_native_ack(self):
        harness = GeminiImageDeliveryTests()
        await harness.asyncSetUp()
        self.addAsyncCleanup(harness.asyncTearDown)
        capture, repository = self.make_capture(
            "gemini", owner=harness.context.owner_id, session_id=harness.context.jarvis_session_id,
        )
        harness.session._context_capture = capture
        await harness.turn()
        await capture.flush()
        observation = repository.of_kind("observation")[0].content
        self.assertEqual(observation["images"][0]["frame_id"], str(harness.frame.frame_id))
        self.assertEqual(observation["delivery_state"], "socket_written")
        self.assertFalse(observation["provider_acknowledged"])
        self.assertEqual(observation["delivery_id_basis"], "application_image_write")

    async def test_gemini_explicit_image_tool_also_records_its_actual_write(self):
        harness = GeminiImageDeliveryTests()
        await harness.asyncSetUp()
        self.addAsyncCleanup(harness.asyncTearDown)
        harness.settings.VISUAL_REALTIME_ENABLED = False
        capture, repository = self.make_capture(
            "gemini", owner=harness.context.owner_id, session_id=harness.context.jarvis_session_id,
        )
        harness.session._context_capture = capture
        await harness.turn()
        await harness.request("get_current_camera_view", {})
        await capture.flush()
        observation = repository.of_kind("observation")[0].content
        self.assertEqual(observation["images"][0]["frame_id"], str(harness.frame.frame_id))
        self.assertEqual(observation["delivery_state"], "socket_written")
        self.assertFalse(observation["provider_acknowledged"])
