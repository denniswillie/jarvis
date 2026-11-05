import unittest
from dataclasses import replace
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

from app.images import ImageUnavailableError, ResolvedImage
from app.perception.current_image import CurrentImageService
from app.perception.frame_window import RecentVisualFrameWindow, SensoryVisualFrame
from app.tasks.context_images import TaskContextImages
from app.tasks.models import ClaimLost
from test_current_image_service import _Clock, _jpeg
from test_second_brain_context_window import source, claim


class TaskContextImageTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.clock = _Clock()
        self.claim = claim()
        self.claim.task.session_id = uuid4()
        self.window = RecentVisualFrameWindow(
            max_frames_per_session=4, ttl_seconds=30,
            now=self.clock.now, monotonic=self.clock.monotonic,
        )
        self.lease = self.window.begin_stream(self.claim.task.owner_id, self.claim.task.session_id, uuid4())
        self.current = CurrentImageService(
            frame_window=self.window, max_age_seconds=10, max_image_bytes=4096,
            max_width=1024, max_height=1024, now=self.clock.now, monotonic=self.clock.monotonic,
        )
        self.frame = SensoryVisualFrame(
            frame_id=uuid4(), owner_id=self.claim.task.owner_id, session_id=self.claim.task.session_id,
            audio_session_id=None, visual_session_id=self.lease.visual_session_id, frame_sequence=1,
            source="glasses", captured_at=self.clock.wall, server_received_at=self.clock.wall,
            content_type="image/jpeg", width=48, height=32, storage_bucket="fixture",
            storage_path="fixture/image.jpg", raw_image_expires_at=self.clock.wall + timedelta(seconds=60),
            image_bytes=_jpeg(),
        )
        self.window.put(self.frame, stream_lease=self.lease)
        self.reference = self.current.select_current_image(
            owner_id=self.claim.task.owner_id, session_id=self.claim.task.session_id,
        )
        self.record = source(
            1, kind="tool_outcome", trust="tool_receipt",
            content={"result": {"status": "success", "image_status": "selected", "image": self.reference.to_metadata()}},
        )
        self.record = self.record.model_copy(update={"session_id": self.claim.task.session_id})
        self.repository = AsyncMock()
        self.repository.source_for_claim.return_value = self.record
        self.images = TaskContextImages(self.repository, current=self.current)

    async def test_exact_original_pixels_are_transient_and_revalidated_before_transport(self):
        selection = await self.images.select(self.claim, self.record.id)
        result = await self.images.resolve(self.claim, selection)
        self.assertEqual(result.image_bytes, self.frame.image_bytes)
        self.assertEqual(result.reference, self.reference)
        self.assertNotIn("image_bytes", selection)
        await result.revalidate()
        self.clock.advance(11)
        with self.assertRaises(ImageUnavailableError) as caught:
            await result.revalidate()
        self.assertEqual(caught.exception.code, "stale_frame")

    async def test_source_removal_or_claim_loss_blocks_queued_pixels(self):
        selection = await self.images.select(self.claim, self.record.id)
        result = await self.images.resolve(self.claim, selection)
        self.repository.source_for_claim.side_effect = ClaimLost()
        with self.assertRaises(ClaimLost):
            await result.revalidate()

    async def test_new_stream_never_silently_substitutes_another_frame(self):
        selection = await self.images.select(self.claim, self.record.id)
        self.window.begin_stream(self.claim.task.owner_id, self.claim.task.session_id, self.lease.visual_session_id)
        with self.assertRaises(ImageUnavailableError) as caught:
            await self.images.resolve(self.claim, selection)
        self.assertEqual(caught.exception.code, "source_changed")

    async def test_model_provided_image_metadata_cannot_replace_selected_evidence(self):
        selection = await self.images.select(self.claim, self.record.id)
        selection["image"]["frame_id"] = str(uuid4())
        with self.assertRaises(ImageUnavailableError):
            await self.images.resolve(self.claim, selection)

    async def test_description_or_user_claim_is_not_original_image_evidence(self):
        for changes in (
            {"kind": "user_transcript", "trust": "user_input"},
            {"content": {"description": "A code editor is visible."}},
            {"content": None, "availability": "expired"},
        ):
            self.repository.source_for_claim.return_value = self.record.model_copy(update=changes)
            with self.assertRaises(ImageUnavailableError):
                await self.images.select(self.claim, self.record.id)

    async def test_current_image_must_match_its_authorized_source_session(self):
        self.repository.source_for_claim.return_value = self.record.model_copy(update={"session_id": uuid4()})
        with self.assertRaises(ImageUnavailableError):
            await self.images.select(self.claim, self.record.id)

    async def test_attached_followup_session_pixels_do_not_become_origin_session_pixels(self):
        # The repository has already admitted this source from a later session.
        self.claim.task.session_id = uuid4()
        selection = await self.images.select(self.claim, self.record.id)
        resolved = await self.images.resolve(self.claim, selection)
        self.assertEqual(resolved.reference.session_id, self.record.session_id)
        self.assertNotEqual(resolved.reference.session_id, self.claim.task.session_id)

    async def test_historical_pixels_keep_historical_provenance_and_policy_recheck(self):
        historical_ref = replace(
            self.reference, source="historical", session_id=uuid4(),
            record_type="experience_event", record_id=uuid4(),
        )
        self.repository.source_for_claim.return_value = self.record.model_copy(update={
            "content": {"result": {"status": "success", "image_status": "selected", "image": historical_ref.to_metadata()}},
        })
        policy = AsyncMock()
        historical = SimpleNamespace(resolve_image=AsyncMock(return_value=ResolvedImage(
            reference=historical_ref, image_bytes=b"original-historical-fixture", revalidate=policy,
        )))
        images = TaskContextImages(self.repository, current=self.current, historical=historical)
        selection = await images.select(self.claim, self.record.id)
        result = await images.resolve(self.claim, selection)
        self.assertEqual(result.reference.source, "historical")
        self.assertEqual(result.reference.session_id, historical_ref.session_id)
        self.assertEqual(result.image_bytes, b"original-historical-fixture")
        policy.assert_awaited()
        policy.side_effect = ImageUnavailableError("frame_unavailable", "Fixture deleted")
        with self.assertRaises(ImageUnavailableError):
            await result.revalidate()
