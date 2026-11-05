from __future__ import annotations

import asyncio
import json
import unittest
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

from app.errors import JarvisHTTPError
from app.images import ImageReference, ImageUnavailableError
from app.memory.retrieval.evidence_repository import HistoricalImageEvidence
from app.memory.retrieval.image_evidence import HistoricalImageService
from test_current_image_service import _jpeg


NOW = datetime(2026, 9, 9, 12, tzinfo=UTC)


def _evidence(*, now=NOW, owner_id="owner", record_type="episode", record_id=None):
    return HistoricalImageEvidence(
        reference=ImageReference(
            owner_id=owner_id, source="historical", frame_id=uuid4(), session_id=uuid4(),
            captured_at=now - timedelta(hours=1), content_type="image/jpeg", width=48, height=32,
            raw_image_expires_at=now + timedelta(minutes=5),
            record_type=record_type, record_id=record_id or uuid4(),
        ),
        storage_bucket="private-frames", storage_path="private/evidence.jpg",
    )


def _service(repository, image_loader, *, now=lambda: NOW, timeout=1.0, max_frames=6):
    return HistoricalImageService(
        repository=repository, image_loader=image_loader,
        max_image_bytes=512_000, max_width=1024, max_height=1024,
        request_timeout_seconds=timeout, max_frames=max_frames, now=now,
    )


class HistoricalImageServiceTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.evidence = _evidence()
        self.reference = self.evidence.reference
        self.repository = SimpleNamespace(
            get_image_evidence=AsyncMock(return_value=self.evidence),
            list_image_evidence=AsyncMock(return_value=(self.evidence,)),
        )
        self.loader = SimpleNamespace(download_object=AsyncMock(return_value=_jpeg()))
        self.service = _service(self.repository, self.loader)

    async def select(self, **overrides):
        return await self.service.select_image(**{
            "owner_id": "owner", "record_id": self.reference.record_id,
            "record_type": self.reference.record_type, **overrides,
        })

    async def resolve(self, **overrides):
        return await self.service.resolve_image(**{
            "owner_id": "owner", "reference": self.reference, **overrides,
        })

    async def test_selection_is_metadata_only_and_preserves_original_time_and_session(self):
        reference = await self.select()
        self.assertEqual(reference, self.reference)
        self.assertLess(reference.captured_at, NOW)
        self.loader.download_object.assert_not_awaited()
        self.repository.get_image_evidence.assert_awaited_once_with(
            owner_id="owner", record_id=reference.record_id,
            record_type=reference.record_type, frame_id=None,
        )
        payload = json.dumps(reference.to_metadata())
        for private in ("private-frames", "private/evidence.jpg", "owner_id", "confidence"):
            self.assertNotIn(private, payload)

    async def test_resolution_rechecks_before_and_after_download_and_at_writer(self):
        image = await self.resolve()
        self.assertEqual(image.image_bytes, _jpeg())
        self.assertEqual(image.reference, self.reference)
        self.assertEqual(self.repository.get_image_evidence.await_count, 2)
        self.loader.download_object.assert_awaited_once_with(
            bucket=self.evidence.storage_bucket, path=self.evidence.storage_path, max_bytes=512_000,
        )
        await image.revalidate()
        self.assertEqual(self.repository.get_image_evidence.await_count, 3)
        self.assertNotIn("revalidate", repr(image))
        self.assertNotIn("image_bytes", repr(image))

    async def test_owner_and_record_identity_are_not_taken_on_trust(self):
        for reference in (
            replace(self.reference, owner_id="other"),
            replace(self.reference, record_id=uuid4()),
            replace(self.reference, record_type="semantic_fact"),
        ):
            with self.subTest(reference=reference), self.assertRaises(ImageUnavailableError) as caught:
                await self.resolve(reference=reference)
            self.assertEqual(caught.exception.code, "frame_unavailable")
        self.loader.download_object.assert_not_awaited()

    async def test_missing_foreign_forgotten_and_text_only_records_use_generic_unavailable(self):
        self.repository.get_image_evidence.return_value = None
        for owner_id, record_id in (("owner", uuid4()), ("other", uuid4()), ("owner", self.reference.record_id)):
            with self.subTest(owner_id=owner_id), self.assertRaises(ImageUnavailableError) as caught:
                await self.select(owner_id=owner_id, record_id=record_id)
            self.assertEqual(caught.exception.code, "frame_unavailable")
            self.assertNotIn(str(record_id), str(caught.exception))
        self.loader.download_object.assert_not_awaited()

    async def test_explicit_unavailable_frame_is_never_substituted(self):
        with self.assertRaises(ImageUnavailableError) as caught:
            await self.select(frame_id=uuid4())
        self.assertEqual(caught.exception.code, "frame_unavailable")
        self.loader.download_object.assert_not_awaited()

    async def test_replayed_reference_cannot_change_expiry_capture_or_dimensions(self):
        for change in (
            {"raw_image_expires_at": self.reference.raw_image_expires_at + timedelta(days=1)},
            {"captured_at": NOW}, {"width": 49}, {"session_id": uuid4()},
        ):
            with self.subTest(change=change), self.assertRaises(ImageUnavailableError):
                await self.resolve(reference=replace(self.reference, **change))
        self.loader.download_object.assert_not_awaited()

    async def test_forgetting_during_download_rejects_pixels(self):
        async def download(**kwargs):
            self.repository.get_image_evidence.return_value = None
            return _jpeg()

        self.loader.download_object.side_effect = download
        with self.assertRaises(ImageUnavailableError) as caught:
            await self.resolve()
        self.assertEqual(caught.exception.code, "frame_unavailable")
        self.assertEqual(self.repository.get_image_evidence.await_count, 2)

    async def test_changed_storage_relationship_after_download_is_rejected(self):
        self.repository.get_image_evidence.side_effect = [
            self.evidence, replace(self.evidence, storage_path="different/evidence.jpg"),
        ]
        with self.assertRaises(ImageUnavailableError) as caught:
            await self.resolve()
        self.assertEqual(caught.exception.code, "frame_unavailable")
        self.assertNotIn("different", str(caught.exception))

    async def test_expiry_during_download_is_rechecked(self):
        now = [NOW]
        self.service = _service(self.repository, self.loader, now=lambda: now[0])

        async def download(**kwargs):
            now[0] = self.reference.raw_image_expires_at
            return _jpeg()

        self.loader.download_object.side_effect = download
        with self.assertRaises(ImageUnavailableError) as caught:
            await self.resolve()
        self.assertEqual(caught.exception.code, "frame_unavailable")

    async def test_writer_revalidation_rejects_a_later_forgotten_record(self):
        image = await self.resolve()
        self.repository.get_image_evidence.return_value = None
        with self.assertRaises(ImageUnavailableError) as caught:
            await image.revalidate()
        self.assertEqual(caught.exception.code, "frame_unavailable")
        self.loader.download_object.assert_awaited_once()

    async def test_invalid_oversized_and_dimension_mismatched_pixels_fail(self):
        for content in (b"", b"not a JPEG", _jpeg(49, 32), _jpeg() + b"x" * 512_000):
            with self.subTest(size=len(content)):
                self.loader.download_object.return_value = content
                with self.assertRaises(ImageUnavailableError) as caught:
                    await self.resolve()
                self.assertEqual(caught.exception.code, "invalid_image")

    async def test_expired_or_oversized_metadata_is_rejected_before_download(self):
        for changes, code in (
            ({"raw_image_expires_at": NOW}, "frame_unavailable"),
            ({"width": 2048}, "invalid_image"),
        ):
            self.repository.get_image_evidence.return_value = replace(
                self.evidence, reference=replace(self.reference, **changes),
            )
            with self.subTest(changes=changes), self.assertRaises(ImageUnavailableError) as caught:
                await self.select()
            self.assertEqual(caught.exception.code, code)
        self.loader.download_object.assert_not_awaited()

    async def test_storage_errors_are_structured_and_do_not_leak_locators(self):
        for error, code in (
            (JarvisHTTPError(502, "visual_frame_storage_download_failed", "SECRET", details={"status_code": 404}), "frame_unavailable"),
            (JarvisHTTPError(502, "visual_frame_storage_download_too_large", "SECRET"), "invalid_image"),
            (RuntimeError("SECRET_STORAGE_URL"), "image_storage_failed"),
        ):
            self.loader.download_object.side_effect = error
            with self.subTest(code=code), self.assertRaises(ImageUnavailableError) as caught:
                await self.resolve()
            self.assertEqual(caught.exception.code, code)
            self.assertNotIn("SECRET", str(caught.exception))

    async def test_download_timeout_cancels_loader(self):
        cancelled = asyncio.Event()

        async def download(**kwargs):
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        self.service = _service(self.repository, self.loader, timeout=0.02)
        self.loader.download_object.side_effect = download
        with self.assertRaises(ImageUnavailableError) as caught:
            await self.resolve()
        self.assertEqual(caught.exception.code, "image_resolution_timeout")
        self.assertTrue(cancelled.is_set())

    async def test_cancellation_propagates_and_closes_loader_work(self):
        entered, closed = asyncio.Event(), asyncio.Event()

        async def download(**kwargs):
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                closed.set()

        self.loader.download_object.side_effect = download
        task = asyncio.create_task(self.resolve())
        await asyncio.wait_for(entered.wait(), 1)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertTrue(closed.is_set())

    async def test_metadata_failure_is_safe(self):
        self.repository.get_image_evidence.side_effect = RuntimeError("SECRET_DB_DETAIL")
        with self.assertRaises(ImageUnavailableError) as caught:
            await self.select()
        self.assertEqual(caught.exception.code, "image_lookup_failed")
        self.assertNotIn("SECRET", str(caught.exception))
        self.loader.download_object.assert_not_awaited()

    async def test_preview_bounds_unique_frames_but_preserves_shared_relationships(self):
        shared = replace(self.evidence, reference=replace(
            self.reference, record_id=uuid4(), record_type="semantic_fact",
        ))
        excess = replace(self.evidence, reference=replace(self.reference, frame_id=uuid4()))
        self.repository.list_image_evidence.return_value = (self.evidence, self.evidence, shared, excess)
        service = _service(self.repository, self.loader, max_frames=1)
        records = (
            (self.reference.record_type, self.reference.record_id),
            (shared.reference.record_type, shared.reference.record_id),
        )
        previews = await service.image_previews(owner_id="owner", records=records)
        self.assertEqual(set(previews), set(records))
        self.assertEqual({item["frame_id"] for refs in previews.values() for item in refs}, {str(self.reference.frame_id)})
        self.assertEqual([len(refs) for refs in previews.values()], [1, 1])
        self.repository.list_image_evidence.assert_awaited_once_with(owner_id="owner", records=records, limit=1)
        self.loader.download_object.assert_not_awaited()
        self.assertNotIn("private", json.dumps(list(previews.values())))

    async def test_storage_not_configured_has_no_fallback_or_download(self):
        service = _service(self.repository, None)
        with self.assertRaises(ImageUnavailableError):
            await service.select_image(owner_id="owner", record_id=self.reference.record_id, record_type="episode")
        self.assertEqual(await service.image_previews(owner_id="owner", records=(("episode", self.reference.record_id),)), {})
        self.repository.get_image_evidence.assert_not_awaited()
        self.repository.list_image_evidence.assert_not_awaited()
