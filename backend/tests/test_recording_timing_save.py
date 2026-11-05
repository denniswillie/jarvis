"""Save orchestration with fake storage/persistence; real media tested separately."""
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from app.files.local_file import _snapshot_chunks
from app.files.storage import JarvisFileStorageClient
from app.recordings.models import freeze_manifest
from app.recordings.saved_timing import configured_timing_options
from app.recordings.timing import RecordingTimingOptions, RecordingTimingResult
import test_recording_ingress as legacy


class TimingSaveTests(legacy.RecordingIngressTests):
    async def enable_timing(self):
        self.ingress.timing_options = RecordingTimingOptions(video_delay_ms=1000)
        self.storage = SimpleNamespace(bucket="files")
        self.storage.build_file_path = lambda **kw: JarvisFileStorageClient.build_file_path(self.storage, **kw)
        self.ingress.storage = self.storage
        self.upload_id, _, self.raw_path = self.ingress._upload_target(self.scope)
        self.objects = {self.raw_path: self.data}
        self.corrected_data = b"fixture corrected MP4; audio unchanged by the fake renderer"
        self.handles = []
        self.published = []
        self.operation_id = uuid4()

        async def snapshot(data):
            async def chunks():
                yield data
            result = await _snapshot_chunks(chunks(), max_bytes=1024)
            self.handles.append(result)
            return result

        async def read(**kw):
            self.assertEqual(kw["path"], self.raw_path)
            return await snapshot(self.objects[kw["path"]])

        async def delete(**kw):
            self.assertIsNotNone(self.saved, "Never remove raw upload before completion")
            self.assertIn(self.raw_path + "/synced", self.objects)
            self.assertEqual(kw, {"bucket": "files", "path": self.raw_path})
            self.objects.pop(self.raw_path, None)

        async def correct(original, options):
            self.assertEqual(options.video_delay_ms, 1000)
            local = await snapshot(self.corrected_data)
            return RecordingTimingResult("corrected", "validated", local, {
                "format": "jarvis.recording_timing.v1", "video_delay_ms": 1000,
                "original_sha256": original.content_sha256,
                "corrected_sha256": local.content_sha256, "audio_reencoded": False})

        async def inspect(local, policy):
            data = b"".join([part async for part in local.iter_chunks()])
            self.assertIn(data, (self.data, self.corrected_data))
            return self.inspection.model_copy(update={
                "content_sha256": local.content_sha256, "size_bytes": local.size_bytes,
                "duration_ms": 2000 if data == self.corrected_data else 1000})

        async def publish(**kw):
            data = b"".join([part async for part in kw["local_file"].iter_chunks()])
            manifest = kw["spec"].metadata["recording_original"]["finalization_manifest"]
            path = self.raw_path + ("/synced" if "timing_correction" in manifest else "")
            self.objects[path] = data
            self.published.append(manifest)
            self.saved = SimpleNamespace(model_dump=lambda **k: {"outcome": "saved", "recording_id": str(self.recording)})
            self.prior = {"id": self.operation_id, "phase": "available"}
            return SimpleNamespace(phase="available", operation_id=self.operation_id)

        self.storage.read_object_file = AsyncMock(side_effect=read)
        self.storage.delete_object = AsyncMock(side_effect=delete)
        self.ingress.inspect_file = AsyncMock(side_effect=inspect)
        self.publications.publish_file = AsyncMock(side_effect=publish)
        self.publications.adopt_recording_upload = AsyncMock(side_effect=publish)
        self.publications.reconcile = AsyncMock(return_value=SimpleNamespace(phase="available", operation_id=self.operation_id))
        self.renderer = patch("app.recordings.saved_timing.correct_recording_timing", side_effect=correct).start()
        self.addCleanup(patch.stopall)

    async def direct(self):
        return await self.ingress.finalize_upload(owner_id=self.owner, session_id=self.session,
            recording_id=self.recording, upload_id=self.upload_id, upload=self.upload)

    def assert_closed(self):
        self.assertTrue(all(not handle.path.exists() for handle in self.handles))

    async def test_corrected_only_and_retry_without_raw_upload(self):
        await self.enable_timing()
        first = await self.direct()
        self.assertEqual(first["timing_processing"]["status"], "corrected")
        self.assertEqual(self.objects, {self.raw_path + "/synced": self.corrected_data})
        manifest = self.frozen.manifest
        self.assertEqual(manifest["uploaded_original"]["content_sha256"], self.upload.content_sha256)
        self.assertEqual(manifest["client_report"]["timeline_end_ms"], 1000)
        self.assertEqual(manifest["duration_ms"], 2000)
        self.assertEqual((await self.direct())["outcome"], "saved")
        self.renderer.assert_awaited_once()
        self.publications.publish_file.assert_awaited_once()
        self.publications.adopt_recording_upload.assert_not_awaited()
        self.assert_closed()

    async def test_failed_correction_saves_original_once(self):
        await self.enable_timing()
        self.renderer.side_effect = None
        self.renderer.return_value = RecordingTimingResult("failed", "timeout")
        result = await self.direct()
        self.assertEqual(result["outcome"], "saved")
        self.assertEqual(result["timing_processing"], {"status": "failed", "reason": "timeout"})
        self.assertEqual(self.objects, {self.raw_path: self.data})
        self.storage.delete_object.assert_not_awaited()
        self.publications.adopt_recording_upload.assert_awaited_once()
        self.publications.publish_file.assert_not_awaited()
        self.assert_closed()

    async def test_corrected_inspection_failure_preserves_original(self):
        await self.enable_timing()
        self.ingress.inspect_file.side_effect = [self.inspection, ValueError("bad corrected file")]
        result = await self.direct()
        self.assertEqual(result["outcome"], "saved")
        self.assertEqual(self.objects, {self.raw_path: self.data})
        self.storage.delete_object.assert_not_awaited()
        self.assert_closed()

    async def test_unconfirmed_publication_retains_raw_until_reconcile(self):
        await self.enable_timing()
        publish = self.publications.publish_file.side_effect

        async def unknown(**kw):
            await publish(**kw)
            self.prior["phase"] = "outcome_unknown"
            return SimpleNamespace(phase="outcome_unknown", operation_id=self.operation_id)

        self.publications.publish_file.side_effect = unknown
        result = await self.direct()
        self.assertEqual(result["outcome"], "publication_unresolved")
        self.assertEqual(len(self.objects), 2)
        self.storage.delete_object.assert_not_awaited()
        self.assertEqual((await self.ingress.reconcile(owner_id=self.owner, session_id=self.session,
            recording_id=self.recording))["outcome"], "saved")
        self.assertEqual(self.objects, {self.raw_path + "/synced": self.corrected_data})
        self.renderer.assert_awaited_once()

    async def test_cleanup_failure_does_not_fail_save_and_retries(self):
        await self.enable_timing()
        delete = self.storage.delete_object.side_effect
        self.storage.delete_object.side_effect = TimeoutError()
        result = await self.direct()
        self.assertEqual(result["outcome"], "saved")
        self.assertTrue(result["temporary_upload_cleanup_pending"])
        self.assertEqual(len(self.objects), 2)
        self.storage.delete_object.side_effect = delete
        self.assertNotIn("temporary_upload_cleanup_pending", await self.direct())
        self.assertEqual(self.objects, {self.raw_path + "/synced": self.corrected_data})

    async def test_bound_retry_reuses_calibration_even_if_disabled_later(self):
        await self.enable_timing()
        publish = self.publications.publish_file.side_effect
        self.publications.publish_file.side_effect = RuntimeError("before publication")
        with self.assertRaises(RuntimeError):
            await self.direct()
        self.assertIn(self.raw_path, self.objects)
        self.ingress.timing_options = RecordingTimingOptions(video_delay_ms=0)
        self.publications.publish_file.side_effect = publish
        self.assertEqual((await self.direct())["outcome"], "saved")
        self.assertEqual(self.renderer.await_count, 2)
        self.assert_closed()

    async def test_correction_cannot_expand_capture_or_change_report(self):
        await self.enable_timing()
        await self.direct()
        for change in ({"video_delay_ms": 9000}, {"audio_reencoded": True}, {"original_sha256": "0" * 64}):
            manifest = self.frozen.manifest
            manifest["timing_correction"].update(change)
            with self.assertRaises(ValueError):
                freeze_manifest(self.scope, manifest)
        manifest = self.frozen.manifest
        manifest["client_report"]["timeline_end_ms"] = 2000
        with self.assertRaises(ValueError):
            freeze_manifest(self.scope, manifest)


class TimingConfigurationTests(unittest.TestCase):
    def test_opt_in_and_invalid_configuration(self):
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(configured_timing_options().video_delay_ms, 0)
        with patch.dict("os.environ", {"JARVIS_RECORDING_VIDEO_DELAY_MS": "1000"}, clear=True):
            self.assertEqual(configured_timing_options().video_delay_ms, 1000)
            self.assertEqual(configured_timing_options().timeout_seconds, 20)
        with patch.dict("os.environ", {"JARVIS_RECORDING_VIDEO_DELAY_MS": "invalid"}, clear=True):
            self.assertEqual(configured_timing_options().video_delay_ms, 0)
