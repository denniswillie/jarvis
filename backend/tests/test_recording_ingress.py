import hashlib
from contextlib import asynccontextmanager
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from app.files.local_file import LocalFileError
from app.files.recording_ingress import RecordingIngress, RecordingUpload
from app.files.transfer import FileTransferPolicy, MP4Inspection
from app.knowledge.models import SourceError
from app.recordings.models import RecordingScope, freeze_manifest


class RecordingIngressTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.data = b"synthetic unit bytes; inspector explicitly substituted"
        self.owner, self.session, self.recording = "recording-unit-owner", uuid4(), uuid4()
        self.scope = RecordingScope(recording_id=self.recording, owner_id=self.owner, session_id=self.session,
            area_id=uuid4(), include_jarvis_audio=True, max_bytes=1024, max_duration_ms=60_000,
            admission_event_id=uuid4(), admission_payload_id=uuid4())
        self.upload = RecordingUpload(filename="original.mp4", description="Client recording",
            size_bytes=len(self.data), content_sha256=hashlib.sha256(self.data).hexdigest(), client_report={
                "generation_id": str(uuid4()), "timeline_end_ms": 1000, "video_frames": 24,
                "source_audio_frames": 47040, "padded_audio_frames": 960,
                "missing_intervals": [{"track": "audio", "start_us": 0, "end_us": 20000,
                                       "reason": "missingInput"}]})
        self.inspection = MP4Inspection(content_sha256=self.upload.content_sha256, size_bytes=len(self.data),
            duration_ms=1000, width=320, height=240, video_codec="h264", audio_codec="aac")
        self.frozen, self.prior, self.saved = None, None, None
        self.files_seen, self.bodies_read = [], 0
        self.connection = SimpleNamespace(fetchrow=AsyncMock(side_effect=lambda *a: self.prior))

        @asynccontextmanager
        async def transaction(owner):
            self.assertEqual(owner, self.owner)
            yield self.connection
        self.sources = SimpleNamespace(transaction=transaction)
        self.admissions = SimpleNamespace(sources=self.sources, tasks=SimpleNamespace(lock_owner=AsyncMock()),
            _authorize=lambda owner: self.assertEqual(owner, self.owner),
            require_admitted_recording=AsyncMock(return_value=self.scope),
            get_finalization=AsyncMock(side_effect=lambda *a: self.frozen),
            bind_finalization=AsyncMock(side_effect=self.bind),
            get_completion=AsyncMock(side_effect=lambda *a: self.saved))
        self.inspector = AsyncMock(side_effect=self.inspect)
        self.policy = FileTransferPolicy("mp4_capture_v1", 1024)
        self.ingress = RecordingIngress(self.sources, object(), self.admissions,
            policy=self.policy, inspect_file=self.inspector)
        self.publications = SimpleNamespace(publish_file=AsyncMock(side_effect=self.publish),
            reconcile=AsyncMock(return_value=SimpleNamespace(phase="outcome_unknown", operation_id=uuid4())))
        self.adapter = SimpleNamespace(media_policy=self.policy, publications=self.publications)
        self.patch = patch("app.files.recording_ingress._RecordingFiles", return_value=self.adapter)
        self.patch.start(); self.addCleanup(self.patch.stop)

    def bind(self, connection, scope, manifest):
        self.assertIs(connection, self.connection)
        candidate = freeze_manifest(scope, manifest)
        if self.frozen is not None and candidate != self.frozen:
            raise SourceError("recording_finalization_conflict", "Different exact data")
        self.frozen = candidate
        return candidate

    async def inspect(self, local, policy):
        self.files_seen.append(local)
        self.assertEqual(b"".join([b async for b in local.iter_chunks()]), self.data)
        return self.inspection

    async def publish(self, **kwargs):
        self.assertEqual(kwargs["operation_key"], "recording:" + str(self.recording))
        self.assertEqual(kwargs["source_payload_ids"], (self.scope.admission_payload_id,))
        self.assertEqual(kwargs["spec"].metadata["recording_original"]["finalization_manifest"], self.frozen.manifest)
        self.assertEqual(kwargs["local_file"].content_sha256, self.upload.content_sha256)
        self.saved = SimpleNamespace(model_dump=lambda **k: {"outcome": "saved", "recording_id": str(self.recording)})
        return SimpleNamespace(phase="available", operation_id=uuid4())

    async def chunks(self, data=None):
        self.bodies_read += 1
        content = self.data if data is None else data
        yield content[:7]
        yield content[7:]

    async def finalize(self, upload=None, data=None):
        return await self.ingress.finalize(owner_id=self.owner, session_id=self.session,
            recording_id=self.recording, upload=upload or self.upload, chunks=self.chunks(data))

    def freeze(self):
        self.frozen = freeze_manifest(self.scope, {"format": "jarvis.recording_finalization.v1",
            **self.upload.model_dump(mode="json"), "duration_ms": self.inspection.duration_ms,
            "media_inspection": self.inspection.model_dump(mode="json")})

    def assert_private_files_closed(self):
        for local in self.files_seen:
            self.assertFalse(local.path.exists())
            with self.assertRaises(LocalFileError):
                local._check()

    async def test_exact_stream_inspection_frozen_binding_and_existing_publication(self):
        result = await self.finalize()
        self.assertEqual(result["outcome"], "saved")
        self.assertEqual(self.bodies_read, 1)
        self.publications.publish_file.assert_awaited_once()
        self.assertEqual(self.frozen.manifest["client_report"], self.upload.client_report.model_dump(mode="json"))
        self.assert_private_files_closed()

    async def test_unknown_effect_reconciles_same_identity_without_reading_or_uploading(self):
        self.freeze(); self.prior = {"id": uuid4(), "phase": "outcome_unknown"}
        result = await self.finalize()
        self.assertEqual(result["outcome"], "publication_unresolved")
        self.assertEqual(self.bodies_read, 0)
        self.inspector.assert_not_awaited(); self.publications.publish_file.assert_not_awaited()
        self.publications.reconcile.assert_awaited_once_with(owner_id=self.owner, operation_id=self.prior["id"])

    async def test_replay_cannot_replace_metadata_even_when_original_upload_was_unknown(self):
        self.freeze(); self.prior = {"id": uuid4(), "phase": "outcome_unknown"}
        with self.assertRaises(SourceError):
            await self.finalize(self.upload.model_copy(update={"description": "Different intent"}))
        self.assertEqual(self.bodies_read, 0)
        self.publications.reconcile.assert_not_awaited()

    async def test_changed_hash_or_truncated_body_never_reaches_inspection_or_binding(self):
        for bad in (self.data[:-1], b"x" * len(self.data), self.data + b"x"):
            with self.subTest(length=len(bad)), self.assertRaises(LocalFileError):
                await self.finalize(data=bad)
        self.inspector.assert_not_awaited(); self.admissions.bind_finalization.assert_not_awaited()
        self.publications.publish_file.assert_not_awaited()

    async def test_authority_changed_during_inspection_prevents_publication_and_closes_bytes(self):
        self.admissions.require_admitted_recording.side_effect = [self.scope, SourceError("recording_revoked", "Revoked", 403)]
        with self.assertRaises(SourceError):
            await self.finalize()
        self.publications.publish_file.assert_not_awaited(); self.assert_private_files_closed()

    async def test_requested_audio_cannot_be_replaced_with_silent_movie(self):
        self.inspection = self.inspection.model_copy(update={"audio_codec": None})
        with self.assertRaises(SourceError):
            await self.finalize()
        self.admissions.bind_finalization.assert_not_awaited(); self.assert_private_files_closed()

    async def test_current_deployment_cap_can_prevent_new_upload_without_rewriting_admission(self):
        self.ingress.policy = FileTransferPolicy("mp4_capture_v1", len(self.data) - 1)
        with self.assertRaises(SourceError):
            await self.finalize()
        self.assertEqual(self.bodies_read, 0)
        self.assertEqual(self.scope.max_bytes, 1024)

    async def test_publication_failure_closes_private_snapshot_and_preserves_bound_identity(self):
        self.publications.publish_file.side_effect = RuntimeError("Synthetic post-binding failure")
        with self.assertRaises(RuntimeError):
            await self.finalize()
        self.assertIsNotNone(self.frozen)
        self.assert_private_files_closed()

    async def test_available_publication_without_atomic_recording_receipt_is_not_saved(self):
        self.freeze(); self.prior = {"id": uuid4(), "phase": "available"}
        self.publications.reconcile.return_value = SimpleNamespace(phase="available", operation_id=self.prior["id"])
        with self.assertRaises(SourceError):
            await self.finalize()
        self.assertEqual(self.bodies_read, 0)


if __name__ == "__main__":
    unittest.main()
