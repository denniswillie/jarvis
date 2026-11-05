"""Prepared Core SQL/media case, never run without the existing explicit lane.

Uses inherited manifested SQL fixture and pinned real inspector. DiskStorage
simulates phone delivery; this is not a hosted TUS or physical capture test.
Run only the new named test, not inherited legacy cases, when allocated.
"""
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock
from uuid import UUID

from app.files.local_file import snapshot_file
from app.files.publication import file_id_for_operation
from app.files.storage import SignedRecordingUpload
import test_recording_ingress_sql as legacy


class DirectRecordingIngressSQLTests(legacy.RecordingIngressSQLTests):
    async def test_direct_object_adoption_exact_identity_no_backend_reupload(self):
        scope = await self.admitted()
        self.storage.sign_recording_upload = AsyncMock(side_effect=lambda path: SignedRecordingUpload(
            self.storage.bucket, path, "https://fixture.invalid/storage/v1/upload/resumable/sign",
            datetime.now(timezone.utc) + timedelta(hours=1), "fixture-direct-upload-token"))
        target = await self.ingress.create_upload_target(owner_id=self.f.owner, session_id=self.session,
            recording_id=scope.recording_id)
        expected = file_id_for_operation(self.f.owner, "recording:" + str(scope.recording_id))
        self.assertEqual(target["upload_id"], str(expected))
        self.assertEqual(target["max_bytes"], legacy.FIXTURE_LIMIT)
        # Explicitly stands in for the external Phone upload, before finalization.
        local = await snapshot_file(self.original, max_bytes=legacy.FIXTURE_LIMIT,
            expected_size=legacy.EXPECTED_SIZE, expected_sha256=legacy.EXPECTED_SHA256)
        try:
            self.storage.lose_ack = False
            await self.storage.upload_file(path=target["object_path"], local_file=local, content_type="video/mp4")
        finally:
            local.close()
        self.assertEqual(self.storage.uploads, 1)
        stopped = await self.admissions.stop(self.command("recordings.stop", {"recording_id": str(scope.recording_id)}))
        self.assertEqual(stopped.outcome, "capture_closed")
        result = await self.ingress.finalize_upload(owner_id=self.f.owner, session_id=self.session,
            recording_id=scope.recording_id, upload_id=UUID(target["upload_id"]), upload=self.upload)
        self.assertEqual(result["outcome"], "saved")
        self.assertEqual(result["file_id"], str(expected))
        self.assertIsNotNone(result["source_id"]); self.assertIsNotNone(result["version_id"])
        self.assertEqual(self.storage.uploads, 1, "Finalization must not upload/reupload bytes")
        replay = await self.ingress.finalize_upload(owner_id=self.f.owner, session_id=self.session,
            recording_id=scope.recording_id, upload_id=expected, upload=self.upload)
        self.assertEqual(replay["file_id"], result["file_id"])
        self.assertEqual(replay["version_id"], result["version_id"])
        self.assertEqual(self.storage.uploads, 1)
