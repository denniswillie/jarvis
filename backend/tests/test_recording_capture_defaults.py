"""Offline defaults and task-grant boundaries; no SQL, native media or providers."""
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock
from uuid import uuid4

from app.files.transfer import CAPTURE_MAX_BYTES, CAPTURE_MAX_DURATION_MS, FileManifest, FileTransferPolicy
from app.migrations.runner import load_migrations, render_fresh_schema
from app.recordings.models import RecordingScope, freeze_manifest, FINALIZATION_FORMAT
from app.recordings.voice import LeanRecordingDispatcher
from app.runtime.feature_config import ArtifactMediaConfiguration, MediaInspectorConfiguration, SecondBrainSettings
from app.sandbox.inputs import SandboxInputService, SavedFileInput
from app.tasks.media_editing import MediaInput
from app.tasks.models import TaskError


class CaptureDefaultsTests(unittest.TestCase):
    def test_finite_defaults_and_explicit_smaller_storage_grant(self):
        fields = dict(ffprobe_path="/fixture/ffprobe", ffprobe_sha256="a" * 64)
        self.assertEqual(MediaInspectorConfiguration(**fields).storage_max_file_bytes, 1024**3)
        self.assertEqual(SecondBrainSettings(_env_file=None).recording_storage_max_file_bytes, str(CAPTURE_MAX_BYTES))
        self.assertEqual(MediaInspectorConfiguration(**fields, storage_max_file_bytes=20_000_000).storage_max_file_bytes,
                         20_000_000)
        for invalid in (0, CAPTURE_MAX_BYTES + 1):
            with self.assertRaises(ValueError):
                MediaInspectorConfiguration(**fields, storage_max_file_bytes=invalid)
        with self.assertRaises(ValueError):
            ArtifactMediaConfiguration(**fields, storage_max_file_bytes=32 * 1024**2 + 1)

    def test_scope_accepts_five_minutes_but_immutable_smaller_manifest_bound_remains(self):
        fields = dict(recording_id=uuid4(), owner_id="fixture", session_id=uuid4(), area_id=uuid4(),
            include_jarvis_audio=False, admission_event_id=uuid4(), admission_payload_id=uuid4())
        scope = RecordingScope(**fields, max_bytes=CAPTURE_MAX_BYTES, max_duration_ms=CAPTURE_MAX_DURATION_MS)
        manifest = dict(format=FINALIZATION_FORMAT, size_bytes=CAPTURE_MAX_BYTES,
                        duration_ms=CAPTURE_MAX_DURATION_MS, content_sha256="a" * 64,
                        client_report={"timeline_end_ms": CAPTURE_MAX_DURATION_MS})
        self.assertEqual(freeze_manifest(scope, manifest).manifest["duration_ms"], 1_800_000)
        smaller = RecordingScope(**fields, max_bytes=20_000_000, max_duration_ms=60_000)
        for value in (manifest, manifest | dict(size_bytes=1, duration_ms=60_001)):
            with self.assertRaises(ValueError):
                freeze_manifest(smaller, value)
        for changes in (dict(max_bytes=CAPTURE_MAX_BYTES + 1), dict(max_duration_ms=1_800_001),
                        dict(max_duration_ms=0)):
            with self.assertRaises(ValueError):
                RecordingScope.model_validate(scope.model_dump() | changes)

    def test_native_offer_is_positive_finite_and_current(self):
        dispatch = LeanRecordingDispatcher(owner_id="fixture", session_id=uuid4(), admissions=None,
            policy=FileTransferPolicy("mp4_capture_v1", CAPTURE_MAX_BYTES),
            provider_session_key=lambda: "fixture", is_active=lambda: True, on_command=lambda _: True)
        offer = dict(type="recording.capabilities", version=1, generation_id=str(uuid4()), available=True,
                     include_jarvis_audio=True, max_bytes=CAPTURE_MAX_BYTES, max_duration_ms=CAPTURE_MAX_DURATION_MS)
        self.assertTrue(dispatch.configure_device(offer))
        for changes in (dict(max_bytes=0), dict(max_bytes=CAPTURE_MAX_BYTES + 1),
                        dict(max_duration_ms=1_800_001), dict(max_duration_ms=0), dict(max_duration_ms=True)):
            self.assertFalse(dispatch.configure_device(offer | changes))
            self.assertIsNone(dispatch.capabilities)

    def test_task_input_representation_accepts_capture_ceiling_only(self):
        fields = dict(file_id=uuid4(), source_id=uuid4(), version_id="fixture", receipt_event_id=uuid4(),
            owner_id="fixture", session_id=uuid4(), recording_id=uuid4(), area_id=uuid4(), source_payload_id=uuid4(),
            receipt_payload_id=uuid4(), admission_payload_id=uuid4(), content_sha256="a" * 64)
        self.assertEqual(MediaInput(**fields, size_bytes=CAPTURE_MAX_BYTES).size_bytes, CAPTURE_MAX_BYTES)
        with self.assertRaises(ValueError):
            MediaInput(**fields, size_bytes=CAPTURE_MAX_BYTES + 1)

    def test_forward_migration_and_canonical_mirror_preserve_old_grants(self):
        migrations = load_migrations()
        upgrade = next(item for item in migrations if item.version == 51)
        self.assertEqual(upgrade.name, "recording_capture_limits")
        self.assertIn("max_bytes between 1 and 268435456", upgrade.sql)
        self.assertIn("max_duration_ms between 1 and 300000", upgrade.sql)
        self.assertIn("'document_v1' and transfer_limit_bytes between 1 and 8388608", upgrade.sql)
        self.assertIn("'mp4_output_v1' and transfer_limit_bytes between 1 and 33554432", upgrade.sql)
        self.assertNotIn("update jarvis_", upgrade.sql)
        enlarged = next(item for item in migrations if item.version == 54)
        self.assertEqual(enlarged.name, "recording_thirty_minute_limits")
        self.assertIn("max_bytes between 1 and 1073741824", enlarged.sql)
        self.assertIn("max_duration_ms between 1 and 1800000", enlarged.sql)
        self.assertIn("'document_v1' and transfer_limit_bytes between 1 and 8388608", enlarged.sql)
        self.assertIn("'mp4_output_v1' and transfer_limit_bytes between 1 and 33554432", enlarged.sql)
        self.assertNotIn("update jarvis_", enlarged.sql)
        self.assertEqual(Path("app/schema.sql").read_text(), render_fresh_schema())


class CaptureTaskInputLimitTests(unittest.IsolatedAsyncioTestCase):
    async def test_smaller_task_grant_rejects_before_reading_original_bytes(self):
        manifest = FileManifest(size_bytes=20_000_001, content_sha256="a" * 64)
        file = SimpleNamespace(content_type="video/mp4", size_bytes=manifest.size_bytes, memory_source={
            "source_id": uuid4(), "revision": 1, "generation": 1,
            "content_digest": "a" * 64, "is_current": True})
        files = SimpleNamespace(get_file=AsyncMock(return_value=file), read_file=AsyncMock(),
            publications=SimpleNamespace(read_manifest=AsyncMock(return_value=(
                manifest, FileTransferPolicy("mp4_capture_v1", CAPTURE_MAX_BYTES)))))
        bridge = SimpleNamespace(context=SimpleNamespace(task=SimpleNamespace(owner_id="fixture")))
        inputs = SandboxInputService(bridge, None, files=files)
        with self.assertRaises(TaskError) as rejected:
            await inputs.saved_file(SavedFileInput(file_id=uuid4()), max_bytes=20_000_000)
        self.assertEqual(rejected.exception.code, "media_source_limit")
        files.read_file.assert_not_awaited()
