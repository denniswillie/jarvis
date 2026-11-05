"""Opt-in composition checks; no model, database, storage or actual probe run."""
import hashlib
import json
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from app.files.generated import ArtifactPolicy
from app.files.local_file import snapshot_file
from app.files.media_inspection import MediaInspectionError
from app.files.media_runtime import prepare_artifact_media
from app.files.transfer import FileTransferPolicy
from app.runtime.feature_config import ArtifactConfiguration, ArtifactMediaConfiguration, Capability, SecondBrainSettings


def flags(**overrides):
    values = dict(_env_file=None, enabled="true", artifacts_enabled="true", sandbox_enabled="true",
        knowledge_enabled="true", hindsight_api_key="fixture-configured-key",
        hindsight_url="https://memory.example.com", knowledge_area_id=str(uuid4()))
    return SecondBrainSettings(**(values | overrides))


class ArtifactMediaFlagTests(unittest.TestCase):
    def resolve(self, **values):
        return flags(**values).resolve(Capability.ARTIFACTS, database_url="postgresql://configured")

    def test_unselected_media_fields_stay_lazy(self):
        result = self.resolve(artifact_mp4_max_file_bytes="bad", artifact_media_storage_max_file_bytes="bad",
                              artifact_media_probe_timeout_seconds="bad")
        self.assertTrue(result.configured)
        self.assertIsNone(result.configuration.media)
        self.assertEqual(result.configuration.required_capabilities, ("files.publish_generated",))

    def test_media_selection_requires_explicit_bounds_and_pin(self):
        self.assertFalse(self.resolve(artifact_formats="text,mp4").configured)
        self.assertFalse(self.resolve(artifact_formats="text,mp4", artifact_mp4_max_file_bytes="32000000").configured)

    def test_api_configuration_does_not_require_a_local_executable(self):
        result = self.resolve(artifact_formats="text,mp4", artifact_mp4_max_file_bytes="32000000",
            artifact_media_storage_max_file_bytes="20000000", artifact_media_ffprobe_path="/not-installed/ffprobe",
            artifact_media_ffprobe_sha256="a" * 64)
        self.assertTrue(result.configured, result)
        self.assertEqual(result.configuration.required_capabilities,
                         ("files.publish_generated", "files.publish_generated_mp4"))
        self.assertEqual(result.configuration.policy.limit_for("text"), 8 * 1024 * 1024)
        self.assertEqual(result.configuration.policy.limit_for("mp4"), 32000000)

    def test_relative_binary_or_excessive_ceiling_rejected(self):
        for overrides in ({"artifact_media_ffprobe_path": "relative"},
                          {"artifact_media_storage_max_file_bytes": str(33 * 1024 * 1024)}):
            args = dict(artifact_formats="mp4", artifact_mp4_max_file_bytes="32000000",
                artifact_media_storage_max_file_bytes="20000000", artifact_media_ffprobe_path="/opt/ffprobe",
                artifact_media_ffprobe_sha256="a" * 64)
            self.assertFalse(self.resolve(**(args | overrides)).configured)


class ArtifactMediaRuntimeTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix="jarvis-media-runtime-")
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "fixture-inspector"
        self.path.write_bytes(b"fixture executable is never invoked\n")
        self.path.chmod(0o700)
        self.pin = hashlib.sha256(self.path.read_bytes()).hexdigest()
        self.media = ArtifactMediaConfiguration(storage_max_file_bytes=20000000,
            ffprobe_path=str(self.path), ffprobe_sha256=self.pin)
        self.config = ArtifactConfiguration(area_id=uuid4(), policy=ArtifactPolicy(
            formats=("text", "mp4"), mp4_max_file_bytes=32000000), media=self.media)

    async def test_document_only_and_disabled_never_probe(self):
        with patch("app.files.media_runtime.PinnedMP4Inspector.verify", new_callable=AsyncMock) as verify:
            for value in (None, ArtifactConfiguration(area_id=uuid4())):
                self.assertEqual(await prepare_artifact_media(value), (None, None))
            verify.assert_not_awaited()

    async def test_policy_is_narrowed_and_same_inspector_is_callable(self):
        policy, inspector = await prepare_artifact_media(self.config)
        self.assertEqual(policy.max_bytes, 20000000)
        local = object()
        with patch("app.files.media_runtime.inspect_mp4_file", new_callable=AsyncMock, return_value="measured") as probe:
            self.assertEqual(await inspector(local, policy), "measured")
            probe.assert_awaited_once_with(local, policy, ffprobe_path=str(self.path), timeout=10)

    async def test_missing_configuration_fails_closed(self):
        with self.assertRaises(MediaInspectionError):
            await prepare_artifact_media(self.config.model_copy(update={"media": None}))

    async def test_modified_binary_is_rechecked_before_each_inspection(self):
        policy, inspector = await prepare_artifact_media(self.config)
        self.path.write_bytes(b"changed binary\n")
        with patch("app.files.media_runtime.inspect_mp4_file", new_callable=AsyncMock) as probe:
            with self.assertRaises(MediaInspectionError):
                await inspector(object(), policy)
            probe.assert_not_awaited()

    async def test_missing_nonexecutable_and_symlink_are_not_ready(self):
        self.path.chmod(0o600)
        with self.assertRaises(MediaInspectionError):
            await prepare_artifact_media(self.config)
        self.path.unlink()
        with self.assertRaises(MediaInspectionError):
            await prepare_artifact_media(self.config)
        target = self.path.with_name("target")
        target.write_bytes(b"fixture executable is never invoked\n")
        target.chmod(0o700)
        self.path.symlink_to(target)
        with self.assertRaises(MediaInspectionError):
            await prepare_artifact_media(self.config)

    async def test_inspection_cannot_widen_or_change_transfer_kind(self):
        _, inspector = await prepare_artifact_media(self.config)
        for policy in (FileTransferPolicy(), FileTransferPolicy("mp4_capture_v1", 20000000),
                       FileTransferPolicy("mp4_output_v1", 20000001)):
            with self.assertRaises(MediaInspectionError):
                await inspector(object(), policy)

    async def retained_fixture(self, duration="45.0"):
        original = self.path.resolve().with_name("original.mp4")
        original.write_bytes(b"synthetic metadata fixture, not an actual video")
        local = await snapshot_file(original, max_bytes=1024)
        self.addCleanup(local.close)
        metadata = {"format": {"format_name": "mp4", "duration": duration,
            "size": str(local.size_bytes), "tags": {"major_brand": "isom"}},
            "streams": [{"index": 0, "codec_type": "video", "codec_name": "h264",
                "width": 720, "height": 1280, "duration": duration,
                "disposition": {"attached_pic": 0}}]}
        probe = self.enterContext(patch("app.files.media_inspection._probe",
            new_callable=AsyncMock, return_value=json.dumps(metadata).encode()))
        return local, probe

    async def test_retained_capture_uses_its_policy_without_widening_publication(self):
        output, inspector = await prepare_artifact_media(self.config)
        local, probe = await self.retained_fixture()
        capture = FileTransferPolicy("mp4_capture_v1", 64 * 1024**2)
        measured = await inspector.inspect_retained_input(local, capture)
        self.assertEqual(measured.duration_ms, 45000)
        self.assertEqual(measured.content_sha256, local.content_sha256)
        probe.assert_awaited_once_with(str(self.path), local.path)
        self.assertEqual(inspector.policy, output)
        with self.assertRaisesRegex(MediaInspectionError, "media_inspection_policy_exceeded"):
            await inspector(local, capture)
        with self.assertRaisesRegex(MediaInspectionError, "media_inspection_policy_exceeded"):
            await inspector(local, FileTransferPolicy("mp4_output_v1", 20000001))
        with self.assertRaisesRegex(MediaInspectionError, "media_duration_exceeded"):
            await inspector(local, output)

    async def test_retained_output_keeps_its_own_bounds(self):
        _, inspector = await prepare_artifact_media(self.config)
        local, _ = await self.retained_fixture("10.0")
        retained = FileTransferPolicy("mp4_output_v1", 32 * 1024**2)
        self.assertEqual((await inspector.inspect_retained_input(local, retained)).duration_ms, 10000)
        self.assertEqual(inspector.policy.max_bytes, 20000000)

    async def test_retained_input_rejects_non_media_and_non_policy(self):
        _, inspector = await prepare_artifact_media(self.config)
        with patch("app.files.media_runtime.inspect_mp4_file", new_callable=AsyncMock) as probe:
            for policy in (FileTransferPolicy(), None, {"policy_id": "mp4_capture_v1"}):
                with self.subTest(policy=policy):
                    with self.assertRaisesRegex(MediaInspectionError, "media_inspection_policy_exceeded"):
                        await inspector.inspect_retained_input(object(), policy)
            probe.assert_not_awaited()

    async def test_retained_input_enforces_deployment_and_hard_size_limits(self):
        _, inspector = await prepare_artifact_media(self.config)
        local, probe = await self.retained_fixture()
        for kind, ceiling in (("mp4_capture_v1", 256 * 1024**2), ("mp4_output_v1", 32 * 1024**2)):
            with self.subTest(kind=kind):
                with self.assertRaisesRegex(MediaInspectionError, "media_size_exceeded"):
                    await inspector.inspect_retained_input(local, FileTransferPolicy(kind, local.size_bytes - 1))
                # An oversized manifest is rejected before any read or process.
                oversized = replace(local, size_bytes=ceiling + 1)
                # The synthetic manifest does not own the shared descriptor.
                object.__setattr__(oversized, "_closed", True)
                with self.assertRaisesRegex(MediaInspectionError, "media_size_exceeded"):
                    await inspector.inspect_retained_input(oversized, FileTransferPolicy(kind, ceiling * 2))
        probe.assert_not_awaited()

    async def test_retained_capture_duration_stays_bounded(self):
        _, inspector = await prepare_artifact_media(self.config)
        local, _ = await self.retained_fixture("300.100001")
        with self.assertRaisesRegex(MediaInspectionError, "media_duration_exceeded"):
            await inspector.inspect_retained_input(local, FileTransferPolicy("mp4_capture_v1", 64 * 1024**2))

    async def test_retained_inspection_rechecks_binary_and_uses_configured_timeout(self):
        _, inspector = await prepare_artifact_media(self.config)
        capture = FileTransferPolicy("mp4_capture_v1", 64 * 1024**2)
        local = object()
        with patch("app.files.media_runtime.inspect_mp4_file", new_callable=AsyncMock) as probe:
            await inspector.inspect_retained_input(local, capture)
            probe.assert_awaited_once_with(local, capture, ffprobe_path=str(self.path), timeout=10)
            self.path.write_bytes(b"changed executable\n")
            probe.reset_mock()
            with self.assertRaisesRegex(MediaInspectionError, "media_probe_binary_mismatch"):
                await inspector.inspect_retained_input(local, capture)
            probe.assert_not_awaited()
