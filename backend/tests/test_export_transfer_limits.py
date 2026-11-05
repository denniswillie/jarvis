"""Export ceilings use metadata only; remote guards stop before filesystem I/O."""
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock
from uuid import uuid4

from app.files.generated import ArtifactInspection, ArtifactPolicy, ArtifactRequest, ArtifactSnapshot
from app.files.transfer import (
    DOCUMENT_POLICY, FileManifest, FileTransferPolicy, MP4Inspection,
    MP4_OUTPUT_MAX_BYTES, MP4_OUTPUT_MAX_DURATION_MS, MP4_OUTPUT_POLICY,
)
from app.migrations.runner import load_migrations, render_fresh_schema
from app.sandbox.file_transfer import FREEZE_OUTPUT, INSTALL_INPUT_FILE, MAX_TRANSFER_BYTES


EXPORT_BYTES = 500_000_000
EXPORT_MS = 1_800_000
DIGEST = "a" * 64


class ReachedFilesystemBoundary(Exception):
    pass


def definitions(program):
    scope = {"__name__": "export_boundary_fixture"}
    exec(compile(program, "<trusted-export-program>", "exec"), scope)
    return scope


class ExportTransferLimitsTests(unittest.TestCase):
    def test_export_migration_and_rendered_schema_preserve_other_limits(self):
        migrations = load_migrations()
        upgrade = next(item for item in migrations if item.version == 55)
        self.assertEqual(upgrade.name, "edited_video_limits")
        for policy, ceiling in (("document_v1", 8388608), ("mp4_capture_v1", 1073741824),
                                ("mp4_output_v1", EXPORT_BYTES)):
            self.assertIn(f"'{policy}' and transfer_limit_bytes between 1 and {ceiling}", upgrade.sql)
        self.assertNotIn("update jarvis_", upgrade.sql.lower())
        rendered = render_fresh_schema(migrations)
        self.assertIn(upgrade.sql.rstrip(), rendered)
        self.assertIn(upgrade.checksum, rendered)
        schema = Path(__file__).resolve().parents[1] / "app" / "schema.sql"
        self.assertEqual(schema.read_text(encoding="utf-8"), rendered)

    def measurement(self, **changes):
        return MP4Inspection(**(dict(size_bytes=EXPORT_BYTES, content_sha256=DIGEST,
            duration_ms=EXPORT_MS, width=1920, height=1080, video_codec="h264",
            audio_codec="aac") | changes))

    def test_exact_export_bytes_and_duration_and_each_plus_one(self):
        self.assertEqual((MP4_OUTPUT_MAX_BYTES, MP4_OUTPUT_MAX_DURATION_MS),
                         (EXPORT_BYTES, EXPORT_MS))
        policy = FileTransferPolicy(MP4_OUTPUT_POLICY, EXPORT_BYTES)
        manifest = FileManifest(size_bytes=EXPORT_BYTES, content_sha256=DIGEST)
        spec = SimpleNamespace(filename="export.mp4", content_type="video/mp4")
        policy.validate(manifest, spec)
        self.measurement().validate_file(manifest, policy)
        self.assertEqual(FileTransferPolicy(MP4_OUTPUT_POLICY, EXPORT_BYTES + 1).max_bytes,
                         EXPORT_BYTES)
        with self.assertRaises(ValueError):
            policy.validate(FileManifest(size_bytes=EXPORT_BYTES + 1, content_sha256=DIGEST), spec)
        with self.assertRaises(ValueError):
            self.measurement(duration_ms=EXPORT_MS + 1).validate_file(manifest, policy)
        with self.assertRaises(ValueError):
            self.measurement().validate_file(manifest,
                FileTransferPolicy(MP4_OUTPUT_POLICY, EXPORT_BYTES - 1))

    def test_generated_and_mixed_grants_keep_exact_export_bounds(self):
        legacy = dict(transfer_policy=MP4_OUTPUT_POLICY, formats=("mp4",),
                      max_file_bytes=EXPORT_BYTES, max_total_bytes=EXPORT_BYTES)
        mixed = dict(formats=("text", "mp4"), mp4_max_file_bytes=EXPORT_BYTES,
                     max_total_bytes=EXPORT_BYTES)
        for kwargs in (legacy, mixed):
            policy = ArtifactPolicy(**kwargs)
            self.assertEqual(policy.for_format("mp4").max_bytes, EXPORT_BYTES)
            self.assertEqual(policy.for_format("mp4", deployment_limit_bytes=20_000_000).max_bytes,
                             20_000_000)
            with self.assertRaises(ValueError):
                ArtifactPolicy(**(kwargs | {"max_total_bytes": EXPORT_BYTES + 1}))
        with self.assertRaises(ValueError):
            ArtifactPolicy(**(legacy | {"max_file_bytes": EXPORT_BYTES + 1}))
        with self.assertRaises(ValueError):
            ArtifactPolicy(**(mixed | {"mp4_max_file_bytes": EXPORT_BYTES + 1}))
        inspection = ArtifactInspection(format="mp4", size_bytes=EXPORT_BYTES,
            content_sha256=DIGEST, media_inspection=self.measurement())
        values = dict(request=ArtifactRequest(operation_key="export", filename="export.mp4",
            format="mp4", description="Metadata-only export boundary"), inspection=inspection,
            parent_grant_id=uuid4(), task_id=uuid4(), evidence_payload_id=uuid4())
        snapshot = ArtifactSnapshot(**values, transfer_limit_bytes=EXPORT_BYTES)
        self.assertEqual(snapshot.publication_policy.max_bytes, EXPORT_BYTES)
        with self.assertRaises(ValueError):
            ArtifactSnapshot(**values, transfer_limit_bytes=EXPORT_BYTES + 1)

    def test_document_per_file_and_aggregate_limits_are_unchanged(self):
        document_bytes = 8 * 1024**2
        self.assertEqual(MAX_TRANSFER_BYTES, document_bytes)
        policy = FileTransferPolicy(DOCUMENT_POLICY, EXPORT_BYTES)
        self.assertEqual(policy.max_bytes, document_bytes)
        ArtifactPolicy(max_file_bytes=document_bytes, max_total_bytes=128 * 1024**2)
        for changes in ({"max_file_bytes": document_bytes + 1},
                        {"max_total_bytes": 128 * 1024**2 + 1}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                ArtifactPolicy(**changes)
        spec = SimpleNamespace(filename="document.txt", content_type="text/plain")
        policy.validate(FileManifest(size_bytes=document_bytes, content_sha256=DIGEST), spec)
        with self.assertRaises(ValueError):
            policy.validate(FileManifest(size_bytes=document_bytes + 1, content_sha256=DIGEST), spec)

    def test_remote_freeze_accepts_exact_ceiling_and_rejects_plus_one_before_io(self):
        scope = definitions(FREEZE_OUTPUT)
        boundary = scope["stage_directory"] = Mock(side_effect=ReachedFilesystemBoundary)
        request = {"stage_id": "b" * 32, "path": "export.mp4"}
        for limit in (EXPORT_BYTES, 20_000_000):
            boundary.reset_mock()
            with self.subTest(limit=limit), self.assertRaises(ReachedFilesystemBoundary):
                scope["freeze"](request | {"limit": limit})
            boundary.assert_called_once_with(request["stage_id"])
        for limit in (EXPORT_BYTES + 1, 0, True):
            boundary.reset_mock()
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                scope["freeze"](request | {"limit": limit})
            boundary.assert_not_called()

    def test_remote_installer_enforces_export_ceiling_and_smaller_grants_before_io(self):
        scope = definitions(INSTALL_INPUT_FILE)
        boundary = scope["protected_directory"] = Mock(side_effect=ReachedFilesystemBoundary)
        base = {"identity": DIGEST, "stage_id": "b" * 32, "sha256": DIGEST,
                "policy": MP4_OUTPUT_POLICY}
        for limit in (EXPORT_BYTES, 20_000_000):
            request = base | {"limit": limit, "size": limit}
            boundary.reset_mock()
            with self.subTest(limit=limit), self.assertRaises(ReachedFilesystemBoundary):
                scope["install"](request)
            boundary.assert_called_once_with("/var/lib/jarvis/incoming")
            boundary.reset_mock()
            with self.assertRaises(ValueError):
                scope["install"](request | {"size": limit + 1})
            boundary.assert_not_called()
        boundary.reset_mock()
        with self.assertRaises(ValueError):
            scope["install"](base | {"limit": EXPORT_BYTES + 1, "size": 1})
        boundary.assert_not_called()
