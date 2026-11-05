"""Per-format authority and concrete publication limits, with legacy fingerprints."""
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock
from uuid import uuid4

from pydantic import ValidationError

from app.files.generated import ArtifactInspection, ArtifactPolicy, ArtifactRequest, ArtifactSnapshot, inspect_artifact
from app.files.publication import FilePublicationService, FilePublicationSpec
from app.files.transfer import FileTransferPolicy
from app.runtime.feature_config import ArtifactConfiguration
from app.tasks.models import digest
from app.tasks.worker import WorkerContextFactory

MIB = 1024 * 1024


class MixedArtifactPolicyTests(unittest.TestCase):
    def mixed(self, **changes):
        return ArtifactPolicy(**({"formats": ("text", "pdf", "mp4"), "mp4_max_file_bytes": 20 * MIB} | changes))

    def test_old_document_and_mp4_policy_encodings_are_unchanged(self):
        expected = {"max_files": 8, "max_file_bytes": 8 * MIB, "max_total_bytes": 32 * MIB,
                    "formats": ["text", "csv", "json", "pdf", "xlsx"]}
        self.assertEqual(ArtifactPolicy().model_dump(mode="json"), expected)
        self.assertEqual(digest(ArtifactPolicy().model_dump(mode="json")), digest(expected))
        legacy = ArtifactPolicy(formats=("mp4",), transfer_policy="mp4_output_v1", max_file_bytes=20 * MIB)
        self.assertNotIn("mp4_max_file_bytes", legacy.model_dump(mode="json"))

    def test_media_is_explicit_and_document_ceiling_never_expands(self):
        for value in ({"formats": ("text", "mp4")}, {"mp4_max_file_bytes": MIB},
                      {"formats": ("text", "mp4"), "mp4_max_file_bytes": MIB, "max_file_bytes": 9 * MIB}):
            with self.subTest(value=value), self.assertRaises(ValidationError):
                ArtifactPolicy(**value)
        policy = self.mixed()
        self.assertEqual((policy.limit_for("text"), policy.limit_for("mp4")), (8 * MIB, 20 * MIB))
        self.assertEqual(policy.for_format("mp4", deployment_limit_bytes=10 * MIB).max_bytes, 10 * MIB)
        self.assertEqual(policy.for_format("text").policy_id, "document_v1")
        self.assertEqual(policy.for_format("mp4").policy_id, "mp4_output_v1")

    def test_mixed_worker_accepts_narrower_legacy_grants_but_does_not_regrant(self):
        worker = self.mixed()
        doc = ArtifactPolicy(formats=("text",), max_files=1, max_total_bytes=MIB)
        media = ArtifactPolicy(formats=("mp4",), transfer_policy="mp4_output_v1", max_file_bytes=12 * MIB)
        self.assertTrue(worker.accepts(doc))
        self.assertTrue(worker.accepts(media))
        self.assertFalse(doc.accepts(worker))
        with self.assertRaises(ValueError):
            doc.for_format("mp4")
        self.assertFalse(worker.accepts(media.model_copy(update={"max_file_bytes": 21 * MIB})))
        self.assertFalse(worker.accepts(doc.model_copy(update={"max_total_bytes": 33 * MIB})))

    def test_each_document_uses_its_smaller_granted_limit(self):
        policy = self.mixed(max_file_bytes=3)
        request = ArtifactRequest(operation_key="note", filename="note.txt", format="text", description="Fixture")
        self.assertEqual(inspect_artifact(request, b"abc", policy).size_bytes, 3)
        with self.assertRaises(ValueError):
            inspect_artifact(request, b"abcd", policy)

    def snapshot(self, limit=None):
        request = ArtifactRequest(operation_key="note", filename="note.txt", format="text", description="Fixture")
        return ArtifactSnapshot(request=request, inspection=ArtifactInspection(format="text", size_bytes=3,
            content_sha256="a" * 64), parent_grant_id=uuid4(), task_id=uuid4(), evidence_payload_id=uuid4(),
            transfer_limit_bytes=limit)

    def test_old_snapshot_has_no_new_fingerprint_field(self):
        original = self.snapshot().model_dump(mode="json")
        self.assertNotIn("transfer_limit_bytes", original)
        self.assertEqual(ArtifactSnapshot.model_validate(original).model_dump(mode="json"), original)

    def test_new_snapshot_freezes_exact_nonwidening_concrete_limit(self):
        value = self.snapshot(4)
        self.assertEqual(value.publication_policy, FileTransferPolicy("document_v1", 4))
        self.assertEqual(value.model_dump(mode="json")["transfer_limit_bytes"], 4)
        for limit in (2, 9 * MIB):
            with self.subTest(limit=limit), self.assertRaises(ValidationError):
                self.snapshot(limit)

    def test_mixed_worker_requires_and_preserves_the_trusted_media_dependencies(self):
        configuration = ArtifactConfiguration(area_id=uuid4(), policy=self.mixed())
        dependencies = dict(knowledge=object(), file_storage=object(), sandbox_provider=object(),
                            artifact_configuration=configuration)
        with self.assertRaises(ValueError):
            WorkerContextFactory(object(), **dependencies)
        factory = WorkerContextFactory(object(), **dependencies,
            media_policy=FileTransferPolicy("mp4_output_v1", 10 * MIB), media_inspector=AsyncMock())
        self.assertEqual(factory.media_policy.max_bytes, 10 * MIB)


class ConcretePublicationPolicyTests(unittest.IsolatedAsyncioTestCase):
    def fixture(self):
        value = object.__new__(FilePublicationService)
        value.owner_id, value.access_check = "fixture", None
        value.transfer_policy = FileTransferPolicy("mp4_output_v1", 20 * MIB)
        value._prepare_manifest = AsyncMock(return_value=object())
        return value

    async def test_bytes_remain_documents_in_a_media_capable_repository(self):
        service = self.fixture()
        spec = FilePublicationSpec(session_id=None, filename="note.txt", content_type="text/plain",
                                   description="Fixture", source="assistant")
        await service.prepare(owner_id="fixture", operation_key="note", spec=spec, content=b"x")
        self.assertEqual(service._prepare_manifest.call_args.kwargs["policy"], FileTransferPolicy())
        await service.prepare(owner_id="fixture", operation_key="note", spec=spec, content=b"x",
                              transfer_policy=FileTransferPolicy("document_v1", 100))
        self.assertEqual(service._prepare_manifest.call_args.kwargs["policy"].max_bytes, 100)

    def test_override_cannot_widen_or_swap_the_configured_concrete_policy(self):
        service = self.fixture()
        for selected in (FileTransferPolicy("mp4_output_v1", 21 * MIB),
                         FileTransferPolicy("mp4_capture_v1", 20 * MIB), FileTransferPolicy()):
            with self.subTest(selected=selected), self.assertRaises(ValueError):
                service._selected_policy(selected, file_backed=True)
        with self.assertRaises(ValueError):
            service._selected_policy(FileTransferPolicy("mp4_output_v1", MIB), file_backed=False)
