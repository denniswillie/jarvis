"""Opt-in worker/recovery dependencies do not broaden document authority."""
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock
from uuid import uuid4

from app.files.generated import ArtifactPolicy
from app.files.transfer import FileTransferPolicy, MP4_CAPTURE_POLICY, MP4_OUTPUT_POLICY
from app.runtime.feature_config import ArtifactConfiguration
from app.tasks.file_recovery import TaskFileRecovery
from app.tasks.worker import WorkerContextFactory


class WorkerMediaConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.policy = FileTransferPolicy(MP4_OUTPUT_POLICY, 20 * 1024 * 1024)
        self.inspector = AsyncMock()
        self.configuration = ArtifactConfiguration(area_id=uuid4(), policy=ArtifactPolicy(
            transfer_policy=MP4_OUTPUT_POLICY, formats=("mp4",), max_file_bytes=32 * 1024 * 1024))
        self.dependencies = dict(knowledge=object(), file_storage=object(), sandbox_provider=object(),
                                 artifact_configuration=self.configuration)

    def test_document_worker_remains_lazy_and_media_free(self):
        factory = WorkerContextFactory(object())
        self.assertIsNone(factory.media_policy)
        self.assertIsNone(factory.media_inspector)

    def test_media_allowance_requires_both_trusted_dependencies(self):
        for extra in ({}, {"media_policy": self.policy}, {"media_inspector": self.inspector}):
            with self.subTest(extra=tuple(extra)), self.assertRaises(ValueError):
                WorkerContextFactory(object(), **self.dependencies, **extra)

    def test_selected_media_worker_preserves_verified_deployment_limit(self):
        factory = WorkerContextFactory(object(), **self.dependencies,
            media_policy=self.policy, media_inspector=self.inspector)
        self.assertIs(factory.media_policy, self.policy)
        self.assertIs(factory.media_inspector, self.inspector)
        self.assertEqual(factory.media_policy.max_bytes, 20 * 1024 * 1024)
        self.assertEqual(factory.artifact_configuration.policy.max_file_bytes, 32 * 1024 * 1024)

    def test_capture_or_document_policy_cannot_authorize_generated_media(self):
        for policy in (FileTransferPolicy(), FileTransferPolicy(MP4_CAPTURE_POLICY, 64 * 1024 * 1024)):
            with self.subTest(policy=policy), self.assertRaises(ValueError):
                WorkerContextFactory(object(), **self.dependencies, media_policy=policy, media_inspector=self.inspector)

    def test_media_dependencies_cannot_silently_enable_a_document_worker(self):
        with self.assertRaises(ValueError):
            WorkerContextFactory(object(), **(self.dependencies | {
                "artifact_configuration": ArtifactConfiguration(area_id=uuid4())}),
                media_policy=self.policy, media_inspector=self.inspector)

    def test_recovery_reconstructs_repository_with_the_same_media_dependencies(self):
        pool = object()
        sources, tasks = SimpleNamespace(pool=pool), SimpleNamespace(pool=pool)
        recovery = TaskFileRecovery(sources, tasks, object(), owner_id="fixture", area_id=uuid4(),
            media_policy=self.policy, media_inspector=self.inspector)
        files = recovery._files(recovery.area_id)
        self.assertIs(files.media_policy, self.policy)
        self.assertIs(files.media_inspector, self.inspector)
        self.assertIs(files.task_recovery, recovery)
        self.assertEqual(files.publications.transfer_policy.max_bytes, 20 * 1024 * 1024)

    def test_recovery_rejects_partial_or_wrong_policy_configuration(self):
        pool = object()
        for extra in ({"media_policy": self.policy}, {"media_inspector": self.inspector},
                {"media_policy": FileTransferPolicy(), "media_inspector": self.inspector}):
            with self.subTest(extra=tuple(extra)), self.assertRaises(ValueError):
                TaskFileRecovery(SimpleNamespace(pool=pool), SimpleNamespace(pool=pool), object(),
                                 owner_id="fixture", area_id=uuid4(), **extra)
