"""Offline original-allocation staging and input/output role separation."""
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock
from uuid import uuid4

from app.files.generated import ArtifactPolicy
from app.files.transfer import FileTransferPolicy
from app.runtime.feature_config import ArtifactConfiguration
from app.sandbox.inputs import SandboxInputService
from app.tasks.models import TaskError
from app.tasks.worker import WorkerContextFactory


class OriginalAllocationStagingTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.identity = uuid4()
        self.row = {"id": self.identity, "phase": "ready"}
        self.claim = object()
        self.context = SimpleNamespace(task=SimpleNamespace(owner_id="synthetic-owner"), claim=self.claim,
            reserve=AsyncMock(return_value={"id": uuid4()}), budgets=SimpleNamespace(submit=AsyncMock()),
            settle=AsyncMock())
        self.bridge = SimpleNamespace(context=self.context, _check=AsyncMock())
        self.handle = object()
        self.broker = SimpleNamespace(
            allocation=AsyncMock(side_effect=AssertionError("Recovery cannot reserve an allocation")),
            _ready=AsyncMock(side_effect=AssertionError("Recovery cannot prepare or replace a sandbox")),
            allocations=SimpleNamespace(current=AsyncMock(return_value=self.row),
                handle=lambda row: self.handle, request_cleanup=AsyncMock()),
            provider=SimpleNamespace(stage_input=AsyncMock(return_value={"path": "/synthetic-input"})))
        self.revalidate = AsyncMock()
        self.source = AsyncMock(return_value=(b"exact-original", {"kind": "synthetic"}, self.revalidate))
        self.inputs = SandboxInputService(self.bridge, self.broker)

    async def test_ready_original_is_used_without_reserving_or_preparing(self):
        result = await self.inputs.stage(self.source, object(), allocation_id=self.identity)
        self.broker.allocations.current.assert_awaited_once_with(self.claim, self.identity)
        self.broker.allocation.assert_not_awaited()
        self.broker._ready.assert_not_awaited()
        self.assertIs(self.broker.provider.stage_input.call_args.args[0], self.handle)
        self.assertEqual(self.broker.provider.stage_input.call_args.args[2], b"exact-original")
        self.assertTrue(result["sandbox_input"]["read_only"])
        self.assertEqual(self.revalidate.await_count, 2)

    async def test_nonready_original_is_not_prepared_or_cleaned_as_if_staged(self):
        for phase in ("reserved", "creating", "uncertain", "preparing", "cleanup_requested"):
            self.broker.allocations.current.return_value = {"id": self.identity, "phase": phase}
            with self.subTest(phase=phase), self.assertRaises(TaskError) as error:
                await self.inputs.stage(self.source, object(), allocation_id=self.identity)
            self.assertEqual(error.exception.code, "sandbox_original_allocation_unavailable")
        self.broker.allocation.assert_not_awaited()
        self.broker._ready.assert_not_awaited()
        self.broker.provider.stage_input.assert_not_awaited()
        self.broker.allocations.request_cleanup.assert_not_awaited()

    async def test_missing_original_cannot_allocate_replacement(self):
        self.broker.allocations.current.side_effect = TaskError("allocation_unavailable", "Unavailable")
        with self.assertRaises(TaskError):
            await self.inputs.stage(self.source, object(), allocation_id=self.identity)
        self.broker.allocation.assert_not_awaited()
        self.broker.provider.stage_input.assert_not_awaited()

    async def test_lost_source_after_staging_revokes_only_original_allocation(self):
        self.revalidate.side_effect = [None, TaskError("source_withdrawn", "Withdrawn")]
        with self.assertRaises(TaskError):
            await self.inputs.stage(self.source, object(), allocation_id=self.identity)
        self.broker.allocations.request_cleanup.assert_awaited_once_with("synthetic-owner", self.identity)
        self.broker.allocation.assert_not_awaited()
        self.context.settle.assert_awaited_once()

    async def test_invalid_identity_has_no_reservation_or_source_read(self):
        with self.assertRaises(TaskError):
            await self.inputs.stage(self.source, object(), allocation_id=str(self.identity))
        self.context.reserve.assert_not_awaited()
        self.source.assert_not_awaited()

    async def test_ordinary_staging_retains_existing_allocation_flow(self):
        self.broker.allocation.side_effect = None
        self.broker.allocation.return_value = self.row
        self.broker._ready.side_effect = None
        self.broker._ready.return_value = self.row
        await self.inputs.stage(self.source, object())
        self.broker.allocation.assert_awaited_once()
        self.broker._ready.assert_awaited_once_with(self.row)
        self.broker.allocations.current.assert_not_awaited()


class MediaInspectorRoleTests(unittest.TestCase):
    def test_output_inspector_alone_cannot_advertise_media_worker(self):
        class OutputInspector:
            async def __call__(self, local_file, policy):
                raise AssertionError("Composition must not inspect media")
        inspector = OutputInspector()
        provider = SimpleNamespace(materialize_media_render=AsyncMock(), decode_media_render_input=AsyncMock())
        configuration = ArtifactConfiguration(area_id=uuid4(),
            policy=ArtifactPolicy(formats=("json", "mp4"), mp4_max_file_bytes=1024))
        options = dict(knowledge=object(), file_storage=object(), sandbox_provider=provider,
            artifact_configuration=configuration, media_policy=FileTransferPolicy("mp4_output_v1", 1024),
            media_inspector=inspector, managed_plugins=SimpleNamespace(capabilities=("plugins.video_editor",)))
        with self.assertRaisesRegex(ValueError, "input inspection"):
            WorkerContextFactory(None, **options)
        inspector.inspect_retained_input = AsyncMock()
        self.assertIn("plugins.video_editor", WorkerContextFactory(None, **options).capabilities)
        inspector.inspect_retained_input.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
