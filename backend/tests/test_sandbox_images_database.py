"""Real claim/grant revocation around workspace pixels, with no native VM."""
from io import BytesIO
import os
from pathlib import Path
import unittest
from uuid import uuid4

from PIL import Image

from app.tasks.actions import TaskActionRepository
from app.tasks.budgets import TaskBudgetRepository
from app.tasks.models import ClaimLost
from app.tasks.runner import WorkerContext
from app.tasks.worker import WorkerContextFactory
from evals.second_brain.sandbox_fixture import sandbox_task_fixture
from test_sandbox_commands import CommandProvider


class ImageProvider(CommandProvider):
    def __init__(self):
        super().__init__()
        stream = BytesIO()
        Image.new("RGB", (64, 32), "navy").save(stream, "PNG")
        self.content = stream.getvalue()
        self.reads = 0

    async def read_workspace_file(self, handle, path, *, max_bytes):
        self.reads += 1
        assert path == "screen.png" and len(self.content) <= max_bytes
        return self.content


@unittest.skipUnless(os.getenv("JARVIS_SECOND_BRAIN_TEST_DB_URL") and
                     os.getenv("JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"), "Manifested SQL fixtures required")
class WorkspaceImageDatabaseTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.f = await self.enterAsyncContext(sandbox_task_fixture(os.environ["JARVIS_SECOND_BRAIN_TEST_DB_URL"],
            Path(os.environ["JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"]), test_name=self.id()))
        self.context = WorkerContext(self.f.claim, self.f.tasks, TaskBudgetRepository(self.f.tasks),
                                     TaskActionRepository(self.f.tasks))
        self.provider = ImageProvider()
        self.bridge = (await WorkerContextFactory(self.f.source.pool,
                       sandbox_provider=self.provider)(self.context)).tools

    async def select(self):
        return await self.bridge.invoke("sandbox_view_image", {"path": "screen.png"},
                                       provider_call_id="fixture-image-" + uuid4().hex)

    async def test_exact_pixels_and_same_attempt_snapshot_survive_file_change(self):
        selected = await self.select()
        self.assertEqual(selected["status"], "success", selected)
        original = self.provider.content
        self.provider.content = b"changed after inspection"
        resolved = await self.bridge.resolve_image("sandbox_view_image", selected)
        self.assertEqual(resolved.image_bytes, original)
        self.assertEqual(self.provider.reads, 1)
        self.assertEqual(resolved.reference.owner_id, self.f.source.owner)
        self.assertEqual(resolved.reference.attempt_id, self.f.claim.attempt_id)
        self.assertEqual((await self.context.budgets.usage(self.f.source.owner, self.f.task.id))["steps"], 1)

    async def test_revoked_grant_fences_selected_pixels_and_further_reads(self):
        resolved = await self.bridge.resolve_image("sandbox_view_image", await self.select())
        async with self.f.source.pool.acquire() as connection:
            await connection.execute("update jarvis_operation_grants set status='revoked' where id=$1", self.f.grant)
        with self.assertRaises(ClaimLost):
            await resolved.revalidate()
        with self.assertRaises(ClaimLost):
            await self.select()
        self.assertEqual(self.provider.reads, 1)

    async def test_deleted_task_context_fences_pixels_already_registered_for_model_transport(self):
        resolved = await self.bridge.resolve_image("sandbox_view_image", await self.select())
        async with self.f.source.pool.acquire() as connection:
            await connection.execute("""update jarvis_context_payloads set content=null,availability='deleted'
                where id=$1""", self.f.task.description_payload_id)
        with self.assertRaises(ClaimLost):
            await resolved.revalidate()

    async def test_revocation_between_native_read_and_registration_exposes_no_pixels(self):
        read = self.provider.read_workspace_file
        async def revoke(*args, **kwargs):
            content = await read(*args, **kwargs)
            async with self.f.source.pool.acquire() as connection:
                await connection.execute("update jarvis_operation_grants set status='revoked' where id=$1", self.f.grant)
            return content
        self.provider.read_workspace_file = revoke
        with self.assertRaises(ClaimLost):
            await self.select()
        self.assertEqual(self.bridge.images.selections, {})
