"""Input-service authority/lifetime checks; no VM, SQL or provider calls."""
import asyncio
import hashlib
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

from app.files.local_file import _snapshot_chunks
from app.files.transfer import FileManifest, FileTransferPolicy
from app.knowledge.models import SourceError
from app.sandbox.inputs import SandboxInputService, SavedFileInput
from app.sandbox.models import SandboxError
from app.tasks.models import ClaimLost, TaskError


class SandboxMediaInputTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.size=9*1024*1024+17
        self.policy=FileTransferPolicy("mp4_capture_v1",20_000_000)
        self.snapshots=[]
        self.transfers=[]
        self.reservation=uuid4()
        self.task=SimpleNamespace(owner_id="synthetic-media-owner")
        self.context=SimpleNamespace(task=self.task,claim=object(),
            reserve=AsyncMock(return_value={"id":self.reservation}),
            budgets=SimpleNamespace(submit=AsyncMock()),settle=AsyncMock())
        self.bridge=SimpleNamespace(context=self.context,_check=AsyncMock(),
            read_tools=("get_file_metadata",))
        self.allocation={"id":uuid4()}
        self.broker=SimpleNamespace(allocation=AsyncMock(return_value=self.allocation),
            _ready=AsyncMock(side_effect=lambda value:value),
            allocations=SimpleNamespace(handle=Mock(return_value="same-vm"),request_cleanup=AsyncMock()),
            provider=SimpleNamespace(stage_input=AsyncMock(),stage_input_file=AsyncMock(side_effect=self.transfer)))
        self.file=SimpleNamespace(id=uuid4(),filename="recording.mp4",content_type="video/mp4",
            size_bytes=self.size,memory_source={"source_id":str(uuid4()),"revision":1,"generation":1,
                "content_digest":"d"*64,"is_current":True},task_evidence={"fixture":"exact-source"})
        # These arbitrary bytes test streaming/identity only, not MP4 decoding.
        local=await self.snapshot()
        self.manifest=FileManifest(size_bytes=local.size_bytes,content_sha256=local.content_sha256)
        local.close()
        self.files=SimpleNamespace(get_file=AsyncMock(side_effect=lambda **_:self.file),
            read_content=AsyncMock(side_effect=AssertionError("media must not use whole bytes")),
            read_file=AsyncMock(),
            publications=SimpleNamespace(read_manifest=AsyncMock(return_value=(self.manifest,self.policy))))
        async def read_file(**_):
            return await self.snapshot()
        self.files.read_file.side_effect=read_file
        self.service=SandboxInputService(self.bridge,self.broker,files=self.files)
        self.request=SavedFileInput(file_id=self.file.id)

    async def snapshot(self):
        async def chunks():
            remaining=self.size
            while remaining:
                count=min(65536,remaining)
                yield b"m"*count
                remaining-=count
        local=await _snapshot_chunks(chunks(),max_bytes=self.policy.max_bytes)
        self.snapshots.append(local)
        return local

    async def transfer(self,handle,identity,local,*,policy):
        self.assertEqual(handle,"same-vm")
        self.assertEqual(policy,self.policy)
        digest=hashlib.sha256(); size=0
        async for block in local.iter_chunks():
            self.assertLessEqual(len(block),65536)
            digest.update(block); size+=len(block)
        self.transfers.append((identity,size,digest.hexdigest()))
        return {"path":"/home/jarvis/inputs/"+identity,"size_bytes":size,"sha256":digest.hexdigest()}

    async def asyncTearDown(self):
        leaked=[local for local in self.snapshots if local.path.exists()]
        for local in leaked:
            local.close()
        self.assertEqual(leaked,[],"Every private input must close on success, denial or cancellation")

    async def stage(self):
        return await self.service.stage(self.service.saved_file,self.request)

    async def test_large_retained_video_streams_and_preserves_exact_source(self):
        result=await self.stage()
        saved=result["sandbox_input"]
        self.assertEqual(saved["size_bytes"],self.size)
        self.assertEqual(saved["sha256"],self.manifest.content_sha256)
        self.assertEqual(saved["source"]["source"],self.file.memory_source)
        self.assertEqual(saved["source"]["task_evidence"],self.file.task_evidence)
        self.assertTrue(saved["read_only"])
        self.files.read_content.assert_not_called()
        self.broker.provider.stage_input.assert_not_called()
        self.broker.allocations.request_cleanup.assert_not_called()
        self.context.settle.assert_awaited_once()
        await self.service.revalidate_staged()
        repeated=await self.stage()
        self.assertEqual(repeated["sandbox_input"]["path"],saved["path"])

    async def test_unavailable_source_does_not_allocate_or_read_bytes(self):
        self.files.get_file.side_effect=None
        self.files.get_file.return_value=None
        with self.assertRaises(SourceError):
            await self.stage()
        self.files.read_file.assert_not_called()
        self.broker.allocation.assert_not_called()
        self.context.settle.assert_awaited_once()

    async def test_wrong_policy_or_narrow_limit_rejects_before_read(self):
        for policy in (FileTransferPolicy(),FileTransferPolicy("mp4_capture_v1",1024)):
            with self.subTest(policy=policy):
                self.files.publications.read_manifest.return_value=(self.manifest,policy)
                with self.assertRaises(TaskError):
                    await self.stage()
        self.files.read_file.assert_not_called()
        self.broker.allocation.assert_not_called()

    async def test_snapshot_hash_mismatch_closes_before_allocation(self):
        self.files.publications.read_manifest.return_value=(
            FileManifest(size_bytes=self.size,content_sha256="f"*64),self.policy)
        with self.assertRaises(SourceError):
            await self.stage()
        self.broker.allocation.assert_not_called()

    async def test_revision_change_during_local_read_closes_before_allocation(self):
        async def changed_read(**_):
            local=await self.snapshot()
            self.file.memory_source=dict(self.file.memory_source,revision=2,content_digest="e"*64)
            return local
        self.files.read_file.side_effect=changed_read
        with self.assertRaises(SourceError):
            await self.stage()
        self.broker.allocation.assert_not_called()

    async def test_revoked_during_upload_revokes_same_vm_and_returns_no_input(self):
        async def revoke(*args,**kwargs):
            receipt=await self.transfer(*args,**kwargs)
            self.files.get_file.side_effect=None
            self.files.get_file.return_value=None
            return receipt
        self.broker.provider.stage_input_file.side_effect=revoke
        with self.assertRaises(SourceError):
            await self.stage()
        self.broker.allocations.request_cleanup.assert_awaited_once_with(
            self.task.owner_id,self.allocation["id"])
        self.assertEqual(self.service._staged_revalidators,[])

    async def test_claim_loss_after_upload_revokes_same_vm(self):
        self.bridge._check.side_effect=[None,ClaimLost()]
        with self.assertRaises(ClaimLost):
            await self.stage()
        self.assertEqual(len(self.transfers),1)
        self.broker.allocations.request_cleanup.assert_awaited_once()

    async def test_uncertain_transport_revokes_without_retry(self):
        self.broker.provider.stage_input_file.side_effect=SandboxError("sandbox_request_failed")
        with self.assertRaises(SandboxError):
            await self.stage()
        self.broker.provider.stage_input_file.assert_awaited_once()
        self.broker.allocations.request_cleanup.assert_awaited_once()
        self.broker.allocation.assert_awaited_once()

    async def test_cancelled_upload_closes_input_and_requests_cleanup(self):
        started=asyncio.Event()
        async def waiting_upload(*args,**kwargs):
            started.set()
            await asyncio.Future()
        self.broker.provider.stage_input_file.side_effect=waiting_upload
        task=asyncio.create_task(self.stage())
        try:
            async with asyncio.timeout(2):
                await started.wait()
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task,return_exceptions=True)
        self.broker.allocations.request_cleanup.assert_awaited_once()
        self.context.settle.assert_awaited_once()

    async def test_unavailable_local_snapshot_is_a_clear_tool_error(self):
        async def closed_read(**_):
            local=await self.snapshot()
            local.close()
            return local
        self.files.read_file.side_effect=closed_read
        with self.assertRaises(TaskError) as raised:
            await self.stage()
        self.assertEqual(raised.exception.code,"sandbox_input_changed")
        self.broker.allocation.assert_not_called()

    async def test_cleanup_failure_is_not_hidden_as_success(self):
        self.broker.provider.stage_input_file.side_effect=SandboxError("sandbox_request_failed")
        self.broker.allocations.request_cleanup.side_effect=RuntimeError("cleanup outcome unknown")
        with self.assertRaisesRegex(RuntimeError,"cleanup outcome unknown"):
            await self.stage()
        self.context.settle.assert_awaited_once()

    async def test_document_uses_unchanged_bytes_path_without_media_manifest(self):
        self.file.content_type="text/plain"
        self.files.read_content.side_effect=None
        self.files.read_content.return_value=b"exact document"
        self.broker.provider.stage_input.return_value={"path":"/home/jarvis/inputs/document",
            "size_bytes":14,"sha256":hashlib.sha256(b"exact document").hexdigest()}
        result=await self.stage()
        self.assertEqual(result["status"],"success")
        self.files.read_file.assert_not_called()
        self.files.publications.read_manifest.assert_not_called()
        self.broker.provider.stage_input_file.assert_not_called()
        self.assertEqual(self.broker.provider.stage_input.call_args.args[-1],b"exact document")

    async def test_document_cannot_inherit_media_limit(self):
        self.file.content_type="text/plain"
        self.files.read_content.side_effect=None
        self.files.read_content.return_value=b"d"*(8*1024*1024+1)
        with self.assertRaises(TaskError):
            await self.stage()
        self.broker.allocation.assert_not_called()


if __name__ == "__main__":
    unittest.main()
