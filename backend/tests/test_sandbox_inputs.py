"""Exact source staging, bounded native transport and real task/source fences."""
import asyncio
import base64
from datetime import UTC, datetime, timedelta
import hashlib
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
import stat
import sys
from contextlib import redirect_stdout
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

from app.files.tools import FileToolModule
from app.knowledge.files import SourceFileRepository
from app.sandbox.e2b_provider import E2BProvider
from app.sandbox.file_transfer import MAX_TRANSFER_BYTES, READ_OUTPUT, workspace_file
from app.sandbox.inputs import ContextInput, ImageInput, SandboxInputService
from app.sandbox.models import E2BConfiguration, SandboxError, SandboxInfo
from app.sandbox.tools import SandboxToolBridge
from app.tasks.context_repository import ContextRepository
from app.tasks.memory_evidence import TaskMemoryEvidence
from app.tasks.models import ClaimLost, TaskError
from app.tasks.runner import TaskRunner
from app.tasks.tool_bridge import TaskToolBridge
from app.tasks.worker import WorkerContextFactory
from evals.second_brain.file_storage_fixture import MemoryStorage
from evals.second_brain.sandbox_fixture import sandbox_task_fixture


class FileProvider:
    def __init__(self):
        self.resources, self.inputs = {}, {}
        self.after_stage = None

    async def create(self, scope, policy):
        handle=SandboxInfo(sandbox_id="fixture-"+scope.allocation_id.hex,scope=scope,policy=policy,
            state="running",expires_at=datetime.now(UTC)+timedelta(minutes=5),template_id="fixture")
        self.resources[handle.sandbox_id]=handle
        return handle

    async def prepare(self, handle):
        return handle

    async def stage_input(self, handle, identity, content):
        self.inputs[identity]=content
        if self.after_stage is not None:
            await self.after_stage()
        return {"path":"/home/jarvis/inputs/"+identity,"size_bytes":len(content),
                "sha256":hashlib.sha256(content).hexdigest()}


class SandboxTransferUnitTests(unittest.IsolatedAsyncioTestCase):
    async def test_native_reader_rejects_a_file_modified_during_the_copy(self):
        class Stream(io.BytesIO):
            def fileno(self):
                return 19
        before=SimpleNamespace(st_mode=stat.S_IFREG|0o600,st_size=4,st_dev=1,st_ino=2,st_mtime_ns=3,st_ctime_ns=4)
        after=SimpleNamespace(**vars(before)|{"st_mtime_ns":5,"st_ctime_ns":6})
        # Run the actual fixed native program with all OS effects substituted.
        # No host file, process identity or permission is changed by this test.
        system=SimpleNamespace(setgroups=Mock(),setgid=Mock(),setuid=Mock(),close=Mock(),
            open=Mock(return_value=19),fdopen=Mock(return_value=Stream(b"data")),
            fstat=Mock(side_effect=[before,after]),O_RDONLY=0,O_DIRECTORY=0,O_NOFOLLOW=0,O_NONBLOCK=0)
        accounts=SimpleNamespace(getpwnam=Mock(return_value=SimpleNamespace(pw_uid=1000,pw_gid=1000)))
        with patch.dict(sys.modules,{"os":system,"pwd":accounts}),patch.object(
                sys,"argv",["reader",json.dumps({"path":"output.txt","limit":16})]),redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError,"file changed during read"):
                exec(READ_OUTPUT,{})
        self.assertEqual(system.fstat.call_count,2)

    def provider(self):
        provider=E2BProvider(E2BConfiguration(api_key="fixture"),sdk=SimpleNamespace())
        box=SimpleNamespace(files=SimpleNamespace(write=AsyncMock()),commands=SimpleNamespace(run=AsyncMock()))
        provider._connect=AsyncMock(return_value=box)
        return provider,box

    async def test_input_bytes_use_sdk_not_shell_and_receipt_is_exact(self):
        provider,box=self.provider()
        content=b"private synthetic\x00bytes"
        receipt={"path":"/home/jarvis/inputs/"+"a"*64,"size_bytes":len(content),
                 "sha256":hashlib.sha256(content).hexdigest()}
        box.commands.run.return_value=SimpleNamespace(stdout=json.dumps(receipt))
        self.assertEqual(await provider.stage_input(object(),"a"*64,content),receipt)
        self.assertEqual(box.files.write.call_args.args[1],content)
        self.assertEqual(box.files.write.call_args.kwargs["user"],"root")
        self.assertNotIn(content.decode(),box.commands.run.call_args.args[0])
        box.commands.run.return_value.stdout=json.dumps(receipt|{"path":"/wrong"})
        with self.assertRaises(SandboxError):
            await provider.stage_input(object(),"a"*64,content)

    async def test_output_is_bounded_hash_verified_and_relative(self):
        provider,box=self.provider()
        content=b"CSV\x00\xff"
        receipt={"content_base64":base64.b64encode(content).decode(),"size_bytes":len(content),
                 "sha256":hashlib.sha256(content).hexdigest()}
        box.commands.run.return_value=SimpleNamespace(stdout=json.dumps(receipt))
        self.assertEqual(await provider.read_workspace_file(object(),"reports/result.bin"),content)
        for values in ({"sha256":"0"*64},{"size_bytes":999},{"content_base64":"!!"}):
            with self.subTest(values=values),self.assertRaises(SandboxError):
                box.commands.run.return_value.stdout=json.dumps(receipt|values)
                await provider.read_workspace_file(object(),"result.bin")
        box.commands.run.return_value.stdout=json.dumps(receipt)
        with self.assertRaises(SandboxError):
            await provider.read_workspace_file(object(),"result.bin",max_bytes=1)
        for path in ("/etc/passwd","../private","a/../../private",".","","nul\x00name"):
            with self.subTest(path=path),self.assertRaises(ValueError):
                workspace_file(path)
        self.assertEqual(workspace_file("reports/./data.csv"),"reports/data.csv")

    async def test_invalid_input_bounds_do_not_touch_native_provider(self):
        provider,box=self.provider()
        for identity,content in (("../x",b"x"),("a"*64,b"x"*(MAX_TRANSFER_BYTES+1)),("a"*64,"text")):
            with self.subTest(identity=identity),self.assertRaises(ValueError):
                await provider.stage_input(object(),identity,content)
        provider._connect.assert_not_awaited()

    async def test_context_reader_reassembles_only_same_revision(self):
        source={"availability":"available","revision":2,"trust":"user_input"}
        content='{"text":"synthetic"}'
        pages=[{"status":"success","source":source,"encoding":"json_ascii","content_json":content[:8],
                "offset":0,"next_offset":8,"total_characters":len(content)},
               {"status":"success","source":source,"encoding":"json_ascii","content_json":content[8:],
                "offset":8,"next_offset":None,"total_characters":len(content)}]
        service=SandboxInputService(SimpleNamespace(),None)
        service.read_tool=AsyncMock(side_effect=pages)
        data,metadata,_=await service.context_source(ContextInput(event_id=uuid4()))
        self.assertEqual(data,content.encode())
        self.assertEqual(metadata["source"],source)
        service.read_tool=AsyncMock(side_effect=[pages[0],pages[1]|{"source":source|{"revision":3}}])
        with self.assertRaises(TaskError):
            await service.context_source(ContextInput(event_id=uuid4()))

    async def test_context_paging_rejects_stalls_and_unavailable_sources(self):
        page={"status":"success","source":{"availability":"available"},"encoding":"json_ascii",
              "content_json":"","offset":0,"next_offset":0,"total_characters":20}
        service=SandboxInputService(SimpleNamespace(),None)
        for changed in ({},{"source":{"availability":"deleted"}},{"total_characters":MAX_TRANSFER_BYTES+1}):
            service.read_tool=AsyncMock(return_value=page|changed)
            with self.subTest(changed=changed),self.assertRaises(TaskError):
                await service.context_source(ContextInput(event_id=uuid4()))

    async def test_image_requires_actual_pixels_and_policy_revalidation(self):
        selection={"status":"success","image_view":"captured_snapshot","source_revision":3}
        check=AsyncMock()
        resolved=SimpleNamespace(image_bytes=b"synthetic pixels",reference=SimpleNamespace(content_type="image/jpeg"),
                                 revalidate=check)
        bridge=SimpleNamespace(resolve_image=AsyncMock(return_value=resolved))
        service=SandboxInputService(bridge,None)
        service.read_tool=AsyncMock(return_value=selection)
        data,metadata,revalidate=await service.context_image(ImageInput(event_id=uuid4()))
        self.assertEqual(data,resolved.image_bytes)
        self.assertEqual(metadata["selection"],selection)
        await revalidate()
        check.assert_awaited_once()
        resolved.revalidate=None
        with self.assertRaises(TaskError):
            await service.context_image(ImageInput(event_id=uuid4()))


@unittest.skipUnless(os.getenv("JARVIS_SECOND_BRAIN_TEST_DB_URL") and
                     os.getenv("JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"),"Explicit manifested database required")
class SandboxInputDatabaseTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.f=await self.enterAsyncContext(sandbox_task_fixture(os.environ["JARVIS_SECOND_BRAIN_TEST_DB_URL"],
            Path(os.environ["JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"]),test_name=self.id(),with_session=True))
        self.context=TaskRunner(self.f.tasks,{}).context(self.f.claim)
        self.provider=FileProvider()
        self.bridge=(await WorkerContextFactory(self.f.source.pool,sandbox_provider=self.provider)(self.context)).tools

    async def invoke(self, name, arguments):
        return await self.bridge.invoke(name,arguments,provider_call_id="fixture-"+uuid4().hex)

    async def test_session_source_exact_bytes_and_repeated_staging_share_input(self):
        args={"event_id":str(self.f.task.request_event_id)}
        page=await self.bridge.inputs.read_tool("read_task_context_source",args)
        first=await self.invoke("sandbox_stage_context",args)
        second=await self.invoke("sandbox_stage_context",args)
        self.assertEqual(first["status"],"success",first)
        self.assertEqual(first,second)
        self.assertEqual(list(self.provider.inputs.values()),[page["content_json"].encode("ascii")])
        self.assertEqual(first["sandbox_input"]["source"]["source"],page["source"])
        async with self.f.source.pool.acquire() as connection:
            self.assertEqual(await connection.fetchval("select count(*) from jarvis_sandbox_allocations"),1)
            self.assertEqual(await connection.fetchval("select count(*) from jarvis_task_reservations where status='settled'"),2)

    async def test_missing_source_never_allocates_or_accepts_replacement_bytes(self):
        for args in ({"event_id":str(uuid4())},{"event_id":str(self.f.task.request_event_id),"content":"invented"}):
            result=await self.invoke("sandbox_stage_context",args)
            self.assertEqual(result["status"],"failed")
        self.assertEqual(self.provider.resources,{})
        self.assertEqual(self.provider.inputs,{})

    async def test_source_deleted_during_transfer_requests_vm_cleanup(self):
        async def remove():
            async with self.f.source.pool.acquire() as connection:
                await connection.execute("""update jarvis_context_payloads set availability='deleted',content=null
                    where id=(select payload_id from jarvis_context_events where id=$1)""",self.f.task.request_event_id)
        self.provider.after_stage=remove
        with self.assertRaises(ClaimLost):
            await self.invoke("sandbox_stage_context",{"event_id":str(self.f.task.request_event_id)})
        async with self.f.source.pool.acquire() as connection:
            self.assertEqual(await connection.fetchval("select phase from jarvis_sandbox_allocations"),"cleanup_requested")

    async def saved_files(self):
        self.storage=MemoryStorage()
        foreground=SourceFileRepository(self.f.source.sources,self.storage,
            owner_id=self.f.source.owner,area_id=self.f.source.area)
        file=await foreground.create_text_file(owner_id=self.f.source.owner,session_id=self.f.task.session_id,
            filename="synthetic.txt",description="Synthetic source.",content="Exact synthetic source.\n",
            operation_key="fixture-"+uuid4().hex)
        evidence=TaskMemoryEvidence(self.context,self.f.source.sources,area_id=self.f.source.area)
        files=SourceFileRepository(self.f.source.sources,self.storage,owner_id=self.f.source.owner,
                                   area_id=self.f.source.area,task_evidence=evidence)
        base=await TaskToolBridge.from_modules(self.context,(FileToolModule(files,self.storage),),
            read_tools=("list_files","get_file_metadata"),memory_evidence=evidence)
        self.bridge=await SandboxToolBridge.attach(base,self.provider,files=files)
        return foreground,file

    async def test_saved_file_uses_original_bytes_and_binds_evidence(self):
        _,file=await self.saved_files()
        result=await self.invoke("sandbox_stage_file",{"file_id":str(file.id)})
        self.assertEqual(result["status"],"success",result)
        self.assertEqual(list(self.provider.inputs.values()),[b"Exact synthetic source.\n"])
        self.assertIsNotNone(result["sandbox_input"]["source"]["task_evidence"])
        self.assertEqual(self.storage.uploads,1)
        async with self.f.source.pool.acquire() as connection:
            self.assertGreater(await connection.fetchval("select count(*) from jarvis_task_memory_sources"),0)

    async def test_unavailable_saved_file_never_allocates(self):
        await self.saved_files()
        result=await self.invoke("sandbox_stage_file",{"file_id":str(uuid4())})
        self.assertEqual(result["status"],"failed")
        self.assertEqual(self.provider.resources,{})
