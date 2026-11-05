"""Retained snapshots through actual storage and a separate frontier worker."""
import asyncio
from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from io import BytesIO
import json
import os
from pathlib import Path
import signal
import sys
from tempfile import TemporaryDirectory
from types import MethodType, SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode
from uuid import uuid4

import httpx

from PIL import Image, ImageDraw, ImageFont
from app.images import ImageReference, ImageUnavailableError
from app.runtime.feature_config import TaskLimits
from app.settings import JarvisSettings
from app.tasks.catalog import TaskWorkerRegistry, ConfiguredTaskProvider
from app.tasks.context_events import ContextIngress
from app.tasks.context_repository import ContextRepository
from app.tasks.context_images import TaskContextImages
from app.tasks.model_gateway import TaskModelGateway
from app.tasks.models import TaskCreate, TaskStatus
from app.tasks.repository import TaskRepository
from app.tasks.runner import TaskRunner
from app.tasks.service import TaskService
from app.tasks.stored_images import StoredFrameImageService, StoredTaskImageFactory
from app.visual_memory.models import ValidatedVisualFrame, VisualFrameStorageObject
from app.visual_memory.repository import VisualFrameRepository
from app.visual_memory.storage import VisualFrameStorageClient
from evals.second_brain.source_fixture import source_fixture
from evals.second_brain.stored_image_fixture import require_absent_object, stored_image_fixture
from test_second_brain_model_gateway import context as model_context, Upstream


def pixels(label="Synthetic"):
    image = Image.new("RGB",(640,220),"white")
    ImageDraw.Draw(image).text((24,70),label,font=ImageFont.load_default(size=40),fill="black")
    output = BytesIO()
    image.save(output,format="JPEG",quality=95)
    return output.getvalue()


class StoredImageUnitTests(unittest.IsolatedAsyncioTestCase):
    async def test_absence_probe_requires_exact_structured_missing_object(self):
        config = SimpleNamespace(supabase_url="https://fixture.invalid",service_role_key="fixture-only-key")
        for status,body,accepted in (
            (400,{"code":"NoSuchKey","statusCode":"404"},True),
            (404,{"code":"NoSuchKey","statusCode":404},True),
            (400,{"code":"AccessDenied","statusCode":"403"},False),
            (404,{"code":"NoSuchBucket","statusCode":"404"},False),
            (200,{"code":"NoSuchKey","statusCode":"404"},False),
            (404,["NoSuchKey"],False),
        ):
            with self.subTest(status=status,body=body):
                def response(request):
                    self.assertEqual(request.method,"GET")
                    self.assertTrue(request.url.params.get("jarvis_fixture_absence"))
                    self.assertIn("no-cache",request.headers["cache-control"])
                    self.assertEqual(request.url.path,"/storage/v1/object/authenticated/files/fixture/image.jpg")
                    return httpx.Response(status,json=body)
                async with httpx.AsyncClient(transport=httpx.MockTransport(response)) as client:
                    if accepted:
                        await require_absent_object(client,config,bucket="files",path="fixture/image.jpg")
                    else:
                        with self.assertRaises(RuntimeError):
                            await require_absent_object(client,config,bucket="files",path="fixture/image.jpg")

    async def test_absence_probe_rejects_unstructured_or_oversized_errors(self):
        config = SimpleNamespace(supabase_url="https://fixture.invalid",service_role_key="fixture-only-key")
        for content in (b"not JSON",b"x"*2049):
            with self.subTest(length=len(content)):
                async with httpx.AsyncClient(transport=httpx.MockTransport(
                        lambda request:httpx.Response(400,content=content))) as client:
                    with self.assertRaises(RuntimeError):
                        await require_absent_object(client,config,bucket="files",path="fixture/image.jpg")

    async def test_cleanup_attempts_every_owned_object_after_first_failure(self):
        @asynccontextmanager
        async def acquire():
            yield SimpleNamespace(execute=AsyncMock())
        owner = "direct-image-eval-second-brain-cleanup-unit"
        fixture = SimpleNamespace(owner=owner,pool=SimpleNamespace(acquire=acquire))
        frames = [SimpleNamespace(id=uuid4()),SimpleNamespace(id=uuid4())]
        repository = SimpleNamespace(create_pending_frame=AsyncMock(side_effect=frames),
            mark_frame_available=AsyncMock(side_effect=frames))
        storage = SimpleNamespace(bucket="fixture-frames",
            upload_frame=AsyncMock(),delete_objects=AsyncMock(),close=AsyncMock())
        storage.build_storage_path = MethodType(VisualFrameStorageClient.build_storage_path,storage)
        target = "evals.second_brain.stored_image_fixture."
        absence = AsyncMock(side_effect=[None,None,RuntimeError("Synthetic first cleanup failure"),None])
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with (patch(target+"get_supabase_storage_config",return_value=SimpleNamespace()),
                  patch(target+"VisualFrameStorageClient",return_value=storage),
                  patch(target+"VisualFrameRepository",return_value=repository),
                  patch(target+"require_absent_object",absence)):
                with self.assertRaisesRegex(RuntimeError,"cleanup incomplete"):
                    async with stored_image_fixture(fixture,root,None,session_id=uuid4(),
                                                    images=[pixels("First"),pixels("Second")]):
                        pass
            manifest = json.loads(next(root.glob("image-objects-*.json")).read_text())
        self.assertEqual(storage.delete_objects.await_count,2)
        storage.close.assert_awaited_once()
        self.assertEqual(manifest["status"],"cleanup_unverified")
        self.assertEqual(manifest["objects"][0]["phase"],"cleanup_unverified")
        self.assertEqual(manifest["objects"][1]["phase"],"deleted")
        self.assertTrue(manifest["objects"][1]["backing_object_absence_verified"])

    def test_invalid_time_and_size_bounds_are_rejected(self):
        for changes in ({"timeout_seconds":float("inf")},{"timeout_seconds":float("nan")},
                        {"max_bytes":True},{"max_width":0}):
            with self.subTest(changes=changes),self.assertRaises(ValueError):
                StoredFrameImageService(None,None,**({
                    "max_bytes":512000,"max_width":1024,"max_height":1024,"timeout_seconds":5,
                }|changes))

    async def test_image_loss_at_terminal_fences_answer_after_accounting_usage(self):
        ctx = model_context()
        url = "data:image/jpeg;base64,c3ludGhldGlj"
        validate = AsyncMock(side_effect=[None,None,ImageUnavailableError("frame_unavailable","Deleted")])
        gateway = TaskModelGateway(ctx,Upstream(),images={url:validate})
        events = []
        with self.assertRaises(ImageUnavailableError):
            async for value in gateway.stream({"model":"fixture-model","input":[{"role":"user","content":[
                {"type":"input_image","image_url":url}]}]}):
                events.append(value)
        ctx.settle.assert_awaited_once()
        self.assertEqual(validate.await_count,3)
        self.assertFalse(any(b'"type": "response.completed"' in event for event in events))


@unittest.skipUnless(os.getenv("JARVIS_SECOND_BRAIN_TEST_DB_URL") and
                     os.getenv("JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"),"Explicit manifested database required")
class StoredImageDatabaseTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.root = Path(os.environ["JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"])
        self.f = await self.enterAsyncContext(source_fixture(
            os.environ["JARVIS_SECOND_BRAIN_TEST_DB_URL"],self.root,test_name=self.id()))
        self.tasks = TaskRepository(self.f.pool,TaskLimits(lease_seconds=300))
        async with self.f.pool.acquire() as connection:
            self.session = await connection.fetchval("insert into jarvis_sessions(owner_id) values($1) returning id",self.f.owner)
        class Storage:
            bucket = "fixture-frames"
            build_storage_path = VisualFrameStorageClient.build_storage_path
            async def download_object(inner,**arguments):
                return inner.image
        self.storage = Storage()
        self.storage.image = pixels()
        now,visual_session = datetime.now(UTC),uuid4()
        path = self.storage.build_storage_path(owner_id=self.f.owner,session_id=self.session,
                                               visual_session_id=visual_session,frame_sequence=0)
        frames = VisualFrameRepository(self.f.pool)
        pending = await frames.create_pending_frame(
            owner_id=self.f.owner,session_id=self.session,audio_session_id=None,visual_session_id=visual_session,
            frame=ValidatedVisualFrame(0,now,"glasses","image/jpeg",640,220,self.storage.image),
            server_received_at=now,storage_bucket=self.storage.bucket,storage_path=path,
            raw_image_expires_at=now+timedelta(minutes=5),
        )
        self.frame = await frames.mark_frame_available(owner_id=self.f.owner,frame_id=pending.id,
                    stored_object=VisualFrameStorageObject(self.storage.bucket,path,len(self.storage.image)))
        self.service = StoredFrameImageService(self.f.pool,self.storage,max_bytes=512000,max_width=1024,
                                               max_height=1024,timeout_seconds=5)

    def reference(self, frame=None, *, current=False):
        frame = frame or self.frame
        values = dict(frame_id=frame.id,owner_id=self.f.owner,session_id=frame.session_id,
            source="current" if current else "historical",captured_at=frame.captured_at,
            content_type=frame.content_type,width=frame.width,height=frame.height,
            raw_image_expires_at=frame.raw_image_expires_at,visual_session_id=frame.visual_session_id,
            frame_sequence=frame.frame_sequence,server_received_at=frame.server_received_at)
        if current:
            values.update(stream_connection_id=uuid4(),sampled_at=frame.server_received_at,
                          sampled_monotonic=1e12)
        return ImageReference(**values)

    async def prepared(self, reference):
        repository = ContextRepository(self.tasks)
        event = await repository.append(self.f.owner,ContextIngress(
            session_id=self.session,source_namespace="fixture-stored-image",source_event_id=str(uuid4()),
            receipt_id=str(uuid4()),kind="observation",trust="sensor",payload={"images":[reference.to_metadata()]}))
        task = await self.tasks.create(self.f.owner,TaskCreate(request_id=uuid4(),title="Synthetic image",
            requested_outcome="Read the selected snapshot.",session_id=self.session),model="fixture-model")
        claim = await self.tasks.claim_next("stored-image-fixture",("frontier",))
        ctx = TaskRunner(self.tasks,{}).context(claim)
        from app.tasks.context import TaskContextBuilder
        await TaskContextBuilder(repository).build(claim)
        return ctx,event,TaskContextImages(repository,stored=self.service)

    async def test_current_reference_becomes_exact_snapshot_without_foreign_clock_or_live_claim(self):
        original = self.reference(current=True)
        ctx,event,images = await self.prepared(original)
        selection = await images.select(ctx.claim,event.id)
        self.assertEqual(selection["image_view"],"captured_snapshot")
        self.assertFalse(selection["current_view_verified"])
        self.assertEqual(selection["original_source"],"current")
        self.assertEqual(selection["image"]["source"],"historical")
        self.assertNotIn("sampled_monotonic",selection["image"])
        self.assertNotIn("stream_connection_id",selection["image"])
        resolved = await images.resolve(ctx.claim,selection)
        self.assertEqual(resolved.reference.frame_id,original.frame_id)
        self.assertEqual(resolved.reference.captured_at,original.captured_at)
        self.assertEqual(resolved.image_bytes,self.storage.image)

    async def test_owner_session_and_metadata_substitution_are_denied(self):
        reference = self.reference()
        for changed in (replace(reference,owner_id=self.f.other),replace(reference,session_id=uuid4()),
                        replace(reference,width=600),replace(reference,frame_sequence=9)):
            with self.subTest(reference=changed),self.assertRaises(ImageUnavailableError):
                await self.service.resolve_image(owner_id=self.f.owner,reference=changed)

    async def test_retention_change_after_download_blocks_queued_pixels(self):
        resolved = await self.service.resolve_image(owner_id=self.f.owner,reference=self.reference())
        async with self.f.pool.acquire() as connection:
            await connection.execute("update visual_frames set raw_image_status='deleted' where id=$1",self.frame.id)
        with self.assertRaises(ImageUnavailableError):
            await resolved.revalidate()

    async def test_database_expiry_wins_even_if_worker_wall_clock_rolls_back(self):
        reference = self.reference()
        self.service.now = lambda:datetime.now(UTC)-timedelta(days=1)
        async with self.f.pool.acquire() as connection:
            await connection.execute("update visual_frames set raw_image_expires_at=clock_timestamp()-interval '1 second' where id=$1",
                                     self.frame.id)
        with self.assertRaises(ImageUnavailableError):
            await self.service.resolve_image(owner_id=self.f.owner,reference=reference)

    async def test_storage_locator_substitution_and_invalid_pixels_are_rejected(self):
        self.storage.image = b"not a JPEG"
        with self.assertRaises(ImageUnavailableError):
            await self.service.resolve_image(owner_id=self.f.owner,reference=self.reference())
        self.storage.image = pixels()
        async with self.f.pool.acquire() as connection:
            await connection.execute("update visual_frames set storage_path='someone-elses/object.jpg' where id=$1",self.frame.id)
        with self.assertRaises(ImageUnavailableError):
            await self.service.resolve_image(owner_id=self.f.owner,reference=self.reference())

    async def test_record_associated_history_keeps_existing_relation_authorization(self):
        reference = replace(self.reference(),record_type="fact",record_id=uuid4())
        ctx,event,images = await self.prepared(reference)
        historical = SimpleNamespace(resolve_image=AsyncMock(side_effect=ImageUnavailableError("frame_unavailable","Forgotten")))
        images.historical = historical
        selection = await images.select(ctx.claim,event.id)
        with self.assertRaises(ImageUnavailableError):
            await images.resolve(ctx.claim,selection)
        historical.resolve_image.assert_awaited_once()

    @unittest.skipUnless(os.getenv("JARVIS_HERMES_PYTHON") and os.getenv("JARVIS_S04_STORAGE_LIVE")=="1",
                         "Explicit actual storage and native worker opt-in required")
    async def test_separate_cli_reads_real_stored_current_and_historical_snapshots(self):
        settings = JarvisSettings()
        labels = ["HARBOR "+str(uuid4().int%90000+10000),"CANYON "+str(uuid4().int%90000+10000)]
        async with stored_image_fixture(self.f,self.root,settings,session_id=self.session,
                                        images=[pixels(label) for label in labels]) as records:
            repository = ContextRepository(self.tasks)
            events = []
            for index,frame in enumerate(records):
                events.append(await repository.append(self.f.owner,ContextIngress(
                    session_id=self.session,source_namespace="fixture-cli-pixels",source_event_id=str(uuid4()),
                    receipt_id=str(uuid4()),kind="observation",trust="sensor",
                    payload={"images":[self.reference(frame,current=index==0).to_metadata()]})))
            await self.run_cli(labels,records,events,settings)

    async def run_cli(self, labels, records, events, settings, *, reasoning_effort="medium"):
        model = os.environ["JARVIS_S04_MODEL"]
        state = self.root/("cli-"+uuid4().hex)
        state.mkdir(mode=0o700)
        async with self.f.pool.acquire() as connection:
            schema = await connection.fetchval("select current_schema()")
        url = urlsplit(os.environ["JARVIS_SECOND_BRAIN_TEST_DB_URL"])
        query = dict(parse_qsl(url.query))
        query["search_path"]=schema+",pg_catalog,public,extensions"
        dsn = urlunsplit((url.scheme,url.netloc,url.path,urlencode(query),url.fragment))
        environment = {
            "PATH":"/usr/bin:/bin","PYTHONPATH":str(Path(__file__).resolve().parents[1]),
            "PYTHONUNBUFFERED":"1","JARVIS_OWNER_ID":self.f.owner,"SUPABASE_DB_URL":dsn,
            "OPENAI_API_KEY":settings.OPENAI_API_KEY,"SUPABASE_URL":settings.SUPABASE_URL,
            "SUPABASE_SERVICE_ROLE_KEY":settings.SUPABASE_SERVICE_ROLE_KEY,
            "SUPABASE_STORAGE_BUCKET":settings.SUPABASE_STORAGE_BUCKET,
            "SECOND_BRAIN_ENABLED":"true","SECOND_BRAIN_TASKS_ENABLED":"true","SECOND_BRAIN_FRONTIER_ENABLED":"true",
            "SECOND_BRAIN_FRONTIER_MODEL":model,"SECOND_BRAIN_FRONTIER_REASONING_EFFORT":reasoning_effort,
            "SECOND_BRAIN_HERMES_PYTHON":os.environ["JARVIS_HERMES_PYTHON"],
            "SECOND_BRAIN_HERMES_WORK_ROOT":str(state/"native-state"),"SECOND_BRAIN_WORKER_HOST_ID":"fixture-image-cli",
            "SECOND_BRAIN_WORKER_HEARTBEAT_SECONDS":"1","SECOND_BRAIN_POLL_SECONDS":"0.2",
            "SECOND_BRAIN_REQUEST_OUTPUT_TOKENS":"4096","SECOND_BRAIN_KNOWLEDGE_ENABLED":"false",
        }
        process = await asyncio.create_subprocess_exec(sys.executable,"-m","app.tasks.worker","run",
            cwd=state,env=environment,stdout=asyncio.subprocess.PIPE,stderr=asyncio.subprocess.PIPE,start_new_session=True)
        drain = asyncio.create_task(process.communicate())
        resource = {"format":"jarvis.worker_role_fixture.v1","pid":process.pid,"state":str(state),"phase":"running"}
        receipt = state/"worker-resources.json"
        receipt.write_text(json.dumps(resource))
        registry = TaskWorkerRegistry(self.f.pool)
        try:
            for _ in range(180):
                if await registry.ready(self.f.owner,"frontier",model): break
                if process.returncode is not None: self.fail("Image worker stopped before readiness")
                await asyncio.sleep(.25)
            self.assertTrue(await registry.ready(self.f.owner,"frontier",model))
            service = TaskService(self.tasks,{"frontier":ConfiguredTaskProvider(model)},registry=registry)
            task = await service.create(self.f.owner,TaskCreate(
                request_id=uuid4(),title="Read two stored labels",session_id=self.session,
                requested_outcome="Read the exact label in each of the two original images referenced by the observations. "
                    "Use the image tool for both, and quote both labels with their source citations. "
                    "These are captured snapshots, not a claim that the camera is currently live.",
            ))
            for _ in range(600):
                current = await self.tasks.get(self.f.owner,task.id)
                if current.status in (TaskStatus.SUCCEEDED,TaskStatus.PAUSED,TaskStatus.FAILED,TaskStatus.WAITING): break
                await asyncio.sleep(.25)
            resource.update(task_id=str(task.id), task_status=current.status.value, task_reason=current.reason)
            if current.result_payload_id is not None:
                async with self.f.pool.acquire() as connection:
                    payload = await connection.fetchval(
                        "select content::text from jarvis_context_payloads where owner_id=$1 and id=$2",
                        self.f.owner, current.result_payload_id)
                resource["observed_result"] = json.loads(payload) if payload is not None else None
            self.assertEqual(current.status,TaskStatus.SUCCEEDED,current.reason)
            async with self.f.pool.acquire() as connection:
                result = json.loads(await connection.fetchval(
                    "select content::text from jarvis_context_payloads where owner_id=$1 and id=$2",
                    self.f.owner,current.result_payload_id))
                for frame in records:
                    self.assertTrue(await connection.fetchval(
                        "select exists(select 1 from jarvis_context_payloads where owner_id=$1 "
                        "and content->>'format'='jarvis.hermes.model_request.v1' and content->'images' @> $2::jsonb)",
                        self.f.owner,json.dumps([{"frame_id":str(frame.id),"source":"historical"}])))
            for label in labels:
                self.assertIn(label,result["answer"])
            self.assertEqual({item["event_id"] for item in result["evidence"]},{str(event.id) for event in events})
            self.assertGreaterEqual(sum(item["name"]=="read_task_context_image" for item in result["tool_receipts"]),2)
            resource.update(task_id=str(task.id),result=result)
        finally:
            if process.returncode is None:
                process.send_signal(signal.SIGTERM)
                try: await asyncio.wait_for(asyncio.shield(drain),12)
                except TimeoutError: os.killpg(process.pid,signal.SIGKILL)
            stdout,stderr = await drain
            def sanitized(value):
                text = value.decode(errors="replace")[-12000:]
                for secret in (settings.OPENAI_API_KEY,settings.SUPABASE_SERVICE_ROLE_KEY,dsn,
                               os.environ["JARVIS_SECOND_BRAIN_TEST_DB_URL"]):
                    if secret: text=text.replace(secret,"[redacted]")
                return text
            resource.update(phase="stopped",exit_code=process.returncode,
                            stdout=sanitized(stdout),stderr=sanitized(stderr))
            receipt.write_text(json.dumps(resource))
        self.assertEqual(process.returncode,0)
        self.assertFalse(await registry.ready(self.f.owner,"frontier",model))
