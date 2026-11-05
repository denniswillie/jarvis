"""Core-only generated MP4 source -> Files HTTP -> C0 receipt qualification.

Requires manifested disposable SQL plus an explicitly hash-pinned real synthetic
MP4 and provisioned ffprobe. Parent grant admission and the Phone saved ACK are
explicit fixture inputs; exact snapshot preparation uses the real publisher.
This does not prove worker/model admission, rendering,
cloud storage, sensor provenance, real network memory use or native Phone save.
No dummy bytes/inspection fallback and no production configuration changes.
"""
import hashlib
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from uuid import UUID, uuid4

from fastapi import FastAPI
from fastapi.responses import JSONResponse
import httpx

from app.errors import JarvisHTTPError
from app.files.generated import ArtifactPolicy, ArtifactRequest
from app.files.local_file import CHUNK_BYTES, snapshot_file
from app.files.media_inspection import inspect_mp4_file
from app.files.router import router as files_router
from app.files.tools import FileToolModule
from app.files.transfer import FileTransferPolicy
from app.files.workspace_access import WorkspaceFileAccess
from app.knowledge.files import SourceFileRepository
from app.knowledge.product_files import GENERATED_ORIGINAL_FORMAT
from app.knowledge.session_areas import SessionKnowledgeAreas
from app.runtime.feature_config import MemoryAuthority
from app.runtime.workspace_delivery import WorkspaceContentExporter, WorkspaceContentResolver
from app.settings import JarvisSettings, get_settings
from app.tasks.generated_artifacts import GeneratedArtifactPublisher
from app.tasks.memory_evidence import TaskMemoryEvidence
from app.tasks.models import json_object
from app.tasks.runner import TaskRunner
from app.tasks.tool_bridge import TaskToolBridge
from app.tasks.worker import WorkerContextFactory
from app.workspace.models import WorkspaceContext
from app.workspace.presentation import LocalReceipt, PresentationRepository, PresentationService
from app.workspace.router import router as workspace_router
from evals.second_brain.sandbox_fixture import sandbox_task_fixture
from test_file_backed_publication_sql import DiskStorage


LIMIT = 20_000_000  # Explicit fixture ceiling, not a qualified deployment limit.


def sha256_file(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


@unittest.skipUnless(os.getenv("JARVIS_SECOND_BRAIN_TEST_DB_URL") and
                     os.getenv("JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"),
                     "Explicit manifested Core database lane required")
class GeneratedMediaSourceSQLTests(unittest.IsolatedAsyncioTestCase):
    storage_class = DiskStorage

    def publication_allowance(self):
        return ArtifactPolicy(max_files=1, max_file_bytes=LIMIT, max_total_bytes=LIMIT,
                              formats=("mp4",), transfer_policy="mp4_output_v1")

    async def asyncSetUp(self):
        names = ("JARVIS_MEDIA_FIXTURE_FILE", "JARVIS_MEDIA_FIXTURE_SHA256",
                 "JARVIS_MEDIA_FFPROBE_PATH", "JARVIS_MEDIA_FFPROBE_SHA256")
        self.assertTrue(all(os.getenv(name) for name in names), "All four pinned media/probe inputs are required")
        self.original = Path(os.environ[names[0]])
        self.probe = Path(os.environ[names[2]])
        self.assertTrue(self.original.is_absolute() and self.probe.is_absolute(), "Explicit absolute paths required")
        self.assertTrue(self.original.is_file() and self.probe.is_file(), "Provisioned media and executable required")
        self.assertFalse(self.original.is_symlink() or self.probe.is_symlink(), "Pinned originals must be regular files")
        self.expected_hash = os.environ[names[1]]
        self.assertEqual(sha256_file(self.original), self.expected_hash)
        self.assertEqual(sha256_file(self.probe), os.environ[names[3]])
        self.assertGreater(self.original.stat().st_size, 8 * 1024 * 1024)
        self.policy = FileTransferPolicy("mp4_output_v1", LIMIT)
        self.local = await snapshot_file(self.original, max_bytes=LIMIT, expected_sha256=self.expected_hash)
        self.addCleanup(self.local.close)

        async def inspector(local_file, policy):
            # The immutable executable is rechecked at every fixture inspection.
            self.assertEqual(sha256_file(self.probe), os.environ[names[3]])
            return await inspect_mp4_file(local_file, policy, ffprobe_path=self.probe, timeout=10)

        self.inspector = inspector
        self.measured = await inspector(self.local, self.policy)
        # Fail input/probe qualification before creating disposable SQL state.
        self.f = await self.enterAsyncContext(sandbox_task_fixture(
            os.environ["JARVIS_SECOND_BRAIN_TEST_DB_URL"],
            Path(os.environ["JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"]), test_name=self.id(),
            with_session=True, requested_outcome="Fixture: retain the exact synthetic generated MP4."))
        self.context = TaskRunner(self.f.tasks, {}).context(self.f.claim)
        await WorkerContextFactory(self.f.source.pool)(self.context)
        directory = tempfile.TemporaryDirectory(prefix="jarvis-generated-media-sql-")
        self.addCleanup(directory.cleanup)
        self.storage = self.storage_class(Path(directory.name).resolve() / "objects")
        self.storage.lose_ack = False
        evidence = TaskMemoryEvidence(self.context, self.f.source.sources, area_id=self.f.source.area)
        self.files = SourceFileRepository(self.f.source.sources, self.storage, owner_id=self.f.source.owner,
            area_id=self.f.source.area, task_evidence=evidence, media_policy=self.policy, media_inspector=inspector)
        self.bridge = await TaskToolBridge.from_modules(self.context, (FileToolModule(self.files, self.storage),),
            read_tools=("list_files", "get_file_metadata"), memory_evidence=evidence,
            receipt_revalidator=self.files.revalidate_receipt)
        self.allowance = self.publication_allowance()
        self.request = ArtifactRequest(operation_key="fixture-exact-mp4", filename="synthetic-generated.mp4",
            format="mp4", description="Synthetic generated MP4 fixture; not camera or model output.")
        async with self.f.tasks.transaction(self.f.source.owner) as connection:
            scope = await self.f.tasks.payload(connection, self.f.source.owner, self.allowance.model_dump(mode="json"))
            self.parent = await connection.fetchval("""insert into jarvis_operation_grants
                (owner_id,capability,operation_key,request_event_id,request_revision,scope_revision,scope_payload_id,expires_at)
                values($1,'files.publish_generated',$2,$3,1,1,$4,clock_timestamp()+interval '1 hour') returning id""",
                self.f.source.owner, str(uuid4()), self.f.event, scope)
            await connection.execute("insert into jarvis_task_grants(owner_id,task_id,grant_id) values($1,$2,$3)",
                                     self.f.source.owner, self.f.task.id, self.parent)
        self.publisher = GeneratedArtifactPublisher(self.bridge, self.files, self.parent)
        # Parent allowance is explicit fixture authorization. Everything after
        # it, including inspection, exact snapshot/grant and evidence freezing,
        # follows the production fresh-producer path before the replay checks.
        action, self.snapshot = await self.publisher.prepare(self.request, self.local)
        self.snapshot_grant = action.grant_id
        self.assertEqual(self.snapshot.parent_grant_id, self.parent)
        self.assertEqual(self.snapshot.task_id, self.f.task.id)

    def application(self, owner, area):
        selections = SessionKnowledgeAreas(self.f.source.sources, self.f.tasks, area)
        repository = SourceFileRepository(self.f.source.sources, self.storage, owner_id=owner, area_id=area,
            selections=selections, media_policy=self.policy, media_inspector=self.inspector)
        access = WorkspaceFileAccess(repository)
        presentations = PresentationService(PresentationRepository(self.f.source.pool),
            resolver=WorkspaceContentResolver(None, access), exporter=WorkspaceContentExporter(None, access, None))
        token = "fixture-media-" + uuid4().hex
        settings = JarvisSettings(_env_file=None, JARVIS_OWNER_ID=owner, JARVIS_STATIC_BEARER_TOKEN=token)
        app = FastAPI()
        app.state.runtime_services = SimpleNamespace(settings=settings, memory_authority=MemoryAuthority.HINDSIGHT,
            tool_modules=(FileToolModule(repository, self.storage),), workspace_presentations=presentations)
        app.dependency_overrides[get_settings] = lambda: settings
        app.include_router(files_router)
        app.include_router(workspace_router)

        @app.exception_handler(JarvisHTTPError)
        async def failure(request, error):
            return JSONResponse({"error": {"code": error.code}}, status_code=error.status_code)

        traces = []

        async def observed(scope, receive, send):
            # ASGITransport itself buffers: observe actual response chunks here.
            # This proves the route's streaming behavior, not network peak RAM.
            status, sizes, checksum = None, [], hashlib.sha256()

            async def record(message):
                nonlocal status
                if message["type"] == "http.response.start":
                    status = message["status"]
                if (message["type"] == "http.response.body" and status == 200
                        and scope["path"].endswith("/content")):
                    body = message.get("body", b"")
                    if body:
                        sizes.append(len(body)); checksum.update(body)
                await send(message)

            await app(scope, receive, record)
            if sizes:
                traces.append({"sizes": sizes, "sha256": checksum.hexdigest()})

        return repository, presentations, observed, token, traces

    async def test_exact_generated_source_http_download_and_c0_receipt(self):
        action, frozen = await self.publisher.prepare(self.request, self.local)
        self.assertEqual((action.grant_id, frozen), (self.snapshot_grant, self.snapshot))
        saved = await self.publisher.publish(self.request, self.local, provider_call_id="fixture-generated-mp4-1")
        self.assertEqual(saved["status"], "success", saved)
        replay = await self.publisher.publish(self.request, provider_call_id="fixture-generated-mp4-replay")
        self.assertTrue(replay["task_operation"]["replayed"])
        self.assertEqual(replay["file"], saved["file"])
        self.assertEqual(self.storage.uploads, 1)
        file_id = UUID(saved["file"]["id"])
        owner, session = self.f.source.owner, self.f.task.session_id
        repository, presentations, app, token, traces = self.application(owner, self.f.source.area)
        file = await repository.for_session(session).get_file(owner_id=owner, file_id=file_id)
        version = file.memory_source["document_id"]
        source = await self.f.source.sources.read(owner, self.f.source.area, UUID(file.memory_source["source_id"]))
        self.assertEqual((source.source_type, source.trust), ("generated_file", "assistant_output"))
        self.assertEqual(source.document_id, version)
        self.assertEqual(source.content["format"], GENERATED_ORIGINAL_FORMAT)
        self.assertEqual(source.content["content_sha256"], self.expected_hash)
        self.assertEqual(source.content["validation"]["media_inspection"], self.measured.model_dump(mode="json"))
        self.assertNotIn("pixel_origin", source.content)
        self.assertNotIn("image", source.content)
        self.assertEqual(sha256_file(self.storage.root / file.storage_path), self.expected_hash)
        async with self.f.source.pool.acquire() as connection:
            action_row = await connection.fetchrow("""select a.status,p.content from jarvis_task_actions a
                join jarvis_context_payloads p on p.owner_id=a.owner_id and p.id=a.receipt_payload_id
                where a.owner_id=$1 and a.task_id=$2""", owner, self.f.task.id)
            self.assertEqual(action_row["status"], "accepted")
            self.assertEqual(json_object(action_row["content"])["result"]["file"]["id"], str(file_id))
            self.assertEqual(await connection.fetchval("select count(*) from jarvis_task_actions where owner_id=$1", owner), 1)
            foreign_area = await connection.fetchval("""insert into jarvis_knowledge_areas(owner_id,name,domain)
                values($1,'Foreign media fixture','test') returning id""", self.f.source.other)
            foreign_session = await connection.fetchval("insert into jarvis_sessions(owner_id) values($1) returning id",
                                                        self.f.source.other)

        params = {"session_id": str(session), "version_id": version}
        base = "/files/" + str(file_id)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://fixture.invalid",
                                     headers={"Authorization": "Bearer " + token}) as client:
            metadata = await client.get(base, params=params)
            self.assertEqual(metadata.status_code, 200, metadata.text)
            metadata = metadata.json()["file"]
            self.assertEqual((metadata["version_id"], metadata["content_sha256"], metadata["transfer_limit_bytes"]),
                             (version, self.expected_hash, LIMIT))
            for suffix in ("", "/content"):
                for invalid, status in ((params | {"session_id": str(foreign_session)}, 404),
                                         (params | {"version_id": "wrong-version"}, 404),
                                         ({"version_id": version}, 403)):
                    response = await client.get(base + suffix, params=invalid)
                    self.assertEqual(response.status_code, status, response.text)
            self.assertEqual((await client.get(base + "/content", params={"session_id": str(session)})).status_code, 422)
            self.assertEqual((await client.get(base, params=params, headers={"Authorization": "Bearer wrong"})).status_code, 403)
            self.assertEqual(traces, [])
            _, _, other_app, other_token, _ = self.application(self.f.source.other, foreign_area)
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=other_app), base_url="http://fixture.invalid",
                    headers={"Authorization": "Bearer " + other_token}) as foreign_client:
                for suffix in ("", "/content"):
                    response = await foreign_client.get(base + suffix,
                        params={"session_id": str(foreign_session), "version_id": version})
                    self.assertEqual(response.status_code, 404, response.text)

            # A synthetic admitted presentation context; actual C0 service,
            # durable SQL request, authenticated polling/ACK and exact byte route.
            ctx = WorkspaceContext(owner, session, uuid4())
            await presentations.present(ctx, action="download", artifact_kind="file", artifact_id=file_id, version_id=version)
            pending = await presentations.repository.get(ctx, ctx.operation_id)
            self.assertTrue(pending.content_ready)
            self.assertEqual((pending.transfer_policy, pending.transfer_limit_bytes, pending.byte_count),
                             ("mp4_output_v1", LIMIT, self.local.size_bytes))
            self.assertEqual(pending.content_sha256, self.expected_hash)
            await presentations.present(ctx, action="download", artifact_kind="file", artifact_id=file_id, version_id=version)
            self.assertEqual(await presentations.repository.get(ctx, ctx.operation_id), pending)
            listed = await client.get("/workspace/requests", params={"session_id": str(session)})
            self.assertEqual(listed.status_code, 200, listed.text)
            self.assertEqual(listed.json()["items"], [pending.wire()])
            ack_url = "/workspace/requests/" + str(ctx.operation_id) + "/ack"
            claimed = await client.post(ack_url, json={"session_id": str(session), "state": "claimed"})
            self.assertEqual(claimed.status_code, 200, claimed.text)
            self.assertEqual(claimed.json()["request"]["state"], "claimed")

            digest, size = hashlib.sha256(), 0
            async with client.stream("GET", base + "/content", params=params) as response:
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.headers["Content-Type"], "video/mp4")
                self.assertEqual(response.headers["Content-Length"], str(self.local.size_bytes))
                self.assertEqual(response.headers["X-Jarvis-Version-Id"], version)
                self.assertEqual(response.headers["X-Content-SHA256"], self.expected_hash)
                self.assertEqual(response.headers["ETag"], '"' + self.expected_hash + '"')
                self.assertEqual(response.headers["Cache-Control"], "private, no-store")
                async for chunk in response.aiter_bytes(chunk_size=CHUNK_BYTES):
                    digest.update(chunk); size += len(chunk)
            self.assertEqual((size, digest.hexdigest()), (self.local.size_bytes, self.expected_hash))
            self.assertEqual(len(traces), 1)
            self.assertEqual(sum(traces[0]["sizes"]), size)
            self.assertLessEqual(max(traces[0]["sizes"]), CHUNK_BYTES)
            self.assertGreater(len(traces[0]["sizes"]), 128)
            self.assertEqual(traces[0]["sha256"], self.expected_hash)
            receipt = LocalReceipt(file_id=file_id, version_id=version, byte_count=size,
                                   content_sha256=digest.hexdigest(), destination="app_storage")
            body = {"session_id": str(session), "state": "saved_local", "local_receipt": receipt.model_dump(mode="json")}
            bad = body | {"local_receipt": receipt.model_dump(mode="json") | {"content_sha256": "0" * 64}}
            self.assertEqual((await client.post(ack_url, json=bad)).status_code, 409)
            acknowledged = await client.post(ack_url, json=body)
            self.assertEqual(acknowledged.status_code, 200, acknowledged.text)
            self.assertEqual((await client.post(ack_url, json=body)).json(), acknowledged.json())
            persisted = await presentations.repository.get(ctx, ctx.operation_id)
            self.assertEqual((persisted.state, persisted.local_receipt), ("saved_local", receipt))
            self.assertEqual((persisted.version_id, persisted.transfer_policy, persisted.transfer_limit_bytes),
                             (version, "mp4_output_v1", LIMIT))
            async with self.f.source.pool.acquire() as connection:
                raw = await connection.fetchval("""select payload from jarvis_workspace_requests
                    where owner_id=$1 and session_id=$2 and request_id=$3""", owner, session, ctx.operation_id)
            self.assertEqual(json_object(raw), persisted.wire())
        self.assertEqual(self.storage.uploads, 1)
        self.assertEqual(sha256_file(self.original), self.expected_hash)
        self.assertEqual(sha256_file(self.storage.root / file.storage_path), self.expected_hash)


if __name__ == "__main__":
    unittest.main()
