"""Core-only manifested recording retention qualification; never auto-run live.

Exact synthetic MP4 and actual pinned ffprobe are mandatory. Server command
carriers and client timing reports are explicit fixture inputs, not proof of
native voice admission, physical capture, Jarvis hearing, lock continuity or
either full concert/gym demo. DiskStorage is not cloud/deployed storage. Actual
admission, canonical context, source/Files adoption and HTTP routes are retained.
Uses the centrally discovered migrations through the existing fixture only.
"""
import hashlib
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from uuid import UUID, uuid4

from fastapi import FastAPI
from fastapi.responses import JSONResponse
import httpx

from app.errors import JarvisHTTPError
from app.files.local_file import CHUNK_BYTES, snapshot_file
from app.files.media_inspection import inspect_mp4_file
from app.files.recording_ingress import RecordingClientReport, RecordingIngress, RecordingUpload
from app.files.recording_router import router as recording_router
from app.files.router import router as files_router
from app.files.tools import FileToolModule
from app.files.transfer import FileTransferPolicy
from app.knowledge.files import SourceFileRepository
from app.knowledge.models import SourceError
from app.knowledge.product_files import FILE_NAMESPACE, RECORDING_ORIGINAL_FORMAT
from app.knowledge.session_areas import SessionKnowledgeAreas
from app.live_state.admission import CommandAdmission
from app.migrations.runner import load_migrations
from app.recordings.admission import RecordingAdmission
from app.runtime.feature_config import MemoryAuthority, TaskLimits
from app.settings import JarvisSettings, get_settings
from app.tasks.models import json_object
from app.tasks.repository import TaskRepository
from evals.second_brain.source_fixture import source_fixture
from test_file_backed_publication_sql import DiskStorage


EXPECTED_SIZE = 15_431_453
EXPECTED_SHA256 = "f4f885c568d5bee8310e1dea75ec5b36860c954b03dc37d2654e9dee4a001ce5"
FIXTURE_LIMIT = 20_000_000  # Explicit fixture bound, not a verified deployment.


def file_sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class TrackedDiskStorage(DiskStorage):
    """The existing disk double, with handles observed only for cleanup proof."""
    def __init__(self, root):
        super().__init__(root)
        self.lose_ack = False
        self.returned = []

    async def read_object_file(self, **kwargs):
        local = await super().read_object_file(**kwargs)
        if local is not None:
            self.returned.append(local)
        return local

    def close_handles(self):
        for local in self.returned:
            local.close()


@unittest.skipUnless(os.getenv("JARVIS_SECOND_BRAIN_TEST_DB_URL") and
                     os.getenv("JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"),
                     "Explicit manifested Core SQL execution lane required")
class RecordingIngressSQLTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        names = ("JARVIS_MEDIA_FIXTURE_FILE", "JARVIS_MEDIA_FIXTURE_SHA256",
                 "JARVIS_MEDIA_FFPROBE_PATH", "JARVIS_MEDIA_FFPROBE_SHA256")
        self.assertTrue(all(os.getenv(name) for name in names), "All four pinned media/probe inputs are required")
        self.original, self.probe = Path(os.environ[names[0]]), Path(os.environ[names[2]])
        manifest_dir = Path(os.environ["JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"])
        self.assertTrue(manifest_dir.is_absolute(), "An absolute manifested fixture directory is required")
        for path in (self.original, self.probe):
            self.assertTrue(path.is_absolute() and path.is_file() and not path.is_symlink(), "Pinned regular files required")
        self.assertEqual(os.environ[names[1]], EXPECTED_SHA256)
        self.assertEqual(file_sha256(self.original), EXPECTED_SHA256)
        self.assertEqual(self.original.stat().st_size, EXPECTED_SIZE)
        self.probe_sha = os.environ[names[3]]
        self.assertEqual(file_sha256(self.probe), self.probe_sha)
        self.assertIn((48, "recording_admission"), {(item.version, item.name) for item in load_migrations()},
                      "Core must release auto-discovered0048 and canonical schema qualification before execution")
        self.policy = FileTransferPolicy("mp4_capture_v1", FIXTURE_LIMIT)
        self.inspected = []

        async def inspector(local, policy):
            self.assertEqual(file_sha256(self.probe), self.probe_sha)
            self.inspected.append(local.path)
            return await inspect_mp4_file(local, policy, ffprobe_path=self.probe, timeout=10)

        self.inspector = inspector
        local = await snapshot_file(self.original, max_bytes=FIXTURE_LIMIT,
                                    expected_size=EXPECTED_SIZE, expected_sha256=EXPECTED_SHA256)
        try:
            self.measured = await inspector(local, self.policy)
            self.assertEqual((self.measured.video_codec, self.measured.audio_codec), ("h264", "aac"))
        finally:
            local.close()
        # Input/probe qualification precedes any disposable SQL allocation.
        self.f = await self.enterAsyncContext(source_fixture(os.environ["JARVIS_SECOND_BRAIN_TEST_DB_URL"],
                                                            manifest_dir, test_name=self.id()))
        self.tasks = TaskRepository(self.f.pool, TaskLimits())
        async with self.f.pool.acquire() as connection:
            self.session = await connection.fetchval("insert into jarvis_sessions(owner_id) values($1) returning id", self.f.owner)
            self.foreign_session = await connection.fetchval("insert into jarvis_sessions(owner_id) values($1) returning id", self.f.other)
        self.selections = SessionKnowledgeAreas(self.f.sources, self.tasks, self.f.area)
        self.admissions = RecordingAdmission(self.f.sources, self.tasks, self.selections, owner_id=self.f.owner)
        directory = tempfile.TemporaryDirectory(prefix="jarvis-recording-sql-")
        self.addCleanup(directory.cleanup)
        self.storage = TrackedDiskStorage(Path(directory.name).resolve() / "objects")
        self.addCleanup(self.storage.close_handles)
        self.ingress = RecordingIngress(self.f.sources, self.storage, self.admissions,
                                        policy=self.policy, inspect_file=inspector)
        self.upload = RecordingUpload(filename="synthetic-recording.mp4",
            description="Synthetic client-reported recording fixture; not physical glasses capture.",
            size_bytes=EXPECTED_SIZE, content_sha256=EXPECTED_SHA256,
            client_report=RecordingClientReport(generation_id=uuid4(), timeline_end_ms=self.measured.duration_ms,
                video_frames=1, source_audio_frames=0, padded_audio_frames=0))
        self.token = "fixture-recording-" + uuid4().hex
        self.body_reads = 0
        self.content_traces = []
        self.app = self.application()

    async def asyncTearDown(self):
        self.assertEqual(file_sha256(self.original), EXPECTED_SHA256)
        self.assertTrue(all(not path.exists() for path in self.inspected), "Every inspected private snapshot must close")
        self.assertTrue(all(not local.path.exists() for local in self.storage.returned), "Every storage snapshot must close")

    def command(self, capability, arguments):
        # Explicit synthetic trusted carrier; no claim of a native/provider call.
        return CommandAdmission(owner_id=self.f.owner, session_id=self.session,
            provider_session_key="fixture-recording-session", provider_call_id="fixture-" + uuid4().hex,
            tool_name="fixture_recording_command", capability=capability, normalized_arguments=arguments)

    async def admitted(self):
        command = self.command("recordings.start", {"include_jarvis_audio": True})
        receipt = await self.admissions.admit(command, area_id=self.f.area, policy=self.policy)
        self.assertEqual(receipt.outcome, "admitted")
        self.assertIsNone(receipt.file_id)
        self.assertEqual(receipt.scope.recording_id, command.operation_id)
        return receipt.scope

    async def body(self):
        with self.original.open("rb") as stream:
            while chunk := stream.read(CHUNK_BYTES):
                self.body_reads += 1
                yield chunk

    async def forbidden_body(self):
        raise AssertionError("An existing/denied finalization must not consume client media again")
        yield b""  # Makes this an async iterator without ever supplying bytes.

    def upload_headers(self, upload=None):
        return {"X-Jarvis-Recording-Manifest": json.dumps((upload or self.upload).model_dump(mode="json"),
                    ensure_ascii=True, separators=(",", ":")),
                "Content-Type": "video/mp4", "Content-Length": str(EXPECTED_SIZE)}

    def application(self):
        repository = SourceFileRepository(self.f.sources, self.storage, owner_id=self.f.owner, area_id=self.f.area,
            selections=self.selections, media_policy=self.policy, media_inspector=self.inspector)
        settings = JarvisSettings(_env_file=None, JARVIS_OWNER_ID=self.f.owner, JARVIS_STATIC_BEARER_TOKEN=self.token)
        app = FastAPI()
        app.state.runtime_services = SimpleNamespace(settings=settings, memory_authority=MemoryAuthority.HINDSIGHT,
            tool_modules=(FileToolModule(repository, self.storage),), recording_ingress=self.ingress)
        app.dependency_overrides[get_settings] = lambda: settings
        app.include_router(recording_router)
        app.include_router(files_router)

        @app.exception_handler(JarvisHTTPError)
        async def failure(request, error):
            return JSONResponse({"error": {"code": error.code}}, status_code=error.status_code)

        async def observed(scope, receive, send):
            sizes, checksum, status = [], hashlib.sha256(), None

            async def record(message):
                nonlocal status
                if message["type"] == "http.response.start":
                    status = message["status"]
                if (message["type"] == "http.response.body" and status == 200
                        and scope["path"].startswith("/files/") and scope["path"].endswith("/content")):
                    body = message.get("body", b"")
                    if body:
                        sizes.append(len(body)); checksum.update(body)
                await send(message)

            await app(scope, receive, record)
            if sizes:
                self.content_traces.append({"sizes": sizes, "sha256": checksum.hexdigest()})

        return observed

    async def test_admitted_stopped_stream_saved_exact_http_retry_and_revocation(self):
        scope = await self.admitted()
        stopped = await self.admissions.stop(self.command("recordings.stop", {"recording_id": str(scope.recording_id)}))
        self.assertEqual(stopped.outcome, "capture_closed"); self.assertIsNone(stopped.file_id)
        async with self.f.sources.transaction(self.f.owner) as connection:
            await self.tasks.lock_owner(connection, self.f.owner)
            with self.assertRaises(SourceError):
                await self.admissions.require_admitted_recording(connection, self.f.owner, self.session, scope.recording_id, for_capture=True)
        params = {"session_id": str(self.session)}
        target = "/recordings/" + str(scope.recording_id) + "/content"
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app), base_url="http://fixture.invalid",
                                     headers={"Authorization": "Bearer " + self.token}) as client:
            rejected = await client.put(target, params=params,
                headers=self.upload_headers() | {"Authorization": "Bearer invalid"}, content=self.forbidden_body())
            self.assertEqual(rejected.status_code, 403)
            rejected = await client.put(target, params={"session_id": str(self.foreign_session)},
                headers=self.upload_headers(), content=self.forbidden_body())
            self.assertEqual(rejected.status_code, 404, rejected.text)
            rejected = await client.put("/recordings/" + str(uuid4()) + "/content", params=params,
                headers=self.upload_headers(), content=self.forbidden_body())
            self.assertEqual(rejected.status_code, 404, rejected.text)
            self.assertEqual(self.body_reads, 0)
            response = await client.put(target, params=params, headers=self.upload_headers(), content=self.body())
            self.assertEqual(response.status_code, 200, response.text)
            saved = response.json()
            self.assertEqual(saved["outcome"], "saved")
            self.assertFalse(saved["capture_allowed"])
            self.assertGreater(self.body_reads, 128)
            reads = self.body_reads
            again = await client.put(target, params=params, headers=self.upload_headers(), content=self.forbidden_body())
            self.assertEqual(again.status_code, 200, again.text)
            self.assertEqual(again.json()["receipt_event_id"], saved["receipt_event_id"])
            self.assertEqual(self.body_reads, reads); self.assertEqual(self.storage.uploads, 1)
            for changes in ({"description": "Different metadata"}, {"content_sha256": "0" * 64}):
                changed = self.upload.model_copy(update=changes)
                conflict = await client.put(target, params=params, headers=self.upload_headers(changed), content=self.forbidden_body())
                self.assertEqual(conflict.status_code, 409, conflict.text)
                self.assertEqual(conflict.json()["error"]["code"], "recording_finalization_conflict")

            file_id, source_id = UUID(saved["file_id"]), UUID(saved["source_id"])
            source = await self.f.sources.read(self.f.owner, self.f.area, source_id)
            self.assertEqual((source.source_type, source.trust), ("recording_file", "user_input"))
            self.assertEqual(source.document_id, saved["version_id"])
            self.assertEqual(source.content["format"], RECORDING_ORIGINAL_FORMAT)
            self.assertEqual(source.content["media_origin"], "client_reported")
            self.assertEqual(source.content["recording_id"], str(scope.recording_id))
            self.assertEqual(source.content["media_inspection"], self.measured.model_dump(mode="json"))
            self.assertEqual(source.content["content_sha256"], EXPECTED_SHA256)
            self.assertNotIn("pixel_origin", source.content); self.assertNotIn("heard_state", source.content)
            async with self.f.pool.acquire() as connection:
                recording = await connection.fetchrow("select * from jarvis_recordings where owner_id=$1 and recording_id=$2",
                                                     self.f.owner, scope.recording_id)
                publication = await connection.fetchrow("select * from jarvis_file_publications where owner_id=$1 and id=$2",
                                                       self.f.owner, UUID(saved["publication_id"]))
                self.assertEqual((publication["phase"], publication["upload_settled"], publication["file_id"]),
                                 ("available", True, file_id))
                self.assertEqual(publication["operation_key"], "recording:" + str(scope.recording_id))
                self.assertEqual((publication["transfer_policy"], publication["transfer_limit_bytes"]),
                                 ("mp4_capture_v1", FIXTURE_LIMIT))
                self.assertEqual(recording["completion_event_id"], UUID(saved["receipt_event_id"]))
                finalization = json_object(recording["final_manifest_json"])
                self.assertEqual(finalization, source.content["finalization_manifest"])
                self.assertEqual(finalization["client_report"], self.upload.client_report.model_dump(mode="json"))
                event = await connection.fetchrow("""select e.kind,e.trust,p.content from jarvis_context_events e
                    join jarvis_context_payloads p on p.owner_id=e.owner_id and p.id=e.payload_id
                    where e.owner_id=$1 and e.id=$2""", self.f.owner, recording["completion_event_id"])
                self.assertEqual((event["kind"], event["trust"]), ("tool_outcome", "tool_receipt"))
                self.assertEqual(json_object(event["content"])["file_id"], str(file_id))
                for parent, child in ((scope.admission_payload_id, publication["metadata_payload_id"]),
                                      (publication["metadata_payload_id"], source.payload_id),
                                      (source.payload_id, recording["completion_payload_id"])):
                    self.assertTrue(await connection.fetchval("""select exists(select 1 from jarvis_context_payload_dependencies
                        where owner_id=$1 and source_payload_id=$2 and dependent_payload_id=$3)""", self.f.owner, parent, child))

            exact = params | {"version_id": saved["version_id"]}
            base = "/files/" + str(file_id)
            metadata = await client.get(base, params=exact)
            self.assertEqual(metadata.status_code, 200, metadata.text)
            self.assertEqual(metadata.json()["file"]["content_sha256"], EXPECTED_SHA256)
            self.assertEqual(metadata.json()["file"]["media_inspection"], self.measured.model_dump(mode="json"))
            for invalid, status in ((exact | {"session_id": str(self.foreign_session)}, 404),
                                    (exact | {"version_id": "unavailable-exact-version"}, 404),
                                    ({"version_id": saved["version_id"]}, 403)):
                denied = await client.get(base + "/content", params=invalid)
                self.assertEqual(denied.status_code, status, denied.text)
            self.assertEqual((await client.get(base, params=exact, headers={"Authorization": "Bearer invalid"})).status_code, 403)
            self.assertEqual(self.content_traces, [])
            async with client.stream("GET", base + "/content", params=exact) as content:
                self.assertEqual(content.status_code, 200)
                self.assertEqual(content.headers["content-type"], "video/mp4")
                self.assertEqual(content.headers["content-length"], str(EXPECTED_SIZE))
                self.assertEqual(content.headers["x-content-sha256"], EXPECTED_SHA256)
                self.assertEqual(content.headers["x-jarvis-version-id"], saved["version_id"])
                self.assertEqual(content.headers["cache-control"], "private, no-store")
                size, checksum = 0, hashlib.sha256()
                async for chunk in content.aiter_bytes(chunk_size=CHUNK_BYTES):
                    size += len(chunk); checksum.update(chunk)
            self.assertEqual((size, checksum.hexdigest()), (EXPECTED_SIZE, EXPECTED_SHA256))
            self.assertEqual(len(self.content_traces), 1)
            self.assertLessEqual(max(self.content_traces[0]["sizes"]), CHUNK_BYTES)
            self.assertEqual(sum(self.content_traces[0]["sizes"]), EXPECTED_SIZE)
            self.assertEqual(self.content_traces[0]["sha256"], EXPECTED_SHA256)

        # A separate UNSAVED admission proves revocation blocks fresh effects;
        # a completed recording was already closed before revocation.
        revoked = await self.admitted()
        async with self.f.sources.transaction(self.f.owner) as connection:
            await self.tasks.lock_owner(connection, self.f.owner)
            await connection.execute("""update jarvis_context_payloads set availability='deleted',content=null
                where owner_id=$1 and id=$2""", self.f.owner, revoked.admission_payload_id)
            self.assertTrue(await connection.fetchval("select revoked from jarvis_recordings where owner_id=$1 and recording_id=$2",
                                                      self.f.owner, revoked.recording_id))
            with self.assertRaises(SourceError):
                await self.admissions.require_admitted_recording(connection, self.f.owner, self.session, revoked.recording_id, for_capture=True)
        with self.assertRaises(SourceError):
            await self.ingress.finalize(owner_id=self.f.owner, session_id=self.session, recording_id=revoked.recording_id,
                                        upload=self.upload, chunks=self.forbidden_body())
        self.assertEqual(self.storage.uploads, 1)
        self.assertEqual(file_sha256(self.storage.root / publication["storage_path"]), EXPECTED_SHA256)

    async def test_failure_after_real_completion_rolls_back_source_and_receipt_then_reconciles(self):
        scope = await self.admitted()
        complete = self.admissions.complete_in_transaction
        provisional = []

        async def fail_after_real_completion(*args, **kwargs):
            receipt = await complete(*args, **kwargs)
            provisional.append(receipt)
            raise RuntimeError("Injected fixture failure after real recording completion")

        with patch.object(self.admissions, "complete_in_transaction", new=fail_after_real_completion):
            with self.assertRaisesRegex(RuntimeError, "Injected fixture failure"):
                await self.ingress.finalize(owner_id=self.f.owner, session_id=self.session, recording_id=scope.recording_id,
                                            upload=self.upload, chunks=self.body())
        self.assertEqual(len(provisional), 1)
        async with self.f.pool.acquire() as connection:
            recording = await connection.fetchrow("select * from jarvis_recordings where owner_id=$1 and recording_id=$2",
                                                 self.f.owner, scope.recording_id)
            self.assertIsNotNone(recording["final_manifest_digest"])
            self.assertIsNone(recording["completion_event_id"]); self.assertIsNone(recording["file_id"])
            self.assertEqual(await connection.fetchval("select count(*) from jarvis_files where owner_id=$1", self.f.owner), 0)
            self.assertEqual(await connection.fetchval("select count(*) from jarvis_knowledge_sources where owner_id=$1 and namespace=$2",
                                                       self.f.owner, FILE_NAMESPACE), 0)
            self.assertEqual(await connection.fetchval("""select count(*) from jarvis_context_events
                where owner_id=$1 and source_namespace='recording-saved'""", self.f.owner), 0)
            publication = await connection.fetchrow("select * from jarvis_file_publications where owner_id=$1 and operation_key=$2",
                                                   self.f.owner, "recording:" + str(scope.recording_id))
            self.assertEqual((publication["phase"], publication["upload_settled"]), ("uploaded", True))
        self.assertEqual(self.storage.uploads, 1)
        self.assertEqual(file_sha256(self.storage.root / publication["storage_path"]), EXPECTED_SHA256)
        reads = self.body_reads
        saved = await self.ingress.finalize(owner_id=self.f.owner, session_id=self.session, recording_id=scope.recording_id,
                                           upload=self.upload, chunks=self.forbidden_body())
        self.assertEqual(saved["outcome"], "saved")
        self.assertEqual(UUID(saved["publication_id"]), publication["id"])
        self.assertEqual(UUID(saved["file_id"]), publication["file_id"])
        self.assertEqual(self.storage.uploads, 1); self.assertEqual(self.body_reads, reads)
        async with self.f.pool.acquire() as connection:
            self.assertEqual(await connection.fetchval("""select count(*) from jarvis_context_events
                where owner_id=$1 and source_namespace='recording-saved'""", self.f.owner), 1)
            self.assertEqual(await connection.fetchval("select phase from jarvis_file_publications where owner_id=$1 and id=$2",
                                                       self.f.owner, publication["id"]), "available")


if __name__ == "__main__":
    unittest.main()
