"""Prepared real-SQL/cloud/HTTP fixture for one synthetic 300-second Phone capture.

Execute only in Root's allocated lane after the native harness is ready. This
module starts no model or voice service. Synthetic admission/frame/audio input
does not establish actual glasses capture, hearing, lock continuity or editing.
The Phone must close its uploader before writing the native result marker.
"""
from __future__ import annotations

import argparse
import asyncio
from contextlib import AsyncExitStack
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import secrets
from time import monotonic
from types import SimpleNamespace
from urllib.parse import parse_qs
from uuid import UUID, uuid4

from fastapi import Depends, Request
from fastapi.responses import JSONResponse

from app.assistant.profile_service import BehaviorProfileService
from app.assistant.repository import BehaviorProfileRepository
from app.auth.client_context import storage_namespace
from app.auth.static_bearer import JarvisPrincipal, require_static_bearer
from app.files.media_runtime import prepare_media_inspector
from app.files.recording_ingress import RecordingIngress, RecordingUpload
from app.files.recording_router import router as recording_router
from app.files.storage import JarvisFileStorageClient
from app.files.tools import FileToolModule
from app.files.transfer import CAPTURE_MAX_BYTES, CAPTURE_MAX_DURATION_MS, CAPTURE_MAX_ENCODED_DURATION_MS, FileTransferPolicy, MP4_CAPTURE_POLICY
from app.knowledge.files import SourceFileRepository
from app.knowledge.notes import SourceNoteRepository
from app.knowledge.session_areas import SessionKnowledgeAreas
from app.notes.tools import NoteToolModule
from app.recordings.runtime import initialize_recording_runtime
from app.recordings.voice import LeanRecordingDispatcher
from app.runtime.feature_config import MediaInspectorConfiguration, MemoryAuthority, SecondBrainSettings
from app.runtime.workspace import initialize_workspace_notes
from app.runtime.workspace_delivery import initialize_workspace_presentations
from app.sessions.models import StartSessionRequest
from app.sessions.service import SessionService
from app.settings import JarvisSettings
from app.storage.supabase_config import get_supabase_storage_config
from app.tasks.repository import TaskRepository
from app.workspace.models import WorkspaceContext
from evals.second_brain.note_copy_phone_fixture import application as note_application, endpoint, save
from evals.second_brain.source_fixture import source_fixture


SETUP_SECONDS, NATIVE_SECONDS, CLEANUP_SECONDS, PARENT_SECONDS = 45, 600, 60, 720
MAX_TUS_PATCHES = 43
MAX_JSON_BYTES = 16_384
MAX_HTTP_REQUESTS, REQUEST_SECONDS = 16, 120


def uuid_text(value):
    try:
        return str(UUID(str(value)))
    except (ValueError, TypeError, AttributeError):
        return None


def bounded_json(path):
    with path.open("rb") as stream:
        raw = stream.read(MAX_JSON_BYTES + 1)
    if len(raw) > MAX_JSON_BYTES:
        raise ValueError("Fixture control JSON exceeds its bound")
    return json.loads(raw)


class RecordingStorage(JarvisFileStorageClient):
    """One manifested object; count actual readbacks and forbid backend uploads."""
    def __init__(self, config, state, persist):
        super().__init__(config)
        self.state, self.persist = state, persist
        self.path = None
        self.signs = 0

    def bind(self, owner, path):
        if (self.path is not None or not owner.startswith("direct-image-eval-second-brain-")
                or not path.startswith("owners/" + owner + "/sessions/")):
            raise ValueError("Only one disposable fixture object may be bound")
        self.path = path
        self.state["cloud_object"] = {"owner_id": owner, "bucket": self.bucket, "path": path}
        self.persist()  # Identity is durable before any signed upload capability.

    def require_target(self, bucket, path):
        if self.path is None or (bucket, path) != (self.bucket, self.path):
            raise ValueError("Foreign cloud object denied by the fixture")

    async def sign_recording_upload(self, *, path):
        self.require_target(self.bucket, path)
        if self.signs:
            raise ValueError("This fixture permits one signing attempt and no recreation")
        self.signs += 1
        target = await super().sign_recording_upload(path=path)
        self.state["credential_expires_at"] = target.expires_at.isoformat()
        self.persist()
        return target

    async def upload_bytes(self, **kwargs):
        raise AssertionError("The Phone must upload the original directly")

    async def upload_file(self, **kwargs):
        raise AssertionError("Backend reupload is forbidden in this fixture")

    async def read_object_file(self, *, bucket, path, max_bytes):
        self.require_target(bucket, path)
        if type(max_bytes) is not int or not 0 < max_bytes <= CAPTURE_MAX_BYTES:
            raise ValueError("Capture read bound exceeded")
        reads = self.state.setdefault("cloud_reads", [])
        if len(reads) >= 4:
            raise ValueError("The four expected cloud snapshots are exhausted")
        entry = {"max_bytes": max_bytes, "started_at": datetime.now(UTC).isoformat()}
        reads.append(entry)
        started = monotonic()
        try:
            local = await super().read_object_file(bucket=bucket, path=path, max_bytes=max_bytes)
            entry["size_bytes"] = None if local is None else local.size_bytes
            entry["content_sha256"] = None if local is None else local.content_sha256
            return local
        finally:
            entry["seconds"] = monotonic() - started
            self.persist()

    async def cleanup_object(self):
        if self.path is None:
            return
        await super().delete_object(bucket=self.bucket, path=self.path)
        # A strict absence check, never an additional whole-object readback.
        remaining = await super().read_object_file(bucket=self.bucket, path=self.path, max_bytes=1)
        if remaining is not None:
            remaining.close()
            raise RuntimeError("Fixture cloud object remains after deletion")
        self.state["cloud_cleanup"] = "object_absent"
        self.state["signed_capability_revoked"] = False
        self.persist()


class RecordingTrace:
    """Capture only finalization metadata and streaming digest, never credentials."""
    def __init__(self, app, *, state, persist):
        self.app, self.state, self.persist = app, state, persist

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        path, method = scope["path"], scope["method"]
        self.state["http_requests_started"] = self.state.get("http_requests_started", 0) + 1
        if self.state["http_requests_started"] > MAX_HTTP_REQUESTS:
            return await JSONResponse({"error": "fixture_request_limit"}, status_code=429)(scope, receive, send)
        if method == "PUT" and path.startswith("/recordings/"):
            return await JSONResponse({"error": "direct_upload_required"}, status_code=403)(scope, receive, send)
        finalizing = path.startswith("/recordings/") and path.endswith("/finalize-upload")
        if finalizing:
            self.state["finalization_attempts"] = self.state.get("finalization_attempts", 0) + 1
            if self.state["finalization_attempts"] > 1:
                return await JSONResponse({"error": "fixture_no_retry"}, status_code=429)(scope, receive, send)
        reading = path.startswith("/files/") and path.endswith("/content")
        incoming, outgoing, checksum = bytearray(), bytearray(), hashlib.sha256()
        entry = {"path": path, "method": method, "status": None, "response_bytes": 0,
                 "response_complete": False, "started_at": datetime.now(UTC).isoformat()}
        params = parse_qs(scope.get("query_string", b"").decode("ascii", errors="replace"))
        entry.update(session_id=uuid_text(params.get("session_id", [None])[0]),
                     version_id=params.get("version_id", [None])[0])
        started = monotonic()

        async def observed_receive():
            message = await receive()
            if finalizing and message["type"] == "http.request":
                incoming.extend(message.get("body", b"")[:max(0, MAX_JSON_BYTES + 1 - len(incoming))])
            return message

        async def observed_send(message):
            await send(message)
            if message["type"] == "http.response.start":
                entry["status"] = message["status"]
                headers = dict(message.get("headers", []))
                entry["content_sha256"] = headers.get(b"x-content-sha256", b"").decode("ascii", errors="replace")
                entry["response_version_id"] = headers.get(b"x-jarvis-version-id", b"").decode("ascii", errors="replace")
            elif message["type"] == "http.response.body":
                block = message.get("body", b"")
                entry["response_bytes"] += len(block)
                if reading:
                    checksum.update(block)
                if finalizing:
                    outgoing.extend(block[:max(0, MAX_JSON_BYTES + 1 - len(outgoing))])
                if not message.get("more_body", False):
                    entry["response_complete"] = True

        try:
            async with asyncio.timeout(REQUEST_SECONDS):
                await self.app(scope, observed_receive, observed_send)
        finally:
            entry["seconds"] = monotonic() - started
            if finalizing:
                try:
                    if max(len(incoming), len(outgoing)) > MAX_JSON_BYTES:
                        raise ValueError("Trace bound exceeded")
                    request, response = json.loads(incoming), json.loads(outgoing)
                    descriptor = RecordingUpload.model_validate(request["descriptor"])
                    entry["descriptor"] = descriptor.model_dump(mode="json")
                    entry["result"] = {key: response.get(key) for key in (
                        "outcome", "recording_id", "file_id", "source_id", "version_id",
                        "download_request_id", "download_status", "download_outcome")}
                    # RecordingReceipt owns this identity inside its immutable scope.
                    entry["result"]["recording_id"] = response.get("scope", {}).get("recording_id")
                except (ValueError, KeyError, TypeError):
                    entry["trace_decode_failed"] = True
                self.state.setdefault("finalizations", []).append(entry)
            if reading:
                entry["observed_sha256"] = checksum.hexdigest()
                self.state.setdefault("content_reads", []).append(entry)
            if finalizing or reading:
                self.persist()


def build_manifest(*, root, run_id, owner, namespace, session, token, command):
    return {"schema_version": 1, "run_id": run_id, "backend_url": "", "session_token": token,
        "owner_id": owner, "backend_namespace": namespace, "session_id": str(session),
        "start_command": command, "result_path": str(root / "native-result.json"),
        "report_url": "/fixture/recording-report",
        "capture": {"duration_ms": CAPTURE_MAX_DURATION_MS, "video_frames": 7200, "synthetic_input": True},
        "budgets": {"setup_seconds": SETUP_SECONDS, "native_seconds": NATIVE_SECONDS,
            "cleanup_seconds": CLEANUP_SECONDS, "parent_seconds": PARENT_SECONDS,
            "request_seconds": REQUEST_SECONDS, "max_http_requests": MAX_HTTP_REQUESTS,
            "max_tus_patches": MAX_TUS_PATCHES, "max_backend_read_bytes": 4 * CAPTURE_MAX_BYTES}}


async def prepare(source, settings, storage, configuration, state, persist, root):
    sessions = SessionService(source.pool, settings)
    initial = await sessions.start_session(JarvisPrincipal(owner_id=source.owner), StartSessionRequest())
    session = initial.session.id
    services = SimpleNamespace(settings=settings, db_pool=source.pool, memory_authority=MemoryAuthority.LEGACY,
        knowledge_runtime=None, tool_modules=(), jarvis_file_storage=storage,
        behavior_profile_service=BehaviorProfileService(BehaviorProfileRepository(source.pool)))
    flags = SecondBrainSettings(_env_file=None, enabled="true", recordings_enabled="true",
        knowledge_enabled="false", recording_area_id=str(source.area),
        recording_storage_max_file_bytes=str(CAPTURE_MAX_BYTES),
        recording_ffprobe_path=configuration.ffprobe_path, recording_ffprobe_sha256=configuration.ffprobe_sha256)
    await initialize_recording_runtime(services, flags=flags)
    if services.recording_ingress is None:
        raise RuntimeError("Recording composition unavailable: " + str(services.recording_issue))
    policy, admissions = services.recording_policy, services.recording_admission
    inspector = services.recording_ingress.inspect_file
    async def inspect(local, selected):
        result = await inspector(local, selected)
        state.setdefault("inspections", []).append(result.model_dump(mode="json"))
        persist()
        return result
    services.recording_ingress.inspect_file = inspect
    state["composition"] = {"memory_authority": str(services.memory_authority),
        "knowledge_runtime_started": services.knowledge_runtime is not None,
        "notes_required": False, "production_recording_initializer": True}
    initialize_workspace_presentations(services)
    if services.workspace_presentations is None:
        raise RuntimeError("Production presentation composition is unavailable")
    audio_session, generation = uuid4(), uuid4()
    commands = []

    def enqueue(payload):
        commands.append({"type": "recording.command", "audio_session_id": str(audio_session), **payload})
        return True

    dispatcher = LeanRecordingDispatcher(owner_id=source.owner, session_id=session, admissions=admissions,
        policy=policy, provider_session_key=lambda: "synthetic-recording-phone-fixture", is_active=lambda: True,
        on_command=enqueue)
    dispatcher.configure_device({"type": "recording.capabilities", "version": 1,
        "generation_id": str(generation), "available": True, "include_jarvis_audio": True,
        "max_bytes": CAPTURE_MAX_BYTES, "max_duration_ms": CAPTURE_MAX_DURATION_MS})
    result = await dispatcher.execute({"action": "start", "include_jarvis_audio": True},
                                      provider_call_id="synthetic-fixture-start-" + state["run_id"])
    if result["outcome"] != "admitted" or len(commands) != 1:
        raise RuntimeError("Synthetic fixture command did not receive real admission")
    async with source.sources.transaction(source.owner) as connection:
        scope = await admissions.require_admitted_recording(connection, source.owner, session, dispatcher.current_id)
        schema = await connection.fetchval("select current_schema()")
    _, bucket, path = services.recording_ingress._upload_target(scope)
    storage.bind(source.owner, path)
    manifest = build_manifest(root=root, run_id=state["run_id"], owner=source.owner,
        namespace=storage_namespace(settings.SUPABASE_DB_URL, schema=schema), session=session,
        token=settings.JARVIS_STATIC_BEARER_TOKEN, command=commands[0])
    return SimpleNamespace(services=services, dispatcher=dispatcher, manifest=manifest, scope=scope,
                           ctx=WorkspaceContext(source.owner, session), bucket=bucket)


def application(prepared, state, persist):
    app = note_application(prepared.services, state, persist)
    app.include_router(recording_router)
    app.add_middleware(RecordingTrace, state=state, persist=persist)

    @app.post("/fixture/recording-report", dependencies=[Depends(require_static_bearer)])
    async def report(request: Request):
        reports = state.setdefault("device_reports", [])
        raw = bytearray()
        async for block in request.stream():
            if len(raw) + len(block) > 1024:
                return JSONResponse({"error": "report_too_large"}, status_code=413)
            raw.extend(block)
        if len(reports) >= 4:
            return JSONResponse({"error": "report_limit"}, status_code=429)
        try:
            result = prepared.dispatcher.accept_report(json.loads(raw))
        except (ValueError, TypeError):
            result = None
        if result is None:
            return JSONResponse({"error": "report_rejected"}, status_code=422)
        reports.append(result)
        persist()
        return {"report": result}
    return app


def evaluate(manifest, state, request):
    command = manifest["start_command"]
    finalizations, reads = state.get("finalizations", []), state.get("content_reads", [])
    final = finalizations[0] if len(finalizations) == 1 else {}
    result, descriptor = final.get("result", {}), final.get("descriptor", {})
    receipt = request.get("local_receipt", {})
    expected = {"file_id": result.get("file_id"), "version_id": result.get("version_id"),
        "byte_count": descriptor.get("size_bytes"), "content_sha256": descriptor.get("content_sha256"),
        "destination": "app_storage"}
    target = "/files/" + str(result.get("file_id")) + "/content"
    matches = [read for read in reads if read["path"].lower() == target.lower()]
    acknowledgements = [item for item in state["requests"] if item["path"].lower() ==
        "/workspace/requests/" + str(result.get("download_request_id")).lower() + "/ack"
        and item["status"] == 200 and item["authenticated"]]
    return {
        "real_saved_finalization": final.get("status") == 200 and final.get("response_complete") is True
            and result.get("outcome") == "saved" and uuid_text(result.get("recording_id")) == uuid_text(command["recording_id"]),
        "exact_300s_client_report": descriptor.get("client_report", {}).get("timeline_end_ms") == 300_000
            and descriptor.get("client_report", {}).get("video_frames") == 7200
            and descriptor.get("client_report", {}).get("source_audio_frames", 0) > 0,
        "trusted_video_and_audio": bool(state.get("inspections")) and all(
            item["video_codec"] == "h264" and item["audio_codec"] == "aac"
            and 0 < item["duration_ms"] <= CAPTURE_MAX_ENCODED_DURATION_MS for item in state.get("inspections", [])),
        "production_download_request": bool(result.get("download_request_id"))
            and uuid_text(request.get("request_id")) == uuid_text(result.get("download_request_id"))
            and request.get("artifact_kind") == "file" and request.get("action") == "download"
            and request.get("content_ready") is True and request.get("transfer_policy") == MP4_CAPTURE_POLICY,
        "exact_saved_local_ack": request.get("state") == "saved_local" and receipt == expected
            and [item.get("ack_state") for item in acknowledgements] == ["claimed", "saved_local"]
            and all(uuid_text(item["session_id"]) == uuid_text(manifest["session_id"]) for item in acknowledgements),
        "one_exact_completed_content_response": len(matches) == 1 and matches[0]["status"] == 200
            and matches[0]["response_complete"] is True and matches[0]["response_bytes"] == expected["byte_count"]
            and matches[0]["observed_sha256"] == expected["content_sha256"]
            and matches[0]["content_sha256"] == expected["content_sha256"]
            and matches[0]["response_version_id"] == expected["version_id"]
            and matches[0]["version_id"] == expected["version_id"]
            and matches[0]["session_id"] == uuid_text(manifest["session_id"]),
        "four_bounded_cloud_snapshots": len(state.get("cloud_reads", [])) == 4 and all(
            item.get("size_bytes") == expected["byte_count"] and item.get("content_sha256") == expected["content_sha256"]
            for item in state.get("cloud_reads", [])),
    }


async def serve(prepared, root, state, persist):
    app = application(prepared, state, persist)
    async with endpoint(app) as running:
        manifest = prepared.manifest
        manifest["backend_url"] = "http://127.0.0.1:" + str(running.port)
        prepared.services.settings.PUBLIC_BASE_URL = manifest["backend_url"]
        state.update(status="ready", phase="native")
        save(root / "fixture.json", manifest)
        persist()
        print("RECORDING_PHONE_READY " + str(root / "fixture.json"), flush=True)
        while not (root / "native-result.json").exists():
            if running.task.done():
                await running.task
                raise RuntimeError("Fixture endpoint exited early")
            await asyncio.sleep(0.2)
        native = bounded_json(root / "native-result.json")
        if (native.get("run_id") != manifest["run_id"] or native.get("uploader_closed") is not True
                or native.get("result") not in {"passed", "failed"}):
            raise ValueError("Native result must identify this run and a closed uploader")
        state["native_result_reported_by_operator"] = native
    result = (state.get("finalizations") or [{}])[-1].get("result", {})
    request = None
    if uuid_text(result.get("download_request_id")):
        request = await prepared.services.workspace_presentations.repository.get(
            prepared.ctx, UUID(result["download_request_id"]))
    state["presentation_receipt"] = {} if request is None else request.wire()
    state["server_checks"] = evaluate(prepared.manifest, state, state["presentation_receipt"])
    persist()
    if native["result"] != "passed" or not all(state["server_checks"].values()):
        raise AssertionError("Combined server/native evidence is incomplete")
    state["status"] = "server_checks_passed_native_review_required"


async def run(directory):
    dsn = os.environ.get("JARVIS_SECOND_BRAIN_TEST_DB_URL")
    if not dsn or os.environ.get("JARVIS_RECORDING_PHONE_ALLOW_CLOUD") != "one-fixture-object":
        raise ValueError("Explicit SQL lane and one-fixture-object cloud allocation are required")
    configuration = MediaInspectorConfiguration(storage_max_file_bytes=CAPTURE_MAX_BYTES,
        ffprobe_path=os.environ["JARVIS_MEDIA_FFPROBE_PATH"], ffprobe_sha256=os.environ["JARVIS_MEDIA_FFPROBE_SHA256"])
    cloud_settings = JarvisSettings(_env_file=None)
    config = get_supabase_storage_config(cloud_settings)
    root = directory.resolve()
    root.mkdir(mode=0o700, parents=True, exist_ok=False)
    state = {"format": "jarvis.recording_phone_fixture.v1", "run_id": str(uuid4()).upper(),
        "status": "preparing", "phase": "setup", "requests": [], "cloud_reads": [],
        "boundary": "synthetic_frames_audio_admission_real_sql_hosted_upload_finalization_phone_download",
        "physical_glasses_voice_lock_editing_proven": False, "native_playback_review": "required"}
    persist = lambda: save(root / "server-evidence.json", state)
    persist()
    stack, storage, prepared = AsyncExitStack(), None, None
    try:
        async with asyncio.timeout(SETUP_SECONDS):
            source = await stack.enter_async_context(source_fixture(dsn, root / "database", test_name="recording_phone_fixture"))
            state["database_manifest"] = str(source.fixture.path)
            settings = cloud_settings.model_copy(update={"JARVIS_OWNER_ID": source.owner,
                "SUPABASE_DB_URL": dsn, "JARVIS_STATIC_BEARER_TOKEN": "recording-phone-" + secrets.token_hex(24)})
            storage = RecordingStorage(config, state, persist)
            prepared = await prepare(source, settings, storage, configuration, state, persist, root)
        async with asyncio.timeout(NATIVE_SECONDS):
            await serve(prepared, root, state, persist)
    except BaseException as error:
        state.update(status="failed", error_type=type(error).__name__)
        raise
    finally:
        # Outer owner must terminate native upload before this process if the
        # native deadline expires. No fixture can revoke the signed capability.
        try:
            async with asyncio.timeout(CLEANUP_SECONDS):
                try:
                    if prepared is not None:
                        await prepared.services.recording_source_access.close()
                    if storage is not None:
                        await storage.cleanup_object()
                finally:
                    try:
                        if storage is not None:
                            await storage.close()
                    finally:
                        await stack.aclose()
                        state["database_cleanup_completed"] = True
        except BaseException as error:
            state.update(status="cleanup_failed", cleanup_error_type=type(error).__name__)
            raise
        finally:
            # The fixture token belongs only to the stopped loopback server.
            (root / "fixture.json").unlink(missing_ok=True)
            persist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--execute-allocated-run", action="store_true", required=True)
    args = parser.parse_args()
    asyncio.run(run(args.directory))


if __name__ == "__main__":
    main()
