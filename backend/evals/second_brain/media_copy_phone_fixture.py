"""Disposable generated-media backend for the connected native Open/Save case.

Core runs this only AFTER the native build is ready; five-minute offer expiry is
unchanged. Explicit fixture grants and disk storage are not worker admission,
rendering, cloud storage or sensor evidence. Optional --preflight makes backend
ASGI/auth reads only. No native/app action is automated. Server checks and an
operator report never replace XCTest, screenshots and real UI trace review.
"""
from __future__ import annotations

import argparse
import asyncio
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import secrets
import tempfile
from time import monotonic_ns
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit
from uuid import UUID, uuid4

import httpx

from app.assistant.profile_service import BehaviorProfileService
from app.assistant.repository import BehaviorProfileRepository
from app.auth.client_context import storage_namespace
from app.auth.static_bearer import JarvisPrincipal
from app.files.generated import ArtifactPolicy, ArtifactRequest
from app.files.local_file import snapshot_file
from app.files.media_inspection import inspect_mp4_file
from app.files.models import StoredFileObject
from app.files.storage import JarvisFileStorageClient
from app.files.tools import FileToolModule
from app.files.transfer import FileTransferPolicy
from app.knowledge.files import SourceFileRepository
from app.knowledge.notes import SourceNoteRepository
from app.knowledge.session_areas import SessionKnowledgeAreas
from app.notes.tools import NoteToolModule
from app.runtime.feature_config import MemoryAuthority
from app.runtime.workspace import initialize_workspace_notes
from app.runtime.workspace_delivery import initialize_workspace_presentations
from app.sessions.models import StartSessionRequest
from app.sessions.service import SessionService
from app.settings import JarvisSettings
from app.tasks.generated_artifacts import GeneratedArtifactPublisher
from app.tasks.memory_evidence import TaskMemoryEvidence
from app.tasks.runner import TaskRunner
from app.tasks.tool_bridge import TaskToolBridge
from app.tasks.worker import WorkerContextFactory
from app.workspace.models import WorkspaceContext
from app.workspace.presentation import LocalReceipt, PresentationRequest
from evals.second_brain.note_copy_phone_fixture import RequestTrace as NoteRequestTrace
from evals.second_brain.note_copy_phone_fixture import application as note_application, endpoint, save
from evals.second_brain.sandbox_fixture import sandbox_task_fixture


EXPECTED_SIZE = 15_431_453
EXPECTED_SHA256 = "f4f885c568d5bee8310e1dea75ec5b36860c954b03dc37d2654e9dee4a001ce5"
FIXTURE_LIMIT = 20_000_000
STAGES = ("verify-account", "open-workspace", "open-video", "save", "take-offline", "open-offline",
          "restore-online", "verify-after-reauthentication", "open-after-reauthentication", "finished-awaiting-review")
REOPEN_STAGES = frozenset(STAGES[4:])


def uuid_text(value):
    try:
        return str(UUID(str(value)))
    except (ValueError, TypeError, AttributeError):
        return None


def file_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def qualified_inputs():
    names = ("JARVIS_MEDIA_FIXTURE_FILE", "JARVIS_MEDIA_FIXTURE_SHA256",
             "JARVIS_MEDIA_FFPROBE_PATH", "JARVIS_MEDIA_FFPROBE_SHA256")
    if not all(os.environ.get(name) for name in names):
        raise ValueError("All four explicit media/probe inputs are required")
    original, probe = Path(os.environ[names[0]]), Path(os.environ[names[2]])
    if any(not path.is_absolute() or not path.is_file() or path.is_symlink() for path in (original, probe)):
        raise ValueError("Explicit regular, absolute media/probe files are required")
    if (os.environ[names[1]] != EXPECTED_SHA256 or original.stat().st_size != EXPECTED_SIZE
            or file_hash(original) != EXPECTED_SHA256 or file_hash(probe) != os.environ[names[3]]):
        raise ValueError("Media/probe input hash or size differs from its pinned identity")
    return SimpleNamespace(original=original, probe=probe, probe_sha256=os.environ[names[3]])


def validate_manifest(value):
    expected = {"schema_version", "run_id", "base_url", "credential_path", "expected_owner_id",
                "expected_backend_namespace", "initial_session_id", "open_request", "download_request", "filename", "content_type"}
    if set(value) != expected or type(value["schema_version"]) is not int or value["schema_version"] != 1:
        raise ValueError("Invalid media fixture fields")
    parsed = urlsplit(value["base_url"])
    if (parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
            or parsed.username is not None or parsed.password is not None or parsed.query or parsed.fragment
            or parsed.path not in {"", "/"}):
        raise ValueError("A credential-free loopback origin is required")
    if (not uuid_text(value["run_id"]) or value["run_id"] != str(UUID(value["run_id"])).upper()
            or not uuid_text(value["initial_session_id"]) or not Path(value["credential_path"]).is_absolute()
            or not value["expected_owner_id"] or not value["expected_backend_namespace"]
            or not value["filename"] or value["content_type"] != "video/mp4"):
        raise ValueError("Invalid fixture identity")
    opened, saved = (PresentationRequest.model_validate(value[key]) for key in ("open_request", "download_request"))
    save_fields = ("file_id", "byte_count", "content_sha256", "transfer_policy", "transfer_limit_bytes", "local_receipt", "error_code")
    if (opened.action != "open" or saved.action != "download" or opened.request_id == saved.request_id
            or any(request.artifact_kind != "file" or request.state != "pending" or not request.content_ready
                   or request.format is not None for request in (opened, saved))
            or any(getattr(opened, field) is not None for field in save_fields)
            or (opened.artifact_id, opened.version_id) != (saved.artifact_id, saved.version_id)
            or saved.file_id != saved.artifact_id or saved.byte_count != EXPECTED_SIZE
            or saved.content_sha256 != EXPECTED_SHA256 or saved.transfer_policy != "mp4_output_v1"
            or saved.transfer_limit_bytes is None or not EXPECTED_SIZE <= saved.transfer_limit_bytes <= 32 * 1024**2
            or saved.local_receipt is not None or saved.error_code is not None):
        raise ValueError("Paired exact pinned-media pending offers are required")
    return value


def read_stage(path, run_id):
    if not path.exists():
        return {"phase": "startup", "session_id": None}
    with path.open("rb") as stream:
        raw = stream.read(16_385)
    if len(raw) > 16_384:
        raise ValueError("Native stage exceeds its bound")
    value = json.loads(raw)
    if not isinstance(value, dict) or value.get("run_id") != run_id or value.get("phase") not in STAGES:
        raise ValueError("Native stage does not match this fixture run")
    session = value.get("session_id")
    if session != "none" and uuid_text(session) is None:
        raise ValueError("Invalid native stage session")
    return {"phase": value["phase"], "session_id": uuid_text(session)}


def receipt_fields(value):
    if value is None:
        return None
    # Strict schema prevents arbitrary body fields from entering trace evidence.
    return LocalReceipt.model_validate(value).model_dump(mode="json")


class MediaRequestTrace:
    def __init__(self, app, *, state, persist, token, stage_path, run_id):
        self.app, self.state, self.persist, self.token = app, state, persist, token
        self.stage_path, self.run_id = stage_path, run_id

    def _event(self):
        order = self.state.get("next_trace_event", 0)
        self.state["next_trace_event"] = order + 1
        return order, monotonic_ns()

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        path = scope["path"]
        query = parse_qs(scope.get("query_string", b"").decode("ascii", errors="replace"))
        sequence = self.state.get("next_request_sequence", 0)
        self.state["next_request_sequence"] = sequence + 1
        started_order, started_ns = self._event()
        entry = {"sequence": sequence, "phase": self.state["phase"], "path": path, "method": scope["method"],
            "request_started_order": started_order, "request_started_ns": started_ns,
            "response_completed_order": None, "response_completed_ns": None,
            "session_id": uuid_text(query.get("session_id", [None])[0]),
            "version_id": query.get("version_id", [None])[0], "status": None, "response_complete": False,
            "response_bytes": 0, "at": datetime.now(UTC).isoformat(),
            "authenticated": dict(scope["headers"]).get(b"authorization") == ("Bearer " + self.token).encode()}
        try:
            entry["stage"] = read_stage(self.stage_path, self.run_id)["phase"] if entry["phase"] == "native" else "preflight"
        except (ValueError, OSError):
            entry.update(stage="unverified", trace_decode_failed=True)
        incoming, outgoing, checksum = bytearray(), bytearray(), hashlib.sha256()
        capture = path == "/sessions/start" or path == "/client-context" or path.endswith("/ack")

        async def observed_receive():
            message = await receive()
            if capture and message["type"] == "http.request":
                block = message.get("body", b"")
                incoming.extend(block[:max(0, 16_385 - len(incoming))])
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
                if path.startswith("/files/") and path.endswith("/content"):
                    checksum.update(block)
                if capture:
                    outgoing.extend(block[:max(0, 16_385 - len(outgoing))])
                if not message.get("more_body", False):
                    entry["response_completed_order"], entry["response_completed_ns"] = self._event()
                    entry["response_complete"] = True

        try:
            await self.app(scope, observed_receive, observed_send)
        finally:
            if path.startswith("/files/") and path.endswith("/content"):
                entry["observed_sha256"] = checksum.hexdigest()
            if capture and entry["status"] == 200:
                try:
                    if max(len(incoming), len(outgoing)) > 16_384:
                        raise ValueError("Trace body bound")
                    response = json.loads(outgoing)
                    if path.endswith("/ack"):
                        request = json.loads(incoming)
                        entry.update(request_id=uuid_text(path.split("/")[-2]), ack_state=request.get("state"),
                            session_id=uuid_text(request.get("session_id")), local_receipt=receipt_fields(request.get("local_receipt")),
                            response_ack_state=response["request"]["state"],
                            response_local_receipt=receipt_fields(response["request"].get("local_receipt")))
                    elif path == "/sessions/start":
                        entry["response_session_id"] = uuid_text(response["session"]["id"])
                    else:
                        entry["context"] = {key: response[key] for key in ("owner_id", "backend_namespace", "session_id")}
                        entry["context"]["session_id"] = uuid_text(entry["context"]["session_id"])
                except (ValueError, KeyError, TypeError):
                    entry["trace_decode_failed"] = True
            self.state["requests"].append(entry)
            self.persist()


def application(services, state, persist, *, stage_path, run_id):
    app = note_application(services, state, persist)
    # Replace exactly the inherited trace, before middleware stack creation.
    if len(app.user_middleware) != 1 or app.user_middleware[0].cls is not NoteRequestTrace:
        raise RuntimeError("Unexpected shared fixture middleware; do not double count")
    app.user_middleware.clear()
    app.add_middleware(MediaRequestTrace, state=state, persist=persist,
        token=services.settings.JARVIS_STATIC_BEARER_TOKEN, stage_path=stage_path, run_id=run_id)
    return app


def evaluate_trace(manifest, requests, receipts, network_transitions):
    """Server evidence only: failed offline connection attempts are not observable here."""
    initial = uuid_text(manifest["initial_session_id"])
    opened, saved = manifest["open_request"], manifest["download_request"]
    live = sorted((entry for entry in requests if entry["phase"] == "native"), key=lambda entry: entry["sequence"])
    reads = [entry for entry in live if entry["path"].startswith("/files/") and entry["path"].endswith("/content")]
    acks = [entry for entry in live if entry["path"].startswith("/workspace/requests/") and entry["path"].endswith("/ack")]
    expected_receipt = {key: saved[key] for key in ("file_id", "version_id", "byte_count", "content_sha256")}
    expected_receipt["destination"] = "app_storage"

    def completed_before_start(first, second):
        # Starts alone cannot prove that the claim or transferred body finished.
        values = [first.get("request_started_order"), first.get("response_completed_order"),
                  second.get("request_started_order"), first.get("request_started_ns"),
                  first.get("response_completed_ns"), second.get("request_started_ns")]
        return first["status"] == 200 and first["authenticated"] and first["response_complete"] \
            and all(type(value) is int and value >= 0 for value in values) \
            and values[0] < values[1] <= values[2] and values[3] <= values[4] <= values[5]

    checks = {"trace_complete": all(not item.get("trace_decode_failed") for item in live),
        "native_exact_metadata_read": any(item["method"] == "GET" and item["path"] == "/files/" + saved["file_id"]
            and item["status"] == 200 and item["authenticated"] and item["response_complete"]
            and item["session_id"] == initial and item.get("version_id") == saved["version_id"] for item in live),
        "exactly_two_native_content_reads": len(reads) == 2 and all(
            item["method"] == "GET" and item["path"] == "/files/" + saved["file_id"] + "/content"
            and item["status"] == 200 and item["authenticated"] and item["response_complete"]
            and item["session_id"] == initial and item.get("version_id") == saved["version_id"]
            and item.get("response_version_id") == saved["version_id"] and item["response_bytes"] == EXPECTED_SIZE
            and item.get("content_sha256") == item.get("observed_sha256") == EXPECTED_SHA256
            for item in reads) and [item["stage"] for item in reads] == ["open-video", "save"],
        "no_copy_or_ack_on_reopen": not any(item["stage"] in REOPEN_STAGES for item in reads + acks),
        "no_unrelated_ack": all(item.get("request_id") in {opened["request_id"], saved["request_id"]} for item in acks),
        "real_offline_then_online": network_transitions == ["offline", "online"]}
    for name, target, terminal, stage in (("open", opened, "displayed", "open-video"),
                                          ("download", saved, "saved_local", "save")):
        matching = [item for item in acks if item.get("request_id") == target["request_id"]]
        states = []
        for item in matching:
            if not states or states[-1] != item.get("ack_state"):
                states.append(item.get("ack_state"))
        checks[name + "_claimed_then_terminal"] = states == ["claimed", terminal] and all(
            item["status"] == 200 and item["authenticated"] and item["response_complete"]
            and item["session_id"] == initial and item["stage"] == stage
            and item.get("response_ack_state") == item.get("ack_state")
            and item.get("local_receipt") == item.get("response_local_receipt") == (
                expected_receipt if item.get("ack_state") == "saved_local" else None) for item in matching)
        checks[name + "_read_between_claim_and_receipt"] = any(
            claim.get("ack_state") == "claimed" and final.get("ack_state") == terminal
            and claim["sequence"] < read["sequence"] < final["sequence"]
            and completed_before_start(claim, read) and completed_before_start(read, final)
            for claim in matching for read in reads if read["stage"] == stage for final in matching)
        record = receipts.get(name, {})
        checks[name + "_persisted_exact_outcome"] = all(record.get(key) == target.get(key) for key in (
            "request_id", "action", "artifact_kind", "artifact_id", "version_id", "file_id", "byte_count",
            "content_sha256", "transfer_policy", "transfer_limit_bytes")) and record.get("state") == terminal \
            and record.get("local_receipt") == (expected_receipt if name == "download" else None)
    starts = [item for item in live if item["path"] == "/sessions/start" and item["method"] == "POST"
              and item["status"] == 200 and item["authenticated"] and item["response_complete"]]
    closes = [item for item in live if item["path"].lower() == "/sessions/" + initial + "/close"
              and item["method"] == "POST" and item["status"] == 200 and item["authenticated"] and item["response_complete"]]
    verified = [item for item in live if item["path"] == "/client-context" and item["status"] == 200
                and item["authenticated"] and item["response_complete"] and item.get("context") == {
                    "owner_id": manifest["expected_owner_id"], "backend_namespace": manifest["expected_backend_namespace"],
                    "session_id": item["session_id"]}]
    checks["initial_reused_then_new_verified_session"] = bool(closes) and any(
        start.get("response_session_id") == initial and start["sequence"] < close["sequence"] < fresh["sequence"] < check["sequence"]
        and fresh.get("response_session_id") not in {None, initial} and fresh["response_session_id"] == check["session_id"]
        for start in starts for close in closes for fresh in starts for check in verified) \
        and any(check["session_id"] == initial for check in verified)
    def unexpected(item):
        path = item["path"].lower()
        if any(word in path for word in ("audio", "voice", "realtime", "perception", "visual", "camera", "stream")):
            return True
        return item["method"] not in {"GET", "HEAD"} and not (
            item["method"] == "POST" and (path == "/sessions/start" or path.startswith("/sessions/") and path.endswith("/close")
                                          or path.startswith("/workspace/requests/") and path.endswith("/ack")))
    checks["no_audio_perception_or_other_mutation"] = not any(unexpected(item) for item in live)
    return checks


class DiskStorage:
    """Private fixture objects only; never a configured cloud bucket."""
    bucket = "synthetic-media"
    build_file_path = JarvisFileStorageClient.build_file_path

    def __init__(self, root):
        self.root, self.uploads, self.staged = root, 0, []

    async def upload_file(self, *, path, local_file, content_type):
        self.uploads += 1
        destination = self.root / path
        destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        descriptor = os.open(destination, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(descriptor, "wb") as output:
            async for block in local_file.iter_chunks():
                output.write(block)
        return StoredFileObject(bucket=self.bucket, path=path, size_bytes=local_file.size_bytes)

    async def read_object_file(self, *, bucket, path, max_bytes):
        if bucket != self.bucket:
            raise ValueError("Foreign fixture bucket")
        target = self.root / path
        if not target.exists():
            return None
        local = await snapshot_file(target, max_bytes=max_bytes)
        self.staged.append(local)
        return local

    async def delete_object(self, *, bucket, path):
        if bucket != self.bucket:
            raise ValueError("Foreign fixture bucket")
        (self.root / path).unlink(missing_ok=True)


async def seed(fixture, settings, inputs, storage):
    source, tasks = fixture.source, fixture.tasks
    context = TaskRunner(tasks, {}).context(fixture.claim)
    await WorkerContextFactory(source.pool)(context)
    transfer = FileTransferPolicy("mp4_output_v1", FIXTURE_LIMIT)

    async def inspector(local, policy):
        if file_hash(inputs.probe) != inputs.probe_sha256:
            raise ValueError("Provisioned inspector hash changed")
        return await inspect_mp4_file(local, policy, ffprobe_path=inputs.probe, timeout=10)

    evidence = TaskMemoryEvidence(context, source.sources, area_id=source.area)
    worker_files = SourceFileRepository(source.sources, storage, owner_id=source.owner, area_id=source.area,
        task_evidence=evidence, media_policy=transfer, media_inspector=inspector)
    bridge = await TaskToolBridge.from_modules(context, (FileToolModule(worker_files, storage),),
        read_tools=("list_files", "get_file_metadata"), memory_evidence=evidence,
        receipt_revalidator=worker_files.revalidate_receipt)
    allowance = ArtifactPolicy(max_files=1, max_file_bytes=FIXTURE_LIMIT, max_total_bytes=FIXTURE_LIMIT,
                               formats=("mp4",), transfer_policy="mp4_output_v1")
    async with tasks.transaction(source.owner) as connection:
        scope = await tasks.payload(connection, source.owner, allowance.model_dump(mode="json"))
        parent = await connection.fetchval("""insert into jarvis_operation_grants
            (owner_id,capability,operation_key,request_event_id,request_revision,scope_revision,scope_payload_id,expires_at)
            values($1,'files.publish_generated',$2,$3,1,1,$4,clock_timestamp()+interval '1 hour') returning id""",
            source.owner, str(uuid4()), fixture.event, scope)
        await connection.execute("insert into jarvis_task_grants(owner_id,task_id,grant_id) values($1,$2,$3)",
                                 source.owner, fixture.task.id, parent)
    publisher = GeneratedArtifactPublisher(bridge, worker_files, parent)
    request = ArtifactRequest(operation_key="connected-media-fixture", filename="synthetic-generated.mp4", format="mp4",
                              description="Pinned synthetic generated media fixture; not sensor or model output.")
    local = await snapshot_file(inputs.original, max_bytes=FIXTURE_LIMIT, expected_size=EXPECTED_SIZE, expected_sha256=EXPECTED_SHA256)
    try:
        result = await publisher.publish(request, local, provider_call_id="fixture-media-fresh-prepare")
        replay = await publisher.publish(request, provider_call_id="fixture-media-existing-receipt")
        if result.get("status") != "success" or not replay.get("task_operation", {}).get("replayed") or storage.uploads != 1:
            raise AssertionError("Real generated-media publication/replay did not settle once")
    finally:
        local.close()
    file_id = UUID(result["file"]["id"])
    sessions = SessionService(source.pool, settings)
    initial = await sessions.start_session(JarvisPrincipal(owner_id=source.owner), StartSessionRequest())
    if initial.session.id != fixture.task.session_id:
        raise AssertionError("Fixture session was not resumed")
    foreign = await sessions.start_session(JarvisPrincipal(owner_id=source.other), StartSessionRequest())
    selections = SessionKnowledgeAreas(source.sources, tasks, source.area)
    notes = SourceNoteRepository(source.sources, owner_id=source.owner, area_id=source.area, selections=selections)
    files = SourceFileRepository(source.sources, storage, owner_id=source.owner, area_id=source.area,
        selections=selections, media_policy=transfer, media_inspector=inspector)
    services = SimpleNamespace(settings=settings, db_pool=source.pool, memory_authority=MemoryAuthority.HINDSIGHT,
        knowledge_runtime=SimpleNamespace(sources=source.sources), tool_modules=(NoteToolModule(notes), FileToolModule(files, storage)),
        behavior_profile_service=BehaviorProfileService(BehaviorProfileRepository(source.pool)))
    initialize_workspace_notes(services)
    initialize_workspace_presentations(services)
    file = await files.for_session(initial.session.id).get_file(owner_id=source.owner, file_id=file_id)
    version = file.memory_source["document_id"]
    if (file.memory_source["source_type"], file.memory_source["trust"]) != ("generated_file", "assistant_output"):
        raise AssertionError("Generated origin was not retained truthfully")
    contexts, offers = {}, {}
    for action in ("open", "download"):
        ctx = WorkspaceContext(source.owner, initial.session.id, uuid4())
        await services.workspace_presentations.present(ctx, action=action, artifact_kind="file", artifact_id=file_id, version_id=version)
        contexts[action] = ctx
        offers[action] = (await services.workspace_presentations.repository.get(ctx, ctx.operation_id)).wire()
    async with source.pool.acquire() as connection:
        schema = await connection.fetchval("select current_schema()")
    manifest = {"schema_version": 1, "run_id": str(uuid4()).upper(), "base_url": "", "credential_path": "",
        "expected_owner_id": source.owner, "expected_backend_namespace": storage_namespace(settings.SUPABASE_DB_URL, schema=schema),
        "initial_session_id": str(initial.session.id), "open_request": offers["open"], "download_request": offers["download"],
        "filename": file.filename, "content_type": file.content_type}
    return SimpleNamespace(services=services, contexts=contexts, manifest=manifest, foreign_session=foreign.session.id)


async def preflight(app, prepared, state):
    """Optional in-process backend/auth checks only; never a native launch."""
    manifest = prepared.manifest
    params = {"session_id": manifest["initial_session_id"], "version_id": manifest["download_request"]["version_id"]}
    path = "/files/" + manifest["download_request"]["file_id"]
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://fixture.invalid",
            headers={"Authorization": "Bearer " + prepared.services.settings.JARVIS_STATIC_BEARER_TOKEN}) as client:
        checks = {}
        for suffix in ("", "/content"):
            results = [await client.get(path + suffix, params=params, headers={"Authorization": "Bearer invalid"}),
                       await client.get(path + suffix, params={"version_id": params["version_id"]}),
                       await client.get(path + suffix, params=params | {"session_id": str(prepared.foreign_session)}),
                       await client.get(path + suffix, params=params | {"version_id": "wrong-version"})]
            checks["fences" + ("_content" if suffix else "_metadata")] = [item.status_code for item in results] == [403, 403, 404, 404]
        context = await client.get("/client-context", params={"session_id": params["session_id"]})
        checks["verified_context"] = context.status_code == 200 and context.json() == {
            "owner_id": manifest["expected_owner_id"], "backend_namespace": manifest["expected_backend_namespace"],
            "session_id": params["session_id"]}
        metadata = await client.get(path, params=params)
        checks["exact_metadata"] = metadata.status_code == 200 and metadata.json()["file"]["content_sha256"] == EXPECTED_SHA256
        note, profile = await client.get("/workspace/note"), await client.get("/assistant/profile")
        checks["actual_note_profile_routes"] = note.status_code == profile.status_code == 200 and note.json() == {"note": None}
    state["preflight_checks"] = checks
    if not all(checks.values()):
        raise AssertionError("Media backend preflight failed")


async def drive_network(app, prepared, root, state, persist, *, timeout):
    manifest = prepared.manifest
    phone_root = root / ("phone-media-" + manifest["run_id"])
    current = endpoint(app)
    running = await current.__aenter__()
    try:
        port = running.port
        url = "http://127.0.0.1:" + str(port)
        manifest["base_url"] = url
        prepared.services.settings.PUBLIC_BASE_URL = url
        validate_manifest(manifest)
        state.update(status="ready", phase="native", network_transitions=[], stages=[])
        save(root / "fixture.json", manifest)
        persist()
        print("MEDIA_COPY_PHONE_READY " + str(root / "fixture.json"), flush=True)
        async with asyncio.timeout(timeout):
            while True:
                if current is not None and running.task.done():
                    await running.task
                    raise RuntimeError("Fixture endpoint stopped unexpectedly")
                stage = read_stage(phone_root / "stage.json", manifest["run_id"])
                phase = stage["phase"]
                if phase != "startup" and (not state["stages"] or state["stages"][-1]["phase"] != phase):
                    state["stages"].append(stage); persist()
                if (root / "stop.json").exists():
                    with (root / "stop.json").open("rb") as stream:
                        raw = stream.read(4097)
                    if len(raw) > 4096:
                        raise ValueError("Stop record exceeds its bound")
                    stop = json.loads(raw)
                    if (not isinstance(stop, dict) or stop.get("run_id") != manifest["run_id"]
                            or stop.get("native_result") not in {"passed", "failed"}):
                        raise ValueError("Stop record must identify this run and operator result")
                    state["native_result_reported_by_operator"] = stop["native_result"]
                    break
                if phase == "take-offline" and not state["network_transitions"]:
                    await current.__aexit__(None, None, None)
                    current = None
                    async with httpx.AsyncClient(trust_env=False, timeout=2) as client:
                        try:
                            await client.get(url + "/health")
                        except httpx.ConnectError:
                            pass
                        else:
                            raise AssertionError("Fixture endpoint remained reachable")
                    state["network_transitions"].append("offline"); persist()
                    save(phone_root / "control.json", {"run_id": manifest["run_id"], "phase": "offline"})
                elif phase == "restore-online" and state["network_transitions"] == ["offline"]:
                    current = endpoint(app, port)
                    running = await current.__aenter__()
                    async with httpx.AsyncClient(trust_env=False, timeout=5) as client:
                        (await client.get(url + "/health")).raise_for_status()
                    state["network_transitions"].append("online"); persist()
                    save(phone_root / "control.json", {"run_id": manifest["run_id"], "phase": "online"})
                await asyncio.sleep(0.2)
    finally:
        if current is not None:
            await current.__aexit__(None, None, None)


async def run(directory, *, timeout=600, backend_preflight=False):
    dsn, manifest_directory = os.getenv("JARVIS_SECOND_BRAIN_TEST_DB_URL"), os.getenv("JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR")
    if not dsn or not manifest_directory or not Path(manifest_directory).is_absolute():
        raise ValueError("Explicit database and absolute manifested fixture directory are required")
    inputs = qualified_inputs()
    root = directory.resolve()
    root.mkdir(mode=0o700, parents=True, exist_ok=False)
    state = {"format": "jarvis.media_copy_phone_fixture.v1", "status": "preparing", "phase": "preflight", "requests": [],
        "boundary": "real_generated_source_sql_http_with_fixture_grants_and_private_disk_storage",
        "native_ui_review": "required", "native_qualification": "unverified", "production_media_enabled": False,
        "input_hashes": {"media_sha256": EXPECTED_SHA256, "size_bytes": EXPECTED_SIZE, "ffprobe_sha256": inputs.probe_sha256},
        "preflight_requested": backend_preflight}
    persist = lambda: save(root / "server-evidence.json", state)
    persist()
    storage = None
    try:
        with tempfile.TemporaryDirectory(prefix="objects-", dir=root) as storage_directory:
            storage = DiskStorage(Path(storage_directory))
            try:
                async with sandbox_task_fixture(dsn, Path(manifest_directory), test_name="media_copy_phone_fixture",
                        with_session=True, requested_outcome="Fixture: retain exact synthetic generated media for native controls.") as fixture:
                    state["database_manifest"] = str(fixture.source.fixture.path)
                    credential = root / "credential.txt"
                    token = "media-copy-" + secrets.token_hex(24)
                    descriptor = os.open(credential, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                    with os.fdopen(descriptor, "w") as output:
                        output.write(token + "\n")
                    settings = JarvisSettings(_env_file=None, JARVIS_OWNER_ID=fixture.source.owner,
                        SUPABASE_DB_URL=dsn, JARVIS_STATIC_BEARER_TOKEN=token)
                    prepared = await seed(fixture, settings, inputs, storage)
                    prepared.manifest["credential_path"] = str(credential)
                    stage_path = root / ("phone-media-" + prepared.manifest["run_id"]) / "stage.json"
                    app = application(prepared.services, state, persist, stage_path=stage_path, run_id=prepared.manifest["run_id"])
                    if backend_preflight:
                        await preflight(app, prepared, state)
                    await drive_network(app, prepared, root, state, persist, timeout=timeout)
                    receipts = {}
                    for action, ctx in prepared.contexts.items():
                        receipts[action] = (await prepared.services.workspace_presentations.repository.get(ctx, ctx.operation_id)).wire()
                    state["receipts"] = receipts
                    state["server_checks"] = evaluate_trace(prepared.manifest, state["requests"], receipts, state["network_transitions"])
                    state["single_storage_publication"] = storage.uploads == 1
                    state["staged_files_closed"] = all(not handle.path.exists() for handle in storage.staged)
                    if (not all(state["server_checks"].values()) or not state["single_storage_publication"]
                            or not state["staged_files_closed"] or state.get("native_result_reported_by_operator") != "passed"
                            or not state["stages"] or state["stages"][-1]["phase"] != "finished-awaiting-review"):
                        raise AssertionError("Server evidence or reported native outcome is incomplete")
                    state["status"] = "server_evidence_complete_native_review_required"
                state["database_cleanup_completed"] = True
            finally:
                for handle in storage.staged:
                    handle.close()
        state["temporary_storage_removed"] = True
    except BaseException as error:
        state.update(status="failed", error_type=type(error).__name__)
        raise
    finally:
        (root / "credential.txt").unlink(missing_ok=True)
        try:
            state["original_unchanged"] = file_hash(inputs.original) == EXPECTED_SHA256
        except OSError:
            state["original_unchanged"] = False
        changed_after_success = not state["original_unchanged"] and state["status"] != "failed"
        if not state["original_unchanged"]:
            state["status"] = "failed"
            state["input_error"] = "original_changed"
        persist()
        if changed_after_success:
            raise RuntimeError("Pinned original changed during the fixture")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, help="New nonexistent private evidence directory")
    parser.add_argument("--timeout", type=int, default=600, choices=range(60, 1201), metavar="60..1200")
    parser.add_argument("--preflight", action="store_true", help="Backend/auth ASGI checks only; no native actions")
    parser.add_argument("--check-inputs", action="store_true", help="Read-only pinned media/probe hashes; no DB or ffprobe execution")
    args = parser.parse_args()
    if args.check_inputs:
        inputs = qualified_inputs()
        print(json.dumps({"media_sha256": EXPECTED_SHA256, "size_bytes": EXPECTED_SIZE, "ffprobe_sha256": inputs.probe_sha256}))
        return
    if args.directory is None:
        parser.error("--directory is required for the disposable fixture")
    asyncio.run(run(args.directory, timeout=args.timeout, backend_preflight=args.preflight))


if __name__ == "__main__":
    main()
