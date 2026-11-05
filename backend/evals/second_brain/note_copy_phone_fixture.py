"""Disposable real-SQL/HTTP backend for Phone's exact-note-copy native case.

Core owns execution. No provider, audio, production owner, app control or deployed
object store is used. Deterministic note setup is not a voice/admission test.
Run the native case in a fresh process only after fixture.json is ready. The
runner responds to its two network stage files; all UI actions remain real taps.
"""
from __future__ import annotations

import argparse
import asyncio
from contextlib import asynccontextmanager
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import secrets
import socket
from types import SimpleNamespace
from uuid import UUID, uuid4

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse
import httpx
import uvicorn

from app.assistant.profile_router import router as profile_router
from app.assistant.profile_service import BehaviorProfileService
from app.assistant.repository import BehaviorProfileRepository
from app.auth.client_context import router as context_router, storage_namespace
from app.auth.static_bearer import JarvisPrincipal, require_static_bearer
from app.errors import JarvisHTTPError
from app.files.router import router as files_router
from app.files.tools import FileToolModule
from app.knowledge.files import SourceFileRepository
from app.knowledge.notes import SourceNoteRepository
from app.knowledge.session_areas import SessionKnowledgeAreas
from app.notes.tools import NoteToolModule
from app.runtime.feature_config import MemoryAuthority, TaskLimits
from app.runtime.workspace import initialize_workspace_notes
from app.runtime.workspace_delivery import initialize_workspace_presentations
from app.sessions.models import StartSessionRequest
from app.sessions.router import router as sessions_router
from app.sessions.service import SessionService
from app.settings import JarvisSettings, get_settings
from app.tasks.repository import TaskRepository
from app.workspace.models import WorkspaceContext
from app.workspace.router import router as workspace_router
from evals.second_brain.file_phone_fixture import FixtureServer
from evals.second_brain.file_storage_fixture import MemoryStorage
from evals.second_brain.source_fixture import source_fixture


def save(path, value):
    temporary = path.with_suffix(".tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
    temporary.replace(path)


class RequestTrace:
    """Observe actual routes, without retaining bearer values or file bodies."""
    def __init__(self, app, *, state, persist, token):
        self.app, self.state, self.persist, self.token = app, state, persist, token

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        from urllib.parse import parse_qs
        query = parse_qs(scope.get("query_string", b"").decode("ascii"))
        path = scope["path"]
        headers = dict(scope["headers"])
        entry = {"phase": self.state["phase"], "method": scope["method"], "path": path,
                 "session_id": query.get("session_id", [None])[0],
                 "at": datetime.now(UTC).isoformat(), "status": None,
                 "authenticated": headers.get(b"authorization") == ("Bearer " + self.token).encode()}
        request_body, response_body = bytearray(), bytearray()
        capture = path == "/sessions/start" or path.endswith("/ack")

        async def observed_receive():
            message = await receive()
            if capture and message["type"] == "http.request":
                request_body.extend(message.get("body", b"")[:max(0, 16384 - len(request_body))])
            return message

        async def observed_send(message):
            if message["type"] == "http.response.start":
                entry["status"] = message["status"]
                response_headers = dict(message.get("headers", []))
                entry["content_sha256"] = response_headers.get(b"x-content-sha256", b"").decode("ascii")
            elif message["type"] == "http.response.body":
                body = message.get("body", b"")
                entry["response_bytes"] = entry.get("response_bytes", 0) + len(body)
                if capture:
                    response_body.extend(body[:max(0, 16384 - len(response_body))])
            await send(message)

        try:
            await self.app(scope, observed_receive, observed_send)
        finally:
            if capture:
                try:
                    incoming = json.loads(request_body)
                    outgoing = json.loads(response_body)
                    if path.endswith("/ack"):
                        entry.update(ack_state=incoming.get("state"), session_id=incoming.get("session_id"))
                    elif entry["status"] == 200:
                        entry["response_session_id"] = outgoing["session"]["id"]
                except (ValueError, KeyError, TypeError):
                    entry["trace_decode_failed"] = True
            self.state["requests"].append(entry)
            self.persist()


def application(services, state, persist):
    app = FastAPI()
    app.state.runtime_services = services
    app.dependency_overrides[get_settings] = lambda: services.settings
    for router in (sessions_router, context_router, workspace_router, files_router, profile_router):
        app.include_router(router)

    @app.exception_handler(JarvisHTTPError)
    async def errors(request: Request, error: JarvisHTTPError):
        return JSONResponse({"error": {"code": error.code, "message": error.message}}, status_code=error.status_code)

    # Only unrelated bootstrap surfaces are empty. Tested routes above are real.
    protected = [Depends(require_static_bearer)]
    @app.get("/reminders/active", dependencies=protected)
    async def reminders(): return {"reminders": []}
    @app.get("/reminders/events", dependencies=protected)
    async def reminder_events(): return {"events": []}
    @app.get("/oauth/google/status", dependencies=protected)
    async def calendar(): return {"connected": False, "expires_at": None, "scopes": []}
    @app.get("/memory/recent", dependencies=protected)
    async def memory(): return {"items": [], "next_cursor": None, "view": "original_sources"}
    @app.get("/memory/working-state", dependencies=protected)
    async def working(): return {"session": None, "owner": None}
    @app.get("/health")
    async def health(): return {"status": "ok", "synthetic_only": True}
    app.add_middleware(RequestTrace, state=state, persist=persist, token=services.settings.JARVIS_STATIC_BEARER_TOKEN)
    return app


@asynccontextmanager
async def endpoint(app, port=0):
    """Restart the identical loopback endpoint without restarting its SQL owner."""
    listener = socket.socket()
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", port))
    listener.listen()
    actual_port = listener.getsockname()[1]
    server = FixtureServer(uvicorn.Config(app, log_level="warning", access_log=False, timeout_graceful_shutdown=5))
    task = asyncio.create_task(server.serve(sockets=[listener]))
    try:
        async with asyncio.timeout(10):
            while not server.started:
                if task.done():
                    await task
                    raise RuntimeError("note_copy_endpoint_start_failed")
                await asyncio.sleep(0.02)
        yield SimpleNamespace(port=actual_port, task=task)
    finally:
        server.should_exit = True
        try:
            await asyncio.wait_for(asyncio.shield(task), 10)
        except TimeoutError:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            raise RuntimeError("note_copy_endpoint_stop_timeout") from None
        finally:
            listener.close()


async def seed(source, settings, format):
    """Use real domain publication with an explicit synthetic fixture guard."""
    sessions = SessionService(source.pool, settings)
    initial = await sessions.start_session(JarvisPrincipal(owner_id=source.owner), StartSessionRequest())
    foreign = await sessions.start_session(JarvisPrincipal(owner_id=source.other), StartSessionRequest())
    selections = SessionKnowledgeAreas(source.sources, TaskRepository(source.pool, TaskLimits()), source.area)
    storage = MemoryStorage()
    notes = SourceNoteRepository(source.sources, owner_id=source.owner, area_id=source.area, selections=selections)
    files = SourceFileRepository(source.sources, storage, owner_id=source.owner, area_id=source.area, selections=selections)
    services = SimpleNamespace(settings=settings, db_pool=source.pool, memory_authority=MemoryAuthority.HINDSIGHT,
        knowledge_runtime=SimpleNamespace(sources=source.sources), tool_modules=(NoteToolModule(notes), FileToolModule(files, storage)),
        behavior_profile_service=BehaviorProfileService(BehaviorProfileRepository(source.pool)))
    initialize_workspace_notes(services)
    initialize_workspace_presentations(services)
    original = await source.publish("Deterministic exact-note-copy setup; no model or speech.", source_key="note-copy-setup")

    async def fixture_guard(connection, current, candidate):
        if source.owner not in source.fixture.owners:
            raise AssertionError("Only the manifested disposable owner may be seeded")
        return (original.payload_id,)

    ctx = lambda operation=None: WorkspaceContext(source.owner, initial.session.id, operation)
    v1_marker, v2_marker = "SAVED COPY VERSION ONE", "CURRENT NOTE VERSION TWO"
    created = await services.workspace_notes.publish_account_edit(ctx(uuid4()), None, [
        {"op": "append_block", "block": {"type": "text", "id": "copy-marker", "text": v1_marker}},
        {"op": "append_block", "block": {"type": "table", "id": "observations",
            "columns": [{"id": "item", "label": "Item"}, {"id": "count", "label": "Count"}],
            "rows": [{"id": "row-one", "cells": {"item": "First", "count": 8}},
                     {"id": "row-two", "cells": {"item": "Unobserved", "count": None}}]}},
    ], fixture_guard)
    artifact_id, v1 = UUID(created["artifact"]["artifact_id"]), created["artifact"]["version_id"]
    download_ctx = ctx(uuid4())
    await services.workspace_presentations.present(download_ctx, action="download", artifact_kind="note",
        artifact_id=artifact_id, version_id=v1, format=format)
    request = await services.workspace_presentations.repository.get(download_ctx, download_ctx.operation_id)
    if request is None or not request.content_ready or request.state != "pending":
        raise AssertionError("Exact export was not prepared by the real publisher")
    changed = await services.workspace_notes.publish_account_edit(ctx(uuid4()), v1,
        [{"op": "set_text", "block_id": "copy-marker", "text": v2_marker}], fixture_guard)
    v2 = changed["artifact"]["version_id"]
    scoped_files = files.for_session(initial.session.id)
    raw = await scoped_files.read_content(owner_id=source.owner, file_id=request.file_id)
    file = await scoped_files.get_file(owner_id=source.owner, file_id=request.file_id)
    if (v1 == v2 or v1_marker.encode() not in raw or v2_marker.encode() in raw or storage.uploads != 1
            or len(raw) != request.byte_count or hashlib.sha256(raw).hexdigest() != request.content_sha256):
        raise AssertionError("Exact version and single-export evidence disagree")
    async with source.pool.acquire() as connection:
        schema = await connection.fetchval("select current_schema()")
    manifest = {"schema_version": 1, "run_id": str(uuid4()).upper(), "base_url": "",
        "credential_path": "", "expected_owner_id": source.owner,
        "expected_backend_namespace": storage_namespace(settings.SUPABASE_DB_URL, schema=schema),
        "initial_session_id": str(initial.session.id), "request": request.wire(),
        "current_version_id": v2, "filename": file.filename, "content_type": file.content_type,
        "download_marker": v1_marker, "current_marker": v2_marker}
    return SimpleNamespace(services=services, storage=storage, manifest=manifest, raw=raw,
        ctx=download_ctx, foreign_session=foreign.session.id)


async def preflight(app, prepared, state):
    """Prove production session/auth fences without consuming a valid HTTP copy."""
    manifest = prepared.manifest
    token = prepared.services.settings.JARVIS_STATIC_BEARER_TOKEN
    file_path = "/files/" + manifest["request"]["file_id"]
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://fixture",
            headers={"Authorization": "Bearer " + token}) as client:
        checks = {}
        for suffix in ("", "/content"):
            path = file_path + suffix
            wrong = await client.get(path, headers={"Authorization": "Bearer invalid"})
            missing = await client.get(path)
            foreign = await client.get(path, params={"session_id": str(prepared.foreign_session)})
            checks["metadata" if not suffix else "content"] = (
                wrong.status_code == 403 and missing.status_code == 403 and foreign.status_code == 404)
        context = await client.get("/client-context", params={"session_id": manifest["initial_session_id"]})
        checks["verified_context"] = context.status_code == 200 and context.json() == {
            "owner_id": manifest["expected_owner_id"], "backend_namespace": manifest["expected_backend_namespace"],
            "session_id": manifest["initial_session_id"]}
        metadata = await client.get(file_path, params={"session_id": manifest["initial_session_id"]})
        checks["owned_metadata"] = metadata.status_code == 200 and metadata.json()["file"]["filename"] == manifest["filename"]
        current = await client.get("/workspace/note")
        checks["current_v2"] = current.status_code == 200 and current.json()["note"]["artifact"]["version_id"] == manifest["current_version_id"]
        state["preflight_checks"] = checks
        if not all(checks.values()):
            raise AssertionError("Connected copy preflight failed")


def evaluate_trace(manifest, requests, receipt, network_transitions):
    """Conservative server evidence, never a substitute for native/UI review."""
    def session_identity(value):
        # Swift serializes UUIDs uppercase; the server serializes lowercase.
        # Normalize UUID identity only. Preserve raw evidence and opaque versions.
        try:
            return UUID(value) if isinstance(value, str) else None
        except ValueError:
            return None

    initial = session_identity(manifest["initial_session_id"])
    target = manifest["request"]
    live = [item for item in requests if item["phase"] == "native"]
    path = "/files/" + target["file_id"]
    reads = [item for item in live if item["path"] == path + "/content" and item["status"] == 200]
    acks = [item for item in live if item["path"] == "/workspace/requests/" + target["request_id"] + "/ack"
            and item["status"] == 200 and item["authenticated"]]
    new_sessions = {session_identity(item.get("response_session_id")) for item in live
                    if item["path"] == "/sessions/start" and item["status"] == 200 and item["authenticated"]}
    new_sessions.discard(None)
    other_sessions = new_sessions - {initial}
    verified = {session_identity(item["session_id"]) for item in live if item["path"] == "/client-context"
                and item["status"] == 200 and item["authenticated"]}
    closed = initial is not None and any(item["path"].startswith("/sessions/") and item["path"].endswith("/close")
                 and session_identity(item["path"][len("/sessions/"):-len("/close")]) == initial
                 and item["status"] == 200 and item["authenticated"] for item in live)
    ack_states = []
    for item in acks:
        if not ack_states or ack_states[-1] != item.get("ack_state"):
            ack_states.append(item.get("ack_state"))
    expected_receipt = {key: target[key] for key in ("file_id", "version_id", "byte_count", "content_sha256")}
    expected_receipt["destination"] = "app_storage"
    return {
        "one_exact_content_read": initial is not None and len(reads) == 1 and reads[0]["authenticated"]
            and session_identity(reads[0]["session_id"]) == initial
            and reads[0].get("response_bytes") == target["byte_count"]
            and reads[0].get("content_sha256") == target["content_sha256"],
        "claimed_then_saved": initial is not None and ack_states == ["claimed", "saved_local"]
            and all(session_identity(item["session_id"]) == initial for item in acks),
        "persisted_exact_receipt": receipt.get("state") == "saved_local" and receipt.get("local_receipt") == expected_receipt,
        "new_session_reverified": closed and initial in new_sessions and bool(other_sessions & verified),
        "real_offline_then_online": network_transitions == ["offline", "online"],
    }


async def drive_network(app, prepared, root, state, persist, *, timeout):
    """Stage-driven network changes only; never call a Phone model action."""
    manifest = prepared.manifest
    run_id = manifest["run_id"]
    phone_root = root / ("phone-copy-" + run_id)
    current_endpoint = endpoint(app)
    running = await current_endpoint.__aenter__()
    port = running.port
    try:
        url = "http://127.0.0.1:" + str(port)
        manifest["base_url"] = url
        prepared.services.settings.PUBLIC_BASE_URL = url
        state.update(status="ready", phase="native", network_transitions=[])
        save(root / "fixture.json", manifest)
        persist()
        print("NOTE_COPY_PHONE_READY " + str(root / "fixture.json"), flush=True)
        async with asyncio.timeout(timeout):
            while True:
                if current_endpoint is not None and running.task.done():
                    await running.task
                    raise RuntimeError("Fixture endpoint stopped unexpectedly")
                if (root / "stop.json").exists():
                    stop = json.loads((root / "stop.json").read_text())
                    if stop.get("run_id") != run_id or stop.get("native_result") not in {"passed", "failed"}:
                        raise ValueError("Stop record must identify this run and native result")
                    state["native_result_reported_by_operator"] = stop["native_result"]
                    break
                stage = phone_root / "stage.json"
                if stage.exists():
                    value = json.loads(stage.read_text())
                    if value.get("run_id") != run_id:
                        raise ValueError("Foreign native stage")
                    phase = value.get("phase")
                    if phase == "take-offline" and not state["network_transitions"]:
                        await current_endpoint.__aexit__(None, None, None)
                        current_endpoint = None
                        async with httpx.AsyncClient(trust_env=False, timeout=2) as client:
                            try:
                                await client.get(url + "/health")
                            except httpx.ConnectError:
                                pass
                            else:
                                raise AssertionError("Fixture endpoint did not go offline")
                        state["network_transitions"].append("offline")
                        persist()
                        save(phone_root / "control.json", {"run_id": run_id, "phase": "offline"})
                    elif phase == "restore-online" and state["network_transitions"] == ["offline"]:
                        current_endpoint = endpoint(app, port)
                        running = await current_endpoint.__aenter__()
                        async with httpx.AsyncClient(trust_env=False, timeout=5) as client:
                            response = await client.get(url + "/health")
                            response.raise_for_status()
                        state["network_transitions"].append("online")
                        persist()
                        save(phone_root / "control.json", {"run_id": run_id, "phase": "online"})
                await asyncio.sleep(0.2)
    finally:
        if current_endpoint is not None:
            await current_endpoint.__aexit__(None, None, None)


async def run(directory, *, format="text", timeout=600):
    dsn = os.environ.get("JARVIS_SECOND_BRAIN_TEST_DB_URL")
    if not dsn:
        raise ValueError("Core's explicit JARVIS_SECOND_BRAIN_TEST_DB_URL lane is required")
    root = directory.resolve()
    root.mkdir(mode=0o700, parents=True, exist_ok=False)
    state = {"format": "jarvis.note_copy_phone_fixture.v1", "status": "preparing", "phase": "preflight",
        "boundary": "real_sql_http_native_case_with_memory_storage_and_synthetic_note_setup",
        "requests": [], "native_ui_review": "required", "production_file_bytes_enabled": False}
    persist = lambda: save(root / "server-evidence.json", state)
    persist()
    try:
        async with source_fixture(dsn, root / "database", test_name="note_copy_phone_fixture") as source:
            state["database_manifest"] = str(source.fixture.path)
            token = "note-copy-" + secrets.token_hex(24)
            credential = root / "credential.txt"
            descriptor = os.open(credential, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(descriptor, "w") as stream:
                stream.write(token + "\n")
            settings = JarvisSettings(_env_file=None, JARVIS_OWNER_ID=source.owner,
                SUPABASE_DB_URL=dsn, JARVIS_STATIC_BEARER_TOKEN=token)
            prepared = await seed(source, settings, format)
            prepared.manifest["credential_path"] = str(credential)
            (root / "expected-copy.bin").write_bytes(prepared.raw)
            app = application(prepared.services, state, persist)
            await preflight(app, prepared, state)
            await drive_network(app, prepared, root, state, persist, timeout=timeout)
            request = await prepared.services.workspace_presentations.repository.get(prepared.ctx, prepared.ctx.operation_id)
            state["receipt"] = request.wire()
            state["server_checks"] = evaluate_trace(prepared.manifest, state["requests"], request.wire(), state["network_transitions"])
            state["single_storage_publication"] = prepared.storage.uploads == 1
            if (state["native_result_reported_by_operator"] != "passed" or not all(state["server_checks"].values())
                    or not state["single_storage_publication"]):
                raise AssertionError("Native or connected server evidence is incomplete")
            state["status"] = "server_checks_passed_native_review_required"
        state["database_cleanup_completed"] = True
    except BaseException as error:
        state.update(status="failed", error_type=type(error).__name__)
        raise
    finally:
        # The ephemeral fixture token is invalid after server teardown. Keep no
        # secret file beyond the run; native/store evidence remains for review.
        (root / "credential.txt").unlink(missing_ok=True)
        persist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True, help="New, nonexistent run directory")
    parser.add_argument("--format", choices=("text", "json"), default="text")
    parser.add_argument("--timeout", type=int, default=600, choices=range(60, 1201), metavar="60..1200")
    args = parser.parse_args()
    asyncio.run(run(args.directory, format=args.format, timeout=args.timeout))


if __name__ == "__main__":
    main()
