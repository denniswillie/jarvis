"""Local phone access to a completed native artifact probe, never seeded task answers."""
import asyncio
from contextlib import asynccontextmanager
from datetime import UTC, datetime
import hashlib
import secrets
import socket
from types import SimpleNamespace
from uuid import UUID

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse
import httpx
import uvicorn

from app.attention.repository import AttentionRepository
from app.attention.router import router as attention_router
from app.attention.service import AttentionService
from app.auth.static_bearer import require_static_bearer
from app.errors import JarvisHTTPError
from app.files.router import router as files_router
from app.files.tools import FileToolModule, _file_payload
from app.knowledge.files import SourceFileRepository
from app.runtime.feature_config import MemoryAuthority
from app.settings import JarvisSettings, get_settings
from app.tasks.router import router as tasks_router
from evals.second_brain.file_phone_fixture import FixtureServer
from evals.second_brain.hindsight_service import save


def application(settings, files, storage, service, state, persist, *, stop):
    """Use real task/inbox/file routers; only unrelated home bootstrap is empty."""
    app = FastAPI()
    for router in (tasks_router, attention_router, files_router):
        app.include_router(router)
    app.dependency_overrides[get_settings] = lambda: settings
    app.state.runtime_services = SimpleNamespace(settings=settings,
        memory_authority=MemoryAuthority.HINDSIGHT, tool_modules=(FileToolModule(files, storage),),
        background_task_service=service, attention_service=AttentionService(AttentionRepository(service.repository)))

    @app.exception_handler(JarvisHTTPError)
    async def errors(request: Request, error: JarvisHTTPError):
        return JSONResponse({"error": {"code": error.code, "message": error.message}}, status_code=error.status_code)

    @app.middleware("http")
    async def record(request, call_next):
        response = await call_next(request)
        state.setdefault("requests", []).append({"method": request.method, "path": request.url.path,
            "status": response.status_code, "time": datetime.now(UTC).isoformat(),
            "authorized": request.headers.get("authorization") == "Bearer " + settings.JARVIS_STATIC_BEARER_TOKEN})
        state["requests"] = state["requests"][-500:]
        persist()
        return response

    protected = [Depends(require_static_bearer)]
    @app.get("/sessions/active", dependencies=protected)
    async def active(): return {"session": None, "realtime": None}
    @app.get("/reminders/active", dependencies=protected)
    async def reminders(): return {"reminders": []}
    @app.get("/reminders/events", dependencies=protected)
    async def events(): return {"events": []}
    @app.get("/oauth/google/status", dependencies=protected)
    async def calendar(): return {"connected": False, "expires_at": None, "scopes": []}
    @app.get("/memory/recent", dependencies=protected)
    async def memory(): return {"items": [], "next_cursor": None, "view": "original_sources"}
    @app.get("/memory/working-state", dependencies=protected)
    async def working(): return {"session": None, "owner": None}
    @app.get("/health")
    async def health(): return {"status": "ok", "synthetic_only": True}
    @app.get("/__fixture/status", dependencies=protected)
    async def status():
        return {"files": [_file_payload(item) for item in await files.list_files(
            owner_id=settings.JARVIS_OWNER_ID, query=None, limit=8)], "checks": state.get("checks", {})}
    @app.post("/__fixture/revoke/{file_id}", dependencies=protected)
    async def revoke(file_id: UUID):
        return await files.delete_source_file(owner_id=settings.JARVIS_OWNER_ID, file_id=file_id)
    @app.post("/__fixture/stop", dependencies=protected)
    async def shutdown():
        stop()
        return {"stopping": True}
    return app


@asynccontextmanager
async def live_http(app):
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    server = FixtureServer(uvicorn.Config(app, log_level="warning", access_log=False, timeout_graceful_shutdown=5))
    task = asyncio.create_task(server.serve(sockets=[listener]))
    try:
        async with asyncio.timeout(10):
            while not server.started:
                if task.done():
                    await task
                    raise RuntimeError("phone_fixture_start_failed")
                await asyncio.sleep(0.02)
        yield SimpleNamespace(url="http://localhost:" + str(listener.getsockname()[1]), server=server, task=task)
    finally:
        server.should_exit = True
        try:
            await asyncio.wait_for(asyncio.shield(task), 10)
        except TimeoutError:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            raise RuntimeError("phone_fixture_shutdown_timeout") from None
        finally:
            listener.close()


async def serve_task_files(probe, source, service, files, storage, task_id, *, timeout_seconds=1200):
    """Check actual HTTP, then keep the same retained originals available to the simulator."""
    if (source.owner not in source.fixture.owners or files.owner_id != source.owner
            or probe.record.get("checks", {}).get("task_cleanup_confirmed", {}).get("passed") is not True):
        raise ValueError("A cleaned sandbox and its manifested synthetic owner are required")
    root = probe.root / "phone"
    root.mkdir(mode=0o700)
    state = {"format": "jarvis.native_task_phone_fixture.v1", "synthetic_only": True,
             "parent_manifest": str(probe.path), "status": "starting", "checks": {}, "requests": [],
             "task_id": str(task_id), "sandbox_destroyed_before_http": True}
    persist = lambda: save(root / "phone-fixture.json", state)
    persist()

    def check(name, passed):
        state["checks"][name] = bool(passed)
        persist()
        if not passed:
            raise AssertionError(name)

    wanted = {item["file_id"]: item for item in probe.record["downloaded_artifacts"]}
    intents = await AttentionRepository(service.repository).list(source.owner, limit=20, offset=0,
        include_dismissed=False, task_id=task_id)
    check("one_current_task_intent", len(intents) == 1)
    intent_id = str(intents[0]["id"])
    routes = ["/tasks/" + str(task_id), "/attention/" + intent_id]
    routes += [route for identity in wanted for route in ("/files/" + identity, "/files/" + identity + "/content")]
    try:
        # A separate actual HTTP server uses the other manifested owner, not a
        # special auth bypass or an invented foreign UUID on the primary server.
        foreign_token = secrets.token_hex(24)
        foreign_settings = JarvisSettings(_env_file=None, JARVIS_OWNER_ID=source.other,
            JARVIS_STATIC_BEARER_TOKEN=foreign_token)
        foreign_files = SourceFileRepository(source.sources, storage, owner_id=source.other, area_id=source.area)
        foreign_state = {"requests": []}
        foreign = application(foreign_settings, foreign_files, storage, service, foreign_state, lambda: None,
                              stop=lambda: None)
        async with live_http(foreign) as server:
            async with httpx.AsyncClient(base_url=server.url, trust_env=False, timeout=30,
                    headers={"Authorization": "Bearer " + foreign_token}) as client:
                denied = [await client.get(route) for route in routes]
                check("foreign_owner_denied", all(response.status_code in {403, 404} for response in denied))
        state["foreign_requests"] = foreign_state["requests"]

        token = "phone-native-" + secrets.token_hex(24)
        settings = JarvisSettings(_env_file=None, JARVIS_OWNER_ID=source.owner, JARVIS_STATIC_BEARER_TOKEN=token)
        controller = {}
        app = application(settings, files, storage, service, state, persist,
                          stop=lambda: setattr(controller["server"], "should_exit", True))
        async with live_http(app) as server:
            controller["server"] = server.server
            async with httpx.AsyncClient(base_url=server.url, trust_env=False, timeout=30,
                    headers={"Authorization": "Bearer " + token}) as client:
                denied = [await client.get(route, headers={"Authorization": "Bearer invalid-fixture-token"})
                          for route in routes]
                check("wrong_token_denied", all(response.status_code == 403 and
                      response.json()["error"]["code"] == "jarvis_auth_invalid" for response in denied))
                task = await client.get("/tasks/" + str(task_id))
                check("completed_task_over_http", task.status_code == 200 and task.json()["status"] == "succeeded")
                inbox = await client.get("/attention", params={"task_id": str(task_id)})
                check("task_inbox_over_http", inbox.status_code == 200 and
                      [item["id"] for item in inbox.json()] == [intent_id])
                detail = await client.get("/attention/" + intent_id)
                check("inbox_links_exact_native_files", detail.status_code == 200 and
                      detail.json()["source_available"] is True and
                      {item["file_id"] for item in detail.json()["artifacts"]} == set(wanted))
                for identity, expected in wanted.items():
                    metadata = await client.get("/files/" + identity)
                    check(expected["filename"] + "_http_metadata", metadata.status_code == 200 and
                          metadata.json()["file"]["filename"] == expected["filename"])
                    content = await client.get("/files/" + identity + "/content")
                    check(expected["filename"] + "_exact_http_original", content.status_code == 200 and
                          len(content.content) == expected["size_bytes"] and
                          hashlib.sha256(content.content).hexdigest() == expected["sha256"] and
                          content.headers.get("cache-control") == "private, no-store" and
                          content.headers.get("x-content-type-options") == "nosniff" and
                          content.headers.get("content-disposition", "").startswith("attachment;"))
            state.update(status="ready", url=server.url, intent_id=intent_id, files=list(wanted.values()))
            save(root / "client.json", {"token": token, "url": server.url, "task_id": str(task_id), "intent_id": intent_id})
            persist()
            print("TASK_FILE_PHONE_READY " + str(root), flush=True)
            await asyncio.wait_for(asyncio.shield(server.task), timeout_seconds)
        state["status"] = "stopped"
    except BaseException as error:
        state["status"] = "failed"
        state["error"] = {"type": type(error).__name__, "code": getattr(error, "code", None)}
        raise
    finally:
        persist()
        probe.record["phone_fixture"] = {"path": str(root / "phone-fixture.json"),
            "status": state["status"], "http_checks": state["checks"], "interactive_review": "required"}
        probe.persist()
