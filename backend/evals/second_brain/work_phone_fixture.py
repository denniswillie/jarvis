"""Model-free phone work/inbox fixture using real authenticated routes and isolated storage."""
import argparse
import asyncio
from datetime import UTC, datetime, timedelta
import json
import os
from pathlib import Path
import secrets
import socket
from types import SimpleNamespace
from uuid import UUID, uuid4

from fastapi import Depends, FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
import uvicorn

from app.attention.repository import AttentionRepository
from app.attention.router import router as attention_router
from app.attention.service import AttentionService
from app.attention.tts import OpenAISpeechProvider
from app.auth.static_bearer import require_static_bearer
from app.errors import JarvisHTTPError
from app.realtime.session_manager import RealtimeSessionManager
from app.reminders.delivery import ReminderDeliveryService
from app.reminders.repository import ReminderRepository
from app.runtime.feature_config import TaskLimits
from app.settings import JarvisSettings, get_settings
from app.tasks.catalog import ConfiguredTaskProvider
from app.tasks.hermes_result import HermesTaskResult
from app.tasks.models import TaskCreate
from app.tasks.repository import TaskRepository
from app.tasks.router import router as tasks_router
from app.tasks.runner import TaskRunner
from app.tasks.service import TaskService
from evals.second_brain.file_phone_fixture import FixtureServer
from evals.second_brain.hindsight_service import save
from evals.second_brain.source_fixture import source_fixture


async def fixture_snapshot(app, owner):
    """Use one JSON-safe shape for HTTP inspection and the final disk receipt."""
    services = app.state.runtime_services
    return jsonable_encoder({
        "tasks": [task.model_dump(mode="json") for task in
                  await services.background_task_service.repository.list(owner)],
        "inbox": await services.attention_service.repository.list(owner, include_dismissed=True),
    })


async def create_app(fixture, token):
    tasks = TaskRepository(fixture.pool, TaskLimits(lease_seconds=300))
    inbox = AttentionRepository(tasks)
    async with fixture.pool.acquire() as connection:
        session = await connection.fetchval("insert into jarvis_sessions(owner_id) values($1) returning id", fixture.owner)

    async def completed(title, answer, status):
        task = await tasks.create(fixture.owner, TaskCreate(request_id=uuid4(),
            title=title, requested_outcome="Synthetic work only. Do not perform external actions.",
            session_id=session), model="fixture-model")
        claim = await tasks.claim_next("phone-fixture-worker", ("frontier",), owner_id=fixture.owner)
        assert claim.task.id == task.id
        context = TaskRunner(tasks, {}).context(claim)
        result = await HermesTaskResult(status=status, answer=answer,
            speech=[{"text": answer}], evidence=[], completed_operation_refs=[],
            uncertainties=["Synthetic fixture; no real-world observation."]).verified(context, SimpleNamespace())
        assert await tasks.finish(claim, result)
        item = (await inbox.list(fixture.owner, task_id=task.id))[0]
        return task, item["id"]

    completed_task, unknown = await completed("Binary search review",
        "Use parentheses to group the sum before division:\n\n    mid = (low + high) // 2\n\nThis is a synthetic answer, not a review of your current screen.", "succeeded")
    speech = OpenAISpeechProvider(SimpleNamespace(with_options=lambda **kwargs: SimpleNamespace()))
    claim = await inbox.claim_next(fixture.owner, session, "phone-fixture-delivery")
    assert claim.intent_id == unknown
    await inbox.prepare(claim, speech)
    await inbox.submitted(claim)
    await inbox.settle(claim, reason="fixture_connection_lost")
    waiting_task, question = await completed("Choose a robotics project",
        "Which robot and controller are you using? I need that information before recommending a wiring plan.",
        "waiting_for_input")
    queued = await tasks.create(fixture.owner, TaskCreate(request_id=uuid4(), title="Workshop notes",
        requested_outcome="Write a synthetic workshop outline.", session_id=session), model="fixture-model")
    reminders = ReminderRepository(fixture.pool)
    reminder = await reminders.create_reminder(owner_id=fixture.owner, session_id=session,
        title="Bring the synthetic notebook", details="Fixture reminder, not proof that you packed anything.",
        remind_at=datetime.now(UTC)-timedelta(seconds=1), condition=None)
    manager = RealtimeSessionManager()
    manager.attention_notifications_enabled = True
    manager.attention_repository = inbox
    due = await reminders.claim_due_reminders(owner_id=fixture.owner, claim_timeout_seconds=120, max_attempts=3)
    delivery = ReminderDeliveryService(owner_id=fixture.owner, repository=reminders,
        session_repository=None, realtime_session_manager=manager, generator=None)
    await delivery.deliver(next(row for row in due if row.id == reminder.id))
    settings = JarvisSettings(_env_file=None, JARVIS_OWNER_ID=fixture.owner, JARVIS_STATIC_BEARER_TOKEN=token)
    app = FastAPI()
    app.include_router(tasks_router)
    app.include_router(attention_router)
    app.dependency_overrides[get_settings] = lambda: settings
    app.state.runtime_services = SimpleNamespace(settings=settings,
        background_task_service=TaskService(tasks, {"frontier": ConfiguredTaskProvider("fixture-model")}),
        attention_service=AttentionService(inbox))
    state = {"format": "jarvis.work_phone_fixture.v1", "pid": os.getpid(),
        "synthetic_only": True, "native_models_contacted": False, "external_actions": False,
        "task_id": str(completed_task.id), "waiting_task_id": str(waiting_task.id),
        "queued_task_id": str(queued.id), "unknown_intent_id": str(unknown),
        "question_intent_id": str(question), "requests": []}

    @app.exception_handler(JarvisHTTPError)
    async def errors(request: Request, error: JarvisHTTPError):
        return JSONResponse({"error": {"code": error.code, "message": error.message}}, status_code=error.status_code)

    protected = [Depends(require_static_bearer)]
    @app.get("/sessions/active", dependencies=protected)
    async def active(): return {"session": None, "realtime": None}
    @app.get("/reminders/active", dependencies=protected)
    async def active_reminders(): return {"reminders": []}
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
        return await fixture_snapshot(app, fixture.owner)
    @app.post("/__fixture/revoke/{intent_id}", dependencies=protected)
    async def revoke(intent_id: UUID):
        async with tasks.transaction(fixture.owner) as connection:
            row, _ = await inbox._entry(connection, fixture.owner, intent_id, lock=True)
            await connection.execute("update jarvis_context_payloads set content=null,availability='deleted' where owner_id=$1 and id=$2",
                                     fixture.owner, row["source_payload_id"])
        return {"revoked": True}
    return app, state


async def serve(root):
    root.mkdir(mode=0o700, parents=True, exist_ok=False)
    state = {"status": "starting"}
    token = "work-fixture-" + secrets.token_hex(24)
    try:
        async with source_fixture(JarvisSettings().SUPABASE_DB_URL, root / "manifests",
                                  test_name="work-phone") as fixture:
            app, state = await create_app(fixture, token)
            @app.middleware("http")
            async def record(request, call_next):
                response = await call_next(request)
                state["requests"].append({"method": request.method, "path": request.url.path,
                    "status": response.status_code, "at": datetime.now(UTC).isoformat(),
                    "authorized": request.headers.get("authorization") == "Bearer " + token})
                state["requests"] = state["requests"][-500:]
                save(root / "result.json", state)
                return response
            listener = socket.socket()
            listener.bind(("127.0.0.1", 0))
            listener.listen()
            state.update(url="http://localhost:" + str(listener.getsockname()[1]), status="ready")
            save(root / "client.json", {"url": state["url"], "token": token})
            save(root / "result.json", state)
            print("WORK_PHONE_READY " + state["url"], flush=True)
            server = FixtureServer(uvicorn.Config(app, log_level="warning", timeout_graceful_shutdown=5))
            try:
                await server.serve(sockets=[listener])
            finally:
                listener.close()
                snapshot = await fixture_snapshot(app, fixture.owner)
                state.update(final_tasks=snapshot["tasks"], final_inbox=snapshot["inbox"])
        state["status"] = "cleaned"
    except BaseException as error:
        state.update(status="failed", error_type=type(error).__name__)
        raise
    finally:
        save(root / "result.json", state)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    asyncio.run(serve(parser.parse_args().root))
