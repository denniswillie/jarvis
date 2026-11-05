"""Injected fixture worker for real-process crash/restart tests, not a model.

The endpoint must be a loopback fixture server. DB credentials are passed in the
test process environment, never argv or logs. This module is not registered in
the production app's provider map.
"""
import asyncio
import json
import os
import signal
from urllib.parse import urlsplit

import asyncpg
import httpx

from app.migrations.runner import quote_schema
from app.runtime.feature_config import TaskLimits
from app.tasks.models import Reconciliation, TaskStatus, WorkerOutcome
from app.tasks.repository import TaskRepository
from app.tasks.runner import TaskRunner


class FixtureHTTPProvider:
    model = "fixture-only"

    def __init__(self, base_url):
        if urlsplit(base_url).hostname not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("fixture provider requires a loopback endpoint")
        self.client = httpx.AsyncClient(base_url=base_url, timeout=2)

    async def execute(self, context):
        if context.task.title == "fixture-wait":
            return WorkerOutcome(TaskStatus.WAITING, {"question": "Fixture input needed."})
        if context.task.title == "fixture-action":
            async with context.tasks.pool.acquire() as connection:
                grant_id = await connection.fetchval("select grant_id from jarvis_task_grants where owner_id=$1 and task_id=$2", context.task.owner_id, context.task.id)
            action = await context.actions.prepare(context.claim, "fixture-send", grant_id, "fixture.send",
                {"recipient": "fixture@example.invalid", "artifact": "synthetic-only"})
            if action["status"] in {"accepted", "reconciled"}:
                return WorkerOutcome(TaskStatus.SUCCEEDED, {"reused_receipt": str(action["id"])})
            await context.actions.authorize(context.claim, action["id"])
            await context.actions.begin_submission(context.claim, action["id"])
            response = await self.client.post("/effects", json={"action": str(action["id"])})
            response.raise_for_status()
            # Deliberately stop between the remote effect and its durable receipt.
            # The parent kills this process and independently reconciles the effect.
            await asyncio.Event().wait()
        reservation = await context.reserve("fixture-computation", input_tokens=2, output_tokens=2)
        await context.dispatch(reservation["id"])
        response = await self.client.post("/jobs", json={"task": str(context.task.id)})
        response.raise_for_status()
        handle = response.json()["id"]
        await context.checkpoint({"reservation_id": str(reservation["id"])}, provider_handle=handle)
        while True:
            state = await self._state(handle, False)
            if state != "running":
                await context.settle(reservation["id"], input_tokens=2, output_tokens=1, tool_steps=0, provider_terminal=True)
                return WorkerOutcome(TaskStatus.SUCCEEDED, {"fixture_result": handle})
            await asyncio.sleep(0.05)

    async def _state(self, handle, cancel):
        response = await (self.client.post(f"/jobs/{handle}/cancel") if cancel else self.client.get(f"/jobs/{handle}"))
        response.raise_for_status()
        return response.json()["state"]

    async def reconcile(self, context, *, cancel):
        handle = context.claim.provider_handle
        if handle is None:
            return Reconciliation("unknown")
        state = await self._state(handle, cancel)
        if state == "running":
            return Reconciliation("running")
        async with context.tasks.pool.acquire() as connection:
            reservation = await connection.fetchrow("select id from jarvis_task_reservations where owner_id=$1 and task_id=$2 and attempt_id=$3 and operation_key='fixture-computation'", context.task.owner_id, context.task.id, context.claim.attempt_id)
        await context.settle(reservation["id"], input_tokens=2, output_tokens=1, tool_steps=0, provider_terminal=True)
        return Reconciliation("terminal", WorkerOutcome(TaskStatus.FAILED if cancel else TaskStatus.SUCCEEDED, {"fixture_result": handle}))


async def main():
    schema = os.environ["JARVIS_TASK_PROBE_SCHEMA"]
    if not schema.startswith("jarvis_sb_eval_"):
        raise ValueError("probe requires a disposable fixture schema")
    quote_schema(schema)
    async def setup_fixture_connection(connection):
        await connection.execute(f'set search_path to {quote_schema(schema)},pg_catalog,public,extensions')
    pool = await asyncpg.create_pool(os.environ["JARVIS_SECOND_BRAIN_TEST_DB_URL"], min_size=1, max_size=2,
        timeout=10, command_timeout=20, setup=setup_fixture_connection)
    provider = FixtureHTTPProvider(os.environ["JARVIS_TASK_PROBE_URL"])
    runner = TaskRunner(TaskRepository(pool, TaskLimits(**json.loads(os.environ["JARVIS_TASK_PROBE_LIMITS"]))), {"frontier": provider})
    stopped = asyncio.Event()
    loop = asyncio.get_running_loop()
    for signum in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(signum, stopped.set)
    running = asyncio.create_task(runner.run())
    try:
        await stopped.wait()
    finally:
        await runner.stop()
        await running
        await provider.client.aclose()
        await pool.close()


if __name__ == "__main__":
    asyncio.run(main())
