"""Bounded SQL-only child for lost-wakeup qualification, not an agent runtime.

Reads its scoped fixture identity over stdin; credentials come only from the
explicit test environment. No bus forwarding, models, tools or services run.
"""
import asyncio
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import sys
import time
import traceback
from types import SimpleNamespace
from uuid import UUID, uuid4

import asyncpg

from app.migrations.runner import quote_schema
from app.runtime.feature_config import TaskLimits
from app.tasks.context_repository import ContextRepository
from app.tasks.hermes_result import HermesTaskResult
from app.tasks.models import TaskMutation
from app.tasks.repository import TaskRepository
from app.tasks.runner import TaskRunner
from evals.second_brain.fixtures import validate_manifest


def emit(value):
    print(json.dumps(value, default=str), flush=True)


def sanitized(error):
    return {"error_type": type(error).__name__, "code": getattr(error, "code", None),
            "sqlstate": getattr(error, "sqlstate", None),
            "frames": [{"file": Path(frame.filename).name, "line": frame.lineno,
                        "function": frame.name}
                       for frame in traceback.extract_tb(error.__traceback__)[-6:]]}


async def main():
    config = json.loads(await asyncio.to_thread(sys.stdin.readline))
    manifest = json.loads(Path(config["manifest"]).read_text())
    validate_manifest(manifest)
    owner, session = config["owner"], UUID(config["session"])
    if owner not in manifest["owners"] or len(manifest["schemas"]) != 1:
        raise ValueError("Writer requires exactly one manifested fixture schema/owner")
    identities = {UUID(value) for value in config["tasks"]}
    if len(identities) != 2:
        raise ValueError("Writer is bounded to exactly two synthetic tasks")
    schema = manifest["schemas"][0]

    async def setup(connection):
        await connection.execute(f"set search_path to {quote_schema(schema)},pg_catalog,extensions")
        await connection.execute("set lock_timeout='5s'; set statement_timeout='15s'")

    pool = await asyncpg.create_pool(os.environ["JARVIS_SECOND_BRAIN_TEST_DB_URL"],
        min_size=1, max_size=2, timeout=10, command_timeout=20, setup=setup)
    tasks = TaskRepository(pool, TaskLimits(lease_seconds=300))
    context = ContextRepository(tasks)
    claims = {}
    try:
        emit({"kind": "ready", "pid": os.getpid(), "bus_forwarding": False})
        for _ in range(12):
            line = await asyncio.to_thread(sys.stdin.readline)
            if not line:
                break
            command = json.loads(line)
            if command["operation"] == "stop":
                break
            identity = UUID(command["task"])
            if identity not in identities:
                raise ValueError("Task is outside the manifested writer scope")
            task = await tasks.get(owner, identity)
            if task.session_id != session:
                raise ValueError("Task session differs from fixture scope")
            operation = command["operation"]
            if operation == "claim":
                claim = await tasks.claim_next("lost-wakeup-" + manifest["run_id"],
                    ("frontier",), owner_id=owner, provider_models={"frontier": task.model})
                if claim is None or claim.task.id != identity:
                    raise AssertionError("Canonical worker claim did not select the exact fixture task")
                claims[identity] = claim
            elif operation == "finish":
                claim = claims[identity]
                request = await context.get(owner, task.request_event_id, session_id=session)
                await context.store_manifest(claim, watermark=task.context_watermark,
                    records=(request,), content={"fixture": "lost_wakeup", "records": [request.as_context()]})
                answer = "Synthetic result " + ("37." if identity == UUID(config["tasks"][0]) else "41.")
                outcome = await HermesTaskResult(status="succeeded", answer=answer,
                    speech=[{"text": answer}], evidence=[], completed_operation_refs=[],
                    uncertainties=[]).verified(TaskRunner(tasks, {}).context(claim), SimpleNamespace())
                if not await tasks.finish(claim, outcome):
                    raise AssertionError("Canonical result commit was rejected")
            elif operation == "update":
                await tasks.mutate(owner, identity, "update", TaskMutation(request_id=uuid4(),
                    expected_revision=task.revision, message="Use the corrected synthetic question."))
            else:
                raise ValueError("Unsupported qualification operation")
            committed = time.monotonic_ns()
            # This is a new post-commit read, not a notification or a provider receipt.
            async with tasks.transaction(owner) as connection:
                fresh, detail = await tasks.live_context_in_transaction(connection, owner, identity)
                source = await context._record(connection, owner, fresh.request_event_id)
            emit({"kind": "committed", "operation": operation, "pid": os.getpid(),
                  "committed_monotonic_ns": committed, "task": fresh.model_dump(mode="json"),
                  "event_sequence": detail["last_event_sequence"],
                  "result_recorded_at": detail["result_recorded_at"],
                  "result": detail["result"], "request_source_time": source.source_time,
                  "observed_at": datetime.now(UTC)})
        else:
            raise AssertionError("Writer command budget exceeded")
    finally:
        try:
            await asyncio.wait_for(pool.close(), 10)
        except BaseException:
            pool.terminate()
            raise
    emit({"kind": "stopped", "pid": os.getpid(), "pool_closed": True})


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as error:
        emit({"kind": "failed", **sanitized(error)})
        raise SystemExit(1) from None
