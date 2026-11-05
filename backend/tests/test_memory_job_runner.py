from __future__ import annotations

import asyncio
import unittest
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from app.memory.jobs.models import MemoryJob, MemoryJobQueueDiagnostics
from app.memory.jobs.reconciler import MemoryJobReconciler
from app.memory.jobs.registry import (
    DuplicateMemoryJobHandlerError,
    MemoryJobHandlerRegistry,
)
from app.memory.jobs.runner import MemoryJobRunner, MemoryJobRunnerConfig


class _Transaction:
    def __init__(self, connection: "_Connection") -> None:
        self.connection = connection
        self._event_count = 0

    async def __aenter__(self) -> "_Connection":
        self._event_count = len(self.connection.events)
        self.connection.in_transaction = True
        return self.connection

    async def __aexit__(self, error_type: object, *_: object) -> None:
        if error_type is not None:
            del self.connection.events[self._event_count :]
        self.connection.in_transaction = False


class _Acquire:
    def __init__(self, connection: "_Connection") -> None:
        self.connection = connection

    async def __aenter__(self) -> "_Connection":
        return self.connection

    async def __aexit__(self, *_: object) -> None:
        return None


class _Connection:
    def __init__(self) -> None:
        self.in_transaction = False
        self.events: list[str] = []

    def transaction(self) -> _Transaction:
        return _Transaction(self)


class _Pool:
    def __init__(self) -> None:
        self.connection = _Connection()

    def acquire(self) -> _Acquire:
        return _Acquire(self.connection)


class _Repository:
    def __init__(self, jobs: list[MemoryJob]) -> None:
        self.jobs = jobs
        self.renewed = 0
        self.completed = 0
        self.failed: list[dict[str, Any]] = []
        self.commit_connection: _Connection | None = None
        self.maintenance_calls = 0
        self.maintenance_error_count = 0
        self.maintenance_recovered = asyncio.Event()
        self.renew_error: Exception | None = None

    async def recover_stale_claims(self, **_: Any) -> list[MemoryJob]:
        self.maintenance_calls += 1
        if self.maintenance_error_count > 0:
            self.maintenance_error_count -= 1
            raise RuntimeError("database temporarily unavailable")
        if self.maintenance_calls > 1:
            self.maintenance_recovered.set()
        return []

    async def requeue_due_failed(self, **_: Any) -> list[MemoryJob]:
        return []

    async def claim_next(self, **_: Any) -> MemoryJob | None:
        return self.jobs.pop(0) if self.jobs else None

    async def renew_claim(self, **_: Any) -> MemoryJob:
        self.renewed += 1
        if self.renew_error is not None:
            raise self.renew_error
        return _job()

    async def lock_claim(self, *, connection: _Connection, **_: Any) -> MemoryJob:
        self.assert_transaction(connection)
        return _job()

    async def complete(self, *, connection: _Connection, **_: Any) -> MemoryJob:
        self.assert_transaction(connection)
        self.commit_connection = connection
        connection.events.append("completed")
        self.completed += 1
        return _job(status="completed")

    async def fail(self, **kwargs: Any) -> MemoryJob:
        self.failed.append(kwargs)
        return _job(status="dead" if kwargs["terminal"] else "failed")

    async def get_diagnostics(self) -> MemoryJobQueueDiagnostics:
        return MemoryJobQueueDiagnostics(
            counts_by_status={
                "queued": 0,
                "claimed": 0,
                "completed": self.completed,
                "failed": 0,
                "dead": 0,
            },
            oldest_queued_age_seconds=None,
            active_job_types=(),
            last_completed_at=None,
        )

    @staticmethod
    def assert_transaction(connection: _Connection) -> None:
        if not connection.in_transaction:
            raise AssertionError("commit work must use an active transaction")


class _Handler:
    job_type = "test"

    def __init__(
        self,
        pool: _Pool,
        *,
        gate: asyncio.Event | None = None,
        commit_error: Exception | None = None,
    ) -> None:
        self.pool = pool
        self.gate = gate
        self.commit_error = commit_error
        self.prepare_saw_transaction = False
        self.commit_connection: _Connection | None = None
        self.prepare_started = asyncio.Event()
        self.prepare_cancelled = asyncio.Event()

    async def prepare(self, job: MemoryJob) -> dict[str, str]:
        self.prepare_saw_transaction = self.pool.connection.in_transaction
        self.prepare_started.set()
        try:
            if self.gate is not None:
                await self.gate.wait()
        except asyncio.CancelledError:
            self.prepare_cancelled.set()
            raise
        return {"event_id": str(job.id)}

    async def commit(
        self,
        job: MemoryJob,
        prepared_result: object,
        connection: _Connection,
    ) -> dict[str, str]:
        _Repository.assert_transaction(connection)
        self.commit_connection = connection
        connection.events.extend(("domain", "successor"))
        if self.commit_error is not None:
            raise self.commit_error
        return {"event_id": str(job.id)}


class MemoryJobRunnerTests(unittest.IsolatedAsyncioTestCase):
    async def test_shutdown_interruption_is_informational_and_remains_retryable(self) -> None:
        pool = _Pool()
        repository = _Repository([_job()])
        handler = _Handler(pool, gate=asyncio.Event())
        runner = _runner(pool, repository, handler)

        with self.assertLogs("app.memory.jobs.runner", level="INFO") as logs:
            task = asyncio.create_task(runner.run())
            await asyncio.wait_for(handler.prepare_started.wait(), timeout=1)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task

        self.assertTrue(handler.prepare_cancelled.is_set())
        self.assertEqual(len(repository.failed), 1)
        self.assertFalse(repository.failed[0]["terminal"])
        self.assertIn("memory job worker stopped", repository.failed[0]["error"])
        self.assertTrue(any("reason=worker_stopped" in line for line in logs.output))
        self.assertTrue(all(record.levelname == "INFO" for record in logs.records))

    async def test_genuine_runtime_error_is_still_a_warning(self) -> None:
        pool = _Pool()
        repository = _Repository([_job()])
        runner = _runner(pool, repository, _Handler(pool, commit_error=RuntimeError("private failure detail")))

        with self.assertLogs("app.memory.jobs.runner", level="WARNING") as logs:
            await runner.run_cycle()
            await runner.wait_until_idle()

        self.assertIn("error_type=RuntimeError", logs.output[0])
        self.assertNotIn("private failure detail", logs.output[0])
        self.assertIn("private failure detail", repository.failed[0]["error"])

    async def test_shutdown_exhausting_retries_is_still_a_warning(self) -> None:
        pool = _Pool()
        repository = _Repository([])
        runner = _runner(pool, repository, _Handler(pool))

        with patch.object(repository, "fail", AsyncMock(return_value=_job(status="dead"))):
            with self.assertLogs("app.memory.jobs.runner", level="WARNING") as logs:
                await runner._record_failure(
                    _job(), RuntimeError("memory job worker stopped"), terminal=False,
                    retry_immediately=True, shutdown_interruption=True,
                )

        self.assertIn("status=dead", logs.output[0])

    async def test_prepare_is_outside_transaction_and_commit_is_atomic(self) -> None:
        pool = _Pool()
        repository = _Repository([_job()])
        handler = _Handler(pool)
        runner = _runner(pool, repository, handler)

        await runner.run_cycle()
        await runner.wait_until_idle()

        self.assertFalse(handler.prepare_saw_transaction)
        self.assertIs(handler.commit_connection, pool.connection)
        self.assertIs(repository.commit_connection, pool.connection)
        self.assertEqual(repository.completed, 1)
        self.assertEqual(pool.connection.events, ["domain", "successor", "completed"])

    async def test_claim_is_renewed_while_prepare_is_blocked(self) -> None:
        pool = _Pool()
        repository = _Repository([_job()])
        gate = asyncio.Event()
        handler = _Handler(pool, gate=gate)
        runner = _runner(pool, repository, handler, claim_timeout=0.15)

        await runner.run_cycle()
        async with asyncio.timeout(1):
            while repository.renewed < 1:
                await asyncio.sleep(0.01)
        self.assertGreaterEqual(repository.renewed, 1)
        gate.set()
        await runner.wait_until_idle()
        renewed_after_completion = repository.renewed
        await asyncio.sleep(0.06)
        self.assertEqual(repository.renewed, renewed_after_completion)

    def test_duplicate_handler_fails_startup(self) -> None:
        pool = _Pool()
        with self.assertRaises(DuplicateMemoryJobHandlerError):
            MemoryJobHandlerRegistry((_Handler(pool), _Handler(pool)))

    async def test_commit_failure_rolls_back_domain_and_successor_writes(self) -> None:
        pool = _Pool()
        repository = _Repository([_job()])
        handler = _Handler(pool, commit_error=RuntimeError("commit failed"))
        runner = _runner(pool, repository, handler)

        await runner.run_cycle()
        await runner.wait_until_idle()

        self.assertEqual(pool.connection.events, [])
        self.assertEqual(repository.completed, 0)
        self.assertEqual(len(repository.failed), 1)

    async def test_runner_survives_transient_maintenance_failure(self) -> None:
        pool = _Pool()
        repository = _Repository([])
        repository.maintenance_error_count = 1
        runner = _runner(pool, repository, _Handler(pool))
        run_task = asyncio.create_task(runner.run())

        await asyncio.wait_for(repository.maintenance_recovered.wait(), timeout=0.5)
        await runner.stop()
        await run_task

        self.assertGreaterEqual(repository.maintenance_calls, 2)

    async def test_runner_cancellation_releases_claim_and_stops_heartbeat(self) -> None:
        pool = _Pool()
        repository = _Repository([_job()])
        gate = asyncio.Event()
        handler = _Handler(pool, gate=gate)
        runner = _runner(pool, repository, handler, claim_timeout=0.15)
        run_task = asyncio.create_task(runner.run())

        await asyncio.wait_for(handler.prepare_started.wait(), timeout=0.5)
        await asyncio.sleep(0.06)
        run_task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await run_task

        self.assertTrue(handler.prepare_cancelled.is_set())
        self.assertEqual(len(repository.failed), 1)
        self.assertFalse(repository.failed[0]["terminal"])
        renewed_after_cancel = repository.renewed
        await asyncio.sleep(0.06)
        self.assertEqual(repository.renewed, renewed_after_cancel)

    async def test_heartbeat_claim_loss_cancels_prepare_without_retry(self) -> None:
        from app.memory.jobs.repository import MemoryJobClaimLostError

        pool = _Pool()
        repository = _Repository([_job()])
        repository.renew_error = MemoryJobClaimLostError()
        handler = _Handler(pool, gate=asyncio.Event())
        runner = _runner(pool, repository, handler, claim_timeout=0.15)

        await runner.run_cycle()
        await runner.wait_until_idle()

        self.assertTrue(handler.prepare_cancelled.is_set())
        self.assertEqual(repository.failed, [])

    async def test_concurrent_runners_process_shared_claim_once(self) -> None:
        repository = _Repository([_job()])
        first_pool = _Pool()
        second_pool = _Pool()
        first = _runner(first_pool, repository, _Handler(first_pool))
        second = _runner(second_pool, repository, _Handler(second_pool))

        await asyncio.gather(first.run_cycle(), second.run_cycle())
        await asyncio.gather(first.wait_until_idle(), second.wait_until_idle())

        self.assertEqual(repository.completed, 1)


def _runner(
    pool: _Pool,
    repository: _Repository,
    handler: _Handler,
    *,
    claim_timeout: float = 120,
) -> MemoryJobRunner:
    registry = MemoryJobHandlerRegistry((handler,))
    reconciler = MemoryJobReconciler(
        pool=pool,  # type: ignore[arg-type]
        repository=repository,  # type: ignore[arg-type]
    )
    return MemoryJobRunner(
        pool=pool,  # type: ignore[arg-type]
        repository=repository,  # type: ignore[arg-type]
        registry=registry,
        reconciler=reconciler,
        config=MemoryJobRunnerConfig(
            poll_interval_seconds=0.01,
            max_concurrency=1,
            claim_timeout_seconds=claim_timeout,
            reconcile_interval_seconds=60,
        ),
    )


def _job(*, status: str = "claimed") -> MemoryJob:
    now = datetime.now(UTC)
    completed = status == "completed"
    failed = status in {"failed", "dead"}
    max_attempts = 3
    return MemoryJob(
        id=uuid4(),
        owner_id="owner",
        session_id=uuid4(),
        job_type="test",
        dedupe_key="test:1",
        payload={},
        status=status,  # type: ignore[arg-type]
        priority=0,
        attempt_count=max_attempts if status == "dead" else 1,
        max_attempts=max_attempts,
        available_at=now,
        claimed_at=now if status == "claimed" else None,
        claimed_by="worker" if status == "claimed" else None,
        claim_token=uuid4() if status == "claimed" else None,
        result={} if completed else None,
        last_error="failed" if failed else None,
        completed_at=now if completed else None,
        created_at=now,
        updated_at=now,
    )
