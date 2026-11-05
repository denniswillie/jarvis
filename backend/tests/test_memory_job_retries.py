from __future__ import annotations

import asyncio
import unittest
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from app.memory.jobs.handlers import (
    MemoryJobTransientError,
    MemoryJobValidationError,
)
from app.memory.jobs.models import MemoryJob, MemoryJobQueueDiagnostics
from app.memory.jobs.reconciler import MemoryJobReconciler
from app.memory.jobs.registry import MemoryJobHandlerRegistry
from app.memory.jobs.runner import MemoryJobRunner, MemoryJobRunnerConfig


class _Context:
    def __init__(self, value: object) -> None:
        self.value = value

    async def __aenter__(self) -> object:
        return self.value

    async def __aexit__(self, *_: object) -> None:
        return None


class _Connection:
    def transaction(self) -> _Context:
        return _Context(self)


class _Pool:
    def __init__(self) -> None:
        self.connection = _Connection()

    def acquire(self) -> _Context:
        return _Context(self.connection)


class _Repository:
    def __init__(self, job: MemoryJob) -> None:
        self.job = job
        self.claimed = False
        self.failures: list[dict[str, Any]] = []
        self.renewed = 0

    async def recover_stale_claims(self, **_: Any) -> list[MemoryJob]:
        return []

    async def requeue_due_failed(self, **_: Any) -> list[MemoryJob]:
        return []

    async def claim_next(self, **_: Any) -> MemoryJob | None:
        if self.claimed:
            return None
        self.claimed = True
        return self.job

    async def renew_claim(self, **_: Any) -> MemoryJob:
        self.renewed += 1
        return self.job

    async def fail(self, **kwargs: Any) -> MemoryJob:
        self.failures.append(kwargs)
        return _job(status="dead" if kwargs["terminal"] else "failed")

    async def get_diagnostics(self) -> MemoryJobQueueDiagnostics:
        raise AssertionError("not used")


class _FailingHandler:
    job_type = "test"

    def __init__(self, error: Exception) -> None:
        self.error = error

    async def prepare(self, _: MemoryJob) -> object:
        raise self.error

    async def commit(self, *_: object) -> None:
        raise AssertionError("commit must not run after preparation failure")


class MemoryJobRetryTests(unittest.IsolatedAsyncioTestCase):
    async def test_transient_failure_is_scheduled_with_backoff(self) -> None:
        repository = _Repository(_job())
        now = datetime.now(UTC)
        runner = _runner(
            repository,
            _FailingHandler(MemoryJobTransientError("provider unavailable")),
            now=now,
        )

        await runner.run_cycle()
        await runner.wait_until_idle()

        failure = repository.failures[0]
        self.assertFalse(failure["terminal"])
        self.assertGreater(failure["retry_at"], now)
        renewed_after_failure = repository.renewed
        await asyncio.sleep(0.06)
        self.assertEqual(repository.renewed, renewed_after_failure)

    async def test_validation_failure_is_terminal_immediately(self) -> None:
        repository = _Repository(_job())
        runner = _runner(
            repository,
            _FailingHandler(MemoryJobValidationError("invalid durable payload")),
            now=datetime.now(UTC),
        )

        await runner.run_cycle()
        await runner.wait_until_idle()

        self.assertTrue(repository.failures[0]["terminal"])


def _runner(
    repository: _Repository,
    handler: _FailingHandler,
    *,
    now: datetime,
) -> MemoryJobRunner:
    pool = _Pool()
    reconciler = MemoryJobReconciler(
        pool=pool,  # type: ignore[arg-type]
        repository=repository,  # type: ignore[arg-type]
    )
    return MemoryJobRunner(
        pool=pool,  # type: ignore[arg-type]
        repository=repository,  # type: ignore[arg-type]
        registry=MemoryJobHandlerRegistry((handler,)),
        reconciler=reconciler,
        config=MemoryJobRunnerConfig(poll_interval_seconds=2),
        now=lambda: now,
    )


def _job(*, status: str = "claimed") -> MemoryJob:
    now = datetime.now(UTC)
    failed = status in {"failed", "dead"}
    return MemoryJob(
        id=uuid4(),
        owner_id="owner",
        session_id=uuid4(),
        job_type="test",
        dedupe_key="test:1",
        payload={},
        status=status,  # type: ignore[arg-type]
        priority=0,
        attempt_count=3 if status == "dead" else 1,
        max_attempts=3,
        available_at=now,
        claimed_at=now if status == "claimed" else None,
        claimed_by="worker" if status == "claimed" else None,
        claim_token=uuid4() if status == "claimed" else None,
        result=None,
        last_error="failed" if failed else None,
        completed_at=None,
        created_at=now,
        updated_at=now,
    )
