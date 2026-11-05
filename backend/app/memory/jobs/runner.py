from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Mapping
from uuid import uuid4

import asyncpg

from app.memory.jobs.handlers import (
    MemoryJobTransientError,
    MemoryJobValidationError,
)
from app.memory.jobs.models import MemoryJob, normalize_job_json_object
from app.memory.jobs.reconciler import MemoryJobReconciler
from app.memory.jobs.registry import MemoryJobHandlerRegistry
from app.memory.jobs.repository import (
    MemoryJobClaimLostError,
    MemoryJobConflictError,
    MemoryJobRepository,
    MemoryJobStateError,
)


LOGGER = logging.getLogger(__name__)
MAX_RETRY_DELAY_SECONDS = 300.0


@dataclass(frozen=True, slots=True)
class MemoryJobRunnerConfig:
    poll_interval_seconds: float = 2.0
    max_concurrency: int = 2
    claim_timeout_seconds: float = 120.0
    reconcile_interval_seconds: float = 60.0

    def __post_init__(self) -> None:
        if self.poll_interval_seconds <= 0:
            raise ValueError("memory job poll interval must be positive")
        if self.max_concurrency < 1:
            raise ValueError("memory job concurrency must be positive")
        if self.claim_timeout_seconds <= 0:
            raise ValueError("memory job claim timeout must be positive")
        if self.reconcile_interval_seconds <= 0:
            raise ValueError("memory job reconcile interval must be positive")


class MemoryJobRunner:
    """Small in-process executor for Postgres-backed durable memory jobs."""

    def __init__(
        self,
        *,
        pool: asyncpg.Pool,
        repository: MemoryJobRepository,
        registry: MemoryJobHandlerRegistry,
        reconciler: MemoryJobReconciler,
        config: MemoryJobRunnerConfig,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._pool = pool
        self._repository = repository
        self._registry = registry
        self._reconciler = reconciler
        self._config = config
        self._now = now
        self._worker_id = f"memory-{uuid4()}"
        self._semaphore = asyncio.Semaphore(config.max_concurrency)
        self._stop_event = asyncio.Event()
        self._active_tasks: set[asyncio.Task[None]] = set()
        self._running = False
        self._last_reconcile_at: datetime | None = None
        self._last_completion_at: datetime | None = None

    @property
    def worker_id(self) -> str:
        return self._worker_id

    @property
    def running(self) -> bool:
        return self._running

    async def run(self) -> None:
        if self._running:
            raise RuntimeError("memory job runner is already running")
        self._running = True
        self._stop_event.clear()
        LOGGER.info(
            "memory job runner started worker=%s handlers=%s",
            self._worker_id,
            len(self._registry.job_types),
        )
        try:
            while not self._stop_event.is_set():
                try:
                    await self.run_cycle()
                except asyncio.CancelledError:
                    raise
                except Exception:
                    LOGGER.exception(
                        "memory job runner cycle failed worker=%s",
                        self._worker_id,
                    )
                try:
                    await asyncio.wait_for(
                        self._stop_event.wait(),
                        timeout=self._config.poll_interval_seconds,
                    )
                except TimeoutError:
                    pass
        finally:
            self._running = False
            await self._cancel_active_tasks()
            LOGGER.info("memory job runner stopped worker=%s", self._worker_id)

    async def stop(self) -> None:
        self._stop_event.set()
        await self.wait_until_idle()

    async def run_cycle(self) -> None:
        await self._run_maintenance()
        job_types = self._registry.job_types
        if not job_types:
            return

        capacity = self._config.max_concurrency - len(self._active_tasks)
        for _ in range(max(0, capacity)):
            job = await self._repository.claim_next(
                worker_id=self._worker_id,
                job_types=job_types,
            )
            if job is None:
                break
            task = asyncio.create_task(self._run_claimed_job(job))
            self._active_tasks.add(task)
            task.add_done_callback(self._finish_active_task)

    async def wait_until_idle(self) -> None:
        while self._active_tasks:
            await asyncio.gather(*tuple(self._active_tasks), return_exceptions=True)

    async def diagnostics(self, *, enabled: bool = True) -> dict[str, Any]:
        queue = await self._repository.get_diagnostics()
        return {
            "enabled": enabled,
            "running": self.running,
            "worker_id": self.worker_id,
            "registered_job_types": self._registry.job_types,
            "active_job_types": queue.active_job_types,
            "active_task_count": len(self._active_tasks),
            "counts_by_status": dict(queue.counts_by_status),
            "oldest_queued_age_seconds": queue.oldest_queued_age_seconds,
            "last_completion_at": queue.last_completed_at or self._last_completion_at,
            "reconciliation": {
                "rule_names": self._reconciler.rule_names,
                "last_run_at": self._reconciler.last_run_at,
                "last_results": self._reconciler.last_results,
                "last_errors": self._reconciler.last_errors,
            },
        }

    async def _run_maintenance(self) -> None:
        now = self._now()
        await self._repository.recover_stale_claims(
            claimed_before=now
            - timedelta(seconds=self._config.claim_timeout_seconds),
            retry_at=now,
        )
        await self._repository.requeue_due_failed()
        if (
            self._last_reconcile_at is None
            or (now - self._last_reconcile_at).total_seconds()
            >= self._config.reconcile_interval_seconds
        ):
            await self._reconciler.run_once(now=now)
            self._last_reconcile_at = now

    async def _run_claimed_job(self, job: MemoryJob) -> None:
        async with self._semaphore:
            started_at = time.perf_counter()
            try:
                handler = self._registry.get(job.job_type)
                if handler is None:
                    raise MemoryJobValidationError(
                        f"no handler registered for job type {job.job_type}"
                    )
                prepared = await self._prepare_with_heartbeat(job, handler.prepare)
                completed = await self._commit(job, handler.commit, prepared)
                self._last_completion_at = completed.completed_at
                LOGGER.info(
                    "memory job completed id=%s type=%s attempt=%s duration_ms=%.2f",
                    job.id,
                    job.job_type,
                    job.attempt_count,
                    (time.perf_counter() - started_at) * 1_000,
                )
            except asyncio.CancelledError:
                await asyncio.shield(
                    self._record_failure(
                        job,
                        RuntimeError("memory job worker stopped"),
                        terminal=False,
                        retry_immediately=True,
                        shutdown_interruption=True,
                    )
                )
                raise
            except MemoryJobValidationError as error:
                await self._record_failure(job, error, terminal=True)
            except (MemoryJobClaimLostError, MemoryJobStateError):
                LOGGER.warning(
                    "memory job claim lost id=%s type=%s attempt=%s",
                    job.id,
                    job.job_type,
                    job.attempt_count,
                )
            except MemoryJobTransientError as error:
                await self._record_failure(job, error, terminal=False)
            except Exception as error:
                await self._record_failure(job, error, terminal=False)

    async def _prepare_with_heartbeat(
        self,
        job: MemoryJob,
        prepare: Callable[[MemoryJob], Any],
    ) -> Any:
        prepare_task = asyncio.create_task(prepare(job))
        heartbeat_task = asyncio.create_task(self._renew_claim_loop(job))
        try:
            done, _ = await asyncio.wait(
                (prepare_task, heartbeat_task),
                return_when=asyncio.FIRST_COMPLETED,
            )
            if heartbeat_task in done:
                error = heartbeat_task.exception()
                if error is None:
                    raise MemoryJobClaimLostError()
                raise error
            return await prepare_task
        finally:
            if not prepare_task.done():
                prepare_task.cancel()
                await asyncio.gather(prepare_task, return_exceptions=True)
            heartbeat_task.cancel()
            await asyncio.gather(heartbeat_task, return_exceptions=True)

    async def _renew_claim_loop(self, job: MemoryJob) -> None:
        claim_token = _require_claim_token(job)
        interval = max(0.05, self._config.claim_timeout_seconds / 3)
        while True:
            await asyncio.sleep(interval)
            await self._repository.renew_claim(
                owner_id=job.owner_id,
                job_id=job.id,
                claim_token=claim_token,
            )

    async def _commit(
        self,
        job: MemoryJob,
        commit: Callable[[MemoryJob, Any, asyncpg.Connection], Any],
        prepared: Any,
    ) -> MemoryJob:
        claim_token = _require_claim_token(job)
        async with self._pool.acquire() as connection:
            async with connection.transaction():
                await self._repository.lock_claim(
                    owner_id=job.owner_id,
                    job_id=job.id,
                    claim_token=claim_token,
                    connection=connection,
                )
                result = await commit(job, prepared, connection)
                safe_result: Mapping[str, Any] | None = None
                if result is not None:
                    safe_result = normalize_job_json_object(
                        result,
                        label="memory job result",
                    )
                return await self._repository.complete(
                    owner_id=job.owner_id,
                    job_id=job.id,
                    claim_token=claim_token,
                    completed_at=self._now(),
                    result=safe_result,
                    connection=connection,
                )

    async def _record_failure(
        self,
        job: MemoryJob,
        error: Exception,
        *,
        terminal: bool,
        retry_immediately: bool = False,
        shutdown_interruption: bool = False,
    ) -> None:
        claim_token = _require_claim_token(job)
        retry_delay = 0.0 if retry_immediately else self._retry_delay(job.attempt_count)
        error_text = f"{type(error).__name__}: {error}"
        try:
            failed = await self._repository.fail(
                owner_id=job.owner_id,
                job_id=job.id,
                claim_token=claim_token,
                error=error_text,
                retry_at=self._now() + timedelta(seconds=retry_delay),
                terminal=terminal,
            )
        except (MemoryJobClaimLostError, MemoryJobConflictError, MemoryJobStateError):
            LOGGER.warning(
                "memory job failure lost claim id=%s type=%s attempt=%s",
                job.id,
                job.job_type,
                job.attempt_count,
            )
            return
        if shutdown_interruption and failed.status != "dead":
            LOGGER.info(
                "memory job interrupted id=%s type=%s attempt=%s status=%s reason=worker_stopped",
                job.id,
                job.job_type,
                job.attempt_count,
                failed.status,
            )
        else:
            LOGGER.warning(
                "memory job failed id=%s type=%s attempt=%s status=%s error_type=%s",
                job.id,
                job.job_type,
                job.attempt_count,
                failed.status,
                type(error).__name__,
            )

    def _retry_delay(self, attempt_count: int) -> float:
        base = max(1.0, self._config.poll_interval_seconds)
        return min(MAX_RETRY_DELAY_SECONDS, base * (2 ** max(0, attempt_count - 1)))

    def _finish_active_task(self, task: asyncio.Task[None]) -> None:
        self._active_tasks.discard(task)
        if task.cancelled():
            return
        try:
            task.result()
        except Exception:
            LOGGER.exception("memory job task escaped runner error handling")

    async def _cancel_active_tasks(self) -> None:
        tasks = tuple(self._active_tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)


def _require_claim_token(job: MemoryJob):  # type: ignore[no-untyped-def]
    if job.claim_token is None:
        raise MemoryJobStateError("claimed memory job has no claim token")
    return job.claim_token
