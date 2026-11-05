from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping
from datetime import UTC, datetime

from app.memory.jobs.repository import MemoryJobRepository


LOGGER = logging.getLogger(__name__)


class MemoryMaintenanceScheduler:
    """Periodically enqueue idempotent maintenance work for the durable runner."""

    def __init__(
        self,
        *,
        repository: MemoryJobRepository,
        owner_id: str,
        job_intervals: Mapping[str, int],
    ) -> None:
        normalized_owner = owner_id.strip()
        if not normalized_owner:
            raise ValueError("maintenance owner_id cannot be blank")
        normalized_intervals = {
            job_type.strip(): interval
            for job_type, interval in job_intervals.items()
        }
        if not normalized_intervals or any(not item for item in normalized_intervals):
            raise ValueError("maintenance scheduler requires job types")
        if any(interval < 1 for interval in normalized_intervals.values()):
            raise ValueError("maintenance intervals must be positive")
        self._repository = repository
        self._owner_id = normalized_owner
        self._job_intervals = normalized_intervals
        self._tick_seconds = min(normalized_intervals.values())
        self._last_enqueued_buckets: dict[str, int] = {}

    @property
    def job_types(self) -> tuple[str, ...]:
        return tuple(self._job_intervals)

    @property
    def tick_seconds(self) -> int:
        return self._tick_seconds

    async def enqueue_due_jobs(self, *, now: datetime | None = None) -> int:
        scheduled_at = _utc(now or datetime.now(UTC))
        enqueued = 0
        for job_type, interval_seconds in self._job_intervals.items():
            bucket = int(scheduled_at.timestamp()) // interval_seconds
            if self._last_enqueued_buckets.get(job_type) == bucket:
                continue
            result = await self._repository.enqueue_unless_job_type_active(
                owner_id=self._owner_id,
                session_id=None,
                job_type=job_type,
                dedupe_key=f"maintenance:{job_type}:{bucket}",
                payload={
                    "schedule_bucket": bucket,
                    "schedule_interval_seconds": interval_seconds,
                },
                priority=-100,
                available_at=scheduled_at,
            )
            self._last_enqueued_buckets[job_type] = bucket
            if result.created:
                enqueued += 1
        return enqueued


async def run_memory_maintenance_scheduler_loop(
    scheduler: MemoryMaintenanceScheduler,
) -> None:
    while True:
        try:
            await scheduler.enqueue_due_jobs()
        except asyncio.CancelledError:
            raise
        except Exception:
            LOGGER.exception("memory maintenance scheduling failed")
        await asyncio.sleep(scheduler.tick_seconds)


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("maintenance timestamps must be timezone-aware")
    return value.astimezone(UTC)
