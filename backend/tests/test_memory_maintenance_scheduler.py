from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from app.memory.jobs.scheduler import MemoryMaintenanceScheduler


pytestmark = pytest.mark.anyio


class RecordingJobRepository:
    def __init__(self, *, created: bool = True) -> None:
        self.enqueues: list[dict[str, object]] = []
        self.created = created

    async def enqueue_unless_job_type_active(self, **kwargs):
        self.enqueues.append(kwargs)
        return SimpleNamespace(created=self.created)


async def test_scheduler_uses_stable_bucketed_dedupe_keys() -> None:
    repository = RecordingJobRepository()
    scheduler = MemoryMaintenanceScheduler(
        repository=repository,
        owner_id="owner",
        job_intervals={"memory.first": 60, "memory.second": 300},
    )
    now = datetime(2026, 8, 28, 12, 2, tzinfo=UTC)

    count = await scheduler.enqueue_due_jobs(now=now)

    assert count == 2
    assert [item["job_type"] for item in repository.enqueues] == [
        "memory.first",
        "memory.second",
    ]
    assert repository.enqueues[0]["payload"] == {
        "schedule_bucket": int(now.timestamp()) // 60,
        "schedule_interval_seconds": 60,
    }
    assert repository.enqueues[1]["payload"] == {
        "schedule_bucket": int(now.timestamp()) // 300,
        "schedule_interval_seconds": 300,
    }


async def test_scheduler_only_reenqueues_jobs_whose_bucket_advanced() -> None:
    repository = RecordingJobRepository()
    scheduler = MemoryMaintenanceScheduler(
        repository=repository,
        owner_id="owner",
        job_intervals={"memory.fast": 60, "memory.slow": 300},
    )
    first = datetime(2026, 8, 28, 12, 2, tzinfo=UTC)

    assert await scheduler.enqueue_due_jobs(now=first) == 2
    assert await scheduler.enqueue_due_jobs(now=first) == 0
    assert await scheduler.enqueue_due_jobs(
        now=datetime(2026, 8, 28, 12, 3, tzinfo=UTC)
    ) == 1
    assert [item["job_type"] for item in repository.enqueues] == [
        "memory.fast",
        "memory.slow",
        "memory.fast",
    ]


async def test_scheduler_does_not_count_an_existing_active_job() -> None:
    repository = RecordingJobRepository(created=False)
    scheduler = MemoryMaintenanceScheduler(
        repository=repository,
        owner_id="owner",
        job_intervals={"memory.cleanup": 60},
    )
    now = datetime(2026, 8, 28, 12, 2, tzinfo=UTC)

    assert await scheduler.enqueue_due_jobs(now=now) == 0
    assert await scheduler.enqueue_due_jobs(now=now) == 0
    assert len(repository.enqueues) == 1
