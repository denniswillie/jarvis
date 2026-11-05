from __future__ import annotations

import unittest
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid4

from app.memory.jobs.models import InvalidMemoryJobPayloadError, MemoryJob
from app.memory.jobs.repository import (
    MemoryJobConflictError,
    MemoryJobRepository,
    MemoryJobTransactionRequiredError,
)


class _AsyncContext:
    def __init__(self, value: object) -> None:
        self.value = value

    async def __aenter__(self) -> object:
        return self.value

    async def __aexit__(self, *_: object) -> None:
        return None


class _Connection:
    def __init__(
        self,
        *,
        in_transaction: bool = False,
        fetchrow_results: list[dict[str, Any] | None] | None = None,
        fetch_results: list[list[dict[str, Any]]] | None = None,
        fetchval_results: list[object] | None = None,
    ) -> None:
        self._in_transaction = in_transaction
        self.fetchrow_results = list(fetchrow_results or [])
        self.fetch_results = list(fetch_results or [])
        self.fetchval_results = list(fetchval_results or [])
        self.calls: list[tuple[str, str, tuple[object, ...]]] = []
        self.transaction_count = 0

    def is_in_transaction(self) -> bool:
        return self._in_transaction

    def transaction(self) -> _AsyncContext:
        self.transaction_count += 1
        return _AsyncContext(self)

    async def execute(self, query: str, *args: object) -> str:
        self.calls.append(("execute", query, args))
        return "SELECT 1"

    async def fetchrow(self, query: str, *args: object) -> dict[str, Any] | None:
        self.calls.append(("fetchrow", query, args))
        return self.fetchrow_results.pop(0)

    async def fetch(self, query: str, *args: object) -> list[dict[str, Any]]:
        self.calls.append(("fetch", query, args))
        return self.fetch_results.pop(0)

    async def fetchval(self, query: str, *args: object) -> object:
        self.calls.append(("fetchval", query, args))
        return self.fetchval_results.pop(0)


class _Pool:
    def __init__(self, connection: _Connection) -> None:
        self.connection = connection
        self.acquire_count = 0

    def acquire(self) -> _AsyncContext:
        self.acquire_count += 1
        return _AsyncContext(self.connection)


class MemoryJobRepositoryTests(unittest.IsolatedAsyncioTestCase):
    async def test_enqueue_is_locked_and_creates_one_job(self) -> None:
        row = _job_row()
        connection = _Connection(fetchrow_results=[None, row])
        repository = MemoryJobRepository(_Pool(connection))  # type: ignore[arg-type]

        result = await repository.enqueue(
            owner_id=row["owner_id"],
            session_id=row["session_id"],
            job_type=row["job_type"],
            dedupe_key=row["dedupe_key"],
            payload={"event_id": "event-1"},
            priority=row["priority"],
            max_attempts=row["max_attempts"],
            available_at=row["available_at"],
        )

        self.assertTrue(result.created)
        self.assertEqual(result.job.id, row["id"])
        self.assertIn("pg_advisory_xact_lock", connection.calls[0][1])
        self.assertEqual(
            connection.calls[0][2][0],
            '["owner","event_consolidation","event:1"]',
        )
        self.assertNotIn("\0", connection.calls[0][2][0])
        self.assertEqual(connection.transaction_count, 1)

    async def test_enqueue_returns_existing_terminal_job(self) -> None:
        row = _job_row(status="dead")
        connection = _Connection(fetchrow_results=[row])
        repository = MemoryJobRepository(_Pool(connection))  # type: ignore[arg-type]

        result = await repository.enqueue(
            owner_id=row["owner_id"],
            session_id=row["session_id"],
            job_type=row["job_type"],
            dedupe_key=row["dedupe_key"],
            payload={"event_id": "event-1"},
            priority=row["priority"],
            max_attempts=row["max_attempts"],
            available_at=row["available_at"],
        )

        self.assertFalse(result.created)
        self.assertEqual(result.job.status, "dead")
        self.assertEqual(len(connection.fetchrow_results), 0)

    async def test_enqueue_rejects_conflicting_durable_intent(self) -> None:
        for conflicting_field in ("payload", "session_id"):
            with self.subTest(conflicting_field=conflicting_field):
                row = _job_row(status="completed")
                connection = _Connection(fetchrow_results=[row])
                repository = MemoryJobRepository(_Pool(connection))  # type: ignore[arg-type]

                with self.assertRaisesRegex(MemoryJobConflictError, conflicting_field):
                    await repository.enqueue(
                        owner_id=row["owner_id"],
                        session_id=(
                            uuid4() if conflicting_field == "session_id" else row["session_id"]
                        ),
                        job_type=row["job_type"],
                        dedupe_key=row["dedupe_key"],
                        payload=(
                            {"event_id": "different-event"}
                            if conflicting_field == "payload"
                            else {"event_id": "event-1"}
                        ),
                        priority=row["priority"],
                        max_attempts=row["max_attempts"],
                        available_at=row["available_at"],
                    )

    async def test_caller_owned_transaction_is_used_without_nested_transaction(self) -> None:
        row = _job_row()
        connection = _Connection(in_transaction=True, fetchrow_results=[None, row])
        pool = _Pool(_Connection())
        repository = MemoryJobRepository(pool)  # type: ignore[arg-type]

        result = await repository.enqueue(
            owner_id=row["owner_id"],
            session_id=row["session_id"],
            job_type=row["job_type"],
            dedupe_key=row["dedupe_key"],
            payload={"event_id": "event-1"},
            available_at=row["available_at"],
            connection=connection,  # type: ignore[arg-type]
        )

        self.assertTrue(result.created)
        self.assertEqual(pool.acquire_count, 0)
        self.assertEqual(connection.transaction_count, 0)

    async def test_caller_owned_enqueue_requires_active_transaction(self) -> None:
        connection = _Connection(in_transaction=False)
        repository = MemoryJobRepository(_Pool(_Connection()))  # type: ignore[arg-type]

        with self.assertRaises(MemoryJobTransactionRequiredError):
            await repository.enqueue(
                owner_id="owner",
                session_id=None,
                job_type="event_consolidation",
                dedupe_key="event:1",
                payload={},
                available_at=datetime.now(UTC),
                connection=connection,  # type: ignore[arg-type]
            )

        self.assertEqual(connection.calls, [])

    async def test_claim_uses_skip_locked_and_increments_attempt(self) -> None:
        row = _job_row(status="claimed")
        connection = _Connection(fetchrow_results=[row])
        repository = MemoryJobRepository(_Pool(connection))  # type: ignore[arg-type]

        claimed = await repository.claim_next(
            worker_id="worker-1",
            job_types=("event_consolidation",),
        )

        self.assertIsNotNone(claimed)
        self.assertEqual(claimed.status, "claimed")  # type: ignore[union-attr]
        query = connection.calls[0][1].lower()
        self.assertIn("for update skip locked", query)
        self.assertIn("attempt_count = jobs.attempt_count + 1", query)

    async def test_failure_query_selects_retryable_or_dead_state_atomically(self) -> None:
        row = _job_row(status="failed")
        connection = _Connection(fetchrow_results=[row])
        repository = MemoryJobRepository(_Pool(connection))  # type: ignore[arg-type]

        failed = await repository.fail(
            owner_id=row["owner_id"],
            job_id=row["id"],
            claim_token=uuid4(),
            error="provider unavailable",
            retry_at=row["available_at"],
        )

        self.assertEqual(failed.status, "failed")
        query = connection.calls[0][1].lower()
        self.assertIn("attempt_count >= max_attempts then 'dead'", query)
        self.assertIn("else 'failed'", query)

    async def test_completion_replay_returns_identical_terminal_outcome(self) -> None:
        row = _job_row(status="completed")
        connection = _Connection(fetchrow_results=[None, row])
        repository = MemoryJobRepository(_Pool(connection))  # type: ignore[arg-type]

        completed = await repository.complete(
            owner_id=row["owner_id"],
            job_id=row["id"],
            claim_token=uuid4(),
            completed_at=row["completed_at"],
            result=row["result"],
        )

        self.assertEqual(completed.id, row["id"])
        self.assertEqual(completed.status, "completed")

    async def test_completion_replay_rejects_different_result(self) -> None:
        row = _job_row(status="completed")
        connection = _Connection(fetchrow_results=[None, row])
        repository = MemoryJobRepository(_Pool(connection))  # type: ignore[arg-type]

        with self.assertRaises(MemoryJobConflictError):
            await repository.complete(
                owner_id=row["owner_id"],
                job_id=row["id"],
                claim_token=uuid4(),
                completed_at=row["completed_at"],
                result={"memory_id": "different-memory"},
            )

    async def test_failure_replay_returns_identical_failed_outcome(self) -> None:
        row = _job_row(status="failed")
        connection = _Connection(fetchrow_results=[None, row])
        repository = MemoryJobRepository(_Pool(connection))  # type: ignore[arg-type]

        failed = await repository.fail(
            owner_id=row["owner_id"],
            job_id=row["id"],
            claim_token=uuid4(),
            error=row["last_error"],
            retry_at=row["available_at"],
        )

        self.assertEqual(failed.id, row["id"])
        self.assertEqual(failed.status, "failed")

    async def test_renew_claim_is_token_guarded_and_uses_server_time(self) -> None:
        row = _job_row(status="claimed")
        connection = _Connection(fetchrow_results=[row])
        repository = MemoryJobRepository(_Pool(connection))  # type: ignore[arg-type]

        renewed = await repository.renew_claim(
            owner_id=row["owner_id"],
            job_id=row["id"],
            claim_token=row["claim_token"],
        )

        self.assertEqual(renewed.id, row["id"])
        query = connection.calls[0][1].lower()
        self.assertIn("claimed_at = now()", query)
        self.assertIn("claim_token = $3", query)

    async def test_retry_and_stale_recovery_are_skip_locked_batches(self) -> None:
        connection = _Connection(fetch_results=[[], []])
        repository = MemoryJobRepository(_Pool(connection))  # type: ignore[arg-type]
        now = datetime.now(UTC)

        self.assertEqual(await repository.requeue_due_failed(limit=10), [])
        self.assertEqual(
            await repository.recover_stale_claims(
                claimed_before=now - timedelta(minutes=5),
                retry_at=now,
                limit=10,
            ),
            [],
        )

        self.assertIn("for update skip locked", connection.calls[0][1].lower())
        self.assertIn("for update skip locked", connection.calls[1][1].lower())

    async def test_history_cleanup_only_deletes_completed_jobs(self) -> None:
        connection = _Connection(fetchval_results=[2])
        repository = MemoryJobRepository(_Pool(connection))  # type: ignore[arg-type]

        removed = await repository.cleanup_completed_before(
            completed_before=datetime.now(UTC),
            limit=20,
        )

        self.assertEqual(removed, 2)
        query = connection.calls[0][1].lower()
        self.assertIn("status = 'completed'", query)
        self.assertNotIn("status = 'dead'", query)

    async def test_payload_rejects_raw_bytes_before_database_access(self) -> None:
        pool = _Pool(_Connection())
        repository = MemoryJobRepository(pool)  # type: ignore[arg-type]

        with self.assertRaises(InvalidMemoryJobPayloadError):
            await repository.enqueue(
                owner_id="owner",
                session_id=None,
                job_type="event_consolidation",
                dedupe_key="event:1",
                payload={"image": b"jpeg"},
                available_at=datetime.now(UTC),
            )

        self.assertEqual(pool.acquire_count, 0)

    def test_dead_job_model_preserves_terminal_knowledge(self) -> None:
        job = MemoryJob(**_job_row(status="dead"))

        self.assertTrue(job.is_terminal)
        self.assertFalse(job.is_active)
        self.assertEqual(job.last_error, "attempt failed")


def _job_row(*, status: str = "queued") -> dict[str, Any]:
    now = datetime.now(UTC)
    claimed = status == "claimed"
    failed = status in {"failed", "dead"}
    completed = status == "completed"
    max_attempts = 3
    if status == "queued":
        attempt_count = 0
    elif status == "dead":
        attempt_count = max_attempts
    else:
        attempt_count = 1
    return {
        "id": uuid4(),
        "owner_id": "owner",
        "session_id": uuid4(),
        "job_type": "event_consolidation",
        "dedupe_key": "event:1",
        "payload": {"event_id": "event-1"},
        "status": status,
        "priority": 10,
        "attempt_count": attempt_count,
        "max_attempts": max_attempts,
        "available_at": now,
        "claimed_at": now if claimed else None,
        "claimed_by": "worker-1" if claimed else None,
        "claim_token": uuid4() if claimed else None,
        "result": {"memory_id": "memory-1"} if completed else None,
        "last_error": "attempt failed" if failed else None,
        "completed_at": now if completed else None,
        "created_at": now,
        "updated_at": now,
    }
