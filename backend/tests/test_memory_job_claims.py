from __future__ import annotations

import unittest
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from app.memory.jobs.repository import (
    MemoryJobRepository,
    MemoryJobTransactionRequiredError,
)


class _Context:
    def __init__(self, value: object) -> None:
        self.value = value

    async def __aenter__(self) -> object:
        return self.value

    async def __aexit__(self, *_: object) -> None:
        return None


class _Connection:
    def __init__(self, rows: list[dict[str, Any] | None], *, active: bool) -> None:
        self.rows = rows
        self.active = active
        self.calls: list[tuple[str, tuple[object, ...]]] = []

    def is_in_transaction(self) -> bool:
        return self.active

    async def execute(self, query: str, *args: object) -> str:
        self.calls.append((query, args))
        return "SELECT 1"

    async def fetchrow(self, query: str, *args: object):  # type: ignore[no-untyped-def]
        self.calls.append((query, args))
        return self.rows.pop(0)


class _Pool:
    def acquire(self) -> _Context:
        raise AssertionError("caller-owned transaction should be used")


class MemoryJobClaimTests(unittest.IsolatedAsyncioTestCase):
    async def test_repository_default_attempts_are_used_for_enqueue(self) -> None:
        row = _job_row(None)
        row.update(
            status="queued",
            attempt_count=0,
            max_attempts=7,
            claimed_at=None,
            claimed_by=None,
            claim_token=None,
        )
        connection = _Connection([None, row], active=True)
        repository = MemoryJobRepository(  # type: ignore[arg-type]
            _Pool(),
            default_max_attempts=7,
        )

        result = await repository.enqueue(
            owner_id=row["owner_id"],
            session_id=row["session_id"],
            job_type=row["job_type"],
            dedupe_key=row["dedupe_key"],
            payload=row["payload"],
            available_at=row["available_at"],
            connection=connection,  # type: ignore[arg-type]
        )

        self.assertTrue(result.created)
        self.assertEqual(result.job.max_attempts, 7)
        self.assertEqual(connection.calls[-1][1][6], 7)

    async def test_lock_claim_revalidates_inside_caller_transaction(self) -> None:
        token = uuid4()
        row = _job_row(token)
        connection = _Connection([row], active=True)
        repository = MemoryJobRepository(_Pool())  # type: ignore[arg-type]

        locked = await repository.lock_claim(
            owner_id=row["owner_id"],
            job_id=row["id"],
            claim_token=token,
            connection=connection,  # type: ignore[arg-type]
        )

        self.assertEqual(locked.id, row["id"])
        self.assertIn("for update", connection.calls[0][0].lower())

    async def test_completion_requires_active_caller_transaction(self) -> None:
        token = uuid4()
        row = _job_row(token)
        repository = MemoryJobRepository(_Pool())  # type: ignore[arg-type]

        with self.assertRaises(MemoryJobTransactionRequiredError):
            await repository.complete(
                owner_id=row["owner_id"],
                job_id=row["id"],
                claim_token=token,
                completed_at=datetime.now(UTC),
                connection=_Connection([], active=False),  # type: ignore[arg-type]
            )


def _job_row(token):  # type: ignore[no-untyped-def]
    now = datetime.now(UTC)
    return {
        "id": uuid4(),
        "owner_id": "owner",
        "session_id": uuid4(),
        "job_type": "test",
        "dedupe_key": "test:1",
        "payload": {},
        "status": "claimed",
        "priority": 0,
        "attempt_count": 1,
        "max_attempts": 3,
        "available_at": now,
        "claimed_at": now,
        "claimed_by": "worker",
        "claim_token": token,
        "result": None,
        "last_error": None,
        "completed_at": None,
        "created_at": now,
        "updated_at": now,
    }
