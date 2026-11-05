from __future__ import annotations

import unittest
from types import SimpleNamespace
from uuid import uuid4

from app.reminders.repository import ReminderRepository


class _AsyncContext:
    def __init__(self, value: object) -> None:
        self._value = value

    async def __aenter__(self) -> object:
        return self._value

    async def __aexit__(self, *_: object) -> None:
        return None


class _Connection:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def transaction(self) -> _AsyncContext:
        return _AsyncContext(self)

    async def fetchrow(self, query: str, *_: object) -> None:
        self.queries.append(query)
        return None


class _Pool:
    def __init__(self, connection: _Connection) -> None:
        self._connection = connection

    def acquire(self) -> _AsyncContext:
        return _AsyncContext(self._connection)


class ReminderRepositoryClaimTests(unittest.IsolatedAsyncioTestCase):
    async def test_finalize_requires_active_reminder(self) -> None:
        connection = _Connection()
        repository = ReminderRepository(_Pool(connection))  # type: ignore[arg-type]

        with self.assertRaises(RuntimeError):
            await repository.finalize_trigger(
                owner_id="owner",
                reminder_id=uuid4(),
                claim_token=uuid4(),
                session_id=None,
                message="message",
            )

        self.assertIn("and status = 'active'", connection.queries[0])

    async def test_manual_completion_and_deletion_invalidate_claim(self) -> None:
        connection = _Connection()
        repository = ReminderRepository(_Pool(connection))  # type: ignore[arg-type]

        await repository.complete_reminder(owner_id="owner", reminder_id=uuid4())
        await repository.delete_reminder(owner_id="owner", reminder_id=uuid4())

        for query in connection.queries:
            self.assertIn("trigger_claimed_at = null", query)
            self.assertIn("trigger_claim_token = null", query)

    async def test_failure_update_requires_active_reminder(self) -> None:
        connection = _Connection()
        repository = ReminderRepository(_Pool(connection))  # type: ignore[arg-type]
        reminder = SimpleNamespace(
            id=uuid4(),
            trigger_claim_token=uuid4(),
            trigger_attempt_count=3,
        )

        recorded = await repository.record_trigger_failure(
            owner_id="owner",
            reminder=reminder,
            error_code="RuntimeError",
            terminal=True,
        )

        self.assertFalse(recorded)
        self.assertIn("and status = 'active'", connection.queries[0])


if __name__ == "__main__":
    unittest.main()
