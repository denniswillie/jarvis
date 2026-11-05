from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

from app.reminders.scheduler import ReminderScheduler


class ReminderSchedulerTests(unittest.IsolatedAsyncioTestCase):
    async def test_terminal_delivery_failure_is_recorded_once(self) -> None:
        reminder = SimpleNamespace(id=uuid4(), trigger_attempt_count=3, remind_at=None)
        repository = SimpleNamespace(
            claim_due_reminders=AsyncMock(return_value=[reminder]),
            record_trigger_failure=AsyncMock(return_value=True),
        )
        delivery = SimpleNamespace(deliver=AsyncMock(side_effect=RuntimeError("failed")))
        settings = SimpleNamespace(
            JARVIS_OWNER_ID="owner",
            REMINDER_TRIGGER_CLAIM_TIMEOUT_SECONDS=120,
            REMINDER_TRIGGER_MAX_ATTEMPTS=3,
        )

        scheduler = ReminderScheduler(
            settings=settings,
            repository=repository,
            delivery=delivery,
        )

        processed = await scheduler.process_due_reminders()

        self.assertEqual(processed, 1)
        repository.record_trigger_failure.assert_awaited_once_with(
            owner_id="owner",
            reminder=reminder,
            error_code="RuntimeError",
            terminal=True,
        )

    async def test_nonterminal_delivery_failure_releases_claim_for_retry(self) -> None:
        reminder = SimpleNamespace(id=uuid4(), trigger_attempt_count=1, remind_at=None)
        repository = SimpleNamespace(
            claim_due_reminders=AsyncMock(return_value=[reminder]),
            record_trigger_failure=AsyncMock(return_value=True),
        )
        delivery = SimpleNamespace(deliver=AsyncMock(side_effect=RuntimeError("failed")))
        settings = SimpleNamespace(
            JARVIS_OWNER_ID="owner",
            REMINDER_TRIGGER_CLAIM_TIMEOUT_SECONDS=120,
            REMINDER_TRIGGER_MAX_ATTEMPTS=3,
        )

        scheduler = ReminderScheduler(
            settings=settings,
            repository=repository,
            delivery=delivery,
        )

        await scheduler.process_due_reminders()

        repository.record_trigger_failure.assert_awaited_once_with(
            owner_id="owner",
            reminder=reminder,
            error_code="RuntimeError",
            terminal=False,
        )


if __name__ == "__main__":
    unittest.main()
