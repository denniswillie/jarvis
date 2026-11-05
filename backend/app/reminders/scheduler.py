from __future__ import annotations

import asyncio
import logging
from typing import Any, Protocol


LOGGER = logging.getLogger(__name__)
CLAIM_BATCH_SIZE = 1


class ReminderSchedulerSettings(Protocol):
    JARVIS_OWNER_ID: str
    REMINDER_TRIGGER_CLAIM_TIMEOUT_SECONDS: int
    REMINDER_TRIGGER_MAX_ATTEMPTS: int


class ReminderSchedulerRepository(Protocol):
    async def claim_due_reminders(self, **kwargs: Any) -> list[Any]: ...

    async def record_trigger_failure(self, **kwargs: Any) -> bool: ...


class ReminderSchedulerDelivery(Protocol):
    async def deliver(self, reminder: Any) -> None: ...


class ReminderScheduler:
    def __init__(
        self,
        *,
        settings: ReminderSchedulerSettings,
        repository: ReminderSchedulerRepository,
        delivery: ReminderSchedulerDelivery,
    ) -> None:
        self._settings = settings
        self._repository = repository
        self._delivery = delivery

    async def process_due_reminders(self) -> int:
        reminders = await self._repository.claim_due_reminders(
            owner_id=self._settings.JARVIS_OWNER_ID,
            claim_timeout_seconds=self._settings.REMINDER_TRIGGER_CLAIM_TIMEOUT_SECONDS,
            max_attempts=self._settings.REMINDER_TRIGGER_MAX_ATTEMPTS,
            limit=CLAIM_BATCH_SIZE,
        )
        for reminder in reminders:
            LOGGER.info(
                "due reminder claimed reminder_id=%s attempt=%s scheduled_at=%s",
                reminder.id,
                reminder.trigger_attempt_count,
                reminder.remind_at,
            )
            try:
                await self._delivery.deliver(reminder)
            except Exception as error:
                terminal = reminder.trigger_attempt_count >= self._settings.REMINDER_TRIGGER_MAX_ATTEMPTS
                LOGGER.exception(
                    "reminder trigger failed reminder_id=%s attempt=%s terminal=%s",
                    reminder.id,
                    reminder.trigger_attempt_count,
                    terminal,
                )
                recorded = await self._repository.record_trigger_failure(
                    owner_id=self._settings.JARVIS_OWNER_ID,
                    reminder=reminder,
                    error_code=type(error).__name__,
                    terminal=terminal,
                )
                if not recorded:
                    LOGGER.info(
                        "reminder trigger claim no longer owned reminder_id=%s attempt=%s",
                        reminder.id,
                        reminder.trigger_attempt_count,
                    )
        return len(reminders)


async def run_reminder_scheduler_loop(
    scheduler: ReminderScheduler,
    *,
    interval_seconds: int,
) -> None:
    LOGGER.info("reminder scheduler started interval_seconds=%s", interval_seconds)
    try:
        while True:
            try:
                await scheduler.process_due_reminders()
            except Exception:
                LOGGER.exception("reminder scheduler iteration failed")
            await asyncio.sleep(max(1, interval_seconds))
    finally:
        LOGGER.info("reminder scheduler stopped")
