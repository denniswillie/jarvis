from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from typing import Any

from app.experience.service import ExperienceEventService


LOGGER = logging.getLogger(__name__)


class ExperienceEventSweeper:
    def __init__(self, service: ExperienceEventService) -> None:
        self._service = service
        self.last_run_at: datetime | None = None
        self.last_success_at: datetime | None = None
        self.last_error: str | None = None
        self.closed_count = 0

    async def run_once(self) -> int:
        self.last_run_at = datetime.now(UTC)
        try:
            closed = await self._service.sweep_due_events(now=self.last_run_at)
        except Exception as exc:
            self.last_error = type(exc).__name__
            raise
        self.last_success_at = datetime.now(UTC)
        self.last_error = None
        self.closed_count += closed
        return closed

    def diagnostics(self) -> dict[str, Any]:
        return {
            "last_run_at": self.last_run_at,
            "last_success_at": self.last_success_at,
            "last_error": self.last_error,
            "closed_count": self.closed_count,
        }


async def run_experience_event_sweeper_loop(
    sweeper: ExperienceEventSweeper,
    *,
    interval_seconds: int,
) -> None:
    while True:
        try:
            await sweeper.run_once()
        except asyncio.CancelledError:
            raise
        except Exception:
            LOGGER.exception("experience event sweeper failed")
        await asyncio.sleep(interval_seconds)
