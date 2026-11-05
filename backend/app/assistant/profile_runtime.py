"""Bounded, best-effort delivery of committed preference changes.

The repository remains authoritative. This process-local notifier is not a
durable outbox: reconnect reads the current profile rather than replaying edits.
"""
from __future__ import annotations

import asyncio
import logging


LOGGER = logging.getLogger(__name__)


class BehaviorProfileUpdates:
    def __init__(self, sessions, *, max_owners=32, timeout_seconds=2.0):
        self.sessions = sessions
        self.max_owners = max_owners
        self.timeout_seconds = timeout_seconds
        self.pending = {}
        self.task = None
        self.closed = False

    def submit(self, event):
        """Synchronous callback after commit; never waits for a phone/model."""
        if self.closed:
            return False
        previous = self.pending.get(event.owner_id)
        if previous is not None and previous.revision >= event.revision:
            return False
        if previous is None and len(self.pending) >= self.max_owners:
            LOGGER.warning("profile notification capacity reached; reconnect reload required")
            return False
        self.pending[event.owner_id] = event
        if self.task is None or self.task.done():
            self.task = asyncio.create_task(self._drain())
        return True

    async def _drain(self):
        while self.pending and not self.closed:
            owner = next(iter(self.pending))
            event = self.pending.pop(owner)
            try:
                async with asyncio.timeout(self.timeout_seconds):
                    await self.sessions.handle_behavior_profile_committed(event)
            except asyncio.CancelledError:
                raise
            except Exception as error:
                LOGGER.warning("profile notification unavailable error_type=%s", type(error).__name__)

    async def close(self):
        self.closed = True
        self.pending.clear()
        if self.task is not None and not self.task.done():
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)
