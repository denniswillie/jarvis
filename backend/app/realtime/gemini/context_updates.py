"""Coalesced Jarvis context refreshes without unsolicited Gemini activity."""
from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any

from app.assistant.memory_context import (
    owner_working_memory_revision,
    session_working_memory_revision,
    working_memory_app_event_payload,
    working_memory_context_available,
    working_memory_next_expires_at,
)
from app.realtime.context_refresh import RealtimeContextRefreshCoordinator
from app.realtime.gemini.client import GeminiTransportError
from app.runtime.events import DomainEvent, MemoryItemChangedPayload


class GeminiContextUpdates:
    def __init__(
        self, *, context, build: Callable[[], Awaitable[dict | None]],
        emit: Callable[[dict], Awaitable[None]], closed: Callable[[], bool],
        timeout_seconds: float, minimum_interval_seconds: float = 2.0,
    ) -> None:
        self.context, self._build, self._emit = context, build, emit
        self._session_closed, self._timeout_seconds = closed, timeout_seconds
        self._stopped = False
        self._session_revision = 0
        self._owner_revision = 0
        self._expiry_task: asyncio.Task | None = None
        self.coordinator = RealtimeContextRefreshCoordinator(
            self._refresh, minimum_interval_seconds=minimum_interval_seconds,
        )

    @property
    def closed(self) -> bool:
        return self._stopped or self._session_closed()

    def initialize(self, context: dict | None) -> None:
        if self.closed or not isinstance(context, dict):
            return
        working = context.get("working_memory")
        if not working_memory_context_available(working):
            return
        self._record(working)
        self.coordinator.initialize(session_working_memory_revision(working))

    async def notify_working_memory_updated(self, revision: int) -> None:
        if not self.closed:
            await self.coordinator.notify(revision)

    async def notify_owner_working_memory_updated(self, revision: int) -> None:
        if not self.closed and revision > self._owner_revision:
            await self.coordinator.invalidate()

    async def notify_memory_item_changed(self, event: DomainEvent) -> None:
        payload = event.payload
        if (self.closed or not isinstance(payload, MemoryItemChangedPayload)
                or payload.owner_id != self.context.owner_id
                or payload.session_id not in {None, self.context.jarvis_session_id}):
            return
        await self._emit({
            "type": "memory.updated", "change_id": str(event.event_id),
            "changed_at": event.occurred_at.isoformat(),
            "affected_memory_ids": [str(payload.memory_item_id)],
            "affected_memory_types": [payload.memory_type] if payload.memory_type is not None else [],
            "reason": payload.reason or payload.change_type, "change_type": payload.change_type,
        })
        if not self.closed:
            await self.coordinator.invalidate()

    async def _refresh(self, requested_revision: int | None) -> int | None:
        if self.closed:
            return None
        async with asyncio.timeout(self._timeout_seconds):
            context = await self._build()
        if self.closed:
            return None
        working = context.get("working_memory") if isinstance(context, dict) else None
        if not working_memory_context_available(working):
            raise GeminiTransportError("gemini_memory_context_unavailable")
        revision = session_working_memory_revision(working)
        owner_revision = owner_working_memory_revision(working)
        if requested_revision is not None and (revision is None or revision < requested_revision):
            return None
        if ((revision is not None and revision < self._session_revision)
                or (owner_revision is not None and owner_revision < self._owner_revision)):
            raise GeminiTransportError("gemini_memory_revision_regressed")
        expires = working_memory_next_expires_at(working)
        if expires is not None and expires <= datetime.now(UTC):
            raise GeminiTransportError("gemini_memory_context_expired")
        await self._emit({
            "type": "working_memory.updated", "session_id": str(self.context.jarvis_session_id),
            **working_memory_app_event_payload(working),
        })
        if self.closed:
            return None
        self._record(working)
        # This is readiness of shared application context, not a claim that a
        # native model has acknowledged a prompt update. The session obtains fresh
        # provider context at its next response-preparation boundary. No realtime
        # text, activity markers, or inference is sent merely for a revision.
        return revision

    def _record(self, working: dict[str, Any]) -> None:
        self._session_revision = max(self._session_revision, session_working_memory_revision(working) or 0)
        self._owner_revision = max(self._owner_revision, owner_working_memory_revision(working) or 0)
        if self._expiry_task is not None and self._expiry_task is not asyncio.current_task():
            self._expiry_task.cancel()
        self._expiry_task = None
        expires = working_memory_next_expires_at(working)
        if expires is not None and not self.closed:
            self._expiry_task = asyncio.create_task(self._expire(expires))

    async def _expire(self, expires: datetime) -> None:
        try:
            await asyncio.sleep(max(0, (expires - datetime.now(UTC)).total_seconds()))
            if not self.closed:
                await self.coordinator.invalidate()
        finally:
            if self._expiry_task is asyncio.current_task():
                self._expiry_task = None

    def stop(self) -> None:
        self._stopped = True
        self.coordinator.stop()
        if self._expiry_task is not None:
            self._expiry_task.cancel()

    async def close(self) -> None:
        self.stop()
        await self.coordinator.close()
        if self._expiry_task is not None:
            await asyncio.gather(self._expiry_task, return_exceptions=True)
            self._expiry_task = None

    def diagnostics(self) -> dict[str, Any]:
        return {"session_revision": self._session_revision, "owner_revision": self._owner_revision,
                "expiry_pending": self._expiry_task is not None and not self._expiry_task.done(),
                "stopped": self._stopped}
