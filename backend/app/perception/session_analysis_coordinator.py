from __future__ import annotations

import asyncio
import logging
from collections import deque
from collections.abc import Awaitable, Callable
from typing import Generic, TypeVar
from uuid import UUID


LOGGER = logging.getLogger(__name__)
T = TypeVar("T")


class SessionAnalysisCoordinator(Generic[T]):
    """Serialize work per session and retain at most one pending latest item."""

    def __init__(
        self,
        processor: Callable[[T], Awaitable[None]],
        *,
        coalesce_latest: bool = True,
    ) -> None:
        self._processor = processor
        self._coalesce_latest = coalesce_latest
        self._lock = asyncio.Lock()
        self._active_sessions: set[UUID] = set()
        self._pending: dict[UUID, deque[T]] = {}
        self._tasks: set[asyncio.Task[None]] = set()

    async def submit(self, session_id: UUID, item: T) -> T | None:
        async with self._lock:
            if session_id in self._active_sessions:
                queue = self._pending.setdefault(session_id, deque())
                replaced = queue[-1] if self._coalesce_latest and queue else None
                if self._coalesce_latest:
                    queue.clear()
                queue.append(item)
                return replaced
            self._active_sessions.add(session_id)
            task = asyncio.create_task(self._run_session(session_id, item))
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)
            return None

    async def pending_count(self, session_id: UUID) -> int:
        async with self._lock:
            return int(session_id in self._active_sessions) + int(
                len(self._pending.get(session_id, ()))
            )

    async def drain(self) -> None:
        while self._tasks:
            await asyncio.gather(*tuple(self._tasks), return_exceptions=True)

    async def close(self) -> None:
        tasks = tuple(self._tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        async with self._lock:
            self._pending.clear()
            self._active_sessions.clear()

    async def _run_session(self, session_id: UUID, first: T) -> None:
        current = first
        try:
            while True:
                try:
                    await self._processor(current)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    LOGGER.exception(
                        "session analysis processor failed",
                        extra={"session_id": str(session_id)},
                    )
                async with self._lock:
                    queue = self._pending.get(session_id)
                    if not queue:
                        self._pending.pop(session_id, None)
                        self._active_sessions.discard(session_id)
                        return
                    current = queue.popleft()
                    if not queue:
                        self._pending.pop(session_id, None)
        finally:
            async with self._lock:
                self._pending.pop(session_id, None)
                self._active_sessions.discard(session_id)
