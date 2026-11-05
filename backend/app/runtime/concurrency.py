from __future__ import annotations

import asyncio
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass
from enum import Enum
from typing import AsyncIterator


class ModelCallPriority(str, Enum):
    HIGH = "high"
    NORMAL = "normal"


@dataclass(frozen=True)
class _Waiter:
    sequence: int
    priority: ModelCallPriority


class PriorityAsyncLimiter:
    """Process-local concurrency limiter with high and normal FIFO wait classes.

    High-priority work is selected before queued normal work. Calls that already
    hold a permit are never preempted.
    """

    def __init__(self, max_concurrency: int) -> None:
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be at least 1")
        self._max_concurrency = max_concurrency
        self._active_count = 0
        self._next_sequence = 0
        self._high_waiters: deque[_Waiter] = deque()
        self._normal_waiters: deque[_Waiter] = deque()
        self._condition = asyncio.Condition()

    @property
    def max_concurrency(self) -> int:
        return self._max_concurrency

    @property
    def active_count(self) -> int:
        return self._active_count

    @property
    def waiting_count(self) -> int:
        return len(self._high_waiters) + len(self._normal_waiters)

    async def acquire(self, priority: ModelCallPriority = ModelCallPriority.NORMAL) -> None:
        async with self._condition:
            waiter = _Waiter(sequence=self._next_sequence, priority=priority)
            self._next_sequence += 1
            queue = self._queue(priority)
            queue.append(waiter)
            try:
                await self._condition.wait_for(lambda: self._can_acquire(waiter))
            except BaseException:
                self._remove_waiter(waiter)
                self._condition.notify_all()
                raise
            queue.popleft()
            self._active_count += 1

    async def release(self) -> None:
        async with self._condition:
            if self._active_count < 1:
                raise RuntimeError("PriorityAsyncLimiter released without an active permit")
            self._active_count -= 1
            self._condition.notify_all()

    @asynccontextmanager
    async def limit(
        self,
        priority: ModelCallPriority = ModelCallPriority.NORMAL,
    ) -> AsyncIterator[None]:
        await self.acquire(priority)
        try:
            yield
        finally:
            await self.release()

    def _can_acquire(self, waiter: _Waiter) -> bool:
        if self._active_count >= self._max_concurrency:
            return False
        if waiter.priority is ModelCallPriority.HIGH:
            return bool(self._high_waiters and self._high_waiters[0] is waiter)
        return (
            not self._high_waiters
            and bool(self._normal_waiters)
            and self._normal_waiters[0] is waiter
        )

    def _queue(self, priority: ModelCallPriority) -> deque[_Waiter]:
        return self._high_waiters if priority is ModelCallPriority.HIGH else self._normal_waiters

    def _remove_waiter(self, waiter: _Waiter) -> None:
        queue = self._queue(waiter.priority)
        try:
            queue.remove(waiter)
        except ValueError:
            return
