from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable


LOGGER = logging.getLogger(__name__)
RefreshCallback = Callable[[int | None], Awaitable[int | None]]


class RealtimeContextRefreshCoordinator:
    """Coalesce monotonic working-memory revisions into bounded prompt refreshes."""

    def __init__(
        self,
        refresh: RefreshCallback,
        *,
        minimum_interval_seconds: float = 2.0,
        retry_delay_seconds: float = 0.5,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        if minimum_interval_seconds < 0:
            raise ValueError("minimum_interval_seconds cannot be negative")
        if retry_delay_seconds < 0:
            raise ValueError("retry_delay_seconds cannot be negative")
        self._refresh = refresh
        self._minimum_interval_seconds = minimum_interval_seconds
        self._retry_delay_seconds = retry_delay_seconds
        self._monotonic = monotonic
        self._sleep = sleep
        self._lock = asyncio.Lock()
        self._pending_revision: int | None = None
        self._invalidation_pending = False
        self._applied_revision = 0
        self._last_attempt_at: float | None = None
        self._task: asyncio.Task[None] | None = None
        self._closed = False

    @property
    def applied_revision(self) -> int:
        return self._applied_revision

    def initialize(self, revision: int | None) -> None:
        if revision is None:
            return
        self._applied_revision = max(self._applied_revision, revision)
        self._last_attempt_at = self._monotonic()

    async def notify(self, revision: int) -> bool:
        if revision < 1:
            return False
        async with self._lock:
            if self._closed or revision <= self._applied_revision:
                return False
            if self._pending_revision is None or revision > self._pending_revision:
                self._pending_revision = revision
            if self._task is None:
                self._task = asyncio.create_task(self._run())
            return True

    async def invalidate(self) -> bool:
        """Request a refresh that is not tied to a newer session-memory revision."""

        async with self._lock:
            if self._closed:
                return False
            self._invalidation_pending = True
            if self._task is None:
                self._task = asyncio.create_task(self._run())
            return True

    def stop(self) -> None:
        """Stop producing before transport teardown, without waiting for a write."""
        if self._closed:
            return
        # State changes never suspend while holding _lock, so this is atomic on
        # the coordinator's event loop. Keep the task until close() can join it.
        self._closed = True
        self._pending_revision = None
        self._invalidation_pending = False
        task = self._task
        if task is not None and task is not asyncio.current_task():
            task.cancel()

    async def close(self) -> None:
        self.stop()
        task = self._task
        if task is not None and task is not asyncio.current_task():
            try:
                await task
            except asyncio.CancelledError:
                pass

    async def wait_idle(self) -> None:
        while True:
            async with self._lock:
                task = self._task
            if task is None:
                return
            await asyncio.shield(task)

    async def _run(self) -> None:
        try:
            while True:
                async with self._lock:
                    target_revision = self._pending_revision
                    invalidation_requested = self._invalidation_pending
                    self._pending_revision = None
                    self._invalidation_pending = False
                    if target_revision is None and not invalidation_requested:
                        self._task = None
                        return
                    if (
                        target_revision is not None
                        and target_revision <= self._applied_revision
                        and not invalidation_requested
                    ):
                        continue

                await self._wait_for_throttle()

                async with self._lock:
                    if self._pending_revision is not None:
                        target_revision = max(
                            target_revision or 0,
                            self._pending_revision,
                        )
                        self._pending_revision = None
                    invalidation_requested = (
                        invalidation_requested or self._invalidation_pending
                    )
                    self._invalidation_pending = False
                    if self._closed:
                        return

                succeeded, applied_revision = await self._attempt_refresh(target_revision)
                if self._closed:
                    return
                if not succeeded:
                    await self._wait_for_retry()
                    async with self._lock:
                        if self._pending_revision is not None:
                            target_revision = max(
                                target_revision or 0,
                                self._pending_revision,
                            )
                            self._pending_revision = None
                        invalidation_requested = (
                            invalidation_requested or self._invalidation_pending
                        )
                        self._invalidation_pending = False
                        if self._closed:
                            return
                    succeeded, applied_revision = await self._attempt_refresh(
                        target_revision
                    )
                if not succeeded:
                    continue
                if (
                    target_revision is not None
                    and applied_revision is not None
                    and applied_revision >= target_revision
                ):
                    self._applied_revision = max(
                        self._applied_revision,
                        applied_revision,
                    )
        finally:
            async with self._lock:
                if self._task is asyncio.current_task():
                    self._task = None

    async def _wait_for_throttle(self) -> None:
        if self._last_attempt_at is None:
            return
        elapsed = self._monotonic() - self._last_attempt_at
        remaining = self._minimum_interval_seconds - elapsed
        if remaining > 0:
            await self._sleep(remaining)

    async def _wait_for_retry(self) -> None:
        elapsed = (
            self._monotonic() - self._last_attempt_at
            if self._last_attempt_at is not None
            else 0.0
        )
        await self._sleep(
            max(
                self._retry_delay_seconds,
                self._minimum_interval_seconds - elapsed,
                0.0,
            )
        )

    async def _attempt_refresh(
        self,
        target_revision: int | None,
    ) -> tuple[bool, int | None]:
        if self._closed:
            return False, None
        self._last_attempt_at = self._monotonic()
        try:
            applied_revision = await self._refresh(target_revision)
        except asyncio.CancelledError:
            raise
        except Exception:
            if self._closed:
                return False, None
            LOGGER.exception(
                "realtime context refresh failed revision=%s",
                target_revision,
            )
            return False, None
        if target_revision is None:
            return True, applied_revision
        return (
            applied_revision is not None and applied_revision >= target_revision,
            applied_revision,
        )
