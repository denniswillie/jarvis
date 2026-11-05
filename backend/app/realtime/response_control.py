"""Bounded, session-owned cancellation writes outside the event readers.

This lane does not cancel application tasks. It fences response submission until
old speech controls have drained. Unknown response identities are resolved from
the matching creation acknowledgment, never by cancelling an arbitrary response.
"""
from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass


class ResponseControlPending(Exception):
    """The response factory lost a race to a control before any response write."""


class ResponsePrewriteRejected(Exception):
    """The native writer withheld this exact response before socket submission."""

    def __init__(self, request_id: str, reason: str):
        if reason not in {"route_not_ready", "notification_not_current"}:
            raise ValueError("Unsupported response prewrite rejection.")
        self.request_id, self.reason = request_id, reason
        super().__init__(reason)


@dataclass
class _Control:
    kind: str
    request_id: str | None
    response: asyncio.Future[str | None] | None
    key: tuple[str, str]
    fence: asyncio.Future | None = None


class ResponseControlLane:
    def __init__(
        self, *, send: Callable[[str, str | None, str | None], Awaitable[None]],
        fail: Callable[[str], None], deadline: float, capacity: int = 32,
    ) -> None:
        self._send, self._fail = send, fail
        self._deadline, self._capacity = deadline, capacity
        self._queue: deque[_Control] = deque()
        self._pending: dict[tuple[str, str], _Control] = {}
        self._idle = asyncio.Event()
        self._idle.set()
        self._task: asyncio.Task[None] | None = None
        self._closed = False

    @property
    def busy(self) -> bool:
        return bool(self._pending)

    def cancel(self, response_id: str | None, request_id: str | None) -> None:
        if self._closed:
            return
        if response_id is None and request_id is None:
            self._fail("response_control_identity_missing")
            self.retire()
            return
        key = ("request", request_id) if request_id else ("response", response_id)
        existing = self._pending.get(key)
        if existing is not None:
            if response_id is not None and not existing.response.done():
                existing.response.set_result(response_id)
            return
        response = asyncio.get_running_loop().create_future()
        if response_id is not None:
            response.set_result(response_id)
        self._enqueue(_Control("cancel", request_id, response, key))

    def reject(self, request_id: str) -> None:
        control = self._pending.get(("request", request_id))
        if control is not None and not control.response.done():
            control.response.set_result(None)

    def clear_output(self) -> None:
        key = ("clear", "output")
        if key not in self._pending:
            self._enqueue(_Control("clear", None, None, key))

    def hold_fence(self, identity: str, fence: asyncio.Future) -> None:
        """Gate newer responses on an already-owned fence, without running it here."""
        key = ("fence", identity)
        if key not in self._pending:
            self._enqueue(_Control("fence", None, None, key, fence=fence))

    def _enqueue(self, control: _Control) -> None:
        if self._closed:
            return
        if len(self._pending) >= self._capacity:
            self._fail("response_control_queue_exhausted")
            self.retire()
            return
        self._pending[control.key] = control
        self._queue.append(control)
        self._idle.clear()
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run(), name="realtime-response-controls")

    async def _run(self) -> None:
        try:
            while self._queue and not self._closed:
                control = self._queue.popleft()
                async with asyncio.timeout(self._deadline):
                    if control.fence is not None:
                        # Delivery owns and joins this task. A lane timeout or
                        # shutdown must not abandon its speaker-fence cleanup.
                        await asyncio.shield(control.fence)
                    else:
                        response_id = await control.response if control.response is not None else None
                        if control.kind != "cancel" or response_id is not None:
                            await self._send(control.kind, response_id, control.request_id)
                self._pending.pop(control.key, None)
        except asyncio.CancelledError:
            raise
        except Exception:
            self._closed = True
            self._fail("response_control_failed")
        finally:
            self._queue.clear()
            for control in self._pending.values():
                if control.response is not None and not control.response.done():
                    control.response.cancel()
            self._pending.clear()
            self._idle.set()

    async def wait_idle(self) -> None:
        await self._idle.wait()
        if self._closed:
            raise RuntimeError("response control lane closed")

    def retire(self) -> None:
        self._closed = True
        if self._task is not None and not self._task.done():
            self._task.cancel()

    async def close(self) -> None:
        self.retire()
        if self._task is not None:
            await asyncio.gather(self._task, return_exceptions=True)
        # A task cancelled before entry never executes its finally block.
        self._queue.clear()
        for control in self._pending.values():
            if control.response is not None and not control.response.done():
                control.response.cancel()
        self._pending.clear()
        self._idle.set()
