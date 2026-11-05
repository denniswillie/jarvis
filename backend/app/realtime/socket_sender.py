from __future__ import annotations

import asyncio
import json
from contextlib import AsyncExitStack
from collections.abc import Awaitable, Callable

from fastapi import WebSocket, WebSocketDisconnect

from app.realtime.telemetry import RealtimeTrace


# Starlette and Uvicorn use these precise errors when the peer closes between
# a state check and the awaited ASGI send. Other RuntimeErrors are bugs.
_CLOSED_SEND_ERRORS = frozenset({
    'Cannot call "send" once a close message has been sent.',
    "Unexpected ASGI message 'websocket.send', after sending 'websocket.close' or response already completed.",
    "Unexpected ASGI message 'websocket.close', after sending 'websocket.close' or response already completed.",
})


def is_closed_websocket_error(error: BaseException) -> bool:
    return isinstance(error, (WebSocketDisconnect, OSError)) or (
        isinstance(error, RuntimeError) and str(error) in _CLOSED_SEND_ERRORS
    )


class WebSocketEventSender:
    """Serialize JSON events/close and retire a departed peer exactly once."""

    def __init__(
        self, websocket: WebSocket, *, on_closed: Callable[[], None] | None = None,
        trace: RealtimeTrace | None = None,
    ) -> None:
        self._websocket = websocket
        self._on_closed = on_closed
        self._lock = asyncio.Lock()
        self._closed = False
        self._trace = trace

    @property
    def closed(self) -> bool:
        return self._closed

    def mark_closed(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._on_closed is not None:
            self._on_closed()

    async def send_event(
        self, payload: dict[str, object], *, should_send: Callable[[], bool] | None = None,
    ) -> bool:
        sent = await self._send(lambda: self._websocket.send_json(payload), should_send=should_send)
        if sent and self._trace is not None:
            self._trace.forwarded(payload)
        return sent

    async def send_text(
        self, payload: str, *, should_send: Callable[[], bool] | None = None,
        write_guard=None,
    ) -> bool:
        """Send an already-encoded envelope through the same send/close boundary."""
        sent = await self._send(lambda: self._websocket.send_text(payload), should_send=should_send, write_guard=write_guard)
        if sent and self._trace is not None:
            try:
                event = json.loads(payload)
            except (ValueError, TypeError):
                return sent
            if isinstance(event, dict):
                self._trace.forwarded(event)
        return sent

    async def _send(
        self, send: Callable[[], Awaitable[None]], *,
        should_send: Callable[[], bool] | None = None,
        write_guard=None,
    ) -> bool:
        async with self._lock:
            if self._closed or (should_send is not None and not should_send()):
                return False
            try:
                async with AsyncExitStack() as stack:
                    mark_sent = await stack.enter_async_context(write_guard()) if write_guard is not None else None
                    # Revocation/barge-in may have happened while acquiring the
                    # durable source fence, even though this writer is held.
                    if self._closed or (should_send is not None and not should_send()):
                        return False
                    await send()
                    if mark_sent is not None:
                        mark_sent()
            except (WebSocketDisconnect, OSError, RuntimeError) as error:
                if not is_closed_websocket_error(error):
                    if self._trace is not None:
                        self._trace.event("socket_write_failed", reason="write_outcome_unknown", error_type=type(error).__name__)
                    raise
                self.mark_closed()
                if self._trace is not None:
                    self._trace.event("socket_write_failed", reason="peer_closed", error_type=type(error).__name__)
                return False
            except (Exception, asyncio.CancelledError) as error:
                if self._trace is not None:
                    self._trace.event("socket_write_failed", reason="write_outcome_unknown", error_type=type(error).__name__)
                raise
            return True

    async def close(self, *, code: int = 1000, reason: str = "") -> None:
        async with self._lock:
            if self._closed:
                return
            self.mark_closed()
            try:
                await self._websocket.close(code=code, reason=reason)
            except (WebSocketDisconnect, OSError, RuntimeError) as error:
                if not is_closed_websocket_error(error):
                    raise
