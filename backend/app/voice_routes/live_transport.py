"""One server-authenticated GPT Live sideband, with direct client WebRTC media.

No connection is made by construction. There is no retry, alternate endpoint,
audio forwarding, automatic reconnection, task execution, or SDK agent here.
The route owns the sole reader and protocol receipts. Closing this socket alone
does NOT stop the phone's primary media connection or prove provider finalization.
"""
from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass
import json
import logging
import math
import ssl
from typing import Any, Protocol

import certifi
import httpx
from websockets.asyncio.client import connect

from app.attention.quiet_delivery import QuietDeliverySuppressed
from app.voice_routes.live_protocol import (
    LIVE_SESSION_URL, LiveSessionAnswer, parse_webrtc_session_answer,
)


Guard = Callable[[], AbstractAsyncContextManager[Any]]


class LiveTransportError(RuntimeError):
    """Diagnostic codes only; never retain credentials, SDP or provider bodies."""


@dataclass(frozen=True)
class LiveSendEvidence:
    """Local write-boundary observations, NOT provider ACK or recovery authority."""
    stage: str
    outcome: str
    socket_handoff: bool
    socket_send_completed: bool
    error_type: str | None
    error_code: str | None


def _send_error_fields(error):
    if error is None:
        return None, None
    name = type(error).__name__
    if name not in {"SourceError", "LiveContextError", "TaskError", "TimeoutError", "CancelledError",
            "QuietDeliverySuppressed", "LiveTransportError", "PermissionError", "ValueError", "RuntimeError", "OSError"}:
        name = "unrecognized_type"
    code = getattr(error, "code", None)
    if type(error) is LiveTransportError and error.args:
        code = error.args[0]
    if code is not None and (type(code) is not str or code not in {
            "live_context_not_current", "live_context_revoked", "live_context_changed",
            "live_state_runtime_unavailable", "live_state_scope_unavailable", "source_payload_unavailable",
            "voice_context_expired", "live_start_context_unavailable", "live_notification_unavailable",
            "command_too_large", "sideband_write_unconfirmed"}):
        code = "unrecognized_code"
    return name, code


class LiveSocket(Protocol):
    async def send(self, message: str) -> None: ...
    async def recv(self) -> str | bytes: ...
    async def close(self) -> None: ...


@asynccontextmanager
async def _unguarded_control():
    yield


class LiveBackendTransport:
    """HTTP bootstrap and one bounded, serialized sideband writer.

An injected HTTP client is borrowed. The WebSocket is always owned. Injected
connectors are for deployment/test composition, not a second provider receiver.
Use only the answer created by this instance for attachment. ``current`` is
checked after the canonical guard and immediately before the actual write.
"""

    def __init__(
        self, api_key: str, *, project: str | None = None,
        organization: str | None = None, http_client: httpx.AsyncClient | None = None,
        connector: Callable[..., Awaitable[LiveSocket]] = connect,
        io_timeout: float = 3.0, open_timeout: float = 10.0,
        max_message_bytes: int = 524288,
    ) -> None:
        for value in (api_key, project, organization):
            if value is not None and (not isinstance(value, str) or not value.strip()
                                      or any(ord(c) < 32 for c in value)):
                raise ValueError("Invalid authentication configuration.")
        if not api_key:
            raise ValueError("A server API key is required.")
        if any(type(v) not in (int, float) or not math.isfinite(v) or not 0 < v <= 120
               for v in (io_timeout, open_timeout)):
            raise ValueError("Invalid transport deadline.")
        if type(max_message_bytes) is not int or not 65536 <= max_message_bytes <= 2097152:
            raise ValueError("Invalid message bound.")
        self._headers = {"Authorization": "Bearer " + api_key}
        if project:
            self._headers["OpenAI-Project"] = project
        if organization:
            self._headers["OpenAI-Organization"] = organization
        self._http = http_client
        self._owns_http = http_client is None
        # Some local Python builds have no usable default public CA bundle.
        # Keep system trust and add the same public roots used by HTTPX.
        self._ssl_context = ssl.create_default_context()
        self._ssl_context.load_verify_locations(cafile=certifi.where())
        self._connector = connector
        self._io_timeout, self._open_timeout = io_timeout, open_timeout
        self._max_bytes = max_message_bytes
        self._socket: LiveSocket | None = None
        self._writer = asyncio.Lock()
        self._receiving = self._created = self._attaching = self._closed = False
        self.creation_submitted = False
        self.answer: LiveSessionAnswer | None = None

    @property
    def attached(self) -> bool:
        return self._socket is not None and not self._closed

    async def create(self, body: dict, *, guard: Guard,
                     current: Callable[[], bool]) -> LiveSessionAnswer:
        if self._created or self._closed:
            raise LiveTransportError("bootstrap_reused_or_closed")
        self._created = True
        if self._http is None:
            self._http = httpx.AsyncClient(timeout=self._open_timeout, follow_redirects=False)
        try:
            async with asyncio.timeout(self._open_timeout):
                async with guard():
                    if self._closed or not current():
                        raise LiveTransportError("bootstrap_fenced")
                    self.creation_submitted = True
                    async with self._http.stream("POST", LIVE_SESSION_URL, json=body,
                                                 headers=self._headers, follow_redirects=False) as response:
                        if response.status_code != 201:
                            raise LiveTransportError("session_creation_rejected")
                        data = bytearray()
                        async for part in response.aiter_bytes():
                            if len(data) + len(part) > self._max_bytes:
                                raise LiveTransportError("session_answer_too_large")
                            data.extend(part)
                    self.answer = parse_webrtc_session_answer(json.loads(data))
                    if self._closed or not current():
                        raise LiveTransportError("bootstrap_fenced_after_submission")
                    return self.answer
        except asyncio.CancelledError:
            raise
        except LiveTransportError:
            raise
        except Exception:
            raise LiveTransportError("session_creation_unconfirmed") from None

    async def attach(self, answer: LiveSessionAnswer) -> None:
        if self._closed or self._attaching or self._socket is not None or answer is not self.answer:
            raise LiveTransportError("attachment_not_owned")
        self._attaching = True
        socket = None
        try:
            async with asyncio.timeout(self._open_timeout):
                socket = await self._connector(
                    answer.sideband_url, additional_headers=dict(self._headers),
                    ssl=self._ssl_context,
                    open_timeout=self._open_timeout, close_timeout=self._io_timeout,
                    max_size=self._max_bytes, max_queue=16, write_limit=32768,
                    compression=None, ping_interval=20, ping_timeout=20,
                )
                if self._closed:
                    raise LiveTransportError("attachment_fenced")
                self._socket = socket
        except BaseException as error:
            if socket is not None and self._socket is not socket:
                try:
                    async with asyncio.timeout(self._io_timeout):
                        await socket.close()
                except (Exception, asyncio.CancelledError):
                    pass
            if isinstance(error, asyncio.CancelledError):
                raise
            logging.getLogger(__name__).warning(
                "Live sideband attachment failed (%s)", type(error).__name__)
            raise LiveTransportError("attachment_unconfirmed") from None
        finally:
            self._attaching = False

    async def send(self, payload: Callable[[], dict | None], *,
                   guard: Guard = _unguarded_control,
                   current: Callable[[], bool] = lambda: True,
                   observe: Callable[[LiveSendEvidence], None] | None = None) -> bool:
        socket_handoff = False
        socket_send_completed = False
        stage, outcome, error = "writer_wait", "error", None
        try:
            async with asyncio.timeout(self._io_timeout):
                async with self._writer:
                    stage = "guard_entry"
                    async with guard():
                        stage = "pre_factory_current"
                        if not self.attached or not current():
                            outcome = "not_written"
                            return False
                        stage = "payload_factory"
                        value = payload()
                        stage = "post_factory_current"
                        if value is None or not self.attached or not current():
                            outcome = "not_written"
                            return False
                        stage = "encode"
                        data = json.dumps(value, ensure_ascii=True, allow_nan=False)
                        if len(data.encode()) > self._max_bytes:
                            raise LiveTransportError("command_too_large")
                        # A socket call may expose output even if it later raises.
                        # Guard-exit errors cannot retroactively prove no handoff.
                        stage = "socket_send"
                        socket_handoff = True
                        await self._socket.send(data)
                        socket_send_completed = True
                        stage, outcome = "guard_exit", "written"
                        return True
        except asyncio.CancelledError as caught:
            outcome, error = "cancelled", caught
            raise
        except QuietDeliverySuppressed as caught:
            outcome, error = "error", caught
            if not socket_handoff:
                # Preserve canonical prewrite refusal through the adapter. It is
                # not source withdrawal, unknown output, or task cancellation.
                raise
            raise LiveTransportError("sideband_write_unconfirmed") from None
        except Exception as caught:
            outcome, error = "error", caught
            raise LiveTransportError("sideband_write_unconfirmed") from None
        finally:
            # One bounded synchronous observation per invocation. The callback
            # never receives payloads or exceptions, and cannot change policy.
            if observe is not None:
                try:
                    error_type, error_code = _send_error_fields(error)
                    observe(LiveSendEvidence(stage, outcome, socket_handoff,
                        socket_send_completed, error_type, error_code))
                except Exception:
                    pass

    async def receive(self) -> dict:
        if not self.attached or self._receiving:
            raise LiveTransportError("receiver_not_owned")
        self._receiving = True
        try:
            raw = await self._socket.recv()
            if not isinstance(raw, (str, bytes)) or len(raw) > self._max_bytes:
                raise LiveTransportError("invalid_provider_message")
            if isinstance(raw, str) and len(raw.encode()) > self._max_bytes:
                raise LiveTransportError("invalid_provider_message")
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise LiveTransportError("invalid_provider_message")
            return value
        except asyncio.CancelledError:
            raise
        except Exception:
            raise LiveTransportError("sideband_receive_failed") from None
        finally:
            self._receiving = False

    async def aclose(self) -> bool:
        self._closed = True
        success = True
        if self._socket is not None:
            try:
                async with asyncio.timeout(self._io_timeout):
                    await self._socket.close()
                self._socket = None
            except Exception:
                success = False
        if self._owns_http and self._http is not None:
            try:
                async with asyncio.timeout(self._io_timeout):
                    await self._http.aclose()
            except Exception:
                success = False
        return success
