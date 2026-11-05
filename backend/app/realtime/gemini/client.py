from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Sequence
from contextlib import suppress
from copy import deepcopy
import json
from typing import Any

import websockets

from app.realtime.gemini.config import GeminiLiveConfig


LIVE_ENDPOINT = (
    "wss://generativelanguage.googleapis.com/ws/"
    "google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"
)


class GeminiTransportError(RuntimeError):
    """Only fixed application codes escape this transport, never native payloads."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def build_setup_event(
    config: GeminiLiveConfig, *, instructions: str,
    tools: Sequence[dict[str, Any]] = (), resume_handle: str | None = None,
) -> dict[str, Any]:
    setup: dict[str, Any] = {
        "model": config.model if config.model.startswith("models/") else f"models/{config.model}",
        "generationConfig": {
            "responseModalities": ["AUDIO"],
            "maxOutputTokens": 4096,
            "speechConfig": {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": config.voice}}},
        },
        "systemInstruction": {"parts": [{"text": instructions}]},
        "inputAudioTranscription": {},
        "outputAudioTranscription": {},
        "realtimeInputConfig": {
            "automaticActivityDetection": {"disabled": True},
            "activityHandling": "START_OF_ACTIVITY_INTERRUPTS",
            "turnCoverage": "TURN_INCLUDES_ALL_INPUT",
        },
        "sessionResumption": {"handle": resume_handle} if resume_handle else {},
        "contextWindowCompression": {"slidingWindow": {}},
    }
    if not resume_handle:
        setup["historyConfig"] = {"initialHistoryInClientContent": True}
    if tools:
        setup["tools"] = [{"functionDeclarations": deepcopy(list(tools))}]
    return {"setup": setup}


class GeminiLiveConnection:
    """One native socket, with bounded I/O and no implicit reconnect or fallback.

    A session owns the logical ordering of writes. The transport serializes actual
    writes and permits exactly one reader at a time, including setup. Cancellation
    during a write is terminal: partially written media is never blindly retried.
    """

    def __init__(self, *, config: GeminiLiveConfig) -> None:
        self.config = config
        self._socket: Any = None
        self._write_lock = asyncio.Lock()
        self._read_lock = asyncio.Lock()
        self._started = False
        self._closed = False
        self._ready = False
        self._pending: deque[dict[str, Any]] = deque()
        self._close_task: asyncio.Task[None] | None = None

    @property
    def ready(self) -> bool:
        return self._ready and not self._closed

    async def connect(
        self, *, instructions: str, tools: Sequence[dict[str, Any]] = (),
        history: Sequence[dict[str, Any]] = (), resume_handle: str | None = None,
    ) -> None:
        if self._started or self._closed:
            raise GeminiTransportError("gemini_connection_not_reusable")
        if resume_handle and history:
            raise GeminiTransportError("gemini_resume_history_conflict")
        self._started = True
        try:
            async with asyncio.timeout(self.config.setup_timeout_seconds):
                socket = await websockets.connect(
                    LIVE_ENDPOINT,
                    additional_headers={"x-goog-api-key": self.config.api_key},
                    open_timeout=self.config.setup_timeout_seconds,
                    close_timeout=self.config.close_timeout_seconds,
                    max_size=self.config.max_provider_message_bytes,
                    max_queue=self.config.receive_queue_size,
                )
                if self._closed:
                    await socket.close()
                    raise GeminiTransportError("gemini_connection_closed")
                self._socket = socket
                await self._write(build_setup_event(
                    self.config, instructions=instructions, tools=tools,
                    resume_handle=resume_handle,
                ))
                while True:
                    event = await self._receive()
                    if "setupComplete" in event:
                        break
                    if len(self._pending) >= self.config.receive_queue_size:
                        raise GeminiTransportError("gemini_setup_event_overflow")
                    self._pending.append(event)
                if not resume_handle:
                    turns = deepcopy(list(history)) or [{
                        "role": "user",
                        "parts": [{"text": "A new Jarvis audio run has started."}],
                    }]
                    await self._write({"clientContent": {"turns": turns, "turnComplete": True}})
                self._ready = True
        except BaseException as error:
            await self.close()
            if isinstance(error, (asyncio.CancelledError, GeminiTransportError)):
                raise
            raise GeminiTransportError("gemini_setup_failed") from None

    async def send_event(self, event: dict[str, Any]) -> None:
        if not self.ready:
            raise GeminiTransportError("gemini_connection_not_ready")
        await self._write(event)

    async def _write(self, event: dict[str, Any]) -> None:
        try:
            async with asyncio.timeout(self.config.write_timeout_seconds):
                async with self._write_lock:
                    if self._closed or self._socket is None:
                        raise GeminiTransportError("gemini_connection_closed")
                    await self._socket.send(json.dumps(event, separators=(",", ":"), allow_nan=False))
        except BaseException as error:
            await self.close()
            if isinstance(error, (asyncio.CancelledError, GeminiTransportError)):
                raise
            raise GeminiTransportError("gemini_write_failed") from None

    async def receive_event(self) -> dict[str, Any]:
        if not self.ready:
            raise GeminiTransportError("gemini_connection_not_ready")
        async with self._read_lock:
            if self._pending:
                return self._pending.popleft()
            return await self._receive()

    async def _receive(self) -> dict[str, Any]:
        try:
            if self._closed or self._socket is None:
                raise GeminiTransportError("gemini_connection_closed")
            raw = await self._socket.recv()
            event = json.loads(raw)
            if not isinstance(event, dict):
                raise GeminiTransportError("gemini_invalid_provider_event")
            if "error" in event:
                raise GeminiTransportError("gemini_provider_error")
            return event
        except asyncio.CancelledError:
            raise
        except Exception as error:
            await self.close()
            if isinstance(error, GeminiTransportError):
                raise
            raise GeminiTransportError("gemini_receive_failed") from None

    async def close(self) -> None:
        self._closed = True
        self._ready = False
        if self._close_task is None:
            self._close_task = asyncio.create_task(self._close_socket())
        await asyncio.shield(self._close_task)

    async def _close_socket(self) -> None:
        socket, self._socket = self._socket, None
        self._pending.clear()
        if socket is not None:
            with suppress(Exception):
                async with asyncio.timeout(self.config.close_timeout_seconds):
                    await socket.close()
