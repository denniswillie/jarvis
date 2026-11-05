"""Native Live transport and bounded tool bridge for isolated acceptance probes."""
from __future__ import annotations

import asyncio
import base64
import json
import time
from dataclasses import dataclass, field
from typing import Any

import websockets

from app.tools.executor import JarvisToolExecutor
from app.tools.registry import JarvisToolRegistry


ENDPOINT = (
    "wss://generativelanguage.googleapis.com/ws/"
    "google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"
)


def declarations(registry: JarvisToolRegistry) -> list[dict[str, Any]]:
    return [{
        "name": tool.name,
        "description": tool.description,
        "parametersJsonSchema": tool.input_schema,
    } for tool in registry.list()]


class ToolBridge:
    """Use Jarvis's executor, keeping execution separate from speech cancellation.

    This cache is only connection-local protection for the evaluation. Production
    must retain its existing durable execution repository across reconnects.
    """

    def __init__(self, registry: JarvisToolRegistry, *, response_delay: float = 0,
                 response_images: dict[str, bytes] | None = None):
        self.registry = registry
        self.executor = JarvisToolExecutor(registry)
        self.response_delay = response_delay
        self.response_images = response_images or {}
        self.tasks: dict[str, asyncio.Task] = {}
        self.arguments: dict[str, str] = {}
        self.cancelled: set[str] = set()
        self.trace: list[dict[str, Any]] = []
        self.called = asyncio.Event()

    async def dispatch(self, call: dict, send) -> None:
        call_id, name = call.get("id"), call.get("name")
        args = call.get("args", {})
        if not isinstance(call_id, str) or not isinstance(name, str) or not isinstance(args, dict):
            raise ValueError("Malformed tool call")
        signature = json.dumps([name, args], sort_keys=True)
        if call_id in self.arguments:
            if self.arguments[call_id] != signature:
                raise ValueError("Provider reused a call ID with different arguments")
            return
        if len(self.tasks) >= 8:
            raise RuntimeError("Evaluation tool-call limit reached")
        self.arguments[call_id] = signature

        async def execute():
            if call_id in self.cancelled:
                return
            result = await self.executor.execute(name, args, provider_call_id=call_id)
            self.trace.append({"id": call_id, "name": name, "arguments": args, "result": result})
            self.called.set()
            if self.response_delay:
                await asyncio.sleep(self.response_delay)
            if call_id not in self.cancelled:
                response = {"id": call_id, "name": name, "response": result}
                if name in self.response_images:
                    response["parts"] = [{"inlineData": {
                        "mimeType": "image/jpeg",
                        "data": base64.b64encode(self.response_images[name]).decode(),
                    }}]
                await send({"toolResponse": {"functionResponses": [response]}})

        self.tasks[call_id] = asyncio.create_task(execute())

    def cancel(self, ids: list[str]) -> None:
        self.cancelled.update(ids)

    def raise_if_failed(self) -> None:
        for task in self.tasks.values():
            if task.done() and not task.cancelled() and task.exception() is not None:
                raise task.exception()

    async def close(self) -> None:
        # Never cancel an executing mutation merely because speech was stopped.
        # The outer scenario timeout bounds this isolated test's handlers.
        if self.tasks:
            await asyncio.gather(*self.tasks.values(), return_exceptions=True)


@dataclass
class Turn:
    started: float = field(default_factory=time.monotonic)
    input_end: float | None = None
    first_audio: float | None = None
    transcript: str = ""
    input_transcript: str = ""
    audio: bytearray = field(default_factory=bytearray)
    audio_types: set[str] = field(default_factory=set)
    usage: list[dict] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)
    completed: bool = False
    interrupted: bool = False
    interruption_at: float | None = None
    first_audio_event: asyncio.Event = field(default_factory=asyncio.Event)

    def add(self, event: dict) -> None:
        now = time.monotonic()
        content = event.get("serverContent", {})
        parts = content.get("modelTurn", {}).get("parts", [])
        self.events.append({
            "seconds": round(now - self.started, 4),
            "fields": list(event), "content_fields": list(content),
            "part_fields": [list(part) for part in parts],
        })
        self.transcript += content.get("outputTranscription", {}).get("text", "")
        self.input_transcript += content.get("inputTranscription", {}).get("text", "")
        for part in parts:
            inline = part.get("inlineData", {})
            mime = inline.get("mimeType", "")
            if mime.startswith("audio/") and inline.get("data"):
                if self.first_audio is None:
                    self.first_audio = now
                    self.first_audio_event.set()
                self.audio.extend(base64.b64decode(inline["data"], validate=True))
                self.audio_types.add(mime)
        if event.get("usageMetadata"):
            self.usage.append(event["usageMetadata"])
        if content.get("interrupted"):
            self.interrupted = True
            self.interruption_at = now
        if content.get("turnComplete"):
            self.completed = True

    def summary(self) -> dict:
        modalities: dict[str, int] = {}
        for usage in self.usage:
            for item in usage.get("promptTokensDetails", []):
                name = item.get("modality", "UNKNOWN")
                modalities[name] = max(modalities.get(name, 0), item.get("tokenCount", 0))
        return {
            "complete": self.completed, "interrupted": self.interrupted,
            "transcript": self.transcript, "input_transcript": self.input_transcript,
            "audio_bytes": len(self.audio), "audio_types": sorted(self.audio_types),
            "audio_seconds": len(self.audio) / 48000,
            "first_audio_after_input_end_seconds": (
                round(self.first_audio - self.input_end, 3)
                if self.first_audio is not None and self.input_end is not None else None
            ),
            "input_modalities": modalities,
            "usage": self.usage, "events": self.events,
        }


class LiveProbe:
    def __init__(self, *, key: str, model: str = "gemini-3.1-flash-live-preview",
                 instructions: str = "", manual_vad: bool = True,
                 bridge: ToolBridge | None = None, thinking: str | None = None,
                 media_resolution: str | None = None):
        self.key, self.model = key, model
        self.instructions, self.manual_vad, self.bridge = instructions, manual_vad, bridge
        self.thinking, self.media_resolution = thinking, media_resolution
        self.socket = None
        self.handle: str | None = None
        self.send_lock = asyncio.Lock()
        self.receive_lock = asyncio.Lock()

    async def connect(self, *, image: bytes | None = None, resume: bool = False):
        handle = self.handle if resume else None
        if resume and not handle:
            raise RuntimeError("No resumable handle was received")
        self.socket = await websockets.connect(
            ENDPOINT, additional_headers={"x-goog-api-key": self.key},
            open_timeout=15, close_timeout=3, max_size=8 * 1024 * 1024,
        )
        setup: dict[str, Any] = {
            "model": "models/" + self.model,
            "generationConfig": {
                "responseModalities": ["AUDIO"], "maxOutputTokens": 4096,
                "speechConfig": {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": "Puck"}}},
            },
            "inputAudioTranscription": {}, "outputAudioTranscription": {},
            "realtimeInputConfig": {
                "automaticActivityDetection": {"disabled": self.manual_vad},
                "turnCoverage": "TURN_INCLUDES_ALL_INPUT",
            },
            "sessionResumption": {"handle": handle} if handle else {},
            "contextWindowCompression": {"slidingWindow": {}},
        }
        if not handle:
            setup["historyConfig"] = {"initialHistoryInClientContent": True}
        if self.thinking:
            setup["generationConfig"]["thinkingConfig"] = {"thinkingLevel": self.thinking}
        if self.media_resolution:
            setup["generationConfig"]["mediaResolution"] = self.media_resolution
        if self.instructions:
            setup["systemInstruction"] = {"parts": [{"text": self.instructions}]}
        if self.bridge:
            setup["tools"] = [{"functionDeclarations": declarations(self.bridge.registry)}]
        await self.send({"setup": setup})
        async with asyncio.timeout(20):
            while "setupComplete" not in await self.receive():
                pass
        if not handle:
            parts = [{"inlineData": {
                "mimeType": "image/jpeg", "data": base64.b64encode(image).decode(),
            }}] if image else [{"text": "A new isolated acceptance-test conversation."}]
            await self.send({"clientContent": {
                "turns": [{"role": "user", "parts": parts}], "turnComplete": True,
            }})

    async def close(self):
        if self.socket:
            await self.socket.close()
            self.socket = None

    async def send(self, event: dict):
        async with self.send_lock:
            if self.socket is None:
                raise RuntimeError("Live session is closed")
            await self.socket.send(json.dumps(event))

    async def receive(self) -> dict:
        async with self.receive_lock:
            event = json.loads(await self.socket.recv())
        if "error" in event:
            # Provider errors can echo inputs; never log the raw error or key.
            raise RuntimeError("Gemini returned an error event")
        update = event.get("sessionResumptionUpdate")
        if update is not None:
            self.handle = update.get("newHandle") if update.get("resumable") else None
        return event

    async def text(self, text: str, turn: Turn):
        await self.send({"realtimeInput": {"text": text}})
        turn.input_end = time.monotonic()

    async def speech(self, pcm: bytes, turn: Turn, *, image: bytes | None = None,
                     chunk_ms: int = 100, flush: bool = True):
        if self.manual_vad:
            await self.send({"realtimeInput": {"activityStart": {}}})
        if image:
            await self.send({"realtimeInput": {"video": {
                "mimeType": "image/jpeg", "data": base64.b64encode(image).decode(),
            }}})
        size = 16000 * 2 * chunk_ms // 1000
        started = time.monotonic()
        for offset in range(0, len(pcm), size):
            chunk = pcm[offset:offset + size]
            await self.send({"realtimeInput": {"audio": {
                "mimeType": "audio/pcm;rate=16000", "data": base64.b64encode(chunk).decode(),
            }}})
            await asyncio.sleep(max(0, started + (offset + len(chunk)) / 32000 - time.monotonic()))
        turn.input_end = time.monotonic()
        if self.manual_vad:
            await self.send({"realtimeInput": {"activityEnd": {}}})
        elif flush:
            await self.send({"realtimeInput": {"audioStreamEnd": True}})
        else:
            # Continue silence as a microphone would, to exercise server end-of-turn VAD.
            for _ in range(15):
                await self.send({"realtimeInput": {"audio": {
                    "mimeType": "audio/pcm;rate=16000",
                    "data": base64.b64encode(bytes(3200)).decode(),
                }}})
                await asyncio.sleep(.1)

    async def collect(self, turn: Turn, *, timeout: float = 65):
        async with asyncio.timeout(timeout):
            while not turn.completed:
                if self.bridge:
                    self.bridge.raise_if_failed()
                event = await self.receive()
                turn.add(event)
                if self.bridge:
                    self.bridge.cancel(event.get("toolCallCancellation", {}).get("ids", []))
                    for call in event.get("toolCall", {}).get("functionCalls", []):
                        await self.bridge.dispatch(call, self.send)
        return turn
