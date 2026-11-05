from __future__ import annotations

import base64
import binascii
from dataclasses import dataclass, field
import hashlib
from typing import Any

from app.realtime.gemini.client import GeminiTransportError


def decode_pcm16(data: object, *, max_bytes: int) -> bytes:
    if not isinstance(data, str) or len(data) > 4 * ((max_bytes + 2) // 3):
        raise ValueError("PCM input is missing or exceeds its byte limit")
    try:
        pcm = base64.b64decode(data, validate=True)
    except (ValueError, binascii.Error):
        raise ValueError("PCM must be valid base64") from None
    if not pcm or len(pcm) % 2 or len(pcm) > max_bytes:
        raise ValueError("PCM16 must contain bounded, complete samples")
    return pcm


@dataclass(slots=True)
class GeminiOutputTurn:
    """Translate one already-correlated native output, never infer turn ownership.

    A session must route late native terminal events to their original instance.
    Generation completion is not logical completion or device playback completion.
    PCM is hashed/counted, not retained in production memory.
    """

    audio_session_id: str
    connection_generation: int
    turn_number: int
    transcript: str = ""
    # Text observed while audio was being forwarded, not word-aligned heard text.
    # Stop extending it when output is interrupted or no longer forwarded.
    context_transcript: str = ""
    interrupted: bool = False
    completed: bool = False
    audio_done: bool = False
    transcript_done: bool = False
    provider_audio_bytes: int = 0
    forwarded_audio_bytes: int = 0
    audio_chunks: int = 0
    forwarded_chunks: int = 0
    _provider_hash: Any = field(default_factory=hashlib.sha256, repr=False)
    _forwarded_hash: Any = field(default_factory=hashlib.sha256, repr=False)

    @property
    def response_id(self) -> str:
        return f"{self.audio_session_id}:g{self.connection_generation}:r{self.turn_number}"

    @property
    def item_id(self) -> str:
        return f"{self.response_id}:audio"

    def _event(self, kind: str, **values: Any) -> dict[str, Any]:
        return {
            "type": kind, "response_id": self.response_id,
            "item_id": self.item_id, **values,
        }

    def interrupt(self, *, source: str) -> list[dict[str, Any]]:
        if self.interrupted or self.completed:
            return []
        self.interrupted = True
        return [self._event("audio.output.clear", provider="gemini", source=source)]

    def consume(self, event: dict[str, Any]) -> list[dict[str, Any]]:
        if self.completed:
            return []
        content = event.get("serverContent")
        if not isinstance(content, dict):
            return []
        result: list[dict[str, Any]] = []
        if content.get("interrupted"):
            result.extend(self.interrupt(source="provider"))
        transcription = content.get("outputTranscription", {})
        text = transcription.get("text", "") if isinstance(transcription, dict) else ""
        if isinstance(text, str) and text and not self.transcript_done:
            self.transcript += text
            if not self.interrupted:
                result.append(self._event("transcript.delta", role="assistant", text=text))
        model_turn = content.get("modelTurn", {})
        parts = model_turn.get("parts", []) if isinstance(model_turn, dict) else []
        for part in parts:
            inline = part.get("inlineData") if isinstance(part, dict) else None
            if not isinstance(inline, dict):
                continue
            mime = inline.get("mimeType", "")
            if not isinstance(mime, str) or not mime.startswith("audio/"):
                continue
            if mime.replace(" ", "") not in {"audio/pcm", "audio/pcm;rate=24000"}:
                raise GeminiTransportError("gemini_unsupported_output_audio")
            try:
                pcm = decode_pcm16(inline.get("data"), max_bytes=8 * 1024 * 1024)
            except ValueError:
                raise GeminiTransportError("gemini_invalid_output_audio") from None
            self.provider_audio_bytes += len(pcm)
            self._provider_hash.update(pcm)
            self.audio_chunks += 1
            if not self.interrupted and not self.audio_done:
                result.append(self._event(
                    "audio.output.delta", audio_base64=inline["data"],
                    sample_rate=24000, audio_format="pcm16",
                    audio_chunk_index=self.audio_chunks,
                ))
        if content.get("generationComplete") or content.get("turnComplete"):
            if not self.audio_done:
                self.audio_done = True
                result.append(self._event("audio.output.done", interrupted=self.interrupted))
            if not self.transcript_done and self.transcript:
                self.transcript_done = True
                result.append(self._event(
                    "transcript.done", role="assistant", text=self.transcript,
                    interrupted=self.interrupted, delivery_status="generated",
                ))
        if content.get("turnComplete"):
            self.completed = True
            result.append(self._event(
                "response.done", status="cancelled" if self.interrupted else "completed",
                interrupted=self.interrupted,
            ))
        return result

    def cancel_locally(self, *, reason: str) -> list[dict[str, Any]]:
        """Settle a retired connection without fabricating a native terminal event."""
        if self.completed:
            return []
        result = self.interrupt(source="application")
        if not self.audio_done:
            self.audio_done = True
            result.append(self._event("audio.output.done", interrupted=True))
        if self.transcript and not self.transcript_done:
            self.transcript_done = True
            result.append(self._event("transcript.done", role="assistant", text=self.transcript,
                                      interrupted=True, delivery_status="generated"))
        self.completed = True
        result.append(self._event("response.done", status="cancelled", interrupted=True,
                                  completion_source="application", reason=reason))
        return result

    def record_forwarded(self, event: dict[str, Any]) -> None:
        """Call only after the client send succeeds; playback remains unconfirmed."""
        if event.get("type") == "transcript.delta":
            if self.forwarded_audio_bytes and not self.interrupted:
                self.context_transcript = self.transcript
            return
        if event.get("type") != "audio.output.delta":
            return
        if (event.get("item_id") != self.item_id
                or event.get("audio_chunk_index") != self.forwarded_chunks + 1):
            raise ValueError("Forwarded audio must retain its original identity and order")
        pcm = decode_pcm16(event.get("audio_base64"), max_bytes=8 * 1024 * 1024)
        self.forwarded_audio_bytes += len(pcm)
        self.forwarded_chunks += 1
        self._forwarded_hash.update(pcm)
        self.context_transcript = self.transcript

    def diagnostics(self) -> dict[str, Any]:
        return {
            "response_id": self.response_id, "item_id": self.item_id,
            "interrupted": self.interrupted, "completed": self.completed,
            "provider_audio_bytes": self.provider_audio_bytes,
            "forwarded_audio_bytes": self.forwarded_audio_bytes,
            "provider_audio_sha256": self._provider_hash.hexdigest(),
            "forwarded_audio_sha256": self._forwarded_hash.hexdigest(),
            "audio_chunks": self.audio_chunks, "forwarded_chunks": self.forwarded_chunks,
        }
