"""Dialogue document construction and batch submission, plus visual deltas.

Live uses intake.LiveDialogueIntake and canonical caption revisions, NOT the
finalized-turn convenience adapter. Live has no provider-final turn marker.
Batching groups HTTP work, not document identities. No transcript database is
introduced here; replay always comes from the existing canonical source archive.
"""
from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass
from datetime import datetime
import json
import time

from .client import MemoryDocument, NativeMemory, source_document_id, timestamp


@dataclass(frozen=True)
class CanonicalCaption:
    sequence: int
    session_id: str
    source_namespace: str
    source_event_id: str
    revision: int
    kind: str
    source_time: datetime | None
    recorded_at: datetime
    availability: str
    content: dict | None
    session_closed_at: datetime | None = None

    @property
    def document_id(self) -> str:
        identity = json.dumps([self.source_namespace, self.source_event_id], separators=(",", ":"))
        return source_document_id("live-caption", self.session_id, identity)

    @property
    def usable(self) -> bool:
        payload = self.content or {}
        return (
            self.availability in {"available", "not_finalized"}
            and payload.get("voice_route") == "live"
            and self.kind in {"user_transcript", "assistant_output"}
            and payload.get("speaker") == ("user" if self.kind == "user_transcript" else "assistant")
            and isinstance(payload.get("text"), str) and bool(payload["text"].strip())
            and bool(payload.get("caption_id"))
        )

    def document(self) -> MemoryDocument:
        """A whole settled caption, not a received fragment or final-turn claim."""
        if not self.usable:
            raise ValueError("Not an eligible canonical Live caption")
        payload = self.content
        observed = self.source_time or self.recorded_at
        offsets = [{key: part.get(key) for key in ("provider_event_id", "start_ms", "end_ms")}
                   for part in payload.get("fragments", [])]
        label = "User" if payload["speaker"] == "user" else "GPT-Live"
        return MemoryDocument(
            self.document_id, f"[{timestamp(observed)}] {label}: {payload['text']}", observed,
            context=("Application-settled Live caption snapshot, NOT a provider-final utterance. "
                     "Times are approximate application ingress and original provider offsets; "
                     "assistant playback/hearing is unconfirmed."),
            metadata={
                "stream": "dialogue", "session_id": self.session_id,
                "source_namespace": self.source_namespace, "source_event_id": self.source_event_id,
                "canonical_sequence": str(self.sequence), "canonical_revision": str(self.revision),
                "caption_id": str(payload["caption_id"]),
                "caption_revision": str(payload.get("caption_revision", self.revision)),
                "speaker": payload["speaker"], "speech_finality": "provisional",
                "settlement": "application_settled_not_provider_final",
                "source_time": timestamp(observed), "latest_recorded_at": timestamp(self.recorded_at),
                "source_time_basis": ("application_ingress_not_physical_capture" if self.source_time
                                      else "canonical_recorded_at_fallback"),
                "provider_timing": "approximate_not_a_turn_boundary",
                "fragment_offsets": json.dumps(offsets, separators=(",", ":")),
            },
        )


@dataclass(frozen=True)
class DialogueTurn:
    source_id: str
    speaker: str
    text: str
    started_at: datetime
    ended_at: datetime
    finalized: bool = True

    def document(self, session_id: str) -> MemoryDocument:
        if self.speaker not in {"user", "assistant"}:
            raise ValueError("Only finalized User/Voice dialogue is auto-retained")
        start, end = timestamp(self.started_at), timestamp(self.ended_at)
        if self.ended_at < self.started_at or not self.text.strip():
            raise ValueError("Valid original times and nonempty dialogue required")
        label = "User" if self.speaker == "user" else "GPT-Live"
        return MemoryDocument(
            source_document_id("dialogue", session_id, self.source_id),
            f"[{start} -- {end}] {label}: {self.text}", self.started_at,
            context="Finalized active Jarvis session dialogue; speaker attribution is evidence, not endorsement.",
            metadata={"stream": "dialogue", "session_id": session_id, "source_id": self.source_id,
                      "speaker": self.speaker, "started_at": start, "ended_at": end},
        )


class DialogueBatcher:
    @staticmethod
    async def publish(memory, documents, *, operation_id=None):
        """Shared native submission seam; Live persists the operation ID first.

        The stable native operation ID also deduplicates a lost acknowledgement,
        unlike document IDs alone, which deduplicate stored content only.
        """
        if operation_id is None:
            return await memory.retain(documents, asynchronous=True)
        return await memory._request("POST", "/memories", body={
            "items": [document.payload() for document in documents],
            "async": True, "operation_id": operation_id,
        })

    def __init__(
        self, memory: NativeMemory, session_id: str, *,
        checkpoint: Callable[[tuple[str, ...]], Awaitable[None]],
        accepted_source_ids: Iterable[str] = (), settle_seconds: float = 18,
        clock: Callable[[], float] = time.monotonic,
    ):
        if not session_id or not 15 <= settle_seconds <= 20:
            raise ValueError("Session ID and a 15-20 second settle interval required")
        self.memory, self.session_id = memory, session_id
        self.checkpoint, self.clock = checkpoint, clock
        self.settle_seconds = settle_seconds
        self._accepted = set(accepted_source_ids)
        self._pending: dict[str, MemoryDocument] = {}
        self._last_activity = clock()
        self._flush_lock = asyncio.Lock()
        self._closed = False

    def enqueue(self, turn: DialogueTurn, *, active=True) -> bool:
        if self._closed:
            raise RuntimeError("Dialogue input is closed")
        if not active or not turn.finalized or turn.source_id in self._accepted:
            return False
        document = turn.document(self.session_id)
        prior = self._pending.get(turn.source_id)
        if prior is not None:
            if prior != document:
                raise ValueError("A finalized source ID cannot change; use explicit document correction")
            return False
        self._pending[turn.source_id] = document
        self.note_activity()
        return True

    def note_activity(self) -> None:
        """Optional speech-activity hook so a long unfinished turn delays settling."""
        self._last_activity = self.clock()

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    async def flush_due(self, *, force=False) -> dict | None:
        async with self._flush_lock:
            if not self._pending or (not force and self.clock() - self._last_activity < self.settle_seconds):
                return None
            batch = sorted(self._pending.items(), key=lambda item: (item[1].observed_at, item[0]))
            ids = tuple(source_id for source_id, _ in batch)
            receipt = await self.publish(self.memory, [document for _, document in batch])
            # If this fails, retain pending sources and retry with identical IDs.
            # Native document replacement prevents a second stored source copy.
            await self.checkpoint(ids)
            self._accepted.update(ids)
            for source_id in ids:
                self._pending.pop(source_id)
            return receipt

    async def close(self) -> dict | None:
        self._closed = True
        return await self.flush_due(force=True)


@dataclass(frozen=True)
class ObservationDelta:
    source_id: str
    session_id: str
    text: str
    observed_from: datetime
    observed_until: datetime
    analyzed_at: datetime
    frame_ids: tuple[str, ...] = ()
    file_ids: tuple[str, ...] = ()

    def document(self) -> MemoryDocument:
        start, end, analyzed = timestamp(self.observed_from), timestamp(self.observed_until), timestamp(self.analyzed_at)
        if self.observed_until < self.observed_from or not self.text.strip():
            raise ValueError("Nonempty meaningful observation and valid observation window required")
        return MemoryDocument(
            source_document_id("observation", self.session_id, self.source_id), self.text, self.observed_from,
            context="Memory-worthy situational observation from the analyzer, not confirmed user speech. Preserve uncertainty.",
            metadata={"stream": "observation", "source_id": self.source_id, "session_id": self.session_id,
                      "observed_from": start, "observed_until": end, "analyzed_at": analyzed,
                      "frame_ids": json.dumps(self.frame_ids), "file_ids": json.dumps(self.file_ids)},
        )


async def retain_observation(memory: NativeMemory, delta: ObservationDelta | None) -> dict | None:
    """Pass only EventAnalyzer.memory_worthy_observation; null means no ingestion.

    Do not pass static frames, the human-state snapshot, or a duplicate dialogue
    summary. Replay uses the original analyzer-result ID, never a new random ID.
    """
    if delta is None:
        return None
    return await memory.retain([delta.document()], asynchronous=True)
