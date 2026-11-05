"""Bounded, run-owned conversation persistence, independent of speech delivery."""
from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import logging
from typing import Any
from uuid import UUID

from app.realtime.gemini.client import GeminiTransportError


LOGGER = logging.getLogger(__name__)
MAX_PENDING_MESSAGES = 64
MAX_RETAINED_MESSAGES = 512


@dataclass(frozen=True, slots=True)
class FinalTranscript:
    logical_item_id: str
    role: str
    content: str
    occurred_at: datetime
    provider_item_id: str
    provider_response_id: str | None = None
    eligible_for_experience: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


class GeminiConversationJournal:
    def __init__(
        self, *, repository, experience_service, owner_id: str, session_id: UUID,
        audio_session_id: str, on_failure: Callable[[str], Awaitable[None]],
        attempt_timeout_seconds: float = 3.0, max_attempts: int = 3,
    ) -> None:
        self.repository, self.experience_service = repository, experience_service
        self.owner_id, self.session_id, self.audio_session_id = owner_id, session_id, audio_session_id
        self.on_failure = on_failure
        self.attempt_timeout_seconds, self.max_attempts = attempt_timeout_seconds, max_attempts
        self._queue: asyncio.Queue[FinalTranscript | None] = asyncio.Queue(MAX_PENDING_MESSAGES)
        self._worker: asyncio.Task | None = None
        self._accepting = True
        self._fingerprints: dict[tuple[str, str], str] = {}
        self._states: dict[tuple[str, str], str] = {}
        self._failures: dict[tuple[str, str], dict[str, Any]] = {}

    def submit(self, transcript: FinalTranscript) -> None:
        if not transcript.content.strip():
            return
        if not self._accepting:
            raise GeminiTransportError("gemini_message_journal_closed")
        key = (transcript.logical_item_id, transcript.role)
        fingerprint = hashlib.sha256(transcript.content.encode()).hexdigest()
        if previous := self._fingerprints.get(key):
            if previous != fingerprint:
                raise GeminiTransportError("gemini_conflicting_final_transcript")
            return
        if len(self._fingerprints) >= MAX_RETAINED_MESSAGES or self._queue.full():
            raise GeminiTransportError("gemini_message_backlog_limit")
        self._fingerprints[key] = fingerprint
        self._states[key] = "accepted"
        self._queue.put_nowait(transcript)
        if self._worker is None:
            self._worker = asyncio.create_task(self._consume())

    async def _consume(self) -> None:
        while True:
            transcript = await self._queue.get()
            try:
                if transcript is None:
                    return
                await self._persist(transcript)
            finally:
                self._queue.task_done()

    async def _persist(self, transcript: FinalTranscript) -> None:
        key = (transcript.logical_item_id, transcript.role)
        message = None
        for attempt in range(self.max_attempts):
            phase, rejection = "message", None
            try:
                async with asyncio.timeout(self.attempt_timeout_seconds):
                    if message is None:
                        message = await self.repository.create_or_reuse_gemini_message(
                            owner_id=self.owner_id, session_id=self.session_id,
                            audio_session_id=self.audio_session_id,
                            logical_item_id=transcript.logical_item_id,
                            role=transcript.role, content=transcript.content,
                            occurred_at=transcript.occurred_at,
                            provider_item_id=transcript.provider_item_id,
                            provider_response_id=transcript.provider_response_id,
                            metadata=transcript.metadata,
                        )
                    self._states[key] = "message_persisted"
                    if not transcript.eligible_for_experience:
                        self._states[key] = "diagnostic_only"
                        return
                    if self.experience_service is None:
                        raise GeminiTransportError("gemini_experience_service_unavailable")
                    phase = "evidence_readback"
                    attached = await self.repository.gemini_message_has_experience_evidence(
                        owner_id=self.owner_id, session_id=self.session_id, message_id=message.id,
                    )
                    if not attached:
                        # The same durable row is retried after an attachment failure.
                        # A read across event boundaries handles an acknowledgement
                        # lost just before the original experience event was closed.
                        phase = "evidence_attachment"
                        outcome = await self.experience_service.attach_message(
                            owner_id=self.owner_id, message=message,
                        )
                        if not outcome.attached:
                            # Only known application classifications, never arbitrary
                            # exception messages, transcript contents or provider prose.
                            reason = getattr(outcome, "reason", None)
                            rejection = reason if reason in {
                                "before_open_event", "finalized_boundary", "session_not_active",
                                "event_closed_during_attach", "event_changed_during_attach",
                            } else "unclassified_attachment_rejection"
                            raise GeminiTransportError("gemini_message_evidence_not_attached")
                    self._states[key] = "attached"
                    return
            except asyncio.CancelledError:
                self._states[key] = "incomplete"
                raise
            except Exception as error:
                self._failures[key] = {
                    "phase": phase, "attempt": attempt + 1,
                    "error_type": type(error).__name__, "attachment_reason": rejection,
                }
                # Never put speech, owner instructions, or database exceptions in logs.
                if attempt + 1 < self.max_attempts:
                    await asyncio.sleep(.05 * (attempt + 1))
        self._states[key] = "failed"
        LOGGER.warning("Gemini conversation persistence incomplete audio_session_id=%s item=%s role=%s",
                       self.audio_session_id, transcript.logical_item_id, transcript.role)
        await self.on_failure("gemini_memory_persistence_failed")

    async def drain(self, *, timeout_seconds: float) -> None:
        self._accepting = False
        if self._worker is None:
            return
        try:
            async with asyncio.timeout(timeout_seconds):
                await self._queue.put(None)
                await asyncio.shield(self._worker)
        except TimeoutError:
            self._worker.cancel()
            await asyncio.gather(self._worker, return_exceptions=True)
            for key, state in self._states.items():
                if state not in {"attached", "diagnostic_only", "failed"}:
                    self._states[key] = "incomplete"
            await self.on_failure("gemini_memory_drain_timeout")
        finally:
            # Drop queued transcript content once this worker can no longer consume it.
            while not self._queue.empty():
                self._queue.get_nowait()
                self._queue.task_done()

    def diagnostics(self) -> dict[str, Any]:
        counts: dict[str, int] = {}
        for state in self._states.values():
            counts[state] = counts.get(state, 0) + 1
        return {"states": counts, "queued": self._queue.qsize(),
                "worker_running": self._worker is not None and not self._worker.done(),
                "degraded": any(state in {"failed", "incomplete"} for state in self._states.values()),
                "failures": [{"role": key[1], "state": self._states[key], **failure}
                             for key, failure in self._failures.items()]}
