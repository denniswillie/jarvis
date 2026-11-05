from __future__ import annotations

import asyncio
import logging
from contextlib import suppress
from collections.abc import Awaitable, Callable
from datetime import datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID

from app.attention.admission_diagnostics import record_admission
if TYPE_CHECKING:
    from app.memory.prospective.delivery import ProspectiveDeliveryAttempt

from app.runtime.events import (
    DomainEvent,
    MemoryItemChangedPayload,
    OwnerWorkingMemoryUpdatedPayload,
    WorkingMemoryUpdatedPayload,
    VisualFrameAcceptedPayload,
)


LOGGER = logging.getLogger(__name__)


class RealtimeSessionManager:
    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._sessions_by_audio_id: dict[str, Any] = {}
        self._audio_ids_by_jarvis_session_id: dict[UUID, str] = {}
        self._audio_ids_by_owner_id: dict[str, list[str]] = {}
        self._jarvis_session_ids_by_audio_id: dict[str, UUID] = {}
        self._resources_closing = False
        self.attention_notifications_enabled = False
        self.attention_repository = None

    async def register(
        self,
        *,
        audio_session_id: str,
        owner_id: str,
        jarvis_session_id: UUID,
        session: Any,
    ) -> bool:
        async with self._lock:
            if self._resources_closing and callable(getattr(session, "prepare_for_resource_close", None)):
                return False
            if jarvis_session_id in self._audio_ids_by_jarvis_session_id:
                return False
            self._sessions_by_audio_id[audio_session_id] = session
            self._audio_ids_by_jarvis_session_id[jarvis_session_id] = audio_session_id
            owner_audio_ids = self._audio_ids_by_owner_id.setdefault(owner_id, [])
            if audio_session_id in owner_audio_ids:
                owner_audio_ids.remove(audio_session_id)
            owner_audio_ids.append(audio_session_id)
            self._jarvis_session_ids_by_audio_id[audio_session_id] = jarvis_session_id
            return True

    async def unregister(
        self,
        *,
        audio_session_id: str,
        owner_id: str,
        jarvis_session_id: UUID,
    ) -> None:
        async with self._lock:
            self._sessions_by_audio_id.pop(audio_session_id, None)
            if self._audio_ids_by_jarvis_session_id.get(jarvis_session_id) == audio_session_id:
                self._audio_ids_by_jarvis_session_id.pop(jarvis_session_id, None)
            owner_audio_ids = self._audio_ids_by_owner_id.get(owner_id)
            if owner_audio_ids is not None:
                with suppress(ValueError):
                    owner_audio_ids.remove(audio_session_id)
                if not owner_audio_ids:
                    self._audio_ids_by_owner_id.pop(owner_id, None)
            self._jarvis_session_ids_by_audio_id.pop(audio_session_id, None)

    async def prepare_for_resource_close(self) -> None:
        """Drain opt-in sessions before their storage closes; leave GPT alone."""
        async with self._lock:
            self._resources_closing = True
            preparing = [
                (audio_id, prepare)
                for audio_id, session in self._sessions_by_audio_id.items()
                if callable(prepare := getattr(session, "prepare_for_resource_close", None))
            ]
        failures = 0
        for audio_id, prepare in preparing:
            try:
                await prepare()
            except Exception:
                failures += 1
                LOGGER.error("realtime resource preparation failed audio_session_id=%s", audio_id)
        if failures:
            raise RuntimeError("Realtime resource preparation failed for one or more sessions.")

    async def close_all(self) -> None:
        async with self._lock:
            sessions = list(self._sessions_by_audio_id.values())
            self._sessions_by_audio_id.clear()
            self._audio_ids_by_jarvis_session_id.clear()
            self._audio_ids_by_owner_id.clear()
            self._jarvis_session_ids_by_audio_id.clear()

        for session in sessions:
            close = getattr(session, "close", None)
            if close is None:
                continue
            with suppress(Exception):
                result = close("service_shutdown")
                if asyncio.iscoroutine(result):
                    await result

    async def get_session_diagnostics(self, jarvis_session_id: UUID) -> dict[str, Any]:
        """Return a redacted snapshot of process-local realtime state."""

        async with self._lock:
            audio_session_id = self._audio_ids_by_jarvis_session_id.get(jarvis_session_id)
            connected = (
                audio_session_id is not None
                and audio_session_id in self._sessions_by_audio_id
            )
        return {
            "realtime_connected": connected,
            "audio_session_id": audio_session_id if connected else None,
        }

    async def send_companion_notice(self, owner_id, notice):
        """Select one active voice route; never create a session just to speak."""
        async with self._lock:
            sessions = [self._sessions_by_audio_id.get(identity)
                for identity in reversed(self._audio_ids_by_owner_id.get(owner_id, []))]
        for session in sessions:
            send = getattr(session, "speak_companion_notice", None)
            if callable(send) and getattr(session, "ready", False):
                return await send(notice)
        return {"status": "unavailable", "provider_received": False}

    async def handle_visual_frame_accepted(self, event: DomainEvent) -> None:
        payload = event.payload
        if not isinstance(payload, VisualFrameAcceptedPayload):
            return
        async with self._lock:
            audio_id = self._audio_ids_by_jarvis_session_id.get(payload.session_id)
            session = (
                self._sessions_by_audio_id.get(audio_id)
                if audio_id in self._audio_ids_by_owner_id.get(payload.owner_id, [])
                else None
            )
        # This synchronous wakeup must never await image transport or inference.
        notify = getattr(session, "notify_visual_frame_available", None)
        if notify is not None:
            notify()

    async def handle_working_memory_updated(self, event: DomainEvent) -> None:
        payload = event.payload
        if not isinstance(payload, WorkingMemoryUpdatedPayload):
            return
        async with self._lock:
            audio_session_id = self._audio_ids_by_jarvis_session_id.get(payload.session_id)
            owner_audio_ids = self._audio_ids_by_owner_id.get(payload.owner_id, [])
            session = (
                self._sessions_by_audio_id.get(audio_session_id)
                if audio_session_id in owner_audio_ids
                else None
            )
        if session is None:
            return
        notify = getattr(session, "notify_working_memory_updated", None)
        if notify is None:
            return
        try:
            result = notify(payload.revision)
            if asyncio.iscoroutine(result):
                await result
        except Exception:
            LOGGER.exception(
                "failed to notify realtime session of working-memory update session_id=%s",
                payload.session_id,
            )

    async def handle_memory_item_changed(self, event: DomainEvent) -> None:
        payload = event.payload
        if not isinstance(payload, MemoryItemChangedPayload):
            return
        async with self._lock:
            if payload.session_id is not None:
                audio_session_id = self._audio_ids_by_jarvis_session_id.get(
                    payload.session_id
                )
                owner_audio_ids = self._audio_ids_by_owner_id.get(payload.owner_id, [])
                audio_session_ids = (
                    [audio_session_id]
                    if audio_session_id is not None and audio_session_id in owner_audio_ids
                    else []
                )
            else:
                audio_session_ids = list(
                    self._audio_ids_by_owner_id.get(payload.owner_id, [])
                )
            sessions = [
                session
                for audio_session_id in audio_session_ids
                if (session := self._sessions_by_audio_id.get(audio_session_id))
                is not None
            ]
        for session in sessions:
            notify = getattr(session, "notify_memory_item_changed", None)
            if notify is None:
                continue
            try:
                result = notify(event)
                if asyncio.iscoroutine(result):
                    await result
            except Exception:
                LOGGER.exception(
                    "failed to notify realtime session of memory change memory_id=%s",
                    payload.memory_item_id,
                )

    async def handle_owner_working_memory_updated(self, event: DomainEvent) -> None:
        payload = event.payload
        if not isinstance(payload, OwnerWorkingMemoryUpdatedPayload):
            return
        async with self._lock:
            sessions = [
                session
                for audio_session_id in self._audio_ids_by_owner_id.get(
                    payload.owner_id,
                    [],
                )
                if (session := self._sessions_by_audio_id.get(audio_session_id))
                is not None
            ]
        for session in sessions:
            notify = getattr(session, "notify_owner_working_memory_updated", None)
            if notify is None:
                continue
            try:
                result = notify(payload.revision)
                if asyncio.iscoroutine(result):
                    await result
            except Exception:
                LOGGER.exception(
                    "failed to notify realtime session of owner working-memory update owner_id=%s",
                    payload.owner_id,
                )

    async def handle_behavior_profile_committed(self, event):
        """Owner-scoped commit fan-out; transport identity belongs to the target."""
        async with self._lock:
            if self._resources_closing:
                return
            sessions = [self._sessions_by_audio_id[audio_id]
                        for audio_id in self._audio_ids_by_owner_id.get(event.owner_id, ())
                        if audio_id in self._sessions_by_audio_id]
        for session in sessions:
            try:
                closed = getattr(session, "_closed", None)
                if closed is not None and closed.is_set():
                    continue
                payload = event.as_payload()
                kind = payload.pop("type")
                payload.update(session_id=str(session.context.jarvis_session_id),
                               audio_session_id=session.context.audio_session_id)
                offer = getattr(session, "_offer_app_event", None)
                if offer is not None:
                    offer(kind, **payload)
                else:
                    await session._send_app_event(kind, **payload)
                notify = getattr(session, "notify_behavior_profile_committed", None)
                if notify is not None:
                    result = notify(event)
                    if asyncio.iscoroutine(result):
                        await result
            except asyncio.CancelledError:
                raise
            except Exception as error:
                LOGGER.warning("profile session delivery unavailable error_type=%s", type(error).__name__)

    async def speak_visual_interrupt(
        self,
        *,
        owner_id: str,
        audio_session_id: str | None,
        jarvis_session_id: UUID,
        source_received_at: datetime | None,
        message: str,
        interrupt_id: UUID,
        visual_frame_id: UUID | None = None,
        source_captured_at: datetime | None = None,
        reason: str | None = None,
        confidence: float | None = None,
        source_payload_id: UUID | None = None,
    ) -> bool:
        async with self._lock:
            owner_audio_ids = self._audio_ids_by_owner_id.get(owner_id, [])
            candidates = (
                audio_session_id,
                self._audio_ids_by_jarvis_session_id.get(jarvis_session_id),
            )
            session = next(
                (
                    self._sessions_by_audio_id[candidate]
                    for candidate in candidates
                    if candidate is not None
                    and candidate in owner_audio_ids
                    and self._jarvis_session_ids_by_audio_id.get(candidate) == jarvis_session_id
                    and candidate in self._sessions_by_audio_id
                ),
                None,
            )
        if session is None:
            return False
        speak = getattr(session, "speak_visual_interrupt", None)
        if speak is None:
            return False
        accepted = speak(
            owner_id=owner_id,
            jarvis_session_id=jarvis_session_id,
            source_received_at=source_received_at,
            message=message,
            interrupt_id=interrupt_id,
            visual_frame_id=visual_frame_id,
            source_captured_at=source_captured_at,
            reason=reason,
            confidence=confidence,
            **({"source_payload_id": source_payload_id} if source_payload_id is not None else {}),
        )
        if asyncio.iscoroutine(accepted):
            accepted = await accepted
        return accepted is True

    async def speak_reminder(
        self,
        *,
        owner_id: str,
        preferred_jarvis_session_id: UUID | None,
        message: str,
        reminder_id: UUID,
        reminder_event_id: UUID,
    ) -> UUID | None:
        async with self._lock:
            audio_session_id = None
            if preferred_jarvis_session_id is not None:
                audio_session_id = self._audio_ids_by_jarvis_session_id.get(
                    preferred_jarvis_session_id
                )
            if audio_session_id not in self._audio_ids_by_owner_id.get(owner_id, []):
                audio_session_id = None
            if audio_session_id is None:
                owner_audio_ids = self._audio_ids_by_owner_id.get(owner_id, [])
                audio_session_id = owner_audio_ids[-1] if owner_audio_ids else None
            session = (
                self._sessions_by_audio_id.get(audio_session_id)
                if audio_session_id is not None
                else None
            )
            jarvis_session_id = (
                self._jarvis_session_ids_by_audio_id.get(audio_session_id)
                if audio_session_id is not None
                else None
            )

        if session is None or jarvis_session_id is None:
            return None
        speak = getattr(session, "speak_reminder", None)
        if speak is None:
            return None
        result = speak(
            message=message,
            reminder_id=reminder_id,
            reminder_event_id=reminder_event_id,
        )
        if asyncio.iscoroutine(result):
            await result
        return jarvis_session_id

    async def has_connected_voice(self, owner_id):
        """Transport presence only, never evidence of location or human hearing."""
        async with self._lock:
            return not self._resources_closing and any(
                callable(getattr(self._sessions_by_audio_id.get(identity), "try_attention", None))
                for identity in self._audio_ids_by_owner_id.get(owner_id, ()))

    async def deliver_attention(self, *, owner_id, repository, speech, worker_id, receipt_seconds):
        async with self._lock:
            if self._resources_closing:
                record_admission("route_not_ready", owner=owner_id)
                return None
            owned = self._audio_ids_by_owner_id.get(owner_id, [])
            session = self._sessions_by_audio_id.get(owned[-1]) if owned else None
        deliver = getattr(session, "try_attention", None)
        if deliver is None:
            record_admission("no_session", owner=owner_id)
            return None
        if (speech is None and getattr(session, "_voice_route_runtime", None) is None
                and getattr(session, "_lean_attention", None) is None):
            record_admission("route_not_ready", owner=owner_id)
            return None
        return await deliver(repository, speech, worker_id=worker_id, receipt_seconds=receipt_seconds)

    async def cancel_visual_attention(self, *, owner_id, focus_id=None, except_focus_id=None, reason):
        async with self._lock:
            deliveries = []
            for audio_id in self._audio_ids_by_owner_id.get(owner_id, ()):
                session = self._sessions_by_audio_id.get(audio_id)
                delivery = getattr(session, "_attention_delivery", None)
                if (delivery is not None and delivery.claim.subject_type == "visual_focus"
                        and (focus_id is None or delivery.claim.subject_id == focus_id)
                        and delivery.claim.subject_id != except_focus_id):
                    deliveries.append(delivery)
        for delivery in deliveries:
            await delivery.stop(reason)

    async def speak_prospective_interrupt(
        self,
        *,
        owner_id: str,
        preferred_jarvis_session_id: UUID | None,
        message: str,
        event_id: UUID,
        subject_type: str,
        subject_id: UUID,
        on_outcome: Callable[[str, str | None], Awaitable[None]],
        delivery_attempt: ProspectiveDeliveryAttempt | None = None,
        source_payload_id: UUID | None = None,
    ) -> UUID | None:
        async with self._lock:
            audio_session_id = None
            if preferred_jarvis_session_id is not None:
                audio_session_id = self._audio_ids_by_jarvis_session_id.get(
                    preferred_jarvis_session_id
                )
            owner_audio_ids = self._audio_ids_by_owner_id.get(owner_id, [])
            if audio_session_id not in owner_audio_ids:
                audio_session_id = None
            if audio_session_id is None:
                owner_audio_ids = self._audio_ids_by_owner_id.get(owner_id, [])
                audio_session_id = owner_audio_ids[-1] if owner_audio_ids else None
            session = (
                self._sessions_by_audio_id.get(audio_session_id)
                if audio_session_id is not None
                else None
            )
            jarvis_session_id = (
                self._jarvis_session_ids_by_audio_id.get(audio_session_id)
                if audio_session_id is not None
                else None
            )
        if session is None or jarvis_session_id is None:
            if delivery_attempt is not None:
                delivery_attempt.reason = "no_connected_session"
            return None
        speak = getattr(session, "speak_prospective_interrupt", None)
        if speak is None:
            if delivery_attempt is not None:
                delivery_attempt.reason = "session_cannot_deliver"
            return None
        if delivery_attempt is not None:
            delivery_attempt.session_id = jarvis_session_id
        accepted = speak(
            message=message,
            event_id=event_id,
            subject_type=subject_type,
            subject_id=subject_id,
            on_outcome=on_outcome,
            **({"source_payload_id": source_payload_id} if source_payload_id is not None else {}),
            **({"delivery_attempt": delivery_attempt} if delivery_attempt is not None else {}),
        )
        if asyncio.iscoroutine(accepted):
            accepted = await accepted
        return jarvis_session_id if accepted else None
