from __future__ import annotations

import asyncio
import hashlib
import logging
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from uuid import UUID
from weakref import WeakValueDictionary

from app.attention.notifications import selected_notification_repository
from app.memory.prospective.models import (
    ProspectiveDeliveryOutcome,
    VisualFocus,
    VisualFocusMode,
)
from app.memory.prospective.delivery import ProspectiveDeliveryAttempt
from app.memory.prospective.repository import ProspectiveRepository
from app.perception.contracts import FastPerceptionRequest, FastProspectiveMatch
from app.realtime.session_manager import RealtimeSessionManager
from app.settings import JarvisSettings


LOGGER = logging.getLogger(__name__)
REVOCATION_NOTIFICATION_TIMEOUT_SECONDS = 3.0


class ProspectiveMemoryService:
    def __init__(
        self,
        *,
        repository: ProspectiveRepository,
        realtime_session_manager: RealtimeSessionManager,
        settings: JarvisSettings,
    ) -> None:
        self._repository = repository
        self._realtime_session_manager = realtime_session_manager
        self._settings = settings
        self._pending_deliveries: dict[UUID, ProspectiveDeliveryAttempt] = {}
        self._focus_change_locks: WeakValueDictionary[str, asyncio.Lock] = WeakValueDictionary()

    async def set_visual_focus(
        self,
        *,
        owner_id: str,
        target: str,
        description: str,
        purpose: str,
        mode: VisualFocusMode,
        expires_in_seconds: int | None,
    ) -> VisualFocus:
        default_ttl = (
            self._settings.PROSPECTIVE_FIND_ONCE_DEFAULT_TTL_SECONDS
            if mode == "find_once"
            else self._settings.PROSPECTIVE_MONITOR_DEFAULT_TTL_SECONDS
        )
        ttl = expires_in_seconds or default_ttl
        lock = self._focus_change_locks.setdefault(owner_id, asyncio.Lock())
        async with lock:
            focus = await self._repository.set_visual_focus(
                owner_id=owner_id,
                target=target,
                description=description,
                purpose=purpose,
                mode=mode,
                expires_at=datetime.now(UTC) + timedelta(seconds=ttl),
            )
            revoked = self._revoke_focus_deliveries(owner_id, except_focus_id=focus.id, reason="focus_replaced")
        await self._notify_revoked(revoked)
        await self._cancel_attention(owner_id, except_focus_id=focus.id, reason="focus_replaced")
        return focus

    async def get_visual_focus(self, *, owner_id: str) -> VisualFocus | None:
        return await self._repository.get_active_visual_focus(owner_id=owner_id)

    async def clear_visual_focus(
        self, *, owner_id: str, focus_id: UUID | None = None,
    ) -> VisualFocus | None:
        revoked = []
        lock = self._focus_change_locks.setdefault(owner_id, asyncio.Lock())
        async with lock:
            if focus_id is None:
                # Internal explicit-stop callers retain their existing behavior.
                # Model tools must supply an identity instead of reaching here.
                focus = await self._repository.clear_active_visual_focus(owner_id=owner_id)
            else:
                focus = await self._repository.clear_active_visual_focus(
                    owner_id=owner_id, expected_focus_id=focus_id,
                )
            if focus_id is not None:
                # An already inactive/expired target may still have pending audio,
                # but its stale stop must never revoke a newer focus's delivery.
                revoked = self._revoke_focus_deliveries(owner_id, focus_id=focus_id, reason="focus_cleared")
            elif focus is not None:
                revoked = self._revoke_focus_deliveries(owner_id, focus_id=focus.id, reason="focus_cleared")
            else:
                # Expired or already-completed focuses can still have queued
                # delivery work. An explicit stop applies to that work too.
                revoked = self._revoke_focus_deliveries(owner_id, reason="focus_cleared")
        await self._notify_revoked(revoked)
        await self._cancel_attention(owner_id, focus_id=focus_id or (focus.id if focus else None), reason="focus_cleared")
        return focus

    async def _cancel_attention(self, owner_id, **kwargs):
        cancel = getattr(self._realtime_session_manager, "cancel_visual_attention", None)
        if cancel is not None:
            try:
                async with asyncio.timeout(REVOCATION_NOTIFICATION_TIMEOUT_SECONDS):
                    await cancel(owner_id=owner_id, **kwargs)
            except Exception as error:
                LOGGER.warning("Visual attention clear pending error_type=%s", type(error).__name__)

    def _revoke_focus_deliveries(
        self, owner_id: str, *, reason: str, focus_id: UUID | None = None,
        except_focus_id: UUID | None = None,
    ) -> list[ProspectiveDeliveryAttempt]:
        revoked = []
        for attempt in self._pending_deliveries.values():
            if (
                attempt.owner_id == owner_id and attempt.subject_type == "visual_focus"
                and (focus_id is None or attempt.subject_id == focus_id)
                and attempt.subject_id != except_focus_id
            ):
                attempt.revoke(reason)
                revoked.append(attempt)
        return revoked

    async def _notify_revoked(self, attempts: list[ProspectiveDeliveryAttempt]) -> None:
        # Flags are already set and the focus-change lock is released. A slow
        # phone must not hold the owner's database/tool mutation lock.
        for attempt in attempts:
            try:
                async with asyncio.timeout(REVOCATION_NOTIFICATION_TIMEOUT_SECONDS):
                    await attempt.notify_revoked()
            except Exception as error:
                LOGGER.warning(
                    "prospective playback clear failed event_id=%s error_type=%s",
                    attempt.event_id, type(error).__name__,
                )

    async def process_matches(
        self,
        *,
        request: FastPerceptionRequest,
        matches: list[FastProspectiveMatch],
        source_payload_id: UUID | None = None,
        effect_guard=None,
        focus_decision=None,
    ) -> None:
        attention = selected_notification_repository(self._realtime_session_manager)
        if effect_guard is not None and attention is None:
            from app.perception.applicability import PerceptionEffectUnavailable
            raise PerceptionEffectUnavailable("perception_effect_delivery_unavailable")
        candidates = sorted(
            matches,
            key=lambda item: item.confidence,
            reverse=True,
        )[: self._settings.PROSPECTIVE_MAX_CANDIDATES_PER_ANALYSIS]
        for candidate in candidates:
            if focus_decision is not None and (candidate.subject_type != "visual_focus"
                    or candidate.subject_id != focus_decision.focus_id):
                continue
            if candidate.confidence < self._settings.VISUAL_INTERRUPT_CONFIDENCE_THRESHOLD:
                continue
            evidence = _validated_evidence(
                request,
                candidate,
                max_age_seconds=self._settings.PROSPECTIVE_MAX_EVIDENCE_AGE_SECONDS,
            )
            if evidence is None:
                continue
            source_args = ({"source_payload_id": source_payload_id} if source_payload_id is not None else {})
            if source_payload_id is not None:
                evidence["source_payload_id"] = str(source_payload_id)
            claim_token = await self._repository.try_claim_subject(
                owner_id=request.owner_id,
                subject_type=candidate.subject_type,
                subject_id=candidate.subject_id,
                claim_timeout_seconds=(
                    self._settings.PROSPECTIVE_TRIGGER_CLAIM_LEASE_SECONDS
                ),
            )
            if claim_token is None:
                if focus_decision is not None:
                    await self._repository.record_focus_decision(owner_id=request.owner_id,
                        decision=focus_decision, effect_guard=effect_guard)
                continue
            occurrence_key = _occurrence_key(
                candidate,
            )
            try:
                event = await self._repository.reserve_delivery_event(
                    owner_id=request.owner_id,
                    session_id=request.session_id,
                    subject_type=candidate.subject_type,
                    subject_id=candidate.subject_id,
                    claim_token=claim_token,
                    occurrence_key=occurrence_key,
                    message=candidate.suggested_message,
                    evidence=evidence,
                    **({"attention":attention, "max_age_seconds":self._settings.PROSPECTIVE_MAX_EVIDENCE_AGE_SECONDS}
                       if attention is not None else {}),
                    **source_args,
                    **({"effect_guard": effect_guard} if effect_guard is not None else {}),
                    **({"focus_decision": focus_decision} if focus_decision is not None else {}),
                    dedupe_window_seconds=(
                        self._settings.VISUAL_INTERRUPT_DEDUPE_SECONDS
                    ),
                )
            except (Exception, asyncio.CancelledError):
                await self._repository.release_subject_claim(
                    owner_id=request.owner_id, subject_type=candidate.subject_type,
                    subject_id=candidate.subject_id, claim_token=claim_token)
                raise
            if event is None:
                await self._repository.release_subject_claim(
                    owner_id=request.owner_id,
                    subject_type=candidate.subject_type,
                    subject_id=candidate.subject_id,
                    claim_token=claim_token,
                )
                continue

            if attention is not None:
                # The transaction's intent now owns all delivery and recovery.
                return

            async def persist_outcome(
                outcome: ProspectiveDeliveryOutcome,
                provider_response_id: str | None,
                *,
                event_id: UUID = event.id,
            ) -> None:
                await self._repository.resolve_delivery_event(
                    owner_id=request.owner_id,
                    event_id=event_id,
                    outcome=outcome,
                    provider_response_id=provider_response_id,
                    **source_args,
                    metadata={
                        "delivery_route": "realtime",
                        "reason": attempt.reason,
                        "playback_status": "unconfirmed",
                    },
                )
                self._pending_deliveries.pop(event_id, None)

            async def preflight(event_id: UUID = event.id) -> bool:
                return await self._repository.begin_delivery(
                    owner_id=request.owner_id, event_id=event_id,
                    session_id=attempt.session_id or request.session_id,
                    **source_args,
                )

            attempt = ProspectiveDeliveryAttempt(
                owner_id=request.owner_id, event_id=event.id,
                subject_type=candidate.subject_type, subject_id=candidate.subject_id,
                preflight=preflight, persist=persist_outcome,
                expires_at=_focus_expiry(request, candidate),
            )
            self._pending_deliveries[event.id] = attempt

            try:
                delivered_session_id = await self._realtime_session_manager.speak_prospective_interrupt(
                    owner_id=request.owner_id,
                    preferred_jarvis_session_id=request.session_id,
                    message=candidate.suggested_message,
                    event_id=event.id,
                    subject_type=candidate.subject_type,
                    subject_id=candidate.subject_id,
                    on_outcome=attempt.on_outcome,
                    delivery_attempt=attempt,
                    **source_args,
                )
            except (Exception, asyncio.CancelledError) as error:
                if not attempt.provider_owned:
                    attempt.reason = (
                        "admission_cancelled" if isinstance(error, asyncio.CancelledError)
                        else "admission_failed"
                    )
                    await attempt.on_outcome("failed", None)
                raise
            if delivered_session_id is None:
                if not attempt.provider_owned:
                    attempt.reason = attempt.reason or "no_delivery_admission"
                    await attempt.on_outcome("missed", None)
                return
            await self._repository.mark_event_dispatched(
                owner_id=request.owner_id,
                event_id=event.id,
                session_id=delivered_session_id,
            )
            return


def _focus_expiry(request: FastPerceptionRequest, candidate: FastProspectiveMatch) -> datetime | None:
    # This is application-owned focus metadata, not model prose. The database
    # independently checks the current subject/claim immediately before dispatch.
    focus = request.active_focus
    if (candidate.subject_type != "visual_focus" or not isinstance(focus, Mapping)
            or str(focus.get("id")) != str(candidate.subject_id)):
        return None
    value = focus.get("expires_at")
    if not isinstance(value, str):
        return None
    try:
        expires_at = datetime.fromisoformat(value)
    except ValueError:
        return None
    return expires_at if expires_at.tzinfo is not None else None


def _validated_evidence(
    request: FastPerceptionRequest,
    candidate: FastProspectiveMatch,
    *,
    max_age_seconds: int,
) -> dict[str, object] | None:
    by_id = {frame.frame_id: frame for frame in request.frames}
    frames = [by_id.get(frame_id) for frame_id in candidate.supporting_frame_ids]
    if not frames or any(frame is None for frame in frames):
        return None
    if any(
        frame.captured_at > request.requested_at
        or (request.requested_at - frame.captured_at).total_seconds() > max_age_seconds
        for frame in frames
        if frame is not None
    ):
        return None
    return {
        "observed_condition": candidate.observed_condition,
        "confidence": candidate.confidence,
        "supporting_frame_ids": [str(frame_id) for frame_id in candidate.supporting_frame_ids],
        "requested_at": request.requested_at.isoformat(),
    }


def _occurrence_key(
    candidate: FastProspectiveMatch,
) -> str:
    material = "|".join(
        [
            candidate.subject_type,
            str(candidate.subject_id),
            *(str(frame_id) for frame_id in candidate.supporting_frame_ids),
        ]
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()
