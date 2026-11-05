from __future__ import annotations

import logging
from datetime import datetime
from uuid import UUID

from app.realtime.session_manager import RealtimeSessionManager
from app.attention.notifications import selected_notification_repository
from app.visual_memory.dedupe import visual_interrupt_dedupe_bucket
from app.visual_memory.models import VisualInterruptRecord
from app.visual_memory.repository import VisualFrameRepository
from app.perception.applicability import PerceptionEffectUnavailable


logger = logging.getLogger(__name__)


class VisualInterruptService:
    def __init__(
        self,
        *,
        repository: VisualFrameRepository,
        realtime_session_manager: RealtimeSessionManager,
        dedupe_seconds: int,
        max_age_seconds: float | None = None,
    ) -> None:
        self._repository = repository
        self._realtime_session_manager = realtime_session_manager
        self._dedupe_seconds = dedupe_seconds
        self._max_age_seconds = max_age_seconds

    async def maybe_interrupt(
        self,
        *,
        owner_id: str,
        session_id: UUID,
        audio_session_id: str | None,
        visual_frame_id: UUID,
        source_received_at: datetime | None,
        message: str,
        reason: str,
        confidence: float,
        source_captured_at: datetime | None = None,
        source_payload_id: UUID | None = None,
        effect_guard=None,
    ) -> VisualInterruptRecord | None:
        attention = selected_notification_repository(self._realtime_session_manager)
        if effect_guard is not None and attention is None:
            raise PerceptionEffectUnavailable("perception_effect_delivery_unavailable")
        dedupe_bucket = visual_interrupt_dedupe_bucket(message)
        if await self._repository.recent_visual_interrupt_exists(
            owner_id=owner_id,
            session_id=session_id,
            dedupe_bucket=dedupe_bucket,
            window_seconds=self._dedupe_seconds,
        ):
            return None

        interrupt = await self._repository.create_visual_interrupt(
            owner_id=owner_id,
            session_id=session_id,
            visual_frame_id=visual_frame_id,
            message=message,
            reason=reason,
            confidence=confidence,
            dedupe_bucket=dedupe_bucket,
            **({"attention":attention, "max_age_seconds":self._max_age_seconds} if attention is not None else {}),
            **({"source_payload_id": source_payload_id} if source_payload_id is not None else {}),
            **({"effect_guard": effect_guard} if effect_guard is not None else {}),
        )
        if attention is not None:
            return interrupt
        try:
            accepted = await self._realtime_session_manager.speak_visual_interrupt(
                owner_id=owner_id,
                source_received_at=source_received_at,
                audio_session_id=audio_session_id,
                jarvis_session_id=session_id,
                message=message,
                interrupt_id=interrupt.id,
                visual_frame_id=visual_frame_id,
                source_captured_at=source_captured_at,
                reason=reason,
                confidence=confidence,
                **({"source_payload_id": source_payload_id} if source_payload_id is not None else {}),
            )
        except Exception:
            logger.info("visual interrupt %s delivery failed", interrupt.id, exc_info=True)
            accepted = False
        if not accepted:
            logger.info("visual interrupt %s candidate recorded; speech not admitted", interrupt.id)
        return interrupt
