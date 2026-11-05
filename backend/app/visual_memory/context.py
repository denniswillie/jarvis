from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Mapping
from uuid import UUID

from app.assistant.context import merge_behavior_instructions
from app.assistant.repository import BehaviorProfileRepository
from app.sessions.models import JarvisMessage, JarvisSession
from app.sessions.repository import SessionRepository
from app.settings import JarvisSettings
from app.perception.services import SessionWorkingMemoryReader
from app.memory.prospective.context import ProspectiveContextBuilder
from app.memory.prospective.models import ProspectiveContext
from app.visual_memory.models import (
    ActiveVisualReminder,
    RecentVisualInterrupt,
    VisualFrameRecord,
)
from app.visual_memory.repository import VisualFrameRepository


@dataclass(frozen=True)
class VisualAnalyzerContext:
    owner_id: str
    session_id: UUID
    audio_session_id: str | None
    frame_id: UUID
    frame_sequence: int
    captured_at: datetime
    current_utc_time: datetime
    behavior_instructions: str
    recent_audio_transcripts: list[JarvisMessage]
    recent_visual_interrupts: list[RecentVisualInterrupt]
    active_reminders: list[ActiveVisualReminder]
    previous_working_state: Mapping[str, Any]
    active_focus: Any | None


class VisualAnalyzerContextBuilder:
    def __init__(
        self,
        *,
        settings: JarvisSettings,
        visual_repository: VisualFrameRepository,
        session_repository: SessionRepository,
        behavior_repository: BehaviorProfileRepository,
        working_memory_reader: SessionWorkingMemoryReader | None = None,
        prospective_context_builder: ProspectiveContextBuilder | None = None,
    ) -> None:
        self._settings = settings
        self._visual_repository = visual_repository
        self._session_repository = session_repository
        self._behavior_repository = behavior_repository
        self._working_memory_reader = working_memory_reader
        self._prospective_context_builder = prospective_context_builder

    async def build(self, frame: VisualFrameRecord) -> VisualAnalyzerContext | None:
        session = await self._session_repository.get_session(frame.owner_id, frame.session_id)
        if session is None:
            return None
        recent_messages = await self._session_repository.get_recent_messages(
            frame.owner_id,
            frame.session_id,
            limit=self._settings.JARVIS_REALTIME_RECENT_MESSAGES_LIMIT,
        )
        behavior_profile = await self._behavior_repository.get(owner_id=frame.owner_id)
        working_memory = (
            await self._working_memory_reader.get_session_working_memory(
                owner_id=frame.owner_id,
                session_id=frame.session_id,
            )
            if self._working_memory_reader is not None
            else None
        )
        prospective = (
            await self._prospective_context_builder.build(owner_id=frame.owner_id)
            if self._prospective_context_builder is not None
            else None
        )
        active_focus = _resolve_active_focus(
            prospective=prospective,
            legacy_focus=session.metadata.get("active_focus"),
        )
        return VisualAnalyzerContext(
            owner_id=frame.owner_id,
            session_id=frame.session_id,
            audio_session_id=frame.audio_session_id,
            frame_id=frame.id,
            frame_sequence=frame.frame_sequence,
            captured_at=frame.captured_at,
            current_utc_time=datetime.now(UTC),
            behavior_instructions=merge_behavior_instructions(
                persistent=behavior_profile.behavior_instructions if behavior_profile else "",
                session_override=session.behavior_instructions,
            ).combined,
            recent_audio_transcripts=[
                message
                for message in recent_messages
                if message.modality == "audio"
                and message.content.strip()
                and message.created_at <= frame.captured_at
            ],
            recent_visual_interrupts=await self._visual_repository.list_recent_visual_interrupts(
                owner_id=frame.owner_id,
                session_id=frame.session_id,
                limit=8,
            ),
            active_reminders=(
                list(prospective.conditional_reminders)
                if prospective is not None
                else await self._visual_repository.list_active_visual_reminders(
                    owner_id=frame.owner_id,
                    limit=20,
                )
            ),
            previous_working_state=working_memory.state if working_memory else {},
            active_focus=active_focus,
        )


def _resolve_active_focus(
    *,
    prospective: ProspectiveContext | None,
    legacy_focus: Any,
) -> Any | None:
    if prospective is None:
        return legacy_focus
    if prospective.visual_focus is None:
        return None
    return prospective.visual_focus.as_prompt_payload()
