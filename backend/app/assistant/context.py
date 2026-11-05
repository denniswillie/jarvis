from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from app.assistant.context_providers import (
    DEFAULT_BEHAVIOR_INSTRUCTIONS,
    AssistantContextProvider,
    BehaviorInstructions,
    ContextBuildRequest,
    ContextSection,
    merge_behavior_instructions,
)
from app.preferences.models import JarvisPreference
from app.memory.prospective.models import ProspectiveContext
from app.reminders.models import JarvisReminder
from app.sessions.models import JarvisMessage, JarvisSession
from app.visual_memory.models import RecentVisualInterrupt


LOGGER = logging.getLogger(__name__)


class AssistantContextBuildError(RuntimeError):
    pass


@dataclass(frozen=True)
class AssistantContext:
    current_utc_time: datetime
    session: JarvisSession | None
    behavior: BehaviorInstructions
    recent_messages: list[JarvisMessage]
    active_reminders: list[JarvisReminder]
    preferences: list[JarvisPreference]
    recent_visual_interrupts: list[RecentVisualInterrupt]
    working_memory: dict[str, Any] | None = None
    prospective: ProspectiveContext | None = None
    second_brain: dict[str, Any] | None = None
    knowledge_sources: dict[str, Any] | None = None
    voice_continuity: dict[str, Any] | None = None
    voice_continuity_source_ids: tuple[UUID, ...] = ()
    provider_source_payload_ids: tuple[UUID, ...] = ()

    @property
    def source_payload_ids(self) -> tuple:
        return tuple(sorted(set(self.voice_continuity_source_ids) | set(self.provider_source_payload_ids) | {
            item.source_payload_id for item in self.recent_messages
            if item.content.strip() and _include_message_in_prompt(item) and item.source_payload_id is not None}))

    def as_prompt_payload(self) -> dict[str, Any]:
        payload = {
            "current_utc_time": self.current_utc_time.isoformat(),
            "session": (
                {"id": str(self.session.id), "title": self.session.title}
                if self.session is not None
                else None
            ),
            "behavior_instructions": self.behavior.as_payload(),
            "recent_messages": [
                _message_prompt_payload(item)
                for item in self.recent_messages
                if item.content.strip() and _include_message_in_prompt(item)
            ],
            "active_reminders": [
                {
                    "id": str(item.id),
                    "title": item.title,
                    "details": item.details,
                    "remind_at": item.remind_at.isoformat() if item.remind_at else None,
                    "condition": item.condition,
                }
                for item in self.active_reminders
            ],
            "preferences": [
                {"key": item.key, "value": item.value, "source": item.source}
                for item in self.preferences
            ],
            "working_memory": self.working_memory,
            "recent_visual_interrupts": [
                {
                    "message": item.message,
                    "reason": item.reason,
                    "confidence": item.confidence,
                    "created_at": item.created_at.isoformat(),
                }
                for item in self.recent_visual_interrupts
            ],
            "prospective_memory": (
                self.prospective.as_prompt_payload()
                if self.prospective is not None
                else None
            ),
        }
        if self.second_brain is not None:
            payload["second_brain"] = self.second_brain
        if self.knowledge_sources is not None:
            payload["knowledge_sources"] = self.knowledge_sources
        if self.voice_continuity is not None:
            payload["voice_continuity"] = self.voice_continuity
        return payload


def _include_message_in_prompt(message: JarvisMessage) -> bool:
    delivery = message.metadata.get("realtime_output")
    # Legacy messages remain eligible. Even late-classified commentary can
    # have been forwarded; exclude only explicitly suppressed assistant output.
    return not (
        message.role == "assistant"
        and isinstance(delivery, dict)
        and delivery.get("delivery_disposition") == "suppressed"
    )


def _message_prompt_payload(message: JarvisMessage) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "role": message.role,
        "modality": message.modality,
        "content": message.content,
        "created_at": message.created_at.isoformat(),
    }
    delivery = message.metadata.get("realtime_output")
    if message.role == "assistant" and isinstance(delivery, dict):
        payload["realtime_output"] = {
            "phase": delivery.get("phase"),
            "delivery_disposition": delivery.get("delivery_disposition"),
        }
    return payload


class AssistantContextBuilder:
    def __init__(
        self,
        *,
        providers: Sequence[AssistantContextProvider],
    ) -> None:
        self._providers = tuple(providers)

    async def build(
        self,
        *,
        owner_id: str,
        session: JarvisSession | None,
        recent_messages: list[JarvisMessage],
        current_utc_time: datetime | None = None,
    ) -> AssistantContext:
        sections = await self.build_sections(
            owner_id=owner_id,
            session=session,
            recent_messages=recent_messages,
            current_utc_time=current_utc_time,
        )
        return _format_legacy_assistant_context(sections)

    async def build_sections(
        self,
        *,
        owner_id: str,
        session: JarvisSession | None,
        recent_messages: list[JarvisMessage],
        current_utc_time: datetime | None = None,
    ) -> tuple[ContextSection, ...]:
        request = ContextBuildRequest(
            owner_id=owner_id,
            session=session,
            recent_messages=tuple(recent_messages),
            requested_at_utc=current_utc_time or datetime.now(UTC),
        )
        results = await asyncio.gather(
            *(provider.build_section(request) for provider in self._providers),
            return_exceptions=True,
        )
        sections: list[ContextSection] = []
        for provider, result in zip(self._providers, results, strict=True):
            if isinstance(result, asyncio.CancelledError):
                raise result
            if isinstance(result, Exception):
                if provider.critical:
                    raise AssistantContextBuildError(
                        f"Critical context provider failed: {provider.key}"
                    ) from result
                LOGGER.warning(
                    "optional assistant context provider failed provider=%s",
                    provider.key,
                    exc_info=(type(result), result, result.__traceback__),
                )
                result = provider.fallback_section(request)
            if result.key != provider.key or result.priority != provider.priority:
                raise AssistantContextBuildError(
                    f"Context provider returned inconsistent metadata: {provider.key}"
                )
            sections.append(result)

        ordered = tuple(sorted(sections, key=lambda item: (item.priority, item.key)))
        keys = [section.key for section in ordered]
        if len(keys) != len(set(keys)):
            raise AssistantContextBuildError("Assistant context provider keys must be unique.")
        return ordered


def _format_legacy_assistant_context(
    sections: Sequence[ContextSection],
) -> AssistantContext:
    by_key = {section.key: section.payload for section in sections}
    current_utc_time = by_key.get("current_utc_time")
    if not isinstance(current_utc_time, datetime):
        raise AssistantContextBuildError("Current UTC time context is required.")
    session = by_key.get("session")
    if session is not None and not isinstance(session, JarvisSession):
        raise AssistantContextBuildError("Session context is invalid.")
    behavior = by_key.get("behavior")
    if not isinstance(behavior, BehaviorInstructions):
        behavior = merge_behavior_instructions(
            persistent="",
            session_override=session.behavior_instructions if session else "",
        )
    return AssistantContext(
        provider_source_payload_ids=tuple(sorted({source for section in sections for source in section.source_payload_ids})),
        current_utc_time=current_utc_time,
        session=session,
        behavior=behavior,
        recent_messages=_list_payload(by_key, "recent_messages"),
        active_reminders=_list_payload(by_key, "active_reminders"),
        preferences=_list_payload(by_key, "preferences"),
        recent_visual_interrupts=_list_payload(by_key, "recent_visual_interrupts"),
        working_memory=(
            dict(by_key["working_memory"])
            if isinstance(by_key.get("working_memory"), dict)
            else None
        ),
        second_brain=(
            dict(by_key["second_brain"])
            if isinstance(by_key.get("second_brain"), dict)
            else None
        ),
        knowledge_sources=(
            dict(by_key["knowledge_sources"])
            if isinstance(by_key.get("knowledge_sources"), dict)
            else None
        ),
        prospective=(
            by_key.get("prospective")
            if isinstance(by_key.get("prospective"), ProspectiveContext)
            else None
        ),
    )


def _list_payload(payloads: dict[str, Any], key: str) -> list[Any]:
    value = payloads.get(key, [])
    return list(value) if isinstance(value, (list, tuple)) else []
