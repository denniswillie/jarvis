from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, ClassVar, Protocol

from app.assistant.repository import BehaviorProfileRepository
from app.preferences.repository import PreferenceRepository
from app.reminders.repository import ReminderRepository
from app.sessions.models import JarvisMessage, JarvisSession
from app.visual_memory.repository import VisualFrameRepository


DEFAULT_BEHAVIOR_INSTRUCTIONS = (
    "Be a useful daily-life companion. Proactively interrupt only when the available evidence is "
    "high-confidence and the interruption is immediately useful."
)


@dataclass(frozen=True)
class BehaviorInstructions:
    persistent: str
    session_override: str

    @property
    def combined(self) -> str:
        sections: list[str] = []
        if self.persistent.strip():
            sections.append("Persistent owner instructions:\n" + self.persistent.strip())
        if self.session_override.strip():
            sections.append("Current session override:\n" + self.session_override.strip())
        return "\n\n".join(sections) or DEFAULT_BEHAVIOR_INSTRUCTIONS

    def as_payload(self) -> dict[str, str]:
        return {
            "persistent": self.persistent,
            "session_override": self.session_override,
            "effective": self.combined,
        }


def merge_behavior_instructions(
    *, persistent: str | None, session_override: str | None
) -> BehaviorInstructions:
    return BehaviorInstructions(
        persistent=(persistent or "").strip(),
        session_override=(session_override or "").strip(),
    )


@dataclass(frozen=True)
class ContextBuildRequest:
    owner_id: str
    session: JarvisSession | None
    recent_messages: tuple[JarvisMessage, ...]
    requested_at_utc: datetime


@dataclass(frozen=True)
class ContextSection:
    key: str
    payload: Any
    priority: int
    freshness: datetime
    size: int | None = None
    source_payload_ids: tuple[UUID, ...] = ()


class AssistantContextProvider(Protocol):
    key: str
    priority: int
    critical: bool

    async def build_section(self, request: ContextBuildRequest) -> ContextSection: ...

    def fallback_section(self, request: ContextBuildRequest) -> ContextSection: ...


@dataclass(frozen=True)
class CurrentTimeContextProvider:
    key: ClassVar[str] = "current_utc_time"
    priority: ClassVar[int] = 10
    critical: ClassVar[bool] = True

    async def build_section(self, request: ContextBuildRequest) -> ContextSection:
        return self.fallback_section(request)

    def fallback_section(self, request: ContextBuildRequest) -> ContextSection:
        return _section(self, request, request.requested_at_utc, size=1)


@dataclass(frozen=True)
class SessionContextProvider:
    key: ClassVar[str] = "session"
    priority: ClassVar[int] = 20
    critical: ClassVar[bool] = True

    async def build_section(self, request: ContextBuildRequest) -> ContextSection:
        if not request.owner_id.strip():
            raise ValueError("owner_id is required to build assistant context")
        return self.fallback_section(request)

    def fallback_section(self, request: ContextBuildRequest) -> ContextSection:
        return _section(self, request, request.session, size=1 if request.session else 0)


@dataclass(frozen=True)
class BehaviorContextProvider:
    repository: BehaviorProfileRepository | None
    key: ClassVar[str] = "behavior"
    priority: ClassVar[int] = 30
    critical: ClassVar[bool] = False

    async def build_section(self, request: ContextBuildRequest) -> ContextSection:
        profile = (
            await self.repository.get(owner_id=request.owner_id)
            if self.repository is not None
            else None
        )
        behavior = merge_behavior_instructions(
            persistent=profile.behavior_instructions if profile else "",
            session_override=(
                request.session.behavior_instructions if request.session is not None else ""
            ),
        )
        return _section(self, request, behavior, size=len(behavior.combined))

    def fallback_section(self, request: ContextBuildRequest) -> ContextSection:
        behavior = merge_behavior_instructions(
            persistent="",
            session_override=(
                request.session.behavior_instructions if request.session is not None else ""
            ),
        )
        return _section(self, request, behavior, size=len(behavior.combined))


@dataclass(frozen=True)
class RecentMessagesContextProvider:
    max_items: int
    key: ClassVar[str] = "recent_messages"
    priority: ClassVar[int] = 40
    critical: ClassVar[bool] = False

    async def build_section(self, request: ContextBuildRequest) -> ContextSection:
        messages = list(request.recent_messages[-self.max_items :])
        return _section(self, request, messages, size=len(messages))

    def fallback_section(self, request: ContextBuildRequest) -> ContextSection:
        return _section(self, request, [], size=0)


@dataclass(frozen=True)
class ActiveRemindersContextProvider:
    repository: ReminderRepository | None
    max_items: int = 20
    key: ClassVar[str] = "active_reminders"
    priority: ClassVar[int] = 50
    critical: ClassVar[bool] = False

    async def build_section(self, request: ContextBuildRequest) -> ContextSection:
        reminders = (
            await self.repository.list_active_reminders(owner_id=request.owner_id)
            if self.repository is not None
            else []
        )
        bounded = list(reminders[: self.max_items])
        return _section(self, request, bounded, size=len(bounded))

    def fallback_section(self, request: ContextBuildRequest) -> ContextSection:
        return _section(self, request, [], size=0)


@dataclass(frozen=True)
class PreferencesContextProvider:
    repository: PreferenceRepository | None
    max_items: int = 20
    key: ClassVar[str] = "preferences"
    priority: ClassVar[int] = 60
    critical: ClassVar[bool] = False

    async def build_section(self, request: ContextBuildRequest) -> ContextSection:
        preferences = (
            await self.repository.search_preferences(
                owner_id=request.owner_id,
                query=None,
                limit=self.max_items,
            )
            if self.repository is not None
            else []
        )
        bounded = list(preferences[: self.max_items])
        return _section(self, request, bounded, size=len(bounded))

    def fallback_section(self, request: ContextBuildRequest) -> ContextSection:
        return _section(self, request, [], size=0)


@dataclass(frozen=True)
class RecentVisualInterruptsContextProvider:
    repository: VisualFrameRepository | None
    max_items: int = 8
    key: ClassVar[str] = "recent_visual_interrupts"
    priority: ClassVar[int] = 80
    critical: ClassVar[bool] = False

    async def build_section(self, request: ContextBuildRequest) -> ContextSection:
        interruptions = []
        if self.repository is not None and request.session is not None:
            interruptions = await self.repository.list_recent_visual_interrupts(
                owner_id=request.owner_id,
                session_id=request.session.id,
                limit=self.max_items,
            )
        bounded = list(interruptions[: self.max_items])
        return _section(self, request, bounded, size=len(bounded))

    def fallback_section(self, request: ContextBuildRequest) -> ContextSection:
        return _section(self, request, [], size=0)


def _section(
    provider: AssistantContextProvider,
    request: ContextBuildRequest,
    payload: Any,
    *,
    size: int | None,
) -> ContextSection:
    return ContextSection(
        key=provider.key,
        payload=payload,
        priority=provider.priority,
        freshness=request.requested_at_utc,
        size=size,
    )
