from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, ClassVar, Protocol

from app.assistant.context_providers import ContextBuildRequest, ContextSection
from app.memory.models import MemoryItem
from app.memory.prospective.models import (
    ProspectiveContext,
    ProspectiveMemorySummary,
)
from app.perception.models import OwnerWorkingMemoryState
from app.reminders.models import JarvisReminder


LOGGER = logging.getLogger(__name__)


class FocusReader(Protocol):
    async def get_active_visual_focus(self, *, owner_id: str): ...


class ReminderReader(Protocol):
    async def list_active_reminders(self, *, owner_id: str) -> list[JarvisReminder]: ...


class GoalReader(Protocol):
    async def list_active_goals(
        self, *, owner_id: str, limit: int
    ) -> list[MemoryItem]: ...


class OwnerWorkingMemoryReader(Protocol):
    async def get_owner_working_memory(
        self, *, owner_id: str
    ) -> OwnerWorkingMemoryState | None: ...


class ProspectiveContextBuilder:
    def __init__(
        self,
        *,
        focus_reader: FocusReader,
        reminder_reader: ReminderReader,
        goal_reader: GoalReader | None,
        owner_working_memory_reader: OwnerWorkingMemoryReader,
        max_conditional_reminders: int,
        max_goals: int,
        max_commitments: int,
    ) -> None:
        self._focus_reader = focus_reader
        self._reminder_reader = reminder_reader
        self._goal_reader = goal_reader
        self._owner_working_memory_reader = owner_working_memory_reader
        self._max_conditional_reminders = max_conditional_reminders
        self._max_goals = max_goals
        self._max_commitments = max_commitments

    async def build(
        self,
        *,
        owner_id: str,
        generated_at: datetime | None = None,
    ) -> ProspectiveContext:
        now = generated_at or datetime.now(UTC)
        async def legacy_goals():
            # This is an optional legacy projection, not source-backed recall or
            # the responsibilities authority. Never query it after selection.
            if self._goal_reader is None:
                return []
            return await self._goal_reader.list_active_goals(owner_id=owner_id, limit=self._max_goals)

        results = await asyncio.gather(
            self._focus_reader.get_active_visual_focus(owner_id=owner_id),
            self._reminder_reader.list_active_reminders(owner_id=owner_id),
            legacy_goals(),
            self._owner_working_memory_reader.get_owner_working_memory(
                owner_id=owner_id
            ),
            return_exceptions=True,
        )
        focus = _result_or_default(results[0], None, "visual_focus")
        reminders = _result_or_default(results[1], [], "conditional_reminders")
        goals = _result_or_default(results[2], [], "goals")
        owner_state = _result_or_default(results[3], None, "commitments")
        conditional = tuple(
            reminder
            for reminder in reminders
            if reminder.status == "active"
            and reminder.condition is not None
            and reminder.condition.strip()
        )[: self._max_conditional_reminders]
        summaries = tuple(
            ProspectiveMemorySummary(
                id=item.id,
                title=item.title,
                content=item.content,
                importance=item.importance,
                updated_at=item.updated_at,
            )
            for item in goals[: self._max_goals]
        )
        return ProspectiveContext(
            visual_focus=focus,
            conditional_reminders=conditional,
            goals=summaries,
            commitments=_commitments(owner_state, self._max_commitments),
            generated_at=now,
        )


@dataclass(frozen=True)
class ProspectiveContextProvider:
    builder: ProspectiveContextBuilder
    key: ClassVar[str] = "prospective"
    priority: ClassVar[int] = 55
    critical: ClassVar[bool] = False

    async def build_section(self, request: ContextBuildRequest) -> ContextSection:
        context = await self.builder.build(
            owner_id=request.owner_id,
            generated_at=request.requested_at_utc,
        )
        return ContextSection(
            key=self.key,
            payload=context,
            priority=self.priority,
            freshness=request.requested_at_utc,
            size=(
                int(context.visual_focus is not None)
                + len(context.conditional_reminders)
                + len(context.goals)
                + len(context.commitments)
            ),
        )

    def fallback_section(self, request: ContextBuildRequest) -> ContextSection:
        return ContextSection(
            key=self.key,
            payload=ProspectiveContext(
                visual_focus=None,
                conditional_reminders=(),
                goals=(),
                commitments=(),
                generated_at=request.requested_at_utc,
            ),
            priority=self.priority,
            freshness=request.requested_at_utc,
            size=0,
        )


def _result_or_default(value: Any, default: Any, source: str) -> Any:
    if isinstance(value, BaseException):
        LOGGER.warning(
            "prospective context source failed source=%s",
            source,
            exc_info=(type(value), value, value.__traceback__),
        )
        return default
    return value


def _commitments(
    state: OwnerWorkingMemoryState | None,
    limit: int,
) -> tuple[str, ...]:
    if state is None:
        return ()
    entry = state.state.get("pending_commitments")
    if not isinstance(entry, dict):
        return ()
    value = entry.get("value")
    if not isinstance(value, list):
        return ()
    normalized = [str(item).strip() for item in value if str(item).strip()]
    return tuple(normalized[:limit])
