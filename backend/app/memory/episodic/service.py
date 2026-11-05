from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from datetime import datetime
from typing import Any
from uuid import UUID

from app.experience.models import (
    ExperienceEvent,
    ExperienceEventEvidence,
    ExperienceEventKeyframe,
)
from app.memory.consolidation.contracts import ConsolidationResult


@dataclass(frozen=True, slots=True)
class ReminderDeliveryEvidence:
    reminder_event_id: UUID
    reminder_id: UUID
    occurred_at: datetime
    message: str
    metadata: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class EpisodePromotionContext:
    event: ExperienceEvent
    result: ConsolidationResult
    evidence: tuple[ExperienceEventEvidence, ...]
    source_payloads: Mapping[tuple[str, UUID], Mapping[str, Any]]
    context_messages: tuple[Mapping[str, Any], ...]
    provisional_keyframes: tuple[ExperienceEventKeyframe, ...]
    keyframe_capture_times: Mapping[UUID, datetime]
    has_explicit_memory_evidence: bool
    reminder_deliveries: tuple[ReminderDeliveryEvidence, ...]

    @property
    def has_reminder_delivery(self) -> bool:
        return bool(self.reminder_deliveries)


@dataclass(frozen=True, slots=True)
class FinalKeyframe:
    visual_frame_id: UUID
    captured_at: datetime
    rank: int
    reason: str
    quality_score: float
    selected_by: str


@dataclass(frozen=True, slots=True)
class EpisodePromotionPlan:
    event_id: UUID
    owner_id: str
    session_id: UUID
    promote: bool
    reason: str
    is_explicit: bool
    retain_keyframes_permanently: bool
    final_keyframes: tuple[FinalKeyframe, ...]


class EpisodePromotionService:
    """Apply deterministic, code-owned episode promotion policy."""

    def __init__(
        self,
        *,
        minimum_importance: float,
        important_keyframe_minimum_importance: float,
        final_keyframe_limit: int = 3,
    ) -> None:
        _require_score(minimum_importance, "minimum_importance")
        _require_score(
            important_keyframe_minimum_importance,
            "important_keyframe_minimum_importance",
        )
        if final_keyframe_limit < 1 or final_keyframe_limit > 3:
            raise ValueError("final_keyframe_limit must be between one and three")
        self._minimum_importance = minimum_importance
        self._important_keyframe_minimum_importance = (
            important_keyframe_minimum_importance
        )
        self._final_keyframe_limit = final_keyframe_limit

    def plan(self, context: EpisodePromotionContext) -> EpisodePromotionPlan:
        event = context.event
        if event.status != "consolidated" or event.consolidation_result is None:
            raise ValueError("episode promotion requires a consolidated event")
        if event.ended_at is None:
            raise ValueError("episode promotion requires an event end time")

        mutating_tools = _completed_mutating_tools(context)
        important_by_policy = (
            context.result.episode_importance
            >= self._important_keyframe_minimum_importance
        )
        model_gate = (
            context.result.should_promote_to_episode
            and context.result.episode_importance >= self._minimum_importance
        )

        if context.has_explicit_memory_evidence:
            promote, reason = True, "explicit memory request"
        elif context.has_reminder_delivery:
            promote, reason = True, "reminder delivered during event"
        elif mutating_tools:
            promote, reason = (
                True,
                "completed user-visible mutation: " + ", ".join(mutating_tools),
            )
        elif important_by_policy:
            promote, reason = True, "event met the important-memory policy"
        elif model_gate:
            promote, reason = True, "validated consolidation recommended promotion"
        else:
            promote, reason = False, "routine event below promotion policy"

        return EpisodePromotionPlan(
            event_id=event.id,
            owner_id=event.owner_id,
            session_id=event.session_id,
            promote=promote,
            reason=reason,
            is_explicit=context.has_explicit_memory_evidence,
            retain_keyframes_permanently=(
                context.has_explicit_memory_evidence or important_by_policy
            ),
            final_keyframes=_select_final_keyframes(
                result=context.result,
                candidates=context.provisional_keyframes,
                capture_times=context.keyframe_capture_times,
                limit=self._final_keyframe_limit,
            ),
        )


def _completed_mutating_tools(context: EpisodePromotionContext) -> tuple[str, ...]:
    names: set[str] = set()
    for evidence in context.evidence:
        if evidence.evidence_type != "tool_execution":
            continue
        payload = context.source_payloads.get(
            (evidence.evidence_type, evidence.evidence_ref_id)
        )
        if payload is None or payload.get("status") != "completed":
            continue
        tool_name = payload.get("tool_name")
        if isinstance(tool_name, str) and payload.get("mutates_state") is True:
            names.add(tool_name)
    return tuple(sorted(names))


def _select_final_keyframes(
    *,
    result: ConsolidationResult,
    candidates: Sequence[ExperienceEventKeyframe],
    capture_times: Mapping[UUID, datetime],
    limit: int,
) -> tuple[FinalKeyframe, ...]:
    by_id = {candidate.visual_frame_id: candidate for candidate in candidates}
    selected: list[tuple[ExperienceEventKeyframe, str]] = []
    selected_ids: set[UUID] = set()

    for raw_frame_id in result.selected_keyframe_ids:
        frame_id = UUID(str(raw_frame_id))
        candidate = by_id.get(frame_id)
        if candidate is None or frame_id in selected_ids:
            continue
        selected.append((candidate, "gpt"))
        selected_ids.add(frame_id)
        if len(selected) == limit:
            break

    for candidate in sorted(candidates, key=lambda item: (item.rank, str(item.visual_frame_id))):
        if len(selected) == limit:
            break
        if candidate.visual_frame_id in selected_ids:
            continue
        selected.append((candidate, "deterministic"))
        selected_ids.add(candidate.visual_frame_id)

    return tuple(
        FinalKeyframe(
            visual_frame_id=candidate.visual_frame_id,
            captured_at=capture_times[candidate.visual_frame_id],
            rank=index,
            reason=candidate.reason,
            quality_score=candidate.quality_score,
            selected_by=selected_by,
        )
        for index, (candidate, selected_by) in enumerate(selected, start=1)
    )


def evidence_excerpt(
    evidence: ExperienceEventEvidence,
    payload: Mapping[str, Any] | None,
) -> str:
    if payload is None:
        return ""
    if evidence.evidence_type == "message":
        content = payload.get("content")
        return _bounded_text(content)
    if evidence.evidence_type == "tool_execution":
        tool_name = payload.get("tool_name")
        status = payload.get("status")
        if isinstance(tool_name, str):
            return _bounded_text(f"{tool_name}: {status or 'unknown'}")
    return ""


def _bounded_text(value: object, *, limit: int = 1_000) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip()[:limit]


def _require_score(value: float, label: str) -> None:
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{label} must be between zero and one")
