from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from math import isfinite
from uuid import UUID


@dataclass(frozen=True)
class EventKeyframeCandidate:
    visual_frame_id: UUID
    captured_at: datetime
    reason: str
    analyzer_score: float
    observation_confidence: float
    has_readable_text: bool
    has_state_change: bool
    entity_count: int

    def __post_init__(self) -> None:
        if not self.reason.strip():
            raise ValueError("keyframe candidate reason cannot be blank")
        _require_score(self.analyzer_score, "analyzer_score")
        _require_score(self.observation_confidence, "observation_confidence")
        if self.entity_count < 0:
            raise ValueError("entity_count cannot be negative")


@dataclass(frozen=True)
class ScoredEventKeyframe:
    candidate: EventKeyframeCandidate
    quality_score: float

    def __post_init__(self) -> None:
        _require_score(self.quality_score, "quality_score")


@dataclass(frozen=True)
class KeyframeRetentionPolicy:
    final_keyframe_max_count: int = 3
    consolidation_candidate_max_count: int = 6
    open_event_candidate_max_count: int = 12
    candidate_retention_seconds: int = 1_800
    ordinary_keyframe_retention_seconds: int = 86_400

    def __post_init__(self) -> None:
        values = {
            "final_keyframe_max_count": self.final_keyframe_max_count,
            "consolidation_candidate_max_count": self.consolidation_candidate_max_count,
            "open_event_candidate_max_count": self.open_event_candidate_max_count,
            "candidate_retention_seconds": self.candidate_retention_seconds,
            "ordinary_keyframe_retention_seconds": self.ordinary_keyframe_retention_seconds,
        }
        for label, value in values.items():
            if value < 1:
                raise ValueError(f"{label} must be positive")
        if self.final_keyframe_max_count > self.consolidation_candidate_max_count:
            raise ValueError(
                "final keyframe count cannot exceed the consolidation candidate count"
            )
        if self.consolidation_candidate_max_count > self.open_event_candidate_max_count:
            raise ValueError(
                "consolidation candidate count cannot exceed the open-event candidate count"
            )


class DeterministicKeyframeSelector:
    """Rank event frames without a model call or mutable process state."""

    def select(
        self,
        candidates: Sequence[EventKeyframeCandidate],
        *,
        limit: int,
    ) -> tuple[ScoredEventKeyframe, ...]:
        if limit < 1 or not candidates:
            return ()
        remaining = list(_deduplicate_candidates(candidates))
        event_start = min(item.captured_at for item in remaining)
        event_end = max(item.captured_at for item in remaining)
        span_seconds = max(1.0, (event_end - event_start).total_seconds())
        selected: list[ScoredEventKeyframe] = []

        while remaining and len(selected) < limit:
            scored = [
                ScoredEventKeyframe(
                    candidate=item,
                    quality_score=_selection_score(
                        item,
                        selected=selected,
                        event_start=event_start,
                        event_end=event_end,
                        span_seconds=span_seconds,
                    ),
                )
                for item in remaining
            ]
            scored.sort(
                key=lambda item: (
                    -item.quality_score,
                    item.candidate.captured_at,
                    str(item.candidate.visual_frame_id),
                )
            )
            winner = scored[0]
            selected.append(winner)
            remaining = [
                item
                for item in remaining
                if item.visual_frame_id != winner.candidate.visual_frame_id
            ]
        return tuple(selected)


def _deduplicate_candidates(
    candidates: Sequence[EventKeyframeCandidate],
) -> tuple[EventKeyframeCandidate, ...]:
    by_frame: dict[UUID, EventKeyframeCandidate] = {}
    for candidate in sorted(
        candidates,
        key=lambda item: (item.captured_at, str(item.visual_frame_id), item.reason),
    ):
        current = by_frame.get(candidate.visual_frame_id)
        if current is None or _base_score(candidate) > _base_score(current):
            by_frame[candidate.visual_frame_id] = candidate
    return tuple(by_frame.values())


def _selection_score(
    candidate: EventKeyframeCandidate,
    *,
    selected: Sequence[ScoredEventKeyframe],
    event_start: datetime,
    event_end: datetime,
    span_seconds: float,
) -> float:
    progress = (candidate.captured_at - event_start).total_seconds() / span_seconds
    if candidate.captured_at == event_end:
        temporal_role_bonus = 0.06
    elif candidate.captured_at == event_start:
        temporal_role_bonus = 0.04
    elif 0.35 <= progress <= 0.65:
        temporal_role_bonus = 0.03
    else:
        temporal_role_bonus = 0.0

    if selected:
        diversity = min(
            abs((candidate.captured_at - item.candidate.captured_at).total_seconds())
            / span_seconds
            for item in selected
        )
        repeated_reasons = sum(
            _normalized_reason(candidate.reason)
            == _normalized_reason(item.candidate.reason)
            for item in selected
        )
    else:
        diversity = 0.0
        repeated_reasons = 0

    score = (
        _base_score(candidate)
        + min(0.12, diversity * 0.12)
        + temporal_role_bonus
        - min(0.16, repeated_reasons * 0.08)
    )
    return max(0.0, min(1.0, score))


def _base_score(candidate: EventKeyframeCandidate) -> float:
    return min(
        1.0,
        candidate.analyzer_score * 0.34
        + candidate.observation_confidence * 0.18
        + (0.14 if candidate.has_readable_text else 0.0)
        + (0.14 if candidate.has_state_change else 0.0)
        + min(candidate.entity_count, 4) * 0.025,
    )


def _normalized_reason(value: str) -> str:
    return " ".join(value.casefold().split())


def _require_score(value: float, label: str) -> None:
    if not isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{label} must be finite and between zero and one")
