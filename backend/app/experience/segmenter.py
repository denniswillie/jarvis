from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Literal, Mapping

from app.experience.evidence import EventEvidenceDescriptor, event_last_evidence_at
from app.experience.models import ExperienceEvent


SegmentationAction = Literal["start", "join", "split", "unattached"]


@dataclass(frozen=True)
class EventSegmentationPolicy:
    idle_close_seconds: int
    max_event_seconds: int

    def __post_init__(self) -> None:
        if self.idle_close_seconds < 1:
            raise ValueError("idle_close_seconds must be positive")
        if self.max_event_seconds < 1:
            raise ValueError("max_event_seconds must be positive")


@dataclass(frozen=True)
class SegmentationDecision:
    action: SegmentationAction
    reason: str
    previous_event_ended_at: datetime | None = None


class ExperienceEventSegmenter:
    def __init__(self, policy: EventSegmentationPolicy) -> None:
        self.policy = policy

    @property
    def policy_idle_delta(self) -> timedelta:
        return timedelta(seconds=self.policy.idle_close_seconds)

    @property
    def policy_max_delta(self) -> timedelta:
        return timedelta(seconds=self.policy.max_event_seconds)

    def decide(
        self,
        *,
        current: ExperienceEvent | None,
        evidence: EventEvidenceDescriptor,
        latest_closed: ExperienceEvent | None = None,
    ) -> SegmentationDecision:
        if current is None:
            if (
                latest_closed is not None
                and latest_closed.ended_at is not None
                and evidence.occurred_at <= latest_closed.ended_at
            ):
                return SegmentationDecision("unattached", "finalized_boundary")
            return SegmentationDecision("start", "first_evidence")

        if evidence.occurred_at < current.started_at:
            # Independent transcript/tool writers can finish in a different order
            # from their source timestamps. Extend only the still-open window;
            # never reopen finalized history or invent a new scene boundary.
            if (latest_closed is not None and latest_closed.ended_at is not None
                    and evidence.occurred_at <= latest_closed.ended_at):
                return SegmentationDecision("unattached", "finalized_boundary")
            if (not evidence.starts_new_event
                    and current.started_at - evidence.occurred_at <= self.policy_idle_delta
                    and event_last_evidence_at(current) - evidence.occurred_at < self.policy_max_delta):
                return SegmentationDecision("join", "late_open_evidence")
            return SegmentationDecision("unattached", "before_open_event")

        last_evidence_at = event_last_evidence_at(current)
        if evidence.starts_new_event and _has_evidence(current.state):
            return SegmentationDecision(
                "split",
                evidence.boundary_reason or "analyzer_boundary",
                last_evidence_at,
            )
        if evidence.occurred_at - last_evidence_at > timedelta(
            seconds=self.policy.idle_close_seconds
        ):
            return SegmentationDecision("split", "idle_gap", last_evidence_at)
        if evidence.occurred_at - current.started_at >= timedelta(
            seconds=self.policy.max_event_seconds
        ):
            return SegmentationDecision("split", "max_duration", last_evidence_at)
        return SegmentationDecision("join", "same_event")

    def due_close_reason(
        self,
        event: ExperienceEvent,
        *,
        now: datetime,
    ) -> str | None:
        if now - event.started_at >= timedelta(seconds=self.policy.max_event_seconds):
            return "max_duration"
        if now - event_last_evidence_at(event) >= timedelta(
            seconds=self.policy.idle_close_seconds
        ):
            return "idle_timeout"
        return None



def _has_evidence(state: Mapping[str, object]) -> bool:
    counts = state.get("evidence_counts")
    return isinstance(counts, Mapping) and any(
        isinstance(value, int) and value > 0 for value in counts.values()
    )
