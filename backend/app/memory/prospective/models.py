from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal, Mapping
from uuid import UUID

from app.reminders.models import JarvisReminder
from app.live_state.models import SourceDependency
from app.live_state.perception import CheckLifecycle, CheckResolution, SituationalCheck


VisualFocusMode = Literal["find_once", "monitor"]
ProspectiveSubjectType = Literal["visual_focus", "conditional_reminder"]
ProspectiveDeliveryOutcome = Literal[
    "delivered", "missed", "failed", "interrupted"
]


@dataclass(frozen=True, slots=True)
class VisualFocus:
    id: UUID
    owner_id: str
    target: str
    description: str
    purpose: str
    mode: VisualFocusMode
    created_from_message_id: UUID | None
    created_at: datetime
    expires_at: datetime
    completed_at: datetime | None
    cleared_at: datetime | None
    completion_evidence: Mapping[str, Any] | None
    trigger_claim_token: UUID | None
    trigger_claimed_at: datetime | None
    delivery_status: Mapping[str, Any] | None = None
    revision: int = 1
    episode_revision: int = 1
    check_resolution: CheckResolution | None = None

    def as_check(self, scope, *, now: datetime) -> SituationalCheck:
        """Project observation tracking, never physical-action completion.

        A monitor retains its last qualified evidence without ending tracking.
        Delivery-only completion of a legacy watch supplies no typed resolution.
        Dismissal/expiry take precedence over historical resolution metadata.
        """
        if scope.owner_id != self.owner_id:
            raise ValueError("Focus belongs to another owner")
        owner = SourceDependency(source_namespace="visual_focus", source_id=self.id, revision=self.revision)
        episode = SourceDependency(source_namespace="episode", source_id=self.id, revision=self.episode_revision)
        resolution = self.check_resolution
        if self.cleared_at is not None:
            lifecycle = CheckLifecycle.DISMISSED
        elif now >= self.expires_at:
            lifecycle = CheckLifecycle.EXPIRED
        elif self.mode == "find_once" and resolution is not None:
            lifecycle = CheckLifecycle.RESOLVED
        elif self.completed_at is not None:
            lifecycle = CheckLifecycle.EXPIRED
        else:
            lifecycle = CheckLifecycle.TRACKING
        return SituationalCheck(id=self.id, revision=self.revision, scope=scope,
            episode_or_object_ref=episode, origin="explicit_request", continuity="durable",
            canonical_owner_ref=owner, question="Inspect the requested target: " + self.target,
            dependency_versions=(owner, episode), lifecycle=lifecycle,
            relevance_now=lifecycle == CheckLifecycle.TRACKING,
            evidence_refs=resolution.source_refs if resolution else (),
            evidence_availability="unknown", observation_window_expires_at=self.expires_at,
            resolution=resolution if lifecycle == CheckLifecycle.RESOLVED else None)

    def as_prompt_payload(self) -> dict[str, Any]:
        return {
            "id": str(self.id),
            "target": self.target,
            "description": self.description,
            "purpose": self.purpose,
            "mode": self.mode,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "revision": self.revision,
            "episode_revision": self.episode_revision,
            "check_resolution": self.check_resolution.model_dump(mode="json") if self.check_resolution else None,
            "delivery_status": dict(self.delivery_status) if self.delivery_status is not None else None,
        }


@dataclass(frozen=True, slots=True)
class ProspectiveMemorySummary:
    id: UUID
    title: str
    content: str
    importance: float
    updated_at: datetime

    def as_prompt_payload(self) -> dict[str, Any]:
        return {
            "id": str(self.id),
            "title": self.title,
            "content": self.content,
            "importance": self.importance,
            "updated_at": self.updated_at.isoformat(),
        }


@dataclass(frozen=True, slots=True)
class ProspectiveContext:
    visual_focus: VisualFocus | None
    conditional_reminders: tuple[JarvisReminder, ...]
    goals: tuple[ProspectiveMemorySummary, ...]
    commitments: tuple[str, ...]
    generated_at: datetime

    def as_prompt_payload(self) -> dict[str, Any]:
        return {
            "visual_focus": (
                self.visual_focus.as_prompt_payload()
                if self.visual_focus is not None
                else None
            ),
            "conditional_reminders": [
                {
                    "id": str(item.id),
                    "title": item.title,
                    "details": item.details,
                    "condition": item.condition,
                    "updated_at": item.updated_at.isoformat(),
                }
                for item in self.conditional_reminders
            ],
            "goals": [item.as_prompt_payload() for item in self.goals],
            "commitments": list(self.commitments),
            "generated_at": self.generated_at.isoformat(),
        }


@dataclass(frozen=True, slots=True)
class ProspectiveDeliveryEvent:
    id: UUID
    owner_id: str
    session_id: UUID | None
    visual_focus_id: UUID | None
    reminder_id: UUID | None
    claim_token: UUID
    occurrence_key: str
    message: str
    event_type: str
    evidence: Mapping[str, Any]
    provider_response_id: str | None
    created_at: datetime
    updated_at: datetime
