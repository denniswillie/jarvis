"""Typed episode commands for the existing responsibility owner, not a new store."""
from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import AwareDatetime, Field, model_serializer, model_validator

from app.live_state.models import ContractModel, Revision, SourceDependency
from app.live_state.perception import CheckResolution, ObservationText, RequestTargetSpan


class EpisodeCheckSpec(ContractModel):
    kind: Literal["episode_check"] = "episode_check"
    title: str = Field(min_length=1, max_length=200)
    details: str = Field(default="", max_length=2000)
    basis: Literal["explicit_request"] = "explicit_request"


class VisualResolutionPolicy(ContractModel):
    """Explicit opt-in, versioned by the owning occurrence's check_revision.

    schema_version versions interpretation, not a model-invented owner counter.
    A condition/policy edit advances the existing check revision atomically.
    """
    schema_version: Literal[1] = 1
    authorization: Literal["explicit_user_request"]
    condition: str = Field(min_length=1, max_length=1500)
    minimum_confidence: float = Field(default=.95, ge=.9, le=1)
    max_evidence_age_seconds: int = Field(default=20, ge=1, le=30, strict=True)


class EpisodeCheckDraft(ContractModel):
    question: ObservationText
    origin: Literal["explicit_request"] = "explicit_request"
    object_association: str | None = Field(default=None, min_length=1, max_length=200)
    activation_cues: tuple[ObservationText, ...] = Field(default=(), max_length=3)
    observation_window_expires_at: AwareDatetime | None = None
    dependencies: tuple[SourceDependency, ...] = Field(default=(), max_length=16)
    # Original inferred cue provenance is retained, but grants no authorization.
    cue_origin: Literal["explicit_request", "policy_permitted_inference"] | None = None
    visual_resolution: VisualResolutionPolicy | None = None
    target_spans: tuple[RequestTargetSpan, ...] = Field(default=(), max_length=4)

    @model_serializer(mode="wrap")
    def preserve_legacy_fingerprints(self, handler):
        value = handler(self)
        if self.visual_resolution is None:
            value.pop("visual_resolution", None)
        if not self.target_spans:
            value.pop("target_spans", None)
        if getattr(self, "visual_request", None) is None:
            value.pop("visual_request", None)
        return value

    @model_validator(mode="after")
    def supported_dependencies(self):
        if any(d.source_namespace != "context_event" or d.generation is not None for d in self.dependencies):
            raise ValueError("Only canonical context-event dependencies are qualified by this owner.")
        if len({d.source_id for d in self.dependencies}) != len(self.dependencies):
            raise ValueError("Dependencies must have unique identities and versions.")
        if self.cue_origin is not None and not self.dependencies:
            raise ValueError("A prior cue requires its original evidence dependencies.")
        return self


class EpisodeCheckPayload(EpisodeCheckDraft):
    admission_request: SourceDependency
    last_request: SourceDependency
    episode_started_at: AwareDatetime
    snoozed_until: AwareDatetime | None = None
    # Server-only marker/reference. Never persist a second authoritative quote.
    # question/object_association remain the original model interpretation.
    visual_request: SourceDependency | None = None


class AdmitEpisodeCheck(ContractModel):
    spec: EpisodeCheckSpec
    check: EpisodeCheckDraft
    responsibility_id: UUID | None = None
    expected_revision: Revision | None = None
    request_slot: int = Field(default=1, ge=1, le=20, strict=True)

    @model_validator(mode="after")
    def paired_rule(self):
        if (self.responsibility_id is None) != (self.expected_revision is None):
            raise ValueError("A new episode of an existing rule needs its expected revision.")
        return self


class ChangeEpisodeCheck(ContractModel):
    responsibility_id: UUID
    occurrence_id: UUID
    expected_revision: Revision
    expected_check_revision: Revision
    episode_id: UUID
    expected_episode_revision: Revision
    operation: Literal["revise", "resolve", "cancel", "snooze", "expire"]
    check: EpisodeCheckDraft | None = None
    rebind_episode: bool = Field(default=False, strict=True)
    resolution: CheckResolution | None = None
    until: AwareDatetime | None = None
    request_slot: int = Field(default=1, ge=1, le=20, strict=True)

    @model_validator(mode="after")
    def exact_operation_shape(self):
        if (self.operation == "revise") != (self.check is not None):
            raise ValueError("Only revise supplies a replacement check.")
        if self.rebind_episode and self.operation != "revise":
            raise ValueError("Episode rebinding is an explicit revision, not a new episode.")
        if (self.operation == "resolve") != (self.resolution is not None):
            raise ValueError("Only resolve supplies evidence-based resolution.")
        if (self.operation == "snooze") != (self.until is not None):
            raise ValueError("Only snooze supplies until.")
        return self
