"""Opt-in contextual-perception contracts, not a lifecycle or persistence engine.

References are application-owned identities after request-local handle resolution.
SourceDependency keeps content revision and optional source generation distinct;
scope and camera epochs are not substitutes for either. Owners must still validate
current, relevant versions, source availability and inference policy before effects.
"""
from __future__ import annotations

from enum import Enum
from typing import Annotated, Literal
from uuid import UUID

from pydantic import AwareDatetime, Field, model_validator

from app.live_state.models import ContractModel, SessionScope, SourceDependency


MAX_QUESTIONS = 4
MAX_WINDOW_FRAMES = 4
MAX_SCENE_CHANGE_CUES = 3
ObservationText = Annotated[str, Field(min_length=1, max_length=1000)]
ShortLabel = Annotated[str, Field(min_length=1, max_length=160)]
Visibility = Literal["visible", "not_visible", "unclear"]


class CheckLifecycle(str, Enum):
    """Check tracking state, never the underlying responsibility's completion.

    DISMISSED may represent cancelled tracking, not merely a dismissed alert.
    Neither DISMISSED nor EXPIRED establishes that a real-world action occurred.
    """

    CANDIDATE = "candidate"
    TRACKING = "tracking"
    RESOLVED = "resolved"
    DISMISSED = "dismissed"
    EXPIRED = "expired"


class DurableResponsibilityCapability(ContractModel):
    """Current availability of the selected canonical owner, not its stored state."""

    owner_namespace: ShortLabel
    status: Literal["available", "disabled", "unavailable"]
    reason: ObservationText | None = None


class PersistentTrackingOutcome(ContractModel):
    """An admission receipt; only the owner can establish actual acceptance."""

    scope: SessionScope
    capability: DurableResponsibilityCapability
    status: Literal["accepted", "unavailable"]
    canonical_owner_ref: SourceDependency | None = None
    check_ref: SourceDependency | None = None

    @model_validator(mode="after")
    def require_available_owner_for_acceptance(self) -> PersistentTrackingOutcome:
        if self.canonical_owner_ref is not None:
            if self.canonical_owner_ref.source_namespace != self.capability.owner_namespace:
                raise ValueError("canonical owner must match the reported capability")
        if self.status == "accepted":
            if self.capability.status != "available":
                raise ValueError("an unavailable owner cannot promise persistent tracking")
            if self.canonical_owner_ref is None or self.check_ref is None:
                raise ValueError("accepted tracking requires canonical owner and check references")
        return self


class EvidenceReference(ContractModel):
    """Original evidence provenance; reuse or rehydration never refreshes its age."""

    source: SourceDependency
    captured_at: AwareDatetime
    episode_or_object_ref: SourceDependency | None = None
    frame_id: UUID | None = None
    visibility: Visibility | None = None

    @model_validator(mode="after")
    def bind_visual_visibility(self) -> EvidenceReference:
        if (self.frame_id is None) != (self.visibility is None):
            raise ValueError("visual evidence requires both a frame and its visibility finding")
        return self


class PriorEvidence(ContractModel):
    reference: EvidenceReference
    finding: ObservationText
    confidence: float = Field(ge=0, le=1)
    uncertainty: ObservationText | None = None


class RequestTargetSpan(ContractModel):
    """Unicode code-point offsets into the exact finalized request, not an entity ID."""

    start: int = Field(ge=0, le=1000, strict=True)
    end: int = Field(gt=0, le=1000, strict=True)

    @model_validator(mode="after")
    def ordered(self):
        if self.end <= self.start:
            raise ValueError("target span must be nonempty and ordered")
        return self


class VisualRequestGrounding(ContractModel):
    """Ephemeral owner projection. Any materialized copy must retain source guards.

    Literal support is only wording provenance, not intent, negation, coreference,
    an identified physical instance, or authorization to resolve or notify.
    """

    request_ref: SourceDependency
    request_session_id: UUID
    payload_id: UUID
    wording_sha256: str = Field(pattern="^[0-9a-f]{64}$")
    user_wording: ObservationText
    target_spans: tuple[RequestTargetSpan, ...] = Field(default=(), max_length=4)
    status: Literal["literal_supported", "unresolved"]
    reason: Literal["literal_span_validated", "model_target_not_supported", "no_target_span",
                    "invalid_target_span", "ambiguous_literal_match", "deictic_reference_unresolved",
                    "multiple_targets_unresolved"]
    clarification_needed: bool
    intent_verified: Literal[False] = False
    referent_resolved: Literal[False] = False

    @model_validator(mode="after")
    def source_bound(self):
        if self.request_ref.source_namespace != "context_event" or self.request_ref.generation is not None:
            raise ValueError("visual wording requires an exact canonical request revision")
        if any(span.end > len(self.user_wording) for span in self.target_spans):
            raise ValueError("target span exceeds the original wording")
        if self.clarification_needed != (self.status == "unresolved"):
            raise ValueError("unresolved target must remain explicit")
        if self.status == "literal_supported" and len(self.target_spans) != 1:
            raise ValueError("literal support requires one validated span")
        return self


class ObservationalQuestion(ContractModel):
    """A literal evidence question, not authorization to notify or settle a duty."""

    id: UUID
    question: ObservationText
    priority: Literal["explicit_inspection", "high", "normal", "low"] = "normal"
    check_ref: SourceDependency | None = None
    episode_or_object_ref: SourceDependency | None = None
    focus_ref: SourceDependency | None = None
    request_grounding: VisualRequestGrounding | None = None

    @model_validator(mode="after")
    def bind_check_context(self) -> ObservationalQuestion:
        if self.check_ref is not None and self.episode_or_object_ref is None:
            raise ValueError("check questions require a versioned episode or object")
        return self


class PerceptionActivity(ContractModel):
    label: ShortLabel
    stage: ShortLabel | None = None
    confidence: float = Field(ge=0, le=1)
    uncertainty: ObservationText | None = None


class PerceptionBrief(ContractModel):
    id: UUID
    version: int = Field(ge=1)
    scope: SessionScope
    scope_epoch: UUID
    # Diagnostic provenance only, never a global compare-and-swap acceptance gate.
    based_on_live_revision: int = Field(ge=0)
    activity: PerceptionActivity | None = None
    intent_ref: SourceDependency | None = None
    episode_or_object_ref: SourceDependency | None = None
    questions: tuple[ObservationalQuestion, ...] = Field(default=(), max_length=MAX_QUESTIONS)
    prior_evidence: tuple[PriorEvidence, ...] = Field(default=(), max_length=8)
    scene_change_cues: tuple[ObservationText, ...] = Field(default=(), max_length=MAX_SCENE_CHANGE_CUES)
    dependencies: tuple[SourceDependency, ...] = Field(default=(), max_length=16)
    durable_responsibility: DurableResponsibilityCapability
    created_at: AwareDatetime
    fresh_until: AwareDatetime
    expires_at: AwareDatetime

    @model_validator(mode="after")
    def validate_assignment(self) -> PerceptionBrief:
        if not self.created_at <= self.fresh_until <= self.expires_at:
            raise ValueError("brief times must satisfy created_at <= fresh_until <= expires_at")
        if len({question.id for question in self.questions}) != len(self.questions):
            raise ValueError("brief question identities must be unique")
        if any(item.reference.captured_at > self.created_at for item in self.prior_evidence):
            raise ValueError("prior evidence cannot postdate brief creation")
        return self


class PerceptionFrameReference(ContractModel):
    frame_id: UUID
    frame_sequence: int = Field(ge=0)
    captured_at: AwareDatetime
    source: SourceDependency | None = None


class PerceptionFinding(ContractModel):
    question_id: UUID
    check_ref: SourceDependency | None = None
    visibility: Visibility
    literal_finding: ObservationText
    confidence: float = Field(ge=0, le=1)
    evidence_frame_ids: tuple[UUID, ...] = Field(min_length=1, max_length=MAX_WINDOW_FRAMES)


class PerceptionEvidence(ContractModel):
    """Capture one immutable brief and actual window, including relevant versions.

    A late result can remain historical; passing this schema does not make it fresh
    or authorize lifecycle effects. Current acceptance is the owning subsystem's
    decision, not equality with based_on_live_revision or the newest whole brief.
    """

    brief: PerceptionBrief
    window_id: UUID
    camera_stream_epoch: UUID
    frames: tuple[PerceptionFrameReference, ...] = Field(min_length=1, max_length=MAX_WINDOW_FRAMES)
    requested_at: AwareDatetime
    analyzed_at: AwareDatetime
    findings: tuple[PerceptionFinding, ...] = Field(default=(), max_length=MAX_QUESTIONS)
    meaningful_scene_or_activity_changes: tuple[ObservationText, ...] = Field(
        default=(), max_length=MAX_SCENE_CHANGE_CUES,
    )
    input_gaps: tuple[ObservationText, ...] = Field(default=(), max_length=8)

    @model_validator(mode="after")
    def validate_window_and_references(self) -> PerceptionEvidence:
        if not self.brief.created_at <= self.requested_at <= self.analyzed_at:
            raise ValueError("analysis must follow its fixed brief and request")
        frame_ids = {frame.frame_id for frame in self.frames}
        if len(frame_ids) != len(self.frames):
            raise ValueError("window frame identities must be unique")
        order = tuple((frame.captured_at, frame.frame_sequence) for frame in self.frames)
        if order != tuple(sorted(order)):
            raise ValueError("window frames must be chronological")
        if any(frame.captured_at > self.requested_at for frame in self.frames):
            raise ValueError("a supplied frame cannot postdate its request")
        questions = {question.id: question for question in self.brief.questions}
        if len({finding.question_id for finding in self.findings}) != len(self.findings):
            raise ValueError("return at most one finding per assigned question")
        for finding in self.findings:
            question = questions.get(finding.question_id)
            if question is None or finding.check_ref != question.check_ref:
                raise ValueError("findings must retain their assigned question and check version")
            if not set(finding.evidence_frame_ids).issubset(frame_ids):
                raise ValueError("findings may cite only frames supplied in this window")
            if len(set(finding.evidence_frame_ids)) != len(finding.evidence_frame_ids):
                raise ValueError("finding frame references must be unique")
        return self


class CheckResolution(ContractModel):
    """Owner-attributed resolution evidence, not a model-issued completion command."""

    basis: Literal["user_report", "visual_observation", "trusted_source"]
    source_refs: tuple[EvidenceReference, ...] = Field(min_length=1, max_length=8)
    resolved_at: AwareDatetime
    applicable_episode_or_object_ref: SourceDependency

    @model_validator(mode="after")
    def validate_resolution_evidence(self) -> CheckResolution:
        for reference in self.source_refs:
            if reference.episode_or_object_ref != self.applicable_episode_or_object_ref:
                raise ValueError("resolution evidence must belong to the applicable episode or object version")
            if reference.captured_at > self.resolved_at:
                raise ValueError("resolution evidence cannot postdate resolution")
        if self.basis == "visual_observation":
            visual = tuple(reference for reference in self.source_refs if reference.frame_id is not None)
            if not visual or any(reference.visibility != "visible" for reference in visual):
                raise ValueError("visual resolution requires visible evidence, not unclear or out-of-view findings")
        return self


class SituationalCheck(ContractModel):
    """A projection of tracking, not a second responsibility record or scheduler.

    Durable projections may remain TRACKING while their owner is unavailable.
    That preserves canonical state without issuing a PersistentTrackingOutcome.
    Expiry, snooze, dismissal and relevance changes never synthesize resolution.
    """

    id: UUID
    revision: int = Field(ge=1)
    scope: SessionScope
    episode_or_object_ref: SourceDependency
    origin: Literal["explicit_request", "policy_permitted_inference"]
    continuity: Literal["ephemeral", "durable"] = "ephemeral"
    canonical_owner_ref: SourceDependency | None = None
    question: ObservationText
    request_grounding: VisualRequestGrounding | None = None
    activation_cues: tuple[ObservationText, ...] = Field(default=(), max_length=3)
    dependency_versions: tuple[SourceDependency, ...] = Field(default=(), max_length=16)
    lifecycle: CheckLifecycle = CheckLifecycle.CANDIDATE
    relevance_now: bool = False
    evidence_availability: Literal["available", "unavailable", "unknown"] = "unknown"
    evidence_refs: tuple[EvidenceReference, ...] = Field(default=(), max_length=8)
    last_observed_at: AwareDatetime | None = None
    observation_window_expires_at: AwareDatetime | None = None
    snoozed_until: AwareDatetime | None = None
    resolution: CheckResolution | None = None

    @model_validator(mode="after")
    def validate_projection(self) -> SituationalCheck:
        if self.continuity == "durable" and self.canonical_owner_ref is None:
            raise ValueError("durable continuity requires a canonical owner reference")
        if (self.lifecycle == CheckLifecycle.RESOLVED) != (self.resolution is not None):
            raise ValueError("only resolved checks carry resolution, and resolution requires explicit evidence")
        if self.resolution is not None:
            if self.resolution.applicable_episode_or_object_ref != self.episode_or_object_ref:
                raise ValueError("resolution cannot settle another episode or object version")
        if self.last_observed_at is not None and self.observation_window_expires_at is not None:
            if self.observation_window_expires_at < self.last_observed_at:
                raise ValueError("an observation window cannot expire before its last observation")
        return self


class FutureCue(ContractModel):
    """A bounded relevance cue, never proof of an obligation or a recurring duty."""

    id: UUID
    revision: int = Field(ge=1)
    scope: SessionScope
    episode_or_object_ref: SourceDependency
    question: ObservationText
    activation_cues: tuple[ObservationText, ...] = Field(min_length=1, max_length=3)
    check_ref: SourceDependency | None = None
    dependencies: tuple[SourceDependency, ...] = Field(default=(), max_length=16)
    created_at: AwareDatetime
    expires_at: AwareDatetime

    @model_validator(mode="after")
    def validate_lifetime(self) -> FutureCue:
        if self.expires_at <= self.created_at:
            raise ValueError("a future cue must expire after creation")
        return self
