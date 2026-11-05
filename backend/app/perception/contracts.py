from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal, Mapping
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from app.live_state.models import GuardedPayloadBinding
from app.live_state.perception import MAX_QUESTIONS, ObservationalQuestion, PerceptionBrief, PerceptionFinding


FAST_PERCEPTION_CONTRACT_VERSION = 1


class FastPerceptionOutputError(ValueError):
    pass


@dataclass(frozen=True)
class FastPerceptionFrame:
    frame_id: UUID
    frame_sequence: int
    captured_at: datetime
    source: str
    content_type: str
    image_bytes: bytes


@dataclass(frozen=True)
class FastPerceptionTranscript:
    role: str
    content: str
    created_at: datetime


@dataclass(frozen=True)
class FastPerceptionRequest:
    owner_id: str
    session_id: UUID
    window_id: UUID
    frames: tuple[FastPerceptionFrame, ...]
    previous_working_state: Mapping[str, Any]
    existing_entity_ids: frozenset[str]
    recent_audio_transcripts: tuple[FastPerceptionTranscript, ...]
    behavior_instructions: str
    active_reminders: tuple[Mapping[str, Any], ...]
    active_focus: Any | None
    recent_interrupts: tuple[Mapping[str, Any], ...]
    requested_at: datetime
    voice_context: Mapping[str, Any] | None = None
    input_context_gaps: tuple[Mapping[str, Any], ...] = ()
    perception_brief: PerceptionBrief | None = None
    perception_brief_binding: GuardedPayloadBinding | None = None
    # Set only after canonical capture includes the exact brief and its parents.
    # This is an input fingerprint, not an authorization or a model output field.
    perception_brief_input_digest: str | None = None
    # Server-captured camera ownership, distinct from a reusable visual session
    # ID. Canonical input capture includes both before any provider work.
    camera_visual_session_id: UUID | None = None
    camera_connection_id: UUID | None = None

    def __post_init__(self) -> None:
        if (self.camera_visual_session_id is None) != (self.camera_connection_id is None):
            raise ValueError("camera ownership requires both session and connection identities")
        if self.camera_connection_id is not None and (
                not isinstance(self.camera_connection_id, UUID)
                or not isinstance(self.camera_visual_session_id, UUID)):
            raise ValueError("camera ownership identities must be UUIDs")
        if not self.owner_id.strip():
            raise ValueError("owner_id cannot be blank")
        if not self.frames and not any(item.content.strip() for item in self.recent_audio_transcripts):
            raise ValueError("fast perception requires frames or original dialogue")
        ordering = [(frame.captured_at, frame.frame_sequence) for frame in self.frames]
        if ordering != sorted(ordering):
            raise ValueError("fast perception frames must be ordered chronologically")
        if len({frame.frame_id for frame in self.frames}) != len(self.frames):
            raise ValueError("fast perception frame IDs must be unique")
        if any(not frame.image_bytes for frame in self.frames):
            raise ValueError("fast perception frames must contain image bytes")
        if any(frame.source not in {"glasses", "iphone_camera"} for frame in self.frames):
            raise ValueError("fast perception frames have an unsupported source")
        if (self.perception_brief is None) != (self.perception_brief_binding is None):
            raise ValueError("a perception brief requires its canonical payload binding")
        if self.perception_brief is None:
            if self.perception_brief_input_digest is not None:
                raise ValueError("an input receipt requires a perception brief")
        else:
            brief, binding = self.perception_brief, self.perception_brief_binding
            if ((brief.scope.owner_id, brief.scope.session_id) != (self.owner_id, self.session_id)
                    or binding.scope != brief.scope
                    or set(binding.dependencies) != set(brief.dependencies)):
                raise ValueError("perception assignment scope or canonical dependencies disagree")
            fingerprint = self.perception_brief_input_digest
            if fingerprint is not None and (
                    len(fingerprint) != 64 or any(char not in "0123456789abcdef" for char in fingerprint)):
                raise ValueError("invalid perception input fingerprint")

    @property
    def primary_frame(self) -> FastPerceptionFrame:
        if not self.frames:
            raise ValueError("dialogue-only requests use the multimodal analyzer, not a primary frame")
        return self.frames[-1]

    @property
    def allowed_frame_ids(self) -> frozenset[UUID]:
        return frozenset(frame.frame_id for frame in self.frames)


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ConfidenceLabel(_StrictModel):
    label: str = Field(min_length=1, max_length=240)
    confidence: float = Field(ge=0, le=1)


class FastPerceptionEntity(_StrictModel):
    existing_entity_id: str | None = Field(default=None, min_length=1, max_length=128)
    new_local_key: str | None = Field(
        default=None,
        min_length=1,
        max_length=64,
        pattern=r"^[a-z][a-z0-9_]*$",
    )
    type: str = Field(min_length=1, max_length=80)
    name: str = Field(min_length=1, max_length=160)
    state: str | None = Field(default=None, max_length=500)
    attributes: dict[str, Any] = Field(default_factory=dict)
    last_visible_frame_id: UUID
    confidence: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def require_exactly_one_identity(self) -> FastPerceptionEntity:
        if (self.existing_entity_id is None) == (self.new_local_key is None):
            raise ValueError(
                "entity requires exactly one of existing_entity_id or new_local_key"
            )
        return self


class FastVisibleText(_StrictModel):
    text: str = Field(min_length=1, max_length=1_000)
    context: str | None = Field(default=None, max_length=240)
    frame_id: UUID
    confidence: float = Field(ge=0, le=1)


class FastStateChange(_StrictModel):
    subject_local_key: str = Field(min_length=1, max_length=128)
    change: str = Field(min_length=1, max_length=500)
    confidence: float = Field(ge=0, le=1)


class FastKeyframeCandidate(_StrictModel):
    frame_id: UUID
    reason: str = Field(min_length=1, max_length=500)
    score: float = Field(ge=0, le=1)


class FastInterruptCandidate(_StrictModel):
    message: str = Field(min_length=1, max_length=500)
    reason: str = Field(min_length=1, max_length=500)
    confidence: float = Field(ge=0, le=1)
    # Old stored observations can omit this marker. Native active-watch results
    # must explicitly classify a generic alert before it can bypass tracked delivery.
    scope: Literal["general"] | None = None


class FastProspectiveMatch(_StrictModel):
    subject_type: Literal["visual_focus", "conditional_reminder"]
    subject_id: UUID
    confidence: float = Field(ge=0, le=1)
    observed_condition: str = Field(min_length=1, max_length=500)
    supporting_frame_ids: list[UUID] = Field(min_length=1, max_length=4)
    suggested_message: str = Field(min_length=1, max_length=500)


class FastPerceptionOutput(_StrictModel):
    summary: str = Field(min_length=1, max_length=2_000)
    scene: ConfidenceLabel | None = None
    activity: ConfidenceLabel | None = None
    entities: list[FastPerceptionEntity] = Field(default_factory=list, max_length=40)
    visible_text: list[FastVisibleText] = Field(default_factory=list, max_length=40)
    state_changes: list[FastStateChange] = Field(default_factory=list, max_length=30)
    memory_tags: list[str] = Field(default_factory=list, max_length=20)
    keyframe_candidates: list[FastKeyframeCandidate] = Field(
        default_factory=list,
        max_length=12,
    )
    starts_new_event: bool
    boundary_reason: str | None = Field(default=None, max_length=500)
    interrupt_candidate: FastInterruptCandidate | None = None
    prospective_matches: list[FastProspectiveMatch] = Field(
        default_factory=list,
        max_length=8,
    )
    # Observations only. The parser assigns exact question/check versions from
    # the fixed request; neither a finding nor visibility settles a check.
    perception_findings: list[PerceptionFinding] = Field(default_factory=list, max_length=MAX_QUESTIONS)
    confidence: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def validate_local_keys(self) -> FastPerceptionOutput:
        keys = [entity.new_local_key for entity in self.entities if entity.new_local_key]
        if len(keys) != len(set(keys)):
            raise ValueError("new_local_key values must be unique")
        return self


def parse_fast_perception_output(
    raw: str,
    *,
    request: FastPerceptionRequest,
) -> FastPerceptionOutput:
    try:
        output = FastPerceptionOutput.model_validate_json(_strip_json_fence(raw))
    except ValidationError as error:
        raise FastPerceptionOutputError(
            f"Fast perception output failed contract validation: {error}"
        ) from error

    supplied_frames = request.allowed_frame_ids
    referenced_frames = {
        entity.last_visible_frame_id for entity in output.entities
    } | {item.frame_id for item in output.visible_text} | {
        item.frame_id for item in output.keyframe_candidates
    }
    unknown_frames = referenced_frames - supplied_frames
    if unknown_frames:
        raise FastPerceptionOutputError(
            "Fast perception output referenced frames outside the supplied window."
        )

    prospective_frames = {
        frame_id
        for candidate in output.prospective_matches
        for frame_id in candidate.supporting_frame_ids
    }
    if prospective_frames - supplied_frames:
        raise FastPerceptionOutputError(
            "Fast perception prospective match referenced frames outside the supplied window."
        )

    allowed_subjects = _allowed_prospective_subjects(request)
    if any(
        (candidate.subject_type, candidate.subject_id) not in allowed_subjects
        for candidate in output.prospective_matches
    ):
        raise FastPerceptionOutputError(
            "Fast perception prospective match referenced a subject outside the supplied context."
        )

    questions = bound_perception_questions(request)
    seen_questions = set()
    for finding in output.perception_findings:
        validate_perception_finding(finding, request=request, questions=questions)
        if finding.question_id in seen_questions:
            raise FastPerceptionOutputError("duplicate_question_finding")
        seen_questions.add(finding.question_id)

    unknown_entities = {
        entity.existing_entity_id
        for entity in output.entities
        if entity.existing_entity_id is not None
        and entity.existing_entity_id not in request.existing_entity_ids
    }
    if unknown_entities:
        raise FastPerceptionOutputError(
            "Fast perception output referenced unknown existing entity IDs."
        )

    entity_references = {
        value
        for entity in output.entities
        for value in (entity.existing_entity_id, entity.new_local_key)
        if value is not None
    }
    if any(
        change.subject_local_key not in entity_references
        for change in output.state_changes
    ):
        raise FastPerceptionOutputError(
            "Fast perception state changes must reference an output entity."
        )

    normalized_tags: list[str] = []
    for value in output.memory_tags:
        tag = value.strip().lower()[:80]
        if tag and tag not in normalized_tags:
            normalized_tags.append(tag)
    return output.model_copy(update={"memory_tags": normalized_tags})


def bound_perception_questions(request: FastPerceptionRequest) -> dict[UUID, ObservationalQuestion]:
    """Validate assignment membership, not current source authority or freshness.

    Results can remain historical after their request becomes stale. Canonical
    publication/effect guards still own availability and current applicability.
    All observational references must belong to the captured brief's binding;
    a source UUID without its exact revision/generation is not a valid handle.
    """
    brief, binding = request.perception_brief, request.perception_brief_binding
    if brief is None:
        if binding is not None:
            raise FastPerceptionOutputError("perception_assignment_scope_mismatch")
        return {}
    if (binding is None or (brief.scope.owner_id, brief.scope.session_id) != (request.owner_id, request.session_id)
            or binding.scope != brief.scope or set(binding.dependencies) != set(brief.dependencies)):
        raise FastPerceptionOutputError("perception_assignment_scope_mismatch")
    dependencies = set(brief.dependencies)
    questions = {}
    checks = {}
    for question in brief.questions:
        if question.id in questions:
            raise FastPerceptionOutputError("duplicate_question_reference")
        for source in (question.check_ref, question.focus_ref, question.episode_or_object_ref):
            if source is not None and source not in dependencies:
                raise FastPerceptionOutputError("unbound_question_reference")
        if question.check_ref is not None:
            previous = checks.get(question.check_ref.source_id)
            if previous is not None and previous != question.check_ref:
                raise FastPerceptionOutputError("ambiguous_check_reference")
            checks[question.check_ref.source_id] = question.check_ref
        questions[question.id] = question
    return questions


def validate_perception_finding(finding: PerceptionFinding, *, request: FastPerceptionRequest,
                                questions: Mapping[UUID, ObservationalQuestion] | None = None) -> None:
    """Reusable strict checks for both native handle mapping and durable JSON."""
    if questions is None:
        questions = bound_perception_questions(request)
    question = questions.get(finding.question_id)
    if question is None or finding.check_ref != question.check_ref:
        raise FastPerceptionOutputError("finding_assignment_mismatch")
    if not set(finding.evidence_frame_ids) <= request.allowed_frame_ids:
        raise FastPerceptionOutputError("unknown_finding_frame_reference")
    if len(set(finding.evidence_frame_ids)) != len(finding.evidence_frame_ids):
        raise FastPerceptionOutputError("duplicate_finding_frame_reference")


def _strip_json_fence(raw: str) -> str:
    text = raw.strip()
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _allowed_prospective_subjects(
    request: FastPerceptionRequest,
) -> set[tuple[str, UUID]]:
    allowed: set[tuple[str, UUID]] = set()
    if isinstance(request.active_focus, Mapping):
        focus_id = _uuid_value(request.active_focus.get("id"))
        if focus_id is not None:
            allowed.add(("visual_focus", focus_id))
    for reminder in request.active_reminders:
        reminder_id = _uuid_value(reminder.get("id"))
        if reminder_id is not None:
            allowed.add(("conditional_reminder", reminder_id))
    return allowed


def _uuid_value(value: object) -> UUID | None:
    if isinstance(value, UUID):
        return value
    if not isinstance(value, str):
        return None
    try:
        return UUID(value)
    except ValueError:
        return None
