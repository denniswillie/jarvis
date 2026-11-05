from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal, Mapping, Sequence, get_args
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from app.experience.models import ExperienceEvent, ExperienceEvidenceType
from app.memory.consolidation.evidence import CitationCatalog, build_citation_catalog


MAX_ACTIONS = 16
MAX_ENTITY_CANDIDATES = 24
MAX_VISIBLE_TEXT_ITEMS = 24
MAX_SEMANTIC_FACT_CANDIDATES = 20
MAX_RELATION_CANDIDATES = 20
MAX_REFLECTION_CANDIDATES = 8
MAX_PROCEDURE_CANDIDATES = 8
MAX_PROCEDURE_STEPS = 16

SemanticPredicate = Literal[
    "is",
    "has",
    "owns",
    "uses",
    "prefers",
    "likes",
    "dislikes",
    "located_at",
    "works_at",
    "lives_at",
    "part_of",
]
SEMANTIC_PREDICATE_VALUES = frozenset(get_args(SemanticPredicate))
EXCLUSIVE_SEMANTIC_PREDICATES = frozenset({"located_at"})
MULTI_VALUE_SEMANTIC_PREDICATES = (
    SEMANTIC_PREDICATE_VALUES - EXCLUSIVE_SEMANTIC_PREDICATES
)
FactChangeIntent = Literal["assert", "replace", "retract", "transition"]
FactTemporalScope = Literal["durable", "event", "state"]
RelationPredicate = Literal[
    "owns",
    "located_at",
    "uses",
    "with",
    "part_of",
    "works_at",
    "lives_at",
    "related_to",
]


CitationErrorCode = Literal[
    "invalid_contract", "invalid_reference", "uncitable_source",
    "missing_literal_quotation", "invalid_explicit_user_attribution", "invalid_keyframe",
    "unsupported_evidence_basis",
    "invalid_state_transition",
]


class ConsolidationValidationError(ValueError):
    def __init__(
        self, message: str, *, code: CitationErrorCode = "invalid_contract",
        location: str = "result", evidence_type: str | None = None,
        evidence_ref_id: UUID | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.location = location
        self.evidence_type = evidence_type
        self.evidence_ref_id = evidence_ref_id
        self.model_attempt: int | None = None

    def safe_diagnostic(self) -> str:
        # Deliberately excludes raw model output, quotation text, and source payloads.
        attempt = f"; model_generation={self.model_attempt}" if self.model_attempt else ""
        return f"{self.code} at {self.location}{attempt}"


@dataclass(frozen=True, slots=True)
class ConsolidationEvidenceInput:
    evidence_id: UUID | None
    evidence_type: ExperienceEvidenceType
    evidence_ref_id: UUID
    occurred_at: datetime
    source_sequence: int | None
    payload: Mapping[str, Any]
    citation_allowed: bool = True


@dataclass(frozen=True, slots=True)
class ConsolidationKeyframeInput:
    visual_frame_id: UUID
    rank: int
    reason: str
    quality_score: float
    captured_at: datetime
    frame_description: str | None
    storage_bucket: str
    storage_path: str
    content_type: str
    image_bytes: bytes | None


@dataclass(frozen=True, slots=True)
class ConsolidationRequest:
    event: ExperienceEvent
    evidence: tuple[ConsolidationEvidenceInput, ...]
    context_messages: tuple[ConsolidationEvidenceInput, ...]
    keyframes: tuple[ConsolidationKeyframeInput, ...]
    event_end_working_state: Mapping[str, Any]
    owner_working_memory: Mapping[str, Any]
    active_reminders: tuple[Mapping[str, Any], ...]
    semantic_fact_context: tuple[Mapping[str, Any], ...] = ()
    semantic_fact_context_truncated: bool = False


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class EvidenceCitation(_StrictModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=False)

    evidence_type: Literal["observation", "message", "tool_execution"]
    evidence_ref_id: str = Field(min_length=1, max_length=64)
    supporting_text: str = Field(min_length=1, max_length=500)


class EntityCandidate(_StrictModel):
    name: str = Field(min_length=1, max_length=160)
    category: str | None
    confidence: float = Field(ge=0, le=1)
    evidence: list[EvidenceCitation] = Field(min_length=1, max_length=8)


class VisibleTextCandidate(_StrictModel):
    text: str = Field(min_length=1, max_length=1000)
    context: str | None
    confidence: float = Field(ge=0, le=1)
    evidence: list[EvidenceCitation] = Field(min_length=1, max_length=8)


class SemanticFactCandidate(_StrictModel):
    subject: str = Field(min_length=1, max_length=160)
    predicate: SemanticPredicate
    value: str = Field(min_length=1, max_length=1000)
    confidence: float = Field(ge=0, le=1)
    explicitly_user_stated: bool
    change_intent: FactChangeIntent = "assert"
    temporal_scope: FactTemporalScope = "durable"
    previous_value: str | None = Field(default=None, min_length=1, max_length=1000)
    valid_from_hint: str | None
    valid_to_hint: str | None
    evidence: list[EvidenceCitation] = Field(min_length=1, max_length=8)

    @property
    def source_classification(self) -> Literal["user_statement", "inferred_observation"]:
        return "user_statement" if self.explicitly_user_stated else "inferred_observation"

    @model_validator(mode="after")
    def validate_change_intent(self) -> SemanticFactCandidate:
        if self.change_intent in {"replace", "retract"} and not self.explicitly_user_stated:
            raise ValueError("replacement and retraction must be explicitly user stated")
        if self.change_intent not in {"replace", "transition"} and self.previous_value is not None:
            raise ValueError("previous_value is only valid for replacement or transition")
        if self.temporal_scope == "event" and self.change_intent != "assert":
            raise ValueError("event-only observations cannot replace persistent state")
        if self.change_intent == "transition" and (
            self.temporal_scope != "state" or self.previous_value is None
        ):
            raise ValueError("state transitions require state scope and an exact previous_value")
        if (
            self.change_intent == "replace"
            and self.predicate in MULTI_VALUE_SEMANTIC_PREDICATES
            and self.previous_value is None
        ):
            raise ValueError("multi-valued replacement requires previous_value")
        return self


class RelationCandidate(_StrictModel):
    source_entity: str = Field(min_length=1, max_length=160)
    predicate: RelationPredicate
    target_entity: str = Field(min_length=1, max_length=160)
    confidence: float = Field(ge=0, le=1)
    explicitly_user_stated: bool
    temporal_scope: FactTemporalScope = "durable"
    valid_from_hint: str | None
    valid_to_hint: str | None
    evidence: list[EvidenceCitation] = Field(min_length=1, max_length=8)

    @property
    def source_classification(self) -> Literal["user_statement", "inferred_observation"]:
        return "user_statement" if self.explicitly_user_stated else "inferred_observation"


class ReflectionCandidate(_StrictModel):
    content: str = Field(min_length=1, max_length=1500)
    confidence: float = Field(ge=0, le=1)
    evidence: list[EvidenceCitation] = Field(min_length=1, max_length=12)


class ProcedureCandidate(_StrictModel):
    title: str = Field(min_length=1, max_length=200)
    steps: list[str] = Field(min_length=1, max_length=MAX_PROCEDURE_STEPS)
    confidence: float = Field(ge=0, le=1)
    explicitly_user_stated: bool
    evidence: list[EvidenceCitation] = Field(min_length=1, max_length=12)


class TaskOutcomeCandidate(_StrictModel):
    task: str = Field(min_length=1, max_length=240)
    state: Literal["in_progress", "paused", "abandoned", "completed", "unknown"]
    evidence_basis: Literal["user_report", "observation", "tool_result", "uncertain"]
    details: str = Field(min_length=1, max_length=1000)
    evidence: list[EvidenceCitation] = Field(min_length=1, max_length=8)

    @model_validator(mode="after")
    def completion_requires_a_basis(self) -> TaskOutcomeCandidate:
        if self.state == "completed" and self.evidence_basis == "uncertain":
            raise ValueError("uncertain evidence cannot establish task completion")
        return self


class ConsolidationResult(_StrictModel):
    # Default only for reading previously stored results. Strict native output
    # includes this field, even when no task outcome can be established.
    task_outcomes: list[TaskOutcomeCandidate] = Field(default_factory=list, max_length=8)
    event_title: str = Field(min_length=1, max_length=240)
    event_summary: str = Field(min_length=1, max_length=3000)
    location: str | None
    actions: list[str] = Field(max_length=MAX_ACTIONS)
    entity_candidates: list[EntityCandidate] = Field(max_length=MAX_ENTITY_CANDIDATES)
    visible_text: list[VisibleTextCandidate] = Field(max_length=MAX_VISIBLE_TEXT_ITEMS)
    episode_importance: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)
    should_promote_to_episode: bool
    promotion_reason: str = Field(min_length=1, max_length=1000)
    selected_keyframe_ids: list[str] = Field(max_length=6)
    semantic_fact_candidates: list[SemanticFactCandidate] = Field(
        max_length=MAX_SEMANTIC_FACT_CANDIDATES
    )
    relation_candidates: list[RelationCandidate] = Field(
        max_length=MAX_RELATION_CANDIDATES
    )
    reflection_candidates: list[ReflectionCandidate] = Field(
        max_length=MAX_REFLECTION_CANDIDATES
    )
    procedure_candidates: list[ProcedureCandidate] = Field(
        max_length=MAX_PROCEDURE_CANDIDATES
    )


def parse_consolidation_result(
    value: str | Mapping[str, Any],
    *,
    request: ConsolidationRequest,
) -> ConsolidationResult:
    try:
        result = (
            ConsolidationResult.model_validate_json(value)
            if isinstance(value, str)
            else ConsolidationResult.model_validate(value)
        )
    except (ValidationError, ValueError, TypeError) as error:
        raise ConsolidationValidationError(
            "consolidation output does not match the required contract"
        ) from error
    validate_consolidation_result(result, request=request)
    return result


def validate_consolidation_result(
    result: ConsolidationResult,
    *,
    request: ConsolidationRequest,
) -> None:
    available_keyframes = {item.visual_frame_id for item in request.keyframes}
    selected: list[UUID] = []
    for index, raw_id in enumerate(result.selected_keyframe_ids):
        location = f"selected_keyframe_ids[{index}]"
        try:
            frame_id = UUID(raw_id)
        except ValueError as error:
            raise ConsolidationValidationError(
                "selected keyframe id is not a UUID",
                code="invalid_keyframe", location=location,
            ) from error
        if frame_id not in available_keyframes:
            raise ConsolidationValidationError(
                "selected keyframe does not belong to the event",
                code="invalid_keyframe", location=location,
            )
        selected.append(frame_id)
    if len(selected) != len(set(selected)):
        raise ConsolidationValidationError(
            "selected keyframe ids must be unique",
            code="invalid_keyframe", location="selected_keyframe_ids",
        )

    catalog = build_citation_catalog(request)
    uncitable = {
        (item.evidence_type, item.evidence_ref_id)
        for item in (*request.evidence, *request.context_messages)
        if (item.evidence_type, item.evidence_ref_id) not in catalog
    }
    for group_name in (
        "entity_candidates", "visible_text", "semantic_fact_candidates",
        "relation_candidates", "reflection_candidates", "procedure_candidates", "task_outcomes",
    ):
        for index, candidate in enumerate(getattr(result, group_name)):
            location = f"{group_name}[{index}]"
            _validate_citations(
                candidate.evidence, catalog=catalog, uncitable=uncitable,
                location=f"{location}.evidence",
            )
            if group_name == "visible_text" and not any(
                citation.evidence_type == "observation"
                and catalog[("observation", UUID(citation.evidence_ref_id))].contains_quote(candidate.text)
                for citation in candidate.evidence
            ):
                raise ConsolidationValidationError(
                    "visible text requires literal text from cited observation evidence",
                    code="unsupported_evidence_basis", location=location,
                )
            if group_name == "semantic_fact_candidates" and candidate.change_intent == "transition":
                def normalized(value: object) -> str:
                    return " ".join(str(value or "").casefold().split())

                if not any(
                    normalized(item.get("subject")) == normalized(candidate.subject)
                    and item.get("predicate") == candidate.predicate
                    and normalized(item.get("value")) == normalized(candidate.previous_value)
                    and item.get("temporal_scope") != "event"
                    for item in request.semantic_fact_context
                ):
                    raise ConsolidationValidationError(
                        "state transition must target a supplied prior subject, predicate, and value",
                        code="invalid_state_transition", location=location,
                    )
            if getattr(candidate, "explicitly_user_stated", False) and not (
                _has_user_message_citation(candidate.evidence, catalog=catalog)
            ):
                raise ConsolidationValidationError(
                    "explicit user attribution must cite an attached user message",
                    code="invalid_explicit_user_attribution",
                    location=f"{location}.explicitly_user_stated",
                )
            if group_name in {"semantic_fact_candidates", "relation_candidates", "task_outcomes"}:
                sources = [
                    catalog[(citation.evidence_type, UUID(citation.evidence_ref_id))].source
                    for citation in candidate.evidence
                ]
                basis = getattr(candidate, "evidence_basis", None)
                if not any(_supports_evidence_basis(source, basis=basis) for source in sources):
                    raise ConsolidationValidationError(
                        "candidate requires matching non-assistant evidence",
                        code="unsupported_evidence_basis", location=location,
                    )


def consolidation_result_payload(result: ConsolidationResult) -> dict[str, Any]:
    return result.model_dump(mode="json")


def _supports_evidence_basis(
    source: ConsolidationEvidenceInput, *, basis: str | None,
) -> bool:
    if source.evidence_type == "observation":
        return basis in {None, "observation", "uncertain"}
    if source.evidence_type == "message":
        return (
            source.payload.get("role") == "user"
            and source.payload.get("modality") in {"audio", "text"}
            and basis in {None, "user_report", "uncertain"}
        )
    if source.evidence_type == "tool_execution":
        result = source.payload.get("result")
        return (
            source.payload.get("status") == "completed"
            and isinstance(result, Mapping)
            and result.get("status") == "success"
            and basis in {None, "tool_result", "uncertain"}
        )
    return False


def _validate_citations(
    citations: Sequence[EvidenceCitation],
    *,
    catalog: CitationCatalog,
    uncitable: set[tuple[str, UUID]],
    location: str,
) -> None:
    for index, citation in enumerate(citations):
        citation_location = f"{location}[{index}]"
        try:
            reference_id = UUID(citation.evidence_ref_id)
        except ValueError as error:
            raise ConsolidationValidationError(
                "citation id is not a UUID", code="invalid_reference",
                location=citation_location,
            ) from error
        key = (citation.evidence_type, reference_id)
        source = catalog.get(key)
        if source is None:
            raise ConsolidationValidationError(
                "citation does not reference attached event evidence",
                code="uncitable_source" if key in uncitable else "invalid_reference",
                location=citation_location, evidence_type=citation.evidence_type,
                evidence_ref_id=reference_id,
            )
        if not source.contains_quote(citation.supporting_text):
            raise ConsolidationValidationError(
                "citation supporting text is not present in its evidence",
                code="missing_literal_quotation", location=citation_location,
                evidence_type=citation.evidence_type, evidence_ref_id=reference_id,
            )


def _has_user_message_citation(
    citations: Sequence[EvidenceCitation],
    *,
    catalog: CitationCatalog,
) -> bool:
    for citation in citations:
        if citation.evidence_type != "message":
            continue
        try:
            reference_id = UUID(citation.evidence_ref_id)
        except ValueError:
            continue
        entry = catalog.get(("message", reference_id))
        if (
            entry is not None
            and entry.source.payload.get("role") == "user"
            and entry.source.payload.get("modality") in {"audio", "text"}
        ):
            return True
    return False
