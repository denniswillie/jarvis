from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from math import isfinite
from typing import Any, Literal, Mapping
from uuid import UUID


PERCEPTION_JSON_CONTRACT_VERSION = 1
WORKING_MEMORY_JSON_CONTRACT_VERSION = 1
_WORKING_MEMORY_VERSION_KEY = "_contract_version"

ObservationSource = Literal["glasses", "iphone_camera"]
ObservationFrameRole = Literal["context", "primary", "key_candidate"]


@dataclass(frozen=True)
class ObservationEntity:
    name: str
    category: str | None
    confidence: float
    attributes: Mapping[str, Any]

    def __post_init__(self) -> None:
        _require_text(self.name, "entity name")
        _require_confidence(self.confidence)


@dataclass(frozen=True)
class VisibleTextObservation:
    text: str
    context: str | None
    confidence: float

    def __post_init__(self) -> None:
        _require_text(self.text, "visible text")
        _require_confidence(self.confidence)


@dataclass(frozen=True)
class ObservationStateChange:
    subject: str
    attribute: str
    before: Any
    after: Any
    confidence: float

    def __post_init__(self) -> None:
        _require_text(self.subject, "state-change subject")
        _require_text(self.attribute, "state-change attribute")
        _require_confidence(self.confidence)


@dataclass(frozen=True)
class KeyframeCandidate:
    visual_frame_id: UUID
    reason: str
    confidence: float

    def __post_init__(self) -> None:
        _require_text(self.reason, "keyframe reason")
        _require_confidence(self.confidence)


@dataclass(frozen=True)
class ObservationFrameLink:
    visual_frame_id: UUID
    ordinal: int
    role: ObservationFrameRole

    def __post_init__(self) -> None:
        if self.ordinal < 0:
            raise ValueError("observation frame ordinal cannot be negative")
        if self.role not in {"context", "primary", "key_candidate"}:
            raise ValueError(f"unsupported observation frame role: {self.role}")


@dataclass(frozen=True)
class VisualObservation:
    id: UUID
    owner_id: str
    session_id: UUID
    source: ObservationSource
    started_at: datetime
    ended_at: datetime
    primary_visual_frame_id: UUID | None
    summary: str
    scene: str | None
    activity: str | None
    entities: tuple[ObservationEntity, ...]
    visible_text: tuple[VisibleTextObservation, ...]
    state_changes: tuple[ObservationStateChange, ...]
    memory_tags: tuple[str, ...]
    keyframe_candidates: tuple[KeyframeCandidate, ...]
    interrupt_candidate: Mapping[str, Any] | None
    starts_new_event: bool
    boundary_reason: str | None
    confidence: float
    analyzer_provider: str
    analyzer_model: str
    analyzer_contract_version: int
    compacted_at: datetime | None
    created_at: datetime
    scene_confidence: float | None = None
    activity_confidence: float | None = None

    def __post_init__(self) -> None:
        _require_text(self.owner_id, "owner id")
        _require_text(self.summary, "observation summary")
        _require_text(self.analyzer_provider, "analyzer provider")
        _require_text(self.analyzer_model, "analyzer model")
        _require_confidence(self.confidence)
        if self.scene_confidence is not None:
            _require_confidence(self.scene_confidence)
        if self.activity_confidence is not None:
            _require_confidence(self.activity_confidence)
        if self.source not in {"glasses", "iphone_camera"}:
            raise ValueError(f"unsupported observation source: {self.source}")
        if self.ended_at < self.started_at:
            raise ValueError("observation ended_at cannot be before started_at")
        if self.analyzer_contract_version < 1:
            raise ValueError("analyzer_contract_version must be positive")
        if self.compacted_at is not None and (
            self.entities
            or self.visible_text
            or self.state_changes
            or self.keyframe_candidates
            or self.interrupt_candidate is not None
        ):
            raise ValueError("compacted observations cannot retain bulky structured detail")

    @property
    def is_compacted(self) -> bool:
        return self.compacted_at is not None


@dataclass(frozen=True)
class SessionWorkingMemoryState:
    session_id: UUID
    owner_id: str
    revision: int
    last_observation_id: UUID | None
    last_observation_ended_at: datetime | None
    state: Mapping[str, Any]
    expires_at: datetime
    created_at: datetime
    updated_at: datetime
    contract_version: int = WORKING_MEMORY_JSON_CONTRACT_VERSION


@dataclass(frozen=True)
class OwnerWorkingMemoryState:
    owner_id: str
    revision: int
    state: Mapping[str, Any]
    expires_at: datetime | None
    created_at: datetime
    updated_at: datetime
    contract_version: int = WORKING_MEMORY_JSON_CONTRACT_VERSION


WorkingMemoryState = SessionWorkingMemoryState | OwnerWorkingMemoryState


@dataclass(frozen=True)
class WorkingMemoryWriteResult:
    applied: bool
    state: WorkingMemoryState


def serialize_entities(values: tuple[ObservationEntity, ...]) -> list[dict[str, Any]]:
    return [
        {
            "name": value.name,
            "category": value.category,
            "confidence": value.confidence,
            "attributes": dict(value.attributes),
        }
        for value in values
    ]


def deserialize_entities(value: object) -> tuple[ObservationEntity, ...]:
    return tuple(
        ObservationEntity(
            name=_required_string(item, "name"),
            category=_optional_string(item.get("category")),
            confidence=_number(item, "confidence"),
            attributes=_mapping(item.get("attributes")),
        )
        for item in _object_array(value)
    )


def serialize_visible_text(values: tuple[VisibleTextObservation, ...]) -> list[dict[str, Any]]:
    return [
        {"text": value.text, "context": value.context, "confidence": value.confidence}
        for value in values
    ]


def deserialize_visible_text(value: object) -> tuple[VisibleTextObservation, ...]:
    return tuple(
        VisibleTextObservation(
            text=_required_string(item, "text"),
            context=_optional_string(item.get("context")),
            confidence=_number(item, "confidence"),
        )
        for item in _object_array(value)
    )


def serialize_state_changes(values: tuple[ObservationStateChange, ...]) -> list[dict[str, Any]]:
    return [
        {
            "subject": value.subject,
            "attribute": value.attribute,
            "before": value.before,
            "after": value.after,
            "confidence": value.confidence,
        }
        for value in values
    ]


def deserialize_state_changes(value: object) -> tuple[ObservationStateChange, ...]:
    return tuple(
        ObservationStateChange(
            subject=_required_string(item, "subject"),
            attribute=_required_string(item, "attribute"),
            before=item.get("before"),
            after=item.get("after"),
            confidence=_number(item, "confidence"),
        )
        for item in _object_array(value)
    )


def serialize_keyframe_candidates(
    values: tuple[KeyframeCandidate, ...],
) -> list[dict[str, Any]]:
    return [
        {
            "visual_frame_id": str(value.visual_frame_id),
            "reason": value.reason,
            "confidence": value.confidence,
        }
        for value in values
    ]


def deserialize_keyframe_candidates(value: object) -> tuple[KeyframeCandidate, ...]:
    return tuple(
        KeyframeCandidate(
            visual_frame_id=UUID(_required_string(item, "visual_frame_id")),
            reason=_required_string(item, "reason"),
            confidence=_number(item, "confidence"),
        )
        for item in _object_array(value)
    )


def serialize_working_memory_state(
    state: Mapping[str, Any],
    *,
    contract_version: int = WORKING_MEMORY_JSON_CONTRACT_VERSION,
) -> dict[str, Any]:
    if contract_version < 1:
        raise ValueError("working-memory contract version must be positive")
    payload = dict(state)
    if _WORKING_MEMORY_VERSION_KEY in payload:
        raise ValueError(f"{_WORKING_MEMORY_VERSION_KEY} is reserved")
    return {_WORKING_MEMORY_VERSION_KEY: contract_version, **payload}


def deserialize_working_memory_state(value: object) -> tuple[int, dict[str, Any]]:
    payload = _mapping(value)
    raw_version = payload.pop(_WORKING_MEMORY_VERSION_KEY, WORKING_MEMORY_JSON_CONTRACT_VERSION)
    if not isinstance(raw_version, int) or isinstance(raw_version, bool) or raw_version < 1:
        raise ValueError("working-memory contract version must be a positive integer")
    return raw_version, payload


def _require_text(value: str, label: str) -> None:
    if not value.strip():
        raise ValueError(f"{label} cannot be blank")


def _require_confidence(value: float) -> None:
    if not isfinite(value) or value < 0 or value > 1:
        raise ValueError("confidence must be between zero and one")


def _object_array(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise TypeError("expected a JSON array")
    return [_mapping(item) for item in value]


def _mapping(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("expected a JSON object")
    return dict(value)


def _required_string(value: Mapping[str, Any], key: str) -> str:
    raw = value.get(key)
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"{key} must be a non-blank string")
    return raw


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError("expected a string or null")
    return value


def _number(value: Mapping[str, Any], key: str) -> float:
    raw = value.get(key)
    if not isinstance(raw, (int, float)) or isinstance(raw, bool):
        raise TypeError(f"{key} must be numeric")
    return float(raw)
