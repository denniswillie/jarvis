from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal
from uuid import UUID


MemoryType = Literal["episode", "semantic_fact", "reflection", "procedure", "goal"]
ExplicitMemoryKind = Literal["fact", "experience", "procedure", "goal"]
MemoryStatus = Literal["active", "superseded", "retracted", "forgotten"]
MemoryEvidenceType = Literal[
    "frame",
    "observation",
    "experience_event",
    "message",
    "tool_execution",
    "reminder",
    "calendar_event",
    "file",
    "memory",
]
PromotionDecision = Literal["pending", "promoted", "skipped"]

MEMORY_TYPES = frozenset({"episode", "semantic_fact", "reflection", "procedure", "goal"})
MEMORY_STATUSES = frozenset({"active", "superseded", "retracted", "forgotten"})
MEMORY_EVIDENCE_TYPES = frozenset(
    {
        "frame",
        "observation",
        "experience_event",
        "message",
        "tool_execution",
        "reminder",
        "calendar_event",
        "file",
        "memory",
    }
)
EMBEDDING_DIMENSIONS = 1536


@dataclass(frozen=True, slots=True)
class MemoryItem:
    id: UUID
    owner_id: str
    session_id: UUID | None
    memory_type: MemoryType
    status: MemoryStatus
    title: str
    content: str
    summary: str
    source_experience_event_id: UUID | None
    occurred_at: datetime | None
    valid_from: datetime | None
    valid_to: datetime | None
    importance: float
    confidence: float
    source: str
    is_explicit: bool
    superseded_by_memory_id: UUID | None
    metadata: Mapping[str, Any]
    created_at: datetime
    updated_at: datetime

    def __post_init__(self) -> None:
        _require_text(self.owner_id, "owner_id")
        _require_text(self.title, "title")
        _require_text(self.content, "content")
        _require_text(self.source, "source")
        if self.memory_type not in MEMORY_TYPES:
            raise ValueError(f"unsupported memory type: {self.memory_type}")
        if self.status not in MEMORY_STATUSES:
            raise ValueError(f"unsupported memory status: {self.status}")
        if self.memory_type == "episode" and self.source_experience_event_id is None:
            raise ValueError("episode memory requires a source experience event")
        if (
            self.valid_from is not None
            and self.valid_to is not None
            and self.valid_to < self.valid_from
        ):
            raise ValueError("memory validity cannot end before it begins")
        _require_score(self.importance, "importance")
        _require_score(self.confidence, "confidence")
        if self.status == "superseded":
            if self.superseded_by_memory_id is None:
                raise ValueError("superseded memory requires its replacement")
            if self.valid_to is None:
                raise ValueError("superseded memory requires a validity end")
        elif self.status == "retracted":
            if self.valid_to is None:
                raise ValueError("retracted memory requires a validity end")
            if self.superseded_by_memory_id is not None:
                raise ValueError("retracted memory cannot have a replacement")
        elif self.superseded_by_memory_id is not None:
            raise ValueError(f"{self.status} memory cannot have a replacement")
        if self.superseded_by_memory_id == self.id:
            raise ValueError("memory cannot supersede itself")


@dataclass(frozen=True, slots=True)
class CalendarEventSnapshot:
    id: UUID
    owner_id: str
    provider: str
    calendar_id: str
    provider_event_id: str
    event_title: str
    starts_at: datetime | None
    ends_at: datetime | None
    observed_at: datetime
    snapshot: Mapping[str, Any]
    content_hash: str
    created_at: datetime

    def __post_init__(self) -> None:
        _require_text(self.owner_id, "owner_id")
        _require_text(self.provider, "provider")
        _require_text(self.calendar_id, "calendar_id")
        _require_text(self.provider_event_id, "provider_event_id")
        if (
            self.starts_at is not None
            and self.ends_at is not None
            and self.ends_at < self.starts_at
        ):
            raise ValueError("calendar event cannot end before it begins")
        if self.content_hash != json_object_hash(self.snapshot):
            raise ValueError("calendar snapshot hash does not match its content")


@dataclass(frozen=True, slots=True)
class MemoryEvidence:
    id: UUID
    owner_id: str
    memory_item_id: UUID
    evidence_type: MemoryEvidenceType
    evidence_ref_id: str
    evidence_time: datetime
    excerpt: str
    confidence: float
    metadata: Mapping[str, Any]
    created_at: datetime

    def __post_init__(self) -> None:
        _require_text(self.owner_id, "owner_id")
        _require_text(self.evidence_ref_id, "evidence_ref_id")
        if self.evidence_type not in MEMORY_EVIDENCE_TYPES:
            raise ValueError(f"unsupported memory evidence type: {self.evidence_type}")
        _require_score(self.confidence, "confidence")


@dataclass(frozen=True, slots=True)
class ExplicitMemoryEvidenceInput:
    evidence_type: MemoryEvidenceType
    evidence_ref_id: str
    evidence_time: datetime
    excerpt: str = ""
    confidence: float = 1.0
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.evidence_type not in MEMORY_EVIDENCE_TYPES:
            raise ValueError(f"unsupported memory evidence type: {self.evidence_type}")
        _require_text(self.evidence_ref_id, "evidence_ref_id")
        _require_score(self.confidence, "confidence")


@dataclass(frozen=True, slots=True)
class MemoryEvidenceDetail:
    evidence: MemoryEvidence
    source_available: bool
    retained_image_available: bool | None


@dataclass(frozen=True, slots=True)
class ExplicitMemoryWriteResult:
    memory: MemoryItem
    evidence: tuple[MemoryEvidence, ...]
    indexing_status: str
    replayed: bool
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class MemoryForgettingTombstone:
    id: UUID
    owner_id: str
    memory_type: MemoryType
    identity_fingerprint: str
    source_memory_id: UUID
    reason: str
    forgotten_at: datetime
    overridden_at: datetime | None
    override_memory_id: UUID | None
    override_metadata: Mapping[str, Any]
    suppressed_recreation_count: int
    last_suppressed_at: datetime | None
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class ForgetMemoryResult:
    memory: MemoryItem
    tombstone: MemoryForgettingTombstone
    replayed: bool
    removed_embedding_count: int
    closed_memory_link_count: int
    closed_entity_relation_count: int
    removed_retention_reason_count: int
    cancelled_embedding_job_count: int


@dataclass(frozen=True, slots=True)
class MemoryLink:
    id: UUID
    owner_id: str
    source_memory_id: UUID
    target_memory_id: UUID
    relationship: str
    confidence: float
    valid_from: datetime
    valid_to: datetime | None
    supporting_memory_id: UUID | None
    metadata: Mapping[str, Any]
    created_at: datetime
    updated_at: datetime

    def __post_init__(self) -> None:
        _require_text(self.owner_id, "owner_id")
        _require_text(self.relationship, "relationship")
        _require_score(self.confidence, "confidence")
        if self.source_memory_id == self.target_memory_id:
            raise ValueError("memory link cannot point to itself")
        if self.valid_to is not None and self.valid_to < self.valid_from:
            raise ValueError("memory link cannot end before it begins")


@dataclass(frozen=True, slots=True)
class MemoryExplanation:
    memory: MemoryItem
    evidence: tuple[MemoryEvidenceDetail, ...]
    links: tuple[MemoryLink, ...]
    related_memories: tuple[MemoryItem, ...]


@dataclass(frozen=True, slots=True)
class SemanticFactHistoryEntry:
    fact: MemoryItem
    evidence: tuple[MemoryEvidence, ...]
    links: tuple[MemoryLink, ...]

    def __post_init__(self) -> None:
        if self.fact.memory_type != "semantic_fact":
            raise ValueError("semantic fact history requires a semantic fact")
        if any(
            item.owner_id != self.fact.owner_id
            or item.memory_item_id != self.fact.id
            for item in self.evidence
        ):
            raise ValueError("semantic fact evidence must match the history fact")
        if any(
            link.owner_id != self.fact.owner_id
            or self.fact.id not in {link.source_memory_id, link.target_memory_id}
            for link in self.links
        ):
            raise ValueError("semantic fact links must involve the history fact")


@dataclass(frozen=True, slots=True)
class MemoryEmbedding:
    subject_id: UUID
    embedding_kind: str
    provider: str
    model: str
    dimensions: int
    embedded_content: str
    content_hash: str
    embedding: tuple[float, ...]
    embedded_at: datetime
    updated_at: datetime

    def __post_init__(self) -> None:
        _require_text(self.embedding_kind, "embedding_kind")
        _require_text(self.provider, "provider")
        _require_text(self.model, "model")
        if self.dimensions != EMBEDDING_DIMENSIONS:
            raise ValueError(f"embedding dimensions must be {EMBEDDING_DIMENSIONS}")
        normalized = normalize_embedding(self.embedding)
        if normalized != self.embedding:
            raise ValueError("embedding must be stored as a normalized float tuple")
        if self.content_hash != content_hash(self.embedded_content):
            raise ValueError("embedding content hash does not match embedded content")


def normalize_embedding(values: Sequence[float]) -> tuple[float, ...]:
    if len(values) != EMBEDDING_DIMENSIONS:
        raise ValueError(f"embedding must contain {EMBEDDING_DIMENSIONS} values")
    normalized = tuple(float(value) for value in values)
    if any(not math.isfinite(value) for value in normalized):
        raise ValueError("embedding cannot contain non-finite values")
    return normalized


def content_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def json_object_hash(value: Mapping[str, Any]) -> str:
    serialized = json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return content_hash(serialized)


def _require_text(value: str, label: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{label} cannot be blank")
    return normalized


def _require_score(value: float, label: str) -> None:
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{label} must be between 0 and 1")
