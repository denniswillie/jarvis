from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from app.memory.models import (
    EMBEDDING_DIMENSIONS,
    content_hash,
    normalize_embedding,
)


EntityRelationStatus = Literal["active", "superseded", "retracted"]
ENTITY_RELATION_STATUSES = frozenset({"active", "superseded", "retracted"})


@dataclass(frozen=True, slots=True)
class MemoryEntity:
    id: UUID
    owner_id: str
    entity_type: str
    canonical_name: str
    normalized_name: str
    aliases: tuple[str, ...]
    description: str
    confidence: float
    external_namespace: str | None
    external_id: str | None
    metadata: Mapping[str, Any]
    created_at: datetime
    updated_at: datetime

    def __post_init__(self) -> None:
        _require_text(self.owner_id, "owner_id")
        _require_text(self.entity_type, "entity_type")
        _require_text(self.canonical_name, "canonical_name")
        _require_text(self.normalized_name, "normalized_name")
        _require_score(self.confidence, "confidence")
        if normalize_aliases(self.aliases) != self.aliases:
            raise ValueError("entity aliases must be normalized and deduplicated")
        _require_external_identity(self.external_namespace, self.external_id)


@dataclass(frozen=True, slots=True)
class MemoryEntityMention:
    id: UUID
    owner_id: str
    entity_id: UUID
    memory_item_id: UUID
    role: str
    confidence: float
    created_at: datetime

    def __post_init__(self) -> None:
        _require_text(self.owner_id, "owner_id")
        _require_text(self.role, "role")
        _require_score(self.confidence, "confidence")


@dataclass(frozen=True, slots=True)
class MemoryEntityEmbedding:
    entity_id: UUID
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
        if normalize_embedding(self.embedding) != self.embedding:
            raise ValueError("embedding must be stored as a normalized float tuple")
        if self.content_hash != content_hash(self.embedded_content):
            raise ValueError("embedding content hash does not match embedded content")


@dataclass(frozen=True, slots=True)
class MemoryEntityRelation:
    id: UUID
    owner_id: str
    subject_entity_id: UUID
    predicate: str
    object_entity_id: UUID | None
    object_value: Any | None
    status: EntityRelationStatus
    confidence: float
    valid_from: datetime
    valid_to: datetime | None
    supporting_memory_id: UUID
    superseded_by_relation_id: UUID | None
    metadata: Mapping[str, Any]
    created_at: datetime
    updated_at: datetime

    def __post_init__(self) -> None:
        _require_text(self.owner_id, "owner_id")
        _require_text(self.predicate, "predicate")
        _require_score(self.confidence, "confidence")
        if self.status not in ENTITY_RELATION_STATUSES:
            raise ValueError(f"unsupported entity relation status: {self.status}")
        if (self.object_entity_id is None) == (self.object_value is None):
            raise ValueError("relation requires exactly one entity or value object")
        if self.object_entity_id == self.subject_entity_id:
            raise ValueError("entity relation cannot point to itself")
        if self.valid_to is not None and self.valid_to < self.valid_from:
            raise ValueError("entity relation cannot end before it begins")
        if self.superseded_by_relation_id == self.id:
            raise ValueError("entity relation cannot supersede itself")
        if self.status == "active":
            if self.valid_to is not None or self.superseded_by_relation_id is not None:
                raise ValueError("active entity relation cannot be closed or superseded")
        elif self.status == "superseded":
            if self.valid_to is None or self.superseded_by_relation_id is None:
                raise ValueError("superseded entity relation requires closure and replacement")
        elif self.valid_to is None or self.superseded_by_relation_id is not None:
            raise ValueError("retracted entity relation requires closure without replacement")


def normalize_entity_name(value: str) -> str:
    normalized = " ".join(_require_text(value, "entity name").casefold().split())
    if not normalized:
        raise ValueError("entity name cannot be blank")
    return normalized


def normalize_aliases(values: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        alias = normalize_entity_name(value)
        if alias not in seen:
            seen.add(alias)
            normalized.append(alias)
    if len(normalized) > 50:
        raise ValueError("entity aliases cannot contain more than 50 values")
    return tuple(normalized)


def _require_external_identity(namespace: str | None, external_id: str | None) -> None:
    if namespace is None and external_id is None:
        return
    if namespace is None or external_id is None:
        raise ValueError("external entity identity requires both namespace and id")
    _require_text(namespace, "external_namespace")
    _require_text(external_id, "external_id")


def _require_text(value: str, label: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{label} cannot be blank")
    return normalized


def _require_score(value: float, label: str) -> None:
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{label} must be between 0 and 1")
