from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Literal
from uuid import UUID

from app.memory.consolidation.contracts import ConsolidationResult
from app.memory.consolidation.contracts import FactChangeIntent, FactTemporalScope
from app.memory.entities.models import MemoryEntity, normalize_aliases, normalize_entity_name
from app.memory.models import MemoryItem, content_hash


RECONCILE_SEMANTIC_MEMORY_JOB = "reconcile_semantic_memory"
EMBED_MEMORY_ENTITY_JOB = "embed_memory_entity"
ENTITY_IDENTITY_EMBEDDING_KIND = "entity_identity"
SemanticSource = Literal["user_statement", "inferred_observation"]


@dataclass(frozen=True, slots=True)
class SemanticEvidence:
    evidence_type: str
    evidence_ref_id: UUID
    occurred_at: datetime
    excerpt: str
    message_role: str | None = None


@dataclass(frozen=True, slots=True)
class SemanticReconciliationContext:
    owner_id: str
    session_id: UUID
    event_id: UUID
    episode_memory_id: UUID
    occurred_at: datetime
    result: ConsolidationResult
    evidence: tuple[SemanticEvidence, ...]
    semantic_reconciled_at: datetime | None = None
    suppressed_memory_fingerprints: frozenset[str] = frozenset()
    event_started_at: datetime | None = None

    @property
    def evidence_by_reference(self) -> Mapping[tuple[str, UUID], SemanticEvidence]:
        return {
            (item.evidence_type, item.evidence_ref_id): item
            for item in self.evidence
        }


@dataclass(frozen=True, slots=True)
class SemanticEntityIdentity:
    name: str
    entity_type: str
    confidence: float
    aliases: tuple[str, ...] = ()
    external_namespace: str | None = None
    external_id: str | None = None

    def __post_init__(self) -> None:
        if not self.name.strip() or not self.entity_type.strip():
            raise ValueError("semantic entity requires a name and type")
        if not 0 <= self.confidence <= 1:
            raise ValueError("semantic entity confidence must be between zero and one")
        if (self.external_namespace is None) != (self.external_id is None):
            raise ValueError("semantic entity external identity must be complete")
        if normalize_aliases(self.aliases) != self.aliases:
            raise ValueError("semantic entity aliases must be normalized")

    @property
    def normalized_name(self) -> str:
        return normalize_entity_name(self.name)

    @property
    def identity_key(self) -> str:
        if self.external_namespace is not None:
            return f"external:{self.external_namespace}:{self.external_id}"
        return f"name:{self.entity_type}:{self.normalized_name}"


@dataclass(frozen=True, slots=True)
class SemanticFactProposal:
    subject: SemanticEntityIdentity
    predicate: str
    value: str
    confidence: float
    source: SemanticSource
    change_intent: FactChangeIntent
    previous_value: str | None
    valid_from: datetime
    valid_to: datetime | None
    evidence: tuple[SemanticEvidence, ...]
    temporal_scope: FactTemporalScope = "durable"

    @property
    def is_explicit(self) -> bool:
        return self.source == "user_statement"


@dataclass(frozen=True, slots=True)
class SemanticRelationProposal:
    subject: SemanticEntityIdentity
    predicate: str
    target: SemanticEntityIdentity
    confidence: float
    source: SemanticSource
    valid_from: datetime
    valid_to: datetime | None
    evidence: tuple[SemanticEvidence, ...]
    temporal_scope: FactTemporalScope = "durable"


@dataclass(frozen=True, slots=True)
class SemanticReconciliationPlan:
    owner_id: str
    event_id: UUID
    episode_memory_id: UUID
    entities: tuple[SemanticEntityIdentity, ...]
    facts: tuple[SemanticFactProposal, ...]
    relations: tuple[SemanticRelationProposal, ...]


@dataclass(frozen=True, slots=True)
class SemanticReconciliationOutcome:
    created_entities: tuple[MemoryEntity, ...]
    created_facts: tuple[MemoryItem, ...]
    updated_facts: tuple[MemoryItem, ...]
    created_relation_count: int
    already_reconciled: bool = False


def semantic_reconciliation_dedupe_key(event_id: UUID) -> str:
    return f"experience-event:{event_id}:reconcile-semantic"


def build_entity_embedding_content(entity: MemoryEntity) -> str:
    return json.dumps(
        {
            "schema": "jarvis.memory_entity_embedding.v1",
            "entity_type": entity.entity_type,
            "canonical_name": entity.canonical_name,
            "aliases": list(entity.aliases),
            "description": entity.description,
        },
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def entity_embedding_dedupe_key(entity: MemoryEntity) -> str:
    digest = content_hash(build_entity_embedding_content(entity))
    return f"memory-entity:{entity.id}:embed:{digest}"
