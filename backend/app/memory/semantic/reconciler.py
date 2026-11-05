from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
from uuid import UUID

from app.memory.consolidation.contracts import EvidenceCitation
from app.memory.entities.models import normalize_entity_name
from app.memory.forgetting import semantic_fact_identity_fingerprint
from app.memory.semantic.contracts import (
    SemanticEntityIdentity,
    SemanticEvidence,
    SemanticFactProposal,
    SemanticReconciliationContext,
    SemanticReconciliationPlan,
    SemanticRelationProposal,
    SemanticSource,
)


_PREFERENCE_PREDICATES = frozenset({"prefers", "likes", "dislikes"})
_PERSON_CATEGORIES = frozenset({"person", "people", "human", "user", "contact"})
_PLACE_CATEGORIES = frozenset({"place", "location", "room", "venue", "address"})
_OWNER_NAMES = frozenset({"user", "me", "myself", "i"})


class SemanticMemoryReconciler:
    """Turn validated model suggestions into conservative, evidence-bound plans."""

    def __init__(
        self,
        *,
        minimum_explicit_confidence: float = 0.55,
        minimum_inferred_confidence: float = 0.85,
        minimum_entity_confidence: float = 0.70,
    ) -> None:
        for name, value in (
            ("explicit confidence", minimum_explicit_confidence),
            ("inferred confidence", minimum_inferred_confidence),
            ("entity confidence", minimum_entity_confidence),
        ):
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be between zero and one")
        if minimum_explicit_confidence > minimum_inferred_confidence:
            raise ValueError("explicit evidence must not require more confidence")
        self._minimum_explicit_confidence = minimum_explicit_confidence
        self._minimum_inferred_confidence = minimum_inferred_confidence
        self._minimum_entity_confidence = minimum_entity_confidence

    def plan(self, context: SemanticReconciliationContext) -> SemanticReconciliationPlan:
        evidence_by_reference = context.evidence_by_reference
        identities: dict[str, SemanticEntityIdentity] = {}
        category_by_name: dict[str, str] = {}
        eligible_entities: list[SemanticEntityIdentity] = []

        for candidate in context.result.entity_candidates:
            if candidate.confidence < self._minimum_entity_confidence:
                continue
            if not _resolve_evidence(candidate.evidence, evidence_by_reference):
                continue
            entity_type = _normalize_entity_type(candidate.category)
            identity = self._entity_identity(
                owner_id=context.owner_id,
                name=candidate.name,
                entity_type=entity_type,
                confidence=candidate.confidence,
            )
            category_by_name[identity.normalized_name] = identity.entity_type
            eligible_entities.append(identity)

        person_name_counts = Counter(
            identity.normalized_name
            for identity in eligible_entities
            if identity.entity_type == "person" and identity.external_namespace is None
        )
        ambiguous_people = frozenset(
            name for name, count in person_name_counts.items() if count > 1
        )
        for identity in eligible_entities:
            if identity.normalized_name in ambiguous_people:
                continue
            identities.setdefault(identity.identity_key, identity)

        facts: list[SemanticFactProposal] = []
        seen_facts: set[tuple[object, ...]] = set()
        for candidate in context.result.semantic_fact_candidates:
            if normalize_entity_name(candidate.subject) in ambiguous_people:
                continue
            source = candidate.source_classification
            if not self._is_confident(candidate.confidence, source):
                continue
            if candidate.predicate in _PREFERENCE_PREDICATES and source != "user_statement":
                continue
            evidence = _resolve_evidence(candidate.evidence, evidence_by_reference)
            if not evidence:
                continue
            if source == "user_statement" and not _contains_user_message(evidence):
                continue
            validity = _resolve_validity(
                candidate.valid_from_hint,
                candidate.valid_to_hint,
                default=context.occurred_at,
                event_start=context.event_started_at or min(item.occurred_at for item in evidence),
                temporal_scope=candidate.temporal_scope,
            )
            if validity is None:
                continue
            subject = self._entity_identity(
                owner_id=context.owner_id,
                name=candidate.subject,
                entity_type=category_by_name.get(
                    normalize_entity_name(candidate.subject), "object"
                ),
                confidence=candidate.confidence,
            )
            if semantic_fact_identity_fingerprint(
                subject_identity_key=subject.identity_key,
                predicate=candidate.predicate,
                value=candidate.value,
            ) in context.suppressed_memory_fingerprints:
                continue
            identity = (
                subject.identity_key,
                candidate.predicate,
                normalize_entity_name(candidate.value),
                candidate.change_intent,
                (
                    normalize_entity_name(candidate.previous_value)
                    if candidate.previous_value is not None
                    else ""
                ),
                candidate.temporal_scope, *validity,
            )
            if identity in seen_facts:
                continue
            seen_facts.add(identity)
            identities.setdefault(subject.identity_key, subject)
            facts.append(
                SemanticFactProposal(
                    subject=subject,
                    predicate=candidate.predicate,
                    value=candidate.value.strip(),
                    confidence=candidate.confidence,
                    source=source,
                    change_intent=candidate.change_intent,
                    previous_value=(
                        candidate.previous_value.strip()
                        if candidate.previous_value is not None
                        else None
                    ),
                    valid_from=validity[0],
                    valid_to=validity[1],
                    evidence=evidence,
                    temporal_scope=candidate.temporal_scope,
                )
            )

        relations: list[SemanticRelationProposal] = []
        seen_relations: set[tuple[object, ...]] = set()
        for candidate in context.result.relation_candidates:
            source_name = normalize_entity_name(candidate.source_entity)
            target_name = normalize_entity_name(candidate.target_entity)
            if source_name in ambiguous_people or target_name in ambiguous_people:
                continue
            evidence = _resolve_evidence(candidate.evidence, evidence_by_reference)
            if not evidence:
                continue
            source: SemanticSource = candidate.source_classification
            if not self._is_confident(candidate.confidence, source):
                continue
            if source == "user_statement" and not _contains_user_message(evidence):
                continue
            validity = _resolve_validity(
                candidate.valid_from_hint,
                candidate.valid_to_hint,
                default=context.occurred_at,
                event_start=context.event_started_at or min(item.occurred_at for item in evidence),
                temporal_scope=candidate.temporal_scope,
            )
            if validity is None:
                continue
            subject = self._entity_identity(
                owner_id=context.owner_id,
                name=candidate.source_entity,
                entity_type=category_by_name.get(source_name, "object"),
                confidence=candidate.confidence,
            )
            target = self._entity_identity(
                owner_id=context.owner_id,
                name=candidate.target_entity,
                entity_type=category_by_name.get(target_name, "object"),
                confidence=candidate.confidence,
            )
            if subject.identity_key == target.identity_key:
                continue
            identity = (
                subject.identity_key, candidate.predicate, target.identity_key,
                candidate.temporal_scope, *validity,
            )
            if identity in seen_relations:
                continue
            seen_relations.add(identity)
            identities.setdefault(subject.identity_key, subject)
            identities.setdefault(target.identity_key, target)
            relations.append(
                SemanticRelationProposal(
                    subject=subject,
                    predicate=candidate.predicate,
                    target=target,
                    confidence=candidate.confidence,
                    source=source,
                    valid_from=validity[0],
                    valid_to=validity[1],
                    evidence=evidence,
                    temporal_scope=candidate.temporal_scope,
                )
            )

        return SemanticReconciliationPlan(
            owner_id=context.owner_id,
            event_id=context.event_id,
            episode_memory_id=context.episode_memory_id,
            entities=tuple(identities.values()),
            facts=tuple(facts),
            relations=tuple(relations),
        )

    def _is_confident(self, confidence: float, source: SemanticSource) -> bool:
        threshold = (
            self._minimum_explicit_confidence
            if source == "user_statement"
            else self._minimum_inferred_confidence
        )
        return confidence >= threshold

    @staticmethod
    def _entity_identity(
        *,
        owner_id: str,
        name: str,
        entity_type: str,
        confidence: float,
    ) -> SemanticEntityIdentity:
        if normalize_entity_name(name) in _OWNER_NAMES:
            return SemanticEntityIdentity(
                name="User",
                entity_type="person",
                confidence=confidence,
                external_namespace="jarvis.owner",
                external_id=owner_id,
            )
        return SemanticEntityIdentity(
            name=name.strip(),
            entity_type=entity_type,
            confidence=confidence,
        )


def _normalize_entity_type(category: str | None) -> str:
    normalized = " ".join((category or "object").casefold().split())
    if normalized in _PERSON_CATEGORIES:
        return "person"
    if normalized in _PLACE_CATEGORIES:
        return "place"
    return normalized or "object"


def _resolve_evidence(
    citations: Sequence[EvidenceCitation],
    available: Mapping[tuple[str, UUID], SemanticEvidence],
) -> tuple[SemanticEvidence, ...]:
    resolved: list[SemanticEvidence] = []
    seen: set[tuple[str, UUID]] = set()
    for citation in citations:
        try:
            reference = (citation.evidence_type, UUID(citation.evidence_ref_id))
        except ValueError:
            return ()
        item = available.get(reference)
        if item is None:
            return ()
        if reference not in seen:
            seen.add(reference)
            resolved.append(item)
    return tuple(resolved)


def _contains_user_message(evidence: Iterable[SemanticEvidence]) -> bool:
    return any(
        item.evidence_type == "message" and item.message_role == "user"
        for item in evidence
    )


def _resolve_validity(
    start_hint: str | None,
    end_hint: str | None,
    *,
    default: datetime,
    event_start: datetime | None = None,
    temporal_scope: str = "durable",
) -> tuple[datetime, datetime | None] | None:
    try:
        start = _parse_timestamp(start_hint) if start_hint else (
            event_start if temporal_scope == "event" and event_start is not None else default
        )
        end = _parse_timestamp(end_hint) if end_hint else None
    except ValueError:
        return None
    if start.tzinfo is None or (end is not None and end.tzinfo is None):
        return None
    if temporal_scope == "event":
        start = max(start, event_start or default)
        end = min(end, default) if end is not None else default
    if end is not None and end < start:
        return None
    return start, end


def _parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))
