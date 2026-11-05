from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from typing import Literal
from uuid import UUID

from app.memory.consolidation.contracts import (
    EXCLUSIVE_SEMANTIC_PREDICATES,
    MULTI_VALUE_SEMANTIC_PREDICATES,
    SEMANTIC_PREDICATE_VALUES,
)
from app.memory.semantic.contracts import SemanticFactProposal


FactCardinality = Literal["exclusive", "multi"]


@dataclass(frozen=True, slots=True)
class ActiveSemanticFact:
    id: UUID
    predicate: str
    value: str
    valid_from: datetime
    is_explicit: bool = False
    valid_to: datetime | None = None
    confidence: float = 1.0
    change_intent: str = "assert"
    previous_value: str | None = None


@dataclass(frozen=True, slots=True)
class FactMutationDecision:
    create_new: bool = False
    reinforce_fact_id: UUID | None = None
    supersede_fact_ids: tuple[UUID, ...] = ()
    retract_fact_ids: tuple[UUID, ...] = ()
    contradict_fact_ids: tuple[UUID, ...] = ()
    reason: str = "no_change"


class SemanticContradictionPolicy:
    """Pure policy for preserving or replacing conflicting semantic facts."""

    def __init__(self, *, supersede_min_confidence: float = 0.90) -> None:
        if not 0 <= supersede_min_confidence <= 1:
            raise ValueError("supersede confidence must be between zero and one")
        if EXCLUSIVE_SEMANTIC_PREDICATES | MULTI_VALUE_SEMANTIC_PREDICATES != SEMANTIC_PREDICATE_VALUES:
            raise ValueError("semantic predicate cardinality policy is incomplete")
        if EXCLUSIVE_SEMANTIC_PREDICATES & MULTI_VALUE_SEMANTIC_PREDICATES:
            raise ValueError("semantic predicate cardinality policy overlaps")
        self._supersede_min_confidence = supersede_min_confidence

    def decide(
        self,
        *,
        proposal: SemanticFactProposal,
        active_facts: tuple[ActiveSemanticFact, ...],
    ) -> FactMutationDecision:
        if proposal.predicate not in SEMANTIC_PREDICATE_VALUES:
            raise ValueError(f"unsupported semantic predicate: {proposal.predicate}")
        if proposal.change_intent in {"replace", "retract"} and not proposal.is_explicit:
            raise ValueError("replacement and retraction require explicit user evidence")
        if proposal.temporal_scope == "event" and proposal.change_intent != "assert":
            raise ValueError("event-only evidence cannot replace persistent state")

        same_predicate = tuple(
            fact for fact in active_facts if fact.predicate == proposal.predicate
        )
        exact = tuple(
            fact
            for fact in same_predicate
            if normalize_fact_value(fact.value) == normalize_fact_value(proposal.value)
        )
        equivalent_interval = tuple(fact for fact in exact if fact.valid_to == proposal.valid_to)

        if proposal.change_intent == "retract":
            return FactMutationDecision(
                retract_fact_ids=tuple(fact.id for fact in exact),
                reason="explicit_retraction" if exact else "retraction_target_not_active",
            )

        if proposal.change_intent in {"replace", "transition"}:
            if proposal.change_intent == "transition":
                if proposal.temporal_scope != "state" or proposal.previous_value is None:
                    raise ValueError("transition requires state scope and an exact previous value")
                if not proposal.is_explicit and proposal.confidence < self._supersede_min_confidence:
                    return FactMutationDecision(reason="uncertain_state_transition")
            targets = self._replacement_targets(proposal, same_predicate)
            if not targets:
                return FactMutationDecision(reason="replacement_target_not_active")
            exact = equivalent_interval
            return FactMutationDecision(
                create_new=not exact,
                reinforce_fact_id=exact[0].id if exact else None,
                supersede_fact_ids=tuple(
                    fact.id
                    for fact in targets
                    if not exact or fact.id != exact[0].id
                ),
                reason="confirmed_state_transition" if proposal.change_intent == "transition" else "explicit_replacement",
            )

        if equivalent_interval:
            return FactMutationDecision(
                reinforce_fact_id=equivalent_interval[0].id,
                reason="same_fact_reinforced",
            )
        if exact:
            return FactMutationDecision(create_new=True, reason="same_value_distinct_interval")

        if proposal.temporal_scope == "event":
            return FactMutationDecision(create_new=True, reason="bounded_event_observation")

        if proposal.predicate in EXCLUSIVE_SEMANTIC_PREDICATES and same_predicate:
            explicit_conflicts = tuple(
                fact.id for fact in same_predicate if fact.is_explicit
            )
            inferred_conflicts = tuple(
                fact.id for fact in same_predicate if not fact.is_explicit
            )
            if proposal.is_explicit:
                return FactMutationDecision(
                    create_new=True,
                    supersede_fact_ids=explicit_conflicts + inferred_conflicts,
                    reason="exclusive_value_superseded",
                )
            if proposal.confidence >= self._supersede_min_confidence:
                return FactMutationDecision(
                    create_new=True,
                    supersede_fact_ids=inferred_conflicts,
                    contradict_fact_ids=explicit_conflicts,
                    reason="exclusive_inference_reconciled",
                )
            return FactMutationDecision(
                create_new=True,
                contradict_fact_ids=explicit_conflicts + inferred_conflicts,
                reason="exclusive_value_conflict_preserved",
            )

        opposite = _opposite_predicate(proposal.predicate)
        contradictions = tuple(
            fact.id
            for fact in active_facts
            if opposite is not None
            and fact.predicate == opposite
            and normalize_fact_value(fact.value) == normalize_fact_value(proposal.value)
        )
        return FactMutationDecision(
            create_new=True,
            contradict_fact_ids=contradictions,
            reason="multi_value_assertion",
        )

    def bound_validity(
        self, *, proposal: SemanticFactProposal, known_facts: tuple[ActiveSemanticFact, ...],
        replaced_ids: tuple[UUID, ...] = (),
    ) -> SemanticFactProposal:
        """A late write may fill an earlier interval, never run through a known successor."""
        ends = [proposal.valid_to] if proposal.valid_to is not None else []
        for fact in known_facts:
            if fact.id in replaced_ids and fact.valid_to is not None:
                ends.append(fact.valid_to)
            if proposal.temporal_scope == "event" or fact.valid_from <= proposal.valid_from:
                continue
            same_predicate = fact.predicate == proposal.predicate
            explicit_successor = (
                fact.change_intent in {"replace", "transition"}
                and fact.previous_value is not None
                and normalize_fact_value(fact.previous_value) == normalize_fact_value(proposal.value)
                and (fact.is_explicit or fact.confidence >= self._supersede_min_confidence)
            )
            exclusive_successor = (
                proposal.predicate in EXCLUSIVE_SEMANTIC_PREDICATES
                and (fact.is_explicit or (
                    not proposal.is_explicit and fact.confidence >= self._supersede_min_confidence
                ))
            )
            if same_predicate and (explicit_successor or exclusive_successor):
                ends.append(fact.valid_from)
        return replace(proposal, valid_to=min(ends) if ends else None)

    @staticmethod
    def _replacement_targets(
        proposal: SemanticFactProposal,
        active_facts: tuple[ActiveSemanticFact, ...],
    ) -> tuple[ActiveSemanticFact, ...]:
        if proposal.previous_value is None:
            if proposal.predicate in MULTI_VALUE_SEMANTIC_PREDICATES:
                raise ValueError("multi-valued replacement requires previous_value")
            return active_facts
        previous = normalize_fact_value(proposal.previous_value)
        return tuple(
            fact for fact in active_facts if normalize_fact_value(fact.value) == previous
        )


def normalize_fact_value(value: str) -> str:
    return " ".join(value.casefold().split())


def semantic_fact_conflict_domain(predicate: str) -> str:
    """Return the lock domain shared by predicates that can conflict."""

    if predicate in {"likes", "dislikes"}:
        return "likes_or_dislikes"
    return predicate


def _opposite_predicate(predicate: str) -> str | None:
    if predicate == "likes":
        return "dislikes"
    if predicate == "dislikes":
        return "likes"
    return None
