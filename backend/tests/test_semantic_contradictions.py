from __future__ import annotations

import unittest
from datetime import UTC, datetime
from uuid import uuid4

from app.memory.consolidation.contracts import (
    EXCLUSIVE_SEMANTIC_PREDICATES,
    MULTI_VALUE_SEMANTIC_PREDICATES,
    SEMANTIC_PREDICATE_VALUES,
)
from app.memory.semantic.contracts import SemanticEntityIdentity, SemanticFactProposal
from app.memory.semantic.contradictions import (
    ActiveSemanticFact,
    SemanticContradictionPolicy,
    semantic_fact_conflict_domain,
)


class SemanticContradictionPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = SemanticContradictionPolicy(supersede_min_confidence=0.90)

    def test_predicate_cardinality_is_exhaustive_and_disjoint(self) -> None:
        self.assertEqual(
            EXCLUSIVE_SEMANTIC_PREDICATES | MULTI_VALUE_SEMANTIC_PREDICATES,
            SEMANTIC_PREDICATE_VALUES,
        )
        self.assertFalse(EXCLUSIVE_SEMANTIC_PREDICATES & MULTI_VALUE_SEMANTIC_PREDICATES)

    def test_multi_value_assertions_coexist(self) -> None:
        existing = _active("owns", "silver keys")
        decision = self.policy.decide(
            proposal=_proposal("owns", "blue bicycle"),
            active_facts=(existing,),
        )
        self.assertTrue(decision.create_new)
        self.assertEqual(decision.supersede_fact_ids, ())

    def test_explicit_exclusive_assertion_supersedes_prior_value(self) -> None:
        existing = _active("located_at", "kitchen")
        decision = self.policy.decide(
            proposal=_proposal("located_at", "office", explicit=True),
            active_facts=(existing,),
        )
        self.assertEqual(decision.supersede_fact_ids, (existing.id,))

    def test_weak_exclusive_inference_preserves_contradiction(self) -> None:
        existing = _active("located_at", "kitchen")
        decision = self.policy.decide(
            proposal=_proposal("located_at", "office", confidence=0.86),
            active_facts=(existing,),
        )
        self.assertEqual(decision.contradict_fact_ids, (existing.id,))
        self.assertEqual(decision.supersede_fact_ids, ())

    def test_strong_exclusive_inference_supersedes_prior_value(self) -> None:
        existing = _active("located_at", "kitchen")
        decision = self.policy.decide(
            proposal=_proposal("located_at", "office", confidence=0.95),
            active_facts=(existing,),
        )
        self.assertEqual(decision.supersede_fact_ids, (existing.id,))

    def test_strong_inference_preserves_conflicting_explicit_fact(self) -> None:
        existing = _active("located_at", "kitchen", explicit=True)
        decision = self.policy.decide(
            proposal=_proposal("located_at", "office", confidence=0.95),
            active_facts=(existing,),
        )
        self.assertEqual(decision.supersede_fact_ids, ())
        self.assertEqual(decision.contradict_fact_ids, (existing.id,))

    def test_strong_inference_only_supersedes_inferred_conflicts(self) -> None:
        explicit = _active("located_at", "kitchen", explicit=True)
        inferred = _active("located_at", "hallway")
        decision = self.policy.decide(
            proposal=_proposal("located_at", "office", confidence=0.95),
            active_facts=(explicit, inferred),
        )
        self.assertEqual(decision.supersede_fact_ids, (inferred.id,))
        self.assertEqual(decision.contradict_fact_ids, (explicit.id,))

    def test_explicit_multi_value_replacement_targets_previous_value(self) -> None:
        old = _active("owns", "silver keys")
        other = _active("owns", "blue bicycle")
        decision = self.policy.decide(
            proposal=_proposal(
                "owns",
                "bronze keys",
                explicit=True,
                change_intent="replace",
                previous_value="silver keys",
            ),
            active_facts=(old, other),
        )
        self.assertEqual(decision.supersede_fact_ids, (old.id,))

    def test_replacement_reuses_an_already_active_new_value(self) -> None:
        old = _active("owns", "silver keys")
        replacement = _active("owns", "bronze keys")
        decision = self.policy.decide(
            proposal=_proposal(
                "owns",
                "bronze keys",
                explicit=True,
                change_intent="replace",
                previous_value="silver keys",
            ),
            active_facts=(old, replacement),
        )
        self.assertFalse(decision.create_new)
        self.assertEqual(decision.reinforce_fact_id, replacement.id)
        self.assertEqual(decision.supersede_fact_ids, (old.id,))

    def test_replacement_with_missing_previous_value_is_a_no_op(self) -> None:
        existing = _active("located_at", "kitchen", explicit=True)
        decision = self.policy.decide(
            proposal=_proposal(
                "located_at",
                "office",
                explicit=True,
                change_intent="replace",
                previous_value="bedroom",
            ),
            active_facts=(existing,),
        )
        self.assertFalse(decision.create_new)
        self.assertIsNone(decision.reinforce_fact_id)
        self.assertEqual(decision.supersede_fact_ids, ())
        self.assertEqual(decision.reason, "replacement_target_not_active")

    def test_retraction_targets_exact_active_value_without_creating_fact(self) -> None:
        old = _active("owns", "silver keys")
        decision = self.policy.decide(
            proposal=_proposal(
                "owns", "silver keys", explicit=True, change_intent="retract"
            ),
            active_facts=(old,),
        )
        self.assertFalse(decision.create_new)
        self.assertEqual(decision.retract_fact_ids, (old.id,))

    def test_same_value_reinforces_existing_fact(self) -> None:
        existing = _active("owns", "Silver   Keys")
        decision = self.policy.decide(
            proposal=_proposal("owns", "silver keys"),
            active_facts=(existing,),
        )
        self.assertEqual(decision.reinforce_fact_id, existing.id)

    def test_likes_and_dislikes_are_linked_as_contradictions(self) -> None:
        existing = _active("likes", "loud cafes")
        decision = self.policy.decide(
            proposal=_proposal("dislikes", "loud cafes", explicit=True),
            active_facts=(existing,),
        )
        self.assertEqual(decision.contradict_fact_ids, (existing.id,))

    def test_likes_and_dislikes_share_a_conflict_lock_domain(self) -> None:
        self.assertEqual(
            semantic_fact_conflict_domain("likes"),
            semantic_fact_conflict_domain("dislikes"),
        )
        self.assertNotEqual(
            semantic_fact_conflict_domain("likes"),
            semantic_fact_conflict_domain("owns"),
        )


def _active(
    predicate: str,
    value: str,
    *,
    explicit: bool = False,
) -> ActiveSemanticFact:
    return ActiveSemanticFact(
        id=uuid4(),
        predicate=predicate,
        value=value,
        valid_from=datetime.now(UTC),
        is_explicit=explicit,
    )


def _proposal(
    predicate: str,
    value: str,
    *,
    explicit: bool = False,
    confidence: float = 0.88,
    change_intent: str = "assert",
    previous_value: str | None = None,
) -> SemanticFactProposal:
    now = datetime.now(UTC)
    return SemanticFactProposal(
        subject=SemanticEntityIdentity(
            name="User",
            entity_type="person",
            confidence=0.99,
            external_namespace="jarvis.owner",
            external_id="owner",
        ),
        predicate=predicate,
        value=value,
        confidence=confidence,
        source="user_statement" if explicit else "inferred_observation",
        change_intent=change_intent,  # type: ignore[arg-type]
        previous_value=previous_value,
        valid_from=now,
        valid_to=None,
        evidence=(),
    )


if __name__ == "__main__":
    unittest.main()
