from __future__ import annotations

import unittest
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

from app.memory.consolidation.contracts import ConsolidationResult
from app.memory.repository import MemoryRepository
from app.memory.semantic.contracts import SemanticEvidence, SemanticReconciliationContext
from app.memory.semantic.job_handler import SemanticMemoryRecoveryRule
from app.memory.semantic.reconciler import SemanticMemoryReconciler
from test_consolidation_contracts import _output, _request


class SemanticFactPromotionTests(unittest.TestCase):
    def test_explicit_user_statement_becomes_an_evidence_backed_fact(self) -> None:
        context = _context()

        plan = SemanticMemoryReconciler().plan(context)

        self.assertEqual(len(plan.facts), 1)
        fact = plan.facts[0]
        self.assertTrue(fact.is_explicit)
        self.assertEqual(fact.source, "user_statement")
        self.assertEqual(fact.subject.external_namespace, "jarvis.owner")
        self.assertEqual(fact.subject.external_id, context.owner_id)
        self.assertEqual(fact.evidence[0].evidence_ref_id, context.evidence[0].evidence_ref_id)

    def test_weak_inferred_fact_is_rejected(self) -> None:
        context = _context(
            fact_updates={"explicitly_user_stated": False, "confidence": 0.70}
        )

        self.assertEqual(SemanticMemoryReconciler().plan(context).facts, ())

    def test_inferred_preference_is_not_promoted(self) -> None:
        context = _context(
            fact_updates={
                "predicate": "prefers",
                "explicitly_user_stated": False,
                "confidence": 0.99,
            }
        )

        self.assertEqual(SemanticMemoryReconciler().plan(context).facts, ())

    def test_explicit_preference_remains_evidence_backed(self) -> None:
        context = _context(fact_updates={"predicate": "prefers"})

        facts = SemanticMemoryReconciler().plan(context).facts

        self.assertEqual(len(facts), 1)
        self.assertEqual(facts[0].predicate, "prefers")

    def test_fact_without_matching_episode_evidence_is_rejected(self) -> None:
        context = _context(include_evidence=False)

        self.assertEqual(SemanticMemoryReconciler().plan(context).facts, ())

    def test_invalid_temporal_hint_is_rejected(self) -> None:
        context = _context(fact_updates={"valid_from_hint": "not-a-timestamp"})

        self.assertEqual(SemanticMemoryReconciler().plan(context).facts, ())

    def test_ambiguous_same_name_people_are_not_merged(self) -> None:
        context = _context(duplicate_person_name="Alex Smith")

        plan = SemanticMemoryReconciler().plan(context)

        self.assertFalse(
            any(entity.normalized_name == "alex smith" for entity in plan.entities)
        )
        self.assertEqual(plan.facts, ())

    def test_inferred_relation_keeps_inferred_threshold_with_user_message(self) -> None:
        context = _context(
            relation_updates={"explicitly_user_stated": False, "confidence": 0.70}
        )

        self.assertEqual(SemanticMemoryReconciler().plan(context).relations, ())

    def test_explicit_relation_uses_explicit_threshold(self) -> None:
        context = _context(
            relation_updates={"explicitly_user_stated": True, "confidence": 0.70}
        )

        relations = SemanticMemoryReconciler().plan(context).relations

        self.assertEqual(len(relations), 1)
        self.assertEqual(relations[0].source, "user_statement")


class SemanticRecoveryTests(unittest.IsolatedAsyncioTestCase):
    async def test_embedded_unreconciled_episode_is_recovered(self) -> None:
        owner_id = "owner"
        event_id = uuid4()
        session_id = uuid4()
        connection = AsyncMock()
        connection.fetch.return_value = [
            {"owner_id": owner_id, "session_id": session_id, "id": event_id}
        ]
        repository = AsyncMock()
        repository.enqueue.return_value.created = True

        recovered = await SemanticMemoryRecoveryRule().reconcile(
            repository=repository,
            connection=connection,
            now=datetime.now(UTC),
        )

        self.assertEqual(recovered, 1)
        self.assertEqual(
            repository.enqueue.await_args.kwargs["job_type"],
            "reconcile_semantic_memory",
        )
        self.assertIs(repository.enqueue.await_args.kwargs["connection"], connection)


class SemanticFactRepositoryTests(unittest.IsolatedAsyncioTestCase):
    async def test_active_facts_exclude_ended_validity_intervals(self) -> None:
        pool = MagicMock()
        connection = AsyncMock()
        pool.acquire.return_value.__aenter__.return_value = connection
        connection.fetch.return_value = []

        await MemoryRepository(pool).list_active_semantic_facts(owner_id="owner")

        query = connection.fetch.await_args.args[0]
        self.assertIn("valid_to is null or valid_to > now()", query)


def _context(
    *,
    fact_updates: dict[str, object] | None = None,
    include_evidence: bool = True,
    relation_updates: dict[str, object] | None = None,
    duplicate_person_name: str | None = None,
) -> SemanticReconciliationContext:
    request = _request()
    payload = _output(request)
    if fact_updates:
        payload["semantic_fact_candidates"][0].update(fact_updates)  # type: ignore[index,union-attr]
    citation = payload["semantic_fact_candidates"][0]["evidence"][0]  # type: ignore[index]
    if relation_updates is not None:
        relation = {
            "source_entity": "User",
            "predicate": "located_at",
            "target_entity": "Kitchen",
            "confidence": 0.95,
            "explicitly_user_stated": False,
            "valid_from_hint": None,
            "valid_to_hint": None,
            "evidence": [citation],
        }
        relation.update(relation_updates)
        payload["relation_candidates"] = [relation]
    if duplicate_person_name is not None:
        person = {
            "name": duplicate_person_name,
            "category": "person",
            "confidence": 0.95,
            "evidence": [citation],
        }
        payload["entity_candidates"] = [person, dict(person)]
        payload["semantic_fact_candidates"][0]["subject"] = duplicate_person_name  # type: ignore[index]
    result = ConsolidationResult.model_validate(payload)
    source = request.evidence[0]
    evidence = (
        SemanticEvidence(
            evidence_type=source.evidence_type,
            evidence_ref_id=source.evidence_ref_id,
            occurred_at=source.occurred_at,
            excerpt="I keep my silver keys on the kitchen counter",
            message_role="user",
        ),
    )
    return SemanticReconciliationContext(
        owner_id=request.event.owner_id,
        session_id=request.event.session_id,
        event_id=request.event.id,
        episode_memory_id=uuid4(),
        occurred_at=request.event.started_at,
        result=result,
        evidence=evidence if include_evidence else (),
    )
