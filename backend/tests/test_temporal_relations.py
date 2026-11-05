from __future__ import annotations

import unittest
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock
from uuid import uuid4

from app.memory.semantic.contracts import (
    SemanticEntityIdentity,
    SemanticEvidence,
    SemanticRelationProposal,
)
from app.memory.semantic.repository import SemanticMemoryRepository


class TemporalRelationTests(unittest.IsolatedAsyncioTestCase):
    async def test_location_replacement_closes_previous_validity_interval(self) -> None:
        connection = AsyncMock()
        previous_id = uuid4()
        previous_target = uuid4()
        now = datetime.now(UTC)
        connection.fetch.return_value = [
            {
                "id": previous_id,
                "object_entity_id": previous_target,
                "valid_from": now - timedelta(minutes=5),
            }
        ]
        replacement_id = uuid4()
        connection.fetchval.side_effect = [True, replacement_id]
        repository = SemanticMemoryRepository(AsyncMock())

        created = await repository.apply_relation(
            connection,
            owner_id="owner",
            supporting_memory_id=uuid4(),
            subject_entity_id=uuid4(),
            object_entity_id=uuid4(),
            proposal=_proposal("located_at", now),
        )

        self.assertTrue(created)
        self.assertEqual(connection.execute.await_args.args[1], "owner")
        self.assertEqual(connection.execute.await_args.args[2], previous_id)
        self.assertEqual(connection.execute.await_args.args[3], now)
        self.assertEqual(connection.execute.await_args.args[4], replacement_id)
        self.assertIn(
            "pg_advisory_xact_lock",
            connection.execute.await_args_list[0].args[0],
        )

    async def test_nonexclusive_relation_preserves_existing_active_relation(self) -> None:
        connection = AsyncMock()
        connection.fetch.return_value = [
            {"id": uuid4(), "object_entity_id": uuid4(), "valid_from": datetime.now(UTC)}
        ]
        connection.fetchval.side_effect = [True, uuid4()]

        created = await SemanticMemoryRepository(AsyncMock()).apply_relation(
            connection,
            owner_id="owner",
            supporting_memory_id=uuid4(),
            subject_entity_id=uuid4(),
            object_entity_id=uuid4(),
            proposal=_proposal("owns", datetime.now(UTC)),
        )

        self.assertTrue(created)
        self.assertEqual(connection.execute.await_count, 1)
        self.assertIn("pg_advisory_xact_lock", connection.execute.await_args.args[0])

    async def test_duplicate_active_target_is_idempotent(self) -> None:
        connection = AsyncMock()
        target_id = uuid4()
        connection.fetch.return_value = [
            {"id": uuid4(), "object_entity_id": target_id, "valid_from": datetime.now(UTC)}
        ]
        connection.fetchval.return_value = True

        created = await SemanticMemoryRepository(AsyncMock()).apply_relation(
            connection,
            owner_id="owner",
            supporting_memory_id=uuid4(),
            subject_entity_id=uuid4(),
            object_entity_id=target_id,
            proposal=_proposal("located_at", datetime.now(UTC)),
        )

        self.assertFalse(created)
        self.assertEqual(connection.fetchval.await_count, 1)

    async def test_older_location_cannot_replace_a_newer_location(self) -> None:
        connection = AsyncMock()
        now = datetime.now(UTC)
        connection.fetch.return_value = [
            {"id": uuid4(), "object_entity_id": uuid4(), "valid_from": now}
        ]
        connection.fetchval.return_value = True

        created = await SemanticMemoryRepository(AsyncMock()).apply_relation(
            connection,
            owner_id="owner",
            supporting_memory_id=uuid4(),
            subject_entity_id=uuid4(),
            object_entity_id=uuid4(),
            proposal=_proposal("located_at", now - timedelta(minutes=1)),
        )

        self.assertFalse(created)
        self.assertEqual(connection.fetchval.await_count, 1)

    async def test_cross_owner_relation_references_are_rejected(self) -> None:
        connection = AsyncMock()
        connection.fetchval.return_value = False

        with self.assertRaisesRegex(ValueError, "outside the owner scope"):
            await SemanticMemoryRepository(AsyncMock()).apply_relation(
                connection,
                owner_id="owner",
                supporting_memory_id=uuid4(),
                subject_entity_id=uuid4(),
                object_entity_id=uuid4(),
                proposal=_proposal("located_at", datetime.now(UTC)),
            )

        connection.fetch.assert_not_awaited()

    async def test_cross_owner_mentions_are_rejected(self) -> None:
        connection = AsyncMock()
        connection.fetchval.return_value = False

        with self.assertRaisesRegex(ValueError, "outside the owner scope"):
            await SemanticMemoryRepository._attach_mention(
                connection,
                owner_id="owner",
                entity_id=uuid4(),
                memory_item_id=uuid4(),
                role="subject",
                confidence=0.9,
            )


def _proposal(predicate: str, valid_from: datetime) -> SemanticRelationProposal:
    return SemanticRelationProposal(
        subject=SemanticEntityIdentity(name="User", entity_type="person", confidence=0.95),
        predicate=predicate,
        target=SemanticEntityIdentity(name="Office", entity_type="place", confidence=0.95),
        confidence=0.95,
        source="inferred_observation",
        valid_from=valid_from,
        valid_to=None,
        evidence=(
            SemanticEvidence(
                evidence_type="observation",
                evidence_ref_id=uuid4(),
                occurred_at=valid_from,
                excerpt="Office visible",
            ),
        ),
    )
