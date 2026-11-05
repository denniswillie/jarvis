from __future__ import annotations

import os
import unittest
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import asyncpg

from app.memory.entities.repository import EntityRepository
from app.memory.models import EMBEDDING_DIMENSIONS
from app.memory.repository import MemoryRepository


@unittest.skipUnless(
    os.getenv("JARVIS_TEST_DB_URL"),
    "JARVIS_TEST_DB_URL is required for entity repository integration tests",
)
class EntityRepositoryIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.owner_id = f"entity-integration-{uuid4()}"
        self.pool = await asyncpg.create_pool(
            os.environ["JARVIS_TEST_DB_URL"],
            min_size=1,
            max_size=2,
        )

    async def asyncTearDown(self) -> None:
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    "delete from memory_retrieval_runs where owner_id = $1",
                    self.owner_id,
                )
                await conn.execute(
                    "delete from memory_entity_relations where owner_id = $1",
                    self.owner_id,
                )
                await conn.execute(
                    "delete from memory_entity_mentions where owner_id = $1",
                    self.owner_id,
                )
                await conn.execute(
                    "delete from memory_entities where owner_id = $1",
                    self.owner_id,
                )
                await conn.execute(
                    "delete from memory_items where owner_id = $1",
                    self.owner_id,
                )
                await conn.execute(
                    "delete from jarvis_sessions where owner_id = $1",
                    self.owner_id,
                )
        await self.pool.close()

    async def test_entity_graph_and_retrieval_audit_execute_against_postgres(self) -> None:
        now = datetime.now(UTC)
        async with self.pool.acquire() as conn:
            session_id = await conn.fetchval(
                """
                insert into jarvis_sessions (owner_id, title)
                values ($1, 'Entity integration')
                returning id
                """,
                self.owner_id,
            )

        memory_repository = MemoryRepository(self.pool)
        supporting_memory = await memory_repository.create_memory_item(
            owner_id=self.owner_id,
            session_id=session_id,
            memory_type="semantic_fact",
            title="Location observation",
            content="Dennis moved from the kitchen to the office.",
            importance=0.7,
            confidence=0.9,
            source="integration_test",
        )
        entity_repository = EntityRepository(self.pool)
        dennis = await entity_repository.create_entity(
            owner_id=self.owner_id,
            entity_type="person",
            canonical_name="Dennis",
            aliases=["Dennis Willie", "dennis willie"],
            confidence=0.95,
            external_namespace="contacts",
            external_id="dennis",
        )
        other_dennis = await entity_repository.create_entity(
            owner_id=self.owner_id,
            entity_type="person",
            canonical_name="Dennis",
            confidence=0.5,
        )
        kitchen = await entity_repository.create_entity(
            owner_id=self.owner_id,
            entity_type="place",
            canonical_name="Kitchen",
            confidence=0.9,
        )
        office = await entity_repository.create_entity(
            owner_id=self.owner_id,
            entity_type="place",
            canonical_name="Office",
            confidence=0.9,
        )

        same_name = await entity_repository.find_by_name(
            owner_id=self.owner_id,
            name="DENNIS",
        )
        self.assertEqual({item.id for item in same_name}, {dennis.id, other_dennis.id})

        mention = await entity_repository.attach_mention(
            owner_id=self.owner_id,
            entity_id=dennis.id,
            memory_item_id=supporting_memory.id,
            role="subject",
            confidence=0.95,
        )
        vector = (0.01,) * EMBEDDING_DIMENSIONS
        embedding = await entity_repository.upsert_embedding(
            owner_id=self.owner_id,
            entity_id=dennis.id,
            embedding_kind="identity",
            provider="openai",
            model="text-embedding-3-small",
            embedded_content="Dennis | Dennis Willie",
            embedding=vector,
        )
        original = await entity_repository.create_relation(
            owner_id=self.owner_id,
            subject_entity_id=dennis.id,
            predicate="located_at",
            object_entity_id=kitchen.id,
            confidence=0.9,
            valid_from=now,
            supporting_memory_id=supporting_memory.id,
        )
        replacement = await entity_repository.create_relation(
            owner_id=self.owner_id,
            subject_entity_id=dennis.id,
            predicate="located_at",
            object_entity_id=office.id,
            confidence=0.9,
            valid_from=now + timedelta(minutes=5),
            supporting_memory_id=supporting_memory.id,
        )
        superseded = await entity_repository.supersede_relation(
            owner_id=self.owner_id,
            relation_id=original.id,
            replacement_relation_id=replacement.id,
        )
        value_relation = await entity_repository.create_relation(
            owner_id=self.owner_id,
            subject_entity_id=dennis.id,
            predicate="prefers",
            object_value={"drink": "tea"},
            confidence=0.8,
            valid_from=now,
            supporting_memory_id=supporting_memory.id,
        )
        audit_id = await memory_repository.record_retrieval_run(
            owner_id=self.owner_id,
            session_id=session_id,
            query="Where was Dennis?",
            constraints={"entity_ids": [str(dennis.id)]},
            candidate_ids=[supporting_memory.id],
            result_ids=[supporting_memory.id],
            reinspection_performed=False,
            provider_spaces=["openai:text-embedding-3-small"],
            latency_ms=12,
        )

        self.assertEqual(mention.entity_id, dennis.id)
        self.assertEqual(embedding.entity_id, dennis.id)
        self.assertEqual(superseded.status, "superseded")
        self.assertEqual(superseded.superseded_by_relation_id, replacement.id)
        self.assertEqual(value_relation.object_value, {"drink": "tea"})
        async with self.pool.acquire() as conn:
            stored_audit = await conn.fetchrow(
                "select * from memory_retrieval_runs where id = $1",
                audit_id,
            )
        self.assertIsNotNone(stored_audit)
        self.assertEqual(stored_audit["candidate_ids"], [supporting_memory.id])
