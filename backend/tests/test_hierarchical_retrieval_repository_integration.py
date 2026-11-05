from __future__ import annotations

import os
import unittest
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

import asyncpg

from app.memory.entities.repository import EntityRepository
from app.memory.models import EMBEDDING_DIMENSIONS
from app.memory.repository import MemoryRepository
from app.memory.retrieval.query import MemoryRetrievalQuery, normalize_memory_query
from app.memory.semantic.contracts import ENTITY_IDENTITY_EMBEDDING_KIND


@unittest.skipUnless(
    os.getenv("JARVIS_TEST_DB_URL"),
    "JARVIS_TEST_DB_URL is required for retrieval repository integration tests",
)
class HierarchicalRetrievalRepositoryIntegrationTests(
    unittest.IsolatedAsyncioTestCase
):
    async def asyncSetUp(self) -> None:
        self.owner_id = f"retrieval-integration-{uuid4()}"
        self.now = datetime.now(UTC)
        self.pool = await asyncpg.create_pool(
            os.environ["JARVIS_TEST_DB_URL"], min_size=1, max_size=2
        )
        async with self.pool.acquire() as conn:
            self.session_id = await conn.fetchval(
                """
                insert into jarvis_sessions (owner_id, title)
                values ($1, 'Retrieval integration')
                returning id
                """,
                self.owner_id,
            )
        self.repository = MemoryRepository(self.pool)

    async def asyncTearDown(self) -> None:
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    "delete from memory_retrieval_runs where owner_id = $1",
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

    async def test_active_and_historical_queries_respect_memory_lifecycle(self) -> None:
        current_id = await self._insert_memory("Current fact")
        replacement_id = await self._insert_memory("Replacement fact")
        expired_id = await self._insert_memory(
            "Expired active fact",
            valid_from=self.now - timedelta(hours=2),
            valid_to=self.now - timedelta(hours=1),
        )
        superseded_id = await self._insert_memory(
            "Superseded fact",
            status="superseded",
            valid_from=self.now - timedelta(hours=2),
            valid_to=self.now - timedelta(hours=1),
            superseded_by_memory_id=replacement_id,
        )
        retracted_id = await self._insert_memory(
            "Retracted fact",
            status="retracted",
            valid_from=self.now - timedelta(hours=2),
            valid_to=self.now - timedelta(hours=1),
        )
        forgotten_id = await self._insert_memory(
            "Forgotten fact",
            status="forgotten",
        )

        active_hits = await self.repository.list_time_retrieval_hits(
            owner_id=self.owner_id,
            query=self._query(),
            limit=40,
        )
        historical_hits = await self.repository.list_time_retrieval_hits(
            owner_id=self.owner_id,
            query=self._query(include_superseded=True),
            limit=40,
        )

        self.assertEqual(
            {hit.document.id for hit in active_hits},
            {current_id, replacement_id},
        )
        self.assertEqual(
            {hit.document.id for hit in historical_hits},
            {
                current_id,
                replacement_id,
                expired_id,
                superseded_id,
                retracted_id,
            },
        )
        self.assertNotIn(forgotten_id, {hit.document.id for hit in historical_hits})

    async def test_unicode_lexical_fallback_matches_without_embeddings(self) -> None:
        memory_id = await self._insert_memory(
            "東京駅のポスター",
            content="東京駅で青い案内ポスターを見た。",
        )

        hits = await self.repository.list_lexical_retrieval_hits(
            owner_id=self.owner_id,
            query=self._query(query="東京"),
            limit=40,
        )

        self.assertIn(memory_id, {hit.document.id for hit in hits})

    async def test_entity_embedding_limit_is_applied_after_session_scope(self) -> None:
        async with self.pool.acquire() as conn:
            other_session_id = await conn.fetchval(
                """
                insert into jarvis_sessions (owner_id, title)
                values ($1, 'Out-of-scope retrieval integration')
                returning id
                """,
                self.owner_id,
            )
        relevant_memory = await self.repository.create_memory_item(
            owner_id=self.owner_id,
            session_id=self.session_id,
            memory_type="semantic_fact",
            title="Relevant person",
            content="The relevant person was in this session.",
            occurred_at=self.now,
            importance=0.7,
            confidence=0.9,
            source="integration_test",
        )
        distractor_memory = await self.repository.create_memory_item(
            owner_id=self.owner_id,
            session_id=other_session_id,
            memory_type="semantic_fact",
            title="Out-of-scope person",
            content="The closest entity belongs to another session.",
            occurred_at=self.now,
            importance=0.7,
            confidence=0.9,
            source="integration_test",
        )
        entities = EntityRepository(self.pool)
        relevant = await entities.create_entity(
            owner_id=self.owner_id,
            entity_type="person",
            canonical_name="Relevant Person",
            confidence=0.9,
        )
        distractor = await entities.create_entity(
            owner_id=self.owner_id,
            entity_type="person",
            canonical_name="Distractor Person",
            confidence=0.9,
        )
        await entities.attach_mention(
            owner_id=self.owner_id,
            entity_id=relevant.id,
            memory_item_id=relevant_memory.id,
            role="subject",
            confidence=0.9,
        )
        await entities.attach_mention(
            owner_id=self.owner_id,
            entity_id=distractor.id,
            memory_item_id=distractor_memory.id,
            role="subject",
            confidence=0.9,
        )
        query_vector = _vector(1.0, 0.0)
        await entities.upsert_embedding(
            owner_id=self.owner_id,
            entity_id=distractor.id,
            embedding_kind=ENTITY_IDENTITY_EMBEDDING_KIND,
            provider="test",
            model="retrieval",
            embedded_content="Distractor Person",
            embedding=query_vector,
        )
        await entities.upsert_embedding(
            owner_id=self.owner_id,
            entity_id=relevant.id,
            embedding_kind=ENTITY_IDENTITY_EMBEDDING_KIND,
            provider="test",
            model="retrieval",
            embedded_content="Relevant Person",
            embedding=_vector(0.8, 0.6),
        )

        hits = await self.repository.list_entity_embedding_retrieval_hits(
            owner_id=self.owner_id,
            query=self._query(query="person"),
            embedding=query_vector,
            provider="test",
            model="retrieval",
            limit=1,
        )

        self.assertEqual([hit.document.id for hit in hits], [relevant_memory.id])

    async def test_temporal_adjacency_returns_nearest_before_and_after(self) -> None:
        before_id = await self._insert_memory(
            "Before", occurred_at=self.now - timedelta(minutes=3)
        )
        anchor_id = await self._insert_memory(
            "Anchor", occurred_at=self.now - timedelta(minutes=2)
        )
        after_id = await self._insert_memory(
            "After", occurred_at=self.now - timedelta(minutes=1)
        )

        hits = await self.repository.list_adjacent_retrieval_hits(
            owner_id=self.owner_id,
            query=self._query(),
            anchor_ids=[anchor_id],
            limit=10,
        )

        self.assertEqual({hit.document.id for hit in hits}, {before_id, after_id})

    def _query(
        self,
        *,
        query: str | None = None,
        include_superseded: bool = False,
    ):
        return normalize_memory_query(
            MemoryRetrievalQuery(
                query=query,
                seconds_before_now=86_400,
                memory_types=("semantic_fact",),
                session_id=self.session_id,
                include_superseded=include_superseded,
            ),
            now=self.now + timedelta(seconds=1),
            default_seconds=600,
            max_days=3650,
        )

    async def _insert_memory(
        self,
        title: str,
        *,
        content: str | None = None,
        status: str = "active",
        occurred_at: datetime | None = None,
        valid_from: datetime | None = None,
        valid_to: datetime | None = None,
        superseded_by_memory_id: UUID | None = None,
    ) -> UUID:
        async with self.pool.acquire() as conn:
            return await conn.fetchval(
                """
                insert into memory_items (
                    owner_id, session_id, memory_type, status, title, content,
                    occurred_at, valid_from, valid_to, importance, confidence,
                    source, superseded_by_memory_id
                )
                values (
                    $1, $2, 'semantic_fact', $3, $4, $5, $6, $7, $8,
                    0.7, 0.9, 'integration_test', $9
                )
                returning id
                """,
                self.owner_id,
                self.session_id,
                status,
                title,
                content or title,
                occurred_at or self.now - timedelta(minutes=5),
                valid_from,
                valid_to,
                superseded_by_memory_id,
            )


def _vector(first: float, second: float) -> tuple[float, ...]:
    return (first, second) + (0.0,) * (EMBEDDING_DIMENSIONS - 2)
