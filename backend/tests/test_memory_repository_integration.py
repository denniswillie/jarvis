from __future__ import annotations

import os
import asyncio
import unittest
from datetime import UTC, datetime
from uuid import uuid4

import asyncpg

from app.memory.jobs.repository import MemoryJobRepository
from app.memory.models import (
    EMBEDDING_DIMENSIONS,
    ExplicitMemoryEvidenceInput,
)
from app.memory.repository import MemoryMissingError, MemoryRepository


@unittest.skipUnless(
    os.getenv("JARVIS_TEST_DB_URL"),
    "JARVIS_TEST_DB_URL is required for memory repository integration tests",
)
class MemoryRepositoryIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.owner_id = f"memory-integration-{uuid4()}"
        self.pool = await asyncpg.create_pool(
            os.environ["JARVIS_TEST_DB_URL"],
            min_size=1,
            max_size=2,
        )

    async def asyncTearDown(self) -> None:
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    "delete from memory_forgetting_tombstones where owner_id = $1",
                    self.owner_id,
                )
                await conn.execute(
                    """
                    update experience_events
                    set promotion_decision = 'pending',
                        promotion_reason = null,
                        promoted_episode_memory_id = null,
                        promotion_decided_at = null
                    where owner_id = $1
                    """,
                    self.owner_id,
                )
                await conn.execute(
                    "delete from memory_items where owner_id = $1",
                    self.owner_id,
                )
                await conn.execute(
                    "delete from memory_jobs where owner_id = $1",
                    self.owner_id,
                )
                await conn.execute(
                    "delete from memory_entities where owner_id = $1",
                    self.owner_id,
                )
                await conn.execute(
                    "delete from calendar_event_snapshots where owner_id = $1",
                    self.owner_id,
                )
                await conn.execute(
                    "delete from jarvis_sessions where owner_id = $1",
                    self.owner_id,
                )
        await self.pool.close()

    async def test_durable_memory_flow_executes_against_postgres(self) -> None:
        now = datetime.now(UTC)
        async with self.pool.acquire() as conn:
            session_id = await conn.fetchval(
                """
                insert into jarvis_sessions (owner_id, title)
                values ($1, 'Memory integration')
                returning id
                """,
                self.owner_id,
            )
            event_id = await conn.fetchval(
                """
                insert into experience_events (
                    owner_id,
                    session_id,
                    status,
                    event_kind,
                    modalities,
                    started_at,
                    ended_at,
                    boundary_reason,
                    keyframes_finalized_at,
                    consolidation_ready_at
                )
                values (
                    $1, $2, 'closed', 'visual', array['visual'],
                    $3, $3, 'integration_test', $3, $3
                )
                returning id
                """,
                self.owner_id,
                session_id,
                now,
            )

        repository = MemoryRepository(self.pool)
        episode = await repository.promote_event_to_episode(
            owner_id=self.owner_id,
            event_id=event_id,
            title="Desk work",
            content="The user worked at a desk.",
            summary="Desk work",
            importance=0.7,
            confidence=0.9,
            source="integration_test",
            promotion_reason="durable event",
        )
        fact = await repository.create_memory_item(
            owner_id=self.owner_id,
            session_id=session_id,
            memory_type="semantic_fact",
            title="Workspace preference",
            content="The user often works at a desk.",
            importance=0.6,
            confidence=0.8,
            source="integration_test",
        )
        snapshot = await repository.record_calendar_event_snapshot(
            owner_id=self.owner_id,
            provider="google",
            calendar_id="primary",
            provider_event_id="event-1",
            event_title="Focus block",
            starts_at=now,
            ends_at=now,
            observed_at=now,
            snapshot={"summary": "Focus block", "status": "confirmed"},
        )
        await repository.attach_evidence(
            owner_id=self.owner_id,
            memory_item_id=fact.id,
            evidence_type="calendar_event",
            evidence_ref_id=snapshot.id,
            evidence_time=now,
            excerpt="Focus block",
            confidence=1.0,
        )
        await repository.create_link(
            owner_id=self.owner_id,
            source_memory_id=episode.id,
            target_memory_id=fact.id,
            relationship="supports",
            confidence=0.9,
            valid_from=now,
            supporting_memory_id=episode.id,
        )
        vector = (0.01,) * EMBEDDING_DIMENSIONS
        await repository.upsert_event_embedding(
            owner_id=self.owner_id,
            event_id=event_id,
            embedding_kind="semantic",
            provider="openai",
            model="text-embedding-3-small",
            embedded_content="Desk work event",
            embedding=vector,
        )
        await repository.upsert_memory_embedding(
            owner_id=self.owner_id,
            memory_item_id=fact.id,
            embedding_kind="semantic",
            provider="openai",
            model="text-embedding-3-small",
            embedded_content=fact.content,
            embedding=vector,
        )
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                insert into memory_jobs (
                    owner_id, session_id, job_type, dedupe_key, payload
                )
                values ($1, $2, 'embed_memory_item', $3, $4::jsonb)
                """,
                self.owner_id,
                session_id,
                f"memory:{fact.id}:semantic",
                '{"memory_item_id":"' + str(fact.id) + '"}',
            )

        with self.assertRaises(MemoryMissingError):
            await repository.forget_derived_memory(
                owner_id=f"other-{self.owner_id}",
                memory_id=fact.id,
                reason="Cross-owner request must not reveal the memory.",
            )

        forgotten_fact = await repository.forget_derived_memory(
            owner_id=self.owner_id,
            memory_id=fact.id,
            reason="The user no longer wants this derived belief recalled.",
        )
        replayed_fact = await repository.forget_derived_memory(
            owner_id=self.owner_id,
            memory_id=fact.id,
            reason="Repeated request.",
        )
        suppressed_recreation = await repository.create_memory_item(
            owner_id=self.owner_id,
            session_id=session_id,
            memory_type="semantic_fact",
            title="Workspace preference recreated automatically",
            content="The user often works at a desk.",
            importance=0.6,
            confidence=0.8,
            source="integration_test",
        )
        explicit_override = await repository.create_memory_item(
            owner_id=self.owner_id,
            session_id=session_id,
            memory_type="semantic_fact",
            title="Workspace preference explicitly restored",
            content="The user often works at a desk.",
            importance=0.6,
            confidence=1.0,
            source="explicit_user_request",
            is_explicit=True,
        )
        replay_after_override = await repository.forget_derived_memory(
            owner_id=self.owner_id,
            memory_id=fact.id,
            reason="Repeated request after explicit override.",
        )
        forgotten = await repository.forget_memory(
            owner_id=self.owner_id,
            memory_id=episode.id,
        )
        replayed = await repository.promote_event_to_episode(
            owner_id=self.owner_id,
            event_id=event_id,
            title="Desk work",
            content="The user worked at a desk.",
            summary="Desk work",
            importance=0.7,
            confidence=0.9,
            source="integration_test",
            promotion_reason="durable event",
        )

        self.assertEqual(forgotten_fact.memory.status, "forgotten")
        self.assertFalse(forgotten_fact.replayed)
        self.assertEqual(forgotten_fact.removed_embedding_count, 1)
        self.assertEqual(forgotten_fact.closed_memory_link_count, 1)
        self.assertEqual(forgotten_fact.cancelled_embedding_job_count, 1)
        self.assertTrue(replayed_fact.replayed)
        self.assertEqual(replayed_fact.tombstone.id, forgotten_fact.tombstone.id)
        self.assertEqual(suppressed_recreation.id, fact.id)
        self.assertEqual(suppressed_recreation.status, "forgotten")
        self.assertNotEqual(explicit_override.id, fact.id)
        self.assertEqual(explicit_override.status, "active")
        self.assertTrue(replay_after_override.replayed)
        self.assertEqual(
            replay_after_override.tombstone.id,
            forgotten_fact.tombstone.id,
        )
        self.assertEqual(forgotten.status, "forgotten")
        self.assertEqual(replayed.id, episode.id)
        self.assertEqual(replayed.status, "forgotten")

        async with self.pool.acquire() as conn:
            evidence_count = await conn.fetchval(
                "select count(*) from memory_evidence where memory_item_id = $1",
                fact.id,
            )
            snapshot_count = await conn.fetchval(
                "select count(*) from calendar_event_snapshots where id = $1",
                snapshot.id,
            )
            embedding_count = await conn.fetchval(
                "select count(*) from memory_item_embeddings where memory_item_id = $1",
                fact.id,
            )
            link_is_closed = await conn.fetchval(
                "select valid_to is not null from memory_links where owner_id = $1 and target_memory_id = $2",
                self.owner_id,
                fact.id,
            )
            job_status = await conn.fetchval(
                """
                select status from memory_jobs
                where owner_id = $1
                  and payload->>'memory_item_id' = $2
                """,
                self.owner_id,
                str(fact.id),
            )
            tombstone = await conn.fetchrow(
                """
                select overridden_at, override_memory_id, suppressed_recreation_count
                from memory_forgetting_tombstones
                where owner_id = $1 and source_memory_id = $2
                """,
                self.owner_id,
                fact.id,
            )
        self.assertEqual(evidence_count, 1)
        self.assertEqual(snapshot_count, 1)
        self.assertEqual(embedding_count, 0)
        self.assertTrue(link_is_closed)
        self.assertEqual(job_status, "dead")
        self.assertIsNotNone(tombstone["overridden_at"])
        self.assertEqual(tombstone["override_memory_id"], explicit_override.id)
        self.assertEqual(tombstone["suppressed_recreation_count"], 1)

    async def test_explicit_memory_retry_is_atomic_and_idempotent(self) -> None:
        now = datetime.now(UTC)
        async with self.pool.acquire() as conn:
            session_id = await conn.fetchval(
                """
                insert into jarvis_sessions (owner_id, title)
                values ($1, 'Explicit memory integration')
                returning id
                """,
                self.owner_id,
            )
            message_id = await conn.fetchval(
                """
                insert into jarvis_messages (
                    owner_id, session_id, role, modality, content
                )
                values ($1, $2, 'user', 'text', 'Remember the blue station poster.')
                returning id
                """,
                self.owner_id,
                session_id,
            )
            entity_id = await conn.fetchval(
                """
                insert into memory_entities (
                    owner_id, entity_type, canonical_name, normalized_name,
                    aliases, description, confidence
                )
                values ($1, 'place', 'Station', 'station', '{}', '', 1.0)
                returning id
                """,
                self.owner_id,
            )

        repository = MemoryRepository(self.pool)
        jobs = MemoryJobRepository(self.pool)

        async def write_memory():
            return await repository.create_explicit_memory(
                owner_id=self.owner_id,
                session_id=session_id,
                memory_type="semantic_fact",
                title="Remembered fact: station poster",
                content="The station poster is blue.",
                occurred_at=now,
                idempotency_key="provider-call-explicit-memory",
                evidence=(
                    ExplicitMemoryEvidenceInput(
                        evidence_type="message",
                        evidence_ref_id=str(message_id),
                        evidence_time=now,
                        excerpt="Remember the blue station poster.",
                    ),
                ),
                entity_ids=(entity_id,),
                metadata={"entity_references": ["Station"]},
                job_repository=jobs,
            )

        first, second = await asyncio.gather(write_memory(), write_memory())

        self.assertEqual(first.memory.id, second.memory.id)
        self.assertEqual({first.replayed, second.replayed}, {False, True})
        async with self.pool.acquire() as conn:
            memory_count = await conn.fetchval(
                """
                select count(*) from memory_items
                where owner_id = $1
                  and metadata->>'explicit_idempotency_key' = $2
                """,
                self.owner_id,
                "provider-call-explicit-memory",
            )
            mention_count = await conn.fetchval(
                """
                select count(*) from memory_entity_mentions
                where owner_id = $1 and memory_item_id = $2
                """,
                self.owner_id,
                first.memory.id,
            )
            job_count = await conn.fetchval(
                """
                select count(*) from memory_jobs
                where owner_id = $1 and job_type = 'embed_memory_item'
                """,
                self.owner_id,
            )
        self.assertEqual(memory_count, 1)
        self.assertEqual(mention_count, 1)
        self.assertEqual(job_count, 1)
