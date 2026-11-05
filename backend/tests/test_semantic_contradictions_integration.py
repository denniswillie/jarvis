from __future__ import annotations

import asyncio
import os
import unittest
from dataclasses import replace
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import asyncpg

from app.memory.entities.repository import EntityRepository
from app.memory.jobs.repository import MemoryJobRepository
from app.memory.repository import MemoryRepository
from app.memory.semantic.contracts import (
    SemanticEntityIdentity,
    SemanticFactProposal,
    SemanticReconciliationPlan,
)
from app.memory.semantic.job_handler import (
    PreparedSemanticReconciliation,
    ReconcileSemanticMemoryJobHandler,
)
from app.memory.semantic.repository import SemanticMemoryRepository


@unittest.skipUnless(
    os.getenv("JARVIS_TEST_DB_URL"),
    "JARVIS_TEST_DB_URL is required for semantic contradiction integration tests",
)
class SemanticContradictionIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.owner_id = f"semantic-contradiction-{uuid4()}"
        self.pool = await asyncpg.create_pool(
            os.environ["JARVIS_TEST_DB_URL"],
            min_size=1,
            max_size=6,
        )
        self.now = datetime.now(UTC)
        self.context = await self._new_context()
        self.session_id = self.context.session_id
        self.event_id = self.context.event_id
        self.subject = await EntityRepository(self.pool).create_entity(
            owner_id=self.owner_id,
            entity_type="person",
            canonical_name="Dennis",
            confidence=0.99,
            external_namespace="jarvis.owner",
            external_id=self.owner_id,
        )
        self.semantic_repository = SemanticMemoryRepository(
            self.pool,
            supersede_min_confidence=0.90,
        )
        self.jobs = MemoryJobRepository(self.pool)

    async def _new_context(self) -> SimpleNamespace:
        async with self.pool.acquire() as connection:
            session_id = await connection.fetchval(
                """
                insert into jarvis_sessions (owner_id, title)
                values ($1, 'Semantic contradiction integration')
                returning id
                """,
                self.owner_id,
            )
            event_id = await connection.fetchval(
                """
                insert into experience_events (
                    owner_id, session_id, status, event_kind, modalities,
                    started_at, ended_at, boundary_reason,
                    keyframes_finalized_at, consolidation_ready_at
                )
                values (
                    $1, $2, 'closed', 'interaction', array['text'],
                    $3, $3, 'integration_test', $3, $3
                )
                returning id
                """,
                self.owner_id,
                session_id,
                self.now,
            )
        memory_repository = MemoryRepository(self.pool)
        episode = await memory_repository.promote_event_to_episode(
            owner_id=self.owner_id,
            event_id=event_id,
            title="Semantic evidence",
            content="Semantic contradiction integration evidence.",
            summary="Semantic evidence",
            importance=0.8,
            confidence=0.95,
            source="integration_test",
            promotion_reason="semantic contradiction test",
        )
        return SimpleNamespace(
            owner_id=self.owner_id,
            session_id=session_id,
            event_id=event_id,
            episode_memory_id=episode.id,
            occurred_at=self.now,
            result=SimpleNamespace(episode_importance=0.8),
            evidence=(),
            suppressed_memory_fingerprints=frozenset(),
        )

    async def asyncTearDown(self) -> None:
        async with self.pool.acquire() as connection:
            async with connection.transaction():
                await connection.execute(
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
                await connection.execute(
                    "delete from memory_items where owner_id = $1",
                    self.owner_id,
                )
                await connection.execute(
                    "delete from memory_entities where owner_id = $1",
                    self.owner_id,
                )
                await connection.execute(
                    "delete from experience_events where owner_id = $1",
                    self.owner_id,
                )
                await connection.execute(
                    "delete from jarvis_sessions where owner_id = $1",
                    self.owner_id,
                )
                # Source deletions enqueue profile refreshes; remove derived rows last.
                await connection.execute(
                    "delete from memory_jobs where owner_id = $1",
                    self.owner_id,
                )
                await connection.execute(
                    "delete from memory_profile_projections where owner_id = $1",
                    self.owner_id,
                )
        await self.pool.close()

    async def test_replacement_retraction_and_history_are_atomic_and_enriched(self) -> None:
        kitchen, _ = await self._apply(_proposal("located_at", "kitchen", explicit=True))
        self.assertIsNotNone(kitchen)
        office, updated = await self._apply(
            _proposal(
                "located_at",
                "office",
                explicit=True,
                change_intent="replace",
                previous_value="kitchen",
            )
        )
        self.assertIsNotNone(office)
        self.assertEqual({item.status for item in updated}, {"superseded"})

        history = await MemoryRepository(self.pool).list_semantic_fact_history(
            owner_id=self.owner_id,
            subject_entity_id=self.subject.id,
            predicate="located_at",
        )
        self.assertEqual({entry.fact.status for entry in history}, {"active", "superseded"})
        self.assertTrue(all(entry.evidence for entry in history))
        self.assertTrue(
            any(
                link.relationship == "supersedes"
                for entry in history
                for link in entry.links
            )
        )

        _, retracted = await self._apply(
            _proposal(
                "located_at",
                "office",
                explicit=True,
                change_intent="retract",
            )
        )
        self.assertEqual({item.status for item in retracted}, {"retracted"})

    async def test_concurrent_exclusive_writes_leave_one_active_value(self) -> None:
        await asyncio.gather(
            self._apply(_proposal("located_at", "kitchen", explicit=True)),
            self._apply(_proposal("located_at", "office", explicit=True)),
        )
        rows = await self._fact_rows("located_at")
        self.assertEqual(sum(row["status"] == "active" for row in rows), 1)
        self.assertEqual(sum(row["status"] == "superseded" for row in rows), 1)

    async def test_concurrent_likes_and_dislikes_create_contradiction_link(self) -> None:
        await asyncio.gather(
            self._apply(_proposal("likes", "loud cafes", explicit=True)),
            self._apply(_proposal("dislikes", "loud cafes", explicit=True)),
        )
        async with self.pool.acquire() as connection:
            active_count = await connection.fetchval(
                """
                select count(*)
                from memory_items
                where owner_id = $1
                  and memory_type = 'semantic_fact'
                  and status = 'active'
                  and metadata->>'predicate' = any($2::text[])
                """,
                self.owner_id,
                ["likes", "dislikes"],
            )
            contradiction_count = await connection.fetchval(
                """
                select count(*)
                from memory_links
                where owner_id = $1 and relationship = 'contradicts'
                """,
                self.owner_id,
            )
        self.assertEqual(active_count, 2)
        self.assertEqual(contradiction_count, 1)

    async def test_unresolved_replacement_does_not_mutate_memory(self) -> None:
        await self._apply(_proposal("located_at", "kitchen", explicit=True))
        created, updated = await self._apply(
            _proposal(
                "located_at",
                "office",
                explicit=True,
                change_intent="replace",
                previous_value="bedroom",
            )
        )
        self.assertIsNone(created)
        self.assertEqual(updated, ())
        rows = await self._fact_rows("located_at")
        self.assertEqual(
            [(row["status"], row["value"]) for row in rows],
            [("active", "kitchen")],
        )

    async def test_transaction_rolls_back_when_downstream_enqueue_fails(self) -> None:
        failing_jobs = AsyncMock()
        failing_jobs.enqueue.side_effect = RuntimeError("embedding enqueue failed")
        with self.assertRaisesRegex(RuntimeError, "embedding enqueue failed"):
            await self._commit(
                self.context,
                _proposal("owns", "silver keys", explicit=True),
                job_repository=failing_jobs,
            )
        failing_jobs.enqueue.assert_awaited_once()
        self.assertEqual(await self._fact_rows("owns"), [])
        async with self.pool.acquire() as connection:
            reconciled_at = await connection.fetchval(
                "select semantic_reconciled_at from experience_events where owner_id = $1 and id = $2",
                self.owner_id,
                self.event_id,
            )
        self.assertIsNone(reconciled_at)

    async def test_cross_session_reinforcement_reuses_queued_and_completed_job(self) -> None:
        proposal = _proposal("owns", "silver keys", explicit=True)
        await self._commit(self.context, proposal)
        initial_jobs = await self._embedding_jobs()
        self.assertEqual(len(initial_jobs), 1)
        initial_job = initial_jobs[0]

        await self._commit(await self._new_context(), proposal)
        queued_jobs = await self._embedding_jobs()
        self.assertEqual(len(queued_jobs), 1)
        self.assertEqual(queued_jobs[0]["id"], initial_job["id"])
        self.assertEqual(queued_jobs[0]["session_id"], self.session_id)
        self.assertEqual(queued_jobs[0]["status"], "queued")

        # Claim only this test owner's job, never another user's queued work.
        claim_token = uuid4()
        async with self.pool.acquire() as connection:
            async with connection.transaction():
                await connection.execute(
                    """
                    update memory_jobs
                    set status = 'claimed', attempt_count = attempt_count + 1,
                        claimed_at = now(), claimed_by = 'integration-test', claim_token = $3
                    where owner_id = $1 and id = $2
                    """,
                    self.owner_id,
                    initial_job["id"],
                    claim_token,
                )
                await self.jobs.complete(
                    owner_id=self.owner_id,
                    job_id=initial_job["id"],
                    claim_token=claim_token,
                    completed_at=datetime.now(UTC),
                    connection=connection,
                )

        await self._commit(await self._new_context(), proposal)
        completed_jobs = await self._embedding_jobs()
        self.assertEqual(len(completed_jobs), 1)
        self.assertEqual(completed_jobs[0]["id"], initial_job["id"])
        self.assertEqual(completed_jobs[0]["session_id"], self.session_id)
        self.assertEqual(completed_jobs[0]["status"], "completed")
        self.assertEqual(len(await self._fact_rows("owns")), 1)

    async def test_changed_fact_content_enqueues_new_key_with_origin_session(self) -> None:
        proposal = _proposal("owns", "silver keys", explicit=True)
        await self._commit(self.context, proposal)
        await self._commit(
            await self._new_context(), replace(proposal, confidence=0.99)
        )

        jobs = await self._embedding_jobs()
        self.assertEqual(len(jobs), 2)
        self.assertEqual(len({job["dedupe_key"] for job in jobs}), 2)
        self.assertEqual({job["session_id"] for job in jobs}, {self.session_id})
        self.assertEqual(jobs[0]["payload"], jobs[1]["payload"])
        self.assertEqual(len(await self._fact_rows("owns")), 1)

    async def test_concurrent_sessions_reinforce_one_embedding_job_atomically(self) -> None:
        proposal = _proposal("owns", "silver keys", explicit=True)
        await self._commit(self.context, proposal)
        contexts = (await self._new_context(), await self._new_context())

        outcomes = await asyncio.gather(
            *(self._commit(context, proposal) for context in contexts)
        )

        self.assertEqual([outcome["updated_facts"] for outcome in outcomes], [1, 1])
        jobs = await self._embedding_jobs()
        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0]["session_id"], self.session_id)
        self.assertEqual(len(await self._fact_rows("owns")), 1)
        async with self.pool.acquire() as connection:
            reconciled_count = await connection.fetchval(
                """
                select count(*) from experience_events
                where owner_id = $1 and semantic_reconciled_at is not null
                """,
                self.owner_id,
            )
        self.assertEqual(reconciled_count, 3)

    async def test_retry_is_idempotent_and_history_is_owner_scoped(self) -> None:
        proposal = _proposal("owns", "silver keys", explicit=True)
        await self._apply(proposal)
        await self._apply(proposal)
        self.assertEqual(len(await self._fact_rows("owns")), 1)
        other_history = await MemoryRepository(self.pool).list_semantic_fact_history(
            owner_id=f"other-{self.owner_id}",
            subject_entity_id=self.subject.id,
            predicate="owns",
        )
        self.assertEqual(other_history, [])

    async def _commit(
        self,
        context: SimpleNamespace,
        proposal: SemanticFactProposal,
        *,
        job_repository: MemoryJobRepository | None = None,
    ):
        proposal = replace(
            proposal, subject=replace(proposal.subject, external_id=self.owner_id)
        )
        handler = ReconcileSemanticMemoryJobHandler(
            repository=self.semantic_repository,
            reconciler=AsyncMock(),
            job_repository=job_repository if job_repository is not None else self.jobs,
            embedding_model=SimpleNamespace(provider="gemini", model="embedding"),
        )
        job = SimpleNamespace(
            owner_id=self.owner_id,
            session_id=context.session_id,
            payload={"event_id": str(context.event_id)},
        )
        prepared = PreparedSemanticReconciliation(
            context=context,
            plan=SemanticReconciliationPlan(
                owner_id=self.owner_id,
                event_id=context.event_id,
                episode_memory_id=context.episode_memory_id,
                entities=(proposal.subject,),
                facts=(proposal,),
                relations=(),
            ),
            entity_embeddings={},
        )
        async with self.pool.acquire() as connection:
            async with connection.transaction():
                return await handler.commit(job, prepared, connection)

    async def _embedding_jobs(self) -> list[asyncpg.Record]:
        async with self.pool.acquire() as connection:
            return await connection.fetch(
                """
                select id, session_id, status, dedupe_key, payload
                from memory_jobs
                where owner_id = $1 and job_type = 'embed_memory_item'
                order by created_at, id
                """,
                self.owner_id,
            )

    async def _apply(self, proposal: SemanticFactProposal):
        async with self.pool.acquire() as connection:
            async with connection.transaction():
                return await self.semantic_repository._apply_fact_proposal(
                    connection,
                    context=self.context,
                    subject=self.subject,
                    proposal=proposal,
                )

    async def _fact_rows(self, predicate: str) -> list[asyncpg.Record]:
        async with self.pool.acquire() as connection:
            return await connection.fetch(
                """
                select status, metadata->>'value' as value
                from memory_items
                where owner_id = $1
                  and memory_type = 'semantic_fact'
                  and metadata->>'subject_entity_id' = $2
                  and metadata->>'predicate' = $3
                order by created_at, id
                """,
                self.owner_id,
                str(self.subject.id),
                predicate,
            )


def _proposal(
    predicate: str,
    value: str,
    *,
    explicit: bool = False,
    confidence: float = 0.95,
    change_intent: str = "assert",
    previous_value: str | None = None,
) -> SemanticFactProposal:
    return SemanticFactProposal(
        subject=SemanticEntityIdentity(
            name="Dennis",
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
        valid_from=datetime.now(UTC),
        valid_to=None,
        evidence=(),
    )


if __name__ == "__main__":
    unittest.main()
