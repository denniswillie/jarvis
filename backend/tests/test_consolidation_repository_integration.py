from __future__ import annotations

import os
import unittest
from datetime import UTC, datetime
from typing import Any, cast
from uuid import UUID, uuid4

import asyncpg

from app.experience.repository import ExperienceRepository
from app.memory.consolidation.contracts import ConsolidationResult
from app.memory.consolidation.job_handler import (
    CONSOLIDATE_EXPERIENCE_EVENT_JOB,
    ConsolidateExperienceEventJobHandler,
    ExperienceConsolidationRecoveryRule,
    consolidation_dedupe_key,
)
from app.memory.consolidation.service import PreparedConsolidation
from app.memory.jobs.reconciler import MemoryJobReconciler
from app.memory.jobs.repository import MemoryJobRepository


class _RejectingJobRepository:
    async def enqueue(self, **_: Any) -> None:
        raise RuntimeError("simulated enqueue failure")


class _PreparedConsolidationService:
    provider = "openai"
    model = "integration-test-model"

    async def consolidate(
        self,
        *,
        owner_id: str,
        session_id: UUID,
        event_id: UUID,
    ) -> PreparedConsolidation:
        return PreparedConsolidation(
            event_id=event_id,
            owner_id=owner_id,
            session_id=session_id,
            result=ConsolidationResult(
                event_title="Consolidated event",
                event_summary="A transactionally persisted consolidation result.",
                location=None,
                actions=[],
                entity_candidates=[],
                visible_text=[],
                episode_importance=0.5,
                confidence=0.9,
                should_promote_to_episode=False,
                promotion_reason="Integration test",
                selected_keyframe_ids=[],
                semantic_fact_candidates=[],
                relation_candidates=[],
                reflection_candidates=[],
                procedure_candidates=[],
            ),
            provider=self.provider,
            model=self.model,
        )


@unittest.skipUnless(
    os.getenv("JARVIS_TEST_DB_URL"),
    "JARVIS_TEST_DB_URL is required for consolidation integration tests",
)
class ConsolidationRepositoryIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.owner_id = f"consolidation-integration-{uuid4()}"
        self.now = datetime.now(UTC)
        self.pool = await asyncpg.create_pool(
            os.environ["JARVIS_TEST_DB_URL"],
            min_size=1,
            max_size=2,
        )

    async def asyncTearDown(self) -> None:
        async with self.pool.acquire() as connection:
            async with connection.transaction():
                await connection.execute(
                    "delete from memory_jobs where owner_id = $1",
                    self.owner_id,
                )
                await connection.execute(
                    "delete from jarvis_sessions where owner_id = $1",
                    self.owner_id,
                )
        await self.pool.close()

    async def test_close_and_enqueue_roll_back_together(self) -> None:
        session_id, event_id = await self._insert_open_event("rollback")
        repository = ExperienceRepository(
            self.pool,
            consolidation_job_repository=cast(
                MemoryJobRepository,
                _RejectingJobRepository(),
            ),
        )

        with self.assertRaisesRegex(RuntimeError, "simulated enqueue failure"):
            await self._close(repository, session_id=session_id, event_id=event_id)

        async with self.pool.acquire() as connection:
            status = await connection.fetchval(
                "select status from experience_events where id = $1",
                event_id,
            )
            job_count = await connection.fetchval(
                "select count(*) from memory_jobs where owner_id = $1",
                self.owner_id,
            )
        self.assertEqual(status, "open")
        self.assertEqual(job_count, 0)

    async def test_result_and_job_completion_are_atomic_and_recoverable(self) -> None:
        session_id, event_id = await self._insert_open_event("commit")
        job_repository = MemoryJobRepository(self.pool, default_max_attempts=3)
        experience_repository = ExperienceRepository(
            self.pool,
            consolidation_job_repository=job_repository,
        )
        await self._close(
            experience_repository,
            session_id=session_id,
            event_id=event_id,
        )
        handler = ConsolidateExperienceEventJobHandler(
            repository=experience_repository,
            service=cast(Any, _PreparedConsolidationService()),
            candidate_lease_seconds=1800,
        )

        first_job = await job_repository.claim_next(
            worker_id="integration-worker-1",
            job_types=(CONSOLIDATE_EXPERIENCE_EVENT_JOB,),
        )
        self.assertIsNotNone(first_job)
        assert first_job is not None and first_job.claim_token is not None
        prepared = await handler.prepare(first_job)
        with self.assertRaisesRegex(RuntimeError, "simulated completion failure"):
            async with self.pool.acquire() as connection:
                async with connection.transaction():
                    await job_repository.lock_claim(
                        owner_id=self.owner_id,
                        job_id=first_job.id,
                        claim_token=first_job.claim_token,
                        connection=connection,
                    )
                    await handler.commit(first_job, prepared, connection)
                    raise RuntimeError("simulated completion failure")

        async with self.pool.acquire() as connection:
            rolled_back = await connection.fetchrow(
                "select status, consolidation_result from experience_events where id = $1",
                event_id,
            )
        self.assertEqual(rolled_back["status"], "consolidating")
        self.assertIsNone(rolled_back["consolidation_result"])

        await job_repository.fail(
            owner_id=self.owner_id,
            job_id=first_job.id,
            claim_token=first_job.claim_token,
            error="simulated completion failure",
            retry_at=self.now,
        )
        reconciler = MemoryJobReconciler(
            pool=self.pool,
            repository=job_repository,
            rules=(ExperienceConsolidationRecoveryRule(candidate_lease_seconds=1800),),
        )
        await reconciler.run_once(now=self.now)
        await job_repository.requeue_due_failed()

        second_job = await job_repository.claim_next(
            worker_id="integration-worker-2",
            job_types=(CONSOLIDATE_EXPERIENCE_EVENT_JOB,),
        )
        self.assertIsNotNone(second_job)
        assert second_job is not None and second_job.claim_token is not None
        prepared = await handler.prepare(second_job)
        async with self.pool.acquire() as connection:
            async with connection.transaction():
                await job_repository.lock_claim(
                    owner_id=self.owner_id,
                    job_id=second_job.id,
                    claim_token=second_job.claim_token,
                    connection=connection,
                )
                result = await handler.commit(second_job, prepared, connection)
                await job_repository.complete(
                    owner_id=self.owner_id,
                    job_id=second_job.id,
                    claim_token=second_job.claim_token,
                    completed_at=self.now,
                    result=result,
                    connection=connection,
                )

        async with self.pool.acquire() as connection:
            persisted = await connection.fetchrow(
                "select status, title, consolidation_result from experience_events where id = $1",
                event_id,
            )
            completed_job = await connection.fetchrow(
                "select status from memory_jobs where id = $1",
                second_job.id,
            )
        self.assertEqual(persisted["status"], "consolidated")
        self.assertEqual(persisted["title"], "Consolidated event")
        self.assertIsNotNone(persisted["consolidation_result"])
        self.assertEqual(completed_job["status"], "completed")

    async def _insert_open_event(self, suffix: str) -> tuple[UUID, UUID]:
        async with self.pool.acquire() as connection:
            session_id = await connection.fetchval(
                """
                insert into jarvis_sessions (owner_id, title)
                values ($1, $2)
                returning id
                """,
                self.owner_id,
                f"Consolidation {suffix}",
            )
            event_id = await connection.fetchval(
                """
                insert into experience_events (
                    owner_id,
                    session_id,
                    event_kind,
                    modalities,
                    started_at,
                    title,
                    summary,
                    state
                )
                values ($1, $2, 'visual', array['visual'], $3, '', '', '{}'::jsonb)
                returning id
                """,
                self.owner_id,
                session_id,
                self.now,
            )
        return session_id, event_id

    async def _close(
        self,
        repository: ExperienceRepository,
        *,
        session_id: UUID,
        event_id: UUID,
    ) -> None:
        await repository.close_event(
            owner_id=self.owner_id,
            session_id=session_id,
            event_id=event_id,
            ended_at=self.now,
            boundary_reason="integration_test",
            keyframes_finalized_at=self.now,
            consolidation_ready_at=self.now,
        )


if __name__ == "__main__":
    unittest.main()
