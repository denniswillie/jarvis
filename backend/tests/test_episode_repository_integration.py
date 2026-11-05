from __future__ import annotations

import json
import os
import unittest
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

import asyncpg

from app.experience.repository import ExperienceRepository
from app.memory.episodic.job_handler import (
    PromoteExperienceEventJobHandler,
    promotion_dedupe_key,
)
from app.memory.episodic.repository import EpisodicMemoryRepository
from app.memory.episodic.service import EpisodePromotionService
from app.memory.jobs.models import MemoryJob
from app.memory.jobs.repository import MemoryJobRepository
from app.memory.repository import MemoryRepository


@unittest.skipUnless(
    os.getenv("JARVIS_TEST_DB_URL"),
    "JARVIS_TEST_DB_URL is required for episode integration tests",
)
class EpisodeRepositoryIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.owner_id = f"episode-integration-{uuid4()}"
        self.started_at = datetime.now(UTC) - timedelta(seconds=10)
        self.ended_at = self.started_at + timedelta(seconds=8)
        self.frame_captured_at = self.started_at + timedelta(seconds=3)
        self.reminder_delivered_at = self.started_at + timedelta(seconds=5)
        self.pool = await asyncpg.create_pool(
            os.environ["JARVIS_TEST_DB_URL"], min_size=1, max_size=2
        )
        await self._insert_fixture()

    async def asyncTearDown(self) -> None:
        async with self.pool.acquire() as connection:
            async with connection.transaction():
                await connection.execute(
                    "delete from memory_jobs where owner_id = $1", self.owner_id
                )
                await connection.execute(
                    "delete from jarvis_reminders where owner_id = $1", self.owner_id
                )
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
                    "delete from memory_items where owner_id = $1", self.owner_id
                )
                await connection.execute(
                    "delete from jarvis_sessions where owner_id = $1", self.owner_id
                )
        await self.pool.close()

    async def test_promotion_rolls_back_then_replays_idempotently_with_evidence(self) -> None:
        experience_repository = ExperienceRepository(self.pool)
        episodic_repository = EpisodicMemoryRepository(
            self.pool,
            experience_repository=experience_repository,
            memory_repository=MemoryRepository(self.pool),
        )
        job_repository = MemoryJobRepository(self.pool, default_max_attempts=3)
        handler = PromoteExperienceEventJobHandler(
            repository=episodic_repository,
            service=EpisodePromotionService(
                minimum_importance=0.55,
                important_keyframe_minimum_importance=0.8,
            ),
            embedding_job_repository=job_repository,
            ordinary_keyframe_retention_seconds=86_400,
            skipped_event_retention_days=90,
        )
        job = self._job()
        prepared = await handler.prepare(job)

        with self.assertRaisesRegex(RuntimeError, "simulated rollback"):
            async with self.pool.acquire() as connection:
                async with connection.transaction():
                    await handler.commit(job, prepared, connection)
                    raise RuntimeError("simulated rollback")

        await self._assert_counts(decision="pending", episodes=0, embed_jobs=0)

        async with self.pool.acquire() as connection:
            async with connection.transaction():
                first = await handler.commit(job, prepared, connection)
        async with self.pool.acquire() as connection:
            async with connection.transaction():
                replay = await handler.commit(job, prepared, connection)

        self.assertEqual(first["episode_memory_id"], replay["episode_memory_id"])
        await self._assert_counts(decision="promoted", episodes=1, embed_jobs=1)
        async with self.pool.acquire() as connection:
            evidence = await connection.fetch(
                """
                select evidence_type, evidence_ref_id, evidence_time, metadata
                from memory_evidence
                where owner_id = $1 and evidence_type in ('reminder', 'frame')
                order by evidence_type
                """,
                self.owner_id,
            )
        by_type = {row["evidence_type"]: row for row in evidence}
        self.assertEqual(by_type["frame"]["evidence_time"], self.frame_captured_at)
        self.assertEqual(
            by_type["reminder"]["evidence_time"], self.reminder_delivered_at
        )
        self.assertEqual(
            UUID(json.loads(by_type["reminder"]["metadata"])["reminder_id"]),
            self.reminder_id,
        )

    async def _insert_fixture(self) -> None:
        result = {
            "event_title": "Reminder event",
            "event_summary": "Jarvis delivered a reminder while viewing a scene.",
            "location": None,
            "actions": [],
            "entity_candidates": [],
            "visible_text": [],
            "episode_importance": 0.1,
            "confidence": 0.9,
            "should_promote_to_episode": False,
            "promotion_reason": "Below model threshold",
            "selected_keyframe_ids": [],
            "semantic_fact_candidates": [],
            "relation_candidates": [],
            "reflection_candidates": [],
            "procedure_candidates": [],
        }
        async with self.pool.acquire() as connection:
            self.session_id = await connection.fetchval(
                "insert into jarvis_sessions (owner_id, title) values ($1, 'Episode') returning id",
                self.owner_id,
            )
            self.frame_id = uuid4()
            await connection.execute(
                """
                insert into visual_frames (
                    id, owner_id, session_id, visual_session_id, frame_sequence,
                    source, captured_at, content_type, width, height, storage_bucket,
                    storage_path, raw_image_expires_at
                ) values (
                    $1, $2, $3, $4, 1, 'glasses', $5, 'image/jpeg', 768, 432,
                    'test-frames', $6, $7
                )
                """,
                self.frame_id,
                self.owner_id,
                self.session_id,
                uuid4(),
                self.frame_captured_at,
                f"integration/{self.frame_id}.jpg",
                self.frame_captured_at + timedelta(minutes=30),
            )
            self.event_id = await connection.fetchval(
                """
                insert into experience_events (
                    owner_id, session_id, status, event_kind, modalities, started_at,
                    ended_at, title, summary, boundary_reason, importance, confidence,
                    keyframes_finalized_at, consolidation_ready_at, consolidation_result,
                    consolidation_provider, consolidation_model, consolidated_at
                ) values (
                    $1, $2, 'consolidated', 'mixed', array['visual','audio']::text[],
                    $3, $4, $5, $6, 'integration_test', 0.1, 0.9, $4, $4,
                    $7::jsonb, 'openai', 'integration-test', $4
                ) returning id
                """,
                self.owner_id,
                self.session_id,
                self.started_at,
                self.ended_at,
                result["event_title"],
                result["event_summary"],
                json.dumps(result),
            )
            await connection.execute(
                """
                insert into experience_event_keyframes (
                    event_id, visual_frame_id, rank, reason, quality_score, selected_by
                ) values ($1, $2, 1, 'clear view', 0.9, 'deterministic')
                """,
                self.event_id,
                self.frame_id,
            )
            await connection.execute(
                """
                insert into visual_frame_retention_reasons (
                    owner_id, visual_frame_id, reason_type, reason_ref_id, expires_at
                ) values ($1, $2, 'event_candidate', $3, $4)
                """,
                self.owner_id,
                self.frame_id,
                self.event_id,
                self.ended_at + timedelta(minutes=30),
            )
            self.reminder_id = await connection.fetchval(
                """
                insert into jarvis_reminders (owner_id, session_id, title, remind_at)
                values ($1, $2, 'Integration reminder', $3) returning id
                """,
                self.owner_id,
                self.session_id,
                self.reminder_delivered_at,
            )
            await connection.execute(
                """
                insert into jarvis_reminder_events (
                    owner_id, reminder_id, session_id, event_type, message, created_at, metadata
                ) values ($1, $2, $3, 'delivered_to_session', 'Take a break.', $4, $5::jsonb)
                """,
                self.owner_id,
                self.reminder_id,
                self.session_id,
                self.reminder_delivered_at,
                json.dumps({"source": "integration"}),
            )

    def _job(self) -> MemoryJob:
        now = datetime.now(UTC)
        return MemoryJob(
            id=uuid4(), owner_id=self.owner_id, session_id=self.session_id,
            job_type="promote_experience_event",
            dedupe_key=promotion_dedupe_key(self.event_id),
            payload={"event_id": str(self.event_id)}, status="claimed", priority=0,
            attempt_count=1, max_attempts=3, available_at=now, claimed_at=now,
            claimed_by="integration", claim_token=uuid4(), result=None,
            last_error=None, completed_at=None, created_at=now, updated_at=now,
        )

    async def _assert_counts(
        self, *, decision: str, episodes: int, embed_jobs: int
    ) -> None:
        async with self.pool.acquire() as connection:
            actual_decision = await connection.fetchval(
                "select promotion_decision from experience_events where id = $1",
                self.event_id,
            )
            episode_count = await connection.fetchval(
                "select count(*) from memory_items where owner_id = $1 and memory_type = 'episode'",
                self.owner_id,
            )
            job_count = await connection.fetchval(
                "select count(*) from memory_jobs where owner_id = $1 and job_type = 'embed_experience_event'",
                self.owner_id,
            )
        self.assertEqual(actual_decision, decision)
        self.assertEqual(episode_count, episodes)
        self.assertEqual(job_count, embed_jobs)
