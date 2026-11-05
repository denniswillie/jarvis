from __future__ import annotations

import asyncio
import os
import unittest
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import asyncpg

from app.memory.reflection.contracts import ReflectionResult
from app.memory.jobs.repository import MemoryJobRepository
from app.memory.reflection.repository import ReflectionRepository
from app.preferences.repository import PreferenceRepository


class _EmbeddingJobRepository:
    def __init__(self) -> None:
        self.memory_item_ids = []

    async def enqueue(self, **kwargs):
        self.memory_item_ids.append(kwargs["payload"]["memory_item_id"])
        return None


class _FailingEmbeddingJobRepository:
    def __init__(self, repository: MemoryJobRepository, fail_on_call: int) -> None:
        self.repository = repository
        self.fail_on_call = fail_on_call
        self.calls = 0

    async def enqueue(self, **kwargs):
        self.calls += 1
        result = await self.repository.enqueue(**kwargs)
        if self.calls == self.fail_on_call:
            raise RuntimeError("forced embedding enqueue failure")
        return result


@unittest.skipUnless(
    os.getenv("JARVIS_TEST_DB_URL"),
    "JARVIS_TEST_DB_URL is required for reflection integration tests",
)
class ReflectionRepositoryIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.owner_id = f"reflection-integration-{uuid4()}"
        self.pool = await asyncpg.create_pool(
            os.environ["JARVIS_TEST_DB_URL"],
            min_size=1,
            max_size=2,
        )
        self.episode_ids = await self._create_episode_range()

    async def asyncTearDown(self) -> None:
        async with self.pool.acquire() as connection:
            async with connection.transaction():
                await connection.execute(
                    "delete from memory_jobs where owner_id = $1",
                    self.owner_id,
                )
                await connection.execute(
                    "delete from jarvis_user_preferences where owner_id = $1",
                    self.owner_id,
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
                    "delete from memory_items where owner_id = $1",
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
        await self.pool.close()

    async def test_reflection_preference_and_procedure_commit_atomically(self) -> None:
        repository = ReflectionRepository(self.pool)
        episode_range = await repository.load_episode_range(
            owner_id=self.owner_id,
            episode_ids=self.episode_ids,
        )
        result = ReflectionResult(
            title="Recurring morning routine",
            summary="Tea repeatedly follows breakfast.",
            content="Across three mornings, the user made tea after breakfast.",
            importance=0.7,
            confidence=0.92,
            supporting_episode_ids=[str(value) for value in self.episode_ids],
            preference_candidates=[
                {
                    "key": "breakfast_drink",
                    "value": "tea",
                    "confidence": 0.92,
                    "supporting_episode_ids": [
                        str(value) for value in self.episode_ids
                    ],
                }
            ],
            procedure_candidates=[
                {
                    "procedure_key": "morning tea",
                    "title": "Make morning tea",
                    "summary": "The repeated morning tea routine.",
                    "steps": ["Boil water", "Steep tea"],
                    "confidence": 0.9,
                    "is_explicit": False,
                    "supporting_episode_ids": [
                        str(value) for value in self.episode_ids
                    ],
                }
            ],
        )
        jobs = MemoryJobRepository(self.pool, default_max_attempts=3)
        now = datetime.now(UTC)
        async with self.pool.acquire() as connection:
            async with connection.transaction():
                outcome = await repository.apply_reflection(
                    connection=connection,
                    episode_range=episode_range,
                    result=result,
                    preference_repository=PreferenceRepository(self.pool),
                    job_repository=jobs,
                    minimum_preference_episodes=3,
                    minimum_preference_confidence=0.85,
                    minimum_procedure_episodes=3,
                    created_at=now,
                )

        self.assertTrue(outcome.created)
        self.assertEqual(len(outcome.preference_ids), 1)
        self.assertEqual(len(outcome.procedure_memory_ids), 1)
        async with self.pool.acquire() as connection:
            embedding_jobs = await connection.fetchval(
                """
                select count(*) from memory_jobs
                where owner_id = $1 and job_type = 'embed_memory_item'
                """,
                self.owner_id,
            )
        self.assertEqual(embedding_jobs, 2)

        async with self.pool.acquire() as connection:
            async with connection.transaction():
                replay = await repository.apply_reflection(
                    connection=connection,
                    episode_range=episode_range,
                    result=result,
                    preference_repository=PreferenceRepository(self.pool),
                    job_repository=jobs,
                    minimum_preference_episodes=3,
                    minimum_preference_confidence=0.85,
                    minimum_procedure_episodes=3,
                    created_at=now,
                )
        self.assertFalse(replay.created)
        self.assertEqual(replay.reflection_memory_id, outcome.reflection_memory_id)

    async def test_foreign_owner_episode_evidence_is_rejected(self) -> None:
        repository = ReflectionRepository(self.pool)

        with self.assertRaisesRegex(ValueError, "foreign-owner"):
            await repository.load_episode_range(
                owner_id=f"other-{self.owner_id}",
                episode_ids=self.episode_ids,
            )

    async def test_transaction_rolls_back_all_reflection_outputs_and_jobs(self) -> None:
        repository = ReflectionRepository(self.pool)
        episode_range = await repository.load_episode_range(
            owner_id=self.owner_id,
            episode_ids=self.episode_ids,
        )
        jobs = _FailingEmbeddingJobRepository(
            MemoryJobRepository(self.pool, default_max_attempts=3),
            fail_on_call=2,
        )
        async with self.pool.acquire() as connection:
            jobs_before = await connection.fetchval(
                "select count(*) from memory_jobs where owner_id = $1",
                self.owner_id,
            )
        with self.assertRaisesRegex(RuntimeError, "forced embedding"):
            async with self.pool.acquire() as connection:
                async with connection.transaction():
                    await repository.apply_reflection(
                        connection=connection,
                        episode_range=episode_range,
                        result=_reflection_result(self.episode_ids),
                        preference_repository=PreferenceRepository(self.pool),
                        job_repository=jobs,
                        minimum_preference_episodes=3,
                        minimum_preference_confidence=0.85,
                        minimum_procedure_episodes=3,
                        created_at=datetime.now(UTC),
                    )

        async with self.pool.acquire() as connection:
            derived_memories = await connection.fetchval(
                """
                select count(*) from memory_items
                where owner_id = $1 and memory_type in ('reflection', 'procedure')
                """,
                self.owner_id,
            )
            preferences = await connection.fetchval(
                "select count(*) from jarvis_user_preferences where owner_id = $1",
                self.owner_id,
            )
            jobs_count = await connection.fetchval(
                "select count(*) from memory_jobs where owner_id = $1",
                self.owner_id,
            )
        self.assertEqual(
            (derived_memories, preferences, jobs_count),
            (0, 0, jobs_before),
        )

    async def test_explicit_preference_wins_a_concurrent_inferred_write(self) -> None:
        repository = PreferenceRepository(self.pool)

        async def write_explicit() -> None:
            async with self.pool.acquire() as connection:
                async with connection.transaction():
                    await repository.set_preference(
                        owner_id=self.owner_id,
                        key="breakfast_drink",
                        value="coffee",
                        connection=connection,
                    )

        async def write_inferred() -> None:
            async with self.pool.acquire() as connection:
                async with connection.transaction():
                    await repository.set_inferred_preference(
                        owner_id=self.owner_id,
                        key="breakfast_drink",
                        value="tea",
                        confidence=0.95,
                        evidence_episode_ids=self.episode_ids,
                        connection=connection,
                    )

        await asyncio.gather(write_explicit(), write_inferred())

        async with self.pool.acquire() as connection:
            row = await connection.fetchrow(
                """
                select value, source, confidence, evidence_episode_ids
                from jarvis_user_preferences
                where owner_id = $1 and key = 'breakfast_drink'
                """,
                self.owner_id,
            )
        self.assertEqual(row["value"], "coffee")
        self.assertEqual(row["source"], "assistant")
        self.assertEqual(row["confidence"], 1.0)
        self.assertEqual(tuple(row["evidence_episode_ids"]), ())

    async def test_procedure_reinforcement_and_material_versioning(self) -> None:
        repository = ReflectionRepository(self.pool)
        jobs = _EmbeddingJobRepository()
        ranges = [self.episode_ids]
        ranges.append(await self._create_episode_range(start_index=3))
        ranges.append(await self._create_episode_range(start_index=6))
        outcomes = []
        for index, episode_ids in enumerate(ranges):
            episode_range = await repository.load_episode_range(
                owner_id=self.owner_id,
                episode_ids=episode_ids,
            )
            result = _reflection_result(
                episode_ids,
                steps=(
                    ["Boil water", "Steep tea"]
                    if index < 2
                    else ["Boil water", "Steep tea", "Add lemon"]
                ),
            )
            async with self.pool.acquire() as connection:
                async with connection.transaction():
                    outcomes.append(
                        await repository.apply_reflection(
                            connection=connection,
                            episode_range=episode_range,
                            result=result,
                            preference_repository=PreferenceRepository(self.pool),
                            job_repository=jobs,
                            minimum_preference_episodes=3,
                            minimum_preference_confidence=0.85,
                            minimum_procedure_episodes=3,
                            created_at=datetime.now(UTC),
                        )
                    )

        first_id = outcomes[0].procedure_memory_ids[0]
        self.assertEqual(outcomes[1].procedure_memory_ids, (first_id,))
        second_id = outcomes[2].procedure_memory_ids[0]
        self.assertNotEqual(second_id, first_id)
        async with self.pool.acquire() as connection:
            rows = await connection.fetch(
                """
                select id, status, superseded_by_memory_id,
                       (metadata->>'version')::integer as version
                from memory_items
                where owner_id = $1 and memory_type = 'procedure'
                order by (metadata->>'version')::integer
                """,
                self.owner_id,
            )
            first_evidence = await connection.fetchval(
                """
                select count(*) from memory_evidence
                where owner_id = $1 and memory_item_id = $2
                """,
                self.owner_id,
                first_id,
            )
            supersession_links = await connection.fetchval(
                """
                select count(*) from memory_links
                where owner_id = $1 and source_memory_id = $2
                  and target_memory_id = $3 and relationship = 'supersedes'
                """,
                self.owner_id,
                second_id,
                first_id,
            )
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["status"], "superseded")
        self.assertEqual(rows[0]["superseded_by_memory_id"], second_id)
        self.assertEqual(rows[0]["version"], 1)
        self.assertEqual(rows[1]["status"], "active")
        self.assertEqual(rows[1]["version"], 2)
        self.assertEqual(first_evidence, 6)
        self.assertEqual(supersession_links, 1)

    async def _create_episode_range(self, *, start_index: int = 0):
        now = datetime.now(UTC) + timedelta(minutes=start_index)
        async with self.pool.acquire() as connection:
            async with connection.transaction():
                session_id = await connection.fetchval(
                    """
                    insert into jarvis_sessions (owner_id, title)
                    values ($1, 'Reflection integration')
                    returning id
                    """,
                    self.owner_id,
                )
                episode_ids = []
                for index in range(start_index, start_index + 3):
                    occurred_at = now + timedelta(seconds=index)
                    event_id = await connection.fetchval(
                        """
                        insert into experience_events (
                            owner_id, session_id, status, event_kind, modalities,
                            started_at, ended_at, boundary_reason,
                            keyframes_finalized_at, consolidation_ready_at
                        )
                        values (
                            $1, $2, 'closed', 'mixed', array['audio'],
                            $3, $3, 'integration_test', $3, $3
                        )
                        returning id
                        """,
                        self.owner_id,
                        session_id,
                        occurred_at,
                    )
                    memory_id = await connection.fetchval(
                        """
                        insert into memory_items (
                            owner_id, session_id, memory_type, title, content,
                            source_experience_event_id, occurred_at, valid_from,
                            created_at, importance, confidence, source, is_explicit
                        )
                        values (
                            $1, $2, 'episode', $3, $4, $5, $6, $6, $6,
                            0.7, 0.9, 'integration_test', false
                        )
                        returning id
                        """,
                        self.owner_id,
                        session_id,
                        f"Morning {index + 1}",
                        "The user made tea after breakfast.",
                        event_id,
                        occurred_at,
                    )
                    await connection.execute(
                        """
                        update experience_events
                        set promotion_decision = 'promoted',
                            promotion_reason = 'integration_test',
                            promoted_episode_memory_id = $3,
                            promotion_decided_at = $4
                        where owner_id = $1 and id = $2
                        """,
                        self.owner_id,
                        event_id,
                        memory_id,
                        occurred_at,
                    )
                    episode_ids.append(memory_id)
        return tuple(episode_ids)


def _reflection_result(
    episode_ids,
    *,
    steps: list[str] | None = None,
) -> ReflectionResult:
    return ReflectionResult(
        title="Recurring morning routine",
        summary="Tea repeatedly follows breakfast.",
        content="Across three mornings, the user made tea after breakfast.",
        importance=0.7,
        confidence=0.92,
        supporting_episode_ids=[str(value) for value in episode_ids],
        preference_candidates=[
            {
                "key": "breakfast_drink",
                "value": "tea",
                "confidence": 0.92,
                "supporting_episode_ids": [str(value) for value in episode_ids],
            }
        ],
        procedure_candidates=[
            {
                "procedure_key": "morning tea",
                "title": "Make morning tea",
                "summary": "The repeated morning tea routine.",
                "steps": steps or ["Boil water", "Steep tea"],
                "confidence": 0.9,
                "is_explicit": False,
                "supporting_episode_ids": [str(value) for value in episode_ids],
            }
        ],
    )


if __name__ == "__main__":
    unittest.main()
