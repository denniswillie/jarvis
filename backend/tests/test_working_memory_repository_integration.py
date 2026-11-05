from __future__ import annotations

import asyncio
import os
import unittest
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import asyncpg

from app.perception.models import VisualObservation
from app.perception.repository import PerceptionRepository


@unittest.skipUnless(
    os.getenv("JARVIS_TEST_DB_URL"),
    "JARVIS_TEST_DB_URL is required for working-memory integration tests",
)
class WorkingMemoryRepositoryIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.owner_id = f"working-memory-integration-{uuid4()}"
        self.pool = await asyncpg.create_pool(
            os.environ["JARVIS_TEST_DB_URL"], min_size=1, max_size=3
        )
        async with self.pool.acquire() as conn:
            self.session_id = await conn.fetchval(
                """
                insert into jarvis_sessions (owner_id, title)
                values ($1, 'Working-memory integration')
                returning id
                """,
                self.owner_id,
            )

    async def asyncTearDown(self) -> None:
        async with self.pool.acquire() as conn:
            await conn.execute(
                "delete from jarvis_sessions where owner_id = $1", self.owner_id
            )
        await self.pool.close()

    async def test_database_cas_allows_only_one_writer_for_a_revision(self) -> None:
        repository = PerceptionRepository(self.pool)
        first = _observation(self.owner_id, self.session_id, datetime.now(UTC))
        second = _observation(
            self.owner_id, self.session_id, first.ended_at + timedelta(seconds=1)
        )
        await repository.insert_observation(observation=first, frame_links=())
        await repository.insert_observation(observation=second, frame_links=())

        first_write, second_write = await asyncio.gather(
            repository.compare_and_swap_session_working_memory(
                owner_id=self.owner_id, session_id=self.session_id,
                expected_revision=0, observation_id=first.id,
                state={"summary": "first", "updated_at": first.ended_at.isoformat()},
                expires_at=first.ended_at + timedelta(minutes=2),
            ),
            repository.compare_and_swap_session_working_memory(
                owner_id=self.owner_id, session_id=self.session_id,
                expected_revision=0, observation_id=second.id,
                state={"summary": "second", "updated_at": second.ended_at.isoformat()},
                expires_at=second.ended_at + timedelta(minutes=2),
            ),
        )

        self.assertEqual(sum(result.applied for result in (first_write, second_write)), 1)
        stored = await repository.get_session_working_memory(
            owner_id=self.owner_id, session_id=self.session_id
        )
        self.assertIsNotNone(stored)
        assert stored is not None
        self.assertEqual(stored.revision, 1)

    async def test_latest_experience_event_pointer_accepts_uuid(self) -> None:
        repository = PerceptionRepository(self.pool)
        event_id = uuid4()
        expires_at = datetime.now(UTC) + timedelta(minutes=2)

        stored = await repository.set_latest_experience_event_id(
            owner_id=self.owner_id,
            session_id=self.session_id,
            event_id=event_id,
            expires_at=expires_at,
        )

        self.assertIsNotNone(stored)
        assert stored is not None
        self.assertEqual(stored.state["latest_experience_event_id"], str(event_id))


def _observation(owner_id, session_id, ended_at) -> VisualObservation:
    return VisualObservation(
        id=uuid4(), owner_id=owner_id, session_id=session_id, source="glasses",
        started_at=ended_at, ended_at=ended_at, primary_visual_frame_id=None,
        summary="Integration observation", scene="room", activity="testing",
        entities=(), visible_text=(), state_changes=(), memory_tags=(),
        keyframe_candidates=(), interrupt_candidate=None, starts_new_event=False,
        boundary_reason=None, confidence=0.9, analyzer_provider="test",
        analyzer_model="test", analyzer_contract_version=1, compacted_at=None,
        created_at=ended_at, scene_confidence=0.9, activity_confidence=0.9,
    )
