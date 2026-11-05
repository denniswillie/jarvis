from __future__ import annotations

import json
import os
import unittest
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import asyncpg

from app.experience.keyframes import DeterministicKeyframeSelector
from app.experience.repository import ExperienceRepository


@unittest.skipUnless(
    os.getenv("JARVIS_TEST_DB_URL"),
    "JARVIS_TEST_DB_URL is required for keyframe retention integration tests",
)
class KeyframeRetentionIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.owner_id = f"keyframe-retention-integration-{uuid4()}"
        self.pool = await asyncpg.create_pool(
            os.environ["JARVIS_TEST_DB_URL"], min_size=1, max_size=3
        )
        async with self.pool.acquire() as conn:
            self.session_id = await conn.fetchval(
                """
                insert into jarvis_sessions (owner_id, title)
                values ($1, 'Keyframe retention integration')
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

    async def test_close_rolls_back_provisional_keyframes_and_retention(self) -> None:
        started_at = datetime.now(UTC)
        visual_session_id = uuid4()
        frame_ids = (uuid4(), uuid4())
        observation_id = uuid4()

        async with self.pool.acquire() as conn:
            await conn.executemany(
                """
                insert into visual_frames (
                    id,
                    owner_id,
                    session_id,
                    visual_session_id,
                    frame_sequence,
                    source,
                    captured_at,
                    content_type,
                    width,
                    height,
                    storage_bucket,
                    storage_path,
                    raw_image_expires_at
                )
                values (
                    $1, $2, $3, $4, $5, 'glasses', $6,
                    'image/jpeg', 768, 432, 'test-frames', $7, $8
                )
                """,
                [
                    (
                        frame_id,
                        self.owner_id,
                        self.session_id,
                        visual_session_id,
                        index,
                        started_at + timedelta(seconds=index),
                        f"integration/{frame_id}.jpg",
                        started_at + timedelta(minutes=5),
                    )
                    for index, frame_id in enumerate(frame_ids, start=1)
                ],
            )
            candidates = json.dumps(
                [
                    {
                        "visual_frame_id": str(frame_id),
                        "reason": f"candidate {index}",
                        "confidence": 0.9,
                    }
                    for index, frame_id in enumerate(frame_ids, start=1)
                ]
            )
            await conn.execute(
                """
                insert into perception_observations (
                    id,
                    owner_id,
                    session_id,
                    source,
                    started_at,
                    ended_at,
                    primary_visual_frame_id,
                    summary,
                    keyframe_candidates,
                    confidence,
                    analyzer_provider,
                    analyzer_model,
                    analyzer_contract_version
                )
                values (
                    $1, $2, $3, 'glasses', $4, $5, $6,
                    'Rollback integration observation', $7::jsonb,
                    0.9, 'test', 'test', 1
                )
                """,
                observation_id,
                self.owner_id,
                self.session_id,
                started_at,
                started_at + timedelta(seconds=2),
                frame_ids[0],
                candidates,
            )
            await conn.executemany(
                """
                insert into perception_observation_frames (
                    observation_id,
                    visual_frame_id,
                    ordinal,
                    role
                )
                values ($1, $2, $3, $4)
                """,
                [
                    (observation_id, frame_ids[0], 0, "primary"),
                    (observation_id, frame_ids[1], 1, "key_candidate"),
                ],
            )
            event_id = await conn.fetchval(
                """
                insert into experience_events (
                    owner_id,
                    session_id,
                    event_kind,
                    modalities,
                    started_at
                )
                values ($1, $2, 'visual', array['visual']::text[], $3)
                returning id
                """,
                self.owner_id,
                self.session_id,
                started_at,
            )
            await conn.execute(
                """
                insert into experience_event_evidence (
                    event_id,
                    evidence_type,
                    evidence_ref_id,
                    occurred_at
                )
                values ($1, 'observation', $2, $3)
                """,
                event_id,
                observation_id,
                started_at,
            )

        repository = ExperienceRepository(self.pool)
        selector = DeterministicKeyframeSelector()
        with self.assertRaises(asyncpg.CheckViolationError):
            await repository.close_event_with_keyframes(
                owner_id=self.owner_id,
                session_id=self.session_id,
                event_id=event_id,
                ended_at=started_at + timedelta(seconds=30),
                boundary_reason="idle_gap",
                finalized_at=None,  # type: ignore[arg-type]
                candidate_expires_at=started_at + timedelta(minutes=30),
                max_count=6,
                selector=lambda values: selector.select(values, limit=6),
            )

        async with self.pool.acquire() as conn:
            status = await conn.fetchval(
                "select status from experience_events where id = $1", event_id
            )
            keyframe_count = await conn.fetchval(
                "select count(*) from experience_event_keyframes where event_id = $1",
                event_id,
            )
            retention_count = await conn.fetchval(
                """
                select count(*)
                from visual_frame_retention_reasons
                where reason_type = 'event_candidate' and reason_ref_id = $1
                """,
                event_id,
            )

        self.assertEqual(status, "open")
        self.assertEqual(keyframe_count, 0)
        self.assertEqual(retention_count, 0)
