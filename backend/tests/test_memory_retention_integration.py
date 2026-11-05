from __future__ import annotations

import os
import unittest
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

import asyncpg

from app.memory.models import ExplicitMemoryEvidenceInput
from app.memory.repository import MemoryRepository
from app.memory.retention import MemoryRetentionRepository


@unittest.skipUnless(
    os.getenv("JARVIS_TEST_DB_URL"),
    "JARVIS_TEST_DB_URL is required for memory retention integration tests",
)
class MemoryRetentionIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.owner_id = f"memory-retention-integration-{uuid4()}"
        self.pool = await asyncpg.create_pool(
            os.environ["JARVIS_TEST_DB_URL"], min_size=1, max_size=3
        )
        async with self.pool.acquire() as conn:
            self.session_id = await conn.fetchval(
                """
                insert into jarvis_sessions (owner_id, title)
                values ($1, 'Memory retention integration')
                returning id
                """,
                self.owner_id,
            )
        self.repository = MemoryRetentionRepository()

    async def asyncTearDown(self) -> None:
        async with self.pool.acquire() as conn:
            await conn.execute("delete from memory_items where owner_id = $1", self.owner_id)
            await conn.execute("delete from jarvis_sessions where owner_id = $1", self.owner_id)
        await self.pool.close()

    async def _insert_frame(
        self,
        conn: asyncpg.Connection,
        *,
        captured_at: datetime,
        expired: bool,
    ) -> UUID:
        frame_id = uuid4()
        await conn.execute(
            """
            insert into visual_frames (
                id, owner_id, session_id, visual_session_id, frame_sequence,
                source, captured_at, content_type, width, height,
                storage_bucket, storage_path, raw_image_expires_at,
                raw_image_status
            )
            values (
                $1, $2, $3, $4, $5, 'glasses', $6, 'image/jpeg', 768, 432,
                'test-frames', $7, $8, $9
            )
            """,
            frame_id,
            self.owner_id,
            self.session_id,
            uuid4(),
            int(captured_at.timestamp()),
            captured_at,
            f"integration/{frame_id}.jpg",
            captured_at + timedelta(minutes=5),
            "expired" if expired else "available",
        )
        return frame_id

    async def _insert_memory(
        self,
        conn: asyncpg.Connection,
        *,
        status: str,
        source_event_id: UUID | None = None,
    ) -> UUID:
        return await conn.fetchval(
            """
            insert into memory_items (
                owner_id, session_id, memory_type, status, title, content,
                source_experience_event_id, occurred_at, valid_from,
                importance, confidence, source, is_explicit
            )
            values (
                $1, $2, 'semantic_fact', $3, 'Retention fixture',
                'Retention fixture content', $4, now(), now(),
                0.5, 1.0, 'integration_test', false
            )
            returning id
            """,
            self.owner_id,
            self.session_id,
            status,
            source_event_id,
        )

    async def test_visual_cleanup_removes_inactive_evidence_only(self) -> None:
        old = datetime.now(UTC) - timedelta(days=45)
        async with self.pool.acquire() as conn:
            removable_frame = await self._insert_frame(conn, captured_at=old, expired=True)
            protected_frame = await self._insert_frame(conn, captured_at=old, expired=True)
            forgotten_memory = await self._insert_memory(conn, status="forgotten")
            active_memory = await self._insert_memory(conn, status="active")
            await conn.executemany(
                """
                insert into memory_evidence (
                    owner_id, memory_item_id, evidence_type, evidence_ref_id,
                    evidence_time, excerpt, confidence
                )
                values ($1, $2, 'frame', $3, now(), 'Frame evidence', 1.0)
                """,
                [
                    (self.owner_id, forgotten_memory, str(removable_frame)),
                    (self.owner_id, active_memory, str(protected_frame)),
                ],
            )
            async with conn.transaction():
                result = await self.repository.delete_unreferenced_visual_frames(
                    owner_id=self.owner_id,
                    captured_before=datetime.now(UTC) - timedelta(days=30),
                    now=datetime.now(UTC),
                    limit=50,
                    connection=conn,
                )
            remaining_ids = {
                row["id"]
                for row in await conn.fetch(
                    "select id from visual_frames where id = any($1::uuid[])",
                    [removable_frame, protected_frame],
                )
            }
            stale_evidence = await conn.fetchval(
                """
                select count(*) from memory_evidence
                where evidence_type = 'frame' and evidence_ref_id = $1
                """,
                str(removable_frame),
            )

        self.assertEqual(result["deleted_frames"], 1)
        self.assertEqual(result["deleted_memory_evidence"], 1)
        self.assertNotIn(removable_frame, remaining_ids)
        self.assertIn(protected_frame, remaining_ids)
        self.assertEqual(stale_evidence, 0)

    async def test_skipped_event_cleanup_respects_exact_durable_references(self) -> None:
        old = datetime.now(UTC) - timedelta(days=120)
        async with self.pool.acquire() as conn:
            event_ids: list[UUID] = []
            for label in ("removable", "source-protected", "reason-protected"):
                event_ids.append(
                    await conn.fetchval(
                        """
                        insert into experience_events (
                            owner_id, session_id, status, event_kind, modalities,
                            started_at, ended_at, title, summary,
                            keyframes_finalized_at, consolidation_ready_at,
                            retention_expires_at, promotion_decision,
                            promotion_reason, promotion_decided_at
                        )
                        values (
                            $1, $2, 'closed', 'interaction', array['text']::text[],
                            $3, $4, $5, $5, $4, $4, $4,
                            'skipped', 'integration test', $4
                        )
                        returning id
                        """,
                        self.owner_id,
                        self.session_id,
                        old,
                        old + timedelta(minutes=1),
                        label,
                    )
                )
            await self._insert_memory(
                conn,
                status="forgotten",
                source_event_id=event_ids[1],
            )
            await conn.execute(
                """
                insert into experience_event_retention_reasons (
                    owner_id, experience_event_id, reason_type,
                    reason_ref_id, expires_at
                )
                values ($1, $2, 'explicit_request', 'integration-test', null)
                """,
                self.owner_id,
                event_ids[2],
            )
            async with conn.transaction():
                result = await self.repository.delete_expired_skipped_events(
                    owner_id=self.owner_id,
                    now=datetime.now(UTC),
                    limit=50,
                    connection=conn,
                )
            remaining_ids = {
                row["id"]
                for row in await conn.fetch(
                    "select id from experience_events where id = any($1::uuid[])",
                    event_ids,
                )
            }

        self.assertEqual(result["deleted_events"], 1)
        self.assertNotIn(event_ids[0], remaining_ids)
        self.assertIn(event_ids[1], remaining_ids)
        self.assertIn(event_ids[2], remaining_ids)

    async def test_explicit_memory_retains_exact_experience_event(self) -> None:
        now = datetime.now(UTC)
        async with self.pool.acquire() as conn:
            event_id = await conn.fetchval(
                """
                insert into experience_events (
                    owner_id, session_id, status, event_kind, modalities,
                    started_at, ended_at, title, summary,
                    keyframes_finalized_at, consolidation_ready_at
                )
                values (
                    $1, $2, 'closed', 'interaction', array['text']::text[],
                    $3, $3, 'Exact event', 'Exact event evidence', $3, $3
                )
                returning id
                """,
                self.owner_id,
                self.session_id,
                now,
            )

        result = await MemoryRepository(self.pool).create_explicit_memory(
            owner_id=self.owner_id,
            session_id=self.session_id,
            memory_type="semantic_fact",
            title="Remembered exact event",
            content="The exact event should remain available as memory evidence.",
            occurred_at=now,
            idempotency_key=f"retention-integration-{uuid4()}",
            evidence=(
                ExplicitMemoryEvidenceInput(
                    evidence_type="experience_event",
                    evidence_ref_id=str(event_id),
                    evidence_time=now,
                    excerpt="Exact event evidence",
                ),
            ),
        )

        async with self.pool.acquire() as conn:
            reason_ref_id = await conn.fetchval(
                """
                select reason_ref_id
                from experience_event_retention_reasons
                where owner_id = $1
                  and experience_event_id = $2
                  and reason_type = 'explicit_memory'
                """,
                self.owner_id,
                event_id,
            )

        self.assertEqual(reason_ref_id, str(result.memory.id))

    async def test_observation_compaction_preserves_active_memory_evidence(self) -> None:
        old = datetime.now(UTC) - timedelta(days=10)
        async with self.pool.acquire() as conn:
            frame_ids = [
                await self._insert_frame(conn, captured_at=old, expired=False)
                for _ in range(2)
            ]
            observation_ids: list[UUID] = []
            for index, frame_id in enumerate(frame_ids):
                observation_id = uuid4()
                observation_ids.append(observation_id)
                await conn.execute(
                    """
                    insert into perception_observations (
                        id, owner_id, session_id, source, started_at, ended_at,
                        primary_visual_frame_id, summary, confidence,
                        analyzer_provider, analyzer_model, analyzer_contract_version
                    )
                    values (
                        $1, $2, $3, 'glasses', $4, $5, $6, $7,
                        0.9, 'test', 'test', 1
                    )
                    """,
                    observation_id,
                    self.owner_id,
                    self.session_id,
                    old + timedelta(seconds=index),
                    old + timedelta(seconds=index + 1),
                    frame_id,
                    f"Observation {index}",
                )
                await conn.execute(
                    """
                    insert into perception_observation_frames (
                        observation_id, visual_frame_id, ordinal, role
                    )
                    values ($1, $2, 0, 'primary')
                    """,
                    observation_id,
                    frame_id,
                )
            active_memory = await self._insert_memory(conn, status="active")
            await conn.execute(
                """
                insert into memory_evidence (
                    owner_id, memory_item_id, evidence_type,
                    evidence_ref_id, evidence_time, excerpt, confidence
                )
                values (
                    $1, $2, 'observation', $3, now(),
                    'Protected observation', 1.0
                )
                """,
                self.owner_id,
                active_memory,
                str(observation_ids[1]),
            )
            async with conn.transaction():
                compacted = await self.repository.compact_observations(
                    owner_id=self.owner_id,
                    ended_before=datetime.now(UTC) - timedelta(days=7),
                    compacted_at=datetime.now(UTC),
                    limit=50,
                    connection=conn,
                )
            rows = {
                row["id"]: row
                for row in await conn.fetch(
                    """
                    select id, compacted_at, primary_visual_frame_id
                    from perception_observations where id = any($1::uuid[])
                    """,
                    observation_ids,
                )
            }

        self.assertEqual(compacted, 1)
        self.assertIsNotNone(rows[observation_ids[0]]["compacted_at"])
        self.assertIsNone(rows[observation_ids[0]]["primary_visual_frame_id"])
        self.assertIsNone(rows[observation_ids[1]]["compacted_at"])
        self.assertEqual(
            rows[observation_ids[1]]["primary_visual_frame_id"],
            frame_ids[1],
        )
