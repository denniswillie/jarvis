from __future__ import annotations

import os
import unittest
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

import asyncpg

from app.memory.repository import MemoryRepository
from app.memory.retrieval.evidence_inspector import HistoricalEvidenceRepository
from app.memory.retrieval.models import FusedRetrievalCandidate, RetrievalDocument
from evals.perception_memory.fixture_cleanup import cleanup_fixture_owners


@unittest.skipUnless(
    os.getenv("JARVIS_TEST_DB_URL"),
    "JARVIS_TEST_DB_URL is required for evidence reinspection integration tests",
)
class HistoricalEvidenceRepositoryIntegrationTests(
    unittest.IsolatedAsyncioTestCase
):
    async def asyncSetUp(self) -> None:
        self.owner_id = f"evidence-reinspection-{uuid4()}"
        self.other_owner_id = f"evidence-reinspection-other-{uuid4()}"
        self.now = datetime.now(UTC)
        self.addAsyncCleanup(self._cleanup_fixture_owners)
        self.pool = await asyncpg.create_pool(
            os.environ["JARVIS_TEST_DB_URL"], min_size=1, max_size=2
        )
        async with self.pool.acquire() as connection:
            self.session_id = await connection.fetchval(
                """
                insert into jarvis_sessions (owner_id, title)
                values ($1, 'Evidence reinspection integration')
                returning id
                """,
                self.owner_id,
            )
            self.other_session_id = await connection.fetchval(
                """
                insert into jarvis_sessions (owner_id, title)
                values ($1, 'Other owner evidence')
                returning id
                """,
                self.other_owner_id,
            )
        self.memory_repository = MemoryRepository(self.pool)
        self.repository = HistoricalEvidenceRepository(self.pool)

    async def _cleanup_fixture_owners(self) -> None:
        pool = getattr(self, "pool", None)
        if pool is not None:
            await pool.close()
        result = await cleanup_fixture_owners(
            os.environ["JARVIS_TEST_DB_URL"], {self.owner_id, self.other_owner_id},
        )
        self.assertEqual(result["remaining_rows"], 0)

    async def test_cleanup_removes_promoted_episode_without_breaking_checked_state(self) -> None:
        frame_id = await self._insert_frame(sequence=1)
        event_id = await self._insert_event(frame_ids=[frame_id])
        await self.memory_repository.promote_event_to_episode(
            owner_id=self.owner_id, event_id=event_id,
            title="Disposable promoted episode", content="Retained fixture evidence.",
            importance=0.5, confidence=0.8, source="integration_test",
            promotion_reason="Exercise checked promotion cleanup",
        )
        result = await cleanup_fixture_owners(os.environ["JARVIS_TEST_DB_URL"], {self.owner_id})
        self.assertEqual(result["remaining_rows"], 0)
        self.assertEqual(result["removed_rows"]["memory_items"], 1)
        self.assertEqual(result["removed_rows"]["experience_events"], 1)

    async def test_resolves_only_owner_scoped_available_pixels(self) -> None:
        direct_event_frame = await self._insert_frame(sequence=1)
        direct_memory_frame = await self._insert_frame(sequence=2)
        memory_event_frame = await self._insert_frame(sequence=3)
        expired_frame = await self._insert_frame(
            sequence=4,
            expires_at=self.now - timedelta(seconds=1),
        )
        unavailable_frame = await self._insert_frame(
            sequence=5,
            raw_image_status="expiring",
        )
        other_owner_frame = await self._insert_frame(
            sequence=1,
            owner_id=self.other_owner_id,
            session_id=self.other_session_id,
        )

        direct_event = await self._insert_event(
            frame_ids=[direct_event_frame, expired_frame, unavailable_frame]
        )
        memory_event = await self._insert_event(frame_ids=[memory_event_frame])
        other_event = await self._insert_event(
            frame_ids=[other_owner_frame],
            owner_id=self.other_owner_id,
            session_id=self.other_session_id,
        )

        direct_memory = await self._create_memory(
            memory_type="semantic_fact",
            title="Direct frame evidence",
        )
        await self.memory_repository.attach_evidence(
            owner_id=self.owner_id,
            memory_item_id=direct_memory.id,
            evidence_type="frame",
            evidence_ref_id=direct_memory_frame,
            evidence_time=self.now,
            confidence=1.0,
        )
        event_memory = await self._create_memory(
            memory_type="semantic_fact",
            title="Event-backed memory",
        )
        await self.memory_repository.attach_evidence(
            owner_id=self.owner_id,
            memory_item_id=event_memory.id,
            evidence_type="experience_event",
            evidence_ref_id=memory_event,
            evidence_time=self.now,
            confidence=1.0,
        )

        frames = await self.repository.list_retained_evidence_frames(
            owner_id=self.owner_id,
            candidates=(
                self._candidate("experience_event", direct_event),
                self._candidate("semantic_fact", direct_memory.id),
                self._candidate("episode", event_memory.id),
                self._candidate("experience_event", other_event),
            ),
            limit=6,
        )

        self.assertEqual(
            [frame.visual_frame_id for frame in frames],
            [direct_event_frame, direct_memory_frame, memory_event_frame],
        )
        returned_ids = {frame.visual_frame_id for frame in frames}
        self.assertNotIn(expired_frame, returned_ids)
        self.assertNotIn(unavailable_frame, returned_ids)
        self.assertNotIn(other_owner_frame, returned_ids)

    async def test_deduplicates_frames_and_caps_results_at_six(self) -> None:
        event_frames = [
            await self._insert_frame(sequence=index) for index in range(1, 7)
        ]
        seventh_frame = await self._insert_frame(sequence=7)
        event_id = await self._insert_event(frame_ids=event_frames)
        event_memory = await self._create_memory(
            memory_type="semantic_fact",
            title="Duplicate event evidence",
        )
        await self.memory_repository.attach_evidence(
            owner_id=self.owner_id,
            memory_item_id=event_memory.id,
            evidence_type="experience_event",
            evidence_ref_id=event_id,
            evidence_time=self.now,
            confidence=1.0,
        )
        direct_memory = await self._create_memory(
            memory_type="semantic_fact",
            title="Seventh frame evidence",
        )
        await self.memory_repository.attach_evidence(
            owner_id=self.owner_id,
            memory_item_id=direct_memory.id,
            evidence_type="frame",
            evidence_ref_id=seventh_frame,
            evidence_time=self.now,
            confidence=1.0,
        )

        frames = await self.repository.list_retained_evidence_frames(
            owner_id=self.owner_id,
            candidates=(
                self._candidate("experience_event", event_id),
                self._candidate("episode", event_memory.id),
                self._candidate("semantic_fact", direct_memory.id),
            ),
            limit=6,
        )

        self.assertEqual([frame.visual_frame_id for frame in frames], event_frames)
        self.assertEqual(len({frame.visual_frame_id for frame in frames}), 6)
        self.assertNotIn(seventh_frame, {frame.visual_frame_id for frame in frames})

    async def test_record_identity_requires_actual_type_and_related_evidence(self) -> None:
        direct_frame = await self._insert_frame(sequence=1)
        event_frame = await self._insert_frame(sequence=2)
        unrelated_frame = await self._insert_frame(sequence=3)
        event_id = await self._insert_event(frame_ids=[event_frame])
        direct_memory = await self._create_memory(memory_type="semantic_fact", title="Direct image")
        event_memory = await self._create_memory(memory_type="reflection", title="Event image")
        await self.memory_repository.attach_evidence(
            owner_id=self.owner_id, memory_item_id=direct_memory.id,
            evidence_type="frame", evidence_ref_id=direct_frame,
            evidence_time=self.now, confidence=1.0,
        )
        await self.memory_repository.attach_evidence(
            owner_id=self.owner_id, memory_item_id=event_memory.id,
            evidence_type="experience_event", evidence_ref_id=event_id,
            evidence_time=self.now, confidence=1.0,
        )
        for record_type, record_id, frame_id in (
            ("experience_event", event_id, event_frame),
            ("semantic_fact", direct_memory.id, direct_frame),
            ("reflection", event_memory.id, event_frame),
        ):
            with self.subTest(record_type=record_type):
                evidence = await self.repository.get_image_evidence(
                    owner_id=self.owner_id, record_type=record_type, record_id=record_id,
                )
                self.assertEqual(evidence.reference.frame_id, frame_id)
                self.assertEqual(evidence.reference.record_id, record_id)
                self.assertEqual(evidence.reference.record_type, record_type)
                self.assertEqual(evidence.reference.source, "historical")
                self.assertIsNone(await self.repository.get_image_evidence(
                    owner_id=self.owner_id, record_type=record_type,
                    record_id=record_id, frame_id=unrelated_frame,
                ))
                self.assertIsNone(await self.repository.get_image_evidence(
                    owner_id=self.other_owner_id, record_type=record_type, record_id=record_id,
                ))
        self.assertIsNone(await self.repository.get_image_evidence(
            owner_id=self.owner_id, record_type="episode", record_id=direct_memory.id,
        ))
        self.assertIsNone(await self.repository.get_image_evidence(
            owner_id=self.owner_id, record_type="semantic_fact", record_id=uuid4(),
        ))

    async def test_forgotten_derived_memory_cannot_resurrect_independent_source(self) -> None:
        frame_id = await self._insert_frame(sequence=1)
        event_id = await self._insert_event(frame_ids=[frame_id])
        memory = await self._create_memory(memory_type="reflection", title="Forgettable reflection")
        await self.memory_repository.attach_evidence(
            owner_id=self.owner_id, memory_item_id=memory.id,
            evidence_type="experience_event", evidence_ref_id=event_id,
            evidence_time=self.now, confidence=1.0,
        )
        self.assertIsNotNone(await self.repository.get_image_evidence(
            owner_id=self.owner_id, record_type="reflection", record_id=memory.id,
        ))
        async with self.pool.acquire() as connection:
            await connection.execute(
                "update memory_items set status = 'forgotten' where id = $1 and owner_id = $2",
                memory.id, self.owner_id,
            )
        self.assertIsNone(await self.repository.get_image_evidence(
            owner_id=self.owner_id, record_type="reflection", record_id=memory.id,
        ))
        self.assertEqual(await self.repository.list_retained_evidence_frames(
            owner_id=self.owner_id, candidates=(self._candidate("reflection", memory.id),), limit=6,
        ), ())
        source = await self.repository.get_image_evidence(
            owner_id=self.owner_id, record_type="experience_event", record_id=event_id,
        )
        self.assertEqual(source.reference.frame_id, frame_id)

    async def test_bounded_preview_keeps_each_shared_record_relationship(self) -> None:
        frame_id = await self._insert_frame(sequence=1)
        other_frame = await self._insert_frame(sequence=2)
        event_id = await self._insert_event(frame_ids=[frame_id, other_frame])
        memory = await self._create_memory(memory_type="semantic_fact", title="Shared event evidence")
        await self.memory_repository.attach_evidence(
            owner_id=self.owner_id, memory_item_id=memory.id,
            evidence_type="experience_event", evidence_ref_id=event_id,
            evidence_time=self.now, confidence=1.0,
        )
        records = (("experience_event", event_id), ("semantic_fact", memory.id))
        images = await self.repository.list_image_evidence(
            owner_id=self.owner_id, records=records, limit=1,
        )
        self.assertEqual({image.reference.frame_id for image in images}, {frame_id})
        self.assertEqual(
            {(image.reference.record_type, image.reference.record_id) for image in images},
            set(records),
        )
        self.assertEqual(len(images), 2)

    async def test_explicit_frame_can_be_selected_outside_bounded_preview(self) -> None:
        frame_ids = [await self._insert_frame(sequence=index) for index in range(1, 8)]
        event_id = await self._insert_event(frame_ids=frame_ids)
        preview = await self.repository.list_image_evidence(
            owner_id=self.owner_id, records=(("experience_event", event_id),), limit=6,
        )
        self.assertEqual([item.reference.frame_id for item in preview], frame_ids[:6])
        chosen = await self.repository.get_image_evidence(
            owner_id=self.owner_id, record_type="experience_event",
            record_id=event_id, frame_id=frame_ids[-1],
        )
        self.assertEqual(chosen.reference.frame_id, frame_ids[-1])

    async def test_available_object_metadata_does_not_bypass_expiry_or_expiring_status(self) -> None:
        available = await self._insert_frame(sequence=1)
        expired = await self._insert_frame(sequence=2, expires_at=self.now - timedelta(seconds=1))
        expiring = await self._insert_frame(sequence=3, raw_image_status="expiring")
        event_id = await self._insert_event(frame_ids=[expired, expiring, available])
        primary = await self.repository.get_image_evidence(
            owner_id=self.owner_id, record_type="experience_event", record_id=event_id,
        )
        self.assertEqual(primary.reference.frame_id, available)
        for frame_id in (expired, expiring):
            with self.subTest(frame_id=frame_id):
                self.assertIsNone(await self.repository.get_image_evidence(
                    owner_id=self.owner_id, record_type="experience_event",
                    record_id=event_id, frame_id=frame_id,
                ))

    async def test_same_owner_earlier_session_is_allowed_but_cross_session_event_join_is_not(self) -> None:
        async with self.pool.acquire() as connection:
            earlier_session_id = await connection.fetchval(
                "insert into jarvis_sessions (owner_id, title) values ($1, 'Earlier evidence session') returning id",
                self.owner_id,
            )
        earlier_frame = await self._insert_frame(sequence=1, session_id=earlier_session_id)
        current_frame = await self._insert_frame(sequence=1)
        event_id = await self._insert_event(frame_ids=[earlier_frame, current_frame], session_id=earlier_session_id)
        image = await self.repository.get_image_evidence(
            owner_id=self.owner_id, record_type="experience_event", record_id=event_id,
        )
        self.assertEqual(image.reference.session_id, earlier_session_id)
        self.assertEqual(image.reference.frame_id, earlier_frame)
        self.assertIsNone(await self.repository.get_image_evidence(
            owner_id=self.owner_id, record_type="experience_event", record_id=event_id, frame_id=current_frame,
        ))

    async def _create_memory(self, *, memory_type: str, title: str):
        return await self.memory_repository.create_memory_item(
            owner_id=self.owner_id,
            session_id=self.session_id,
            memory_type=memory_type,
            title=title,
            content=f"{title} content.",
            importance=0.8,
            confidence=0.9,
            source="integration_test",
        )

    async def _insert_frame(
        self,
        *,
        sequence: int,
        owner_id: str | None = None,
        session_id: UUID | None = None,
        expires_at: datetime | None = None,
        raw_image_status: str = "available",
    ) -> UUID:
        frame_id = uuid4()
        async with self.pool.acquire() as connection:
            await connection.execute(
                """
                insert into visual_frames (
                    id, owner_id, session_id, visual_session_id, frame_sequence,
                    source, captured_at, content_type, width, height, storage_bucket,
                    storage_path, raw_image_status, raw_image_expires_at
                ) values (
                    $1, $2, $3, $4, $5, 'glasses', $6, 'image/jpeg', 768, 432,
                    'test-frames', $7, $8, $9
                )
                """,
                frame_id,
                owner_id or self.owner_id,
                session_id or self.session_id,
                uuid4(),
                sequence,
                self.now + timedelta(seconds=sequence),
                f"integration/{frame_id}.jpg",
                raw_image_status,
                expires_at or self.now + timedelta(minutes=30),
            )
        return frame_id

    async def _insert_event(
        self,
        *,
        frame_ids: list[UUID],
        owner_id: str | None = None,
        session_id: UUID | None = None,
    ) -> UUID:
        async with self.pool.acquire() as connection:
            event_id = await connection.fetchval(
                """
                insert into experience_events (
                    owner_id, session_id, status, event_kind, modalities,
                    started_at, ended_at, boundary_reason,
                    keyframes_finalized_at, consolidation_ready_at
                ) values (
                    $1, $2, 'closed', 'visual', array['visual'],
                    $3, $3, 'integration_test', $3, $3
                ) returning id
                """,
                owner_id or self.owner_id,
                session_id or self.session_id,
                self.now,
            )
            await connection.executemany(
                """
                insert into experience_event_keyframes (
                    event_id, visual_frame_id, rank, reason, quality_score, selected_by
                ) values ($1, $2, $3, 'integration test', 0.9, 'deterministic')
                """,
                [
                    (event_id, frame_id, rank)
                    for rank, frame_id in enumerate(frame_ids, start=1)
                ],
            )
        return event_id

    def _candidate(
        self,
        record_type: str,
        document_id: UUID,
    ) -> FusedRetrievalCandidate:
        return FusedRetrievalCandidate(
            document=RetrievalDocument(
                id=document_id,
                owner_id=self.owner_id,
                session_id=self.session_id,
                record_type=record_type,
                occurred_at=self.now,
                valid_from=self.now,
                valid_to=None,
                title=f"{record_type} candidate",
                summary="Retained evidence candidate.",
                content="Retained evidence candidate content.",
                importance=0.8,
                confidence=0.9,
                record_status="active",
                is_explicit=False,
                evidence_count=1,
                keyframe_count=1,
                evidence_types=("frame",),
            ),
            score=0.9,
            sources=("integration_test",),
            exact_match=True,
        )
