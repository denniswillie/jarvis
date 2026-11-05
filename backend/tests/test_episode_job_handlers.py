from __future__ import annotations

import unittest
from datetime import UTC, datetime
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

from app.memory.embeddings import PreparedEmbeddingWrite, PreparedEventEmbeddings
from app.memory.embeddings.job_handlers import (
    EMBED_MEMORY_ITEM_JOB,
    EmbedExperienceEventJobHandler,
)
from app.memory.episodic.job_handler import (
    PreparedEpisodePromotion,
    PromoteExperienceEventJobHandler,
)
from app.memory.episodic.repository import EpisodePromotionOutcome
from app.memory.episodic.service import EpisodePromotionPlan
from app.memory.jobs.models import MemoryJob
from app.memory.models import EMBEDDING_DIMENSIONS


class EpisodeJobHandlerTests(unittest.IsolatedAsyncioTestCase):
    async def test_promotion_commit_enqueues_event_embedding_atomically(self) -> None:
        job = _job("promote_experience_event", "event_id")
        event_id = UUID(str(job.payload["event_id"]))
        plan = EpisodePromotionPlan(
            event_id=event_id,
            owner_id=job.owner_id,
            session_id=job.session_id,  # type: ignore[arg-type]
            promote=False,
            reason="routine",
            is_explicit=False,
            retain_keyframes_permanently=False,
            final_keyframes=(),
        )
        repository = AsyncMock()
        repository.apply_promotion.return_value = EpisodePromotionOutcome(
            event_id=event_id,
            decision="skipped",
            reason="routine",
            episode_memory_id=None,
            selected_keyframe_ids=(),
        )
        job_repository = AsyncMock()
        handler = PromoteExperienceEventJobHandler(
            repository=repository,
            service=AsyncMock(),
            embedding_job_repository=job_repository,
            ordinary_keyframe_retention_seconds=86_400,
            skipped_event_retention_days=90,
        )
        prepared = PreparedEpisodePromotion(
            context=object(),  # type: ignore[arg-type]
            plan=plan,
        )
        connection = AsyncMock()

        result = await handler.commit(job, prepared, connection)

        self.assertEqual(result["promotion_decision"], "skipped")
        job_repository.enqueue.assert_awaited_once()
        self.assertIs(
            job_repository.enqueue.await_args.kwargs["connection"],
            connection,
        )

    async def test_promotion_commit_succeeds_when_embeddings_are_disabled(self) -> None:
        job = _job("promote_experience_event", "event_id")
        event_id = UUID(str(job.payload["event_id"]))
        plan = EpisodePromotionPlan(
            event_id=event_id,
            owner_id=job.owner_id,
            session_id=job.session_id,  # type: ignore[arg-type]
            promote=True,
            reason="important",
            is_explicit=False,
            retain_keyframes_permanently=True,
            final_keyframes=(),
        )
        repository = AsyncMock()
        repository.apply_promotion.return_value = EpisodePromotionOutcome(
            event_id=event_id,
            decision="promoted",
            reason="important",
            episode_memory_id=uuid4(),
            selected_keyframe_ids=(),
        )
        handler = PromoteExperienceEventJobHandler(
            repository=repository,
            service=AsyncMock(),
            embedding_job_repository=None,
            ordinary_keyframe_retention_seconds=86_400,
            skipped_event_retention_days=90,
        )

        result = await handler.commit(
            job,
            PreparedEpisodePromotion(context=object(), plan=plan),  # type: ignore[arg-type]
            AsyncMock(),
        )

        self.assertEqual(result["promotion_decision"], "promoted")

    async def test_event_embedding_commit_enqueues_promoted_memory_embedding(self) -> None:
        job = _job("embed_experience_event", "event_id")
        event_id = UUID(str(job.payload["event_id"]))
        episode_id = uuid4()
        repository = AsyncMock()
        repository.promoted_episode_id.return_value = episode_id
        job_repository = AsyncMock()
        handler = EmbedExperienceEventJobHandler(
            repository=repository,
            service=AsyncMock(),
            image_loader=None,
            job_repository=job_repository,
        )
        prepared = PreparedEventEmbeddings(
            event_id=event_id,
            owner_id=job.owner_id,
            text_existing=None,
            text_write=_embedding_write(event_id),
            multimodal_existing=None,
            multimodal_write=None,
        )
        connection = AsyncMock()

        result = await handler.commit(
            job,
            type(
                "Prepared",
                (),
                {"session_id": job.session_id, "embeddings": prepared},
            )(),
            connection,
        )

        self.assertEqual(result["episode_memory_id"], str(episode_id))
        repository.store_event_embedding.assert_awaited_once()
        self.assertEqual(
            job_repository.enqueue.await_args.kwargs["job_type"],
            EMBED_MEMORY_ITEM_JOB,
        )
        self.assertIs(
            job_repository.enqueue.await_args.kwargs["connection"],
            connection,
        )


def _job(job_type: str, payload_key: str) -> MemoryJob:
    now = datetime.now(UTC)
    source_id = uuid4()
    return MemoryJob(
        id=uuid4(),
        owner_id="owner",
        session_id=uuid4(),
        job_type=job_type,
        dedupe_key=f"{job_type}:{source_id}",
        payload={payload_key: str(source_id)},
        status="claimed",
        priority=0,
        attempt_count=1,
        max_attempts=3,
        available_at=now,
        claimed_at=now,
        claimed_by="worker",
        claim_token=uuid4(),
        result=None,
        last_error=None,
        completed_at=None,
        created_at=now,
        updated_at=now,
    )


def _embedding_write(subject_id: UUID) -> PreparedEmbeddingWrite:
    return PreparedEmbeddingWrite(
        subject_id=subject_id,
        embedding_kind="event_text",
        provider="gemini",
        model="gemini-embedding-2",
        embedded_content="content",
        embedding=tuple([0.01] * EMBEDDING_DIMENSIONS),
    )
