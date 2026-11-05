from __future__ import annotations

import unittest
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

from app.memory.embeddings import PreparedEventEmbeddings
from app.memory.embeddings.job_handlers import EmbedExperienceEventJobHandler
from test_episode_job_handlers import _embedding_write, _job


class EmbeddingJobHandlerTests(unittest.IsolatedAsyncioTestCase):
    async def test_event_embedding_without_episode_does_not_enqueue_memory_job(self) -> None:
        job = _job("embed_experience_event", "event_id")
        event_id = UUID(str(job.payload["event_id"]))
        repository = AsyncMock()
        repository.promoted_episode_id.return_value = None
        job_repository = AsyncMock()
        handler = EmbedExperienceEventJobHandler(
            repository=repository,
            service=AsyncMock(),
            image_loader=None,
            job_repository=job_repository,
        )
        prepared = type(
            "Prepared",
            (),
            {
                "session_id": job.session_id,
                "embeddings": PreparedEventEmbeddings(
                    event_id=event_id,
                    owner_id=job.owner_id,
                    text_existing=None,
                    text_write=_embedding_write(event_id),
                    multimodal_existing=None,
                    multimodal_write=None,
                ),
            },
        )()

        result = await handler.commit(job, prepared, AsyncMock())

        self.assertIsNone(result["episode_memory_id"])
        job_repository.enqueue.assert_not_awaited()
