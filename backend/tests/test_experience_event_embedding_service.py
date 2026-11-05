from __future__ import annotations

import unittest
from datetime import UTC, datetime
from typing import Any, Sequence
from uuid import UUID, uuid4

from app.experience.models import ExperienceEvent, ExperienceEventKeyframe
from app.memory.embeddings import (
    EVENT_MULTIMODAL_EMBEDDING_KIND,
    EVENT_TEXT_EMBEDDING_KIND,
    EventEmbeddingKeyframe,
    ExperienceEventEmbeddingService,
)
from app.memory.models import EMBEDDING_DIMENSIONS, MemoryEmbedding, content_hash
from app.model_providers.protocols import VisualModelImage


_JPEG_A = b"\xff\xd8\xff\xe0first\xff\xd9"
_JPEG_B = b"\xff\xd8\xff\xe0second\xff\xd9"


class _EmbeddingModel:
    provider = "gemini"
    model = "gemini-embedding-2"

    def __init__(self, *, fail_on_call: int | None = None) -> None:
        self.calls: list[tuple[str | None, tuple[VisualModelImage, ...]]] = []
        self.fail_on_call = fail_on_call
        self.closed = False

    async def embed_multimodal(
        self,
        *,
        text: str | None,
        images: Sequence[VisualModelImage],
    ) -> list[float]:
        self.calls.append((text, tuple(images)))
        if self.fail_on_call == len(self.calls):
            raise RuntimeError("provider failed")
        return [0.01] * EMBEDDING_DIMENSIONS

    async def close(self) -> None:
        self.closed = True


class _Repository:
    def __init__(self) -> None:
        self.embeddings: dict[str, MemoryEmbedding] = {}
        self.writes: list[dict[str, Any]] = []

    async def get_event_embedding(
        self,
        *,
        owner_id: str,
        event_id: UUID,
        embedding_kind: str,
        provider: str,
        model: str,
    ) -> MemoryEmbedding | None:
        return self.embeddings.get(embedding_kind)

    async def upsert_event_embedding(self, **values: Any) -> MemoryEmbedding:
        self.writes.append(values)
        now = datetime.now(UTC)
        result = MemoryEmbedding(
            subject_id=values["event_id"],
            embedding_kind=values["embedding_kind"],
            provider=values["provider"],
            model=values["model"],
            dimensions=EMBEDDING_DIMENSIONS,
            embedded_content=values["embedded_content"],
            content_hash=content_hash(values["embedded_content"]),
            embedding=tuple(values["embedding"]),
            embedded_at=now,
            updated_at=now,
        )
        self.embeddings[result.embedding_kind] = result
        return result


class ExperienceEventEmbeddingServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_finalized_event_creates_tagged_text_and_multimodal_embeddings(self) -> None:
        event = _event()
        repository = _Repository()
        model = _EmbeddingModel()
        service = ExperienceEventEmbeddingService(repository, model)

        result = await service.embed_event(
            event=event,
            keyframes=(_keyframe(event, 2, _JPEG_B), _keyframe(event, 1, _JPEG_A)),
        )

        self.assertEqual(
            [write["embedding_kind"] for write in repository.writes],
            [EVENT_TEXT_EMBEDDING_KIND, EVENT_MULTIMODAL_EMBEDDING_KIND],
        )
        self.assertEqual(result.text_embedding.provider, "gemini")
        self.assertEqual(result.multimodal_embedding.model, "gemini-embedding-2")
        self.assertTrue(model.calls[0][0].startswith("title: Walk through a cafe | text:"))
        self.assertNotIn(str(event.id), model.calls[0][0])
        self.assertEqual([image.label for image in model.calls[1][1]], [
            "Event keyframe rank 1: reason 1",
            "Event keyframe rank 2: reason 2",
        ])
        multimodal_content = repository.writes[1]["embedded_content"]
        self.assertIn("image_sha256", multimodal_content)
        self.assertNotIn(_JPEG_A.hex(), multimodal_content)
        self.assertNotIn("image_sha256", model.calls[1][0])

    async def test_unchanged_content_skips_provider_and_database_writes(self) -> None:
        event = _event()
        repository = _Repository()
        model = _EmbeddingModel()
        service = ExperienceEventEmbeddingService(repository, model)
        keyframes = (_keyframe(event, 1, _JPEG_A),)
        await service.embed_event(event=event, keyframes=keyframes)
        model.calls.clear()
        repository.writes.clear()

        await service.embed_event(event=event, keyframes=keyframes)

        self.assertEqual(model.calls, [])
        self.assertEqual(repository.writes, [])

    async def test_changed_image_only_recomputes_multimodal_embedding(self) -> None:
        event = _event()
        repository = _Repository()
        model = _EmbeddingModel()
        service = ExperienceEventEmbeddingService(repository, model)
        await service.embed_event(event=event, keyframes=(_keyframe(event, 1, _JPEG_A),))
        model.calls.clear()
        repository.writes.clear()

        await service.embed_event(event=event, keyframes=(_keyframe(event, 1, _JPEG_B),))

        self.assertEqual(len(model.calls), 1)
        self.assertEqual(len(model.calls[0][1]), 1)
        self.assertEqual(
            [write["embedding_kind"] for write in repository.writes],
            [EVENT_MULTIMODAL_EMBEDDING_KIND],
        )

    async def test_provider_failure_happens_before_any_database_write(self) -> None:
        event = _event()
        repository = _Repository()
        service = ExperienceEventEmbeddingService(
            repository,
            _EmbeddingModel(fail_on_call=2),
        )

        with self.assertRaisesRegex(RuntimeError, "provider failed"):
            await service.embed_event(
                event=event,
                keyframes=(_keyframe(event, 1, _JPEG_A),),
            )

        self.assertEqual(repository.writes, [])
        self.assertEqual(repository.embeddings, {})

    async def test_open_event_and_invalid_keyframes_are_rejected(self) -> None:
        repository = _Repository()
        model = _EmbeddingModel()
        service = ExperienceEventEmbeddingService(repository, model)
        open_event = _event(open_event=True)

        with self.assertRaisesRegex(ValueError, "finalized"):
            await service.embed_event(event=open_event)

        event = _event()
        wrong_event = _event()
        with self.assertRaisesRegex(ValueError, "different event"):
            await service.embed_event(
                event=event,
                keyframes=(_keyframe(wrong_event, 1, _JPEG_A),),
            )
        with self.assertRaisesRegex(ValueError, "not a JPEG"):
            await service.embed_event(
                event=event,
                keyframes=(_keyframe(event, 1, b"bad"),),
            )

        self.assertEqual(model.calls, [])


def _event(*, open_event: bool = False) -> ExperienceEvent:
    now = datetime.now(UTC)
    marker = None if open_event else now
    return ExperienceEvent(
        id=uuid4(),
        owner_id="owner",
        session_id=uuid4(),
        status="open" if open_event else "closed",
        event_kind="visual",
        modalities=("visual",),
        started_at=now,
        ended_at=marker,
        title="Walk through a cafe",
        summary="The user entered a quiet cafe.",
        scene="cafe",
        activity="walking",
        state={"visible_text": ["Menu"]},
        boundary_reason=None if open_event else "scene_changed",
        importance=0.7,
        confidence=0.9,
        keyframes_finalized_at=marker,
        consolidation_ready_at=marker,
        evidence_reduced_at=marker,
        retention_expires_at=marker,
        consolidation_result=None,
        consolidation_provider=None,
        consolidation_model=None,
        consolidated_at=None,
        created_at=now,
        updated_at=now,
    )


def _keyframe(
    event: ExperienceEvent,
    rank: int,
    image_bytes: bytes,
) -> EventEmbeddingKeyframe:
    return EventEmbeddingKeyframe(
        keyframe=ExperienceEventKeyframe(
            event_id=event.id,
            visual_frame_id=uuid4(),
            rank=rank,
            reason=f"reason {rank}",
            quality_score=0.8,
            selected_by="deterministic",
            created_at=datetime.now(UTC),
        ),
        image_bytes=image_bytes,
    )
