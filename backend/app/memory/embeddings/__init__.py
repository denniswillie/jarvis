from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol
from uuid import UUID

from app.experience.models import ExperienceEvent, ExperienceEventKeyframe
from app.memory.models import MemoryEmbedding, MemoryItem, content_hash
from app.model_providers.protocols import MultimodalEmbeddingModel, VisualModelImage


EVENT_TEXT_EMBEDDING_KIND = "event_text"
EVENT_MULTIMODAL_EMBEDDING_KIND = "event_multimodal"
EVENT_EMBEDDING_SCHEMA = "jarvis.experience_event_embedding.v1"
EVENT_EMBEDDING_MAX_KEYFRAMES = 6
MEMORY_TEXT_EMBEDDING_KIND = "memory_text"
MEMORY_EMBEDDING_SCHEMA = "jarvis.memory_item_embedding.v1"


class EventEmbeddingRepository(Protocol):
    async def get_event_embedding(
        self,
        *,
        owner_id: str,
        event_id: UUID,
        embedding_kind: str,
        provider: str,
        model: str,
    ) -> MemoryEmbedding | None: ...

    async def upsert_event_embedding(
        self,
        *,
        owner_id: str,
        event_id: UUID,
        embedding_kind: str,
        provider: str,
        model: str,
        embedded_content: str,
        embedding: Sequence[float],
    ) -> MemoryEmbedding: ...

    async def get_memory_embedding(
        self,
        *,
        owner_id: str,
        memory_item_id: UUID,
        embedding_kind: str,
        provider: str,
        model: str,
    ) -> MemoryEmbedding | None: ...

    async def upsert_memory_embedding(
        self,
        *,
        owner_id: str,
        memory_item_id: UUID,
        embedding_kind: str,
        provider: str,
        model: str,
        embedded_content: str,
        embedding: Sequence[float],
    ) -> MemoryEmbedding: ...


@dataclass(frozen=True, slots=True)
class EventEmbeddingKeyframe:
    keyframe: ExperienceEventKeyframe
    image_bytes: bytes
    content_type: str = "image/jpeg"


@dataclass(frozen=True, slots=True)
class EventEmbeddingResult:
    text_embedding: MemoryEmbedding
    multimodal_embedding: MemoryEmbedding | None


@dataclass(frozen=True, slots=True)
class PreparedEmbeddingWrite:
    subject_id: UUID
    embedding_kind: str
    provider: str
    model: str
    embedded_content: str
    embedding: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class PreparedEventEmbeddings:
    event_id: UUID
    owner_id: str
    text_existing: MemoryEmbedding | None
    text_write: PreparedEmbeddingWrite | None
    multimodal_existing: MemoryEmbedding | None
    multimodal_write: PreparedEmbeddingWrite | None


@dataclass(frozen=True, slots=True)
class PreparedMemoryEmbedding:
    memory_item_id: UUID
    owner_id: str
    existing: MemoryEmbedding | None
    write: PreparedEmbeddingWrite | None


class ExperienceEventEmbeddingService:
    """Embeds finalized events without owning scheduling, storage, or retention."""

    def __init__(
        self,
        repository: EventEmbeddingRepository,
        model: MultimodalEmbeddingModel,
    ) -> None:
        self._repository = repository
        self._model = model

    async def embed_event(
        self,
        *,
        event: ExperienceEvent,
        keyframes: Sequence[EventEmbeddingKeyframe] = (),
    ) -> EventEmbeddingResult:
        prepared = await self.prepare_event_embeddings(
            event=event,
            keyframes=keyframes,
        )
        text_embedding = prepared.text_existing
        if prepared.text_write is not None:
            text_embedding = await self._repository.upsert_event_embedding(
                owner_id=prepared.owner_id,
                event_id=prepared.event_id,
                embedding_kind=prepared.text_write.embedding_kind,
                provider=prepared.text_write.provider,
                model=prepared.text_write.model,
                embedded_content=prepared.text_write.embedded_content,
                embedding=prepared.text_write.embedding,
            )
        if text_embedding is None:
            raise AssertionError("event text embedding was not produced")

        multimodal_embedding = prepared.multimodal_existing
        if prepared.multimodal_write is not None:
            multimodal_embedding = await self._repository.upsert_event_embedding(
                owner_id=prepared.owner_id,
                event_id=prepared.event_id,
                embedding_kind=prepared.multimodal_write.embedding_kind,
                provider=prepared.multimodal_write.provider,
                model=prepared.multimodal_write.model,
                embedded_content=prepared.multimodal_write.embedded_content,
                embedding=prepared.multimodal_write.embedding,
            )
        return EventEmbeddingResult(
            text_embedding=text_embedding,
            multimodal_embedding=multimodal_embedding,
        )

    async def prepare_event_embeddings(
        self,
        *,
        event: ExperienceEvent,
        keyframes: Sequence[EventEmbeddingKeyframe] = (),
    ) -> PreparedEventEmbeddings:
        """Run provider work without persisting, for a durable job commit."""

        ordered_keyframes = _validate_keyframes(event, keyframes)
        text_content = build_event_text_embedding_content(event)
        semantic_document = build_event_retrieval_document(event)
        multimodal_content = (
            build_event_multimodal_embedding_content(event, ordered_keyframes)
            if ordered_keyframes
            else None
        )

        existing_text = await self._repository.get_event_embedding(
            owner_id=event.owner_id,
            event_id=event.id,
            embedding_kind=EVENT_TEXT_EMBEDDING_KIND,
            provider=self._model.provider,
            model=self._model.model,
        )
        existing_multimodal = None
        if multimodal_content is not None:
            existing_multimodal = await self._repository.get_event_embedding(
                owner_id=event.owner_id,
                event_id=event.id,
                embedding_kind=EVENT_MULTIMODAL_EMBEDDING_KIND,
                provider=self._model.provider,
                model=self._model.model,
            )

        text_changed = not _matches_content(existing_text, text_content)
        multimodal_changed = multimodal_content is not None and not _matches_content(
            existing_multimodal,
            multimodal_content,
        )

        text_vector: Sequence[float] | None = None
        multimodal_vector: Sequence[float] | None = None
        if text_changed:
            text_vector = await self._model.embed_multimodal(
                text=semantic_document,
                images=(),
            )
        if multimodal_changed:
            multimodal_vector = await self._model.embed_multimodal(
                text=semantic_document,
                images=tuple(
                    VisualModelImage(
                        label=(
                            f"Event keyframe rank {item.keyframe.rank}: "
                            f"{item.keyframe.reason}"
                        ),
                        image_bytes=item.image_bytes,
                        content_type=item.content_type,
                    )
                    for item in ordered_keyframes
                ),
            )

        return PreparedEventEmbeddings(
            event_id=event.id,
            owner_id=event.owner_id,
            text_existing=existing_text,
            text_write=(
                _prepared_write(
                    subject_id=event.id,
                    kind=EVENT_TEXT_EMBEDDING_KIND,
                    provider=self._model.provider,
                    model=self._model.model,
                    content=text_content,
                    vector=text_vector,
                )
                if text_changed
                else None
            ),
            multimodal_existing=existing_multimodal,
            multimodal_write=(
                _prepared_write(
                    subject_id=event.id,
                    kind=EVENT_MULTIMODAL_EMBEDDING_KIND,
                    provider=self._model.provider,
                    model=self._model.model,
                    content=multimodal_content,
                    vector=multimodal_vector,
                )
                if multimodal_changed
                else None
            ),
        )

    async def prepare_memory_item_embedding(
        self,
        *,
        memory: MemoryItem,
    ) -> PreparedMemoryEmbedding:
        content = build_memory_item_embedding_content(memory)
        existing = await self._repository.get_memory_embedding(
            owner_id=memory.owner_id,
            memory_item_id=memory.id,
            embedding_kind=MEMORY_TEXT_EMBEDDING_KIND,
            provider=self._model.provider,
            model=self._model.model,
        )
        if _matches_content(existing, content):
            return PreparedMemoryEmbedding(
                memory_item_id=memory.id,
                owner_id=memory.owner_id,
                existing=existing,
                write=None,
            )
        vector = await self._model.embed_multimodal(
            text=build_memory_item_retrieval_document(memory),
            images=(),
        )
        return PreparedMemoryEmbedding(
            memory_item_id=memory.id,
            owner_id=memory.owner_id,
            existing=existing,
            write=_prepared_write(
                subject_id=memory.id,
                kind=MEMORY_TEXT_EMBEDDING_KIND,
                provider=self._model.provider,
                model=self._model.model,
                content=content,
                vector=vector,
            ),
        )

    async def close(self) -> None:
        await self._model.close()


def build_event_text_embedding_content(event: ExperienceEvent) -> str:
    _require_finalized_event(event)
    return _canonical_json(
        {
            "schema": EVENT_EMBEDDING_SCHEMA,
            "kind": EVENT_TEXT_EMBEDDING_KIND,
            "event": _event_content(event),
        }
    )


def build_event_retrieval_document(event: ExperienceEvent) -> str:
    """Return Gemini Embedding 2's asymmetric document representation."""

    _require_finalized_event(event)
    title = event.title.strip() or "none"
    details = (
        f"event kind: {event.event_kind}",
        f"modalities: {', '.join(event.modalities) or 'none'}",
        f"started at: {event.started_at.isoformat()}",
        f"ended at: {event.ended_at.isoformat() if event.ended_at else 'unknown'}",
        f"summary: {event.summary.strip() or 'none'}",
        f"scene: {event.scene.strip() if event.scene else 'none'}",
        f"activity: {event.activity.strip() if event.activity else 'none'}",
        f"state: {_canonical_json(dict(event.state))}",
        f"boundary reason: {event.boundary_reason or 'none'}",
    )
    return f"title: {title} | text: {' | '.join(details)}"


def build_event_multimodal_embedding_content(
    event: ExperienceEvent,
    keyframes: Sequence[EventEmbeddingKeyframe],
) -> str:
    ordered_keyframes = _validate_keyframes(event, keyframes)
    if not ordered_keyframes:
        raise ValueError("multimodal event embedding requires at least one keyframe")
    return _canonical_json(
        {
            "schema": EVENT_EMBEDDING_SCHEMA,
            "kind": EVENT_MULTIMODAL_EMBEDDING_KIND,
            "event": _event_content(event),
            "keyframes": [
                {
                    "visual_frame_id": str(item.keyframe.visual_frame_id),
                    "rank": item.keyframe.rank,
                    "reason": item.keyframe.reason,
                    "quality_score": item.keyframe.quality_score,
                    "selected_by": item.keyframe.selected_by,
                    "content_type": item.content_type,
                    "image_sha256": hashlib.sha256(item.image_bytes).hexdigest(),
                }
                for item in ordered_keyframes
            ],
        }
    )


def build_memory_item_embedding_content(memory: MemoryItem) -> str:
    return _canonical_json(
        {
            "schema": MEMORY_EMBEDDING_SCHEMA,
            "kind": MEMORY_TEXT_EMBEDDING_KIND,
            "memory": {
                "memory_type": memory.memory_type,
                "title": memory.title,
                "content": memory.content,
                "summary": memory.summary,
                "occurred_at": (
                    memory.occurred_at.isoformat() if memory.occurred_at else None
                ),
                "valid_from": (
                    memory.valid_from.isoformat() if memory.valid_from else None
                ),
                "valid_to": memory.valid_to.isoformat() if memory.valid_to else None,
                "importance": memory.importance,
                "confidence": memory.confidence,
                "is_explicit": memory.is_explicit,
            },
        }
    )


def build_memory_item_retrieval_document(memory: MemoryItem) -> str:
    return (
        f"title: {memory.title} | text: memory type: {memory.memory_type} | "
        f"content: {memory.content} | summary: {memory.summary or 'none'} | "
        f"occurred at: {memory.occurred_at.isoformat() if memory.occurred_at else 'unknown'}"
    )


def _event_content(event: ExperienceEvent) -> Mapping[str, object]:
    return {
        "id": str(event.id),
        "event_kind": event.event_kind,
        "modalities": list(event.modalities),
        "started_at": event.started_at.isoformat(),
        "ended_at": event.ended_at.isoformat() if event.ended_at else None,
        "title": event.title,
        "summary": event.summary,
        "scene": event.scene,
        "activity": event.activity,
        "state": event.state,
        "boundary_reason": event.boundary_reason,
        "importance": event.importance,
        "confidence": event.confidence,
    }


def _validate_keyframes(
    event: ExperienceEvent,
    keyframes: Sequence[EventEmbeddingKeyframe],
) -> tuple[EventEmbeddingKeyframe, ...]:
    _require_finalized_event(event)
    ordered = tuple(sorted(keyframes, key=lambda item: item.keyframe.rank))
    if len(ordered) > EVENT_EMBEDDING_MAX_KEYFRAMES:
        raise ValueError(
            f"event embedding accepts at most {EVENT_EMBEDDING_MAX_KEYFRAMES} keyframes"
        )
    frame_ids: set[UUID] = set()
    ranks: set[int] = set()
    for item in ordered:
        if item.keyframe.event_id != event.id:
            raise ValueError("event embedding keyframe belongs to a different event")
        if item.keyframe.visual_frame_id in frame_ids:
            raise ValueError("event embedding keyframe IDs must be unique")
        if item.keyframe.rank in ranks:
            raise ValueError("event embedding keyframe ranks must be unique")
        if item.content_type != "image/jpeg":
            raise ValueError("event embedding keyframes must be JPEG")
        if not _is_jpeg(item.image_bytes):
            raise ValueError("event embedding keyframe bytes are not a JPEG")
        frame_ids.add(item.keyframe.visual_frame_id)
        ranks.add(item.keyframe.rank)
    return ordered


def _require_finalized_event(event: ExperienceEvent) -> None:
    if (
        event.status == "open"
        or event.ended_at is None
        or event.keyframes_finalized_at is None
        or event.consolidation_ready_at is None
    ):
        raise ValueError("event must be finalized before embedding")


def _matches_content(embedding: MemoryEmbedding | None, content: str) -> bool:
    return embedding is not None and embedding.content_hash == content_hash(content)


def _prepared_write(
    *,
    subject_id: UUID,
    kind: str,
    provider: str,
    model: str,
    content: str | None,
    vector: Sequence[float] | None,
) -> PreparedEmbeddingWrite:
    if content is None or vector is None:
        raise AssertionError("changed embedding has no content or vector")
    return PreparedEmbeddingWrite(
        subject_id=subject_id,
        embedding_kind=kind,
        provider=provider,
        model=model,
        embedded_content=content,
        embedding=tuple(float(value) for value in vector),
    )


def _canonical_json(value: Mapping[str, object]) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _is_jpeg(value: bytes) -> bool:
    return len(value) >= 4 and value[:3] == b"\xff\xd8\xff" and value[-2:] == b"\xff\xd9"
