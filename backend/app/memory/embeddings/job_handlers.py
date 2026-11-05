from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Mapping, Protocol
from uuid import UUID

import asyncpg

from app.experience.models import ExperienceEventKeyframe
from app.memory.embeddings import (
    EventEmbeddingKeyframe,
    ExperienceEventEmbeddingService,
    PreparedEventEmbeddings,
    PreparedEmbeddingWrite,
    PreparedMemoryEmbedding,
)
from app.memory.episodic.repository import EpisodicMemoryRepository
from app.memory.jobs.handlers import (
    MemoryJobTransientError,
    MemoryJobValidationError,
)
from app.memory.jobs.models import MemoryJob
from app.memory.jobs.repository import MemoryJobRepository
from app.memory.models import content_hash
from app.memory.semantic.contracts import (
    EMBED_MEMORY_ENTITY_JOB,
    ENTITY_IDENTITY_EMBEDDING_KIND,
    RECONCILE_SEMANTIC_MEMORY_JOB,
    build_entity_embedding_content,
    semantic_reconciliation_dedupe_key,
)
from app.memory.semantic.repository import SemanticMemoryRepository
from app.model_providers.gemini_multimodal_embeddings import (
    GeminiMultimodalEmbeddingError,
)
from app.model_providers.protocols import MultimodalEmbeddingModel


EMBED_EXPERIENCE_EVENT_JOB = "embed_experience_event"
EMBED_MEMORY_ITEM_JOB = "embed_memory_item"


def event_embedding_dedupe_key(event_id: UUID) -> str:
    return f"experience-event:{event_id}:embed"


def memory_embedding_dedupe_key(
    memory_item_id: UUID,
    *,
    content_digest: str | None = None,
) -> str:
    suffix = f":{content_digest}" if content_digest is not None else ""
    return f"memory-item:{memory_item_id}:embed{suffix}"


class EmbeddingImageLoader(Protocol):
    async def download_object(self, *, bucket: str, path: str) -> bytes: ...


@dataclass(frozen=True, slots=True)
class PreparedEventEmbeddingJob:
    session_id: UUID
    embeddings: PreparedEventEmbeddings


class EmbedExperienceEventJobHandler:
    job_type = EMBED_EXPERIENCE_EVENT_JOB

    def __init__(
        self,
        *,
        repository: EpisodicMemoryRepository,
        service: ExperienceEventEmbeddingService,
        image_loader: EmbeddingImageLoader | None,
        job_repository: MemoryJobRepository,
    ) -> None:
        self._repository = repository
        self._service = service
        self._image_loader = image_loader
        self._job_repository = job_repository

    async def prepare(self, job: MemoryJob) -> PreparedEventEmbeddingJob:
        event_id = _required_uuid(job.payload, "event_id")
        session_id = _required_session_id(job)
        try:
            event = await self._repository.get_event_for_embedding(
                owner_id=job.owner_id,
                session_id=session_id,
                event_id=event_id,
            )
            rows = await self._repository.list_event_keyframe_sources(
                owner_id=job.owner_id,
                session_id=session_id,
                event_id=event_id,
            )
            keyframes: list[EventEmbeddingKeyframe] = []
            if self._image_loader is not None:
                for row in rows:
                    if row["raw_image_status"] != "available":
                        continue
                    image_bytes = await self._image_loader.download_object(
                        bucket=row["storage_bucket"],
                        path=row["storage_path"],
                    )
                    keyframes.append(
                        EventEmbeddingKeyframe(
                            keyframe=ExperienceEventKeyframe(
                                event_id=row["event_id"],
                                visual_frame_id=row["visual_frame_id"],
                                rank=row["rank"],
                                reason=row["reason"],
                                quality_score=float(row["quality_score"]),
                                selected_by=row["selected_by"],
                                created_at=row["created_at"],
                            ),
                            image_bytes=image_bytes,
                            content_type=row["content_type"],
                        )
                    )
            embeddings = await self._service.prepare_event_embeddings(
                event=event,
                keyframes=tuple(keyframes),
            )
            return PreparedEventEmbeddingJob(
                session_id=session_id,
                embeddings=embeddings,
            )
        except ValueError as error:
            raise MemoryJobValidationError(str(error)) from error
        except GeminiMultimodalEmbeddingError as error:
            raise MemoryJobTransientError(str(error)) from error

    async def commit(
        self,
        job: MemoryJob,
        prepared_result: PreparedEventEmbeddingJob,
        connection: asyncpg.Connection,
    ) -> Mapping[str, Any]:
        embeddings = prepared_result.embeddings
        if embeddings.owner_id != job.owner_id:
            raise MemoryJobValidationError("prepared event embedding owner changed")
        if embeddings.event_id != _required_uuid(job.payload, "event_id"):
            raise MemoryJobValidationError("prepared event embedding source changed")
        writes = tuple(
            write
            for write in (embeddings.text_write, embeddings.multimodal_write)
            if write is not None
        )
        for write in writes:
            await self._repository.store_event_embedding(
                owner_id=job.owner_id,
                write=write,
                connection=connection,
            )
        episode_id = await self._repository.promoted_episode_id(
            owner_id=job.owner_id,
            event_id=embeddings.event_id,
            connection=connection,
        )
        if episode_id is not None:
            await self._job_repository.enqueue(
                owner_id=job.owner_id,
                session_id=prepared_result.session_id,
                job_type=EMBED_MEMORY_ITEM_JOB,
                dedupe_key=memory_embedding_dedupe_key(episode_id),
                payload={"memory_item_id": str(episode_id)},
                available_at=datetime.now(UTC),
                connection=connection,
            )
        return {
            "event_id": str(embeddings.event_id),
            "embedding_writes": len(writes),
            "episode_memory_id": str(episode_id) if episode_id else None,
        }


class EmbedMemoryItemJobHandler:
    job_type = EMBED_MEMORY_ITEM_JOB

    def __init__(
        self,
        *,
        repository: EpisodicMemoryRepository,
        service: ExperienceEventEmbeddingService,
        job_repository: MemoryJobRepository | None = None,
    ) -> None:
        self._repository = repository
        self._service = service
        self._job_repository = job_repository

    async def prepare(self, job: MemoryJob) -> PreparedMemoryEmbedding:
        memory_item_id = _required_uuid(job.payload, "memory_item_id")
        try:
            memory = await self._repository.get_memory_item(
                owner_id=job.owner_id,
                memory_item_id=memory_item_id,
            )
            return await self._service.prepare_memory_item_embedding(memory=memory)
        except ValueError as error:
            raise MemoryJobValidationError(str(error)) from error
        except GeminiMultimodalEmbeddingError as error:
            raise MemoryJobTransientError(str(error)) from error

    async def commit(
        self,
        job: MemoryJob,
        prepared_result: PreparedMemoryEmbedding,
        connection: asyncpg.Connection,
    ) -> Mapping[str, Any]:
        memory_item_id = _required_uuid(job.payload, "memory_item_id")
        if prepared_result.owner_id != job.owner_id:
            raise MemoryJobValidationError("prepared memory embedding owner changed")
        if prepared_result.memory_item_id != memory_item_id:
            raise MemoryJobValidationError("prepared memory embedding source changed")
        if prepared_result.write is not None:
            await self._repository.store_memory_embedding(
                owner_id=job.owner_id,
                write=prepared_result.write,
                connection=connection,
            )
        semantic_event_id: UUID | None = None
        if self._job_repository is not None:
            semantic_event = await connection.fetchrow(
                """
                select event.id, event.session_id
                from memory_items memory
                join experience_events event
                  on event.promoted_episode_memory_id = memory.id
                 and event.owner_id = memory.owner_id
                where memory.owner_id = $1
                  and memory.id = $2
                  and memory.memory_type = 'episode'
                  and event.semantic_reconciled_at is null
                """,
                job.owner_id,
                memory_item_id,
            )
            if semantic_event is not None:
                semantic_event_id = semantic_event["id"]
                await self._job_repository.enqueue(
                    owner_id=job.owner_id,
                    session_id=semantic_event["session_id"],
                    job_type=RECONCILE_SEMANTIC_MEMORY_JOB,
                    dedupe_key=semantic_reconciliation_dedupe_key(semantic_event_id),
                    payload={"event_id": str(semantic_event_id)},
                    available_at=datetime.now(UTC),
                    connection=connection,
                )
        return {
            "memory_item_id": str(memory_item_id),
            "embedding_written": prepared_result.write is not None,
            "semantic_event_id": str(semantic_event_id) if semantic_event_id else None,
        }


@dataclass(frozen=True, slots=True)
class PreparedEntityEmbedding:
    owner_id: str
    entity_id: UUID
    expected_content_hash: str
    write: PreparedEmbeddingWrite | None
    stale: bool = False


class EmbedMemoryEntityJobHandler:
    job_type = EMBED_MEMORY_ENTITY_JOB

    def __init__(
        self,
        *,
        repository: SemanticMemoryRepository,
        model: MultimodalEmbeddingModel,
    ) -> None:
        self._repository = repository
        self._model = model

    async def prepare(self, job: MemoryJob) -> PreparedEntityEmbedding:
        entity_id = _required_uuid(job.payload, "entity_id")
        expected_hash = job.payload.get("content_hash")
        if not isinstance(expected_hash, str) or len(expected_hash) != 64:
            raise MemoryJobValidationError("entity content_hash must be a SHA-256 digest")
        try:
            entity = await self._repository.get_entity(
                owner_id=job.owner_id,
                entity_id=entity_id,
            )
            content = build_entity_embedding_content(entity)
            actual_hash = content_hash(content)
            if actual_hash != expected_hash:
                return PreparedEntityEmbedding(
                    owner_id=job.owner_id,
                    entity_id=entity_id,
                    expected_content_hash=expected_hash,
                    write=None,
                    stale=True,
                )
            existing_hash = await self._repository.get_entity_embedding_hash(
                owner_id=job.owner_id,
                entity_id=entity_id,
                provider=self._model.provider,
                model=self._model.model,
            )
            if existing_hash == actual_hash:
                return PreparedEntityEmbedding(
                    owner_id=job.owner_id,
                    entity_id=entity_id,
                    expected_content_hash=actual_hash,
                    write=None,
                )
            aliases = ", ".join(entity.aliases) or "none"
            vector = await self._model.embed_multimodal(
                text=(
                    f"entity type: {entity.entity_type} | "
                    f"name: {entity.canonical_name} | aliases: {aliases} | "
                    f"description: {entity.description or 'none'}"
                ),
                images=(),
            )
            return PreparedEntityEmbedding(
                owner_id=job.owner_id,
                entity_id=entity_id,
                expected_content_hash=actual_hash,
                write=PreparedEmbeddingWrite(
                    subject_id=entity_id,
                    embedding_kind=ENTITY_IDENTITY_EMBEDDING_KIND,
                    provider=self._model.provider,
                    model=self._model.model,
                    embedded_content=content,
                    embedding=tuple(float(value) for value in vector),
                ),
            )
        except ValueError as error:
            raise MemoryJobValidationError(str(error)) from error
        except GeminiMultimodalEmbeddingError as error:
            raise MemoryJobTransientError(str(error)) from error

    async def commit(
        self,
        job: MemoryJob,
        prepared_result: PreparedEntityEmbedding,
        connection: asyncpg.Connection,
    ) -> Mapping[str, Any]:
        entity_id = _required_uuid(job.payload, "entity_id")
        if prepared_result.owner_id != job.owner_id:
            raise MemoryJobValidationError("prepared entity embedding owner changed")
        if prepared_result.entity_id != entity_id:
            raise MemoryJobValidationError("prepared entity embedding source changed")
        written = False
        if prepared_result.write is not None:
            written = await self._repository.store_entity_embedding(
                owner_id=job.owner_id,
                write=prepared_result.write,
                expected_content_hash=prepared_result.expected_content_hash,
                connection=connection,
            )
        return {
            "entity_id": str(entity_id),
            "embedding_written": written,
            "stale": prepared_result.stale,
        }


def _required_session_id(job: MemoryJob) -> UUID:
    if job.session_id is None:
        raise MemoryJobValidationError("event embedding job requires a session")
    return job.session_id


def _required_uuid(payload: Mapping[str, Any], key: str) -> UUID:
    value = payload.get(key)
    if not isinstance(value, str):
        raise MemoryJobValidationError(f"{key} must be a UUID string")
    try:
        return UUID(value)
    except ValueError as error:
        raise MemoryJobValidationError(f"{key} must be a UUID string") from error
