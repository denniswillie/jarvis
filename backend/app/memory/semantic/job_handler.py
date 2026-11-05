from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

import asyncpg

from app.memory.embeddings import build_memory_item_embedding_content
from app.memory.embeddings.job_handlers import EMBED_MEMORY_ITEM_JOB, memory_embedding_dedupe_key
from app.memory.jobs.handlers import MemoryJobTransientError, MemoryJobValidationError
from app.memory.jobs.models import MemoryJob
from app.memory.jobs.repository import MemoryJobRepository
from app.memory.models import content_hash
from app.memory.semantic.contracts import (
    EMBED_MEMORY_ENTITY_JOB,
    RECONCILE_SEMANTIC_MEMORY_JOB,
    SemanticReconciliationContext,
    SemanticReconciliationPlan,
    entity_embedding_dedupe_key,
    semantic_reconciliation_dedupe_key,
)
from app.memory.semantic.entity_resolver import SemanticEntityResolver
from app.memory.semantic.reconciler import SemanticMemoryReconciler
from app.memory.semantic.repository import SemanticMemoryRepository
from app.model_providers.gemini_multimodal_embeddings import GeminiMultimodalEmbeddingError
from app.model_providers.protocols import MultimodalEmbeddingModel


@dataclass(frozen=True, slots=True)
class PreparedSemanticReconciliation:
    context: SemanticReconciliationContext
    plan: SemanticReconciliationPlan
    entity_embeddings: Mapping[str, Sequence[float]]


class ReconcileSemanticMemoryJobHandler:
    job_type = RECONCILE_SEMANTIC_MEMORY_JOB

    def __init__(
        self,
        *,
        repository: SemanticMemoryRepository,
        reconciler: SemanticMemoryReconciler,
        job_repository: MemoryJobRepository,
        embedding_model: MultimodalEmbeddingModel,
    ) -> None:
        self._repository = repository
        self._reconciler = reconciler
        self._job_repository = job_repository
        self._embedding_model = embedding_model

    async def prepare(self, job: MemoryJob) -> PreparedSemanticReconciliation:
        event_id = _required_event_id(job)
        session_id = _required_session_id(job)
        try:
            context = await self._repository.load_context(
                owner_id=job.owner_id,
                session_id=session_id,
                event_id=event_id,
            )
            plan = self._reconciler.plan(context)
            entity_embeddings: dict[str, Sequence[float]] = {}
            if context.semantic_reconciled_at is None:
                for identity in plan.entities:
                    needs_embedding = await self._repository.requires_embedding_lookup(
                        owner_id=job.owner_id,
                        identity=identity,
                        provider=self._embedding_model.provider,
                        model=self._embedding_model.model,
                    )
                    if needs_embedding:
                        entity_embeddings[identity.identity_key] = (
                            await self._embedding_model.embed_multimodal(
                                text=(
                                    f"entity type: {identity.entity_type} | "
                                    f"name: {identity.name}"
                                ),
                                images=(),
                            )
                        )
            return PreparedSemanticReconciliation(
                context=context,
                plan=plan,
                entity_embeddings=entity_embeddings,
            )
        except ValueError as error:
            raise MemoryJobValidationError(str(error)) from error
        except GeminiMultimodalEmbeddingError as error:
            raise MemoryJobTransientError(str(error)) from error

    async def commit(
        self,
        job: MemoryJob,
        prepared_result: PreparedSemanticReconciliation,
        connection: asyncpg.Connection,
    ) -> Mapping[str, Any]:
        context = prepared_result.context
        if context.owner_id != job.owner_id:
            raise MemoryJobValidationError("prepared semantic owner changed")
        if context.session_id != _required_session_id(job):
            raise MemoryJobValidationError("prepared semantic session changed")
        if context.event_id != _required_event_id(job):
            raise MemoryJobValidationError("prepared semantic event changed")

        now = datetime.now(UTC)
        resolver = SemanticEntityResolver(
            self._repository,
            embedding_provider=self._embedding_model.provider,
            embedding_model=self._embedding_model.model,
        )
        outcome = await self._repository.apply_plan(
            context=context,
            plan=prepared_result.plan,
            resolver=resolver,
            entity_embeddings=prepared_result.entity_embeddings,
            reconciled_at=now,
            connection=connection,
        )
        for entity in outcome.created_entities:
            await self._job_repository.enqueue(
                owner_id=job.owner_id,
                session_id=context.session_id,
                job_type=EMBED_MEMORY_ENTITY_JOB,
                dedupe_key=entity_embedding_dedupe_key(entity),
                payload={
                    "entity_id": str(entity.id),
                    "content_hash": entity_embedding_dedupe_key(entity).rsplit(":", 1)[-1],
                },
                available_at=now,
                connection=connection,
            )
        facts_to_embed = {
            fact.id: fact for fact in (*outcome.created_facts, *outcome.updated_facts)
        }
        for fact in facts_to_embed.values():
            digest = content_hash(build_memory_item_embedding_content(fact))
            await self._job_repository.enqueue(
                owner_id=job.owner_id,
                # Reinforcement can come from another session without changing the fact.
                session_id=fact.session_id,
                job_type=EMBED_MEMORY_ITEM_JOB,
                dedupe_key=memory_embedding_dedupe_key(fact.id, content_digest=digest),
                payload={"memory_item_id": str(fact.id)},
                available_at=now,
                connection=connection,
            )
        return {
            "event_id": str(context.event_id),
            "episode_memory_id": str(context.episode_memory_id),
            "created_entities": len(outcome.created_entities),
            "created_facts": len(outcome.created_facts),
            "updated_facts": len(outcome.updated_facts),
            "created_relations": outcome.created_relation_count,
            "already_reconciled": outcome.already_reconciled,
        }


class SemanticMemoryRecoveryRule:
    name = "semantic_memory_reconciliation"

    def __init__(self, *, limit: int = 100) -> None:
        if limit < 1:
            raise ValueError("semantic recovery limit must be positive")
        self._limit = limit

    async def reconcile(
        self,
        *,
        repository: MemoryJobRepository,
        connection: asyncpg.Connection,
        now: datetime,
    ) -> int:
        rows = await connection.fetch(
            """
            select event.owner_id, event.session_id, event.id
            from experience_events event
            where event.promotion_decision = 'promoted'
              and event.semantic_reconciled_at is null
              and exists (
                  select 1
                  from memory_item_embeddings embedding
                  where embedding.memory_item_id = event.promoted_episode_memory_id
              )
              and not exists (
                  select 1
                  from memory_jobs job
                  where job.owner_id = event.owner_id
                    and job.job_type = 'reconcile_semantic_memory'
                    and job.dedupe_key =
                        'experience-event:' || event.id::text || ':reconcile-semantic'
                    and job.status in ('queued', 'claimed', 'failed', 'dead')
              )
            order by event.promotion_decided_at asc, event.id asc
            limit $1
            """,
            self._limit,
        )
        recovered = 0
        for row in rows:
            result = await repository.enqueue(
                owner_id=row["owner_id"],
                session_id=row["session_id"],
                job_type=RECONCILE_SEMANTIC_MEMORY_JOB,
                dedupe_key=semantic_reconciliation_dedupe_key(row["id"]),
                payload={"event_id": str(row["id"])},
                available_at=now,
                connection=connection,
            )
            recovered += int(result.created)
        return recovered


def _required_event_id(job: MemoryJob) -> UUID:
    value = job.payload.get("event_id")
    if not isinstance(value, str):
        raise MemoryJobValidationError("event_id must be a UUID string")
    try:
        return UUID(value)
    except ValueError as error:
        raise MemoryJobValidationError("event_id must be a UUID string") from error


def _required_session_id(job: MemoryJob) -> UUID:
    if job.session_id is None:
        raise MemoryJobValidationError("semantic reconciliation requires a session")
    return job.session_id
