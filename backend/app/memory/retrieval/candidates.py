from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol
from uuid import UUID

from app.memory.retrieval.fusion import group_rankings
from app.memory.retrieval.models import RetrievalDocument, RetrievalSourceHit
from app.memory.retrieval.query import NormalizedMemoryRetrievalQuery, build_embedding_query
from app.model_providers.gemini_multimodal_embeddings import (
    GeminiMultimodalEmbeddingError,
)
from app.model_providers.protocols import MultimodalEmbeddingModel


class RetrievalMemoryRepository(Protocol):
    async def list_time_retrieval_hits(
        self, *, owner_id: str, query: NormalizedMemoryRetrievalQuery, limit: int
    ) -> list[RetrievalSourceHit]: ...

    async def list_lexical_retrieval_hits(
        self, *, owner_id: str, query: NormalizedMemoryRetrievalQuery, limit: int
    ) -> list[RetrievalSourceHit]: ...

    async def list_embedding_retrieval_hits(
        self,
        *,
        owner_id: str,
        query: NormalizedMemoryRetrievalQuery,
        embedding: Sequence[float],
        provider: str,
        model: str,
        limit: int,
    ) -> list[RetrievalSourceHit]: ...

    async def list_exact_entity_retrieval_hits(
        self, *, owner_id: str, query: NormalizedMemoryRetrievalQuery, limit: int
    ) -> list[RetrievalSourceHit]: ...

    async def list_entity_embedding_retrieval_hits(
        self,
        *,
        owner_id: str,
        query: NormalizedMemoryRetrievalQuery,
        embedding: Sequence[float],
        provider: str,
        model: str,
        limit: int,
    ) -> list[RetrievalSourceHit]: ...

    async def list_adjacent_retrieval_hits(
        self,
        *,
        owner_id: str,
        query: NormalizedMemoryRetrievalQuery,
        anchor_ids: Sequence[UUID],
        limit: int,
    ) -> list[RetrievalSourceHit]: ...

    async def get_retrieval_documents_by_ids(
        self, *, owner_id: str, document_ids: Sequence[UUID]
    ) -> list[RetrievalDocument]: ...


class RetrievalPerceptionRepository(Protocol):
    async def rank_retrieval_visible_text_events(
        self,
        *,
        owner_id: str,
        session_id: UUID | None,
        start_time: datetime,
        end_time: datetime,
        query: str,
        limit: int,
    ) -> list[UUID]: ...


@dataclass(frozen=True, slots=True)
class CandidateCollection:
    rankings: dict[str, list[RetrievalSourceHit]]
    provider_spaces: tuple[str, ...]


class RetrievalCandidateCollector:
    def __init__(
        self,
        *,
        memory_repository: RetrievalMemoryRepository,
        perception_repository: RetrievalPerceptionRepository,
        embedding_model: MultimodalEmbeddingModel | None,
    ) -> None:
        self._memory_repository = memory_repository
        self._perception_repository = perception_repository
        self._embedding_model = embedding_model

    async def collect(
        self,
        *,
        owner_id: str,
        query: NormalizedMemoryRetrievalQuery,
        limit: int,
    ) -> CandidateCollection:
        hits = await self._memory_repository.list_time_retrieval_hits(
            owner_id=owner_id,
            query=query,
            limit=limit,
        )
        provider_spaces: set[str] = set()

        if query.query:
            hits.extend(
                await self._memory_repository.list_lexical_retrieval_hits(
                    owner_id=owner_id,
                    query=query,
                    limit=limit,
                )
            )
            hits.extend(
                await self._memory_repository.list_exact_entity_retrieval_hits(
                    owner_id=owner_id,
                    query=query,
                    limit=limit,
                )
            )
            hits.extend(
                await self._visible_text_hits(
                    owner_id=owner_id,
                    query=query,
                    limit=limit,
                )
            )

            embedding = await self._query_embedding(query.query)
            if embedding is not None and self._embedding_model is not None:
                provider = self._embedding_model.provider
                model = self._embedding_model.model
                provider_spaces.add(f"{provider}:{model}:query")
                embedding_hits = await self._memory_repository.list_embedding_retrieval_hits(
                    owner_id=owner_id,
                    query=query,
                    embedding=embedding,
                    provider=provider,
                    model=model,
                    limit=limit,
                )
                entity_hits = (
                    await self._memory_repository.list_entity_embedding_retrieval_hits(
                        owner_id=owner_id,
                        query=query,
                        embedding=embedding,
                        provider=provider,
                        model=model,
                        limit=limit,
                    )
                )
                hits.extend(embedding_hits)
                hits.extend(entity_hits)
                provider_spaces.update(
                    f"{provider}:{model}:{hit.source}" for hit in embedding_hits + entity_hits
                )

        rankings = group_rankings(hits)
        anchor_ids = _strong_anchor_ids(rankings, limit=8)
        adjacency = (
            await self._memory_repository.list_adjacent_retrieval_hits(
                owner_id=owner_id,
                query=query,
                anchor_ids=anchor_ids,
                limit=limit,
            )
            if anchor_ids
            else []
        )
        if adjacency:
            rankings["temporal_adjacency"] = adjacency
        return CandidateCollection(
            rankings=rankings,
            provider_spaces=tuple(sorted(provider_spaces)),
        )

    async def _query_embedding(self, query: str) -> list[float] | None:
        if self._embedding_model is None:
            return None
        try:
            return await self._embedding_model.embed_multimodal(
                text=build_embedding_query(query),
                images=(),
            )
        except (GeminiMultimodalEmbeddingError, ValueError):
            return None

    async def _visible_text_hits(
        self,
        *,
        owner_id: str,
        query: NormalizedMemoryRetrievalQuery,
        limit: int,
    ) -> list[RetrievalSourceHit]:
        event_ids = await self._perception_repository.rank_retrieval_visible_text_events(
            owner_id=owner_id,
            session_id=query.session_id,
            start_time=query.start_time,
            end_time=query.end_time,
            query=query.query,
            limit=limit,
        )
        if not event_ids:
            return []
        documents = await self._memory_repository.get_retrieval_documents_by_ids(
            owner_id=owner_id,
            document_ids=event_ids,
        )
        by_id = {document.id: document for document in documents}
        return [
            RetrievalSourceHit(
                document=by_id[event_id],
                source="visible_text_exact",
                raw_score=1.0,
                exact_match=True,
            )
            for event_id in event_ids
            if event_id in by_id
        ]


def _strong_anchor_ids(
    rankings: dict[str, list[RetrievalSourceHit]],
    *,
    limit: int,
) -> tuple[UUID, ...]:
    if limit < 1:
        raise ValueError("anchor limit must be positive")
    anchor_ids: list[UUID] = []
    seen: set[tuple[str, UUID]] = set()
    for source, hits in rankings.items():
        if source == "time":
            continue
        for hit in hits:
            key = (hit.document.record_type, hit.document.id)
            if key in seen:
                continue
            seen.add(key)
            anchor_ids.append(hit.document.id)
            if len(anchor_ids) == limit:
                return tuple(anchor_ids)
    return tuple(anchor_ids)
