import asyncio
from datetime import UTC, datetime
from uuid import UUID

from app.memory.retrieval.candidates import RetrievalCandidateCollector
from app.memory.retrieval.models import RetrievalDocument, RetrievalSourceHit
from app.memory.retrieval.query import MemoryRetrievalQuery, normalize_memory_query
from app.model_providers.gemini_multimodal_embeddings import (
    GeminiMultimodalEmbeddingError,
)


NOW = datetime(2026, 8, 27, 12, 0, tzinfo=UTC)


def document(value: int, *, minute: int = 0) -> RetrievalDocument:
    return RetrievalDocument(
        id=UUID(int=value),
        owner_id="owner",
        session_id=UUID(int=99),
        record_type="experience_event",
        occurred_at=datetime(2026, 8, 27, 11, minute, tzinfo=UTC),
        valid_from=None,
        valid_to=None,
        title=f"Event {value}",
        summary="summary",
        content="content",
        importance=0.5,
        confidence=0.8,
        record_status="historical",
        is_explicit=False,
        evidence_count=1,
        keyframe_count=1,
        evidence_types=("observation",),
    )


class FakeMemoryRepository:
    def __init__(self) -> None:
        self.adjacency_anchor_ids = ()
        self.time = [
            RetrievalSourceHit(document(1, minute=1), "time", 0),
            RetrievalSourceHit(document(2, minute=2), "time", 0),
            RetrievalSourceHit(document(3, minute=3), "time", 0),
        ]
        self.documents = {hit.document.id: hit.document for hit in self.time}

    async def list_time_retrieval_hits(self, **_kwargs):
        return list(self.time)

    async def list_lexical_retrieval_hits(self, **_kwargs):
        return [RetrievalSourceHit(self.documents[UUID(int=2)], "lexical", 1, True)]

    async def list_exact_entity_retrieval_hits(self, **_kwargs):
        return []

    async def list_embedding_retrieval_hits(self, **_kwargs):
        return [
            RetrievalSourceHit(
                self.documents[UUID(int=2)], "embedding:event_multimodal", 0.9
            )
        ]

    async def list_entity_embedding_retrieval_hits(self, **_kwargs):
        return []

    async def list_adjacent_retrieval_hits(self, *, anchor_ids, **_kwargs):
        self.adjacency_anchor_ids = tuple(anchor_ids)
        return [
            RetrievalSourceHit(
                self.documents[UUID(int=1)],
                "temporal_adjacency",
                0.5,
            )
        ]

    async def get_retrieval_documents_by_ids(self, *, document_ids, **_kwargs):
        return [self.documents[item] for item in document_ids if item in self.documents]


class FakePerceptionRepository:
    async def rank_retrieval_visible_text_events(self, **_kwargs):
        return [UUID(int=3)]


class FakeEmbeddingModel:
    provider = "gemini"
    model = "gemini-embedding-2"

    async def embed_multimodal(self, *, text, images):
        assert text.startswith("task: retrieve")
        assert images == ()
        return [0.0] * 1536

    async def close(self):
        return None


class FailingEmbeddingModel(FakeEmbeddingModel):
    async def embed_multimodal(self, *, text, images):
        raise GeminiMultimodalEmbeddingError("provider unavailable")


def normalized_query():
    return normalize_memory_query(
        MemoryRetrievalQuery(query="poster"),
        now=NOW,
        default_seconds=600,
        max_days=3650,
    )


def test_collector_combines_text_visual_embedding_and_adjacency_sources() -> None:
    repository = FakeMemoryRepository()
    collector = RetrievalCandidateCollector(
        memory_repository=repository,
        perception_repository=FakePerceptionRepository(),
        embedding_model=FakeEmbeddingModel(),
    )

    result = asyncio.run(
        collector.collect(owner_id="owner", query=normalized_query(), limit=40)
    )

    assert "time" in result.rankings
    assert "lexical" in result.rankings
    assert "visible_text_exact" in result.rankings
    assert "embedding:event_multimodal" in result.rankings
    assert "temporal_adjacency" in result.rankings
    assert repository.adjacency_anchor_ids == (UUID(int=2), UUID(int=3))
    assert result.provider_spaces == (
        "gemini:gemini-embedding-2:embedding:event_multimodal",
        "gemini:gemini-embedding-2:query",
    )


def test_collector_falls_back_when_embedding_provider_is_unavailable() -> None:
    collector = RetrievalCandidateCollector(
        memory_repository=FakeMemoryRepository(),
        perception_repository=FakePerceptionRepository(),
        embedding_model=FailingEmbeddingModel(),
    )

    result = asyncio.run(
        collector.collect(owner_id="owner", query=normalized_query(), limit=40)
    )

    assert "lexical" in result.rankings
    assert "visible_text_exact" in result.rankings
    assert not any(source.startswith("embedding:") for source in result.rankings)
    assert result.provider_spaces == ()


class EmptyEmbeddingMemoryRepository(FakeMemoryRepository):
    async def list_embedding_retrieval_hits(self, **_kwargs):
        return []


def test_collector_audits_embedding_space_even_without_database_matches() -> None:
    collector = RetrievalCandidateCollector(
        memory_repository=EmptyEmbeddingMemoryRepository(),
        perception_repository=FakePerceptionRepository(),
        embedding_model=FakeEmbeddingModel(),
    )

    result = asyncio.run(
        collector.collect(owner_id="owner", query=normalized_query(), limit=40)
    )

    assert result.provider_spaces == ("gemini:gemini-embedding-2:query",)
