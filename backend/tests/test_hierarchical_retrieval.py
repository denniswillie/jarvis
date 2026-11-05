import asyncio
from datetime import UTC, datetime
from uuid import UUID
from types import SimpleNamespace
from unittest.mock import AsyncMock

from app.memory.retrieval.candidates import CandidateCollection
from app.memory.retrieval.models import RetrievalDocument, RetrievalSourceHit
from app.memory.retrieval.query import MemoryRetrievalQuery
from app.memory.retrieval.service import HierarchicalMemoryRetrievalService
from app.images import ImageUnavailableError
from app.memory.retrieval.models import EvidenceReinspectionResult


NOW = datetime(2026, 8, 27, 12, 0, tzinfo=UTC)


def document() -> RetrievalDocument:
    return RetrievalDocument(
        id=UUID(int=1),
        owner_id="owner",
        session_id=UUID(int=2),
        record_type="episode",
        occurred_at=NOW,
        valid_from=NOW,
        valid_to=None,
        title="Saw the station poster",
        summary="A blue departure poster was beside the platform.",
        content="Detailed episode",
        importance=0.7,
        confidence=0.9,
        record_status="active",
        is_explicit=True,
        evidence_count=2,
        keyframe_count=1,
        evidence_types=("experience_event", "frame"),
    )


class FakeCollector:
    async def collect(self, **_kwargs):
        item = document()
        return CandidateCollection(
            rankings={
                "lexical": [
                    RetrievalSourceHit(item, "lexical", 1.0, exact_match=True)
                ],
                "embedding:event_multimodal": [
                    RetrievalSourceHit(item, "embedding:event_multimodal", 0.9)
                ],
            },
            provider_spaces=("gemini:gemini-embedding-2:event_multimodal",),
        )


class FakeMemoryRepository:
    def __init__(self) -> None:
        self.audit = None

    async def record_retrieval_run(self, **kwargs):
        self.audit = kwargs
        return UUID(int=50)


class FakePerceptionRepository:
    pass


def test_service_returns_evidence_aware_results_and_audits() -> None:
    repository = FakeMemoryRepository()
    service = HierarchicalMemoryRetrievalService(
        memory_repository=repository,  # type: ignore[arg-type]
        perception_repository=FakePerceptionRepository(),  # type: ignore[arg-type]
        candidate_collector=FakeCollector(),  # type: ignore[arg-type]
    )

    response = asyncio.run(
        service.search(
            owner_id="owner",
            request=MemoryRetrievalQuery(
                query="station poster",
                seconds_before_now=600,
                session_id=UUID(int=2),
            ),
            now=NOW,
        )
    )

    assert response.retrieval_run_id == UUID(int=50)
    assert response.candidate_count == 1
    assert response.results[0].evidence_available is True
    assert response.results[0].keyframe_count == 1
    assert response.results[0].evidence_metadata["evidence_count"] == 2
    assert "matched memory text" in response.results[0].match_explanation
    assert repository.audit["candidate_ids"] == [UUID(int=1)]
    assert repository.audit["result_ids"] == [UUID(int=1)]
    assert repository.audit["reinspection_performed"] is False


def test_service_can_hide_evidence_metadata_without_hiding_availability() -> None:
    service = HierarchicalMemoryRetrievalService(
        memory_repository=FakeMemoryRepository(),  # type: ignore[arg-type]
        perception_repository=FakePerceptionRepository(),  # type: ignore[arg-type]
        candidate_collector=FakeCollector(),  # type: ignore[arg-type]
    )

    response = asyncio.run(
        service.search(
            owner_id="owner",
            request=MemoryRetrievalQuery(
                query="poster",
                include_evidence_metadata=False,
            ),
            now=NOW,
        )
    )

    assert response.results[0].evidence_available is True
    assert response.results[0].evidence_metadata == {}


def test_image_preview_is_optional_batched_and_preserves_ranking_and_audit() -> None:
    async def scenario():
        repository = FakeMemoryRepository()
        previews = SimpleNamespace(preview_limit=2, image_previews=AsyncMock(return_value={
            ("episode", UUID(int=1)): [{
                "frame_id": str(UUID(int=100)), "captured_at": NOW.isoformat(), "availability": "available",
            }],
        }))
        service = HierarchicalMemoryRetrievalService(
            memory_repository=repository, perception_repository=FakePerceptionRepository(),
            candidate_collector=FakeCollector(), image_service=previews,
        )
        result = await service.search(owner_id="owner", request=MemoryRetrievalQuery(query="poster"), now=NOW)
        previews.image_previews.assert_awaited_once_with(owner_id="owner", records=(("episode", UUID(int=1)),))
        metadata = result.results[0].evidence_metadata
        assert metadata["image_refs"][0]["frame_id"] == str(UUID(int=100))
        assert metadata["image_refs_status"] == "metadata_only"
        assert metadata["image_refs_bounded"] is True
        assert metadata["image_refs_unique_frame_limit"] == 2
        assert repository.audit["result_ids"] == [UUID(int=1)]
        assert repository.audit["reinspection_performed"] is False
        assert result.provider_spaces == ("gemini:gemini-embedding-2:event_multimodal",)
        previews.image_previews.reset_mock()
        hidden = await service.search(
            owner_id="owner", request=MemoryRetrievalQuery(query="poster", include_evidence_metadata=False), now=NOW,
        )
        assert hidden.results[0].evidence_metadata == {}
        previews.image_previews.assert_not_awaited()

    asyncio.run(scenario())


def test_unavailable_optional_images_do_not_fail_text_search() -> None:
    previews = SimpleNamespace(preview_limit=6, image_previews=AsyncMock(
        side_effect=ImageUnavailableError("image_lookup_failed", "Unavailable"),
    ))
    repository = FakeMemoryRepository()
    service = HierarchicalMemoryRetrievalService(
        memory_repository=repository, perception_repository=FakePerceptionRepository(),
        candidate_collector=FakeCollector(), image_service=previews,
    )
    result = asyncio.run(service.search(owner_id="owner", request=MemoryRetrievalQuery(query="poster"), now=NOW))
    assert result.results[0].summary == document().summary
    assert result.results[0].evidence_metadata["image_refs_status"] == "unavailable"
    assert result.results[0].evidence_metadata["image_refs"] == []
    assert repository.audit["result_ids"] == [UUID(int=1)]


def test_explicit_service_api_retains_legacy_historical_reinspection() -> None:
    inspection = EvidenceReinspectionResult(
        status="verified", verified_details="A retained detail.", uncertainty=None,
        confidence=0.9, pixel_evidence_available=True, supporting_frame_ids=(UUID(int=100),),
        provider="legacy_provider", model="legacy_model", fallback_reason=None,
    )
    inspector = SimpleNamespace(inspect=AsyncMock(return_value=inspection))
    repository = FakeMemoryRepository()
    service = HierarchicalMemoryRetrievalService(
        memory_repository=repository, perception_repository=FakePerceptionRepository(),
        candidate_collector=FakeCollector(), evidence_inspector=inspector,
    )
    result = asyncio.run(service.search(
        owner_id="owner", request=MemoryRetrievalQuery(query="poster"),
        now=NOW, reinspect_evidence=True,
    ))
    inspector.inspect.assert_awaited_once()
    assert result.evidence_reinspection is inspection
    assert repository.audit["reinspection_performed"] is True
    assert "legacy_provider:legacy_model:historical_evidence" in result.provider_spaces
