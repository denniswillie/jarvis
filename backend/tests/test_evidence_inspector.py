import asyncio
import json
from datetime import UTC, datetime
from uuid import UUID

import pytest

from app.memory.retrieval.evidence_inspector import (
    HistoricalEvidenceInspector,
    RetainedEvidenceFrame,
)
from app.memory.retrieval.models import FusedRetrievalCandidate, RetrievalDocument
from app.model_providers.gemini_fast_visual import GeminiFastVisualModelError
from app.runtime.concurrency import ModelCallPriority, PriorityAsyncLimiter


NOW = datetime(2026, 8, 27, 12, 0, tzinfo=UTC)
JPEG = b"\xff\xd8historical-frame\xff\xd9"


def candidate() -> FusedRetrievalCandidate:
    return FusedRetrievalCandidate(
        document=RetrievalDocument(
            id=UUID(int=1),
            owner_id="owner",
            session_id=UUID(int=2),
            record_type="episode",
            occurred_at=NOW,
            valid_from=NOW,
            valid_to=None,
            title="Station poster",
            summary="A blue departure poster was near the platform.",
            content="The poster was visible while waiting for a train.",
            importance=0.7,
            confidence=0.9,
            record_status="active",
            is_explicit=False,
            evidence_count=2,
            keyframe_count=1,
            evidence_types=("frame", "experience_event"),
        ),
        score=0.8,
        sources=("lexical",),
        exact_match=True,
    )


def retained_frame() -> RetainedEvidenceFrame:
    return RetainedEvidenceFrame(
        candidate_id=UUID(int=1),
        visual_frame_id=UUID(int=10),
        captured_at=NOW,
        storage_bucket="frames",
        storage_path="private/frame.jpg",
        content_type="image/jpeg",
        existing_description="A blue poster.",
    )


class FakeRepository:
    def __init__(self, frames=(retained_frame(),)) -> None:
        self.frames = tuple(frames)

    async def list_retained_evidence_frames(self, **_kwargs):
        return self.frames


class FakeLoader:
    def __init__(self, content: bytes = JPEG) -> None:
        self.content = content
        self.requested = []
        self.loaded = asyncio.Event()

    async def download_object(self, *, bucket, path):
        self.requested.append((bucket, path))
        self.loaded.set()
        return self.content


class FakeModel:
    provider = "google_gemini"
    model = "gemini-test"

    def __init__(self, *, order=None) -> None:
        self.prompt = None
        self.images = None
        self.order = order

    async def generate_historical_evidence_json(self, *, prompt, images):
        self.prompt = prompt
        self.images = tuple(images)
        if self.order is not None:
            self.order.append("historical")
        return json.dumps(
            {
                "evidence_sufficient": True,
                "verified_details": "The departure code on the poster is A17.",
                "uncertainty": None,
                "confidence": 0.96,
                "supporting_frame_ids": [str(UUID(int=10))],
            }
        )


class FailingModel(FakeModel):
    async def generate_historical_evidence_json(self, *, prompt, images):
        raise GeminiFastVisualModelError("provider unavailable")


class InconclusiveModel(FakeModel):
    async def generate_historical_evidence_json(self, *, prompt, images):
        return json.dumps(
            {
                "evidence_sufficient": False,
                "verified_details": "The retained frame does not show the departure code.",
                "uncertainty": "The relevant part of the poster is outside the frame.",
                "confidence": 0.2,
                "supporting_frame_ids": [],
            }
        )


def make_inspector(*, repository=None, loader=None, model=None, limiter=None):
    return HistoricalEvidenceInspector(
        repository=repository or FakeRepository(),
        image_loader=loader or FakeLoader(),
        model=model or FakeModel(),
        model_limiter=limiter or PriorityAsyncLimiter(1),
    )


def test_reinspection_recovers_detail_omitted_from_structured_memory() -> None:
    model = FakeModel()
    loader = FakeLoader()
    result = asyncio.run(
        make_inspector(loader=loader, model=model).inspect(
            owner_id="owner",
            query="What was the departure code?",
            candidates=(candidate(),),
        )
    )

    assert result.status == "verified"
    assert result.verified_details == "The departure code on the poster is A17."
    assert result.supporting_frame_ids == (UUID(int=10),)
    assert result.pixel_evidence_available is True
    assert result.provider == "google_gemini"
    assert loader.requested == [("frames", "private/frame.jpg")]
    assert "What was the departure code?" in model.prompt
    assert "private/frame.jpg" not in model.prompt
    assert len(model.images) == 1


def test_no_keyframe_falls_back_to_lower_confidence_structured_evidence() -> None:
    result = asyncio.run(
        make_inspector(repository=FakeRepository(frames=())).inspect(
            owner_id="owner",
            query="What did I see?",
            candidates=(candidate(),),
        )
    )

    assert result.status == "structured_only"
    assert result.fallback_reason == "no_retained_keyframes"
    assert result.pixel_evidence_available is False
    assert result.supporting_frame_ids == ()
    assert result.confidence == 0.585
    assert "blue departure poster" in result.verified_details


def test_provider_failure_degrades_without_failing_retrieval() -> None:
    result = asyncio.run(
        make_inspector(model=FailingModel()).inspect(
            owner_id="owner",
            query="Read the poster.",
            candidates=(candidate(),),
        )
    )

    assert result.status == "structured_only"
    assert result.fallback_reason == "provider_inspection_failed"
    assert result.pixel_evidence_available is True
    assert result.provider is None


def test_inconclusive_pixels_are_not_reported_as_verified() -> None:
    result = asyncio.run(
        make_inspector(model=InconclusiveModel()).inspect(
            owner_id="owner",
            query="What was the departure code?",
            candidates=(candidate(),),
        )
    )

    assert result.status == "structured_only"
    assert result.fallback_reason == "pixel_evidence_inconclusive"
    assert result.pixel_evidence_available is True
    assert result.supporting_frame_ids == ()
    assert result.provider is None


def test_reinspection_result_rejects_impossible_states() -> None:
    from app.memory.retrieval.models import EvidenceReinspectionResult

    with pytest.raises(ValueError, match="available pixel evidence"):
        EvidenceReinspectionResult(
            status="verified",
            verified_details="Verified detail",
            uncertainty=None,
            confidence=0.9,
            pixel_evidence_available=False,
            supporting_frame_ids=(UUID(int=10),),
            provider="google_gemini",
            model="gemini-test",
            fallback_reason=None,
        )
    with pytest.raises(ValueError, match="provider and model"):
        EvidenceReinspectionResult(
            status="verified",
            verified_details="Verified detail",
            uncertainty=None,
            confidence=0.9,
            pixel_evidence_available=True,
            supporting_frame_ids=(UUID(int=10),),
            provider=None,
            model=None,
            fallback_reason=None,
        )
    with pytest.raises(ValueError, match="fallback reason"):
        EvidenceReinspectionResult(
            status="structured_only",
            verified_details="Structured detail",
            uncertainty="Pixels unavailable",
            confidence=0.4,
            pixel_evidence_available=False,
            supporting_frame_ids=(),
            provider=None,
            model=None,
            fallback_reason=None,
        )


def test_high_priority_current_view_enters_before_waiting_reinspection() -> None:
    async def scenario() -> list[str]:
        limiter = PriorityAsyncLimiter(1)
        loader = FakeLoader()
        order: list[str] = []
        inspector = make_inspector(
            loader=loader,
            model=FakeModel(order=order),
            limiter=limiter,
        )
        blocker_entered = asyncio.Event()
        release_blocker = asyncio.Event()

        async def blocker() -> None:
            async with limiter.limit(ModelCallPriority.NORMAL):
                blocker_entered.set()
                await release_blocker.wait()

        async def current_view() -> None:
            async with limiter.limit(ModelCallPriority.HIGH):
                order.append("current-view")

        blocker_task = asyncio.create_task(blocker())
        await blocker_entered.wait()
        inspection_task = asyncio.create_task(
            inspector.inspect(
                owner_id="owner",
                query="Read the poster.",
                candidates=(candidate(),),
            )
        )
        await loader.loaded.wait()
        await asyncio.sleep(0)
        current_view_task = asyncio.create_task(current_view())
        await asyncio.sleep(0)
        release_blocker.set()
        await asyncio.gather(blocker_task, current_view_task, inspection_task)
        return order

    assert asyncio.run(scenario()) == ["current-view", "historical"]
