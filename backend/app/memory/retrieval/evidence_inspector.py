from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from datetime import datetime
from typing import Protocol
from uuid import UUID

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from app.memory.retrieval.models import (
    EvidenceReinspectionResult,
    FusedRetrievalCandidate,
)
from app.memory.retrieval.prompts import (
    HistoricalEvidencePromptCandidate,
    HistoricalEvidencePromptFrame,
    build_historical_evidence_prompt,
)
from app.model_providers.gemini_fast_visual import GeminiFastVisualModelError
from app.model_providers.protocols import VisualModelImage
from app.runtime.concurrency import ModelCallPriority, PriorityAsyncLimiter


LOGGER = logging.getLogger(__name__)
STRUCTURED_CONFIDENCE_FACTOR = 0.65
MAX_STRUCTURED_DETAILS_CHARS = 4_000


from app.memory.retrieval.evidence_repository import (
    HistoricalEvidenceRepository,
    RetainedEvidenceFrame,
)


class HistoricalEvidenceFrameReader(Protocol):
    async def list_retained_evidence_frames(
        self,
        *,
        owner_id: str,
        candidates: Sequence[FusedRetrievalCandidate],
        limit: int,
    ) -> tuple[RetainedEvidenceFrame, ...]: ...


class HistoricalEvidenceImageLoader(Protocol):
    async def download_object(self, *, bucket: str, path: str) -> bytes: ...


class HistoricalEvidenceModel(Protocol):
    @property
    def provider(self) -> str: ...

    @property
    def model(self) -> str: ...

    async def generate_historical_evidence_json(
        self,
        *,
        prompt: str,
        images: Sequence[VisualModelImage],
    ) -> str: ...


class _ProviderInspectionResult(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    evidence_sufficient: bool
    verified_details: str = Field(min_length=1, max_length=4_000)
    uncertainty: str | None = Field(default=None, max_length=1_000)
    confidence: float = Field(ge=0, le=1)
    supporting_frame_ids: list[UUID] = Field(default_factory=list, max_length=6)

    @field_validator("supporting_frame_ids")
    @classmethod
    def require_unique_frame_ids(cls, value: list[UUID]) -> list[UUID]:
        if len(value) != len(set(value)):
            raise ValueError("supporting_frame_ids must be unique")
        return value

    @model_validator(mode="after")
    def require_consistent_evidence_state(self) -> _ProviderInspectionResult:
        if self.evidence_sufficient and not self.supporting_frame_ids:
            raise ValueError(
                "supporting_frame_ids are required when evidence is sufficient"
            )
        if not self.evidence_sufficient and self.supporting_frame_ids:
            raise ValueError(
                "supporting_frame_ids must be empty when evidence is insufficient"
            )
        return self


class HistoricalEvidenceInspector:
    def __init__(
        self,
        *,
        repository: HistoricalEvidenceFrameReader,
        image_loader: HistoricalEvidenceImageLoader | None,
        model: HistoricalEvidenceModel | None,
        model_limiter: PriorityAsyncLimiter,
        max_frames: int = 6,
    ) -> None:
        if max_frames < 1 or max_frames > 6:
            raise ValueError("max_frames must be between 1 and 6")
        self._repository = repository
        self._image_loader = image_loader
        self._model = model
        self._model_limiter = model_limiter
        self._max_frames = max_frames

    async def inspect(
        self,
        *,
        owner_id: str,
        query: str,
        candidates: Sequence[FusedRetrievalCandidate],
    ) -> EvidenceReinspectionResult:
        normalized_owner_id = owner_id.strip()
        if not normalized_owner_id:
            raise ValueError("owner_id cannot be blank")
        ordered_candidates = tuple(candidates)
        if not ordered_candidates:
            return build_structured_evidence_fallback(
                candidates=(),
                reason="no_retrieval_candidates",
            )
        if self._image_loader is None or self._model is None:
            return build_structured_evidence_fallback(
                candidates=ordered_candidates,
                reason="pixel_reinspection_not_configured",
            )

        try:
            frame_refs = await self._repository.list_retained_evidence_frames(
                owner_id=normalized_owner_id,
                candidates=ordered_candidates,
                limit=self._max_frames,
            )
        except Exception:
            LOGGER.exception("historical evidence lookup failed")
            return build_structured_evidence_fallback(
                candidates=ordered_candidates,
                reason="evidence_lookup_failed",
            )
        if not frame_refs:
            return build_structured_evidence_fallback(
                candidates=ordered_candidates,
                reason="no_retained_keyframes",
            )

        loaded = await self._load_images(frame_refs)
        if not loaded:
            return build_structured_evidence_fallback(
                candidates=ordered_candidates,
                reason="retained_keyframes_unavailable",
            )

        prompt_candidates = tuple(
            HistoricalEvidencePromptCandidate(
                id=candidate.document.id,
                record_type=candidate.document.record_type,
                occurred_at=candidate.document.occurred_at,
                title=candidate.document.title,
                summary=candidate.document.summary or candidate.document.content[:500],
            )
            for candidate in ordered_candidates
        )
        prompt_frames = tuple(
            HistoricalEvidencePromptFrame(
                id=frame.visual_frame_id,
                candidate_id=frame.candidate_id,
                captured_at=frame.captured_at,
                existing_description=frame.existing_description,
            )
            for frame, _ in loaded
        )
        images = tuple(
            VisualModelImage(
                label=(
                    f"Historical frame {frame.visual_frame_id} captured at "
                    f"{frame.captured_at.isoformat()} for candidate {frame.candidate_id}."
                ),
                image_bytes=image_bytes,
                content_type=frame.content_type,
            )
            for frame, image_bytes in loaded
        )
        prompt = build_historical_evidence_prompt(
            query=query,
            candidates=prompt_candidates,
            frames=prompt_frames,
        )
        try:
            async with self._model_limiter.limit(ModelCallPriority.NORMAL):
                raw = await self._model.generate_historical_evidence_json(
                    prompt=prompt,
                    images=images,
                )
            parsed = _parse_provider_result(
                raw,
                allowed_frame_ids=frozenset(frame.id for frame in prompt_frames),
            )
        except (GeminiFastVisualModelError, ValidationError, ValueError, json.JSONDecodeError):
            LOGGER.exception("historical evidence model inspection failed")
            return build_structured_evidence_fallback(
                candidates=ordered_candidates,
                reason="provider_inspection_failed",
                pixel_evidence_available=True,
            )

        if not parsed.evidence_sufficient:
            return EvidenceReinspectionResult(
                status="structured_only",
                verified_details=parsed.verified_details,
                uncertainty=parsed.uncertainty,
                confidence=parsed.confidence,
                pixel_evidence_available=True,
                supporting_frame_ids=(),
                provider=None,
                model=None,
                fallback_reason="pixel_evidence_inconclusive",
            )

        return EvidenceReinspectionResult(
            status="verified",
            verified_details=parsed.verified_details,
            uncertainty=parsed.uncertainty,
            confidence=parsed.confidence,
            pixel_evidence_available=True,
            supporting_frame_ids=tuple(parsed.supporting_frame_ids),
            provider=self._model.provider,
            model=self._model.model,
            fallback_reason=None,
        )

    async def _load_images(
        self,
        frames: Sequence[RetainedEvidenceFrame],
    ) -> tuple[tuple[RetainedEvidenceFrame, bytes], ...]:
        assert self._image_loader is not None
        loaded: list[tuple[RetainedEvidenceFrame, bytes]] = []
        for frame in frames[: self._max_frames]:
            if frame.content_type != "image/jpeg":
                continue
            try:
                image_bytes = await self._image_loader.download_object(
                    bucket=frame.storage_bucket,
                    path=frame.storage_path,
                )
            except Exception:
                LOGGER.warning(
                    "historical evidence frame download failed frame_id=%s",
                    frame.visual_frame_id,
                    exc_info=True,
                )
                continue
            if not _is_jpeg(image_bytes):
                LOGGER.warning(
                    "historical evidence frame is not a JPEG frame_id=%s",
                    frame.visual_frame_id,
                )
                continue
            loaded.append((frame, image_bytes))
        return tuple(loaded)


def build_structured_evidence_fallback(
    *,
    candidates: Sequence[FusedRetrievalCandidate],
    reason: str,
    pixel_evidence_available: bool = False,
) -> EvidenceReinspectionResult:
    details = _structured_details(candidates)
    base_confidence = max(
        (candidate.document.confidence for candidate in candidates),
        default=0.0,
    )
    return EvidenceReinspectionResult(
        status="structured_only",
        verified_details=details,
        uncertainty=(
            "Retained pixels could not be verified; this answer relies on structured "
            "memory evidence."
        ),
        confidence=round(base_confidence * STRUCTURED_CONFIDENCE_FACTOR, 4),
        pixel_evidence_available=pixel_evidence_available,
        supporting_frame_ids=(),
        provider=None,
        model=None,
        fallback_reason=reason,
    )


def _structured_details(candidates: Sequence[FusedRetrievalCandidate]) -> str:
    parts: list[str] = []
    for candidate in candidates[:6]:
        document = candidate.document
        title = document.title.strip()
        summary = (document.summary or document.content[:500]).strip()
        text = f"{title}: {summary}" if title and summary else title or summary
        if text:
            parts.append(text)
    combined = "\n".join(parts).strip()
    return (combined or "Structured memory evidence was retrieved.")[
        :MAX_STRUCTURED_DETAILS_CHARS
    ]


def _parse_provider_result(
    raw: str,
    *,
    allowed_frame_ids: frozenset[UUID],
) -> _ProviderInspectionResult:
    candidate = raw.strip()
    if candidate.startswith("```"):
        candidate = candidate.removeprefix("```json").removeprefix("```")
        candidate = candidate.removesuffix("```").strip()
    decoded = json.loads(candidate)
    parsed = _ProviderInspectionResult.model_validate(decoded)
    unsupported = set(parsed.supporting_frame_ids) - allowed_frame_ids
    if unsupported:
        raise ValueError("provider referenced a frame outside the evidence bundle")
    return parsed


def _is_jpeg(value: bytes) -> bool:
    return len(value) >= 4 and value.startswith(b"\xff\xd8") and value.endswith(b"\xff\xd9")
