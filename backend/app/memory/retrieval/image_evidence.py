from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from math import isfinite
from typing import Any, cast
from uuid import UUID

from app.errors import JarvisHTTPError
from app.images import ImageReference, ImageUnavailableError, ResolvedImage
from app.memory.retrieval.evidence_repository import (
    MAX_HISTORICAL_EVIDENCE_FRAMES,
    HistoricalEvidenceRepository,
    HistoricalImageEvidence,
)
from app.memory.retrieval.models import RETRIEVAL_RECORD_TYPES, RetrievalRecordType
from app.visual_memory.storage import VisualFrameStorageClient
from app.visual_memory.validation import VisualFrameValidationError, jpeg_dimensions


class HistoricalImageService:
    """Read existing, authorized historical pixels without interpreting or retaining them."""

    def __init__(
        self, *, repository: HistoricalEvidenceRepository,
        image_loader: VisualFrameStorageClient | None,
        max_image_bytes: int, max_width: int, max_height: int,
        request_timeout_seconds: float,
        max_frames: int = MAX_HISTORICAL_EVIDENCE_FRAMES,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        if min(max_image_bytes, max_width, max_height) < 1:
            raise ValueError("Historical image byte and dimension limits must be positive.")
        if not isfinite(request_timeout_seconds) or request_timeout_seconds <= 0:
            raise ValueError("Historical image timeout must be finite and positive.")
        if not 1 <= max_frames <= MAX_HISTORICAL_EVIDENCE_FRAMES:
            raise ValueError("Historical image preview limit must be between 1 and 6.")
        self._repository = repository
        self._image_loader = image_loader
        self._max_image_bytes = max_image_bytes
        self._max_width = max_width
        self._max_height = max_height
        self._timeout = request_timeout_seconds
        self._max_frames = max_frames
        self._now = now

    @property
    def preview_limit(self) -> int:
        return self._max_frames

    async def select_image(
        self, *, owner_id: str, record_id: UUID,
        record_type: RetrievalRecordType, frame_id: UUID | None = None,
    ) -> ImageReference:
        if self._image_loader is None:
            raise ImageUnavailableError("frame_unavailable", "Historical image storage is unavailable.")
        evidence = await self._read_evidence(
            owner_id=owner_id, record_id=record_id, record_type=record_type, frame_id=frame_id,
        )
        return evidence.reference

    async def resolve_image(self, *, owner_id: str, reference: ImageReference) -> ResolvedImage:
        if (
            reference.source != "historical" or reference.owner_id != owner_id.strip()
            or reference.record_id is None or reference.record_type not in RETRIEVAL_RECORD_TYPES
            or self._image_loader is None
        ):
            raise self._unavailable()
        try:
            async with asyncio.timeout(self._timeout):
                evidence = await self._read_evidence(
                    owner_id=owner_id, record_id=reference.record_id,
                    record_type=cast(RetrievalRecordType, reference.record_type),
                    frame_id=reference.frame_id,
                )
                if evidence.reference != reference:
                    raise self._unavailable()
                try:
                    image_bytes = await self._image_loader.download_object(
                        bucket=evidence.storage_bucket, path=evidence.storage_path,
                        max_bytes=self._max_image_bytes,
                    )
                except JarvisHTTPError as error:
                    if error.details.get("status_code") == 404:
                        raise self._unavailable() from error
                    if error.code == "visual_frame_storage_download_too_large":
                        raise ImageUnavailableError("invalid_image", "The historical image exceeds the byte limit.") from error
                    raise ImageUnavailableError("image_storage_failed", "Historical image storage is temporarily unavailable.") from error
                except TimeoutError:
                    raise
                except Exception as error:
                    raise ImageUnavailableError("image_storage_failed", "Historical image storage is temporarily unavailable.") from error
                self._validate_pixels(reference, image_bytes)
                await self._revalidate(evidence)
        except TimeoutError as error:
            raise ImageUnavailableError("image_resolution_timeout", "The historical image could not be loaded in time.") from error

        async def revalidate() -> None:
            # The common writer invokes this after its queue wait, before sending pixels.
            await self._revalidate(evidence)

        return ResolvedImage(reference=reference, image_bytes=image_bytes, revalidate=revalidate)

    async def image_previews(
        self, *, owner_id: str, records: Sequence[tuple[RetrievalRecordType, UUID]],
    ) -> dict[tuple[RetrievalRecordType, UUID], list[dict[str, Any]]]:
        if not records or self._image_loader is None:
            return {}
        try:
            async with asyncio.timeout(self._timeout):
                evidence = await self._repository.list_image_evidence(
                    owner_id=owner_id, records=records, limit=self._max_frames,
                )
        except TimeoutError as error:
            raise ImageUnavailableError("image_lookup_timeout", "Historical image metadata could not be loaded in time.") from error
        except Exception as error:
            raise ImageUnavailableError("image_lookup_failed", "Historical image metadata is temporarily unavailable.") from error
        requested = set(records)
        seen_frames: set[UUID] = set()
        seen_relations: set[tuple[str, UUID, UUID]] = set()
        previews: dict[tuple[RetrievalRecordType, UUID], list[dict[str, Any]]] = {}
        for item in evidence:
            reference = item.reference
            key = (cast(RetrievalRecordType, reference.record_type), reference.record_id)
            if key not in requested:
                continue
            try:
                self._validate_evidence(item, owner_id=owner_id, record_type=key[0], record_id=key[1])
            except ImageUnavailableError:
                continue
            relation = (reference.record_type, reference.record_id, reference.frame_id)
            if relation in seen_relations:
                continue
            if reference.frame_id not in seen_frames and len(seen_frames) >= self._max_frames:
                continue
            seen_frames.add(reference.frame_id)
            seen_relations.add(relation)
            previews.setdefault(key, []).append({
                "frame_id": str(reference.frame_id),
                "captured_at": reference.captured_at.isoformat(),
                "availability": "available",
            })
        return previews

    async def _read_evidence(
        self, *, owner_id: str, record_id: UUID, record_type: RetrievalRecordType,
        frame_id: UUID | None = None,
    ) -> HistoricalImageEvidence:
        try:
            async with asyncio.timeout(self._timeout):
                evidence = await self._repository.get_image_evidence(
                    owner_id=owner_id, record_id=record_id, record_type=record_type, frame_id=frame_id,
                )
        except TimeoutError as error:
            raise ImageUnavailableError("image_lookup_timeout", "Historical image metadata could not be loaded in time.") from error
        except Exception as error:
            raise ImageUnavailableError("image_lookup_failed", "Historical image metadata is temporarily unavailable.") from error
        if evidence is None:
            raise self._unavailable()
        self._validate_evidence(evidence, owner_id=owner_id, record_id=record_id, record_type=record_type)
        if frame_id is not None and evidence.reference.frame_id != frame_id:
            raise self._unavailable()
        return evidence

    def _validate_evidence(
        self, evidence: HistoricalImageEvidence, *, owner_id: str,
        record_id: UUID, record_type: RetrievalRecordType,
    ) -> None:
        reference = evidence.reference
        if (
            reference.source != "historical" or reference.owner_id != owner_id.strip()
            or reference.record_id != record_id or reference.record_type != record_type
            or record_type not in RETRIEVAL_RECORD_TYPES
            or reference.raw_image_expires_at is None
            or reference.raw_image_expires_at <= self._now()
        ):
            raise self._unavailable()
        if reference.width > self._max_width or reference.height > self._max_height:
            raise ImageUnavailableError("invalid_image", "The historical image dimensions exceed the accepted limits.")

    async def _revalidate(self, evidence: HistoricalImageEvidence) -> None:
        reference = evidence.reference
        current = await self._read_evidence(
            owner_id=reference.owner_id, record_id=reference.record_id,
            record_type=cast(RetrievalRecordType, reference.record_type), frame_id=reference.frame_id,
        )
        if current != evidence:
            raise self._unavailable()

    def _validate_pixels(self, reference: ImageReference, image_bytes: bytes) -> None:
        if not image_bytes or len(image_bytes) > self._max_image_bytes:
            raise ImageUnavailableError("invalid_image", "The historical image exceeds the accepted byte limits.")
        try:
            dimensions = jpeg_dimensions(image_bytes)
        except VisualFrameValidationError as error:
            raise ImageUnavailableError("invalid_image", "The historical image is not a valid JPEG.") from error
        if dimensions != (reference.width, reference.height):
            raise ImageUnavailableError("invalid_image", "The historical image dimensions do not match its metadata.")

    @staticmethod
    def _unavailable() -> ImageUnavailableError:
        # Missing, foreign, unrelated and forgotten records intentionally look the same.
        return ImageUnavailableError("frame_unavailable", "The selected historical image is unavailable.")
