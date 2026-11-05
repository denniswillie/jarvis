from __future__ import annotations

import time
from collections.abc import Callable
from datetime import UTC, datetime
from math import isfinite
from uuid import UUID

from app.images import ImageReference, ImageUnavailableError, ResolvedImage
from app.perception.frame_window import (
    RecentVisualFrameWindow,
    SensoryVisualFrame,
    SensoryVisualFrameSample,
    VisualStreamLease,
)
from app.visual_memory.validation import (
    JPEG_CONTENT_TYPE,
    VisualFrameValidationError,
    jpeg_dimensions,
)


class CurrentImageService:
    """Retrieve exact current pixels without a model, storage lookup, or new cache."""

    def __init__(
        self,
        *,
        frame_window: RecentVisualFrameWindow,
        max_age_seconds: float,
        max_image_bytes: int,
        max_width: int,
        max_height: int,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        if not isfinite(max_age_seconds) or max_age_seconds <= 0:
            raise ValueError("Current image maximum age must be finite and positive.")
        if min(max_image_bytes, max_width, max_height) < 1:
            raise ValueError("Current image byte and dimension limits must be positive.")
        self._frame_window = frame_window
        self._max_age_seconds = max_age_seconds
        self._max_image_bytes = max_image_bytes
        self._max_width = max_width
        self._max_height = max_height
        self._now = now
        self._monotonic = monotonic

    def select_current_image(self, *, owner_id: str, session_id: UUID) -> ImageReference:
        lease = self._require_stream(owner_id, session_id)
        frame = self._frame_window.latest(owner_id, session_id)
        if frame is None:
            raise ImageUnavailableError("no_frame", "The active camera stream has no image available.")
        sample = self._require_sample(owner_id, session_id, frame.frame_id)
        return self._validated_reference(sample, lease)

    def resolve_current_image(
        self,
        *,
        owner_id: str,
        session_id: UUID,
        reference: ImageReference,
    ) -> ResolvedImage:
        if (
            reference.source != "current"
            or reference.owner_id != owner_id
            or reference.session_id != session_id
        ):
            raise ImageUnavailableError("frame_unavailable", "The selected image is unavailable.")
        lease = self._require_stream(owner_id, session_id)
        if (
            reference.visual_session_id != lease.visual_session_id
            or reference.stream_connection_id != lease.connection_id
        ):
            raise ImageUnavailableError("source_changed", "The camera stream has changed.")
        sample = self._require_sample(owner_id, session_id, reference.frame_id)
        current_reference = self._validated_reference(sample, lease)
        if current_reference != reference:
            raise ImageUnavailableError("frame_unavailable", "The selected image has changed.")
        return ResolvedImage(reference=reference, image_bytes=sample.frame.image_bytes)

    def _require_stream(self, owner_id: str, session_id: UUID) -> VisualStreamLease:
        lease = self._frame_window.stream_lease(owner_id, session_id)
        if lease is None:
            raise ImageUnavailableError("no_active_stream", "There is no active camera stream.")
        return lease

    def _require_sample(
        self, owner_id: str, session_id: UUID, frame_id: UUID,
    ) -> SensoryVisualFrameSample:
        sample = self._frame_window.get_sample(owner_id, session_id, frame_id=frame_id)
        if sample is None:
            raise ImageUnavailableError("frame_unavailable", "The selected image is no longer available.")
        return sample

    def _validated_reference(
        self, sample: SensoryVisualFrameSample, lease: VisualStreamLease,
    ) -> ImageReference:
        frame = sample.frame
        if (
            frame.owner_id != lease.owner_id
            or frame.session_id != lease.session_id
            or frame.visual_session_id != lease.visual_session_id
            or not self._frame_window.owns_stream(lease)
        ):
            raise ImageUnavailableError("source_changed", "The camera stream has changed.")
        self._validate_image(frame)
        now = self._now()
        monotonic_now = self._monotonic()
        try:
            reference = ImageReference(
                frame_id=frame.frame_id,
                owner_id=frame.owner_id,
                session_id=frame.session_id,
                source="current",
                captured_at=frame.captured_at,
                content_type=frame.content_type,
                width=frame.width,
                height=frame.height,
                raw_image_expires_at=frame.raw_image_expires_at,
                visual_session_id=frame.visual_session_id,
                frame_sequence=frame.frame_sequence,
                stream_connection_id=lease.connection_id,
                server_received_at=frame.server_received_at,
                sampled_at=sample.sampled_at,
                sampled_monotonic=sample.sampled_monotonic,
            )
            freshness = reference.freshness(
                now=now,
                monotonic_now=monotonic_now,
                max_age_seconds=self._max_age_seconds,
            )
        except ValueError as error:
            raise ImageUnavailableError(
                "clock_uncertain", "The image capture timing cannot establish a current view.",
            ) from error
        if freshness.status == "uncertain":
            raise ImageUnavailableError(
                "clock_uncertain", "The image capture timing cannot establish a current view.",
            )
        if freshness.status == "stale":
            raise ImageUnavailableError("stale_frame", "The available camera image is too old.")
        # Wall-clock rollback must not postpone an already-sampled expiry.
        remaining_at_sample = (frame.raw_image_expires_at - sample.sampled_at).total_seconds()
        if (
            now >= frame.raw_image_expires_at
            or monotonic_now - sample.sampled_monotonic >= remaining_at_sample
        ):
            raise ImageUnavailableError("frame_unavailable", "The selected image has expired.")
        if not self._frame_window.owns_stream(lease):
            raise ImageUnavailableError("source_changed", "The camera stream has changed.")
        if not self._frame_window.contains(frame):
            raise ImageUnavailableError("frame_unavailable", "The selected image is no longer available.")
        return reference

    def _validate_image(self, frame: SensoryVisualFrame) -> None:
        if (
            frame.content_type != JPEG_CONTENT_TYPE
            or not frame.image_bytes
            or len(frame.image_bytes) > self._max_image_bytes
            or frame.frame_sequence < 0
        ):
            raise ImageUnavailableError("invalid_image", "The camera image is not a supported JPEG.")
        try:
            width, height = jpeg_dimensions(frame.image_bytes)
        except VisualFrameValidationError as error:
            raise ImageUnavailableError("invalid_image", "The camera image is not a valid JPEG.") from error
        if (
            width != frame.width
            or height != frame.height
            or width > self._max_width
            or height > self._max_height
        ):
            raise ImageUnavailableError(
                "invalid_image", "The camera image dimensions do not match the accepted limits.",
            )
