from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from math import isfinite
from typing import Any, Literal
from uuid import UUID

from app.perception.freshness import (
    FrameFreshness,
    assess_frame_freshness,
    require_aware_timestamp,
)


ImageSource = Literal["current", "historical"]
ImageUnavailableReason = Literal[
    "no_active_stream",
    "no_frame",
    "stale_frame",
    "clock_uncertain",
    "source_changed",
    "frame_unavailable",
    "invalid_image",
    "image_lookup_failed",
    "image_lookup_timeout",
    "image_storage_failed",
    "image_resolution_timeout",
]


class ImageUnavailableError(Exception):
    """A recoverable evidence failure, not a reason to invoke another model."""

    def __init__(self, code: ImageUnavailableReason, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True, slots=True)
class ImageReference:
    """Authorized source metadata; provider attachment state is deliberately absent."""

    frame_id: UUID
    owner_id: str
    session_id: UUID
    source: ImageSource
    captured_at: datetime
    content_type: str
    width: int
    height: int
    raw_image_expires_at: datetime | None = None
    visual_session_id: UUID | None = None
    frame_sequence: int | None = None
    stream_connection_id: UUID | None = None
    server_received_at: datetime | None = None
    sampled_at: datetime | None = None
    sampled_monotonic: float | None = None
    retained_file_id: UUID | None = None
    record_type: str | None = None
    record_id: UUID | None = None

    def __post_init__(self) -> None:
        if not self.owner_id.strip():
            raise ValueError("Image owner_id cannot be blank.")
        if self.source not in {"current", "historical"}:
            raise ValueError("Unsupported image source.")
        if self.content_type != "image/jpeg" or self.width <= 0 or self.height <= 0:
            raise ValueError("Image references require a JPEG with positive dimensions.")
        for timestamp in (
            self.captured_at,
            self.raw_image_expires_at,
            self.server_received_at,
            self.sampled_at,
        ):
            if timestamp is not None:
                require_aware_timestamp(timestamp)
        if self.frame_sequence is not None and self.frame_sequence < 0:
            raise ValueError("Image frame_sequence cannot be negative.")
        if self.sampled_monotonic is not None and not isfinite(self.sampled_monotonic):
            raise ValueError("Image sample time must be finite.")
        if self.source == "current" and any(
            value is None
            for value in (
                self.visual_session_id,
                self.frame_sequence,
                self.stream_connection_id,
                self.server_received_at,
                self.sampled_at,
                self.sampled_monotonic,
                self.raw_image_expires_at,
            )
        ):
            raise ValueError("Current images require stream identity and original timing.")
        if self.retained_file_id is not None and (self.source != "historical" or self.raw_image_expires_at is not None):
            raise ValueError("Retained files are historical snapshots with independent storage lifetime.")
        if self.record_type == "source_evidence" and self.source != "historical":
            raise ValueError("Source-backed images are historical evidence, not a current view.")
        if (self.record_type is None) != (self.record_id is None):
            raise ValueError("Historical record type and ID must be supplied together.")

    @property
    def source_payload_ids(self) -> tuple[UUID, ...]:
        """A source-image record ID identifies its removable evidence dependency."""
        return (self.record_id,) if self.record_type == "source_evidence" and self.record_id is not None else ()

    def freshness(
        self,
        *,
        now: datetime,
        monotonic_now: float,
        max_age_seconds: float,
    ) -> FrameFreshness:
        if self.source != "current":
            raise ValueError("Historical images cannot establish current-view freshness.")
        assert self.server_received_at is not None
        assert self.sampled_at is not None
        assert self.sampled_monotonic is not None
        return assess_frame_freshness(
            captured_at=self.captured_at,
            server_received_at=self.server_received_at,
            sampled_at=self.sampled_at,
            sampled_monotonic=self.sampled_monotonic,
            now=now,
            monotonic_now=monotonic_now,
            max_age_seconds=max_age_seconds,
        )

    @classmethod
    def from_metadata(cls, metadata: Mapping[str, Any], *, owner_id: str) -> ImageReference:
        """Restore a durable selection using trusted invocation ownership.

        This decodes source metadata only. The source resolver must still check
        authorization, freshness, expiry, and the exact selected frame.
        """
        values = dict(metadata)
        values["owner_id"] = owner_id
        for name in ("frame_id", "session_id", "visual_session_id", "stream_connection_id", "record_id", "retained_file_id"):
            if values.get(name) is not None:
                values[name] = UUID(str(values[name]))
        for name in ("captured_at", "raw_image_expires_at", "server_received_at", "sampled_at"):
            if values.get(name) is not None:
                values[name] = datetime.fromisoformat(values[name])
        return cls(**values)

    def to_metadata(self) -> dict[str, Any]:
        """Safe source metadata, without bytes, storage locators, or owner credentials."""
        metadata: dict[str, Any] = {
            "frame_id": str(self.frame_id),
            "session_id": str(self.session_id),
            "source": self.source,
            "captured_at": self.captured_at.isoformat(),
            "content_type": self.content_type,
            "width": self.width,
            "height": self.height,
        }
        for name in ("visual_session_id", "stream_connection_id", "record_id", "retained_file_id"):
            value = getattr(self, name)
            if value is not None:
                metadata[name] = str(value)
        for name in ("raw_image_expires_at", "server_received_at", "sampled_at"):
            value = getattr(self, name)
            if value is not None:
                metadata[name] = value.isoformat()
        for name in ("frame_sequence", "sampled_monotonic", "record_type"):
            value = getattr(self, name)
            if value is not None:
                metadata[name] = value
        return metadata


@dataclass(frozen=True, slots=True)
class WorkspaceImageReference:
    """An inspected sandbox file, not a camera frame or proof of current UI state."""

    owner_id: str
    task_id: UUID
    attempt_id: UUID
    allocation_id: UUID
    path: str
    observed_at: datetime
    sha256: str
    content_type: str
    width: int
    height: int

    def __post_init__(self) -> None:
        require_aware_timestamp(self.observed_at)
        if (not self.owner_id.strip() or self.content_type not in {"image/png", "image/jpeg"}
                or self.width <= 0 or self.height <= 0
                or len(self.sha256) != 64 or any(c not in "0123456789abcdef" for c in self.sha256)):
            raise ValueError("Invalid workspace image metadata")

    def to_metadata(self) -> dict[str, Any]:
        return {"source": "sandbox_workspace", "task_id": str(self.task_id),
                "attempt_id": str(self.attempt_id), "allocation_id": str(self.allocation_id),
                "path": self.path, "observed_at": self.observed_at.isoformat(),
                "sha256": self.sha256, "content_type": self.content_type,
                "width": self.width, "height": self.height}


@dataclass(frozen=True, slots=True)
class RetainedWorkspaceImageReference:
    """A retained conversation's sandbox snapshot, not a fabricated task/frame."""

    owner_id: str
    logical_session_id: str
    input_id: str
    sandbox_id: str
    path: str
    observed_at: datetime
    sha256: str
    content_type: str
    width: int
    height: int

    def to_metadata(self) -> dict[str, Any]:
        return {"source": "retained_sandbox_workspace", "session_id": self.logical_session_id,
                "input_id": self.input_id, "sandbox_id": self.sandbox_id, "path": self.path,
                "observed_at": self.observed_at.isoformat(), "sha256": self.sha256,
                "content_type": self.content_type, "width": self.width, "height": self.height}


@dataclass(frozen=True, slots=True)
class ResolvedImage:
    """Transient image bytes for transport, never a JSON tool-execution result."""

    reference: ImageReference | WorkspaceImageReference | RetainedWorkspaceImageReference
    image_bytes: bytes = field(repr=False)
    # Transient source-policy check for queued writes; never stored in the ledger.
    revalidate: Callable[[], Awaitable[None]] | None = field(
        default=None, repr=False, compare=False,
    )


def image_source_payload_ids(result: Mapping[str, Any], *, owner_id: str) -> tuple[UUID, ...]:
    """Decode trusted image-tool metadata; callers still enforce owned source availability."""
    metadata = result.get("image")
    if not isinstance(metadata, dict) or metadata.get("record_type") != "source_evidence":
        return ()
    return ImageReference.from_metadata(metadata, owner_id=owner_id).source_payload_ids


ImageResolver = Callable[[dict[str, Any]], Awaitable[ResolvedImage]]
