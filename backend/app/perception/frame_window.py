from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import Callable
from uuid import UUID, uuid4

from app.visual_memory.models import BufferedVisualFrame, VisualFrameSource


@dataclass(frozen=True)
class SensoryVisualFrame:
    """Immutable, short-lived frame snapshot used for immediate perception."""

    frame_id: UUID
    owner_id: str
    session_id: UUID
    audio_session_id: str | None
    visual_session_id: UUID
    frame_sequence: int
    source: VisualFrameSource
    captured_at: datetime
    server_received_at: datetime
    content_type: str
    width: int
    height: int
    storage_bucket: str
    storage_path: str
    raw_image_expires_at: datetime
    image_bytes: bytes

    @classmethod
    def from_buffered_frame(
        cls,
        frame: BufferedVisualFrame,
        *,
        image_bytes: bytes,
    ) -> SensoryVisualFrame:
        return cls(
            frame_id=frame.frame_id,
            owner_id=frame.owner_id,
            session_id=frame.session_id,
            audio_session_id=frame.audio_session_id,
            visual_session_id=frame.visual_session_id,
            frame_sequence=frame.frame_sequence,
            source=frame.source,
            captured_at=frame.captured_at,
            server_received_at=frame.server_received_at,
            content_type=frame.content_type,
            width=frame.width,
            height=frame.height,
            storage_bucket=frame.storage_bucket,
            storage_path=frame.storage_path,
            raw_image_expires_at=frame.raw_image_expires_at,
            image_bytes=bytes(image_bytes),
        )

    def as_buffered_frame(self) -> BufferedVisualFrame:
        return BufferedVisualFrame(
            frame_id=self.frame_id,
            owner_id=self.owner_id,
            session_id=self.session_id,
            audio_session_id=self.audio_session_id,
            visual_session_id=self.visual_session_id,
            frame_sequence=self.frame_sequence,
            source=self.source,
            captured_at=self.captured_at,
            server_received_at=self.server_received_at,
            content_type=self.content_type,
            width=self.width,
            height=self.height,
            storage_bucket=self.storage_bucket,
            storage_path=self.storage_path,
            raw_image_expires_at=self.raw_image_expires_at,
        )


@dataclass(frozen=True)
class VisualStreamLease:
    """Socket ownership is distinct from the app's reusable visual-session ID."""

    owner_id: str
    session_id: UUID
    visual_session_id: UUID
    connection_id: UUID = field(default_factory=uuid4)


@dataclass(frozen=True)
class _WindowEntry:
    frame: SensoryVisualFrame
    stored_monotonic: float
    sampled_at: datetime


@dataclass(frozen=True)
class SensoryVisualFrameSample:
    """Original local timing travels with a frame instead of restarting on reads."""

    frame: SensoryVisualFrame
    sampled_at: datetime
    sampled_monotonic: float


class RecentVisualFrameWindow:
    """Owner-bound sensory memory with strict per-session size and age bounds."""

    def __init__(
        self,
        *,
        max_frames_per_session: int,
        ttl_seconds: float,
        monotonic: Callable[[], float] = time.monotonic,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        if max_frames_per_session < 1:
            raise ValueError("max_frames_per_session must be at least 1")
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be greater than 0")
        self._max_frames_per_session = max_frames_per_session
        self._ttl_seconds = ttl_seconds
        self._monotonic = monotonic
        self._now = now
        self._entries: dict[tuple[str, UUID], dict[int, _WindowEntry]] = {}
        self._active_streams: dict[tuple[str, UUID], VisualStreamLease] = {}

    def begin_stream(
        self, owner_id: str, session_id: UUID, visual_session_id: UUID,
    ) -> VisualStreamLease:
        """Replace socket ownership even when the app reuses its stream ID."""
        self._evict_expired(self._monotonic())
        key = (owner_id, session_id)
        lease = VisualStreamLease(owner_id, session_id, visual_session_id)
        self._entries.pop(key, None)
        self._active_streams[key] = lease
        return lease

    def active_stream(self, owner_id: str, session_id: UUID) -> UUID | None:
        lease = self._active_streams.get((owner_id, session_id))
        return lease.visual_session_id if lease is not None else None

    def stream_lease(self, owner_id: str, session_id: UUID) -> VisualStreamLease | None:
        """A read-only source identity, including reconnects that reuse a stream ID."""
        return self._active_streams.get((owner_id, session_id))

    def contains(self, frame: SensoryVisualFrame) -> bool:
        """Recheck retention without requiring the inspected frame to remain latest."""
        self._evict_expired(self._monotonic())
        entry = self._entries.get((frame.owner_id, frame.session_id), {}).get(frame.frame_sequence)
        return entry is not None and entry.frame.frame_id == frame.frame_id

    def owns_stream(self, lease: VisualStreamLease) -> bool:
        return self._active_streams.get((lease.owner_id, lease.session_id)) == lease

    def end_stream(self, lease: VisualStreamLease) -> None:
        if not self.owns_stream(lease):
            return
        key = (lease.owner_id, lease.session_id)
        self._active_streams.pop(key, None)
        self._entries.pop(key, None)
        self._evict_expired(self._monotonic())

    def put(
        self, frame: SensoryVisualFrame, *, stream_lease: VisualStreamLease | None = None,
    ) -> bool:
        active = self._active_streams.get((frame.owner_id, frame.session_id))
        if active is not None:
            if stream_lease != active or frame.visual_session_id != active.visual_session_id:
                return False
        elif stream_lease is not None:
            # A retired socket cannot resurrect a cleared session's frame window.
            return False
        now = self._monotonic()
        self._evict_expired(now)
        entries = self._entries.setdefault((frame.owner_id, frame.session_id), {})
        previous = entries.get(frame.frame_sequence)
        same_frame = previous is not None and previous.frame.frame_id == frame.frame_id
        entries[frame.frame_sequence] = _WindowEntry(
            frame=replace(frame, image_bytes=bytes(frame.image_bytes)),
            stored_monotonic=previous.stored_monotonic if same_frame else now,
            sampled_at=previous.sampled_at if same_frame else self._now(),
        )
        while len(entries) > self._max_frames_per_session:
            oldest_sequence = min(
                entries,
                key=lambda sequence: (
                    entries[sequence].frame.captured_at,
                    entries[sequence].stored_monotonic,
                    sequence,
                ),
            )
            del entries[oldest_sequence]
        self._evict_expired(self._monotonic())
        return True

    def latest(
        self,
        owner_id: str,
        session_id: UUID,
        *,
        max_age_seconds: float | None = None,
    ) -> SensoryVisualFrame | None:
        frames = self.window(
            owner_id,
            session_id,
            max_frames=1,
            max_age_seconds=max_age_seconds,
        )
        return frames[-1] if frames else None

    def window(
        self,
        owner_id: str,
        session_id: UUID,
        *,
        max_frames: int | None = None,
        max_age_seconds: float | None = None,
    ) -> tuple[SensoryVisualFrame, ...]:
        if max_frames is not None and max_frames < 1:
            raise ValueError("max_frames must be at least 1")
        if max_age_seconds is not None and max_age_seconds < 0:
            raise ValueError("max_age_seconds cannot be negative")
        now = self._monotonic()
        self._evict_expired(now)
        eligible = [
            entry
            for entry in self._entries.get((owner_id, session_id), {}).values()
            if max_age_seconds is None
            or now - entry.stored_monotonic <= max_age_seconds
        ]
        eligible.sort(
            key=lambda entry: (
                entry.frame.captured_at,
                entry.stored_monotonic,
                entry.frame.frame_sequence,
            )
        )
        if max_frames is not None:
            eligible = eligible[-max_frames:]
        result = tuple(entry.frame for entry in eligible)
        self._evict_expired(self._monotonic())
        return result

    def get_sample(
        self,
        owner_id: str,
        session_id: UUID,
        *,
        frame_id: UUID,
    ) -> SensoryVisualFrameSample | None:
        """Resolve an exact owner-bound frame with its original local timing."""
        self._evict_expired(self._monotonic())
        entries = self._entries.get((owner_id, session_id), {})
        for entry in entries.values():
            if entry.frame.frame_id == frame_id:
                return SensoryVisualFrameSample(
                    frame=entry.frame,
                    sampled_at=entry.sampled_at,
                    sampled_monotonic=entry.stored_monotonic,
                )
        return None

    def latest_for_session(
        self,
        session_id: UUID,
        *,
        max_age_seconds: float | None = None,
    ) -> SensoryVisualFrame | None:
        now = self._monotonic()
        self._evict_expired(now)
        candidates = [
            entry
            for (_, candidate_session_id), entries in self._entries.items()
            if candidate_session_id == session_id
            for entry in entries.values()
            if max_age_seconds is None
            or now - entry.stored_monotonic <= max_age_seconds
        ]
        candidates.sort(
            key=lambda entry: (
                entry.frame.captured_at,
                entry.stored_monotonic,
                entry.frame.frame_sequence,
            )
        )
        result = candidates[-1].frame if candidates else None
        self._evict_expired(self._monotonic())
        return result

    def clear(self, owner_id: str, session_id: UUID) -> None:
        self._evict_expired(self._monotonic())
        self._entries.pop((owner_id, session_id), None)
        self._active_streams.pop((owner_id, session_id), None)
        self._evict_expired(self._monotonic())

    def clear_session(self, session_id: UUID) -> None:
        self._evict_expired(self._monotonic())
        for key in [key for key in self._active_streams if key[1] == session_id]:
            self._active_streams.pop(key, None)
        for key in [key for key in self._entries if key[1] == session_id]:
            self._entries.pop(key, None)
        self._evict_expired(self._monotonic())

    def diagnostics(self) -> dict[str, int | float | None]:
        """Return bounded window health without exposing frame bytes or identities."""

        now = self._monotonic()
        self._evict_expired(now)
        entries = [entry for session in self._entries.values() for entry in session.values()]
        ages = [max(0.0, now - entry.stored_monotonic) for entry in entries]
        result: dict[str, int | float | None] = {
            "session_count": len(self._entries),
            "frame_count": len(entries),
            "max_frames_per_session": self._max_frames_per_session,
            "ttl_seconds": self._ttl_seconds,
            "oldest_frame_age_seconds": max(ages) if ages else None,
            "newest_frame_age_seconds": min(ages) if ages else None,
        }
        self._evict_expired(self._monotonic())
        return result

    def _evict_expired(self, now: float) -> None:
        for key, entries in list(self._entries.items()):
            expired = [
                sequence
                for sequence, entry in entries.items()
                if now - entry.stored_monotonic > self._ttl_seconds
            ]
            for sequence in expired:
                entries.pop(sequence, None)
            if not entries:
                self._entries.pop(key, None)
