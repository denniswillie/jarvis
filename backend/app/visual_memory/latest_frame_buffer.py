from __future__ import annotations

from uuid import UUID

from app.perception.frame_window import RecentVisualFrameWindow, SensoryVisualFrame
from app.visual_memory.models import BufferedVisualFrame


class LatestVisualFrameBuffer:
    """Legacy metadata view backed by the bounded sensory frame window."""

    def __init__(
        self,
        *,
        ttl_seconds: int,
        frame_window: RecentVisualFrameWindow | None = None,
    ) -> None:
        self._ttl_seconds = max(1, ttl_seconds)
        self._frame_window = frame_window or RecentVisualFrameWindow(
            max_frames_per_session=1,
            ttl_seconds=self._ttl_seconds,
        )

    @property
    def frame_window(self) -> RecentVisualFrameWindow:
        return self._frame_window

    def put(self, frame: BufferedVisualFrame, *, image_bytes: bytes = b"") -> None:
        self._frame_window.put(
            SensoryVisualFrame.from_buffered_frame(frame, image_bytes=image_bytes)
        )

    def get(self, session_id: UUID) -> BufferedVisualFrame | None:
        frame = self._frame_window.latest_for_session(
            session_id,
            max_age_seconds=self._ttl_seconds,
        )
        return frame.as_buffered_frame() if frame is not None else None

    def remove(self, session_id: UUID) -> None:
        self._frame_window.clear_session(session_id)
