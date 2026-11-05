"""Shared metadata/frame fixture; JPEG headers exercise parsing, not vision accuracy."""

from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

from app.perception.frame_window import SensoryVisualFrame
from test_current_image_service import _jpeg


def _frame(owner_id: str, session_id: UUID) -> SensoryVisualFrame:
    now = datetime.now(UTC)
    return SensoryVisualFrame(
        frame_id=uuid4(), owner_id=owner_id, session_id=session_id,
        audio_session_id=None, visual_session_id=uuid4(), frame_sequence=1,
        source="glasses", captured_at=now, server_received_at=now,
        content_type="image/jpeg", width=640, height=360,
        storage_bucket="frames", storage_path="frame.jpg",
        raw_image_expires_at=now + timedelta(minutes=5), image_bytes=_jpeg(640, 360),
    )
