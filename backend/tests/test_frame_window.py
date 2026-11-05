from __future__ import annotations

import unittest
from datetime import datetime, timezone
from uuid import UUID, uuid4

from app.perception.frame_window import RecentVisualFrameWindow, SensoryVisualFrame


def _frame(owner_id: str, session_id: UUID, sequence: int) -> SensoryVisualFrame:
    now = datetime.now(timezone.utc)
    return SensoryVisualFrame(
        frame_id=uuid4(), owner_id=owner_id, session_id=session_id,
        audio_session_id=None, visual_session_id=uuid4(), frame_sequence=sequence,
        source="glasses", captured_at=now, server_received_at=now,
        content_type="image/jpeg", width=640, height=360,
        storage_bucket="frames", storage_path=f"{sequence}.jpg",
        raw_image_expires_at=now, image_bytes=f"jpeg-{sequence}".encode(),
    )


class FrameWindowTests(unittest.TestCase):
    def test_window_is_owner_bound_bounded_and_sequence_idempotent(self) -> None:
        clock = [100.0]
        window = RecentVisualFrameWindow(
            max_frames_per_session=2, ttl_seconds=10, monotonic=lambda: clock[0]
        )
        session_id = uuid4()
        window.put(_frame("owner-a", session_id, 1))
        window.put(_frame("owner-a", session_id, 2))
        window.put(_frame("owner-a", session_id, 2))
        window.put(_frame("owner-a", session_id, 3))
        self.assertEqual(
            [frame.frame_sequence for frame in window.window("owner-a", session_id)],
            [2, 3],
        )
        self.assertEqual(window.window("owner-b", session_id), ())

    def test_every_public_read_evicts_expired_frames(self) -> None:
        clock = [100.0]
        session_id = uuid4()
        window = RecentVisualFrameWindow(
            max_frames_per_session=4, ttl_seconds=10, monotonic=lambda: clock[0]
        )
        window.put(_frame("owner", session_id, 1))
        clock[0] = 111.0
        self.assertIsNone(window.latest("owner", session_id))
        self.assertEqual(window.window("owner", session_id), ())

    def test_clear_removes_only_requested_owner_session(self) -> None:
        window = RecentVisualFrameWindow(max_frames_per_session=4, ttl_seconds=10)
        session_id = uuid4()
        window.put(_frame("owner-a", session_id, 1))
        window.put(_frame("owner-b", session_id, 1))
        window.clear("owner-a", session_id)
        self.assertIsNone(window.latest("owner-a", session_id))
        self.assertIsNotNone(window.latest("owner-b", session_id))


if __name__ == "__main__":
    unittest.main()
