from __future__ import annotations

import unittest
from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from uuid import uuid4

from app.images import ImageReference, ImageUnavailableError
from app.perception.current_image import CurrentImageService
from app.perception.frame_window import RecentVisualFrameWindow, SensoryVisualFrame


def _jpeg(width: int = 48, height: int = 32) -> bytes:
    """Header fixture for the existing dimension parser, not an image decoder."""
    return (
        b"\xff\xd8\xff\xc0\x00\x11\x08"
        + height.to_bytes(2, "big")
        + width.to_bytes(2, "big")
        + b"\x03\x01\x11\x00\x02\x11\x00\x03\x11\x00\xff\xd9"
    )


class _Clock:
    def __init__(self) -> None:
        self.wall = datetime(2026, 9, 9, 12, 0, tzinfo=UTC)
        self.ticks = 100.0

    def now(self) -> datetime:
        return self.wall

    def monotonic(self) -> float:
        return self.ticks

    def advance(self, seconds: float) -> None:
        self.wall += timedelta(seconds=seconds)
        self.ticks += seconds


class CurrentImageServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.clock = _Clock()
        self.owner = "owner-a"
        self.session = uuid4()
        self.window = RecentVisualFrameWindow(
            max_frames_per_session=4,
            ttl_seconds=30,
            now=self.clock.now,
            monotonic=self.clock.monotonic,
        )
        self.lease = self.window.begin_stream(self.owner, self.session, uuid4())
        self.service = self._service()

    def _service(self, **changes) -> CurrentImageService:
        options = {
            "frame_window": self.window,
            "max_age_seconds": 10,
            "max_image_bytes": 4096,
            "max_width": 1024,
            "max_height": 1024,
            "now": self.clock.now,
            "monotonic": self.clock.monotonic,
        }
        options.update(changes)
        return CurrentImageService(**options)

    def _frame(self, **changes) -> SensoryVisualFrame:
        frame = SensoryVisualFrame(
            frame_id=uuid4(),
            owner_id=self.owner,
            session_id=self.session,
            audio_session_id=None,
            visual_session_id=self.lease.visual_session_id,
            frame_sequence=1,
            source="glasses",
            captured_at=self.clock.wall - timedelta(seconds=0.5),
            server_received_at=self.clock.wall - timedelta(seconds=0.25),
            content_type="image/jpeg",
            width=48,
            height=32,
            storage_bucket="private-frames",
            storage_path="private/owner/frame.jpg",
            raw_image_expires_at=self.clock.wall + timedelta(seconds=60),
            image_bytes=_jpeg(),
        )
        return replace(frame, **changes)

    def _put(self, frame: SensoryVisualFrame | None = None) -> SensoryVisualFrame:
        frame = frame if frame is not None else self._frame()
        self.assertTrue(self.window.put(frame, stream_lease=self.lease))
        return frame

    def _select(self) -> ImageReference:
        return self.service.select_current_image(owner_id=self.owner, session_id=self.session)

    def _resolve(self, reference: ImageReference):
        return self.service.resolve_current_image(
            owner_id=self.owner, session_id=self.session, reference=reference,
        )

    def _unavailable(self, code: str, operation: Callable[[], object]) -> None:
        with self.assertRaises(ImageUnavailableError) as caught:
            operation()
        self.assertEqual(caught.exception.code, code)
        self.assertTrue(caught.exception.message)

    def test_selects_and_resolves_original_current_pixels(self) -> None:
        frame = self._put()
        reference = self._select()
        resolved = self._resolve(reference)
        self.assertEqual(reference.frame_id, frame.frame_id)
        self.assertEqual(reference.source, "current")
        self.assertEqual(reference.stream_connection_id, self.lease.connection_id)
        self.assertEqual(reference.captured_at, frame.captured_at)
        self.assertEqual((reference.width, reference.height), (48, 32))
        self.assertEqual(resolved.reference, reference)
        self.assertEqual(resolved.image_bytes, frame.image_bytes)

    def test_selection_uses_latest_frame(self) -> None:
        self._put()
        newer = self._put(self._frame(frame_sequence=2, captured_at=self.clock.wall))
        self.assertEqual(self._select().frame_id, newer.frame_id)

    def test_newer_frame_does_not_replace_an_exact_selected_reference(self) -> None:
        original = self._put()
        selected = self._select()
        self.clock.advance(1)
        newer = self._put(self._frame(frame_sequence=2, width=96, image_bytes=_jpeg(96, 32)))
        self.assertEqual(self._select().frame_id, newer.frame_id)
        self.assertEqual(self._resolve(selected).image_bytes, original.image_bytes)

    def test_empty_active_stream_is_distinct_from_no_stream(self) -> None:
        self._unavailable("no_frame", self._select)
        self.window.end_stream(self.lease)
        self._unavailable("no_active_stream", self._select)

    def test_unleased_buffered_frame_does_not_establish_a_current_stream(self) -> None:
        frame = self._frame()
        self.window.end_stream(self.lease)
        self.assertTrue(self.window.put(frame))
        self._unavailable("no_active_stream", self._select)

    def test_selection_and_lookup_are_owner_scoped(self) -> None:
        own_frame = self._put()
        other_lease = self.window.begin_stream("owner-b", self.session, uuid4())
        other_frame = self._frame(
            owner_id="owner-b", visual_session_id=other_lease.visual_session_id,
            captured_at=self.clock.wall,
        )
        self.assertTrue(self.window.put(other_frame, stream_lease=other_lease))
        self.assertEqual(self._select().frame_id, own_frame.frame_id)
        self.assertIsNone(self.window.get_sample(self.owner, self.session, frame_id=other_frame.frame_id))
        self._unavailable(
            "frame_unavailable",
            lambda: self._resolve(replace(self._select(), frame_id=other_frame.frame_id)),
        )

    def test_selection_is_session_scoped(self) -> None:
        own_frame = self._put()
        other_session = uuid4()
        other_lease = self.window.begin_stream(self.owner, other_session, uuid4())
        other_frame = self._frame(
            session_id=other_session, visual_session_id=other_lease.visual_session_id,
            captured_at=self.clock.wall,
        )
        self.assertTrue(self.window.put(other_frame, stream_lease=other_lease))
        self.assertEqual(self._select().frame_id, own_frame.frame_id)

    def test_resolution_rejects_wrong_owner_or_session_before_stream_lookup(self) -> None:
        self._put()
        selected = self._select()
        for owner, session in (("owner-b", self.session), (self.owner, uuid4())):
            with self.subTest(owner=owner, session=session):
                self._unavailable(
                    "frame_unavailable",
                    lambda: self.service.resolve_current_image(
                        owner_id=owner, session_id=session, reference=selected,
                    ),
                )

    def test_reconnect_reusing_visual_session_id_invalidates_reference(self) -> None:
        frame = self._put()
        selected = self._select()
        replacement = self.window.begin_stream(self.owner, self.session, self.lease.visual_session_id)
        self.assertNotEqual(replacement.connection_id, self.lease.connection_id)
        self.assertFalse(self.window.put(frame, stream_lease=self.lease))
        self._unavailable("source_changed", lambda: self._resolve(selected))

    def test_stream_end_invalidates_reference(self) -> None:
        self._put()
        selected = self._select()
        self.window.end_stream(self.lease)
        self._unavailable("no_active_stream", lambda: self._resolve(selected))

    def test_recent_upload_does_not_make_old_capture_fresh(self) -> None:
        self._put(self._frame(
            captured_at=self.clock.wall - timedelta(seconds=11),
            server_received_at=self.clock.wall,
        ))
        self._unavailable("stale_frame", self._select)

    def test_future_capture_time_is_uncertain(self) -> None:
        self._put(self._frame(captured_at=self.clock.wall + timedelta(seconds=2)))
        self._unavailable("clock_uncertain", self._select)

    def test_future_receipt_time_is_uncertain(self) -> None:
        self._put(self._frame(server_received_at=self.clock.wall + timedelta(seconds=2)))
        self._unavailable("clock_uncertain", self._select)

    def test_naive_capture_timestamp_is_uncertain(self) -> None:
        self._put(self._frame(captured_at=self.clock.wall.replace(tzinfo=None)))
        self._unavailable("clock_uncertain", self._select)

    def test_monotonic_regression_is_uncertain(self) -> None:
        self._put()
        selected = self._select()
        self.clock.ticks -= 1
        self._unavailable("clock_uncertain", lambda: self._resolve(selected))

    def test_large_wall_clock_rollback_is_uncertain(self) -> None:
        self._put()
        selected = self._select()
        self.clock.wall -= timedelta(seconds=2)
        self.clock.ticks += 2
        self._unavailable("clock_uncertain", lambda: self._resolve(selected))

    def test_small_clock_rollback_does_not_rejuvenate_a_reselected_frame(self) -> None:
        self._put(self._frame(captured_at=self.clock.wall, server_received_at=self.clock.wall))
        self.clock.wall -= timedelta(seconds=0.5)
        self.clock.ticks += 10.1
        self._unavailable("stale_frame", self._select)

    def test_freshness_boundary_is_shared_with_realtime_policy(self) -> None:
        self._put(self._frame(captured_at=self.clock.wall - timedelta(seconds=10)))
        selected = self._select()
        self.clock.advance(0.000001)
        self._unavailable("stale_frame", lambda: self._resolve(selected))

    def test_eviction_does_not_silently_substitute_another_frame(self) -> None:
        self._put()
        selected = self._select()
        for sequence in range(2, 7):
            self._put(self._frame(frame_sequence=sequence))
        self._unavailable("frame_unavailable", lambda: self._resolve(selected))

    def test_exact_lookup_evicts_expired_window_entries(self) -> None:
        self._put()
        selected = self._select()
        self.clock.advance(30.1)
        self.assertIsNone(self.window.get_sample(self.owner, self.session, frame_id=selected.frame_id))
        self._unavailable("frame_unavailable", lambda: self._resolve(selected))

    def test_raw_image_expiry_is_enforced_at_its_boundary(self) -> None:
        self._put(self._frame(raw_image_expires_at=self.clock.wall + timedelta(seconds=1)))
        selected = self._select()
        self.clock.advance(1)
        self._unavailable("frame_unavailable", lambda: self._resolve(selected))

    def test_wall_clock_rollback_does_not_postpone_raw_expiry(self) -> None:
        self._put(self._frame(raw_image_expires_at=self.clock.wall + timedelta(seconds=1)))
        selected = self._select()
        self.clock.wall -= timedelta(seconds=0.5)
        self.clock.ticks += 2
        self._unavailable("frame_unavailable", lambda: self._resolve(selected))

    def test_repeated_selection_and_resolution_preserve_original_timing(self) -> None:
        self._put()
        original = self._select()
        self.clock.advance(2)
        self.assertEqual(self._select(), original)
        self.assertEqual(self._resolve(original).reference, original)

    def test_duplicate_put_preserves_original_sample_and_window_expiry(self) -> None:
        frame = self._put()
        original = self.window.get_sample(self.owner, self.session, frame_id=frame.frame_id)
        self.clock.advance(29)
        self._put(frame)
        self.assertEqual(self.window.get_sample(self.owner, self.session, frame_id=frame.frame_id), original)
        self.clock.advance(1.1)
        self.assertIsNone(self.window.get_sample(self.owner, self.session, frame_id=frame.frame_id))

    def test_new_frame_at_same_sequence_gets_new_sample_timing(self) -> None:
        original = self._put()
        self.clock.advance(2)
        replacement = self._put(self._frame())
        sample = self.window.get_sample(self.owner, self.session, frame_id=replacement.frame_id)
        self.assertIsNotNone(sample)
        self.assertEqual(sample.sampled_at, self.clock.wall)
        self.assertEqual(sample.sampled_monotonic, self.clock.ticks)
        self.assertIsNone(self.window.get_sample(self.owner, self.session, frame_id=original.frame_id))

    def test_resolution_rejects_changed_reference_metadata(self) -> None:
        self._put()
        selected = self._select()
        for changes in ({"width": 49}, {"sampled_monotonic": 101.0}):
            with self.subTest(changes=changes):
                changed = replace(selected, **changes)
                self._unavailable("frame_unavailable", lambda: self._resolve(changed))

    def test_historical_reference_is_not_a_current_image(self) -> None:
        self._put()
        historical = replace(self._select(), source="historical")
        self._unavailable("frame_unavailable", lambda: self._resolve(historical))

    def test_invalid_image_payloads_and_dimensions_are_rejected(self) -> None:
        cases = (
            {"content_type": "image/png"},
            {"image_bytes": b""},
            {"image_bytes": b"not an image"},
            {"image_bytes": b"\xff\xd8\xff\xc0\x00\x11"},
            {"image_bytes": _jpeg() + b"x" * 4096},
            {"width": 49},
            {"height": 0},
            {"frame_sequence": -1},
            {"width": 1100, "image_bytes": _jpeg(1100, 32)},
            {"height": 1100, "image_bytes": _jpeg(48, 1100)},
        )
        for changes in cases:
            with self.subTest(changes=tuple(changes)):
                self.lease = self.window.begin_stream(self.owner, self.session, uuid4())
                self._put(self._frame(**changes))
                self._unavailable("invalid_image", self._select)

    def test_payload_at_exact_configured_limits_is_accepted(self) -> None:
        frame = self._put()
        service = self._service(max_image_bytes=len(frame.image_bytes), max_width=48, max_height=32)
        self.assertEqual(
            service.select_current_image(owner_id=self.owner, session_id=self.session).frame_id,
            frame.frame_id,
        )

    def test_invalid_service_limits_are_rejected(self) -> None:
        for changes in (
            {"max_age_seconds": 0}, {"max_age_seconds": -1},
            {"max_age_seconds": float("nan")}, {"max_age_seconds": float("inf")},
            {"max_image_bytes": 0}, {"max_width": 0}, {"max_height": -1},
        ):
            with self.subTest(changes=changes):
                with self.assertRaises(ValueError):
                    self._service(**changes)


if __name__ == "__main__":
    unittest.main()
