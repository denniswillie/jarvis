from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from datetime import datetime, timedelta, timezone
import json
import os
import unittest
from unittest.mock import patch
from uuid import uuid4

from pydantic import ValidationError

from app.perception.frame_window import SensoryVisualFrame
from app.perception.freshness import CAPTURE_CLOCK_TOLERANCE_SECONDS
from app.realtime.visual_context import (
    ProviderImageStatus,
    VisualAvailabilityReason,
    VisualAvailabilityStatus,
    VisualConnectionKey,
    VisualContextNotReadyError,
    VisualContextState,
    VisualDeliveryPolicy,
    VisualFrameMetadata,
    VisualSyncDeadline,
)
from app.settings import JarvisSettings


NOW = datetime(2026, 9, 6, 12, 0, tzinfo=timezone.utc)


class VisualDeliveryConfigurationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.enterContext(patch.dict(os.environ, {}, clear=True))

    def test_defaults_and_explicit_overrides(self) -> None:
        self.assertEqual(
            VisualDeliveryPolicy.from_settings(JarvisSettings(_env_file=None)),
            VisualDeliveryPolicy(),
        )
        settings = JarvisSettings(
            _env_file=None, VISUAL_REALTIME_ENABLED=False,
            VISUAL_REALTIME_MIN_INTERVAL_SECONDS=2,
            VISUAL_REALTIME_MAX_CONTEXT_AGE_SECONDS=4,
            VISUAL_REALTIME_SYNC_TIMEOUT_SECONDS=0.25,
            VISUAL_REALTIME_MAX_IMAGE_ITEMS=6,
        )
        self.assertEqual(
            VisualDeliveryPolicy.from_settings(settings),
            VisualDeliveryPolicy(False, 2, 4, 0.25, 6),
        )

    def test_invalid_duration_and_capacity_settings_are_rejected(self) -> None:
        for field in (
            "VISUAL_REALTIME_MIN_INTERVAL_SECONDS",
            "VISUAL_REALTIME_MAX_CONTEXT_AGE_SECONDS",
            "VISUAL_REALTIME_SYNC_TIMEOUT_SECONDS",
        ):
            for value in (0, -1, float("nan"), float("inf"), float("-inf")):
                with self.subTest(field=field, value=value), self.assertRaises(ValidationError):
                    JarvisSettings(_env_file=None, **{field: value})
        with self.assertRaises(ValidationError):
            JarvisSettings(_env_file=None, VISUAL_REALTIME_MAX_IMAGE_ITEMS=3)
        with self.assertRaises(ValueError):
            VisualDeliveryPolicy(max_image_items=3)
        with self.assertRaises(ValueError):
            VisualDeliveryPolicy(sync_timeout_seconds=float("inf"))

    def test_one_deadline_is_shared_across_operations(self) -> None:
        deadline = VisualSyncDeadline(100, 0.5)
        self.assertAlmostEqual(deadline.remaining(100.2), 0.3)
        self.assertAlmostEqual(deadline.remaining(100.4), 0.1)
        self.assertEqual(deadline.remaining(101), 0)
        with self.assertRaises(ValueError):
            deadline.remaining(99)


class VisualContextStateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.connection = VisualConnectionKey("owner", uuid4(), uuid4())
        self.stream_id = uuid4()
        self.state = VisualContextState(
            connection=self.connection, policy=VisualDeliveryPolicy(),
            visual_session_id=self.stream_id,
        )

    def frame(self, sequence=1, received=0, captured=None, connection=None, stream_id=None):
        connection = connection or self.connection
        sensory = SensoryVisualFrame(
            frame_id=uuid4(), owner_id=connection.owner_id,
            session_id=connection.session_id, audio_session_id="audio-session",
            visual_session_id=stream_id or self.stream_id, frame_sequence=sequence,
            source="glasses", captured_at=NOW + timedelta(seconds=received if captured is None else captured),
            server_received_at=NOW + timedelta(seconds=received),
            content_type="image/jpeg", width=640, height=360,
            storage_bucket="private-frames", storage_path="https://private.invalid/secret-path",
            raw_image_expires_at=NOW + timedelta(minutes=5), image_bytes=b"SECRET_IMAGE_BYTES",
        )
        return VisualFrameMetadata.from_sensory_frame(
            sensory, connection=connection,
            sampled_at=NOW + timedelta(seconds=received), sampled_monotonic=100 + received,
        )

    def reserve(self, at=0):
        return self.state.reserve_create(now=NOW + timedelta(seconds=at), monotonic_now=100 + at)

    def submit(self, item, at=0):
        self.assertTrue(self.state.mark_submitted(
            item_id=item.item_id, now=NOW + timedelta(seconds=at), monotonic_now=100 + at,
        ))

    def acknowledge(self, item, at=0):
        self.assertTrue(self.state.acknowledge_create(
            connection_epoch=self.connection.connection_epoch,
            item_id=item.item_id, acknowledged_at=NOW + timedelta(seconds=at),
        ))

    def accepted_frame(self, sequence=1, at=0):
        frame = self.frame(sequence=sequence, received=at)
        self.assertTrue(self.state.offer_frame(frame))
        item = self.reserve(at)
        self.assertIsNotNone(item)
        self.submit(item, at)
        self.acknowledge(item, at)
        return item

    def availability(self, at=0):
        return self.state.availability(now=NOW + timedelta(seconds=at), monotonic_now=100 + at)

    def prepare(self, at=0, allow_degraded=False):
        return self.state.prepare_response(
            turn_id="turn-1", generation_id=uuid4(),
            now=NOW + timedelta(seconds=at), monotonic_now=100 + at,
            allow_degraded=allow_degraded,
        )

    def test_disabled_and_no_vision_are_unavailable_without_speech_side_effects(self) -> None:
        self.state = VisualContextState(
            connection=self.connection, policy=VisualDeliveryPolicy(enabled=False),
            visual_session_id=self.stream_id,
        )
        self.assertFalse(self.state.offer_frame(self.frame()))
        self.assertIsNone(self.reserve())
        self.assertEqual(self.availability().reason, VisualAvailabilityReason.DISABLED)
        self.assertEqual(self.prepare().availability.status, VisualAvailabilityStatus.UNAVAILABLE)
        state = VisualContextState(connection=self.connection, policy=VisualDeliveryPolicy())
        self.assertEqual(
            state.availability(now=NOW, monotonic_now=100).reason,
            VisualAvailabilityReason.NO_VISUAL_STREAM,
        )

    def test_owner_session_epoch_and_stream_are_separate_boundaries(self) -> None:
        for connection in (
            replace(self.connection, owner_id="other"),
            replace(self.connection, session_id=uuid4()),
            replace(self.connection, connection_epoch=uuid4()),
        ):
            with self.subTest(connection=connection):
                self.assertFalse(self.state.offer_frame(self.frame(connection=connection)))
        self.assertFalse(self.state.offer_frame(self.frame(stream_id=uuid4())))
        self.assertEqual(self.state.items, ())

    def test_duplicate_and_reordered_frames_do_not_refresh_or_rewind(self) -> None:
        frame = self.frame(sequence=2, received=1)
        self.assertTrue(self.state.offer_frame(frame))
        self.assertFalse(self.state.offer_frame(replace(frame, sampled_at=NOW + timedelta(seconds=4), sampled_monotonic=104)))
        self.assertFalse(self.state.offer_frame(self.frame(sequence=1, received=2)))
        self.assertFalse(self.state.offer_frame(self.frame(sequence=3, received=0)))
        self.assertEqual(self.state.latest_local, frame)

    def test_only_one_create_and_one_replaceable_latest_candidate(self) -> None:
        self.state.offer_frame(self.frame())
        reserved = self.reserve()
        for sequence in range(2, 100):
            self.assertTrue(self.state.offer_frame(self.frame(sequence=sequence)))
        self.assertIsNone(self.reserve())
        self.assertFalse(self.state.mark_submitted(item_id=reserved.item_id, now=NOW, monotonic_now=100))
        latest = self.reserve()
        self.assertEqual(latest.frame.frame_sequence, 99)
        self.submit(latest)
        self.state.offer_frame(self.frame(sequence=100))
        self.assertIsNone(self.reserve())
        self.assertEqual(len(self.state.items), 1)
        self.assertEqual(self.state.pending_candidate(now=NOW, monotonic_now=100).frame_sequence, 100)
        self.assertIsNone(self.state.active_snapshot)

    def test_submission_is_not_acknowledgement_or_a_user_turn(self) -> None:
        self.state.offer_frame(self.frame())
        item = self.reserve()
        self.submit(item)
        self.assertIsNotNone(self.state.latest_submitted)
        self.assertIsNone(self.state.latest_acknowledged)
        self.assertEqual(self.availability().reason, VisualAvailabilityReason.AWAITING_ACKNOWLEDGEMENT)
        self.assertIsNone(self.state.active_snapshot)
        self.acknowledge(item)
        self.assertEqual(self.availability().status, VisualAvailabilityStatus.FRESH)
        self.assertEqual(self.state.to_metadata(now=NOW, monotonic_now=100)["input_kind"], "ambient_visual")

    def test_unknown_create_keeps_its_slot_and_late_acceptance_is_reconciled(self) -> None:
        self.state.offer_frame(self.frame())
        item = self.reserve()
        self.submit(item)
        self.assertTrue(self.state.mark_create_timeout(connection_epoch=self.connection.connection_epoch, item_id=item.item_id))
        self.state.offer_frame(self.frame(sequence=2, received=1))
        self.assertIsNone(self.reserve(1))
        self.assertEqual(self.state.free_slots, 3)
        self.assertEqual(self.availability(1).reason, VisualAvailabilityReason.PROVIDER_OUTCOME_UNKNOWN)
        self.assertFalse(self.state.acknowledge_delete(
            connection_epoch=self.connection.connection_epoch,
            item_id=item.item_id, request_id="speculative-not-found",
        ))
        self.acknowledge(item, 1)
        newer = self.reserve(1)
        self.submit(newer, 1)
        self.acknowledge(newer, 1)
        self.assertFalse(self.state.acknowledge_create(
            connection_epoch=self.connection.connection_epoch,
            item_id=item.item_id, acknowledged_at=NOW + timedelta(seconds=2),
        ))
        self.assertEqual(self.state.latest_acknowledged, newer.frame)

    def test_rejection_and_never_written_cancellation_release_only_valid_reservations(self) -> None:
        self.state.offer_frame(self.frame())
        first = self.reserve()
        self.assertTrue(self.state.cancel_reserved(connection_epoch=self.connection.connection_epoch, item_id=first.item_id))
        second = self.reserve()
        self.assertNotEqual(first.item_id, second.item_id)
        self.submit(second)
        self.assertFalse(self.state.cancel_reserved(connection_epoch=self.connection.connection_epoch, item_id=second.item_id))
        self.assertTrue(self.state.reject_create(connection_epoch=self.connection.connection_epoch, item_id=second.item_id))
        self.assertEqual(self.state.free_slots, 4)
        self.assertEqual(self.availability().reason, VisualAvailabilityReason.PROVIDER_REJECTED)
        self.assertFalse(self.state.acknowledge_create(
            connection_epoch=self.connection.connection_epoch,
            item_id=second.item_id, acknowledged_at=NOW,
        ))

    def test_two_pins_leave_room_for_advancing_images_and_unknown_deletes_count(self) -> None:
        first = self.accepted_frame(1, 0)
        second = self.accepted_frame(2, 1)
        snapshot = self.prepare(1)
        third = self.accepted_frame(3, 2)
        fourth = self.accepted_frame(4, 3)
        self.state.offer_frame(self.frame(sequence=5, received=4))
        self.assertIsNone(self.reserve(4))
        self.assertIsNone(self.state.begin_delete(connection_epoch=self.connection.connection_epoch, item_id=first.item_id))
        self.assertEqual(
            self.state.retirable_item_ids(now=NOW + timedelta(seconds=4), monotonic_now=104),
            (third.item_id,),
        )
        deletion = self.state.begin_delete(connection_epoch=self.connection.connection_epoch, item_id=third.item_id)
        self.assertTrue(self.state.mark_delete_timeout(connection_epoch=self.connection.connection_epoch, item_id=third.item_id, request_id=deletion))
        self.assertIsNone(self.reserve(4))
        self.assertFalse(self.state.acknowledge_delete(connection_epoch=self.connection.connection_epoch, item_id=third.item_id, request_id="wrong"))
        self.assertTrue(self.state.acknowledge_delete(connection_epoch=self.connection.connection_epoch, item_id=third.item_id, request_id=deletion))
        fifth = self.reserve(4)
        self.submit(fifth, 4)
        self.acknowledge(fifth, 4)
        self.assertEqual(len(self.state.items), 4)
        self.assertEqual(snapshot.pinned_item_ids, frozenset((first.item_id, second.item_id)))
        self.assertEqual(snapshot.current_frame, second.frame)
        self.assertFalse(self.state.settle_response(connection_epoch=self.connection.connection_epoch, generation_id=uuid4()))
        self.assertTrue(self.state.settle_response(connection_epoch=self.connection.connection_epoch, generation_id=snapshot.generation_id))
        self.assertEqual(
            self.state.retirable_item_ids(now=NOW + timedelta(seconds=4), monotonic_now=104),
            (first.item_id, second.item_id),
        )
        self.assertIn(fourth.item_id, {item.item_id for item in self.state.items})

    def test_full_or_unresolved_context_cannot_claim_a_healthy_cutoff(self) -> None:
        for sequence in range(1, 5):
            self.accepted_frame(sequence, sequence - 1)
        with self.assertRaises(VisualContextNotReadyError):
            self.prepare(3)
        degraded = self.prepare(3, allow_degraded=True)
        self.assertFalse(degraded.synchronized)
        self.assertEqual(degraded.availability.status, VisualAvailabilityStatus.DEGRADED)
        self.assertEqual(len(degraded.pinned_item_ids), 4)
        self.assertIsNone(degraded.current_frame)
        with self.assertRaises(VisualContextNotReadyError):
            self.prepare(3, allow_degraded=True)

    def test_degraded_snapshot_separates_acknowledged_and_possibly_present_images(self) -> None:
        first = self.accepted_frame()
        self.state.offer_frame(self.frame(sequence=2, received=1))
        unknown = self.reserve(1)
        self.submit(unknown, 1)
        self.state.mark_create_timeout(connection_epoch=self.connection.connection_epoch, item_id=unknown.item_id)
        snapshot = self.prepare(1, allow_degraded=True)
        self.assertEqual(tuple(item.item_id for item in snapshot.acknowledged_items), (first.item_id,))
        self.assertEqual(tuple(item.item_id for item in snapshot.possibly_present_items), (unknown.item_id,))
        self.assertIsNone(snapshot.current_frame)
        with self.assertRaises(FrozenInstanceError):
            snapshot.synchronized = True

    def test_more_capacity_does_not_enlarge_the_normal_snapshot_budget(self) -> None:
        self.state = VisualContextState(
            connection=self.connection, policy=VisualDeliveryPolicy(max_image_items=8),
            visual_session_id=self.stream_id,
        )
        for sequence in range(1, 4):
            self.accepted_frame(sequence, sequence - 1)
        with self.assertRaises(VisualContextNotReadyError):
            self.prepare(2)

    def test_stream_restart_allows_sequence_reset_but_requires_old_item_cleanup(self) -> None:
        self.state.offer_frame(self.frame(sequence=20))
        old = self.reserve()
        self.submit(old)
        new_stream = uuid4()
        self.state.activate_visual_stream(new_stream)
        self.acknowledge(old, 1)
        self.assertIsNone(self.state.latest_acknowledged)
        self.assertFalse(self.state.offer_frame(self.frame(sequence=21, received=1)))
        new = self.frame(sequence=0, received=1, stream_id=new_stream)
        self.assertTrue(self.state.offer_frame(new))
        self.assertIsNone(self.reserve(1))
        self.assertEqual(self.availability(1).reason, VisualAvailabilityReason.STREAM_CLEANUP_REQUIRED)
        deletion = self.state.begin_delete(connection_epoch=self.connection.connection_epoch, item_id=old.item_id)
        self.state.acknowledge_delete(connection_epoch=self.connection.connection_epoch, item_id=old.item_id, request_id=deletion)
        self.assertEqual(self.reserve(1).frame, new)

    def test_close_and_old_connection_events_cannot_repopulate_state(self) -> None:
        item = self.accepted_frame()
        self.assertFalse(self.state.mark_create_timeout(connection_epoch=uuid4(), item_id=item.item_id))
        self.state.close()
        self.assertEqual(self.state.items, ())
        self.assertFalse(self.state.offer_frame(self.frame(sequence=2)))
        self.assertFalse(self.state.acknowledge_create(connection_epoch=self.connection.connection_epoch, item_id=item.item_id, acknowledged_at=NOW))
        self.assertEqual(self.availability().reason, VisualAvailabilityReason.CONNECTION_CLOSED)
        with self.assertRaises(RuntimeError):
            self.state.activate_visual_stream(uuid4())

    def test_fresh_at_submission_can_expire_before_the_question(self) -> None:
        item = self.accepted_frame()
        self.assertEqual(self.availability(5).status, VisualAvailabilityStatus.FRESH)
        self.assertEqual(self.availability(6).status, VisualAvailabilityStatus.STALE)
        self.assertEqual(self.state.retirable_item_ids(now=NOW + timedelta(seconds=6), monotonic_now=106), (item.item_id,))
        with self.assertRaises(VisualContextNotReadyError):
            self.prepare(6)
        self.assertIsNone(self.prepare(6, allow_degraded=True).current_frame)

    def test_delayed_working_memory_uses_observation_time_not_update_arrival(self) -> None:
        item = self.accepted_frame(1, 2)
        snapshot = self.prepare(3)
        self.assertEqual(snapshot.current_frame, item.frame)
        self.assertTrue(snapshot.observation_is_older(observed_at=NOW + timedelta(seconds=1)))
        self.assertFalse(snapshot.observation_is_older(observed_at=NOW + timedelta(seconds=3)))

    def test_metadata_does_not_retain_or_serialize_image_bytes_or_storage_secrets(self) -> None:
        self.accepted_frame()
        self.prepare()
        encoded = json.dumps(self.state.to_metadata(now=NOW, monotonic_now=100))
        for forbidden in ("SECRET_IMAGE_BYTES", "private-frames", "private.invalid", "storage_path", "image_bytes", "base64"):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, encoded)
        self.assertIn("ambient_visual", encoded)

    def test_old_uploads_future_capture_and_clock_changes_do_not_manufacture_freshness(self) -> None:
        old = self.frame(captured=-30)
        self.assertEqual(old.freshness(now=NOW, monotonic_now=100, max_age_seconds=5).status, "stale")
        future = self.frame(captured=CAPTURE_CLOCK_TOLERANCE_SECONDS + 1)
        self.assertEqual(future.freshness(now=NOW, monotonic_now=100, max_age_seconds=5).status, "uncertain")
        tolerated = self.frame(captured=CAPTURE_CLOCK_TOLERANCE_SECONDS / 2)
        self.assertEqual(tolerated.freshness(now=NOW, monotonic_now=100, max_age_seconds=5).status, "fresh")
        self.assertEqual(tolerated.captured_at, NOW + timedelta(seconds=CAPTURE_CLOCK_TOLERANCE_SECONDS / 2))
        frame = self.frame()
        self.assertEqual(frame.freshness(now=NOW + timedelta(seconds=1), monotonic_now=107, max_age_seconds=5).status, "stale")
        self.assertEqual(frame.freshness(now=NOW, monotonic_now=99, max_age_seconds=5).status, "uncertain")
        with self.assertRaises(ValueError):
            replace(frame, captured_at=NOW.replace(tzinfo=None))


if __name__ == "__main__":
    unittest.main()
