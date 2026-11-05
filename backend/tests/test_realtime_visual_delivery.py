from __future__ import annotations

import asyncio
import json
import time
import unittest
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import Mock
from uuid import uuid4

from app.perception.frame_window import RecentVisualFrameWindow, SensoryVisualFrame
from app.realtime.openai_client import OpenAIRealtimeConnection
from app.realtime.session_manager import RealtimeSessionManager
from app.realtime.visual_context import (
    ProviderImageStatus, VisualAvailabilityStatus, VisualConnectionKey,
    VisualContextNotReadyError, VisualContextState, VisualDeliveryPolicy,
)
from app.realtime.visual_delivery import RealtimeVisualCoordinator
from app.runtime.events import DomainEvent, DomainEventBus, DomainEventName, VisualFrameAcceptedPayload
from app.settings import JarvisSettings


class _Clock:
    def __init__(self):
        self.origin = time.monotonic()
        self.offset = 0.0

    def monotonic(self):
        return time.monotonic() - self.origin + self.offset

    def now(self):
        return datetime(2026, 9, 6, 12, tzinfo=UTC) + timedelta(seconds=self.monotonic())


class _Provider:
    def __init__(self):
        self.events = []
        self.auto_ack = True
        self.before_write = None
        self.after_write = None
        self.coordinator = None

    async def send_visual_event(self, factory, *, timeout_seconds):
        async with asyncio.timeout(timeout_seconds):
            if self.before_write is not None:
                await self.before_write.wait()
            event = factory()
            if event is None:
                return False
            self.events.append(event)
            if self.after_write is not None:
                await self.after_write.wait()
            if self.auto_ack:
                self.ack(event)
        return True

    async def send_event(self, event):
        self.events.append(event)

    def ack(self, event):
        if event["type"] == "conversation.item.create":
            return self.coordinator.handle_provider_event({
                "type": "conversation.item.done", "item": {"id": event["item"]["id"]},
            })
        return self.coordinator.handle_provider_event({
            "type": "conversation.item.deleted", "item_id": event["item_id"],
        })


class VisualDeliveryTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.clock = _Clock()
        self.key = VisualConnectionKey("owner", uuid4(), uuid4())
        self.stream = uuid4()
        self.window = RecentVisualFrameWindow(
            max_frames_per_session=4, ttl_seconds=10, monotonic=self.clock.monotonic,
        )
        self.stream_lease = self.window.begin_stream("owner", self.key.session_id, self.stream)
        self.state = VisualContextState(
            connection=self.key,
            policy=VisualDeliveryPolicy(min_interval_seconds=0.01, sync_timeout_seconds=0.04),
        )
        self.provider = _Provider()
        self.coordinator = RealtimeVisualCoordinator(
            state=self.state, frame_window=self.window, provider=self.provider,
            max_image_bytes=512_000, now=self.clock.now, monotonic=self.clock.monotonic,
            ack_timeout_seconds=0.06, write_timeout_seconds=0.02,
        )
        self.provider.coordinator = self.coordinator

    async def asyncTearDown(self):
        await self.coordinator.close()

    def frame(self, sequence, *, stream=None, owner="owner", session=None):
        now = self.clock.now()
        return SensoryVisualFrame(
            frame_id=uuid4(), owner_id=owner, session_id=session or self.key.session_id,
            audio_session_id="audio", visual_session_id=stream or self.stream,
            frame_sequence=sequence, source="glasses", captured_at=now,
            server_received_at=now, content_type="image/jpeg", width=640, height=360,
            storage_bucket="private", storage_path="private-secret-path",
            raw_image_expires_at=now + timedelta(minutes=5),
            image_bytes=("synthetic-pixels-" + str(sequence)).encode(),
        )

    def offer(self, sequence):
        self.clock.offset += 0.02
        frame = self.frame(sequence)
        self.window.put(frame, stream_lease=self.stream_lease)
        self.coordinator.notify_frame_available()
        return frame

    async def cycle(self):
        self.coordinator._tick()
        task = self.coordinator._write_task
        if task is not None:
            await asyncio.wait_for(asyncio.shield(task), timeout=1)

    def creates(self):
        return [event for event in self.provider.events if event["type"] == "conversation.item.create"]

    async def wait_until(self, predicate):
        async with asyncio.timeout(1):
            while not predicate():
                await asyncio.sleep(0.001)

    async def test_silence_delivers_pixels_without_model_analysis_or_speech(self):
        frame = self.offer(1)
        self.coordinator.start()
        await self.wait_until(lambda: self.state.latest_acknowledged is not None)
        event = self.creates()[0]
        self.assertEqual(event["item"]["role"], "user")
        parts = event["item"]["content"]
        self.assertEqual(parts[1]["type"], "input_image")
        self.assertTrue(parts[1]["image_url"].startswith("data:image/jpeg;base64,"))
        self.assertIn(str(frame.frame_id), parts[0]["text"])
        self.assertIn("not a user utterance", parts[0]["text"])
        self.assertNotIn("private-secret-path", json.dumps(event))
        self.assertFalse(any(event["type"] == "response.create" for event in self.provider.events))

    async def test_duplicate_missing_and_out_of_order_hints_use_latest_window(self):
        first = self.offer(1)
        latest = self.offer(3)
        # A delayed older frame/hint cannot replace the latest observation.
        self.window.put(first, stream_lease=self.stream_lease)
        await self.cycle()
        for _ in range(3):
            self.coordinator.notify_frame_available()
            await self.cycle()
        self.assertEqual(len(self.creates()), 1)
        self.assertEqual(self.state.latest_acknowledged.frame_id, latest.frame_id)
        self.assertNotEqual(first.frame_id, latest.frame_id)

    async def test_queued_create_coalesces_before_building_image_payload(self):
        release = asyncio.Event()
        self.provider.before_write = release
        self.offer(1)
        self.coordinator._tick()
        await asyncio.sleep(0)
        latest = self.offer(3)
        release.set()
        await self.cycle()
        self.assertEqual(self.creates(), [])
        self.clock.offset += 0.3
        await self.cycle()
        self.assertEqual(self.state.latest_acknowledged.frame_id, latest.frame_id)
        self.assertEqual(len(self.creates()), 1)

    async def test_timeout_before_write_reclaims_slot_but_after_write_retains_it(self):
        self.provider.before_write = asyncio.Event()
        self.offer(1)
        await self.cycle()
        self.assertEqual(self.state.items, ())
        self.provider.before_write = None
        self.provider.after_write = asyncio.Event()
        self.clock.offset += 0.3
        await self.cycle()
        self.assertEqual(len(self.state.items), 1)
        self.assertEqual(self.state.items[0].status, ProviderImageStatus.CREATE_UNKNOWN)
        original = self.creates()[0]
        self.offer(2)
        await self.cycle()
        self.assertEqual(len(self.creates()), 1)
        self.provider.after_write = None
        self.provider.ack(original)
        await self.cycle()
        self.assertEqual(len(self.creates()), 2)

    async def test_create_timeout_and_late_ack_keep_original_identity(self):
        self.provider.auto_ack = False
        self.offer(1)
        await self.cycle()
        original = self.creates()[0]
        self.clock.offset += 0.1
        self.offer(2)
        await self.cycle()
        self.assertEqual(len(self.creates()), 1)
        self.assertEqual(self.state.items[0].status, ProviderImageStatus.CREATE_UNKNOWN)
        self.provider.ack(original)
        self.provider.auto_ack = True
        await self.cycle()
        self.assertEqual(len(self.creates()), 2)
        self.assertEqual(self.state.latest_acknowledged.frame_sequence, 2)

    async def test_partial_added_event_does_not_claim_full_acceptance(self):
        self.provider.auto_ack = False
        self.offer(1)
        await self.cycle()
        item_id = self.creates()[0]["item"]["id"]
        self.assertTrue(self.coordinator.handle_provider_event({
            "type": "conversation.item.added", "item": {"id": item_id},
        }))
        self.assertIsNone(self.state.latest_acknowledged)
        self.provider.ack(self.creates()[0])
        self.assertEqual(self.state.latest_acknowledged.frame_sequence, 1)

    async def test_delete_timeout_retains_capacity_and_not_found_requires_known_create(self):
        for sequence in (1, 2, 3):
            self.offer(sequence)
            await self.cycle()
        self.provider.auto_ack = False
        await self.cycle()
        deletion = self.provider.events[-1]
        self.assertEqual(deletion["type"], "conversation.item.delete")
        count = len(self.state.items)
        self.clock.offset += 0.1
        await self.cycle()
        self.assertEqual(len(self.state.items), count)
        self.assertTrue(any(item.status == ProviderImageStatus.DELETE_UNKNOWN for item in self.state.items))
        self.coordinator.handle_provider_event({
            "type": "error", "error": {
                "event_id": deletion["event_id"], "code": "item_not_found",
            },
        })
        self.assertEqual(len(self.state.items), count - 1)

    async def test_unknown_create_is_not_freed_by_speculative_delete_ack(self):
        self.provider.auto_ack = False
        self.offer(1)
        await self.cycle()
        item_id = self.creates()[0]["item"]["id"]
        self.coordinator.handle_provider_event({
            "type": "conversation.item.deleted", "item_id": item_id,
        })
        self.assertEqual(len(self.state.items), 1)
        self.assertEqual(self.state.items[0].status, ProviderImageStatus.SUBMITTED)

    async def test_silence_expiry_retires_provider_images(self):
        self.offer(1)
        await self.cycle()
        self.clock.offset += 6
        await self.cycle()
        self.assertEqual(self.state.items, ())
        self.assertNotEqual(
            self.state.availability(now=self.clock.now(), monotonic_now=self.clock.monotonic()).status,
            VisualAvailabilityStatus.FRESH,
        )

    async def test_restarts_reject_old_frames_and_old_close_cannot_clear_new_stream(self):
        old_lease = self.stream_lease
        old_frame = self.offer(1)
        await self.cycle()
        self.stream = uuid4()
        self.stream_lease = self.window.begin_stream("owner", self.key.session_id, self.stream)
        new_frame = self.offer(1)
        self.assertFalse(self.window.put(old_frame, stream_lease=old_lease))
        self.window.end_stream(old_lease)
        self.assertEqual(self.window.latest("owner", self.key.session_id).frame_id, new_frame.frame_id)
        await self.cycle()  # Retire the old provider stream first.
        await self.cycle()
        self.assertEqual(self.state.latest_acknowledged.visual_session_id, self.stream)
        self.window.end_stream(self.stream_lease)
        await self.cycle()
        self.assertIsNone(self.state.visual_session_id)
        self.assertEqual(self.state.items, ())

    async def test_same_stream_reconnect_is_not_invalidated_by_old_socket(self):
        old_lease = self.stream_lease
        self.offer(1)
        await self.cycle()
        self.stream_lease = self.window.begin_stream(
            "owner", self.key.session_id, self.stream,
        )
        self.assertNotEqual(old_lease.connection_id, self.stream_lease.connection_id)
        self.assertFalse(self.window.owns_stream(old_lease))
        self.assertFalse(self.window.put(self.frame(2), stream_lease=old_lease))

        replacement_frame = self.offer(2)
        self.window.end_stream(old_lease)
        self.window.end_stream(old_lease)  # Repeated late cleanup is harmless.
        self.assertTrue(self.window.owns_stream(self.stream_lease))
        self.assertEqual(
            self.window.latest("owner", self.key.session_id).frame_id,
            replacement_frame.frame_id,
        )
        await self.cycle()
        self.assertEqual(self.state.latest_acknowledged.frame_id, replacement_frame.frame_id)

        self.window.end_stream(self.stream_lease)
        self.assertFalse(self.window.put(self.frame(3), stream_lease=self.stream_lease))
        self.assertIsNone(self.window.latest("owner", self.key.session_id))
        await self.cycle()
        await self.cycle()
        self.assertIsNone(self.state.visual_session_id)
        self.assertEqual(self.state.items, ())

    async def test_stream_lease_cannot_write_into_another_owner_or_session(self):
        self.assertFalse(self.window.put(self.frame(1)))
        self.assertFalse(self.window.put(
            self.frame(1, owner="other-owner"), stream_lease=self.stream_lease,
        ))
        self.assertFalse(self.window.put(
            self.frame(1, session=uuid4()), stream_lease=self.stream_lease,
        ))
        self.assertFalse(self.window.put(
            self.frame(1, stream=uuid4()), stream_lease=self.stream_lease,
        ))
        self.assertTrue(self.window.put(self.frame(1), stream_lease=self.stream_lease))

    async def test_retired_connection_ignores_late_ack_and_reconnect_uses_fresh_scope(self):
        self.provider.auto_ack = False
        self.offer(1)
        await self.cycle()
        old_event = self.creates()[0]
        await self.coordinator.close()
        self.assertFalse(self.provider.ack(old_event))
        self.assertEqual(self.state.items, ())
        other_state = VisualContextState(
            connection=VisualConnectionKey("other-owner", self.key.session_id, uuid4()),
            policy=VisualDeliveryPolicy(),
        )
        other = RealtimeVisualCoordinator(
            state=other_state, frame_window=self.window, provider=self.provider,
            max_image_bytes=512_000, now=self.clock.now, monotonic=self.clock.monotonic,
        )
        try:
            other._tick()
            self.assertEqual(other_state.items, ())
            self.assertFalse(other.handle_provider_event({
                "type": "conversation.item.done", "item": {"id": old_event["item"]["id"]},
            }))
        finally:
            await other.close()

    async def test_rejection_is_redacted_and_disables_images_without_disabling_audio(self):
        self.provider.auto_ack = False
        self.offer(1)
        await self.cycle()
        event = self.creates()[0]
        with self.assertLogs("app.realtime.visual_delivery", level="WARNING") as logs:
            self.assertTrue(self.coordinator.handle_provider_event({
                "type": "error", "error": {
                    "event_id": event["event_id"], "code": "invalid_value",
                    "message": "PRIVATE_KEY_AND_BASE64",
                },
            }))
        self.offer(2)
        await self.cycle()
        await self.provider.send_event({"type": "input_audio_buffer.append", "audio": "test"})
        self.assertEqual(len(self.creates()), 1)
        self.assertEqual(self.provider.events[-1]["type"], "input_audio_buffer.append")
        serialized = json.dumps(self.coordinator.diagnostics()) + repr(logs.output)
        self.assertNotIn("PRIVATE_KEY_AND_BASE64", serialized)
        self.assertNotIn("synthetic-pixels", serialized)
        self.assertNotIn("private-secret-path", serialized)

    async def test_vision_departure_and_late_delete_error_keep_audio_usable(self):
        self.offer(1)
        await self.cycle()
        self.provider.auto_ack = False
        self.window.end_stream(self.stream_lease)
        await self.cycle()
        deletion = next(
            event for event in self.provider.events
            if event["type"] == "conversation.item.delete"
        )
        error = {
            "type": "error", "error": {
                "event_id": deletion["event_id"], "code": "invalid_request_error",
                "message": "PRIVATE_KEY_AND_IMAGE_BYTES",
            },
        }
        self.assertTrue(self.coordinator.handle_provider_event(error))
        self.assertNotEqual(
            self.state.availability(now=self.clock.now(), monotonic_now=self.clock.monotonic()).status,
            VisualAvailabilityStatus.FRESH,
        )
        self.assertTrue(self.provider.ack(deletion))
        self.assertEqual(self.state.items, ())
        # A late duplicate belongs to this retired image, not a spoken response.
        self.assertTrue(self.coordinator.handle_provider_event(error))
        await self.provider.send_event({"type": "input_audio_buffer.append", "audio": "synthetic"})
        self.assertEqual(self.provider.events[-1]["type"], "input_audio_buffer.append")
        self.assertFalse(any(event["type"] == "response.create" for event in self.provider.events))
        self.assertNotIn("PRIVATE_KEY_AND_IMAGE_BYTES", json.dumps(self.coordinator.diagnostics()))
        await self.coordinator.close()
        self.assertFalse(self.provider.ack(deletion))
        self.assertFalse(self.coordinator.handle_provider_event(error))
        self.assertEqual(self.state.items, ())

    async def test_prepared_generation_fences_mutation_until_correlated_response_created(self):
        for sequence in (1, 2):
            self.offer(sequence)
            await self.cycle()
        snapshot = await self.coordinator.prepare_response(turn_id="turn", generation_id=uuid4())
        pins = snapshot.pinned_item_ids
        self.assertEqual(len(pins), 2)
        self.assertTrue(snapshot.synchronized)
        self.assertTrue(self.coordinator.mark_response_submitted(snapshot, request_id="request"))
        self.offer(3)
        await self.cycle()
        self.assertEqual(len(self.creates()), 2)
        self.assertFalse(self.coordinator.response_created(request_id="old", response_id="old"))
        self.assertTrue(self.coordinator.response_created(request_id="request", response_id="response"))
        await self.cycle()
        for sequence in range(4, 15):
            self.offer(sequence)
            await self.cycle()
            await self.cycle()
            self.assertLessEqual(len(self.state.items), 4)
            self.assertTrue(pins.issubset({item.item_id for item in self.state.items}))
        self.assertEqual(self.state.latest_acknowledged.frame_sequence, 14)
        self.assertTrue(self.coordinator.settle_response(request_id="request", response_id="response"))
        await self.cycle()
        await self.cycle()
        newer = await self.coordinator.prepare_response(turn_id="next", generation_id=uuid4())
        self.assertLessEqual(len(newer.acknowledged_items), 2)
        self.assertTrue(newer.synchronized)
        self.assertFalse(self.coordinator.mark_response_submitted(snapshot, request_id="reused"))
        self.assertTrue(self.coordinator.cancel_preparation(newer))

    async def test_shared_preparation_deadline_degrades_unknown_images_without_new_response(self):
        self.provider.auto_ack = False
        self.offer(1)
        await self.cycle()
        started = time.monotonic()
        snapshot = await self.coordinator.prepare_response(turn_id="turn", generation_id=uuid4())
        self.assertLess(time.monotonic() - started, 0.15)
        self.assertFalse(snapshot.synchronized)
        self.assertEqual(len(snapshot.possibly_present_items), 1)
        self.assertIsNone(snapshot.current_frame)
        self.assertFalse(any(event["type"] == "response.create" for event in self.provider.events))
        self.assertTrue(self.coordinator.mark_response_submitted(snapshot, request_id="request"))
        self.assertFalse(self.coordinator.cancel_preparation(snapshot))
        self.clock.offset += 1
        self.offer(2)
        await self.cycle()
        self.assertEqual(len(self.creates()), 1)
        self.assertTrue(self.coordinator.settle_response(request_id="request"))

    async def test_preparation_cancellation_before_submission_preserves_unknown_create(self):
        self.provider.auto_ack = False
        self.offer(1)
        await self.cycle()
        task = asyncio.create_task(self.coordinator.prepare_response(
            turn_id="cancelled", generation_id=uuid4(),
        ))
        await asyncio.sleep(0)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertIsNone(self.state.active_snapshot)
        self.assertFalse(self.coordinator.diagnostics()["image_fenced"])
        self.assertEqual(len(self.state.items), 1)

    async def test_close_wakes_preparation_without_resurrecting_epoch(self):
        self.provider.auto_ack = False
        self.offer(1)
        await self.cycle()
        task = asyncio.create_task(self.coordinator.prepare_response(
            turn_id="closing", generation_id=uuid4(),
        ))
        await asyncio.sleep(0)
        await self.coordinator.close()
        with self.assertRaises(VisualContextNotReadyError):
            await asyncio.wait_for(task, timeout=1)
        self.assertEqual(self.state.items, ())

    async def test_publish_only_wakes_matching_session_while_provider_is_blocked(self):
        manager = RealtimeSessionManager()
        notify = Mock(side_effect=self.coordinator.notify_frame_available)
        await manager.register(
            audio_session_id="audio", owner_id="owner", jarvis_session_id=self.key.session_id,
            session=SimpleNamespace(notify_visual_frame_available=notify),
        )
        bus = DomainEventBus()
        bus.subscribe(DomainEventName.VISUAL_FRAME_ACCEPTED, manager.handle_visual_frame_accepted)
        self.provider.before_write = asyncio.Event()
        frame = self.offer(1)
        self.coordinator._tick()
        for owner, session in (("wrong", self.key.session_id), ("owner", uuid4()), ("owner", self.key.session_id)):
            await asyncio.wait_for(bus.publish(DomainEvent(
                name=DomainEventName.VISUAL_FRAME_ACCEPTED,
                payload=VisualFrameAcceptedPayload(
                    owner_id=owner, session_id=session, frame_id=frame.frame_id, frame_sequence=1,
                ),
            )), timeout=0.1)
        notify.assert_called_once()
        self.assertEqual(self.creates(), [])


class VisualWriterTests(unittest.IsolatedAsyncioTestCase):
    async def test_foreground_audio_order_wins_over_queued_image(self):
        provider = OpenAIRealtimeConnection(settings=JarvisSettings(_env_file=None), owner_id="owner")
        entered, release = asyncio.Event(), asyncio.Event()
        sent = []

        async def send(raw):
            event = json.loads(raw)
            sent.append(event["type"])
            if event["type"] == "audio-1":
                entered.set()
                await release.wait()

        provider._websocket = SimpleNamespace(send=send)
        first = asyncio.create_task(provider.send_event({"type": "audio-1"}))
        await entered.wait()
        factory = Mock(return_value={"type": "image"})
        image = asyncio.create_task(provider.send_visual_event(factory, timeout_seconds=1))
        await asyncio.sleep(0)
        second = asyncio.create_task(provider.send_event({"type": "audio-2"}))
        await asyncio.sleep(0)
        release.set()
        self.assertEqual(await asyncio.gather(first, image, second), [True, False, True])
        self.assertEqual(sent, ["audio-1", "audio-2"])
        factory.assert_not_called()

    async def test_bounded_image_write_releases_writer_for_control(self):
        provider = OpenAIRealtimeConnection(settings=JarvisSettings(_env_file=None), owner_id="owner")
        sent = []

        async def send(raw):
            event = json.loads(raw)
            sent.append(event["type"])
            if event["type"] == "image":
                await asyncio.Event().wait()

        provider._websocket = SimpleNamespace(send=send)
        with self.assertRaises(TimeoutError):
            await provider.send_visual_event(lambda: {"type": "image"}, timeout_seconds=0.01)
        await asyncio.wait_for(provider.send_event({"type": "response.cancel"}), timeout=1)
        self.assertEqual(sent, ["image", "response.cancel"])


if __name__ == "__main__":
    unittest.main()
