from __future__ import annotations

import asyncio
import base64
import json
import unittest
from dataclasses import replace
from datetime import timedelta
from types import SimpleNamespace
from uuid import uuid4

from app.images import ImageReference, ImageUnavailableError, ResolvedImage
from app.perception.current_image import CurrentImageService
from app.perception.frame_window import RecentVisualFrameWindow, SensoryVisualFrame
from app.realtime.openai_client import OpenAIRealtimeConnection, build_requested_image_event
from app.realtime.visual_context import (
    ProviderImageStatus,
    RequestedImageDeliveryError,
    VisualAvailabilityStatus,
    VisualConnectionKey,
    VisualContextNotReadyError,
    VisualContextState,
    VisualDeliveryPolicy,
    VisualFrameMetadata,
)
from app.realtime.visual_delivery import RealtimeVisualCoordinator
from test_current_image_service import _Clock, _jpeg


class _Images:
    def __init__(self, *, enabled=True, capacity=4):
        self.clock = _Clock()
        self.key = VisualConnectionKey("owner", uuid4(), uuid4())
        self.window = RecentVisualFrameWindow(
            max_frames_per_session=8, ttl_seconds=30,
            now=self.clock.now, monotonic=self.clock.monotonic,
        )
        self.lease = self.window.begin_stream(self.key.owner_id, self.key.session_id, uuid4())
        self.state = VisualContextState(
            connection=self.key, visual_session_id=self.lease.visual_session_id,
            policy=VisualDeliveryPolicy(
                enabled=enabled, min_interval_seconds=0.01,
                sync_timeout_seconds=0.1, max_image_items=capacity,
            ),
        )
        self.reader = CurrentImageService(
            frame_window=self.window, max_age_seconds=5, max_image_bytes=4096,
            max_width=1024, max_height=1024,
            now=self.clock.now, monotonic=self.clock.monotonic,
        )

    def current(self, sequence=1, **changes):
        now = self.clock.now()
        frame = SensoryVisualFrame(
            frame_id=uuid4(), owner_id=self.key.owner_id, session_id=self.key.session_id,
            audio_session_id="audio", visual_session_id=self.lease.visual_session_id,
            frame_sequence=sequence, source="glasses", captured_at=now,
            server_received_at=now, content_type="image/jpeg", width=48, height=32,
            storage_bucket="private-bucket", storage_path="private-image-path",
            raw_image_expires_at=now + timedelta(seconds=60), image_bytes=_jpeg(),
        )
        frame = replace(frame, **changes)
        if not self.window.put(frame, stream_lease=self.lease):
            raise AssertionError("Fixture frame was not admitted.")
        reference = self.reader.select_current_image(
            owner_id=self.key.owner_id, session_id=self.key.session_id,
        )
        return self.reader.resolve_current_image(
            owner_id=self.key.owner_id, session_id=self.key.session_id, reference=reference,
        )

    def historical(self, **changes):
        reference = ImageReference(
            frame_id=uuid4(), owner_id=self.key.owner_id, session_id=uuid4(),
            source="historical", captured_at=self.clock.now() - timedelta(days=1),
            content_type="image/jpeg", width=48, height=32,
            raw_image_expires_at=self.clock.now() + timedelta(seconds=60),
            record_type="experience_event", record_id=uuid4(),
        )
        return ResolvedImage(reference=replace(reference, **changes), image_bytes=_jpeg())

    def metadata(self, image):
        sample = self.window.get_sample(
            self.key.owner_id, self.key.session_id, frame_id=image.reference.frame_id,
        )
        if sample is None:
            raise AssertionError("Fixture frame is unavailable.")
        return VisualFrameMetadata.from_sensory_frame(
            sample.frame, connection=self.key,
            sampled_at=sample.sampled_at, sampled_monotonic=sample.sampled_monotonic,
        )

    def reserve(self, image, call_id="call", generation=7):
        return self.state.reserve_requested_create(
            reference=image.reference,
            current_frame=self.metadata(image) if image.reference.source == "current" else None,
            provider_call_id=call_id, turn_generation=generation,
            now=self.clock.now(), monotonic_now=self.clock.monotonic(),
        )

    def acknowledge(self, item):
        if not self.state.mark_submitted(
            item_id=item.item_id, now=self.clock.now(), monotonic_now=self.clock.monotonic(),
        ):
            raise AssertionError("Fixture image was not submitted.")
        if not self.state.acknowledge_create(
            connection_epoch=self.key.connection_epoch,
            item_id=item.item_id, acknowledged_at=self.clock.now(),
        ):
            raise AssertionError("Fixture image was not acknowledged.")
        return next(entry for entry in self.state.items if entry.item_id == item.item_id)

    def passive(self, sequence=1):
        image = self.current(sequence)
        self.state.offer_frame(self.metadata(image))
        item = self.state.reserve_create(now=self.clock.now(), monotonic_now=self.clock.monotonic())
        return image, self.acknowledge(item)

    def prepare(self, calls=(), *, turn="7", generation=None, degraded=False):
        return self.state.prepare_response(
            turn_id=turn, generation_id=generation or uuid4(),
            now=self.clock.now(), monotonic_now=self.clock.monotonic(),
            requested_call_ids=calls, allow_degraded=degraded,
        )


class RequestedImageStateTests(unittest.TestCase):
    def setUp(self):
        self.images = _Images()
        self.state = self.images.state

    def test_reservation_and_submission_are_not_acknowledgements(self):
        item = self.images.reserve(self.images.current())
        self.assertEqual(item.status, ProviderImageStatus.RESERVED)
        self.assertIn(item.item_id, self.state.pinned_item_ids)
        self.assertTrue(self.state.mark_submitted(
            item_id=item.item_id, now=self.images.clock.now(), monotonic_now=self.images.clock.monotonic(),
        ))
        self.assertEqual(self.state.requested_item("call").status, ProviderImageStatus.SUBMITTED)
        with self.assertRaises(VisualContextNotReadyError):
            self.images.prepare(("call",))

    def test_history_never_replaces_live_identity_or_freshness(self):
        _, live = self.images.passive()
        before = (self.state.latest_local, self.state.latest_submitted, self.state.latest_acknowledged)
        available = self.state.availability(now=self.images.clock.now(), monotonic_now=self.images.clock.monotonic())
        self.images.acknowledge(self.images.reserve(self.images.historical()))
        self.assertEqual(before, (self.state.latest_local, self.state.latest_submitted, self.state.latest_acknowledged))
        self.assertEqual(
            self.state.availability(now=self.images.clock.now(), monotonic_now=self.images.clock.monotonic()), available,
        )
        self.assertEqual(self.state.latest_acknowledged, live.frame)

    def test_exact_current_image_reuses_acknowledged_passive_item(self):
        image, passive = self.images.passive()
        requested = self.images.reserve(image)
        self.assertEqual(requested.item_id, passive.item_id)
        self.assertEqual(requested.delivery_origin, "passive")
        self.assertEqual(len(self.state.items), 1)
        self.assertIsNone(self.state.begin_delete(
            connection_epoch=self.images.key.connection_epoch, item_id=passive.item_id,
        ))

    def test_same_pixels_requested_as_history_get_a_distinct_temporal_label(self):
        current, passive = self.images.passive()
        historical = replace(current, reference=replace(current.reference, source="historical"))
        item = self.images.acknowledge(self.images.reserve(historical))
        self.assertNotEqual(item.item_id, passive.item_id)
        self.assertTrue(item.is_historical)
        self.assertEqual(self.state.latest_acknowledged, passive.frame)

    def test_repeated_call_identity_reuses_reservation_and_rejects_conflicts(self):
        image = self.images.historical()
        reserved = self.images.reserve(image)
        self.assertEqual(self.images.reserve(image).item_id, reserved.item_id)
        for candidate, generation in ((self.images.historical(), 7), (image, 8)):
            with self.subTest(generation=generation), self.assertRaises(RequestedImageDeliveryError) as caught:
                self.images.reserve(candidate, generation=generation)
            self.assertEqual(caught.exception.code, "request_conflict")
        self.assertEqual(len(self.state.items), 1)

    def test_foreign_owner_cannot_reserve_historical_evidence(self):
        with self.assertRaises(RequestedImageDeliveryError) as caught:
            self.images.reserve(self.images.historical(owner_id="other-owner"))
        self.assertEqual(caught.exception.code, "frame_unavailable")
        self.assertEqual(self.state.items, ())

    def test_pending_delete_does_not_free_a_slot_for_requested_images(self):
        _, item = self.images.passive()
        request_id = self.state.begin_delete(
            connection_epoch=self.images.key.connection_epoch, item_id=item.item_id,
        )
        self.assertIsNotNone(request_id)
        self.assertIsNone(self.images.reserve(self.images.historical()))
        self.assertEqual(self.state.free_slots, 3)

    def test_all_pinned_slots_remain_unavailable(self):
        for sequence in range(1, 5):
            self.images.passive(sequence)
        snapshot = self.images.prepare(degraded=True)
        self.assertEqual(len(snapshot.pinned_item_ids), 4)
        self.assertIsNone(self.images.reserve(self.images.historical()))
        self.assertEqual(len(self.state.items), 4)

    def test_unknown_historical_create_occupies_capacity_without_changing_live_age(self):
        _, live = self.images.passive()
        item = self.images.reserve(self.images.historical())
        self.state.mark_submitted(item_id=item.item_id, now=self.images.clock.now(), monotonic_now=100)
        self.state.mark_create_timeout(connection_epoch=self.images.key.connection_epoch, item_id=item.item_id)
        self.assertEqual(self.state.free_slots, 2)
        self.assertEqual(self.state.latest_acknowledged, live.frame)
        self.assertEqual(
            self.state.availability(now=self.images.clock.now(), monotonic_now=100).status,
            VisualAvailabilityStatus.FRESH,
        )

    def test_wrong_epoch_and_duplicate_acknowledgements_do_not_change_request(self):
        item = self.images.reserve(self.images.historical())
        self.state.mark_submitted(item_id=item.item_id, now=self.images.clock.now(), monotonic_now=100)
        self.assertFalse(self.state.acknowledge_create(
            connection_epoch=uuid4(), item_id=item.item_id, acknowledged_at=self.images.clock.now(),
        ))
        self.assertTrue(self.state.acknowledge_create(
            connection_epoch=self.images.key.connection_epoch,
            item_id=item.item_id, acknowledged_at=self.images.clock.now(),
        ))
        self.assertFalse(self.state.acknowledge_create(
            connection_epoch=self.images.key.connection_epoch,
            item_id=item.item_id, acknowledged_at=self.images.clock.now(),
        ))

    def test_pins_transfer_to_correlated_response_and_survive_until_settlement(self):
        _, live = self.images.passive()
        historical = self.images.acknowledge(self.images.reserve(self.images.historical()))
        snapshot = self.images.prepare(("call",))
        self.assertTrue(snapshot.synchronized)
        self.assertEqual(snapshot.current_frame, live.frame)
        self.assertEqual(snapshot.requested_images[0].item_id, historical.item_id)
        self.assertIsNone(self.state.requested_pin("call"))
        self.assertIn(historical.item_id, self.state.pinned_item_ids)
        self.assertIsNone(self.state.begin_delete(
            connection_epoch=self.images.key.connection_epoch, item_id=historical.item_id,
        ))
        self.assertTrue(self.state.settle_response(
            connection_epoch=self.images.key.connection_epoch, generation_id=snapshot.generation_id,
        ))
        self.assertIn(historical.item_id, self.state.retirable_item_ids(now=self.images.clock.now(), monotonic_now=100))

    def test_unrelated_response_cannot_take_requested_image_pins(self):
        self.images.acknowledge(self.images.reserve(self.images.historical()))
        for calls, turn in (((), "7"), (("call",), "8"), (("wrong",), "7")):
            with self.subTest(calls=calls, turn=turn), self.assertRaises(VisualContextNotReadyError):
                self.images.prepare(calls, turn=turn)
        self.assertIsNotNone(self.state.requested_pin("call"))

    def test_cancellation_is_generation_scoped(self):
        item = self.images.reserve(self.images.historical())
        self.assertFalse(self.state.release_requested_image(provider_call_id="call", turn_generation=8))
        self.assertIn(item.item_id, self.state.pinned_item_ids)
        self.assertTrue(self.state.release_requested_image(provider_call_id="call", turn_generation=7))
        self.assertEqual(self.state.items, ())

    def test_expired_history_cannot_enter_an_answer_snapshot(self):
        self.images.acknowledge(self.images.reserve(self.images.historical()))
        self.images.clock.advance(60)
        with self.assertRaises(VisualContextNotReadyError):
            self.images.prepare(("call",))

    def test_larger_capacity_allows_explicit_inputs_without_enlarging_default_budget(self):
        images = _Images(capacity=6)
        for index in range(3):
            images.acknowledge(images.reserve(images.historical(), call_id=f"call-{index}"))
        snapshot = images.prepare(tuple(f"call-{index}" for index in range(3)))
        self.assertTrue(snapshot.synchronized)
        self.assertEqual(len(snapshot.acknowledged_items), 3)
        self.assertIsNone(snapshot.current_frame)

    def test_response_metadata_labels_history_without_media_or_storage_locators(self):
        self.images.acknowledge(self.images.reserve(self.images.historical()))
        snapshot = self.images.prepare(("call",))
        metadata = snapshot.to_metadata()
        self.assertEqual(len(metadata["historical_images"]), 1)
        self.assertEqual(metadata["requested_images"][0]["image"]["source"], "historical")
        serialized = json.dumps(metadata)
        for forbidden in ("image_bytes", "image_url", "storage_path", "private-image-path", "base64"):
            self.assertNotIn(forbidden, serialized)

    def test_requested_payload_contains_exact_pixels_not_a_new_spoken_request(self):
        image = self.images.historical()
        item = self.images.reserve(image)
        event = build_requested_image_event(item, image)
        self.assertEqual(event["item"]["role"], "user")
        text, pixels = event["item"]["content"]
        self.assertIn("NOT the current camera view", text["text"])
        self.assertIn("not a new user utterance", text["text"])
        self.assertIn("untrusted scene content", text["text"])
        self.assertEqual(pixels["type"], "input_image")
        self.assertEqual(base64.b64decode(pixels["image_url"].split(",", 1)[1]), image.image_bytes)
        self.assertNotIn("output", event["item"])


class _Wire:
    def __init__(self):
        self.events = []
        self.queue = asyncio.Queue()
        self.send_entered = asyncio.Event()
        self.release_send = asyncio.Event()
        self.release_send.set()
        self.ack_creates = True
        self.ack_deletes = True
        self.coordinator = None

    async def send(self, raw):
        self.send_entered.set()
        await self.release_send.wait()
        event = json.loads(raw)
        self.events.append(event)
        self.queue.put_nowait(event)
        if (event["type"] == "conversation.item.create" and self.ack_creates) or (
            event["type"] == "conversation.item.delete" and self.ack_deletes
        ):
            self.ack(event)

    def ack(self, event):
        if event["type"] == "conversation.item.create":
            return self.coordinator.handle_provider_event({
                "type": "conversation.item.done", "item": {"id": event["item"]["id"]},
            })
        return self.coordinator.handle_provider_event({
            "type": "conversation.item.deleted", "item_id": event["item_id"],
        })

    async def next_event(self, event_type):
        async with asyncio.timeout(1):
            while True:
                event = await self.queue.get()
                if event["type"] == event_type:
                    return event


class _Provider(OpenAIRealtimeConnection):
    def __init__(self, wire):
        super().__init__(
            settings=SimpleNamespace(OPENAI_REALTIME_WRITE_TIMEOUT_SECONDS=10), owner_id="owner",
        )
        self._websocket = wire
        self.image_waiting = asyncio.Event()

    async def send_event(self, event):
        if callable(event):
            self.image_waiting.set()
        return await super().send_event(event)


class RequestedImageDeliveryTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tasks = []
        self.configure()

    def configure(self, *, enabled=True):
        self.images = _Images(enabled=enabled)
        self.state = self.images.state
        self.wire = _Wire()
        self.provider = _Provider(self.wire)
        self.coordinator = RealtimeVisualCoordinator(
            state=self.state, frame_window=self.images.window, provider=self.provider,
            max_image_bytes=4096, now=self.images.clock.now, monotonic=self.images.clock.monotonic,
            ack_timeout_seconds=0.05, write_timeout_seconds=0.04,
        )
        self.wire.coordinator = self.coordinator
        self.active = True
        self.coordinator.start()

    async def asyncTearDown(self):
        await self.coordinator.close()
        for task in self.tasks:
            if not task.done():
                task.cancel()
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)

    def request(self, image, *, call="call", generation=7):
        task = asyncio.create_task(self.coordinator.deliver_requested_image(
            image, provider_call_id=call, turn_generation=generation,
            is_request_active=lambda: self.active,
        ))
        self.tasks.append(task)
        return task

    def creates(self):
        return [event for event in self.wire.events if event["type"] == "conversation.item.create"]

    async def cycle(self):
        self.coordinator._tick()
        task = self.coordinator._write_task
        if task is not None:
            await asyncio.wait_for(asyncio.shield(task), timeout=1)

    async def test_current_image_uses_real_ordered_writer_and_requires_ack(self):
        self.wire.ack_creates = False
        image = self.images.current()
        task = self.request(image)
        event = await self.wire.next_event("conversation.item.create")
        self.assertFalse(task.done())
        self.assertEqual(self.state.requested_item("call").status, ProviderImageStatus.SUBMITTED)
        self.assertTrue(self.wire.ack(event))
        item = await task
        self.assertEqual(item.status, ProviderImageStatus.ACKNOWLEDGED)
        self.assertEqual(item.frame.frame_id, image.reference.frame_id)
        self.assertEqual(len(self.creates()), 1)
        self.assertFalse(any(event["type"] == "response.create" for event in self.wire.events))

    async def test_partial_and_wrong_item_ack_do_not_complete_request(self):
        self.wire.ack_creates = False
        task = self.request(self.images.historical())
        event = await self.wire.next_event("conversation.item.create")
        self.assertTrue(self.coordinator.handle_provider_event({
            "type": "conversation.item.added", "item": {"id": event["item"]["id"]},
        }))
        self.assertFalse(self.coordinator.handle_provider_event({
            "type": "conversation.item.done", "item": {"id": uuid4().hex},
        }))
        self.assertFalse(task.done())
        self.assertEqual(self.state.requested_item("call").status, ProviderImageStatus.SUBMITTED)
        self.wire.ack(event)
        self.assertEqual((await task).status, ProviderImageStatus.ACKNOWLEDGED)

    async def test_audio_progresses_while_requested_image_awaits_ack(self):
        self.wire.ack_creates = False
        task = self.request(self.images.historical())
        event = await self.wire.next_event("conversation.item.create")
        self.assertTrue(await self.provider.send_event({"type": "input_audio_buffer.append", "audio": "synthetic"}))
        self.assertEqual(self.wire.events[-1]["type"], "input_audio_buffer.append")
        self.assertFalse(task.done())
        self.wire.ack(event)
        await task

    async def test_historical_image_works_without_a_current_stream(self):
        self.images.window.end_stream(self.images.lease)
        item = await self.request(self.images.historical())
        self.assertTrue(item.is_historical)
        self.assertIsNone(self.state.latest_acknowledged)
        self.assertIsNone(self.state.visual_session_id)
        self.assertNotEqual(
            self.state.availability(now=self.images.clock.now(), monotonic_now=100).status,
            VisualAvailabilityStatus.FRESH,
        )

    async def test_passive_disabled_still_allows_explicit_current_image(self):
        await self.coordinator.close()
        self.configure(enabled=False)
        self.assertIsNone(self.coordinator._worker)
        item = await self.request(self.images.current())
        self.assertEqual(item.status, ProviderImageStatus.ACKNOWLEDGED)
        self.assertEqual(len(self.creates()), 1)
        self.assertEqual(self.state.latest_acknowledged.frame_id, item.frame.frame_id)

    async def test_existing_passive_image_is_reused_without_another_create(self):
        image = self.images.current()
        self.coordinator.notify_frame_available()
        passive = await self.wire.next_event("conversation.item.create")
        item = await self.request(image)
        self.assertEqual(item.item_id, passive["item"]["id"])
        self.assertEqual(len(self.creates()), 1)

    async def test_duplicate_pending_and_completed_requests_never_duplicate_images(self):
        self.wire.ack_creates = False
        image = self.images.historical()
        first = self.request(image)
        event = await self.wire.next_event("conversation.item.create")
        with self.assertRaises(RequestedImageDeliveryError) as caught:
            await self.request(image)
        self.assertEqual(caught.exception.code, "request_in_progress")
        self.wire.ack(event)
        item = await first
        self.assertEqual((await self.request(image)).item_id, item.item_id)
        self.assertEqual(len(self.creates()), 1)

    async def test_two_concurrent_requests_share_the_existing_inventory(self):
        first, second = self.request(self.images.historical(), call="one"), self.request(self.images.historical(), call="two")
        items = await asyncio.gather(first, second)
        self.assertEqual(len({item.item_id for item in items}), 2)
        self.assertEqual(len(self.state.items), 2)
        self.assertEqual(len(self.state.requested_pins), 2)
        self.assertEqual(self.state.free_slots, 2)

    async def test_full_pinned_inventory_fails_boundedly_without_eviction(self):
        for sequence in range(1, 5):
            self.images.passive(sequence)
        snapshot = self.images.prepare(degraded=True)
        with self.assertRaises(RequestedImageDeliveryError) as caught:
            await self.request(self.images.historical())
        self.assertEqual(caught.exception.code, "image_delivery_timeout")
        self.assertEqual(self.state.pinned_item_ids, snapshot.pinned_item_ids)
        self.assertEqual(len(self.state.items), 4)
        self.assertEqual(self.creates(), [])

    async def test_rejection_is_correlated_redacted_and_audio_remains_usable(self):
        self.wire.ack_creates = False
        task = self.request(self.images.historical())
        event = await self.wire.next_event("conversation.item.create")
        with self.assertLogs("app.realtime.visual_delivery", level="WARNING") as logs:
            self.assertTrue(self.coordinator.handle_provider_event({
                "type": "error", "error": {
                    "event_id": event["event_id"], "code": "invalid_value",
                    "message": "PRIVATE_PROVIDER_MESSAGE",
                },
            }))
        with self.assertRaises(RequestedImageDeliveryError) as caught:
            await task
        self.assertEqual(caught.exception.code, "provider_rejected_image_input")
        self.assertEqual(self.state.items, ())
        await self.provider.send_event({"type": "input_audio_buffer.append", "audio": "synthetic"})
        self.assertEqual(self.wire.events[-1]["type"], "input_audio_buffer.append")
        self.assertNotIn("PRIVATE_PROVIDER_MESSAGE", json.dumps(self.coordinator.diagnostics()) + repr(logs.output))

    async def test_timeout_retains_unknown_create_until_late_ack_and_cleanup(self):
        self.wire.ack_creates = False
        task = self.request(self.images.historical())
        event = await self.wire.next_event("conversation.item.create")
        with self.assertRaises(RequestedImageDeliveryError):
            await task
        self.assertEqual(self.state.items[0].status, ProviderImageStatus.CREATE_UNKNOWN)
        self.assertEqual(self.state.requested_pins, ())
        self.wire.ack(event)
        await self.wire.next_event("conversation.item.delete")
        self.assertEqual(self.state.items, ())
        self.assertEqual(len(self.creates()), 1)

    async def test_passive_disabled_late_ack_still_triggers_automatic_cleanup(self):
        await self.coordinator.close()
        self.configure(enabled=False)
        self.wire.ack_creates = False
        task = self.request(self.images.historical())
        event = await self.wire.next_event("conversation.item.create")
        with self.assertRaises(RequestedImageDeliveryError):
            await task
        self.assertEqual(self.state.items[0].status, ProviderImageStatus.CREATE_UNKNOWN)
        self.wire.ack(event)
        await self.wire.next_event("conversation.item.delete")
        self.assertEqual(self.state.items, ())

    async def test_cancellation_before_writer_admission_never_sends_pixels(self):
        await self.provider._send_lock.acquire()
        task = self.request(self.images.current())
        try:
            await asyncio.wait_for(self.provider.image_waiting.wait(), timeout=1)
            self.assertTrue(self.coordinator.cancel_requested_image(provider_call_id="call", turn_generation=7))
            with self.assertRaises(asyncio.CancelledError):
                await task
        finally:
            self.provider._send_lock.release()
        self.assertEqual(self.creates(), [])
        self.assertEqual(self.state.requested_pins, ())
        self.assertFalse(any(item.delivery_origin == "requested" for item in self.state.items))

    async def test_guard_revocation_before_write_cancels_request(self):
        await self.provider._send_lock.acquire()
        task = self.request(self.images.historical())
        try:
            await asyncio.wait_for(self.provider.image_waiting.wait(), timeout=1)
            self.active = False
            self.coordinator._tick()
            with self.assertRaises(asyncio.CancelledError):
                await task
        finally:
            self.provider._send_lock.release()
        self.assertEqual(self.creates(), [])
        self.assertEqual(self.state.requested_pins, ())

    async def test_cancelled_written_image_keeps_unknown_slot_until_late_ack(self):
        self.wire.ack_creates = False
        task = self.request(self.images.historical())
        event = await self.wire.next_event("conversation.item.create")
        self.assertFalse(self.coordinator.cancel_requested_image(provider_call_id="call", turn_generation=8))
        self.assertTrue(self.coordinator.cancel_requested_image(provider_call_id="call", turn_generation=7))
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(self.state.items[0].status, ProviderImageStatus.CREATE_UNKNOWN)
        self.wire.ack(event)
        await self.wire.next_event("conversation.item.delete")
        self.assertEqual(self.state.items, ())

    async def test_close_cancels_waiter_and_rejects_late_ack(self):
        self.wire.ack_creates = False
        task = self.request(self.images.historical())
        event = await self.wire.next_event("conversation.item.create")
        await self.coordinator.close()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(self.state.items, ())
        self.assertFalse(self.wire.ack(event))

    async def test_stream_replacement_before_write_prevents_old_current_image(self):
        await self.provider._send_lock.acquire()
        task = self.request(self.images.current())
        try:
            await asyncio.wait_for(self.provider.image_waiting.wait(), timeout=1)
            self.images.lease = self.images.window.begin_stream(
                self.images.key.owner_id, self.images.key.session_id, self.images.lease.visual_session_id,
            )
        finally:
            self.provider._send_lock.release()
        with self.assertRaises((ImageUnavailableError, RequestedImageDeliveryError, asyncio.CancelledError)):
            await task
        self.assertEqual(self.creates(), [])

    async def test_stream_replacement_does_not_invalidate_authorized_history(self):
        await self.provider._send_lock.acquire()
        task = self.request(self.images.historical())
        try:
            await asyncio.wait_for(self.provider.image_waiting.wait(), timeout=1)
            self.images.lease = self.images.window.begin_stream(self.images.key.owner_id, self.images.key.session_id, uuid4())
        finally:
            self.provider._send_lock.release()
        self.assertTrue((await task).is_historical)
        self.assertIsNone(self.state.latest_acknowledged)

    async def test_selected_current_image_is_not_replaced_by_newer_pixels_at_writer(self):
        await self.provider._send_lock.acquire()
        original = self.images.current()
        task = self.request(original)
        try:
            await asyncio.wait_for(self.provider.image_waiting.wait(), timeout=1)
            self.images.current(2, width=96, image_bytes=_jpeg(96, 32))
        finally:
            self.provider._send_lock.release()
        item = await task
        self.assertEqual(item.frame.frame_id, original.reference.frame_id)
        encoded = self.creates()[0]["item"]["content"][1]["image_url"].split(",", 1)[1]
        self.assertEqual(base64.b64decode(encoded), original.image_bytes)

    async def test_response_transfers_requested_pins_then_settlement_retires_image(self):
        item = await self.request(self.images.historical())
        snapshot = await self.coordinator.prepare_response(
            turn_id="7", generation_id=uuid4(), requested_call_ids=("call",),
        )
        self.assertTrue(snapshot.synchronized)
        self.assertIn(item.item_id, snapshot.pinned_item_ids)
        self.assertIsNone(self.state.requested_pin("call"))
        self.assertTrue(self.coordinator.mark_response_submitted(snapshot, request_id="response-request"))
        self.assertTrue(self.coordinator.response_created(request_id="response-request", response_id="response"))
        self.assertTrue(self.coordinator.settle_response(request_id="response-request", response_id="response"))
        await self.wire.next_event("conversation.item.delete")
        self.assertEqual(self.state.items, ())

    async def test_acknowledged_current_image_can_expire_before_answer_submission(self):
        await self.request(self.images.current())
        snapshot = await self.coordinator.prepare_response(
            turn_id="7", generation_id=uuid4(), requested_call_ids=("call",),
        )
        self.images.clock.advance(6)
        self.assertFalse(self.coordinator.mark_response_submitted(snapshot, request_id="response-request"))
        self.assertTrue(self.coordinator.cancel_preparation(snapshot))

    async def test_pending_delete_ack_continues_to_consume_capacity(self):
        item = await self.request(self.images.historical())
        self.wire.ack_deletes = False
        self.coordinator.cancel_requested_image(provider_call_id="call", turn_generation=7)
        deletion = await self.wire.next_event("conversation.item.delete")
        self.assertEqual(deletion["item_id"], item.item_id)
        self.images.clock.advance(0.06)
        self.coordinator._tick()
        self.assertEqual(self.state.items[0].status, ProviderImageStatus.DELETE_UNKNOWN)
        self.assertEqual(self.state.free_slots, 3)
        self.assertTrue(self.wire.ack(deletion))
        self.assertEqual(self.state.items, ())

    async def test_completed_call_cannot_be_replayed_after_response_pin_transfer(self):
        image = self.images.historical()
        await self.request(image)
        snapshot = await self.coordinator.prepare_response(
            turn_id="7", generation_id=uuid4(), requested_call_ids=("call",),
        )
        with self.assertRaises(RequestedImageDeliveryError) as caught:
            await self.request(image)
        self.assertEqual(caught.exception.code, "request_settled")
        self.assertEqual(len(self.creates()), 1)
        self.coordinator.cancel_preparation(snapshot)

    async def test_invalid_owner_and_oversized_pixels_never_reach_provider(self):
        cases = (
            self.images.historical(owner_id="other-owner"),
            replace(self.images.historical(), image_bytes=b"x" * 4097),
        )
        for index, image in enumerate(cases):
            with self.subTest(index=index), self.assertRaises(ImageUnavailableError):
                await self.request(image, call=f"invalid-{index}")
        self.assertEqual(self.wire.events, [])

    async def test_invalid_jpeg_is_rejected_without_claiming_writer_contention(self):
        image = replace(self.images.historical(), image_bytes=b"not a JPEG")
        with self.assertRaises((ImageUnavailableError, RequestedImageDeliveryError)) as caught:
            await self.request(image)
        self.assertEqual(caught.exception.code, "invalid_image")
        self.assertEqual(self.creates(), [])

    async def test_expired_current_source_is_not_reused_as_fresh(self):
        image = self.images.current()
        self.images.clock.advance(6)
        with self.assertRaises(ImageUnavailableError) as caught:
            await self.request(image)
        self.assertEqual(caught.exception.code, "stale_frame")
        self.assertEqual(self.creates(), [])


if __name__ == "__main__":
    unittest.main()
