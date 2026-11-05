from __future__ import annotations

import asyncio
import json
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

from app.assistant.base_prompt import RESPONSE_STYLE_PROMPT_LINES, build_base_prompt
from app.perception.frame_window import RecentVisualFrameWindow
from app.realtime.openai_client import OpenAIRealtimeConnection
from app.realtime.visual_delivery import RealtimeVisualCoordinator

from visual_frame_fixtures import _frame
from test_generic_visual_speech import GenericVisualSpeechTests
from test_realtime_visual_delivery import _Clock


class _Wire:
    """Fake provider state behind the real ordered OpenAI transport."""

    def __init__(self, session):
        self.session = session
        self.events = []
        self.images = set()
        self.responses = []
        self.incoming = asyncio.Queue()
        self.auto_image_ack = True
        self.auto_delete_ack = True
        self.auto_response_ack = True
        self.before_response = None
        self.close = AsyncMock()

    async def send(self, raw):
        event = json.loads(raw)
        self.events.append(event)
        item = event.get("item", {})
        is_image = any(content.get("type") == "input_image" for content in item.get("content", []))
        if event["type"] == "conversation.item.create" and is_image:
            self.images.add(item["id"])
            if self.auto_image_ack:
                self.ack_image(event)
        elif event["type"] == "conversation.item.delete":
            self.images.discard(event["item_id"])
            if self.auto_delete_ack:
                self.incoming.put_nowait({
                    "type": "conversation.item.deleted", "item_id": event["item_id"],
                })
        elif event["type"] == "response.create":
            if self.before_response is not None:
                self.before_response()
            self.responses.append({
                "event": event,
                "images": frozenset(self.images),
                "cutoff": self.session._visual_context.active_snapshot,
            })
            if self.auto_response_ack:
                self.ack_response(self.responses[-1], f"response-{len(self.responses)}")

    def ack_image(self, event):
        self.incoming.put_nowait({
            "type": "conversation.item.done", "item": {"id": event["item"]["id"]},
        })

    def ack_response(self, recorded, response_id):
        self.incoming.put_nowait({
            "type": "response.created",
            "response": {
                "id": response_id, "metadata": recorded["event"]["response"]["metadata"],
            },
        })


class SharedViewSessionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.fixture = GenericVisualSpeechTests()
        await self.fixture.asyncSetUp()
        self.session = self.fixture.session
        self.clock = _Clock()
        self.session._now = self.clock.now
        self.session._monotonic = self.clock.monotonic
        self.state = self.session._visual_context
        self.state.policy = replace(
            self.state.policy, min_interval_seconds=0.001, sync_timeout_seconds=0.05,
        )
        self.window = RecentVisualFrameWindow(
            max_frames_per_session=8, ttl_seconds=30, monotonic=self.clock.monotonic,
        )
        self.stream = uuid4()
        self.lease = self.window.begin_stream("owner", self.session.context.jarvis_session_id, self.stream)
        self.provider = OpenAIRealtimeConnection(settings=self.session.settings, owner_id="owner")
        self.wire = _Wire(self.session)
        self.provider._websocket = self.wire
        self.session.openai = self.provider
        self.delivery = RealtimeVisualCoordinator(
            state=self.state, frame_window=self.window, provider=self.provider,
            max_image_bytes=512_000, now=self.clock.now, monotonic=self.clock.monotonic,
        )
        self.session._visual_delivery = self.delivery
        self.receiver = asyncio.create_task(self.receive())
        self.delivery.start()

    async def asyncTearDown(self):
        await self.session.close("test_complete")
        self.receiver.cancel()
        await asyncio.gather(self.receiver, return_exceptions=True)

    async def receive(self):
        self.session._provider_receive_task = asyncio.current_task()
        while True:
            event = await self.wire.incoming.get()
            await self.session._handle_openai_event(event)

    async def until(self, predicate):
        async with asyncio.timeout(2):
            while not predicate():
                if self.receiver.done():
                    await self.receiver
                await asyncio.sleep(0.001)

    def offer(self, sequence):
        self.clock.offset += 0.01
        frame = replace(
            _frame("owner", self.session.context.jarvis_session_id),
            visual_session_id=self.stream, frame_sequence=sequence,
            captured_at=self.clock.now(), server_received_at=self.clock.now(),
            image_bytes=f"synthetic-frame-{sequence}".encode(),
        )
        self.window.put(frame, stream_lease=self.lease)
        self.delivery.notify_frame_available()
        return frame

    async def ready(self, sequence):
        frame = self.offer(sequence)
        await self.until(
            lambda: self.state.latest_acknowledged is not None
            and self.state.latest_acknowledged.frame_id == frame.frame_id
        )
        return frame

    async def request(self, *, receive_loop=False):
        await self.fixture.append()
        await self.session._handle_client_event({"type": "audio.input.commit"})
        previous = self.session._client_receive_task
        if receive_loop:
            self.session._client_receive_task = asyncio.current_task()
        try:
            await self.session._handle_client_event({"type": "response.create"})
        finally:
            self.session._client_receive_task = previous

    async def acknowledged(self, response_id="response-1"):
        await self.until(lambda: response_id in self.session._active_responses)
        await self.until(lambda: self.delivery._fence_generation is None)

    def assert_shared_response_policy(self, recorded):
        instructions = recorded["event"]["response"]["instructions"]
        self.assertIn(build_base_prompt(), instructions)
        self.assertEqual(instructions.count("\n\n".join(RESPONSE_STYLE_PROMPT_LINES)), 1)

    def assert_healthy_cutoff(self, recorded):
        self.assert_shared_response_policy(recorded)
        snapshot = recorded["cutoff"]
        self.assertTrue(snapshot.synchronized)
        self.assertEqual(
            frozenset(item.item_id for item in snapshot.acknowledged_items),
            recorded["images"],
        )
        self.assertFalse(snapshot.possibly_present_items)
        self.assertIsNotNone(snapshot.response_submitted_at)
        self.assertNotIn("conversation", recorded["event"]["response"])
        self.assertNotIn("input", recorded["event"]["response"])

    async def test_cutoff_preserves_spoken_request_context_and_default_tool_history(self):
        first = await self.ready(1)
        context = SimpleNamespace(as_prompt_payload=lambda: {
            "working_memory": {"marker": "older-temporal-state"},
            "behavior_instructions": "Be concise.",
        })
        await self.session._send_complete_session_update(context)
        await self.provider.send_event({
            "type": "conversation.item.create",
            "item": {"type": "function_call_output", "call_id": "prior-tool", "output": "prior-result"},
        })
        await self.request()
        await self.acknowledged()
        recorded = self.wire.responses[0]
        self.assert_healthy_cutoff(recorded)
        instructions = recorded["event"]["response"]["instructions"]
        self.assertIn(build_base_prompt(), instructions)
        self.assertIn("older-temporal-state", instructions)
        self.assertIn("response_cutoff", instructions)
        types = [event["type"] for event in self.wire.events]
        self.assertLess(types.index("input_audio_buffer.append"), types.index("response.create"))
        self.assertTrue(any(event.get("item", {}).get("call_id") == "prior-tool" for event in self.wire.events))
        self.assertEqual(recorded["cutoff"].current_frame.frame_id, first.frame_id)
        await self.ready(2)
        self.assertEqual(recorded["cutoff"].current_frame.frame_id, first.frame_id)
        self.assertEqual(len(self.wire.responses), 1)

    async def test_response_without_attached_images_keeps_the_shared_policy(self):
        await self.request()
        await self.acknowledged()
        recorded = self.wire.responses[0]
        self.assert_shared_response_policy(recorded)
        self.assertFalse(recorded["images"])
        self.assertIsNone(recorded["cutoff"].current_frame)

    async def test_receive_loop_and_audio_continue_while_waiting_for_image_ack(self):
        self.wire.auto_image_ack = False
        self.offer(1)
        await self.until(lambda: bool(self.wire.images))
        await self.request(receive_loop=True)
        task = self.session._response_preparation_task
        self.assertIsNotNone(task)
        self.assertFalse(task.done())
        # The same receive path remains free to process image acknowledgements.
        image = next(event for event in self.wire.events if event.get("item", {}).get("id") in self.wire.images)
        self.wire.ack_image(image)
        await asyncio.wait_for(asyncio.shield(task), 1)
        await self.acknowledged()
        self.assert_healthy_cutoff(self.wire.responses[0])

    async def test_frame_arriving_at_response_write_cannot_rewrite_prepared_input(self):
        first = await self.ready(1)
        self.wire.before_response = lambda: self.offer(2)
        await self.request()
        await self.acknowledged()
        self.wire.before_response = None
        self.assert_healthy_cutoff(self.wire.responses[0])
        self.assertEqual(self.wire.responses[0]["cutoff"].current_frame.frame_id, first.frame_id)
        await self.until(lambda: self.state.latest_acknowledged.frame_sequence == 2)
        self.assertEqual(len(self.wire.responses), 1)

    async def test_delayed_image_ack_produces_degraded_not_certified_cutoff(self):
        self.wire.auto_image_ack = False
        self.offer(1)
        await self.until(lambda: bool(self.wire.images))
        await self.request()
        await self.acknowledged()
        snapshot = self.wire.responses[0]["cutoff"]
        self.assert_shared_response_policy(self.wire.responses[0])
        self.assertFalse(snapshot.synchronized)
        self.assertIsNone(snapshot.current_frame)
        self.assertTrue(snapshot.possibly_present_items)
        self.assertTrue(self.wire.responses[0]["images"].issubset(snapshot.pinned_item_ids))
        self.assertEqual(len(self.wire.responses), 1)

    async def test_delayed_delete_ack_does_not_claim_exact_provider_image_set(self):
        await self.ready(1)
        self.wire.auto_delete_ack = False
        for sequence in (2, 3):
            await self.ready(sequence)
        await self.until(lambda: any(event["type"] == "conversation.item.delete" for event in self.wire.events))
        await self.request()
        await self.acknowledged()
        self.assertFalse(self.wire.responses[0]["cutoff"].synchronized)
        self.assertIsNone(self.wire.responses[0]["cutoff"].current_frame)

    async def test_barge_in_before_submission_releases_fence_without_duplicate_response(self):
        self.wire.auto_image_ack = False
        self.offer(1)
        await self.until(lambda: bool(self.wire.images))
        request = asyncio.create_task(self.request())
        await self.until(lambda: self.delivery._fence_generation is not None)
        await self.fixture.append()
        await asyncio.wait_for(request, 1)
        self.assertFalse(self.wire.responses)
        self.assertIsNone(self.delivery._fence_generation)
        self.assertFalse(self.state.pinned_item_ids)
        self.assertEqual(request.cancelling(), 0)

    async def test_late_response_ack_keeps_prior_pins_until_its_own_settlement(self):
        await self.ready(1)
        self.wire.auto_response_ack = False
        await self.request()
        first = self.wire.responses[0]
        await self.request()
        second = self.wire.responses[1]
        self.assertFalse(second["cutoff"].synchronized)
        first_generation = first["cutoff"].generation_id
        self.assertIn(first_generation, self.state._held_response_pins)
        self.wire.ack_response(first, "late-first")
        await self.until(lambda: "late-first" in self.session._active_responses)
        self.wire.incoming.put_nowait({
            "type": "response.done", "response": {"id": "late-first", "status": "cancelled"},
        })
        await self.until(lambda: first_generation not in self.state._held_response_pins)
        self.assertEqual(self.state.active_snapshot.generation_id, second["cutoff"].generation_id)
        self.wire.ack_response(second, "current-second")
        await self.acknowledged("current-second")
        self.assertEqual(self.session._current_response_id, "current-second")
        self.assertEqual(len(self.wire.responses), 2)

    async def test_tool_follow_up_uses_new_image_without_losing_tool_output(self):
        await self.ready(1)
        await self.request()
        await self.acknowledged()
        newest = await self.ready(2)
        self.session.tool_executor.execute = AsyncMock(return_value={
            "status": "success", "result": "ordinary-tool-result",
        })
        self.wire.incoming.put_nowait({
            "type": "response.output_item.done", "response_id": "response-1",
            "item": {
                "type": "function_call", "call_id": "inspection-1",
                "name": "ordinary_example", "arguments": '{}',
            },
        })
        await self.until(lambda: self.session._tool_followup is not None)
        self.wire.incoming.put_nowait({
            "type": "response.done", "response": {"id": "response-1", "status": "completed"},
        })
        await self.until(lambda: len(self.wire.responses) == 2)
        await self.acknowledged("response-2")
        self.assert_healthy_cutoff(self.wire.responses[1])
        self.assertEqual(self.wire.responses[1]["cutoff"].current_frame.frame_id, newest.frame_id)
        outputs = [event for event in self.wire.events if event.get("item", {}).get("call_id") == "inspection-1"]
        self.assertEqual(len(outputs), 1)
        self.assertIn("ordinary-tool-result", outputs[0]["item"]["output"])

    async def test_queued_response_rechecks_freshness_at_real_writer(self):
        await self.ready(1)
        await self.provider._send_lock.acquire()
        # Start from an admitted turn so audio append is not itself held by the writer.
        async with self.session._response_admission_lock:
            reservation = self.session._reserve_response_locked("visual_interrupt", {})
        task = asyncio.create_task(self.session._send_reserved_response(reservation))
        try:
            await self.until(lambda: self.provider._foreground_waiters > 0)
            self.clock.offset += 6
        finally:
            self.provider._send_lock.release()
        await asyncio.wait_for(task, 1)
        self.assertEqual(self.wire.responses[0]["cutoff"].availability.status.value, "stale")
        self.assertIsNone(self.wire.responses[0]["cutoff"].current_frame)

    async def test_source_replaced_after_preparation_revokes_freshness(self):
        await self.ready(1)
        await self.provider._send_lock.acquire()
        async with self.session._response_admission_lock:
            reservation = self.session._reserve_response_locked("visual_interrupt", {})
        task = asyncio.create_task(self.session._send_reserved_response(reservation))
        try:
            await self.until(lambda: self.provider._foreground_waiters > 0)
            self.window.end_stream(self.lease)
        finally:
            self.provider._send_lock.release()
        await asyncio.wait_for(task, 1)
        self.assertFalse(self.wire.responses[0]["cutoff"].synchronized)
        self.assertIsNone(self.wire.responses[0]["cutoff"].current_frame)
