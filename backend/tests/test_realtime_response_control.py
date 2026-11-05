"""Cancellation liveness, identity and writer ordering without provider calls."""
import asyncio
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock

from app.realtime.response_control import ResponseControlLane
import test_realtime_image_tools as fixtures


class ResponseControlLaneTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.send = AsyncMock()
        self.failures = []
        self.lane = ResponseControlLane(send=self.send, fail=self.failures.append,
                                        deadline=0.1, capacity=2)

    async def asyncTearDown(self):
        await self.lane.close()

    async def test_unknown_identity_waits_for_exact_ack_and_deduplicates(self):
        self.lane.cancel(None, "old-request")
        await asyncio.sleep(0)
        self.send.assert_not_awaited()
        self.lane.cancel("old-response", "old-request")
        self.lane.cancel("old-response", "old-request")
        await self.lane.wait_idle()
        self.send.assert_awaited_once_with("cancel", "old-response", "old-request")

    async def test_rejected_creation_does_not_send_untargeted_cancel(self):
        self.lane.cancel(None, "rejected")
        self.lane.reject("rejected")
        await self.lane.wait_idle()
        self.send.assert_not_awaited()

    async def test_unknown_identity_timeout_fails_closed(self):
        self.lane.cancel(None, "never-acknowledged")
        with self.assertRaises(RuntimeError):
            await asyncio.wait_for(self.lane.wait_idle(), 1)
        self.assertEqual(self.failures, ["response_control_failed"])
        self.send.assert_not_awaited()

    async def test_missing_identity_never_sends_global_cancel(self):
        self.lane.cancel(None, None)
        self.assertEqual(self.failures, ["response_control_identity_missing"])
        self.send.assert_not_awaited()

    async def test_capacity_includes_running_work_and_close_joins(self):
        self.lane.cancel(None, "one")
        await asyncio.sleep(0)
        self.lane.cancel(None, "two")
        self.lane.cancel(None, "three")
        self.assertEqual(self.failures, ["response_control_queue_exhausted"])
        await self.lane.close()
        self.assertTrue(self.lane._task.done())
        self.assertFalse(self.lane.busy)

    async def test_immediate_close_cleans_before_worker_entry(self):
        self.lane.cancel(None, "one")
        await self.lane.close()
        self.assertFalse(self.lane.busy)
        self.assertTrue(self.lane._task.done())
        self.send.assert_not_awaited()

    async def test_failed_write_is_not_retried(self):
        self.send.side_effect = RuntimeError("fixture write failure")
        self.lane.cancel("old", "request")
        with self.assertRaises(RuntimeError):
            await self.lane.wait_idle()
        self.assertEqual(self.send.await_count, 1)
        self.assertEqual(self.failures, ["response_control_failed"])


class SessionResponseControlTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.fixture = fixtures.RealtimeImageToolTests()
        await self.fixture.asyncSetUp()
        self.session = self.fixture.session
        await self.fixture.start()
        self.session._voice_route_runtime = SimpleNamespace(
            ready=True,
            before_session_close=AsyncMock(return_value=True),
            after_session_close=AsyncMock(return_value=True))

    async def asyncTearDown(self):
        await self.fixture.asyncTearDown()

    async def test_cancel_releases_admission_and_client_while_writer_is_held(self):
        lock = self.session.openai._send_lock
        await lock.acquire()
        try:
            await asyncio.wait_for(self.session._handle_client_event({"type": "response.cancel"}), 0.2)
            await asyncio.wait_for(self.session._handle_client_event({"type": "response.cancel"}), 0.2)
            self.assertFalse(self.session._response_admission_lock.locked())
            self.assertTrue(self.session._response_controls.busy)
            self.assertTrue(self.session._active_responses["response-1"].interrupted)
        finally:
            lock.release()
        await self.session._response_controls.wait_idle()
        cancels = [event for event in self.fixture.wire.events if event["type"] == "response.cancel"]
        self.assertEqual(len(cancels), 1)
        self.assertEqual(cancels[0]["response_id"], "response-1")
        self.assertIn(cancels[0]["event_id"], self.session._recent_cancellations)

    async def test_response_factory_yields_to_control_queued_while_waiting_for_writer(self):
        await self.session._handle_client_event({"type": "response.cancel"})
        await self.session._response_controls.wait_idle()
        async with self.session._response_admission_lock:
            reservation = self.session._reserve_response_locked("normal_user_turn", {})
        lock = self.session.openai._send_lock
        await lock.acquire()
        sending = asyncio.create_task(self.session._send_reserved_response_locked(reservation))
        try:
            await asyncio.sleep(0)
            self.session._response_controls.clear_output()
        finally:
            lock.release()
        self.assertTrue(await asyncio.wait_for(sending, 1))
        events = self.fixture.wire.events
        clear = next(index for index, event in enumerate(events) if event["type"] == "output_audio_buffer.clear")
        create = next(index for index, event in enumerate(events)
                      if event["type"] == "response.create" and event["event_id"] == reservation.request_id)
        self.assertLess(clear, create)
        self.assertEqual(sum(event["type"] == "response.create" and event["event_id"] == reservation.request_id
                             for event in events), 1)

    async def test_direct_clear_is_owned_and_ordered_after_targeted_cancel(self):
        from app.realtime.direct_session import DirectRealtimeAudioSession
        original = self.session.__class__
        self.session.__class__ = DirectRealtimeAudioSession
        self.session._direct_audio_responses = {"response-1"}
        self.session._direct_output_pending = {"response-1"}
        lock = self.session.openai._send_lock
        await lock.acquire()
        try:
            await asyncio.wait_for(self.session._handle_client_event({"type": "response.cancel"}), 0.2)
            self.assertFalse(self.session._direct_audio_responses)
            self.assertFalse(self.session._response_admission_lock.locked())
        finally:
            self.session.__class__ = original
            lock.release()
        await self.session._response_controls.wait_idle()
        controls = [event["type"] for event in self.fixture.wire.events
                    if event["type"] in {"response.cancel", "output_audio_buffer.clear"}]
        self.assertEqual(controls, ["response.cancel", "output_audio_buffer.clear"])

    async def test_close_joins_cancellation_blocked_on_writer(self):
        lock = self.session.openai._send_lock
        await lock.acquire()
        try:
            await self.session._handle_client_event({"type": "response.cancel"})
            await asyncio.sleep(0)
            await asyncio.wait_for(self.session.close("fixture_shutdown"), 1)
            self.assertFalse(self.session._response_controls.busy)
            self.assertTrue(self.session._response_controls._task.done())
        finally:
            lock.release()
