"""Exact notification cancellation through the real reservation/event owner."""
import asyncio
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock
from uuid import uuid4

from app.realtime.models import RealtimeResponseState
from app.realtime.voice_route_runtime import RealtimeVoiceRouteRuntime, _Dispatch
from app.voice_routes.contracts import InterruptSpeechCommand
import test_realtime_image_tools as fixtures
import test_realtime_voice_route as routes


class TargetedNotificationCancellationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.h = fixtures.RealtimeImageToolTests()
        await self.h.asyncSetUp()
        self.session = self.h.session
        self.session._voice_route_runtime = SimpleNamespace(
            before_session_close=AsyncMock(return_value=True), after_session_close=AsyncMock(return_value=True))

    async def asyncTearDown(self):
        await self.h.asyncTearDown()

    def cancellations(self):
        return [event for event in self.h.wire.events if event["type"] == "response.cancel"]

    async def test_exact_old_response_cancel_preserves_newer_turn_and_wrong_offer_is_noop(self):
        await self.h.start()
        old = self.session._active_responses["response-1"]
        request_id = old.metadata["jarvis_request_id"]
        offer_id = str(uuid4())
        old.metadata["voice_route_offer_id"] = offer_id
        new = RealtimeResponseState(response_id="newer", kind="normal_user_turn",
                                    metadata={"jarvis_request_id": "new-request"})
        self.session._active_responses["newer"] = new
        self.session._current_response_id = "newer"
        generation = self.session._turn_generation
        await self.session.cancel_route_notification(request_id=request_id, offer_id="wrong-offer")
        self.assertFalse(old.interrupted)
        self.assertFalse(self.cancellations())
        await self.session.cancel_route_notification(request_id=request_id, offer_id=offer_id)
        await self.session.cancel_route_notification(request_id=request_id, offer_id=offer_id)
        await self.session._response_controls.wait_idle()
        self.assertEqual([event["response_id"] for event in self.cancellations()], ["response-1"])
        self.assertTrue(old.interrupted)
        self.assertFalse(new.interrupted)
        self.assertEqual(self.session._current_response_id, "newer")
        self.assertEqual(self.session._turn_generation, generation)

    async def test_pre_ack_cancellation_waits_for_exact_identity_before_new_response(self):
        self.h.receiver.cancel()
        await asyncio.gather(self.h.receiver, return_exceptions=True)
        offer_id = str(uuid4())
        async with self.session._response_admission_lock:
            old = self.session._reserve_response_locked("prospective", {"voice_route_offer_id": offer_id})
        await self.session._send_reserved_response(old)
        old_ack = await self.h.wire.incoming.get()
        generation = self.session._turn_generation
        await self.session.cancel_route_notification(request_id=old.request_id, offer_id=offer_id)
        self.assertIn(old.request_id, self.session._aborted_continuation_requests)
        self.assertTrue(self.session._response_controls.busy)
        self.assertFalse(self.cancellations())
        async with self.session._response_admission_lock:
            newer = self.session._reserve_response_locked("normal_user_turn", {})
        sending = asyncio.create_task(self.session._send_reserved_response(newer))
        try:
            await asyncio.sleep(0)
            self.assertEqual(len(self.h.wire.responses), 1)
            await self.session._handle_openai_event(old_ack)
            await asyncio.wait_for(sending, 1)
            await self.session._response_controls.wait_idle()
            self.assertEqual(len(self.h.wire.responses), 2)
            self.assertEqual([event["response_id"] for event in self.cancellations()], ["response-1"])
            self.assertTrue(self.session._active_responses["response-1"].interrupted)
            self.assertEqual(self.session._turn_generation, generation)
            self.assertIs(self.session._pending_response, newer)
        finally:
            if not sending.done():
                sending.cancel()
            await asyncio.gather(sending, return_exceptions=True)

    async def test_runtime_source_fence_uses_exact_dispatch_not_latest_turn(self):
        offer = routes.make_offer()
        other = routes.make_offer()
        owner = SimpleNamespace(revoked=True, offer=offer)
        cancel = AsyncMock()
        adapter = SimpleNamespace(fence_output_only=AsyncMock(return_value="fenced"))
        first = _Dispatch(offer, "dispatch-old", None, lambda: True,
                          response_request_id="exact-old-provider-request")
        newer = _Dispatch(other, "dispatch-new", None, lambda: True,
                          response_request_id="other-provider-request")
        runtime = SimpleNamespace(request=SimpleNamespace(route=offer.route), adapter=adapter,
            session=SimpleNamespace(_attention_delivery=owner, cancel_route_notification=cancel),
            _dispatches={"old": first, "new": newer})
        command = InterruptSpeechCommand(route=offer.route, command_id=uuid4(),
            requested_at=routes.NOW, reason="source_withdrawn")
        self.assertEqual(await RealtimeVoiceRouteRuntime.fence_notification_output(
            runtime, command, delivery=owner), "fenced")
        cancel.assert_awaited_once_with(request_id="exact-old-provider-request", offer_id=str(offer.offer_id))
        adapter.fence_output_only.assert_awaited_once_with(command)
