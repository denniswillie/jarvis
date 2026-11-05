"""Notification fencing preserves ownership without recursive turn cancellation."""
import asyncio
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock
from uuid import uuid4

from app.realtime.response_control import ResponseControlLane
from app.realtime.voice_route_runtime import RealtimeVoiceRouteRuntime, VoiceRouteRuntimeError
from app.voice_routes.contracts import InterruptSpeechCommand
import test_native_attention_delivery as native
import test_realtime_voice_route as route_fixtures


class NativeFenceLaneTests(unittest.IsolatedAsyncioTestCase):
    async def test_fence_only_does_not_schedule_another_generic_cancel(self):
        control = route_fixtures.FakeControl()
        adapter = route_fixtures.make_adapter(control)
        try:
            await adapter.start(route_fixtures.make_start())
            command = InterruptSpeechCommand(route=adapter.identity, command_id=uuid4(),
                requested_at=route_fixtures.NOW, reason="notification_revoked")
            epoch = adapter._speech_epoch
            result = await adapter.fence_output_only(command)
            await asyncio.sleep(0)
            self.assertEqual(result.status, "completed")
            self.assertTrue(result.output_fenced)
            self.assertEqual(adapter._speech_epoch, epoch + 1)
            self.assertFalse(control.cancel_started.is_set())
            self.assertFalse(any(value["type"] == "response.cancel" for value in control.writes))
        finally:
            await route_fixtures.stop_route(adapter)

    async def test_runtime_requires_exact_revoked_owner(self):
        route = route_fixtures.make_identity()
        owner = SimpleNamespace(revoked=True)
        adapter = SimpleNamespace(fence_output_only=AsyncMock(return_value="receipt"))
        runtime = SimpleNamespace(request=SimpleNamespace(route=route), adapter=adapter, _dispatches={},
                                  session=SimpleNamespace(_attention_delivery=owner))
        command = InterruptSpeechCommand(route=route, command_id=uuid4(),
            requested_at=route_fixtures.NOW, reason="notification_revoked")
        self.assertEqual(await RealtimeVoiceRouteRuntime.fence_notification_output(
            runtime, command, delivery=owner), "receipt")
        with self.assertRaises(VoiceRouteRuntimeError):
            await RealtimeVoiceRouteRuntime.fence_notification_output(
                runtime, command, delivery=SimpleNamespace(revoked=True))
        owner.revoked = False
        with self.assertRaises(VoiceRouteRuntimeError):
            await RealtimeVoiceRouteRuntime.fence_notification_output(runtime, command, delivery=owner)
        adapter.fence_output_only.assert_awaited_once()

    async def test_owned_fence_blocks_new_response_barrier_but_not_request_stop(self):
        fixture = native.NativeDeliveryTests()
        await fixture.asyncSetUp()
        offering, fencing, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
        original_interrupt = fixture.runtime.coordinator.interrupt_speech.side_effect
        async def offer(_):
            offering.set()
            await asyncio.Future()
        async def fence(command, *, delivery):
            self.assertIs(fixture.session._attention_delivery, delivery)
            fencing.set()
            await release.wait()
            return await original_interrupt(command)
        fixture.runtime.offer_notification.side_effect = offer
        fixture.runtime.fence_notification_output = AsyncMock(side_effect=fence)
        lane = ResponseControlLane(send=AsyncMock(), fail=lambda reason: self.fail(reason), deadline=1)
        delivery = fixture.sink.create_delivery(fixture.f, None, fixture.f.claim,
                                                fixture.session, receipt_seconds=1)
        fixture.session._attention_delivery = delivery
        delivery.task = asyncio.create_task(delivery.run())
        try:
            await offering.wait()
            first = delivery.request_stop("user_interrupted")
            lane.hold_fence("delivery", first)
            await fencing.wait()
            self.assertIs(delivery.request_stop("user_interrupted"), first)
            lane.hold_fence("delivery", first)
            barrier = asyncio.create_task(lane.wait_idle())
            await asyncio.sleep(0)
            self.assertFalse(barrier.done())
            self.assertIs(fixture.session._attention_delivery, delivery)
            self.assertFalse(delivery.task.done())
            release.set()
            await asyncio.wait_for(asyncio.gather(barrier, delivery.task), 1)
            self.assertIsNone(fixture.session._attention_delivery)
            self.assertEqual(fixture.f.attempt["phase"], "interrupted")
            fixture.runtime.fence_notification_output.assert_awaited_once()
            fixture.runtime.coordinator.interrupt_speech.assert_not_awaited()
        finally:
            release.set()
            await lane.close()
            await fixture.asyncTearDown()

    async def test_lane_shutdown_does_not_cancel_delivery_owned_fence(self):
        release = asyncio.Event()
        fence = asyncio.create_task(release.wait())
        lane = ResponseControlLane(send=AsyncMock(), fail=lambda _: None, deadline=1)
        lane.hold_fence("owned", fence)
        await asyncio.sleep(0)
        await lane.close()
        self.assertFalse(fence.cancelled())
        release.set()
        await fence
