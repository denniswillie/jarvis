"""Real publisher/runtime/adapter flow over offline transport; no native proof."""
import asyncio
from datetime import timedelta
import json
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

from app.live_state.service import LiveStateService
from app.realtime.live_state_bridge import SessionLiveStateBridge
from app.realtime.voice_route_runtime import RealtimeVoiceRouteRuntime
from app.voice_routes.coordinator import CoordinationPhase, VoiceRouteCoordinator
from test_live_state_voice_attachment import BINDING, NOW, SCOPE, request, snapshot
from test_realtime_voice_route import FakeControl, make_adapter
import test_response_route_prewrite_admission as response_fixture


class ContextExpiryRecoveryTests(unittest.IsolatedAsyncioTestCase):
    async def setup_route(self, *, enabled=True, timeout=0.15):
        self.now = NOW
        self.current = snapshot()
        self.fresh_valid = True
        self.expire_revisions = {1}
        self.request = request()
        self.control = FakeControl(self.request.route)
        self.adapter = make_adapter(self.control, clock=lambda: self.now)

        async def validity(update):
            if update.cursor.revision in self.expire_revisions:
                self.now = update.expires_at + timedelta(milliseconds=1)
                return False
            return self.fresh_valid

        self.coordinator = VoiceRouteCoordinator(SCOPE, validity=validity,
            publish_timeout=timeout, control_timeout=0.5, clock=lambda: self.now,
            preflight_expiry_refresh=enabled)
        await self.coordinator.start(self.adapter, self.request)
        self.session = SimpleNamespace(_closed=asyncio.Event(), _context_revoked=False)

        async def close_session(reason):
            self.session._closed.set()
            await self.coordinator.close()
        self.session.close = AsyncMock(side_effect=close_session)
        self.runtime = RealtimeVoiceRouteRuntime.__new__(RealtimeVoiceRouteRuntime)
        self.runtime.session, self.runtime.request = self.session, self.request
        self.runtime.coordinator = self.coordinator
        self.runtime.channel = SimpleNamespace(ready=True)
        self.runtime._installed = True
        self.runtime._closing = self.runtime._closed = False
        self.service = LiveStateService(SCOPE, clock=lambda: self.now)
        await self.service.start()
        self.service.snapshot = lambda: self.current
        self.live = SimpleNamespace(service=self.service, requires_reconstruction=False,
            bind=AsyncMock(return_value=BINDING), close=AsyncMock(side_effect=self.service.close))
        self.attachment = SessionLiveStateBridge(self.session, self.live, self.runtime,
                                                 clock=lambda: self.now)
        self.addAsyncCleanup(self.attachment.close)
        logger = patch("app.voice_routes.bridge._PUBLICATION_LOGGER.warning")
        self.log = logger.start()
        self.addCleanup(logger.stop)
        self.attachment.start()

    async def until(self, predicate):
        async with asyncio.timeout(2):
            while not predicate():
                await asyncio.sleep(0)

    def creates(self):
        return [row for row in self.control.writes if row["type"] == "conversation.item.create"]

    def deletes(self):
        return [row for row in self.control.writes if row["type"] == "conversation.item.delete"]

    def diagnostics(self, stage):
        return [row for call in self.log.call_args_list
                if (row := json.loads(call.args[1]))["stage"] == stage]

    async def test_expired_false_preflight_blocks_speech_but_actual_publisher_recovers(self):
        await self.setup_route()
        await self.until(lambda: self.runtime.context_refresh_pending)
        self.assertFalse(self.runtime.ready)
        self.assertTrue(self.runtime.context_publication_ready)
        self.assertEqual(self.coordinator.status.phase, CoordinationPhase.ACTIVE)
        self.assertEqual(len(self.creates()), 1)
        self.assertEqual(self.deletes(), [])
        row = self.diagnostics("context_refresh_wait")[0]
        self.assertEqual(row["outcome"], "expired_preflight")
        self.assertFalse(row["adapter_entered"])
        waiter = asyncio.create_task(self.runtime.wait_for_context_ready(1))
        self.current = snapshot(2, text="fresh complete snapshot", expiry=self.now + timedelta(seconds=10))
        self.assertTrue(await waiter)
        self.assertTrue(self.runtime.ready)
        self.assertEqual(self.adapter._context.cursor.revision, 2)
        self.assertEqual(len(self.creates()), 2)
        self.assertEqual(len(self.deletes()), 1)
        self.assertEqual(self.runtime.context_publication_status.last_receipt.status, "written")
        self.assertEqual(len(self.diagnostics("context_refresh_return")), 1)

    async def test_routes_without_opt_in_keep_existing_valid_context_retention(self):
        await self.setup_route(enabled=False)
        await self.until(lambda: self.coordinator._bridge.status.retry_update_id is not None)
        self.assertFalse(self.runtime.context_refresh_pending)
        self.assertTrue(self.runtime.ready)
        self.assertEqual(self.coordinator.status.phase, CoordinationPhase.ACTIVE)
        self.assertEqual(self.coordinator._bridge.status.reason, "context_publication_deferred_unwritten")
        self.assertEqual(self.coordinator._bridge.status.retry_update_id, self.attachment._last_update.update_id)
        self.assertEqual(self.adapter._context, self.request.initial_context)
        self.assertTrue(self.adapter.can_retain_context(self.request.initial_context))
        self.assertEqual(len(self.creates()), 1)
        self.assertEqual(self.deletes(), [])

    async def test_freshness_wait_is_bounded_without_retiring_healthy_route(self):
        await self.setup_route(timeout=0.03)
        await self.until(lambda: self.runtime.context_refresh_pending)
        self.assertFalse(await self.runtime.wait_for_context_ready(0.08))
        self.assertEqual(self.coordinator.status.phase, CoordinationPhase.ACTIVE)
        self.assertIsNone(self.coordinator.status.reason)
        self.assertFalse(self.runtime.ready)
        self.assertEqual(len(self.creates()), 1)
        self.assertEqual(self.deletes(), [])
        self.current = snapshot(2, expiry=self.now + timedelta(seconds=10))
        self.assertTrue(await self.runtime.wait_for_context_ready(1))
        self.assertTrue(self.runtime.ready)

    async def test_fresh_current_source_rejection_does_not_end_hold_or_write(self):
        await self.setup_route()
        await self.until(lambda: self.runtime.context_refresh_pending)
        self.fresh_valid = False
        self.current = snapshot(2, expiry=self.now + timedelta(seconds=10))
        await self.until(lambda: self.coordinator.status.phase == CoordinationPhase.DEGRADED)
        self.assertEqual(self.coordinator.status.reason, "context_validity_rejected")
        self.assertEqual(len(self.creates()), 1)
        self.assertEqual(self.deletes(), [])
        self.assertFalse(await self.runtime.wait_for_context_ready(0.1))

    async def test_possible_delete_without_ack_still_fences_and_never_retries(self):
        await self.setup_route()
        await self.until(lambda: self.runtime.context_refresh_pending)
        self.control.item_event = None
        self.current = snapshot(2, expiry=self.now + timedelta(seconds=10))
        await self.until(lambda: self.coordinator.status.phase == CoordinationPhase.DEGRADED)
        self.assertFalse(self.runtime.ready)
        self.assertEqual(len(self.creates()), 1)
        self.assertEqual(len(self.deletes()), 1)
        self.assertEqual(self.diagnostics("context_refresh_return"), [])

    async def test_close_releases_waiter_and_joins_owned_work(self):
        await self.setup_route()
        await self.until(lambda: self.runtime.context_refresh_pending)
        waiter = asyncio.create_task(self.runtime.wait_for_context_ready(1))
        bridge = self.coordinator._bridge
        await self.attachment.close()
        self.assertFalse(await waiter)
        self.assertTrue(all(job.done() for job in bridge._jobs))
        self.assertFalse(self.runtime.ready)

    async def test_more_expired_snapshots_do_not_retry_or_retire_the_connection(self):
        await self.setup_route(timeout=0.2)
        await self.until(lambda: self.runtime.context_refresh_pending)
        self.expire_revisions.add(2)
        self.current = snapshot(2, expiry=self.now + timedelta(seconds=10))
        await self.until(lambda: len(self.diagnostics("context_refresh_wait")) == 2)
        self.assertFalse(await self.runtime.wait_for_context_ready(0.05))
        self.assertEqual(self.coordinator.status.phase, CoordinationPhase.ACTIVE)
        self.assertIsNone(self.coordinator.status.reason)
        self.assertEqual(len(self.creates()), 1)

    async def test_actual_publisher_coalesces_at_response_boundary(self):
        await self.setup_route()
        await self.until(lambda: self.runtime.context_refresh_pending)
        self.session._active_responses = {"response": object()}
        self.current = snapshot(2, text="middle", expiry=self.now + timedelta(seconds=10))
        await self.attachment.publish_latest()
        self.current = snapshot(3, text="latest", expiry=self.now + timedelta(seconds=10))
        await self.attachment.publish_latest()
        self.assertEqual(len(self.creates()), 1)
        self.assertEqual(self.deletes(), [])
        self.session._active_responses.clear()
        self.assertTrue(await self.runtime.wait_for_context_ready(1))
        self.assertEqual(self.adapter._context.cursor.revision, 3)
        self.assertEqual(len(self.creates()), 2)

    async def test_waiting_unwritten_turn_allows_needed_freshness_publication(self):
        await self.setup_route()
        await self.until(lambda: self.runtime.context_refresh_pending)
        self.session._pending_response = SimpleNamespace(request_id="request", submitted=False)
        self.assertFalse(self.runtime.context_publication_idle)
        self.session._context_waiting_request_id = "request"
        self.assertTrue(self.runtime.context_publication_idle)
        self.current = snapshot(2, expiry=self.now + timedelta(seconds=10))
        self.assertTrue(await self.runtime.wait_for_context_ready(1))


class ResponseExpiryWaitTests(unittest.IsolatedAsyncioTestCase):
    setUp = response_fixture.ResponseRoutePrewriteAdmissionTests.setUp
    voice = response_fixture.ResponseRoutePrewriteAdmissionTests.voice
    records = response_fixture.ResponseRoutePrewriteAdmissionTests.records

    def waiting_voice(self, *, pending=True):
        voice, runtime, reservation, socket = self.voice()
        voice._now = lambda: NOW
        voice._monotonic = asyncio.get_running_loop().time
        voice._response_admission_lock = asyncio.Lock()
        voice.prepare_visual_response = AsyncMock(return_value=None)
        voice.cancel_visual_response_preparation = Mock()
        voice._send_error = AsyncMock()
        released = asyncio.Event()
        entered = asyncio.Event()
        bridge = SimpleNamespace(awaiting_fresh_context=pending)

        async def wait(timeout):
            entered.set()
            try:
                async with asyncio.timeout(timeout):
                    await released.wait()
            except TimeoutError:
                return False
            bridge.awaiting_fresh_context = False
            return True
        bridge.wait_for_fresh_context = wait
        runtime.coordinator._bridge = bridge
        return voice, runtime, reservation, socket, bridge, entered, released

    async def test_accepted_turn_waits_off_reader_and_writes_same_request_once(self):
        voice, runtime, reservation, socket, bridge, entered, released = self.waiting_voice()
        task = asyncio.create_task(voice._prepare_and_send_response(reservation, None, None))
        try:
            await asyncio.wait_for(entered.wait(), 0.5)
            socket.send.assert_not_awaited()
            voice.prepare_visual_response.assert_not_awaited()
            self.assertFalse(reservation.submitted)
            released.set()
            self.assertTrue(await task)
            self.assertEqual(json.loads(socket.send.call_args.args[0])["event_id"], reservation.request_id)
            socket.send.assert_awaited_once()
            voice.prepare_visual_response.assert_awaited_once()
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def test_waiter_barge_in_or_route_replacement_cannot_submit_stale_turn(self):
        for change in ("cancel", "turn", "route"):
            with self.subTest(change=change):
                voice, runtime, reservation, socket, bridge, entered, released = self.waiting_voice()
                task = asyncio.create_task(voice._prepare_and_send_response(reservation, None, None))
                try:
                    await asyncio.wait_for(entered.wait(), 0.5)
                    if change == "cancel":
                        task.cancel()
                    elif change == "turn":
                        voice._turn_generation += 1
                    else:
                        voice._voice_route_runtime = None
                    released.set()
                    result = await asyncio.gather(task, return_exceptions=True)
                    self.assertIsInstance(result[0], asyncio.CancelledError) if change == "cancel" else self.assertFalse(result[0])
                    socket.send.assert_not_awaited()
                    self.assertFalse(reservation.submitted)
                finally:
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)

    async def test_finite_wait_reports_unavailable_not_fake_submission(self):
        voice, runtime, reservation, socket, bridge, entered, released = self.waiting_voice()
        with self.assertRaises(TimeoutError):
            await voice._prepare_and_send_response(reservation, None, None)
        self.assertFalse(reservation.submitted)
        socket.send.assert_not_awaited()
        self.assertIsNone(voice._pending_response)
        self.assertEqual(voice._last_user_turn_settled_at, NOW)
        self.assertIsInstance(voice._last_user_turn_settled_monotonic, float)
        voice._send_error.assert_awaited_once()

    async def test_hold_during_visual_preparation_reprepares_only_unwritten_normal_turn_once(self):
        voice, runtime, reservation, socket, bridge, entered, released = self.waiting_voice(pending=False)

        async def prepare(**kwargs):
            if voice.prepare_visual_response.await_count == 1:
                bridge.awaiting_fresh_context = True
            return None
        voice.prepare_visual_response.side_effect = prepare
        task = asyncio.create_task(voice._prepare_and_send_response(reservation, None, None))
        try:
            await asyncio.wait_for(entered.wait(), 0.5)
            self.assertFalse(reservation.submitted)
            socket.send.assert_not_awaited()
            released.set()
            self.assertTrue(await task)
            self.assertEqual(voice.prepare_visual_response.await_count, 2)
            socket.send.assert_awaited_once()
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def test_unknown_provider_write_is_not_retried_after_refresh(self):
        voice, runtime, reservation, socket, bridge, entered, released = self.waiting_voice()
        socket.send.side_effect = RuntimeError("private transport failure")
        released.set()
        with self.assertRaises(RuntimeError):
            await voice._prepare_and_send_response(reservation, None, None)
        self.assertTrue(reservation.submitted)
        socket.send.assert_awaited_once()
        voice.prepare_visual_response.assert_awaited_once()
