"""Actual publisher tick/projection/runtime/coordinator over controlled transport.

Canonical validity and binding are controlled, not SQL or native qualification.
Tests never manually offer a context: SessionLiveStateBridge owns every offer.
"""
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


class LiveStatePublicationRetryTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.now = NOW
        self.current = snapshot()
        self.previous_valid = True
        self.validations = []
        self.request = request()
        self.control = FakeControl(self.request.route)
        self.adapter = make_adapter(self.control, clock=lambda: self.now)
        self.coordinator = VoiceRouteCoordinator(SCOPE, validity=self.validity,
            publish_timeout=0.03, control_timeout=0.5, clock=lambda: self.now)
        await self.coordinator.start(self.adapter, self.request)
        self.session = SimpleNamespace(_closed=asyncio.Event(), _context_revoked=False)

        async def close_session(reason):
            self.session._closed.set()
            await self.coordinator.close()
        self.session.close = AsyncMock(side_effect=close_session)
        # Exercise the real ready/status/offer methods without creating native
        # channels, a provider connection, or an independent runtime owner.
        self.runtime = RealtimeVoiceRouteRuntime.__new__(RealtimeVoiceRouteRuntime)
        self.runtime.session, self.runtime.request = self.session, self.request
        self.runtime.coordinator = self.coordinator
        self.runtime.channel = SimpleNamespace(ready=True)
        self.runtime._installed = True
        self.runtime._closing = self.runtime._closed = False
        self.runtime.offer_context = Mock(wraps=self.runtime.offer_context)
        self.service = LiveStateService(SCOPE, clock=lambda: self.now)
        await self.service.start()
        self.service.snapshot = lambda: self.current
        self.live = SimpleNamespace(service=self.service, requires_reconstruction=False,
            bind=AsyncMock(return_value=BINDING), close=AsyncMock(side_effect=self.service.close))
        self.attachment = SessionLiveStateBridge(self.session, self.live, self.runtime,
                                                 clock=lambda: self.now)
        logger = patch("app.voice_routes.bridge._PUBLICATION_LOGGER.warning")
        self.log = logger.start()
        self.addCleanup(logger.stop)

    async def asyncTearDown(self):
        if self.control.lock.locked():
            self.control.lock.release()
        await asyncio.wait_for(self.attachment.close(), 2)

    async def validity(self, update):
        self.validations.append(update.update_id)
        return self.previous_valid if update == self.request.initial_context else True

    async def until(self, predicate):
        async with asyncio.timeout(2):
            while not predicate():
                await asyncio.sleep(0)

    def deferred(self):
        status = self.runtime.context_publication_status
        return status is not None and status.retry_update_id is not None

    def creates(self):
        return [item for item in self.control.writes if item["type"] == "conversation.item.create"]

    def deletes(self):
        return [item for item in self.control.writes if item["type"] == "conversation.item.delete"]

    def rows(self, stage):
        return [row for call in self.log.call_args_list
                if (row := json.loads(call.args[1]))["stage"] == stage]

    async def test_unchanged_content_eventually_reoffers_from_actual_tick_without_manual_offer(self):
        await self.control.lock.acquire()
        self.attachment.start()
        await self.until(self.deferred)
        offered = self.attachment._last_update
        self.assertIsNone(self.attachment._last_written_update_id)
        self.assertEqual(len(self.creates()), 1)
        self.assertEqual(self.rows("publication_deadline")[0]["outcome"], "cancellation_pending")
        self.assertEqual(self.rows("publication_fenced"), [])
        self.control.lock.release()
        await self.until(lambda: self.attachment._last_written_update_id == offered.update_id)
        self.assertEqual(self.runtime.context_publication_status.last_receipt.status, "written")
        self.assertEqual(self.adapter._context.update_id, offered.update_id)
        self.assertEqual([call.args[0].update_id for call in self.runtime.offer_context.call_args_list],
                         [offered.update_id, offered.update_id])
        self.live.bind.assert_awaited_once()
        self.assertEqual(len(self.creates()), 2)
        self.assertEqual(len(self.deletes()), 1)
        self.assertIn(self.request.initial_context.update_id, self.validations)
        await asyncio.sleep(0.22)
        self.assertEqual(self.runtime.offer_context.call_count, 2)

    async def test_same_selection_has_two_retry_budget_not_unlimited_tick_loop(self):
        await self.control.lock.acquire()
        self.attachment.start()
        await self.until(lambda: self.attachment.last_omission == "safe_context_retry_exhausted")
        self.assertEqual(self.attachment._unwritten_retry_count, 2)
        self.assertEqual(self.runtime.offer_context.call_count, 3)
        self.assertEqual(len(self.creates()), 1)
        self.assertEqual(self.deletes(), [])
        await asyncio.sleep(0.22)
        self.assertEqual(self.runtime.offer_context.call_count, 3)
        self.assertEqual(self.coordinator.status.phase, CoordinationPhase.ACTIVE)
        self.assertTrue(self.adapter.can_retain_context(self.request.initial_context))

    async def test_actual_tick_coalesces_changed_selection_and_only_latest_is_written(self):
        await self.control.lock.acquire()
        self.attachment.start()
        await self.until(self.deferred)
        self.current = snapshot(2, text="middle context")
        await self.until(lambda: self.attachment._last_update.cursor.revision == 2 and self.deferred())
        self.current = snapshot(3, text="newest context")
        await self.until(lambda: self.attachment._last_update.cursor.revision == 3 and self.deferred())
        self.control.lock.release()
        await self.until(lambda: self.attachment._last_written_update_id == self.attachment._last_update.update_id)
        self.assertEqual(self.adapter._context.cursor.revision, 3)
        self.assertIn("newest context", self.adapter._context.text)
        self.assertNotIn("middle context", json.dumps(self.creates()))
        self.assertEqual(len(self.creates()), 2)
        self.assertEqual(len(self.deletes()), 1)

    async def test_expired_deferred_update_is_not_retried_and_fresh_revision_can_replace_it(self):
        await self.control.lock.acquire()
        self.attachment.start()
        await self.until(self.deferred)
        stale = self.attachment._last_update
        self.assertIsNotNone(stale.expires_at)
        self.now = stale.expires_at + timedelta(milliseconds=1)
        await self.until(lambda: self.attachment.last_omission == "deferred_context_expired")
        self.control.lock.release()
        await asyncio.sleep(0.12)
        self.assertEqual(self.runtime.offer_context.call_count, 1)
        self.assertEqual(len(self.creates()), 1)
        self.current = snapshot(2, expiry=self.now + timedelta(seconds=10))
        await self.until(lambda: self.adapter._context.cursor.revision == 2)
        self.assertNotEqual(self.adapter._context.update_id, stale.update_id)
        self.assertGreater(self.adapter._context.expires_at, self.now)
        self.assertEqual(len(self.creates()), 2)
        self.live.bind.assert_awaited_once()

    async def test_shutdown_joins_publisher_and_prevents_deferred_tick_retry(self):
        await self.control.lock.acquire()
        self.attachment.start()
        await self.until(self.deferred)
        self.control.lock.release()
        await self.attachment.close()
        await asyncio.sleep(0.12)
        self.assertEqual(self.runtime.offer_context.call_count, 1)
        self.assertEqual(len(self.creates()), 1)
        self.assertTrue(all(task.done() for task in self.attachment._tasks))
        self.assertEqual(self.coordinator.status.phase, CoordinationPhase.CLOSED)

    async def test_possible_delete_without_ack_never_causes_publisher_retry(self):
        self.control.item_event = None
        self.attachment.start()
        await self.until(lambda: self.coordinator.status.phase == CoordinationPhase.DEGRADED)
        await asyncio.sleep(0.22)
        self.assertEqual(self.runtime.offer_context.call_count, 1)
        self.assertEqual(len(self.deletes()), 1)
        self.assertEqual(len(self.creates()), 1)
        self.assertIsNone(self.attachment._last_written_update_id)
        self.assertEqual(self.rows("publication_deferred"), [])
        self.assertEqual(self.rows("publication_fenced")[0]["outcome"], "failed_closed")

    async def test_previous_context_revocation_before_tick_retry_fences_without_mutation(self):
        await self.control.lock.acquire()
        self.attachment.start()
        await self.until(self.deferred)
        self.previous_valid = False
        self.control.lock.release()
        await self.until(lambda: self.coordinator.status.phase == CoordinationPhase.DEGRADED)
        self.assertEqual(self.coordinator.status.reason, "retained_context_unavailable_before_retry")
        self.assertEqual(len(self.creates()), 1)
        self.assertEqual(self.deletes(), [])
        self.assertIsNone(self.attachment._last_written_update_id)


if __name__ == "__main__":
    unittest.main()
