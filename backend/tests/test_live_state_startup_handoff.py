"""Startup dedup over the actual coordinator/adapter with controlled transport.

This is not SQL/native qualification. No receipt, readiness, or startup ACK is
invented by the attachment; the real coordinator must first confirm its start.
"""
import asyncio
from dataclasses import replace
from datetime import timedelta
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock

from app.live_state.models import LiveSnapshot, StateCursor
from app.live_state.projection import prepare_voice_context, select_voice_context
from app.live_state.service import LiveStateService
from app.realtime.live_state_bridge import SessionLiveStateBridge
from app.realtime.voice_route_runtime import RealtimeVoiceRouteRuntime
from app.voice_routes.contracts import RouteStart
from app.voice_routes.coordinator import VoiceRouteCoordinator
from test_live_state_voice_attachment import BINDING, EPOCH, NOW, SCOPE, request, snapshot
from test_realtime_voice_route import FakeControl, make_adapter


def empty_snapshot(revision=1, *, readiness="ready", epoch=EPOCH):
    return LiveSnapshot(scope=SCOPE, cursor=StateCursor(epoch=epoch, revision=revision),
                        as_of=NOW, readiness=readiness, entries=())


class StartupHandoffTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.now = NOW
        self.attachment = self.coordinator = self.service = None

    async def boot(self, current=None, *, start=True, handoff=True):
        self.current = current if current is not None else empty_snapshot()
        self.service = LiveStateService(SCOPE, clock=lambda: self.now)
        await self.service.start()
        self.service.snapshot = lambda: self.current
        self.live = SimpleNamespace(service=self.service, requires_reconstruction=False,
            bind=AsyncMock(return_value=BINDING), close=AsyncMock(side_effect=self.service.close))
        selected = select_voice_context(self.current, now=self.now)
        route = request().route
        initial = await prepare_voice_context(selected, route, bind=self.live.bind,
                                              clock=lambda: self.now)
        self.request = RouteStart(route=route, command_id=request().command_id,
            requested_at=self.now, authorization_ref=SCOPE.session_id, initial_context=initial)
        self.control = FakeControl(route)
        self.adapter = make_adapter(self.control, clock=lambda: self.now)
        self.coordinator = VoiceRouteCoordinator(SCOPE, validity=AsyncMock(return_value=True),
            control_timeout=0.5, clock=lambda: self.now, preflight_expiry_refresh=True)
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
        self.runtime.offer_context = Mock(wraps=self.runtime.offer_context)
        self.selected = selected
        self.attachment = SessionLiveStateBridge(self.session, self.live, self.runtime,
            clock=lambda: self.now, initial_selection=selected if handoff else None)
        if start:
            await self.coordinator.start(self.adapter, self.request)

    async def asyncTearDown(self):
        if self.attachment is not None:
            await asyncio.wait_for(self.attachment.close(), 2)
        elif self.coordinator is not None:
            await self.coordinator.close()
        if self.service is not None:
            await self.service.close()

    def creates(self):
        return [row for row in self.control.writes if row["type"] == "conversation.item.create"]

    async def published(self, revision):
        async with asyncio.timeout(2):
            while self.adapter._context.cursor.revision != revision:
                await asyncio.sleep(0)

    async def test_empty_initial_selection_is_not_republished_for_cursor_progress(self):
        await self.boot()
        self.current = empty_snapshot(10)
        await self.attachment.publish_latest()
        self.runtime.offer_context.assert_not_called()
        self.live.bind.assert_not_awaited()
        self.assertEqual(len(self.creates()), 1)
        self.assertIs(self.attachment._last_update, self.request.initial_context)
        self.assertIsNone(self.attachment._last_written_update_id)
        self.assertIsNone(self.runtime.context_publication_status.last_receipt)

    async def test_source_backed_initial_selection_does_not_rebind(self):
        await self.boot(snapshot())
        self.current = snapshot(10)
        await self.attachment.publish_latest()
        self.runtime.offer_context.assert_not_called()
        self.live.bind.assert_awaited_once()
        self.assertIs(self.attachment._last_update.guard_binding, BINDING)
        self.assertEqual(len(self.creates()), 1)

    async def test_no_dedup_authority_before_confirmed_start(self):
        await self.boot(start=False)
        await self.attachment.publish_latest()
        self.assertIsNone(self.attachment._last_update)
        self.assertEqual(self.creates(), [])
        await self.coordinator.start(self.adapter, self.request)
        self.current = empty_snapshot(2)
        await self.attachment.publish_latest()
        self.runtime.offer_context.assert_not_called()
        self.assertIs(self.attachment._last_update, self.request.initial_context)

    async def test_changed_text_during_start_is_published(self):
        await self.boot(snapshot())
        self.current = snapshot(2, text="The replacement request")
        await self.attachment.publish_latest()
        await self.published(2)
        self.assertIn("The replacement request", self.adapter._context.text)
        self.assertEqual(self.live.bind.await_count, 2)
        self.assertEqual(len(self.creates()), 2)

    async def test_changed_readiness_is_not_an_empty_context_duplicate(self):
        await self.boot(empty_snapshot(readiness="warming"))
        self.current = empty_snapshot(2)
        await self.attachment.publish_latest()
        await self.published(2)
        self.assertNotEqual(self.adapter._context.text, self.request.initial_context.text)
        self.assertEqual(len(self.creates()), 2)

    async def test_removed_source_is_published(self):
        await self.boot(snapshot())
        self.current = empty_snapshot(2)
        await self.attachment.publish_latest()
        await self.published(2)
        self.assertEqual(self.adapter._context.dependencies, ())
        self.assertNotIn("Check the number", self.adapter._context.text)
        self.assertEqual(len(self.creates()), 2)

    async def test_near_expiry_renews_using_original_binding(self):
        await self.boot(snapshot())
        self.now += timedelta(seconds=9)
        self.current = snapshot(2, expiry=NOW + timedelta(seconds=20))
        await self.attachment.publish_latest()
        await self.published(2)
        self.live.bind.assert_awaited_once()
        self.assertEqual(self.adapter._context.expires_at, NOW + timedelta(seconds=20))
        self.assertEqual(len(self.creates()), 2)

    async def test_expired_initial_selection_cannot_seed_dedup(self):
        await self.boot(snapshot())
        self.now += timedelta(seconds=11)
        self.current = snapshot(2, expiry=NOW + timedelta(seconds=20))
        await self.attachment.publish_latest()
        await self.published(2)
        self.assertEqual(self.live.bind.await_count, 2)
        self.assertEqual(len(self.creates()), 2)

    async def test_foreign_epoch_cannot_seed_or_replace_current_route(self):
        await self.boot()
        self.current = empty_snapshot(10, epoch=request().command_id)
        await self.attachment.publish_latest()
        self.assertIsNone(self.attachment._last_update)
        self.assertEqual(self.attachment.last_omission, "epoch_requires_new_route")
        self.assertEqual(len(self.creates()), 1)

    async def test_mismatched_initial_selection_is_rejected(self):
        await self.boot()
        with self.assertRaisesRegex(ValueError, "exact prepared startup context"):
            SessionLiveStateBridge(self.session, self.live, self.runtime,
                clock=lambda: self.now, initial_selection=replace(self.selected, text="Not installed"))

    async def test_callers_without_exact_handoff_keep_existing_publication_behavior(self):
        await self.boot(handoff=False)
        self.current = empty_snapshot(2)
        await self.attachment.publish_latest()
        await self.published(2)
        self.runtime.offer_context.assert_called_once()
        self.assertEqual(len(self.creates()), 2)
