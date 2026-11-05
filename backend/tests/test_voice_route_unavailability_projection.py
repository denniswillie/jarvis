"""Offline actual-runtime/channel status projection; no network or database."""

import asyncio
import json
import unittest
from contextlib import asynccontextmanager
from unittest.mock import patch
from uuid import uuid4

from app.realtime.session import RealtimeAudioSession
from app.realtime.voice_route_runtime import RealtimeVoiceRouteRuntime
from app.voice_routes.coordinator import CoordinationPhase
from test_voice_route_runtime import CLIENT_CAPS, NOW, ROOT, Session, eventually, request_for


class VoiceRouteUnavailabilityTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.session = Session()
        self.request = request_for(self.session)
        self.guard_active = False

        @asynccontextmanager
        async def atomic(_):
            yield

        async def valid(_):
            return True

        self.runtime = RealtimeVoiceRouteRuntime(
            self.session, self.request, CLIENT_CAPS, atomic_guard=atomic,
            payload_roots=lambda _: (ROOT,), validity=valid, authorize_start=valid,
            clock=lambda: NOW, ready_timeout=0.2, control_timeout=0.05,
        )
        RealtimeAudioSession.install_voice_route_runtime(self.session, self.runtime)
        self.session._realtime_context_ready.set()
        await eventually(lambda: self.runtime.ready or self.runtime.failure is not None)
        self.assertTrue(self.runtime.ready, self.runtime.failure)

    async def asyncTearDown(self):
        await RealtimeAudioSession.close(self.session, "fixture_cleanup")

    def errors(self):
        return [item["payload"] for item in self.session._app_sender.messages
                if item.get("type") == "voice_route.event"
                and item.get("payload", {}).get("kind") == "route_error"]

    def degrade(self):
        self.runtime.coordinator._publication_failed(self.request.route, "context_receipt_unconfirmed")

    async def test_silent_no_write_factory_reports_once_after_original_guard_and_lock_exit(self):
        self.degrade()
        original = self.runtime.channel.send_route_event

        async def observed(event):
            self.assertFalse(self.guard_active)
            self.assertFalse(self.session.openai.lock.locked())
            await original(event)

        @asynccontextmanager
        async def guard():
            self.guard_active = True
            try:
                yield
            finally:
                self.guard_active = False

        before = list(self.session.openai.writes)
        with patch.object(self.runtime.channel, "send_route_event", side_effect=observed):
            self.assertFalse(await self.session.openai.send_event(lambda: None, write_guard=guard))
            self.assertFalse(await self.session.openai.send_event(lambda: None, write_guard=guard))
        self.assertEqual(self.session.openai.writes, before)
        self.assertEqual(len(self.errors()), 1)
        self.assertEqual(self.errors()[0]["route"], self.request.route.model_dump(mode="json"))
        self.assertEqual(self.errors()[0]["code"], "voice_route_unavailable")
        self.assertIs(self.errors()[0]["recoverable"], False)
        self.assertEqual(self.runtime.coordinator.status.phase, CoordinationPhase.DEGRADED)
        self.assertFalse(self.session.resources_closed)
        self.assertEqual(self.session.cancels, 0)
        self.assertIsNone(self.runtime.failure)

    async def test_response_rejection_still_rejects_without_provider_write(self):
        self.degrade()
        before = list(self.session.openai.writes)
        self.assertFalse(await self.session._request_normal_response(turn_generation=2))
        self.assertEqual(self.session.openai.writes, before)
        self.assertEqual(len(self.errors()), 1)

    async def test_active_refresh_is_not_a_terminal_notice(self):
        self.runtime._context_replacement_active = True
        try:
            self.assertFalse(await self.session.openai.send_event(lambda: None))
            self.assertEqual(self.errors(), [])
            self.assertFalse(self.runtime._unavailability_reported)
            self.assertEqual(self.runtime.coordinator.status.phase, CoordinationPhase.ACTIVE)
        finally:
            self.runtime._context_replacement_active = False
        self.assertTrue(self.runtime.ready)

    async def test_current_replacement_must_settle_before_notice(self):
        self.degrade()
        self.runtime._context_replacement_active = True
        try:
            await self.runtime._report_unavailable_status()
            self.assertEqual(self.errors(), [])
        finally:
            self.runtime._context_replacement_active = False
        await self.runtime._report_unavailable_status()
        self.assertEqual(len(self.errors()), 1)

    async def test_other_runtime_or_route_cannot_publish(self):
        self.degrade()
        with patch.object(self.session, "_voice_route_runtime", object()):
            await self.runtime._report_unavailable_status()
        identity = self.runtime.coordinator._identity
        self.runtime.coordinator._identity = identity.model_copy(update={"route_id": uuid4()})
        try:
            await self.runtime._report_unavailable_status()
        finally:
            self.runtime.coordinator._identity = identity
        self.assertEqual(self.errors(), [])
        self.assertFalse(self.runtime._unavailability_reported)

    async def test_closing_closed_and_disconnected_are_not_relabelled(self):
        self.degrade()
        for attribute in ("_closing", "_closed"):
            with patch.object(self.runtime, attribute, True):
                await self.runtime._report_unavailable_status()
        self.session._closed.set()
        await self.runtime._report_unavailable_status()
        self.session._closed.clear()
        self.runtime.channel.close()
        await self.runtime._report_unavailable_status()
        self.assertEqual(self.errors(), [])

    async def test_failed_notice_write_does_not_retry_or_change_send_outcome(self):
        self.degrade()
        with patch.object(self.runtime.channel, "send_route_event", side_effect=TimeoutError("private failure")) as send:
            self.assertFalse(await self.session.openai.send_event(lambda: None))
            self.assertFalse(await self.session.openai.send_event(lambda: None))
        self.assertEqual(send.await_count, 1)
        self.assertIsNone(self.runtime.failure)
        self.assertFalse(self.session.resources_closed)

    async def test_original_provider_exception_is_preserved_without_extra_status_work(self):
        self.degrade()
        with patch.object(self.runtime, "_owner_provider_send", side_effect=ValueError("fixture")):
            with self.assertRaisesRegex(ValueError, "fixture"):
                await self.session.openai.send_event(lambda: None)
        self.assertEqual(self.errors(), [])
        self.assertFalse(self.runtime._unavailability_reported)

    async def test_status_cancellation_remains_cancellation_no_retry_or_shutdown(self):
        self.degrade()
        with patch.object(self.runtime.channel, "send_route_event", side_effect=asyncio.CancelledError):
            with self.assertRaises(asyncio.CancelledError):
                await self.session.openai.send_event(lambda: None)
        self.assertTrue(self.runtime._unavailability_reported)
        self.assertIsNone(self.runtime.failure)

    async def test_projection_does_not_expose_internal_failure_reason(self):
        self.runtime.coordinator._publication_failed(self.request.route, "private context URL token")
        await self.session.openai.send_event(lambda: None)
        projected = json.dumps(self.errors())
        self.assertNotIn("private context", projected)
        self.assertNotIn("URL", projected)
        self.assertNotIn("token", projected)
        self.assertEqual(len(self.errors()), 1)
