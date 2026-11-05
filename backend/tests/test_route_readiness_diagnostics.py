"""Native readiness evidence and awaited-factory races, no provider/SQL I/O."""
import asyncio
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from app.realtime.response_control import ResponsePrewriteRejected
from app.voice_routes.coordinator import CoordinationPhase
from tests import test_response_route_prewrite_admission as fixtures


class RouteReadinessDiagnosticsTests(unittest.IsolatedAsyncioTestCase):
    setUp = fixtures.ResponseRoutePrewriteAdmissionTests.setUp
    voice = fixtures.ResponseRoutePrewriteAdmissionTests.voice
    records = fixtures.ResponseRoutePrewriteAdmissionTests.records

    async def delayed(self, *, before_factory, degrade):
        voice, runtime, reservation, socket = self.voice()
        entered, release = asyncio.Event(), asyncio.Event()
        async def send(factory, *, write_guard=None):
            async def delayed_factory():
                payload = None if before_factory else factory()
                entered.set()
                await release.wait()
                return factory() if before_factory else payload
            return await runtime._provider_send(delayed_factory, write_guard=write_guard)
        voice.openai.send_event = send
        task = asyncio.create_task(voice._send_reserved_response_locked(reservation))
        try:
            await asyncio.wait_for(entered.wait(), 0.5)
            self.assertEqual(reservation.submitted, not before_factory)
            if degrade:
                runtime.coordinator.status = SimpleNamespace(
                    phase=CoordinationPhase.DEGRADED, reason="context_receipt_unconfirmed")
            release.set()
            sent = await asyncio.wait_for(task, 0.5)
            self.assertEqual(sent, not degrade)
            self.assertEqual(reservation.submitted, not degrade)
            self.assertEqual(socket.send.await_count, int(not degrade))
            if degrade:
                self.assertIsNone(voice._response_start_task)
                voice._response_preparation_finished(task, reservation)
                self.assertNotIn(reservation.request_id, voice._unacknowledged_response_requests)
                self.assertEqual(self.records(voice, "response_write_finished")[0]["reason"], "route_not_ready")
            return voice
        finally:
            release.set()
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def test_readiness_flip_before_or_after_awaited_factory_never_leaves_false_submitted(self):
        for before in (False, True):
            for degrade in (False, True):
                with self.subTest(before_factory=before, degrade=degrade):
                    await self.delayed(before_factory=before, degrade=degrade)

    async def test_foreign_prewrite_rejection_cannot_clear_this_requests_unknown_state(self):
        voice, runtime, reservation, socket = self.voice()
        async def send(factory, **kwargs):
            factory()
            raise ResponsePrewriteRejected("foreign", "route_not_ready")
        voice.openai.send_event = send
        with self.assertRaises(ResponsePrewriteRejected):
            await voice._send_reserved_response_locked(reservation)
        self.assertTrue(reservation.submitted)
        socket.send.assert_not_awaited()

    async def test_readiness_observation_deduplicates_bounds_and_redacts(self):
        voice, runtime, reservation, socket = self.voice()
        with patch("app.realtime.voice_route_runtime._READINESS_LOGGER.warning") as log:
            for _ in range(10):
                self.assertTrue(runtime.ready)
            self.assertEqual(log.call_count, 1)
            runtime.coordinator.status = SimpleNamespace(phase="PRIVATE_PHASE", reason="sk-PRIVATE")
            self.assertFalse(runtime.ready)
            for _ in range(100):
                runtime._readiness_diagnostic("response_prewrite_rejected", request_id="sk-PRIVATE")
            rows = [json.loads(call.args[1]) for call in log.call_args_list]
        self.assertEqual(len(rows), 65)
        self.assertEqual(rows[-1]["stage"], "truncated")
        self.assertTrue(rows[-1]["absence_after_limit_is_inconclusive"])
        self.assertNotIn("PRIVATE", json.dumps(rows))
        self.assertEqual(rows[1]["phase"], "unrecognized_code")
        self.assertEqual(rows[1]["reason"], "unrecognized_code")
        self.assertFalse(rows[1]["context_revoked"])
        self.assertTrue(rows[1]["channel_ready"])

    async def test_logging_failure_does_not_change_readiness_or_write(self):
        voice, runtime, reservation, socket = self.voice()
        with patch("app.realtime.voice_route_runtime._READINESS_LOGGER.warning", side_effect=RuntimeError("private")):
            self.assertTrue(runtime.ready)
            self.assertTrue(await voice._send_reserved_response_locked(reservation))
        socket.send.assert_awaited_once()

