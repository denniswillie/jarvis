"""Actual native writer/runtime admission with offline socket/source fixtures."""
import asyncio
from contextlib import asynccontextmanager
from contextvars import ContextVar
import json
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

from app.realtime.openai_client import OpenAIRealtimeConnection
from app.realtime.session import RealtimeAudioSession
from app.realtime.source_context import LiveContextError
from app.realtime.voice_route_runtime import RealtimeVoiceRouteRuntime
from app.realtime.webrtc import OpenAIWebRTCConnection
from app.settings import JarvisSettings
from app.voice_routes.coordinator import CoordinationPhase


class ResponseRoutePrewriteAdmissionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        instructions = patch("app.realtime.session.build_realtime_instructions", return_value="fixture")
        instructions.start()
        self.addCleanup(instructions.stop)

    def voice(self):
        voice = object.__new__(RealtimeAudioSession)
        voice.settings = JarvisSettings(_env_file=None, OPENAI_REALTIME_WRITE_TIMEOUT_SECONDS=0.2)
        voice._source_context = SimpleNamespace(has_sources=False, prepare=AsyncMock())
        voice._closed = asyncio.Event()
        voice._context_revoked = False
        voice._turn_generation = 1
        voice._pending_response = None
        voice._response_controls = SimpleNamespace(busy=False, wait_idle=AsyncMock())
        voice._prospective_attempt_for = Mock(return_value=None)
        voice._trace = Mock()
        voice._assistant_context = None
        voice._visual_context = SimpleNamespace(active_snapshot=None)
        voice._visual_prompt_metadata = Mock(return_value=None)
        voice.mark_visual_response_submitted = Mock(return_value=True)
        voice.tool_registry = SimpleNamespace(get=Mock(return_value=None))
        voice.context = SimpleNamespace(audio_session_id=str(uuid4()), jarvis_session_id=uuid4())
        voice._response_start_task = None
        voice._response_preparation_task = None
        voice._unacknowledged_response_requests = {}
        voice._release_tool_images = Mock()
        voice._cancel_response_start_timer = Mock()
        voice._watch_response_start = AsyncMock()
        voice._begin_context_rebuild = Mock()
        # Reuse both production send wrappers without creating any HTTP client,
        # opening a provider connection, or constructing a second media owner.
        provider = object.__new__(OpenAIWebRTCConnection)
        OpenAIRealtimeConnection.__init__(provider, settings=voice.settings, owner_id="offline-owner")
        socket = SimpleNamespace(send=AsyncMock())
        provider._websocket = socket
        voice.openai = provider
        runtime = object.__new__(RealtimeVoiceRouteRuntime)
        runtime.session = voice
        runtime._installed = True
        runtime._closing = runtime._closed = False
        runtime.channel = SimpleNamespace(ready=True)
        runtime.coordinator = SimpleNamespace(status=SimpleNamespace(
            phase=CoordinationPhase.ACTIVE, reason=None))
        runtime._dispatch_context = ContextVar("offline_response_dispatch", default=None)
        runtime._owner_provider_send = provider.send_event
        # This partial runtime qualifies writer admission, not status delivery.
        # The real once-only projector has its own unavailable-projection suite.
        runtime._report_unavailable_status = AsyncMock()
        voice._voice_route_runtime = runtime
        provider.send_event = runtime._provider_send
        reservation = voice._reserve_response_locked("normal_user_turn", {})
        voice._unacknowledged_response_requests[reservation.request_id] = reservation
        voice._trace.reset_mock()

        async def cleanup():
            if voice._response_start_task is not None:
                voice._response_start_task.cancel()
                await asyncio.gather(voice._response_start_task, return_exceptions=True)
        self.addAsyncCleanup(cleanup)
        return voice, runtime, reservation, socket

    def records(self, voice, name):
        return [call.kwargs for call in voice._trace.event.call_args_list if call.args == (name,)]

    async def test_unready_runtime_rejects_before_visual_or_submitted_mutation(self):
        for phase in CoordinationPhase:
            if phase == CoordinationPhase.ACTIVE:
                continue
            with self.subTest(phase=phase.value):
                voice, runtime, reservation, socket = self.voice()
                runtime.coordinator.status = SimpleNamespace(phase=phase, reason="context_receipt_unconfirmed")
                self.assertFalse(await voice._send_reserved_response_locked(reservation, preparation=object()))
                self.assertFalse(reservation.submitted)
                voice.mark_visual_response_submitted.assert_not_called()
                socket.send.assert_not_awaited()
                self.assertIsNone(voice._response_start_task)
                voice._begin_context_rebuild.assert_not_called()
                self.assertEqual(self.records(voice, "response_write_finished")[0]["reason"], "route_not_ready")
                self.assertEqual(self.records(voice, "response_route_state")[0]["status"], phase.value)

    async def test_route_degrades_during_source_guard_and_remains_proven_unwritten(self):
        voice, runtime, reservation, socket = self.voice()
        entered, release, exited = asyncio.Event(), asyncio.Event(), asyncio.Event()
        voice._source_context.has_sources = True
        voice._source_context.prepare.return_value = object()

        @asynccontextmanager
        async def guard(_):
            entered.set()
            try:
                await release.wait()
                yield
            finally:
                exited.set()
        voice._source_context.guard = guard
        sending = asyncio.create_task(voice._send_reserved_response_locked(reservation))
        try:
            await asyncio.wait_for(entered.wait(), 0.5)
            runtime.coordinator.status = SimpleNamespace(
                phase=CoordinationPhase.DEGRADED, reason="context_publication_deadline")
            release.set()
            self.assertFalse(await asyncio.wait_for(sending, 0.5))
            self.assertTrue(exited.is_set())
            self.assertFalse(reservation.submitted)
            socket.send.assert_not_awaited()
            voice._response_preparation_finished(sending, reservation)
            self.assertNotIn(reservation.request_id, voice._unacknowledged_response_requests)
            self.assertIsNone(voice._response_start_task)
            self.assertEqual(self.records(voice, "response_route_state")[0]["reason"],
                             "context_publication_deadline")
        finally:
            release.set()
            if not sending.done():
                sending.cancel()
            await asyncio.gather(sending, return_exceptions=True)

    async def test_source_withdrawal_still_fails_closed_before_route_check(self):
        voice, runtime, reservation, socket = self.voice()
        runtime.coordinator.status.phase = CoordinationPhase.DEGRADED
        voice._source_context.has_sources = True
        voice._source_context.prepare.return_value = object()

        @asynccontextmanager
        async def guard(_):
            raise LiveContextError("live_context_revoked")
            yield
        voice._source_context.guard = guard
        with self.assertRaises(LiveContextError):
            await voice._send_reserved_response_locked(reservation)
        voice._begin_context_rebuild.assert_called_once()
        self.assertFalse(reservation.submitted)
        socket.send.assert_not_awaited()

    async def test_ready_route_writes_once_and_unknown_send_error_is_not_retried(self):
        for fail in (False, True):
            with self.subTest(fail=fail):
                voice, runtime, reservation, socket = self.voice()
                if fail:
                    socket.send.side_effect = RuntimeError("private transport detail")
                    with self.assertRaises(RuntimeError):
                        await voice._send_reserved_response_locked(reservation)
                else:
                    self.assertTrue(await voice._send_reserved_response_locked(reservation))
                socket.send.assert_awaited_once()
                self.assertTrue(reservation.submitted)
                self.assertEqual(json.loads(socket.send.call_args.args[0])["event_id"], reservation.request_id)

    async def test_unknown_phase_reason_and_logging_errors_cannot_leak_or_admit(self):
        for fail_logging in (False, True):
            with self.subTest(fail_logging=fail_logging):
                voice, runtime, reservation, socket = self.voice()
                runtime.coordinator.status = SimpleNamespace(phase="PRIVATE_PHASE", reason="PRIVATE_REASON")
                if fail_logging:
                    voice._trace.event.side_effect = RuntimeError("PRIVATE_LOGGER")
                self.assertFalse(await voice._send_reserved_response_locked(reservation))
                self.assertFalse(reservation.submitted)
                socket.send.assert_not_awaited()
                self.assertNotIn("PRIVATE", repr(voice._trace.event.call_args_list))
                state = self.records(voice, "response_route_state")[0]
                self.assertEqual((state["status"], state["reason"]), ("unknown", "unknown"))

    async def test_legacy_route_without_runtime_is_unchanged(self):
        voice, runtime, reservation, socket = self.voice()
        voice._voice_route_runtime = None
        voice.openai.send_event = runtime._owner_provider_send
        self.assertTrue(await voice._send_reserved_response_locked(reservation))
        self.assertTrue(reservation.submitted)
        socket.send.assert_awaited_once()
