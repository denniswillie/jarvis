"""Offline response factory diagnostics; real ordered writer, fake socket/storage."""
import asyncio
from contextlib import asynccontextmanager
import json
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

from app.realtime.openai_client import OpenAIRealtimeConnection
from app.realtime.session import RealtimeAudioSession
from app.realtime.source_context import LiveContextError
from app.realtime.telemetry import RealtimeTrace
from app.settings import JarvisSettings


class ResponseWriteGateDiagnosticsTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        instructions = patch("app.realtime.session.build_realtime_instructions",
                             return_value="PRIVATE fixture instructions")
        instructions.start()
        self.addCleanup(instructions.stop)

    def voice(self):
        voice = object.__new__(RealtimeAudioSession)
        voice.settings = JarvisSettings(OPENAI_REALTIME_WRITE_TIMEOUT_SECONDS=0.2)
        voice._source_context = SimpleNamespace(has_sources=False, prepare=AsyncMock())
        voice._closed = asyncio.Event()
        voice._context_revoked = False
        voice._turn_generation = 1
        voice._pending_response = None
        voice._voice_route_runtime = SimpleNamespace(ready=True)
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
        voice._cancel_response_start_timer = Mock()
        voice._watch_response_start = AsyncMock()
        voice._begin_context_rebuild = Mock()
        voice.openai = OpenAIRealtimeConnection(settings=voice.settings, owner_id="offline-owner")
        socket = SimpleNamespace(send=AsyncMock())
        voice.openai._websocket = socket
        reservation = voice._reserve_response_locked("normal_user_turn", {})
        voice._trace.reset_mock()

        async def cleanup():
            if voice._response_start_task is not None:
                voice._response_start_task.cancel()
                await asyncio.gather(voice._response_start_task, return_exceptions=True)
        self.addAsyncCleanup(cleanup)
        return voice, reservation, socket

    def records(self, voice, name):
        return [call.kwargs for call in voice._trace.event.call_args_list if call.args == (name,)]

    def mutate(self, voice, reason):
        if reason == "prospective_revoked":
            voice._prospective_attempt_for.return_value = SimpleNamespace(revoked=True)
        elif reason == "session_closed":
            voice._closed.set()
        elif reason == "context_revoked":
            voice._context_revoked = True
        elif reason == "reservation_replaced":
            voice._pending_response = None
        elif reason == "turn_generation_mismatch":
            voice._turn_generation += 1
        elif reason == "route_runtime_replaced":
            voice._voice_route_runtime = SimpleNamespace()
        elif reason == "provider_replaced":
            voice.openai = SimpleNamespace()
        else:
            raise AssertionError("Unknown offline mutation")

    async def test_each_guard_reason_at_preflight_prepared_factory_and_retry_boundaries(self):
        reasons = ("prospective_revoked", "session_closed", "context_revoked",
                   "reservation_replaced", "turn_generation_mismatch",
                   "route_runtime_replaced", "provider_replaced")
        boundaries = ("preparation_preflight", "prepared_preflight", "payload_at_write",
                      "source_retry_preflight")
        for boundary in boundaries:
            for reason in reasons:
                with self.subTest(boundary=boundary, reason=reason):
                    voice, reservation, socket = self.voice()
                    if boundary == "preparation_preflight":
                        def attempt(_):
                            voice._prospective_attempt_for.side_effect = None
                            self.mutate(voice, reason)
                            return voice._prospective_attempt_for.return_value
                        voice._prospective_attempt_for.side_effect = attempt
                    else:
                        voice._source_context.has_sources = True
                        async def prepare(*args, **kwargs):
                            if boundary == "prepared_preflight":
                                self.mutate(voice, reason)
                            return object()
                        voice._source_context.prepare.side_effect = prepare
                        @asynccontextmanager
                        async def guard(_):
                            self.mutate(voice, reason)
                            if boundary == "source_retry_preflight":
                                raise LiveContextError("live_context_changed")
                            yield
                        voice._source_context.guard = guard
                    self.assertFalse(await voice._send_reserved_response_locked(reservation))
                    self.assertFalse(reservation.submitted)
                    socket.send.assert_not_awaited()
                    voice._begin_context_rebuild.assert_not_called()
                    expected = dict(request_id=reservation.request_id,
                                    turn_generation=reservation.turn_generation,
                                    reason=reason, boundary=boundary)
                    self.assertEqual(self.records(voice, "response_write_gate"),
                                     [dict(expected, status="rejected")])
                    self.assertEqual(self.records(voice, "response_write_finished"),
                                     [dict(expected, status="not_submitted")])

    async def test_visual_preparation_rejection_without_requested_images(self):
        voice, reservation, socket = self.voice()
        preparation = object()
        voice.mark_visual_response_submitted.return_value = False
        self.assertFalse(await voice._send_reserved_response_locked(reservation, preparation=preparation))
        voice.mark_visual_response_submitted.assert_called_once_with(preparation, request_id=reservation.request_id)
        self.assertEqual(self.records(voice, "response_write_finished")[0]["reason"],
                         "visual_preparation_ineligible")
        self.assertFalse(reservation.submitted)
        socket.send.assert_not_awaited()

    async def test_visual_rejection_with_requested_images_still_raises(self):
        voice, reservation, socket = self.voice()
        reservation.requested_image_call_ids = ("fixture-image-call",)
        voice.mark_visual_response_submitted.return_value = False
        with self.assertRaises(Exception) as caught:
            await voice._send_reserved_response_locked(reservation, preparation=object())
        self.assertEqual(type(caught.exception).__name__, "RequestedImageDeliveryError")
        self.assertEqual(self.records(voice, "response_write_gate")[0]["reason"],
                         "visual_preparation_ineligible")
        self.assertEqual(self.records(voice, "response_write_finished"), [])
        self.assertFalse(reservation.submitted)
        socket.send.assert_not_awaited()

    async def test_success_reports_factory_admission_then_transport_write_not_playback(self):
        voice, reservation, socket = self.voice()
        self.assertTrue(await voice._send_reserved_response_locked(reservation))
        socket.send.assert_awaited_once()
        payload = json.loads(socket.send.call_args.args[0])
        self.assertEqual(payload["event_id"], reservation.request_id)
        self.assertEqual(payload["type"], "response.create")
        self.assertTrue(reservation.submitted)
        self.assertEqual(self.records(voice, "response_write_gate")[0]["reason"], "payload_prepared")
        self.assertEqual(self.records(voice, "response_write_finished")[0]["reason"],
                         "transport_write_completed")
        self.assertEqual(self.records(voice, "response_write_finished")[0]["boundary"], "transport_return")

    async def test_false_transport_without_factory_reason_does_not_invent_rejection(self):
        for invoke_factory in (False, True):
            with self.subTest(invoke_factory=invoke_factory):
                voice, reservation, socket = self.voice()
                async def transport(factory, **kwargs):
                    if invoke_factory:
                        self.assertIsNotNone(factory())
                    return False
                voice.openai.send_event = transport
                self.assertFalse(await voice._send_reserved_response_locked(reservation))
                self.assertEqual(reservation.submitted, invoke_factory)
                socket.send.assert_not_awaited()
                final = self.records(voice, "response_write_finished")[0]
                self.assertEqual(final["reason"], "transport_not_written_without_factory_reason")
                self.assertEqual(final["status"], "not_submitted")
                self.assertEqual(final["boundary"], "transport_return")
                self.assertFalse(any(row["status"] == "rejected"
                                     for row in self.records(voice, "response_write_gate")))

    async def test_control_pending_is_deferred_not_a_terminal_rejection_or_source_retry(self):
        voice, reservation, socket = self.voice()
        voice._source_context.has_sources = True
        attempts = 0
        @asynccontextmanager
        async def guard(_):
            nonlocal attempts
            attempts += 1
            voice._response_controls.busy = attempts == 1
            yield
        voice._source_context.guard = guard
        async def idle():
            voice._response_controls.busy = False
        voice._response_controls.wait_idle.side_effect = idle
        self.assertTrue(await voice._send_reserved_response_locked(reservation))
        self.assertEqual(attempts, 2)
        self.assertEqual([(row["reason"], row["status"]) for row in self.records(voice, "response_write_gate")],
                         [("response_control_pending", "deferred"), ("payload_prepared", "accepted")])
        self.assertEqual(self.records(voice, "response_source_preparation"), [])
        self.assertEqual(self.records(voice, "response_write_finished")[0]["reason"],
                         "transport_write_completed")
        socket.send.assert_awaited_once()

    async def test_control_deferral_diagnostics_are_bounded_and_do_not_hide_terminal_outcome(self):
        voice, reservation, socket = self.voice()
        voice._source_context.has_sources = True
        attempts = 0
        @asynccontextmanager
        async def guard(_):
            nonlocal attempts
            attempts += 1
            voice._response_controls.busy = attempts < 13
            yield
        voice._source_context.guard = guard
        async def idle():
            voice._response_controls.busy = False
        voice._response_controls.wait_idle.side_effect = idle
        self.assertTrue(await voice._send_reserved_response_locked(reservation))
        gates = self.records(voice, "response_write_gate")
        self.assertEqual(len(gates), 9)
        self.assertEqual(gates[-1]["status"], "truncated")
        self.assertEqual(gates[-1]["reason"], "diagnostic_limit_reached")
        self.assertEqual(self.records(voice, "response_write_finished")[0]["status"], "written")
        socket.send.assert_awaited_once()

    async def test_transport_error_after_factory_is_still_unknown_not_false_or_retried(self):
        voice, reservation, socket = self.voice()
        socket.send.side_effect = RuntimeError("PRIVATE transport failure")
        with self.assertRaises(RuntimeError):
            await voice._send_reserved_response_locked(reservation)
        self.assertTrue(reservation.submitted)
        socket.send.assert_awaited_once()
        self.assertEqual(self.records(voice, "response_write_finished"), [])
        self.assertEqual(self.records(voice, "response_source_preparation"), [])

    async def test_actual_sanitized_trace_has_static_codes_and_ids_not_instructions_or_metadata(self):
        for reject in (False, True):
            with self.subTest(reject=reject):
                voice, reservation, socket = self.voice()
                reservation.metadata["private"] = "PRIVATE_TOKEN_AND_TRANSCRIPT"
                voice._trace = RealtimeTrace(provider="openai", audio_session_id=voice.context.audio_session_id,
                                             jarvis_session_id=voice.context.jarvis_session_id)
                if reject:
                    voice._closed.set()
                with self.assertLogs("app.realtime.telemetry", level="INFO") as captured:
                    self.assertEqual(await voice._send_reserved_response_locked(reservation), not reject)
                self.assertNotIn("PRIVATE", " ".join(captured.output))
                rows = [json.loads(record.getMessage().split("realtime_trace ", 1)[1])
                        for record in captured.records]
                self.assertTrue({row["event"] for row in rows} <= {
                    "response_preparation_latency", "response_write_gate", "response_write_finished"})
                # Additional bounded latency phases do not replace either gate
                # receipt; privacy was checked over the entire captured stream.
                rows = [row for row in rows if row["event"] in {
                    "response_write_gate", "response_write_finished"}]
                self.assertEqual(len(rows), 2)
                for row in rows:
                    self.assertEqual(row["request_id"], reservation.request_id)
                    self.assertEqual(row["audio_session_id"], voice.context.audio_session_id)
                    self.assertEqual(row["jarvis_session_id"], str(voice.context.jarvis_session_id))
                    self.assertTrue(set(row) <= {"event", "schema", "process_id", "at", "monotonic_s",
                        "provider", "audio_session_id", "jarvis_session_id", "request_id", "turn_generation",
                        "reason", "boundary", "status"})

    async def test_diagnostic_sink_failure_cannot_change_admission(self):
        for reject in (False, True):
            with self.subTest(reject=reject):
                voice, reservation, socket = self.voice()
                voice._trace.event.side_effect = RuntimeError("PRIVATE logger failure")
                if reject:
                    voice._closed.set()
                self.assertEqual(await voice._send_reserved_response_locked(reservation), not reject)
                self.assertEqual(socket.send.await_count, 0 if reject else 1)
                voice._begin_context_rebuild.assert_not_called()
