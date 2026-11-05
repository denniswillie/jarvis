"""Bounded native-control diagnostics through real owner/runtime/writer seams.

Provider receipts and persistence are offline fixtures. These tests establish no
provider, playback, hearing, or historical failed-run causal claims.
"""

import asyncio
from contextlib import asynccontextmanager
from datetime import timedelta
import json
import unittest
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from app.live_state.models import SessionScope, StateCursor
from app.realtime import direct_session
from app.realtime.session import RealtimeAudioSession
from app.realtime.voice_route_runtime import RealtimeVoiceRouteRuntime
from app.voice_routes.contracts import CommandReceipt, RouteIdentity, RouteStart, VoiceContextUpdate
from tests import test_realtime_auto_vad_request_evidence as capture_fixture
from tests import test_voice_route_runtime as route_fixture


class NativeControlDiagnosticsTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.harness = capture_fixture.NativeAutoVADRequestEvidenceTests(
            "test_native_auto_vad_binds_exact_final_request_through_tool_executor")
        await self.harness.asyncSetUp()
        self.addAsyncCleanup(self.harness.asyncTearDown)
        logger_patch = patch.object(direct_session._NATIVE_CONTROL_LOGGER, "warning")
        self.log = logger_patch.start()
        self.addCleanup(logger_patch.stop)
        session = self.session = self.harness.session
        session._realtime_context_ready.clear()
        self.provider = self.harness.provider = session.openai = route_fixture.Provider()
        self.provider.capture_mode = "auto"
        self.provider.events = self.provider.writes
        self.provider.item_event = lambda payload: session._handle_openai_event(payload)
        session._send_app_event = RealtimeAudioSession._send_app_event.__get__(session)
        self.outbound = []

        async def record(payload):
            self.outbound.append(payload)
            if payload["type"] == "voice_route.selected":
                await session._handle_client_event({"type": "voice_route.ready", "protocol_version": 1,
                    "audio_session_id": payload["audio_session_id"], "route": payload["route"]})
            elif payload["type"] in ("voice_route.interrupt", "voice_route.close"):
                command = payload["payload"]
                receipt = CommandReceipt(route=command["route"], command_id=command["command_id"],
                    operation="close" if payload["type"] == "voice_route.close" else "interrupt_speech",
                    observed_at=route_fixture.NOW, status="completed", output_fenced=True)
                await session._handle_client_event({"type": "voice_route.event", "protocol_version": 1,
                    "audio_session_id": payload["audio_session_id"], "payload": receipt.model_dump(mode="json")})

        async def text_out(text):
            await record(json.loads(text))

        session._app_sender._websocket.send_text = AsyncMock(side_effect=text_out)
        session._app_sender._websocket.send_json = AsyncMock(side_effect=record)
        self.route = RouteIdentity(scope=SessionScope(owner_id=session.context.owner_id,
            session_id=session.context.jarvis_session_id, area_id=None), route_id=uuid4(), route_generation=1)
        context = VoiceContextUpdate(route=self.route, update_id=uuid4(),
            cursor=StateCursor(epoch=uuid4(), revision=0), mode="snapshot",
            text="No source-specific context is selected.", dependencies=(),
            created_at=route_fixture.NOW, expires_at=route_fixture.NOW + timedelta(minutes=5))
        request = RouteStart(route=self.route, command_id=uuid4(), requested_at=route_fixture.NOW,
            authorization_ref=uuid4(), initial_context=context)

        @asynccontextmanager
        async def atomic(_):
            yield

        async def valid(_):
            return True

        self.runtime = RealtimeVoiceRouteRuntime(session, request, route_fixture.CLIENT_CAPS,
            atomic_guard=atomic, payload_roots=lambda _: (), validity=valid,
            authorize_start=valid, on_event=lambda _: None, clock=lambda: route_fixture.NOW,
            ready_timeout=1, control_timeout=0.2)
        session.install_voice_route_runtime(self.runtime)
        session._realtime_context_ready.set()
        await self.harness.until(lambda: self.runtime.ready or self.runtime.failure is not None)
        self.assertTrue(self.runtime.ready, self.runtime.failure)

    def records(self, *, stage=None, event_type=None):
        records = [json.loads(call.args[1]) for call in self.log.call_args_list]
        return [record for record in records if (stage is None or record["stage"] == stage)
            and (event_type is None or record.get("event_type") == event_type)]

    async def response(self):
        await self.harness.start("fixture-input")
        await self.harness.commit_and_ack_response("fixture-input", "resp_fixture1")

    async def output_item(self, *, response_id="resp_fixture1", **extra):
        await self.harness.event("response.output_item.added", response_id=response_id, output_index=0,
            item={"id": "fixture-item", "type": "message", "phase": "final_answer", **extra})

    async def test_healthy_actual_owner_runtime_and_writer_keep_exact_envelopes(self):
        await self.response()
        await self.output_item(content=[{"text": "PRIVATE_TRANSCRIPT"}])
        await self.harness.event("output_audio_buffer.started", response_id="resp_fixture1")
        await self.harness.event("output_audio_buffer.stopped", response_id="resp_fixture1")
        media = [p for p in self.outbound if p["type"].startswith("audio.webrtc.")]
        self.assertEqual([p["type"] for p in media], ["audio.webrtc.response", "audio.webrtc.output_item",
            "audio.webrtc.started", "audio.webrtc.stopped"])
        for payload in media:
            self.assertEqual(payload["protocol_version"], 1)
            self.assertEqual(payload["audio_session_id"], self.session.context.audio_session_id)
            self.assertEqual(payload["route"], self.route.model_dump(mode="json"))
        self.assertTrue(self.records(stage="provider_handler_received", event_type="response.output_item.added"))
        rows = self.records(stage="runtime_send_result")
        self.assertEqual(len(rows), 4)
        self.assertTrue(all(row["send_result"] is True and row["runtime_ready"] for row in rows))
        self.assertTrue(all(row["coordinator_phase"] == "active" for row in rows))
        self.assertNotIn("PRIVATE_TRANSCRIPT", str(self.log.call_args_list))
        self.harness.assert_no_explicit_commit()

    async def test_degraded_route_records_new_media_rejection_and_end_control_success(self):
        await self.response()
        self.runtime.coordinator._publication_failed(self.route, "context_receipt_unconfirmed")
        await self.output_item()
        await self.harness.event("output_audio_buffer.started", response_id="resp_fixture1")
        await self.harness.event("output_audio_buffer.stopped", response_id="resp_fixture1")
        await self.harness.event("output_audio_buffer.cleared", response_id="resp_fixture1")
        rejected = self.records(stage="runtime_rejected")
        self.assertEqual(len(rejected), 2)
        self.assertEqual({row["event_type"] for row in rejected},
                         {"audio.webrtc.output_item", "audio.webrtc.started"})
        for row in rejected:
            self.assertFalse(row["runtime_ready"])
            self.assertEqual(row["coordinator_phase"], "degraded")
            self.assertEqual(row["coordinator_reason"], "context_receipt_unconfirmed")
            self.assertEqual(row["reason"], "runtime_not_ready")
            self.assertIs(row["send_result"], False)
        self.assertEqual(len(self.records(stage="runtime_send_result")), 3)
        for kind in ("audio.webrtc.stopped", "audio.webrtc.cleared"):
            sent = self.records(stage="runtime_send_result", event_type=kind)
            self.assertEqual(len(sent), 1)
            self.assertIs(sent[0]["send_result"], True)
            self.assertFalse(sent[0]["runtime_ready"])
            self.assertEqual(sent[0]["coordinator_phase"], "degraded")
        self.assertTrue(self.records(stage="direct_admitted", event_type="audio.webrtc.output_item"))
        self.assertFalse(self.records(stage="direct_send_result", event_type="audio.webrtc.output_item")[0]["send_result"])
        self.assertTrue(await self.session._send_app_event("transcript.done", text="PRIVATE_TEXT"))

    async def test_direct_precheck_rejection_is_distinct_from_runtime_rejection(self):
        await self.harness.event("output_audio_buffer.started", response_id="resp_unknown1")
        await self.output_item(response_id="resp_unknown1")
        rejected = self.records(stage="direct_precheck_rejected")
        self.assertEqual(len(rejected), 2)
        self.assertTrue(all(row["reason"] == "response_not_pending" for row in rejected))
        self.assertEqual(self.records(stage="runtime_attempt"), [])

    async def assert_context_rejection_reason(self, reason):
        await self.response()
        self.runtime.coordinator._publication_failed(self.route, reason)
        await self.output_item()
        rows = self.records(stage="runtime_rejected", event_type="audio.webrtc.output_item")
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["coordinator_reason"], reason)
        self.assertEqual(row["coordinator_phase"], "degraded")
        self.assertEqual(row["reason"], "runtime_not_ready")
        self.assertIs(row["runtime_ready"], False)
        self.assertIs(row["send_result"], False)
        self.assertEqual(len(self.records(stage="runtime_send_result")), 1)

    async def test_context_validity_rejection_is_not_redacted_as_unknown(self):
        await self.assert_context_rejection_reason("context_validity_rejected")

    async def test_context_expiry_before_write_is_not_redacted_as_unknown(self):
        await self.assert_context_rejection_reason("context_expired_before_write")

    async def test_known_context_reason_prefix_does_not_allow_private_suffix(self):
        self.runtime.coordinator._publication_failed(self.route, "context_validity_rejected")
        for reason in ("context_validity_rejected", "context_expired_before_write"):
            with self.subTest(reason=reason):
                private = reason + ": PRIVATE_ERROR Bearer sk-secret SDP transcript"
                with patch.object(self.runtime.coordinator, "_reason", private):
                    self.session._native_control_diagnostic(
                        "runtime_rejected", "audio.webrtc.started", "resp_fixture1",
                        reason="runtime_not_ready", send_result=False)
                row = self.records(stage="runtime_rejected")[-1]
                self.assertEqual(row["coordinator_reason"], "unrecognized_code")
                self.assertEqual(row["reason"], "runtime_not_ready")
                self.assertNotIn(private, str(self.log.call_args_list))
                self.assertNotIn("PRIVATE_ERROR", str(self.log.call_args_list))

    async def test_runtime_only_attempt_does_not_fabricate_provider_receipt(self):
        self.assertTrue(await self.session._send_native_control("audio.webrtc.stopped", response_id="resp_fixture1"))
        self.assertEqual(self.records(stage="provider_handler_received"), [])
        self.assertTrue(self.records(stage="runtime_send_result")[0]["send_result"])

    async def test_writer_recheck_records_degradation_without_changing_guard(self):
        original = self.runtime._owner_send_text

        async def degrade_before_write(text, **kwargs):
            self.runtime.coordinator._publication_failed(self.route, "context_publication_deadline")
            return await original(text, **kwargs)

        self.runtime._owner_send_text = degrade_before_write
        self.assertFalse(await self.session._send_native_control("audio.webrtc.started", response_id="resp_fixture1"))
        rows = self.records(stage="runtime_write_guard_rejected")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["reason"], "runtime_not_ready")
        self.assertFalse(self.records(stage="runtime_send_result")[0]["send_result"])

    async def test_owner_guard_and_actual_false_result_are_not_success(self):
        event = json.dumps({"type": "audio.webrtc.started", "response_id": "resp_fixture1"})
        guard = unittest.mock.Mock(return_value=False)
        self.assertFalse(await self.runtime._app_send_text(event, should_send=guard))
        guard.assert_called_once_with()
        self.assertEqual(self.records(stage="runtime_write_guard_rejected")[0]["reason"], "owner_send_guard")
        self.assertFalse(self.records(stage="runtime_send_result")[0]["send_result"])

    async def test_write_exception_preserves_exception_and_unknown_outcome_without_error_text(self):
        error = RuntimeError("PRIVATE_ERROR sk-secret Authorization SDP")
        self.runtime._owner_send_text = AsyncMock(side_effect=error)
        with self.assertRaises(RuntimeError) as caught:
            await self.session._send_native_control("audio.webrtc.started", response_id="resp_fixture1")
        self.assertIs(caught.exception, error)
        for stage in ("runtime_send_raised", "direct_send_raised"):
            row = self.records(stage=stage)[0]
            self.assertIsNone(row["send_result"])
            self.assertEqual(row["reason"], "write_exception")
        self.assertNotIn("PRIVATE_ERROR", str(self.log.call_args_list))
        self.assertEqual(self.records(stage="runtime_send_result"), [])

    async def test_cancellation_is_propagated_and_never_claimed_not_sent(self):
        original_writer = self.runtime._owner_send_text
        self.runtime._owner_send_text = AsyncMock(side_effect=asyncio.CancelledError())
        try:
            with self.assertRaises(asyncio.CancelledError):
                await self.session._send_native_control("audio.webrtc.started", response_id="resp_fixture1")
        finally:
            # The injected cancellation is scoped to this write, not the
            # session-close drain that the shared fixture performs afterward.
            self.runtime._owner_send_text = original_writer
        self.assertIs(self.runtime._owner_send_text, original_writer)
        for stage in ("runtime_send_raised", "direct_send_raised"):
            row = self.records(stage=stage)[0]
            self.assertEqual(row["reason"], "write_cancelled")
            self.assertIsNone(row["send_result"])

    async def test_private_values_are_redacted_and_arbitrary_event_types_are_not_logged(self):
        private = "PRIVATE_SECRET Bearer sk-secret v=0 SDP transcript"
        self.runtime.coordinator._publication_failed(self.route, private)
        self.session._native_control_diagnostic("runtime_attempt", "audio.webrtc.started", private,
            reason=private, send_result=private)
        row = self.records()[0]
        self.assertIsNone(row["response_id"])
        self.assertTrue(row["response_id_redacted"])
        self.assertEqual(row["reason"], "unrecognized_code")
        self.assertEqual(row["coordinator_reason"], "unrecognized_code")
        self.assertIsNone(row["send_result"])
        self.session._native_control_diagnostic("provider_handler_received", private, "resp_fixture1")
        self.assertEqual(len(self.records()), 1)
        self.assertNotIn(private, str(self.log.call_args_list))

    async def test_fixed_session_bound_emits_one_marker_without_disabling_control(self):
        for _ in range(direct_session._NATIVE_CONTROL_LOG_LIMIT + 20):
            self.session._native_control_diagnostic("provider_handler_received", "response.created", "resp_fixture1")
        self.assertEqual(len(self.records()), direct_session._NATIVE_CONTROL_LOG_LIMIT + 1)
        self.assertEqual(len(self.records(stage="truncated")), 1)
        self.assertTrue(self.records(stage="truncated")[0]["absence_after_limit_is_inconclusive"])
        self.assertTrue(await self.session._send_native_control("audio.webrtc.stopped", response_id="resp_fixture1"))
        self.assertEqual(len(self.records()), direct_session._NATIVE_CONTROL_LOG_LIMIT + 1)
        self.assertEqual(self.outbound[-1]["type"], "audio.webrtc.stopped")

    async def test_logging_failure_does_not_change_send(self):
        self.log.side_effect = RuntimeError("PRIVATE_LOGGER_FAILURE")
        self.assertTrue(await self.session._send_native_control("audio.webrtc.stopped", response_id="resp_fixture1"))
        self.assertEqual(self.outbound[-1]["type"], "audio.webrtc.stopped")
