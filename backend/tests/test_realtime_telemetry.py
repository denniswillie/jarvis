from __future__ import annotations

import asyncio
from datetime import UTC, datetime
import json
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock
from uuid import uuid4

from fastapi import WebSocketDisconnect

from app.realtime.socket_sender import WebSocketEventSender
from app.realtime.telemetry import MAX_RECENT_AUDIO_ITEMS, RealtimeTrace, safe_fields
from evals.glasses_reliability.baselines import CASES
from evals.glasses_reliability.report import summarize
from test_realtime_normal_close import _transport_session
import test_gemini_audio_session as gemini_fixture


def trace():
    return RealtimeTrace(provider="gemini", audio_session_id="fixture-audio", jarvis_session_id=uuid4())


def records(logs):
    return [json.loads(record.getMessage().split("realtime_trace ", 1)[1]) for record in logs.records]


class RealtimeTelemetryTests(unittest.IsolatedAsyncioTestCase):
    async def test_first_audio_only_after_success_and_once_per_item_on_both_encodings(self):
        sink = SimpleNamespace(send_json=AsyncMock(), send_text=AsyncMock())
        sender = WebSocketEventSender(sink, trace=trace())
        payload = {"type": "audio.output.delta", "response_id": "r1", "item_id": "i1", "audio_base64": "PRIVATE_AUDIO"}
        with self.assertLogs("app.realtime.telemetry", "INFO") as logs:
            await sender.send_event(payload)
            await sender.send_text(json.dumps(payload))
            await sender.send_event({**payload, "item_id": "i2"})
        rows = records(logs)
        self.assertEqual([r["item_id"] for r in rows], ["i1", "i2"])
        self.assertEqual({r["provider"] for r in rows}, {"gemini"})
        self.assertTrue(all(r["audio_session_id"] == "fixture-audio" and r["monotonic_s"] > 0 for r in rows))
        self.assertNotIn("PRIVATE_AUDIO", str(rows))
        sink.send_json.assert_any_await(payload)

    async def test_departure_and_retired_writes_never_claim_forwarding(self):
        sink = SimpleNamespace(send_json=AsyncMock(side_effect=WebSocketDisconnect()))
        sender = WebSocketEventSender(sink, trace=trace())
        with self.assertLogs("app.realtime.telemetry", "INFO") as logs:
            for _ in range(3):
                await sender.send_event({"type": "audio.output.delta", "response_id": "r", "audio_base64": "AA=="})
        self.assertEqual([r["event"] for r in records(logs)], ["socket_write_failed"])
        sink.send_json.assert_awaited_once()

    async def test_cancelled_write_is_unknown_not_forwarded_and_still_raises(self):
        sink = SimpleNamespace(send_json=AsyncMock(side_effect=asyncio.CancelledError()))
        sender = WebSocketEventSender(sink, trace=trace())
        with self.assertLogs("app.realtime.telemetry", "INFO") as logs:
            with self.assertRaises(asyncio.CancelledError):
                await sender.send_event({"type": "audio.output.delta", "response_id": "r", "audio_base64": "AA=="})
        self.assertEqual(records(logs)[0]["reason"], "write_outcome_unknown")
        self.assertFalse(sender.closed)

    def test_redaction_and_finite_bounded_fields(self):
        result = safe_fields({"audio_session_id": "fixture", "frame_id": "https://private/image?token=secret",
                              "reason": "Bearer private", "item_id": "x" * 161, "duration_ms": float("nan"),
                              "transcript": "PRIVATE", "audio_base64": "PRIVATE", "image": {"url": "PRIVATE"},
                              "authorization": "PRIVATE", "api_key": "PRIVATE", "cutoff_at": "PRIVATE",
                              "captured_at": datetime(2026, 1, 1, tzinfo=UTC)})
        self.assertEqual(set(result), {"audio_session_id", "captured_at"})
        self.assertEqual(safe_fields({"audio_end_ms": 10 ** 1000}), {})
        recorder = trace()
        recorder.client_event({"type": []})
        recorder.forwarded({"type": []})

    def test_long_session_state_is_bounded_and_missing_identity_is_not_measured(self):
        recorder = trace()
        with self.assertLogs("app.realtime.telemetry", "INFO"):
            for index in range(MAX_RECENT_AUDIO_ITEMS + 3):
                recorder.forwarded({"type": "audio.output.delta", "response_id": f"r{index}", "audio_base64": "AA=="})
        self.assertEqual(len(recorder._audio_items), MAX_RECENT_AUDIO_ITEMS)
        with self.assertNoLogs("app.realtime.telemetry", "INFO"):
            recorder.forwarded({"type": "audio.output.delta", "audio_base64": "AA=="})

    async def test_gpt_production_control_and_forwarding_are_correlated_without_payload_changes(self):
        phone = SimpleNamespace(send_text=AsyncMock(), close=AsyncMock())
        session = _transport_session(phone)
        session.openai = SimpleNamespace(send_event=AsyncMock(return_value=True), close=AsyncMock())
        try:
            with self.assertLogs("app.realtime.telemetry", "INFO") as logs:
                await session._handle_client_event({"type": "audio.input.commit", "segment_id": "segment"})
                await session._handle_client_event({"type": "response.cancel"})
                await session._send_app_event("audio.output.delta", response_id="response", item_id="item", audio_base64="AA==")
                await session._request_normal_response(turn_generation=-1)
            rows = records(logs)
            self.assertTrue({"input_commit_received", "input_commit_admitted", "response_cancel_requested",
                             "first_audio_forwarded", "response_rejected"}.issubset({r["event"] for r in rows}))
            self.assertEqual({r["provider"] for r in rows}, {"openai"})
            self.assertEqual({r["audio_session_id"] for r in rows}, {"audio"})
        finally:
            await session.close("test_complete")

    def test_regression_manifest_and_honest_report(self):
        self.assertEqual([case.case_id for case in CASES], [f"R{i:02}" for i in range(1, 12)])
        self.assertTrue(all(case.oracle and case.limitation for case in CASES))
        report = summarize(["realtime_trace invalid", "not a trace"])
        self.assertEqual(report["malformed_traces"], 1)
        self.assertIsNone(report["hardware_first_audible_ms"])

    def test_report_correlates_requests_but_never_subtracts_cross_process_clocks(self):
        def line(event, clock, **fields):
            return "realtime_trace " + json.dumps({"schema": 1, "event": event, "monotonic_s": clock,
                                                   "process_id": "server", "audio_session_id": "a", **fields})
        report = summarize([
            line("response_write_finished", 10, request_id="q", status="written"),
            line("provider_response_started", 11, request_id="q", response_id="r"),
            line("first_audio_forwarded", 12, response_id="r", item_id="i"),
            line("first_audio_forwarded", 13, response_id="r", item_id="i"),
            line("provider_response_started", 1, process_id="other", response_id="r"),
        ])
        self.assertEqual(report["distinct_audio_items_forwarded"], 1)
        self.assertEqual(report["responses"][0]["response_write_to_first_forwarded_ms"], 2000)
        self.assertEqual(report["responses"][0]["provider_start_to_first_forwarded_ms"], 1000)
        self.assertIsNone(report["responses"][1]["provider_start_to_first_forwarded_ms"])

    async def test_gemini_real_dispatcher_traces_unavailable_image_audio_and_reset(self):
        case = gemini_fixture.GeminiAudioSessionTests()
        await case.asyncSetUp()
        try:
            with self.assertLogs("app.realtime.telemetry", "INFO") as logs:
                await case.start()
                await case.append()
                await case.commit()
                await case.answer()
                await gemini_fixture.until(lambda: bool(case.events("response.done")))
                await case.session._reset_before_new_input()
            rows = records(logs)
            self.assertTrue({"input_commit_admitted", "preparation_started", "preparation_finished",
                             "image_selected", "image_prewrite", "response_admitted", "response_write_finished",
                             "provider_response_started", "first_audio_forwarded", "provider_response_completed",
                             "provider_reset_started", "provider_reconnected_forwarded"}.issubset({r["event"] for r in rows}))
            admitted = next(r for r in rows if r["event"] == "response_admitted")
            audio = next(r for r in rows if r["event"] == "first_audio_forwarded")
            self.assertEqual(admitted["response_id"], audio["response_id"])
            self.assertEqual(admitted["segment_id"], "a")
            self.assertEqual(case.session._turns["a"].evidence_status, "unavailable")
            self.assertEqual(next(r for r in rows if r["event"] == "provider_reconnected_forwarded")["connection_generation"], 2)
        finally:
            await case.asyncTearDown()

    async def test_gpt_disabled_visual_preparation_still_writes_and_records_outcome(self):
        session = _transport_session(SimpleNamespace(send_text=AsyncMock(), close=AsyncMock()))
        session.openai = SimpleNamespace(send_event=AsyncMock(return_value=True), close=AsyncMock())
        session.prepare_visual_response = AsyncMock(return_value=None)
        try:
            with self.assertLogs("app.realtime.telemetry", "INFO") as logs:
                reservation = session._reserve_response_locked("normal_user_turn", {})
                self.assertTrue(await session._prepare_and_send_response(reservation, None, None))
            rows = records(logs)
            self.assertEqual(next(r for r in rows if r["event"] == "image_context_prepared")["status"], "disabled")
            self.assertEqual(next(r for r in rows if r["event"] == "preparation_finished")["status"], "written")
        finally:
            await session.close("test_complete")
