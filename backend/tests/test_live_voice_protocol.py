"""Offline protocol fixtures, not live-provider, playback, or latency evidence."""
from dataclasses import asdict
from datetime import datetime, timezone
import unittest
from uuid import uuid4

from app.voice_routes.live_protocol import (
    LiveAppendKind as Kind, LiveCommandStatus as Status, LiveControlProtocol,
    LiveDelegationRequest, LiveHistoryMessage, LiveProtocolError, LiveSessionConfiguration,
    LiveSessionFinalization, LiveTranscriptFragment, parse_webrtc_session_answer,
    webrtc_session_request,
)


class LiveSessionRequestTests(unittest.TestCase):
    def test_session_has_current_clock_without_adding_domain_context(self):
        body = webrtc_session_request(
            LiveSessionConfiguration("Be factual."), "v=0", count_tokens=len,
            now=datetime(2026, 11, 4, 14, 23, tzinfo=timezone.utc),
        )
        self.assertEqual(body["session"]["instructions"],
            "Be factual.\n\nCurrent date/time at session start: "
            "2026-11-04T14:23:00+00:00 (Europe/Dublin).")
        self.assertEqual(body["session"]["input"], [])

    def test_clock_uses_local_date_and_daylight_saving_offset(self):
        body = webrtc_session_request(
            LiveSessionConfiguration(""), "v=0", count_tokens=len,
            now=datetime(2026, 7, 1, 23, 30, tzinfo=timezone.utc),
        )
        self.assertIn("2026-07-02T00:30:00+01:00 (Europe/Dublin)",
            body["session"]["instructions"])

    def test_clock_is_current_for_each_new_session_not_prior_history(self):
        history = (LiveHistoryMessage("assistant", "Today is December 16th, 2025."),)
        dates = []
        for day in (4, 5):
            body = webrtc_session_request(
                LiveSessionConfiguration("Be factual."), "v=0", count_tokens=len,
                history=history, now=datetime(2026, 11, day, tzinfo=timezone.utc),
            )
            dates.append(body["session"]["instructions"])
            self.assertEqual(body["session"]["input"][0]["content"][0]["text"], history[0].text)
        self.assertIn("2026-11-04T00:00:00", dates[0])
        self.assertIn("2026-11-05T00:00:00", dates[1])

    def test_clock_validation_and_metadata_budget(self):
        with self.assertRaisesRegex(LiveProtocolError, "invalid_session_clock"):
            webrtc_session_request(LiveSessionConfiguration(""), "v=0",
                count_tokens=len, now=datetime(2026, 11, 4))
        with self.assertRaisesRegex(LiveProtocolError, "invalid_session_timezone"):
            webrtc_session_request(LiveSessionConfiguration("", time_zone="Not/AZone"),
                "v=0", count_tokens=len)
        with self.assertRaisesRegex(LiveProtocolError, "token_budget_exceeded"):
            webrtc_session_request(LiveSessionConfiguration("x" * 16383),
                "v=0", count_tokens=len)

    def test_client_delegation_webrtc_and_no_recording(self):
        body = webrtc_session_request(LiveSessionConfiguration("Be factual."), "v=0\r\n",
            history=(LiveHistoryMessage("user", "Earlier question"),
                     LiveHistoryMessage("assistant", "Earlier generated answer")), count_tokens=len)
        self.assertEqual(body["session"]["delegation"], {"type": "client"})
        self.assertFalse(body["session"]["store"])
        self.assertEqual(body["transport"], {"type": "webrtc", "sdp": "v=0\r\n"})
        self.assertNotIn("format", body["session"]["audio"])
        self.assertEqual(body["session"]["input"][1]["content"][0]["type"], "output_text")
        self.assertNotIn("tools", body["session"])

    def test_budgets_reject_without_truncating(self):
        with self.subTest("instructions"), self.assertRaises(LiveProtocolError):
            webrtc_session_request(LiveSessionConfiguration("x" * 16385), "v=0", count_tokens=len)
        with self.subTest("history_count"), self.assertRaises(LiveProtocolError):
            webrtc_session_request(LiveSessionConfiguration(""), "v=0", count_tokens=len,
                history=tuple(LiveHistoryMessage("user", "x") for _ in range(129)))
        with self.subTest("history_tokens"), self.assertRaises(LiveProtocolError):
            webrtc_session_request(LiveSessionConfiguration(""), "v=0", count_tokens=len,
                history=(LiveHistoryMessage("user", "x" * 8193),))
        with self.subTest("sdp"), self.assertRaises(LiveProtocolError):
            webrtc_session_request(LiveSessionConfiguration(""), " " * 10, count_tokens=len)
        with self.subTest("role"), self.assertRaises(LiveProtocolError):
            webrtc_session_request(LiveSessionConfiguration(""), "v=0", count_tokens=len,
                history=(LiveHistoryMessage("tool", "untrusted output"),))

    def test_answer_preserves_opaque_id_and_encodes_url_path(self):
        answer = parse_webrtc_session_answer({"session": {"id": "live_a/b?query"},
                                              "transport": {"type": "webrtc", "sdp": "answer"}})
        self.assertEqual(answer.session_id, "live_a/b?query")
        self.assertEqual(answer.sideband_url, "wss://api.openai.com/v1/live/sessions/live_a%2Fb%3Fquery/attach")
        self.assertEqual(answer.answer_sdp, "answer")

    def test_bad_answers_rejected(self):
        for raw in (None, {}, {"session": {"id": "live_x"}, "transport": {"type": "websocket", "sdp": "x"}},
                    {"session": {"id": "live_x\n"}, "transport": {"type": "webrtc", "sdp": "x"}}):
            with self.subTest(raw=raw), self.assertRaises(LiveProtocolError):
                parse_webrtc_session_answer(raw)


class LiveControlProtocolTests(unittest.TestCase):
    def setUp(self):
        self.now = 10.0
        self.protocol = self.make()

    def make(self, *, attached=True, **limits):
        result = LiveControlProtocol(route_id=uuid4(), route_generation=1, provider_session_id="live_fixture",
            count_tokens=len, clock=lambda: self.now, **limits)
        if attached:
            result.mark_attached(authorized_session_id="live_fixture")
        return result

    def accept(self, raw, protocol=None):
        return (protocol or self.protocol).accept(raw, provider_session_id="live_fixture")

    def pending(self, kind=Kind.THINKING, **kwargs):
        command = self.protocol.prepare_append(kind, "fact", **kwargs)
        event = self.protocol.take_for_write(command.command_id)
        self.protocol.mark_written(command.command_id)
        return command, event

    def ack(self, command, *, event_id="provider_ack", **fields):
        return {"type": command.kind + "ed", "event_id": event_id, "client_event_id": command.command_id,
                "start_ms": 100, "end_ms": 200, **fields}

    def test_not_ready_before_authenticated_attach(self):
        protocol = self.make(attached=False)
        with self.assertRaises(LiveProtocolError):
            protocol.prepare_append(Kind.THINKING, "fact")
        with self.assertRaises(LiveProtocolError):
            protocol.mark_attached(authorized_session_id="foreign")
        protocol.mark_attached(authorized_session_id="live_fixture")
        self.assertTrue(protocol.ready)

    def test_all_append_kinds_require_explicit_delegation_field(self):
        for kind in Kind:
            with self.subTest(kind=kind):
                command, event = self.pending(kind)
                self.assertIn("delegation_id", event)
                self.assertIsNone(event["delegation_id"])
                result = self.accept(self.ack(command, event_id=str(uuid4())))
                self.assertEqual(result.status, Status.ACCEPTED)
                self.assertEqual(result.end_ms, 200)
                self.assertNotIn("heard", asdict(result))
                self.assertNotIn("output_id", asdict(result))

    def test_payload_can_be_taken_only_once_and_write_is_not_ack(self):
        command, event = self.pending()
        self.assertEqual(self.protocol.observation(command.command_id).status, Status.WRITTEN)
        self.assertEqual(event["content"], "fact")
        with self.assertRaises(LiveProtocolError):
            self.protocol.take_for_write(command.command_id)
        with self.assertRaises(LiveProtocolError):
            self.protocol.prepare_append(Kind.THINKING, "fact", command_id=command.command_id)

    def test_ack_before_submission_is_rejected(self):
        command = self.protocol.prepare_append(Kind.THINKING, "fact")
        with self.assertRaises(LiveProtocolError):
            self.accept(self.ack(command))
        self.assertEqual(self.protocol.observation(command.command_id).status, Status.PREPARED)

    def test_ack_can_arrive_before_writer_returns(self):
        command = self.protocol.prepare_append(Kind.THINKING, "fact")
        self.protocol.take_for_write(command.command_id)
        self.assertEqual(self.accept(self.ack(command)).status, Status.ACCEPTED)
        self.assertEqual(self.protocol.mark_written(command.command_id).status, Status.ACCEPTED)

    def test_wrong_ack_kind_and_delegation_rejected(self):
        command, _ = self.pending()
        with self.assertRaises(LiveProtocolError):
            self.accept(self.ack(command, type="session.commentary.appended"))
        with self.assertRaises(LiveProtocolError):
            self.accept(self.ack(command, event_id="ack_2", delegation_id="unknown"))
        self.assertEqual(self.protocol.observation(command.command_id).status, Status.WRITTEN)

    def test_foreign_session_never_changes_state(self):
        command, _ = self.pending()
        with self.assertRaises(LiveProtocolError):
            self.protocol.accept(self.ack(command), provider_session_id="foreign")
        self.assertEqual(self.protocol.observation(command.command_id).status, Status.WRITTEN)

    def test_timeout_is_unknown_and_late_ack_cannot_promote(self):
        command, _ = self.pending(timeout=2)
        self.now = 12
        self.assertEqual(self.protocol.expire()[0].status, Status.UNKNOWN)
        self.assertIsNone(self.accept(self.ack(command)))
        self.assertEqual(self.protocol.observation(command.command_id).status, Status.UNKNOWN)

    def test_prepared_timeout_cannot_write(self):
        command = self.protocol.prepare_append(Kind.THINKING, "fact", timeout=1)
        self.now = 11
        with self.assertRaises(LiveProtocolError):
            self.protocol.take_for_write(command.command_id)

    def test_token_and_timeout_limits(self):
        with self.assertRaises(LiveProtocolError):
            self.protocol.prepare_append(Kind.THINKING, "x" * 501)
        for timeout in (0, -1, float("nan"), float("inf"), True, 121):
            with self.subTest(timeout=timeout), self.assertRaises(LiveProtocolError):
                self.protocol.prepare_append(Kind.THINKING, "fact", timeout=timeout)
        with self.assertRaises(LiveProtocolError):
            self.protocol.prepare_append("session.thinking.append", "fact")

    def test_delegation_carries_no_inferred_task_text(self):
        raw = {"type": "session.delegation.created", "event_id": "delegation_event", "offset_ms": 123,
               "delegation": {"id": "item_fixture", "type": "delegation", "target": "client"}}
        result = self.accept(raw)
        self.assertIsInstance(result, LiveDelegationRequest)
        self.assertNotIn("text", asdict(result))
        command = self.protocol.prepare_append(Kind.COMMENTARY, "verified result", delegation_id=result.delegation_id)
        self.assertEqual(self.protocol.take_for_write(command.command_id)["delegation_id"], "item_fixture")
        self.assertIsNone(self.accept({**raw, "event_id": "same_delegation_new_event"}))

    def test_unknown_delegation_not_treated_as_session_context(self):
        with self.assertRaises(LiveProtocolError):
            self.protocol.prepare_append(Kind.COMMENTARY, "result", delegation_id="foreign")

    def test_unexpected_managed_delegation_retires_route(self):
        with self.assertRaises(LiveProtocolError):
            self.accept({"type": "session.delegation.created", "event_id": "wrong_mode", "offset_ms": 0,
                         "delegation": {"id": "item_wrong", "type": "delegation", "target": "responses"}})
        self.assertTrue(self.protocol.reconnect_required)

    def test_transcripts_preserve_spaces_repetition_and_timeline(self):
        pieces = []
        for i, delta in enumerate((" yes", " yes ")):
            item = self.accept({"type": "session.input_transcript.delta", "event_id": f"transcript_{i}",
                                "delta": delta, "start_ms": 1000 + i * 100, "end_ms": 1100 + i * 100})
            self.assertIsInstance(item, LiveTranscriptFragment)
            self.assertNotIn("final", asdict(item))
            pieces.append(item.delta)
        self.assertEqual("".join(pieces), " yes yes ")

    def test_invalid_timeline_rejected(self):
        for start, end in ((True, 5), (-1, 5), (6, 5), (0, 1.2)):
            with self.subTest(start=start, end=end), self.assertRaises(LiveProtocolError):
                self.accept({"type": "session.input_transcript.delta", "event_id": str(uuid4()),
                             "delta": "x", "start_ms": start, "end_ms": end})

    def test_duplicate_event_is_noop_but_conflicting_reuse_retires(self):
        command, _ = self.pending()
        raw = self.ack(command)
        self.accept(raw)
        self.assertIsNone(self.accept(raw))
        with self.assertRaises(LiveProtocolError):
            self.accept({**raw, "end_ms": 201})
        self.assertTrue(self.protocol.reconnect_required)

    def test_error_is_correlated_and_does_not_expose_provider_message(self):
        command, _ = self.pending()
        result = self.accept({"type": "error", "event_id": "error_1", "error": {
            "code": "invalid_request", "client_event_id": command.command_id, "message": "private fixture text"}})
        self.assertEqual(result.command_id, command.command_id)
        self.assertEqual(self.protocol.observation(command.command_id).status, Status.REJECTED)
        self.assertNotIn("private fixture text", repr(result))
        self.assertIsNone(self.accept(self.ack(command)))

    def test_uncorrelated_error_is_not_assigned_to_any_command(self):
        command, _ = self.pending()
        result = self.accept({"type": "error", "event_id": "error_2", "error": {
            "code": "private provider text with spaces", "client_event_id": "missing"}})
        self.assertIsNone(result.command_id)
        self.assertEqual(result.code, "provider_error")
        self.assertEqual(self.protocol.observation(command.command_id).status, Status.WRITTEN)

    def test_input_mute_ack_does_not_claim_output_interruption(self):
        command = self.protocol.prepare_input_mute(muted=True)
        self.protocol.take_for_write(command.command_id)
        result = self.accept({"type": "session.input_audio.muted", "event_id": "muted",
                              "client_event_id": command.command_id})
        self.assertEqual(result.status, Status.ACCEPTED)
        self.assertNotIn("output_fenced", asdict(result))
        self.assertTrue(self.protocol.ready)

    def test_close_waits_for_matching_session_finalization(self):
        pending, _ = self.pending()
        command = self.protocol.prepare_close()
        self.assertFalse(self.protocol.ready)
        with self.assertRaises(LiveProtocolError):
            self.protocol.prepare_append(Kind.THINKING, "new fact")
        self.assertEqual(self.protocol.take_for_write(command.command_id)["type"], "session.close")
        with self.assertRaises(LiveProtocolError):
            self.accept({"type": "session.closed", "event_id": "foreign_close", "session": {"id": "foreign"}})
        result = self.accept({"type": "session.closed", "event_id": "closed", "session": {"id": "live_fixture"},
                              "reason": "close_requested", "usage": {"seconds": 12}})
        self.assertIsInstance(result, LiveSessionFinalization)
        self.assertEqual(result.usage_seconds, 12)
        self.assertEqual(self.protocol.observation(command.command_id).status, Status.ACCEPTED)
        self.assertEqual(self.protocol.observation(pending.command_id).status, Status.UNKNOWN)
        self.assertNotIn("output_fenced", asdict(result))

    def test_spontaneous_close_does_not_claim_unsent_close_command_was_accepted(self):
        command = self.protocol.prepare_close()
        self.accept({"type": "session.closed", "event_id": "remote_close", "session": {"id": "live_fixture"},
                     "reason": "remote_hangup"})
        self.assertEqual(self.protocol.observation(command.command_id).status, Status.UNKNOWN)

    def test_disconnect_and_retirement_prevent_late_results_and_reuse(self):
        command, _ = self.pending()
        self.protocol.transport_lost()
        self.assertEqual(self.protocol.observation(command.command_id).status, Status.UNKNOWN)
        self.assertIsNone(self.accept(self.ack(command)))
        with self.assertRaises(LiveProtocolError):
            self.protocol.mark_attached(authorized_session_id="live_fixture")

    def test_pending_capacity_is_backpressure_not_silent_drop(self):
        protocol = self.make(max_pending=1)
        first = protocol.prepare_append(Kind.THINKING, "first")
        with self.assertRaises(LiveProtocolError):
            protocol.prepare_append(Kind.THINKING, "second")
        self.assertEqual(protocol.observation(first.command_id).status, Status.PREPARED)

    def test_command_capacity_requires_reconnection_instead_of_forgetting_ids(self):
        protocol = self.make(max_commands=1, max_pending=1)
        protocol.prepare_append(Kind.THINKING, "first", timeout=1)
        self.now += 2
        with self.assertRaises(LiveProtocolError):
            protocol.prepare_append(Kind.THINKING, "second")
        self.assertTrue(protocol.reconnect_required)

    def test_event_capacity_is_explicit(self):
        protocol = self.make(max_events=1)
        raw = {"type": "session.output_transcript.delta", "event_id": "one", "delta": "a", "start_ms": 0, "end_ms": 1}
        self.accept(raw, protocol)
        with self.assertRaises(LiveProtocolError):
            self.accept({**raw, "event_id": "two"}, protocol)
        self.assertTrue(protocol.reconnect_required)

    def test_no_audio_or_realtime_events_are_misinterpreted(self):
        for kind in ("session.output_audio.delta", "response.done", "response.created", "session.started"):
            with self.subTest(kind=kind):
                self.assertIsNone(self.accept({"type": kind, "event_id": str(uuid4())}))

    def test_empty_thinking_is_supported_without_silence_guarantee(self):
        command = self.protocol.prepare_append(Kind.THINKING, "")
        self.assertEqual(self.protocol.take_for_write(command.command_id)["content"], "")
        result = self.accept(self.ack(command))
        self.assertEqual(result.status, Status.ACCEPTED)
        self.assertNotIn("silent", asdict(result))
