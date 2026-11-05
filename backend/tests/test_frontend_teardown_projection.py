"""Offline sanitized teardown projection through actual existing parent boundaries."""
import asyncio
import io
import json
from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import patch

import httpx

from app.realtime import webrtc
from app.settings import JarvisSettings
from evals.responsive_voice.frontend_host_qualification import ChildOutputDrain, sanitized_webrtc_diagnostic
from evals.responsive_voice.native_replay_run import provider_evidence


PRIVATE = "Authorization Bearer sk-projection-private https://private.invalid/secret private-body"


def line(value):
    return "INFO app.realtime.webrtc WebRTC remote cleanup diagnostic=" + json.dumps(value)


def record(**changes):
    return {
        "stage": "hangup_http", "creation_outcome": "created", "closure_outcome": "unknown",
        "reason": "hangup_deadline_without_response", "call_id": "rtc_projection_fixture",
        "http_status": None, "creation_http_status": 201, "provider_request_id": None,
        "creation_request_id": "req_creation", "remote_closure_confirmed": True,
        "hangup_elapsed_ms": 5000.125, "hangup_total_timeout_ms": 5000, "http_phase_timeout_ms": 3000,
        "http_phase": "response_headers", "http_phase_state": "started",
        "request_headers_sent": True, "request_body_sent": True, "response_headers_received": False,
        "caller_cancelled": False, "deadline_source": "hangup_total", "exception_type": "TimeoutError",
        "response_body_outcome": "not_read", "response_close_outcome": "not_requested",
        "http_close_outcome": "closed", **changes,
    }


def assert_private_absent(test, value):
    text = json.dumps(value)
    for forbidden in ("Authorization", "Bearer", "sk-projection-private", "https://", "private-body"):
        test.assertNotIn(forbidden, text)


class TeardownProjectionTests(TestCase):
    def test_full_teardown_metadata_is_retained_without_closure_claim(self):
        source = record(body=PRIVATE, headers={"authorization": PRIVATE}, url=PRIVATE, exception_text=PRIVATE)
        result = sanitized_webrtc_diagnostic(line(source))
        for key, expected in record().items():
            with self.subTest(field=key):
                self.assertEqual(result[key], False if key=="remote_closure_confirmed" else expected)
        self.assertEqual(set(result), set(record()))
        assert_private_absent(self, result)

    def test_all_supported_progress_enums_and_teardown_reasons(self):
        fields = {
            "reason": ("call_identity_or_transport_unavailable", "request_started",
                       "request_cancelled_without_response", "request_cancelled_after_response",
                       "request_failed_without_response", "request_failed_after_response",
                       "teardown_accepted_not_confirmed_closed", "undocumented_success_status",
                       "hangup_http_failure", "hangup_deadline_without_response", "hangup_deadline_after_response"),
            "http_phase": ("not_started", "request_dispatch", "connect_tcp", "tls",
                           "request_headers", "request_body", "response_headers", "response_body"),
            "http_phase_state": ("not_started", "started", "complete", "failed"),
            "deadline_source": (None, "http_phase", "hangup_total", "cancelled"),
            "exception_type": (None, *sorted(webrtc.SAFE_TRANSPORT_EXCEPTIONS), "other"),
            "response_body_outcome": ("not_read", "not_needed", "read", "too_large", "invalid_json", "failed", "cancelled"),
            "response_close_outcome": ("not_requested", "closed", "deadline", "failed", "cancelled"),
            "http_close_outcome": ("not_requested", "borrowed", "closed", "deadline", "failed", "cancelled"),
        }
        for key, choices in fields.items():
            for candidate in choices:
                with self.subTest(field=key, value=candidate):
                    self.assertEqual(sanitized_webrtc_diagnostic(line(record(**{key:candidate})))[key], candidate)

    def test_malformed_or_private_enum_values_cannot_raise_or_escape(self):
        fields = ("stage", "creation_outcome", "closure_outcome", "reason", "http_phase", "http_phase_state",
                  "deadline_source", "exception_type", "response_body_outcome", "response_close_outcome", "http_close_outcome")
        for key in fields:
            for candidate in (PRIVATE, {"private":PRIVATE}, [PRIVATE], True, 1, None):
                with self.subTest(field=key, kind=type(candidate).__name__):
                    result = sanitized_webrtc_diagnostic(line(record(**{key:candidate})))
                    if key=="stage":
                        self.assertIsNone(result)
                    else:
                        self.assertIsNone(result[key])
                    assert_private_absent(self, result)

    def test_timing_fields_require_bounded_finite_numbers_not_booleans(self):
        for key in ("hangup_elapsed_ms", "hangup_total_timeout_ms", "http_phase_timeout_ms"):
            for candidate in (0, 1.25, 60000):
                with self.subTest(field=key, valid=candidate):
                    self.assertEqual(sanitized_webrtc_diagnostic(line(record(**{key:candidate})))[key], candidate)
            for candidate in (True, False, -1, 60000.01, 10**100, float("nan"), float("inf"),
                              -float("inf"), "3000", None, [], {}):
                with self.subTest(field=key, invalid_type=type(candidate).__name__):
                    self.assertIsNone(sanitized_webrtc_diagnostic(line(record(**{key:candidate})))[key])

    def test_progress_flags_are_exact_booleans_and_missing_is_unknown(self):
        for key in ("request_headers_sent", "request_body_sent", "response_headers_received", "caller_cancelled"):
            for candidate in (True, False):
                with self.subTest(field=key, valid=candidate):
                    self.assertIs(sanitized_webrtc_diagnostic(line(record(**{key:candidate})))[key], candidate)
            for candidate in (0, 1, "true", "false", [], {}, None):
                with self.subTest(field=key, invalid_type=type(candidate).__name__):
                    self.assertIsNone(sanitized_webrtc_diagnostic(line(record(**{key:candidate})))[key])

    def test_status_and_outcome_never_imply_confirmed_closure(self):
        for status, outcome in ((200,"accepted"), (200,"unknown"), (404,"failed"), (None,"unknown")):
            with self.subTest(status=status, outcome=outcome):
                projected = sanitized_webrtc_diagnostic(line(record(http_status=status, closure_outcome=outcome)))
                evidence = provider_evidence([projected], True)
                self.assertEqual(projected["closure_outcome"], outcome)
                self.assertFalse(projected["remote_closure_confirmed"])
                self.assertFalse(evidence["remote_closure_confirmed"])
                self.assertEqual(evidence["closure_by_call"]["rtc_projection_fixture"], outcome)
        for candidate in (True, False, 99, 600, "200"):
            with self.subTest(invalid_status=candidate):
                projected = sanitized_webrtc_diagnostic(line(record(http_status=candidate)))
                self.assertIsNone(projected["http_status"])

    def test_legacy_setup_unknown_fields_and_invalid_lines_remain_safe(self):
        source = {
            "stage":"create_http", "reason":"unexpected_http_status", "creation_outcome":"rejected",
            "closure_outcome":"not_requested", "http_status":401, "creation_http_status":401,
            "call_id": "rtc_" + PRIVATE, "provider_request_id": "req_sk-projection-private",
            "creation_request_id":"req_safe", "remote_closure_confirmed":True,
            "prompt":PRIVATE, "authorization":PRIVATE, "body":PRIVATE,
        }
        projected = sanitized_webrtc_diagnostic(line(source))
        self.assertEqual(projected["reason"], "unexpected_http_status")
        self.assertEqual(projected["http_status"], 401)
        self.assertEqual(projected["creation_request_id"], "req_safe")
        self.assertIsNone(projected["call_id"])
        self.assertIsNone(projected["provider_request_id"])
        self.assertIsNone(projected["hangup_elapsed_ms"])
        self.assertIsNone(projected["caller_cancelled"])
        self.assertFalse(projected["remote_closure_confirmed"])
        assert_private_absent(self, projected)
        for candidate in (None, b"not text", "unrelated diagnostic={}", "app.realtime.webrtc diagnostic=invalid",
                          line(record(body="x"*8192)), "app.realtime.webrtc diagnostic=[]"):
            with self.subTest(kind=type(candidate).__name__):
                self.assertIsNone(sanitized_webrtc_diagnostic(candidate))


class TeardownProjectionIntegrationTests(IsolatedAsyncioTestCase):
    async def actual_logs(self, handler):
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        self.addAsyncCleanup(client.aclose)
        connection = webrtc.OpenAIWebRTCConnection(
            settings=JarvisSettings(_env_file=None, OPENAI_API_KEY="sk-projection-private"),
            owner_id="offline-owner", offer="v=0", http=client)
        connection._owns_http = True
        connection.created_call_id = connection.call_id = "rtc_projection_fixture"
        connection.creation_outcome = "created"
        connection._creation_http_status = 201
        with self.assertLogs("app.realtime.webrtc", level="INFO") as captured:
            result = await connection.close()
        self.assertTrue(client.is_closed)
        self.assertTrue(connection._close_task.done())
        assert_private_absent(self, captured.output)
        return captured.output, result

    async def through_drain(self, lines):
        stdout, stderr = asyncio.StreamReader(), asyncio.StreamReader()
        child = SimpleNamespace(pid=4321, stdout=stdout, stderr=stderr)
        log = io.BytesIO()
        drain = ChildOutputDrain(child, log, byte_limit=256)
        payload = (b"x"*2048 + b'\nFRONTEND_HOST_READY {"pid":4321}\n' +
                   "\n".join(lines).encode() + b'\nFRONTEND_HOST_EXIT {"status":"passed"}\n')
        # Real drain/control parser and parent evidence collector; only the
        # process pipes and HTTP transport are controlled offline boundaries.
        stdout.feed_data(payload[:777]); stdout.feed_data(payload[777:]); stdout.feed_eof()
        stderr.feed_eof()
        self.assertEqual((await drain.wait_ready(timeout=1))["pid"], 4321)
        result = await drain.wait_closed(timeout=1)
        self.assertTrue(result["complete"]); self.assertTrue(result["truncated"])
        self.assertEqual(result["errors"], [])
        self.assertEqual(result["provider_diagnostics_dropped"], 0)
        self.assertLessEqual(len(log.getvalue()), 256)
        records = result["provider_diagnostics"]
        self.assertEqual(len(records), len(lines))
        assert_private_absent(self, records)
        return records, provider_evidence(records, True)

    async def test_actual_phase_timeout_survives_log_cap_and_parent_unknown_evidence(self):
        requests = []
        async def handler(request):
            requests.append(request)
            await request.extensions["trace"]("http11.send_request_headers.complete", {"secret": PRIVATE})
            await request.extensions["trace"]("http11.receive_response_headers.started", {"url": PRIVATE})
            raise httpx.ReadTimeout(PRIVATE, request=request)
        lines, transport = await self.actual_logs(handler)
        records, evidence = await self.through_drain(lines)
        final = records[-1]
        for key in ("reason","hangup_elapsed_ms","hangup_total_timeout_ms","http_phase_timeout_ms",
                    "http_phase","http_phase_state","request_headers_sent","request_body_sent",
                    "response_headers_received","caller_cancelled","deadline_source","exception_type",
                    "response_body_outcome","response_close_outcome","http_close_outcome"):
            with self.subTest(field=key):
                self.assertEqual(final[key], transport[key])
        self.assertEqual(final["deadline_source"], "http_phase")
        self.assertEqual(final["exception_type"], "ReadTimeout")
        self.assertTrue(final["request_headers_sent"])
        self.assertEqual(evidence["closure_by_call"]["rtc_projection_fixture"], "unknown")
        self.assertFalse(evidence["remote_closure_confirmed"])
        self.assertEqual(len(requests), 1)

    async def test_actual_200_headers_are_accepted_not_confirmed_by_parent(self):
        requests = []
        def handler(request):
            requests.append(request)
            return httpx.Response(200, headers={"x-request-id":"req_header"})
        lines, transport = await self.actual_logs(handler)
        records, evidence = await self.through_drain(lines)
        final = records[-1]
        self.assertEqual(final["http_status"], 200)
        self.assertEqual(final["provider_request_id"], "req_header")
        self.assertEqual(final["reason"], "teardown_accepted_not_confirmed_closed")
        self.assertTrue(final["response_headers_received"])
        self.assertEqual(final["response_body_outcome"], "not_needed")
        self.assertEqual(final["http_close_outcome"], "closed")
        self.assertEqual(final["hangup_elapsed_ms"], transport["hangup_elapsed_ms"])
        self.assertEqual(evidence["closure_by_call"]["rtc_projection_fixture"], "accepted")
        self.assertFalse(evidence["remote_closure_confirmed"])
        self.assertEqual(len(requests), 1)

    async def test_actual_headers_and_total_deadline_both_survive_projection(self):
        class StalledBody(httpx.AsyncByteStream):
            async def __aiter__(self):
                await asyncio.Event().wait()
                yield b""
            async def aclose(self):
                pass
        requests = []
        def handler(request):
            requests.append(request)
            return httpx.Response(404, headers={"x-request-id":"req_not_found"}, stream=StalledBody())
        with patch.object(webrtc, "HANGUP_TIMEOUT_SECONDS", 0.02):
            lines, transport = await self.actual_logs(handler)
        records, evidence = await self.through_drain(lines)
        final = records[-1]
        self.assertEqual(final["http_status"], 404)
        self.assertEqual(final["provider_request_id"], "req_not_found")
        self.assertEqual(final["reason"], "hangup_deadline_after_response")
        self.assertEqual(final["deadline_source"], "hangup_total")
        self.assertEqual(final["exception_type"], "TimeoutError")
        self.assertEqual(final["response_body_outcome"], "cancelled")
        self.assertEqual(final["hangup_elapsed_ms"], transport["hangup_elapsed_ms"])
        self.assertEqual(evidence["closure_by_call"]["rtc_projection_fixture"], "failed")
        self.assertFalse(evidence["remote_closure_confirmed"])
        self.assertEqual(len(requests), 1)
