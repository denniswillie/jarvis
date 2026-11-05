"""Offline hangup progress: actual HTTPX streaming with controlled transport only."""
import asyncio
import json
from unittest import IsolatedAsyncioTestCase
from unittest.mock import patch

import httpx

from app.realtime import webrtc
from app.settings import JarvisSettings


KEY = "sk-teardown-offline-secret"
PRIVATE = "private-body Authorization " + KEY


class Body(httpx.AsyncByteStream):
    def __init__(self, chunks=(), *, stalled=False, close_stalled=False):
        self.chunks = chunks
        self.stalled = stalled
        self.read_started = asyncio.Event()
        self.close_started = asyncio.Event()
        self.close_release = asyncio.Event()
        if not close_stalled:
            self.close_release.set()
        self.close_count = 0

    async def __aiter__(self):
        self.read_started.set()
        for chunk in self.chunks:
            yield chunk
        if self.stalled:
            await asyncio.Event().wait()

    async def aclose(self):
        self.close_count += 1
        self.close_started.set()
        await self.close_release.wait()


class TeardownProgressTests(IsolatedAsyncioTestCase):
    def connection(self, handler):
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        self.addAsyncCleanup(client.aclose)
        connection = webrtc.OpenAIWebRTCConnection(
            settings=JarvisSettings(_env_file=None, OPENAI_API_KEY=KEY),
            owner_id="offline-owner", offer="v=0", http=client)
        connection._owns_http = True
        # Known synthetic identity; no creation or sideband request is performed.
        connection.created_call_id = connection.call_id = "rtc_teardown_fixture"
        connection.creation_outcome = "created"
        connection._creation_http_status = 201
        return connection, client

    async def wait(self, event):
        await asyncio.wait_for(event.wait(), 0.5)

    def safe(self, value):
        text = value if isinstance(value, str) else json.dumps(value)
        for forbidden in (KEY, "Authorization", "private-body", "https://", "api.openai.com"):
            self.assertNotIn(forbidden, text)
        self.assertLess(len(text), 20000)

    async def test_success_headers_survive_stalled_unread_body_and_caller_cancellation(self):
        body = Body(stalled=True, close_stalled=True)
        requests = []
        async def handler(request):
            requests.append(request)
            self.assertEqual(request.extensions["timeout"], dict(connect=3.0, read=3.0, write=3.0, pool=3.0))
            return httpx.Response(200, headers={"x-request-id": "req_headers_first"}, stream=body)
        connection, client = self.connection(handler)
        with self.assertLogs("app.realtime.webrtc", level="INFO") as logs:
            caller = asyncio.create_task(connection.close())
            await self.wait(body.close_started)
            self.assertFalse(body.read_started.is_set())
            self.assertEqual(connection.closure_diagnostic["http_status"], 200)
            self.assertEqual(connection.closure_diagnostic["provider_request_id"], "req_headers_first")
            self.assertFalse(connection.closure_diagnostic["remote_closure_confirmed"])
            caller.cancel()
            await asyncio.sleep(0)
            self.assertFalse(caller.done())
            self.assertFalse(client.is_closed)
            body.close_release.set()
            with self.assertRaises(asyncio.CancelledError):
                await caller
        result = await connection.close()
        self.assertEqual(result["closure_outcome"], "accepted")
        self.assertTrue(result["caller_cancelled"])
        self.assertEqual(result["response_body_outcome"], "not_needed")
        self.assertEqual(result["response_close_outcome"], "closed")
        self.assertEqual(result["http_close_outcome"], "closed")
        self.assertTrue(result["response_headers_received"])
        self.assertEqual(len(requests), 1)
        self.assertEqual(body.close_count, 1)
        self.assertGreater(result["hangup_total_timeout_ms"], result["http_phase_timeout_ms"])
        self.safe("\n".join(logs.output))

    async def test_error_headers_survive_total_deadline_during_optional_body(self):
        body = Body(stalled=True)
        requests = []
        def handler(request):
            requests.append(request)
            return httpx.Response(404, headers={"x-request-id": "req_not_found"}, stream=body)
        connection, client = self.connection(handler)
        with patch.object(webrtc, "HANGUP_TIMEOUT_SECONDS", 0.02):
            caller = asyncio.create_task(connection.close())
            await self.wait(body.read_started)
            self.assertEqual(connection.closure_diagnostic["http_status"], 404)
            self.assertEqual(connection.closure_diagnostic["provider_request_id"], "req_not_found")
            result = await asyncio.wait_for(caller, 0.5)
        self.assertEqual(result["http_status"], 404)
        self.assertEqual(result["closure_outcome"], "failed")
        self.assertEqual(result["reason"], "hangup_deadline_after_response")
        self.assertEqual(result["deadline_source"], "hangup_total")
        self.assertEqual(result["response_body_outcome"], "cancelled")
        self.assertFalse(result["remote_closure_confirmed"])
        self.assertTrue(client.is_closed)
        self.assertEqual(result, await connection.close())
        self.assertEqual(len(requests), 1)
        self.safe(result)

    async def test_phase_timeouts_are_distinct_from_total_deadline_without_retry(self):
        for error_type in (httpx.ConnectTimeout, httpx.ReadTimeout, httpx.WriteTimeout, httpx.PoolTimeout):
            with self.subTest(error_type=error_type.__name__):
                requests = []
                def handler(request):
                    requests.append(request)
                    raise error_type(PRIVATE, request=request)
                connection, client = self.connection(handler)
                result = await connection.close()
                self.assertEqual(result["exception_type"], error_type.__name__)
                self.assertEqual(result["deadline_source"], "http_phase")
                self.assertEqual(result["closure_outcome"], "unknown")
                self.assertIsNone(result["http_status"])
                self.assertFalse(result["response_headers_received"])
                self.assertEqual(len(requests), 1)
                self.assertTrue(client.is_closed)
                self.assertEqual(result, await connection.close())
                self.safe(result)

    async def test_total_deadline_preserves_last_transport_phase_and_write_receipts(self):
        requests = []
        async def handler(request):
            requests.append(request)
            trace = request.extensions["trace"]
            for name in ("connection.connect_tcp.complete", "connection.start_tls.complete",
                         "http11.send_request_headers.complete", "http11.send_request_body.complete",
                         "http11.receive_response_headers.started"):
                await trace(name, {"secret": PRIVATE, "url": str(request.url)})
            await asyncio.Event().wait()
        connection, client = self.connection(handler)
        with patch.object(webrtc, "HANGUP_TIMEOUT_SECONDS", 0.02):
            result = await asyncio.wait_for(connection.close(), 0.5)
        self.assertEqual(result["reason"], "hangup_deadline_without_response")
        self.assertEqual(result["deadline_source"], "hangup_total")
        self.assertEqual(result["http_phase"], "response_headers")
        self.assertEqual(result["http_phase_state"], "started")
        self.assertTrue(result["request_headers_sent"])
        self.assertTrue(result["request_body_sent"])
        self.assertFalse(result["response_headers_received"])
        self.assertFalse(result["remote_closure_confirmed"])
        self.assertEqual(result["exception_type"], "TimeoutError")
        self.assertGreaterEqual(result["hangup_elapsed_ms"], 0)
        self.assertTrue(client.is_closed)
        self.assertEqual(len(requests), 1)
        self.safe(result)

    async def test_error_body_is_bounded_and_only_allowlisted_metadata_survives(self):
        cases = (
            (json.dumps({"error": {"code": "call_not_found", "param": "model", "message": PRIVATE}}).encode(), "read", "call_not_found"),
            (json.dumps({"error": {"code": KEY, "param": PRIVATE}}).encode(), "read", "[redacted]"),
            (b"x" * (webrtc.MAX_ERROR_METADATA_BYTES + 1), "too_large", None),
            (PRIVATE.encode(), "invalid_json", None),
        )
        for payload, outcome, code in cases:
            with self.subTest(outcome=outcome, code=code):
                body = Body((payload,))
                connection, client = self.connection(lambda _: httpx.Response(400,
                    headers={"x-request-id": "req_bounded"}, stream=body))
                with self.assertLogs("app.realtime.webrtc", level="INFO") as logs:
                    result = await connection.close()
                self.assertEqual(result["http_status"], 400)
                self.assertEqual(result["provider_request_id"], "req_bounded")
                self.assertEqual(result["provider_error_code"], code)
                self.assertEqual(result["response_body_outcome"], outcome)
                self.assertEqual(result["response_close_outcome"], "closed")
                self.assertFalse(result["remote_closure_confirmed"])
                self.assertTrue(client.is_closed)
                self.safe("\n".join(logs.output))

    async def test_unknown_exception_and_trace_payload_cannot_become_log_text(self):
        error_type = type("Private_" + KEY, (Exception,), {})
        async def handler(request):
            trace = request.extensions["trace"]
            await trace("http2.send_request_headers.complete", {"headers": PRIVATE})
            await trace(PRIVATE, {"return_value": PRIVATE})
            raise error_type(PRIVATE)
        connection, _ = self.connection(handler)
        with self.assertLogs("app.realtime.webrtc", level="INFO") as logs:
            result = await connection.close()
        self.assertEqual(result["exception_type"], "other")
        self.assertEqual(result["http_phase"], "request_headers")
        self.assertTrue(result["request_headers_sent"])
        self.assertIsNone(result["deadline_source"])
        self.safe("\n".join(logs.output))

    async def test_response_close_has_its_own_bound_and_preserves_observed_200(self):
        body = Body(close_stalled=True)
        requests = []
        def handler(request):
            requests.append(request)
            return httpx.Response(200, headers={"x-request-id": "req_close"}, stream=body)
        connection, client = self.connection(handler)
        with patch.object(webrtc, "RESPONSE_CLOSE_TIMEOUT_SECONDS", 0.02):
            with self.assertRaises(TimeoutError):
                await asyncio.wait_for(connection.close(), 0.5)
        result = await connection.close()
        self.assertEqual(result["response_close_outcome"], "deadline")
        self.assertEqual(result["http_close_outcome"], "closed")
        self.assertEqual(result["closure_outcome"], "accepted")
        self.assertEqual(result["http_status"], 200)
        self.assertEqual(result["provider_request_id"], "req_close")
        self.assertFalse(result["remote_closure_confirmed"])
        self.assertTrue(client.is_closed)
        self.assertEqual(len(requests), 1)
        self.assertTrue(connection._close_task.done())
