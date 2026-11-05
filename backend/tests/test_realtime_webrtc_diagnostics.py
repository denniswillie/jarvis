"""Offline transport diagnostics: fake HTTP and sideband, no provider requests."""
import asyncio
import json
import logging
from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import httpx

from app.realtime.models import RealtimeSessionContext
from app.realtime.session import RealtimeAudioSession
from app.realtime.webrtc import OpenAIWebRTCConnection, WebRTCTransportError, log_webrtc_setup_failure
from app.settings import JarvisSettings


OFFER = "v=0\r\nm=audio 9 UDP/TLS/RTP/SAVPF 111\r\na=secret-fixture-sdp\r\n"
KEY = "sk-test-only-never-send"
PRIVATE = "private prompt, Authorization Bearer " + KEY + " " + OFFER


class WebRTCDiagnosticTests(IsolatedAsyncioTestCase):
    def connection(self, handler):
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        self.addAsyncCleanup(client.aclose)
        connection = OpenAIWebRTCConnection(settings=JarvisSettings(_env_file=None,
            OPENAI_API_KEY=KEY, OPENAI_REALTIME_MODEL="fixture-model"), owner_id="fixture-owner", offer=OFFER, http=client)
        return connection, client

    def created(self):
        return httpx.Response(201, text=OFFER, headers={
            "location": "/v1/realtime/calls/rtc_fixture", "x-request-id": "req_create_fixture"})

    def socket(self):
        return SimpleNamespace(close=AsyncMock(), send=AsyncMock(), recv=AsyncMock())

    def assert_sanitized(self, value):
        text = json.dumps(value) if not isinstance(value, str) else value
        for forbidden in (KEY, "Authorization", "private prompt", "secret-fixture-sdp", "v=0", "m=audio"):
            self.assertNotIn(forbidden, text)
        self.assertLess(len(text), 4000)

    async def test_definite_rejection_has_bounded_code_param_status_and_request_id(self):
        requests = []
        def handler(request):
            requests.append(request)
            return httpx.Response(400, headers={"x-request-id": "req_rejected"}, json={"error": {
                "code": "invalid_value", "param": "session.audio.output.voice", "message": PRIVATE}})
        connection, _ = self.connection(handler)
        with patch("app.realtime.webrtc.websockets.connect", AsyncMock()) as attach:
            with self.assertRaises(WebRTCTransportError) as raised:
                await connection.connect()
            attach.assert_not_awaited()
        diagnostic = raised.exception.diagnostic
        self.assertEqual(diagnostic["stage"], "create_http")
        self.assertEqual(diagnostic["http_status"], 400)
        self.assertEqual(diagnostic["provider_error_code"], "invalid_value")
        self.assertEqual(diagnostic["provider_error_param"], "session.audio.output.voice")
        self.assertEqual(diagnostic["provider_request_id"], "req_rejected")
        self.assertEqual(diagnostic["creation_outcome"], "rejected")
        self.assertIsNone(diagnostic["call_id"])
        self.assert_sanitized(diagnostic)
        self.assert_sanitized(str(raised.exception))
        self.assertTrue(raised.exception.__suppress_context__)
        result = await connection.close()
        self.assertEqual(result["closure_outcome"], "not_requested")
        self.assertEqual(len(requests), 1)

    async def test_missing_malformed_or_foreign_success_identity_remains_unknown(self):
        for location in ("", "/v1/realtime/calls/not-a-call", "https://other.invalid/v1/realtime/calls/rtc_other", "https://["):
            with self.subTest(location=location):
                requests = []
                def handler(request):
                    requests.append(request)
                    return httpx.Response(201, text=PRIVATE, headers={"location": location})
                connection, _ = self.connection(handler)
                with patch("app.realtime.webrtc.websockets.connect", AsyncMock()) as attach:
                    with self.assertRaises(WebRTCTransportError) as raised:
                        await connection.connect()
                    attach.assert_not_awaited()
                self.assertEqual(raised.exception.diagnostic["stage"], "validate_call_identity")
                self.assertIsNone(connection.created_call_id)
                result = await connection.close()
                self.assertEqual(result["creation_outcome"], "unknown")
                self.assertEqual(result["closure_outcome"], "unknown")
                self.assertFalse(result["remote_closure_confirmed"])
                self.assertEqual(len(requests), 1)
                self.assert_sanitized(result)

    async def test_created_identity_survives_invalid_sdp_and_is_used_for_hangup(self):
        requests = []
        def handler(request):
            requests.append(request)
            if request.url.path.endswith("/hangup"):
                return httpx.Response(200, headers={"x-request-id": "req_hangup"})
            return httpx.Response(201, text="invalid " + PRIVATE,
                                  headers={"location": "/v1/realtime/calls/rtc_fixture"})
        connection, _ = self.connection(handler)
        with patch("app.realtime.webrtc.websockets.connect", AsyncMock()) as attach:
            with self.assertRaises(WebRTCTransportError) as raised:
                await connection.connect()
            attach.assert_not_awaited()
        self.assertEqual(raised.exception.diagnostic["stage"], "validate_sdp_answer")
        self.assertEqual(connection.created_call_id, "rtc_fixture")
        result = await connection.close()
        self.assertIsNone(connection.call_id)
        self.assertEqual(connection.created_call_id, "rtc_fixture")
        self.assertEqual(result["closure_outcome"], "accepted")
        self.assertFalse(result["remote_closure_confirmed"])
        self.assertEqual(result["provider_request_id"], "req_hangup")
        self.assertTrue(requests[-1].url.path.endswith("/rtc_fixture/hangup"))

    async def test_lost_create_response_or_server_failure_is_unknown_and_never_retried(self):
        for response_status in (None, 408, 500, 200):
            with self.subTest(status=response_status):
                requests = []
                def handler(request):
                    requests.append(request)
                    if response_status is None:
                        raise httpx.ReadTimeout(PRIVATE, request=request)
                    return httpx.Response(response_status, text=PRIVATE)
                connection, _ = self.connection(handler)
                with self.assertRaises(WebRTCTransportError):
                    await connection.connect()
                self.assertEqual(connection.creation_outcome, "unknown")
                with self.assertRaises(WebRTCTransportError):
                    await connection.connect()
                result = await connection.close()
                self.assertEqual(result["closure_outcome"], "unknown")
                self.assert_sanitized(result)
                self.assertEqual(len(requests), 1)

    async def test_sideband_failure_preserves_creation_and_sanitizes_http_failure(self):
        connection, _ = self.connection(lambda request: httpx.Response(200) if request.url.path.endswith("/hangup") else self.created())
        response = httpx.Response(403, headers={"x-request-id": "req_attach"}, json={"error": {
            "code": "permission_denied", "param": "model", "message": PRIVATE}})
        error = httpx.HTTPStatusError(PRIVATE, request=httpx.Request("GET", "https://example.invalid"), response=response)
        with patch("app.realtime.webrtc.websockets.connect", AsyncMock(side_effect=error)):
            with self.assertRaises(WebRTCTransportError) as raised:
                await connection.connect()
        diagnostic = raised.exception.diagnostic
        self.assertEqual(diagnostic["stage"], "sideband_connect")
        self.assertEqual(diagnostic["http_status"], 403)
        self.assertEqual(diagnostic["creation_http_status"], 201)
        self.assertEqual(diagnostic["call_id"], "rtc_fixture")
        self.assertEqual(diagnostic["provider_request_id"], "req_attach")
        self.assertEqual(diagnostic["creation_request_id"], "req_create_fixture")
        self.assert_sanitized(diagnostic)
        self.assertEqual((await connection.close())["closure_outcome"], "accepted")

    async def test_hangup_status_failure_unknown_and_accepted_remain_distinct(self):
        for status, expected in ((200, "accepted"), (202, "unknown"), (204, "unknown"),
                                 (302, "failed"), (400, "failed"), (404, "failed"), (500, "failed"), (None, "unknown")):
            with self.subTest(status=status):
                requests = []
                def handler(request):
                    requests.append(request)
                    if not request.url.path.endswith("/hangup"):
                        return self.created()
                    self.assertEqual(connection.closure_outcome, "requested")
                    if status is None:
                        raise httpx.ReadTimeout(PRIVATE, request=request)
                    return httpx.Response(status, json={"error": {"code": "server_error", "message": PRIVATE}})
                connection, _ = self.connection(handler)
                socket = self.socket()
                with patch("app.realtime.webrtc.websockets.connect", AsyncMock(return_value=socket)):
                    await connection.connect()
                result = await connection.close()
                self.assertEqual(result["closure_outcome"], expected)
                self.assertFalse(result["remote_closure_confirmed"])
                self.assertEqual(result, await connection.close())
                self.assertEqual(len(requests), 2)
                socket.close.assert_awaited_once()
                self.assert_sanitized(result)

    async def test_sensitive_metadata_is_redacted_and_oversized_body_is_not_parsed(self):
        for body in ({"error": {"code": KEY, "param": "session." + KEY, "message": PRIVATE}},
                     {"error": {"code": "arbitrary_private_text", "param": "unknown_secret_path", "message": PRIVATE}},
                     {"error": {"code": "invalid_value", "param": "model", "message": "x" * 5000}},
                     [PRIVATE]):
            with self.subTest(body_kind=type(body).__name__):
                connection, _ = self.connection(lambda request: httpx.Response(400, json=body,
                    headers={"x-request-id": "req_" + KEY, "authorization": "Bearer " + KEY}))
                with self.assertRaises(WebRTCTransportError) as raised:
                    await connection.connect()
                with self.assertLogs("test.webrtc", level="ERROR") as captured:
                    log_webrtc_setup_failure(logging.getLogger("test.webrtc"), connection, raised.exception)
                self.assert_sanitized(" ".join(captured.output))
                self.assertNotIn("arbitrary_private_text", " ".join(captured.output))
                self.assertNotIn("unknown_secret_path", " ".join(captured.output))
                self.assertEqual(raised.exception.diagnostic["provider_request_id"], "[redacted]")
                if isinstance(body, dict) and len(json.dumps(body)) > 4096:
                    self.assertIsNone(raised.exception.diagnostic["provider_error_code"])
                await connection.close()

    async def test_provider_error_event_metadata_preserves_payload_semantics(self):
        connection, _ = self.connection(lambda request: httpx.Response(200) if request.url.path.endswith("/hangup") else self.created())
        socket = self.socket()
        event = {"type": "error", "error": {"code": "invalid_value", "param": "session.audio.input.format", "message": PRIVATE}}
        socket.recv.return_value = json.dumps(event)
        with patch("app.realtime.webrtc.websockets.connect", AsyncMock(return_value=socket)):
            await connection.connect()
        self.assertEqual(await connection.receive_event(), event)
        diagnostic = connection.setup_failure_diagnostic()
        self.assertEqual(diagnostic["stage"], "provider_control_error")
        self.assertEqual(diagnostic["provider_error_param"], "session.audio.input.format")
        self.assertIsNone(diagnostic["http_status"])
        self.assert_sanitized(diagnostic)
        await connection.close()

    async def test_readiness_failure_uses_actual_session_logging_block_and_cleans_known_call(self):
        connection, _ = self.connection(lambda request: httpx.Response(200) if request.url.path.endswith("/hangup") else self.created())
        session = RealtimeAudioSession(websocket=SimpleNamespace(close=AsyncMock(), send_text=AsyncMock()),
            context=RealtimeSessionContext(audio_session_id="fixture", owner_id="owner", jarvis_session_id=uuid4(),
                client_kind="phone_audio", session=SimpleNamespace()), settings=JarvisSettings(_env_file=None, OPENAI_API_KEY=KEY),
            repository=SimpleNamespace(create_message=AsyncMock(), get_recent_messages=AsyncMock(return_value=[])),
            tool_modules=(), context_providers=())
        session.openai = connection
        session._send_transport_setup = AsyncMock()
        session._send_session_update = AsyncMock(return_value=0)
        session._wait_for_openai_session_ready = AsyncMock(side_effect=TimeoutError(PRIVATE))
        session._send_error = AsyncMock()
        async def close(_reason):
            await connection.close()
        session.close = AsyncMock(side_effect=close)
        with patch("app.realtime.webrtc.websockets.connect", AsyncMock(return_value=self.socket())):
            with self.assertLogs("app.realtime.session", level="ERROR") as captured:
                await session.run()
        text = " ".join(captured.output)
        self.assertIn("post_attach_setup", text)
        self.assertIn("rtc_fixture", text)
        self.assertIn("req_create_fixture", text)
        self.assert_sanitized(text)
        self.assertFalse(session._realtime_context_ready.is_set())
        self.assertEqual(connection.closure_outcome, "accepted")
        session._send_error.assert_awaited_once_with("openai_realtime_connect_failed", "OpenAI Realtime connection failed.", recoverable=True)

    async def test_cancelled_creation_remains_unknown_and_cancelled_hangup_closes_owned_http(self):
        creation_started = asyncio.Event()
        async def lost_create(_request):
            creation_started.set()
            await asyncio.Event().wait()
        connection, _ = self.connection(lost_create)
        task = asyncio.create_task(connection.connect())
        await creation_started.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual((await connection.close())["closure_outcome"], "unknown")

        hangup_started = asyncio.Event()
        async def pending_hangup(request):
            if not request.url.path.endswith("/hangup"):
                return self.created()
            hangup_started.set()
            await asyncio.Event().wait()
        connection, client = self.connection(pending_hangup)
        connection._owns_http = True
        with patch("app.realtime.webrtc.websockets.connect", AsyncMock(return_value=self.socket())):
            await connection.connect()
        task = asyncio.create_task(connection.close())
        await hangup_started.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(connection.closure_outcome, "unknown")
        self.assertTrue(client.is_closed)
        self.assertEqual((await connection.close())["closure_outcome"], "unknown")

    async def test_local_socket_close_failure_still_attempts_remote_hangup(self):
        connection, _ = self.connection(lambda request: httpx.Response(200) if request.url.path.endswith("/hangup") else self.created())
        socket = self.socket()
        socket.close.side_effect = RuntimeError("fixture local close failure")
        with patch("app.realtime.webrtc.websockets.connect", AsyncMock(return_value=socket)):
            await connection.connect()
        with self.assertRaisesRegex(RuntimeError, "fixture local close failure"):
            await connection.close()
        self.assertEqual(connection.closure_outcome, "accepted")
        self.assertFalse(connection.closure_diagnostic["remote_closure_confirmed"])

    async def test_legacy_provider_does_not_acquire_webrtc_diagnostics(self):
        logger = SimpleNamespace(error=lambda *args: self.fail("legacy logger must remain in session"))
        self.assertFalse(log_webrtc_setup_failure(logger, SimpleNamespace(), RuntimeError(PRIVATE)))
