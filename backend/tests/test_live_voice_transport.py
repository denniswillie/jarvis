"""Offline HTTP/WebSocket fixtures for the actual Live transport; no network."""
import asyncio
from contextlib import asynccontextmanager
import json
import ssl
import unittest

import httpx

from app.voice_routes.live_protocol import LiveSessionAnswer
from app.voice_routes.live_transport import LiveBackendTransport, LiveTransportError


@asynccontextmanager
async def guard():
    yield


class Socket:
    def __init__(self):
        self.incoming = asyncio.Queue()
        self.sent = []
        self.closed = False
        self.auto_ack = self.auto_close = True
        self.block_type = None
        self.blocked, self.release = asyncio.Event(), asyncio.Event()

    async def send(self, text):
        value = json.loads(text)
        self.sent.append(value)
        if value["type"] == self.block_type:
            self.blocked.set()
            await self.release.wait()
        if value["type"].endswith(".append") and self.auto_ack:
            await self.incoming.put(json.dumps(dict(type=value["type"] + "ed", event_id="ack_" + value["event_id"],
                client_event_id=value["event_id"], delegation_id=value["delegation_id"], start_ms=1, end_ms=2)))
        if value["type"] == "session.close" and self.auto_close:
            await self.incoming.put(json.dumps(dict(type="session.closed", event_id="closed",
                session={"id": "live_fixture"}, reason="close_requested", usage={"seconds": 1})))

    async def recv(self):
        value = await self.incoming.get()
        if isinstance(value, Exception):
            raise value
        return value

    async def close(self):
        self.closed = True


class TransportFixture:
    def __init__(self, **options):
        self.socket = Socket()
        self.requests, self.connections = [], []
        self.status = 201
        self.body = {"session": {"id": "live_fixture"}, "transport": {"type": "webrtc", "sdp": "v=0 answer"}}
        self.client = httpx.AsyncClient(transport=httpx.MockTransport(self.http))
        self.transport = LiveBackendTransport("fixture-secret", project="fixture-project", organization="fixture-org",
            http_client=self.client, connector=self.connect, io_timeout=.2, open_timeout=.5, **options)

    def http(self, request):
        self.requests.append(request)
        return httpx.Response(self.status, json=self.body)

    async def connect(self, url, **kwargs):
        self.connections.append((url, kwargs))
        return self.socket

    async def start(self):
        answer = await self.transport.create({"session": {"delegation": {"type": "client"}}}, guard=guard, current=lambda: True)
        await self.transport.attach(answer)
        return answer

    async def close(self):
        await self.transport.aclose()
        await self.client.aclose()


class LiveTransportTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.f = TransportFixture()

    async def asyncTearDown(self):
        await self.f.close()

    async def test_real_http_shape_authentication_and_bounded_attach(self):
        answer = await self.f.start()
        request = self.f.requests[0]
        self.assertEqual(str(request.url), "https://api.openai.com/v1/live/sessions")
        self.assertEqual(request.method, "POST")
        url, options = self.f.connections[0]
        self.assertEqual(url, answer.sideband_url)
        self.assertEqual(options["additional_headers"]["Authorization"], request.headers["Authorization"])
        self.assertEqual(options["additional_headers"]["OpenAI-Project"], "fixture-project")
        self.assertEqual(options["max_queue"], 16)
        self.assertIsNone(options["compression"])
        self.assertIsInstance(options["ssl"], ssl.SSLContext)
        self.assertTrue(options["ssl"].check_hostname)
        self.assertEqual(options["ssl"].verify_mode, ssl.CERT_REQUIRED)
        self.assertGreater(options["ssl"].cert_store_stats()["x509_ca"], 0)
        self.assertFalse(self.f.socket.sent)  # No second session.start.

    async def test_guard_and_current_checked_at_write(self):
        await self.f.start()
        valid = True
        entered = []
        @asynccontextmanager
        async def guarded():
            nonlocal valid
            entered.append(True)
            valid = False
            yield
        wrote = await self.f.transport.send(lambda: {"type": "never"}, guard=guarded, current=lambda: valid)
        self.assertFalse(wrote)
        self.assertEqual(entered, [True])
        self.assertEqual(self.f.socket.sent, [])

    async def test_foreign_answer_and_second_attachment_rejected(self):
        answer = await self.f.start()
        for candidate in (answer, LiveSessionAnswer("foreign", "answer")):
            with self.assertRaises(LiveTransportError):
                await self.f.transport.attach(candidate)
        self.assertEqual(len(self.f.connections), 1)

    async def test_http_failure_sanitized_no_retry(self):
        self.f.status = 429
        self.f.body = {"error": "fixture-secret private provider message"}
        with self.assertRaisesRegex(LiveTransportError, "session_creation_rejected") as caught:
            await self.f.start()
        self.assertNotIn("private", str(caught.exception))
        with self.assertRaises(LiveTransportError):
            await self.f.start()
        self.assertEqual(len(self.f.requests), 1)
        self.assertTrue(self.f.transport.creation_submitted)

    async def test_bootstrap_guard_failure_does_not_post(self):
        @asynccontextmanager
        async def withdrawn():
            raise PermissionError("private source")
            yield
        with self.assertRaises(LiveTransportError):
            await self.f.transport.create({}, guard=withdrawn, current=lambda: True)
        self.assertFalse(self.f.requests)

    async def test_receive_has_one_owner_and_rejects_invalid_json(self):
        await self.f.start()
        reader = asyncio.create_task(self.f.transport.receive())
        await asyncio.sleep(0)
        with self.assertRaises(LiveTransportError):
            await self.f.transport.receive()
        self.f.socket.incoming.put_nowait("not JSON with private data")
        with self.assertRaisesRegex(LiveTransportError, "sideband_receive_failed"):
            await reader

    async def test_oversized_response_and_non_dict_event_rejected(self):
        self.f.body["private"] = "x" * 524288
        with self.assertRaisesRegex(LiveTransportError, "session_answer_too_large"):
            await self.f.start()
        self.assertFalse(self.f.connections)

    async def test_close_owns_socket_but_not_borrowed_http(self):
        await self.f.start()
        self.assertTrue(await self.f.transport.aclose())
        self.assertTrue(self.f.socket.closed)
        self.assertFalse(self.f.client.is_closed)
        self.assertFalse(await self.f.transport.send(lambda: {"type": "never"}))

    async def test_write_timeout_is_not_replayed(self):
        await self.f.start()
        self.f.socket.block_type = "blocked"
        with self.assertRaisesRegex(LiveTransportError, "sideband_write_unconfirmed"):
            await self.f.transport.send(lambda: {"type": "blocked"})
        self.assertEqual(len(self.f.socket.sent), 1)
