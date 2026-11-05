"""Offline teardown ownership: no API, socket, provider creation or native I/O."""

import asyncio
from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, patch

import httpx

from app.realtime.webrtc import OpenAIWebRTCConnection, WebRTCTransportError
from app.settings import JarvisSettings


class CloseHTTP:
    def __init__(self, *, status=200):
        self.status = status
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.requests = []
        self.completed = False
        self.closed = False
        self.close_started = asyncio.Event()
        self.close_release = asyncio.Event()
        self.close_release.set()
        self.post_cancelled = False
        self.close_before_post_finished = False

    def build_request(self, method, url, **kwargs):
        return httpx.Request(method, url, headers=kwargs.get("headers"), extensions=kwargs.get("extensions"))

    async def send(self, request, *, stream, follow_redirects):
        assert stream and not follow_redirects
        return await self.post(str(request.url))

    async def post(self, url, **kwargs):
        self.requests.append(url)
        self.started.set()
        try:
            await self.release.wait()
            return httpx.Response(self.status)
        except asyncio.CancelledError:
            self.post_cancelled = True
            raise
        finally:
            self.completed = True

    async def aclose(self):
        self.close_before_post_finished = bool(self.requests) and not self.completed
        self.close_started.set()
        await self.close_release.wait()
        self.closed = True


class WebRTCCloseOwnershipTests(IsolatedAsyncioTestCase):
    def connection(self, http=None, *, owned=True, known=True):
        http = http or CloseHTTP()
        connection = OpenAIWebRTCConnection(settings=JarvisSettings(_env_file=None,
            OPENAI_API_KEY="sk-offline-fixture"), owner_id="fixture-owner",
            offer="v=0\r\nm=audio 9 UDP/TLS/RTP/SAVPF 111\r\n", http=http)
        connection._owns_http = owned
        if known:
            # Explicit fixture identity, not a provider creation or receipt.
            connection.created_call_id = connection.call_id = "rtc_fixture_owned"
            connection.creation_outcome = "created"
            connection._creation_attempted = True
        connection._websocket = SimpleNamespace(close=AsyncMock())
        return connection, http

    async def wait(self, event):
        await asyncio.wait_for(event.wait(), 0.5)

    async def test_cancelled_caller_joins_hangup_then_preserves_cancellation(self):
        connection, http = self.connection()
        caller = asyncio.create_task(connection.close())
        await self.wait(http.started)
        caller.cancel()
        await asyncio.sleep(0)
        self.assertFalse(http.post_cancelled)
        self.assertFalse(http.closed)
        self.assertFalse(caller.done())
        http.release.set()
        with self.assertRaises(asyncio.CancelledError):
            await caller
        self.assertTrue(connection._close_task.done())
        self.assertTrue(http.closed)
        self.assertFalse(http.close_before_post_finished)
        self.assertEqual(connection.closure_outcome, "accepted")
        self.assertFalse(connection.closure_diagnostic["remote_closure_confirmed"])
        self.assertEqual(len(http.requests), 1)

    async def test_repeated_cancellation_and_concurrent_callers_share_one_attempt(self):
        connection, http = self.connection()
        first = asyncio.create_task(connection.close())
        await self.wait(http.started)
        second = asyncio.create_task(connection.close())
        for _ in range(3):
            first.cancel()
            await asyncio.sleep(0)
        self.assertFalse(http.post_cancelled)
        self.assertFalse(http.closed)
        http.release.set()
        with self.assertRaises(asyncio.CancelledError):
            await first
        result = await second
        self.assertEqual(result["closure_outcome"], "accepted")
        self.assertEqual(result, await connection.close())
        self.assertEqual(len(http.requests), 1)
        self.assertTrue(connection._close_task.done())

    async def test_hangup_deadline_is_unknown_joined_and_never_retried(self):
        connection, http = self.connection()
        with patch("app.realtime.webrtc.HANGUP_TIMEOUT_SECONDS", 0.02):
            result = await asyncio.wait_for(connection.close(), 0.5)
        self.assertEqual(result["closure_outcome"], "unknown")
        self.assertEqual(result["reason"], "hangup_deadline_without_response")
        self.assertTrue(http.post_cancelled)
        self.assertTrue(http.completed)
        self.assertTrue(http.closed)
        self.assertFalse(http.close_before_post_finished)
        self.assertTrue(connection._close_task.done())
        self.assertEqual(result, await connection.close())
        self.assertEqual(len(http.requests), 1)

    async def test_cancellation_during_local_close_still_reaches_hangup(self):
        connection, http = self.connection()
        local_started, local_release = asyncio.Event(), asyncio.Event()

        async def close_socket():
            local_started.set()
            await local_release.wait()

        connection._websocket.close = close_socket
        caller = asyncio.create_task(connection.close())
        await self.wait(local_started)
        caller.cancel()
        await asyncio.sleep(0)
        local_release.set()
        await self.wait(http.started)
        http.release.set()
        with self.assertRaises(asyncio.CancelledError):
            await caller
        self.assertTrue(http.closed)
        self.assertEqual(connection.closure_outcome, "accepted")

    async def test_local_close_timeout_does_not_skip_hangup_or_http_cleanup(self):
        connection, http = self.connection()
        connection._websocket.close = AsyncMock(side_effect=lambda: None)

        async def stuck_socket():
            await asyncio.Event().wait()

        connection._websocket.close = stuck_socket
        http.release.set()
        with patch("app.realtime.webrtc.LOCAL_CLOSE_TIMEOUT_SECONDS", 0.02):
            with self.assertRaises(TimeoutError):
                await asyncio.wait_for(connection.close(), 0.5)
        self.assertEqual(len(http.requests), 1)
        self.assertEqual(connection.closure_outcome, "accepted")
        self.assertTrue(http.closed)
        self.assertTrue(connection._close_task.done())

    async def test_caller_cancel_during_http_close_does_not_detach_cleanup(self):
        connection, http = self.connection()
        http.release.set()
        http.close_release.clear()
        caller = asyncio.create_task(connection.close())
        await self.wait(http.close_started)
        caller.cancel()
        await asyncio.sleep(0)
        self.assertFalse(caller.done())
        http.close_release.set()
        with self.assertRaises(asyncio.CancelledError):
            await caller
        self.assertTrue(http.closed)
        self.assertTrue(connection._close_task.done())

    async def test_http_cleanup_deadline_does_not_claim_remote_closure_or_retry(self):
        connection, http = self.connection()
        http.release.set()
        http.close_release.clear()
        with patch("app.realtime.webrtc.HTTP_CLOSE_TIMEOUT_SECONDS", 0.02):
            with self.assertRaises(TimeoutError):
                await asyncio.wait_for(connection.close(), 0.5)
        self.assertFalse(http.closed)
        self.assertTrue(connection._close_task.done())
        self.assertEqual(connection.closure_outcome, "accepted")
        self.assertFalse(connection.closure_diagnostic["remote_closure_confirmed"])
        self.assertEqual((await connection.close())["closure_outcome"], "accepted")
        self.assertEqual(len(http.requests), 1)

    async def test_failed_hangup_and_partial_start_preserve_evidence(self):
        connection, http = self.connection(CloseHTTP(status=500))
        connection._websocket = None
        http.release.set()
        result = await connection.close()
        self.assertEqual(result["closure_outcome"], "failed")
        self.assertEqual(result["http_status"], 500)
        self.assertEqual(connection.created_call_id, "rtc_fixture_owned")
        self.assertFalse(result["remote_closure_confirmed"])
        self.assertEqual(len(http.requests), 1)
        self.assertTrue(http.closed)

    async def test_unknown_create_identity_is_not_guessed_or_retried(self):
        connection, http = self.connection(known=False)
        connection.creation_outcome = "unknown"
        result = await connection.close()
        self.assertEqual(result["closure_outcome"], "unknown")
        self.assertEqual(http.requests, [])
        self.assertTrue(http.closed)

    async def test_closing_transport_cannot_start_a_new_creation(self):
        connection, http = self.connection(known=False)
        close_started, close_release = asyncio.Event(), asyncio.Event()

        async def local_close():
            close_started.set()
            await close_release.wait()

        connection._websocket.close = local_close
        caller = asyncio.create_task(connection.close())
        await self.wait(close_started)
        with self.assertRaises(WebRTCTransportError):
            await connection.connect()
        close_release.set()
        await caller
        self.assertEqual(http.requests, [])

    async def test_borrowed_http_is_not_closed(self):
        connection, http = self.connection(owned=False)
        http.release.set()
        self.assertEqual((await connection.close())["closure_outcome"], "accepted")
        self.assertFalse(http.closed)
        self.assertTrue(connection._close_task.done())
