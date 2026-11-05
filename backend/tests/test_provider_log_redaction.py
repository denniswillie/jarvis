from __future__ import annotations

import asyncio
import base64
import io
import logging
import socket
import unittest
from contextlib import contextmanager
from unittest.mock import patch

import httpx
import uvicorn
import websockets

from app.logging_config import configure_logging
from app.visual_memory.gemini_client import (
    GeminiVisualAnalyzerClient, GeminiVisualAnalyzerConfig, GeminiVisualAnalyzerError,
)


SENTINEL_KEY = "synthetic-api-key-must-not-be-logged"
SENTINEL_IMAGE = b"synthetic-image-must-not-be-logged"
WIRE_LOGGERS = (
    "httpcore", "websockets.client", "websockets.server", "openai", "uvicorn.error",
)


@contextmanager
def capture_debug_logs(*, json_logs: bool = True):
    """Exercise production logging without leaking logger state between tests."""

    stream = io.StringIO()
    root = logging.getLogger()
    root_level, root_handlers = root.level, list(root.handlers)
    names = (*WIRE_LOGGERS, "httpx", "uvicorn", "uvicorn.access")
    logger_state = {
        name: (
            logging.getLogger(name).level,
            list(logging.getLogger(name).handlers),
            logging.getLogger(name).propagate,
        )
        for name in names
    }
    try:
        with patch("sys.stdout", stream):
            configure_logging("DEBUG", json_logs=json_logs)
            logging.getLogger("httpx").setLevel(logging.INFO)
            yield stream
    finally:
        for handler in root.handlers:
            if handler not in root_handlers:
                handler.close()
        root.handlers[:] = root_handlers
        root.setLevel(root_level)
        for name, (level, handlers, propagate) in logger_state.items():
            logger = logging.getLogger(name)
            logger.setLevel(level)
            logger.handlers[:] = handlers
            logger.propagate = propagate


class ProviderLogRedactionTests(unittest.IsolatedAsyncioTestCase):
    async def test_gemini_requests_and_logs_are_safe_on_success_and_failures(self):
        for outcome in ("success", "timeout", "transport", 400, 500, "invalid_json"):
            with self.subTest(outcome=outcome):
                await self.exercise_request(outcome)

    async def test_inbound_websocket_logs_exclude_authorization_and_media(self):
        encoded_image = base64.b64encode(SENTINEL_IMAGE).decode()

        for json_logs in (False, True):
            with self.subTest(json_logs=json_logs):
                received_messages = []

                async def app(scope, receive, send):
                    await receive()
                    await send({"type": "websocket.accept"})
                    while True:
                        event = await receive()
                        if event["type"] == "websocket.disconnect":
                            return
                        received_messages.append(
                            event.get("text") if event.get("text") is not None else event["bytes"]
                        )
                        await send({"type": "websocket.send", "text": "received"})

                with capture_debug_logs(json_logs=json_logs) as stream:
                    with socket.socket() as listener:
                        listener.bind(("127.0.0.1", 0))
                        port = listener.getsockname()[1]
                        server = uvicorn.Server(uvicorn.Config(
                            app, log_config=None, access_log=False, lifespan="off", ws="auto",
                        ))
                        server_task = asyncio.create_task(server.serve(sockets=[listener]))
                        try:
                            async with asyncio.timeout(5):
                                while not server.started:
                                    if server_task.done():
                                        await server_task
                                        self.fail("Websocket test server stopped before startup")
                                    await asyncio.sleep(0.01)

                                async with websockets.connect(
                                    f"ws://127.0.0.1:{port}",
                                    additional_headers={"Authorization": f"Bearer {SENTINEL_KEY}"},
                                ) as websocket:
                                    for media in (encoded_image, SENTINEL_IMAGE):
                                        await websocket.send(media)
                                        self.assertEqual(await websocket.recv(), "received")

                            logging.getLogger("app.provider_test").debug(
                                "application debug remains visible"
                            )
                            logging.getLogger("uvicorn.error").info(
                                "websocket status remains visible"
                            )
                        finally:
                            server.should_exit = True
                            await asyncio.wait_for(server_task, timeout=5)

                self.assertEqual(received_messages, [encoded_image, SENTINEL_IMAGE])
                captured = stream.getvalue()
                for private in (SENTINEL_KEY, encoded_image, SENTINEL_IMAGE.decode()):
                    self.assertNotIn(private, captured)
                self.assertIn("application debug remains visible", captured)
                self.assertIn("websocket status remains visible", captured)

    async def exercise_request(self, outcome):
        requests = []
        encoded_image = base64.b64encode(SENTINEL_IMAGE).decode()

        async def transport(request):
            requests.append(request)
            if outcome == "timeout":
                raise httpx.ReadTimeout(f"{SENTINEL_KEY} {encoded_image}", request=request)
            if outcome == "transport":
                raise httpx.ConnectError(f"{SENTINEL_KEY} {encoded_image}", request=request)
            if isinstance(outcome, int):
                return httpx.Response(outcome, text=f"{SENTINEL_KEY} {encoded_image}")
            if outcome == "invalid_json":
                return httpx.Response(200, text=f"{SENTINEL_KEY} {encoded_image}")
            return httpx.Response(200, json={
                "candidates": [{"content": {"parts": [{"text": "{}"}]}}],
            })

        client = GeminiVisualAnalyzerClient(GeminiVisualAnalyzerConfig(
            api_key=SENTINEL_KEY, model="test-model",
        ))
        # Replace only the HTTP transport; request construction remains real.
        await client._client.aclose()
        client._client = httpx.AsyncClient(transport=httpx.MockTransport(transport))
        observed_error = None
        try:
            with capture_debug_logs() as stream:
                for name in (*WIRE_LOGGERS, "httpcore.http11"):
                    logging.getLogger(name).debug("wire headers/media %s %s", SENTINEL_KEY, encoded_image)
                try:
                    result = await client.generate_json(
                        prompt="Describe this synthetic image.",
                        image_bytes=SENTINEL_IMAGE, content_type="image/jpeg",
                    )
                except (GeminiVisualAnalyzerError, httpx.HTTPError) as error:
                    observed_error = error
                    logging.getLogger("app.provider_test").exception("provider request failed")
                else:
                    self.assertEqual(result, "{}")

            self.assertEqual(len(requests), 1)
            request = requests[0]
            self.assertEqual(request.headers["x-goog-api-key"], SENTINEL_KEY)
            self.assertNotIn("key", request.url.params)
            self.assertNotIn(SENTINEL_KEY, str(request.url))
            self.assertTrue(request.url.path.endswith("/test-model:generateContent"))
            self.assertIn(encoded_image, request.content.decode())
            self.assertNotIn(SENTINEL_KEY, repr(client._config))
            captured = stream.getvalue()
            for private in (SENTINEL_KEY, encoded_image, SENTINEL_IMAGE.decode()):
                self.assertNotIn(private, captured)
                self.assertNotIn(private, str(observed_error))
            if outcome == "success":
                self.assertIsNone(observed_error)
            elif outcome == "timeout":
                self.assertIsInstance(observed_error, httpx.TimeoutException)
            elif outcome == "transport":
                self.assertIsInstance(observed_error, httpx.RequestError)
            else:
                self.assertIsInstance(observed_error, GeminiVisualAnalyzerError)
            if outcome not in {"timeout", "transport"}:
                self.assertIn("HTTP Request:", captured)
            if isinstance(outcome, int):
                self.assertIn(f"status {outcome}", str(observed_error))
        finally:
            await client.close()
