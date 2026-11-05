"""Bounded operator evidence, without retaining native/provider error text."""
import asyncio
import json
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock

import httpx

from app.tasks.hermes import _NativeRun
from app.tasks.model_gateway import ModelGatewayError, OpenAIResponsesHTTP, TaskModelGateway
import test_second_brain_model_gateway as fixtures


class FailureDiagnosticsTests(unittest.IsolatedAsyncioTestCase):
    async def test_counter_rejection_retains_only_status_and_structural_parameter(self):
        upstream = OpenAIResponsesHTTP("fixture-key", transport=httpx.MockTransport(lambda request:
            httpx.Response(400, json={"error": {"param": "tools[2].parameters", "message": "SECRET prompt"}})))
        self.addAsyncCleanup(upstream.aclose)
        with self.assertRaises(ModelGatewayError) as raised:
            await upstream.count({"model": "fixture"})
        error = raised.exception
        self.assertEqual((error.code, error.http_status, error.parameter),
                         ("model_input_count_unavailable", 400, "tools[2].parameters"))
        self.assertNotIn("SECRET", repr(error.__dict__))

    async def test_stream_rejection_is_bounded_and_keeps_no_provider_message(self):
        for body in (b'{"error":{"param":"SECRET-value","message":"SECRET"}}', b"x" * 64001):
            upstream = OpenAIResponsesHTTP("fixture-key", transport=httpx.MockTransport(
                lambda request: httpx.Response(403, content=body)))
            try:
                with self.assertRaises(ModelGatewayError) as raised:
                    [event async for event in upstream.events({"model": "fixture"})]
                self.assertEqual(raised.exception.http_status, 403)
                self.assertIsNone(raised.exception.parameter)
                self.assertNotIn("SECRET", repr(raised.exception.__dict__))
            finally:
                await upstream.aclose()

    async def client(self, gateway):
        client = httpx.AsyncClient(transport=httpx.ASGITransport(app=gateway.application()),
            base_url="http://fixture", headers={"Authorization": "Bearer " + gateway.token})
        self.addAsyncCleanup(client.aclose)
        return client

    async def test_count_failure_is_recorded_before_any_reservation(self):
        context, upstream = fixtures.context(), fixtures.Upstream()
        upstream.count = AsyncMock(side_effect=ModelGatewayError(
            "model_input_count_unavailable", http_status=400, parameter="text.format.schema"))
        gateway = TaskModelGateway(context, upstream)
        client = await self.client(gateway)
        response = await client.post("/v1/responses", json={"model": "fixture-model", "input": "fixture"})
        self.assertIn("model_input_count_unavailable", response.text)
        self.assertEqual(gateway.diagnostics, [{"stage": "model_stream", "error_type": "ModelGatewayError",
            "code": "model_input_count_unavailable", "http_status": 400, "parameter": "text.format.schema"}])
        context.reserve.assert_not_awaited()
        self.assertFalse(upstream.requests)

    async def test_validation_diagnostics_are_bounded_without_request_content(self):
        gateway = TaskModelGateway(fixtures.context(), fixtures.Upstream())
        client = await self.client(gateway)
        for _ in range(10):
            response = await client.post("/v1/responses", json={"model": "SECRET", "input": "SECRET"})
            self.assertEqual(response.status_code, 400)
        self.assertEqual(len(gateway.diagnostics), 8)
        self.assertTrue(all(row["stage"] == "request_validation" for row in gateway.diagnostics))
        self.assertNotIn("SECRET", repr(gateway.diagnostics))

    async def test_unexpected_transport_error_does_not_retain_exception_message(self):
        upstream = fixtures.Upstream()
        upstream.count = AsyncMock(side_effect=RuntimeError("SECRET bearer token"))
        gateway = TaskModelGateway(fixtures.context(), upstream)
        client = await self.client(gateway)
        response = await client.post("/v1/responses", json={"model": "fixture-model", "input": "fixture"})
        self.assertIn("model_transport_failed", response.text)
        self.assertEqual(gateway.diagnostics, [{"stage": "model_stream", "error_type": "RuntimeError"}])
        self.assertNotIn("SECRET", response.text)

    async def test_native_receipt_keeps_gateway_metadata_not_native_prose(self):
        run = object.__new__(_NativeRun)
        run.run_id = "hermes-run:fixture"
        run.marker = {}
        run.gateway = SimpleNamespace(diagnostics=[{"stage": "model_stream", "http_status": 400}])
        run.write_marker = Mock()
        stdout, stderr = asyncio.StreamReader(), asyncio.StreamReader()
        stdout.feed_data((json.dumps({"type": "result", "run_id": run.run_id,
            "completed": False, "failed": True, "api_calls": 1, "final_response": "SECRET"}) + "\n").encode())
        stdout.feed_eof()
        stderr.feed_data(b"SECRET native diagnostics")
        stderr.feed_eof()
        run.process = SimpleNamespace(stdout=stdout, stderr=stderr, wait=AsyncMock(), returncode=0)
        await run.result()
        self.assertEqual(run.marker["model_diagnostics"], run.gateway.diagnostics)
        self.assertNotIn("SECRET", repr(run.marker))
        run.write_marker.assert_called_once()
