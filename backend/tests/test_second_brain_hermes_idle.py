"""Actual pinned Hermes transport with a deterministic quiet provider, no API/DB."""
import asyncio
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from uuid import uuid4

from app.runtime.feature_config import FrontierConfiguration
from app.tasks.hermes import HermesRuntimeConfiguration, _NativeRun
from app.tasks.model_gateway import TaskModelGateway
from test_second_brain_model_gateway import Upstream, context, frame


class QuietUpstream(Upstream):
    def __init__(self):
        super().__init__()
        self.entered = asyncio.Event()

    async def events(self, payload):
        self.requests.append(payload)
        yield self.sequence[0]
        yield frame({"type": "response.output_item.added", "output_index": 0,
                     "item": {"id": "rs_fixture", "type": "reasoning", "summary": []}})
        self.entered.set()
        # A valid reasoning response need not emit text continuously.
        await asyncio.sleep(15)
        answer = "Synthetic completed answer."
        item = {"id": "msg_fixture", "type": "message", "role": "assistant",
                "status": "completed", "content": [
                    {"type": "output_text", "text": answer, "annotations": []}]}
        yield frame({"type": "response.output_item.added", "output_index": 1, "item": item})
        yield frame({"type": "response.output_text.delta", "item_id": "msg_fixture",
                     "output_index": 1, "content_index": 0, "delta": answer})
        yield frame({"type": "response.output_item.done", "output_index": 1, "item": item})
        yield self.sequence[-1]


@unittest.skipUnless(os.getenv("JARVIS_HERMES_PYTHON"), "Explicit pinned native interpreter required")
class HermesQuietStreamTests(unittest.IsolatedAsyncioTestCase):
    async def prepare(self):
        directory = tempfile.TemporaryDirectory(prefix="jarvis-hermes-quiet-")
        self.addCleanup(directory.cleanup)
        ctx = context()
        ctx.task.id = uuid4()
        ctx.task.model = "gpt-6-astra"
        ctx.task.budget.request_output_tokens = 4096
        ctx.claim.attempt_id = uuid4()
        schema = {"name": "fixture_read", "description": "Read a synthetic fixture",
                  "parameters": {"type": "object", "properties": {}}}
        upstream = QuietUpstream()
        gateway = TaskModelGateway(ctx, upstream, tools=(schema,), run_id="hermes-run:" + uuid4().hex)
        prepared = SimpleNamespace(
            manifest=SimpleNamespace(content={"request": "Give the synthetic final answer; no tools needed."}),
            tools=SimpleNamespace(context=ctx, catalog=lambda: [schema]))
        run = _NativeRun(HermesRuntimeConfiguration(
            Path(os.environ["JARVIS_HERMES_PYTHON"]), Path(directory.name)), ctx, gateway)
        self.addAsyncCleanup(run.stop)
        await run.launch(prepared, FrontierConfiguration(
            model="gpt-6-astra", reasoning_effort="medium", api_key="synthetic-unused"))
        return run, ctx, upstream

    async def test_valid_quiet_stream_survives_and_settles_once(self):
        run, ctx, upstream = await self.prepare()
        result = await asyncio.wait_for(run.result(), 45)
        self.assertTrue(result.get("completed"), {
            key: result.get(key) for key in ("completed", "failed", "interrupted", "api_calls")})
        self.assertFalse(result.get("failed"))
        self.assertEqual(result["final_response"], "Synthetic completed answer.")
        self.assertEqual(len(upstream.requests), 1, "Quiet reasoning must not trigger another request")
        ctx.settle.assert_awaited_once()

    async def test_parent_stop_still_interrupts_a_quiet_stream(self):
        run, ctx, upstream = await self.prepare()
        reading = asyncio.create_task(run.result())
        try:
            await asyncio.wait_for(upstream.entered.wait(), 20)
            await asyncio.wait_for(run.stop(), 5)
            self.assertIsNotNone(run.process.returncode)
            self.assertFalse(run.home.exists())
            ctx.settle.assert_not_awaited()
        finally:
            await asyncio.gather(reading, return_exceptions=True)
