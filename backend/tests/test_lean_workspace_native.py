"""Native event-loop integration with controlled domains; no provider/network."""
import asyncio
import json
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock

import test_lean_voice_session as fixtures


class NativeWorkspaceBoundaryTests(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = fixtures.LeanSessionTests.asyncSetUp
    asyncTearDown = fixtures.LeanSessionTests.asyncTearDown
    created = fixtures.LeanSessionTests.created
    done = fixtures.LeanSessionTests.done
    drain = fixtures.LeanSessionTests.drain

    async def send_tool(self, raw='{"action":"create"}'):
        await self.session._handle_openai_event({"type": "response.function_call_arguments.done",
            "response_id": "resp_test", "call_id": "call_note", "name": "live_note", "arguments": raw})

    async def test_domain_work_does_not_hold_provider_writer_or_reader(self):
        entered, release = asyncio.Event(), asyncio.Event()
        in_writer = [False]
        original_send = self.provider.send_event
        async def writer(payload, **kwargs):
            in_writer[0] = True
            try:
                return await original_send(payload, **kwargs)
            finally:
                in_writer[0] = False
        self.provider.send_event = writer
        async def execute(*args, **kwargs):
            self.assertFalse(in_writer[0], "Domain work must happen outside provider writer admission")
            entered.set()
            await release.wait()
            return {"status": "success", "outcome": "completed", "effect_committed": True,
                    "summary": "Saved the fixture note."}
        dispatcher = SimpleNamespace(handles=lambda name: name == "live_note", execute=AsyncMock(side_effect=execute))
        self.session._lean_workspace = dispatcher
        await self.created()
        await self.send_tool()
        await asyncio.wait_for(entered.wait(), .5)
        await self.done()
        # The provider receiver remains usable while the domain operation waits.
        await self.session._handle_openai_event({"type": "input_audio_buffer.speech_started", "item_id": "new_speech"})
        await self.provider.send_event({"type": "fixture.control"})
        self.assertIn("fixture.control", [event["type"] for event in self.provider.sent])
        release.set()
        await self.drain()
        dispatcher.execute.assert_awaited_once()
        outputs = [event for event in self.provider.sent if event["type"] == "conversation.item.create"]
        self.assertEqual(len(outputs), 1)
        self.assertTrue(json.loads(outputs[0]["item"]["output"])["effect_committed"])
        self.assertNotIn("response.create", [event["type"] for event in self.provider.sent])

    async def test_invalid_json_does_not_dispatch_domain(self):
        dispatcher = SimpleNamespace(handles=lambda name: True, execute=AsyncMock())
        self.session._lean_workspace = dispatcher
        await self.created()
        await self.send_tool("not JSON")
        await self.drain()
        dispatcher.execute.assert_not_awaited()
        output = next(event for event in self.provider.sent if event["type"] == "conversation.item.create")
        self.assertEqual(json.loads(output["item"]["output"])["error"]["code"], "invalid_arguments")

    async def test_note_content_has_bounded_room_beyond_briefing_arguments(self):
        dispatcher = SimpleNamespace(handles=lambda name: True, execute=AsyncMock(return_value={
            "status": "success", "outcome": "completed", "effect_committed": True, "summary": "Saved."}))
        self.session._lean_workspace = dispatcher
        await self.created()
        await self.send_tool(json.dumps({"action": "create", "text": "x" * 6000}))
        await self.drain()
        dispatcher.execute.assert_awaited_once()

    async def test_oversized_note_never_reaches_domain(self):
        dispatcher = SimpleNamespace(handles=lambda name: True, execute=AsyncMock())
        self.session._lean_workspace = dispatcher
        await self.created()
        await self.send_tool(json.dumps({"text": "x" * (256 * 1024)}))
        await self.drain()
        dispatcher.execute.assert_not_awaited()
