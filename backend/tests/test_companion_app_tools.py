from types import SimpleNamespace as NS
from unittest.mock import AsyncMock
from uuid import uuid4
import unittest

from app.live_state.companion_tools import CompanionTools
from app.tools.module import ToolBuildContext


class CompanionAppToolsTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.controls = NS(invoke=AsyncMock(return_value={"outcome": "device_command_enqueued"}))
        self.tools = CompanionTools(NS(owner_id="fixture"), app_controls=self.controls)

    async def test_shared_dispatch_keeps_explicit_audio_choice_and_operation_identity(self):
        session, operation = uuid4(), uuid4()
        result = await self.tools.invoke("control_app", {
            "action": "recording.start", "include_jarvis_audio": False},
            owner_id="fixture", session_id=session, operation_id=operation)
        self.assertEqual(result["outcome"], "device_command_enqueued")
        call = self.controls.invoke.await_args
        self.assertEqual(call.args, ("fixture", session))
        self.assertEqual(call.kwargs["operation_id"], operation)
        self.assertIs(call.kwargs["include_jarvis_audio"], False)

    async def test_recording_does_not_default_the_unstated_audio_choice(self):
        with self.assertRaises(ValueError):
            await self.tools.invoke("control_app", {"action": "recording.start"}, owner_id="fixture")
        self.controls.invoke.assert_not_awaited()

    async def test_hermes_uses_same_control_handler(self):
        session = uuid4()
        definition = next(tool for tool in self.tools.build_tools(ToolBuildContext(None, "fixture", session))
                          if tool.name == "control_app")
        view = str(uuid4())
        await definition.handler({"action": "sandbox.stop", "view_id": view})
        self.assertEqual(self.controls.invoke.await_args.kwargs["view_id"], view)

    def test_voice_schema_makes_optional_action_fields_explicit_nulls(self):
        tool = next(tool for tool in self.tools.voice_tools() if tool["name"] == "control_app")
        schema = tool["parameters"]
        self.assertEqual(set(schema["required"]), set(schema["properties"]))
        self.assertFalse(schema["additionalProperties"])
