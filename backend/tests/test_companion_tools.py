from dataclasses import dataclass
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock
import unittest
from uuid import uuid4

from app.live_state.companion import CompanionState
from app.live_state.companion_runtime import initialize_companion, connect_companion_tools
from app.live_state.companion_tools import CompanionTools
from app.realtime.live_router import selected_call
from app.tools.module import ToolBuildContext
from test_companion_state import Memory, Reminders
from test_live_router import response


class OperationalTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.state = CompanionState(owner_id="fixture", working_memory=Memory(), reminders=Reminders())
        await self.state.initialize()
        self.tools = CompanionTools(self.state)

    async def test_disabled_composition_has_no_database_dependency(self):
        services = NS(db_pool=None)
        await initialize_companion(services, environ={})
        self.assertFalse(hasattr(services, "companion_state"))

    async def test_removed_quiet_tool_is_neither_advertised_nor_executable(self):
        args = {"active": True, "duration_seconds": None}
        self.assertNotIn("set_quiet_mode", self.tools.names())
        self.assertNotIn("set_quiet_mode", {tool["name"] for tool in self.tools.voice_tools()})
        for owner in ("fixture", "other"):
            with self.subTest(owner=owner), self.assertRaises(ValueError):
                await self.tools.invoke("set_quiet_mode", args, owner_id=owner)
        self.assertTrue(self.state.notice_allowed())

    async def test_operational_tools_remain_owner_bound(self):
        with self.assertRaises(ValueError):
            await self.tools.invoke("manage_reminder", {
                "action": "create", "title": "Keys", "condition": "Leaving",
                "reminder_id": None, "repeat_until_confirmed": False}, owner_id="other")
        with self.assertRaises(ValueError):
            self.tools.build_tools(ToolBuildContext(None, "other", uuid4()))
        self.assertEqual(self.state.read()["system"]["reminders"], [])

    async def test_no_coercion_or_unrecognized_parameters(self):
        base = {"action": "create", "title": "Keys", "condition": "Leaving",
                "reminder_id": None, "repeat_until_confirmed": False}
        for changes in ({"repeat_until_confirmed": "false"},
                        {"repeat_until_confirmed": 1}, {"owner_id": "other"}):
            with self.assertRaises(ValueError):
                await self.tools.invoke("manage_reminder", {**base, **changes}, owner_id="fixture")
        self.assertEqual(self.state.read()["system"]["reminders"], [])

    async def test_reminder_create_complete_and_exact_id(self):
        created = await self.tools.invoke("manage_reminder", {
            "action": "create", "title": "Umbrella", "condition": "When leaving the office",
            "reminder_id": None, "repeat_until_confirmed": False}, owner_id="fixture")
        self.assertEqual(created["status"], "applied")
        self.assertEqual(len(self.state.read()["system"]["reminders"]), 1)
        done = await self.tools.invoke("manage_reminder", {"action": "complete", "title": None,
            "condition": None, "reminder_id": created["reminder"]["id"],
            "repeat_until_confirmed": False}, owner_id="fixture")
        self.assertEqual(done["reminder"]["status"], "completed")

    async def test_retained_and_voice_share_handlers_but_not_forget_routing(self):
        definitions = self.tools.build_tools(ToolBuildContext(None, "fixture", uuid4()))
        self.assertNotIn("set_quiet_mode", {tool.name for tool in definitions})
        reminder = next(tool for tool in definitions if tool.name == "manage_reminder")
        result = await reminder.handler({"action": "create", "title": "Keys", "condition": "Leaving",
                                        "reminder_id": None, "repeat_until_confirmed": False})
        self.assertEqual(result["status"], "applied")
        self.assertEqual(self.state.read()["system"]["reminders"][0]["title"], "Keys")
        self.assertTrue(reminder.mutates_state)
        self.assertNotIn("forget_memory", {item["name"] for item in self.tools.voice_tools()})

    async def test_forget_uses_single_intake_and_invalidates_matching_intention(self):
        @dataclass
        class Result:
            requested_deletions_applied: bool = True
        intake = NS(forget=AsyncMock(return_value=Result()))
        tools = CompanionTools(self.state, memory_intake=intake)
        row = await self.state.manage_reminder(action="create", title="Check", condition="Leaving")
        result = await tools.invoke("forget_memory", {
            "document_ids": ["actual-source", "actual-source"], "reminder_ids": [row["id"]]},
            owner_id="fixture")
        intake.forget.assert_awaited_once_with(("actual-source",))
        self.assertEqual(result["status"], "applied")
        self.assertEqual(self.state.read()["system"]["reminders"], [])

    def test_operational_schema_selection_is_not_rejected_as_extra_arguments(self):
        tools = self.tools.voice_tools()
        name, args = selected_call(response("manage_reminder",
            '{"action":"create","title":"Keys","condition":"Leaving",'
            '"reminder_id":null,"repeat_until_confirmed":false}'), tools)
        self.assertEqual(name, "manage_reminder")
        self.assertFalse(args["repeat_until_confirmed"])
        for tool in tools:
            schema = tool["parameters"]
            self.assertEqual(set(schema["properties"]), set(schema["required"]))
            self.assertFalse(schema["additionalProperties"])
