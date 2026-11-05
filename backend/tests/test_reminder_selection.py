"""Real routing/handlers, synthetic repositories. --live makes at most two Luna calls."""
import asyncio
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import AsyncMock, Mock
from uuid import UUID

from app.live_state.companion import CompanionState
from app.live_state.companion_tools import CompanionTools
from app.realtime.live_router import LunaLiveRouter
from test_companion_state import Memory, Reminders
from test_live_router import context, response


UTTERANCE = ("Delete only the test reminder to put on my headphones when I say I am "
             "starting the music. Leave other reminders unchanged.")
TARGET = "4efc6c16-170c-50e7-a21c-2f14d73aa969"
OTHER = "b80e65d3-c1ab-50c0-8bf1-71fd8641c252"
SESSION = UUID("1702db23-ec45-4b6c-8aee-f50bf7b61c5c")


def action_args(action="delete", identity=TARGET):
    return {"action": action, "reminder_id": identity, "title": None,
            "condition": None, "repeat_until_confirmed": False}


def scripted(*calls):
    return NS(responses=NS(create=AsyncMock(side_effect=[
        response(name, json.dumps(arguments)) for name, arguments in calls])))


def safe_event(event):
    """Keep fixture requests and tool receipts, never response reasoning/text."""
    item = deepcopy(event)
    if item["stage"] == "router_response":
        value = item["response"]
        item["response"] = {"id": value.get("id"), "status": value.get("status"),
            "calls": [{key: call.get(key) for key in ("name", "arguments", "call_id")}
                      for call in value.get("output", []) if call.get("type") == "function_call"]}
    return item


class ReminderFixture:
    def __init__(self, observer=None):
        self.repository = Reminders()
        self.repository.rows = {identity: {
            "id": identity, "owner_id": "fixture", "title": title,
            "condition": "When I say I am starting the music", "status": "active",
            "repeat_until_confirmed": False} for identity, title in (
                (TARGET, "Test reminder: put on my headphones"),
                (OTHER, "Get my water bottle"))}
        self.before = deepcopy(self.repository.rows)
        self.state = CompanionState(owner_id="fixture", working_memory=Memory(),
                                    reminders=self.repository)
        self.controls = NS(invoke=AsyncMock(side_effect=AssertionError("No app effects allowed")))
        self.tools = CompanionTools(self.state, app_controls=self.controls)
        self.memory = NS(scope=NS(owner_id="fixture"),
                         recall=AsyncMock(side_effect=AssertionError("No memory provider allowed")))
        self.work = AsyncMock(side_effect=AssertionError("No Hermes launch allowed"))
        self.work.route_followups = Mock()
        route = NS(scope=NS(owner_id="fixture"))
        self.packet = NS(route=route, window=NS(request=NS(delegation_id="del-reminder-fixture")),
            as_context=lambda: {"dialogue": [{"speaker": "user", "delta": UTTERANCE}]})
        self.session = NS(work=self.work, request=NS(route=route), ready=True,
            read_live_state=AsyncMock(side_effect=lambda: (self.state.read(), ())),
            services=NS(companion_tools=self.tools, shared_memory=self.memory),
            context=NS(jarvis_session_id=SESSION),
            _source_context=NS(observe=Mock(), context_guard=context))
        self.publish = AsyncMock(return_value=NS(provider_received=False, reason="fixture_only_not_sent"))
        self.events, self.router, self.observer = [], None, observer

    async def initialize(self):
        await self.state.initialize()
        self.state.manage_reminder = AsyncMock(wraps=self.state.manage_reminder)

    def observe(self, event):
        item = safe_event(event)
        self.events.append(item)
        if self.observer:
            self.observer(item)

    async def run(self, client):
        if self.router is None:
            self.router = LunaLiveRouter(self.session, client, self.observe)
        return await self.router(self.packet, publish=self.publish, is_current=lambda: True)

    def checks(self):
        requests = [event for event in self.events if event["stage"] == "router_request"]
        selected = [(event["tool"], event["arguments"]) for event in self.events
                    if event["stage"] == "router_selected"]
        services = [event for event in self.events if event["stage"] == "router_service_result"]
        static = json.dumps([{key: event["request"][key] for key in ("instructions", "tools")}
                             for event in requests])
        return {
            "lookup_then_exact_delete": selected == [
                ("read_live_state", {"section": "system", "purpose": "act"}),
                ("manage_reminder", action_args())],
            "two_selections_one_read_one_mutation": len(requests) == 2
                and self.session.read_live_state.await_count == 1
                and self.state.manage_reminder.await_count == 1,
            "target_deleted": self.repository.rows[TARGET]["status"] == "deleted",
            "unrelated_unchanged": self.repository.rows[OTHER] == self.before[OTHER],
            "only_target_left_active_projection": [row["id"] for row in
                self.state.read()["system"]["reminders"]] == [OTHER],
            "handler_confirmed_applied": any(event["tool"] == "manage_reminder"
                and event["result"].get("status") == "applied" for event in services),
            "first_input_original_dialogue_only": bool(requests) and requests[0]["request"]["input"] == [
                {"role": "user", "content": UTTERANCE}],
            "ids_only_in_lookup_data": len(requests) == 2
                and TARGET in requests[1]["request"]["input"][-1].get("output", "")
                and all(identity not in static for identity in (TARGET, OTHER))
                and "Get my water bottle" not in static,
            "no_removed_quiet_tool": all(tool["name"] != "set_quiet_mode"
                for event in requests for tool in event["request"]["tools"]),
            "no_external_effects_or_speech": self.work.await_count == 0
                and self.controls.invoke.await_count == 0 and self.memory.recall.await_count == 0
                and self.publish.await_count == 1,
        }


class ReminderSelectionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.fixture = ReminderFixture()
        await self.fixture.initialize()

    def lookup_then(self, action="delete"):
        return scripted(("read_live_state", {"section": "system", "purpose": "act"}),
                        ("manage_reminder", action_args(action)))

    async def test_lookup_exact_delete_preserves_unrelated_and_static_instructions(self):
        await self.fixture.run(self.lookup_then())
        self.assertEqual(self.fixture.checks(), dict.fromkeys(self.fixture.checks(), True))

    async def test_same_delegation_does_not_repeat_lookup_or_mutation(self):
        client = self.lookup_then()
        await self.fixture.run(client)
        await self.fixture.run(client)
        self.assertEqual(client.responses.create.await_count, 2)
        self.fixture.state.manage_reminder.assert_awaited_once()
        self.fixture.publish.assert_awaited_once()

    async def test_lookup_complete_preserves_unrelated(self):
        await self.fixture.run(self.lookup_then("complete"))
        self.assertEqual(self.fixture.repository.rows[TARGET]["status"], "completed")
        self.assertEqual(self.fixture.repository.rows[OTHER], self.fixture.before[OTHER])

    async def test_create_still_needs_no_id_lookup(self):
        args = {"action": "create", "reminder_id": None, "title": "Keys",
                "condition": "When leaving home", "repeat_until_confirmed": False}
        await self.fixture.run(scripted(("manage_reminder", args)))
        self.fixture.session.read_live_state.assert_not_awaited()
        self.assertEqual(len(self.fixture.repository.rows), 3)
        self.assertEqual(json.loads(self.fixture.publish.await_args.args[0].text)["status"], "applied")

    async def test_invalid_delete_parameters_fail_closed_without_repository_change(self):
        for changes in ({"reminder_id": None}, {"reminder_id": "headphones"},
                        {"title": "headphones"}, {"condition": "starting music"}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                await self.fixture.tools.invoke("manage_reminder", action_args() | changes,
                                                owner_id="fixture")
        self.fixture.state.manage_reminder.assert_not_awaited()
        self.assertEqual(self.fixture.repository.rows, self.fixture.before)

    async def test_direct_null_id_reproduces_unconfirmed_not_success(self):
        await self.fixture.run(scripted(("manage_reminder", action_args(identity=None))))
        self.assertEqual(json.loads(self.fixture.publish.await_args.args[0].text)["status"], "unconfirmed")
        self.fixture.state.manage_reminder.assert_not_awaited()
        self.assertEqual(self.fixture.repository.rows, self.fixture.before)

    def test_field_contract_is_exposed_in_strict_voice_schema(self):
        schema = next(tool["parameters"] for tool in self.fixture.tools.voice_tools()
                      if tool["name"] == "manage_reminder")
        fields = schema["properties"]
        self.assertEqual(set(schema["required"]), set(fields))
        self.assertFalse(schema["additionalProperties"])
        self.assertIn("read_live_state", fields["reminder_id"]["description"])
        self.assertIn("purpose=act", fields["reminder_id"]["description"])
        for name in ("title", "condition"):
            self.assertIn("Complete/delete: null", fields[name]["description"])


async def live_check(output):
    from dotenv import dotenv_values
    from openai import AsyncOpenAI

    output.mkdir(parents=True, exist_ok=False)
    with (output / "events.jsonl").open("x") as log:
        def observe(event):
            log.write(json.dumps(event, ensure_ascii=True) + "\n")
            log.flush()

        fixture = ReminderFixture(observe)
        await fixture.initialize()
        error = None
        try:
            key = dotenv_values(Path(__file__).resolve().parents[1] / ".env")["OPENAI_API_KEY"]
            async with AsyncOpenAI(api_key=key, max_retries=0) as client:
                await fixture.run(client)
        except Exception as exc:
            error = type(exc).__name__
        checks = fixture.checks()
        requests = {event["step"]: event for event in fixture.events if event["stage"] == "router_request"}
        responses = [event for event in fixture.events if event["stage"] == "router_response"]
        result = {"observed_at": datetime.now(timezone.utc).isoformat(),
            "scope": "Synthetic repositories only; no Live speech, native work, or serving changes",
            "utterance": UTTERANCE, "model": "gpt-6-luna", "provider_error_type": error,
            "passed": error is None and all(checks.values()), "checks": checks,
            "selection_latency_seconds": [round(event["monotonic_s"] -
                requests[event["step"]]["monotonic_s"], 3) for event in responses],
            "selected": [{"tool": event["tool"], "arguments": event["arguments"]}
                for event in fixture.events if event["stage"] == "router_selected"],
            "instructions_sha256": hashlib.sha256(requests[1]["request"]["instructions"].encode()).hexdigest()
                if requests else None,
            "fixture_rows": fixture.repository.rows,
            "limitation": "One isolated selection attempt, not serving cleanup or a reliability estimate; "
                          "original failed trial arguments were not recorded."}
        (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
        return result["passed"]


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--live":
        sys.exit(0 if asyncio.run(live_check(Path(sys.argv[2]))) else 1)
    unittest.main()
