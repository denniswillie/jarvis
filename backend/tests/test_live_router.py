"""Single-pass routing boundaries, without provider calls or database writes."""
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import json
from types import SimpleNamespace as NS
import unittest
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

from app.realtime.live_router import LunaLiveRouter, TOOLS, function, router_input
from app.tasks.models import TaskError


@asynccontextmanager
async def context(**kwargs):
    yield


def response(name, arguments="{}", *, status="completed"):
    return NS(status=status, output=[NS(type="function_call", name=name, arguments=arguments, call_id="call-fixture")],
              model_dump=lambda **kwargs: {"status": status, "tool": name, "arguments": arguments})


class RouterTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.route = NS(scope=NS(owner_id="fixture"))
        self.root = uuid4()
        self.packet = NS(route=self.route, window=NS(request=NS(delegation_id="del-fixture")),
            observation_payload_ids=(uuid4(),),
            as_context=lambda: {"dialogue": [{"speaker": "user", "delta": "What can you see?"}],
                                "window": {"speech_finality": "unavailable"}})
        self.work = AsyncMock()
        self.work.route_followups = Mock()
        self.work.router_directory = Mock(return_value={"sandboxes": [], "reminders": [], "partial": True})
        self.session = NS(work=self.work, request=NS(route=self.route), ready=True,
            read_live_state=AsyncMock(return_value=({"text": "No current visual observation."}, (self.root,))),
            _source_context=NS(observe=Mock(), context_guard=context))
        self.client = NS(responses=NS(create=AsyncMock(return_value=response("read_live_state"))))
        self.events = []
        self.router = LunaLiveRouter(self.session, self.client, self.events.append)
        self.publish = AsyncMock(return_value=NS(provider_received=True, reason=None))

    async def run_route(self, current=lambda: True):
        return await self.router(self.packet, publish=self.publish, is_current=current)

    async def test_direct_read_is_one_model_call_no_hermes_or_payload_write(self):
        await self.run_route()
        self.session.read_live_state.assert_awaited_once_with(purpose="reply")
        self.client.responses.create.assert_awaited_once()
        self.work.assert_not_awaited()
        result = self.publish.await_args.args[0]
        self.assertIsNone(result.task_id)
        self.assertEqual(result.roots, (self.root,))
        self.assertIn("No current visual", result.text)
        self.assertNotIn("No current visual", self.client.responses.create.call_args.kwargs["input"][0]["content"])
        self.work.tasks.payload.assert_not_awaited()
        self.work.router_directory.assert_not_called()

    async def test_lookup_then_action_uses_tool_result_not_instruction_injection(self):
        operational = NS(owner_id="fixture", invoke=AsyncMock(return_value={"status": "completed"}),
            voice_tools=lambda: [function("manage_reminder", "Manage a reminder.",
                {"reminder_id": {"type": "string"}})])
        self.session.services = NS(companion_tools=operational)
        self.session.context = NS(jarvis_session_id=uuid4())
        self.session.read_live_state.return_value = ({"human": None, "system": {
            "reminders": [{"id": "known-reminder", "title": "Take keys"}]}}, ())
        self.client.responses.create.side_effect = [
            response("read_live_state", '{"section":"system","purpose":"act"}'),
            response("manage_reminder", '{"reminder_id":"known-reminder"}')]
        await self.run_route()
        self.assertEqual(self.client.responses.create.await_count, 2)
        second = self.client.responses.create.call_args_list[1].kwargs
        self.assertEqual(second["input"][-1]["type"], "function_call_output")
        self.assertEqual(second["input"][-1]["call_id"], "call-fixture")
        self.assertIn("known-reminder", second["input"][-1]["output"])
        self.assertFalse(any(item.get("role") == "developer" for item in second["input"]))
        self.assertNotIn("read_live_state", {tool["name"] for tool in second["tools"]})
        self.assertEqual(operational.invoke.await_args.args, ("manage_reminder", {"reminder_id": "known-reminder"}))
        self.session.read_live_state.assert_awaited_once_with(purpose="act")
        self.work.assert_not_awaited()
        self.publish.assert_awaited_once()

    async def test_confirmed_recording_start_needs_no_state_lookup(self):
        from app.live_state.companion_tools import SCHEMAS
        command_model, description = SCHEMAS["control_app"]
        arguments = dict(action="recording.start", include_jarvis_audio=True,
                         recording_id=None, allocation_id=None, generation=None,
                         resume=False, view_id=None)
        arguments = command_model.model_validate(arguments).model_dump(mode="json")
        schema = command_model.model_json_schema()
        schema["required"] = list(schema["properties"])
        tool = dict(type="function", name="control_app", strict=True,
                    description=description, parameters=schema)
        operational = NS(owner_id="fixture", invoke=AsyncMock(return_value={"status": "accepted"}),
                         voice_tools=lambda: [tool])
        self.session.services = NS(companion_tools=operational)
        self.session.context = NS(jarvis_session_id=uuid4())
        self.packet.as_context = lambda: {"dialogue": [
            {"speaker": "user", "delta": "Start recording, please"},
            {"speaker": "assistant", "delta": "Include my voice in the recording?"},
            {"speaker": "user", "delta": "Yes, of course"}]}
        self.client.responses.create.return_value = response("control_app", json.dumps(arguments))
        await self.run_route()
        self.client.responses.create.assert_awaited_once()
        self.session.read_live_state.assert_not_awaited()
        self.work.assert_not_awaited()
        self.assertEqual(operational.invoke.await_args.args, ("control_app", arguments))
        request = self.client.responses.create.call_args.kwargs
        self.assertIn("call recording.start directly", request["instructions"])
        self.assertIn("no ID or state lookup", description)
        self.publish.assert_awaited_once()

    async def test_lookup_cannot_start_a_third_selection(self):
        self.client.responses.create.side_effect = [
            response("read_live_state", '{"section":"all","purpose":"act"}'),
            response("read_live_state", '{"section":"all","purpose":"act"}')]
        await self.run_route()
        self.assertEqual(self.client.responses.create.await_count, 2)
        self.session.read_live_state.assert_awaited_once()
        self.work.assert_not_awaited()
        self.assertIn("No action was started", self.publish.await_args.args[0].text)

    async def test_failed_lookup_does_not_select_an_action(self):
        self.client.responses.create.return_value = response("read_live_state", '{"section":"all","purpose":"act"}')
        self.session.read_live_state.side_effect = TimeoutError()
        await self.run_route()
        self.client.responses.create.assert_awaited_once()
        self.work.assert_not_awaited()
        self.assertIn("unavailable", self.publish.await_args.args[0].text)

    async def test_empty_argument_handoff_preserves_original_packet(self):
        self.client.responses.create.return_value = response("send_to_hermes")
        await self.run_route()
        self.assertIs(self.work.await_args.args[0], self.packet)
        self.session.read_live_state.assert_not_awaited()
        self.publish.assert_not_awaited()

    async def test_delivery_modes_reach_work_without_rewriting_dialogue(self):
        for mode in ("steer", "queue", "redirect"):
            with self.subTest(mode=mode):
                self.router.seen.clear()
                self.client.responses.create.return_value = response("send_to_hermes", json.dumps({"delivery": mode}))
                await self.run_route()
                self.assertIs(self.work.await_args.args[0], self.packet)
                self.assertEqual(self.work.await_args.kwargs["delivery"], mode)
        self.session.read_live_state.assert_not_awaited()

    async def test_invalid_delivery_mode_never_executes_work(self):
        self.client.responses.create.return_value = response("send_to_hermes", '{"delivery":"kill"}')
        await self.run_route()
        self.work.assert_not_awaited()

    async def test_human_lookup_projects_state_without_system_metadata(self):
        snapshot = {
            "human": {"revision": 3, "content": {"relevant_details": [{
                "description": "Card number 47", "basis": "observed",
                "evidence_at": "2026-11-04T04:26:36Z", "sources": ["frame:test"]}],
                "uncertainties": []}},
            "system": {"reminders": [{"id": "reminder-1", "title": "Other work"}]}}
        self.session.read_live_state.return_value = (snapshot, (self.root,))
        self.client.responses.create.return_value = response("read_live_state", '{"section":"human"}')
        await self.run_route()
        result = self.publish.await_args.args[0]
        value = json.loads(result.text)
        self.assertIn("Card number 47", result.text)
        self.assertIn("frame:test", result.text)
        self.assertIn("2026-11-04T04:26:36Z", result.text)
        self.assertNotIn("system", value)
        self.assertEqual(snapshot["system"]["reminders"][0]["id"], "reminder-1")
        self.assertEqual(result.roots, (self.root,))
        self.work.assert_not_awaited()

    async def test_invalid_state_section_does_not_read_or_execute(self):
        self.client.responses.create.return_value = response("read_live_state", '{"section":"everything_else"}')
        await self.run_route()
        self.session.read_live_state.assert_not_awaited()
        self.work.assert_not_awaited()
        self.assertIn("No action was started", self.publish.await_args.args[0].text)

    async def test_current_view_requests_fresh_evidence_not_cached_state(self):
        self.session.read_live_state.return_value = ({"human": {"current_view": {
            "status": "fresh", "facts": ["A standing fan is directly ahead."]}}, "system": {}}, ())
        self.client.responses.create.return_value = response("read_live_state",
            '{"section":"human","purpose":"reply","fresh_view":true}')
        await self.run_route()
        self.session.read_live_state.assert_awaited_once_with(fresh_view=True)
        self.client.responses.create.assert_awaited_once()
        self.assertIn("standing fan", self.publish.await_args.args[0].text)
        self.work.assert_not_awaited()

    async def test_current_view_timeout_never_falls_back_to_cached_scene(self):
        self.client.responses.create.return_value = response("read_live_state",
            '{"section":"human","purpose":"reply","fresh_view":true}')
        self.session.read_live_state.side_effect = TimeoutError()
        await self.run_route()
        self.session.read_live_state.assert_awaited_once_with(fresh_view=True)
        self.assertIn("freshness_unavailable", self.publish.await_args.args[0].text)
        self.assertIn("do not describe cached", self.publish.await_args.args[0].text)

    async def test_fresh_view_cannot_be_a_system_action_lookup(self):
        self.client.responses.create.return_value = response("read_live_state",
            '{"section":"system","purpose":"act","fresh_view":true}')
        await self.run_route()
        self.session.read_live_state.assert_not_awaited()
        self.work.assert_not_awaited()

    async def test_native_memory_read_is_direct_and_owner_bound(self):
        memory = NS(scope=NS(owner_id="fixture"), recall=AsyncMock(return_value={
            "results": [{"text": "The keys were last put in the drawer.", "document_id": "source-1"}]}))
        self.session.services = NS(shared_memory=memory)
        self.client.responses.create.return_value = response("recall_memory", '{"query":"Where were my keys?"}')
        now = datetime(2026, 11, 4, 5, 0, tzinfo=timezone.utc)
        with patch("app.realtime.live_router.utc_now", return_value=now):
            await self.run_route()
        self.client.responses.create.assert_awaited_once()
        memory.recall.assert_awaited_once_with("Where were my keys?", max_tokens=1000, budget="low", at=now)
        request = self.client.responses.create.call_args.kwargs
        self.assertNotIn(now.isoformat(), request["instructions"])
        self.assertNotIn(now.isoformat(), json.dumps(request["input"]))
        self.work.assert_not_awaited()
        self.assertIn("source-1", self.publish.await_args.args[0].text)
        self.assertEqual(self.publish.await_args.args[0].roots, ())

    async def test_memory_for_another_owner_is_not_offered_or_read(self):
        memory = NS(scope=NS(owner_id="other"), recall=AsyncMock())
        self.session.services = NS(shared_memory=memory)
        await self.run_route()
        self.assertNotIn("recall_memory", {tool["name"] for tool in self.client.responses.create.call_args.kwargs["tools"]})
        memory.recall.assert_not_awaited()

    async def test_old_task_selection_tools_are_not_offered(self):
        self.assertEqual({t["name"] for t in TOOLS}, {"read_live_state", "send_to_hermes", "no_action"})
        self.assertEqual(set(TOOLS[1]["parameters"]["properties"]), {"delivery"})
        self.assertEqual(TOOLS[1]["parameters"]["properties"]["delivery"]["enum"],
                         ["steer", "queue", "redirect"])
        self.client.responses.create.return_value = response("steer_work", '{"work_id":"invented"}')
        await self.run_route()
        self.work.assert_not_awaited()
        self.assertIn("No action was started", self.publish.await_args.args[0].text)

    async def test_more_than_32_delegations_remain_usable(self):
        for index in range(40):
            self.packet.window.request.delegation_id = "del-" + str(index)
            await self.run_route()
        self.assertEqual(self.session.read_live_state.await_count, 40)

    async def test_duplicate_delegation_does_not_repeat(self):
        await self.run_route()
        await self.run_route()
        self.client.responses.create.assert_awaited_once()
        self.publish.assert_awaited_once()

    async def test_smalltalk_no_service_call(self):
        self.client.responses.create.return_value = response("no_action", '{"reason":"conversation_only"}')
        await self.run_route()
        self.work.assert_not_awaited()
        self.session.read_live_state.assert_not_awaited()
        self.publish.assert_awaited_once()
        result = self.publish.await_args.args[0]
        self.assertFalse(result.speak)
        self.assertEqual(result.delegation_id, "del-fixture")
        self.assertEqual(json.loads(result.text)["status"], "no_action")
        self.assertIn("No backend work was started", result.text)
        await self.run_route()
        self.publish.assert_awaited_once()
        self.client.responses.create.assert_awaited_once()

    async def test_clarification_still_returns_a_speakable_reply_without_work(self):
        self.client.responses.create.return_value = response("no_action", '{"reason":"clarification_needed"}')
        await self.run_route()
        self.work.assert_not_awaited()
        self.session.read_live_state.assert_not_awaited()
        self.publish.assert_awaited_once()
        self.assertTrue(self.publish.await_args.args[0].speak)
        self.assertIn("clarify", self.publish.await_args.args[0].text)

    async def test_route_change_during_selection_prevents_execution(self):
        current = iter((True, False))
        with self.assertRaises(TaskError):
            await self.run_route(lambda: next(current))
        self.session.read_live_state.assert_not_awaited()
        self.work.assert_not_awaited()

    async def test_malformed_or_multiple_calls_do_not_execute(self):
        self.client.responses.create.return_value = response("send_to_hermes", '{"message":"rewrite"}')
        await self.run_route()
        self.work.assert_not_awaited()
        self.packet.window.request.delegation_id = "second"
        value = response("read_live_state")
        value.output.append(NS(type="function_call", name="send_to_hermes", arguments="{}"))
        self.client.responses.create.return_value = value
        await self.run_route()
        self.session.read_live_state.assert_not_awaited()
        self.work.assert_not_awaited()

    async def test_unavailable_state_is_not_fabricated(self):
        self.session.read_live_state.side_effect = TimeoutError()
        await self.run_route()
        self.assertIn("unavailable", self.publish.await_args.args[0].text)
        self.work.assert_not_awaited()

    async def test_provider_error_does_not_retry(self):
        self.client.responses.create.side_effect = TimeoutError()
        await self.run_route()
        self.client.responses.create.assert_awaited_once()
        self.work.assert_not_awaited()

    async def test_single_delivery_guard_preserves_source_invalidation(self):
        await self.run_route()
        result = self.publish.await_args.args[0]
        @asynccontextmanager
        async def withdrawn(**kwargs):
            self.assertEqual(kwargs["extra_sources"], (self.root,))
            raise TaskError("withdrawn", "The source was withdrawn")
            yield
        self.session._source_context.context_guard = withdrawn
        with self.assertRaises(TaskError):
            async with self.router.guard(result):
                self.fail("withdrawn data reached delivery")

    def test_only_original_dialogue_enters_router_input(self):
        original = {"dialogue": [{"speaker": "user", "delta": "first\n"},
                                 {"speaker": "user", "delta": "second"}],
                    "application_context": {"large": "must not copy automatically"}}
        content = router_input(original)
        self.assertEqual(content, [{"role": "user", "content": "first\nsecond"}])
        self.assertNotIn("must not copy", str(content))
