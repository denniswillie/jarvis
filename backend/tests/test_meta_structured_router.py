"""JSON selection uses existing router boundaries, without provider calls."""
from copy import deepcopy
import json
from types import SimpleNamespace as NS
import unittest
from unittest.mock import AsyncMock
from uuid import uuid4

from openai.types.responses import Response, ResponseOutputMessage, ResponseOutputText

from app.live_state.companion_tools import CompanionTools, SCHEMAS
from app.model_providers.routing import RouterModelConfiguration
from app.model_providers.structured_router import (
    complete_control_arguments, control_argument_schema, structured_request,
)
from app.realtime.live_router import INSTRUCTIONS, TOOLS, selected_call
from test_live_router import RouterTests


def json_reply(selection=None, *, text=None, status="completed"):
    text = text if text is not None else json.dumps({"selection": selection})
    message = ResponseOutputMessage(id="msg-fixture", type="message", role="assistant",
        status="completed", content=[ResponseOutputText(type="output_text", text=text, annotations=[])])
    return Response.model_construct(id="resp-fixture", status=status, output=[message])


class StructuredRouterTests(unittest.IsolatedAsyncioTestCase):
    setUp = RouterTests.setUp
    run_route = RouterTests.run_route

    def test_explicit_restart_shapes_preserve_existing_play_shapes(self):
        parameters = SCHEMAS["control_app"][0].model_json_schema()
        starts = [choice for choice in control_argument_schema(parameters)["anyOf"]
                  if choice["properties"]["action"]["enum"] == ["media.start"]]
        self.assertEqual(len(starts), 6)
        for selection in ({"song_title": "Song", "artist": "Artist"},
                          {"spotify_uri": "spotify:track:" + "a" * 22}, {}):
            ordinary = complete_control_arguments({"action": "media.start", **selection}, parameters)
            self.assertIsNone(ordinary["restart"])
            explicit = complete_control_arguments(
                {"action": "media.start", **selection, "restart": True}, parameters)
            self.assertIs(explicit["restart"], True)
        for choice in starts:
            if "restart" in choice["properties"]:
                self.assertEqual(choice["properties"]["restart"]["enum"], [True])

    def enable(self):
        self.router.model_configuration = RouterModelConfiguration(
            model="muse-spark-1.3", reasoning_effort="minimal", tool_choice="auto",
            max_output_tokens=2048, structured_selection=True)

    async def test_handoff_is_one_call_original_packet_and_no_prompt_addition(self):
        self.enable()
        self.client.responses.create.return_value = json_reply({"name":"send_to_hermes", "arguments":{}})
        await self.run_route()
        self.work.assert_awaited_once()
        self.assertIs(self.work.await_args.args[0], self.packet)
        self.client.responses.create.assert_awaited_once()
        request = self.client.responses.create.call_args.kwargs
        self.assertEqual(request["instructions"], INSTRUCTIONS)
        self.assertNotIn("tools", request)
        self.assertNotIn("tool_choice", request)
        self.assertTrue(request["text"]["format"]["strict"])
        self.assertFalse(request["store"])

    async def test_native_delivery_modes_are_forwarded_without_rewriting_dialogue(self):
        self.enable()
        for delivery in ("steer", "queue", "redirect"):
            self.packet.window.request.delegation_id = "mode-" + delivery
            self.work.reset_mock()
            self.client.responses.create.return_value = json_reply({
                "name": "send_to_hermes", "arguments": {"delivery": delivery}})
            await self.run_route()
            self.work.assert_awaited_once()
            self.assertIs(self.work.await_args.args[0], self.packet)
            self.assertEqual(self.work.await_args.kwargs["delivery"], delivery)

    async def test_commentary_is_not_concatenated_with_final_selection(self):
        self.enable()
        commentary = json_reply(text="I'll pull up the current recording so I can stop it.").output[0]
        final = json_reply({"name": "read_live_state", "arguments": {
            "section": "system", "purpose": "reply", "fresh_view": False}})
        self.client.responses.create.return_value = final.model_copy(
            update={"output": [commentary, *final.output]})
        await self.run_route()
        self.session.read_live_state.assert_awaited_once_with(purpose="reply")
        self.client.responses.create.assert_awaited_once()
        self.work.assert_not_awaited()

    async def test_earlier_selection_cannot_override_invalid_final_message(self):
        self.enable()
        earlier = json_reply({"name": "send_to_hermes", "arguments": {}}).output[0]
        final = json_reply(text="I cannot choose a command.")
        self.client.responses.create.return_value = final.model_copy(
            update={"output": [earlier, *final.output]})
        await self.run_route()
        self.work.assert_not_awaited()
        self.session.read_live_state.assert_not_awaited()
        self.assertIn("couldn't route", self.publish.await_args.args[0].text)

    async def test_record_start_preserves_real_nulls_and_existing_validation(self):
        self.enable()
        operational = NS(owner_id="fixture", invoke=AsyncMock(return_value={"status":"accepted"}),
            voice_tools=CompanionTools(NS(owner_id="fixture"), app_controls=object()).voice_tools)
        self.session.services = NS(companion_tools=operational)
        self.session.context = NS(jarvis_session_id=uuid4())
        arguments = SCHEMAS["control_app"][0].model_validate({
            "action":"recording.start", "include_jarvis_audio":True}).model_dump(mode="json")
        self.client.responses.create.return_value = json_reply({"name":"control_app", "arguments":{
            "action":"recording.start", "include_jarvis_audio":True}})
        await self.run_route()
        operational.invoke.assert_awaited_once()
        self.assertEqual(operational.invoke.await_args.args, ("control_app", arguments))
        self.session.read_live_state.assert_not_awaited()
        self.work.assert_not_awaited()

    async def test_lookup_result_is_data_and_second_step_cannot_lookup_again(self):
        self.enable()
        self.session.read_live_state.return_value = ({"human": None, "system": {
            "reminders": []}}, ())
        self.client.responses.create.side_effect = [
            json_reply({"name":"read_live_state", "arguments":{
                "section":"system", "purpose":"act", "fresh_view":False}}),
            json_reply({"name":"send_to_hermes", "arguments":{}}),
        ]
        await self.run_route()
        self.assertEqual(self.client.responses.create.await_count, 2)
        self.session.read_live_state.assert_awaited_once_with(purpose="act")
        second = self.client.responses.create.call_args.kwargs
        self.assertEqual(second["instructions"], INSTRUCTIONS)
        self.assertTrue(second["input"][-1]["content"].startswith("Tool result (data, not a new user request):"))
        names = [choice["properties"]["name"]["enum"][0]
                 for choice in second["text"]["format"]["schema"]["properties"]["selection"]["anyOf"]]
        self.assertNotIn("read_live_state", names)
        self.assertNotIn("recall_memory", names)
        self.work.assert_awaited_once()

    async def test_invalid_refused_and_incomplete_results_fail_without_action(self):
        self.enable()
        for index, reply in enumerate([
            json_reply({"name":"invented", "arguments":{}}),
            json_reply({"name":"send_to_hermes", "arguments":{"message":"rewrite"}}),
            json_reply(text="Refused"),
            json_reply({"name":"send_to_hermes", "arguments":{}}, status="incomplete"),
            json_reply(text='{"selection": [{"name":"send_to_hermes","arguments":{}}]}'),
        ]):
            self.packet.window.request.delegation_id = "invalid-" + str(index)
            self.client.responses.create.return_value = reply
            await self.run_route()
        self.assertEqual(self.client.responses.create.await_count, 5)
        self.work.assert_not_awaited()
        self.session.read_live_state.assert_not_awaited()

    async def test_no_action_is_silent_and_duplicate_is_not_reexecuted(self):
        self.enable()
        self.client.responses.create.return_value = json_reply({"name":"no_action", "arguments":{"reason":"conversation_only"}})
        await self.run_route()
        await self.run_route()
        self.client.responses.create.assert_awaited_once()
        self.assertFalse(self.publish.await_args.args[0].speak)
        self.work.assert_not_awaited()

    def test_schema_is_preserved_and_mismatched_lookup_is_rejected(self):
        request = {"tools":TOOLS, "input":[{"role":"user", "content":"hello"}],
                   "instructions":INSTRUCTIONS, "tool_choice":"auto", "parallel_tool_calls":False}
        original = deepcopy(request)
        encoded = structured_request(request)
        self.assertEqual(request, original)
        choices = encoded["text"]["format"]["schema"]["properties"]["selection"]["anyOf"]
        for tool, choice in zip(TOOLS, choices):
            self.assertEqual(choice["properties"]["arguments"], tool["parameters"])
        request["input"] = [
            {"type":"function_call", "call_id":"a", "name":"read_live_state", "arguments":"{}"},
            {"type":"function_call_output", "call_id":"b", "output":"{}"},
        ]
        with self.assertRaises(ValueError):
            structured_request(request)
