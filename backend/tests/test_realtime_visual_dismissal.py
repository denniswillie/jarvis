from __future__ import annotations

import json
import unittest
from unittest.mock import AsyncMock

import test_realtime_output_phases as fixtures

from app.realtime.control_tools import DISMISS_VISUAL_INTERRUPT


class RealtimeVisualDismissalTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.case = fixtures.RealtimeOutputPhaseTests()
        await self.case.asyncSetUp()
        self.session = self.case.session
        self.fixture = self.case.fixture

    async def asyncTearDown(self):
        await self.case.asyncTearDown()

    async def dismiss(self, response_id="response-1", call_id="dismiss-1", arguments="{}"):
        await self.session._handle_openai_event({
            "type": "response.function_call_arguments.done", "response_id": response_id,
            "call_id": call_id, "name": DISMISS_VISUAL_INTERRUPT, "arguments": arguments,
        })

    def tool_outputs(self):
        return [
            event["item"] for event in self.fixture.provider.events
            if event.get("item", {}).get("type") == "function_call_output"
        ]

    def terminal_events(self):
        return [
            data for kind, data in self.fixture.app_events
            if kind == "visual.interrupt.done"
        ]

    def advertised_tools(self):
        return {tool["name"] for tool in self.fixture.creates()[-1]["response"].get("tools", [])}

    async def test_dismissal_is_available_only_on_the_visual_response(self):
        await self.case.start_durable("visual_interrupt")
        self.assertIn(DISMISS_VISUAL_INTERRUPT, self.advertised_tools())
        self.assertEqual(self.fixture.creates()[-1]["response"]["tool_choice"], "auto")
        self.assertIsNone(self.session.tool_registry.get(DISMISS_VISUAL_INTERRUPT))
        self.assertIn("not a request to describe the scene", self.fixture.creates()[-1]["response"]["instructions"])
        self.assertIn("Otherwise call dismiss_visual_interrupt", self.fixture.creates()[-1]["response"]["instructions"])

    async def test_success_is_silent_terminal_idempotent_and_not_recorded_as_spoken(self):
        await self.case.start_durable("visual_interrupt")
        await self.dismiss()
        await self.dismiss()
        await fixtures.finish(self.session, "response-1")
        await fixtures.finish(self.session, "response-1")
        self.assertEqual(len(self.tool_outputs()), 1)
        self.assertEqual(json.loads(self.tool_outputs()[0]["output"])["status"], "success")
        self.assertEqual(self.case.transcripts(), [])
        self.assertEqual(self.case.saved(), [])
        self.assertEqual(self.case.errors(), [])
        self.session.visual_repository.mark_visual_interrupt_spoken.assert_not_awaited()
        self.assertEqual(len(self.fixture.creates()), 1)
        self.assertEqual(len(self.terminal_events()), 1)
        self.assertTrue(self.terminal_events()[0]["dismissed"])
        self.assertFalse(self.terminal_events()[0]["forwarded_answer"])
        self.assertIsNone(self.session._tool_followup)
        self.assertIsNone(self.session._current_response_id)
        self.assertFalse(self.session._normal_turn_pending)
        self.assertEqual(self.session._tool_budgets, {})

    async def test_dismissal_after_inspection_does_not_start_another_model_call(self):
        await self.case.start_durable("visual_interrupt")
        self.session.tool_executor.execute = AsyncMock(return_value={"status": "success"})
        await self.case.tool()
        await fixtures.finish(self.session, "response-1")
        self.assertEqual(len(self.fixture.creates()), 2)
        self.assertIn(DISMISS_VISUAL_INTERRUPT, self.advertised_tools())
        await self.fixture.acknowledge(response_id="inspection-followup")
        await self.dismiss(response_id="inspection-followup")
        await fixtures.finish(self.session, "inspection-followup")
        self.assertEqual(len(self.fixture.creates()), 2)
        self.assertEqual(self.case.errors(), [])
        self.assertTrue(self.terminal_events()[0]["dismissed"])
        self.assertEqual(self.session._tool_budgets, {})
        self.session.visual_repository.mark_visual_interrupt_spoken.assert_not_awaited()

    async def test_normal_user_turn_cannot_use_dismissal(self):
        await self.case.start()
        self.assertNotIn(DISMISS_VISUAL_INTERRUPT, self.advertised_tools())
        await self.dismiss()
        self.assertFalse(self.session._active_responses["response-1"].dismissed)
        self.assertEqual(json.loads(self.tool_outputs()[0]["output"])["error"]["code"], "tool_unavailable_for_response")
        await fixtures.finish(self.session, "response-1")
        self.assertEqual(len(self.fixture.creates()), 2)
        self.assertNotIn(DISMISS_VISUAL_INTERRUPT, self.advertised_tools())

    async def assert_durable_cannot_dismiss(self, kind):
        await self.case.start_durable(kind)
        self.assertNotIn(DISMISS_VISUAL_INTERRUPT, self.advertised_tools())
        await self.dismiss()
        self.assertFalse(self.session._active_responses["response-1"].dismissed)
        self.assertEqual(json.loads(self.tool_outputs()[0]["output"])["error"]["code"], "tool_unavailable_for_response")
        await fixtures.finish(self.session, "response-1")
        self.assertNotIn(DISMISS_VISUAL_INTERRUPT, self.advertised_tools())
        await self.fixture.acknowledge(response_id="durable-followup")
        await fixtures.output_item(self.session, "durable-followup", "answer", "final_answer")
        await fixtures.speech(self.session, "durable-followup", "answer", "The requested reminder.")
        await fixtures.finish(self.session, "durable-followup")

    async def test_explicit_reminder_still_requires_delivery(self):
        await self.assert_durable_cannot_dismiss("reminder")
        self.assertEqual(
            self.session.reminder_repository.mark_event_delivery.await_args.kwargs["event_type"],
            "delivered_to_session",
        )

    async def test_prospective_reminder_still_requires_delivery(self):
        await self.assert_durable_cannot_dismiss("prospective")
        self.case.outcome.assert_awaited_once_with("delivered", "durable-followup")

    async def test_malformed_or_nonempty_control_arguments_are_not_dismissals(self):
        await self.case.start_durable("visual_interrupt")
        for index, arguments in enumerate(("{", "[]", '"unexpected"', '{"reason":"not a supported parameter"}')):
            with self.subTest(arguments=arguments):
                await self.dismiss(call_id=f"invalid-{index}", arguments=arguments)
                self.assertFalse(self.session._active_responses["response-1"].dismissed)
                self.assertEqual(json.loads(self.tool_outputs()[-1]["output"])["error"]["code"], "invalid_tool_arguments")
        await fixtures.finish(self.session, "response-1")
        self.assertEqual(len(self.fixture.creates()), 2)

    async def test_late_audio_after_dismissal_is_not_forwarded_or_persisted(self):
        await self.case.start_durable("visual_interrupt")
        await self.dismiss()
        await fixtures.output_item(self.session, "response-1", "unwanted", "final_answer")
        await fixtures.speech(self.session, "response-1", "unwanted", "An unwanted explanation.")
        await fixtures.finish(self.session, "response-1")
        self.assertEqual(self.case.transcripts(), [])
        self.assertEqual(self.case.saved(), [])
        self.assertFalse(any(kind.startswith("audio.output") for kind, _ in self.fixture.app_events))
        self.assertTrue(self.terminal_events()[0]["dismissed"])
        self.assertFalse(self.terminal_events()[0]["forwarded_answer"])

    async def test_already_forwarded_speech_is_not_relabelled_as_silence(self):
        await self.case.start_durable("visual_interrupt")
        await fixtures.output_item(self.session, "response-1", "early", "final_answer")
        await fixtures.speech(self.session, "response-1", "early", "Already forwarded.")
        await self.dismiss()
        await fixtures.finish(self.session, "response-1")
        self.assertEqual(self.case.transcripts(), ["Already forwarded."])
        self.assertTrue(self.terminal_events()[0]["forwarded_answer"])
        self.session.visual_repository.mark_visual_interrupt_spoken.assert_awaited_once()

    async def test_tools_after_dismissal_do_not_mutate_or_start_continuations(self):
        await self.case.start_durable("visual_interrupt")
        self.session.tool_executor.execute = AsyncMock(return_value={"status": "success"})
        await self.dismiss()
        await self.case.tool()
        await fixtures.finish(self.session, "response-1")
        self.session.tool_executor.execute.assert_not_awaited()
        self.assertEqual(json.loads(self.tool_outputs()[-1]["output"])["error"]["code"], "response_dismissed")
        self.assertEqual(len(self.fixture.creates()), 1)
        self.assertEqual(self.case.errors(), [])

    async def test_barge_in_keeps_the_new_user_turn_authoritative(self):
        await self.case.start_durable("visual_interrupt")
        await self.fixture.start_user_response()
        await self.dismiss()
        await fixtures.finish(self.session, "response-1", status="cancelled")
        self.assertEqual(self.tool_outputs(), [])
        self.assertTrue(self.session._normal_turn_pending)
        await self.fixture.acknowledge(response_id="current-user")
        await fixtures.output_item(self.session, "current-user", "answer", "final_answer")
        await fixtures.speech(self.session, "current-user", "answer", "Current answer.")
        await fixtures.finish(self.session, "current-user")
        self.assertEqual(self.case.transcripts(), ["Current answer."])

    async def test_next_user_turn_works_after_dismissal(self):
        await self.case.start_durable("visual_interrupt")
        await self.dismiss()
        await fixtures.finish(self.session, "response-1")
        await self.case.start(response_id="next-user")
        await fixtures.output_item(self.session, "next-user", "answer", "final_answer")
        await fixtures.speech(self.session, "next-user", "answer", "Hello.")
        await fixtures.finish(self.session, "next-user")
        self.assertEqual(self.case.transcripts(), ["Hello."])
        self.assertEqual(self.case.errors(), [])
        self.assertNotIn(DISMISS_VISUAL_INTERRUPT, self.advertised_tools())

    async def test_no_output_without_explicit_dismissal_is_still_an_error(self):
        await self.case.start_durable("visual_interrupt")
        await fixtures.finish(self.session, "response-1")
        self.assertEqual(self.case.errors(), ["no_final_answer"])
        self.assertFalse(self.terminal_events()[0]["dismissed"])
