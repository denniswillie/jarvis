from __future__ import annotations

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import test_generic_visual_speech as speech_fixtures
import test_realtime_shared_view as visual_fixtures


async def output_item(session, response_id, item_id, phase, *, index=0, event_type="response.output_item.added"):
    item = {"id": item_id, "type": "message", "role": "assistant"}
    if phase is not None:
        item["phase"] = phase
    await session._handle_openai_event({
        "type": event_type, "response_id": response_id, "output_index": index, "item": item,
    })


async def speech(session, response_id, item_id, text):
    for event_type, fields in (
        ("response.output_audio.delta", {"delta": "AA=="}),
        ("response.output_audio_transcript.delta", {"delta": text}),
        ("response.output_audio_transcript.done", {"transcript": text}),
        ("response.output_audio.done", {}),
    ):
        await session._handle_openai_event({
            "type": event_type, "response_id": response_id, "item_id": item_id, **fields,
        })


async def finish(session, response_id, *, status="completed"):
    await session._handle_openai_event({
        "type": "response.done", "response": {"id": response_id, "status": status},
    })


class RealtimeOutputPhaseTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.fixture = speech_fixtures.GenericVisualSpeechTests()
        await self.fixture.asyncSetUp()
        self.session = self.fixture.session
        self.attach = AsyncMock()
        self.session.experience_event_service = SimpleNamespace(attach_message=self.attach)

    async def asyncTearDown(self):
        await self.fixture.asyncTearDown()

    async def start(self, response_id="response-1"):
        await self.fixture.start_user_response()
        await self.fixture.acknowledge(response_id=response_id)

    def transcripts(self):
        return [
            data["text"] for kind, data in self.fixture.app_events
            if kind == "transcript.done"
        ]

    def errors(self):
        return [data["code"] for kind, data in self.fixture.app_events if kind == "error"]

    def saved(self):
        return [call.kwargs for call in self.fixture.repository.create_message.await_args_list]

    async def tool(self, response_id="response-1", call_id="call-1"):
        item = {"type": "function_call", "id": call_id, "call_id": call_id,
                "name": "example", "arguments": "{}"}
        await self.session._handle_openai_event({
            "type": "response.output_item.done", "response_id": response_id, "item": item,
        })
        return item

    async def test_commentary_tool_continuation_keeps_only_final_speech_and_history(self):
        await self.start()
        await output_item(self.session, "response-1", "intro", "commentary")
        await output_item(self.session, "response-1", "intro", "commentary", event_type="response.output_item.created")
        await speech(self.session, "response-1", "intro", "PRIVATE_SUPPRESSED_PREAMBLE")
        self.session.tool_executor.execute = AsyncMock(return_value={"status": "success", "result": "saved"})
        item = await self.tool()
        await self.session._handle_openai_event({
            **item, "type": "response.function_call_arguments.done", "response_id": "response-1",
        })
        self.session.tool_executor.execute.assert_awaited_once()
        self.assertEqual(self.transcripts(), [])
        self.assertEqual(self.saved(), [])
        self.attach.assert_not_awaited()
        await finish(self.session, "response-1")
        self.assertEqual(len(self.fixture.creates()), 2)
        self.assertNotIn("no_final_answer", self.errors())
        tool_outputs = [
            event["item"] for event in self.fixture.provider.events
            if event.get("item", {}).get("type") == "function_call_output"
        ]
        self.assertEqual(len(tool_outputs), 1)
        self.assertIn("saved", tool_outputs[0]["output"])
        await self.fixture.acknowledge(response_id="response-2")
        await output_item(self.session, "response-2", "answer", "final_answer")
        await speech(self.session, "response-2", "answer", "Saved the note.")
        await finish(self.session, "response-2")
        self.assertEqual(self.transcripts(), ["Saved the note."])
        self.assertEqual([message["content"] for message in self.saved()], ["Saved the note."])
        self.attach.assert_awaited_once()
        self.assertFalse(self.session._normal_turn_pending)
        self.assertFalse(any(event["type"] == "conversation.item.delete" for event in self.fixture.provider.events))

    async def test_mixed_response_preserves_all_final_items_in_provider_order(self):
        await self.start()
        await output_item(self.session, "response-1", "intro", "commentary", index=0)
        await speech(self.session, "response-1", "intro", "hidden")
        # Index, rather than completion order, determines durable ordering.
        for item_id, index, text in (("second", 2, "Second result."), ("first", 1, "First result.")):
            await output_item(self.session, "response-1", item_id, "final_answer", index=index)
            await speech(self.session, "response-1", item_id, text)
            await output_item(self.session, "response-1", item_id, "final_answer", index=index, event_type="response.output_item.done")
            await self.session._handle_openai_event({
                "type": "response.output_audio_transcript.done", "response_id": "response-1",
                "item_id": item_id, "transcript": text,
            })
        await finish(self.session, "response-1")
        await finish(self.session, "response-1")
        self.assertEqual([message["content"] for message in self.saved()], ["First result.", "Second result."])
        self.assertEqual([message["provider_item_id"] for message in self.saved()], ["first", "second"])
        self.assertEqual(len(self.transcripts()), 2)
        self.assertEqual(self.attach.await_count, 2)
        self.assertTrue(all(message["metadata"]["realtime_output"]["phase"] == "final_answer" for message in self.saved()))
        self.assertEqual(self.errors(), [])

    async def test_final_only_missing_phase_and_unknown_phase_keep_legacy_delivery(self):
        for index, phase in enumerate(("final_answer", None, "future_provider_phase")):
            with self.subTest(phase=phase):
                response_id = f"response-{index}"
                await self.start(response_id)
                await output_item(self.session, response_id, f"item-{index}", phase)
                await speech(self.session, response_id, f"item-{index}", f"Answer {index}.")
                await finish(self.session, response_id)
        self.assertEqual(len(self.transcripts()), 3)
        self.assertEqual(len(self.saved()), 3)
        self.assertEqual(self.errors(), [])

    async def test_late_phase_pins_forwarding_and_records_the_anomaly(self):
        await self.start()
        await output_item(self.session, "response-1", "late", None)
        await self.session._handle_openai_event({
            "type": "response.output_audio_transcript.delta",
            "response_id": "response-1", "item_id": "late", "delta": "Already ",
        })
        await output_item(self.session, "response-1", "late", "commentary", event_type="response.output_item.done")
        await speech(self.session, "response-1", "late", "Already started, finish this sentence.")
        await finish(self.session, "response-1")
        self.assertEqual(self.transcripts(), ["Already started, finish this sentence."])
        metadata = self.saved()[0]["metadata"]["realtime_output"]
        self.assertEqual(metadata["delivery_disposition"], "forwarded")
        self.assertTrue(metadata["phase_anomaly"])
        self.assertIn("late_commentary_phase", self.session._output_policy_notices)

    async def test_conflicting_phase_cannot_unmute_a_suppressed_item(self):
        await self.start()
        await output_item(self.session, "response-1", "intro", "commentary")
        await speech(self.session, "response-1", "intro", "hidden")
        await output_item(self.session, "response-1", "intro", "final_answer", event_type="response.output_item.done")
        await finish(self.session, "response-1")
        self.assertEqual(self.transcripts(), [])
        self.assertEqual(self.saved(), [])
        self.assertIn("conflicting_phase", self.session._output_policy_notices)
        self.assertEqual(self.errors(), ["no_final_answer"])
        self.assertEqual(len(self.fixture.creates()), 1)
        self.assertFalse(self.session._normal_turn_pending)

    async def test_terminal_commentary_is_recoverable_and_duplicate_or_retired_output_is_ignored(self):
        await self.start()
        await output_item(self.session, "response-1", "intro", "commentary")
        await speech(self.session, "response-1", "intro", "hidden")
        await finish(self.session, "response-1")
        events = len(self.fixture.app_events)
        await finish(self.session, "response-1")
        await self.fixture.acknowledge(response_id="response-1")
        await speech(self.session, "response-1", "intro", "late output")
        await self.session._handle_openai_event({
            "type": "response.output_audio.delta", "item_id": "intro", "delta": "AA==",
        })
        self.assertEqual(len(self.fixture.app_events), events)
        self.assertEqual(self.errors(), ["no_final_answer"])
        self.assertEqual(len(self.fixture.creates()), 1)
        self.assertNotIn("response-1", self.session._active_responses)
        self.assertEqual(self.saved(), [])

    async def test_barge_in_blocks_old_media_without_settling_the_new_turn(self):
        await self.start()
        await output_item(self.session, "response-1", "intro", "commentary")
        await speech(self.session, "response-1", "intro", "hidden")
        await self.start("new-response")
        await output_item(self.session, "response-1", "late-answer", "final_answer")
        await speech(self.session, "response-1", "late-answer", "obsolete")
        await finish(self.session, "response-1", status="cancelled")
        self.assertEqual(self.transcripts(), [])
        self.assertTrue(self.session._normal_turn_pending)
        self.assertEqual(self.session._current_response_id, "new-response")
        await output_item(self.session, "new-response", "answer", "final_answer")
        await speech(self.session, "new-response", "answer", "Current answer.")
        await finish(self.session, "new-response")
        self.assertEqual([message["content"] for message in self.saved()], ["Current answer."])

    async def test_stop_clears_output_state_and_blocks_late_media(self):
        await self.start()
        await output_item(self.session, "response-1", "intro", "commentary")
        await speech(self.session, "response-1", "intro", "hidden")
        await self.session.close("client_closed")
        count = len(self.fixture.app_events)
        await speech(self.session, "response-1", "answer", "late")
        self.assertEqual(len(self.fixture.app_events), count)
        self.assertEqual(self.session._active_responses, {})
        self.assertEqual(self.saved(), [])

    async def test_cancel_during_tool_work_does_not_start_a_stale_continuation(self):
        await self.start()
        await output_item(self.session, "response-1", "intro", "commentary")
        entered, release = asyncio.Event(), asyncio.Event()

        async def execute(*args, **kwargs):
            entered.set()
            await release.wait()
            return {"status": "success"}

        self.session.tool_executor.execute = AsyncMock(side_effect=execute)
        task = asyncio.create_task(self.tool())
        try:
            await asyncio.wait_for(entered.wait(), 1)
            await self.session._handle_client_event({"type": "response.cancel"})
            release.set()
            await task
            await finish(self.session, "response-1", status="cancelled")
            self.assertEqual(len(self.fixture.creates()), 1)
            self.assertIsNone(self.session._tool_followup)
        finally:
            release.set()
            await asyncio.gather(task, return_exceptions=True)

    async def test_final_tool_failure_qualification_and_requested_detail_are_not_filtered(self):
        await self.start()
        self.session.tool_executor.execute = AsyncMock(return_value={
            "status": "failed", "error": {"code": "unavailable"},
        })
        await self.tool()
        await finish(self.session, "response-1")
        await self.fixture.acknowledge(response_id="followup")
        answer = "The operation failed. The reading is uncertain. Check it before acting. Here are the details you requested."
        await output_item(self.session, "followup", "answer", "final_answer")
        await speech(self.session, "followup", "answer", answer)
        await finish(self.session, "followup")
        self.assertEqual(self.transcripts(), [answer])
        self.assertEqual(self.saved()[0]["content"], answer)

    async def start_durable(self, kind):
        self.outcome = AsyncMock()
        self.session.reminder_repository = SimpleNamespace(
            mark_event_delivery=AsyncMock(), set_event_provider_response_id=AsyncMock(),
        )
        self.session.visual_repository = SimpleNamespace(mark_visual_interrupt_spoken=AsyncMock())
        if kind == "reminder":
            await self.session.speak_reminder(
                message="Take a break.", reminder_id=uuid4(), reminder_event_id=uuid4(),
            )
        elif kind == "prospective":
            await self.session.speak_prospective_interrupt(
                message="Your keys are here.", event_id=uuid4(), subject_type="visual_focus",
                subject_id=uuid4(), on_outcome=self.outcome,
            )
        else:
            self.assertTrue(await self.fixture.candidate())
        await self.fixture.acknowledge()

    async def assert_durable_output(self, kind, phase):
        await self.start_durable(kind)
        await output_item(self.session, "response-1", "speech", phase)
        await speech(self.session, "response-1", "speech", "Spoken result.")
        await finish(self.session, "response-1")
        delivered = phase == "final_answer"
        if kind == "reminder":
            event = self.session.reminder_repository.mark_event_delivery.await_args.kwargs
            self.assertEqual(event["event_type"], "delivered_to_session" if delivered else "failed")
            self.assertEqual(event["metadata"]["forwarded_answer"], delivered)
        elif kind == "prospective":
            self.outcome.assert_awaited_once_with("delivered" if delivered else "failed", "response-1")
        else:
            self.assertEqual(self.session.visual_repository.mark_visual_interrupt_spoken.await_count, int(delivered))
            terminal = [data for event, data in self.fixture.app_events if event == "visual.interrupt.done"][-1]
            self.assertEqual(terminal["forwarded_answer"], delivered)
        self.assertEqual(self.transcripts(), ["Spoken result."] if delivered else [])

    async def test_reminder_final_is_delivered(self):
        await self.assert_durable_output("reminder", "final_answer")

    async def test_reminder_commentary_is_not_delivered(self):
        await self.assert_durable_output("reminder", "commentary")

    async def test_prospective_final_is_delivered(self):
        await self.assert_durable_output("prospective", "final_answer")

    async def test_prospective_commentary_is_not_delivered(self):
        await self.assert_durable_output("prospective", "commentary")

    async def test_visual_interrupt_final_is_delivered(self):
        await self.assert_durable_output("visual_interrupt", "final_answer")

    async def test_visual_interrupt_commentary_is_not_delivered(self):
        await self.assert_durable_output("visual_interrupt", "commentary")

    async def test_reminder_tool_continuation_owns_delivery_until_final_answer(self):
        await self.start_durable("reminder")
        await output_item(self.session, "response-1", "intro", "commentary")
        await speech(self.session, "response-1", "intro", "hidden")
        self.session.tool_executor.execute = AsyncMock(return_value={"status": "success"})
        await self.tool()
        await finish(self.session, "response-1")
        events = self.session.reminder_repository.mark_event_delivery.await_args_list
        self.assertEqual([call.kwargs["event_type"] for call in events], ["dispatched_to_session"])
        self.assertEqual(len(self.fixture.creates()), 2)
        await self.fixture.acknowledge(response_id="followup")
        await output_item(self.session, "followup", "answer", "final_answer")
        await speech(self.session, "followup", "answer", "Final reminder.")
        await finish(self.session, "followup")
        self.assertEqual(self.session.reminder_repository.mark_event_delivery.await_args.kwargs["event_type"], "delivered_to_session")
        self.assertEqual(self.saved()[0]["metadata"]["response_kind"], "reminder")


class SharedViewOutputPhaseTests(unittest.IsolatedAsyncioTestCase):
    async def test_phase_gating_and_context_refresh_preserve_the_response_image_cutoff(self):
        fixture = visual_fixtures.SharedViewSessionTests()
        await fixture.asyncSetUp()
        try:
            first = await fixture.ready(1)
            await fixture.request()
            await fixture.acknowledged()
            recorded = fixture.wire.responses[0]
            snapshot = recorded["cutoff"]
            await output_item(fixture.session, "response-1", "intro", "commentary")
            await speech(fixture.session, "response-1", "intro", "hidden")
            await fixture.ready(2)
            await fixture.session._send_complete_session_update(SimpleNamespace(
                as_prompt_payload=lambda: {"working_memory": {"marker": "updated"}},
            ))
            self.assertEqual(snapshot.current_frame.frame_id, first.frame_id)
            self.assertEqual(fixture.state.active_snapshot.generation_id, snapshot.generation_id)
            await output_item(fixture.session, "response-1", "answer", "final_answer", index=1)
            await speech(fixture.session, "response-1", "answer", "Grounded answer.")
            await finish(fixture.session, "response-1")
            fixture.assert_healthy_cutoff(recorded)
            self.assertEqual(len(fixture.wire.responses), 1)
            transcripts = [data["text"] for kind, data in fixture.fixture.app_events if kind == "transcript.done"]
            self.assertEqual(transcripts, ["Grounded answer."])
        finally:
            await fixture.asyncTearDown()
