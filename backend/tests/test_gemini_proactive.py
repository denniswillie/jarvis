from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
import json
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock
from uuid import uuid4

from app.realtime.gemini.client import GeminiTransportError
from app.realtime.gemini.session import GeminiAudioSession
from test_gemini_audio_session import ENCODED, Phone, Wire, until
from test_gemini_domain_integration import context, settings


class GeminiProactiveTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.phone, self.wires = Phone(), []
        self.reminders = SimpleNamespace(
            mark_event_delivery=AsyncMock(return_value=SimpleNamespace()),
            set_event_provider_response_id=AsyncMock(),
        )

        def factory(**kwargs):
            wire = Wire(**kwargs)
            self.wires.append(wire)
            return wire

        self.session = GeminiAudioSession(
            websocket=self.phone, context=context(), settings=settings(),
            repository=SimpleNamespace(), reminder_repository=self.reminders,
            connection_factory=factory,
        )
        self.session._assistant_transcript_finalized = AsyncMock()
        self.session._input_transcript_finalized = AsyncMock()
        self.run_task = asyncio.create_task(self.session.run())
        await until(lambda: self.session._ready)

    async def asyncTearDown(self):
        await asyncio.wait_for(self.session.close("test_complete"), 2)
        await asyncio.wait_for(self.run_task, 2)

    def events(self, kind):
        return [e for e in self.phone.events if e["type"] == kind]

    def native_deliveries(self):
        return [e for w in self.wires for e in w.sent if e.get("clientContent", {}).get("turnComplete")]

    def reminder_states(self):
        return [call.kwargs["event_type"] for call in self.reminders.mark_event_delivery.await_args_list]

    async def reminder(self, **kwargs):
        await self.session.speak_reminder(message="Check the oven.", reminder_id=uuid4(),
                                         reminder_event_id=uuid4(), **kwargs)

    async def prospective(self, *, event_id=None, callback=None):
        callback = callback or AsyncMock()
        accepted = await self.session.speak_prospective_interrupt(
            message="Your appointment is soon.", event_id=event_id or uuid4(),
            subject_type="reminder", subject_id=uuid4(), on_outcome=callback,
        )
        return accepted, callback

    async def visual(self, **overrides):
        fields = dict(owner_id=self.session.context.owner_id,
                      jarvis_session_id=self.session.context.jarvis_session_id,
                      source_received_at=datetime.now(UTC), message="The door is open.", interrupt_id=uuid4())
        return await self.session.speak_visual_interrupt(**{**fields, **overrides})

    async def append(self, segment="user"):
        await self.phone.send("audio.input.append", segment_id=segment, audio_base64=ENCODED)
        await until(lambda: segment in self.session._turns and self.session._turns[segment].input_bytes)

    async def commit(self, segment="user"):
        await self.phone.send("audio.input.commit", segment_id=segment)
        await self.phone.send("response.create")
        await until(lambda: self.session._turns[segment].released)

    async def answer(self, *, terminal=True):
        await self.wires[-1].publish({
            "modelTurn": {"parts": [{"inlineData": {"mimeType": "audio/pcm;rate=24000", "data": ENCODED}}]},
            "outputTranscription": {"text": "Check the oven."},
            **({"generationComplete": True, "turnComplete": True} if terminal else {}),
        })

    async def test_visual_notification_is_synchronous_and_never_sends_inference(self):
        before = list(self.wires[0].sent)
        self.assertIsNone(self.session.notify_visual_frame_available())
        self.assertEqual(self.session.diagnostics()["visual_frame_notifications"], 1)
        self.assertEqual(self.wires[0].sent, before)

    async def test_reminder_submission_is_not_completion_and_same_event_is_not_repeated(self):
        fields = dict(message="Check the oven.", reminder_id=uuid4(), reminder_event_id=uuid4())
        await self.session.speak_reminder(**fields)
        self.assertEqual(self.reminder_states(), ["dispatched_to_session"])
        self.assertEqual(len(self.events("reminder.started")), 1)
        self.assertFalse(self.events("reminder.done"))
        self.assertIsNone(self.session._active_input)
        self.assertFalse(self.session._pending_transcripts)
        payload = json.loads(self.native_deliveries()[0]["clientContent"]["turns"][0]["parts"][0]["text"])
        self.assertEqual(payload["kind"], "jarvis_proactive_delivery")
        await self.answer()
        await until(lambda: bool(self.events("reminder.done")))
        await self.session.speak_reminder(**fields)
        self.assertEqual(len(self.native_deliveries()), 1)
        self.assertEqual(self.reminder_states(), ["dispatched_to_session", "delivered_to_session"])
        self.assertEqual(self.reminders.set_event_provider_response_id.await_args.kwargs["provider_response_id"],
                         self.events("reminder.done")[0]["response_id"])
        self.session._input_transcript_finalized.assert_not_awaited()

    async def test_refusal_of_reminder_raises_instead_of_returning_false_to_manager(self):
        await self.append()
        with self.assertRaises(GeminiTransportError):
            await self.reminder()
        self.reminders.mark_event_delivery.assert_not_awaited()
        self.assertFalse(self.native_deliveries())

    async def test_prospective_outcome_waits_for_completed_forwarded_audio(self):
        accepted, callback = await self.prospective()
        self.assertTrue(accepted)
        callback.assert_not_awaited()
        await self.answer(terminal=False)
        await until(lambda: bool(self.events("audio.output.delta")))
        callback.assert_not_awaited()
        await self.wires[-1].publish({"generationComplete": True, "turnComplete": True})
        await until(lambda: callback.await_count == 1)
        self.assertEqual(callback.await_args.args[0], "delivered")
        await self.session.close("test_complete")
        callback.assert_awaited_once()

    async def test_barge_in_settles_original_delivery_without_stealing_new_answer(self):
        accepted, callback = await self.prospective()
        self.assertTrue(accepted)
        await self.answer(terminal=False)
        await until(lambda: bool(self.events("audio.output.delta")))
        original_id = self.events("audio.output.delta")[0]["response_id"]
        await self.append()
        await until(lambda: callback.await_count == 1)
        self.assertEqual(callback.await_args.args, ("interrupted", original_id))
        await self.wires[-1].publish({"interrupted": True, "turnComplete": True})
        await self.commit()
        await self.answer()
        await until(lambda: self.session._turns["user"].output.completed)
        callback.assert_awaited_once()
        self.assertNotEqual(self.session._turns["user"].output.response_id, original_id)
        self.assertFalse(self.session._turns["user"].output.interrupted)

    async def test_user_arriving_during_durable_preparation_wins(self):
        entered, release = asyncio.Event(), asyncio.Event()

        async def mark(**kwargs):
            if kwargs["event_type"] == "dispatched_to_session":
                entered.set()
                await release.wait()
            return SimpleNamespace()

        self.reminders.mark_event_delivery.side_effect = mark
        task = asyncio.create_task(self.reminder())
        await entered.wait()
        self.assertFalse(self.session._checkpoint_safe())
        await self.append()
        release.set()
        with self.assertRaises(GeminiTransportError):
            await task
        await until(lambda: "failed" in self.reminder_states())
        self.assertFalse(self.native_deliveries())
        self.assertEqual(len(self.wires), 1)

    async def test_cancelled_request_before_write_does_not_later_speak(self):
        entered, release = asyncio.Event(), asyncio.Event()

        async def prepare(delivery):
            entered.set()
            await release.wait()

        self.session.proactive.prepare = prepare
        task = asyncio.create_task(self.prospective())
        await entered.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        release.set()
        await until(lambda: self.session._preparing_proactive is None)
        self.assertFalse(self.native_deliveries())

    async def test_proactive_tool_call_is_rejected_without_domain_admission(self):
        await self.prospective()
        await self.wires[-1].input.put({"toolCall": {"functionCalls": [
            {"id": "native-action", "name": "remember", "args": {"text": "Do not save this."}},
        ]}})
        await until(lambda: any("toolResponse" in e for e in self.wires[-1].sent))
        call = next(iter(self.session.operations.calls.values()))
        self.assertFalse(call.admitted)
        self.assertTrue(call.delivered)
        response = next(e["toolResponse"] for e in self.wires[-1].sent if "toolResponse" in e)
        self.assertEqual(response["functionResponses"][0]["response"]["error"]["code"], "proactive_speech_only")

    async def test_close_waits_for_preparation_then_records_failed_not_late_dispatched(self):
        entered, release = asyncio.Event(), asyncio.Event()

        async def mark(**kwargs):
            if kwargs["event_type"] == "dispatched_to_session":
                entered.set()
                await release.wait()
            return SimpleNamespace()

        self.reminders.mark_event_delivery.side_effect = mark
        request = asyncio.create_task(self.reminder())
        await entered.wait()
        close = asyncio.create_task(self.session.close("service_shutdown"))
        await until(lambda: self.session._closed.is_set())
        self.assertFalse(close.done())
        release.set()
        with self.assertRaises(GeminiTransportError):
            await request
        await close
        self.assertEqual(self.reminder_states(), ["dispatched_to_session", "failed"])
        self.assertEqual(self.session.proactive.diagnostics()["owned_tasks"], 0)

    async def test_failed_final_write_retries_same_identity_without_another_speech(self):
        calls = 0

        async def mark(**kwargs):
            nonlocal calls
            if kwargs["event_type"] == "delivered_to_session":
                calls += 1
                if calls == 1:
                    raise RuntimeError("Controlled acknowledgement loss")
            return SimpleNamespace()

        self.reminders.mark_event_delivery.side_effect = mark
        await self.reminder()
        await self.answer()
        await until(lambda: bool(self.events("reminder.done")))
        self.assertEqual(calls, 2)
        self.assertEqual(len(self.native_deliveries()), 1)
        self.assertEqual(len(self.events("reminder.done")), 1)
        self.assertFalse(self.session.proactive.diagnostics()["degraded"])

    async def test_visual_owner_freshness_and_quiet_policy_are_preserved(self):
        self.assertFalse(await self.visual(owner_id="foreign-owner"))
        self.assertFalse(await self.visual(source_received_at=datetime.now(UTC) + timedelta(seconds=30)))
        self.session._last_user_turn_settled_at = datetime.now(UTC)
        import time
        self.session._last_user_turn_settled_monotonic = time.monotonic()
        self.assertFalse(await self.visual())
        self.assertFalse(self.native_deliveries())

    async def test_durable_delivery_can_interrupt_started_assistant_but_generic_comment_cannot(self):
        await self.append()
        await self.commit()
        await self.answer(terminal=False)
        await until(lambda: self.session._turns["user"].provider_started)
        self.assertFalse(await self.visual())
        await self.reminder()
        self.assertTrue(self.session._turns["user"].output.interrupted)
        self.assertEqual(len(self.native_deliveries()), 1)
        await self.wires[-1].publish({"interrupted": True, "turnComplete": True})
        await self.answer()
        await until(lambda: bool(self.events("reminder.done")))

    async def test_terminal_without_audio_is_not_a_delivered_message(self):
        _, callback = await self.prospective()
        await self.wires[-1].publish({"outputTranscription": {"text": "Unvoiced text."},
                                      "generationComplete": True, "turnComplete": True})
        await until(lambda: callback.await_count == 1)
        self.assertEqual(callback.await_args.args[0], "failed")

    async def test_ambiguous_native_write_does_not_retry_or_claim_delivery(self):
        self.wires[0].send_event.side_effect = GeminiTransportError("gemini_write_failed")
        with self.assertRaises(GeminiTransportError):
            await self.reminder()
        await self.run_task
        self.assertEqual(self.reminder_states(), ["dispatched_to_session", "interrupted"])
        self.assertFalse(self.events("reminder.done"))
        self.assertEqual(len(self.wires), 1)

    async def test_conflicting_event_identity_is_refused_without_second_delivery(self):
        fields = dict(message="Check the oven.", reminder_id=uuid4(), reminder_event_id=uuid4())
        await self.session.speak_reminder(**fields)
        with self.assertRaises(GeminiTransportError):
            await self.session.speak_reminder(**{**fields, "message": "Different message."})
        self.assertEqual(len(self.native_deliveries()), 1)

    async def test_old_cancelled_tool_receipt_does_not_force_proactive_reconnect(self):
        call, _ = self.session.operations.receive(
            {"id": "old-cancelled-call", "name": "remember", "args": {}},
            namespace=self.session._native_namespace, turn_number=0,
        )
        self.session.operations.cancel(namespace=self.session._native_namespace, native_ids=[call.native_id])
        self.assertTrue((await self.prospective())[0])
        self.assertEqual(len(self.wires), 1)
        self.assertEqual(len(self.native_deliveries()), 1)

    async def test_unresolved_persistence_is_degraded_then_reconciled_on_close(self):
        healthy = False

        async def mark(**kwargs):
            if kwargs["event_type"] != "dispatched_to_session" and not healthy:
                raise RuntimeError("Controlled storage outage")
            return SimpleNamespace()

        self.reminders.mark_event_delivery.side_effect = mark
        await self.reminder()
        await self.answer()
        await until(lambda: self.session.proactive.diagnostics()["degraded"])
        self.assertFalse(self.events("reminder.done"))
        healthy = True
        await self.session.close("service_shutdown")
        self.assertFalse(self.session.proactive.diagnostics()["degraded"])
        self.assertEqual(self.reminder_states()[-1], "delivered_to_session")
        self.assertEqual(len(self.native_deliveries()), 1)

    async def test_prospective_refused_while_user_speaks_leaves_outcome_to_caller(self):
        await self.append()
        accepted, callback = await self.prospective()
        self.assertFalse(accepted)
        await self.session.close("test_complete")
        callback.assert_not_awaited()

    async def test_interrupt_chain_is_bounded_and_user_turn_resets_it(self):
        from app.assistant.proactive_policy import ProactiveInterruptGate
        self.session._proactive_gate = ProactiveInterruptGate(max_chain_length=1)
        self.assertTrue((await self.prospective())[0])
        await self.answer()
        await until(lambda: not self.session._pending_responses)
        self.assertFalse((await self.prospective())[0])
        await self.append()
        await self.commit()
        await self.answer()
        await until(lambda: self.session._turns["user"].output.completed)
        self.assertTrue((await self.prospective())[0])


if __name__ == "__main__":
    unittest.main()
