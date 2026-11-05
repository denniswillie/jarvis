"""Delivery-boundary regressions; no provider, database or phone connection."""
import asyncio
import unittest

import test_lean_voice_session as fixtures


class LeanDeliveryBoundaryTests(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = fixtures.LeanSessionTests.asyncSetUp
    asyncTearDown = fixtures.LeanSessionTests.asyncTearDown
    created = fixtures.LeanSessionTests.created
    tool = fixtures.LeanSessionTests.tool
    done = fixtures.LeanSessionTests.done
    drain = fixtures.LeanSessionTests.drain

    async def queued_cancel(self, *, block_write, advance):
        await self.created()
        entered, release = asyncio.Event(), asyncio.Event()
        original_send = self.provider.send_event
        count = 0

        async def queued_send(payload, **kwargs):
            nonlocal count
            count += 1
            if count == block_write:
                entered.set()
                await release.wait()
            return await original_send(payload, **kwargs)

        self.provider.send_event = queued_send
        await self.session._handle_client_event({"type": "response.cancel"})
        await asyncio.wait_for(entered.wait(), 0.5)
        await advance()
        release.set()
        await self.drain()

    async def test_queued_cancel_does_not_stop_new_response(self):
        await self.queued_cancel(block_write=1,
            advance=lambda: self.created("resp_new"))
        self.assertEqual(self.provider.sent, [])

    async def test_buffer_clear_is_rechecked_after_cancel_write(self):
        await self.queued_cancel(block_write=2,
            advance=lambda: self.created("resp_new"))
        self.assertEqual(self.provider.sent,
            [{"type": "response.cancel", "response_id": "resp_test"}])

    async def test_new_speech_generation_retires_queued_cancel(self):
        async def new_speech():
            await self.session._handle_openai_event({
                "type": "input_audio_buffer.speech_started", "item_id": "new_user"})
        await self.queued_cancel(block_write=1, advance=new_speech)
        self.assertEqual(self.provider.sent, [])

    async def test_no_active_response_does_not_send_unscoped_cancel(self):
        await self.session._handle_client_event({"type": "response.cancel"})
        await self.drain()
        self.assertEqual(self.provider.sent, [])

    async def transcript_item(self):
        await self.created()
        await self.session._handle_openai_event({"type": "response.output_item.added",
            "response_id": "resp_test", "output_index": 0,
            "item": {"id": "item_answer", "type": "message", "role": "assistant", "phase": "final_answer"}})

    async def test_transcript_work_is_bounded_while_phone_is_slow(self):
        await self.transcript_item()
        for _ in range(200):
            await self.session._handle_openai_event({"type": "response.output_audio_transcript.delta",
                "response_id": "resp_test", "item_id": "item_answer", "delta": "a"})
        self.assertLessEqual(len(self.session._lean_work), 128)

    async def test_final_transcript_is_persisted_after_slow_delivery(self):
        await self.transcript_item()
        entered, release = asyncio.Event(), asyncio.Event()
        original_send = self.socket.send_text
        async def slow_send(text):
            entered.set()
            await release.wait()
            await original_send(text)
        self.socket.send_text = slow_send
        self.session._lean_app_writer = self.session._spawn(self.session._write_app_events())
        await self.session._handle_openai_event({"type": "response.output_audio_transcript.done",
            "response_id": "resp_test", "item_id": "item_answer", "transcript": "The number is 42."})
        await asyncio.wait_for(entered.wait(), 0.5)
        await self.done()
        await asyncio.sleep(0)
        release.set()
        await self.drain()
        await asyncio.wait_for(self.session._lean_app_events.join(), 0.5)
        self.repository.create_message.assert_awaited_once()
        self.assertEqual(self.repository.create_message.call_args.kwargs["content"], "The number is 42.")

    async def test_unconfirmed_tool_output_does_not_trigger_followup(self):
        original_send = self.provider.send_event
        async def reject_output(payload, **kwargs):
            if callable(payload):
                payload = payload()
                if asyncio.iscoroutine(payload):
                    payload = await payload
            if payload is not None and payload.get("type") == "conversation.item.create":
                return False
            return await original_send(payload, **kwargs)
        self.provider.send_event = reject_output
        await self.created()
        await self.tool()
        await self.done()
        await self.drain()
        self.assertNotIn("response.create", [event["type"] for event in self.provider.sent])

    async def test_local_briefing_is_read_at_actual_provider_write_boundary(self):
        from test_live_briefing import BriefingTests
        fixture = BriefingTests()
        fixture.setUp()
        from app.live_state.models import SessionScope
        fixture.scope = SessionScope(owner_id=self.session.context.owner_id,
                                     session_id=self.session.context.jarvis_session_id)
        fixture.snapshot = fixture.snapshot.model_copy(update={"scope": fixture.scope,
            "entries": tuple(fixture.entry(index) for index in range(9))})
        self.session._lean_briefing.snapshot = lambda: fixture.snapshot
        self.session._lean_briefing.clock = lambda: fixture.now
        self.assertIn("Visible value", (await self.session._lean_briefing.read({}))["briefing"]["text"])
        entered, release = asyncio.Event(), asyncio.Event()
        original_send = self.provider.send_event
        async def queued_send(payload, **kwargs):
            entered.set()
            await release.wait()
            if callable(payload):
                payload = payload()
                if asyncio.iscoroutine(payload):
                    payload = await payload
            return await original_send(payload, **kwargs)
        self.provider.send_event = queued_send
        await self.created()
        await self.tool()
        await asyncio.wait_for(entered.wait(), 0.5)
        fixture.snapshot = None
        release.set()
        await self.drain()
        output = self.provider.sent[0]["item"]["output"]
        self.assertNotIn("Visible value", output)
        self.assertIn("live_state_unavailable", output)


if __name__ == "__main__":
    unittest.main()
