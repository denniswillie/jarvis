"""Repeated speech cuts during item lifecycle waits; entirely offline.

Real adapter and runtime event dispatch, with explicit fake writer/fence/owner
receipts. This qualifies neither provider interruption nor decoded/heard audio,
and does not assert cancellation of durable work. Test deadlines are bounds,
not responsiveness measurements.
"""

import asyncio
from types import SimpleNamespace
import unittest
from uuid import uuid4

from app.realtime.voice_route_runtime import RealtimeVoiceRouteRuntime
from app.voice_routes.contracts import InterruptSpeechCommand
from app.voice_routes.realtime_adapter import RealtimeRouteError, RealtimeVoiceRoute
from tests.test_realtime_voice_route import (
    FakeControl, NOW, make_context, make_identity, make_start, stop_route,
)


class RepeatedContextInterruptionTests(unittest.IsolatedAsyncioTestCase):
    async def _until(self, predicate):
        async with asyncio.timeout(0.5):
            while not predicate():
                await asyncio.sleep(0)

    async def _case(self, stage, cut_count):
        control = FakeControl()
        route = RealtimeVoiceRoute(control.bindings(), media_transport="webrtc",
            input_codec="opus", output_codec="opus", io_timeout=1.0,
            start_timeout=1.0, clock=lambda: NOW)
        # Unlike the convenience fixture, no automatic ACKs: every provider
        # lifecycle receipt below must traverse the actual runtime dispatcher.
        self.assertIsNone(control.item_event)
        active = {}
        received = []
        completed_responses = []

        async def owner_handler(event):
            received.append(event)
            if event["type"] == "response.created":
                active[event["response"]["id"]] = SimpleNamespace(interrupted=False, dismissed=False)
            elif event["type"] == "response.done":
                response_id = event["response"]["id"]
                self.assertIn(response_id, active)
                completed_responses.append(response_id)
                active.pop(response_id)

        runtime = SimpleNamespace(adapter=route, request=make_start(),
            session=SimpleNamespace(_active_responses=active),
            _owner_provider_handler=owner_handler, _response_offers={},
            _response_dispatch=lambda _: None)

        async def receive(event):
            # A blocked publication must not hold up this receiver wrapper.
            await asyncio.wait_for(RealtimeVoiceRouteRuntime._provider_event(runtime, event), 0.3)

        def writes(kind):
            return [event for event in control.writes if event["type"] == kind]

        publication = None
        try:
            identity = route.identity
            self.assertEqual((await route.start(runtime.request)).status, "completed")
            initial = writes("conversation.item.create")[0]
            original_id = initial["item"]["id"]
            create_ack = {"type": "conversation.item.done", "item": initial["item"]}
            delete_ack = {"type": "conversation.item.deleted", "item_id": original_id}

            if stage != "create_pending":
                await receive(create_ack)
            publication = asyncio.create_task(route.publish_context(make_context(revision=1)))
            await asyncio.sleep(0)
            if stage == "create_pending":
                self.assertEqual(route._items[original_id].state, "create_pending")
                self.assertFalse(writes("conversation.item.delete"))
            else:
                await self._until(lambda: len(writes("conversation.item.delete")) == 1)
                self.assertEqual(route._items[original_id].state, "delete_pending")

            if stage == "late_rejected_delete":
                deletion = writes("conversation.item.delete")[0]
                rejection = {"type": "error", "error": {
                    "code": "item_delete_invalid_item_id", "event_id": deletion["event_id"]}}
                await receive(rejection)
                receipt = await publication
                self.assertEqual(receipt.status, "unknown")
                self.assertIsNone(receipt.provider_receipt_ref)
                self.assertTrue(any(event is rejection for event in received))
                # A late create ACK and late delete ACK cannot undo a rejection
                # or silently authorize another delete. The error stays visible.
                await receive(create_ack)
                await receive(delete_ack)
                self.assertEqual(route._items[original_id].state, "delete_rejected")

            command_ids = []
            for cut in range(1, cut_count + 1):
                if stage != "late_rejected_delete":
                    self.assertFalse(publication.done())
                command = InterruptSpeechCommand(route=identity, command_id=uuid4(),
                    requested_at=NOW, reason="fixture_repeated_speech_cut")
                command_ids.append(command.command_id)
                receipt = await asyncio.wait_for(route.interrupt_speech(command), 0.3)
                self.assertEqual(receipt.route, identity)
                self.assertEqual(receipt.command_id, command.command_id)
                self.assertEqual(receipt.operation, "interrupt_speech")
                self.assertEqual(receipt.status, "completed")
                self.assertTrue(receipt.output_fenced)  # Explicit fake fence only.
                self.assertIsNone(receipt.provider_stopped)
                await self._until(lambda: len(writes("response.cancel")) == cut)

                # Exercise the common serialized writer and actual provider
                # event dispatcher while the background publication is waiting.
                # The response owner is deliberately a local fixture, not a
                # second implementation of production turn admission.
                response_id = f"fixture-{stage}-{cut}"
                self.assertTrue(await asyncio.wait_for(control.send_event({
                    "type": "response.create", "event_id": uuid4().hex}), 0.3))
                await receive({"type": "input_audio_buffer.speech_started", "item_id": f"input-{cut}"})
                await receive({"type": "response.created", "response": {"id": response_id}})
                await receive({"type": "response.output_item.added", "response_id": response_id,
                    "item": {"id": f"output-{cut}", "type": "message", "phase": "final_answer"}})
                await receive({"type": "response.done", "response": {
                    "id": response_id, "status": "completed"}})
                self.assertEqual(len(completed_responses), cut)

                # Speech cuts must not advance the source/route owner. A receipt
                # carrying another route generation cannot free the current item.
                with self.assertRaises(RealtimeRouteError):
                    route.observe_item_event(make_identity(2), create_ack)
                self.assertEqual(route.identity, identity)
                self.assertEqual(route._context_item, original_id)
                self.assertEqual(route._source_epoch, 0)
                self.assertFalse(route._retired)
                self.assertEqual(len(writes("conversation.item.create")), 1)

                if stage == "create_pending":
                    await receive({"type": "conversation.item.added", "item": initial["item"]})
                    self.assertEqual(route._items[original_id].state, "create_pending")
                    self.assertEqual(len(writes("conversation.item.delete")), 0)
                elif stage == "delete_pending":
                    await receive(create_ack)  # Repeated ACK during each cut.
                    await receive({"type": "conversation.item.deleted", "item_id": "foreign-item"})
                    self.assertEqual(route._items[original_id].state, "delete_pending")
                    self.assertEqual(len(writes("conversation.item.delete")), 1)
                else:
                    await receive(create_ack)
                    await receive(delete_ack)
                    retry_receipt = await route.publish_context(make_context(revision=cut + 1))
                    self.assertEqual(retry_receipt.status, "unknown")
                    self.assertIsNone(retry_receipt.provider_receipt_ref)
                    self.assertEqual(route._items[original_id].state, "delete_rejected")
                    self.assertEqual(len(writes("conversation.item.delete")), 1)

            if stage != "late_rejected_delete":
                self.assertFalse(publication.done())
                if stage == "create_pending":
                    await receive(create_ack)
                    await self._until(lambda: len(writes("conversation.item.delete")) == 1)
                await receive(delete_ack)
                receipt = await publication
                self.assertEqual(receipt.status, "written")
                self.assertIsNone(receipt.provider_receipt_ref)
                self.assertEqual(receipt.route, identity)
                self.assertEqual(receipt.cursor.revision, 1)
                self.assertEqual(len(writes("conversation.item.create")), 2)
                replacement = writes("conversation.item.create")[-1]
                self.assertNotEqual(replacement["item"]["id"], original_id)
                self.assertNotIn(original_id, route._items)
                # Duplicated old events after replacement cannot resurrect the
                # old item or change ownership of the replacement.
                await receive(create_ack)
                await receive(delete_ack)
                await receive({"type": "conversation.item.done", "item": replacement["item"]})
                self.assertEqual(route._context_item, replacement["item"]["id"])
                self.assertNotIn(original_id, route._items)

            self.assertEqual([event["item_id"] for event in writes("conversation.item.delete")], [original_id])
            self.assertEqual([command.command_id for command in control.fences[1:]], command_ids)
            self.assertEqual(len(writes("response.cancel")), cut_count)
            self.assertEqual(len(writes("response.create")), cut_count)
            self.assertEqual(len(completed_responses), cut_count)
            self.assertEqual(sum(event["type"] == "input_audio_buffer.speech_started" for event in received), cut_count)
            self.assertFalse(active)
        finally:
            if publication is not None:
                if not publication.done():
                    publication.cancel()
                await asyncio.gather(publication, return_exceptions=True)
            await stop_route(route)

    async def test_one_two_three_cuts_while_create_ack_pending(self):
        for cuts in (1, 2, 3):
            with self.subTest(cuts=cuts, stage="create_pending"):
                await self._case("create_pending", cuts)

    async def test_one_two_three_cuts_while_delete_ack_pending(self):
        for cuts in (1, 2, 3):
            with self.subTest(cuts=cuts, stage="delete_pending"):
                await self._case("delete_pending", cuts)

    async def test_one_two_three_cuts_after_late_error_and_ack(self):
        for cuts in (1, 2, 3):
            with self.subTest(cuts=cuts, stage="late_rejected_delete"):
                await self._case("late_rejected_delete", cuts)


if __name__ == "__main__":
    unittest.main()
