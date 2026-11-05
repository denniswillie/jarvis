"""Foreground ownership through real session/bridge/writer, controlled I/O only."""
import asyncio
import json
import unittest
from datetime import timedelta
from uuid import uuid4

from app.voice_routes.contracts import InterruptSpeechCommand
from app.voice_routes.coordinator import CoordinationPhase
import test_normal_turn_context_readiness as fixtures


class ForegroundContextHandoffTests(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = fixtures.NormalTurnContextTests.asyncSetUp
    cleanup = fixtures.NormalTurnContextTests.cleanup
    until = fixtures.NormalTurnContextTests.until
    rows = fixtures.NormalTurnContextTests.rows
    observation = fixtures.NormalTurnContextTests.observation
    mutations = fixtures.NormalTurnContextTests.mutations
    creates = fixtures.NormalTurnContextTests.creates
    question_and_accept = fixtures.NormalTurnContextTests.question_and_accept
    expire_inside_adapter_guard = fixtures.NormalTurnContextTests.expire_inside_adapter_guard
    held = fixtures.NormalTurnContextTests.held
    busy_context = fixtures.NormalTurnContextTests.busy_context
    ask = fixtures.NormalTurnContextTests.ask

    async def finish_answer(self):
        await self.provider.inject("response.output_audio_transcript.done", response_id="answer",
            item_id="answer-item", transcript="Controlled answer without current briefing.")
        await self.provider.inject("response.done", response={"id": "answer", "status": "completed",
            "output": [{"id": "answer-item", "type": "message", "phase": "final_answer",
                        "content": [{"type": "output_audio"}]}]})
        await self.until(lambda: any(e["type"] == "response.done" for e in self.app.events))
        await self.provider.inject("output_audio_buffer.stopped", response_id="answer")
        await self.until(lambda: not self.runtime._answer_owns_context())

    async def client_cuts(self, cuts, *, explicit=False):
        await self.busy_context()
        await self.question_and_accept()
        self.assertFalse(self.runtime.ready)
        generation = self.session._turn_generation
        for index in range(cuts):
            if explicit:
                command = InterruptSpeechCommand(route=self.runtime.request.route, command_id=uuid4(),
                    requested_at=self.now, reason="fixture_barge_in")
                self.runtime._queue_interrupt(command)
                self.runtime._queue_interrupt(command)  # exact duplicate, not a second cancel
            else:
                await self.session._handle_client_event({"type": "response.cancel"})
            await self.until(lambda: self.session._turn_generation == generation + index + 1)
        await self.until(lambda: any(e["type"] == "voice_route.event" and
            e.get("payload", {}).get("operation") == "interrupt_speech" for e in self.app.events))
        self.assertIsNone(self.runtime.failure)
        self.assertEqual(self.runtime.coordinator.status.phase, CoordinationPhase.ACTIVE)
        self.assertEqual(len(self.creates()), 1)
        self.assertFalse(self.session._context_revoked)
        self.assertTrue(any(e["type"] == "response.cancel" for e in self.provider.writes))

    async def test_one_client_cancel_with_empty_briefing(self):
        await self.client_cuts(1)

    async def test_two_client_cancels_with_empty_briefing(self):
        await self.client_cuts(2)

    async def test_three_client_cancels_with_empty_briefing(self):
        await self.client_cuts(3)

    async def test_explicit_interrupt_identity_is_deduplicated_without_global_ready(self):
        await self.client_cuts(3, explicit=True)

    async def test_foreign_route_cannot_interrupt_current_output(self):
        await self.busy_context()
        await self.question_and_accept()
        command = InterruptSpeechCommand(route=self.runtime.request.route.model_copy(update={"route_id": uuid4()}),
            command_id=uuid4(), requested_at=self.now, reason="foreign_fixture")
        generation = self.session._turn_generation
        self.runtime._queue_interrupt(command)
        self.assertEqual(self.session._turn_generation, generation)
        self.assertEqual(self.runtime.failure, "interrupt_not_admitted")

    async def test_normal_turn_joins_cancelled_optional_preflight_before_clear(self):
        self.now += timedelta(seconds=2)
        entered, cancelled = asyncio.Event(), asyncio.Event()
        original = self.bridge._validity

        async def blocked(update):
            entered.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.set()
                raise
            return await original(update)

        self.bridge._validity = blocked
        await self.observation("pending-preflight")
        await self.until(entered.is_set)
        before = len(self.mutations())
        await self.question_and_accept()
        self.assertTrue(cancelled.is_set())
        self.assertFalse(self.bridge._jobs)
        self.assertIsNone(self.runtime.failure)
        self.assertFalse(self.bridge.status.fenced)
        self.assertEqual([e["type"] for e in self.mutations()[before:]], ["conversation.item.delete"])
        self.bridge._validity = original

    async def test_joined_successful_replacement_is_not_deleted_again(self):
        entered, release = asyncio.Event(), asyncio.Event()
        original = self.provider.send
        before = len(self.mutations())

        async def hold_created(raw):
            result = await original(raw)
            if json.loads(raw)["type"] == "conversation.item.create":
                entered.set()
                await release.wait()
            return result

        self.provider.send = hold_created
        try:
            await self.observation("fresh-while-request-arrives")
            await self.until(entered.is_set)
            self.assertTrue(self.runtime._context_replacement_active)
            ask = asyncio.create_task(self.question_and_accept())
            await self.until(lambda: self.session._pending_response is not None)
            release.set()
            await asyncio.wait_for(ask, 3)
            self.assertEqual([e["type"] for e in self.mutations()[before:]],
                             ["conversation.item.delete", "conversation.item.create"])
            self.assertTrue(self.runtime.ready)
            self.assertIn("fresh-while-request-arrives", self.runtime.adapter._context.text)
            self.assertNotIn("Current Live State briefing is temporarily unavailable",
                             self.creates()[0]["response"]["instructions"])
        finally:
            release.set()
            self.provider.send = original
            if "ask" in locals():
                ask.cancel()
                await asyncio.gather(ask, return_exceptions=True)

    async def test_unchanged_desired_context_refills_only_after_foreground_finishes(self):
        await self.busy_context()
        desired = self.attachment._last_update
        cursor = self.service.snapshot().cursor
        await self.question_and_accept()
        before = len(self.mutations())
        self.busy = False
        await self.attachment.publish_latest()
        self.assertIsNone(self.runtime.adapter._context_item)
        self.assertFalse(self.runtime.ready)
        await self.finish_answer()
        await self.until(lambda: self.runtime.ready and not self.bridge._jobs)
        self.assertEqual(self.service.snapshot().cursor, cursor)
        self.assertEqual(self.runtime.adapter._context, desired)
        self.assertEqual(self.runtime.adapter._context.expires_at, desired.expires_at)
        self.assertEqual([e["type"] for e in self.mutations()[before:]], ["conversation.item.create"])

    async def test_newer_snapshot_consumes_foreground_refill_permission(self):
        await self.busy_context()
        await self.question_and_accept()
        self.busy = False
        await self.observation("new-evidence-after-foreground-clear")
        self.assertIsNone(self.runtime.adapter._context_item)
        await self.finish_answer()
        await self.until(lambda: self.runtime.ready and not self.bridge._jobs and
            "new-evidence-after-foreground-clear" in self.runtime.adapter._context.text)
        self.assertFalse(self.bridge._normal_turn_refill)
        self.assertIsNone(self.bridge.status.refill_update_id)
