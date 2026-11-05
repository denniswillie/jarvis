"""Real session/adapter/writer composition; controlled storage and provider ACKs."""
import asyncio
from contextlib import asynccontextmanager
from datetime import timedelta
import json
import unittest
from unittest.mock import AsyncMock

from app.knowledge.models import SourceError
from app.guard_diagnostics import guard_wait_stage
from test_voice_context_owner_busy import OwnerBusyContextTests


class NormalTurnContextTests(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = OwnerBusyContextTests.asyncSetUp
    cleanup = OwnerBusyContextTests.cleanup
    until = OwnerBusyContextTests.until
    rows = OwnerBusyContextTests.rows
    observation = OwnerBusyContextTests.observation
    mutations = OwnerBusyContextTests.mutations
    creates = OwnerBusyContextTests.creates
    question_and_accept = OwnerBusyContextTests.question_and_accept
    expire_inside_adapter_guard = OwnerBusyContextTests.expire_inside_adapter_guard
    held = OwnerBusyContextTests.held

    async def busy_context(self, *, after_delete=False):
        self.busy, self.after_delete = True, after_delete
        await self.observation("optional-context-not-available")
        await self.held()
        self.runtime.wait_for_context_ready = AsyncMock(side_effect=AssertionError("Optional refresh must not block this turn"))

    async def ask(self):
        await self.provider.inject("input_audio_buffer.speech_started", item_id="question")
        await self.provider.inject("input_audio_buffer.speech_stopped", item_id="question")
        await self.provider.inject("input_audio_buffer.committed", item_id="question")
        await self.provider.inject("conversation.item.input_audio_transcription.completed",
                                   item_id="question", transcript="What can you tell me now?")

    async def test_unwritten_busy_allows_one_guarded_normal_answer_not_notifications(self):
        await self.busy_context()
        before = len(self.mutations())
        roots = set(self.session._source_context._sources)
        await self.question_and_accept()
        self.assertEqual(len(self.creates()), 1)
        self.assertFalse(self.runtime.ready)
        self.assertTrue(self.runtime.context_refresh_pending)
        self.assertEqual([row["type"] for row in self.mutations()[before:]], ["conversation.item.delete"])
        self.assertIsNone(self.runtime.adapter._context_item)
        self.assertTrue(roots.issubset(self.session._source_context._sources))
        self.assertIn("Current Live State briefing is temporarily unavailable", self.creates()[0]["response"]["instructions"])
        self.runtime.wait_for_context_ready.assert_not_awaited()

    async def test_already_acknowledged_empty_slot_needs_no_second_delete(self):
        await self.busy_context(after_delete=True)
        before = len(self.mutations())
        self.assertIsNone(self.runtime.adapter._context_item)
        await self.question_and_accept()
        self.assertEqual(len(self.mutations()), before)
        self.assertFalse(self.runtime.ready)

    async def test_expired_old_item_is_removed_before_response(self):
        await self.busy_context()
        self.now += timedelta(seconds=120)
        await self.question_and_accept()
        types = [r["type"] for r in self.provider.writes]
        self.assertLess(max(i for i,t in enumerate(types) if t == "conversation.item.delete"), types.index("response.create"))

    async def test_exact_delete_ack_is_required_and_late_unknown_ack_cannot_authorize(self):
        await self.busy_context()
        self.runtime.adapter._io_timeout = 0.1
        original = self.provider.send
        deleted = asyncio.Event()
        deletion = None

        async def no_delete_ack(raw):
            nonlocal deletion
            row = json.loads(raw)
            if row["type"] == "conversation.item.delete":
                deletion = row
                self.provider.writes.append(row)
                deleted.set()
                return
            await original(raw)

        self.provider.send = no_delete_ack
        await self.ask()
        await self.until(deleted.is_set)
        self.assertEqual(self.creates(), [])
        await self.until(lambda: self.runtime.failure is not None)
        await self.provider.inject("conversation.item.deleted", item_id=deletion["item_id"])
        self.assertEqual(self.creates(), [])
        self.assertFalse(self.runtime.ready)

    async def test_withdrawn_exposed_roots_still_reject_response_after_deletion(self):
        await self.busy_context()
        self.storage.rejected.update(self.session._source_context._sources)
        await self.ask()
        await self.until(lambda: self.session._context_revoked or self.session._closed.is_set())
        self.assertEqual(self.creates(), [])

    async def test_scope_authority_unavailable_does_not_delete_or_submit(self):
        await self.busy_context()
        before = len(self.mutations())

        @asynccontextmanager
        async def denied():
            raise SourceError("live_state_scope_changed", "Fixture scope no longer authorized")
            yield

        self.runtime._normal_turn_scope_guard = denied
        await self.ask()
        await self.until(lambda: any(e.get("code") == "response_preparation_failed" for e in self.app.events))
        self.assertEqual(len(self.mutations()), before)
        self.assertEqual(self.creates(), [])

    async def test_scope_timeout_records_nested_wait_without_submitting(self):
        await self.busy_context()
        before = len(self.mutations())

        @asynccontextmanager
        async def blocked():
            async with asyncio.timeout(0.02):
                with guard_wait_stage("memory_owner_lock"):
                    await asyncio.Event().wait()
            yield

        self.runtime._normal_turn_scope_guard = blocked
        with self.assertLogs("app.realtime.telemetry", level="INFO") as logs:
            await self.ask()
            await self.until(lambda: any(e.get("code") == "response_preparation_failed" for e in self.app.events))
            await self.until(lambda: self.session._response_preparation_task is None)
        events = [json.loads(row.split("realtime_trace ", 1)[1]) for row in logs.output if "realtime_trace " in row]
        self.assertTrue(any(e.get("event") == "provider_guard_wait_cancelled"
                            and e.get("phase") == "memory_owner_lock" for e in events))
        self.assertTrue(any(e.get("phase") == "normal_context_scope" and e.get("status") == "failed" for e in events))
        self.assertTrue(any(e.get("event") == "response_preparation_failed"
                            and e.get("error_type") == "TimeoutError" for e in events))
        self.assertEqual(len(self.mutations()), before)
        self.assertEqual(self.creates(), [])

    async def test_failure_diagnostic_does_not_log_exception_content(self):
        await self.busy_context()

        @asynccontextmanager
        async def denied():
            raise SourceError("live_state_scope_changed", "PRIVATE_EXCEPTION_CONTENT")
            yield

        self.runtime._normal_turn_scope_guard = denied
        with self.assertLogs("app.realtime.telemetry", level="INFO") as logs:
            await self.ask()
            await self.until(lambda: any(e.get("code") == "response_preparation_failed" for e in self.app.events))
            await self.until(lambda: self.session._response_preparation_task is None)
        self.assertNotIn("PRIVATE_EXCEPTION_CONTENT", "\n".join(logs.output))
        self.assertIn('"error_type":"SourceError"', "\n".join(logs.output))
        self.assertEqual(self.creates(), [])

    async def test_scope_is_rechecked_at_actual_writer_after_acknowledged_delete(self):
        await self.busy_context()
        original = self.runtime._normal_turn_scope_guard
        calls = 0

        @asynccontextmanager
        async def revoked_at_write():
            nonlocal calls
            calls += 1
            if calls == 2:
                raise SourceError("live_state_scope_changed", "Fixture scope revoked after deletion")
            async with original() as connection:
                yield connection

        self.runtime._normal_turn_scope_guard = revoked_at_write
        await self.ask()
        await self.until(lambda: self.session._context_revoked or self.session._closed.is_set())
        self.assertEqual(calls, 2)
        self.assertIsNone(self.runtime.adapter._context_item)
        self.assertEqual(self.creates(), [])

    async def test_replaced_request_is_rejected_at_writer(self):
        await self.busy_context()
        original = self.runtime._normal_turn_scope_guard
        calls = 0

        @asynccontextmanager
        async def supersede_at_write():
            nonlocal calls
            calls += 1
            async with original() as connection:
                if calls == 2:
                    self.session._turn_generation += 1
                yield connection

        self.runtime._normal_turn_scope_guard = supersede_at_write
        await self.ask()
        await self.until(lambda: calls >= 2 and self.session._pending_response is None)
        self.assertEqual(self.creates(), [])

    async def interrupt_preparations(self, cuts):
        await self.busy_context()
        original = self.runtime._normal_turn_scope_guard
        entered = asyncio.Queue()
        never = asyncio.Event()
        calls = cancelled = 0

        @asynccontextmanager
        async def hold_first_requests():
            nonlocal calls, cancelled
            calls += 1
            if calls <= cuts:
                entered.put_nowait(calls)
                try:
                    await never.wait()
                except asyncio.CancelledError:
                    cancelled += 1
                    raise
            async with original() as connection:
                yield connection

        async def utterance(index):
            identity = f"interruption-{index}"
            await self.provider.inject("input_audio_buffer.speech_started", item_id=identity)
            await self.provider.inject("input_audio_buffer.speech_stopped", item_id=identity)
            await self.provider.inject("input_audio_buffer.committed", item_id=identity)
            await self.provider.inject("conversation.item.input_audio_transcription.completed",
                                       item_id=identity, transcript=f"Use request {index} instead.")

        self.runtime._normal_turn_scope_guard = hold_first_requests
        before = len(self.mutations())
        for index in range(cuts):
            await utterance(index)
            self.assertEqual(await asyncio.wait_for(entered.get(), 2), index + 1)
        await utterance(cuts)
        await self.until(lambda: len(self.creates()) == 1)
        self.assertEqual(cancelled, cuts)
        self.assertGreaterEqual(calls, cuts + 2)
        self.assertEqual([row["type"] for row in self.mutations()[before:]], ["conversation.item.delete"])
        self.assertEqual(self.creates()[0]["response"]["metadata"]["jarvis_turn_generation"],
                         str(self.session._turn_generation))
        self.assertFalse(self.session._context_revoked)
        self.assertIsNone(self.runtime.failure)

    async def test_one_interruption_of_unwritten_preparation(self):
        await self.interrupt_preparations(1)

    async def test_two_interruptions_of_unwritten_preparation(self):
        await self.interrupt_preparations(2)

    async def test_three_interruptions_of_unwritten_preparation(self):
        await self.interrupt_preparations(3)


class SourceFreeNormalTurnTests(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = NormalTurnContextTests.asyncSetUp
    cleanup = NormalTurnContextTests.cleanup
    until = NormalTurnContextTests.until
    rows = NormalTurnContextTests.rows
    mutations = NormalTurnContextTests.mutations
    creates = NormalTurnContextTests.creates
    expire_inside_adapter_guard = NormalTurnContextTests.expire_inside_adapter_guard
    held = NormalTurnContextTests.held
    busy_context = NormalTurnContextTests.busy_context
    ask = NormalTurnContextTests.ask

    async def observation(self, summary, **options):
        if summary == "initial-as-of-view":
            return  # Start genuinely source-free, never erase exposed roots.
        return await NormalTurnContextTests.observation(self, summary, **options)

    async def test_source_free_normal_output_still_has_canonical_scope_guard(self):
        await self.busy_context()
        self.assertFalse(self.session._source_context.has_sources)
        original = self.runtime._normal_turn_scope_guard
        calls = 0

        @asynccontextmanager
        async def tracked():
            nonlocal calls
            calls += 1
            async with original() as connection:
                yield connection

        self.runtime._normal_turn_scope_guard = tracked
        await self.ask()
        await self.until(lambda: len(self.creates()) == 1)
        self.assertGreaterEqual(calls, 2)
        request = self.creates()[0]
        before_output = calls
        await self.provider.inject("response.created", response={"id": "answer",
            "metadata": request["response"]["metadata"]})
        await self.provider.inject("response.output_item.added", response_id="answer", output_index=0,
            item={"id": "answer-item", "type": "message", "role": "assistant", "phase": "final_answer"})
        await self.provider.inject("output_audio_buffer.started", response_id="answer")
        await self.until(lambda: any(e["type"] == "audio.webrtc.started" for e in self.app.events))
        self.assertGreater(calls, before_output)
        self.assertIsNone(self.session._active_responses["answer"].source_snapshot)
        self.assertFalse(self.runtime.ready)
