from __future__ import annotations

import asyncio
import base64
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from app.tools.executor import JarvisToolExecutor
from app.tools.registry import JarvisToolRegistry, ToolDefinition
from evals.gemini_live.interruption import (
    Barrier, DatabaseFixtures, Lifecycle, Trace, instrument_registry,
)
from evals.gemini_live.interruption_runner import INTERRUPT, SAVE, STATUS, Provider


class InterruptionLifecycleTests(unittest.IsolatedAsyncioTestCase):
    def make(self, stage=None, *, failure=False):
        self.effects = []
        trace = Trace()
        barrier = Barrier(stage, trace)

        async def handler(arguments):
            if failure:
                return {"status": "failed", "error": {"code": "test_failure", "message": "Fixture failure"}}
            self.effects.append(arguments)
            return {"status": "success", "memory_id": "fixture-memory"}

        registry = JarvisToolRegistry()
        registry.register(ToolDefinition(name="remember", description="Fixture mutation",
                                         input_schema={"type": "object"}, handler=handler, mutates_state=True))
        registry = instrument_registry(registry, barrier, trace)
        self.send = AsyncMock(return_value=True)
        lifecycle = Lifecycle(registry, JarvisToolExecutor(registry), self.send, barrier, trace)
        return lifecycle, barrier, trace

    def call(self, call_id="one"):
        return {"id": call_id, "name": "remember", "args": {"content": "fixture"}}

    async def test_old_call_rejected_before_admission(self):
        lifecycle, barrier, trace = self.make("before_admission")
        task = asyncio.create_task(lifecycle.dispatch(self.call(), "old"))
        await asyncio.wait_for(barrier.reached.wait(), 1)
        lifecycle.interrupt()
        barrier.release.set()
        self.assertFalse(await task)
        self.assertEqual(self.effects, [])
        self.send.assert_not_awaited()
        self.assertIn("tool_call_not_admitted", [event["event"] for event in trace.events])

    async def test_admitted_write_finishes_after_interruption(self):
        lifecycle, barrier, _ = self.make("during_execution")
        task = asyncio.create_task(lifecycle.dispatch(self.call(), "old"))
        await asyncio.wait_for(barrier.reached.wait(), 1)
        self.assertEqual(self.effects, [])
        lifecycle.interrupt()
        barrier.release.set()
        self.assertFalse(await task)
        self.assertEqual(len(self.effects), 1)
        self.send.assert_awaited_once()
        self.assertIsNone(lifecycle.session._tool_followup)
        self.assertFalse(lifecycle.session._normal_turn_pending)

    async def test_committed_outcome_not_dropped_with_old_speech(self):
        lifecycle, barrier, _ = self.make("after_commit")
        task = asyncio.create_task(lifecycle.dispatch(self.call(), "old"))
        await asyncio.wait_for(barrier.reached.wait(), 1)
        self.assertEqual(len(self.effects), 1)
        self.send.assert_not_awaited()
        lifecycle.interrupt()
        barrier.release.set()
        await task
        output = self.send.await_args.args[0]
        self.assertEqual(output["item"]["call_id"], "one")
        self.assertIn("fixture-memory", output["item"]["output"])
        self.assertIsNone(lifecycle.session._tool_followup)

    async def test_same_id_not_reexecuted_in_session(self):
        lifecycle, _, _ = self.make()
        await lifecycle.dispatch(self.call(), "response")
        await lifecycle.dispatch(self.call(), "response")
        self.assertEqual(len(self.effects), 1)
        self.send.assert_awaited_once()

    async def test_new_ids_are_not_silently_semantically_deduplicated(self):
        lifecycle, _, _ = self.make()
        await lifecycle.dispatch(self.call("one"), "response")
        await lifecycle.dispatch(self.call("two"), "response")
        self.assertEqual(len(self.effects), 2)
        self.assertEqual(self.send.await_count, 2)

    async def test_new_turn_can_execute_after_old_turn_interrupted(self):
        lifecycle, _, _ = self.make()
        lifecycle.source("old")
        lifecycle.interrupt()
        self.assertFalse(await lifecycle.dispatch(self.call("old-call"), "old"))
        self.assertTrue(await lifecycle.dispatch(self.call("new-call"), "new"))
        self.assertEqual(len(self.effects), 1)

    async def test_completed_response_cannot_admit_new_mutation(self):
        lifecycle, _, _ = self.make()
        lifecycle.source("done").completed_response = {"status": "completed"}
        await lifecycle.dispatch(self.call(), "done")
        self.assertEqual(self.effects, [])
        self.send.assert_not_awaited()

    async def test_failure_outcome_also_survives_interruption(self):
        lifecycle, barrier, _ = self.make("after_commit", failure=True)
        task = asyncio.create_task(lifecycle.dispatch(self.call(), "old"))
        await asyncio.wait_for(barrier.reached.wait(), 1)
        lifecycle.interrupt()
        barrier.release.set()
        await task
        self.assertEqual(self.effects, [])
        self.assertIn("test_failure", self.send.await_args.args[0]["item"]["output"])

    async def test_transport_failure_is_visible_not_claimed_success(self):
        lifecycle, _, _ = self.make()
        self.send.side_effect = RuntimeError("transport unavailable")
        with self.assertRaisesRegex(RuntimeError, "transport unavailable"):
            await lifecycle.dispatch(self.call(), "response")
        self.assertEqual(len(self.effects), 1)

    async def test_barrier_only_holds_first_matching_boundary(self):
        trace = Trace()
        barrier = Barrier("after_commit", trace)
        await barrier.pause("during_execution")
        self.assertFalse(barrier.reached.is_set())
        task = asyncio.create_task(barrier.pause("after_commit"))
        await asyncio.wait_for(barrier.reached.wait(), 1)
        await asyncio.wait_for(barrier.pause("after_commit"), 1)
        barrier.release.set()
        await task


class GeminiTurnBoundaryTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        registry = JarvisToolRegistry()
        trace = Trace()
        self.provider = Provider("gemini", {}, registry, JarvisToolExecutor(registry),
                                 Barrier(None, trace), trace)
        self.provider.wire = SimpleNamespace(send=AsyncMock())

    def emit(self, text="", audio_parts=(), **flags):
        self.provider.receive_gemini({"serverContent": {
            "outputTranscription": {"text": text},
            "modelTurn": {"parts": [{"inlineData": {
                "mimeType": "audio/pcm;rate=24000",
                "data": base64.b64encode(data).decode("ascii"),
            }} for data in audio_parts]}, **flags,
        }})

    async def start_interrupted_speech(self):
        await self.provider.text(SAVE, "save")
        self.emit("I've remembered", [b"\x00\x00"])
        await self.provider.text(INTERRUPT, "interruption", interrupt=True)

    async def test_separate_interrupted_and_completion_do_not_settle_new_answer(self):
        await self.start_interrupted_speech()
        waiter = asyncio.create_task(self.provider.settle("interruption"))
        try:
            self.emit(" that your", [b"\x01\x00"])
            self.emit(interrupted=True)
            self.emit(turnComplete=True)
            await asyncio.sleep(0)
            self.assertFalse(waiter.done())
            self.assertFalse(self.provider.phase_result("save")["completed"])
            self.assertEqual(self.provider.phase_result("save")["transcript"], "I've remembered that your")
            self.assertEqual(self.provider.phase_result("interruption")["audio"], b"")
            self.assertEqual(self.provider.phase_result("interruption")["transcript"], "")
            self.emit("Four plus five is nine.", [b"\x02\x00"], turnComplete=True)
            await asyncio.wait_for(waiter, 1)
            await self.provider.text(STATUS, "status")
            self.emit("Yes, it completed.", [b"\x03\x00"], turnComplete=True)
            await self.provider.settle("status")
            self.assertEqual(self.provider.phase_result("interruption")["transcript"], "Four plus five is nine.")
            self.assertEqual(self.provider.phase_result("status")["transcript"], "Yes, it completed.")
        finally:
            waiter.cancel()
            await asyncio.gather(waiter, return_exceptions=True)

    async def test_combined_interruption_completion_keeps_all_trailing_audio_on_old_turn(self):
        await self.start_interrupted_speech()
        self.emit(" old tail", [b"\x01\x00", b"\x02\x00"], interrupted=True, turnComplete=True)
        self.assertEqual(self.provider.phase_result("save")["audio"], b"\x00\x00\x01\x00\x02\x00")
        self.assertFalse(self.provider.phase_result("save")["completed"])
        self.assertFalse(self.provider.phase_result("interruption")["completed"])
        self.assertEqual(self.provider.phase_result("interruption")["audio"], b"")
        self.emit("Nine.", [b"\x03\x00"], turnComplete=True)
        self.assertTrue(self.provider.phase_result("interruption")["completed"])

    async def test_natural_old_completion_race_does_not_complete_new_turn(self):
        await self.start_interrupted_speech()
        self.emit(" the name.", [b"\x01\x00"], turnComplete=True)
        self.assertTrue(self.provider.phase_result("save")["completed"])
        self.assertFalse(self.provider.phase_result("interruption")["completed"])
        self.emit("Nine.", [b"\x02\x00"], turnComplete=True)
        self.assertEqual(self.provider.phase_result("interruption")["transcript"], "Nine.")
        self.assertTrue(self.provider.phase_result("interruption")["completed"])

    async def test_old_audio_cannot_complete_a_later_empty_turn(self):
        await self.provider.text(SAVE, "save")
        self.emit("old speech", [b"\x01\x00"])
        self.emit(interrupted=True)
        self.emit(turnComplete=True)
        self.emit(turnComplete=True)
        self.assertFalse(self.provider.phase_result("save")["completed"])

    async def test_early_interruption_without_old_audio_still_accepts_new_answer(self):
        await self.provider.text(SAVE, "save")
        await self.provider.text(INTERRUPT, "interruption", interrupt=True)
        self.emit(interrupted=True)
        self.emit(turnComplete=True)
        self.assertFalse(self.provider.phase_result("interruption")["completed"])
        self.emit("Nine.", [b"\x01\x00"], turnComplete=True)
        self.assertTrue(self.provider.phase_result("interruption")["completed"])
        self.assertEqual(self.provider.phase_result("interruption")["transcript"], "Nine.")

    async def test_next_prompt_rejected_until_old_output_boundary_arrives(self):
        await self.start_interrupted_speech()
        with self.assertRaisesRegex(RuntimeError, "has not ended"):
            await self.provider.text(STATUS, "status")
        self.assertEqual(self.provider.wire.send.await_count, 2)

    async def test_receive_loop_routes_events_through_turn_tracking(self):
        await self.start_interrupted_speech()
        queue = asyncio.Queue()
        self.provider.wire.receive = queue.get
        self.provider.reader = asyncio.create_task(self.provider.receive())
        try:
            await queue.put({"serverContent": {"interrupted": True}})
            await queue.put({"serverContent": {"turnComplete": True}})
            await self.provider.until(lambda: any(
                event["event"] == "gemini_turn_handoff_completed"
                for event in self.provider.trace.events
            ), timeout=1)
            self.assertFalse(self.provider.phase_result("interruption")["completed"])
            await queue.put({"serverContent": {
                "outputTranscription": {"text": "Nine."},
                "modelTurn": {"parts": [{"inlineData": {
                    "mimeType": "audio/pcm;rate=24000", "data": "AAA=",
                }}]}, "turnComplete": True,
            }})
            await asyncio.wait_for(self.provider.settle("interruption"), 1)
        finally:
            self.provider.reader.cancel()
            await asyncio.gather(self.provider.reader, return_exceptions=True)


class FixtureSafetyTests(unittest.IsolatedAsyncioTestCase):
    async def test_nonfixture_snapshot_refused_before_connecting(self):
        with tempfile.TemporaryDirectory() as directory:
            fixtures = DatabaseFixtures("not-a-real-dsn", Path(directory))
            with patch("evals.gemini_live.interruption.asyncpg.connect", new_callable=AsyncMock) as connect:
                with self.assertRaisesRegex(ValueError, "non-fixture"):
                    await fixtures.snapshot("personal-owner", "session")
                connect.assert_not_awaited()

    async def test_cleanup_failure_is_recorded_and_raised(self):
        with tempfile.TemporaryDirectory() as directory:
            fixtures = DatabaseFixtures("not-a-real-dsn", Path(directory))
            fixtures.owners.append("direct-image-eval-interruption-test")
            with patch("evals.gemini_live.interruption.cleanup_fixture_owners", new_callable=AsyncMock) as cleanup:
                cleanup.side_effect = RuntimeError("fixture cleanup failed")
                with self.assertRaisesRegex(RuntimeError, "fixture cleanup failed"):
                    await fixtures.cleanup()
            manifest = (Path(directory) / "cleanup-manifest.json").read_text()
            self.assertIn("cleanup_failed", manifest)
            self.assertNotIn("not-a-real-dsn", manifest)

    async def test_cleanup_uses_only_registered_disposable_owners(self):
        with tempfile.TemporaryDirectory() as directory:
            fixtures = DatabaseFixtures("not-a-real-dsn", Path(directory))
            fixtures.owners.append("direct-image-eval-interruption-test")
            with patch("evals.gemini_live.interruption.cleanup_fixture_owners", new_callable=AsyncMock) as cleanup:
                cleanup.return_value = {"remaining_rows": 0}
                await fixtures.cleanup()
                cleanup.assert_awaited_once_with("not-a-real-dsn", set(fixtures.owners))
            self.assertEqual(fixtures.cleanup_result, {"remaining_rows": 0})


if __name__ == "__main__":
    unittest.main()
