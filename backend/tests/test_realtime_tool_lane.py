"""Real session/event-handler/executor fixtures; controlled slow external work."""
import asyncio
from types import SimpleNamespace
import time
import unittest
from unittest.mock import AsyncMock

import test_realtime_image_tools as fixtures


class ToolLaneTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.fixture = fixtures.RealtimeImageToolTests()
        await self.fixture.asyncSetUp()
        self.session = self.fixture.session
        await self.fixture.start()
        # Activate only the lane gate; transport-specific runtime behavior has
        # its own tests. These are the real existing provider handlers/executor.
        self.session._voice_route_runtime = SimpleNamespace(
            before_session_close=AsyncMock(return_value=True), after_session_close=AsyncMock(return_value=True))
        self.entered, self.release = asyncio.Event(), asyncio.Event()
        self.order = []
        async def work(arguments):
            self.order.append("started")
            self.entered.set()
            await self.release.wait()
            self.order.append("finished")
            return {"status": "success", "value": "canonical operation completed"}
        self.fixture.ordinary.side_effect = work

    async def asyncTearDown(self):
        self.release.set()
        await self.fixture.asyncTearDown()

    async def call_blocked(self, identity="ordinary-1"):
        item = self.fixture.call(identity, name="ordinary_example", arguments={})
        await asyncio.wait_for(self.entered.wait(), 1)
        return item

    async def test_provider_completion_consumed_while_tool_is_blocked(self):
        await self.call_blocked()
        before = time.perf_counter()
        self.fixture.done()
        await self.fixture.until(lambda: self.session._active_responses["response-1"].completed_response is not None)
        elapsed = time.perf_counter() - before
        self.assertLess(elapsed, 0.25)  # Liveness bound, not a production p95 claim.
        self.assertEqual(self.order, ["started"])
        self.assertEqual(self.fixture.outputs(), [])
        self.release.set()
        await self.fixture.continuation()
        self.assertEqual(len(self.fixture.outputs()), 1)
        self.assertEqual(len(self.fixture.executions.finalized), 1)

    async def test_barge_in_preserves_committed_work_but_suppresses_stale_output(self):
        await self.call_blocked()
        await self.session._handle_client_event({"type": "response.cancel"})
        self.assertFalse(self.release.is_set())
        self.release.set()
        await self.fixture.until(lambda: not self.session._ordinary_tool_tasks)
        self.assertEqual(self.order, ["started", "finished"])
        self.assertEqual(len(self.fixture.executions.finalized), 1)
        self.assertEqual(self.fixture.outputs(), [])
        self.assertIsNone(self.session._tool_followup)

    async def test_same_chain_is_ordered_and_finishes_once(self):
        await self.call_blocked()
        self.fixture.call("ordinary-2", name="ordinary_example", arguments={})
        self.fixture.done()
        await self.fixture.until(lambda: len(self.session._ordinary_tool_tasks) == 2)
        self.assertEqual(self.fixture.ordinary.await_count, 1)
        self.release.set()
        await self.fixture.continuation()
        self.assertEqual([identity for identity, _ in self.fixture.outputs()], ["ordinary-1", "ordinary-2"])
        self.assertEqual(len(self.fixture.executions.finalized), 2)
        self.assertEqual(len(self.fixture.wire.responses), 2)

    async def test_cancelled_queued_work_never_enters_canonical_executor(self):
        await self.call_blocked()
        self.fixture.call("ordinary-2", name="ordinary_example", arguments={})
        await self.fixture.until(lambda: len(self.session._ordinary_tool_tasks) == 2)
        await self.session._handle_client_event({"type": "response.cancel"})
        self.release.set()
        await self.fixture.until(lambda: not self.session._ordinary_tool_tasks)
        self.assertEqual(self.fixture.ordinary.await_count, 1)
        self.assertNotIn("ordinary-2", self.fixture.executions.records)
        self.assertEqual(self.fixture.outputs(), [])

    async def test_duplicate_call_does_not_start_another_operation(self):
        await self.call_blocked()
        self.fixture.call("ordinary-1", name="ordinary_example", arguments={})
        self.fixture.done()
        self.release.set()
        await self.fixture.continuation()
        self.assertEqual(self.fixture.ordinary.await_count, 1)
        self.assertEqual(len(self.fixture.executions.finalized), 1)

    async def test_new_turn_tool_can_run_while_old_operation_finishes(self):
        await self.call_blocked()
        await self.session._handle_client_event({"type": "response.cancel"})
        await self.fixture.fixture.append()
        await self.session._handle_client_event({"type": "audio.input.commit"})
        await self.session._handle_client_event({"type": "response.create"})
        await self.fixture.until(lambda: "response-2" in self.session._active_responses)
        newer = asyncio.Event()
        async def new_work(arguments):
            newer.set()
            return {"status": "success", "value": "new turn"}
        self.fixture.ordinary.side_effect = new_work
        self.fixture.call("ordinary-new", name="ordinary_example", response_id="response-2", arguments={})
        await asyncio.wait_for(newer.wait(), 1)
        self.assertFalse(self.release.is_set())
        self.release.set()
        await self.fixture.until(lambda: not self.session._ordinary_tool_tasks)
        self.assertEqual([identity for identity, _ in self.fixture.outputs()], ["ordinary-new"])

    async def test_backpressure_closes_instead_of_building_an_unbounded_queue(self):
        await self.call_blocked()
        for index in range(2, 34):
            self.fixture.call(f"ordinary-{index}", name="ordinary_example", arguments={})
        await self.fixture.until(lambda: self.session._closed.is_set())
        self.assertLessEqual(len(self.session._ordinary_tool_tasks), 32)
        self.assertEqual(self.session._close_reason, "ordinary_tool_queue_exhausted")
        self.assertEqual(self.fixture.ordinary.await_count, 1)

    async def test_running_work_is_bounded_across_independent_chains(self):
        four_running, release = asyncio.Event(), asyncio.Event()
        active = peak = 0
        async def execute(**work):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            if active == 4:
                four_running.set()
            try:
                await release.wait()
            finally:
                active -= 1
        self.session._execute_function_call = execute
        source = self.session._active_responses["response-1"]
        tasks = [asyncio.create_task(self.session._run_ordinary_tool(
            chain_lock=asyncio.Lock(), source_state=source,
            generation=self.session._turn_generation)) for _ in range(5)]
        try:
            await asyncio.wait_for(four_running.wait(), 1)
            await asyncio.sleep(0)
            self.assertEqual(active, 4)
        finally:
            release.set()
            await asyncio.gather(*tasks)
        self.assertEqual(peak, 4)

    async def test_close_joins_owned_tool_tasks(self):
        await self.call_blocked()
        await self.session.close("test_shutdown")
        self.assertFalse(self.session._ordinary_tool_tasks)
        self.assertFalse(self.session._ordinary_tool_chains)
        self.assertFalse(self.session._ordinary_tool_locks)

    async def test_write_rechecks_generation_after_waiting_for_provider_writer(self):
        ready, write = asyncio.Event(), asyncio.Event()
        recorded = []
        async def send(event, **kwargs):
            ready.set()
            await write.wait()
            value = event() if callable(event) else event
            if value is not None:
                recorded.append(value)
                return True
            return False
        self.session.openai.send_event = send
        generation = self.session._turn_generation
        task = asyncio.create_task(self.session._write_tool_output("stale", {"status": "success"},
            is_current=lambda: self.session._turn_generation == generation))
        await ready.wait()
        self.session._turn_generation += 1
        write.set()
        self.assertFalse(await task)
        self.assertEqual(recorded, [])

    async def test_legacy_route_retains_its_existing_inline_behavior(self):
        self.session._voice_route_runtime = None
        await self.call_blocked()
        self.fixture.done()
        await asyncio.sleep(0.01)
        self.assertIsNone(self.session._active_responses["response-1"].completed_response)
        self.release.set()
        await self.fixture.continuation()
        self.assertEqual(len(self.fixture.executions.finalized), 1)
