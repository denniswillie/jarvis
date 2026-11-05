"""Remaining model budget is usable without erasing uncertain charges."""
import os
from pathlib import Path
import unittest
from uuid import uuid4

from app.runtime.feature_config import TaskLimits
from app.tasks.models import BudgetExceeded, TaskCreate, TaskError
from app.tasks.repository import TaskRepository
from app.tasks.runner import TaskRunner
from evals.second_brain.source_fixture import source_fixture


@unittest.skipUnless(os.getenv("JARVIS_SECOND_BRAIN_TEST_DB_URL") and
                     os.getenv("JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"),
                     "Explicit manifested database required")
class ModelOutputBudgetTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.fixture = await self.enterAsyncContext(source_fixture(
            os.environ["JARVIS_SECOND_BRAIN_TEST_DB_URL"],
            Path(os.environ["JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"]), test_name=self.id()))

    async def context(self, **limits):
        tasks = TaskRepository(self.fixture.pool, TaskLimits(**limits))
        task = await tasks.create(self.fixture.owner, TaskCreate(request_id=uuid4(),
            title="Synthetic budget", requested_outcome="No external actions."), model="fixture-model")
        claim = await tasks.claim_next("budget-fixture", ("frontier",))
        self.assertEqual(task.id, claim.task.id)
        return TaskRunner(tasks, {}).context(claim)

    async def test_unknown_16000_plus_settled_92_leaves_15908_not_a_false_pause(self):
        ctx = await self.context()
        old = await ctx.reserve("interrupted", input_tokens=3776, output_tokens=16000)
        await ctx.budgets.submit(ctx.claim, old["id"])
        # Seed the historical uncertainty, not a fake terminal usage receipt.
        async with self.fixture.pool.acquire() as connection:
            await connection.execute("update jarvis_task_reservations set status='unknown' where id=$1", old["id"])
        first = await ctx.reserve("read-context", input_tokens=8571, output_tokens=16000)
        await ctx.budgets.submit(ctx.claim, first["id"])
        await ctx.settle(first["id"], input_tokens=8571, output_tokens=92, tool_steps=0)
        final = await ctx.reserve("answer", input_tokens=9033, output_tokens=16000, output_is_ceiling=True)
        self.assertEqual(final["output_tokens"], 15908)
        self.assertEqual((await ctx.budgets.usage(ctx.task.owner_id, ctx.task.id))["output"], 32000)
        async with self.fixture.pool.acquire() as connection:
            row = await connection.fetchrow("select status,output_tokens,actual_output_tokens from jarvis_task_reservations where id=$1", old["id"])
        self.assertEqual(dict(row), {"status":"unknown", "output_tokens":16000, "actual_output_tokens":None})
        replay = await ctx.reserve("answer", input_tokens=9033, output_tokens=16000, output_is_ceiling=True)
        self.assertEqual(replay["id"], final["id"])
        self.assertEqual(replay["output_tokens"], 15908)
        with self.assertRaises(BudgetExceeded):
            await ctx.reserve("nothing-left", output_tokens=16000, output_is_ceiling=True)

    async def test_owner_daily_capacity_includes_this_request_input(self):
        ctx = await self.context(owner_daily_tokens=120)
        await ctx.reserve("earlier", input_tokens=20, output_tokens=30)
        row = await ctx.reserve("model", input_tokens=11, output_tokens=100, output_is_ceiling=True)
        self.assertEqual(row["output_tokens"], 59)
        with self.assertRaises(BudgetExceeded) as error:
            await ctx.reserve("daily-full", input_tokens=1, output_tokens=100, output_is_ceiling=True)
        self.assertEqual(error.exception.code, "task_owner_daily_budget_reached")

    async def test_exact_reservations_and_input_limits_are_not_weakened(self):
        ctx = await self.context(task_output_tokens=16000, task_input_tokens=10)
        await ctx.reserve("earlier", input_tokens=1, output_tokens=10)
        with self.assertRaises(BudgetExceeded):
            await ctx.reserve("exact", output_tokens=16000)
        with self.assertRaises(BudgetExceeded):
            await ctx.reserve("too-much-input", input_tokens=10, output_tokens=16000, output_is_ceiling=True)
        with self.assertRaises(BudgetExceeded) as error:
            await ctx.reserve("too-high-ceiling", output_tokens=16001, output_is_ceiling=True)
        self.assertEqual(error.exception.code, "task_request_output_budget_reached")

    async def test_ceiling_replay_cannot_change_input_or_raise_existing_allocation(self):
        ctx = await self.context()
        first = await ctx.reserve("same", input_tokens=10, output_tokens=100, output_is_ceiling=True)
        replay = await ctx.reserve("same", input_tokens=10, output_tokens=200, output_is_ceiling=True)
        self.assertEqual(replay["id"], first["id"])
        self.assertEqual(replay["output_tokens"], 100)
        for changes in ({"input_tokens":11, "output_tokens":100}, {"input_tokens":10, "output_tokens":99}):
            with self.subTest(changes=changes), self.assertRaises(TaskError):
                await ctx.reserve("same", **changes, output_is_ceiling=True)
