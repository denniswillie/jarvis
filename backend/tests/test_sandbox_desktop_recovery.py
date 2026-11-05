"""The shared command lifecycle under a desktop grant, using isolated real SQL."""
import asyncio
import os
from pathlib import Path
from unittest.mock import AsyncMock
from uuid import uuid4

import test_sandbox_commands as command_tests

from app.runtime.feature_config import SandboxConfiguration
from app.sandbox.models import CommandRequest, SandboxError, SandboxPolicy
from app.tasks.actions import TaskActionRepository
from app.tasks.budgets import TaskBudgetRepository
from app.tasks.catalog import ConfiguredTaskProvider
from app.tasks.models import TaskCreate, TaskError
from app.tasks.runner import WorkerContext
from app.tasks.service import TaskService
from app.tasks.worker import WorkerContextFactory
from evals.second_brain.sandbox_fixture import sandbox_task_fixture


class DesktopCommandDatabaseTests(command_tests.SandboxCommandDatabaseTests):
    """Reuse every terminal lifecycle assertion, without pretending to run a GUI."""

    async def asyncSetUp(self):
        self.policy = SandboxPolicy(template="desktop")
        self.f = await self.enterAsyncContext(sandbox_task_fixture(
            os.environ["JARVIS_SECOND_BRAIN_TEST_DB_URL"],
            Path(os.environ["JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"]),
            test_name=self.id(), policy=self.policy))
        self.context = WorkerContext(self.f.claim, self.f.tasks, TaskBudgetRepository(self.f.tasks),
                                     TaskActionRepository(self.f.tasks))
        self.provider = command_tests.CommandProvider()
        self.factory = WorkerContextFactory(self.f.source.pool, sandbox_provider=self.provider,
                                            sandbox_policy=self.policy)
        self.bridge = (await self.factory(self.context)).tools
        self.repository = self.bridge.broker.commands
        self.request = CommandRequest(operation_key="synthetic-command", command="xdotool type synthetic-result")
        self.owner = self.f.source.owner

    async def test_parallel_tasks_have_separate_desktops_and_cannot_adopt_each_other(self):
        service = TaskService(self.f.tasks, {"frontier": ConfiguredTaskProvider("fixture-no-model")},
                             sandbox_configuration=SandboxConfiguration(policy=self.policy))
        second = await service.create(self.owner, TaskCreate(request_id=uuid4(), title="Other synthetic UI",
            requested_outcome="Inspect a separate synthetic desktop.", execution_environment="desktop_sandbox"))
        claim = await self.f.tasks.claim_next("second-desktop-worker", ("frontier",), owner_id=self.owner,
                                             worker_capabilities=self.policy.required_capabilities)
        self.assertEqual(claim.task.id, second.id)
        context = WorkerContext(claim, self.f.tasks, TaskBudgetRepository(self.f.tasks),
                                TaskActionRepository(self.f.tasks))
        bridge = (await self.factory(context)).tools
        first_result, second_result = await asyncio.gather(self.invoke(), bridge.invoke(
            "sandbox_execute", self.request.model_dump(), provider_call_id="second-desktop-command"))
        self.assertEqual(first_result["sandbox_command"]["state"], "completed")
        self.assertEqual(second_result["sandbox_command"]["state"], "completed")
        self.assertEqual((self.provider.creates, self.provider.starts), (2, 2))
        handles = list(self.provider.resources.values())
        self.assertEqual({item.scope.task_id for item in handles}, {self.f.task.id, second.id})
        self.assertEqual(len({item.sandbox_id for item in handles}), 2)
        self.assertEqual(len({item.scope.attempt_id for item in handles}), 2)
        foreign = next(item for item in handles if item.scope.task_id == second.id)
        with self.assertRaises(TaskError) as caught:
            await self.f.allocations.current(self.f.claim, foreign.scope.allocation_id)
        self.assertEqual(caught.exception.code, "sandbox_allocation_not_found")

    async def test_active_desktop_cannot_be_claimed_by_a_second_controller(self):
        await self.invoke()
        self.assertIsNone(await self.f.tasks.claim_next("competing-desktop-worker", ("frontier",),
            owner_id=self.owner, worker_capabilities=self.policy.required_capabilities))
        self.assertEqual(self.provider.creates, 1)

    async def test_unavailable_desktop_does_not_report_executed_input(self):
        self.provider.prepare = AsyncMock(side_effect=SandboxError("sandbox_desktop_unavailable"))
        result = await self.invoke()
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["error"]["code"], "sandbox_desktop_unavailable")
        self.assertEqual(self.provider.starts, 0)
        self.assertNotIn("sandbox_command", result)

