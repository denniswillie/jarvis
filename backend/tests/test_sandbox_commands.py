"""Managed commands through real SQL and the ordinary Hermes context/tool path."""
import asyncio
from dataclasses import replace
from datetime import UTC, datetime, timedelta
import os
from pathlib import Path
import unittest
from unittest.mock import AsyncMock
from uuid import uuid4

import asyncpg

from app.sandbox.commands import SandboxCommandRepository
from app.sandbox.models import CommandReceipt, CommandRequest, SandboxError, SandboxInfo
from app.tasks.actions import TaskActionRepository
from app.tasks.budgets import TaskBudgetRepository
from app.tasks.models import ClaimLost, TaskError, TaskMutation
from app.tasks.runner import WorkerContext
from app.tasks.worker import WorkerContextFactory
from evals.second_brain.sandbox_fixture import sandbox_task_fixture


class CommandProvider:
    def __init__(self):
        self.resources, self.receipts = {}, {}
        self.starts = self.creates = 0
        self.lose_create = self.lose_start = False
        self.complete = True

    async def create(self, scope, policy):
        self.creates += 1
        handle = SandboxInfo(sandbox_id="fixture-"+scope.allocation_id.hex, scope=scope, policy=policy,
            state="running", expires_at=datetime.now(UTC)+timedelta(minutes=5), template_id="fixture-base")
        self.resources[handle.sandbox_id] = handle
        if self.lose_create:
            raise SandboxError("sandbox_provider_outcome_uncertain")
        return handle

    async def discover(self, scope, policy):
        return [identity for identity, handle in self.resources.items() if handle.scope==scope and handle.policy==policy]

    async def info(self, identity, scope, policy):
        return self.resources[identity]

    async def prepare(self, handle):
        pass

    async def start_managed(self, handle, request, *, operation_id, request_digest):
        self.starts += 1
        self.receipts[operation_id] = CommandReceipt(format="jarvis.sandbox.command.v1", operation_id=operation_id,
            allocation_id=handle.scope.allocation_id, request_digest=request_digest, pid=4321,
            phase="completed" if self.complete else "running", started=True,
            exit_code=0 if self.complete else None, stdout="synthetic-result", stderr="",
            timed_out=False, output_limited=False, runner_error=None)
        if self.lose_start:
            raise SandboxError("sandbox_provider_outcome_uncertain")
        return 4321

    async def command_receipt(self, handle, identity, fingerprint):
        return self.receipts.get(identity)


@unittest.skipUnless(os.getenv("JARVIS_SECOND_BRAIN_TEST_DB_URL") and
    os.getenv("JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"), "Explicit manifested database fixtures required")
class SandboxCommandDatabaseTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.f = await self.enterAsyncContext(sandbox_task_fixture(os.environ["JARVIS_SECOND_BRAIN_TEST_DB_URL"],
            Path(os.environ["JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"]), test_name=self.id()))
        self.context = WorkerContext(self.f.claim, self.f.tasks, TaskBudgetRepository(self.f.tasks),
                                     TaskActionRepository(self.f.tasks))
        self.provider = CommandProvider()
        self.prepared = await WorkerContextFactory(self.f.source.pool, sandbox_provider=self.provider)(self.context)
        self.bridge = self.prepared.tools
        self.repository = self.bridge.broker.commands
        self.request = CommandRequest(operation_key="synthetic-command", command="printf synthetic-result")
        self.owner = self.f.source.owner

    async def invoke(self, name="sandbox_execute", arguments=None):
        return await self.bridge.invoke(name, arguments if arguments is not None else self.request.model_dump(),
                                        provider_call_id="fixture-"+uuid4().hex)

    async def command(self):
        async with self.f.source.pool.acquire() as connection:
            return dict(await connection.fetchrow("select * from jarvis_sandbox_commands where owner_id=$1", self.owner))

    async def test_default_worker_does_not_expose_sandbox_tools(self):
        plain = await WorkerContextFactory(self.f.source.pool)(self.context)
        self.assertNotIn("sandbox_execute", [tool["name"] for tool in plain.tools.catalog()])
        self.assertIn("read_task_context_source", [tool["name"] for tool in plain.tools.catalog()])

    async def test_real_worker_bridge_reuses_one_receipt_and_one_execution_budget(self):
        self.assertTrue(self.bridge.registry.get("sandbox_execute").mutates_state)
        first, second = await self.invoke(), await self.invoke()
        self.assertEqual(first, second)
        self.assertEqual(first["sandbox_command"]["stdout"], "synthetic-result")
        self.assertEqual(self.provider.starts, 1)
        self.assertEqual(self.provider.creates, 1)
        self.assertEqual((await self.context.budgets.usage(self.owner, self.f.task.id))["steps"], 1)
        lookup = await self.invoke("sandbox_status", {"operation_key": self.request.operation_key})
        self.assertEqual(lookup, first)
        listing = await self.invoke("sandbox_status", {"operation_key": None})
        self.assertEqual(listing["commands"][0]["operation_key"], self.request.operation_key)

    async def test_changed_parameters_cannot_reuse_logical_identity(self):
        await self.invoke()
        result = await self.invoke(arguments=self.request.model_copy(update={"command": "different"}).model_dump())
        self.assertEqual(result["error"]["code"], "sandbox_command_conflict")
        self.assertEqual(self.provider.starts, 1)

    async def test_lost_create_receipt_discovers_existing_vm_without_recreating(self):
        self.provider.lose_create = True
        self.assertEqual((await self.invoke())["status"], "failed")
        result = await self.invoke()
        self.assertEqual(result["sandbox_command"]["state"], "completed")
        self.assertEqual(self.provider.creates, 1)
        self.assertEqual(self.provider.starts, 1)

    async def test_empty_discovery_does_not_authorize_replacement(self):
        self.provider.lose_create = True
        await self.invoke()
        self.provider.resources.clear()
        result = await self.invoke()
        self.assertEqual(result["error"]["code"], "sandbox_creation_outcome_unresolved")
        self.assertEqual(self.provider.creates, 1)
        self.assertEqual(self.provider.starts, 0)

    async def test_lost_command_ack_recovers_terminal_output_without_reexecution(self):
        self.provider.lose_start = True
        self.assertEqual((await self.invoke())["status"], "failed")
        result = await self.invoke()
        self.assertEqual(result["sandbox_command"]["state"], "completed")
        self.assertEqual(self.provider.starts, 1)

    async def test_missing_receipt_remains_unknown_not_replayed(self):
        self.provider.lose_start = True
        await self.invoke()
        self.provider.receipts.clear()
        result = await self.invoke()
        self.assertEqual(result["sandbox_command"]["state"], "unknown")
        self.assertEqual(self.provider.starts, 1)

    async def test_running_receipt_then_completion_does_not_create_new_work(self):
        self.provider.complete = False
        self.assertEqual((await self.invoke())["sandbox_command"]["state"], "running")
        identity, receipt = next(iter(self.provider.receipts.items()))
        self.provider.receipts[identity] = receipt.model_copy(update={"phase": "completed", "exit_code": 0})
        self.assertEqual((await self.invoke())["sandbox_command"]["state"], "completed")
        self.assertEqual(self.provider.starts, 1)

    async def test_only_one_concurrent_command_crosses_submission_boundary(self):
        allocation = await self.bridge.broker.allocation()
        row = await self.repository.prepare(self.f.claim, allocation["id"], self.request)
        await self.bridge.broker._ready(allocation)
        tokens = await asyncio.gather(*(self.repository.begin_submission(self.f.claim, row["id"]) for _ in range(2)))
        self.assertEqual(sum(token is not None for token in tokens), 1)

    async def test_wrong_receipt_and_stale_claim_rejected(self):
        await self.invoke()
        row = await self.command()
        receipt = self.provider.receipts[row["id"]]
        for changed in ({"operation_id": uuid4()}, {"allocation_id": uuid4()}, {"pid": 9999}):
            with self.subTest(changed=changed), self.assertRaises(TaskError):
                await self.repository.record(self.owner, row["id"], row["submission_token"],
                                             receipt.model_copy(update=changed))
        with self.assertRaises(ClaimLost):
            await self.repository.current(replace(self.f.claim, fence=self.f.claim.fence+1), row["id"])

    async def test_source_removal_erases_parameters_and_output_and_blocks_future_commands(self):
        await self.invoke()
        async with self.f.source.pool.acquire() as connection:
            await connection.execute("""update jarvis_context_payloads set availability='deleted',content=null
                where owner_id=$1 and id=$2""", self.owner, self.f.task.description_payload_id)
            row = await connection.fetchrow("""select p.content as parameters,r.content as result
                from jarvis_sandbox_commands c join jarvis_context_payloads p on p.id=c.parameters_payload_id
                left join jarvis_context_payloads r on r.id=c.result_payload_id where c.owner_id=$1""", self.owner)
        self.assertIsNone(row["parameters"])
        self.assertIsNone(row["result"])
        with self.assertRaises(ClaimLost):
            await self.invoke()

    async def test_late_receipt_does_not_resurrect_removed_command(self):
        self.provider.lose_start = True
        await self.invoke()
        row = await self.command()
        async with self.f.source.pool.acquire() as connection:
            await connection.execute("""update jarvis_context_payloads set availability='deleted',content=null
                where owner_id=$1 and id=$2""", self.owner, self.f.task.description_payload_id)
        final = await self.repository.record(self.owner, row["id"], row["submission_token"], self.provider.receipts[row["id"]])
        self.assertEqual(final["phase"], "completed")
        self.assertIsNone(final["result_payload_id"])
        self.assertEqual((await self.context.budgets.usage(self.owner, self.f.task.id))["steps"], 1)

    async def test_replaying_terminal_receipt_repairs_lost_usage_settlement(self):
        self.provider.lose_start = True
        await self.invoke()
        row = await self.command()
        settle = self.repository.budgets.settle
        self.repository.budgets.settle = AsyncMock(side_effect=RuntimeError("synthetic settlement outage"))
        with self.assertRaises(RuntimeError):
            await self.repository.record(self.owner, row["id"], row["submission_token"], self.provider.receipts[row["id"]])
        self.repository.budgets.settle = settle
        await self.repository.record(self.owner, row["id"], row["submission_token"], self.provider.receipts[row["id"]])
        async with self.f.source.pool.acquire() as connection:
            self.assertEqual(await connection.fetchval("select status from jarvis_task_reservations where id=$1",
                                                      row["reservation_id"]), "settled")

    async def test_cancellation_blocks_more_execution_but_admits_late_receipt(self):
        self.provider.lose_start = True
        await self.invoke()
        row = await self.command()
        await self.f.tasks.mutate(self.owner, self.f.task.id, "cancel",
                                 TaskMutation(request_id=uuid4(), expected_revision=1))
        final = await self.repository.record(self.owner, row["id"], row["submission_token"], self.provider.receipts[row["id"]])
        self.assertEqual(final["phase"], "completed")
        with self.assertRaises(ClaimLost):
            await self.invoke()
        self.assertEqual(self.provider.starts, 1)

    async def test_sql_rejects_command_identity_and_final_receipt_replacement(self):
        await self.invoke()
        row = await self.command()
        async with self.f.source.pool.acquire() as connection:
            for change in ("pid=9999", "submission_token=gen_random_uuid()", "phase='running',result_digest=null"):
                with self.subTest(change=change), self.assertRaises(asyncpg.CheckViolationError):
                    await connection.execute("update jarvis_sandbox_commands set "+change+" where id=$1", row["id"])

    async def test_path_escape_is_rejected_before_native_io(self):
        result = await self.invoke(arguments=self.request.model_copy(update={"cwd": "../other"}).model_dump())
        self.assertEqual(result["error"]["code"], "invalid_tool_input")
        self.assertEqual(self.provider.creates, 0)

    async def test_escaped_large_output_is_explicitly_bounded_before_sql_and_model_transport(self):
        self.provider.lose_start = True
        await self.invoke()
        row = await self.command()
        receipt = self.provider.receipts[row["id"]].model_copy(update={"stdout": chr(1)*60000})
        self.provider.receipts[row["id"]] = receipt
        result = await self.invoke()
        self.assertEqual(result["sandbox_command"]["state"], "completed")
        self.assertFalse(result["sandbox_command"]["output_complete"])
        self.assertEqual(len(result["sandbox_command"]["stdout"]), 4000)
        async with self.f.source.pool.acquire() as connection:
            stored = await connection.fetchval("""select octet_length(p.content::text)
                from jarvis_sandbox_commands c join jarvis_context_payloads p on p.id=c.result_payload_id
                where c.id=$1""", row["id"])
        self.assertLess(stored, 262144)

    async def test_revoked_grant_blocks_native_creation(self):
        async with self.f.source.pool.acquire() as connection:
            await connection.execute("update jarvis_operation_grants set status='revoked' where id=$1", self.f.grant)
        with self.assertRaises(ClaimLost):
            await self.invoke()
        self.assertEqual(self.provider.creates, 0)

    async def test_binary_and_surrogate_output_remain_bounded_and_explicitly_incomplete(self):
        self.provider.lose_start = True
        await self.invoke()
        row = await self.command()
        self.provider.receipts[row["id"]] = self.provider.receipts[row["id"]].model_copy(
            update={"stdout": (chr(0)+chr(0x1f600))*12000})
        result = await self.invoke()
        self.assertFalse(result["sandbox_command"]["output_complete"])
        self.assertNotIn(chr(0), result["sandbox_command"]["stdout"])
        self.assertEqual(result["sandbox_command"]["state"], "completed")

    async def test_exhausted_tool_budget_blocks_native_creation(self):
        await self.context.reserve("synthetic-exhaustion", tool_steps=self.f.task.budget.tool_steps)
        result = await self.invoke()
        self.assertEqual(result["status"], "failed")
        self.assertEqual(self.provider.creates, 0)
