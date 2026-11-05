import asyncio
import os
from pathlib import Path
import unittest
from uuid import uuid4

import asyncpg

from app.runtime.feature_config import SandboxConfiguration, TaskLimits
from app.sandbox.allocations import SandboxAllocationRepository
from app.sandbox.models import SandboxPolicy
from app.tasks.catalog import ConfiguredTaskProvider, TaskWorkerRegistry
from app.tasks.models import TaskCreate, TaskError, TaskMutation, TaskStatus, WorkerOutcome, json_object
from app.tasks.repository import TaskRepository
from app.tasks.service import TaskService
from evals.second_brain.source_fixture import source_fixture


@unittest.skipUnless(os.getenv("JARVIS_SECOND_BRAIN_TEST_DB_URL") and
    os.getenv("JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"), "Manifested database fixtures required")
class SandboxWorkerAdmissionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.f = await self.enterAsyncContext(source_fixture(os.environ["JARVIS_SECOND_BRAIN_TEST_DB_URL"],
            Path(os.environ["JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"]), test_name=self.id()))
        self.tasks = TaskRepository(self.f.pool, TaskLimits(lease_seconds=60))
        self.registry = TaskWorkerRegistry(self.f.pool)
        self.providers = {"frontier": ConfiguredTaskProvider("fixture")}
        self.configuration = SandboxConfiguration()
        self.service = TaskService(self.tasks, self.providers, registry=self.registry,
                                   sandbox_configuration=self.configuration)
        self.request = TaskCreate(request_id=uuid4(), title="Synthetic compute", requested_outcome="Compute offline.",
                                  execution_environment="offline_sandbox")

    async def advertise(self, capabilities=("sandbox.execute",)):
        await self.registry.advertise(worker_id="fixture", owner_id=self.f.owner, host_id="fixture",
                                      providers=self.providers, ttl_seconds=60, capabilities=capabilities)

    async def create(self):
        await self.advertise()
        return await self.service.create(self.f.owner, self.request)

    async def grant(self, task):
        async with self.f.pool.acquire() as connection:
            return await connection.fetchval("select grant_id from jarvis_task_grants where owner_id=$1 and task_id=$2",
                                             self.f.owner, task.id)

    async def claim(self, capabilities=("sandbox.execute",)):
        return await self.tasks.claim_next("fixture", ("frontier",), owner_id=self.f.owner,
                                          worker_capabilities=capabilities)

    async def test_plain_readiness_cannot_admit_sandbox_task(self):
        await self.advertise(())
        with self.assertRaises(TaskError) as caught:
            await self.service.create(self.f.owner, self.request)
        self.assertEqual(caught.exception.code, "task_worker_not_ready")
        async with self.f.pool.acquire() as connection:
            self.assertEqual(await connection.fetchval("select count(*) from jarvis_background_tasks"), 0)
            self.assertEqual(await connection.fetchval("select count(*) from jarvis_operation_grants"), 0)

    async def test_atomic_grant_and_replay_survive_worker_and_configuration_loss(self):
        task = await self.create()
        await self.registry.withdraw(self.f.owner, "fixture")
        self.service.sandbox_configuration = None
        replay = await self.service.create(self.f.owner, self.request)
        self.assertEqual(task.id, replay.id)
        async with self.f.pool.acquire() as connection:
            self.assertEqual(await connection.fetchval("select count(*) from jarvis_operation_grants"), 1)
            row = await connection.fetchrow("select * from jarvis_operation_grants")
            self.assertEqual(row["request_event_id"], task.request_event_id)
            self.assertTrue(await connection.fetchval("""select exists(select 1 from jarvis_context_payload_dependencies
                where owner_id=$1 and dependent_payload_id=$2)""", self.f.owner, row["scope_payload_id"]))

    async def test_plain_worker_skips_sandbox_and_can_claim_ordinary_work(self):
        sandbox = await self.create()
        plain = await self.service.create(self.f.owner, self.request.model_copy(
            update={"request_id": uuid4(), "execution_environment": "none"}))
        self.assertEqual((await self.claim(())).task.id, plain.id)
        self.assertEqual((await self.claim()).task.id, sandbox.id)

    async def test_direct_repository_cannot_invent_execution_scope(self):
        with self.assertRaises(TaskError) as caught:
            await self.tasks.create(self.f.owner, self.request, model="fixture")
        self.assertEqual(caught.exception.code, "task_execution_scope_unavailable")

    async def test_source_deletion_removes_issued_scope(self):
        task = await self.create()
        async with self.f.pool.acquire() as connection:
            payload = await connection.fetchval("select payload_id from jarvis_context_events where id=$1", task.request_event_id)
            await connection.execute("update jarvis_context_payloads set content=null,availability='deleted' where id=$1", payload)
            self.assertIsNone(await connection.fetchval("""select p.content from jarvis_operation_grants g
                join jarvis_context_payloads p on p.id=g.scope_payload_id where g.id=$1""", await self.grant(task)))
        self.assertIsNone(await self.claim())

    async def test_cleaned_allocation_allows_same_task_fresh_attempt_without_new_grant(self):
        task = await self.create()
        grant = await self.grant(task)
        claim = await self.claim()
        allocations = SandboxAllocationRepository(self.tasks)
        old = await allocations.reserve(claim, grant)
        await allocations.request_cleanup(self.f.owner, old["id"])
        await self.tasks.finish(claim, WorkerOutcome(TaskStatus.PAUSED, {"reason":"synthetic pause"}))
        await self.service.mutate(self.f.owner, task.id, "resume", TaskMutation(request_id=uuid4(), expected_revision=1))
        new_claim = await self.claim()
        new = await allocations.reserve(new_claim, grant)
        self.assertNotEqual(old["id"], new["id"])
        self.assertEqual(new["grant_id"], old["grant_id"])

    async def test_unknown_old_allocation_blocks_replacement(self):
        task = await self.create()
        grant = await self.grant(task)
        claim = await self.claim()
        allocations = SandboxAllocationRepository(self.tasks)
        old = await allocations.reserve(claim, grant)
        token = await allocations.begin_creation(claim, old["id"])
        await allocations.creation_uncertain(self.f.owner, old["id"], token)
        await self.tasks.finish(claim, WorkerOutcome(TaskStatus.PAUSED, {"reason":"synthetic pause"}))
        await self.service.mutate(self.f.owner, task.id, "resume", TaskMutation(request_id=uuid4(), expected_revision=1))
        with self.assertRaises(TaskError) as caught:
            await allocations.reserve(await self.claim(), grant)
        self.assertEqual(caught.exception.code, "sandbox_cleanup_required")

    async def test_sandbox_grant_cannot_attach_to_another_task_even_before_allocation(self):
        task = await self.create()
        request = self.request.model_copy(update={"request_id":uuid4(), "execution_environment":"none",
                                                  "grant_ids":(await self.grant(task),)})
        with self.assertRaises(asyncpg.CheckViolationError):
            await self.service.create(self.f.owner, request)

    async def test_disabling_execution_keeps_reasoning_admission(self):
        await self.advertise(())
        self.service.sandbox_configuration = None
        with self.assertRaises(TaskError):
            await self.service.create(self.f.owner, self.request)
        task = await self.service.create(self.f.owner, self.request.model_copy(
            update={"request_id":uuid4(), "execution_environment":"none"}))
        self.assertEqual(task.status, TaskStatus.QUEUED)

    async def scope(self, task):
        async with self.f.pool.acquire() as connection:
            content = await connection.fetchval("""select p.content from jarvis_operation_grants g
                join jarvis_context_payloads p on p.id=g.scope_payload_id
                where g.id=$1""", await self.grant(task))
        return json_object(content)

    async def test_network_request_needs_both_configuration_and_worker_capability(self):
        request = self.request.model_copy(update={"execution_environment": "networked_sandbox"})
        await self.advertise(("sandbox.execute", "sandbox.network"))
        with self.assertRaises(TaskError) as caught:
            await self.service.create(self.f.owner, request)
        self.assertEqual(caught.exception.code, "sandbox_network_disabled")
        self.service.sandbox_configuration = SandboxConfiguration(policy=SandboxPolicy(internet_access=True))
        await self.advertise()
        with self.assertRaises(TaskError) as caught:
            await self.service.create(self.f.owner, request)
        self.assertEqual(caught.exception.code, "task_worker_not_ready")
        async with self.f.pool.acquire() as connection:
            self.assertEqual(await connection.fetchval("select count(*) from jarvis_background_tasks"), 0)
            self.assertEqual(await connection.fetchval("select count(*) from jarvis_operation_grants"), 0)
        await self.advertise(("sandbox.execute", "sandbox.network"))
        task = await self.service.create(self.f.owner, request)
        self.assertTrue((await self.scope(task))["internet_access"])
        await self.registry.withdraw(self.f.owner, "fixture")
        self.service.sandbox_configuration = None
        self.assertEqual((await self.service.create(self.f.owner, request)).id, task.id)

    async def test_network_ceiling_does_not_upgrade_offline_and_claims_are_filtered(self):
        self.service.sandbox_configuration = SandboxConfiguration(policy=SandboxPolicy(internet_access=True))
        await self.advertise(("sandbox.execute", "sandbox.network"))
        network = await self.service.create(self.f.owner, self.request.model_copy(
            update={"execution_environment": "networked_sandbox"}))
        offline = await self.service.create(self.f.owner, self.request.model_copy(update={"request_id": uuid4()}))
        self.assertNotIn("internet_access", await self.scope(offline))
        self.assertEqual((await self.claim()).task.id, offline.id)
        self.assertIsNone(await self.claim())
        self.assertEqual((await self.claim(("sandbox.execute", "sandbox.network"))).task.id, network.id)

    async def test_network_resume_keeps_original_scope_and_requires_current_authority(self):
        policy = SandboxPolicy(internet_access=True)
        self.service.sandbox_configuration = SandboxConfiguration(policy=policy)
        await self.advertise(policy.required_capabilities)
        task = await self.service.create(self.f.owner, self.request.model_copy(
            update={"execution_environment": "networked_sandbox"}))
        original_scope = await self.scope(task)
        claim = await self.claim(policy.required_capabilities)
        await self.tasks.finish(claim, WorkerOutcome(TaskStatus.PAUSED, {"reason": "synthetic pause"}))
        request = TaskMutation(request_id=uuid4(), expected_revision=1)
        self.service.sandbox_configuration = SandboxConfiguration()
        with self.assertRaises(TaskError) as caught:
            await self.service.mutate(self.f.owner, task.id, "resume", request)
        self.assertEqual(caught.exception.code, "sandbox_policy_exceeds_worker")
        self.service.sandbox_configuration = SandboxConfiguration(policy=policy)
        await self.advertise()
        with self.assertRaises(TaskError) as caught:
            await self.service.mutate(self.f.owner, task.id, "resume", request)
        self.assertEqual(caught.exception.code, "task_worker_not_ready")
        await self.advertise(policy.required_capabilities)
        await self.service.mutate(self.f.owner, task.id, "resume", request)
        self.assertEqual(await self.scope(task), original_scope)

    async def test_desktop_claim_and_admission_require_explicit_capability(self):
        policy = SandboxPolicy(template="desktop", internet_access=True)
        self.service.sandbox_configuration = SandboxConfiguration(policy=policy)
        request = self.request.model_copy(update={"execution_environment": "networked_desktop_sandbox"})
        await self.advertise(("sandbox.execute", "sandbox.network"))
        with self.assertRaises(TaskError) as caught:
            await self.service.create(self.f.owner, request)
        self.assertEqual(caught.exception.code, "task_worker_not_ready")
        await self.advertise(policy.required_capabilities)
        task = await self.service.create(self.f.owner, request)
        self.assertEqual((await self.scope(task))["template"], "desktop")
        self.assertIsNone(await self.claim(("sandbox.execute", "sandbox.network")))
        self.assertIsNone(await self.claim(("sandbox.execute", "sandbox.desktop")))
        self.assertEqual((await self.claim(policy.required_capabilities)).task.id, task.id)

    async def test_desktop_ceiling_preserves_terminal_and_offline_scopes(self):
        policy = SandboxPolicy(template="desktop", internet_access=True)
        self.service.sandbox_configuration = SandboxConfiguration(policy=policy)
        await self.advertise(policy.required_capabilities)
        desktop = await self.service.create(self.f.owner, self.request.model_copy(
            update={"execution_environment": "desktop_sandbox"}))
        terminal = await self.service.create(self.f.owner, self.request.model_copy(update={"request_id": uuid4()}))
        self.assertEqual((await self.scope(terminal))["template"], "base")
        self.assertNotIn("internet_access", await self.scope(desktop))
        self.assertEqual((await self.claim()).task.id, terminal.id)
        self.assertEqual((await self.claim(("sandbox.execute", "sandbox.desktop"))).task.id, desktop.id)
