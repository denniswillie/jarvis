"""Real SQL qualification of managed task/action authority, without a provider.

The bundle and readiness are explicitly synthetic fixture inputs. Passing this
test establishes neither installed software nor live runtime/plugin activation.
All writes use one manifested disposable schema and synthetic owners.
"""
import asyncio
from dataclasses import replace
from datetime import UTC, datetime, timedelta
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from uuid import uuid4

from app.plugins.authority import PluginAuthority
from app.plugins.deployment import CURATED_DEPLOYMENT_BASELINE_V1
from app.plugins.managed import (
    CAPABILITY, GRANT_PREFIX, FilesDataBundle, FilesDataEvidence,
    ManagedFilesDataRuntime, load_managed_task, submission_receipt,
)
from app.plugins.models import InventoryObservation, ReadinessObservation
from app.plugins.policy import PluginPolicyChange
from app.plugins.repository import PluginPolicyRepository
from app.runtime.feature_config import SandboxConfiguration, TaskLimits
from app.tasks.catalog import ConfiguredTaskProvider, TaskWorkerRegistry
from app.tasks.context_repository import ContextRepository
from app.tasks.models import ClaimLost, TaskCreate, TaskError, json_object
from app.tasks.repository import TaskRepository
from app.tasks.service import TaskService
from evals.second_brain.source_fixture import source_fixture


@unittest.skipUnless(os.getenv("JARVIS_SECOND_BRAIN_TEST_DB_URL") and
                     os.getenv("JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"),
                     "Explicit manifested database qualification is required")
class ManagedFilesDataSQLTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_admission_submission_disable_replay_and_source_withdrawal(self):
        async with asyncio.timeout(180):
            async with source_fixture(
                os.environ["JARVIS_SECOND_BRAIN_TEST_DB_URL"],
                Path(os.environ["JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"]),
                test_name=self.id(),
            ) as fixture:
                await self.exercise(fixture)

    async def exercise(self, fixture):
        tasks = TaskRepository(fixture.pool, TaskLimits(
            owner_concurrency=4, global_concurrency=4, lease_seconds=300))
        contexts = ContextRepository(tasks)
        baseline = CURATED_DEPLOYMENT_BASELINE_V1
        authority = PluginAuthority(PluginPolicyRepository(fixture.pool), contexts,
                                    tasks, deployment_baseline=baseline)
        selected = next(item for item in baseline.policy.entries
                        if item.plugin_id == "files_data")
        bundle = FilesDataBundle(configuration_id=selected.configuration_id,
            bundle_version="synthetic-sql-fixture-v1", bundle_sha256="a" * 64,
            sandbox_template_id="synthetic-no-provider", executable="/fixture/python3")
        now = datetime.now(UTC)
        observation = dict(plugin_id="files_data", configuration_id=bundle.configuration_id,
                           observed_at=now, expires_at=now + timedelta(minutes=10),
                           bundle_sha256=bundle.bundle_sha256)
        evidence = FilesDataEvidence(fixture.owner, bundle.sandbox_template_id,
            bundle.executable,
            InventoryObservation(**observation, state="installed", bundle_version=bundle.bundle_version),
            ReadinessObservation(**observation, state="ready"))

        async def observe(connection, owner, requested_bundle):
            self.assertTrue(connection.is_in_transaction())
            self.assertEqual(owner, fixture.owner)
            self.assertEqual(requested_bundle, bundle)
            return evidence

        runtime = ManagedFilesDataRuntime(authority, bundles=(bundle,), observe=observe)
        registry = TaskWorkerRegistry(fixture.pool)
        providers = {"frontier": ConfiguredTaskProvider("synthetic-sql-only")}
        service = TaskService(tasks, providers, registry=registry,
            sandbox_configuration=SandboxConfiguration(), managed_plugins=runtime)
        capabilities = ("sandbox.execute", CAPABILITY)

        def request(label):
            return TaskCreate(request_id=uuid4(), title=label,
                requested_outcome="Synthetic SQL authority test; execute no provider operation.",
                execution_environment="offline_sandbox")

        async def advertise(caps):
            await registry.advertise(worker_id="managed-sql-fixture", owner_id=fixture.owner,
                host_id="synthetic", providers=providers, ttl_seconds=300, capabilities=caps)

        async def task_count():
            async with fixture.pool.acquire() as connection:
                return await connection.fetchval(
                    "select count(*) from jarvis_background_tasks where owner_id=$1", fixture.owner)

        first_request, queued_request = request("Submitted fixture"), request("Queued fixture")
        with self.subTest("ordinary sandbox readiness cannot admit managed work"):
            await advertise(("sandbox.execute",))
            with self.assertRaises(TaskError) as denied:
                await service.create_files_data(fixture.owner, first_request)
            self.assertEqual(denied.exception.code, "task_worker_not_ready")
            self.assertEqual(await task_count(), 0)

        await advertise(capabilities)
        first = await service.create_files_data(fixture.owner, first_request)
        queued = await service.create_files_data(fixture.owner, queued_request)
        with self.subTest("admission persists exact mapping and grant but no submission"):
            self.assertEqual((await service.create_files_data(fixture.owner, first_request)).id, first.id)
            self.assertEqual(await task_count(), 2)
            async with tasks.transaction(fixture.owner) as connection:
                job, first_grant = await load_managed_task(connection, first)
                queued_job, queued_grant = await load_managed_task(connection, queued)
                self.assertEqual(job.bundle, bundle)
                self.assertEqual(queued_job.bundle, bundle)
                self.assertEqual(job.binding.deployment_baseline, baseline.identity)
                self.assertEqual(await connection.fetchval(
                    "select count(*) from jarvis_task_actions where owner_id=$1", fixture.owner), 0)
                self.assertEqual(await connection.fetchval(
                    "select count(*) from jarvis_operation_grants where owner_id=$1", fixture.owner), 2)

        claim = await tasks.claim_next("managed-sql-fixture", ("frontier",),
            owner_id=fixture.owner, worker_capabilities=capabilities)
        self.assertIsNotNone(claim)
        self.assertEqual(claim.task.id, first.id)
        context = SimpleNamespace(tasks=tasks, task=claim.task, claim=claim)
        self.assertEqual(await runtime.ensure_submitted(context, first_grant), job)
        async with tasks.transaction(fixture.owner) as connection:
            pin = await runtime.submitted_job(connection, tasks, claim.task)
            row = await connection.fetchrow("""select a.*,p.content as receipt
                from jarvis_task_actions a join jarvis_context_payloads p
                  on p.owner_id=a.owner_id and p.id=a.receipt_payload_id
                where a.owner_id=$1 and a.task_id=$2 and a.operation_key=$3""",
                fixture.owner, first.id, GRANT_PREFIX + str(first.id))
            self.assertEqual(row["status"], "accepted")
            self.assertEqual(row["submission_token"], pin.submission_token)
            self.assertEqual(json_object(row["receipt"]),
                             submission_receipt(claim.task, job, pin.submission_token))
            self.assertEqual(json_object(row["receipt"])["effect"], "not_yet_observed")

        await authority.mutate(owner_id=fixture.owner, command=PluginPolicyChange(
            mutation_id=uuid4(), expected_revision=0,
            enabled_changes=({"plugin_id": "files_data", "enabled": False},)))
        with self.subTest("disabled queued work never acquires a submitted pin"):
            queued_claim = await tasks.claim_next("managed-sql-fixture", ("frontier",),
                owner_id=fixture.owner, worker_capabilities=capabilities)
            self.assertIsNotNone(queued_claim)
            self.assertEqual(queued_claim.task.id, queued.id)
            queued_context = SimpleNamespace(tasks=tasks, task=queued_claim.task, claim=queued_claim)
            with self.assertRaises(TaskError) as denied:
                await runtime.ensure_submitted(queued_context, queued_grant)
            self.assertEqual(denied.exception.code, "plugin_disabled")
            async with tasks.transaction(fixture.owner) as connection:
                row = await connection.fetchrow("""select status,submission_token,receipt_payload_id
                    from jarvis_task_actions where owner_id=$1 and task_id=$2""", fixture.owner, queued.id)
                self.assertEqual(row["status"], "authorized")
                self.assertIsNone(row["submission_token"])
                self.assertIsNone(row["receipt_payload_id"])
            with self.assertRaises(TaskError) as denied:
                await service.create_files_data(fixture.owner, request("Disabled new fixture"))
            self.assertEqual(denied.exception.code, "plugin_disabled")
            self.assertEqual(await task_count(), 2)

        with self.subTest("same submitted job survives disable without minting another action"):
            evidence = None
            runtime.bundles.clear()
            self.assertEqual(await runtime.ensure_submitted(context, first_grant), job)
            async with tasks.transaction(fixture.owner) as connection:
                self.assertEqual(await runtime.submitted_job(connection, tasks, claim.task), pin)
                with self.assertRaises(TaskError) as denied:
                    await runtime.submitted_job(connection, tasks, claim.task,
                        expected=replace(pin, submission_token=uuid4()))
                self.assertEqual(denied.exception.code, "managed_submission_mismatch")
                self.assertEqual(await connection.fetchval("""select count(*) from jarvis_task_actions
                    where owner_id=$1 and task_id=$2""", fixture.owner, first.id), 1)

        with self.subTest("ordinary terminal is independent and exact managed replay needs no worker"):
            await advertise(("sandbox.execute",))
            ordinary = await service.create(fixture.owner, request("Ordinary sandbox fixture"))
            async with tasks.transaction(fixture.owner) as connection:
                self.assertIsNone(await load_managed_task(connection, ordinary))
            await registry.withdraw(fixture.owner, "managed-sql-fixture")
            service.managed_plugins = None
            service.providers = {}
            self.assertEqual((await service.create_files_data(fixture.owner, first_request)).id, first.id)
            with self.assertRaises(TaskError) as denied:
                await service.create(fixture.owner, first_request)
            self.assertEqual(denied.exception.code, "task_request_conflict")
            self.assertEqual(await task_count(), 3)

        with self.subTest("withdrawing original request invalidates the submitted continuation"):
            async with tasks.transaction(fixture.owner) as connection:
                await connection.execute("""update jarvis_context_payloads set availability='deleted',content=null
                    where owner_id=$1 and id=(select payload_id from jarvis_context_events
                        where owner_id=$1 and id=$2)""", fixture.owner, first.request_event_id)
            with self.assertRaises((ClaimLost, TaskError)):
                await runtime.ensure_submitted(context, first_grant)


if __name__ == "__main__":
    unittest.main()
