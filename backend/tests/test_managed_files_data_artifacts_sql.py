"""Real managed worker/action/publication SQL with in-memory external services.

No live sandbox, installed-bundle claim, object-storage upload or model call is
made. The existing manifested source fixture independently audits SQL cleanup.
"""
import asyncio
from datetime import UTC, datetime, timedelta
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from uuid import uuid4

from app.plugins.authority import PluginAuthority
from app.plugins.deployment import CURATED_DEPLOYMENT_BASELINE_V1
from app.plugins.managed import CAPABILITY, GRANT_PREFIX, FilesDataBundle, FilesDataEvidence, ManagedFilesDataRuntime
from app.plugins.models import InventoryObservation, ReadinessObservation
from app.plugins.policy import PluginPolicyChange
from app.plugins.repository import PluginPolicyRepository
from app.runtime.feature_config import ArtifactConfiguration, SandboxConfiguration, TaskLimits
from app.tasks.catalog import ConfiguredTaskProvider, TaskWorkerRegistry
from app.tasks.context_repository import ContextRepository
from app.tasks.models import TaskCreate
from app.tasks.repository import TaskRepository
from app.tasks.runner import TaskRunner
from app.tasks.service import TaskService
from app.tasks.worker import WorkerContextFactory
from evals.second_brain.file_storage_fixture import MemoryStorage
from evals.second_brain.source_fixture import source_fixture
from test_sandbox_artifacts import OutputProvider


@unittest.skipUnless(os.getenv("JARVIS_SECOND_BRAIN_TEST_DB_URL") and
                     os.getenv("JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"),
                     "Explicit manifested database qualification is required")
class ManagedFilesDataArtifactSQLTests(unittest.IsolatedAsyncioTestCase):
    async def test_submitted_worker_publishes_after_disable_and_replays_without_another_upload(self):
        async with asyncio.timeout(180):
            async with source_fixture(
                os.environ["JARVIS_SECOND_BRAIN_TEST_DB_URL"],
                Path(os.environ["JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"]),
                test_name=self.id(),
            ) as fixture:
                await self.exercise(fixture)

    async def exercise(self, fixture):
        tasks = TaskRepository(fixture.pool, TaskLimits(lease_seconds=300))
        registry = TaskWorkerRegistry(fixture.pool)
        baseline = CURATED_DEPLOYMENT_BASELINE_V1
        authority = PluginAuthority(PluginPolicyRepository(fixture.pool),
            ContextRepository(tasks), tasks, deployment_baseline=baseline)
        choice = next(item for item in baseline.policy.entries if item.plugin_id == "files_data")
        bundle = FilesDataBundle(configuration_id=choice.configuration_id,
            bundle_version="synthetic-publication-sql-v1", bundle_sha256="b" * 64,
            sandbox_template_id="fixture", executable="/fixture/python3")
        now = datetime.now(UTC)
        fields = dict(plugin_id="files_data", configuration_id=bundle.configuration_id,
            observed_at=now, expires_at=now + timedelta(minutes=10), bundle_sha256=bundle.bundle_sha256)
        evidence = FilesDataEvidence(fixture.owner, bundle.sandbox_template_id, bundle.executable,
            InventoryObservation(**fields, state="installed", bundle_version=bundle.bundle_version),
            ReadinessObservation(**fields, state="ready"))

        async def observe(connection, owner, selected_bundle):
            self.assertTrue(connection.is_in_transaction())
            self.assertEqual((owner, selected_bundle), (fixture.owner, bundle))
            return evidence

        runtime = ManagedFilesDataRuntime(authority, bundles=(bundle,), observe=observe)
        providers = {"frontier": ConfiguredTaskProvider("synthetic-publication-only")}
        artifacts = ArtifactConfiguration(area_id=fixture.area)
        service = TaskService(tasks, providers, registry=registry,
            sandbox_configuration=SandboxConfiguration(), artifact_configuration=artifacts,
            managed_plugins=runtime)
        capabilities = ("sandbox.execute", "files.publish_generated", CAPABILITY)
        await registry.advertise(worker_id="managed-publication-fixture", owner_id=fixture.owner,
            host_id="synthetic", providers=providers, ttl_seconds=300, capabilities=capabilities)
        async with fixture.pool.acquire() as connection:
            session = await connection.fetchval(
                "insert into jarvis_sessions(owner_id) values($1) returning id", fixture.owner)
        task = await service.create_files_data(fixture.owner, TaskCreate(
            request_id=uuid4(), title="Synthetic managed report",
            requested_outcome="Create a synthetic JSON report; no real-world action.",
            session_id=session, execution_environment="offline_sandbox", publish_artifacts=True))
        claim = await tasks.claim_next("managed-publication-fixture", ("frontier",),
            owner_id=fixture.owner, worker_capabilities=capabilities)
        self.assertIsNotNone(claim)
        self.assertEqual(claim.task.id, task.id)
        context = TaskRunner(tasks, {}).context(claim)
        storage, provider = MemoryStorage(), OutputProvider()
        knowledge = SimpleNamespace(sources=fixture.sources,
            retrieval=SimpleNamespace(sources=fixture.sources),
            configuration=SimpleNamespace(area_id=fixture.area), context_for=lambda owner: None)
        factory = WorkerContextFactory(fixture.pool, knowledge=knowledge,
            file_storage=storage, sandbox_provider=provider,
            artifact_configuration=artifacts, managed_plugins=runtime)
        self.assertIn(CAPABILITY, factory.capabilities)
        prepared = await factory(context)
        bridge = prepared.tools
        names = {item["name"] for item in bridge.catalog()}
        self.assertIn("files_data_execute", names)
        self.assertNotIn("sandbox_execute", names)
        self.assertIn("sandbox_publish_file", names)

        allocation = await bridge.broker._ready(await bridge.broker.allocation())
        async with tasks.transaction(fixture.owner) as connection:
            original_pin = await runtime.submitted_job(connection, tasks, claim.task)
        provider.files["summary.json"] = b'{"total":10}\n'
        await authority.mutate(owner_id=fixture.owner, command=PluginPolicyChange(
            mutation_id=uuid4(), expected_revision=0,
            enabled_changes=({"plugin_id": "files_data", "enabled": False},)))
        evidence = None
        runtime.bundles.clear()
        arguments = dict(operation_key="summary", path="summary.json", filename="summary.json",
            format="json", description="Synthetic computed report.")
        first = await bridge.invoke("sandbox_publish_file", arguments, provider_call_id="first-publication")
        self.assertEqual(first["status"], "success", first)
        self.assertEqual(storage.uploads, 1)
        self.assertEqual(provider.reads, 1)
        self.assertEqual(first["task_operation"]["state"], "accepted")
        self.assertIn("file", first)

        # Lose the external workspace after confirmed publication. Recovery uses
        # the same source/action records, not another file read or upload.
        provider.resources.clear()
        await bridge.broker.allocations.request_cleanup(fixture.owner, allocation["id"])
        fresh = await factory(context)
        replay = await fresh.tools.invoke("sandbox_publish_file", arguments, provider_call_id="replay-publication")
        status = await fresh.tools.invoke("sandbox_artifact_status", {"operation_key": "summary"},
                                          provider_call_id="status-publication")
        self.assertEqual(replay["status"], "success", replay)
        self.assertEqual(status["status"], "success", status)
        self.assertEqual(replay["file"], first["file"])
        self.assertEqual(status["file"], first["file"])
        self.assertEqual(replay["task_operation"]["operation_ref"], first["task_operation"]["operation_ref"])
        self.assertEqual((storage.uploads, provider.reads), (1, 1))
        verified = await fresh.tools.verified_operation(first["task_operation"]["operation_ref"])
        self.assertIsNotNone(verified)
        async with tasks.transaction(fixture.owner) as connection:
            self.assertEqual(await runtime.submitted_job(connection, tasks, claim.task), original_pin)
            self.assertEqual(await connection.fetchval("""select count(*) from jarvis_task_actions
                where owner_id=$1 and task_id=$2 and operation_key=$3""",
                fixture.owner, task.id, GRANT_PREFIX + str(task.id)), 1)
            rows = await connection.fetch("""select a.status,a.submission_token from jarvis_task_actions a
                join jarvis_operation_grants g on g.owner_id=a.owner_id and g.id=a.grant_id
                where a.owner_id=$1 and a.task_id=$2 and g.capability='files.publish_snapshot'""",
                fixture.owner, task.id)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["status"], "accepted")
            self.assertIsNotNone(rows[0]["submission_token"])
        await registry.withdraw(fixture.owner, "managed-publication-fixture")


if __name__ == "__main__":
    unittest.main()
