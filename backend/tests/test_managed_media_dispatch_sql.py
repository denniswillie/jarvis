"""Real SQL media admission/claim/submission, with synthetic readiness only.

No rendering, provider allocation, model request, storage upload or native app.
The existing fixture records identities and independently audits SQL cleanup.
"""
import asyncio
from datetime import UTC, datetime, timedelta
import os
from pathlib import Path
import unittest
from uuid import uuid4

from app.files.generated import ArtifactPolicy
from app.plugins.authority import PluginAuthority
from app.plugins.deployment import CURATED_DEPLOYMENT_BASELINE_V1
from app.plugins.managed import (
    CAPABILITY, FilesDataBundle, FilesDataEvidence, ManagedPluginRuntime, load_managed_task,
)
from app.plugins.media import MEDIA_CAPABILITY, MEDIA_GRANT_PREFIX, MediaRendererBundle, MediaRendererEvidence
from app.plugins.models import InventoryObservation, ReadinessObservation
from app.plugins.policy import PluginPolicyChange
from app.plugins.repository import PluginPolicyRepository
from app.runtime.feature_config import ArtifactConfiguration, SandboxConfiguration, TaskLimits
from app.tasks.catalog import ConfiguredTaskProvider, TaskWorkerRegistry
from app.tasks.context_repository import ContextRepository
from app.tasks.models import TaskCreate, TaskStatus
from app.tasks.repository import TaskRepository
from app.tasks.runner import TaskRunner
from app.tasks.service import TaskService
from app.workspace.media_recipe import MediaRecipePolicy, TemplatePin
from app.workspace.media_render import RendererBinding
from evals.second_brain.source_fixture import source_fixture


@unittest.skipUnless(os.getenv("JARVIS_SECOND_BRAIN_TEST_DB_URL") and
                     os.getenv("JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"),
                     "Explicit manifested database qualification is required")
class ManagedMediaDispatchSQLTests(unittest.IsolatedAsyncioTestCase):
    async def test_variant_claim_filter_and_existing_durable_submission(self):
        async with asyncio.timeout(180):
            async with source_fixture(os.environ["JARVIS_SECOND_BRAIN_TEST_DB_URL"],
                    Path(os.environ["JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"]), test_name=self.id()) as fixture:
                await self.exercise(fixture)

    async def exercise(self, fixture):
        tasks = TaskRepository(fixture.pool, TaskLimits(lease_seconds=300))
        registry = TaskWorkerRegistry(fixture.pool)
        baseline = CURATED_DEPLOYMENT_BASELINE_V1
        authority = PluginAuthority(PluginPolicyRepository(fixture.pool), ContextRepository(tasks), tasks,
                                    deployment_baseline=baseline)
        choices = {entry.plugin_id: entry for entry in baseline.policy.entries}
        renderer = RendererBinding(
            template=TemplatePin(id="jarvis.clip.v1", version="synthetic-v1", bundle_sha256="b" * 64),
            physical_template_id="synthetic-media-sql-image", build_id="synthetic-build",
            software_sha256="c" * 64, build_receipt_sha256="d" * 64,
            policy=MediaRecipePolicy(max_source_bytes=1024, max_source_duration_ms=1000, max_output_frames=24),
            max_output_bytes=1024)
        media = MediaRendererBundle(configuration_id=choices["video_editor"].configuration_id,
            bundle_version="synthetic-v1", bundle_sha256="a" * 64, renderer=renderer)
        files = FilesDataBundle(configuration_id=choices["files_data"].configuration_id,
            bundle_version="synthetic-v1", bundle_sha256="e" * 64,
            sandbox_template_id="synthetic-files-sql-image", executable="/synthetic/python3")
        now = datetime.now(UTC)
        observations = {}
        for bundle, plugin in ((files, "files_data"), (media, "video_editor")):
            fields = dict(plugin_id=plugin, configuration_id=bundle.configuration_id, observed_at=now,
                          expires_at=now + timedelta(minutes=10), bundle_sha256=bundle.bundle_sha256)
            inventory = InventoryObservation(**fields, state="installed", bundle_version=bundle.bundle_version)
            readiness = ReadinessObservation(**fields, state="ready")
            observations[bundle.configuration_id] = (
                MediaRendererEvidence(fixture.owner, renderer, inventory, readiness) if plugin == "video_editor" else
                FilesDataEvidence(fixture.owner, files.sandbox_template_id, files.executable, inventory, readiness))

        async def observe(connection, owner, selected):
            self.assertTrue(connection.is_in_transaction())
            self.assertEqual(owner, fixture.owner)
            return observations[selected.configuration_id]

        runtime = ManagedPluginRuntime(authority, bundles=(files, media), observe=observe)
        providers = {"frontier": ConfiguredTaskProvider("synthetic-media-dispatch-only")}
        artifacts = ArtifactConfiguration(area_id=fixture.area,
            policy=ArtifactPolicy(formats=("json", "mp4"), mp4_max_file_bytes=1024))
        service = TaskService(tasks, providers, registry=registry, sandbox_configuration=SandboxConfiguration(),
                              artifact_configuration=artifacts, managed_plugins=runtime)
        base_caps = ("sandbox.execute", "files.publish_generated", "files.publish_generated_mp4")
        await registry.advertise(worker_id="synthetic-media-dispatch", owner_id=fixture.owner,
            host_id="synthetic", providers=providers, ttl_seconds=300,
            capabilities=base_caps + (CAPABILITY, MEDIA_CAPABILITY))
        async with fixture.pool.acquire() as connection:
            session = await connection.fetchval("insert into jarvis_sessions(owner_id) values($1) returning id", fixture.owner)

        media_task = await service.create_media(fixture.owner, TaskCreate(
            request_id=uuid4(), title="Synthetic media task", requested_outcome="Synthetic bounded video edit.",
            session_id=session, execution_environment="offline_sandbox", publish_artifacts=True))
        files_task = await service.create_files_data(fixture.owner, TaskCreate(
            request_id=uuid4(), title="Synthetic data task", requested_outcome="Synthetic offline calculation.",
            session_id=session, execution_environment="offline_sandbox"))

        # Media is older, but a Files-only worker must bypass it rather than
        # claim it under a general shell or publication capability.
        self.assertIsNone(await tasks.claim_next("no-plugin-capability", ("frontier",),
            owner_id=fixture.owner, worker_capabilities=base_caps))
        files_claim = await tasks.claim_next("files-only", ("frontier",),
            owner_id=fixture.owner, worker_capabilities=base_caps + (CAPABILITY,))
        self.assertIsNotNone(files_claim)
        self.assertEqual(files_claim.task.id, files_task.id)
        await tasks.abandon(files_claim, reason="synthetic_claim_checked", target=TaskStatus.PAUSED)
        self.assertIsNone(await tasks.claim_next("files-only", ("frontier",),
            owner_id=fixture.owner, worker_capabilities=base_caps + (CAPABILITY,)))
        media_claim = await tasks.claim_next("media-only", ("frontier",),
            owner_id=fixture.owner, worker_capabilities=base_caps + (MEDIA_CAPABILITY,))
        self.assertIsNotNone(media_claim)
        self.assertEqual(media_claim.task.id, media_task.id)

        async with tasks.transaction(fixture.owner) as connection:
            job, grant_id = await load_managed_task(connection, media_claim.task)
            self.assertEqual(job.bundle, media)
            self.assertEqual(await connection.fetchval(
                "select operation_key from jarvis_operation_grants where owner_id=$1 and id=$2",
                fixture.owner, grant_id), MEDIA_GRANT_PREFIX + str(media_task.id))
        context = TaskRunner(tasks, {}).context(media_claim)
        self.assertEqual(await runtime.ensure_submitted(context, grant_id), job)
        async with tasks.transaction(fixture.owner) as connection:
            original_pin = await runtime.submitted_job(connection, tasks, media_claim.task)

        await authority.mutate(owner_id=fixture.owner, command=PluginPolicyChange(
            mutation_id=uuid4(), expected_revision=0,
            enabled_changes=({"plugin_id": "video_editor", "enabled": False},)))
        observations.clear()
        runtime.bundles.clear()
        self.assertEqual(await runtime.ensure_submitted(context, grant_id), job)
        async with tasks.transaction(fixture.owner) as connection:
            self.assertEqual(await runtime.submitted_job(connection, tasks, media_claim.task), original_pin)
            self.assertEqual(await connection.fetchval("""select count(*) from jarvis_task_actions
                where owner_id=$1 and task_id=$2 and operation_key=$3 and status='accepted'""",
                fixture.owner, media_task.id, MEDIA_GRANT_PREFIX + str(media_task.id)), 1)
            self.assertEqual(await connection.fetchval(
                "select count(*) from jarvis_sandbox_allocations where owner_id=$1", fixture.owner), 0)
        await tasks.abandon(media_claim, reason="synthetic_submission_checked", target=TaskStatus.PAUSED)
        await registry.withdraw(fixture.owner, "synthetic-media-dispatch")


if __name__ == "__main__":
    unittest.main()
