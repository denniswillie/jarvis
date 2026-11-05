"""Core-scheduled real SQL file-admission/worker-heartbeat lock regression.

Real recording/source publication, command/grant admission, worker context,
file evidence binding and TaskRepository.renew. Two gates force the overlap;
no lock acquisition or validation is mocked. Synthetic non-MP4 bytes, inspection,
inventory and provider IDs are explicit fixtures. No codec, VM, model or cloud.
"""
import asyncio
from datetime import UTC, datetime, timedelta
import hashlib
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from uuid import UUID, uuid4

import asyncpg

from app.files.generated import ArtifactPolicy
from app.files.recording_ingress import RecordingIngress, RecordingUpload
from app.files.transfer import FileTransferPolicy, MP4Inspection
from app.files.workspace_access import WorkspaceFileAccess
from app.knowledge import files as file_module
from app.knowledge.files import SourceFileRepository
from app.knowledge.models import SourceError
from app.knowledge.session_areas import SessionKnowledgeAreas
from app.live_state.admission import CommandAdmission
from app.plugins.authority import PluginAuthority
from app.plugins.deployment import CURATED_DEPLOYMENT_BASELINE_V1
from app.plugins.managed import ManagedPluginRuntime
from app.plugins.media import MEDIA_CAPABILITY, MediaRendererBundle, MediaRendererEvidence
from app.plugins.models import InventoryObservation, ReadinessObservation
from app.plugins.repository import PluginPolicyRepository
from app.recordings.admission import RecordingAdmission
from app.runtime.feature_config import ArtifactConfiguration, SandboxConfiguration, TaskLimits
from app.tasks.catalog import ConfiguredTaskProvider, TaskWorkerRegistry
from app.tasks.context_repository import ContextRepository
from app.tasks.media_editing import MEDIA_TOOL
from app.tasks.memory_evidence import TaskMemoryCitation, TaskMemoryEvidence
from app.tasks.repository import TaskRepository
from app.tasks.runner import TaskRunner
from app.tasks.service import TaskService
from app.tasks.worker import WorkerContextFactory
from app.workspace.media_agent import WorkspaceMediaAgent
from app.workspace.media_recipe import MediaRecipePolicy, TemplatePin
from app.workspace.media_render import RendererBinding
from app.workspace.models import WorkspaceContext
from evals.second_brain.source_fixture import source_fixture
from tests.test_file_backed_publication_sql import DiskStorage


@unittest.skipUnless(os.getenv("JARVIS_SECOND_BRAIN_TEST_DB_URL") and
                     os.getenv("JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"),
                     "Core must allocate the manifested SQL lane explicitly")
class MediaPublicationLocksSQLTests(unittest.IsolatedAsyncioTestCase):
    async def test_file_evidence_and_real_heartbeat_complete_without_lock_cycle(self):
        directory = Path(os.environ["JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"])
        self.assertTrue(directory.is_absolute())
        async with asyncio.timeout(180):
            async with source_fixture(os.environ["JARVIS_SECOND_BRAIN_TEST_DB_URL"],
                                      directory, test_name=self.id()) as fixture:
                with tempfile.TemporaryDirectory(prefix="jarvis-media-locks-sql-") as root:
                    await self.exercise(fixture, Path(root).resolve())

    async def exercise(self, fixture, root):
        owner = fixture.owner
        tasks = TaskRepository(fixture.pool, TaskLimits(lease_seconds=300))
        async with fixture.pool.acquire() as connection:
            session = await connection.fetchval(
                "insert into jarvis_sessions(owner_id) values($1) returning id", owner)
        selections = SessionKnowledgeAreas(fixture.sources, tasks, fixture.area)
        recordings = RecordingAdmission(fixture.sources, tasks, selections, owner_id=owner)
        policy = FileTransferPolicy("mp4_capture_v1", 1024)
        storage = DiskStorage(root / "objects")
        storage.lose_ack = False

        async def inspector(local, admitted_policy):
            # Declared metadata only: this test does not qualify media decoding.
            self.assertLessEqual(local.size_bytes, admitted_policy.max_bytes)
            return MP4Inspection(size_bytes=local.size_bytes, content_sha256=local.content_sha256,
                duration_ms=1000, width=320, height=240, video_codec="h264", audio_codec="aac")

        def carrier(tool, capability, arguments, call):
            return CommandAdmission(owner_id=owner, session_id=session,
                provider_session_key="synthetic-lock-regression", provider_call_id=call,
                tool_name=tool, capability=capability, normalized_arguments=arguments)

        started = await recordings.admit(carrier("fixture_recording", "recordings.start",
            {"include_jarvis_audio": True}, "start"), area_id=fixture.area, policy=policy)
        recording_id = started.scope.recording_id
        await recordings.stop(carrier("fixture_recording", "recordings.stop",
                                      {"recording_id": str(recording_id)}, "stop"))
        content = b"Synthetic recording bytes for SQL locks, not a playable MP4."

        async def chunks():
            yield content

        saved = await RecordingIngress(fixture.sources, storage, recordings,
            policy=policy, inspect_file=inspector).finalize(owner_id=owner, session_id=session,
            recording_id=recording_id, upload=RecordingUpload(filename="sql-lock-fixture.mp4",
                description="Synthetic SQL fixture, no codec or physical capture.",
                size_bytes=len(content), content_sha256=hashlib.sha256(content).hexdigest(),
                client_report={"generation_id": str(uuid4()), "timeline_end_ms": 1000,
                    "video_frames": 24, "source_audio_frames": 48000, "padded_audio_frames": 0}),
            chunks=chunks())
        self.assertEqual(saved["outcome"], "saved")
        foreground = SourceFileRepository(fixture.sources, storage, owner_id=owner,
            area_id=fixture.area, selections=selections, media_policy=policy, media_inspector=inspector)
        baseline = CURATED_DEPLOYMENT_BASELINE_V1
        choice = next(item for item in baseline.policy.entries if item.plugin_id == "video_editor")
        authority = PluginAuthority(PluginPolicyRepository(fixture.pool), ContextRepository(tasks), tasks,
                                    deployment_baseline=baseline)
        renderer = RendererBinding(
            template=TemplatePin(id="jarvis.clip.v1", version="synthetic-v1", bundle_sha256="b" * 64),
            physical_template_id="synthetic-lock-test", build_id="synthetic-no-build",
            software_sha256="c" * 64, build_receipt_sha256="d" * 64,
            policy=MediaRecipePolicy(max_source_bytes=1024, max_source_duration_ms=1000, max_output_frames=24),
            max_output_bytes=1024)
        bundle = MediaRendererBundle(configuration_id=choice.configuration_id, bundle_version="synthetic-v1",
                                     bundle_sha256="a" * 64, renderer=renderer)
        now = datetime.now(UTC)
        fields = dict(plugin_id="video_editor", configuration_id=bundle.configuration_id,
            observed_at=now, expires_at=now + timedelta(minutes=10), bundle_sha256=bundle.bundle_sha256)
        readiness = MediaRendererEvidence(owner, renderer,
            InventoryObservation(**fields, state="installed", bundle_version=bundle.bundle_version),
            ReadinessObservation(**fields, state="ready"))

        async def observe(connection, selected_owner, selected):
            self.assertEqual((selected_owner, selected), (owner, bundle))
            return readiness

        managed = ManagedPluginRuntime(authority, bundles=(bundle,), observe=observe)
        registry = TaskWorkerRegistry(fixture.pool)
        providers = {"frontier": ConfiguredTaskProvider("synthetic-no-model-call")}
        service = TaskService(tasks, providers, registry=registry, sandbox_configuration=SandboxConfiguration(),
            artifact_configuration=ArtifactConfiguration(area_id=fixture.area,
                policy=ArtifactPolicy(formats=("json", "mp4"), mp4_max_file_bytes=1024)), managed_plugins=managed)
        worker_id = "media-locks-" + uuid4().hex
        capabilities = ("sandbox.execute", "files.publish_generated", "files.publish_generated_mp4", MEDIA_CAPABILITY)
        await registry.advertise(worker_id=worker_id, owner_id=owner, host_id="synthetic",
                                 providers=providers, ttl_seconds=300, capabilities=capabilities)
        try:
            arguments = {"instruction": "Trim the first half second and preserve its sound.",
                "recording_id": str(recording_id), "saved": {key: saved[key] for key in
                    ("file_id", "source_id", "version_id", "receipt_event_id")}}
            command = carrier(MEDIA_TOOL, "tasks.create", arguments, "edit")
            accepted = await WorkspaceMediaAgent(service, WorkspaceFileAccess(foreground), recordings).submit(
                WorkspaceContext(owner, session, command.operation_id), arguments, command_admission=command)
            self.assertEqual(accepted["outcome"], "accepted")
            claim = await tasks.claim_next(worker_id, ("frontier",), owner_id=owner,
                                          worker_capabilities=capabilities)
            self.assertIsNotNone(claim)
            context = TaskRunner(tasks, {}).context(claim)
            await WorkerContextFactory(fixture.pool)(context)
            evidence = TaskMemoryEvidence(context, fixture.sources, area_id=fixture.area)
            worker_files = SourceFileRepository(fixture.sources, storage, owner_id=owner,
                area_id=fixture.area, task_evidence=evidence, media_policy=policy, media_inspector=inspector)
            file_id = UUID(saved["file_id"])
            locked, heartbeat_owns, read_wants_owner = asyncio.Event(), asyncio.Event(), asyncio.Event()
            original_read, original_lock = file_module.read_file_original, tasks.lock_owner
            read_task, heartbeat = None, None

            async def gated_read(*args, **kwargs):
                value = await original_read(*args, **kwargs)
                if asyncio.current_task() is read_task:
                    locked.set()
                    await heartbeat_owns.wait()
                return value

            async def gated_owner(connection, selected_owner):
                if asyncio.current_task() is read_task:
                    read_wants_owner.set()
                await original_lock(connection, selected_owner)
                if asyncio.current_task() is heartbeat:
                    heartbeat_owns.set()
                    await read_wants_owner.wait()

            # Read holds its real publication guard first. Heartbeat then owns
            # the real task lock before either proceeds. No sleep-based race.
            with patch.object(file_module, "read_file_original", gated_read), \
                    patch.object(tasks, "lock_owner", gated_owner):
                try:
                    async with asyncio.timeout(30):
                        read_task = asyncio.create_task(worker_files.get_file(owner_id=owner, file_id=file_id))
                        await locked.wait()
                        # SHARE permits concurrent validation, but still excludes
                        # a mutation/deletion lock. The attempted lock rolls back.
                        async with fixture.pool.acquire() as connection:
                            with self.assertRaises(asyncpg.LockNotAvailableError):
                                async with connection.transaction():
                                    await connection.fetchrow("select id from jarvis_file_publications "
                                        "where owner_id=$1 and file_id=$2 for update nowait", owner, file_id)
                        heartbeat = asyncio.create_task(tasks.renew(claim))
                        outcomes = await asyncio.gather(read_task, heartbeat, return_exceptions=True)
                        for outcome in outcomes:
                            if isinstance(outcome, BaseException):
                                # Preserve the actual DB DETAIL/stack in Core's
                                # diagnostic receipt, not only an error class.
                                raise outcome
                        file, renewed = outcomes
                finally:
                    children = [item for item in (read_task, heartbeat) if item is not None]
                    for child in children:
                        if not child.done():
                            child.cancel()
                    await asyncio.gather(*children, return_exceptions=True)
            self.assertIs(renewed, True)
            self.assertEqual(file.id, file_id)
            self.assertEqual(file.task_evidence["source_id"], saved["source_id"])
            await evidence.verify(TaskMemoryCitation.model_validate(file.task_evidence))
            # Normal source withdrawal still fences further worker access.
            await foreground.for_session(session).delete_source_file(owner_id=owner, file_id=file_id)
            self.assertIsNone(await worker_files.get_file(owner_id=owner, file_id=file_id))
            with self.assertRaises(SourceError):
                await fixture.sources.read(owner, fixture.area, UUID(saved["source_id"]))
            async with tasks.transaction(owner) as connection:
                current = await tasks.load(connection, owner, claim.task.id)
                self.assertIsNotNone(await tasks.context_problem(connection, current))
        finally:
            await registry.withdraw(owner, worker_id)


if __name__ == "__main__":
    unittest.main()
