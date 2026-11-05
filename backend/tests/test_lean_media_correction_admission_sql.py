"""Core-scheduled accepted-output continuation SQL qualification, no rendering.

Real: RecordingAdmission/Ingress, source provenance, existing task/action/grant
transactions, GeneratedArtifactPublisher/MediaPublicationStore, command admission,
readonly context snapshots and dependency withdrawal. Storage is local DiskStorage.

Synthetic: command carriers (not Voice/provider calls), non-MP4 bytes, metadata and
decode observations used to construct a fixture plan, renderer inventory/readiness,
and a claimed fixture producer (not Hermes). No codec, model, provider or VM runs.
The previous task uses the existing finalized-user-input sandbox fixture; initial
voice media admission is covered separately by test_lean_media_admission_sql.
"""
import asyncio
from datetime import UTC, datetime, timedelta
import hashlib
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock, patch
from uuid import UUID, uuid4

from app.errors import JarvisHTTPError
from app.files.generated import ArtifactPolicy
from app.files.local_file import snapshot_file
from app.files.recording_ingress import RecordingClientReport, RecordingIngress, RecordingUpload
from app.files.tools import FileToolModule
from app.files.transfer import FileTransferPolicy, MP4Inspection
from app.files.workspace_access import WorkspaceFileAccess
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
from app.runtime.feature_config import ArtifactConfiguration, SandboxConfiguration
from app.tasks.catalog import ConfiguredTaskProvider, TaskWorkerRegistry
from app.tasks.context_repository import ContextRepository
from app.tasks.generated_artifacts import GeneratedArtifactPublisher
from app.tasks.media_editing import (
    MEDIA_TOOL, AcceptedOutputTarget, CorrectedMediaInput, media_description,
    require_media_recipe_input, validate_media_input,
)
from app.tasks.memory_evidence import TaskMemoryEvidence
from app.tasks.models import TaskError, TaskStatus, WorkerOutcome
from app.tasks.runner import TaskRunner
from app.tasks.service import TaskService
from app.tasks.tool_bridge import TaskToolBridge
from app.tasks.worker import WorkerContextFactory
from app.workspace.media_agent import WorkspaceMediaAgent
from app.workspace.media_publication import MediaPublicationStore
from app.workspace.media_recipe import MediaRecipeError, RetainedRef
from app.workspace.models import WorkspaceContext
from evals.second_brain.sandbox_fixture import sandbox_task_fixture
from tests import test_workspace_media_publication_sql as publication_fixture


LIMIT = publication_fixture.LIMIT
WORKER = "synthetic-correction-admission-only"


@unittest.skipUnless(os.getenv("JARVIS_SECOND_BRAIN_TEST_DB_URL") and
                     os.getenv("JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"),
                     "Explicit manifested Core SQL lane required")
class LeanMediaCorrectionSQLTests(unittest.IsolatedAsyncioTestCase):
    async def test_accepted_output_session_b_replay_readonly_and_withdrawal(self):
        manifest = Path(os.environ["JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"])
        self.assertTrue(manifest.is_absolute(), "Use an absolute manifested cleanup directory")
        async with asyncio.timeout(180):
            async with sandbox_task_fixture(
                os.environ["JARVIS_SECOND_BRAIN_TEST_DB_URL"], manifest,
                test_name=self.id(), with_session=True,
                requested_outcome="Synthetic retained-media producer for exact continuation admission; no render.",
            ) as fixture:
                with tempfile.TemporaryDirectory(prefix="jarvis-media-correction-sql-") as directory:
                    await self.exercise(fixture, Path(directory).resolve())

    def command(self, session, call, capability, arguments, *, tool=MEDIA_TOOL):
        # Explicit test-owned trusted input, never a fabricated provider receipt.
        return CommandAdmission(owner_id=self.owner, session_id=session,
            provider_session_key="synthetic-sql-session:" + str(session),
            provider_call_id=call, tool_name=tool, capability=capability,
            normalized_arguments=arguments)

    async def exercise(self, fixture, root):
        source, tasks = fixture.source, fixture.tasks
        self.owner, self.task_id = source.owner, fixture.task.id
        capture_a = fixture.task.session_id
        self.assertIsNotNone(capture_a)
        async with source.pool.acquire() as connection:
            command_b = await connection.fetchval(
                "insert into jarvis_sessions(owner_id) values($1) returning id", self.owner)
        self.assertNotEqual(capture_a, command_b)
        selections = SessionKnowledgeAreas(source.sources, tasks, source.area)
        admissions = RecordingAdmission(source.sources, tasks, selections, owner_id=self.owner)
        storage = publication_fixture.MediaPlanDiskStorage(root / "objects")
        storage.lose_ack = False

        async def synthetic_inspector(local, policy):
            # Declared fixture metadata only. These bytes are NOT a decodable MP4.
            self.assertLessEqual(local.size_bytes, policy.max_bytes)
            return MP4Inspection(size_bytes=local.size_bytes, content_sha256=local.content_sha256,
                duration_ms=10000, width=720, height=1280, video_codec="h264", audio_codec="aac")

        capture_policy = FileTransferPolicy("mp4_capture_v1", LIMIT)
        start = self.command(capture_a, "capture-start", "recordings.start",
                             {"include_jarvis_audio": True}, tool="fixture_recording_command")
        admission = await admissions.admit(start, area_id=source.area, policy=capture_policy)
        self.assertEqual(admission.outcome, "admitted")
        recording_id = admission.scope.recording_id
        stopped = await admissions.stop(self.command(capture_a, "capture-stop", "recordings.stop",
            {"recording_id": str(recording_id)}, tool="fixture_recording_command"))
        self.assertEqual(stopped.outcome, "capture_closed")
        capture_bytes = b"Explicit synthetic recording bytes; not a playable MP4."
        capture_hash = hashlib.sha256(capture_bytes).hexdigest()

        async def chunks():
            yield capture_bytes

        ingress = RecordingIngress(source.sources, storage, admissions,
                                  policy=capture_policy, inspect_file=synthetic_inspector)
        captured = await ingress.finalize(owner_id=self.owner, session_id=capture_a,
            recording_id=recording_id, upload=RecordingUpload(filename="synthetic-capture.mp4",
                description="Synthetic bytes with declared inspection; no physical capture or decode.",
                size_bytes=len(capture_bytes), content_sha256=capture_hash,
                client_report=RecordingClientReport(generation_id=uuid4(), timeline_end_ms=10000,
                    video_frames=1, source_audio_frames=0, padded_audio_frames=0)), chunks=chunks())
        self.assertEqual(captured["outcome"], "saved")
        original_ref = RetainedRef(file_id=UUID(captured["file_id"]),
            version_id=captured["version_id"], content_sha256=capture_hash)

        # Reuse the real existing producer/publication fixture path, not a new worker.
        context = TaskRunner(tasks, {}).context(fixture.claim)
        await WorkerContextFactory(source.pool)(context)
        evidence = TaskMemoryEvidence(context, source.sources, area_id=source.area)
        output_policy = FileTransferPolicy("mp4_output_v1", LIMIT)
        worker_files = SourceFileRepository(source.sources, storage, owner_id=self.owner,
            area_id=source.area, task_evidence=evidence, media_policy=output_policy,
            media_inspector=synthetic_inspector)
        foreground = SourceFileRepository(source.sources, storage, owner_id=self.owner,
            area_id=source.area, selections=selections, media_policy=output_policy,
            media_inspector=synthetic_inspector)
        access = WorkspaceFileAccess(foreground)
        writer = MediaPublicationStore(tasks, worker_files)
        reader = MediaPublicationStore(tasks, await access.scoped(self.owner, command_b))
        bridge = await TaskToolBridge.from_modules(context, (FileToolModule(worker_files, storage),),
            read_tools=("list_files", "get_file_metadata"), memory_evidence=evidence,
            receipt_revalidator=worker_files.revalidate_receipt)
        policy = ArtifactPolicy(formats=("json", "mp4"), max_files=2, max_file_bytes=LIMIT,
                                max_total_bytes=2 * LIMIT, mp4_max_file_bytes=LIMIT)
        async with tasks.transaction(self.owner) as connection:
            scope = await tasks.payload(connection, self.owner, policy.model_dump(mode="json"))
            parent = await connection.fetchval("""insert into jarvis_operation_grants
                (owner_id,capability,operation_key,request_event_id,request_revision,scope_revision,scope_payload_id,expires_at)
                values($1,'files.publish_generated',$2,$3,1,1,$4,clock_timestamp()+interval '1 hour') returning id""",
                self.owner, "artifact-task:" + str(self.task_id), fixture.event, scope)
            await connection.execute("insert into jarvis_task_grants(owner_id,task_id,grant_id) values($1,$2,$3)",
                                     self.owner, self.task_id, parent)
        publisher = GeneratedArtifactPublisher(bridge, worker_files, parent)
        # This existing helper explicitly supplies synthetic decode/readiness observations.
        plan = publication_fixture.MediaPublicationSQLTests.plan(
            self, original_ref, len(capture_bytes), "synthetic-original-edit")
        recipe_receipt = await writer.publish_recipe(plan, publisher, provider_call_id="fixture-recipe")
        self.assertEqual(recipe_receipt["status"], "success", recipe_receipt)
        output_path = root / "synthetic-output.mp4"
        output_bytes = b"Explicit synthetic edited output; not a rendered MP4."
        output_path.write_bytes(output_bytes)
        local = await snapshot_file(output_path, max_bytes=LIMIT)
        try:
            output_receipt = await writer.publish_output(plan, publisher, local, provider_call_id="fixture-output")
        finally:
            local.close()
        self.assertEqual(output_receipt["status"], "success", output_receipt)
        accepted = await reader.load_accepted_output(self.owner, self.task_id, plan.digest)
        saved_plan = await reader.load_plan(self.owner, self.task_id, plan.digest)
        self.assertIsNotNone(accepted)
        self.assertIsNotNone(saved_plan)
        self.assertEqual(accepted.file.content_sha256, hashlib.sha256(output_bytes).hexdigest())
        self.assertEqual(accepted.action_id, UUID(output_receipt["task_operation"]["action_id"]))
        self.assertEqual(saved_plan.action_id, UUID(recipe_receipt["task_operation"]["action_id"]))
        self.assertTrue(await tasks.finish(fixture.claim, WorkerOutcome(TaskStatus.SUCCEEDED,
            {"fixture": "Synthetic retained output published; no render or model execution."})))
        self.assertEqual(await reader.load_accepted_output(self.owner, self.task_id, plan.digest), accepted)

        registry = TaskWorkerRegistry(source.pool)
        try:
            service = await self.managed_service(fixture, registry, plan.binding, policy)
            agent = WorkspaceMediaAgent(service, access, admissions)
            arguments = {"instruction": "Keep the same cut and change the caption to Corrected fixture.",
                         "accepted_output": AcceptedOutputTarget.from_output(accepted).model_dump(mode="json")}
            command = self.command(command_b, "new-correction-command", "tasks.create", arguments)
            ctx = WorkspaceContext(self.owner, command_b, command.operation_id)
            result = await agent.submit(ctx, arguments, command_admission=command)
            self.assertEqual(result["outcome"], "accepted")
            self.assertFalse(result["effect_committed"])
            async with source.pool.acquire() as connection:
                request = await connection.fetchrow("select * from jarvis_task_requests where owner_id=$1 and request_id=$2",
                                                    self.owner, command.operation_id)
            self.assertEqual(request["operation"], "create")
            self.assertNotEqual(request["task_id"], self.task_id)
            task_b = await tasks.get(self.owner, request["task_id"])
            self.assertEqual(task_b.session_id, command_b)
            self.assertEqual(task_b.knowledge_area_id, source.area)

            # Real READ ONLY transaction plus real task-scoped snapshot. Any FOR
            # SHARE hidden in the accepted proof/source path makes this fail.
            async with tasks.read_transaction() as connection:
                _, pinned, instruction = await media_description(connection, task_b, lock=False)
                self.assertIsInstance(pinned, CorrectedMediaInput)
                self.assertEqual(await validate_media_input(connection, self.owner, command_b, pinned, lock=False), pinned)
                self.assertEqual(pinned.capture_session_id, capture_a)
                self.assertEqual(pinned.original.session_id, capture_a)
                self.assertEqual(pinned.command_session_id, command_b)
                self.assertEqual(pinned.original.recording_id, recording_id)
                self.assertEqual(pinned.original.file_id, original_ref.file_id)
                self.assertEqual(pinned.original.version_id, original_ref.version_id)
                self.assertEqual(pinned.original.content_sha256, capture_hash)
                self.assertEqual(pinned.edit_id, plan.recipe.recipe.edit_id)
                self.assertEqual(pinned.previous_recipe, saved_plan.file)
                self.assertEqual(pinned.target, AcceptedOutputTarget.from_output(accepted))
                event = await connection.fetchrow("select * from jarvis_context_events where owner_id=$1 and id=$2",
                                                   self.owner, instruction.event_id)
                self.assertEqual((event["source_namespace"], event["kind"], event["trust"], event["session_id"]),
                                 ("voice_command", "command_request", "derived", command_b))
                dependencies = await connection.fetch("select source_payload_id from jarvis_context_payload_dependencies "
                    "where owner_id=$1 and dependent_payload_id=$2", self.owner, event["payload_id"])
                self.assertTrue(set(pinned.dependencies).issubset({row["source_payload_id"] for row in dependencies}))
                grants = await connection.fetch("""select g.id,g.capability from jarvis_task_grants t
                    join jarvis_operation_grants g on g.owner_id=t.owner_id and g.id=t.grant_id
                    where t.owner_id=$1 and t.task_id=$2""", self.owner, task_b.id)
                self.assertEqual({row["capability"] for row in grants}, {"sandbox.execute", "files.publish_generated"})
                self.assertEqual(len(grants), 2)
            snapshot = await ContextRepository(tasks).snapshot(self.owner, command_b,
                max_messages=16, max_pins=8, task_id=task_b.id)
            self.assertTrue(any(record.id == instruction.event_id for record in snapshot["scope_sources"]))

            counts, uploads = await self.counts(source.pool), storage.uploads
            with patch.object(access, "scoped", AsyncMock(side_effect=AssertionError("replay cannot resolve files"))), \
                 patch.object(access, "media_metadata", AsyncMock(side_effect=AssertionError("replay cannot resolve metadata"))):
                self.assertEqual(await agent.submit(ctx, arguments, command_admission=command), result)
            self.assertEqual(await self.counts(source.pool), counts)
            self.assertEqual(storage.uploads, uploads)
            self.assertEqual(await reader.load_accepted_output(self.owner, self.task_id, plan.digest), accepted)
            self.assertEqual((await reader.load_plan(self.owner, self.task_id, plan.digest)).file, saved_plan.file)

            revised = plan.recipe.recipe.model_dump(mode="json")
            revised["previous_recipe"] = saved_plan.file.model_dump(mode="json")
            revised["captions"][0]["text"] = "Corrected fixture"
            async with tasks.transaction(self.owner) as connection:
                await tasks.lock_owner(connection, self.owner)
                for grant in grants:
                    await tasks.validate_grant(connection, self.owner, grant["id"])
                self.assertEqual(await require_media_recipe_input(tasks, connection, task_b, revised), pinned)
                self.assertEqual(await connection.fetchval("select session_id from jarvis_recordings "
                    "where owner_id=$1 and recording_id=$2", self.owner, recording_id), capture_a)
                self.assertEqual(await connection.fetchval("select count(*) from jarvis_sandbox_allocations where owner_id=$1",
                                                          self.owner), 0)

            # Withdraw an exact prior recipe through the real source owner. Its
            # output and correction authority must fail; the capture is independent.
            removed = await foreground.delete_source_file(owner_id=self.owner, file_id=saved_plan.file.file_id)
            self.assertTrue(removed["deleted"], removed)
            async with tasks.read_transaction() as connection:
                with self.assertRaises(TaskError):
                    await validate_media_input(connection, self.owner, command_b, pinned, lock=False)
            for grant in grants:
                with self.assertRaises(TaskError):
                    async with tasks.transaction(self.owner) as connection:
                        await tasks.validate_grant(connection, self.owner, grant["id"])
            with self.assertRaises((TaskError, SourceError, JarvisHTTPError)):
                await agent.submit(ctx, arguments, command_admission=command)
            fresh = self.command(command_b, "after-withdrawal-new-command", "tasks.create", arguments)
            with self.assertRaises((TaskError, SourceError, JarvisHTTPError)):
                await agent.submit(WorkspaceContext(self.owner, command_b, fresh.operation_id), arguments,
                                   command_admission=fresh)
            try:
                unavailable = await reader.load_accepted_output(self.owner, self.task_id, plan.digest)
            except (MediaRecipeError, SourceError, JarvisHTTPError):
                unavailable = None
            self.assertIsNone(unavailable)
            original, metadata = await access.media_metadata(self.owner, command_b,
                original_ref.file_id, original_ref.version_id)
            self.assertEqual(original.id, original_ref.file_id)
            self.assertEqual(metadata["content_sha256"], capture_hash)
            final = await self.counts(source.pool)
            self.assertEqual(final["tasks"], counts["tasks"])
            self.assertEqual(final["requests"], counts["requests"])
            self.assertEqual(final["actions"], counts["actions"])
            self.assertEqual(storage.uploads, uploads)
        finally:
            await registry.withdraw(self.owner, WORKER)

    async def managed_service(self, fixture, registry, renderer, policy):
        source, tasks = fixture.source, fixture.tasks
        baseline = CURATED_DEPLOYMENT_BASELINE_V1
        authority = PluginAuthority(PluginPolicyRepository(source.pool), ContextRepository(tasks), tasks,
                                    deployment_baseline=baseline)
        selected = next(item for item in baseline.policy.entries if item.plugin_id == "video_editor")
        bundle = MediaRendererBundle(configuration_id=selected.configuration_id,
            bundle_version="synthetic-correction-v1", bundle_sha256="a" * 64, renderer=renderer)
        now = datetime.now(UTC)
        fields = dict(plugin_id="video_editor", configuration_id=bundle.configuration_id,
            observed_at=now, expires_at=now + timedelta(minutes=10), bundle_sha256=bundle.bundle_sha256)
        evidence = MediaRendererEvidence(self.owner, renderer,
            InventoryObservation(**fields, state="installed", bundle_version=bundle.bundle_version),
            ReadinessObservation(**fields, state="ready"))

        async def observe(connection, owner, requested):
            self.assertTrue(connection.is_in_transaction())
            self.assertEqual((owner, requested), (self.owner, bundle))
            return evidence  # Explicit synthetic readiness, not installed-image evidence.

        runtime = ManagedPluginRuntime(authority, bundles=(bundle,), observe=observe)
        providers = {"frontier": ConfiguredTaskProvider("synthetic-correction-no-model")}
        await registry.advertise(worker_id=WORKER, owner_id=self.owner, host_id="synthetic",
            providers=providers, ttl_seconds=300, capabilities=("sandbox.execute",
                "files.publish_generated", "files.publish_generated_mp4", MEDIA_CAPABILITY))
        return TaskService(tasks, providers, registry=registry, managed_plugins=runtime,
            sandbox_configuration=SandboxConfiguration(),
            artifact_configuration=ArtifactConfiguration(area_id=source.area, policy=policy))

    async def counts(self, pool):
        async with pool.acquire() as connection:
            return dict(await connection.fetchrow("""select
                (select count(*) from jarvis_background_tasks where owner_id=$1) as tasks,
                (select count(*) from jarvis_task_requests where owner_id=$1) as requests,
                (select count(*) from jarvis_task_actions where owner_id=$1) as actions,
                (select count(*) from jarvis_context_events where owner_id=$1) as events,
                (select count(*) from jarvis_operation_grants where owner_id=$1) as grants,
                (select count(*) from jarvis_file_publications where owner_id=$1) as publications,
                (select count(*) from jarvis_files where owner_id=$1) as files""", self.owner))


if __name__ == "__main__":
    unittest.main()
