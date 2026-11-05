"""One Core-lane SQL case for recording -> English media command admission.

Real: isolated migrations, RecordingAdmission/Ingress, retained source/Files,
WorkspaceMediaAgent, TaskService, request receipts, grant guards and revocation.
Synthetic: byte stream, MP4 inspector, renderer inventory/readiness and worker
lease. No codec, model, provider, VM, native capture or memory-cloud operation.
The existing source_fixture owns manifested SQL cleanup; DiskStorage is local.
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

from app.files.generated import ArtifactPolicy, ArtifactRequest, ArtifactSnapshot, inspect_artifact
from app.files.recording_ingress import RecordingIngress, RecordingUpload
from app.files.transfer import FileTransferPolicy, MP4Inspection
from app.files.workspace_access import WorkspaceFileAccess
from app.knowledge.files import SourceFileRepository
from app.knowledge.session_areas import SessionKnowledgeAreas
from app.live_state.admission import CommandAdmission, command_parameters
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
from app.tasks.media_editing import MEDIA_TOOL, MediaInput
from app.tasks.models import TaskError, digest, json_object
from app.tasks.repository import TaskRepository
from app.tasks.service import TaskService
from app.workspace.media_agent import WorkspaceMediaAgent
from app.workspace.media_recipe import MediaRecipePolicy, TemplatePin
from app.workspace.media_render import RendererBinding
from app.workspace.models import WorkspaceContext
from evals.second_brain.source_fixture import source_fixture
from tests.test_file_backed_publication_sql import DiskStorage


@unittest.skipUnless(os.getenv("JARVIS_SECOND_BRAIN_TEST_DB_URL") and
                     os.getenv("JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"),
                     "Core must allocate the manifested SQL lane explicitly")
class LeanMediaAdmissionSQLTests(unittest.IsolatedAsyncioTestCase):
    async def test_recording_command_receipt_replay_and_grant_fences(self):
        manifest_dir = Path(os.environ["JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"])
        self.assertTrue(manifest_dir.is_absolute())
        async with asyncio.timeout(180):
            async with source_fixture(os.environ["JARVIS_SECOND_BRAIN_TEST_DB_URL"],
                    manifest_dir, test_name=self.id()) as fixture:
                with tempfile.TemporaryDirectory(prefix="jarvis-lean-media-sql-") as directory:
                    await self.exercise(fixture, Path(directory).resolve())

    async def exercise(self, fixture, root):
        tasks = TaskRepository(fixture.pool, TaskLimits())
        context = ContextRepository(tasks)
        async with fixture.pool.acquire() as connection:
            self.assertEqual(await connection.fetchval(
                "select name from jarvis_schema_migrations where version=49"), "media_command_grants")
            session = await connection.fetchval(
                "insert into jarvis_sessions(owner_id) values($1) returning id", fixture.owner)
        selections = SessionKnowledgeAreas(fixture.sources, tasks, fixture.area)
        recordings = RecordingAdmission(fixture.sources, tasks, selections, owner_id=fixture.owner)
        policy = FileTransferPolicy("mp4_capture_v1", 1024)
        storage = DiskStorage(root / "objects")
        storage.lose_ack = False
        inspected = []

        async def synthetic_inspector(local, admitted_policy):
            # Deliberately not MP4/decode qualification. Real retention receives
            # exact synthetic bytes, with the inspector as the only substitution.
            self.assertEqual(admitted_policy.max_bytes, policy.max_bytes)
            inspected.append(local)
            self.addCleanup(local.close)
            return MP4Inspection(size_bytes=local.size_bytes, content_sha256=local.content_sha256,
                duration_ms=1000, width=320, height=240, video_codec="h264", audio_codec="aac")

        ingress = RecordingIngress(fixture.sources, storage, recordings, policy=policy,
                                   inspect_file=synthetic_inspector)

        def carrier(tool, capability, arguments, call):
            # Trusted server construction with explicitly synthetic provider IDs.
            return CommandAdmission(owner_id=fixture.owner, session_id=session,
                provider_session_key=" synthetic SQL provider session ", provider_call_id=call,
                tool_name=tool, capability=capability, normalized_arguments=arguments)

        async def retained(label):
            data = ("Synthetic non-MP4 recording bytes: " + label).encode("ascii")
            start = carrier("live_recording", "recordings.start", {"include_jarvis_audio": True}, "start-" + label)
            admitted = await recordings.admit(start, area_id=fixture.area, policy=policy)
            await recordings.stop(carrier("live_recording", "recordings.stop",
                {"recording_id": str(admitted.scope.recording_id)}, "stop-" + label))
            upload = RecordingUpload(filename=label + ".mp4", description="Synthetic SQL fixture " + label,
                size_bytes=len(data), content_sha256=hashlib.sha256(data).hexdigest(),
                client_report={"generation_id": str(uuid4()), "timeline_end_ms": 1000,
                               "video_frames": 24, "source_audio_frames": 48000, "padded_audio_frames": 0})

            async def chunks():
                yield data

            saved = await ingress.finalize(owner_id=fixture.owner, session_id=session,
                recording_id=admitted.scope.recording_id, upload=upload, chunks=chunks())
            self.assertEqual(saved["outcome"], "saved")
            return admitted.scope, saved, upload

        original, saved, upload = await retained("selected")
        _, other_saved, _ = await retained("other")
        self.assertEqual(storage.uploads, 2)
        files = SourceFileRepository(fixture.sources, storage, owner_id=fixture.owner, area_id=fixture.area,
            selections=selections, media_policy=policy, media_inspector=synthetic_inspector)
        access = WorkspaceFileAccess(files)

        # Reuse the same explicit baseline/observer composition as the existing
        # managed-media SQL fixture. No installed evidence is inferred from it.
        baseline = CURATED_DEPLOYMENT_BASELINE_V1
        choice = next(entry for entry in baseline.policy.entries if entry.plugin_id == "video_editor")
        authority = PluginAuthority(PluginPolicyRepository(fixture.pool), context, tasks,
                                    deployment_baseline=baseline)
        renderer = RendererBinding(
            template=TemplatePin(id="jarvis.clip.v1", version="synthetic-v1", bundle_sha256="b" * 64),
            physical_template_id="synthetic-admission-only", build_id="synthetic-no-build",
            software_sha256="c" * 64, build_receipt_sha256="d" * 64,
            policy=MediaRecipePolicy(max_source_bytes=1024, max_source_duration_ms=1000, max_output_frames=24),
            max_output_bytes=1024)
        bundle = MediaRendererBundle(configuration_id=choice.configuration_id, bundle_version="synthetic-v1",
                                     bundle_sha256="a" * 64, renderer=renderer)
        observed = datetime.now(UTC)
        fields = dict(plugin_id="video_editor", configuration_id=bundle.configuration_id,
            observed_at=observed, expires_at=observed + timedelta(minutes=10), bundle_sha256=bundle.bundle_sha256)
        evidence = MediaRendererEvidence(fixture.owner, renderer,
            InventoryObservation(**fields, state="installed", bundle_version=bundle.bundle_version),
            ReadinessObservation(**fields, state="ready"))

        async def observe(connection, owner, selected):
            self.assertTrue(connection.is_in_transaction())
            self.assertEqual((owner, selected), (fixture.owner, bundle))
            return evidence  # Explicit fixture observation, not deployment truth.

        managed = ManagedPluginRuntime(authority, bundles=(bundle,), observe=observe)
        registry = TaskWorkerRegistry(fixture.pool)
        providers = {"frontier": ConfiguredTaskProvider("synthetic-no-model-call")}
        service = TaskService(tasks, providers, registry=registry, sandbox_configuration=SandboxConfiguration(),
            artifact_configuration=ArtifactConfiguration(area_id=fixture.area,
                policy=ArtifactPolicy(formats=("json", "mp4"), mp4_max_file_bytes=1024)), managed_plugins=managed)
        agent = WorkspaceMediaAgent(service, access, recordings)
        worker_id = "lean-media-sql-" + uuid4().hex
        await registry.advertise(worker_id=worker_id, owner_id=fixture.owner, host_id="synthetic",
            providers=providers, ttl_seconds=300,
            capabilities=("sandbox.execute", "files.publish_generated", "files.publish_generated_mp4", MEDIA_CAPABILITY))
        try:
            ref_keys = ("file_id", "source_id", "version_id", "receipt_event_id")
            args = {"instruction": "Keep the first half second of this recording and preserve its sound.",
                    "recording_id": str(original.recording_id), "saved": {key: saved[key] for key in ref_keys}}
            command = carrier(MEDIA_TOOL, "tasks.create", args, " edit-call-preserved ")
            ctx = WorkspaceContext(fixture.owner, session, command.operation_id)
            accepted = await agent.submit(ctx, args, command_admission=command)
            self.assertEqual(accepted["outcome"], "accepted")
            self.assertIs(accepted["effect_committed"], False)

            async with tasks.transaction(fixture.owner) as connection:
                receipt = await connection.fetchrow("select * from jarvis_task_requests where owner_id=$1 and request_id=$2",
                                                     fixture.owner, command.operation_id)
                self.assertEqual((receipt["operation"], receipt["accepted_revision"]), ("create", 1))
                self.assertEqual(receipt["parameters_digest"], digest(command_parameters(command)))
                task = await tasks.load(connection, fixture.owner, receipt["task_id"], lock="share")
                record = await context._record(connection, fixture.owner, task.request_event_id)
                self.assertEqual((record.source_namespace, record.kind, record.trust),
                                 ("voice_command", "command_request", "derived"))
                self.assertFalse(record.authorizes_request)
                self.assertEqual(record.content["command"], command.parameters())
                pin = MediaInput.model_validate(record.content["media_input"])
                self.assertEqual((pin.owner_id, pin.session_id, pin.area_id, pin.recording_id),
                                 (fixture.owner, session, fixture.area, original.recording_id))
                self.assertEqual((pin.file_id, pin.source_id, pin.version_id, pin.receipt_event_id),
                                 (UUID(saved["file_id"]), UUID(saved["source_id"]), saved["version_id"], UUID(saved["receipt_event_id"])))
                self.assertEqual((pin.content_sha256, pin.size_bytes), (upload.content_sha256, upload.size_bytes))
                description = json_object(await connection.fetchval(
                    "select content from jarvis_context_payloads where owner_id=$1 and id=$2",
                    fixture.owner, task.description_payload_id))
                self.assertEqual(description["media_input"], pin.model_dump(mode="json"))
                self.assertEqual(description["requested_outcome"], args["instruction"])
                self.assertEqual(description["managed_plugin"]["bundle"], bundle.model_dump(mode="json"))
                dependencies = await connection.fetch("""select source_payload_id from jarvis_context_payload_dependencies
                    where owner_id=$1 and dependent_payload_id=$2""", fixture.owner, record.payload_id)
                self.assertTrue(set(pin.dependencies) <= {row["source_payload_id"] for row in dependencies})
                ids = await connection.fetch("select grant_id from jarvis_task_grants where owner_id=$1 and task_id=$2",
                                             fixture.owner, task.id)
                grants = [await tasks.validate_grant(connection, fixture.owner, row["grant_id"]) for row in ids]
                self.assertEqual({row["capability"] for row in grants}, {"sandbox.execute", "files.publish_generated"})
                self.assertEqual(len(grants), 2)
                parents = {row["capability"]: row for row in grants}
                self.assertEqual(parents["sandbox.execute"]["operation_key"], "managed-video-editor:" + str(task.id))
                allowance = ArtifactPolicy.model_validate(json_object(parents["files.publish_generated"]["scope"]))
                self.assertEqual((allowance.formats, allowance.max_files), (("json", "mp4"), 2))

                # Exercise the legitimate child guard, without publishing bytes
                # or fabricating any task action/render/output receipt.
                evidence_payload = await tasks.payload(connection, fixture.owner,
                    {"fixture": "synthetic candidate evidence; no render or publication"}, source_payload_ids=(pin.source_payload_id,))
                candidate = ArtifactRequest(operation_key="synthetic-recipe", filename="recipe.json",
                                            format="json", description="Synthetic scope-only candidate")
                snapshot = ArtifactSnapshot(request=candidate, inspection=inspect_artifact(candidate, b"{}", allowance),
                    task_id=task.id, parent_grant_id=parents["files.publish_generated"]["id"], evidence_payload_id=evidence_payload)
                snapshot_key = "artifact-snapshot:" + task.id.hex + ":" + digest({"operation_key": candidate.operation_key})

                async def insert_grant(capability, key, scope, expires):
                    payload = await tasks.payload(connection, fixture.owner, scope, source_payload_ids=(record.payload_id,))
                    return await connection.fetchval("""insert into jarvis_operation_grants
                        (owner_id,capability,operation_key,request_event_id,request_revision,scope_revision,scope_payload_id,expires_at)
                        values($1,$2,$3,$4,1,1,$5,$6) returning id""",
                        fixture.owner, capability, key, record.id, payload, expires)

                expires = parents["files.publish_generated"]["expires_at"]
                child_id = await insert_grant("files.publish_snapshot", snapshot_key, snapshot.model_dump(mode="json"), expires)
                await connection.execute("insert into jarvis_task_grants(owner_id,task_id,grant_id) values($1,$2,$3)",
                                         fixture.owner, task.id, child_id)
                self.assertEqual((await tasks.validate_grant(connection, fixture.owner, child_id))["capability"], "files.publish_snapshot")
                sandbox_scope = json_object(parents["sandbox.execute"]["scope"])
                cases = [(capability, "forbidden:" + uuid4().hex, {}) for capability in
                         ("files.create_text", "memory.correct_sources", "notes.edit_account")]
                cases += [("sandbox.execute", parents["sandbox.execute"]["operation_key"], dict(sandbox_scope, internet_access=True)),
                          ("sandbox.execute", parents["sandbox.execute"]["operation_key"], dict(sandbox_scope, template="desktop")),
                          ("files.publish_generated", parents["files.publish_generated"]["operation_key"],
                           allowance.model_dump(mode="json") | {"formats": ["json", "mp4", "text"]}),
                          ("files.publish_snapshot", snapshot_key,
                           snapshot.model_dump(mode="json") | {"task_id": str(uuid4())}),
                          ("files.publish_snapshot", snapshot_key,
                           snapshot.model_dump(mode="json") | {"parent_grant_id": str(uuid4())})]
                for capability, key, scope in cases:
                    with self.subTest(denied_capability=capability, scope=scope), self.assertRaises(asyncpg.CheckViolationError):
                        async with connection.transaction():  # Each denied scope rolls its payload back too.
                            await insert_grant(capability, key, scope, expires)

            # The normal worker builds a task-scoped canonical snapshot in a
            # READ ONLY transaction. This must retain source checks without
            # attempting the mutation path's FOR SHARE locks.
            snapshot = await context.snapshot(fixture.owner, session,
                max_messages=64, max_pins=16, task_id=task.id)
            self.assertGreaterEqual(snapshot["watermark"], record.sequence)

            async def counts():
                async with fixture.pool.acquire() as connection:
                    return dict(await connection.fetchrow("""select
                        (select count(*) from jarvis_background_tasks where owner_id=$1) as tasks,
                        (select count(*) from jarvis_task_requests where owner_id=$1) as requests,
                        (select count(*) from jarvis_context_events where owner_id=$1
                            and source_namespace='voice_command' and source_event_id=$2) as commands,
                        (select count(*) from jarvis_operation_grants where owner_id=$1) as grants,
                        (select count(*) from jarvis_task_actions where owner_id=$1) as actions,
                        (select count(*) from jarvis_sandbox_allocations where owner_id=$1) as allocations,
                        (select count(*) from jarvis_file_publications where owner_id=$1) as publications""",
                        fixture.owner, str(command.operation_id)))

            before = await counts()
            self.assertEqual(before, dict(tasks=1, requests=1, commands=1, grants=3, actions=0, allocations=0, publications=2))
            with patch.object(access, "media_metadata", side_effect=AssertionError("Replay resolved input again")), \
                    patch.object(managed, "admit", side_effect=AssertionError("Replay selected a new managed job")):
                replay = await agent.submit(ctx, args, command_admission=command)
            self.assertEqual(replay, accepted)
            self.assertEqual(await counts(), before)
            self.assertEqual(storage.uploads, 2)

            wrong = dict(args, saved={key: other_saved[key] for key in ref_keys})
            wrong_command = carrier(MEDIA_TOOL, "tasks.create", wrong, "wrong-retained-file")
            with self.assertRaises(TaskError) as mismatch:
                await agent.submit(WorkspaceContext(fixture.owner, session, wrong_command.operation_id),
                                   wrong, command_admission=wrong_command)
            self.assertEqual(mismatch.exception.code, "media_recording_reference_changed")
            self.assertEqual(await counts(), before)
            self.assertEqual(storage.uploads, 2)

            # Actual owned source/publication withdrawal, not a mocked validator.
            await files.for_session(session).delete_source_file(owner_id=fixture.owner, file_id=pin.file_id)
            async with tasks.transaction(fixture.owner) as connection:
                for identity in [row["id"] for row in grants] + [child_id]:
                    with self.assertRaises(TaskError) as unavailable:
                        await tasks.validate_grant(connection, fixture.owner, identity)
                    self.assertEqual(unavailable.exception.code, "task_grant_unavailable")
            with self.assertRaises(TaskError):
                await agent.submit(ctx, args, command_admission=command)
            self.assertEqual(await counts(), before)
            self.assertEqual(storage.uploads, 2)
            self.assertTrue(all(not local.path.exists() for local in inspected))
        finally:
            await registry.withdraw(fixture.owner, worker_id)
