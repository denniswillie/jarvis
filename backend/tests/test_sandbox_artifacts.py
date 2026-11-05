"""Optional publication admission and the native output/receipt boundary."""
import asyncio
from datetime import UTC, datetime, timedelta
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from uuid import uuid4
from unittest.mock import patch

from pydantic import ValidationError

from app.files.generated import ArtifactPolicy
from app.knowledge.files import SourceFileRepository
from app.runtime.feature_config import ArtifactConfiguration, Capability, SandboxConfiguration, SecondBrainSettings, TaskLimits
from app.sandbox.models import SandboxError, SandboxInfo
from app.tasks.catalog import ConfiguredTaskProvider, TaskWorkerRegistry
from app.tasks.models import ClaimLost, TaskCreate, TaskError
from app.tasks.repository import TaskRepository
from app.tasks.runner import TaskRunner
from app.tasks.service import TaskService
from app.tasks.worker import WorkerContextFactory
from evals.second_brain.file_storage_fixture import MemoryStorage
from evals.second_brain.source_fixture import source_fixture


class OutputProvider:
    def __init__(self):
        self.resources, self.files, self.reads = {}, {}, 0
        self.after_read = None

    async def create(self, scope, policy):
        handle = SandboxInfo(sandbox_id="fixture-" + scope.allocation_id.hex, scope=scope, policy=policy,
            state="running", expires_at=datetime.now(UTC) + timedelta(minutes=5), template_id="fixture")
        self.resources[handle.sandbox_id] = handle
        return handle

    async def create_bound(self, scope, policy, *, template_id):
        if template_id != "fixture":
            raise SandboxError("sandbox_template_mismatch")
        return await self.create(scope, policy)

    async def prepare(self, handle):
        return handle

    async def read_workspace_file(self, handle, path, *, max_bytes):
        self.reads += 1
        if handle.sandbox_id not in self.resources or path not in self.files:
            raise SandboxError("sandbox_not_found")
        value = self.files[path]
        if len(value) > max_bytes:
            raise SandboxError("sandbox_output_limit")
        if self.after_read is not None:
            await self.after_read()
        return value


class ArtifactConfigurationTests(unittest.TestCase):
    def test_disabled_configuration_is_lazy_and_separate(self):
        flags = SecondBrainSettings(enabled="true", artifacts_enabled="false", artifact_max_files="bad")
        self.assertFalse(flags.resolve(Capability.ARTIFACTS).configured)
        flags = SecondBrainSettings(enabled="true", artifacts_enabled="true", sandbox_enabled="true")
        self.assertEqual(flags.resolve(Capability.ARTIFACTS, database_url="postgresql://configured").issue,
                         "artifact_dependencies_unavailable")
        flags = SecondBrainSettings(enabled="true", artifacts_enabled="true", sandbox_enabled="true",
            knowledge_enabled="true", hindsight_api_key="fixture-configured-key",
            hindsight_url="https://memory.example.com", knowledge_area_id=str(uuid4()))
        result = flags.resolve(Capability.ARTIFACTS, database_url="postgresql://configured")
        self.assertTrue(result.configured, result)
        self.assertEqual(result.configuration.policy.max_files, 8)

    def test_publish_scope_and_baseline_fingerprints(self):
        base = TaskCreate(request_id=uuid4(), title="Synthetic", requested_outcome="Reason only.")
        values = TaskRepository._create_parameters(base)
        self.assertNotIn("publish_artifacts", values)
        self.assertNotIn("execution_environment", values)
        for change in ({"publish_artifacts": True}, {"publish_artifacts": True, "execution_environment": "offline_sandbox"}):
            with self.assertRaises(ValidationError):
                TaskCreate(**(base.model_dump() | change))

    def test_worker_factory_does_not_accept_incomplete_artifact_dependencies(self):
        with self.assertRaises(ValueError):
            WorkerContextFactory(object(), artifact_configuration=ArtifactConfiguration(area_id=uuid4()))




class ArtifactErrorBoundaryTests(unittest.IsolatedAsyncioTestCase):
    async def test_internal_validation_is_not_reported_as_pre_submission_input_failure(self):
        from unittest.mock import AsyncMock
        from app.files.generated import ArtifactRequest
        from app.sandbox.artifacts import SandboxArtifactTools
        from app.sandbox.tools import SandboxToolBridge
        service = object.__new__(SandboxArtifactTools)
        service.context = SimpleNamespace(claim=object(), reserve=AsyncMock(return_value={"id": uuid4()}),
            budgets=SimpleNamespace(submit=AsyncMock()), settle=AsyncMock())
        with self.assertRaises(ValidationError) as invalid:
            ArtifactRequest.model_validate({})
        service.publish = AsyncMock(side_effect=invalid.exception)
        outer = object.__new__(SandboxToolBridge)
        outer.base = SimpleNamespace(_check=AsyncMock())
        outer.inputs = SimpleNamespace(revalidate_staged=AsyncMock())
        outer.artifacts = service
        outer.artifact_names = frozenset({"sandbox_publish_file"})
        outer.input_names = frozenset()
        outer.handlers = {"sandbox_publish_file": object()}
        arguments = {"operation_key": "report", "path": "report.json", "filename": "report.json",
                     "format": "json", "description": "Synthetic report."}
        result = await outer.invoke("sandbox_publish_file", arguments, provider_call_id="fixture-valid")
        self.assertEqual(result["error"]["code"], "sandbox_internal_validation")
        self.assertNotIn("no new file operation", result["error"]["message"].lower())
        service.publish.reset_mock()
        result = await outer.invoke("sandbox_publish_file", {}, provider_call_id="fixture-invalid")
        self.assertEqual(result["error"]["code"], "invalid_tool_input")
        service.publish.assert_not_awaited()

@unittest.skipUnless(os.getenv("JARVIS_SECOND_BRAIN_TEST_DB_URL") and
                     os.getenv("JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"), "isolated database fixture required")
class SandboxArtifactDatabaseTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.s = await self.enterAsyncContext(source_fixture(os.environ["JARVIS_SECOND_BRAIN_TEST_DB_URL"],
            Path(os.environ["JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"]), test_name=self.id()))
        self.tasks = TaskRepository(self.s.pool, TaskLimits(lease_seconds=300))
        self.registry = TaskWorkerRegistry(self.s.pool)
        self.providers = {"frontier": ConfiguredTaskProvider("fixture-model")}
        self.configuration = ArtifactConfiguration(area_id=self.s.area)
        self.service = TaskService(self.tasks, self.providers, registry=self.registry,
            sandbox_configuration=SandboxConfiguration(), artifact_configuration=self.configuration)
        async with self.s.pool.acquire() as connection:
            self.session = await connection.fetchval("insert into jarvis_sessions(owner_id) values($1) returning id", self.s.owner)
        self.storage, self.provider = MemoryStorage(), OutputProvider()
        self.knowledge = SimpleNamespace(sources=self.s.sources, retrieval=SimpleNamespace(sources=self.s.sources),
            configuration=SimpleNamespace(area_id=self.s.area), context_for=lambda owner: None)

    async def advertise(self, publication=True):
        await self.registry.advertise(worker_id="fixture", owner_id=self.s.owner, host_id="fixture-host",
            providers=self.providers, ttl_seconds=300,
            capabilities=("sandbox.execute", "files.publish_generated") if publication else ("sandbox.execute",))

    def request(self, publish=True):
        return TaskCreate(request_id=uuid4(), title="Synthetic report", requested_outcome="Create a synthetic JSON report.",
            session_id=self.session, execution_environment="offline_sandbox", publish_artifacts=publish)

    async def setup_task(self, publish=True):
        await self.advertise()
        self.task = await self.service.create(self.s.owner, self.request(publish))
        self.claim = await self.tasks.claim_next("fixture", ("frontier",),
            worker_capabilities=("sandbox.execute", "files.publish_generated"))
        self.context = TaskRunner(self.tasks, {}).context(self.claim)
        prepared = await WorkerContextFactory(self.s.pool, knowledge=self.knowledge,
            file_storage=self.storage, sandbox_provider=self.provider,
            artifact_configuration=self.configuration)(self.context)
        self.bridge = prepared.tools
        return self.task

    async def invoke(self, name, arguments):
        return await self.bridge.invoke(name, arguments, provider_call_id="fixture-" + uuid4().hex)

    def arguments(self):
        return {"operation_key": "summary", "path": "summary.json", "filename": "summary.json",
                "format": "json", "description": "Synthetic computed report."}

    async def workspace(self):
        self.allocation = await self.bridge.broker._ready(await self.bridge.broker.allocation())
        self.provider.files["summary.json"] = b'{"total":10}\n'

    async def test_capable_worker_and_explicit_publication_are_required(self):
        await self.advertise(False)
        with self.assertRaises(TaskError):
            await self.service.create(self.s.owner, self.request())
        async with self.s.pool.acquire() as connection:
            self.assertEqual(await connection.fetchval("select count(*) from jarvis_background_tasks"), 0)
        await self.setup_task(False)
        names = {item["name"] for item in self.bridge.catalog()}
        self.assertNotIn("sandbox_publish_file", names)
        result = await self.invoke("sandbox_publish_file", self.arguments())
        self.assertEqual(result["status"], "failed")
        self.assertEqual(self.storage.uploads, 0)

    async def test_area_and_request_replay_are_stable_and_claims_are_capability_filtered(self):
        await self.advertise()
        request = self.request()
        task = await self.service.create(self.s.owner, request)
        self.assertEqual(task.knowledge_area_id, self.s.area)
        self.service.artifact_configuration = None
        replay = await self.service.create(self.s.owner, request)
        self.assertEqual(replay.id, task.id)
        self.assertIsNone(await self.tasks.claim_next("incapable", ("frontier",), worker_capabilities=("sandbox.execute",)))
        capable = await self.tasks.claim_next("capable", ("frontier",), worker_capabilities=("sandbox.execute", "files.publish_generated"))
        self.assertEqual(capable.task.id, task.id)
        async with self.s.pool.acquire() as connection:
            self.assertEqual(await connection.fetchval("select count(*) from jarvis_task_grants"), 2)

    async def test_missing_workspace_and_invalid_paths_do_not_allocate_or_upload(self):
        await self.setup_task()
        for arguments in (self.arguments(), self.arguments() | {"path": "/etc/passwd"}):
            result = await self.invoke("sandbox_publish_file", arguments)
            self.assertEqual(result["status"], "failed", result)
        self.assertEqual(self.provider.resources, {})
        self.assertEqual(self.storage.uploads, 0)
        async with self.s.pool.acquire() as connection:
            self.assertEqual(await connection.fetchval("select count(*) from jarvis_sandbox_allocations"), 0)

    async def test_exact_publish_replays_after_vm_loss_and_rehydrates_references(self):
        await self.setup_task()
        await self.workspace()
        first = await self.invoke("sandbox_publish_file", self.arguments())
        self.assertEqual(first["status"], "success", first)
        self.provider.resources.clear()
        await self.bridge.broker.allocations.request_cleanup(self.s.owner, self.allocation["id"])
        second = await self.invoke("sandbox_publish_file", self.arguments())
        self.assertEqual(second["file"], first["file"])
        status = await self.invoke("sandbox_artifact_status", {"operation_key": "summary"})
        self.assertEqual(status["file"], first["file"])
        self.assertEqual(self.provider.reads, 1)
        self.assertEqual(self.storage.uploads, 1)
        prepared = await WorkerContextFactory(self.s.pool, knowledge=self.knowledge,
            file_storage=self.storage, sandbox_provider=self.provider,
            artifact_configuration=self.configuration)(self.context)
        verified = await prepared.tools.verified_operation(first["task_operation"]["operation_ref"])
        self.assertIsNotNone(verified)
        self.assertFalse(any(item["tool"] == "publish_generated_file" for item in prepared.tools.authorized_operations()))

    async def test_status_listing_and_unknown_upload_recovery_need_no_vm_read(self):
        await self.setup_task()
        empty = await self.invoke("sandbox_artifact_status", {"operation_key": None})
        self.assertEqual(empty["artifact_operations"], [])
        await self.workspace()
        self.storage.mode = "fail_after"
        with self.assertRaises(RuntimeError):
            await self.invoke("sandbox_publish_file", self.arguments())
        self.storage.mode = None
        self.provider.resources.clear()
        status = await self.invoke("sandbox_artifact_status", {"operation_key": "summary"})
        self.assertEqual(status["status"], "success", status)
        listing = await self.invoke("sandbox_artifact_status", {"operation_key": None})
        self.assertEqual(listing["artifact_operations"][0]["state"], "accepted")
        self.assertEqual(self.provider.reads, 1)
        self.assertEqual(self.storage.uploads, 1)

    async def test_unsubmitted_snapshot_reuses_exact_workspace_bytes_without_a_second_identity(self):
        from app.files.generated import ArtifactRequest
        await self.setup_task()
        await self.workspace()
        request = ArtifactRequest.model_validate({key: value for key, value in self.arguments().items() if key != "path"})
        action, _ = await self.bridge.artifacts.publisher.prepare(request, self.provider.files["summary.json"])
        status = await self.invoke("sandbox_artifact_status", {"operation_key": "summary"})
        self.assertEqual(status["artifact_operation"]["state"], "not_submitted")
        saved = await self.invoke("sandbox_publish_file", self.arguments())
        self.assertEqual(saved["status"], "success", saved)
        self.assertEqual(saved["task_operation"]["operation_ref"], action.reference)
        self.assertEqual(self.storage.uploads, 1)
        self.assertEqual(self.provider.reads, 1)

    async def test_invalid_delivery_type_is_a_sanitized_failure_before_publication(self):
        await self.setup_task()
        await self.workspace()
        self.provider.files["summary.json"] = b'{"private synthetic invalid'
        result = await self.invoke("sandbox_publish_file", self.arguments() | {"filename": "report.html"})
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["error"]["code"], "artifact_publication_invalid")
        self.assertNotIn("private synthetic", str(result))
        self.assertEqual(self.storage.uploads, 0)

    async def test_task_source_revocation_during_read_fences_upload(self):
        await self.setup_task()
        await self.workspace()
        async def revoke():
            async with self.s.pool.acquire() as connection:
                await connection.execute("""update jarvis_context_payloads set availability='deleted',content=null
                    where id=(select payload_id from jarvis_context_events where id=$1)""", self.task.request_event_id)
        self.provider.after_read = revoke
        with self.assertRaises(ClaimLost):
            await self.invoke("sandbox_publish_file", self.arguments())
        self.assertEqual(self.storage.uploads, 0)
        async with self.s.pool.acquire() as connection:
            self.assertEqual(await connection.fetchval("select phase from jarvis_sandbox_allocations"), "cleanup_requested")
