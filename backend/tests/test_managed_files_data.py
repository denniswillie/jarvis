"""Offline contracts only: substituted SQL/observer/provider, no image qualification."""
import asyncio
from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from app.plugins.authority import ManagedPluginBinding
from app.plugins.managed import (
    CAPABILITY, GRANT_PREFIX, FilesDataBundle, FilesDataEvidence, ManagedFilesDataJob,
    ManagedFilesDataRuntime, load_managed_task, submission_receipt,
)
from app.plugins.models import InventoryObservation, ReadinessObservation
from app.plugins.task_tools import FilesDataTaskToolModule
from app.runtime.feature_config import TaskLimits
from app.sandbox.broker import SandboxCommandBroker
from app.sandbox.models import CommandRequest, SandboxError, SandboxPolicy
from app.sandbox.tools import SandboxToolBridge
from app.tasks.actions import TaskActionRepository
from app.tasks.generated_artifacts import GeneratedArtifactPublisher, PUBLISH_TOOL
from app.tasks.models import TaskCreate, TaskError, TaskMutation, TaskStatus, digest
from app.tasks.repository import TaskRepository
from app.tasks.service import TaskService
from app.tasks.tool_bridge import TaskToolBridge
from app.tools.registry import JarvisToolRegistry


NOW = datetime(2026, 11, 2, tzinfo=UTC)
OWNER = "managed-fixture-owner"


def fixture_scope():
    binding = ManagedPluginBinding(OWNER, "files_data", uuid4(), 2)
    bundle = FilesDataBundle(configuration_id=binding.configuration_id, bundle_version="fixture-v1",
        bundle_sha256="a" * 64, sandbox_template_id="fixture-image-not-qualified",
        executable="/usr/bin/python3")
    fields = dict(plugin_id="files_data", configuration_id=binding.configuration_id,
        observed_at=NOW - timedelta(seconds=1), expires_at=NOW + timedelta(seconds=60), bundle_sha256="a" * 64)
    evidence = FilesDataEvidence(OWNER, bundle.sandbox_template_id, bundle.executable,
        InventoryObservation(**fields, state="installed", bundle_version="fixture-v1"),
        ReadinessObservation(**fields, state="ready"))
    return ManagedFilesDataJob(binding, bundle), evidence


class Connection:
    """Small SQL-protocol substitute, deliberately not SQL acceptance evidence."""
    def __init__(self, tasks):
        self.tasks = tasks
        self.payloads, self.grants, self.actions = {}, {}, {}
        self.active = False

    def is_in_transaction(self):
        return self.active

    async def fetch(self, sql, *args):
        if "managed-files-data:%" in sql:
            return [dict(id=key, operation_key=row["operation_key"]) for key, row in self.grants.items()
                    if row["operation_key"].startswith(GRANT_PREFIX)]
        if "a.capability='sandbox.execute'" in sql:
            return [{"grant_id": key} for key, row in self.grants.items() if row["capability"] == "sandbox.execute"]
        raise AssertionError(sql)

    async def fetchrow(self, sql, *args):
        if "insert into jarvis_task_actions" in sql:
            owner, task, revision, operation, grant, scope_revision, payload, fingerprint = args
            row = dict(id=uuid4(), owner_id=owner, task_id=task, task_revision=revision,
                operation_key=operation, grant_id=grant, scope_revision=scope_revision,
                parameters_payload_id=payload, parameters_digest=fingerprint,
                status="prepared", submission_token=None, receipt_payload_id=None, receipt_digest=None)
            self.actions[row["id"]] = row
            return dict(row)
        if "from jarvis_task_actions" in sql:
            row = next((item for item in self.actions.values() if
                (item["operation_key"] == args[2] if "operation_key=$3" in sql else item["id"] == args[2])), None)
            if row is None:
                return None
            value = dict(row)
            if "as receipt" in sql:
                payload = self.payloads.get(row["receipt_payload_id"], {})
                value.update(receipt=payload.get("content"), receipt_availability=payload.get("availability"))
            return value
        if "from jarvis_context_payloads" in sql:
            return self.payloads.get(args[1])
        raise AssertionError(sql)

    async def fetchval(self, sql, *args):
        if "select exists(select 1 from jarvis_task_grants" in sql:
            return (args[0], args[1], args[2]) in self.tasks.attached
        if "insert into jarvis_background_tasks" in sql:
            self.tasks.task.description_payload_id = args[5]
            return self.tasks.task.id
        if "insert into jarvis_operation_grants" in sql:
            owner, capability, operation_key, event, revision, payload, _seconds = args
            identity = uuid4()
            self.grants[identity] = dict(id=identity, capability=capability, operation_key=operation_key,
                request_event_id=event, request_revision=revision, scope_revision=1, scope_payload_id=payload)
            return identity
        if "select capability from jarvis_operation_grants" in sql:
            return self.grants[args[1]]["capability"]
        if "lock_jarvis_copy_source" in sql:
            return self.payloads[args[1]]["availability"]
        if "select availability='available'" in sql:
            return self.payloads[args[1]]["availability"] == "available"
        if "select exists(select 1 from jarvis_task_actions" in sql:
            rows = list(self.actions.values())
            if "submission_token is not null" in sql:
                return any(row["submission_token"] is not None for row in rows)
            if "status in ('submitting','outcome_unknown')" in sql:
                return any(row["status"] in {"submitting", "outcome_unknown"} for row in rows)
            return any(row["grant_id"] == args[1] for row in rows)
        if "jarvis_sessions" in sql or "jarvis_knowledge_areas" in sql:
            return True
        raise AssertionError(sql)

    async def execute(self, sql, *args):
        if "insert into jarvis_task_grants" in sql:
            self.tasks.attached.add(tuple(args))
        elif "update jarvis_task_actions" in sql:
            if "status='revoked'" in sql:
                for row in self.actions.values():
                    if row["status"] in {"prepared", "authorized"}:
                        row["status"] = "revoked"
                return
            row = self.actions[args[1]]
            if "status='authorized'" in sql:
                row["status"] = "authorized"
            elif "status='submitting'" in sql:
                row.update(status="submitting", submission_token=args[2])
            elif "status=$3,receipt_payload_id" in sql:
                row.update(status=args[2], receipt_payload_id=args[3], receipt_digest=args[4])
            else:
                raise AssertionError(sql)
        elif "update jarvis_background_tasks set revision=revision+1" in sql:
            task = self.tasks.task
            task.revision += 1
            task.fence += 1
            task.status, task.pending_status = TaskStatus(args[2]), args[3]
            task.description_payload_id, task.request_event_id, task.request_revision = args[4:7]
            task.context_watermark = args[7]
        elif not any(text in sql for text in ("jarvis_task_reservations", "insert into jarvis_task_grants")):
            raise AssertionError(sql)


class Tasks(TaskRepository):
    def __init__(self):
        self.limits = TaskLimits()
        self.lock = asyncio.Lock()
        self.connection = Connection(self)
        self.attached = set()
        self.pool = SimpleNamespace(acquire=self.acquire)
        self.requests, self.events = {}, []
        self.source_invalid = False
        self.task = SimpleNamespace(id=uuid4(), owner_id=OWNER, revision=1, fence=1,
            session_id=None, knowledge_area_id=None, backend_kind="frontier", model="fixture-model",
            status=TaskStatus.QUEUED, current_attempt_id=None, pending_status=None,
            title="CSV fixture", mode=None, requested_outcome="Sum the table.",
            description_payload_id=uuid4(), request_event_id=uuid4(), request_revision=1,
            context_watermark=1, budget=self.limits, reason=None, updated_at=NOW)
        self.claim = SimpleNamespace(task=self.task, task_revision=1, attempt_id=uuid4())

    @asynccontextmanager
    async def acquire(self):
        yield self.connection

    @asynccontextmanager
    async def transaction(self, owner_id=None):
        async with self.lock:
            self.connection.active = True
            try:
                yield self.connection
            finally:
                self.connection.active = False

    async def load(self, connection, owner_id, task_id, **kwargs):
        assert owner_id == OWNER and task_id == self.task.id
        return self.task

    async def lock_claim(self, connection, claim, **kwargs):
        return self.task, {"id": claim.attempt_id}

    async def context_problem(self, connection, task):
        return "source_withdrawn" if self.source_invalid else None

    async def validate_grant(self, connection, owner_id, grant_id):
        if self.source_invalid:
            raise TaskError("source_withdrawn", "Fixture source was withdrawn.")
        row = connection.grants.get(grant_id)
        if row is None:
            raise TaskError("grant_unavailable", "The fixture grant is unavailable.", 403)
        return dict(row, scope=connection.payloads[row["scope_payload_id"]]["content"])

    async def payload(self, connection, owner_id, content, **kwargs):
        identity = uuid4()
        connection.payloads[identity] = {"availability": "available", "content": content}
        return identity

    async def _request_event(self, connection, owner_id, request_id, parameters, session_id):
        return dict(id=uuid4(), revision=1, sequence=1,
            payload_id=await self.payload(connection, owner_id, parameters))

    async def _replay(self, connection, owner_id, request_id, operation, parameters, task_id=None):
        old = self.requests.get(request_id)
        if old is not None and old != parameters:
            raise TaskError("task_request_conflict", "The same request has a different scope.")
        return self.task if old is not None else None

    async def _save_request(self, connection, task, request_id, operation, parameters):
        self.requests[request_id] = parameters

    async def _queue_capacity(self, *args):
        pass

    async def _bind_voice_session(self, *args, **kwargs):
        pass

    async def emit(self, connection, task, kind, **kwargs):
        self.events.append(kind)

    async def resolve_action_barrier(self, *args):
        pass


class ManagedFilesDataTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tasks = Tasks()
        self.job, self.evidence = fixture_scope()
        self.enabled = True
        self.checked = 0

        async def select(**kwargs):
            self.assertTrue(kwargs["connection"].is_in_transaction())
            if not self.enabled:
                raise TaskError("plugin_disabled", "Disabled")
            return self.job.binding

        async def check(connection, **kwargs):
            self.assertTrue(connection.is_in_transaction())
            self.checked += 1
            if not self.enabled:
                raise TaskError("plugin_disabled", "Disabled")
            self.assertEqual(kwargs["binding"], self.job.binding)

        self.authority = SimpleNamespace(select_new_work=AsyncMock(side_effect=select),
            require_new_work=AsyncMock(side_effect=check))
        self.observer = AsyncMock(side_effect=lambda *_: self.evidence)
        self.runtime = ManagedFilesDataRuntime(self.authority, bundles=(self.job.bundle,),
            observe=self.observer, clock=lambda: NOW)
        self.registry = SimpleNamespace(require=AsyncMock())
        self.service = TaskService(self.tasks, {"frontier": SimpleNamespace(model="fixture-model")},
            registry=self.registry, sandbox_configuration=SimpleNamespace(policy=SandboxPolicy(), grant_seconds=120),
            managed_plugins=self.runtime)
        self.request = TaskCreate(request_id=uuid4(), title="CSV fixture", requested_outcome="Sum the table.",
            execution_environment="offline_sandbox")
        self.context = SimpleNamespace(tasks=self.tasks, task=self.tasks.task, claim=self.tasks.claim)
        self.budget = patch("app.tasks.actions.TaskBudgetRepository.reserve_locked",
            new=AsyncMock(return_value={"id": uuid4(), "status": "reserved"}))
        self.budget.start()
        self.addCleanup(self.budget.stop)

    async def create(self):
        await self.service.create_files_data(OWNER, self.request)
        return next(iter(self.tasks.connection.grants))

    async def test_create_persists_binding_and_exact_mapping_with_existing_grant(self):
        grant = await self.create()
        row = self.tasks.connection.payloads[self.tasks.task.description_payload_id]
        self.assertEqual(row["content"]["managed_plugin"], self.job.payload())
        self.assertEqual(self.tasks.connection.grants[grant]["operation_key"], GRANT_PREFIX + str(self.tasks.task.id))
        self.assertIn(CAPABILITY, self.registry.require.call_args.kwargs["required_capabilities"])
        self.assertEqual(self.tasks.connection.actions, {})
        self.assertEqual(self.checked, 0)

    async def test_exact_replay_precedes_selection_and_readiness_and_cannot_drop_route(self):
        await self.create()
        self.enabled = False
        self.service.managed_plugins = None
        self.registry.require.side_effect = AssertionError("Replay must not inspect readiness")
        self.assertIs(await self.service.create_files_data(OWNER, self.request), self.tasks.task)
        self.authority.select_new_work.assert_awaited_once()
        self.assertEqual(self.observer.await_count, 1)
        with self.assertRaises(TaskError) as caught:
            await self.service.create(OWNER, self.request)
        self.assertEqual(caught.exception.code, "task_request_conflict")

    async def test_provider_removed_replay_keeps_original_route(self):
        await self.create()
        self.service.providers.clear()
        self.assertIs(await self.service.create_files_data(OWNER, self.request), self.tasks.task)

    async def test_absent_runtime_or_unready_worker_never_accepts_new_managed_work(self):
        self.service.managed_plugins = None
        with self.assertRaises(TaskError):
            await self.create()
        self.assertFalse(self.tasks.connection.grants)
        self.service.managed_plugins = self.runtime
        self.service.registry = None
        with self.assertRaises(TaskError):
            await self.create()
        self.assertFalse(self.tasks.requests)

    async def test_incompatible_model_scope_does_not_select_or_grant(self):
        for changes in ({"execution_environment": "none"}, {"execution_environment": "networked_sandbox"},
                        {"edit_memory": True, "session_id": uuid4()}, {"grant_ids": (uuid4(),)}):
            with self.subTest(changes=changes), self.assertRaises(TaskError):
                await self.service.create_files_data(OWNER, self.request.model_copy(update=changes))
        self.authority.select_new_work.assert_not_awaited()

    async def test_missing_expired_mismatched_or_wrong_owner_evidence_denies(self):
        original = self.evidence
        variants = (None, replace(original, owner_id="other"), replace(original, executable="/opt/other/python"),
            replace(original, sandbox_template_id="other-image"),
            replace(original, inventory=original.inventory.model_copy(update={"state": "absent"})),
            replace(original, inventory=original.inventory.model_copy(update={"bundle_version": "other"})),
            replace(original, readiness=original.readiness.model_copy(update={"state": "unknown"})),
            replace(original, readiness=original.readiness.model_copy(update={"expires_at": NOW})),
            replace(original, readiness=original.readiness.model_copy(update={"configuration_id": uuid4()})),
            replace(original, readiness=original.readiness.model_copy(update={"bundle_sha256": "b" * 64})))
        for evidence in variants:
            self.evidence = evidence
            with self.subTest(evidence=evidence), self.assertRaises(TaskError):
                async with self.tasks.transaction(OWNER) as connection:
                    await self.runtime.admit(connection, OWNER)

    async def test_deployment_provenance_roundtrips_but_does_not_supply_readiness(self):
        binding = replace(self.job.binding, policy_revision=0, deployment_baseline="deployment-v1:" + "a" * 64)
        job = replace(self.job, binding=binding)
        self.assertEqual(ManagedFilesDataJob.from_payload(job.payload(), OWNER), job)
        self.evidence = None
        with self.assertRaises(TaskError):
            async with self.tasks.transaction(OWNER) as connection:
                await self.runtime.admit(connection, OWNER)

    async def test_changed_mapping_after_admission_denies_submission(self):
        grant = await self.create()
        self.runtime.bundles[self.job.binding.configuration_id] = self.job.bundle.model_copy(update={"executable": "/opt/other/python"})
        with self.assertRaises(TaskError) as caught:
            await self.runtime.ensure_submitted(self.context, grant)
        self.assertEqual(caught.exception.code, "managed_bundle_changed")
        self.assertTrue(all(row["submission_token"] is None for row in self.tasks.connection.actions.values()))

    async def test_disabled_queued_or_prepared_job_has_no_submission_pin(self):
        grant = await self.create()
        self.enabled = False
        with self.assertRaises(TaskError) as caught:
            await self.runtime.ensure_submitted(self.context, grant)
        self.assertEqual(caught.exception.code, "plugin_disabled")
        self.assertEqual(len(self.tasks.connection.actions), 1)
        action = next(iter(self.tasks.connection.actions.values()))
        self.assertEqual(action["status"], "authorized")
        self.assertIsNone(action["submission_token"])
        self.assertIsNone(action["receipt_payload_id"])

    async def test_submission_token_and_receipt_commit_before_provider_preparation(self):
        grant = await self.create()

        async def prepare(handle):
            self.assertFalse(self.tasks.connection.active)
            action = next(iter(self.tasks.connection.actions.values()))
            self.assertEqual(action["status"], "accepted")
            self.assertIsNotNone(action["submission_token"])
            receipt = self.tasks.connection.payloads[action["receipt_payload_id"]]["content"]
            self.assertEqual(receipt, submission_receipt(self.tasks.task, self.job, action["submission_token"]))
            self.assertEqual(receipt["effect"], "not_yet_observed")

        provider = SimpleNamespace(prepare=AsyncMock(side_effect=prepare))
        broker = SandboxCommandBroker(self.context, provider, grant, managed_plugins=self.runtime)
        row = {"id": uuid4(), "phase": "preparing"}
        broker.allocations = SimpleNamespace(handle=lambda _: SimpleNamespace(template_id=self.job.bundle.sandbox_template_id),
            ready=AsyncMock(return_value=row), current=AsyncMock(return_value=row))
        await broker._ready(row)
        provider.prepare.assert_awaited_once()
        self.assertEqual(self.checked, 1)

    async def test_submitted_same_job_continues_after_disable_and_observation_expiry(self):
        grant = await self.create()
        await self.runtime.ensure_submitted(self.context, grant)
        original = next(iter(self.tasks.connection.actions.values()))["submission_token"]
        self.enabled, self.evidence = False, None
        self.runtime.bundles.clear()
        self.assertEqual(await self.runtime.ensure_submitted(self.context, grant), self.job)
        self.assertEqual(self.checked, 1)
        self.assertEqual(len(self.tasks.connection.actions), 1)
        self.assertEqual(next(iter(self.tasks.connection.actions.values()))["submission_token"], original)

    async def test_policy_mutation_serializes_after_submission_not_between_check_and_token(self):
        grant = await self.create()
        checked, release, disabled = asyncio.Event(), asyncio.Event(), asyncio.Event()

        async def check(connection, **kwargs):
            self.assertTrue(connection.active)
            self.assertTrue(all(row["submission_token"] is None for row in connection.actions.values()))
            checked.set()
            await release.wait()

        self.authority.require_new_work.side_effect = check
        submitting = asyncio.create_task(self.runtime.ensure_submitted(self.context, grant))
        await asyncio.wait_for(checked.wait(), 1)

        async def disable():
            async with self.tasks.transaction(OWNER):
                self.assertIsNotNone(next(iter(self.tasks.connection.actions.values()))["submission_token"])
                self.enabled = False
                disabled.set()

        changing = asyncio.create_task(disable())
        await asyncio.sleep(0)
        self.assertFalse(disabled.is_set())
        release.set()
        await asyncio.wait_for(asyncio.gather(submitting, changing), 1)

    async def test_raw_action_submission_cannot_drop_managed_authority(self):
        grant = await self.create()
        actions = TaskActionRepository(self.tasks)
        scope = (await self.tasks.validate_grant(self.tasks.connection, OWNER, grant))["scope"]
        action = await actions.prepare(self.tasks.claim, GRANT_PREFIX + str(self.tasks.task.id), grant, "sandbox.execute", scope)
        await actions.authorize(self.tasks.claim, action["id"])
        with self.assertRaises(TaskError) as caught:
            await actions.begin_submission(self.tasks.claim, action["id"])
        self.assertEqual(caught.exception.code, "managed_submission_required")
        self.assertIsNone(self.tasks.connection.actions[action["id"]]["submission_token"])

    async def test_raw_broker_cannot_drop_binding_or_execute_a_shell_fallback(self):
        grant = await self.create()
        request = CommandRequest(operation_key="one", command="echo fallback")
        for runtime, error in ((None, SandboxError), (self.runtime, TaskError)):
            broker = SandboxCommandBroker(self.context, object(), grant, managed_plugins=runtime)
            broker.allocation = AsyncMock()
            with self.assertRaises(error):
                await broker.execute(request)
            broker.allocation.assert_not_awaited()

    async def test_removed_binding_and_wrong_owner_fail_closed(self):
        await self.create()
        row = self.tasks.connection.payloads[self.tasks.task.description_payload_id]
        original = row["content"]["managed_plugin"]
        del row["content"]["managed_plugin"]
        with self.assertRaises(TaskError):
            await load_managed_task(self.tasks.connection, self.tasks.task)
        row["content"]["managed_plugin"] = original
        original["binding"]["owner_id"] = "other-owner"
        with self.assertRaises(TaskError):
            await load_managed_task(self.tasks.connection, self.tasks.task)

    async def test_removed_source_or_submission_receipt_does_not_grant_continuation(self):
        grant = await self.create()
        await self.runtime.ensure_submitted(self.context, grant)
        action = next(iter(self.tasks.connection.actions.values()))
        self.tasks.connection.payloads[action["receipt_payload_id"]]["availability"] = "unavailable"
        with self.assertRaises(TaskError) as caught:
            await self.runtime.ensure_submitted(self.context, grant)
        self.assertEqual(caught.exception.code, "managed_submission_unconfirmed")
        self.tasks.source_invalid = True
        with self.assertRaises(TaskError):
            await self.runtime.ensure_submitted(self.context, grant)

    async def test_queued_update_preserves_binding_but_submitted_update_cannot_reuse_pin(self):
        grant = await self.create()
        await self.tasks.mutate(OWNER, self.tasks.task.id, "update",
            TaskMutation(request_id=uuid4(), expected_revision=1, message="Compute a mean instead."))
        row = self.tasks.connection.payloads[self.tasks.task.description_payload_id]
        self.assertEqual(row["content"]["managed_plugin"], self.job.payload())
        self.tasks.claim.task_revision = self.tasks.task.revision
        await self.runtime.ensure_submitted(self.context, grant)
        with self.assertRaises(TaskError) as caught:
            await self.tasks.mutate(OWNER, self.tasks.task.id, "update", TaskMutation(
                request_id=uuid4(), expected_revision=self.tasks.task.revision, message="Different new work."))
        self.assertEqual(caught.exception.code, "managed_submitted_assignment_immutable")

    async def test_independent_general_terminal_never_selects_or_checks_plugin_policy(self):
        self.enabled = False
        await self.service.create(OWNER, self.request)
        grant = next(iter(self.tasks.connection.grants))
        self.assertTrue(self.tasks.connection.grants[grant]["operation_key"].startswith("sandbox-task:"))
        broker = SandboxCommandBroker(self.context, object(), grant)
        self.assertIsNone(await broker._managed())
        self.authority.select_new_work.assert_not_awaited()
        self.authority.require_new_work.assert_not_awaited()
        self.assertNotIn("managed_plugin_route", self.tasks.requests[self.request.request_id])

    async def test_actual_image_mismatch_prevents_provider_prepare(self):
        grant = await self.create()
        provider = SimpleNamespace(prepare=AsyncMock())
        broker = SandboxCommandBroker(self.context, provider, grant, managed_plugins=self.runtime)
        broker.allocations = SimpleNamespace(handle=lambda _: SimpleNamespace(template_id="wrong-image"))
        with self.assertRaises(TaskError) as caught:
            await broker._ready({"id": uuid4(), "phase": "preparing"})
        self.assertEqual(caught.exception.code, "managed_bundle_image_mismatch")
        provider.prepare.assert_not_awaited()

    def test_explicit_worker_tool_replaces_shell_surface_only_for_managed_tasks(self):
        base = SimpleNamespace(context=self.context, registry=JarvisToolRegistry(),
            catalog=lambda: [], authorized_operations=lambda: [], read_tools=frozenset())
        bridge = SandboxToolBridge(base, object(), uuid4(), SandboxPolicy(), managed_job=self.job, managed_plugins=self.runtime)
        names = {item["name"] for item in bridge.catalog()}
        self.assertIn("files_data_execute", names)
        self.assertNotIn("sandbox_execute", names)
        self.assertEqual(bridge.authorized_operations()[0]["tool"], "files_data_execute")

    def test_executable_mapping_is_used_and_shell_suffix_is_rejected(self):
        request = self.job.bundle.command({"operation_key": "sum", "code": "print(sum([2, 3]))"})
        self.job.bundle.require_command(request)
        self.assertTrue(request.command.startswith("/usr/bin/python3 -I -S -c "))
        for command in (request.command + "; echo bypass", "/other/python -I -S -c 'print(1)'", "python3 -c 'print(1)'"):
            with self.subTest(command=command), self.assertRaises(TaskError):
                self.job.bundle.require_command(request.model_copy(update={"command": command}))

    async def test_foreground_tool_never_accepts_a_model_binding(self):
        module = FilesDataTaskToolModule(self.service)
        tool = module.build_tools(SimpleNamespace(session_id=uuid4(), owner_id=OWNER))[0]
        self.assertEqual(tool.name, "start_files_data_task")
        result = await tool.handler({"title": "CSV", "requested_outcome": "Sum it", "managed_plugin": self.job.payload()})
        self.assertEqual(result["status"], "failed")
        self.authority.select_new_work.assert_not_awaited()

    async def test_foreground_tool_uses_exact_request_not_provider_call_for_identity(self):
        self.service.create_files_data = AsyncMock(return_value=self.tasks.task)
        session = uuid4()
        tool = FilesDataTaskToolModule(self.service).build_tools(SimpleNamespace(session_id=session, owner_id=OWNER))[0]
        invocation = SimpleNamespace(task=None, request_event_id=uuid4(), request_revision=1,
            request_identity=SimpleNamespace(namespace="voice", event_id="exact-source"))
        with patch("app.plugins.task_tools.current_tool_invocation", return_value=invocation):
            first = await tool.handler({"title": "CSV", "requested_outcome": "Sum it"})
            await tool.handler({"title": "CSV", "requested_outcome": "Sum it"})
        self.assertTrue(first["accepted"])
        calls = self.service.create_files_data.await_args_list
        self.assertEqual(calls[0].args[1].request_id, calls[1].args[1].request_id)
        self.assertEqual(calls[0].kwargs["source"].event_id, invocation.request_event_id)
        self.assertEqual(calls[0].args[1].execution_environment, "offline_sandbox")

    async def publication(self, *, managed=True):
        """Real publisher/action bridge, with only grant/storage I/O substituted."""
        from app.files.generated import ArtifactInspection, ArtifactRequest, ArtifactSnapshot

        self.tasks.task.session_id = uuid4()
        request = ArtifactRequest(operation_key="table-output", filename="totals.csv", format="csv",
                                  description="Fixture table totals")
        parent_id, child_id = uuid4(), uuid4()
        snapshot = ArtifactSnapshot(request=request,
            inspection=ArtifactInspection(format="csv", content_sha256="c" * 64, size_bytes=12),
            parent_grant_id=parent_id, task_id=self.tasks.task.id, evidence_payload_id=uuid4())
        connection = self.tasks.connection
        for identity, capability, scope, operation in (
            (parent_id, "files.publish_generated", {"formats": ["csv"]}, "artifact-task:" + str(self.tasks.task.id)),
            (child_id, "files.publish_snapshot", snapshot.model_dump(mode="json"),
             "artifact-snapshot:" + self.tasks.task.id.hex + ":" + digest({"operation_key": request.operation_key})),
        ):
            payload = await self.tasks.payload(connection, OWNER, scope)
            connection.grants[identity] = dict(id=identity, capability=capability, operation_key=operation,
                request_event_id=self.tasks.task.request_event_id, request_revision=self.tasks.task.request_revision,
                scope_revision=1, scope_payload_id=payload)
            self.tasks.attached.add((OWNER, self.tasks.task.id, identity))
        action = GeneratedArtifactPublisher._action(await self.tasks.validate_grant(connection, OWNER, child_id))
        self.context.actions = TaskActionRepository(self.tasks)
        bridge = TaskToolBridge(self.context, JarvisToolRegistry(), read_tools=(), actions=())
        if managed:
            bridge.bind_managed_plugins(self.runtime)
        publisher = object.__new__(GeneratedArtifactPublisher)
        publisher.context, publisher.bridge = self.context, bridge
        publisher.parent_grant_id = parent_id
        publisher.prepare = AsyncMock(return_value=(action, snapshot))
        publisher.files = SimpleNamespace(publish_generated_snapshot=AsyncMock(return_value={"file_id": "fixture-file"}))
        projection = patch("app.tasks.generated_artifacts._file_payload", side_effect=lambda value: value)
        projection.start()
        self.addCleanup(projection.stop)
        return publisher, action, request, snapshot

    async def test_managed_attach_forwards_runtime_to_actual_artifact_action_bridge(self):
        await self.create()
        base = TaskToolBridge(self.context, JarvisToolRegistry(), read_tools=(), actions=())
        from app.tasks.memory_editing import TaskMemoryEditingBridge
        wrapped = TaskMemoryEditingBridge(base, None, uuid4(), None)
        with patch("app.sandbox.tools.SandboxArtifactTools.attach", new=AsyncMock(return_value=None)):
            result = await SandboxToolBridge.attach(wrapped, object(), policy_limit=SandboxPolicy(),
                                                    managed_plugins=self.runtime)
        self.assertIs(base.managed_plugins, self.runtime)
        self.assertIs(result.base, wrapped)
        self.assertFalse(self.tasks.connection.actions)

    async def test_publication_after_disable_uses_existing_pin_and_real_action_receipt(self):
        grant = await self.create()
        await self.runtime.ensure_submitted(self.context, grant)
        publisher, action, request, _ = await self.publication()
        self.enabled, self.evidence = False, None
        self.runtime.bundles.clear()
        result = await publisher.publish(request, b"value\n3\n", provider_call_id="publish-one")
        self.assertEqual(result["status"], "success")
        self.assertFalse(result["task_operation"]["replayed"])
        self.assertEqual(result["task_operation"]["operation_ref"], action.reference)
        record = self.tasks.connection.actions[__import__("uuid").UUID(result["task_operation"]["action_id"])]
        self.assertEqual(record["grant_id"], action.grant_id)
        self.assertEqual(record["parameters_digest"], digest(action.parameters))
        receipt = self.tasks.connection.payloads[record["receipt_payload_id"]]["content"]
        self.assertEqual(receipt["tool"], PUBLISH_TOOL)
        self.assertNotEqual(receipt.get("format"), "jarvis.managed-files-data.submission.v1")
        publisher.files.publish_generated_snapshot.assert_awaited_once()
        self.assertEqual(self.checked, 1)

    async def test_publication_replay_after_disable_never_repeats_effect_or_mints_token(self):
        grant = await self.create()
        await self.runtime.ensure_submitted(self.context, grant)
        publisher, action, request, _ = await self.publication()
        first = await publisher.publish(request, b"value\n3\n", provider_call_id="first-call")
        tokens = {key: row["submission_token"] for key, row in self.tasks.connection.actions.items()}
        self.enabled, self.evidence = False, None
        replay = await publisher.publish(request, provider_call_id="different-call")
        self.assertTrue(replay["task_operation"]["replayed"])
        self.assertEqual(replay["task_operation"]["action_id"], first["task_operation"]["action_id"])
        self.assertEqual(replay["task_operation"]["operation_ref"], action.reference)
        self.assertEqual(tokens, {key: row["submission_token"] for key, row in self.tasks.connection.actions.items()})
        publisher.files.publish_generated_snapshot.assert_awaited_once()

    async def test_publication_cannot_turn_queued_or_prepared_job_into_submitted_pin(self):
        grant = await self.create()
        publisher, _action, request, _ = await self.publication()
        self.enabled = False
        for prepare in (False, True):
            if prepare:
                actions = TaskActionRepository(self.tasks)
                scope = (await self.tasks.validate_grant(self.tasks.connection, OWNER, grant))["scope"]
                record = await actions.prepare(self.tasks.claim, GRANT_PREFIX + str(self.tasks.task.id),
                                               grant, "sandbox.execute", scope)
                await actions.authorize(self.tasks.claim, record["id"])
            with self.subTest(prepared=prepare), self.assertRaises(TaskError) as caught:
                await publisher.publish(request, b"value\n3\n", provider_call_id="not-submitted")
            self.assertEqual(caught.exception.code, "managed_submission_unconfirmed")
            self.assertTrue(all(row["submission_token"] is None for row in self.tasks.connection.actions.values()))
        publisher.files.publish_generated_snapshot.assert_not_awaited()

    async def test_publication_rejects_wrong_task_scope_and_changed_operation(self):
        grant = await self.create()
        await self.runtime.ensure_submitted(self.context, grant)
        publisher, action, request, _ = await self.publication()
        scope = self.tasks.connection.payloads[self.tasks.connection.grants[action.grant_id]["scope_payload_id"]]["content"]
        original_task = scope["task_id"]
        scope["task_id"] = str(uuid4())
        with self.assertRaises(TaskError):
            await publisher.publish(request, b"value\n3\n", provider_call_id="wrong-task")
        scope["task_id"] = original_task
        self.tasks.connection.grants[action.grant_id]["operation_key"] += "-different"
        with self.assertRaises(TaskError):
            await publisher.publish(request, b"value\n3\n", provider_call_id="wrong-operation")
        publisher.files.publish_generated_snapshot.assert_not_awaited()

    async def test_publication_pin_never_replaces_missing_child_or_parent_grant(self):
        grant = await self.create()
        await self.runtime.ensure_submitted(self.context, grant)
        publisher, action, request, snapshot = await self.publication()
        for identity in (action.grant_id, snapshot.parent_grant_id):
            attached = (OWNER, self.tasks.task.id, identity)
            self.tasks.attached.remove(attached)
            with self.subTest(grant=identity), self.assertRaises(TaskError):
                await publisher.publish(request, b"value\n3\n", provider_call_id="unattached")
            self.tasks.attached.add(attached)
            removed = self.tasks.connection.grants.pop(identity)
            with self.subTest(withdrawn=identity), self.assertRaises(TaskError):
                await publisher.publish(request, b"value\n3\n", provider_call_id="withdrawn")
            self.tasks.connection.grants[identity] = removed
        publisher.files.publish_generated_snapshot.assert_not_awaited()

    async def test_publication_exact_scope_change_is_not_covered_by_managed_pin(self):
        grant = await self.create()
        await self.runtime.ensure_submitted(self.context, grant)
        publisher, action, request, _ = await self.publication()
        scope = self.tasks.connection.payloads[self.tasks.connection.grants[action.grant_id]["scope_payload_id"]]["content"]
        # The bridge retains the originally authorized parameters while the
        # substituted repository returns different scope bytes.
        from copy import deepcopy
        original_action = replace(action, parameters=deepcopy(action.parameters))
        publisher.prepare.return_value = (original_action, publisher.prepare.return_value[1])
        scope["inspection"]["content_sha256"] = "d" * 64
        with self.assertRaises(TaskError) as caught:
            await publisher.publish(request, b"value\n3\n", provider_call_id="changed-snapshot")
        self.assertEqual(caught.exception.code, "managed_publication_scope")
        publisher.files.publish_generated_snapshot.assert_not_awaited()

    async def test_durable_artifact_submission_revalidates_forwarded_pin_under_lock(self):
        grant = await self.create()
        await self.runtime.ensure_submitted(self.context, grant)
        _publisher, action, _request, _ = await self.publication()
        actions = self.context.actions
        record = await actions.prepare(self.tasks.claim, "tool-grant:" + action.grant_id.hex,
                                       action.grant_id, action.capability, action.parameters)
        await actions.authorize(self.tasks.claim, record["id"])
        async with self.tasks.transaction(OWNER) as connection:
            pin = await self.runtime.submitted_job(connection, self.tasks, self.tasks.task)
        for wrong in (replace(pin, job_id=uuid4()), replace(pin, submission_token=uuid4()),
                      replace(pin, binding=replace(pin.binding, owner_id="another-owner"))):
            with self.subTest(pin=wrong), self.assertRaises(TaskError) as caught:
                await actions.begin_submission(self.tasks.claim, record["id"],
                                               managed_plugins=self.runtime, managed_pin=wrong)
            self.assertEqual(caught.exception.code, "managed_submission_mismatch")
            self.assertIsNone(self.tasks.connection.actions[record["id"]]["submission_token"])
        self.enabled = False
        token = await actions.begin_submission(self.tasks.claim, record["id"],
                                               managed_plugins=self.runtime, managed_pin=pin)
        self.assertIsNotNone(token)
        self.assertEqual(self.tasks.connection.actions[record["id"]]["status"], "submitting")
        self.assertIsNone(self.tasks.connection.actions[record["id"]]["receipt_payload_id"])

    async def test_pin_cannot_bypass_canonical_artifact_action_operation(self):
        grant = await self.create()
        await self.runtime.ensure_submitted(self.context, grant)
        _publisher, action, _request, _ = await self.publication()
        actions = self.context.actions
        record = await actions.prepare(self.tasks.claim, "alternate-operation", action.grant_id,
                                       action.capability, action.parameters)
        await actions.authorize(self.tasks.claim, record["id"])
        async with self.tasks.transaction(OWNER) as connection:
            pin = await self.runtime.submitted_job(connection, self.tasks, self.tasks.task)
        with self.assertRaises(TaskError) as caught:
            await actions.begin_submission(self.tasks.claim, record["id"], managed_plugins=self.runtime, managed_pin=pin)
        self.assertEqual(caught.exception.code, "managed_submission_required")

    async def test_publication_replay_and_recovery_reject_withdrawn_job_pin(self):
        grant = await self.create()
        await self.runtime.ensure_submitted(self.context, grant)
        publisher, action, request, snapshot = await self.publication()
        first = await publisher.publish(request, b"value\n3\n", provider_call_id="original")
        job_action = next(row for row in self.tasks.connection.actions.values() if row["grant_id"] == grant)
        self.tasks.connection.payloads[job_action["receipt_payload_id"]]["availability"] = "unavailable"
        with self.assertRaises(TaskError):
            await publisher.publish(request, provider_call_id="replay-withdrawn")
        artifact = next(row for row in self.tasks.connection.actions.values() if row["grant_id"] == action.grant_id)
        artifact["status"] = "outcome_unknown"
        with self.assertRaises(TaskError):
            await publisher.publish(request, provider_call_id="recover-withdrawn")
        publisher.files.publish_generated_snapshot.assert_awaited_once()
        self.assertEqual(first["status"], "success")

    async def test_general_terminal_publication_stays_independent_of_plugin_policy(self):
        self.enabled, self.evidence = False, None
        await self.service.create(OWNER, self.request)
        publisher, action, request, _ = await self.publication(managed=False)
        first = await publisher.publish(request, b"value\n3\n", provider_call_id="general-terminal")
        replay = await publisher.publish(request, provider_call_id="general-replay")
        self.assertEqual(first["status"], "success")
        self.assertTrue(replay["task_operation"]["replayed"])
        publisher.files.publish_generated_snapshot.assert_awaited_once()
        self.authority.select_new_work.assert_not_awaited()
        self.authority.require_new_work.assert_not_awaited()
