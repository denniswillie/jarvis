"""Offline contracts for task-only command admission; no database/provider I/O."""
from contextlib import asynccontextmanager
from dataclasses import FrozenInstanceError
from datetime import UTC, datetime
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch
from uuid import UUID

from pydantic import ValidationError

from app.knowledge.session_areas import SessionKnowledgeAreas
from app.live_state.admission import (
    CommandAdmission, append_task_command, command_parameters, decode_task_command,
    require_command, require_command_create, task_command_source,
)
from app.live_state.command_identity import command_arguments_digest, command_operation_id
from app.runtime.feature_config import TaskLimits
from app.tasks.context_events import ContextRecord
from app.tasks.context_repository import ContextRepository
from app.tasks.models import TaskCreate, TaskError, TaskMutation, TaskStatus, digest
from app.tasks.repository import TaskRepository
from app.tasks.service import TaskService


OWNER, SESSION, AREA, TASK, EVENT, PAYLOAD = "owner-opaque", UUID(int=1), UUID(int=2), UUID(int=3), UUID(int=4), UUID(int=5)


def command(operation="create", **changes):
    values = dict(owner_id=OWNER, session_id=SESSION, provider_session_key=" Provider \u00e9 ",
                  provider_call_id=" Call\n ", tool_name="task_" + operation,
                  capability="tasks." + operation, normalized_arguments={"goal": "Read only"})
    values.update(changes)
    return CommandAdmission(**values)


def creation(admission, **changes):
    values = dict(request_id=admission.operation_id, title="Read", requested_outcome="Summarize",
                  session_id=admission.session_id, knowledge_area_id=AREA)
    values.update(changes)
    return TaskCreate(**values)


def task(**changes):
    values = dict(id=TASK, owner_id=OWNER, session_id=SESSION, knowledge_area_id=AREA,
                  backend_kind="frontier", model="fixture", revision=1, request_event_id=EVENT,
                  request_revision=1, title="Read", requested_outcome="Summarize", mode=None,
                  current_attempt_id=None, description_payload_id=PAYLOAD, status=TaskStatus.QUEUED)
    values.update(changes)
    return SimpleNamespace(**values)


def record(admission, **changes):
    request = creation(admission) if admission.capability == "tasks.create" else TaskMutation(
        request_id=admission.operation_id, expected_revision=1)
    operation = admission.capability.split(".")[1]
    values = dict(id=EVENT, owner_id=OWNER, session_id=admission.session_id, sequence=1, origin_sequence=1,
        source_namespace="voice_command", source_event_id=str(admission.operation_id), revision=1,
        supersedes_event_id=None, kind="command_request", trust="derived", finality="final",
        source_time=None, recorded_at=datetime(2026, 11, 1, tzinfo=UTC), payload_id=PAYLOAD,
        availability="available", content_digest="a" * 64, is_current=True,
        content={"format": "jarvis.task_command.v1", "command": admission.parameters(), "operation": operation,
            "task_id": str(TASK), "base_revision": 0 if operation == "create" else 1,
            "area_id": str(AREA), "request": request.model_dump(mode="json"), "mode": None,
            "parameters_digest": digest(command_parameters(admission))})
    values.update(changes)
    return ContextRecord(**values)


def connection():
    return SimpleNamespace(is_in_transaction=Mock(return_value=True), fetchrow=AsyncMock(),
                           fetchval=AsyncMock(return_value=False), fetch=AsyncMock(return_value=[]),
                           execute=AsyncMock())


class CommandCarrierTests(unittest.TestCase):
    def test_exact_identity_and_detached_immutable_arguments(self):
        arguments = {"nested": ["original"]}
        admission = command(normalized_arguments=arguments)
        arguments["nested"].append("later")
        copy = admission.normalized_arguments
        copy["nested"].append("another")
        self.assertEqual(admission.normalized_arguments, {"nested": ["original"]})
        self.assertEqual(admission.operation_id, command_operation_id(
            OWNER, SESSION, " Provider \u00e9 ", " Call\n "))
        self.assertEqual(admission.provider_call_id, " Call\n ")
        self.assertEqual(admission.arguments_digest, command_arguments_digest(
            "task_create", {"nested": ["original"]}))
        with self.assertRaises(FrozenInstanceError):
            admission.capability = "tasks.cancel"

    def test_argument_conflict_keeps_operation_identity(self):
        first, second = command(), command(normalized_arguments={"goal": "Different"})
        self.assertEqual(first.operation_id, second.operation_id)
        self.assertNotEqual(command_parameters(first), command_parameters(second))

    def test_json_cannot_supply_authority_and_model_fields_are_unchanged(self):
        admission = command()
        with self.assertRaises(TaskError):
            require_command(admission.parameters(), OWNER, admission.operation_id, "create")
        for values in (dict(owner_id="other"), dict(session_id=str(SESSION)),
                       dict(normalized_arguments={"x": float("nan")})):
            if "owner_id" in values:
                with self.assertRaises(TaskError):
                    require_command(command(**values), OWNER, admission.operation_id, "create")
            else:
                with self.assertRaises((TypeError, ValueError)):
                    command(**values)
        with self.assertRaises(ValidationError):
            TaskCreate(**(creation(admission).model_dump() | {"command_admission": admission.parameters()}))
        with self.assertRaises(ValidationError):
            TaskMutation(request_id=admission.operation_id, expected_revision=1, command_admission={})

    def test_action_capabilities_and_grant_bearing_creation_are_rejected(self):
        with self.assertRaises(ValueError):
            command(capability="files.publish")
        with self.assertRaises(ValueError):
            command(capability="notes.create")
        admission = command()
        for changes in (dict(backend_kind="sandbox"), dict(execution_environment="offline_sandbox"),
                        dict(edit_memory=True), dict(grant_ids=(UUID(int=20),)),
                        dict(execution_environment="offline_sandbox", publish_artifacts=True)):
            with self.subTest(changes=changes), self.assertRaises(TaskError):
                require_command_create(admission, creation(admission, **changes))

    def test_legacy_fingerprints_are_unchanged(self):
        request = creation(command())
        expected = request.model_dump(mode="json") | {"source_event_id": None}
        for name in ("execution_environment", "edit_memory", "publish_artifacts"):
            expected.pop(name)
        self.assertEqual(TaskRepository._create_parameters(request), expected)
        mutation = TaskMutation(request_id=UUID(int=9), expected_revision=1)
        self.assertEqual(TaskRepository._mutation_parameters(mutation, None), mutation.model_dump(mode="json"))


class CommandSourceTests(unittest.IsolatedAsyncioTestCase):
    def setup_source(self, admission=None, source=None, target=None):
        admission = admission or command()
        source, target = source or record(admission), target or task()
        conn = connection()
        conn.fetchrow.side_effect = [source.model_dump(), {
            "task_id": TASK, "operation": admission.capability.split(".")[1],
            "parameters_digest": digest(command_parameters(admission)),
            "accepted_revision": source.content["base_revision"] + 1,
        }]
        return admission, source, target, conn

    async def test_durable_authority_needs_receipt_not_live_voice_or_user_trust(self):
        admission, source, target, conn = self.setup_source()
        self.assertFalse(source.authorizes_request)
        self.assertEqual(await task_command_source(object(), conn, target), source)
        self.assertIn("for share of e,p", conn.fetchrow.call_args_list[0].args[0])
        self.assertEqual(conn.fetchrow.call_args_list[1].args[1:], (OWNER, admission.operation_id))

    async def test_removed_changed_foreign_and_relabelled_sources_fail_closed(self):
        admission = command()
        for changes in (dict(availability="deleted"), dict(is_current=False), dict(owner_id="other"),
                        dict(trust="user_input"), dict(finality="provisional"),
                        dict(source_namespace="other"), dict(session_id=UUID(int=99))):
            with self.subTest(changes=changes), self.assertRaises(TaskError):
                _, _, target, conn = self.setup_source(admission, record(admission, **changes))
                await task_command_source(object(), conn, target)
        bad = record(admission)
        payload = dict(bad.content)
        payload["command"] = dict(payload["command"], arguments={"goal": "changed"})
        with self.assertRaises(TaskError):
            decode_task_command(bad.model_copy(update={"content": payload}))

    async def test_missing_receipt_wrong_target_and_grants_do_not_authorize(self):
        for kind in ("missing_receipt", "wrong_target", "grants"):
            _, source, target, conn = self.setup_source()
            if kind == "missing_receipt":
                conn.fetchrow.side_effect = [source.model_dump(), None]
            elif kind == "wrong_target":
                target.id = UUID(int=99)
            else:
                conn.fetchval.return_value = True
                conn.fetchrow.side_effect = [*conn.fetchrow.side_effect, None]
            with self.subTest(kind=kind), self.assertRaises(TaskError):
                await task_command_source(object(), conn, target)

    async def test_ordinary_derived_source_is_not_command_authority(self):
        source = record(command(), source_namespace="observer", kind="observation")
        _, _, target, conn = self.setup_source(source=source)
        self.assertIsNone(await task_command_source(object(), conn, target))
        self.assertEqual(conn.fetchrow.await_count, 1)

    async def test_append_reuses_transaction_and_pins_exact_command_target_base_area(self):
        admission, conn = command(), connection()
        with patch.object(ContextRepository, "append_in_transaction", new_callable=AsyncMock) as append:
            append.return_value = record(admission)
            await append_task_command(SimpleNamespace(pool=object()), conn, admission, "create", creation(admission),
                                      task_id=TASK, area_id=AREA, base_revision=0)
            used, owner, ingress = append.call_args.args
            self.assertIs(used, conn)
            self.assertEqual(owner, OWNER)
            self.assertEqual((ingress.source_namespace, ingress.kind, ingress.trust),
                             ("voice_command", "command_request", "derived"))
            self.assertEqual(ingress.expected_revision, 0)
            self.assertEqual(ingress.payload["task_id"], str(TASK))
            self.assertEqual(ingress.payload["area_id"], str(AREA))
            self.assertEqual(ingress.payload["base_revision"], 0)


class CommandRepositoryTests(unittest.IsolatedAsyncioTestCase):
    def repository(self):
        repo, conn = TaskRepository(object(), TaskLimits()), connection()

        @asynccontextmanager
        async def transaction(owner):
            self.assertEqual(owner, OWNER)
            yield conn

        repo.transaction = transaction
        repo._replay = AsyncMock(return_value=None)
        repo._queue_capacity = AsyncMock()
        repo._save_request = AsyncMock()
        repo._bind_voice_session = AsyncMock()
        repo.emit = AsyncMock()
        repo.payload = AsyncMock(return_value=PAYLOAD)
        repo.context_problem = AsyncMock(return_value=None)
        repo.load = AsyncMock(return_value=task())
        return repo, conn

    async def test_create_commits_source_task_receipt_on_one_connection(self):
        admission = command()
        repo, conn = self.repository()
        resolve, ready = AsyncMock(return_value=AREA), AsyncMock()

        async def fetchval(sql, *args):
            if sql.startswith("insert into jarvis_background_tasks"):
                repo.load.return_value.id = args[-1]
                return args[-1]
            return True

        conn.fetchval.side_effect = fetchval
        event = {"id": EVENT, "revision": 1, "sequence": 1, "payload_id": PAYLOAD, "session_id": SESSION}
        with patch("app.tasks.repository.append_task_command", new_callable=AsyncMock, return_value=event) as append:
            result = await repo.create(OWNER, creation(admission), model="fixture", command_admission=admission,
                                       area_resolver=resolve, admission=ready)
            self.assertIs(append.call_args.args[1], conn)
            self.assertEqual(append.call_args.kwargs["task_id"], result.id)
            self.assertEqual(append.call_args.kwargs["area_id"], AREA)
            resolve.assert_awaited_once_with(conn, OWNER, creation(admission), None, command_admission=admission)
            ready.assert_awaited_once_with(conn)
            repo._save_request.assert_awaited_once_with(conn, result, admission.operation_id, "create", command_parameters(admission))
            repo.emit.assert_awaited_once_with(conn, result, "accepted", payload_id=PAYLOAD)

    async def test_create_replay_precedes_area_and_provider_readiness(self):
        admission = command()
        repo, conn = self.repository()
        saved = task()
        repo._replay.return_value = saved
        resolve, ready = AsyncMock(), AsyncMock()
        with patch("app.tasks.repository.task_command_source", new_callable=AsyncMock, return_value=record(admission)):
            result = await repo.create(OWNER, creation(admission, knowledge_area_id=UUID(int=90)), model="fixture",
                                       command_admission=admission, area_resolver=resolve, admission=ready)
        self.assertIs(result, saved)
        resolve.assert_not_awaited()
        ready.assert_not_awaited()
        repo._save_request.assert_not_awaited()

    async def test_mutation_replay_uses_original_target_before_new_base_or_lookup(self):
        admission = command("cancel")
        repo, conn = self.repository()
        saved = task(revision=2, status=TaskStatus.CANCELLED)
        repo._replay.return_value = saved
        request = TaskMutation(request_id=admission.operation_id, expected_revision=99)
        with patch("app.tasks.repository.task_command_source", new_callable=AsyncMock, return_value=record(admission)):
            result = await repo.mutate(OWNER, UUID(int=999), "cancel", request, command_admission=admission)
        self.assertIs(result, saved)
        repo.load.assert_not_awaited()
        repo._replay.assert_awaited_once_with(conn, OWNER, admission.operation_id, "cancel", command_parameters(admission))

    async def test_conflicting_command_digest_uses_existing_receipt_conflict(self):
        admission, conn = command(), connection()
        changed = command(normalized_arguments={"goal": "other"})
        conn.fetchrow.return_value = {"operation": "create", "parameters_digest": digest(command_parameters(admission)), "task_id": TASK}
        repo = TaskRepository(object(), TaskLimits())
        with self.assertRaises(TaskError) as raised:
            await repo._replay(conn, OWNER, changed.operation_id, "create", command_parameters(changed))
        self.assertEqual(raised.exception.code, "task_request_conflict")

    async def test_fresh_cancel_keeps_existing_revision_and_event_lifecycle(self):
        admission = command("cancel")
        repo, conn = self.repository()
        current, cancelled = task(), task(revision=2, status=TaskStatus.CANCELLED)
        repo.load.side_effect = [current, cancelled]
        repo.has_unknown_actions = AsyncMock(return_value=False)
        request = TaskMutation(request_id=admission.operation_id, expected_revision=1)
        event = {"id": EVENT, "revision": 1, "sequence": 2, "payload_id": PAYLOAD, "session_id": SESSION}
        with patch("app.tasks.repository.append_task_command", new_callable=AsyncMock, return_value=event) as append:
            self.assertIs(await repo.mutate(OWNER, TASK, "cancel", request, command_admission=admission), cancelled)
            self.assertEqual(append.call_args.kwargs["base_revision"], 1)
            self.assertEqual(append.call_args.kwargs["task_id"], TASK)
        self.assertIn("fence=fence+1", conn.execute.call_args_list[0].args[0])
        repo._save_request.assert_awaited_once_with(conn, cancelled, admission.operation_id, "cancel", command_parameters(admission))

    async def test_json_carrier_is_rejected_before_transaction(self):
        repo = TaskRepository(object(), TaskLimits())
        admission = command()
        with self.assertRaises(TaskError):
            await repo.create(OWNER, creation(admission), model="fixture", command_admission=admission.parameters())


class CommandServiceAndContextTests(unittest.IsolatedAsyncioTestCase):
    async def test_removed_provider_replays_existing_command_instead_of_accepting_again(self):
        admission = command()
        repo = SimpleNamespace(pool=object(), replay_create=AsyncMock(return_value=task()))
        service = TaskService(repo, {})
        request = creation(admission)
        self.assertIs(await service.create(OWNER, request, command_admission=admission), repo.replay_create.return_value)
        repo.replay_create.assert_awaited_once_with(OWNER, request, command_admission=admission, mode=None)

    async def test_grant_checks_and_transcript_suppression_boundary_stay_closed(self):
        repo = TaskRepository(object(), TaskLimits())
        conn = connection()
        conn.fetchrow.return_value = {"availability": "available", "finality": "final", "trust": "derived", "kind": "command_request"}
        self.assertFalse(await repo._source_available(conn, OWNER, EVENT, 1))
        service = TaskService(SimpleNamespace(pool=object()), {})
        admission = command("cancel")
        with self.assertRaises(TaskError):
            await service.cancel_with_result_suppression(OWNER, TASK,
                TaskMutation(request_id=admission.operation_id, expected_revision=1), source=admission)

    async def test_area_resolution_accepts_only_the_typed_first_admission(self):
        areas = object.__new__(SessionKnowledgeAreas)
        areas.read_in_transaction = AsyncMock(return_value={"area_id": str(AREA), "area_status": "active"})
        admission, conn = command(), connection()
        request = creation(admission, knowledge_area_id=None)
        self.assertEqual(await areas.task_area_in_transaction(conn, OWNER, request, None,
                                                              command_admission=admission), AREA)
        areas.read_in_transaction.assert_awaited_once_with(conn, OWNER, SESSION)
        with self.assertRaises(TaskError):
            await areas.task_area_in_transaction(conn, OWNER, request, None, command_admission=admission.parameters())

    async def test_task_context_uses_receipt_validator_without_changing_record_authority(self):
        admission, source = command(), record(command())
        repository = ContextRepository(SimpleNamespace(pool=object()))
        repository._records = AsyncMock(return_value={EVENT: source})
        conn = connection()
        conn.fetch.return_value = [{"request_event_id": EVENT, "request_revision": 1, "session_id": SESSION}]
        with patch("app.live_state.admission.task_command_source", new_callable=AsyncMock, return_value=source) as validate:
            sessions, sources = await repository._task_sessions(conn, task())
            self.assertEqual(sessions, [SESSION])
            self.assertEqual(sources, [source])
            self.assertFalse(source.authorizes_request)
            self.assertFalse(validate.call_args.kwargs["lock"])
            self.assertFalse(validate.call_args.kwargs["current"])
