"""Offline admission/source contracts. SQL/renderer/provider observations are fakes."""
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch
from uuid import UUID

from pydantic import ValidationError

from app.files.generated import ArtifactPolicy, ArtifactRequest, ArtifactSnapshot, inspect_artifact
from app.live_state.admission import CommandAdmission, append_task_command, decode_task_command, require_command_create
from app.live_state.workspace_dispatch import LeanWorkspaceDispatcher
from app.plugins.managed import require_request
from app.runtime.feature_config import TaskLimits
from app.sandbox.models import SandboxPolicy
from app.tasks.media_editing import (
    MEDIA_TOOL, MediaInput, MediaInstruction, media_command_grant_source_available,
    require_media_input, require_media_recipe_input, validate_media_command_target, validate_media_input,
)
from app.tasks.models import TaskCreate, TaskError, TaskStatus, digest
from app.tasks.repository import TaskRepository
from app.tasks.service import TaskService
from app.workspace.commands import WorkspaceCommands
from app.workspace.media_agent import WorkspaceMediaAgent
from app.workspace.models import WorkspaceContext
from tests.test_lean_task_admission import record as ordinary_record
from tests.test_lean_workspace_dispatch import Journal


OWNER = "opaque-owner"
SESSION, AREA, RECORDING, FILE, SOURCE, RECEIPT, ORIGINAL, RECEIPT_PAYLOAD, ADMISSION, TASK, EVENT, PAYLOAD = [UUID(int=i) for i in range(1, 13)]


def pin(**changes):
    return MediaInput(**(dict(owner_id=OWNER, session_id=SESSION, area_id=AREA, recording_id=RECORDING,
        file_id=FILE, source_id=SOURCE, version_id="retained-version-1", receipt_event_id=RECEIPT,
        source_payload_id=ORIGINAL, receipt_payload_id=RECEIPT_PAYLOAD, admission_payload_id=ADMISSION,
        content_sha256="a" * 64, size_bytes=1000) | changes))


def arguments(**changes):
    return dict(instruction="Trim this recording to the useful section and keep its sound.", recording_id=str(RECORDING),
                saved={key: str(getattr(pin(), key)) for key in ("file_id", "source_id", "version_id", "receipt_event_id")}) | changes


def command(**changes):
    return CommandAdmission(**(dict(owner_id=OWNER, session_id=SESSION, provider_session_key=" Provider é ",
        provider_call_id=" Call\n ", tool_name=MEDIA_TOOL, capability="tasks.create", normalized_arguments=arguments()) | changes))


def request(c=None, **changes):
    c = c or command()
    return TaskCreate(**(dict(request_id=c.operation_id, title="Edit your video", session_id=SESSION,
        knowledge_area_id=AREA, requested_outcome=arguments()["instruction"], backend_kind="frontier",
        execution_environment="offline_sandbox", publish_artifacts=True) | changes))


def task():
    return SimpleNamespace(id=TASK, owner_id=OWNER, session_id=SESSION, knowledge_area_id=AREA,
        backend_kind="frontier", description_payload_id=PAYLOAD, revision=1, status=TaskStatus.QUEUED)


def source_row(value=None):
    value = value or pin()
    row = value.model_dump()
    row["completion_event_id"] = row.pop("receipt_event_id")
    row["completion_payload_id"] = row.pop("receipt_payload_id")
    row["original_sha256"] = value.content_sha256
    return row


@asynccontextmanager
async def transaction(connection):
    yield connection


def connection():
    return SimpleNamespace(is_in_transaction=Mock(return_value=True), fetch=AsyncMock(return_value=[]),
        fetchrow=AsyncMock(), fetchval=AsyncMock(return_value=True), execute=AsyncMock())


def description():
    return {"media_input": pin().model_dump(mode="json"), "managed_plugin": {"synthetic": True},
            "instruction_request": {"event_id": str(EVENT), "revision": 1, "session_id": str(SESSION)}}


def family():
    expires = datetime.now(UTC) + timedelta(minutes=10)
    policy = ArtifactPolicy(formats=("json", "mp4"), max_files=2, mp4_max_file_bytes=10000)
    base = dict(request_event_id=EVENT, request_revision=1, expires_at=expires)
    return [dict(base, id=UUID(int=100), capability="sandbox.execute", operation_key="managed-video-editor:" + str(TASK),
                 scope=SandboxPolicy().model_dump(mode="json")),
            dict(base, id=UUID(int=101), capability="files.publish_generated", operation_key="artifact-task:" + str(TASK),
                 scope=policy.model_dump(mode="json"))]


def child(parent):
    policy = ArtifactPolicy.model_validate(parent["scope"])
    req = ArtifactRequest(operation_key="media-recipe:exact", filename="recipe.json", format="json", description="Recipe")
    snapshot = ArtifactSnapshot(request=req, inspection=inspect_artifact(req, b"{}", policy),
        parent_grant_id=parent["id"], task_id=TASK, evidence_payload_id=UUID(int=102))
    return dict(parent, id=UUID(int=103), capability="files.publish_snapshot",
        operation_key="artifact-snapshot:" + TASK.hex + ":" + digest({"operation_key": req.operation_key}),
        scope=snapshot.model_dump(mode="json"))


class MediaShapeTests(unittest.TestCase):
    def test_exact_english_shape_and_managed_route(self):
        require_command_create(command(), request(), mode="background")
        require_request(request(), "video_editing", command())
        self.assertEqual(require_media_input(command(), pin()), pin())

    def test_authority_recipe_hash_and_paths_are_not_model_fields(self):
        for key in ("owner_id", "command_admission", "recipe", "content_sha256", "input_path", "grant_ids"):
            with self.subTest(key=key), self.assertRaises(ValidationError):
                MediaInstruction.model_validate(arguments(**{key: "untrusted"}))

    def test_wrong_route_tool_and_unrelated_authority_denied(self):
        for change in ({"edit_note": True}, {"edit_memory": True}, {"publish_artifacts": False},
                       {"execution_environment": "networked_sandbox"}, {"grant_ids": [UUID(int=88)]},
                       {"requested_outcome": "different"}, {"session_id": UUID(int=90)}):
            with self.subTest(change=change), self.assertRaises((TaskError, ValueError)):
                require_command_create(command(), request(**change))
        with self.assertRaises(TaskError):
            require_request(request(), "files_data", command())
        with self.assertRaises(TaskError):
            require_command_create(command(tool_name="live_note"), request())

    def test_exact_refs_and_owner_not_model_digest(self):
        for change in ({"owner_id": "other"}, {"session_id": UUID(int=99)}, {"recording_id": UUID(int=99)},
                       {"file_id": UUID(int=99)}, {"version_id": "other"}, {"receipt_event_id": UUID(int=99)}):
            with self.subTest(change=change), self.assertRaises(TaskError):
                require_media_input(command(), pin(**change))

    def test_raw_call_identity_stable_but_instruction_conflicts(self):
        first = command()
        other = command(normalized_arguments=arguments(instruction="Keep the original length."))
        self.assertEqual(first.operation_id, other.operation_id)
        self.assertNotEqual(first.arguments_digest, other.arguments_digest)
        self.assertEqual(first.provider_call_id, " Call\n ")

    def test_existing_command_representation_unchanged(self):
        from tests.test_lean_task_admission import command as ordinary_command
        record = ordinary_record(ordinary_command())
        self.assertIsNotNone(decode_task_command(record))
        self.assertFalse(record.authorizes_request)

    def test_media_advertisement_opt_in_and_journal_required(self):
        ordinary = WorkspaceCommands(files=object())
        self.assertNotIn(MEDIA_TOOL, ordinary.schemas())
        commands = WorkspaceCommands(files=object(), media_editor=object())
        self.assertIn(MEDIA_TOOL, commands.schemas())
        dispatcher = LeanWorkspaceDispatcher(owner_id=OWNER, session_id=SESSION, provider_session_key=lambda: "p",
            is_active=lambda: True, commands=commands)
        self.assertNotIn(MEDIA_TOOL, [tool.name for tool in dispatcher.definitions()])
        dispatcher = LeanWorkspaceDispatcher(owner_id=OWNER, session_id=SESSION, provider_session_key=lambda: "p",
            is_active=lambda: True, commands=commands, journal=Journal())
        self.assertIn(MEDIA_TOOL, [tool.name for tool in dispatcher.definitions()])


class MediaAuthorityTests(unittest.IsolatedAsyncioTestCase):
    async def test_exact_source_recheck_and_withdrawal(self):
        c = connection()
        c.fetchrow.return_value = source_row()
        self.assertEqual(await validate_media_input(c, OWNER, SESSION, pin()), pin())
        for key in ("content_sha256", "original_sha256", "version_id", "source_payload_id", "completion_event_id", "area_id"):
            with self.subTest(key=key):
                c.fetchrow.return_value = dict(source_row(), **{key: "changed"})
                with self.assertRaises(TaskError):
                    await validate_media_input(c, OWNER, SESSION, pin())
        c.fetchrow.return_value = None  # withdrawn/revoked/no longer available exact join
        with self.assertRaises(TaskError):
            await validate_media_input(c, OWNER, SESSION, pin())

    async def test_wrong_owner_session_fails_before_source_query(self):
        c = connection()
        for owner, session in (("other", SESSION), (OWNER, UUID(int=99))):
            with self.assertRaises(TaskError):
                await validate_media_input(c, owner, session, pin())
        c.fetchrow.assert_not_awaited()

    async def test_complete_family_and_exact_child(self):
        c = connection()
        c.fetchrow.return_value = {"availability": "available", "content": description()}
        rows = family()
        rows.append(child(rows[1]))
        c.fetch.return_value = rows
        with patch("app.plugins.managed.decode_managed_job"):
            self.assertEqual(await validate_media_command_target(c, task()), pin())
        # Synthetic recipe only: no installed-template/render observation is
        # claimed here. The same API guards both fresh and recovered plans.
        recipe = dict(schema_version=1, edit_id=str(UUID(int=200)), previous_recipe=None,
            template={"id": "jarvis.clip.v1", "version": "synthetic", "bundle_sha256": "b" * 64},
            sources=[{"source_id": "clip", "file": {"file_id": str(FILE), "version_id": pin().version_id,
                "content_sha256": pin().content_sha256}}],
            segments=[{"segment_id": "one", "source_id": "clip", "in_ms": 0, "out_ms": 1000}],
            captions=[], output_profile="portrait_720p24_v1", fit="contain", audio="preserve")
        instruction = SimpleNamespace(event_id=EVENT, revision=1)
        proof = SimpleNamespace(content={"operation": "create", "command": {"tool_name": MEDIA_TOOL},
                                         "media_input": pin().model_dump(mode="json")})
        tasks = SimpleNamespace(_source_available=AsyncMock(return_value=False))
        with patch("app.tasks.instruction_source.task_instruction_source", AsyncMock(return_value=instruction)), \
                patch("app.live_state.admission.task_command_source", AsyncMock(return_value=proof)) as source:
            for phase in ("fresh", "recovery"):
                with self.subTest(phase=phase):
                    self.assertEqual(await require_media_recipe_input(tasks, c, task(), recipe), pin())
            for key, value in (("file_id", str(UUID(int=99))), ("version_id", "other-version"), ("content_sha256", "c" * 64)):
                changed = dict(recipe, sources=[{"source_id": "clip", "file": dict(recipe["sources"][0]["file"], **{key: value})}])
                with self.subTest(key=key), self.assertRaises(TaskError) as failure:
                    await require_media_recipe_input(tasks, c, task(), changed)
                self.assertEqual(failure.exception.code, "media_command_recipe_mismatch")
            with self.assertRaises(TaskError) as failure:
                await require_media_recipe_input(tasks, c, task(), dict(recipe, previous_recipe=recipe["sources"][0]["file"]))
            self.assertEqual(failure.exception.code, "media_command_predecessor_not_admitted")
            source.return_value = None
            with self.assertRaises(TaskError):
                await require_media_recipe_input(tasks, c, task(), recipe)
            tasks._source_available.return_value = True
            self.assertIsNone(await require_media_recipe_input(tasks, c, task(), recipe))

    async def test_wrong_scope_family_parent_or_child_rejected(self):
        cases = [lambda rows: rows.pop(),
                 lambda rows: rows[0].update(operation_key="sandbox-task:" + str(TASK)),
                 lambda rows: rows[0]["scope"].update(internet_access=True),
                 lambda rows: rows[1]["scope"].update(formats=["json", "mp4", "text"]),
                 lambda rows: rows[1].update(request_event_id=UUID(int=999)),
                 lambda rows: rows.append(dict(rows[1], capability="memory.correct_sources")),
                 lambda rows: rows.append(dict(child(rows[1]), scope=dict(child(rows[1])["scope"], task_id=str(UUID(int=99))))),
                 lambda rows: rows.append(dict(child(rows[1]), scope=dict(child(rows[1])["scope"], parent_grant_id=str(UUID(int=99)))))]
        for change in cases:
            c = connection()
            c.fetchrow.return_value = {"availability": "available", "content": description()}
            rows = family()
            change(rows)
            c.fetch.return_value = rows
            with self.subTest(change=change), patch("app.plugins.managed.decode_managed_job"), self.assertRaises(TaskError):
                await validate_media_command_target(c, task())

    async def test_durable_source_after_disconnect_no_voice_object(self):
        c, tasks = connection(), SimpleNamespace(load=AsyncMock(return_value=task()))
        c.fetch.return_value = [{"task_id": TASK}]
        c.fetchrow.side_effect = [{"availability": "available", "content": description()}, source_row()]
        proof = SimpleNamespace(content={"command": {"tool_name": MEDIA_TOOL}, "media_input": pin().model_dump(mode="json")})
        with patch("app.plugins.managed.decode_managed_job"), patch("app.live_state.admission.task_command_source", AsyncMock(return_value=proof)):
            self.assertTrue(await media_command_grant_source_available(tasks, c, OWNER, family()[0]))
        c.fetchrow.side_effect = [{"availability": "available", "content": description()}, None]
        with patch("app.plugins.managed.decode_managed_job"), patch("app.live_state.admission.task_command_source", AsyncMock(return_value=proof)):
            self.assertFalse(await media_command_grant_source_available(tasks, c, OWNER, family()[0]))

    async def test_pin_recorded_in_same_context_and_dependency_transaction(self):
        c = connection()
        saved = SimpleNamespace(id=EVENT, revision=1, sequence=1, payload_id=PAYLOAD, session_id=SESSION)
        with patch("app.tasks.context_repository.ContextRepository.append_in_transaction", AsyncMock(return_value=saved)) as append:
            await append_task_command(SimpleNamespace(pool=object()), c, command(), "create", request(), task_id=TASK,
                                      area_id=AREA, base_revision=0, mode="background", media_input=pin())
        self.assertIs(append.await_args.args[0], c)
        ingress = append.await_args.args[2]
        self.assertEqual(ingress.payload["media_input"], pin().model_dump(mode="json"))
        self.assertEqual(c.execute.await_count, 3)

    async def test_repository_replay_precedes_pin_readiness_and_resolution(self):
        c, saved = connection(), task()
        repo = TaskRepository(object(), TaskLimits())
        repo.transaction = lambda owner: transaction(c)
        repo._command_replay = AsyncMock(return_value=saved)
        ready, resolve = AsyncMock(), AsyncMock()
        result = await repo.create(OWNER, request(), model="synthetic", command_admission=command(),
            managed_route="video_editing", managed_admission=ready, area_resolver=resolve,
            media_input=None, mode="background")
        self.assertIs(result, saved)
        ready.assert_not_awaited()
        resolve.assert_not_awaited()
        c.fetchrow.assert_not_awaited()

    async def test_create_receipt_precedes_grants_on_same_transaction(self):
        c, saved, order = connection(), task(), []
        repo = TaskRepository(object(), TaskLimits())
        repo.transaction = lambda owner: transaction(c)
        repo._command_replay = AsyncMock(return_value=None)
        repo._queue_capacity = AsyncMock()
        repo.payload = AsyncMock(return_value=PAYLOAD)
        repo.load = AsyncMock(return_value=saved)
        repo._save_request = AsyncMock(side_effect=lambda *a: order.append("receipt"))
        repo.validate_grant = AsyncMock()
        repo.context_problem = AsyncMock(return_value=None)
        repo._bind_voice_session = AsyncMock()
        repo.emit = AsyncMock()
        c.fetchval.return_value = TASK
        async def grants(conn, task_id, event):
            self.assertIs(conn, c)
            order.append("grants")
            return (UUID(int=100), UUID(int=101))
        event = {"id": EVENT, "payload_id": PAYLOAD, "revision": 1, "sequence": 1}
        with patch("app.tasks.repository.append_task_command", AsyncMock(return_value=event)) as append, \
                patch("app.tasks.media_editing.validate_media_input", AsyncMock(return_value=pin())), \
                patch("app.tasks.media_editing.validate_media_command_target", AsyncMock()), \
                patch("app.plugins.managed.decode_managed_job", return_value=SimpleNamespace(payload=lambda: {"synthetic": True})):
            result = await repo.create(OWNER, request(), model="synthetic", command_admission=command(),
                managed_route="video_editing", managed_admission=AsyncMock(return_value={}),
                media_input=pin(), mode="background", grant_factory=grants)
        self.assertIs(result, saved)
        self.assertEqual(order, ["receipt", "grants"])
        self.assertIs(append.await_args.args[1], c)
        self.assertEqual(append.await_args.kwargs["media_input"], pin())
        self.assertEqual(repo.validate_grant.await_count, 2)

    async def test_service_forwards_explicit_pin_and_narrows_only_media_policy(self):
        repo = SimpleNamespace(pool=object(), create=AsyncMock(return_value=task()), payload=AsyncMock(return_value=PAYLOAD))
        artifact = ArtifactPolicy(formats=("text", "json", "mp4"), mp4_max_file_bytes=10000)
        service = TaskService(repo, {"frontier": SimpleNamespace(model="synthetic")},
            sandbox_configuration=SimpleNamespace(policy=SandboxPolicy(), grant_seconds=600),
            artifact_configuration=SimpleNamespace(policy=artifact, grant_seconds=600, area_id=AREA))
        await service.create_media(OWNER, request(), command_admission=command(), media_input=pin(), mode="background")
        opts = repo.create.await_args.kwargs
        self.assertEqual(opts["managed_route"], "video_editing")
        self.assertEqual(opts["media_input"], pin())
        c = connection()
        await opts["grant_factory"](c, TASK, {"id": EVENT, "revision": 1, "payload_id": PAYLOAD})
        scope = repo.payload.await_args.args[2]
        self.assertEqual(scope["formats"], ["json", "mp4"])
        self.assertEqual(scope["max_files"], 2)
        self.assertEqual(artifact.formats, ("text", "json", "mp4"))
        with self.assertRaises(TaskError):
            await service.create(OWNER, request(), command_admission=command(), media_input=pin())


class MediaAdapterTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.c = connection()
        self.c.fetchrow.return_value = source_row()
        self.pool = object()
        sources = SimpleNamespace(pool=self.pool, transaction=lambda owner: transaction(self.c),
            resolve_document_in_transaction=AsyncMock(return_value=SimpleNamespace(payload_id=ORIGINAL)))
        self.repo = SimpleNamespace(pool=self.pool, replay_create=AsyncMock(return_value=None),
            lock_owner=AsyncMock(), reference=TaskRepository.reference)
        self.tasks = SimpleNamespace(repository=self.repo, create_media=AsyncMock(return_value=task()))
        self.files = SimpleNamespace(repository=SimpleNamespace(sources=sources), media_metadata=AsyncMock(
            return_value=(SimpleNamespace(size_bytes=1000), {"content_sha256": "a" * 64})))
        self.recordings = SimpleNamespace(sources=sources,
            require_admitted_recording=AsyncMock(return_value=SimpleNamespace(area_id=AREA, admission_payload_id=ADMISSION)),
            get_completion=AsyncMock(return_value=SimpleNamespace(**{key: getattr(pin(), key) for key in
                ("file_id", "source_id", "version_id", "receipt_event_id", "receipt_payload_id")})))
        self.agent = WorkspaceMediaAgent(self.tasks, self.files, self.recordings)
        self.ctx = WorkspaceContext(OWNER, SESSION, command().operation_id)

    async def test_english_to_service_with_exact_recording_ref(self):
        result = await self.agent.submit(self.ctx, arguments(), command_admission=command())
        self.assertEqual(result["outcome"], "accepted")
        self.assertIs(result["effect_committed"], False)
        self.files.media_metadata.assert_awaited_once_with(OWNER, SESSION, FILE, pin().version_id)
        self.assertEqual(self.tasks.create_media.await_args.kwargs["media_input"], pin())
        self.assertEqual(self.tasks.create_media.await_args.args[1].requested_outcome, arguments()["instruction"])

    async def test_domain_replay_never_resolves_current_metadata(self):
        self.repo.replay_create.return_value = task()
        self.files.media_metadata.side_effect = AssertionError("must not resolve again")
        result = await self.agent.submit(self.ctx, arguments(), command_admission=command())
        self.assertEqual(result["outcome"], "accepted")
        self.tasks.create_media.assert_not_awaited()
        self.recordings.get_completion.assert_not_awaited()

    async def test_missing_or_ambiguous_ref_clarifies_without_global_latest(self):
        for args in ({"instruction": "Edit that recording."}, arguments(saved=None), arguments(recording_id=None)):
            normalized = MediaInstruction.model_validate(args).model_dump(mode="json", exclude_none=True)
            result = await self.agent.submit(self.ctx, normalized, command_admission=command(normalized_arguments=normalized))
            self.assertEqual(result["outcome"], "clarification_required")
            self.assertFalse(result["effect_committed"])
        self.files.media_metadata.assert_not_awaited()
        self.tasks.create_media.assert_not_awaited()

    async def test_changed_completion_or_hash_never_creates(self):
        self.recordings.get_completion.return_value = None
        with self.assertRaises(TaskError):
            await self.agent.submit(self.ctx, arguments(), command_admission=command())
        self.tasks.create_media.assert_not_awaited()

    async def test_real_commands_dispatch_and_raw_identity_replay(self):
        self.agent.receipt = AsyncMock(return_value=None)
        commands = WorkspaceCommands(files=self.files, media_editor=self.agent)
        journal = Journal()
        dispatcher = LeanWorkspaceDispatcher(owner_id=OWNER, session_id=SESSION,
            provider_session_key=lambda: " Provider é ", is_active=lambda: True, commands=commands, journal=journal)
        result = await dispatcher.execute(MEDIA_TOOL, arguments(), provider_call_id=" Call\n ")
        self.assertEqual(result["outcome"], "accepted")
        self.assertEqual(self.tasks.create_media.await_args.kwargs["command_admission"].operation_id, command().operation_id)
        self.agent.receipt.return_value = result
        again = await dispatcher.execute(MEDIA_TOOL, arguments(), provider_call_id=" Call\n ", admit_if=lambda: False)
        self.assertEqual(again, result)
        self.tasks.create_media.assert_awaited_once()
        conflict = await dispatcher.execute(MEDIA_TOOL, arguments(instruction="Different edit."), provider_call_id=" Call\n ")
        self.assertEqual(conflict["outcome"], "conflict")
        self.assertNotIn("instruction", str(journal.starts))
        self.assertNotIn("content_sha256", str(journal.finishes))

    async def test_interruption_before_admission_and_unknown_no_retry(self):
        self.agent.receipt = AsyncMock(return_value=None)
        commands = WorkspaceCommands(files=self.files, media_editor=self.agent)
        dispatcher = LeanWorkspaceDispatcher(owner_id=OWNER, session_id=SESSION,
            provider_session_key=lambda: "p", is_active=lambda: True, commands=commands, journal=Journal())
        result = await dispatcher.execute(MEDIA_TOOL, arguments(), provider_call_id="call", admit_if=lambda: False)
        self.assertEqual(result["error"]["code"], "command_interrupted_before_admission")
        self.files.media_metadata.assert_not_awaited()
        self.tasks.create_media.assert_not_awaited()
        dispatcher.journal.rows.clear()
        dispatcher.journal.fail_finalize = True
        result = await dispatcher.execute(MEDIA_TOOL, arguments(), provider_call_id="unknown")
        self.assertEqual(result["outcome"], "accepted")
        result = await dispatcher.execute(MEDIA_TOOL, arguments(), provider_call_id="unknown")
        self.assertEqual(result["outcome"], "outcome_unknown")
        self.tasks.create_media.assert_awaited_once()


class MediaMigrationSourceTests(unittest.TestCase):
    def test_forward_guard_preserves_note_and_user_input_branch(self):
        # Source-only contract, NOT execution/acceptance of this SQL.
        sql = Path(__file__).parents[1] / "app" / "migrations" / "sql"
        old = (sql / "0046_account_note_command_grants.sql").read_text()
        new = (sql / "0049_media_command_grants.sql").read_text()
        injected = """        if new.capability in ('sandbox.execute','files.publish_generated','files.publish_snapshot')
            and jarvis_media_command_grant_valid(new) is true then
            new.created_at:=clock_timestamp();
            return new;
        end if;
"""
        self.assertTrue(new.endswith(old.replace("        if new.capability not in ('notes.edit_account','notes.publish_account_edit')",
                                                injected + "        if new.capability not in ('notes.edit_account','notes.publish_account_edit')")))
        self.assertNotIn("create table", new.lower())
        self.assertIn("tg_table_name<>'jarvis_operation_grants'", new)
        self.assertIn("tr.accepted_revision=1", new)
        self.assertIn("rec.session_id=e.session_id", new)
        self.assertIn("p.content->>'content_sha256'=fp.content_sha256", new)
        self.assertIn("scope->'formats'='[\"json\",\"mp4\"]'::jsonb", new)


if __name__ == "__main__":
    unittest.main()
