"""Offline executor contracts; all codec/provider/SQL receipts are synthetic.

Real recipe validation, local-file ownership, input staging and tool accounting
run here. These tests do not qualify an image, decode media or execute SQL.
"""
import asyncio
from contextlib import asynccontextmanager
from dataclasses import replace
import hashlib
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch
from uuid import UUID, uuid4

from app.files.generated import ArtifactPolicy
from app.files.local_file import _snapshot_chunks
from app.files.transfer import FileManifest, FileTransferPolicy, MP4Inspection
from app.plugins.authority import ManagedPluginBinding
from app.plugins.media import ManagedMediaJob, MediaRendererBundle
from app.sandbox.inputs import SandboxInputService
from app.sandbox.media import ManagedMediaTools
from app.sandbox.media_decode import MediaDecodeReceipt
from app.tasks.generated_artifacts import GeneratedArtifactPublisher
from app.tasks.models import ClaimLost, TaskError, digest
from app.workspace.media_publication import SavedMediaPlan, publication_requests, restore_plan
from app.workspace.media_recipe import MediaRecipeError, PreviousRecipe, RetainedRef
from app.workspace.media_render import AcceptedMediaOutput, RendererBinding
from tests.test_workspace_media_recipe import recipe


class ManagedMediaToolTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.locals, self.rows, self.saved, self.outputs, self.events = [], {}, {}, {}, []
        self.raw = b"synthetic MP4 bytes: not an actual media qualification"
        self.sha = hashlib.sha256(self.raw).hexdigest()
        self.value = recipe()
        self.value["sources"][0]["file"]["content_sha256"] = self.sha
        self.reference = RetainedRef.model_validate(self.value["sources"][0]["file"])
        self.task = SimpleNamespace(owner_id="media-test-owner", id=uuid4())
        self.claim = SimpleNamespace(task=self.task, attempt_id=uuid4())
        self.allocation = {"id": uuid4(), "phase": "ready"}
        self.connection = SimpleNamespace(fetch=AsyncMock(side_effect=self.fetch))

        @asynccontextmanager
        async def transaction(*_):
            yield self.connection

        self.pool = object()
        self.tasks = SimpleNamespace(pool=self.pool, transaction=transaction,
                                     lock_claim=AsyncMock(return_value=(self.task, {})))
        # This executor fixture doubles SQL/source admission. The real guard
        # wiring and lock semantics have separate input-binding and SQL tests.
        authority = patch("app.tasks.media_editing.require_media_recipe_input", AsyncMock(return_value=None))
        self.authority = authority.start()
        self.addCleanup(authority.stop)
        self.context = SimpleNamespace(task=self.task, claim=self.claim, tasks=self.tasks,
            reserve=AsyncMock(side_effect=lambda *a, **k: {"id": uuid4()}),
            budgets=SimpleNamespace(submit=AsyncMock()), settle=AsyncMock())
        self.bridge = SimpleNamespace(context=self.context, _check=AsyncMock(), read_tools=("get_file_metadata",))
        self.renderer = RendererBinding(template=self.value["template"], physical_template_id="synthetic-renderer",
            build_id="synthetic-build", software_sha256="c" * 64, build_receipt_sha256="d" * 64,
            policy={"max_source_bytes": 100000, "max_source_duration_ms": 30000, "max_output_frames": 240},
            max_output_bytes=100000)
        config = uuid4()
        self.job = ManagedMediaJob(ManagedPluginBinding(self.task.owner_id, "video_editor", config, 1),
            MediaRendererBundle(configuration_id=config, bundle_version="fixture-1", bundle_sha256="b" * 64,
                                renderer=self.renderer))
        self.handle = SimpleNamespace(sandbox_id="synthetic-vm", template_id=self.renderer.physical_template_id,
            scope=SimpleNamespace(allocation_id=self.allocation["id"], owner_id=self.task.owner_id, task_id=self.task.id))
        self.input_policy = FileTransferPolicy("mp4_capture_v1", 100000)
        self.output_policy = FileTransferPolicy("mp4_output_v1", 100000)
        self.file = SimpleNamespace(id=self.reference.file_id, filename="original.mp4", content_type="video/mp4",
            size_bytes=len(self.raw), memory_source={"source_id": str(uuid4()), "revision": 1, "generation": 1,
                "content_digest": "f" * 64, "is_current": True, "document_id": self.reference.version_id},
            task_evidence={"synthetic": True})
        self.measured = MP4Inspection(content_sha256=self.sha, size_bytes=len(self.raw), duration_ms=30000,
                                      width=720, height=1280, video_codec="h264", audio_codec="aac")
        self.inspector = Mock(side_effect=AssertionError("Retained input must not use output inspection"))
        self.inspector.inspect_retained_input = AsyncMock(return_value=self.measured)
        self.files = SimpleNamespace(sources=SimpleNamespace(pool=self.pool, transaction=transaction),
            task_evidence=SimpleNamespace(context=self.context), get_file=AsyncMock(return_value=self.file),
            read_file=AsyncMock(side_effect=self.snapshot), media_inspector=self.inspector,
            media_policy=self.output_policy, publications=SimpleNamespace(read_manifest=AsyncMock(return_value=(
                FileManifest(size_bytes=len(self.raw), content_sha256=self.sha), self.input_policy))))
        self.provider = SimpleNamespace(stage_input_file=AsyncMock(side_effect=self.stage),
            decode_media_render_input=AsyncMock(side_effect=self.decode),
            read_workspace_file_snapshot=AsyncMock(side_effect=self.read_output))
        self.broker = SimpleNamespace(context=self.context, provider=self.provider,
            allocation=AsyncMock(return_value=self.allocation), _ready=AsyncMock(side_effect=lambda r: r),
            allocations=SimpleNamespace(handle=Mock(return_value=self.handle),
                current=AsyncMock(return_value=self.allocation), request_cleanup=AsyncMock()),
            commands=SimpleNamespace(prepare=AsyncMock(side_effect=self.prepare),
                                     current=AsyncMock(side_effect=self.command_current)),
            observe=AsyncMock(side_effect=self.observe), execute_media=AsyncMock(side_effect=self.execute))
        self.inputs = SandboxInputService(self.bridge, self.broker, files=self.files)
        self.inputs.stage = AsyncMock(wraps=self.inputs.stage)
        self.publisher = GeneratedArtifactPublisher(self.bridge, self.files, uuid4())
        self.policy = ArtifactPolicy(formats=("json", "mp4"), mp4_max_file_bytes=100000)
        self.publisher._parent = AsyncMock(return_value=({}, self.policy))
        self.artifacts = SimpleNamespace(publisher=self.publisher, policy=self.policy,
            ready_allocation=AsyncMock(return_value=self.allocation), status=AsyncMock(return_value={
                "status": "success", "artifact_operation": {"state": "not_submitted"}}))
        self.tools = ManagedMediaTools(self.bridge, self.broker, self.inputs, self.artifacts, self.job, self.files)
        self.store = SimpleNamespace(_verify_reference=AsyncMock(side_effect=self.verify_source),
            previous_recipe=AsyncMock(), publish_recipe=AsyncMock(side_effect=self.publish_recipe),
            load_plan=AsyncMock(side_effect=lambda o, t, d: self.saved.get(d)),
            load_accepted_output=AsyncMock(side_effect=lambda o, t, d: self.outputs.get(d)),
            publish_output=AsyncMock(side_effect=self.publish_output))
        self.tools.store = self.store

    async def asyncTearDown(self):
        leaked = [local for local in self.locals if not local._closed]
        for local in leaked:
            local.close()
        self.assertEqual(leaked, [], "Every owned input/output must close, including denial and cancellation")

    async def snapshot(self, **_):
        async def chunks():
            yield self.raw
        local = await _snapshot_chunks(chunks(), max_bytes=100000)
        self.locals.append(local)
        return local

    async def fetch(self, query, owner, task_id, operation_digest):
        self.assertIn("jarvis_sandbox_commands", query)
        self.assertEqual((owner, task_id), (self.task.owner_id, self.task.id))
        self.events.append("lookup")
        return [row.copy() for row in self.rows.values() if row["operation_digest"] == operation_digest][:2]

    async def verify_source(self, owner, reference):
        if owner != self.task.owner_id or reference != self.reference:
            raise MediaRecipeError("media_publication_unavailable", "Exact source unavailable")
        return self.file

    async def stage(self, handle, identity, local, *, policy):
        self.events.append("stage")
        local._check()
        return {"path": "/home/jarvis/inputs/" + identity, "sha256": local.content_sha256,
                "size_bytes": local.size_bytes}

    async def decode(self, handle, *, input_path, content_sha256, size_bytes, binding):
        self.events.append("decode")
        # Synthetic receipt only. No real ffprobe/ffmpeg or provider runs here.
        return MediaDecodeReceipt.model_validate({"format": "jarvis.media-decode.v1", "sandbox_id": handle.sandbox_id,
            "allocation_id": str(handle.scope.allocation_id), "input_path": input_path,
            "content_sha256": content_sha256, "size_bytes": size_bytes, "software_sha256": binding.software_sha256,
            "video_streams": 1, "audio_streams": 1, "video_decoded": True, "video_frames": 720,
            "audio_decoded": True, "audio_duration_us": 30000000, "duration_ms": 30000})

    async def prepare(self, claim, allocation_id, request):
        self.events.append("prepare")
        row = {"id": uuid4(), "parameters": request.model_dump(mode="json"), "availability": "available",
            "request_digest": digest(request.model_dump(mode="json")), "operation_digest": digest({"key": request.operation_key}),
            "attempt_id": claim.attempt_id, "allocation_id": allocation_id, "phase": "prepared"}
        self.rows[request.operation_key] = row
        return row.copy()

    async def command_current(self, claim, identity):
        self.assertIs(claim, self.claim)
        return next(row.copy() for row in self.rows.values() if row["id"] == identity)

    async def publish_recipe(self, plan, publisher, *, provider_call_id):
        self.events.append("recipe")
        self.assertIs(publisher, self.publisher)
        self.assertEqual(provider_call_id, "actual-call")
        restored = restore_plan(plan.provenance_bytes, owner_id=self.task.owner_id,
                                task_id=self.task.id, plan_digest=plan.digest)
        self.saved[plan.digest] = SavedMediaPlan(restored, self.reference, uuid4())
        return {"status": "success", "task_operation": {"state": "accepted", "operation_ref": "op_recipe"}}

    async def execute(self, plan):
        self.events.append("execute")
        self.assertIn(plan.digest, self.saved, "A command must not run before accepted recipe persistence")
        self.rows[plan.operation_key]["phase"] = "running"
        return {"status": "success", "sandbox_command": {"state": "running", "operation_key": plan.operation_key}}

    async def observe(self, identity):
        self.events.append("observe")
        row = next(r for r in self.rows.values() if r["id"] == identity)
        command = {"state": row["phase"], "operation_key": row["parameters"]["operation_key"]}
        if row["phase"] == "completed":
            command.update(started=True, exit_code=0, timed_out=False, runner_error=None)
        return {"status": "success", "sandbox_command": command}

    async def read_output(self, handle, path, **kwargs):
        self.events.append("read-output")
        self.assertTrue(path.endswith("/render/output.mp4"))
        self.assertLessEqual(kwargs["max_bytes"], self.renderer.max_output_bytes)
        return await self.snapshot()

    async def publish_output(self, plan, publisher, content=None, *, provider_call_id):
        self.assertIs(publisher, self.publisher)
        if plan.digest not in self.outputs:
            if content is None:
                raise TaskError("artifact_original_required", "Exact original required")
            content._check()
            self.outputs[plan.digest] = AcceptedMediaOutput(owner_id=self.task.owner_id, task_id=self.task.id,
                action_id=uuid4(), render_plan_sha256=plan.digest, file=self.reference,
                content_type="video/mp4", size_bytes=content.size_bytes)
        return {"status": "success", "task_operation": {"state": "accepted", "operation_ref": "op_output"}}

    async def call(self, name="media_render", **changes):
        arguments = {"operation_key": "edit-one"}
        if name == "media_render":
            arguments["recipe"] = self.value
        return await self.tools.invoke(name, arguments | changes, provider_call_id="actual-call")

    async def completed(self):
        await self.call()
        self.rows["edit-one"]["phase"] = "completed"
        return next(iter(self.saved.values())).plan

    async def test_catalog_and_argument_boundaries(self):
        definitions = self.tools.definitions()
        self.assertEqual([tool.name for tool in definitions], ["media_render", "media_status", "media_publish"])
        for field in ("path", "inspection", "binding", "limits"):
            result = await self.call(**{field: {}})
            self.assertEqual(result["status"], "failed")
        self.broker.allocation.assert_not_called()
        self.assertEqual(self.context.settle.await_count, 4)

    async def test_runtime_supplies_template_and_preserves_canonical_replay(self):
        creative = {key: value for key, value in self.value.items() if key != "template"}
        first = await self.call(recipe=creative)
        self.assertNotIn("template", creative, "Original model arguments remain unchanged")
        saved = next(iter(self.saved.values())).plan
        self.assertEqual(saved.recipe.recipe.template, self.job.bundle.renderer.template)
        self.assertEqual(saved.recipe.recipe.model_dump(mode="json"), self.value)
        replay = await self.call()  # Old explicit matching pin normalizes identically.
        self.assertEqual(first["media_operation"]["plan_sha256"], replay["media_operation"]["plan_sha256"])
        self.provider.stage_input_file.assert_awaited_once()
        self.broker.execute_media.assert_awaited_once()
        self.assertGreater(self.authority.await_count, 0)

    async def test_explicit_wrong_template_cannot_change_an_existing_operation(self):
        creative = {key: value for key, value in self.value.items() if key != "template"}
        await self.call(recipe=creative)
        before = dict(self.rows["edit-one"])
        self.value["template"] = dict(self.value["template"], bundle_sha256="0" * 64)
        with self.assertRaises(TaskError):
            await self.call()
        self.assertEqual(self.rows["edit-one"], before)
        self.broker.execute_media.assert_awaited_once()

    async def test_explicit_null_pin_is_not_replaced_by_runtime(self):
        value = dict(self.value, template=None)
        result = await self.call(recipe=value)
        self.assertEqual(result["status"], "failed")
        self.assertIsNone(value["template"])
        self.broker.allocation.assert_not_called()

    async def test_actual_metadata_plus_separate_decode_precede_persisted_render(self):
        result = await self.call()
        self.assertEqual(result["media_operation"]["state"], "submitted")
        self.assertEqual(self.events, ["lookup", "stage", "decode", "prepare", "recipe", "execute"])
        self.inspector.inspect_retained_input.assert_awaited_once()
        self.inspector.assert_not_called()
        self.assertEqual(self.context.settle.await_count, 2, "Media call and nested input stage both consume tool steps")

    async def test_restart_unknown_and_completed_never_stage_or_execute_again(self):
        await self.call()
        fresh = ManagedMediaTools(self.bridge, self.broker, self.inputs, self.artifacts, self.job, self.files)
        fresh.store = self.store
        self.tools = fresh
        for phase, expected in (("unknown", "unknown"), ("completed", "rendered")):
            self.rows["edit-one"]["phase"] = phase
            result = await self.call()
            self.assertEqual(result["media_operation"]["state"], expected)
        self.provider.stage_input_file.assert_awaited_once()
        self.broker.execute_media.assert_awaited_once()
        self.provider.decode_media_render_input.assert_awaited_once()

    async def test_changed_recipe_same_operation_rejected_before_staging(self):
        await self.call()
        self.value["captions"][0]["text"] = "Changed"
        with self.assertRaises(TaskError) as raised:
            await self.call()
        self.assertEqual(raised.exception.code, "media_operation_conflict")
        self.provider.stage_input_file.assert_awaited_once()

    async def test_missing_recipe_blocks_original_without_reconstructing_authority_from_path(self):
        await self.call()
        self.saved.clear()
        self.rows["edit-one"]["phase"] = "unknown"
        result = await self.call()
        self.assertEqual(result["media_operation"]["state"], "recipe_unresolved")
        self.artifacts.status.assert_awaited_once()
        self.provider.stage_input_file.assert_awaited_once()
        self.broker.execute_media.assert_awaited_once()

    async def test_unknown_recipe_can_reconcile_same_snapshot_then_observe(self):
        await self.call()
        saved = next(iter(self.saved.values()))
        self.saved.clear()
        async def reconcile(*args):
            self.saved[saved.plan.digest] = saved
            return {"status": "success"}
        self.artifacts.status.side_effect = reconcile
        result = await self.call("media_status")
        self.assertEqual(result["media_operation"]["state"], "submitted")
        self.provider.stage_input_file.assert_awaited_once()

    async def test_status_and_publish_absence_never_allocate(self):
        for name in ("media_status", "media_publish"):
            result = await self.call(name)
            self.assertEqual(result["status"], "failed")
        self.broker.allocation.assert_not_called()
        self.files.read_file.assert_not_called()

    async def test_publication_reads_only_successful_original_then_replays_without_vm(self):
        plan = await self.completed()
        result = await self.call("media_publish")
        self.assertEqual(result["media_operation"]["state"], "published")
        self.assertEqual(result["task_operation"]["operation_ref"], "op_output")
        self.artifacts.ready_allocation.side_effect = AssertionError("Replay must not need a VM")
        again = await self.call("media_publish")
        self.assertEqual(again["media_operation"]["output"]["render_plan_sha256"], plan.digest)
        self.provider.read_workspace_file_snapshot.assert_awaited_once()
        self.broker.execute_media.assert_awaited_once()

    async def test_running_and_failed_commands_cannot_publish_bytes(self):
        await self.call()
        result = await self.call("media_publish")
        self.assertEqual(result["media_operation"]["state"], "submitted")
        self.broker.observe.return_value = None
        self.broker.observe.side_effect = None
        for changes in ({"exit_code": 1}, {"exit_code": False}, {"timed_out": True}, {"runner_error": "failed"}, {"started": False}):
            command = {"state": "completed", "started": True, "exit_code": 0, "timed_out": False, "runner_error": None} | changes
            self.broker.observe.return_value = {"sandbox_command": command}
            result = await self.call("media_publish")
            self.assertEqual(result["media_operation"]["state"], "completed_unsuccessfully")
        self.provider.read_workspace_file_snapshot.assert_not_called()

    async def test_unknown_upload_never_readbacks_or_rerenders(self):
        await self.completed()
        self.store.publish_output.side_effect = None
        self.store.publish_output.return_value = {"status": "success", "task_operation": {"state": "outcome_unknown"}}
        result = await self.call("media_publish")
        self.assertNotEqual(result["media_operation"]["state"], "published")
        self.assertEqual(result["task_operation"]["state"], "outcome_unknown")
        self.provider.read_workspace_file_snapshot.assert_not_called()
        self.broker.execute_media.assert_awaited_once()

    async def test_missing_or_wrong_decode_evidence_never_sets_metadata_decodable(self):
        original = self.decode
        for change in ({"video_decoded": False}, {"video_frames": 0}, {"audio_decoded": False},
                       {"audio_duration_us": 0}, {"content_sha256": "0" * 64}, {"software_sha256": "0" * 64},
                       {"allocation_id": str(uuid4())}, {"sandbox_id": "other"}, {"input_path": "/tmp/other"},
                       {"duration_ms": 29999}, {"audio_streams": 0}, {"video_decoded": 1}, {"unexpected": True}):
            async def invalid(*args, **kwargs):
                fields = (await original(*args, **kwargs)).model_dump(mode="python") | change
                # Deliberately malformed synthetic provider objects exercise
                # the adapter's revalidation, not just its isinstance guard.
                if "unexpected" in change:
                    return fields
                return MediaDecodeReceipt.model_construct(**fields)
            self.provider.decode_media_render_input.side_effect = invalid
            with self.subTest(change=change), self.assertRaises(TaskError) as raised:
                await self.call()
            self.assertEqual(raised.exception.code, "media_decode_invalid")
        self.broker.commands.prepare.assert_not_called()
        self.store.publish_recipe.assert_not_called()
        self.assertEqual(self.broker.allocations.request_cleanup.await_count, 13)

    async def test_absent_decoder_rejected_before_allocation(self):
        self.provider.decode_media_render_input = None
        with self.assertRaises(TaskError) as raised:
            await self.call()
        self.assertEqual(raised.exception.code, "media_decode_unavailable")
        self.broker.allocation.assert_not_called()

    async def test_metadata_mismatch_closes_input_and_prevents_staging(self):
        self.inspector.inspect_retained_input.return_value = self.measured.model_copy(update={"content_sha256": "0" * 64})
        with self.assertRaises(ValueError):
            await self.call()
        self.provider.stage_input_file.assert_not_called()

    async def test_exact_reference_and_template_cannot_be_replaced(self):
        self.value["sources"][0]["file"]["version_id"] = "newer"
        with self.assertRaises(TaskError):
            await self.call()
        self.value["sources"][0]["file"]["version_id"] = self.reference.version_id
        self.value["template"]["bundle_sha256"] = "0" * 64
        with self.assertRaises(TaskError):
            await self.call()
        self.broker.allocation.assert_not_called()

    async def test_previous_recipe_identity_and_edit_lineage_are_preserved(self):
        previous = RetainedRef(file_id=uuid4(), version_id="previous-version", content_sha256="e" * 64)
        self.value["previous_recipe"] = previous.model_dump(mode="json")
        self.store.previous_recipe.return_value = PreviousRecipe(file=previous, edit_id=UUID(self.value["edit_id"]))
        await self.call()
        plan = next(iter(self.saved.values())).plan
        self.assertEqual(plan.recipe.recipe.previous_recipe, previous)
        self.store.previous_recipe.return_value = PreviousRecipe(file=previous, edit_id=uuid4())
        with self.assertRaises(TaskError):
            await self.call("media_status")
        self.broker.execute_media.assert_awaited_once()

    async def test_source_withdrawal_blocks_replay(self):
        await self.completed()
        self.store._verify_reference.side_effect = MediaRecipeError("media_publication_unavailable", "Withdrawn")
        with self.assertRaises(TaskError):
            await self.call("media_publish")
        self.provider.read_workspace_file_snapshot.assert_not_called()

    async def test_claim_loss_after_decode_cleans_same_allocation(self):
        async def lost(*args, **kwargs):
            receipt = await self.decode(*args, **kwargs)
            self.bridge._check.side_effect = ClaimLost()
            return receipt
        self.provider.decode_media_render_input.side_effect = lost
        with self.assertRaises(ClaimLost):
            await self.call()
        self.broker.allocations.request_cleanup.assert_awaited_once()
        self.broker.commands.prepare.assert_not_called()

    async def test_revocation_after_output_read_closes_file_and_cleans_original(self):
        await self.completed()
        async def withdrawn(*args, **kwargs):
            local = await self.read_output(*args, **kwargs)
            self.store._verify_reference.side_effect = MediaRecipeError("media_publication_unavailable", "Withdrawn")
            return local
        self.provider.read_workspace_file_snapshot.side_effect = withdrawn
        with self.assertRaises(TaskError):
            await self.call("media_publish")
        self.broker.allocations.request_cleanup.assert_awaited_once()
        self.assertEqual(self.outputs, {})

    async def test_cancelled_publication_closes_owned_output_and_settles(self):
        await self.completed()
        async def cancelled(plan, publisher, content=None, **kwargs):
            if content is None:
                raise TaskError("artifact_original_required", "Original required")
            raise asyncio.CancelledError()
        self.store.publish_output.side_effect = cancelled
        before = self.context.settle.await_count
        with self.assertRaises(asyncio.CancelledError):
            await self.call("media_publish")
        self.assertEqual(self.context.settle.await_count, before + 1)

    async def test_historical_attempt_never_allocates_replacement(self):
        await self.call()
        self.rows["edit-one"]["attempt_id"] = uuid4()
        with self.assertRaises(TaskError) as raised:
            await self.call()
        self.assertEqual(raised.exception.code, "media_original_attempt_unavailable")
        self.broker.allocation.assert_awaited_once()

    async def test_forged_publisher_owner_and_wrong_saved_binding_fail_closed(self):
        with self.assertRaises(ValueError):
            ManagedMediaTools(self.bridge, self.broker, self.inputs,
                SimpleNamespace(publisher=SimpleNamespace()), self.job, self.files)
        await self.call()
        saved = next(iter(self.saved.values()))
        wrong = self.renderer.model_copy(update={"software_sha256": "0" * 64})
        self.saved[saved.plan.digest] = replace(saved, plan=replace(saved.plan, binding=wrong))
        with self.assertRaises(TaskError):
            await self.call("media_status")

    async def test_withdrawn_command_and_conflicting_historical_rows_do_not_stage(self):
        await self.call()
        self.rows["edit-one"]["availability"] = "removed"
        with self.assertRaises(TaskError):
            await self.call()
        self.rows["edit-one"]["availability"] = "available"
        self.rows["duplicate"] = dict(self.rows["edit-one"], id=uuid4())
        with self.assertRaises(TaskError) as raised:
            await self.call()
        self.assertEqual(raised.exception.code, "media_operation_ambiguous")
        self.provider.stage_input_file.assert_awaited_once()
    async def fault_before_recipe(self, *, before_upload=False):
        """Synthetic publication cut points; no actual upload or SQL occurs."""
        frozen = []

        async def fault(plan, publisher, *, provider_call_id):
            self.assertIs(publisher, self.publisher)
            if before_upload:
                frozen.append((publication_requests(plan)[0], plan.provenance_bytes))
                self.artifacts.status.return_value = {"status": "success",
                    "artifact_operation": {"state": "not_submitted"}}
            else:
                self.artifacts.status.return_value = {"status": "failed",
                    "error": {"code": "artifact_operation_not_found"}}
            raise RuntimeError("fault before upload" if before_upload else "fault before recipe publication")

        self.store.publish_recipe.side_effect = fault
        with self.assertRaisesRegex(RuntimeError, "fault before"):
            await self.call()
        original = self.rows["edit-one"].copy()
        self.assertEqual(original["phase"], "prepared")
        self.assertEqual(self.saved, {})
        self.broker.execute_media.assert_not_called()

        async def resume(plan, publisher, *, provider_call_id):
            if frozen:
                self.assertEqual((publication_requests(plan)[0], plan.provenance_bytes), frozen[0])
            return await self.publish_recipe(plan, publisher, provider_call_id=provider_call_id)

        self.store.publish_recipe.side_effect = resume
        return original

    async def assert_same_original_recovery(self, *, before_upload):
        original = await self.fault_before_recipe(before_upload=before_upload)
        # New adapter/input-service instances have no retained plan or validators.
        self.inputs = SandboxInputService(self.bridge, self.broker, files=self.files)
        self.inputs.stage = AsyncMock(wraps=self.inputs.stage)
        self.tools = ManagedMediaTools(self.bridge, self.broker, self.inputs, self.artifacts, self.job, self.files)
        self.tools.store = self.store
        result = await self.call()
        self.assertEqual(result["media_operation"]["state"], "submitted")
        self.assertEqual(self.rows["edit-one"]["id"], original["id"])
        self.assertEqual(self.rows["edit-one"]["request_digest"], original["request_digest"])
        self.assertEqual(self.rows["edit-one"]["parameters"], original["parameters"])
        self.assertEqual(self.inputs.stage.await_args.kwargs, {"allocation_id": original["allocation_id"]})
        self.broker.allocation.assert_awaited_once()
        self.broker._ready.assert_awaited_once()
        self.artifacts.ready_allocation.assert_awaited_once()
        self.broker.commands.prepare.assert_awaited_once()
        self.broker.execute_media.assert_awaited_once()
        self.assertEqual(self.provider.stage_input_file.await_count, 2)
        first, second = self.provider.stage_input_file.await_args_list
        self.assertIs(first.args[0], second.args[0])
        self.assertEqual(first.args[1], second.args[1])
        self.assertEqual(self.provider.decode_media_render_input.await_count, 2)
        self.assertEqual(self.store.publish_recipe.await_count, 2)
        self.artifacts.status.assert_awaited_once()

    async def test_recover_fault_before_recipe_publication_on_original_command_and_vm(self):
        await self.assert_same_original_recovery(before_upload=False)

    async def test_recover_frozen_recipe_before_upload_with_identical_original_bytes(self):
        await self.assert_same_original_recovery(before_upload=True)

    async def test_reconstruction_changed_recipe_cannot_publish_or_change_command(self):
        original = await self.fault_before_recipe(before_upload=True)
        self.value["captions"][0]["text"] = "A changed recipe"
        with self.assertRaises(TaskError) as raised:
            await self.call()
        self.assertEqual(raised.exception.code, "media_operation_conflict")
        self.assertEqual(self.rows["edit-one"], original)
        self.assertEqual(self.store.publish_recipe.await_count, 1)
        self.broker.commands.prepare.assert_awaited_once()
        self.broker.execute_media.assert_not_called()
        self.broker.allocation.assert_awaited_once()

    async def test_reconstruction_other_attempt_and_submitted_token_are_denied(self):
        await self.fault_before_recipe()
        self.rows["edit-one"]["attempt_id"] = uuid4()
        with self.assertRaises(TaskError) as raised:
            await self.call()
        self.assertEqual(raised.exception.code, "media_original_attempt_unavailable")
        self.rows["edit-one"]["attempt_id"] = self.claim.attempt_id
        self.rows["edit-one"]["submission_token"] = uuid4()
        with self.assertRaises(TaskError) as raised:
            await self.call()
        self.assertEqual(raised.exception.code, "media_reconstruction_unavailable")
        self.provider.stage_input_file.assert_awaited_once()
        self.provider.decode_media_render_input.assert_awaited_once()
        self.broker.allocation.assert_awaited_once()

    async def test_reconstruction_missing_or_not_ready_original_never_replaces_vm(self):
        await self.fault_before_recipe()
        for phase in ("reserved", "preparing", "uncertain", "cleanup_requested", "cleaned"):
            self.allocation["phase"] = phase
            with self.subTest(phase=phase), self.assertRaises(TaskError) as raised:
                await self.call()
            self.assertEqual(raised.exception.code, "media_original_allocation_unavailable")
        self.broker.allocations.current.side_effect = TaskError("sandbox_not_current", "Original VM unavailable")
        with self.assertRaises(TaskError) as raised:
            await self.call()
        self.assertEqual(raised.exception.code, "sandbox_not_current")
        self.broker.allocation.assert_awaited_once()
        self.broker._ready.assert_awaited_once()
        self.provider.stage_input_file.assert_awaited_once()
        self.broker.allocations.request_cleanup.assert_not_called()

    async def test_unknown_submitted_completed_or_uncertain_publication_never_reconstruct(self):
        await self.fault_before_recipe()
        for phase in ("unknown", "submitting", "running", "completed"):
            self.rows["edit-one"]["phase"] = phase
            result = await self.call()
            self.assertEqual(result["media_operation"]["state"], "recipe_unresolved")
        self.rows["edit-one"]["phase"] = "prepared"
        self.artifacts.status.return_value = {"status": "success", "task_operation": {"state": "outcome_unknown"}}
        result = await self.call()
        self.assertEqual(result["media_operation"]["state"], "recipe_unresolved")
        self.broker.commands.current.assert_not_called()
        self.provider.stage_input_file.assert_awaited_once()
        self.provider.decode_media_render_input.assert_awaited_once()
        self.broker.execute_media.assert_not_called()

    async def test_status_and_publish_do_not_reconstruct_prepared_original(self):
        await self.fault_before_recipe()
        for name in ("media_status", "media_publish"):
            result = await self.call(name)
            self.assertEqual(result["media_operation"]["state"], "recipe_unresolved")
        self.broker.commands.current.assert_not_called()
        self.provider.stage_input_file.assert_awaited_once()
        self.broker.execute_media.assert_not_called()

    async def test_reconstruction_source_version_hash_and_fingerprint_remain_exact(self):
        await self.fault_before_recipe()
        source = self.value["sources"][0]["file"]
        for field, value in (("version_id", "replacement"), ("content_sha256", "0" * 64), ("file_id", str(uuid4()))):
            old = source[field]
            source[field] = value
            with self.subTest(field=field), self.assertRaises(TaskError):
                await self.call()
            source[field] = old
        self.rows["edit-one"]["request_digest"] = "0" * 64
        with self.assertRaises(TaskError) as raised:
            await self.call()
        self.assertEqual(raised.exception.code, "media_operation_conflict")
        self.provider.stage_input_file.assert_awaited_once()
        self.broker.commands.prepare.assert_awaited_once()
        self.assertEqual(self.store.publish_recipe.await_count, 1)

    async def test_command_advancement_during_decode_blocks_recipe_publication(self):
        await self.fault_before_recipe()
        async def advanced(*args, **kwargs):
            receipt = await self.decode(*args, **kwargs)
            self.rows["edit-one"]["phase"] = "unknown"
            return receipt
        self.provider.decode_media_render_input.side_effect = advanced
        with self.assertRaises(TaskError) as raised:
            await self.call()
        self.assertEqual(raised.exception.code, "media_reconstruction_unavailable")
        self.assertEqual(self.store.publish_recipe.await_count, 1)
        self.broker.execute_media.assert_not_called()

    async def test_missing_retained_input_inspector_fails_before_effects(self):
        self.inspector.inspect_retained_input = None
        with self.assertRaises(TaskError) as raised:
            await self.call()
        self.assertEqual(raised.exception.code, "media_inspection_unavailable")
        self.files.read_file.assert_not_called()
        self.inputs.stage.assert_not_called()
        self.broker.allocation.assert_not_called()
        self.broker.commands.prepare.assert_not_called()
        self.provider.decode_media_render_input.assert_not_called()
        self.inspector.assert_not_called()


if __name__ == "__main__":
    unittest.main()
