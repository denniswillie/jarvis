"""Prepared offline contracts, not SQL/source/storage/renderer qualification.

All source rows, publication-store reads and accepted-action proofs below are
explicit substitutes. No renderer inventory or media decoding is asserted.
"""
import copy
from pathlib import Path
import re
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch
from uuid import UUID

from pydantic import ValidationError

from app.live_state.admission import require_command_create
from app.tasks.media_editing import (
    MEDIA_TOOL, AcceptedMediaProof, AcceptedOutputTarget, CorrectedMediaInput,
    MediaInstruction, RetainedMediaInput, parse_media_input, require_media_input,
    require_media_recipe_input, validate_media_input,
)
from app.tasks.models import TaskError
from app.workspace.media_correction import (
    _original_recording, accepted_media_proof, resolve_retained_media_input,
    validate_retained_media_input,
)
from app.workspace.media_publication import SavedMediaPlan
from app.workspace.media_recipe import MediaEditRecipeV1, RetainedRef
from app.workspace.media_render import AcceptedMediaOutput
from app.workspace.models import WorkspaceContext
from tests import test_lean_media_admission as base


B = UUID(int=900)
PRIOR_TASK, OUTPUT_ACTION, OUTPUT_FILE, RECIPE_FILE, EDIT, RECIPE_ACTION = [UUID(int=i) for i in range(201, 207)]
PLAN_HASH = "c" * 64


def output(**changes):
    return AcceptedMediaOutput(**(dict(owner_id=base.OWNER, task_id=PRIOR_TASK,
        action_id=OUTPUT_ACTION, render_plan_sha256=PLAN_HASH,
        file=RetainedRef(file_id=OUTPUT_FILE, version_id="output-v1", content_sha256="b" * 64),
        content_type="video/mp4", size_bytes=2000) | changes))


def target():
    return AcceptedOutputTarget.from_output(output())


def proof(action, offset):
    return AcceptedMediaProof(action_id=action, grant_id=UUID(int=offset),
        scope_payload_id=UUID(int=offset+1), receipt_payload_id=UUID(int=offset+2),
        parent_scope_payload_id=UUID(int=offset+3), source_payload_id=UUID(int=offset+4),
        metadata_payload_id=UUID(int=offset+5), parameters_digest="f" * 64)


def retained():
    return RetainedMediaInput(original=base.pin(), capture_session_id=base.SESSION,
                              command_session_id=B)


def corrected():
    return CorrectedMediaInput(original=base.pin(), capture_session_id=base.SESSION,
        command_session_id=B, target=target(),
        previous_recipe=RetainedRef(file_id=RECIPE_FILE, version_id="recipe-v1", content_sha256=PLAN_HASH),
        edit_id=EDIT, recipe_action_id=RECIPE_ACTION,
        recipe_proof=proof(RECIPE_ACTION, 300), output_proof=proof(OUTPUT_ACTION, 400))


def args(**changes):
    return dict(instruction="Keep the ending two seconds longer, retaining the earlier choices.",
                accepted_output=target().model_dump(mode="json")) | changes


def command(arguments=None, **changes):
    return base.command(**(dict(session_id=B, normalized_arguments=args() if arguments is None else arguments) | changes))


def recipe():
    pin = corrected()
    return dict(schema_version=1, edit_id=str(pin.edit_id), previous_recipe=pin.previous_recipe.model_dump(mode="json"),
        template={"id": "jarvis.clip.v1", "version": "synthetic", "bundle_sha256": "b" * 64},
        sources=[{"source_id": "clip", "file": {"file_id": str(base.FILE), "version_id": base.pin().version_id,
            "content_sha256": base.pin().content_sha256}}],
        segments=[{"segment_id": "one", "source_id": "clip", "in_ms": 0, "out_ms": 1000}],
        captions=[], output_profile="portrait_720p24_v1", fit="contain", audio="preserve")


class CorrectionShapeTests(unittest.TestCase):
    def test_old_recording_shape_and_fingerprint_unchanged(self):
        normalized = MediaInstruction.model_validate(base.arguments()).model_dump(mode="json", exclude_none=True)
        self.assertEqual(normalized, base.arguments())
        self.assertEqual(base.command(normalized_arguments=normalized).arguments_digest, base.command().arguments_digest)
        self.assertEqual(require_media_input(base.command(), base.pin()), base.pin())

    def test_exact_target_comes_from_accepted_output_not_a_project_head(self):
        self.assertEqual(MediaInstruction.model_validate(args()).accepted_output, target())
        self.assertEqual(set(target().model_dump()), {"task_id", "action_id", "render_plan_sha256", "file"})
        self.assertEqual(require_media_input(command(), corrected()), corrected())
        c = command()
        require_command_create(c, base.request(c, session_id=B, requested_outcome=args()["instruction"]), mode="background")

    def test_targets_mutually_exclusive_and_model_cannot_supply_authority(self):
        for change in ({"recording_id": str(base.RECORDING)}, {"saved": base.arguments()["saved"]},
                       {"recording_profile": "retained_recording.v1"}, {"capture_session_id": str(base.SESSION)},
                       {"command_session_id": str(B)}, {"recipe": recipe()}, {"grant_ids": []}, {"owner_id": base.OWNER}):
            with self.subTest(change=change), self.assertRaises(ValidationError):
                MediaInstruction.model_validate(args(**change))
        for change in ({"task_id": None}, {"action_id": None}, {"render_plan_sha256": "latest"},
                       {"project_head": "current"}):
            with self.subTest(change=change), self.assertRaises(ValidationError):
                MediaInstruction.model_validate(args(accepted_output=target().model_dump(mode="json") | change))

    def test_later_session_recording_requires_explicit_profile(self):
        arguments = base.arguments(recording_profile="retained_recording.v1")
        self.assertEqual(require_media_input(command(arguments), retained()), retained())
        with self.assertRaises(TaskError):
            require_media_input(command(base.arguments()), retained())
        with self.assertRaises(TaskError):
            require_media_input(command(base.arguments()), base.pin())
        with self.assertRaises(TaskError):
            require_media_input(command(arguments), corrected())

    def test_capture_and_command_sessions_are_distinct_and_round_trip(self):
        for value in (retained(), corrected()):
            restored = parse_media_input(value.model_dump(mode="json"))
            self.assertEqual(restored, value)
            self.assertEqual(restored.capture_session_id, base.SESSION)
            self.assertEqual(restored.original.session_id, base.SESSION)
            self.assertEqual(restored.command_session_id, B)
            self.assertEqual(restored.session_id, B)
        with self.assertRaises(ValidationError):
            RetainedMediaInput(original=base.pin(), capture_session_id=B, command_session_id=B)

    def test_wrong_owner_command_or_exact_output_rejected(self):
        for c in (command(owner_id="other"), command(session_id=UUID(int=901)),
                  command(args(accepted_output=target().model_dump(mode="json") | {"action_id": str(UUID(int=902))}))):
            with self.subTest(command=c), self.assertRaises(TaskError):
                require_media_input(c, corrected())

    def test_predecessor_and_action_proofs_are_immutable_exact_bindings(self):
        original = corrected().model_dump(mode="json")
        for change in ({"recipe_action_id": str(OUTPUT_ACTION)},
                       {"previous_recipe": original["previous_recipe"] | {"content_sha256": "d" * 64}},
                       {"output_proof": original["output_proof"] | {"action_id": str(RECIPE_ACTION)}}):
            with self.subTest(change=change), self.assertRaises(ValidationError):
                CorrectedMediaInput.model_validate(original | change)
        expected = {*base.pin().dependencies, *corrected().recipe_proof.dependencies, *corrected().output_proof.dependencies}
        self.assertEqual(set(corrected().dependencies), expected)

    def test_same_raw_call_conflicts_and_new_intent_is_new_bookkeeping(self):
        first = command()
        changed = command(args(instruction="Keep the entire ending."))
        fresh = command(args(instruction="Keep the entire ending."), provider_call_id="second real provider call")
        self.assertEqual(first.operation_id, changed.operation_id)
        self.assertNotEqual(first.arguments_digest, changed.arguments_digest)
        self.assertNotEqual(first.operation_id, fresh.operation_id)
        self.assertEqual(fresh.normalized_arguments["accepted_output"], first.normalized_arguments["accepted_output"])


class CorrectionAdapterTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        base.MediaAdapterTests.setUp(self)
        self.ctx = WorkspaceContext(base.OWNER, B, command().operation_id)
        self.recordings.require_admitted_recording.return_value = SimpleNamespace(
            area_id=base.AREA, admission_payload_id=base.ADMISSION,
            session_id=base.SESSION, recording_id=base.RECORDING)
        self.c.fetchrow.return_value = {"recording_id": base.RECORDING, "session_id": base.SESSION}
        self.files.scoped = AsyncMock(return_value=object())
        self.returned_task = base.task()
        self.returned_task.session_id = B
        self.tasks.create_media.return_value = self.returned_task

    async def test_replay_before_any_retained_target_resolution(self):
        self.repo.replay_create.return_value = self.returned_task
        with patch("app.workspace.media_correction.resolve_retained_media_input", AsyncMock(side_effect=AssertionError("no selection on replay"))) as resolve:
            result = await self.agent.submit(self.ctx, args(), command_admission=command())
        self.assertEqual(result["outcome"], "accepted")
        self.assertFalse(result["effect_committed"])
        resolve.assert_not_awaited()
        self.files.media_metadata.assert_not_awaited()
        self.files.scoped.assert_not_awaited()
        self.tasks.create_media.assert_not_awaited()

    async def test_correction_forwards_same_service_and_preserves_prior_context_pointer(self):
        value = corrected()
        with patch("app.workspace.media_correction.resolve_retained_media_input", AsyncMock(return_value=value)) as resolve:
            result = await self.agent.submit(self.ctx, args(), command_admission=command())
        self.assertEqual(result["outcome"], "accepted")
        self.assertFalse(result["effect_committed"])
        resolve.assert_awaited_once()
        request = self.tasks.create_media.await_args.args[1]
        self.assertEqual(request.session_id, B)
        self.assertEqual(request.requested_outcome, args()["instruction"])
        self.assertEqual(request.request_id, command().operation_id)
        pin = self.tasks.create_media.await_args.kwargs["media_input"]
        self.assertEqual(pin.target.task_id, PRIOR_TASK)  # existing canonical task context, not copied prose
        self.assertEqual(pin.original.session_id, base.SESSION)
        self.assertEqual(pin.command_session_id, B)
        self.assertEqual(pin.edit_id, EDIT)
        self.assertEqual(pin.previous_recipe, corrected().previous_recipe)

    async def test_capture_a_is_server_resolved_while_metadata_access_uses_b(self):
        with patch("app.workspace.media_correction.validate_media_input", AsyncMock(return_value=base.pin())) as validate:
            value = await _original_recording(self.agent, self.ctx, base.FILE, base.pin().version_id,
                recording_id=base.RECORDING,
                expected_saved=MediaInstruction.model_validate(base.arguments()).saved)
        self.assertEqual(value, base.pin())
        self.files.media_metadata.assert_awaited_once_with(base.OWNER, B, base.FILE, base.pin().version_id)
        self.recordings.require_admitted_recording.assert_awaited_once_with(self.c, base.OWNER, base.SESSION, base.RECORDING)
        validate.assert_awaited_once_with(self.c, base.OWNER, base.SESSION, base.pin())

    async def test_missing_capture_or_withdrawn_capture_never_creates_work(self):
        self.c.fetchrow.return_value = None
        with self.assertRaises(TaskError):
            await _original_recording(self.agent, self.ctx, base.FILE, base.pin().version_id)
        self.recordings.require_admitted_recording.assert_not_awaited()
        self.tasks.create_media.assert_not_awaited()
        self.c.fetchrow.return_value = {"recording_id": base.RECORDING, "session_id": base.SESSION}
        self.recordings.require_admitted_recording.side_effect = TaskError("source_unavailable", "Synthetic withdrawal", 403)
        with self.assertRaises(TaskError):
            await _original_recording(self.agent, self.ctx, base.FILE, base.pin().version_id)
        self.tasks.create_media.assert_not_awaited()

    async def test_changed_original_hash_fails_before_recording_lookup(self):
        with self.assertRaises(TaskError):
            await _original_recording(self.agent, self.ctx, base.FILE, base.pin().version_id, expected_hash="d" * 64)
        self.c.fetchrow.assert_not_awaited()
        self.tasks.create_media.assert_not_awaited()

    async def test_accepted_output_and_saved_plan_resolve_original_not_output_as_input(self):
        value = corrected()
        old_recipe = MediaEditRecipeV1.model_validate(recipe() | {"previous_recipe": None})
        # The immutable recipe is real; store bytes/plan authority are explicitly mocked.
        saved = SavedMediaPlan(plan=SimpleNamespace(recipe=SimpleNamespace(recipe=old_recipe)),
                               file=value.previous_recipe, action_id=RECIPE_ACTION)
        store = SimpleNamespace(load_accepted_output=AsyncMock(return_value=output()), load_plan=AsyncMock(return_value=saved))
        with patch("app.workspace.media_correction.MediaPublicationStore", return_value=store), \
             patch("app.workspace.media_correction._original_recording", AsyncMock(return_value=base.pin())) as original, \
             patch("app.workspace.media_correction.accepted_media_proof", AsyncMock(side_effect=[value.recipe_proof, value.output_proof])), \
             patch("app.workspace.media_correction.validate_retained_media_input", AsyncMock(return_value=value)):
            resolved = await resolve_retained_media_input(self.agent, self.ctx, MediaInstruction.model_validate(args()))
        self.assertEqual(resolved, value)
        self.files.scoped.assert_awaited_once_with(base.OWNER, B)
        store.load_accepted_output.assert_awaited_once_with(base.OWNER, PRIOR_TASK, PLAN_HASH)
        store.load_plan.assert_awaited_once_with(base.OWNER, PRIOR_TASK, PLAN_HASH)
        original.assert_awaited_once_with(self.agent, self.ctx, base.FILE, base.pin().version_id,
                                         expected_hash=base.pin().content_sha256)

    async def test_unaccepted_wrong_owner_or_changed_output_never_resolves_capture(self):
        for actual in (None, output(owner_id="other"), output(action_id=UUID(int=999)),
                       output(file=output().file.model_copy(update={"version_id": "other-version"}))):
            store = SimpleNamespace(load_accepted_output=AsyncMock(return_value=actual), load_plan=AsyncMock())
            with self.subTest(output=actual), patch("app.workspace.media_correction.MediaPublicationStore", return_value=store), \
                 patch("app.workspace.media_correction._original_recording", AsyncMock()) as original, self.assertRaises(TaskError):
                await resolve_retained_media_input(self.agent, self.ctx, MediaInstruction.model_validate(args()))
            store.load_plan.assert_not_awaited()
            original.assert_not_awaited()


class CorrectionDurableTests(unittest.IsolatedAsyncioTestCase):
    async def test_legacy_source_lock_false_is_readonly_and_default_still_locks(self):
        for lock in (False, True):
            c = base.connection()
            c.fetchrow.return_value = base.source_row()
            if lock:
                actual = await validate_media_input(c, base.OWNER, base.SESSION, base.pin())
            else:
                actual = await validate_media_input(c, base.OWNER, base.SESSION, base.pin(), lock=False)
            self.assertEqual(actual, base.pin())
            self.assertEqual("for share" in c.fetchrow.await_args.args[0].lower(), lock)

    async def test_continuation_source_lock_false_reaches_every_proof(self):
        value = corrected()
        for lock in (False, True):
            c = base.connection()
            with patch("app.workspace.media_correction.validate_media_input", AsyncMock(return_value=value.original)) as original, \
                 patch("app.workspace.media_correction.accepted_media_proof", AsyncMock(side_effect=[value.recipe_proof, value.output_proof])) as proof_read:
                if lock:
                    actual = await validate_retained_media_input(c, base.OWNER, B, value)
                else:
                    actual = await validate_retained_media_input(c, base.OWNER, B, value, lock=False)
            self.assertEqual(actual, value)
            self.assertEqual("for share" in c.fetchval.await_args_list[0].args[0].lower(), lock)
            original.assert_awaited_once_with(c, base.OWNER, base.SESSION, value.original, lock=lock)
            self.assertEqual([call.kwargs for call in proof_read.await_args_list], [{"lock": lock}, {"lock": lock}])

    async def test_retained_dispatch_forwards_lock_false(self):
        c, value = base.connection(), corrected()
        with patch("app.workspace.media_correction.validate_retained_media_input", AsyncMock(return_value=value)) as validate:
            self.assertEqual(await validate_media_input(c, base.OWNER, B, value, lock=False), value)
        validate.assert_awaited_once_with(c, base.OWNER, B, value, lock=False)
        c.fetchrow.assert_not_awaited()

    async def test_accepted_proof_reader_forwards_explicit_readonly_boolean(self):
        c, value = base.connection(), corrected()
        c.fetchval.return_value = value.output_proof.model_dump(mode="json")
        await accepted_media_proof(c, base.OWNER, base.AREA, PRIOR_TASK, OUTPUT_ACTION,
            PLAN_HASH, target().file, "mp4", EDIT, lock=False)
        self.assertEqual(c.fetchval.await_args.args[-1], False)
        self.assertIn("$9::boolean", c.fetchval.await_args.args[0])

    async def test_current_command_session_and_original_capture_rechecked_after_disconnect(self):
        c, value = base.connection(), corrected()
        with patch("app.workspace.media_correction.validate_media_input", AsyncMock(return_value=value.original)) as original, \
             patch("app.workspace.media_correction.accepted_media_proof", AsyncMock(side_effect=[value.recipe_proof, value.output_proof])) as proof_read:
            self.assertEqual(await validate_retained_media_input(c, base.OWNER, B, value), value)
        original.assert_awaited_once_with(c, base.OWNER, base.SESSION, value.original, lock=True)
        self.assertEqual(proof_read.await_count, 2)
        self.assertEqual(c.fetchval.await_args_list[0].args[1:], (base.OWNER, B))
        self.assertEqual(c.fetchval.await_args.args[1:], (base.OWNER, base.ORIGINAL,
            value.recipe_proof.source_payload_id, value.output_proof.source_payload_id))

    async def test_wrong_owner_or_command_session_denied_before_database(self):
        for owner, session in (("other", B), (base.OWNER, base.SESSION)):
            c = base.connection()
            with self.assertRaises(TaskError):
                await validate_retained_media_input(c, owner, session, corrected())
            c.fetchval.assert_not_awaited()
            c.fetchrow.assert_not_awaited()

    async def test_fabricated_capture_fails_original_row_recheck(self):
        fake_session = UUID(int=990)
        value = RetainedMediaInput(original=base.pin(session_id=fake_session),
            capture_session_id=fake_session, command_session_id=B)
        c = base.connection()
        c.fetchrow.return_value = base.source_row()
        with self.assertRaises(TaskError):
            await validate_media_input(c, base.OWNER, B, value)
        self.assertEqual(c.fetchrow.await_args.args[1:], (base.OWNER, fake_session, base.RECORDING))

    async def test_withdrawn_original_or_accepted_proof_denied(self):
        value = corrected()
        c = base.connection()
        c.fetchrow.return_value = None
        with self.assertRaises(TaskError):
            await validate_media_input(c, base.OWNER, B, value)
        for proofs in (TaskError("source_unavailable", "Synthetic withdrawal", 403),
                       [value.recipe_proof.model_copy(update={"parameters_digest": "e" * 64}), value.output_proof]):
            with patch("app.workspace.media_correction.validate_media_input", AsyncMock(return_value=value.original)), \
                 patch("app.workspace.media_correction.accepted_media_proof", AsyncMock(side_effect=proofs)), self.assertRaises(TaskError):
                await validate_retained_media_input(c, base.OWNER, B, value)

    async def test_missing_current_session_or_broken_dependency_lineage_denied(self):
        value = corrected()
        for answers in ([False], [True, False]):
            c = base.connection()
            c.fetchval.side_effect = answers
            with patch("app.workspace.media_correction.validate_media_input", AsyncMock(return_value=value.original)), \
                 patch("app.workspace.media_correction.accepted_media_proof", AsyncMock(side_effect=[value.recipe_proof, value.output_proof])), self.assertRaises(TaskError):
                await validate_retained_media_input(c, base.OWNER, B, value)

    async def test_sql_proof_reader_is_exact_and_fails_closed(self):
        c, value = base.connection(), corrected()
        c.fetchval.return_value = value.output_proof.model_dump(mode="json")
        self.assertEqual(await accepted_media_proof(c, base.OWNER, base.AREA, PRIOR_TASK, OUTPUT_ACTION,
            PLAN_HASH, target().file, "mp4", EDIT), value.output_proof)
        self.assertEqual(c.fetchval.await_args.args[1:], (base.OWNER, base.AREA, PRIOR_TASK, OUTPUT_ACTION,
            PLAN_HASH, target().file.model_dump_json(), "mp4", EDIT, True))
        c.fetchval.return_value = None
        with self.assertRaises(TaskError):
            await accepted_media_proof(c, base.OWNER, base.AREA, PRIOR_TASK, OUTPUT_ACTION,
                PLAN_HASH, target().file, "mp4", EDIT)

    async def test_recipe_guard_pins_original_predecessor_and_edit_for_fresh_and_recovery(self):
        value, c = corrected(), base.connection()
        record = SimpleNamespace(content={"operation": "create", "command": {"tool_name": MEDIA_TOOL},
                                          "media_input": value.model_dump(mode="json")})
        instruction = SimpleNamespace(event_id=base.EVENT, revision=1)
        tasks = SimpleNamespace(_source_available=AsyncMock(return_value=False))
        with patch("app.tasks.instruction_source.task_instruction_source", AsyncMock(return_value=instruction)), \
             patch("app.live_state.admission.task_command_source", AsyncMock(return_value=record)):
            for phase in ("fresh", "recovery"):
                with self.subTest(phase=phase):
                    self.assertEqual(await require_media_recipe_input(tasks, c, base.task(), recipe()), value)
            changes = [dict(recipe(), previous_recipe=None), dict(recipe(), edit_id=str(UUID(int=991))),
                       dict(recipe(), previous_recipe=target().file.model_dump(mode="json"))]
            for field, replacement in (("file_id", str(OUTPUT_FILE)), ("version_id", "changed"), ("content_sha256", "d" * 64)):
                changed = copy.deepcopy(recipe())
                changed["sources"][0]["file"][field] = replacement
                changes.append(changed)
            for changed in changes:
                with self.subTest(recipe=changed), self.assertRaises(TaskError):
                    await require_media_recipe_input(tasks, c, base.task(), changed)


class CorrectionMigrationSourceTests(unittest.TestCase):
    def test_forward_migration_shares_legacy_checks_without_duplicate_predicates(self):
        # Text contract only; migration/constraint acceptance remains Core's SQL lane.
        root = Path(__file__).parents[1] / "app" / "migrations" / "sql"
        old = (root / "0049_media_command_grants.sql").read_text()
        new = (root / "0050_media_correction_commands.sql").read_text()
        pattern = r"create(?: or replace)? function jarvis_media_command_grant_valid\b.*?\$\$;"
        legacy = re.search(pattern, old, re.I | re.S)
        merged = re.search(pattern, new, re.I | re.S)
        self.assertIsNotNone(legacy)
        self.assertIsNotNone(merged)
        legacy, merged = legacy.group(0), merged.group(0)
        # Entire common grant-family tail and immutable task receipt join stay exact.
        marker = "    if exists(select 1 from jarvis_task_grants tg"
        self.assertEqual(legacy[legacy.index(marker):], merged[merged.index(marker):])
        start = "    select bt.*,d.content as description into t"
        end = "    select rec.*,v.payload_id as source_payload_id"
        self.assertEqual(legacy[legacy.index(start):legacy.index(end)],
                         merged[merged.index(start):merged.index(end)])
        # The source joins/availability predicates stay shared; only target scope varies.
        start = "    from jarvis_recordings rec"
        end = "    where rec.owner_id=g.owner_id"
        self.assertEqual(legacy[legacy.index(start):legacy.index(end)],
                         merged[merged.index(start):merged.index(end)])
        start = "        and a.status='active' and s.status='active'"
        end = "    for share of rec,a,v,s,p,cp,ap,fp,fpp,ce,ae;"
        self.assertEqual(legacy[legacy.index(start):legacy.index(end)+len(end)],
                         merged[merged.index(start):merged.index(end)+len(end)])
        self.assertIn("if not(pin ? 'profile') then", merged)
        self.assertIn("capture_session:=e.session_id::text;", merged)
        self.assertIn("original_pin:=pin;", merged)
        self.assertIn("(c->'arguments')-array['instruction','recording_id','saved']<>'{}'::jsonb", merged)
        self.assertIn("if not(pin ? 'profile') and c#>'{arguments,saved}' is distinct from", merged)
        self.assertIn("if original_pin is distinct from jsonb_build_object", merged)
        self.assertNotIn("jarvis_media_recording_command_grant_valid_v1", new)
        self.assertNotIn("jarvis_media_retained_command_grant_valid", new)
        self.assertEqual(len(re.findall(pattern, new, re.I | re.S)), 1)
        for function in ("jarvis_media_accepted_proof", "jarvis_media_correction_lineage"):
            self.assertIn(function, new)
        for forbidden in ("create table", "drop table", "create or replace function guard_jarvis_explicit_request"):
            self.assertNotIn(forbidden, new.lower())
        for binding in ("capture_session_id", "command_session_id", "retained_recording.v1", "accepted_output.v1"):
            self.assertIn(binding, new)
        self.assertIn("lock_rows boolean default true", new)
        self.assertIn("execute lookup_sql || (case when lock_rows then", new)
        self.assertIn("task-generated-file:", new)
        self.assertIn("edit-", new)


if __name__ == "__main__":
    unittest.main()
