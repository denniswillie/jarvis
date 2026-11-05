"""Exact retained-target resolution over existing media/source/task owners.

No rendering, publication, task mutation or selector ledger. Stored plan bytes
are read only after the caller's exact request replay check, outside task locks.
"""
import json
from uuid import UUID

from app.tasks.media_editing import (
    AcceptedMediaProof, AcceptedOutputTarget, CorrectedMediaInput, MediaInput,
    RetainedMediaInput, SavedRecordingReference, media_error, validate_media_input,
)
from app.tasks.models import json_object
from app.workspace.media_publication import MediaPublicationStore, SavedMediaPlan
from app.workspace.media_render import AcceptedMediaOutput


async def accepted_media_proof(connection, owner_id, area_id, task_id, action_id, plan_digest, reference, format, edit_id, *, lock=True):
    raw = await connection.fetchval(
        "select jarvis_media_accepted_proof($1,$2,$3,$4,$5,$6::jsonb,$7,$8,$9::boolean)",
        owner_id, area_id, task_id, action_id, plan_digest,
        reference.model_dump_json(), format, edit_id, lock)
    if raw is None:
        raise media_error()
    return AcceptedMediaProof.model_validate(json_object(raw))


async def _original_recording(agent, ctx, file_id, version_id, *, recording_id=None, expected_saved=None, expected_hash=None):
    # Current account/session access is checked independently of capture scope.
    file, metadata = await agent.files.media_metadata(ctx.owner_id, ctx.session_id, file_id, version_id)
    if expected_hash is not None and metadata["content_sha256"] != expected_hash:
        raise media_error()
    async with agent.recordings.sources.transaction(ctx.owner_id) as connection:
        await agent.tasks.repository.lock_owner(connection, ctx.owner_id)
        row = await connection.fetchrow("""select recording_id,session_id from jarvis_recordings
            where owner_id=$1 and file_id=$2 and version_id=$3
                and ($4::uuid is null or recording_id=$4) for share""",
            ctx.owner_id, file_id, version_id, recording_id)
        if row is None:
            raise media_error()
        # The capture session comes ONLY from the immutable owned recording.
        scope = await agent.recordings.require_admitted_recording(connection, ctx.owner_id,
            row["session_id"], row["recording_id"])
        saved = await agent.recordings.get_completion(connection, scope)
        if saved is None:
            raise media_error()
        reference = SavedRecordingReference(**{key: getattr(saved, key) for key in SavedRecordingReference.model_fields})
        if expected_saved is not None and reference != expected_saved:
            raise media_error()
        source = await agent.recordings.sources.resolve_document_in_transaction(
            connection, ctx.owner_id, scope.area_id, reference.version_id, historical=True)
        original = MediaInput(**reference.model_dump(), owner_id=ctx.owner_id, session_id=scope.session_id,
            recording_id=scope.recording_id, area_id=scope.area_id, source_payload_id=source.payload_id,
            receipt_payload_id=saved.receipt_payload_id, admission_payload_id=scope.admission_payload_id,
            content_sha256=metadata["content_sha256"], size_bytes=file.size_bytes)
        await validate_media_input(connection, ctx.owner_id, scope.session_id, original)
        return original


async def resolve_retained_media_input(agent, ctx, arguments):
    if arguments.accepted_output is None:
        if arguments.recording_profile != "retained_recording.v1" or arguments.saved is None:
            raise media_error()
        original = await _original_recording(agent, ctx, arguments.saved.file_id, arguments.saved.version_id,
            recording_id=arguments.recording_id, expected_saved=arguments.saved)
        return RetainedMediaInput(original=original, capture_session_id=original.session_id,
                                  command_session_id=ctx.session_id)
    target = arguments.accepted_output
    scoped = await agent.files.scoped(ctx.owner_id, ctx.session_id)
    store = MediaPublicationStore(agent.tasks.repository, scoped)
    output = await store.load_accepted_output(ctx.owner_id, target.task_id, target.render_plan_sha256)
    if (not isinstance(output, AcceptedMediaOutput) or output.owner_id != ctx.owner_id
            or AcceptedOutputTarget.from_output(output) != target):
        raise media_error()
    saved = await store.load_plan(ctx.owner_id, target.task_id, target.render_plan_sha256)
    if not isinstance(saved, SavedMediaPlan):
        raise media_error()
    recipe = saved.plan.recipe.recipe
    original_ref = recipe.sources[0].file
    original = await _original_recording(agent, ctx, original_ref.file_id, original_ref.version_id,
                                         expected_hash=original_ref.content_sha256)
    async with agent.recordings.sources.transaction(ctx.owner_id) as connection:
        await agent.tasks.repository.lock_owner(connection, ctx.owner_id)
        recipe_proof = await accepted_media_proof(connection, ctx.owner_id, original.area_id,
            target.task_id, saved.action_id, target.render_plan_sha256, saved.file, "json", recipe.edit_id)
        output_proof = await accepted_media_proof(connection, ctx.owner_id, original.area_id,
            target.task_id, target.action_id, target.render_plan_sha256, target.file, "mp4", recipe.edit_id)
        pin = CorrectedMediaInput(original=original, capture_session_id=original.session_id,
            command_session_id=ctx.session_id, target=target, previous_recipe=saved.file,
            edit_id=recipe.edit_id, recipe_action_id=saved.action_id,
            recipe_proof=recipe_proof, output_proof=output_proof)
        await validate_retained_media_input(connection, ctx.owner_id, ctx.session_id, pin)
        return pin


async def validate_retained_media_input(connection, owner_id, command_session_id, pin, *, lock=True):
    """Same-connection execution/publication recheck, independent of voice life."""
    if (pin.owner_id != owner_id or pin.command_session_id != command_session_id
            or pin.capture_session_id != pin.original.session_id):
        raise media_error()
    if not await connection.fetchval("select true from jarvis_sessions where owner_id=$1 and id=$2" + (" for share" if lock else ""),
                                     owner_id, command_session_id):
        raise media_error()
    await validate_media_input(connection, owner_id, pin.capture_session_id, pin.original, lock=lock)
    if isinstance(pin, CorrectedMediaInput):
        target = pin.target
        recipe = await accepted_media_proof(connection, owner_id, pin.area_id, target.task_id,
            pin.recipe_action_id, target.render_plan_sha256, pin.previous_recipe, "json", pin.edit_id, lock=lock)
        output = await accepted_media_proof(connection, owner_id, pin.area_id, target.task_id,
            target.action_id, target.render_plan_sha256, target.file, "mp4", pin.edit_id, lock=lock)
        if recipe != pin.recipe_proof or output != pin.output_proof:
            raise media_error()
        if not await connection.fetchval("select jarvis_media_correction_lineage($1,$2,$3,$4)",
                owner_id, pin.original.source_payload_id, recipe.source_payload_id, output.source_payload_id):
            raise media_error()
    return pin
