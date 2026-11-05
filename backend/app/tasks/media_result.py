"""Verify saved media outcomes without rendering, publishing or rewriting prose.

The ordinary action verifier supplies receipts. The media owner reopens their
exact accepted publications; a successful command or saved recipe is not an
edited video. This adds evidence to the existing result, not another ledger.
"""
import re

from app.tasks.models import TaskError


async def verified_media_result(context, bridge, status, receipts):
    media = getattr(bridge, "media", None)
    if media is None:
        return None

    from app.sandbox.media import ManagedMediaTools
    from app.workspace.media_recipe import MediaRecipeError
    from app.workspace.media_render import AcceptedMediaOutput

    if not isinstance(media, ManagedMediaTools) or media.context is not context:
        raise TaskError("task_media_result_scope", "The result has no matching managed media owner.")
    await media.bridge._check()
    outputs = []
    for receipt in receipts:
        operation = receipt.get("task_operation", {})
        action = media.bridge.actions.get(operation.get("operation_ref"))
        if action is None:
            raise TaskError("task_media_result_scope", "The result claims an unattached media action.")
        # This is the already-verified action's scope, not model-authored JSON.
        # The publication store performs the complete snapshot/source validation.
        request = action.parameters.get("request", {})
        if not isinstance(request, dict):
            raise TaskError("task_media_result_scope", "The publication scope is unavailable.")
        key = request.get("operation_key")
        if request.get("format") != "mp4":
            continue
        match = re.fullmatch(r"media-output:([0-9a-f]{64})", key or "")
        if match is None:
            raise TaskError("task_media_result_scope", "The MP4 is not a publication of this managed edit.")
        try:
            output = await media.store.load_accepted_output(
                context.task.owner_id, context.task.id, match.group(1))
        except MediaRecipeError as error:
            raise TaskError(error.code, error.message) from None
        file = receipt.get("file")
        source = file.get("memory_source") if isinstance(file, dict) else None
        if (not isinstance(output, AcceptedMediaOutput) or not isinstance(source, dict)
                or receipt.get("status") != "success"
                or output.owner_id != context.task.owner_id or output.task_id != context.task.id
                or output.render_plan_sha256 != match.group(1)
                or str(output.action_id) != operation.get("action_id")
                or str(output.file.file_id) != file.get("id")
                or output.file.version_id != source.get("document_id")
                or output.content_type != file.get("content_type")
                or output.size_bytes != file.get("size_bytes")):
            raise TaskError("task_media_result_unconfirmed",
                            "No exact accepted edited video matches the claimed publication.")
        outputs.append(output.model_dump(mode="json"))
    await media.bridge._check()
    if status == "succeeded" and not outputs:
        raise TaskError("task_media_result_unconfirmed",
                        "A video-editing task cannot succeed without a confirmed saved output.")
    return {"format": "jarvis.media.result_evidence.v1", "outputs": outputs,
            "meaning": "Accepted saved outputs, not proof of creative quality, phone playback or external delivery."}
