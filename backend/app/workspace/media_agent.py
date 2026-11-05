"""Ordinary English to the existing managed media task owner, not a renderer."""
from app.live_state.admission import require_command, task_command_source
from app.tasks.media_editing import (
    MEDIA_TOOL, MediaInput, MediaInstruction, SavedRecordingReference,
    require_media_input, validate_media_input,
)
from app.tasks.models import TaskCreate, TaskError
from app.workspace.models import WorkspaceContext


class WorkspaceMediaAgent:
    def __init__(self, tasks, files, recordings):
        if (tasks.repository.pool is not recordings.sources.pool
                or files.repository.sources is not recordings.sources):
            raise ValueError("Media admission must share the existing task/source owners")
        self.tasks, self.files, self.recordings = tasks, files, recordings

    @staticmethod
    def _accepted(tasks, task, operation_id):
        return {"status": "accepted", "outcome": "accepted", "effect_committed": False,
                "operation_id": str(operation_id), "task_ref": tasks.reference(task.id),
                "task_revision": task.revision, "task_state": task.status.value,
                "summary": "Your video edit was accepted for background work. This receipt does not confirm a rendered or saved output."}

    async def submit(self, ctx, arguments, *, command_admission=None):
        if not isinstance(ctx, WorkspaceContext):
            raise TypeError("Media editing requires trusted WorkspaceContext")
        command = require_command(command_admission, ctx.owner_id, ctx.operation_id, "create")
        args = MediaInstruction.model_validate(arguments)
        if command.tool_name != MEDIA_TOOL or command.normalized_arguments != args.model_dump(mode="json", exclude_none=True):
            raise TaskError("media_command_mismatch", "The English request differs from its admitted call.", 403)
        if args.accepted_output is None and (args.recording_id is None or args.saved is None):
            return {"status": "success", "outcome": "clarification_required", "effect_committed": False,
                    "operation_id": str(ctx.operation_id),
                    "summary": "Clarify which recording you mean. Use the saved reference from recording status in this conversation; never ask for technical IDs or select the account's latest file."}
        request = TaskCreate(request_id=ctx.operation_id, title="Edit your video", requested_outcome=args.instruction,
            session_id=ctx.session_id, backend_kind="frontier", execution_environment="offline_sandbox", publish_artifacts=True)
        # Resolve neither current targets nor bytes/metadata for an exact replay.
        replay = await self.tasks.repository.replay_create(ctx.owner_id, request, mode="background",
            command_admission=command, managed_route="video_editing")
        if replay is not None:
            return self._accepted(self.tasks.repository, replay, ctx.operation_id)
        if args.accepted_output is not None or args.recording_profile is not None:
            from app.workspace.media_correction import resolve_retained_media_input
            pin = await resolve_retained_media_input(self, ctx, args)
            require_media_input(command, pin)
        else:
            file, metadata = await self.files.media_metadata(ctx.owner_id, ctx.session_id, args.saved.file_id, args.saved.version_id)
            # Metadata uses its normal source guards outside the task transaction.
            # Its result is revalidated on acceptance, never trusted across this gap.
            async with self.recordings.sources.transaction(ctx.owner_id) as connection:
                await self.tasks.repository.lock_owner(connection, ctx.owner_id)
                scope = await self.recordings.require_admitted_recording(connection, ctx.owner_id, ctx.session_id, args.recording_id)
                saved = await self.recordings.get_completion(connection, scope)
                if saved is None or any(getattr(saved, key) != getattr(args.saved, key) for key in SavedRecordingReference.model_fields):
                    raise TaskError("media_recording_reference_changed", "No exact saved recording matches this conversation's reference.", 409)
                source = await self.recordings.sources.resolve_document_in_transaction(
                    connection, ctx.owner_id, scope.area_id, args.saved.version_id, historical=True)
                pin = MediaInput(**args.saved.model_dump(), owner_id=ctx.owner_id, session_id=ctx.session_id,
                    recording_id=args.recording_id, area_id=scope.area_id, source_payload_id=source.payload_id,
                    receipt_payload_id=saved.receipt_payload_id, admission_payload_id=scope.admission_payload_id,
                    content_sha256=metadata["content_sha256"], size_bytes=file.size_bytes)
                require_media_input(command, pin)
                await validate_media_input(connection, ctx.owner_id, ctx.session_id, pin)
        request = request.model_copy(update={"knowledge_area_id": pin.area_id})
        task = await self.tasks.create_media(ctx.owner_id, request, mode="background",
            command_admission=command, media_input=pin)
        return self._accepted(self.tasks.repository, task, ctx.operation_id)

    async def receipt(self, ctx, operation_id):
        tasks = self.tasks.repository
        async with tasks.transaction(ctx.owner_id) as connection:
            receipt = await connection.fetchrow("select * from jarvis_task_requests where owner_id=$1 and request_id=$2",
                                                ctx.owner_id, operation_id)
            if receipt is None or receipt["operation"] != "create":
                return None
            task = await tasks.load(connection, ctx.owner_id, receipt["task_id"], lock="share")
            record = await task_command_source(tasks, connection, task, operation_id=operation_id, current=False)
            if (record is None or record.content["command"]["tool_name"] != MEDIA_TOOL
                    or task.session_id != ctx.session_id):
                return None
            # Publication/result normalization is a separate owner seam. Never
            # turn worker status/prose into a saved-output receipt here.
            return self._accepted(tasks, task, operation_id)
