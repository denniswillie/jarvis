from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from app.files.tools import _file_payload
from app.images import ImageUnavailableError
from app.tasks.models import digest
from app.files.repository import JarvisFileRepository
from app.files.storage import JarvisFileStorageClient
from app.settings import JarvisSettings
from app.tools.inputs import ToolInputError, failed, input_error_payload, optional_string, required_string, success
from app.tools.registry import ToolDefinition
from app.visual_memory.latest_frame_buffer import LatestVisualFrameBuffer
from app.visual_memory.repository import VisualFrameRepository


def create_capture_latest_visual_frame_tool(
    *,
    settings: JarvisSettings,
    latest_frame_buffer: LatestVisualFrameBuffer,
    visual_repository: VisualFrameRepository,
    file_repository: JarvisFileRepository,
    file_storage: JarvisFileStorageClient,
    owner_id: str,
    session_id: UUID,
) -> ToolDefinition:
    async def capture_latest_visual_frame(arguments: dict[str, Any]) -> dict[str, Any]:
        try:
            filename = optional_string(arguments, "filename", max_length=240) or "current-view.jpg"
            description = required_string(arguments, "description", max_length=2000)
        except ToolInputError as error:
            return input_error_payload(error)

        save = getattr(file_repository, "capture_current_frame", None)
        if save is not None:
            try:
                file = await save(owner_id=owner_id, session_id=session_id, filename=filename, description=description)
            except ImageUnavailableError as error:
                return failed(error.code, error.message)
            return success({"file": _file_payload(file), "source_frame_id": str(file.source_visual_frame_id)})
        frame = latest_frame_buffer.get(session_id)
        if frame is None:
            return failed("latest_visual_frame_not_available", "No recent visual frame is available to capture.")
        age_seconds = (datetime.now(UTC) - frame.server_received_at).total_seconds()
        if age_seconds > settings.VISUAL_LATEST_CAPTURE_MAX_AGE_SECONDS:
            return failed(
                "latest_visual_frame_too_old",
                f"Latest visual frame is older than {settings.VISUAL_LATEST_CAPTURE_MAX_AGE_SECONDS} seconds.",
            )

        file_id = uuid4()
        destination_path = file_storage.build_file_path(
            owner_id=owner_id,
            session_id=session_id,
            file_id=file_id,
            filename=filename,
        )
        stored = await file_storage.copy_object(
            source_bucket=frame.storage_bucket,
            source_path=frame.storage_path,
            destination_path=destination_path,
            content_type=frame.content_type,
        )
        file = await file_repository.create_file(
            owner_id=owner_id,
            session_id=session_id,
            filename=filename,
            content_type=frame.content_type,
            storage_bucket=stored.bucket,
            storage_path=stored.path,
            size_bytes=stored.size_bytes,
            description=description,
            source="visual_frame",
            source_visual_frame_id=frame.frame_id,
            metadata={"captured_frame_sequence": frame.frame_sequence},
        )
        return success({"file": _file_payload(file), "source_frame_id": str(frame.frame_id)})

    tool = ToolDefinition(
        name="capture_latest_visual_frame",
        mutates_state=True,
        description=(
            "Capture the current glasses view as a retained Jarvis-managed file. Use when the user asks to save, capture, "
            "or remember what they are looking at now. Requires a visual frame received in the last few seconds. "
            "Returns file metadata only, not image bytes or signed URLs."
        ),
        input_schema={
            "type": "object",
            "additionalProperties": False,
            "required": ["description"],
            "properties": {
                "filename": {"type": ["string", "null"]},
                "description": {"type": "string"},
            },
        },
        handler=capture_latest_visual_frame,
    )
    capture = getattr(file_repository, "capture_tool_result", None)
    if capture is not None:
        tool = replace(tool, result_capture=capture, capture_after_commit=True,
                       audit_arguments=lambda arguments: {"fields": sorted(arguments), "arguments_digest": digest(arguments)})
    return tool
