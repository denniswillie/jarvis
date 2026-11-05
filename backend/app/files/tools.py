from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, ClassVar
from uuid import UUID, uuid4

from app.files.models import JarvisFile
from app.files.repository import JarvisFileRepository
from app.files.storage import JarvisFileStorageClient
from app.tasks.models import digest
from app.tools.inputs import (
    ToolInputError,
    failed,
    input_error_payload,
    optional_positive_int,
    optional_string,
    required_string,
    required_uuid,
    success,
)
from app.tools.module import ToolBuildContext
from app.tools.registry import ToolDefinition


@dataclass(frozen=True)
class FileToolModule:
    repository: JarvisFileRepository
    storage: JarvisFileStorageClient
    name: ClassVar[str] = "files"

    def build_tools(self, context: ToolBuildContext) -> list[ToolDefinition]:
        if context.session_id is None:
            return []
        return create_file_tools(
            repository=self.repository,
            storage=self.storage,
            owner_id=context.owner_id,
            session_id=context.session_id,
        )


def create_file_tools(
    *,
    repository: JarvisFileRepository,
    storage: JarvisFileStorageClient,
    owner_id: str,
    session_id: UUID,
) -> list[ToolDefinition]:
    bind_session = getattr(repository, "for_session", None)
    if bind_session is not None:
        repository = bind_session(session_id)
    list_limit = getattr(repository, "list_limit", 100)

    async def create_text_file(arguments: dict[str, Any]) -> dict[str, Any]:
        try:
            filename, content, description = text_file_parameters(arguments)
        except ToolInputError as error:
            return input_error_payload(error)
        publish = getattr(repository, "create_text_file", None)
        if publish is not None:
            file = await publish(owner_id=owner_id, session_id=session_id, filename=filename,
                                 content=content, description=description)
            return success({"file": _file_payload(file)})
        file_id = uuid4()
        path = storage.build_file_path(owner_id=owner_id, session_id=session_id, file_id=file_id, filename=filename)
        stored = await storage.upload_bytes(
            path=path,
            content=content.encode("utf-8"),
            content_type="text/plain; charset=utf-8",
        )
        file = await repository.create_file(
            owner_id=owner_id,
            session_id=session_id,
            filename=filename,
            content_type="text/plain; charset=utf-8",
            storage_bucket=stored.bucket,
            storage_path=stored.path,
            size_bytes=stored.size_bytes,
            description=description,
            source="assistant",
        )
        return success({"file": _file_payload(file)})

    async def list_files(arguments: dict[str, Any]) -> dict[str, Any]:
        try:
            query = optional_string(arguments, "query", max_length=500)
            limit = optional_positive_int(arguments, "limit", default=min(20, list_limit), max_value=list_limit)
        except ToolInputError as error:
            return input_error_payload(error)
        files = await repository.list_files(owner_id=owner_id, query=query, limit=limit)
        scope = getattr(files, "knowledge_scope", None)
        return success({"files": [_file_payload(file) for file in files],
                        **({"knowledge_scope": scope} if scope is not None else {})})

    async def get_file_metadata(arguments: dict[str, Any]) -> dict[str, Any]:
        try:
            file_id = required_uuid(arguments, "file_id")
        except ToolInputError as error:
            return input_error_payload(error)
        file = await repository.get_file(owner_id=owner_id, file_id=file_id)
        if file is None:
            return failed("file_not_found", "File was not found.")
        return success({"file": _file_payload(file)})

    async def delete_file(arguments: dict[str, Any]) -> dict[str, Any]:
        try:
            file_id = required_uuid(arguments, "file_id")
        except ToolInputError as error:
            return input_error_payload(error)
        remove = getattr(repository, "delete_source_file", None)
        if remove is not None:
            return success(await remove(owner_id=owner_id, file_id=file_id))
        file = await repository.get_file(owner_id=owner_id, file_id=file_id)
        if file is None:
            return failed("file_not_found", "File was not found.")
        await storage.delete_object(bucket=file.storage_bucket, path=file.storage_path)
        await repository.delete_file(owner_id=owner_id, file_id=file_id)
        return success({"deleted": True, "file": _file_payload(file)})

    tools = [
        ToolDefinition(
            name="create_text_file",
            mutates_state=True,
            description="Create a retained Jarvis-managed text file. Returns metadata only, not raw bytes or signed URLs.",
            input_schema={
                "type": "object",
                "additionalProperties": False,
                "required": ["filename", "content"],
                "properties": {
                    "filename": {"type": "string"},
                    "content": {"type": "string"},
                    "description": {"type": ["string", "null"]},
                },
            },
            handler=create_text_file,
        ),
        ToolDefinition(
            name="list_files",
            description="List Jarvis-managed file metadata. Omit query to list recent files. Does not return raw bytes or signed URLs.",
            input_schema={
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "query": {"type": ["string", "null"]},
                    "limit": {"type": ["integer", "null"], "minimum": 1, "maximum": list_limit},
                },
            },
            handler=list_files,
        ),
        ToolDefinition(
            name="get_file_metadata",
            description="Get metadata for a Jarvis-managed file. Does not return raw bytes or signed URLs.",
            input_schema={
                "type": "object",
                "additionalProperties": False,
                "required": ["file_id"],
                "properties": {"file_id": {"type": "string"}},
            },
            handler=get_file_metadata,
        ),
        ToolDefinition(
            name="delete_file",
            mutates_state=True,
            description="Delete a Jarvis-managed file and its retained storage object.",
            input_schema={
                "type": "object",
                "additionalProperties": False,
                "required": ["file_id"],
                "properties": {"file_id": {"type": "string"}},
            },
            handler=delete_file,
        ),
    ]
    capture = getattr(repository, "capture_tool_result", None)
    if capture is not None:
        tools = [replace(tool, result_capture=capture, capture_after_commit=tool.mutates_state,
                         audit_arguments=lambda arguments: {"fields": sorted(arguments), "arguments_digest": digest(arguments)})
                 if tool.name != "delete_file" else tool for tool in tools]
    return tools


def text_file_parameters(arguments):
    """One normalization contract for foreground input and exact task grants."""
    return (required_string(arguments, "filename", max_length=240),
            required_string(arguments, "content", max_length=200000, preserve_whitespace=True),
            optional_string(arguments, "description", max_length=2000) or "Jarvis-created text file.")


def _file_payload(file: JarvisFile) -> dict[str, Any]:
    result = {
        "id": str(file.id),
        "session_id": str(file.session_id) if file.session_id else None,
        "filename": file.filename,
        "content_type": file.content_type,
        "storage_bucket": file.storage_bucket,
        "storage_path": file.storage_path,
        "size_bytes": file.size_bytes,
        "description": file.description,
        "source": file.source,
        "source_visual_frame_id": str(file.source_visual_frame_id) if file.source_visual_frame_id else None,
        "created_at": file.created_at.isoformat(),
        "metadata": file.metadata,
    }
    if file.memory_source is not None:
        result.pop("storage_bucket")
        result.pop("storage_path")
        result.update(metadata=None, memory_source=file.memory_source,
            meaning="Verified retained original; generated content is not independent proof of its claims. "
                    "Use explain_memory for the complete original. This is not a Hindsight ingestion or delivery receipt.")
    if file.task_evidence is not None:
        result["task_evidence"] = file.task_evidence
    return result
