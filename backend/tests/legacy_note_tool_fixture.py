from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, ClassVar
from uuid import UUID

"""Test-only legacy effect fixture. Never registered by the application."""
from app.notes.models import JarvisNote
from app.notes.repository import NoteRepository
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
class NoteToolModule:
    repository: NoteRepository
    name: ClassVar[str] = "notes"

    def build_tools(self, context: ToolBuildContext) -> list[ToolDefinition]:
        if context.session_id is None:
            return []
        return create_note_tools(
            repository=self.repository,
            owner_id=context.owner_id,
            session_id=context.session_id,
        )


def create_note_tools(*, repository: NoteRepository, owner_id: str, session_id: UUID) -> list[ToolDefinition]:
    bind_session = getattr(repository, "for_session", None)
    if bind_session is not None:
        repository = bind_session(session_id)
    search_limit = getattr(repository, "search_limit", 50)

    async def create_note(arguments: dict[str, Any]) -> dict[str, Any]:
        try:
            title = required_string(arguments, "title", max_length=200)
            body = required_string(arguments, "body", max_length=20000)
        except ToolInputError as error:
            return input_error_payload(error)
        note = await repository.create_note(owner_id=owner_id, session_id=session_id, title=title, body=body)
        return success({"note": _note_payload(note)})

    async def search_notes(arguments: dict[str, Any]) -> dict[str, Any]:
        try:
            query = optional_string(arguments, "query", max_length=500)
            limit = optional_positive_int(arguments, "limit", default=min(10, search_limit), max_value=search_limit)
        except ToolInputError as error:
            return input_error_payload(error)
        notes = await repository.search_notes(owner_id=owner_id, query=query, limit=limit)
        scope = getattr(notes, "knowledge_scope", None)
        return success({"notes": [_note_payload(note) for note in notes],
                        **({"knowledge_scope": scope} if scope is not None else {})})

    async def update_note(arguments: dict[str, Any]) -> dict[str, Any]:
        try:
            note_id = required_uuid(arguments, "note_id")
            title = optional_string(arguments, "title", max_length=200)
            body = optional_string(arguments, "body", max_length=20000)
            if title is None and body is None:
                return failed("invalid_tool_input", "title or body is required.")
        except ToolInputError as error:
            return input_error_payload(error)
        note = await repository.update_note(owner_id=owner_id, note_id=note_id, title=title, body=body)
        if note is None:
            return failed("note_not_found", "Note was not found.")
        return success({"note": _note_payload(note)})

    async def delete_note(arguments: dict[str, Any]) -> dict[str, Any]:
        try:
            note_id = required_uuid(arguments, "note_id")
        except ToolInputError as error:
            return input_error_payload(error)
        deleted = await repository.delete_note(owner_id=owner_id, note_id=note_id)
        if not deleted:
            return failed("note_not_found", "Note was not found.")
        return success({"deleted": True, "note_id": str(note_id)})

    tools = [
        ToolDefinition(
            name="create_note",
            mutates_state=True,
            description="Create a Jarvis-owned note for durable personal information. Use for free-form information the user may want later.",
            input_schema={
                "type": "object",
                "additionalProperties": False,
                "required": ["title", "body"],
                "properties": {
                    "title": {"type": "string"},
                    "body": {"type": "string"},
                },
            },
            handler=create_note,
        ),
        ToolDefinition(
            name="search_notes",
            description="Search Jarvis-owned notes by title/body substring. Omit query to list recent notes. Source-backed results may contain previews; use explain_memory with memory_source.source_id for the complete original.",
            input_schema={
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "query": {"type": ["string", "null"]},
                    "limit": {"type": ["integer", "null"], "minimum": 1, "maximum": search_limit},
                },
            },
            handler=search_notes,
        ),
        ToolDefinition(
            name="update_note",
            mutates_state=True,
            description="Update an existing Jarvis-owned note title and/or body.",
            input_schema={
                "type": "object",
                "additionalProperties": False,
                "required": ["note_id"],
                "properties": {
                    "note_id": {"type": "string"},
                    "title": {"type": ["string", "null"]},
                    "body": {"type": ["string", "null"]},
                },
            },
            handler=update_note,
        ),
        ToolDefinition(
            name="delete_note",
            mutates_state=True,
            description="Delete a Jarvis-owned note permanently.",
            input_schema={
                "type": "object",
                "additionalProperties": False,
                "required": ["note_id"],
                "properties": {"note_id": {"type": "string"}},
            },
            handler=delete_note,
        ),
    ]
    capture = getattr(repository, "capture_tool_result", None)
    if capture is not None:
        tools = [replace(tool, result_capture=capture, capture_after_commit=tool.mutates_state,
                         audit_arguments=lambda arguments: {"fields": sorted(arguments), "arguments_digest": digest(arguments)})
                 if tool.name != "delete_note" else tool for tool in tools]
    return tools


def _note_payload(note: JarvisNote) -> dict[str, Any]:
    result = {
        "id": str(note.id),
        "session_id": str(note.session_id) if note.session_id else None,
        "title": note.title,
        "body": note.body,
        "source": note.source,
        "created_at": note.created_at.isoformat(),
        "updated_at": note.updated_at.isoformat(),
        "metadata": note.metadata,
    }

    if note.memory_source is not None:
        # Full originals remain available through the existing paged source reader.
        # Bound search and durable receipts without silently truncating stored text.
        result.update(body=note.body[:1000], body_complete=len(note.body) <= 1000,
                      body_characters=len(note.body), metadata=None,
                      memory_source=note.memory_source,
                      meaning="Exact stored note; body may be a preview and metadata is omitted. "
                              "Use explain_memory for the full original. Storage is not independent proof "
                              "that its statement is true, nor proof that Hindsight ingestion has completed.")
        if note.task_evidence is not None:
            result["task_evidence"] = note.task_evidence
    return result
