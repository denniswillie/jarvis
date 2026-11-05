"""Trusted exact-note export through the existing retained-file lifecycle."""
from hashlib import sha256
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from app.files.publication import FilePublicationSpec, MAX_FILE_BYTES
from app.knowledge.models import SourceError
from app.knowledge.notes import SourceNoteRepository
from app.notes.workspace_content import canonical_content, content_for_note, render_body
from app.notes.workspace_repository import WorkspaceNoteRepository
from app.tasks.models import json_object
from app.tools.invocation import current_tool_invocation
from app.workspace.models import WorkspaceContext
from app.workspace.presentation import PreparedContent, PresentationRequest


class NoteExportBinding(BaseModel):
    """Reference-only producer metadata, never model-supplied authority."""
    model_config = ConfigDict(extra="forbid", frozen=True)
    request_id: UUID
    artifact_id: UUID
    version_id: str = Field(min_length=1, max_length=256)
    area_id: UUID
    source_id: UUID
    source_revision: int = Field(strict=True, ge=1)
    source_payload_id: UUID
    source_content_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    format: Literal["text", "json"]

    @property
    def filename(self):
        return f"note-{self.artifact_id}.{'txt' if self.format == 'text' else 'json'}"

    @property
    def content_type(self):
        return "text/plain; charset=utf-8" if self.format == "text" else "application/json"


async def validate_note_export_original(connection, sources, *, owner_id, area_id,
                                        spec, content, metadata_payload_id=None):
    """Recheck exact source/receipt inside the caller's owner transaction.

    Ordinary note corrections may advance generation without changing an exact
    historical version. Generation/currentness are not publication identity.
    """
    if not connection.is_in_transaction():
        raise RuntimeError("Note export validation requires the owner transaction")
    if (not isinstance(spec, FilePublicationSpec) or set(spec.metadata) != {"workspace_note_export"}
            or spec.source != "assistant" or spec.source_visual_frame_id is not None
            or spec.session_id is None or type(content) is not bytes or len(content) > MAX_FILE_BYTES):
        raise SourceError("workspace_export_manifest", "The file is not a supported exact note export.")
    binding = NoteExportBinding.model_validate(spec.metadata["workspace_note_export"])
    if (binding.area_id != area_id or spec.filename != binding.filename
            or spec.content_type != binding.content_type):
        raise SourceError("workspace_export_manifest", "Export metadata does not match its exact source scope.")
    row = await connection.fetchrow(
        """select session_id,payload from jarvis_workspace_requests
            where owner_id=$1 and request_id=$2 for share""", owner_id, binding.request_id)
    if row is None or row["session_id"] != spec.session_id:
        raise SourceError("workspace_export_not_admitted", "The owned presentation operation is unavailable.", 403)
    request = PresentationRequest.model_validate(json_object(row["payload"]))
    if (request.request_id, request.action, request.artifact_kind, request.artifact_id,
        request.version_id, request.format) != (
            binding.request_id, "download", "note", binding.artifact_id, binding.version_id, binding.format):
        raise SourceError("workspace_export_not_admitted", "The export differs from the admitted presentation.", 403)
    ctx = WorkspaceContext(owner_id, spec.session_id, binding.request_id)
    notes = WorkspaceNoteRepository(SourceNoteRepository(sources, owner_id=owner_id, area_id=area_id))
    snapshot = await notes.read_note_in_transaction(connection, ctx, binding.artifact_id, binding.version_id)
    source = snapshot.source
    if (source.source_id, source.revision, source.payload_id, source.content_digest) != (
            binding.source_id, binding.source_revision, binding.source_payload_id, binding.source_content_digest):
        raise SourceError("workspace_export_source_changed", "The exact note source is unavailable.")
    document = content_for_note(snapshot.note)
    text = render_body(document) if binding.format == "text" else canonical_content(document)
    if "\x00" in text or text.encode("utf-8") != content:
        raise SourceError("workspace_export_bytes_changed", "The bytes are not the exact requested note export.")
    if metadata_payload_id is not None and not await connection.fetchval(
            """select exists(select 1 from jarvis_context_payload_dependencies
                where owner_id=$1 and dependent_payload_id=$2 and source_payload_id=$3)""",
            owner_id, metadata_payload_id, binding.source_payload_id):
        raise SourceError("workspace_export_dependency_missing", "The retained export lacks its source-copy dependency.")
    return text


