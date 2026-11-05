"""Exact note-version export preparation for the trusted file-publication seam.

This does not grant publication authority or upload bytes. Core supplies the
foreground publisher, revalidates source access during adoption, and reconciles
the stable operation key through the existing file-publication service.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Literal, Protocol
from uuid import UUID

from app.errors import JarvisHTTPError
from app.notes.workspace_content import canonical_content, content_for_note, render_body
from app.workspace.models import WorkspaceContext
from app.workspace.presentation import PreparedContent, PresentationRequest


@dataclass(frozen=True, slots=True)
class NoteExport:
    """Trusted preparation data, not a new public API or model-authored payload."""
    owner_id: str
    session_id: UUID
    request_id: UUID
    artifact_id: UUID
    version_id: str
    area_id: UUID
    source_id: UUID
    source_revision: int
    source_payload_id: UUID
    source_content_digest: str
    format: Literal["text", "json"]
    content: bytes

    @property
    def operation_key(self):
        return "workspace-note-export:" + str(self.request_id)

    @property
    def filename(self):
        return f"note-{self.artifact_id}.{'txt' if self.format == 'text' else 'json'}"

    @property
    def content_type(self):
        return "text/plain; charset=utf-8" if self.format == "text" else "application/json"

    @property
    def content_sha256(self):
        return sha256(self.content).hexdigest()


class NoteExportPublisher(Protocol):
    async def __call__(self, ctx: WorkspaceContext, export: NoteExport) -> PreparedContent:
        """Reconcile SAME operation; return only durably adopted exact bytes.

        Must enforce foreground admission, owned session/area and exact source
        revision/payload; retain source-copy dependencies and the 8 MiB ceiling.
        Unknown upload outcomes never authorize a second publication identity.
        """
        ...


class ExactNoteExporter:
    def __init__(self, notes, publisher: NoteExportPublisher):
        self.notes, self.publisher = notes, publisher

    async def __call__(self, ctx: WorkspaceContext, request: PresentationRequest) -> PreparedContent:
        if (request.artifact_kind != "note" or request.action != "download"
                or request.format not in {"text", "json"}):
            raise JarvisHTTPError(422, "unsupported_note_export", "This producer exports saved notes only.",
                                 details={"effect_committed": False})
        if ctx.operation_id != request.request_id:
            raise JarvisHTTPError(403, "workspace_export_scope", "Export must use the admitted presentation operation.",
                                 details={"effect_committed": False})
        snapshot = await self.notes.read_snapshot(ctx, request.artifact_id, request.version_id)
        source, note = snapshot.source, snapshot.note
        if (source.owner_id != ctx.owner_id or source.document_id != request.version_id
                or note.id != request.artifact_id):
            raise JarvisHTTPError(409, "workspace_export_version", "The requested exact note original is unavailable.",
                                 details={"effect_committed": False})
        content = content_for_note(note)
        raw = (render_body(content) if request.format == "text" else canonical_content(content)).encode("utf-8")
        if len(raw) > 8 * 1024 * 1024:
            raise JarvisHTTPError(422, "workspace_export_too_large", "This export exceeds the retained-file bound.",
                                 details={"effect_committed": False})
        prepared = NoteExport(ctx.owner_id, ctx.session_id, request.request_id, note.id, source.document_id,
            source.area_id, source.source_id, source.revision, source.payload_id,
            source.content_digest, request.format, raw)
        receipt = PreparedContent.model_validate(await self.publisher(ctx, prepared))
        if (receipt.byte_count, receipt.content_sha256) != (len(raw), prepared.content_sha256):
            # Publication may already have happened; never label this no-effect.
            raise JarvisHTTPError(409, "workspace_export_receipt_mismatch",
                                 "Publication did not confirm the exact exported bytes.")
        return receipt

from app.files.publication import FilePublicationSpec, MAX_FILE_BYTES
from app.knowledge.models import SourceError
from app.files.workspace_exports import NoteExportBinding

def note_export_spec(ctx, export):
    if (not isinstance(ctx, WorkspaceContext) or not isinstance(export, NoteExport)
            or ctx.operation_id is None
            or (ctx.owner_id, ctx.session_id, ctx.operation_id)
               != (export.owner_id, export.session_id, export.request_id)
            or type(export.content) is not bytes or len(export.content) > MAX_FILE_BYTES):
        raise SourceError("workspace_export_scope", "An exact admitted note export is required.", 403)
    binding = NoteExportBinding(**{
        name: getattr(export, name) for name in NoteExportBinding.model_fields})
    return FilePublicationSpec(session_id=ctx.session_id, filename=binding.filename,
        content_type=binding.content_type, description="Exact export of a saved note version.",
        source="assistant", metadata={"workspace_note_export": binding.model_dump(mode="json")})


