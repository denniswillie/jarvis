"""One account note in the existing note/source store; no agent or task runtime.

Only the trusted publisher calls this module. Its injected guard owns task/action
authority and exact evidence checks; a worker draft alone grants no write right.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Annotated
from uuid import NAMESPACE_URL, UUID, uuid5

from pydantic import Field, TypeAdapter, ValidationError

from app.knowledge.models import SourceError
from app.knowledge.notes import _ConnectionPool
from app.knowledge.product_notes import NOTE_NAMESPACE
from app.notes.repository import NoteRepository, _note_from_record
from app.notes.workspace_content import (
    apply_patches, canonical_content, canonical_json, content_for_note, parameters_digest,
    render_body, validate_content,
)
from app.notes.workspace_repository import _artifact_ref, _version_id
from app.workspace.models import Patch


ACCOUNT_NOTE_MARKER = "jarvis_account_note"
ACCOUNT_NOTE_MARKER_VALUE = "v1"
MAX_PUBLICATION_DEPENDENCIES = 128
_PATCHES = TypeAdapter(Annotated[list[Patch], Field(min_length=1, max_length=32)])


@dataclass(frozen=True, slots=True)
class _AccountOwner:
    """Internal current-read scope, never mutation or conversation authority."""
    owner_id: str


def account_note_id(owner_id: str) -> UUID:
    if not isinstance(owner_id, str) or not owner_id.strip() or "\x00" in owner_id:
        raise ValueError("An exact authenticated owner is required")
    return uuid5(NAMESPACE_URL, "jarvis.account-note:v1:" + canonical_json([owner_id]))


def account_note_edit_parameters(owner_id, base_version_id, changes) -> dict:
    """Stable effect parameters; source/task authority is separately guarded."""
    _version_id(base_version_id)
    try:
        parsed = _PATCHES.validate_python(changes)
    except ValidationError as error:
        raise SourceError("invalid_document", "The document or patch is invalid.", 422) from error
    return {"action": "patch", "artifact_id": str(account_note_id(owner_id)),
            "base_version_id": base_version_id,
            "changes": [change.model_dump(mode="json") for change in parsed]}


@dataclass(frozen=True, slots=True)
class AccountNotePublicationCandidate:
    artifact_id: UUID
    base_version_id: str | None
    parameters_digest: str
    title: str
    changes_json: str
    content_json: str

    @property
    def changes(self):
        # Each access is a copy, so a guard cannot alter the prepared effect.
        return json.loads(self.changes_json)

    @property
    def content(self):
        return validate_content(json.loads(self.content_json))


class AccountNoteRepository:
    def __init__(self, workspace):
        self.workspace = workspace
        self.sources = workspace.sources

    async def read_current_for_owner(self, owner_id):
        # The authenticated account can open its note before voice exists.
        # Existing WorkspaceContext/sessionful APIs remain strict and separate.
        self.workspace.notes._authorize(owner_id)
        notes = self.workspace.notes.for_session(None)
        owner = _AccountOwner(owner_id)
        async with self.sources.read_guard_transaction(owner_id) as connection:
            current, _, _ = await self._current(connection, notes, owner)
            notes._authorize(owner_id)
            return current

    async def _current(self, connection, notes, ctx):
        identity = account_note_id(ctx.owner_id)
        # Known mappings retain their original area across conversation changes.
        area = await notes._area_for(connection, identity)
        mapping = await notes._mapping(connection, identity, required=False, area_id=area)
        row = await connection.fetchrow("select * from jarvis_notes where owner_id=$1 and id=$2",
                                        ctx.owner_id, identity)
        if mapping is None:
            # An absent account note initializes only in the configured default
            # owner area, never a transient session's selected activity area.
            area = self.workspace.notes.area_id
            await self.sources._area(connection, ctx.owner_id, area)
            if row is None:
                return None, area, None
            raise SourceError("account_note_binding_conflict", "The account note has no matching source binding.")
        await self.sources._area(connection, ctx.owner_id, area)
        if mapping.get("namespace") != NOTE_NAMESPACE or mapping.get("source_key") != str(identity):
            raise SourceError("account_note_binding_conflict", "The account note source binding conflicts.")
        if mapping.get("status") != "active" or row is None:
            raise SourceError("account_note_unavailable", "The account note is unavailable; it was not recreated.")
        note = _note_from_record(row)
        if note.session_id is not None or note.metadata.get(ACCOUNT_NOTE_MARKER) != ACCOUNT_NOTE_MARKER_VALUE:
            raise SourceError("account_note_binding_conflict", "An existing note cannot be adopted as the account note.")
        snapshot = await self.workspace._read(connection, notes, ctx, identity, area_id=area)
        return snapshot, area, mapping

    async def read(self, ctx, version_id=None):
        notes = self.workspace._bound(ctx)
        _version_id(version_id)
        async with self.sources.read_guard_transaction(ctx.owner_id) as connection:
            return await self._read_in_transaction(connection, notes, ctx, version_id)

    async def read_in_transaction(self, connection, ctx, version_id=None):
        notes = self.workspace._bound(ctx)
        _version_id(version_id)
        if not connection.is_in_transaction():
            raise RuntimeError("Account-note reads require an active source guard transaction")
        return await self._read_in_transaction(connection, notes, ctx, version_id)

    async def _read_in_transaction(self, connection, notes, ctx, version_id):
        await self.workspace._session(connection, ctx)
        current, area, _ = await self._current(connection, notes, ctx)
        if current is None and version_id is not None:
            raise SourceError("note_version_mismatch", "The account note has no such retained version.", 404)
        if current is not None and version_id is not None:
            current = await self.workspace._read(connection, notes, ctx, current.note.id, version_id, area_id=area)
        notes._authorize(ctx.owner_id)
        return current

    async def publish(self, ctx, base_version_id, changes, publication_guard, *, receipt_recorder=None):
        notes = self.workspace._bound(ctx)
        if ctx.operation_id is None:
            raise SourceError("command_admission_required", "A fixed admitted publication operation is required.", 403)
        parameters = account_note_edit_parameters(ctx.owner_id, base_version_id, changes)
        digest = parameters_digest(parameters)
        identity = UUID(parameters["artifact_id"])
        async with self.sources.transaction(ctx.owner_id) as connection:
            await self.workspace._session(connection, ctx)
            receipt = await self.workspace._receipt(connection, ctx, ctx.operation_id)
            if receipt is not None:
                if (receipt["parameters_digest"] != digest or receipt["session_id"] != ctx.session_id
                        or receipt["artifact_id"] != identity or receipt["operation_kind"] != "patch"):
                    raise SourceError("operation_conflict", "This operation identity was used for a different command.")
                return await self.workspace._receipt_result(connection, notes, ctx, receipt)

            current, area, mapping = await self._current(connection, notes, ctx)
            actual_base = current.current.document_id if current else None
            if actual_base != base_version_id:
                result = {"status": "success", "outcome": "conflict", "effect_committed": False,
                    "operation_id": str(ctx.operation_id), "summary": "The account note changed. Read its current version before preparing another edit.",
                    "error": {"code": "note_version_conflict", "retryable": False}}
                if current is not None:
                    result["artifact"] = _artifact_ref(identity, actual_base, current.note.title)
                if receipt_recorder is not None:
                    await receipt_recorder(connection, result)
                return result
            original_content = content_for_note(current.note) if current else validate_content({"schema_version": 1, "blocks": []})
            title, content = apply_patches(current.note.title if current else "Notebook", original_content,
                                           _PATCHES.validate_python(parameters["changes"]))
            candidate = AccountNotePublicationCandidate(identity, base_version_id, digest, title,
                canonical_json(parameters["changes"]), canonical_content(content))
            if not callable(publication_guard):
                raise SourceError("note_publication_not_authorized", "A trusted publication guard is required.", 403)
            # No invocation-context clearing or foreground mutate fallback. This
            # guard must validate the real task/action/fence and frozen evidence.
            originals = await publication_guard(connection, current, candidate)
            if (type(originals) is not tuple or not originals
                    or any(not isinstance(identity, UUID) for identity in originals)):
                raise SourceError("note_publication_evidence_invalid", "Publication needs validated original evidence.", 403)
            parents = tuple(sorted(set(originals) | ({current.source.payload_id} if current else set()), key=str))
            if len(parents) > MAX_PUBLICATION_DEPENDENCIES:
                raise SourceError("note_publication_evidence_invalid", "Publication evidence exceeds its bound.", 422)
            if not await connection.fetchval("select lock_jarvis_copy_sources($1,$2::uuid[])", ctx.owner_id, list(parents)):
                raise SourceError("note_publication_evidence_unavailable", "Original publication evidence is unavailable.", 403)

            body = render_body(content)
            metadata = dict(current.note.metadata) if current else {ACCOUNT_NOTE_MARKER: ACCOUNT_NOTE_MARKER_VALUE}
            metadata["workspace_v1"] = content.model_dump(mode="json")
            changed = current is None or (title, body, metadata) != (current.note.title, current.note.body, current.note.metadata)
            if current is None:
                note = await NoteRepository(_ConnectionPool(connection)).create_note(owner_id=ctx.owner_id,
                    session_id=None, title=title, body=body, metadata=metadata, note_id=identity)
            elif changed:
                row = await connection.fetchrow("""update jarvis_notes set title=$3,body=$4,metadata=$5::jsonb,
                    updated_at=clock_timestamp() where owner_id=$1 and id=$2 returning *""",
                    ctx.owner_id, identity, title, body, canonical_json(metadata))
                note = _note_from_record(row)
            else:
                note = current.note
            if changed:
                note = await notes._publish(connection, note, mapping, area_id=area)
                version_id = note.memory_source["document_id"]
                payload_id = UUID(str(note.memory_source["payload_id"]))
                await connection.executemany("""insert into jarvis_context_payload_dependencies
                    (owner_id,source_payload_id,dependent_payload_id) values($1,$2,$3) on conflict do nothing""",
                    [(ctx.owner_id, parent, payload_id) for parent in parents])
            else:
                # A no-op gets a receipt, not new content or retroactive edits to
                # the immutable version's provenance. Its task owns request evidence.
                version_id = actual_base
            await connection.execute("""insert into jarvis_note_operations
                (owner_id,operation_id,session_id,area_id,operation_kind,parameters_digest,artifact_id,result_version_id,outcome)
                values($1,$2,$3,$4,$5,$6,$7,$8,'completed')""",
                ctx.owner_id, ctx.operation_id, ctx.session_id, area, "patch", digest, identity, version_id)
            notes._authorize(ctx.owner_id)
            result = {"status": "success", "outcome": "completed", "effect_committed": True,
                    "operation_id": str(ctx.operation_id), "summary": "Updated your note.",
                    "artifact": _artifact_ref(identity, version_id, note.title)}
            if receipt_recorder is not None:
                await receipt_recorder(connection, result)
            return result
