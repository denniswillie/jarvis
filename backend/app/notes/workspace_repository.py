"""Structured note operations composed with existing source-note transactions.

Originals and immutable versions remain in their existing stores. The only new
row is a reference-only operation receipt, committed with the note publication.
"""
from __future__ import annotations

import base64
from dataclasses import dataclass
from datetime import datetime
import json
from uuid import NAMESPACE_URL, UUID, uuid5

from app.knowledge.models import SourceError, SourceVersion
from app.knowledge.notes import SourceNoteRepository, _ConnectionPool
from app.knowledge.product_notes import IMPORTED_NAMESPACE, NOTE_NAMESPACE
from app.notes.models import JarvisNote
from app.notes.repository import NoteRepository, _note_from_record
from app.notes.workspace_content import (
    apply_patches, canonical_json, content_for_note, parameters_digest, render_body,
)
from app.tools.invocation import current_tool_invocation
from app.workspace.models import ArtifactPage, ArtifactSummary, ArtifactView, VersionPage, WorkspaceContext


@dataclass(frozen=True)
class NoteSnapshot:
    note: JarvisNote
    source: SourceVersion
    current: SourceVersion

    def summary(self):
        return ArtifactSummary(artifact_id=self.note.id, version_id=self.source.document_id,
            revision=self.source.revision, current_version_id=self.current.document_id,
            current_revision=self.current.revision, is_current=self.source.is_current,
            title=self.note.title, updated_at=self.note.updated_at)

    def view(self):
        return ArtifactView(artifact=self.summary(), content=content_for_note(self.note))


def _encode_cursor(value):
    return base64.urlsafe_b64encode(canonical_json(value).encode()).decode().rstrip("=")


def _decode_cursor(value, expected, fields):
    try:
        if not isinstance(value, str) or not 1 <= len(value) <= 4096:
            raise ValueError()
        decoded = base64.b64decode(value + "=" * (-len(value) % 4), altchars=b"-_", validate=True)
        result = json.loads(decoded)
        if not isinstance(result, dict) or set(result) != set(expected) | set(fields):
            raise ValueError()
        if any(type(result[key]) is not type(val) or result[key] != val for key, val in expected.items()):
            raise ValueError()
        return result
    except (ValueError, TypeError, UnicodeDecodeError, RecursionError) as error:
        raise SourceError("invalid_cursor", "The pagination cursor is invalid for this scope; restart the list.", 400) from error


def _artifact_ref(note_id, version_id, title=None):
    result = {"artifact_id": str(note_id), "version_id": version_id, "kind": "note"}
    if title is not None:
        result["title"] = title
    return result


def _uuid(value, field):
    if isinstance(value, UUID):
        return value
    try:
        if not isinstance(value, str):
            raise ValueError()
        return UUID(value)
    except ValueError as error:
        raise SourceError("invalid_workspace_identity", f"{field} must be a UUID.", 422) from error


def _version_id(value):
    if value is not None and (not isinstance(value, str) or not 1 <= len(value) <= 200 or "\x00" in value):
        raise SourceError("invalid_workspace_identity", "version_id must be a bounded opaque string.", 422)
    return value


class WorkspaceNoteRepository:
    """Foreground adapter; never substitute it for a worker-granted repository."""

    def __init__(self, notes: SourceNoteRepository):
        if not isinstance(notes, SourceNoteRepository):
            raise TypeError("Workspace requires the source-backed note repository")
        if notes.task_evidence is not None:
            raise ValueError("Workspace cannot replace worker note grants")
        self.notes = notes
        self.sources = notes.sources

    async def read_account_note(self, ctx, version_id=None):
        from app.notes.workspace_account_note import AccountNoteRepository
        return await AccountNoteRepository(self).read(ctx, version_id)

    async def read_current_account_note(self, owner_id):
        from app.notes.workspace_account_note import AccountNoteRepository
        return await AccountNoteRepository(self).read_current_for_owner(owner_id)

    async def read_account_note_in_transaction(self, connection, ctx, version_id=None):
        from app.notes.workspace_account_note import AccountNoteRepository
        return await AccountNoteRepository(self).read_in_transaction(connection, ctx, version_id)

    async def publish_account_edit(self, ctx, base_version_id, changes, publication_guard, *, receipt_recorder=None):
        from app.notes.workspace_account_note import AccountNoteRepository
        return await AccountNoteRepository(self).publish(ctx, base_version_id, changes, publication_guard,
                                                        receipt_recorder=receipt_recorder)

    def _bound(self, ctx):
        if not isinstance(ctx, WorkspaceContext):
            raise TypeError("Workspace scope must be trusted typed context")
        self.notes._authorize(ctx.owner_id)
        notes = self.notes.for_session(ctx.session_id)
        if notes.task_evidence is not None:
            raise SourceError("structured_worker_not_granted", "Structured worker editing is not enabled.", 403)
        return notes

    async def _session(self, connection, ctx):
        owned = await connection.fetchval(
            "select exists(select 1 from jarvis_sessions where owner_id=$1 and id=$2)",
            ctx.owner_id, ctx.session_id)
        if not owned:
            raise SourceError("workspace_session_unavailable", "The owned Workspace session is unavailable.", 404)

    async def _read(self, conn, notes, ctx, note_id, version_id=None, *, area_id=None):
        area_id = await notes._area_for(conn, note_id) if area_id is None else area_id
        row = await conn.fetchrow("select * from jarvis_notes where owner_id=$1 and id=$2", ctx.owner_id, note_id)
        if row is None:
            raise SourceError("note_not_found", "This note is no longer available.", 404)
        note = await notes._expose(conn, _note_from_record(row), area_id=area_id)
        mapping = await notes._mapping(conn, note_id, area_id=area_id)
        current = await self.sources._read(conn, ctx.owner_id, area_id, mapping["id"])
        source = current
        if version_id is not None and version_id != current.document_id:
            source = await self.sources.resolve_document_in_transaction(
                conn, ctx.owner_id, area_id, version_id, historical=True)
            if source.source_id != current.source_id:
                raise SourceError("note_version_mismatch", "This version belongs to another note.")
            try:
                note = JarvisNote.model_validate(source.content["record"])
            except (ValueError, KeyError, TypeError) as error:
                raise SourceError("note_original_invalid", "The historical original is not a valid note.") from error
            if note.id != note_id:
                raise SourceError("note_version_mismatch", "This source version does not contain the requested note.")
        # Enforce body/envelope coherence even for summary and export reads.
        content_for_note(note)
        notes._authorize(ctx.owner_id)
        return NoteSnapshot(note, source, current)

    async def read_note(self, ctx, artifact_id, version_id=None):
        notes = self._bound(ctx)
        artifact_id, version_id = _uuid(artifact_id, "artifact_id"), _version_id(version_id)
        async with self.sources.read_guard_transaction(ctx.owner_id) as conn:
            await self._session(conn, ctx)
            return await self._read(conn, notes, ctx, artifact_id, version_id)

    async def read_note_in_transaction(self, connection, ctx, artifact_id, version_id=None):
        """Read exact evidence while the caller holds this owner's source guard.

        The caller owns the existing transaction and its commit/rollback. This
        seam never acquires a connection, starts a transaction or grants export
        authority; it retains the ordinary note read's scope/source checks.
        """
        notes = self._bound(ctx)
        artifact_id, version_id = _uuid(artifact_id, "artifact_id"), _version_id(version_id)
        if not connection.is_in_transaction():
            raise RuntimeError("Workspace note read requires an existing transaction")
        await self._session(connection, ctx)
        return await self._read(connection, notes, ctx, artifact_id, version_id)

    async def _receipt(self, conn, ctx, operation_id):
        return await conn.fetchrow("select * from jarvis_note_operations where owner_id=$1 and operation_id=$2",
            ctx.owner_id, operation_id)

    async def _receipt_result(self, conn, notes, ctx, receipt):
        ref = _artifact_ref(receipt["artifact_id"], receipt["result_version_id"])
        result = {"status": "success", "outcome": "completed", "effect_committed": True,
            "operation_id": str(receipt["operation_id"]), "summary": "This note operation was applied.", "artifact": ref}
        try:
            snapshot = await self._read(conn, notes, ctx, receipt["artifact_id"], receipt["result_version_id"],
                area_id=receipt["area_id"])
        except SourceError:
            result["summary"] = "This note operation was applied; its saved result is now unavailable."
            result["error"] = {"code": "note_result_unavailable", "retryable": False}
        else:
            ref["title"] = snapshot.note.title
        notes._authorize(ctx.owner_id)
        return result

    async def operation_receipt(self, ctx, operation_id):
        notes = self._bound(ctx)
        operation_id = _uuid(operation_id, "operation_id")
        async with self.sources.read_guard_transaction(ctx.owner_id) as conn:
            await self._session(conn, ctx)
            receipt = await self._receipt(conn, ctx, operation_id)
            return await self._receipt_result(conn, notes, ctx, receipt) if receipt is not None else None

    async def mutate(self, ctx, command):
        notes = self._bound(ctx)
        if ctx.operation_id is None:
            raise SourceError("command_admission_required", "A trusted foreground command admission is required.", 403)
        if current_tool_invocation().task is not None:
            raise SourceError("structured_worker_not_granted", "Structured worker editing is not enabled.", 403)
        normalized = command.model_dump(mode="json")
        digest = parameters_digest(normalized)
        async with self.sources.transaction(ctx.owner_id) as conn:
            await self._session(conn, ctx)
            receipt = await self._receipt(conn, ctx, ctx.operation_id)
            if receipt is not None:
                if receipt["parameters_digest"] != digest or receipt["session_id"] != ctx.session_id:
                    raise SourceError("operation_conflict", "This operation identity was used for a different command.")
                return await self._receipt_result(conn, notes, ctx, receipt)
            if command.action == "create":
                note_id = uuid5(NAMESPACE_URL, "jarvis.workspace.note:" + canonical_json([ctx.owner_id, str(ctx.operation_id)]))
                area_id = await notes._area_for(conn, note_id)
                await self.sources._area(conn, ctx.owner_id, area_id)
                mapping = await notes._mapping(conn, note_id, required=False, area_id=area_id)
                if mapping is not None:
                    raise SourceError("note_original_unavailable", "An existing or removed identity cannot be recreated.")
                note = await NoteRepository(_ConnectionPool(conn)).create_note(
                    owner_id=ctx.owner_id, session_id=ctx.session_id, title=command.title,
                    body=render_body(command.content), metadata={"workspace_v1": command.content.model_dump(mode="json")},
                    note_id=note_id)
                note = await notes._publish(conn, note, None, area_id=area_id)
                version_id = note.memory_source["document_id"]
            else:
                note_id = command.artifact_id
                area_id = await notes._area_for(conn, note_id)
                snapshot = await self._read(conn, notes, ctx, note_id, area_id=area_id)
                if snapshot.current.document_id != command.base_version_id:
                    return {"status": "success", "outcome": "conflict", "effect_committed": False,
                        "operation_id": str(ctx.operation_id), "summary": "The note changed. Read its current version before preparing another edit.",
                        "artifact": _artifact_ref(note_id, snapshot.current.document_id, snapshot.note.title),
                        "error": {"code": "note_version_conflict", "retryable": False}}
                mapping = await notes._mapping(conn, note_id, area_id=area_id)
                note, version_id = snapshot.note, snapshot.current.document_id
                if command.action == "delete":
                    await self.sources.delete_source_in_transaction(conn, ctx.owner_id, area_id, mapping["id"],
                        expected_revision=mapping["current_revision"])
                else:
                    if command.action == "restore":
                        historical = await self._read(conn, notes, ctx, note_id, command.restore_version_id, area_id=area_id)
                        title, content = historical.note.title, content_for_note(historical.note)
                    else:
                        title, content = apply_patches(note.title, content_for_note(note), command.changes)
                    body = render_body(content)
                    metadata = dict(note.metadata) | {"workspace_v1": content.model_dump(mode="json")}
                    if (title, body, metadata) != (note.title, note.body, note.metadata):
                        row = await conn.fetchrow("""update jarvis_notes set title=$3,body=$4,metadata=$5::jsonb,
                            updated_at=clock_timestamp() where owner_id=$1 and id=$2 returning *""",
                            ctx.owner_id, note_id, title, body, canonical_json(metadata))
                        note = await notes._publish(conn, _note_from_record(row), mapping, area_id=area_id)
                        version_id = note.memory_source["document_id"]
            await conn.execute("""insert into jarvis_note_operations
                (owner_id,operation_id,session_id,area_id,operation_kind,parameters_digest,artifact_id,result_version_id,outcome)
                values($1,$2,$3,$4,$5,$6,$7,$8,'completed')""",
                ctx.owner_id, ctx.operation_id, ctx.session_id, area_id, command.action, digest, note_id, version_id)
            notes._authorize(ctx.owner_id)
            verb = "Deleted" if command.action == "delete" else "Saved"
            return {"status": "success", "outcome": "completed", "effect_committed": True,
                "operation_id": str(ctx.operation_id), "summary": f"{verb} {note.title}.",
                "artifact": _artifact_ref(note_id, version_id, note.title)}

    async def list_notes(self, ctx, query=None, limit=8, cursor=None):
        notes = self._bound(ctx)
        if type(limit) is not int or not 1 <= limit <= 8 or (query is not None and (not isinstance(query, str) or len(query) > 500)):
            raise SourceError("invalid_workspace_query", "Note queries require limit 1–8 and at most 500 characters.", 422)
        async with self.sources.read_guard_transaction(ctx.owner_id) as conn:
            await self._session(conn, ctx)
            area_id = await notes._area_for(conn)
            await self.sources._area(conn, ctx.owner_id, area_id)
            scope = {"kind": "notes", "owner": ctx.owner_id, "session": str(ctx.session_id),
                "area": str(area_id), "query": query, "limit": limit}
            before_time = before_id = None
            if cursor is not None:
                decoded = _decode_cursor(cursor, scope, ("before_time", "before_id"))
                try:
                    if not isinstance(decoded["before_time"], str) or not isinstance(decoded["before_id"], str):
                        raise ValueError()
                    before_time, before_id = datetime.fromisoformat(decoded["before_time"]), UUID(decoded["before_id"])
                    if before_time.tzinfo is None:
                        raise ValueError()
                except (ValueError, TypeError) as error:
                    raise SourceError("invalid_cursor", "The list cursor contains an invalid ordering key.", 400) from error
            pattern = f"%{query}%" if query else None
            unmatched = await conn.fetchval("""select exists(select 1 from jarvis_notes n where n.owner_id=$1
                and ($2::text is null or n.title ilike $2 or n.body ilike $2)
                and not exists(select 1 from jarvis_knowledge_sources s where s.owner_id=n.owner_id
                    and ((s.namespace=$3 and s.source_key=n.id::text)
                        or (s.namespace=$4 and s.source_key='jarvis_notes/'||n.id::text))))""",
                ctx.owner_id, pattern, NOTE_NAMESPACE, IMPORTED_NAMESPACE)
            if unmatched:
                raise SourceError("note_import_incomplete", "Matching notes require source import; the result is not an empty list.")
            rows = await conn.fetch("""select n.* from jarvis_notes n where n.owner_id=$1
                and ($2::text is null or n.title ilike $2 or n.body ilike $2)
                and exists(select 1 from jarvis_knowledge_sources s where s.owner_id=n.owner_id
                    and s.area_id=$3 and s.status='active' and ((s.namespace=$4 and s.source_key=n.id::text)
                        or (s.namespace=$5 and s.source_key='jarvis_notes/'||n.id::text)))
                and ($6::timestamptz is null or (n.updated_at,n.id)<($6,$7::uuid))
                order by n.updated_at desc,n.id desc limit $8""",
                ctx.owner_id, pattern, area_id, NOTE_NAMESPACE, IMPORTED_NAMESPACE, before_time, before_id, limit + 1)
            items = [(await self._read(conn, notes, ctx, row["id"], area_id=area_id)).summary() for row in rows[:limit]]
            next_cursor = None
            if len(rows) > limit:
                last = rows[limit - 1]
                next_cursor = _encode_cursor(scope | {"before_time": last["updated_at"].isoformat(), "before_id": str(last["id"])})
            return ArtifactPage(items=items, next_cursor=next_cursor)

    async def list_versions(self, ctx, artifact_id, limit=8, cursor=None):
        notes = self._bound(ctx)
        artifact_id = _uuid(artifact_id, "artifact_id")
        if type(limit) is not int or not 1 <= limit <= 8:
            raise SourceError("invalid_workspace_query", "Version queries require limit 1–8.", 422)
        async with self.sources.read_guard_transaction(ctx.owner_id) as conn:
            await self._session(conn, ctx)
            area_id = await notes._area_for(conn, artifact_id)
            current = await self._read(conn, notes, ctx, artifact_id, area_id=area_id)
            scope = {"kind": "versions", "owner": ctx.owner_id, "session": str(ctx.session_id),
                "area": str(area_id), "artifact": str(artifact_id), "limit": limit}
            before_revision = expected_generation = None
            if cursor is not None:
                decoded = _decode_cursor(cursor, scope, ("before_revision", "expected_generation"))
                before_revision, expected_generation = decoded["before_revision"], decoded["expected_generation"]
                if any(type(v) is not int or v < 1 for v in (before_revision, expected_generation)):
                    raise SourceError("invalid_cursor", "The history cursor is invalid.", 400)
            page = await self.sources.list_versions_in_transaction(conn, ctx.owner_id, area_id, current.current.source_id,
                limit=limit, before_revision=before_revision, expected_generation=expected_generation)
            items = [(await self._read(conn, notes, ctx, artifact_id, version["document_id"], area_id=area_id)).summary()
                for version in page["versions"]]
            next_cursor = None
            if page["next_before_revision"] is not None:
                next_cursor = _encode_cursor(scope | {"before_revision": page["next_before_revision"],
                    "expected_generation": page["expected_generation"]})
            return VersionPage(items=items, next_cursor=next_cursor, unavailable_count=page["unavailable_revisions"])
