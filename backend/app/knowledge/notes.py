"""Editable note originals published once to the selected source-backed memory."""
from contextlib import asynccontextmanager
import json
from uuid import NAMESPACE_URL, UUID, uuid4, uuid5

from app.knowledge.models import MemorySourceReference, SourceError, SourcePublication
from app.knowledge.product_notes import NOTE_NAMESPACE, IMPORTED_NAMESPACE, note_sources
from app.knowledge.read_provenance import SourceReadProvenance
from app.notes.models import JarvisNote
from app.notes.repository import NoteRepository, _note_from_record
from app.notes.serialization import _note_payload
from app.tasks.models import TaskError, digest, json_object
from app.tools.inputs import success
from app.tools.invocation import current_tool_invocation
from app.tools.provenance import SourceBoundToolResult


class _ConnectionPool:
    """Reuse the original CRUD SQL without acquiring a second pool connection."""
    def __init__(self, connection):
        self.connection = connection

    @asynccontextmanager
    async def acquire(self):
        yield self.connection


class SourceNoteSearchResult(list):
    """List-compatible result retaining scope even when a search is empty."""
    def __init__(self, notes, area_id):
        super().__init__(notes)
        self.knowledge_scope = {"area_id": str(area_id), "origin": "session_selection"}


class SourceNoteRepository(NoteRepository):
    search_limit = 8

    def __init__(self, sources, *, owner_id, area_id, access_check=None, task_evidence=None,
                 selections=None, session_id=None):
        super().__init__(sources.pool)
        self.sources, self.owner_id, self.area_id = sources, owner_id, area_id
        self.access_check, self.task_evidence = access_check, task_evidence
        self.legacy_repository = NoteRepository(sources.pool)
        self.provenance = SourceReadProvenance(sources)
        self.selections, self.session_id = selections, session_id
        if selections is not None and (
            selections.sources.pool is not sources.pool or selections.default_area_id != area_id
            or task_evidence is not None
        ):
            raise ValueError("Session notes require the same source/default scope and cannot replace task grants")
        if session_id is not None and not isinstance(session_id, UUID):
            raise ValueError("A trusted note session must be an exact UUID")
        if task_evidence is not None and (
                task_evidence.sources is not sources or task_evidence.area_id != area_id
                or task_evidence.context.task.owner_id != owner_id):
            raise ValueError("Note task evidence must match its source scope")

    def for_session(self, session_id):
        if self.selections is None:
            return self
        return SourceNoteRepository(self.sources, owner_id=self.owner_id, area_id=self.area_id,
            access_check=self.access_check, selections=self.selections, session_id=session_id)

    async def _area_for(self, connection, note_id=None):
        if self.selections is None:
            return self.area_id
        await self.selections._session(connection, self.owner_id, self.session_id)
        if note_id is not None:
            rows = await note_sources(connection, self.owner_id, note_id)
            areas = {row["area_id"] for row in rows}
            if len(areas) > 1:
                raise SourceError("note_source_conflict", "The note has competing area mappings; no area was guessed.")
            if areas:
                return areas.pop()
        selected = await self.selections.read_in_transaction(connection, self.owner_id, self.session_id)
        return UUID(selected["area_id"])

    def _authorize(self, owner_id):
        if owner_id != self.owner_id:
            raise SourceError("knowledge_owner_mismatch", "Notes are outside this runtime owner scope.", 403)
        if self.access_check is not None:
            self.access_check(owner_id)

    async def _mapping(self, connection, note_id, *, required=True, area_id=None):
        area_id = self.area_id if area_id is None else area_id
        rows = await note_sources(connection, self.owner_id, note_id)
        retained = [row for row in rows if row["status"] != "deleted"]
        if len(retained) > 1:
            raise SourceError("note_shared_original",
                "This note has competing source mappings. Resolve area ownership before using or editing it.")
        selected = [row for row in rows if row["area_id"] == area_id]
        if not selected:
            if required:
                raise SourceError("note_source_unavailable",
                    "The note has not been imported into the selected area; do not use an untracked fallback.")
            if rows:
                raise SourceError("note_source_scope", "The note identity belongs to another source area.")
            return None
        if len(selected) != 1:
            raise SourceError("note_source_conflict", "The note identity has competing source mappings.")
        return selected[0]

    async def _expose(self, connection, note, *, area_id=None):
        area_id = self.area_id if area_id is None else area_id
        mapping = await self._mapping(connection, note.id, area_id=area_id)
        source = await self.sources._read(connection, self.owner_id, area_id, mapping["id"])
        try:
            original = JarvisNote.model_validate(source.content["record"])
        except (ValueError, KeyError, TypeError) as error:
            raise SourceError("note_original_invalid", "The source does not contain its original note.") from error
        if original.model_dump(mode="json") != note.model_dump(mode="json"):
            raise SourceError("note_original_changed",
                "The editable note and its source version disagree. Repair publication before using the note.")
        metadata = source.original_evidence()
        metadata.pop("content")
        citation = None
        if self.task_evidence is not None:
            citation = (await self.task_evidence.bind_in_transaction(connection, [metadata]))[0]
        self._authorize(self.owner_id)
        return note.model_copy(update={"memory_source": metadata, "task_evidence": citation})

    async def _publish(self, connection, note, mapping, *, area_id=None):
        area_id = self.area_id if area_id is None else area_id
        payload = await connection.fetchval(
            """insert into jarvis_context_payloads(owner_id,availability,content)
                values($1,'available',$2::jsonb) returning id""", self.owner_id,
            json.dumps({"origin": "jarvis_notes", "record": note.model_dump(mode="json"),
                        "meaning": "Literal editable note, not independent proof of its claims."}, allow_nan=False))
        await self.sources.publish_in_transaction(connection, self.owner_id, SourcePublication(
            area_id=area_id, namespace=mapping["namespace"] if mapping else NOTE_NAMESPACE,
            source_key=mapping["source_key"] if mapping else str(note.id), payload_id=payload,
            source_type="note", trust="external_content", source_time=note.updated_at,
            session_id=note.session_id, expected_revision=mapping["current_revision"] if mapping else 0,
            time_basis="jarvis_notes.updated_at: note editing time, not proof of physical event time."))
        return await self._expose(connection, note, area_id=area_id)

    async def _task_action(self, connection, owner_id, session_id, parameters):
        invocation = current_tool_invocation()
        task = invocation.task
        if self.task_evidence is None:
            if task is not None:
                raise TaskError("note_task_evidence_missing", "Task note writes require source tracking.", 403)
            return None
        ctx = self.task_evidence.context
        if (session_id != ctx.task.session_id or task is None or task.task_id != ctx.task.id
                or task.task_revision != ctx.claim.task_revision or task.attempt_id != ctx.claim.attempt_id
                or task.action_id is None):
            raise TaskError("note_write_scope_mismatch", "A current authorized task action is required.", 403)
        await ctx.tasks.lock_owner(connection, owner_id)
        _, action = await ctx.actions._locked_action(connection, ctx.claim, task.action_id)
        grant = await ctx.tasks.validate_grant(connection, owner_id, action["grant_id"])
        if (action["status"] != "submitting" or action["submission_token"] is None
                or action["operation_key"] != task.operation_key or grant["capability"] != "notes.create"
                or action["parameters_digest"] != digest(parameters)
                or digest(json_object(grant["scope"])) != digest(parameters)
                or invocation.request_event_id != grant["request_event_id"]
                or invocation.request_revision != grant["request_revision"]):
            raise TaskError("note_write_not_authorized", "The note does not match its exact submitted grant.", 403)
        return action

    async def _task_receipt(self, connection, action, note, parameters):
        if action is None:
            return
        ctx = self.task_evidence.context
        async def materialize(conn, originals):
            await ctx.actions.record_receipt_in_transaction(
                conn, self.owner_id, ctx.task.id, action["id"], action["submission_token"], "accepted",
                {"tool": "create_note", "parameters_digest": digest(parameters),
                 "result": success({"note": _note_payload(note)})})
            payload = await conn.fetchval(
                "select receipt_payload_id from jarvis_task_actions where owner_id=$1 and id=$2",
                self.owner_id, action["id"])
            await conn.executemany("""insert into jarvis_context_payload_dependencies
                (owner_id,dependent_payload_id,source_payload_id) values($1,$2,$3) on conflict do nothing""",
                [(self.owner_id, payload, original) for original in originals])
            return None, payload
        await self.provenance.bind_in_transaction(
            connection, self.owner_id, self.area_id, [note.memory_source], materialize)

    async def create_note(self, *, owner_id, session_id, title, body, metadata=None):
        self._authorize(owner_id)
        if self.selections is not None and self.session_id != session_id:
            raise SourceError("knowledge_session_unavailable", "The note belongs to another conversation.", 403)
        parameters = {"title": title, "body": body}
        invocation = current_tool_invocation()
        identity = None
        if invocation.task is not None:
            identity = {"action": str(invocation.task.action_id)}
        elif invocation.request_event_id is not None:
            identity = {"event": str(invocation.request_event_id), "revision": invocation.request_revision,
                        "parameters": parameters}
        elif invocation.request_identity is not None:
            identity = {"namespace": invocation.request_identity.namespace,
                        "event": invocation.request_identity.event_id, "parameters": parameters}
        elif invocation.provider_call_id is not None:
            identity = {"session": str(session_id), "call": invocation.provider_call_id}
        note_id = uuid5(NAMESPACE_URL, "jarvis:note:" + digest({"owner": owner_id, **identity})) if identity else uuid4()
        async with self.sources.transaction(owner_id) as connection:
            area_id = await self._area_for(connection, note_id)
            await self.sources._area(connection, owner_id, area_id)
            action = await self._task_action(connection, owner_id, session_id, parameters)
            existing = await connection.fetchrow(
                "select * from jarvis_notes where owner_id=$1 and id=$2 for update", owner_id, note_id)
            if existing is not None:
                note = _note_from_record(existing)
                if (note.title, note.body, note.session_id, note.metadata) != (title, body, session_id, metadata or {}):
                    raise SourceError("note_create_replay_conflict", "This save identity has already stored different content.")
                note = await self._expose(connection, note, area_id=area_id)
            else:
                mapping = await self._mapping(connection, note_id, required=False, area_id=area_id)
                if mapping is not None:
                    raise SourceError("note_original_unavailable", "A removed original cannot be recreated by replay.")
                note = await NoteRepository(_ConnectionPool(connection)).create_note(
                    owner_id=owner_id, session_id=session_id, title=title, body=body, metadata=metadata, note_id=note_id)
                note = await self._publish(connection, note, None, area_id=area_id)
            await self._task_receipt(connection, action, note, parameters)
            self._authorize(owner_id)
            return note

    async def search_notes(self, *, owner_id, query, limit):
        self._authorize(owner_id)
        if type(limit) is not int or not 1 <= limit <= self.search_limit:
            raise ValueError("Source note search limit must be between 1 and 8")
        pattern = f"%{query}%" if query else None
        async with self.sources.transaction(owner_id) as connection:
            area_id = await self._area_for(connection)
            await self.sources._area(connection, owner_id, area_id)
            # Do not silently pretend an unimported matching original is absent.
            unmatched = await connection.fetchval(
                """select exists(select 1 from jarvis_notes n where n.owner_id=$1
                    and ($2::text is null or n.title ilike $2 or n.body ilike $2)
                    and not exists(select 1 from jarvis_knowledge_sources s where s.owner_id=n.owner_id
                        and ((s.namespace=$3 and s.source_key=n.id::text)
                            or (s.namespace=$4 and s.source_key='jarvis_notes/'||n.id::text))))""",
                owner_id, pattern, NOTE_NAMESPACE, IMPORTED_NAMESPACE)
            if unmatched:
                raise SourceError("note_import_incomplete", "Matching notes still require source import; results are not empty.")
            records = await connection.fetch(
                """select n.* from jarvis_notes n where n.owner_id=$1
                    and ($2::text is null or n.title ilike $2 or n.body ilike $2)
                    and exists(select 1 from jarvis_knowledge_sources s where s.owner_id=n.owner_id
                        and s.area_id=$3 and s.status='active'
                        and ((s.namespace=$4 and s.source_key=n.id::text)
                            or (s.namespace=$5 and s.source_key='jarvis_notes/'||n.id::text)))
                    order by n.updated_at desc,n.id limit $6""",
                owner_id, pattern, area_id, NOTE_NAMESPACE, IMPORTED_NAMESPACE, limit)
            notes = [await self._expose(connection, _note_from_record(record), area_id=area_id) for record in records]
            self._authorize(owner_id)
            return SourceNoteSearchResult(notes, area_id) if self.selections is not None else notes

    async def update_note(self, *, owner_id, note_id, title, body):
        self._authorize(owner_id)
        if self.task_evidence is not None:
            raise TaskError("note_update_not_granted", "Task note editing is not an authorized capability.", 403)
        async with self.sources.transaction(owner_id) as connection:
            area_id = await self._area_for(connection, note_id)
            row = await connection.fetchrow(
                "select * from jarvis_notes where owner_id=$1 and id=$2 for update", owner_id, note_id)
            if row is None:
                return None
            current = await self._expose(connection, _note_from_record(row), area_id=area_id)
            if "workspace_v1" in current.metadata and body is not None and body != current.body:
                raise SourceError(
                    "note_structured_update_required",
                    "Update structured notes through Workspace so content and searchable text remain consistent.",
                )
            mapping = await self._mapping(connection, note_id, area_id=area_id)
            if (title is None or title == current.title) and (body is None or body == current.body):
                return current
            note = await NoteRepository(_ConnectionPool(connection)).update_note(
                owner_id=owner_id, note_id=note_id, title=title, body=body)
            return await self._publish(connection, note, mapping, area_id=area_id)

    async def delete_note(self, *, owner_id, note_id):
        self._authorize(owner_id)
        if self.task_evidence is not None:
            raise TaskError("note_delete_not_granted", "Task note deletion is not an authorized capability.", 403)
        async with self.sources.transaction(owner_id) as connection:
            area_id = await self._area_for(connection, note_id)
            exists = await connection.fetchval(
                "select exists(select 1 from jarvis_notes where owner_id=$1 and id=$2)", owner_id, note_id)
            if not exists:
                return False
            mapping = await self._mapping(connection, note_id, area_id=area_id)
            await self.sources.delete_source_in_transaction(
                connection, owner_id, area_id, mapping["id"], expected_revision=mapping["current_revision"])
            self._authorize(owner_id)
            return True

    async def capture_tool_result(self, result, context):
        self._authorize(context.capture.owner_id)
        if self.sources.pool is not context.capture.repository.pool:
            raise SourceError("knowledge_context_pool_mismatch", "Note capture must use the same database.")
        notes = result["notes"] if context.tool_name == "search_notes" else [result["note"]]
        area_id, scope = self.area_id, {}
        if self.selections is not None:
            if context.capture.session_id != self.session_id:
                raise SourceError("knowledge_session_unavailable", "The note result belongs to another conversation.", 403)
            selected = result["knowledge_scope"] if context.tool_name == "search_notes" else result["note"]["memory_source"]
            area_id = UUID(selected["area_id"])
            if any(UUID(note["memory_source"]["area_id"]) != area_id for note in notes):
                raise SourceError("knowledge_result_scope_mismatch", "A note result contains conflicting area references.")
            scope = {"publication_area_id": area_id}
        async def materialize(connection, originals):
            record = await context.capture.bound_tool_outcome(
                connection, context.call_id, context.tool_name, result, originals, **scope)
            self._authorize(self.owner_id)
            return SourceBoundToolResult(result, record), record.payload_id
        return await self.provenance.bind(
            self.owner_id, area_id, [note["memory_source"] for note in notes], materialize)

    async def revalidate_receipt(self, action, result):
        if action.capability != "notes.create" or result.get("status") != "success":
            return result
        async with self.sources.transaction(self.owner_id) as connection:
            self._authorize(self.owner_id)
            row = await connection.fetchrow("select * from jarvis_notes where owner_id=$1 and id=$2",
                                           self.owner_id, UUID(result["note"]["id"]))
            if row is None:
                raise SourceError("note_receipt_unavailable", "The saved note is no longer available.")
            note = await self._expose(connection, _note_from_record(row))
            if MemorySourceReference.from_metadata(note.memory_source) != MemorySourceReference.from_metadata(
                    result["note"]["memory_source"]):
                raise SourceError("note_receipt_source_changed", "The saved note changed after this receipt.")
            return result
