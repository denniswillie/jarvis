"""Durable per-session defaults and immutable canonical-source area identity."""
from uuid import UUID

from app.evidence.models import OwnedReference
from app.knowledge.models import SourceError
from app.tasks.context_events import ContextIngress
from app.tasks.context_repository import ContextRepository
from app.tasks.models import digest


class SessionKnowledgeAreas:
    def __init__(self, sources, requests, default_area_id):
        if sources.pool is not requests.pool or not isinstance(default_area_id, UUID):
            raise ValueError("Session scope must share the source/request pool and an exact default area")
        self.sources, self.requests, self.default_area_id = sources, requests, default_area_id

    async def _session(self, connection, owner_id, session_id):
        if session_id is None:
            return
        if not isinstance(session_id, UUID):
            raise ValueError("An exact session UUID is required")
        if not await connection.fetchval(
            "select true from jarvis_sessions where owner_id=$1 and id=$2 for share", owner_id, session_id):
            raise SourceError("knowledge_session_unavailable", "The session is outside this owner scope.", 404)

    async def read_in_transaction(self, connection, owner_id, session_id, *, through_sequence=None):
        await self._session(connection, owner_id, session_id)
        # Resolve the receipt and lock its area in one statement. The owner lock
        # still serializes selection changes; no cached/default fallback can
        # conceal a missing explicitly selected area.
        row = await connection.fetchrow("""select a.id as area_id,a.status as area_status,
                selected.revision,selected.receipt_event_id
            from (select 1) anchor left join lateral (
                select s.area_id,s.revision,s.receipt_event_id from jarvis_session_knowledge_selections s
                join jarvis_context_events e on e.owner_id=s.owner_id and e.id=s.receipt_event_id
                where s.owner_id=$1 and s.session_id=$2 and ($3::bigint is null or e.sequence<=$3)
                order by s.revision desc limit 1
            ) selected on true
            join jarvis_knowledge_areas a on a.owner_id=$1
                and a.id=coalesce(selected.area_id,$4::uuid)
            for share of a""", owner_id, session_id, through_sequence, self.default_area_id)
        if row is None:
            raise SourceError("knowledge_area_unavailable", "The selected area is unavailable; no fallback was used.", 404)
        selected = row["revision"] is not None
        return {"area_id": str(row["area_id"]), "area_status": row["area_status"],
                "selection_revision": row["revision"] if selected else 0,
                "selection_event_id": str(row["receipt_event_id"]) if selected else None,
                "origin": "session_selection" if selected else "configured_default"}

    async def read(self, owner_id, session_id):
        """Read one committed snapshot without occupying the mutation lane.

        This is context, not an authorization lease. Effectful callers still
        revalidate their exact request/scope inside their owned transaction.
        Keep the locked read_in_transaction path for composition with writes.
        """
        OwnedReference.validate_owner(owner_id)
        if session_id is not None and not isinstance(session_id, UUID):
            raise ValueError("An exact session UUID is required")
        async with self.sources.pool.acquire() as connection:
            row = await connection.fetchrow("""select
                coalesce(selected.area_id,$3::uuid) as area_id,a.status as area_status,
                selected.revision,selected.receipt_event_id,
                ($2::uuid is null or exists(select 1 from jarvis_sessions
                    where owner_id=$1 and id=$2)) as session_present
                from (select 1) anchor
                left join lateral (
                    select s.area_id,s.revision,s.receipt_event_id
                    from jarvis_session_knowledge_selections s
                    join jarvis_context_events e on e.owner_id=s.owner_id and e.id=s.receipt_event_id
                    where s.owner_id=$1 and s.session_id=$2
                    order by s.revision desc limit 1
                ) selected on true
                left join jarvis_knowledge_areas a
                    on a.owner_id=$1 and a.id=coalesce(selected.area_id,$3::uuid)""",
                owner_id, session_id, self.default_area_id)
        if not row["session_present"]:
            raise SourceError("knowledge_session_unavailable", "The session is outside this owner scope.", 404)
        if row["area_status"] is None:
            raise SourceError("knowledge_area_unavailable", "The selected area is unavailable; no fallback was used.", 404)
        return {"area_id": str(row["area_id"]), "area_status": row["area_status"],
                "selection_revision": row["revision"] if row["revision"] is not None else 0,
                "selection_event_id": str(row["receipt_event_id"]) if row["receipt_event_id"] else None,
                "origin": "session_selection" if row["revision"] is not None else "configured_default"}

    async def task_area_in_transaction(self, connection, owner_id, request, event, *, command_admission=None):
        """Use the original request's scope, never today's default for old speech.

        Task admission already owns its transaction and owner lock. Do not open
        a source transaction or acquire another pool connection from this hook.
        """
        if not connection.is_in_transaction():
            raise SourceError("knowledge_task_scope_invalid", "Task scope requires the admission transaction.", 403)
        if command_admission is not None:
            from app.live_state.admission import require_command, require_command_create
            require_command(command_admission, owner_id, request.request_id, "create")
            require_command_create(command_admission, request)
            if event is not None:
                raise SourceError("knowledge_task_scope_invalid", "Command scope is resolved before its immutable record.", 403)
            if request.knowledge_area_id is not None:
                return request.knowledge_area_id
            selected = await self.read_in_transaction(connection, owner_id, command_admission.session_id)
            if selected["area_status"] != "active":
                raise SourceError("knowledge_area_unavailable", "The selected area is unavailable.", 404)
            return UUID(selected["area_id"])
        record = await ContextRepository(self.requests)._record(connection, owner_id, event["id"])
        if record.session_id != request.session_id or not record.authorizes_request:
            raise SourceError("knowledge_task_scope_invalid", "The exact request is outside this task's session scope.", 403)
        if request.knowledge_area_id is not None:
            return request.knowledge_area_id
        # A compound request may explicitly switch areas before delegating.
        # Honor only that exact request's accepted selection, never a later
        # different utterance's selection. Accepted tasks already replay above.
        selected = await connection.fetchval("""select area_id from jarvis_session_knowledge_selections
            where owner_id=$1 and request_event_id=$2 and request_revision=$3 and session_id=$4""",
            owner_id, record.id, record.revision, record.session_id)
        if selected is not None:
            return selected
        bound = await connection.fetchval("""select area_id from jarvis_knowledge_context_scopes
            where owner_id=$1 and source_namespace=$2 and source_event_id=$3""",
            owner_id, record.source_namespace, record.source_event_id)
        if bound is not None:
            return bound
        if record.source_namespace == "task_api":
            scope = await self.read_in_transaction(connection, owner_id, record.session_id,
                                                   through_sequence=record.origin_sequence)
            return UUID(scope["area_id"])
        # Pre-binding canonical publications can still identify their exact
        # original area. Missing or ambiguous historical evidence is not a
        # reason to substitute the current conversational topic.
        existing = await connection.fetch("""select distinct s.area_id from jarvis_knowledge_context_sources cs
            join jarvis_context_events e on e.owner_id=cs.owner_id and e.id=cs.event_id
            join jarvis_knowledge_sources s on s.owner_id=cs.owner_id and s.id=cs.source_id
            where e.owner_id=$1 and e.source_namespace=$2 and e.source_event_id=$3""",
            owner_id, record.source_namespace, record.source_event_id)
        if len(existing) != 1:
            raise SourceError("knowledge_task_scope_unavailable", "The original request has no unambiguous knowledge area.", 503)
        return existing[0]["area_id"]

    async def source_area(self, owner_id, session_id, source_id):
        """Resolve an exact existing target; session defaults never move it.

        Do not require an active area here. Inspection and already-authorized
        cleanup must remain reachable after archival or deletion; each lifecycle
        operation retains its own state and request checks.
        """
        if not isinstance(source_id, UUID):
            raise ValueError("An exact source UUID is required")
        async with self.sources.transaction(owner_id) as connection:
            await self._session(connection, owner_id, session_id)
            area_id = await connection.fetchval(
                "select area_id from jarvis_knowledge_sources where owner_id=$1 and id=$2",
                owner_id, source_id)
            if area_id is None:
                raise SourceError("knowledge_source_not_found", "No source exists in this owner scope.", 404)
            return area_id

    async def select(self, owner_id, area_id, *, request, expected_revision, access_check=None):
        if not isinstance(area_id, UUID) or type(expected_revision) is not int or expected_revision < 0:
            raise ValueError("Selection requires an exact area UUID and nonnegative selection revision")
        if request.session_id is None:
            raise SourceError("knowledge_session_unavailable", "Selection requires a current voice session.", 403)
        parameters = {"area_id": str(area_id), "expected_revision": expected_revision}
        identity = digest({"operation": "select_knowledge_area", "request": request.parameters()})
        async with self.sources.transaction(owner_id) as connection:
            await self.requests.lock_owner(connection, owner_id)
            source = await self.requests._voice_source(connection, owner_id, request)
            if access_check is not None:
                access_check(owner_id)
            prior = await connection.fetchrow("""select s.*,p.availability from jarvis_session_knowledge_selections s
                join jarvis_context_events e on e.owner_id=s.owner_id and e.id=s.receipt_event_id
                join jarvis_context_payloads p on p.owner_id=e.owner_id and p.id=e.payload_id
                where s.owner_id=$1 and s.request_event_id=$2 and s.request_revision=$3""",
                owner_id, request.event_id, request.revision)
            before = await self.read_in_transaction(connection, owner_id, request.session_id)
            if prior:
                if prior["arguments_digest"] != digest(parameters):
                    raise SourceError("knowledge_selection_request_conflict", "This request already selected another scope or revision.")
                if prior["availability"] != "available":
                    raise SourceError("knowledge_selection_receipt_unavailable", "The earlier selection receipt is unavailable; it was not repeated.")
                applied_revision, receipt_id = prior["revision"], prior["receipt_event_id"]
            else:
                if before["selection_revision"] != expected_revision:
                    raise SourceError("knowledge_selection_revision_conflict", "The session selection changed; read its current state.")
                previous_request = await connection.fetchval("""select request_event_id
                    from jarvis_session_knowledge_selections where owner_id=$1 and session_id=$2 and revision=$3""",
                    owner_id, request.session_id, before["selection_revision"])
                if previous_request is not None:
                    history = ContextRepository(self.requests)
                    previous = await history._record(connection, owner_id, previous_request)
                    incoming = await history._record(connection, owner_id, request.event_id)
                    if incoming.origin_sequence < previous.origin_sequence:
                        raise SourceError("knowledge_selection_request_stale",
                            "A newer user request already selected this session's scope. An older request cannot replace it.")
                await self.sources._area(connection, owner_id, area_id)
                applied_revision = expected_revision + 1
                receipt = await ContextRepository(self.requests).append_in_transaction(connection, owner_id, ContextIngress(
                    session_id=request.session_id, source_namespace="knowledge-session-area", source_event_id=identity,
                    receipt_id=identity, expected_revision=0, kind="tool_outcome", trust="tool_receipt",
                    payload={"area_id": str(area_id), "selection_revision": applied_revision},
                    source_payload_ids=(source["payload_id"],)))
                receipt_id = receipt.id
                await connection.execute("""insert into jarvis_session_knowledge_selections
                    (owner_id,session_id,revision,area_id,request_event_id,request_revision,arguments_digest,receipt_event_id)
                    values($1,$2,$3,$4,$5,$6,$7,$8)""", owner_id, request.session_id, applied_revision,
                    area_id, request.event_id, request.revision, digest(parameters), receipt_id)
            current = await self.read_in_transaction(connection, owner_id, request.session_id)
            if access_check is not None:
                access_check(owner_id)
            return {"selection": current, "applied_area_id": str(area_id), "applied_revision": applied_revision,
                    "receipt_event_id": str(receipt_id), "replayed": prior is not None,
                    "historical_sources_moved": False, "other_sessions_changed": False, "provider_contacted": False}

    async def bind_context(self, connection, owner_id, record, *, requested_area_id=None):
        if not connection.is_in_transaction() or record.owner_id != owner_id:
            raise SourceError("knowledge_source_scope_mismatch", "Scope binding requires the owned canonical transaction.", 403)
        if requested_area_id is not None and not isinstance(requested_area_id, UUID):
            raise ValueError("A trusted publication scope requires an exact area UUID")
        bound = await connection.fetchval("""select area_id from jarvis_knowledge_context_scopes
            where owner_id=$1 and source_namespace=$2 and source_event_id=$3""",
            owner_id, record.source_namespace, record.source_event_id)
        if bound is not None:
            if requested_area_id is not None and requested_area_id != bound:
                raise SourceError("knowledge_context_scope_conflict", "An existing context stream cannot change area.")
            return bound
        # Existing publications outrank today's configured default. This also
        # preserves forgotten sources so a scope change cannot recreate them.
        existing = await connection.fetch("""select distinct s.area_id from jarvis_knowledge_context_sources cs
            join jarvis_context_events e on e.owner_id=cs.owner_id and e.id=cs.event_id
            join jarvis_knowledge_sources s on s.owner_id=cs.owner_id and s.id=cs.source_id
            where e.owner_id=$1 and e.source_namespace=$2 and e.source_event_id=$3""",
            owner_id, record.source_namespace, record.source_event_id)
        if len(existing) > 1:
            raise SourceError("knowledge_context_scope_ambiguous", "Existing source mappings disagree on this stream's area; no scope was guessed.")
        if existing:
            area_id = existing[0]["area_id"]
            if requested_area_id is not None and requested_area_id != area_id:
                raise SourceError("knowledge_context_scope_conflict", "An existing publication cannot change area.")
        elif requested_area_id is not None:
            area_id = requested_area_id
        else:
            scope = await self.read_in_transaction(connection, owner_id, record.session_id,
                                                   through_sequence=record.origin_sequence)
            area_id = UUID(scope["area_id"])
        await connection.execute("""insert into jarvis_knowledge_context_scopes
            (owner_id,source_namespace,source_event_id,area_id,bound_event_id,origin_sequence)
            values($1,$2,$3,$4,$5,$6)""", owner_id, record.source_namespace,
            record.source_event_id, area_id, record.id, record.origin_sequence)
        return area_id
