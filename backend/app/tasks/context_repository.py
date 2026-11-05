"""Owner-scoped canonical history, preserving identity instead of matching text."""
from __future__ import annotations

import json
from uuid import UUID

from app.sessions.repository import SessionRepository
from app.tasks.context_events import ContextIngress, ContextRecord, TaskContextManifest
from app.tasks.context_history import pending_tool_acknowledgment, tool_ack_recovery_instructions
from app.tasks.models import ClaimLost, TaskError, digest, json_object, json_text
from app.tasks.repository import TaskRepository


RECORD_SQL = """select e.*,p.availability,p.content_digest,p.content,
    (select min(first.sequence) from jarvis_context_events first where first.owner_id=e.owner_id
        and first.source_namespace=e.source_namespace and first.source_event_id=e.source_event_id) as origin_sequence,
    not exists(select 1 from jarvis_context_events newer where newer.owner_id=e.owner_id
        and newer.source_namespace=e.source_namespace and newer.source_event_id=e.source_event_id
        and newer.revision>e.revision) as is_current
    from jarvis_context_events e join jarvis_context_payloads p on p.owner_id=e.owner_id and p.id=e.payload_id"""


def context_record(row):
    return ContextRecord.model_validate(dict(row) | {"content": json_object(row["content"]) if row["content"] is not None else None})


class ContextRepository:
    def __init__(self, tasks: TaskRepository, *, publication=None):
        self.tasks = tasks
        self.pool = tasks.pool
        self.publication = publication

    async def _record(self, connection, owner_id, event_id):
        row = await connection.fetchrow(RECORD_SQL + " where e.owner_id=$1 and e.id=$2", owner_id, event_id)
        if row is None:
            raise TaskError("context_event_not_found", "No context event exists in this owner scope.", 404)
        return context_record(row)

    async def _records(self, connection, owner_id, event_ids):
        """Resolve exact source identities in one owner-scoped database request."""
        identities = list(dict.fromkeys(event_ids))
        if not identities:
            return {}
        rows = await connection.fetch(
            RECORD_SQL + " where e.owner_id=$1 and e.id=any($2::uuid[])",
            owner_id, identities,
        )
        records = {row["id"]: context_record(row) for row in rows}
        if any(identity not in records for identity in identities):
            raise TaskError("context_event_not_found", "No context event exists in this owner scope.", 404)
        return records

    async def append(self, owner_id: str, event: ContextIngress) -> ContextRecord:
        transaction = (self.publication.transaction(self.tasks, owner_id) if self.publication is not None
                       else self.tasks.transaction(owner_id))
        async with transaction as connection:
            record = await self.append_in_transaction(connection, owner_id, event)
            await self._project_voice_transcript(connection, owner_id, record)
            return record

    async def _project_voice_transcript(self, connection, owner_id, record):
        """Commit policy-admitted voice messages with their exact originals.

        User ASR revisions replace displayed text. An assistant utterance is
        projected once; later playback telemetry does not become another answer.
        Neither case infers identity from prose or claims that output was heard.
        """
        content = record.content or {}
        if (record.session_id is None or not record.source_namespace.startswith("voice:")
                or content.get("audio_provider") != "openai" or not record.is_current
                or record.availability != "available" or record.finality not in {"final", "revised"}):
            return
        if record.kind == "user_transcript" and record.authorizes_request:
            role, metadata = "user", {}
            item_id, response_id = content.get("provider_item_id"), None
        elif record.kind == "assistant_output":
            metadata = content.get("conversation_message")
            if not isinstance(metadata, dict) or content.get("completion_state") == "interrupted":
                return
            item_id, response_id = content.get("item_id"), content.get("response_id")
            output = metadata.get("realtime_output")
            if (not isinstance(output, dict) or output.get("delivery_disposition") != "forwarded"
                    or not isinstance(item_id, str) or not item_id
                    or output.get("item_id") != item_id
                    or not isinstance(response_id, str) or not response_id
                    or not isinstance(content.get("text"), str) or not content["text"].strip()):
                raise TaskError("context_message_projection_invalid", "A conversation copy requires exact forwarded output.")
            role = "assistant"
        else:
            return
        if await connection.fetchval("""select true from jarvis_messages m
                join jarvis_context_events e on e.owner_id=m.owner_id and e.payload_id=m.source_payload_id
                where m.owner_id=$1 and m.session_id=$2 and m.role=$3
                    and e.session_id=m.session_id and e.source_namespace=$4 and e.source_event_id=$5
                    and ($3='assistant' or e.payload_id=$6)""",
                owner_id, record.session_id, role, record.source_namespace, record.source_event_id, record.payload_id):
            return
        if role == "user":
            # Source-bound rows cannot be rebound. Keep their identity but clear
            # obsolete ASR text; canonical originals retain explicit history.
            await connection.execute("""update jarvis_messages m set content='',
                    metadata=jsonb_build_object('response_kind','source_bound_superseded',
                        'canonical_event_id',e.id,'canonical_revision',e.revision)
                from jarvis_context_events e
                where m.owner_id=$1 and m.session_id=$2 and m.role='user' and m.content<>''
                    and e.owner_id=m.owner_id and e.payload_id=m.source_payload_id
                    and e.session_id=m.session_id and e.source_namespace=$3
                    and e.source_event_id=$4 and e.revision<$5""",
                owner_id, record.session_id, record.source_namespace, record.source_event_id, record.revision)
        await SessionRepository(self.pool).create_message(
            owner_id=owner_id, session_id=record.session_id, role=role, modality="audio",
            content=content["text"].strip(), provider_item_id=item_id, provider_response_id=response_id,
            source_payload_id=record.payload_id, connection=connection,
            metadata={**metadata, "audio_provider":"openai", "audio_session_id":content.get("audio_session_id"),
                      "canonical_event_id":str(record.id), "canonical_revision":record.revision},
        )

    async def append_in_transaction(self, connection, owner_id, event, *, payload_id=None, publication_area_id=None):
        """Internal composition seam; caller holds the source/task owner transaction."""
        if not connection.is_in_transaction():
            raise RuntimeError("Canonical ingress requires a transaction")
        values = event.model_dump(mode="json")
        parents = values.pop("source_payload_ids", [])
        scope = {}
        if publication_area_id is not None:
            if not isinstance(publication_area_id, UUID) or self.publication is None:
                raise TaskError("context_publication_scope_unavailable", "A scoped receipt requires a trusted source publisher.")
            scope = {"publication_area_id": str(publication_area_id)}
        parameters = digest(values | ({"source_payload_ids": parents} if parents else {}) | scope)
        payload_text = json_text(values["payload"])
        receipt = await connection.fetchrow("""select * from jarvis_context_ingress_receipts
            where owner_id=$1 and source_namespace=$2 and source_event_id=$3 and receipt_id=$4""",
            owner_id, event.source_namespace, event.source_event_id, event.receipt_id)
        if receipt:
            if receipt["parameters_digest"] != parameters:
                raise TaskError("context_receipt_conflict", "The same ingress identity has different content.")
            return await self._record(connection, owner_id, receipt["event_id"])
        for parent in event.source_payload_ids:
            status = await connection.fetchval("select lock_jarvis_copy_source($1,$2)", owner_id, parent)
            if status != "available":
                code = "context_source_removed" if status in {"removed", "revoked"} else "context_dependency_unavailable"
                raise TaskError(code, "The original evidence is no longer available for this copy.", 403)
        if event.session_id is not None and not await connection.fetchval(
            "select true from jarvis_sessions where owner_id=$1 and id=$2 for share", owner_id, event.session_id,
        ):
            raise TaskError("context_session_not_found", "The context session is outside this owner scope.", 404)
        previous = await connection.fetchrow(RECORD_SQL + """ where e.owner_id=$1 and e.source_namespace=$2
            and e.source_event_id=$3 order by e.revision desc limit 1""", owner_id, event.source_namespace, event.source_event_id)
        revision = previous["revision"] if previous else 0
        if event.expected_revision is not None and event.expected_revision != revision:
            raise TaskError("context_revision_conflict", "The source has a newer context revision.")
        if previous:
            if (previous["session_id"], previous["kind"], previous["trust"]) != (event.session_id, event.kind, event.trust):
                raise TaskError("context_identity_conflict", "An existing source cannot change session, kind, or origin.")
            if previous["availability"] in {"deleted", "expired", "unavailable"}:
                raise TaskError("context_source_removed", "A late provider event cannot restore removed source content.", 403)
            if previous["finality"] in {"final", "revised"} and event.finality in {"provisional", "unavailable"}:
                raise TaskError("context_finality_conflict", "A finalized source cannot revert to a provisional input.")
        if payload_id is None:
            payload_id = await connection.fetchval("""insert into jarvis_context_payloads(owner_id,availability,content)
                values($1,$2,$3::jsonb) returning id""", owner_id,
                "not_finalized" if event.finality == "provisional" else "available", payload_text)
        else:
            original = await connection.fetchrow(
                "select availability,content from jarvis_context_payloads where owner_id=$1 and id=$2 for share",
                owner_id,payload_id,
            )
            if (event.finality not in {"final","revised"} or original is None
                    or original["availability"] != "available" or json_object(original["content"]) != values["payload"]):
                raise TaskError("context_payload_mismatch", "The original payload does not match canonical ingress.")
        if event.source_payload_ids:
            await connection.executemany("""insert into jarvis_context_payload_dependencies
                (owner_id,source_payload_id,dependent_payload_id) values($1,$2,$3) on conflict do nothing""",
                [(owner_id, parent, payload_id) for parent in event.source_payload_ids])
        identity = await connection.fetchval("""insert into jarvis_context_events(owner_id,session_id,source_namespace,
            source_event_id,revision,supersedes_event_id,kind,finality,trust,source_time,payload_id)
            values($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11) returning id""", owner_id, event.session_id,
            event.source_namespace, event.source_event_id, revision+1, previous["id"] if previous else None,
            event.kind, event.finality, event.trust, event.source_time, payload_id)
        await connection.execute("""insert into jarvis_context_ingress_receipts(owner_id,source_namespace,source_event_id,
            receipt_id,parameters_digest,event_id) values($1,$2,$3,$4,$5,$6)""", owner_id,
            event.source_namespace, event.source_event_id, event.receipt_id, parameters, identity)
        record = await self._record(connection, owner_id, identity)
        if self.publication is not None:
            if publication_area_id is None:
                await self.publication.publish(connection, owner_id, record)
            else:
                await self.publication.publish(connection, owner_id, record, requested_area_id=publication_area_id)
        return record

    async def get(self, owner_id: str, event_id: UUID, *, session_id: UUID | None = None):
        async with self.pool.acquire() as connection:
            record = await self._record(connection, owner_id, event_id)
        if session_id is not None and record.session_id != session_id:
            raise TaskError("context_event_not_found", "No context event exists in this session scope.", 404)
        return record

    async def request_evidence(self, owner_id, session_id, *, source_namespace, source_event_id):
        """Read one committed source snapshot; task mutations revalidate under their owner lock."""
        # This single statement has no side effects or cross-query consistency
        # requirement. Waiting for context-manifest writes here delays foreground
        # reads without extending authority: admission rechecks the exact revision.
        async with self.pool.acquire() as connection:
            row = await connection.fetchrow(RECORD_SQL + """ where e.owner_id=$1 and e.session_id=$2
                and e.source_namespace=$3 and e.source_event_id=$4 order by e.revision desc limit 1""",
                owner_id, session_id, source_namespace, source_event_id)
            if row is None or not context_record(row).authorizes_request:
                raise TaskError("context_request_not_finalized", "This exact input has no current finalized user transcript; request clarification rather than inventing authority.", 409)
            return context_record(row)

    async def voice_history(self, owner_id, session_id, *, limit=64):
        async with self.tasks.read_transaction() as connection:
            return await self.voice_history_in_transaction(connection, owner_id, session_id, limit=limit)

    async def voice_history_in_transaction(self, connection, owner_id, session_id, *, limit=64):
        """Reuse a caller-owned transaction without acquiring another pool slot."""
        if not connection.is_in_transaction():
            raise RuntimeError("Canonical voice history requires a transaction")
        if not 1 <= limit <= 128:
            raise ValueError("invalid voice history bound")
        if not await connection.fetchval("select true from jarvis_sessions where owner_id=$1 and id=$2",
                                         owner_id, session_id):
            raise TaskError("context_session_not_found", "The voice context is outside this owner scope.", 404)
        scope = """ e.owner_id=$1 and e.session_id=$2 and e.source_namespace like 'voice:%'
            and e.kind in ('user_request','user_transcript','assistant_output','played_output_boundary',
                           'tool_outcome','context_gap','correction')
            and not exists(select 1 from jarvis_context_events newer where newer.owner_id=e.owner_id
                and newer.source_namespace=e.source_namespace and newer.source_event_id=e.source_event_id
                and newer.revision>e.revision)"""
        total = await connection.fetchval("select count(*) from jarvis_context_events e where " + scope,
                                          owner_id, session_id)
        rows = await connection.fetch(RECORD_SQL + " where " + scope + " order by origin_sequence desc limit $3",
                                      owner_id, session_id, limit)
        return {"total":total, "records":[context_record(row) for row in rows]}


    async def claim_tool_acknowledgment(self, owner_id, session_id, expected, *, audio_session_id):
        """One durable speech attempt per reset, never a domain-operation replay.

        A crash after this receipt is an uncertain acknowledgment, not a reason
        to repeat effects or claim the user heard anything.
        """
        transaction = (self.publication.transaction(self.tasks, owner_id) if self.publication is not None
                       else self.tasks.transaction(owner_id))
        async with transaction as connection:
            snapshot = await self.voice_history_in_transaction(connection, owner_id, session_id)
            candidate = pending_tool_acknowledgment(snapshot["records"])
            if candidate is None or candidate != expected:
                return None
            identities = {candidate["reset_event_id"], candidate["request_event_id"], *candidate["tool_event_ids"]}
            sources = tuple(record.payload_id for record in snapshot["records"] if str(record.id) in identities)
            await self.append_in_transaction(connection, owner_id, ContextIngress(
                session_id=session_id, source_namespace="voice:tool-acknowledgment",
                source_event_id=candidate["reset_event_id"], receipt_id="attempt", expected_revision=0,
                kind="context_gap", trust="derived", source_payload_ids=sources,
                payload={"reason": "tool_acknowledgment_attempted", "audio_session_id": audio_session_id,
                         **candidate, "meaning": "Speech attempt only; not delivery or another tool operation."}))
            return tool_ack_recovery_instructions(candidate)
    async def archive(self, owner_id, session_id, *, after=0, through=None, limit=100):
        if after < 0 or through is not None and through < 0 or not 1 <= limit <= 200:
            raise ValueError("invalid context archive pagination")
        async with self.pool.acquire() as connection:
            rows = await connection.fetch(RECORD_SQL + """ where e.owner_id=$1 and e.session_id=$2 and e.sequence>$3
                and ($4::bigint is null or e.sequence<=$4) order by e.sequence limit $5""", owner_id, session_id, after, through, limit)
        return [context_record(row) for row in rows]

    async def pin(self, owner_id, session_id, event_id, *, reason):
        if not 1 <= len(reason.strip()) <= 200 or "\x00" in reason:
            raise ValueError("invalid context pin reason")
        async with self.tasks.transaction(owner_id) as connection:
            event = await self._record(connection, owner_id, event_id)
            if event.session_id != session_id or not event.is_current or event.availability != "available":
                raise TaskError("context_pin_unavailable", "A pin requires an available current source in this session.")
            await connection.execute("""insert into jarvis_context_pins(owner_id,session_id,event_id,reason) values($1,$2,$3,$4)
                on conflict(owner_id,session_id,event_id) do update set reason=excluded.reason""", owner_id, session_id, event_id, reason)

    async def _task_sessions(self, connection, task):
        """Original session plus explicitly attached voice sessions, never all owner history."""
        sessions = {task.session_id} if task.session_id is not None else set()
        bindings = await connection.fetch("""select request_event_id,request_revision,session_id from jarvis_task_session_context
            where owner_id=$1 and task_id=$2""", task.owner_id, task.id)
        records = await self._records(connection, task.owner_id, [binding["request_event_id"] for binding in bindings])
        sources = []
        for binding in bindings:
            record = records[binding["request_event_id"]]
            command = None
            if record.source_namespace == "voice_command" or record.kind == "command_request":
                from app.live_state.admission import task_command_source
                command = await task_command_source(self.tasks, connection, task, event_id=record.id,
                    event_revision=binding["request_revision"], current=False, lock=False)
            if (record.authorizes_request or command is not None) and record.session_id == binding["session_id"]:
                sessions.add(record.session_id)
                sources.append(record)
        return list(sessions), sources

    async def snapshot(self, owner_id, session_id, *, max_messages, max_pins, task_id=None):
        if not 1 <= max_messages <= 512 or not 1 <= max_pins <= 128:
            raise ValueError("invalid context snapshot bounds")
        async with self.tasks.read_transaction() as connection:
            if session_id is not None and not await connection.fetchval(
                "select true from jarvis_sessions where owner_id=$1 and id=$2", owner_id, session_id,
            ):
                raise TaskError("context_session_not_found", "The context session is outside this owner scope.", 404)
            sessions = [session_id] if session_id is not None else []
            scope_sources = []
            if task_id is not None:
                task = await self.tasks.load(connection, owner_id, task_id)
                sessions, scope_sources = await self._task_sessions(connection, task)
            current = """ e.owner_id=$1 and e.session_id=any($2::uuid[]) and not exists(select 1 from jarvis_context_events newer
                where newer.owner_id=e.owner_id and newer.source_namespace=e.source_namespace
                    and newer.source_event_id=e.source_event_id and newer.revision>e.revision)"""
            pinned_membership = """exists(select 1 from jarvis_context_pins pin
                join jarvis_context_events original on original.owner_id=pin.owner_id and original.id=pin.event_id
                where pin.owner_id=e.owner_id and pin.session_id=e.session_id and original.source_namespace=e.source_namespace
                    and original.source_event_id=e.source_event_id)"""
            # Batch projection-only reads, not authorization. Count metadata over
            # the full scope, but load payloads only AFTER both windows are bounded.
            # Keep the repeatable-read transaction and downstream source guards.
            row = await connection.fetchrow(
                """with current_events as materialized (
                    select e.id,
                        (select min(first.sequence) from jarvis_context_events first
                         where first.owner_id=e.owner_id and first.source_namespace=e.source_namespace
                             and first.source_event_id=e.source_event_id) as origin_sequence,
                        """ + pinned_membership + """ as is_pinned
                    from jarvis_context_events e where """ + current + """
                ), recent_ids as (
                    select id from current_events order by origin_sequence desc limit $3
                ), pinned_ids as (
                    select id from current_events where is_pinned order by origin_sequence desc limit $4
                ), selected_ids as (
                    select id from recent_ids union select id from pinned_ids
                ), records as materialized (""" + RECORD_SQL + """
                    where e.owner_id=$1 and e.id in (select id from selected_ids)
                ) select
                    coalesce((select last_sequence from jarvis_context_streams where owner_id=$1),0) as watermark,
                    (select count(*) from current_events) as total,
                    (select count(*) from current_events where is_pinned) as pinned_total,
                    coalesce((select jsonb_agg(to_jsonb(r) order by r.origin_sequence desc)
                        from records r join recent_ids using(id)),'[]'::jsonb) as recent,
                    coalesce((select jsonb_agg(to_jsonb(r) order by r.origin_sequence desc)
                        from records r join pinned_ids using(id)),'[]'::jsonb) as pinned,
                    coalesce((select array_agg(c.id) from current_events c
                        join selected_ids using(id) where c.is_pinned),'{}'::uuid[]) as selected_pinned_ids""",
                owner_id, sessions, max_messages, max_pins,
            )
            recent = json.loads(row["recent"]) if isinstance(row["recent"], str) else row["recent"]
            pinned = json.loads(row["pinned"]) if isinstance(row["pinned"], str) else row["pinned"]
        return {"watermark": row["watermark"], "total": row["total"], "recent": [context_record(item) for item in recent],
                "pinned_total": row["pinned_total"], "pinned": [context_record(item) for item in pinned],
                "selected_pinned_ids": set(row["selected_pinned_ids"]), "scope_sources": scope_sources,
                "scope_session_count": len(sessions)}

    async def store_manifest(self, claim, *, watermark, records, content):
        async with self.tasks.transaction(claim.task.owner_id) as connection:
            task, _ = await self.tasks.lock_claim(connection, claim)
            if await self.tasks.context_problem(connection, task):
                raise ClaimLost()
            sessions, scope_sources = await self._task_sessions(connection, task)
            # Even omitted session content depends on its original admission.
            unique = {record.id: record for record in (*scope_sources, *records)}
            if unique:
                # Lock source payloads through the dependency insert. Otherwise
                # deletion could run after validation but before its manifest
                # dependency exists, leaving a copied private payload active.
                await connection.fetch("""select id from jarvis_context_payloads where owner_id=$1
                    and id=any($2::uuid[]) order by id for share""", task.owner_id,
                    [record.payload_id for record in unique.values()])
            current = await self._records(connection, task.owner_id, unique)
            for selected in unique.values():
                record = current[selected.id]
                if ((not record.is_current and not selected.streaming_assistant_snapshot)
                        or record.revision != selected.revision or record.payload_id != selected.payload_id
                        or record.content_digest != selected.content_digest
                        or record.availability != selected.availability or record.session_id not in sessions):
                    raise TaskError("context_snapshot_changed", "Context changed while the worker manifest was assembled.")
            payload_id = await self.tasks.payload(connection, task.owner_id, content)
            row = await connection.fetchrow("""insert into jarvis_task_context_manifests(owner_id,task_id,task_revision,
                context_watermark,payload_id) values($1,$2,$3,$4,$5) returning *""", task.owner_id, task.id, task.revision, watermark, payload_id)
            if unique:
                await connection.executemany(
                    """insert into jarvis_task_context_sources(owner_id,manifest_id,event_id,event_revision,
                        payload_id,content_digest,source_usage) values($1,$2,$3,$4,$5,$6,$7)""",
                    [(task.owner_id, row["id"], record.id, record.revision, record.payload_id,
                      record.content_digest, record.dependency_usage) for record in unique.values()],
                )
            await connection.execute("update jarvis_task_attempts set context_manifest_id=$3 where owner_id=$1 and id=$2", task.owner_id, claim.attempt_id, row["id"])
            await self.tasks.emit(connection, task, "context_manifest_attached", payload={"manifest_id": str(row["id"]), "watermark": watermark}, attempt_id=claim.attempt_id)
        return TaskContextManifest.model_validate(dict(row) | {"content": content})

    async def read_manifest(self, owner_id, task_id, manifest_id):
        async with self.tasks.transaction(owner_id) as connection:
            task = await self.tasks.load(connection, owner_id, task_id, lock=True)
            row = await connection.fetchrow("""select m.*,p.content from jarvis_task_context_manifests m
                join jarvis_context_payloads p on p.owner_id=m.owner_id and p.id=m.payload_id
                where m.owner_id=$1 and m.task_id=$2 and m.id=$3""", owner_id, task_id, manifest_id)
            if row is None:
                raise TaskError("context_manifest_not_found", "No manifest exists for this task.", 404)
            if row["status"] != "active" or row["task_revision"] != task.revision or await self.tasks.context_problem(connection, task):
                raise TaskError("context_manifest_unavailable", "This manifest is no longer current or authorized.", 409)
            return TaskContextManifest.model_validate(dict(row) | {"content": json_object(row["content"]) if row["content"] is not None else None})

    async def _archive_scope(self, connection, claim, through):
        task, attempt = await self.tasks.lock_claim(connection, claim)
        if await self.tasks.context_problem(connection, task):
            raise ClaimLost()
        manifest = await connection.fetchrow(
            """select id,context_watermark,status,task_revision from jarvis_task_context_manifests
            where owner_id=$1 and task_id=$2 and id=$3 for share""",
            task.owner_id, task.id, attempt["context_manifest_id"],
        )
        if manifest is None:
            raise TaskError("context_manifest_required", "Attach a worker manifest before retrieving its source archive.")
        if manifest["status"] != "active" or manifest["task_revision"] != task.revision:
            raise ClaimLost()
        watermark = manifest["context_watermark"]
        if through is not None and through > watermark:
            raise TaskError("context_archive_outside_manifest", "This read exceeds the task's context watermark.")
        return task, manifest["id"], watermark if through is None else through

    async def _bind_archive_sources(self, connection, claim, task, manifest_id, records):
        if records:
            await connection.executemany(
                """insert into jarvis_task_context_sources(owner_id,manifest_id,event_id,event_revision,
                    payload_id,content_digest,source_usage) values($1,$2,$3,$4,$5,$6,$7)
                    on conflict(owner_id,manifest_id,event_id) do nothing""",
                [(task.owner_id, manifest_id, record.id, record.revision, record.payload_id,
                  record.content_digest, record.dependency_usage) for record in records],
            )
            await self.tasks.emit(
                connection, task, "context_archive_read",
                payload={"manifest_id": str(manifest_id), "event_ids": [str(record.id) for record in records]},
                attempt_id=claim.attempt_id,
            )

    async def archive_for_claim(self, claim, *, after=0, through=None, limit=100, pinned_only=False):
        if (isinstance(after, bool) or not isinstance(after, int) or after < 0
                or through is not None and (isinstance(through, bool) or not isinstance(through, int) or through < 0)
                or isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 200
                or not isinstance(pinned_only, bool)):
            raise ValueError("invalid context archive pagination")
        async with self.tasks.transaction(claim.task.owner_id) as connection:
            task, manifest_id, watermark = await self._archive_scope(connection, claim, through)
            pinned_filter = ""
            if pinned_only:
                # One source per pin, selecting its latest revision as of this
                # manifest. Pinning an original also pins its later corrections.
                pinned_filter = """ and not exists(select 1 from jarvis_context_events newer
                    where newer.owner_id=e.owner_id and newer.source_namespace=e.source_namespace
                        and newer.source_event_id=e.source_event_id and newer.revision>e.revision
                        and newer.sequence<=$4)
                    and exists(select 1 from jarvis_context_pins pin
                        join jarvis_context_events original on original.owner_id=pin.owner_id and original.id=pin.event_id
                        where pin.owner_id=e.owner_id and pin.session_id=e.session_id
                            and original.source_namespace=e.source_namespace
                            and original.source_event_id=e.source_event_id)"""
            sessions, _ = await self._task_sessions(connection, task)
            rows = await connection.fetch(
                RECORD_SQL + """ where e.owner_id=$1 and e.session_id=any($2::uuid[]) and e.sequence>$3
                    and e.sequence<=$4""" + pinned_filter + " order by e.sequence limit $5 for share of p",
                task.owner_id, sessions, after, watermark, limit,
            )
            records = [context_record(row) for row in rows]
            await self._bind_archive_sources(connection, claim, task, manifest_id, records)
            return records

    async def source_for_claim(self, claim, event_id):
        async with self.tasks.transaction(claim.task.owner_id) as connection:
            task, manifest_id, watermark = await self._archive_scope(connection, claim, None)
            sessions, _ = await self._task_sessions(connection, task)
            row = await connection.fetchrow(
                RECORD_SQL + """ where e.owner_id=$1 and (e.session_id=any($2::uuid[]) or e.id=$5) and e.id=$3
                    and e.sequence<=$4 for share of p""",
                task.owner_id, sessions, event_id, watermark, task.request_event_id,
            )
            if row is None:
                raise TaskError("context_event_not_found", "No source exists in this task's context snapshot.", 404)
            record = context_record(row)
            await self._bind_archive_sources(connection, claim, task, manifest_id, [record])
            return record
