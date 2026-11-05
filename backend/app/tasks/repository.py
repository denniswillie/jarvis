"""Short PostgreSQL transactions for tasks; provider I/O belongs outside them.

Lock order: scheduler advisory lock (claiming only), owner advisory lock, task
row, source/grant rows. Task revision is the user's work/context revision;
attempt fences change independently when execution ownership changes.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Literal
from uuid import UUID, uuid4

import asyncpg

from app.runtime.feature_config import TaskLimits
from app.live_state.admission import (
    append_task_command, command_parameters, require_command, require_command_create,
    require_command_target, task_command_source,
)
from app.tasks.models import (
    BudgetExceeded, ClaimLost, Reconciliation, TaskClaim, TaskCreate, TaskError, TaskMutation,
    TaskStatus, TaskView, TaskVoiceSource, WorkerOutcome, digest, json_object, json_text, require_transition,
)


ACTIVE_ATTEMPTS = ("running", "recovering", "outcome_unknown")


class TaskRepository:
    def __init__(self, pool: asyncpg.Pool, limits: TaskLimits):
        self.pool = pool
        self.limits = limits

    @asynccontextmanager
    async def transaction(self, owner_id: str | None = None):
        async with self.pool.acquire() as connection:
            async with connection.transaction():
                if owner_id is not None:
                    await self.lock_owner(connection, owner_id)
                yield connection

    @asynccontextmanager
    async def read_transaction(self):
        """Consistent committed reads without blocking the owner's mutation lane."""
        async with self.pool.acquire() as connection:
            async with connection.transaction(isolation="repeatable_read", readonly=True):
                yield connection

    async def lock_owner(self, connection, owner_id):
        await connection.execute(
            "select pg_advisory_xact_lock(hashtextextended(current_schema() || ':jarvis-task-owner:' || $1,0))", owner_id,
        )

    async def try_lock_owner(self, connection, owner_id):
        """Same transaction lock and key, without waiting behind another writer."""
        return await connection.fetchval(
            "select pg_try_advisory_xact_lock(hashtextextended(current_schema() || ':jarvis-task-owner:' || $1,0))",
            owner_id,
        ) is True

    async def lock_owner_shared(self, connection, owner_id):
        await connection.execute(
            "select pg_advisory_xact_lock_shared(hashtextextended(current_schema() || ':jarvis-task-owner:' || $1,0))", owner_id,
        )

    async def try_lock_owner_shared(self, connection, owner_id):
        return await connection.fetchval(
            "select pg_try_advisory_xact_lock_shared(hashtextextended(current_schema() || ':jarvis-task-owner:' || $1,0))",
            owner_id,
        ) is True

    async def load(self, connection, owner_id, task_id, *, lock: bool | Literal["share"] = False) -> TaskView:
        row = await connection.fetchrow("""select t.*,p.content as description from jarvis_background_tasks t
            left join jarvis_context_payloads p on p.owner_id=t.owner_id and p.id=t.description_payload_id
            where t.owner_id=$1 and t.id=$2""" + (
                " for share of t" if lock == "share" else " for update of t" if lock else ""), owner_id, task_id)
        if row is None:
            raise TaskError("task_not_found", "The task was not found for this owner.", 404)
        values = dict(row)
        values["budget"] = json_object(values["budget"])
        if values.pop("description") is not None:
            description = json_object(row["description"])
            values.update(title=description.get("title"), requested_outcome=description.get("requested_outcome"), mode=description.get("mode"))
        return TaskView.model_validate(values)

    async def get(self, owner_id: str, task_id: UUID) -> TaskView:
        async with self.pool.acquire() as connection:
            return await self.load(connection, owner_id, task_id)

    async def list(self, owner_id: str, *, limit: int = 20, offset: int = 0, active_only: bool = False):
        if not 1 <= limit <= 100 or not 0 <= offset <= 100_000:
            raise ValueError("invalid task pagination")
        async with self.pool.acquire() as connection:
            ids = await connection.fetch("""select id from jarvis_background_tasks where owner_id=$1
                and (not $4 or status not in ('succeeded','failed','cancelled'))
                order by created_at desc,id desc limit $2 offset $3""", owner_id, limit, offset, active_only)
            return [await self.load(connection, owner_id, row["id"]) for row in ids]

    @staticmethod
    def reference(task_id: UUID) -> str:
        return "task_" + task_id.hex[:12]

    async def resolve_reference(self, owner_id: str, reference: str) -> UUID:
        # An exact issued token, never fuzzy matching a title or truncated UUID.
        async with self.pool.acquire() as connection:
            rows = await connection.fetch("""select id from jarvis_background_tasks
                where owner_id=$1 and 'task_' || left(replace(id::text,'-',''),12)=$2 limit 2""",
                owner_id, reference)
        if len(rows) != 1:
            raise TaskError("task_reference_unavailable",
                "Copy an exact task_ref from list_background_tasks; this reference is missing or ambiguous.", 404)
        return rows[0]["id"]

    async def context_summaries(self, owner_id: str, *, limit: int = 20) -> dict:
        """Bounded active/recent terminal summaries; no per-task loads or answers."""
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError("invalid task context limit")
        async with self.pool.acquire() as connection:
            rows = await connection.fetch(
                """with ranked as (
                    select t.*,
                        t.status in ('succeeded','failed','cancelled') as is_terminal,
                        count(*) over(partition by t.status in ('succeeded','failed','cancelled')) as category_total,
                        row_number() over(partition by t.status in ('succeeded','failed','cancelled')
                            order by t.updated_at desc,t.id) as position
                    from jarvis_background_tasks t where t.owner_id=$1
                )
                select t.id,t.session_id,t.status,t.revision,t.updated_at,t.is_terminal,t.category_total,
                    p.content->>'title' as title,p.content->>'mode' as mode,t.reason,
                    r.content->>'answer' as pending_question,
                    (select jsonb_build_object('event_type',e.event_type,'sequence',e.sequence)
                     from jarvis_task_events e where e.owner_id=t.owner_id and e.task_id=t.id
                     and e.task_revision=t.revision and e.event_type in
                     ('accepted','running','runtime_started','waiting_for_input','paused','reconciling',
                      'update_requested','cancel_requested','resume_requested','succeeded','failed','cancelled')
                     order by e.sequence desc limit 1) as milestone
                from ranked t
                left join jarvis_context_payloads p on p.owner_id=t.owner_id and p.id=t.description_payload_id
                    and p.availability='available'
                left join jarvis_context_payloads r on r.owner_id=t.owner_id and r.id=t.result_payload_id
                    and r.availability='available' and t.status='waiting_for_input'
                    and exists(select 1 from jarvis_context_events source
                        join jarvis_context_payloads original on original.owner_id=source.owner_id and original.id=source.payload_id
                        where source.owner_id=t.owner_id and source.id=t.request_event_id
                        and source.revision=t.request_revision and source.trust='user_input'
                        and source.finality in ('final','revised') and original.availability='available'
                        and not exists(select 1 from jarvis_context_events newer where newer.owner_id=source.owner_id
                            and newer.source_namespace=source.source_namespace and newer.source_event_id=source.source_event_id
                            and newer.revision>source.revision))
                    and not exists(select 1 from jarvis_task_events result_event
                        join jarvis_task_attempts attempt on attempt.owner_id=result_event.owner_id and attempt.id=result_event.attempt_id
                        join jarvis_task_context_manifests manifest on manifest.owner_id=attempt.owner_id and manifest.id=attempt.context_manifest_id
                        where result_event.owner_id=t.owner_id and result_event.task_id=t.id
                            and result_event.payload_id=t.result_payload_id and manifest.status<>'active')
                where t.position<=$2
                order by t.is_terminal,t.updated_at desc,t.id""", owner_id, limit,
            )
        summaries = [{
                "id": str(row["id"]), "task_ref": self.reference(row["id"]),
                "session_id": str(row["session_id"]) if row["session_id"] is not None else None,
                "status": row["status"], "revision": row["revision"],
                "title": row["title"], "mode": row["mode"], "reason": row["reason"],
                "pending_question": row["pending_question"] if row["pending_question"] is not None
                    and len(row["pending_question"]) <= 1200 else None,
                "pending_question_available": row["pending_question"] is not None,
                "last_milestone": json_object(row["milestone"]) if row["milestone"] else None,
                "updated_at": row["updated_at"].isoformat(),
            } for row in rows]
        return {
            "total": next((row["category_total"] for row in rows if not row["is_terminal"]), 0),
            "terminal_total": next((row["category_total"] for row in rows if row["is_terminal"]), 0),
            "tasks": [summary for row, summary in zip(rows, summaries) if not row["is_terminal"]],
            "recent_terminal_tasks": [summary for row, summary in zip(rows, summaries) if row["is_terminal"]],
        }

    async def voice_detail(self, owner_id: str, task_id: UUID):
        """One committed snapshot; no provider call or owner-wide write lock."""
        async with self.read_transaction() as connection:
            task = await self.load(connection, owner_id, task_id)
            problem = await self.context_problem(connection, task, lock=False)
            # Keep result visibility and its milestone in one statement rather
            # than three network round trips under the same read snapshot.
            detail = await connection.fetchrow("""with result_state as (
                select exists(
                    select 1 from jarvis_task_events e
                    join jarvis_task_attempts a on a.owner_id=e.owner_id and a.id=e.attempt_id
                    join jarvis_task_context_manifests m on m.owner_id=a.owner_id and m.id=a.context_manifest_id
                    where e.owner_id=$1 and e.task_id=$2 and e.task_revision=$3
                    and e.payload_id=$4 and m.status<>'active') as invalid_manifest
                )
                select state.invalid_manifest,p.content as result,
                    (select jsonb_build_object('event_type',e.event_type,'sequence',e.sequence,
                                              'task_revision',e.task_revision)
                     from jarvis_task_events e
                     where e.owner_id=$1 and e.task_id=$2 and e.task_revision=$3 and e.event_type in
                     ('accepted','running','runtime_started','waiting_for_input','paused','reconciling',
                      'update_requested','cancel_requested','resume_requested','succeeded','failed','cancelled')
                     order by e.sequence desc limit 1) as last_milestone
                from result_state state
                left join jarvis_context_payloads p on p.owner_id=$1 and p.id=$4
                    and p.availability='available' and $5::boolean and not state.invalid_manifest""",
                owner_id, task_id, task.revision, task.result_payload_id, problem is None)
            return task, {
                "result": json_object(detail["result"]) if detail["result"] is not None else None,
                "result_unavailable_reason": problem or (
                    "context_manifest_invalidated" if detail["invalid_manifest"] else None),
                "last_milestone": json_object(detail["last_milestone"])
                    if detail["last_milestone"] is not None else None,
            }

    async def live_context_ids(self, owner_id: str, session_id: UUID,
                               area_id: UUID | None, *, limit: int = 8):
        """Candidates only; the Live State binder rereads under canonical locks.

        Apply session/area scope before the bound, so another session's active
        tasks cannot starve this session's status/result projection.
        """
        if type(limit) is not int or not 1 <= limit <= 16:
            raise ValueError("invalid live task context limit")
        async with self.pool.acquire() as connection:
            rows = await connection.fetch("""select id from jarvis_background_tasks
                where owner_id=$1 and session_id=$2
                    and knowledge_area_id is not distinct from $3::uuid
                order by status in ('succeeded','failed','cancelled'),updated_at desc,id
                limit $4""", owner_id, session_id, area_id, limit)
        return tuple(row["id"] for row in rows)

    async def live_context_in_transaction(self, connection, owner_id: str, task_id: UUID):
        """Source-backed reader, not a task owner or status mutation.

        The caller holds the existing memory-owner/task-owner locks. Intent
        revision, execution fence and the actual event sequence remain distinct.
        Unlike a display-only summary, worker text requires a positively matched
        result event, attempt and active manifest for the CURRENT task intent.
        """
        if not connection.is_in_transaction():
            raise ValueError("Live task context requires an explicit transaction")
        # This reader never mutates the task. Compatible row locks also avoid
        # opposing-order task dependency checks deadlocking under shared owners.
        task = await self.load(connection, owner_id, task_id, lock="share")
        problem = await self.context_problem(connection, task)
        if problem is not None:
            raise TaskError("live_task_context_unavailable", problem)
        detail = await connection.fetchrow("""select t.last_event_sequence,
                receipt.event_id,receipt.result_recorded_at,receipt.result_attempt_id,
                receipt.manifest_id,receipt.manifest_status,receipt.manifest_payload_id,
                p.content as result
            from jarvis_background_tasks t
            left join lateral (
                select e.id as event_id,e.created_at as result_recorded_at,
                    a.id as result_attempt_id,m.id as manifest_id,
                    m.status as manifest_status,m.payload_id as manifest_payload_id
                from jarvis_task_events e
                join jarvis_task_attempts a on a.owner_id=e.owner_id and a.id=e.attempt_id
                    and a.task_id=e.task_id and a.task_revision=e.task_revision
                join jarvis_task_context_manifests m on m.owner_id=a.owner_id
                    and m.id=a.context_manifest_id and m.task_id=a.task_id
                    and m.task_revision=a.task_revision
                where e.owner_id=t.owner_id and e.task_id=t.id and e.task_revision=t.revision
                    and e.payload_id=t.result_payload_id and e.event_type=t.status
                    and t.status in ('succeeded','failed','waiting_for_input','paused')
                order by e.sequence desc limit 1
            ) receipt on true
            left join jarvis_context_payloads p on p.owner_id=t.owner_id and p.id=t.result_payload_id
                and p.availability='available' and receipt.manifest_status='active'
            where t.owner_id=$1 and t.id=$2""", owner_id, task_id)
        if detail is None or detail["last_event_sequence"] < 1:
            raise TaskError("live_task_context_unavailable", "The task has no committed event sequence.")
        return task, dict(detail) | {
            "result": json_object(detail["result"]) if detail["result"] is not None else None,
        }

    async def events(self, owner_id: str, task_id: UUID, *, after: int = 0, limit: int = 100):
        if after < 0 or not 1 <= limit <= 100:
            raise ValueError("invalid event pagination")
        async with self.pool.acquire() as connection:
            await self.load(connection, owner_id, task_id)
            rows = await connection.fetch("""select e.*,p.content,p.availability from jarvis_task_events e
                left join jarvis_context_payloads p on p.owner_id=e.owner_id and p.id=e.payload_id
                where e.owner_id=$1 and e.task_id=$2 and e.sequence>$3 order by e.sequence limit $4""",
                owner_id, task_id, after, limit)
        return [dict(row) | {"content": json_object(row["content"]) if row["content"] is not None else None} for row in rows]

    async def payload(self, connection, owner_id: str, content: dict, *, source_payload_ids: tuple[UUID, ...] = ()) -> UUID:
        if source_payload_ids and not connection.is_in_transaction():
            raise ValueError("Source-bound task payloads require an explicit transaction")
        payload_id = await connection.fetchval("""insert into jarvis_context_payloads(owner_id,availability,content)
            values($1,'available',$2::jsonb) returning id""", owner_id, json_text(content))
        for source_id in sorted(set(source_payload_ids)):
            # Reuse the owned-source/late-write guard and recursive removal
            # graph. The payload and all its lineage commit together.
            await connection.execute("""insert into jarvis_context_payload_dependencies
                (owner_id,source_payload_id,dependent_payload_id) values($1,$2,$3)""",
                owner_id, source_id, payload_id)
        return payload_id

    async def emit(self, connection, task: TaskView, kind: str, *, payload: dict | None = None,
                   payload_id: UUID | None = None, attempt_id: UUID | None = None, revision: int | None = None,
                   source_payload_ids: tuple[UUID, ...] = ()):
        if payload is not None:
            payload_id = await self.payload(connection, task.owner_id, payload, source_payload_ids=source_payload_ids)
        sequence = await connection.fetchval("""update jarvis_background_tasks set last_event_sequence=last_event_sequence+1,
            updated_at=clock_timestamp() where owner_id=$1 and id=$2 returning last_event_sequence""", task.owner_id, task.id)
        await connection.execute("""insert into jarvis_task_events(owner_id,task_id,sequence,task_revision,attempt_id,event_type,payload_id)
            values($1,$2,$3,$4,$5,$6,$7)""", task.owner_id, task.id, sequence, revision or task.revision, attempt_id, kind, payload_id)

    async def transition(self, connection, task: TaskView, status: TaskStatus, *, reason: str | None = None):
        require_transition(task.status, status)
        await connection.execute("""update jarvis_background_tasks set status=$3,reason=$4,updated_at=clock_timestamp()
            where owner_id=$1 and id=$2""", task.owner_id, task.id, status.value, reason)

    async def _source_available(self, connection, owner_id, event_id, revision, *, lock=True) -> bool:
        if lock:
            await connection.fetchval("select 1 from jarvis_context_streams where owner_id=$1 for share", owner_id)
        row = await connection.fetchrow("""select e.id,p.availability,e.finality,e.trust,e.kind
            from jarvis_context_events e join jarvis_context_payloads p on p.owner_id=e.owner_id and p.id=e.payload_id
            where e.owner_id=$1 and e.id=$2 and e.revision=$3
            and not exists(select 1 from jarvis_context_events newer where newer.owner_id=e.owner_id
                and newer.source_namespace=e.source_namespace and newer.source_event_id=e.source_event_id and newer.revision>e.revision)
            """ + (" for share of e,p" if lock else ""), owner_id, event_id, revision)
        return bool(row and row["availability"] == "available" and row["trust"] == "user_input"
                    and row["finality"] in ("final", "revised") and row["kind"] in ("user_request", "user_transcript"))

    async def validate_grant(self, connection, owner_id: str, grant_id: UUID, *, lock=True):
        grant = await connection.fetchrow("""select g.*,p.content as scope,p.availability,
            g.expires_at>clock_timestamp() as unexpired from jarvis_operation_grants g
            join jarvis_context_payloads p on p.owner_id=g.owner_id and p.id=g.scope_payload_id
            where g.owner_id=$1 and g.id=$2""" + (" for share of g,p" if lock else ""), owner_id, grant_id)
        if not grant or grant["status"] != "active" or not grant["unexpired"] or grant["availability"] != "available":
            raise TaskError("task_grant_unavailable", "The required grant is missing, revoked, expired, or no longer current.", 403)
        source_available = await self._source_available(
            connection, owner_id, grant["request_event_id"], grant["request_revision"], lock=lock)
        if not source_available and grant["capability"] in {"sandbox.execute", "files.publish_generated", "files.publish_snapshot"}:
            from app.tasks.media_editing import media_command_grant_source_available
            source_available = await media_command_grant_source_available(self, connection, owner_id, grant, lock=lock)
        if not source_available:
            raise TaskError("task_grant_unavailable", "The required grant is missing, revoked, expired, or no longer current.", 403)
        return grant

    async def context_problem(self, connection, task: TaskView, *, lock=True) -> str | None:
        if not await self._source_available(connection, task.owner_id, task.request_event_id, task.request_revision, lock=lock):
            from app.tasks.readonly import readonly_source
            from app.tasks.live_commands import live_command_source
            try:
                command = await task_command_source(self, connection, task, lock=lock)
                if (command is None and await readonly_source(self, connection, task, lock=lock) is None
                        and await live_command_source(self, connection, task, lock=lock) is None):
                    return "request_evidence_unavailable"
            except TaskError as error:
                return error.code
        if task.title is None or task.requested_outcome is None:
            return "task_context_removed"
        if task.knowledge_area_id is not None:
            active = await connection.fetchval("""select status='active' from jarvis_knowledge_areas
                where owner_id=$1 and id=$2""" + (" for share" if lock else ""), task.owner_id, task.knowledge_area_id)
            if not active:
                return "knowledge_area_unavailable"
        grants = await connection.fetch("select grant_id from jarvis_task_grants where owner_id=$1 and task_id=$2", task.owner_id, task.id)
        for row in grants:
            try:
                await self.validate_grant(connection, task.owner_id, row["grant_id"], lock=lock)
            except TaskError:
                return "task_grant_unavailable"
        if task.current_attempt_id is not None:
            manifest = await connection.fetchrow("""select m.status from jarvis_task_attempts a
                join jarvis_task_context_manifests m on m.owner_id=a.owner_id and m.id=a.context_manifest_id
                where a.owner_id=$1 and a.id=$2""", task.owner_id, task.current_attempt_id)
            if manifest is not None and manifest["status"] != "active":
                return "task_context_manifest_invalidated"
        return None

    async def _request_event(self, connection, owner_id, request_id, body, session_id=None):
        payload_id = await self.payload(connection, owner_id, body)
        return await connection.fetchrow("""insert into jarvis_context_events(owner_id,session_id,source_namespace,source_event_id,
            kind,finality,trust,payload_id) values($1,$2,'task_api',$3,'user_request','final','user_input',$4) returning id,revision,sequence,payload_id""",
            owner_id, session_id, str(request_id), payload_id)

    async def _replay(self, connection, owner_id, request_id, operation, parameters, task_id=None):
        row = await connection.fetchrow("select * from jarvis_task_requests where owner_id=$1 and request_id=$2", owner_id, request_id)
        if row is None:
            return None
        if row["operation"] != operation or row["parameters_digest"] != digest(parameters) or (task_id and row["task_id"] != task_id):
            raise TaskError("task_request_conflict", "This request identity was already used for a different operation.")
        return await self.load(connection, owner_id, row["task_id"])

    async def _save_request(self, connection, task, request_id, operation, parameters):
        await connection.execute("""insert into jarvis_task_requests(owner_id,request_id,task_id,operation,parameters_digest,accepted_revision)
            values($1,$2,$3,$4,$5,$6)""", task.owner_id, request_id, task.id, operation, digest(parameters), task.revision)

    async def _queue_capacity(self, connection, owner_id):
        queued = await connection.fetchval("""select count(*) from jarvis_background_tasks where owner_id=$1
            and status in ('queued','waiting_for_input','paused','cancel_requested','outcome_unknown')""", owner_id)
        if queued >= self.limits.owner_queue_limit:
            raise TaskError("task_queue_full", "The pending-task limit has been reached; no new task was accepted.", 429)

    async def _voice_source(self, connection, owner_id, source: TaskVoiceSource):
        if not isinstance(source, TaskVoiceSource):
            raise TaskError("task_request_evidence_missing", "A trusted current user request is required.", 403)
        row = await connection.fetchrow("""select id,revision,sequence,session_id,payload_id from jarvis_context_events
            where owner_id=$1 and id=$2 and revision=$3 and session_id=$4""",
            owner_id, source.event_id, source.revision, source.session_id)
        if row is None or not await self._source_available(connection, owner_id, source.event_id, source.revision):
            raise TaskError("task_request_evidence_missing",
                "This exact user transcript is unavailable or changed; no new operation was accepted.", 403)
        return row

    async def _bind_voice_session(self, connection, task, source, operation, *, command_event=None):
        if (source is None and command_event is None) or operation == "cancel":
            return
        session_id = source.session_id if source is not None else command_event["session_id"]
        event_id = source.event_id if source is not None else command_event["id"]
        revision = source.revision if source is not None else command_event["revision"]
        await connection.execute("""insert into jarvis_task_session_context(owner_id,task_id,session_id,
            request_event_id,request_revision) values($1,$2,$3,$4,$5)
            on conflict(owner_id,task_id,session_id) do update set
            request_event_id=excluded.request_event_id,request_revision=excluded.request_revision""",
            task.owner_id, task.id, session_id, event_id, revision)

    @staticmethod
    def _instruction_metadata(event, session_id):
        # A resume changes control authority, not the instructions that own an
        # effect. Keep their exact canonical reference with the immutable text.
        if session_id is None:
            return {}
        return {"instruction_request": TaskVoiceSource(
            event["id"], event["revision"], session_id).parameters()}

    @staticmethod
    def _create_parameters(request, *, source_event_id=None, source=None, mode=None, command_admission=None,
                           managed_route=None):
        if command_admission is not None:
            return command_parameters(command_admission)
        parameters = request.model_dump(mode="json") | {
            "source_event_id": str(source_event_id) if source_event_id else None}
        if not request.publish_artifacts:
            parameters.pop("publish_artifacts")
        if not request.edit_memory:
            parameters.pop("edit_memory")
        if request.execution_environment == "none":
            # Preserve fingerprints of already accepted baseline requests.
            parameters.pop("execution_environment")
        if source is not None:
            parameters.update(voice_source=source.parameters(), mode=mode)
        if managed_route is not None:
            parameters["managed_plugin_route"] = managed_route
        return parameters

    @staticmethod
    def _mutation_parameters(request, source, command_admission=None):
        if command_admission is not None:
            return command_parameters(command_admission)
        parameters = request.model_dump(mode="json")
        if source is not None:
            parameters["voice_source"] = source.parameters()
        return parameters

    async def _command_replay(self, connection, owner_id, request_id, operation, parameters):
        task = await self._replay(connection, owner_id, request_id, operation, parameters)
        if task is not None:
            await task_command_source(self, connection, task, operation_id=request_id, current=False)
        return task

    async def replay_create(self, owner_id: str, request: TaskCreate, *, source=None, mode=None, command_admission=None,
                            managed_route=None):
        if command_admission is not None:
            require_command(command_admission, owner_id, request.request_id, "create")
            require_command_create(command_admission, request, source=source, mode=mode)
            async with self.transaction(owner_id) as connection:
                return await self._command_replay(connection, owner_id, request.request_id, "create",
                                                  command_parameters(command_admission))
        parameters = self._create_parameters(request, source=source, mode=mode, managed_route=managed_route)
        async with self.transaction(owner_id) as connection:
            if source is not None:
                await self._voice_source(connection, owner_id, source)
            return await self._replay(connection, owner_id, request.request_id, "create", parameters)

    async def create(self, owner_id: str, request: TaskCreate, *, model: str,
                     source_event_id: UUID | None = None, source: TaskVoiceSource | None = None,
                     mode: str | None = None, admission=None, grant_factory=None,
                     default_knowledge_area_id: UUID | None = None, area_resolver=None,
                     command_admission=None, managed_route=None, managed_admission=None, media_input=None) -> TaskView:
        media_command = getattr(command_admission, "tool_name", None) == "start_media_task"
        if (media_command and managed_route != "video_editing") or (media_input is not None and not media_command):
            raise TaskError("managed_route_required", "Media command authority requires its exact managed route.", 403)
        if managed_route is not None:
            from app.plugins.managed import require_request
            require_request(request, managed_route, command_admission)
        elif managed_admission is not None:
            raise TaskError("managed_route_required", "Managed admission requires its explicit route.", 403)
        if command_admission is not None:
            require_command(command_admission, owner_id, request.request_id, "create")
            require_command_create(command_admission, request, source=source, source_event_id=source_event_id,
                                   mode=mode, grant_factory=grant_factory)
        if source is not None and (source_event_id is not None or request.session_id != source.session_id
                                   or mode not in {"consult", "background"}):
            raise TaskError("task_voice_scope_invalid", "Invalid voice task admission scope.", 422)
        parameters = self._create_parameters(request, source_event_id=source_event_id, source=source,
                                             mode=mode, command_admission=command_admission, managed_route=managed_route)
        async with self.transaction(owner_id) as connection:
            voice_event = await self._voice_source(connection, owner_id, source) if source is not None else None
            replay = (await self._command_replay(connection, owner_id, request.request_id, "create", parameters)
                      if command_admission is not None else
                      await self._replay(connection, owner_id, request.request_id, "create", parameters))
            if replay:
                return replay
            if media_command:
                from app.tasks.media_editing import require_media_input, validate_media_input
                media_input = require_media_input(command_admission, media_input)
                await validate_media_input(connection, owner_id, request.session_id, media_input)
                if request.knowledge_area_id != media_input.area_id:
                    raise TaskError("media_command_area_changed", "The task must keep the recorded input's exact area.", 403)
            if request.execution_environment != "none" and grant_factory is None:
                raise TaskError("task_execution_scope_unavailable", "No trusted computation authority was supplied.", 403)
            if request.edit_memory and grant_factory is None:
                raise TaskError("task_memory_edit_scope_unavailable", "No trusted memory-editing authority was supplied.", 403)
            if admission is not None:
                # Only trusted application readiness/configuration checks belong
                # here. No provider I/O; a replay never calls the admission hook.
                await admission(connection)
            managed_payload = None
            if managed_route is not None:
                if managed_admission is None:
                    raise TaskError("managed_admission_required", "No managed authority was supplied.", 403)
                from app.plugins.managed import decode_managed_job
                managed_payload = decode_managed_job(
                    await managed_admission(connection), owner_id, route=managed_route).payload()
            await self._queue_capacity(connection, owner_id)
            if request.session_id is not None and not await connection.fetchval(
                "select true from jarvis_sessions where owner_id=$1 and id=$2 for share", owner_id, request.session_id,
            ):
                raise TaskError("task_session_not_found", "The session was not found for this owner.", 404)
            command_task_id = None
            if command_admission is not None:
                # Resolve only the first acceptance, under this task owner lock.
                # The immutable source and task row pin the same area/identity.
                area_id = (await area_resolver(connection, owner_id, request, None,
                                               command_admission=command_admission)
                           if area_resolver is not None else request.knowledge_area_id or default_knowledge_area_id)
                command_task_id = uuid4()
                event = await append_task_command(self, connection, command_admission, "create", request,
                    task_id=command_task_id, area_id=area_id, base_revision=0, mode=mode,
                    **({"media_input": media_input} if media_command else {}))
            elif voice_event is not None:
                event = voice_event
            elif source_event_id is None:
                event = await self._request_event(connection, owner_id, request.request_id, parameters, request.session_id)
            else:
                event = await connection.fetchrow("select id,revision,sequence,payload_id from jarvis_context_events where owner_id=$1 and id=$2", owner_id, source_event_id)
                if not event or not await self._source_available(connection, owner_id, event["id"], event["revision"]):
                    raise TaskError("task_request_evidence_missing", "A current user request is required.", 403)
            # Resolve only new work, on this connection, after its exact request
            # is known. Replays keep their recorded area without consulting a
            # newer session default or a currently unavailable memory runtime.
            if command_admission is None:
                area_id = (await area_resolver(connection, owner_id, request, event)
                           if area_resolver is not None else request.knowledge_area_id or default_knowledge_area_id)
            if area_id is not None and not await connection.fetchval(
                "select status='active' from jarvis_knowledge_areas where owner_id=$1 and id=$2 for share", owner_id, area_id,
            ):
                raise TaskError("task_area_unavailable", "The knowledge area is not available for this owner.", 404)
            description = await self.payload(connection, owner_id, {"title": request.title,
                "requested_outcome": request.requested_outcome,
                **({"mode": mode} if source is not None or command_admission is not None else {}),
                **self._instruction_metadata(event, request.session_id),
                **({"managed_plugin": managed_payload} if managed_payload is not None else {}),
                **({"media_input": media_input.model_dump(mode="json")} if media_command else {})},
                source_payload_ids=(event["payload_id"],))
            if command_admission is not None:
                identity = await connection.fetchval("""insert into jarvis_background_tasks(id,owner_id,session_id,knowledge_area_id,backend_kind,
                    model,description_payload_id,request_event_id,request_revision,context_watermark,budget)
                    values($11,$1,$2,$3,$4,$5,$6,$7,$8,$9,$10::jsonb) returning id""", owner_id, request.session_id, area_id,
                    request.backend_kind, model, description, event["id"], event["revision"], event["sequence"],
                    self.limits.model_dump_json(), command_task_id)
            else:
                identity = await connection.fetchval("""insert into jarvis_background_tasks(owner_id,session_id,knowledge_area_id,backend_kind,
                model,description_payload_id,request_event_id,request_revision,context_watermark,budget)
                values($1,$2,$3,$4,$5,$6,$7,$8,$9,$10::jsonb) returning id""", owner_id, request.session_id, area_id,
                request.backend_kind, model, description, event["id"], event["revision"], event["sequence"], self.limits.model_dump_json())
            scoped_command = media_command
            if scoped_command:
                # The narrow command-derived note grant checks the canonical
                # request receipt. All rows remain invisible until this whole
                # transaction commits; failed grants roll the receipt back too.
                task = await self.load(connection, owner_id, identity, lock=True)
                await self._save_request(connection, task, request.request_id, "create", parameters)
            generated = await grant_factory(connection, identity, event) if grant_factory is not None else ()
            grant_ids = tuple(dict.fromkeys((*request.grant_ids, *generated)))
            if scoped_command:
                # The family validator needs the complete attached family, not
                # a transient half-attached parent/child set.
                for grant_id in grant_ids:
                    await connection.execute("insert into jarvis_task_grants(owner_id,task_id,grant_id) values($1,$2,$3)", owner_id, identity, grant_id)
                if media_command:
                    from app.tasks.media_editing import validate_media_command_target
                    await validate_media_command_target(connection, task)
                for grant_id in grant_ids:
                    await self.validate_grant(connection, owner_id, grant_id)
            else:
                for grant_id in grant_ids:
                    await self.validate_grant(connection, owner_id, grant_id)
                    await connection.execute("insert into jarvis_task_grants(owner_id,task_id,grant_id) values($1,$2,$3)", owner_id, identity, grant_id)
            task = await self.load(connection, owner_id, identity, lock=True)
            if command_admission is not None and not scoped_command:
                # Validation needs the receipt, but neither is visible until commit.
                await self._save_request(connection, task, request.request_id, "create", parameters)
            problem = await self.context_problem(connection, task)
            if problem:
                raise TaskError(problem, "Task context or authority is unavailable.", 403)
            if command_admission is not None:
                await self._bind_voice_session(connection, task, None, "create", command_event=event)
            else:
                await self._bind_voice_session(connection, task, source, "create")
            await self.emit(connection, task, "accepted", payload_id=description)
            if command_admission is None:
                await self._save_request(connection, task, request.request_id, "create", parameters)
            return task

    async def replay_mutation(self, owner_id, task_id, operation, request, *, source=None, command_admission=None):
        if command_admission is not None:
            require_command(command_admission, owner_id, request.request_id, operation)
            if source is not None:
                raise TaskError("task_command_scope_invalid", "Command and transcript admission cannot be combined.", 403)
            async with self.transaction(owner_id) as connection:
                return await self._command_replay(connection, owner_id, request.request_id, operation,
                                                  command_parameters(command_admission))
        async with self.transaction(owner_id) as connection:
            if source is not None:
                await self._voice_source(connection, owner_id, source)
            return await self._replay(connection,owner_id,request.request_id,operation,
                                     self._mutation_parameters(request, source),task_id)

    async def mutate(self, owner_id: str, task_id: UUID, operation: str, request: TaskMutation, *, source=None,
                     admission=None, command_admission=None) -> TaskView:
        if operation not in {"update", "cancel", "resume"}:
            raise ValueError("unsupported task mutation")
        if operation == "update" and not request.message.strip():
            raise TaskError("task_update_empty", "An update must include the new instructions.", 422)
        if command_admission is not None:
            require_command(command_admission, owner_id, request.request_id, operation)
            if source is not None:
                raise TaskError("task_command_scope_invalid", "Command and transcript admission cannot be combined.", 403)
        parameters = self._mutation_parameters(request, source, command_admission)
        async with self.transaction(owner_id) as connection:
            voice_event = await self._voice_source(connection, owner_id, source) if source is not None else None
            replay = (await self._command_replay(connection, owner_id, request.request_id, operation, parameters)
                      if command_admission is not None else
                      await self._replay(connection, owner_id, request.request_id, operation, parameters, task_id))
            if replay:
                return replay
            task = await self.load(connection, owner_id, task_id, lock=True)
            if command_admission is not None:
                await require_command_target(connection, task)
            if task.revision != request.expected_revision:
                raise TaskError("task_revision_conflict", "The task changed; read its current revision before updating it.")
            if admission is not None:
                await admission(connection, task)
            unresolved = await self.has_unknown_actions(connection, task)
            if operation == "resume" and (task.current_attempt_id is not None or unresolved):
                raise TaskError("task_reconciliation_required", "Existing execution or external actions must be reconciled before resuming.")
            if operation == "resume" and task.status not in {TaskStatus.PAUSED, TaskStatus.WAITING, TaskStatus.FAILED, TaskStatus.CANCELLED, TaskStatus.UNKNOWN}:
                raise TaskError("task_state_conflict", "This task is not in a resumable state.")
            if operation == "resume" and (task.title is None or task.requested_outcome is None):
                raise TaskError("task_context_removed", "The task instructions were removed. Update the task with fresh instructions before resuming.")
            if operation == "update" and task.status == TaskStatus.CANCEL_REQUESTED:
                raise TaskError("task_reconciliation_required", "Cancellation must settle before the task can be updated.")
            if operation != "cancel" and task.status in {TaskStatus.SUCCEEDED, TaskStatus.FAILED, TaskStatus.CANCELLED}:
                await self._queue_capacity(connection, owner_id)
            if command_admission is not None:
                event = await append_task_command(self, connection, command_admission, operation, request,
                    task_id=task.id, area_id=task.knowledge_area_id, base_revision=task.revision)
            else:
                event = voice_event if voice_event is not None else await self._request_event(
                    connection, owner_id, request.request_id,
                    {"operation": operation, "task_id": str(task_id), **parameters}, task.session_id)
            description = task.description_payload_id
            if operation == "update":
                from app.plugins.managed import load_managed_task
                managed = await load_managed_task(connection, task)
                if managed is not None and await connection.fetchval("""select exists(select 1 from jarvis_task_actions
                    where owner_id=$1 and task_id=$2 and grant_id=$3 and submission_token is not null)""",
                    owner_id, task.id, managed[1]):
                    raise TaskError("managed_submitted_assignment_immutable",
                        "A submitted managed assignment cannot become new work; create a separate task.", 409)
                # The inherited title/mode are copies too. If the old content
                # was removed, fresh instructions can repair the task without
                # linking to an unavailable description or restoring its text.
                sources = (event["payload_id"],) + ((description,) if task.title or task.mode else ())
                description = await self.payload(connection, owner_id, {"title": task.title or "Updated task",
                    "requested_outcome": request.message, **({"mode": task.mode} if task.mode else {}),
                    **self._instruction_metadata(event, command_admission.session_id if command_admission is not None
                        else source.session_id if source is not None else task.session_id),
                    **({"managed_plugin": managed[0].payload()} if managed is not None else {})},
                    source_payload_ids=sources)
            if operation == "cancel":
                status = TaskStatus.CANCEL_REQUESTED if task.current_attempt_id is not None else (TaskStatus.UNKNOWN if unresolved else TaskStatus.CANCELLED)
                pending = "cancelled" if task.current_attempt_id is not None or unresolved else None
            else:
                status = TaskStatus.UNKNOWN if unresolved else (TaskStatus.PAUSED if task.current_attempt_id is not None else TaskStatus.QUEUED)
                pending = "queued" if task.current_attempt_id is not None or unresolved else None
            require_transition(task.status, status)
            await connection.execute("""update jarvis_background_tasks set revision=revision+1,fence=fence+1,status=$3,pending_status=$4,
                description_payload_id=$5,request_event_id=$6,request_revision=$7,context_watermark=$8,result_payload_id=null,
                reason=$9,updated_at=clock_timestamp(),available_at=clock_timestamp() where owner_id=$1 and id=$2""",
                owner_id, task_id, status.value, pending, description, event["id"], event["revision"], event["sequence"], operation + "_requested")
            if task.current_attempt_id is not None:
                # The old worker's fence is already invalid. A recovery claimant
                # now owns proving the old provider/action outcome before requeue.
                await connection.execute("update jarvis_task_attempts set lease_until=clock_timestamp() where owner_id=$1 and id=$2", owner_id, task.current_attempt_id)
            if operation in {"cancel", "update"}:
                await connection.execute("""update jarvis_task_actions set status='revoked',updated_at=clock_timestamp()
                    where owner_id=$1 and task_id=$2 and status in ('prepared','authorized')""", owner_id, task_id)
            current = await self.load(connection, owner_id, task_id)
            if command_admission is not None:
                await self._bind_voice_session(connection, current, None, operation, command_event=event)
            else:
                await self._bind_voice_session(connection, current, source, operation)
            await self.emit(connection, current, operation + "_requested", payload={"message": request.message},
                            source_payload_ids=(event["payload_id"],))
            await self._save_request(connection, current, request.request_id, operation, parameters)
            return current

    async def has_unknown_actions(self, connection, task: TaskView) -> bool:
        return bool(await connection.fetchval("""select exists(select 1 from jarvis_task_actions where owner_id=$1 and task_id=$2
            and status in ('submitting','outcome_unknown'))""", task.owner_id, task.id))

    async def _active_count(self, connection, owner_id=None):
        return await connection.fetchval("""select count(*) from jarvis_task_attempts
            where status=any($1::text[]) and ($2::text is null or owner_id=$2)""", list(ACTIVE_ATTEMPTS), owner_id)

    async def scheduler_limits(self, connection):
        await connection.execute("""insert into jarvis_task_scheduler_limits(global_concurrency,owner_concurrency)
            values($1,$2) on conflict(singleton) do nothing""", self.limits.global_concurrency, self.limits.owner_concurrency)
        return await connection.fetchrow("select * from jarvis_task_scheduler_limits where singleton")

    async def configure_scheduler_limits(self):
        """Explicit administrative change. Lower limits drain, not kill, work."""
        async with self.transaction() as connection:
            await connection.execute("select pg_advisory_xact_lock(hashtextextended(current_schema() || ':jarvis-task-claims',0))")
            await connection.execute("""insert into jarvis_task_scheduler_limits(global_concurrency,owner_concurrency)
                values($1,$2) on conflict(singleton) do update set global_concurrency=excluded.global_concurrency,
                owner_concurrency=excluded.owner_concurrency,updated_at=clock_timestamp()""", self.limits.global_concurrency, self.limits.owner_concurrency)
            return {"global_concurrency": self.limits.global_concurrency, "owner_concurrency": self.limits.owner_concurrency,
                    "active_attempts": await self._active_count(connection)}

    async def claim_next(self, worker_id: str, backend_kinds: tuple[str, ...], *, owner_id=None, provider_models=None,
                         worker_capabilities=None) -> TaskClaim | None:
        if not backend_kinds:
            return None
        async with self.transaction() as connection:
            await connection.execute("select pg_advisory_xact_lock(hashtextextended(current_schema() || ':jarvis-task-claims',0))")
            admission = await self.scheduler_limits(connection)
            if await self._active_count(connection) >= admission["global_concurrency"]:
                return None
            candidate = await connection.fetchrow("""select t.owner_id,t.id from jarvis_background_tasks t
                left join jarvis_task_owner_scheduling s on s.owner_id=t.owner_id
                where t.status='queued' and t.current_attempt_id is null and t.available_at<=clock_timestamp()
                    and t.backend_kind=any($1::text[]) and ($4::text is null or t.owner_id=$4)
                    and ($5::jsonb is null or t.model=($5::jsonb->>t.backend_kind))
                    and (not exists(select 1 from jarvis_task_grants mtg
                        join jarvis_operation_grants mg on mg.owner_id=mtg.owner_id and mg.id=mtg.grant_id
                        where mtg.owner_id=t.owner_id and mtg.task_id=t.id
                            and mg.operation_key like 'managed-files-data:%')
                        or ($6::text[] is not null and 'plugins.files_data'=any($6::text[])))
                    and (not exists(select 1 from jarvis_task_grants mtg
                        join jarvis_operation_grants mg on mg.owner_id=mtg.owner_id and mg.id=mtg.grant_id
                        where mtg.owner_id=t.owner_id and mtg.task_id=t.id
                            and mg.operation_key like 'managed-video-editor:%')
                        or ($6::text[] is not null and 'plugins.video_editor'=any($6::text[])))
                    and (not exists(select 1 from jarvis_context_events ro
                        where ro.owner_id=t.owner_id and ro.id=t.request_event_id
                            and ro.source_namespace like 'live-readonly:%')
                        or ($6::text[] is not null and 'live.readonly.v1'=any($6::text[])))
                    and ($6::text[] is null or not exists(select 1 from jarvis_task_grants tg
                        join jarvis_operation_grants g on g.owner_id=tg.owner_id and g.id=tg.grant_id
                        join jarvis_context_payloads p on p.owner_id=g.owner_id and p.id=g.scope_payload_id
                        where tg.owner_id=t.owner_id and tg.task_id=t.id and g.capability in ('sandbox.execute','files.publish_generated','memory.correct_sources','notes.edit_account')
                            and (not g.capability=any($6::text[]) or
                                (g.capability='sandbox.execute'
                                 and (p.content->>'internet_access'='true'
                                      or coalesce(p.content->'allowed_domains','[]'::jsonb)<>'[]'::jsonb)
                                 and not 'sandbox.network'=any($6::text[])) or
                                (g.capability='sandbox.execute' and p.content->>'template'='desktop'
                                 and not 'sandbox.desktop'=any($6::text[])) or
                                (g.capability='files.publish_generated' and p.content->'formats' ? 'mp4'
                                 and not 'files.publish_generated_mp4'=any($6::text[])))))
                    and (select count(*) from jarvis_task_attempts a
                        where a.owner_id=t.owner_id and a.status=any($2::text[]))<$3
                order by coalesce(s.last_claimed_at,'-infinity'::timestamptz),t.created_at,t.id limit 1""",
                list(backend_kinds), list(ACTIVE_ATTEMPTS), admission["owner_concurrency"],owner_id,
                json_text(provider_models) if provider_models is not None else None,
                list(worker_capabilities) if worker_capabilities is not None else None)
            if not candidate:
                return None
            await self.lock_owner(connection, candidate["owner_id"])
            task = await self.load(connection, candidate["owner_id"], candidate["id"], lock=True)
            if task.status != TaskStatus.QUEUED or task.current_attempt_id is not None:
                return None
            problem = await self.context_problem(connection, task)
            if task.attempt_count >= task.budget.max_attempts:
                problem = "task_attempt_budget_reached"
            if task.active_seconds >= task.budget.active_seconds:
                problem = "task_active_budget_reached"
            if problem:
                await self.transition(connection, task, TaskStatus.PAUSED, reason=problem)
                await self.emit(connection, task, "paused", payload={"reason": problem})
                return None
            attempt = await connection.fetchrow("""insert into jarvis_task_attempts(owner_id,task_id,task_revision,attempt_number,
                fence,worker_id,status,lease_until) values($1,$2,$3,$4,$5,$6,'running',clock_timestamp()+$7*interval '1 second') returning *""",
                task.owner_id, task.id, task.revision, task.attempt_count + 1, task.fence + 1, worker_id, float(task.budget.lease_seconds))
            await connection.execute("""update jarvis_background_tasks set status='running',reason=null,current_attempt_id=$3,
                attempt_count=attempt_count+1,fence=fence+1,updated_at=clock_timestamp() where owner_id=$1 and id=$2""", task.owner_id, task.id, attempt["id"])
            await connection.execute("""insert into jarvis_task_owner_scheduling(owner_id,last_claimed_at) values($1,clock_timestamp())
                on conflict(owner_id) do update set last_claimed_at=excluded.last_claimed_at""", task.owner_id)
            current = await self.load(connection, task.owner_id, task.id)
            await self.emit(connection, current, "running", attempt_id=attempt["id"])
            return self._claim(current, attempt)

    @staticmethod
    def _claim(task, attempt, *, recovering=False):
        return TaskClaim(task, attempt["id"], attempt["task_revision"], attempt["fence"], attempt["lease_until"],
                         attempt["provider_state"], attempt["provider_handle"], attempt["checkpoint_payload_id"], recovering)

    async def lock_claim(self, connection, claim: TaskClaim, *, recovery: bool = False):
        task = await self.load(connection, claim.task.owner_id, claim.task.id, lock=True)
        attempt = await connection.fetchrow("select *,lease_until>clock_timestamp() as lease_valid from jarvis_task_attempts where owner_id=$1 and id=$2 for update", task.owner_id, claim.attempt_id)
        if (not attempt or not attempt["lease_valid"] or task.current_attempt_id != claim.attempt_id or task.fence != claim.fence
                or attempt["fence"] != claim.fence or attempt["status"] not in ACTIVE_ATTEMPTS):
            raise ClaimLost()
        if not recovery and (task.status != TaskStatus.RUNNING or task.revision != claim.task_revision):
            raise ClaimLost()
        return task, attempt

    async def _charge_time(self, connection, task, attempt):
        elapsed = await connection.fetchval("""update jarvis_task_attempts set charged_until=least(clock_timestamp(),lease_until)
            where owner_id=$1 and id=$2 returning greatest(0,extract(epoch from (charged_until-$3::timestamptz)))::float8""",
            task.owner_id, attempt["id"], attempt["charged_until"])
        await connection.execute("update jarvis_background_tasks set active_seconds=active_seconds+$3 where owner_id=$1 and id=$2", task.owner_id, task.id, elapsed)
        return task.active_seconds + elapsed

    async def renew(self, claim: TaskClaim) -> bool:
        async with self.transaction(claim.task.owner_id) as connection:
            task, attempt = await self.lock_claim(connection, claim, recovery=claim.recovering)
            used = await self._charge_time(connection, task, attempt)
            problem = await self.context_problem(connection, task)
            if used >= task.budget.active_seconds:
                problem = "task_active_budget_reached"
            if problem:
                await self._abandon_locked(connection, task, attempt, reason=problem, target=TaskStatus.PAUSED)
                return False
            await connection.execute("update jarvis_task_attempts set lease_until=clock_timestamp()+$3*interval '1 second' where owner_id=$1 and id=$2",
                                     task.owner_id, attempt["id"], float(task.budget.lease_seconds))
            return True

    async def start_runtime(self, claim: TaskClaim, provider_handle: str, payload: dict, *, host_id=None):
        """Attach a maintained runtime before it can make nested model/tool calls."""
        if host_id is not None and (not isinstance(host_id,str) or not 1 <= len(host_id) <= 200):
            raise ValueError("Invalid native runtime host")
        if not isinstance(provider_handle, str) or not 1 <= len(provider_handle) <= 512:
            raise ValueError("invalid runtime handle")
        async with self.transaction(claim.task.owner_id) as connection:
            task, attempt = await self.lock_claim(connection, claim)
            if await self.context_problem(connection, task):
                raise ClaimLost()
            if attempt["provider_state"] != "not_started" or attempt["provider_handle"] is not None:
                raise TaskError("task_runtime_already_started", "This attempt already has execution ownership.")
            payload_id = await self.payload(connection, task.owner_id, payload)
            await connection.execute("""update jarvis_task_attempts set provider_state='live',
                provider_handle=$3,checkpoint_payload_id=$4,runtime_host_id=$5 where owner_id=$1 and id=$2""",
                task.owner_id, attempt["id"], provider_handle, payload_id,host_id)
            await self.emit(connection, task, "runtime_started", payload_id=payload_id, attempt_id=attempt["id"])

    async def checkpoint(self, claim: TaskClaim, payload: dict, *, provider_handle: str | None = None):
        if provider_handle is not None and not 1 <= len(provider_handle) <= 512:
            raise ValueError("invalid provider handle")
        async with self.transaction(claim.task.owner_id) as connection:
            task, attempt = await self.lock_claim(connection, claim)
            if await self.context_problem(connection, task):
                raise ClaimLost()
            if provider_handle is not None and attempt["provider_state"] not in {"submitting", "live"}:
                raise TaskError("task_dispatch_not_recorded", "Record dispatch intent before attaching a provider handle.")
            if attempt["provider_handle"] is not None and provider_handle not in {None, attempt["provider_handle"]}:
                raise TaskError("task_handle_conflict", "A provider handle cannot be replaced while it is live.")
            payload_id = await self.payload(connection, task.owner_id, payload)
            await connection.execute("""update jarvis_task_attempts set checkpoint_payload_id=$3,
                provider_handle=coalesce($4,provider_handle),provider_state=case when $4::text is null then provider_state else 'live' end
                where owner_id=$1 and id=$2""", task.owner_id, attempt["id"], payload_id, provider_handle)
            await self.emit(connection, task, "checkpoint", payload_id=payload_id, attempt_id=attempt["id"])

    async def finish(self, claim: TaskClaim, outcome: WorkerOutcome) -> bool:
        async with self.transaction(claim.task.owner_id) as connection:
            task = await self.load(connection, claim.task.owner_id, claim.task.id, lock=True)
            try:
                task, attempt = await self.lock_claim(connection, claim)
            except ClaimLost:
                # Receipt history is allowed, but can never become the current result.
                known = await connection.fetchval("select exists(select 1 from jarvis_task_attempts where owner_id=$1 and task_id=$2 and id=$3)", task.owner_id, task.id, claim.attempt_id)
                if known:
                    problem = await self.context_problem(connection, task)
                    payload = {"outcome_discarded": True, "reason": problem} if problem else outcome.payload
                    await self.emit(connection, task, "stale_attempt_outcome", payload=payload, attempt_id=claim.attempt_id, revision=claim.task_revision)
                return False
            await self._finish_locked(connection, task, attempt, outcome)
            return True

    async def _finish_locked(self, connection, task, attempt, outcome, *, source_payload_ids=()):
        from app.tasks.budgets import USAGE_SQL
        used = await self._charge_time(connection, task, attempt)
        problem = await self.context_problem(connection, task)
        context_removed = problem is not None
        if used > task.budget.active_seconds:
            problem = "task_active_budget_reached"
        await connection.execute("update jarvis_task_reservations set status='released' where owner_id=$1 and attempt_id=$2 and status='reserved'", task.owner_id, attempt["id"])
        await connection.execute("update jarvis_task_reservations set status='unknown' where owner_id=$1 and attempt_id=$2 and status='submitted'", task.owner_id, attempt["id"])
        usage = await connection.fetchrow(USAGE_SQL + " and task_id=$2", task.owner_id, task.id)
        if usage["input"]>task.budget.task_input_tokens or usage["output"]>task.budget.task_output_tokens or usage["steps"]>task.budget.tool_steps:
            problem = "task_usage_overrun"
        unknown = await self.has_unknown_actions(connection, task)
        if unknown:
            await connection.execute("""update jarvis_task_actions set status='outcome_unknown',updated_at=clock_timestamp()
                where owner_id=$1 and task_id=$2 and status='submitting'""", task.owner_id, task.id)
        unsubmitted = await connection.fetchval("""select exists(select 1 from jarvis_task_actions
            where owner_id=$1 and task_id=$2 and status in ('prepared','authorized'))""", task.owner_id, task.id)
        if unsubmitted and outcome.status == TaskStatus.SUCCEEDED:
            problem = "unsubmitted_actions"
            context_removed = True
        status = TaskStatus.UNKNOWN if unknown else (TaskStatus.PAUSED if problem else outcome.status)
        payload_id = await self.payload(connection, task.owner_id,
            {"outcome_discarded": True, "reason": "external_action_unknown" if unknown else problem}
            if context_removed or unknown else outcome.payload,
            source_payload_ids=() if context_removed or unknown else source_payload_ids)
        await connection.execute("""update jarvis_task_attempts set status=$3,provider_state='terminal',ended_at=clock_timestamp()
            where owner_id=$1 and id=$2""", task.owner_id, attempt["id"], "succeeded" if outcome.status == TaskStatus.SUCCEEDED else "failed")
        await self.transition(connection, task, status, reason="external_action_unknown" if unknown else problem or outcome.reason)
        await connection.execute("""update jarvis_background_tasks set current_attempt_id=null,pending_status=null,result_payload_id=$3
            where owner_id=$1 and id=$2""", task.owner_id, task.id, None if problem or unknown else payload_id)
        await self.emit(connection, task, status.value, payload_id=payload_id, attempt_id=attempt["id"])
        return None if problem or unknown else payload_id

    async def abandon(self, claim: TaskClaim, *, reason="worker_stopped", target: TaskStatus = TaskStatus.QUEUED):
        async with self.transaction(claim.task.owner_id) as connection:
            try:
                task, attempt = await self.lock_claim(connection, claim, recovery=claim.recovering)
            except ClaimLost:
                return
            await self._charge_time(connection, task, attempt)
            await self._abandon_locked(connection, task, attempt, reason=reason, target=target)

    async def _abandon_locked(self, connection, task, attempt, *, reason, target):
        remote_uncertain = attempt["provider_state"] in {"submitting", "live"}
        unknown = remote_uncertain or await self.has_unknown_actions(connection, task)
        status = TaskStatus.UNKNOWN if unknown else target
        await connection.execute("""update jarvis_task_attempts set status=$3,lease_until=clock_timestamp(),
            ended_at=case when $4 then null else clock_timestamp() end where owner_id=$1 and id=$2""",
            task.owner_id, attempt["id"], "outcome_unknown" if remote_uncertain else "interrupted", remote_uncertain)
        await connection.execute("""update jarvis_task_actions set status='outcome_unknown',updated_at=clock_timestamp()
            where owner_id=$1 and task_id=$2 and status='submitting'""", task.owner_id, task.id)
        await connection.execute("update jarvis_task_reservations set status='unknown' where owner_id=$1 and attempt_id=$2 and status='submitted'", task.owner_id, attempt["id"])
        await connection.execute("update jarvis_task_reservations set status='released' where owner_id=$1 and attempt_id=$2 and status='reserved'", task.owner_id, attempt["id"])
        await self.transition(connection, task, status, reason=reason)
        await connection.execute("""update jarvis_background_tasks set fence=fence+1,current_attempt_id=$3,
            pending_status=case when $4 then 'paused' when not $5 and not $6 then null else pending_status end,
            available_at=clock_timestamp() where owner_id=$1 and id=$2""", task.owner_id, task.id,
            attempt["id"] if remote_uncertain else None, remote_uncertain and target == TaskStatus.PAUSED, remote_uncertain, unknown)
        await self.emit(connection, task, status.value, payload={"reason": reason}, attempt_id=attempt["id"])

    async def claim_recovery(self, worker_id: str, *, owner_id=None, host_id=None, backend_kinds=None) -> TaskClaim | None:
        async with self.transaction() as connection:
            await connection.execute("select pg_advisory_xact_lock(hashtextextended(current_schema() || ':jarvis-task-claims',0))")
            row = await connection.fetchrow("""select a.owner_id,a.task_id,a.id from jarvis_task_attempts a
                join jarvis_background_tasks t on t.owner_id=a.owner_id and t.current_attempt_id=a.id
                where a.status=any($1::text[]) and a.lease_until<=clock_timestamp()
                    and ($2::text is null or a.owner_id=$2)
                    and ($3::text is null or a.runtime_host_id is null or a.runtime_host_id=$3)
                    and ($4::text[] is null or t.backend_kind=any($4::text[]))
                order by a.lease_until,a.id limit 1""", list(ACTIVE_ATTEMPTS),owner_id,host_id,
                list(backend_kinds) if backend_kinds is not None else None)
            if row is None:
                return None
            await self.lock_owner(connection, row["owner_id"])
            task = await self.load(connection, row["owner_id"], row["task_id"], lock=True)
            attempt = await connection.fetchrow("select *,lease_until<=clock_timestamp() as expired from jarvis_task_attempts where owner_id=$1 and id=$2 for update", row["owner_id"], row["id"])
            if task.current_attempt_id != attempt["id"] or attempt["status"] not in ACTIVE_ATTEMPTS or not attempt["expired"]:
                return None
            used = await self._charge_time(connection, task, attempt)
            problem = await self.context_problem(connection, task)
            if used >= task.budget.active_seconds:
                problem = "task_active_budget_reached"
            if problem:
                # Recovery is still execution. Revocation or exhaustion must
                # request cancellation of an existing remote handle as well.
                # An explicit user cancellation retains its stronger outcome.
                await connection.execute("""update jarvis_background_tasks set
                    pending_status=case when pending_status='cancelled' then pending_status else 'paused' end,
                    reason=$3 where owner_id=$1 and id=$2""", task.owner_id, task.id, problem)
            fence = task.fence + 1
            await connection.execute("update jarvis_background_tasks set fence=$3 where owner_id=$1 and id=$2", task.owner_id, task.id, fence)
            attempt = await connection.fetchrow("""update jarvis_task_attempts set status='recovering',worker_id=$3,fence=$4,
                charged_until=clock_timestamp(),lease_until=clock_timestamp()+$5*interval '1 second'
                where owner_id=$1 and id=$2 returning *""", task.owner_id, attempt["id"], worker_id, fence, float(task.budget.lease_seconds))
            await self.emit(connection, task, "reconciling", attempt_id=attempt["id"])
            return self._claim(await self.load(connection, task.owner_id, task.id), attempt, recovering=True)

    async def resolve_recovery(self, claim: TaskClaim, result: Reconciliation):
        async with self.transaction(claim.task.owner_id) as connection:
            task, attempt = await self.lock_claim(connection, claim, recovery=True)
            if result.state in {"running", "unknown"}:
                await connection.execute("""update jarvis_task_attempts set status=$3,lease_until=clock_timestamp()+$4*interval '1 second'
                    where owner_id=$1 and id=$2""", task.owner_id, attempt["id"], "recovering" if result.state == "running" else "outcome_unknown", max(0.1, task.budget.poll_seconds * 4))
                if result.state == "unknown" and task.status not in {TaskStatus.UNKNOWN, TaskStatus.CANCEL_REQUESTED}:
                    await self.transition(connection, task, TaskStatus.UNKNOWN, reason="provider_outcome_unknown")
                await self.emit(connection, task, "provider_" + result.state, attempt_id=attempt["id"])
                return
            if result.state == "not_started" and attempt["provider_state"] == "not_started":
                await self._abandon_locked(connection, task, attempt, reason="worker_recovered_before_dispatch",
                                           target=TaskStatus(task.pending_status or "queued"))
                return
            if result.state == "not_started":
                # A provider's explicit reconciliation proof may clear an
                # ambiguous dispatch; a missing handle by itself is not proof.
                await connection.execute("update jarvis_task_attempts set provider_state='not_started',provider_handle=null where owner_id=$1 and id=$2", task.owner_id, attempt["id"])
                attempt = dict(attempt) | {"provider_state": "not_started"}
                await connection.execute("""update jarvis_task_reservations set status='released' where owner_id=$1 and attempt_id=$2
                    and is_provider and status in ('submitted','unknown')""", task.owner_id, attempt["id"])
                await self._abandon_locked(connection, task, attempt, reason="provider_confirmed_not_started",
                                           target=TaskStatus(task.pending_status or "queued"))
                return
            if task.revision == attempt["task_revision"] and task.pending_status is None:
                await self._finish_locked(connection, task, attempt, result.outcome)
                return
            await self._charge_time(connection, task, attempt)
            await connection.execute("update jarvis_task_attempts set status='interrupted',provider_state='terminal',ended_at=clock_timestamp() where owner_id=$1 and id=$2", task.owner_id, attempt["id"])
            unknown = await self.has_unknown_actions(connection, task)
            if unknown:
                await connection.execute("""update jarvis_task_actions set status='outcome_unknown',updated_at=clock_timestamp()
                    where owner_id=$1 and task_id=$2 and status='submitting'""", task.owner_id, task.id)
            status = TaskStatus.UNKNOWN if unknown else TaskStatus(task.pending_status or "paused")
            await self.transition(connection, task, status, reason="old_revision_reconciled")
            await connection.execute("""update jarvis_background_tasks set current_attempt_id=null,
                pending_status=case when $3 then pending_status else null end where owner_id=$1 and id=$2""", task.owner_id, task.id, unknown)
            problem = await self.context_problem(connection, task)
            await self.emit(connection, task, "stale_attempt_outcome", payload={"outcome_discarded": True, "reason": problem} if problem else result.outcome.payload,
                            attempt_id=attempt["id"], revision=attempt["task_revision"])

    async def resolve_action_barrier(self, connection, task):
        if task.status != TaskStatus.UNKNOWN or task.current_attempt_id is not None or await self.has_unknown_actions(connection, task):
            return
        target = TaskStatus(task.pending_status or "paused")
        await self.transition(connection, task, target, reason="external_actions_reconciled")
        await connection.execute("update jarvis_background_tasks set pending_status=null where owner_id=$1 and id=$2", task.owner_id, task.id)
        await self.emit(connection, task, target.value, payload={"reason": "external_actions_reconciled"})
