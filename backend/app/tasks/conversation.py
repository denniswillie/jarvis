"""Ordinary messages for explicitly selected work, using existing task storage.

This mailbox does not interpret corrections, grant tools, or decide whether a
model applied a message. Execution ownership and effects remain with TaskService
and the domain services. Message delivery is separate from destructive update.
"""
from dataclasses import dataclass
from uuid import NAMESPACE_URL, UUID, uuid5

from app.tasks.context_repository import ContextRepository
from app.tasks.models import BudgetExceeded, TaskError, TaskStatus, digest, json_object, json_text


INPUT = "work_input"
PROGRESS = "work_progress"
MAX_INPUT_BYTES = 60000  # Leave room for the native steering envelope.


@dataclass(frozen=True)
class WorkMessage:
    sequence: int
    payload_id: UUID
    content: dict


def bounded_content(content):
    value = json_object(content)
    if len(json_text(value).encode("utf-8")) > MAX_INPUT_BYTES:
        raise TaskError("work_input_too_large", "The work message exceeds its transport bound.", 413)
    return value


class TaskConversation:
    def __init__(self, tasks):
        self.tasks = tasks
        self.sources = ContextRepository(tasks)

    async def append_user(self, owner_id, task_id, event_id):
        """Internal ingress from an existing canonical user message, not model text."""
        async with self.tasks.transaction(owner_id) as connection:
            task = await self.tasks.load(connection, owner_id, task_id, lock=True)
            source = await self.sources._record(connection, owner_id, event_id)
            text = (source.content or {}).get("text")
            if (not source.authorizes_request or not isinstance(text, str)
                    or not text.strip() or "\x00" in text):
                raise TaskError("work_input_source_unavailable", "The original user message is unavailable.", 409)
            if source.session_id != task.session_id:
                raise TaskError("work_input_session_mismatch", "The message does not belong to this work session.", 403)
            identity = uuid5(NAMESPACE_URL, "jarvis.work-input:" + str(task.id) + ":" + str(source.id))
            return await self._append(connection, task, identity,
                {"kind": "user_message", "text": text, "source_event_id": str(source.id),
                 "source_revision": source.revision}, (source.payload_id,))

    async def append_live(self, task_id, packet):
        """Reuse exact Live observation checks, never the interpreter admission.

        The adapter must supply the explicitly associated work ID. A complete
        observation window is context, not a fabricated finalized user turn.
        """
        from app.tasks.live_commands import _originals, _parameters, _shape, request_identity
        parameters = _parameters(packet, ())
        route, _, _, _ = _shape(parameters)
        owner = route.scope.owner_id
        async with self.tasks.transaction(owner) as connection:
            task = await self.tasks.load(connection, owner, task_id, lock=True)
            if (task.session_id != route.scope.session_id
                    or task.knowledge_area_id != route.scope.area_id):
                raise TaskError("work_input_scope_mismatch", "The selected work is outside this Live scope.", 403)
            _, roots = await _originals(self.tasks, connection, parameters, lock=True)
            return await self._append(connection, task, request_identity(route, packet.window.request.delegation_id),
                {"kind": "live_dialogue", "context": packet.as_context()}, roots)

    async def _append(self, connection, task, identity, content, roots):
        content = bounded_content(content)
        body = {"message_id": str(identity), "input": content}
        previous = await connection.fetchrow("""select e.sequence,e.payload_id,p.content,p.availability
            from jarvis_task_events e join jarvis_context_payloads p on p.owner_id=e.owner_id and p.id=e.payload_id
            where e.owner_id=$1 and e.task_id=$2 and e.event_type=$3 and p.content->>'message_id'=$4
            order by e.sequence limit 1""", task.owner_id, task.id, INPUT, str(identity))
        if previous is not None:
            if (previous["availability"] != "available" or previous["content"] is None
                    or digest(json_object(previous["content"])) != digest(body)):
                raise TaskError("work_input_conflict", "This work message is unavailable or changed.", 409)
            return WorkMessage(previous["sequence"], previous["payload_id"], content)
        allowed = {TaskStatus.QUEUED, TaskStatus.RUNNING, TaskStatus.SUCCEEDED, TaskStatus.WAITING}
        if task.status not in allowed:
            raise TaskError("work_not_accepting_input", "This work must be resumed or reconciled before new input.", 409)
        problem = await self.tasks.context_problem(connection, task)
        if problem:
            raise TaskError(problem, "The work's existing source or access is no longer available.", 409)
        idle = task.current_attempt_id is None and task.status in {TaskStatus.SUCCEEDED, TaskStatus.WAITING}
        if idle:
            if task.attempt_count >= task.budget.max_attempts or task.active_seconds >= task.budget.active_seconds:
                raise BudgetExceeded()
            if await self.tasks.has_unknown_actions(connection, task):
                raise TaskError("task_reconciliation_required", "Existing effects must be reconciled first.", 409)
            await self.tasks._queue_capacity(connection, task.owner_id)
        payload_id = await self.tasks.payload(connection, task.owner_id, body, source_payload_ids=roots)
        await self.tasks.emit(connection, task, INPUT, payload_id=payload_id)
        sequence = await connection.fetchval("select last_event_sequence from jarvis_background_tasks where owner_id=$1 and id=$2",
                                             task.owner_id, task.id)
        if idle:
            await self._queue(connection, task)
        return WorkMessage(sequence, payload_id, content)

    async def append_live_observations(self, task_id, route, observations):
        """Complete a delegated sentence while work runs, without inventing a turn.

        Only already captured original fragments are accepted. Ordinary idle
        conversation does not wake a worker; another native delegation does.
        """
        from app.tasks.readonly import FRAGMENT_FORMAT
        if not observations or len(observations) > 64:
            raise ValueError("Expected a bounded batch of original observations")
        owner = route.scope.owner_id
        async with self.tasks.transaction(owner) as connection:
            task = await self.tasks.load(connection, owner, task_id, lock=True)
            if task.session_id != route.scope.session_id or task.knowledge_area_id != route.scope.area_id:
                raise TaskError("work_input_scope_mismatch", "The work is outside this conversation.", 403)
            if task.status not in {TaskStatus.QUEUED, TaskStatus.RUNNING}:
                return False
            originals, roots = [], []
            for observed in observations:
                record = await self.sources._record(connection, owner, observed.id)
                body = record.content or {}
                if (record != observed or record.session_id != task.session_id or not record.is_current
                        or record.availability != "available" or record.kind != "observation"
                        or record.trust != "derived" or body.get("format") != FRAGMENT_FORMAT
                        or body.get("route") != route.model_dump(mode="json")
                        or body.get("speech_finality") != "provisional"
                        or body.get("speaker") not in {"user", "assistant"}):
                    raise TaskError("work_input_source_unavailable", "An original speech fragment changed.")
                originals.append({"source_event_id": str(record.id), "source_revision": record.revision,
                    "speaker": body["speaker"], "delta": body["delta"],
                    "start_ms": body["start_ms"], "end_ms": body["end_ms"]})
                roots.append(record.payload_id)
            identity = uuid5(NAMESPACE_URL, "jarvis.work-fragments:" + str(task.id) + ":" + digest({"parts": originals}))
            await self._append(connection, task, identity,
                {"kind": "live_observations", "dialogue": originals, "speech_finality": "provisional"}, tuple(roots))
            return True

    async def history(self, claim, *, limit=128):
        """Existing command receipts and outputs, not a second transcript store."""
        async with self.tasks.transaction(claim.task.owner_id) as connection:
            task, _ = await self.tasks.lock_claim(connection, claim)
            rows = await connection.fetch("""select e.sequence,e.event_type,e.payload_id,p.content,p.availability
                from jarvis_task_events e join jarvis_context_payloads p
                on p.owner_id=e.owner_id and p.id=e.payload_id
                where e.owner_id=$1 and e.task_id=$2 and e.event_type=any($3::text[])
                order by e.sequence desc limit $4""", task.owner_id, task.id,
                ["live_command_requested", "live_command_replied", "succeeded", "waiting_for_input"], limit)
            history = []
            for row in reversed(rows):
                if row["availability"] != "available" or row["content"] is None:
                    raise TaskError("work_history_unavailable", "An earlier work result was withdrawn.")
                history.append({"sequence": row["sequence"], "kind": row["event_type"],
                    "payload_id": str(row["payload_id"]), "content": json_object(row["content"])})
            return history

    async def read_inputs(self, claim, *, after=0, limit=32):
        if type(after) is not int or after < 0 or type(limit) is not int or not 1 <= limit <= 64:
            raise ValueError("invalid work message pagination")
        async with self.tasks.transaction(claim.task.owner_id) as connection:
            task, _ = await self.tasks.lock_claim(connection, claim)
            problem = await self.tasks.context_problem(connection, task)
            if problem:
                raise TaskError(problem, "The work context is no longer available.", 409)
            rows = await connection.fetch("""select e.sequence,e.payload_id,p.content,p.availability
                from jarvis_task_events e join jarvis_context_payloads p on p.owner_id=e.owner_id and p.id=e.payload_id
                where e.owner_id=$1 and e.task_id=$2 and e.event_type=$3 and e.sequence>$4
                order by e.sequence limit $5""", task.owner_id, task.id, INPUT, after, limit)
            messages = []
            for row in rows:
                if (row["availability"] != "available" or row["content"] is None
                        or await connection.fetchval("select lock_jarvis_copy_source($1,$2)",
                                                     task.owner_id, row["payload_id"]) != "available"):
                    raise TaskError("work_input_source_unavailable", "A work message was removed or superseded.", 409)
                messages.append(WorkMessage(row["sequence"], row["payload_id"],
                                            json_object(row["content"])["input"]))
            return messages

    async def publish_progress(self, claim, text, *, source_payload_ids):
        """Ordinary assistant text only; this is not a verified effect receipt."""
        if not isinstance(text, str) or not text.strip() or len(text.encode("utf-8")) > 16384:
            raise ValueError("invalid work progress")
        if not source_payload_ids:
            raise ValueError("work progress requires its actual context sources")
        async with self.tasks.transaction(claim.task.owner_id) as connection:
            task, _ = await self.tasks.lock_claim(connection, claim)
            problem = await self.tasks.context_problem(connection, task)
            if problem:
                raise TaskError(problem, "The work context is no longer available.", 409)
            await self.tasks.emit(connection, task, PROGRESS, attempt_id=claim.attempt_id,
                payload={"text": text, "authorship": "assistant", "effect_receipt": False},
                source_payload_ids=tuple(source_payload_ids))

    async def finish(self, claim, outcome, *, through_sequence):
        """Serialize completion with append; late input stays queued, not dropped.

        through_sequence is the worker's transport cursor, not a statement that
        the model applied a correction. Actual effect/outcome checks run first.
        """
        if type(through_sequence) is not int or through_sequence < 0:
            raise ValueError("invalid work input cursor")
        async with self.tasks.transaction(claim.task.owner_id) as connection:
            task, attempt = await self.tasks.lock_claim(connection, claim)
            rows = await connection.fetch("""select e.payload_id from jarvis_task_events e
                where e.owner_id=$1 and e.task_id=$2 and
                ((e.event_type='work_input' and e.sequence<=$3)
                    or e.event_type in ('live_command_replied','succeeded','waiting_for_input'))
                order by e.sequence desc limit 512""", task.owner_id, task.id, through_sequence)
            roots = {task.description_payload_id, *(row["payload_id"] for row in rows)}
            if attempt["context_manifest_id"] is not None:
                manifest = await connection.fetchval("""select payload_id from jarvis_task_context_manifests
                    where owner_id=$1 and id=$2 and status='active'""", task.owner_id, attempt["context_manifest_id"])
                if manifest is not None:
                    roots.add(manifest)
            await self.tasks._finish_locked(connection, task, attempt, outcome, source_payload_ids=tuple(roots))
            current = await self.tasks.load(connection, task.owner_id, task.id, lock=True)
            if current.status not in {TaskStatus.SUCCEEDED, TaskStatus.WAITING}:
                return True
            pending = await connection.fetchval("""select exists(select 1 from jarvis_task_events
                where owner_id=$1 and task_id=$2 and event_type=$3 and sequence>$4)""",
                task.owner_id, task.id, INPUT, through_sequence)
            if pending:
                # Existing accepted work, not a fresh admission or a budget reset.
                # The normal claim path enforces attempt/time limits before execution.
                await self._queue(connection, current)
            return True

    async def _queue(self, connection, task):
        await self.tasks.transition(connection, task, TaskStatus.QUEUED, reason="work_input_pending")
        await connection.execute("""update jarvis_background_tasks set available_at=clock_timestamp(),result_payload_id=null
            where owner_id=$1 and id=$2""", task.owner_id, task.id)
        await self.tasks.emit(connection, task, "work_continued")
