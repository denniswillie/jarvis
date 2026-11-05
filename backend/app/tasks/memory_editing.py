"""Frontier-selected corrections under a task-scoped user request.

Hermes chooses sources and replacement text. This adapter only derives immutable
exact grants and uses the existing action/receipt boundary for their execution.
"""
from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.knowledge.lifecycle_tools import CorrectSourceArguments
from app.knowledge.models import SourceError
from app.tasks.memory_corrections import MEMORY_CORRECTION_CAPABILITY
from app.tasks.memory_evidence import TaskMemoryCitation
from app.tasks.models import ClaimLost, TaskError, TaskVoiceSource, digest, json_object
from app.tasks.tool_bridge import GrantedToolAction
from app.tools.inputs import failed

MEMORY_EDIT_CAPABILITY = "memory.correct_sources"
CORRECTION_TOOL = "correct_memory_source"


class MemoryEditScope(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    task_id: UUID
    area_id: UUID
    max_sources: int = Field(ge=1, strict=True)


async def admit_memory_editing(tasks, connection, owner_id, task_id, event):
    """Called by trusted task admission, never by the frontier tool itself."""
    area = await connection.fetchval(
        "select knowledge_area_id from jarvis_background_tasks where owner_id=$1 and id=$2",
        owner_id, task_id)
    if area is None:
        raise TaskError("memory_edit_area_required", "Memory editing needs a selected owned knowledge area.", 403)
    scope = MemoryEditScope(task_id=task_id, area_id=area, max_sources=tasks.limits.tool_steps)
    payload = await tasks.payload(connection, owner_id, scope.model_dump(mode="json"),
                                  source_payload_ids=(event["payload_id"],))
    return await connection.fetchval("""insert into jarvis_operation_grants
        (owner_id,capability,operation_key,request_event_id,request_revision,
         scope_revision,scope_payload_id,expires_at)
        values($1,$2,$3,$4,$5,1,$6,clock_timestamp()+$7*interval '1 second') returning id""",
        owner_id, MEMORY_EDIT_CAPABILITY, "memory-edit-task:" + task_id.hex,
        event["id"], event["revision"], payload, float(tasks.limits.memory_edit_grant_seconds))


class TaskMemoryEditingBridge:
    def __init__(self, bridge, backend, grant_id, scope):
        self.base, self.backend = bridge, backend
        self.context, self.grant_id, self.scope = bridge.context, grant_id, scope

    def __getattr__(self, name):
        return getattr(self.base, name)

    @classmethod
    async def attach(cls, bridge, backend):
        ctx = bridge.context
        async with ctx.tasks.transaction(ctx.task.owner_id) as connection:
            task, _ = await ctx.tasks.lock_claim(connection, ctx.claim)
            if await ctx.tasks.context_problem(connection, task):
                raise ClaimLost()
            rows = await connection.fetch("""select g.id from jarvis_task_grants t
                join jarvis_operation_grants g on g.owner_id=t.owner_id and g.id=t.grant_id
                where t.owner_id=$1 and t.task_id=$2 and g.capability=$3""",
                task.owner_id, task.id, MEMORY_EDIT_CAPABILITY)
            if not rows:
                return bridge
            if len(rows) != 1 or backend is None or backend.context is not ctx:
                raise TaskError("memory_edit_scope_unavailable", "A single supported memory-editing scope is required.")
            grant = await ctx.tasks.validate_grant(connection, task.owner_id, rows[0]["id"])
            scope = MemoryEditScope.model_validate(json_object(grant["scope"]))
            if scope.task_id != task.id or scope.area_id != backend.evidence.area_id:
                raise TaskError("memory_edit_scope_mismatch", "The memory-editing area or task does not match.", 403)
        return cls(bridge, backend, grant["id"], scope)

    def catalog(self):
        parameters = {"oneOf": [
            CorrectSourceArguments.model_json_schema(),
            {"type": "object", "additionalProperties": False, "required": ["operation_ref"],
             "properties": {"operation_ref": {"type": "string", "pattern": "^op_[0-9a-f]{32}$"}}},
        ]}
        return [item for item in self.base.catalog() if item["name"] != CORRECTION_TOOL] + [{
            "name": CORRECTION_TOOL,
            "description": (
                "Correct retained source statements as requested in this task. Inspect current originals "
                "using search_memory or explain_memory before choosing a source and formulating its full "
                "replacement. Preserve unrelated facts. If the intended source or change is ambiguous, "
                "ask the user through your normal expert clarification instead of guessing or writing. "
                "A correction is model-formulated, not independently observed truth. This tool does not "
                "edit original notes/files or delete memories. For an exact measurement, replacement is a JSON "
                "object encoded as a string, containing subject, metric, value (decimal string), unit, "
                "measured_on (YYYY-MM-DD), and optional notes. Preserve the other fields. "
                "Supply source_id, expected_revision and "
                "replacement. The request and source own one immutable operation: retry identical "
                "parameters or the returned operation_ref, never a new revision to repeat the same edit. "
                "Use returned current/historical evidence and accepted task_operation references in "
                "your result. Ingestion pending is not searchable completion."),
            "parameters": parameters,
        }]

    def authorized_operations(self):
        return self.base.authorized_operations() + [{
            "tool": CORRECTION_TOOL, "capability": MEMORY_EDIT_CAPABILITY,
            "scope": self.scope.model_dump(mode="json"),
            "meaning": "Task-scoped correction of retained statements for the user's actual request, "
                       "not permission to infer new obligations, edit unrelated originals or delete evidence. "
                       "A completed correction returns its exact operation_ref.",
        }]

    async def _parent(self, connection):
        ctx, tasks, owner = self.context, self.context.tasks, self.context.task.owner_id
        await tasks.lock_owner(connection, owner)
        task, _ = await tasks.lock_claim(connection, ctx.claim)
        if await tasks.context_problem(connection, task):
            raise ClaimLost()
        attached = await connection.fetchval("""select exists(select 1 from jarvis_task_grants
            where owner_id=$1 and task_id=$2 and grant_id=$3)""", owner, task.id, self.grant_id)
        grant = await tasks.validate_grant(connection, owner, self.grant_id)
        if (not attached or grant["capability"] != MEMORY_EDIT_CAPABILITY
                or MemoryEditScope.model_validate(json_object(grant["scope"])) != self.scope
                or task.knowledge_area_id != self.scope.area_id):
            raise TaskError("memory_edit_scope_mismatch", "This task's memory-editing authority changed.", 403)
        # Use the request owning the current instructions, not the latest
        # control event. Resume preserves this payload; update replaces it.
        description = await connection.fetchrow("""select content,availability
            from jarvis_context_payloads where owner_id=$1 and id=$2 for share""",
            owner, task.description_payload_id)
        if description is None or description["availability"] != "available":
            raise TaskError("memory_edit_request_missing", "The effective instruction request is unavailable.", 403)
        try:
            reference = json_object(description["content"])["instruction_request"]
            request = TaskVoiceSource(UUID(reference["event_id"]), reference["revision"],
                                      UUID(reference["session_id"]))
        except (KeyError, TypeError, ValueError, AttributeError):
            raise TaskError("memory_edit_request_missing",
                "This task needs a fresh instruction request; a control event cannot supply edit authority.", 403) from None
        event = await tasks._voice_source(connection, owner, request)
        return task, grant, event

    @staticmethod
    def _action(grant):
        return GrantedToolAction(grant["id"], MEMORY_CORRECTION_CAPABILITY, CORRECTION_TOOL,
            json_object(grant["scope"]), grant["request_event_id"], grant["request_revision"])

    async def prepare(self, arguments):
        parameters = arguments.model_dump(mode="json")
        ctx, tasks, owner = self.context, self.context.tasks, self.context.task.owner_id
        async with self.backend.sources.transaction(owner) as connection:
            task, parent, event = await self._parent(connection)
            key = "memory-edit:" + digest({"request_event_id": str(event["id"]),
                                          "source_id": str(arguments.source_id)})
            previous = await connection.fetchrow("""select g.id,
                exists(select 1 from jarvis_task_grants t where t.owner_id=g.owner_id
                    and t.grant_id=g.id and t.task_id=$3) as attached
                from jarvis_operation_grants g where g.owner_id=$1 and g.operation_key=$2""",
                owner, key, task.id)
            if previous is not None:
                if not previous["attached"]:
                    raise TaskError("memory_edit_already_owned",
                        "Another task owns this request's correction. Inspect that task; do not duplicate it.")
                grant = await tasks.validate_grant(connection, owner, previous["id"])
                if (grant["capability"] != MEMORY_CORRECTION_CAPABILITY
                        or grant["request_event_id"] != event["id"] or grant["request_revision"] != event["revision"]
                        or digest(json_object(grant["scope"])) != digest(parameters)):
                    raise TaskError("memory_edit_operation_conflict",
                        "This request already fixes a different correction. Reuse its exact operation; "
                        "a further change requires a new user request.")
                return self._action(grant)
            count = await connection.fetchval("""select count(*) from jarvis_task_grants t
                join jarvis_operation_grants g on g.owner_id=t.owner_id and g.id=t.grant_id
                where t.owner_id=$1 and t.task_id=$2 and g.capability=$3""",
                owner, task.id, MEMORY_CORRECTION_CAPABILITY)
            if count >= self.scope.max_sources:
                raise TaskError("memory_edit_allowance_exhausted", "This task's correction allowance is exhausted.")
            original = await self.backend.sources._read(
                connection, owner, self.scope.area_id, arguments.source_id)
            if original.revision != arguments.expected_revision:
                raise SourceError("knowledge_source_revision_conflict", "Read the current original before correcting it.")
            if original.content.get("format") == "jarvis.measurement.v1":
                from app.knowledge.measurements import measurement_correction
                measurement_correction(arguments.replacement)
            # This is interpretation, unlike execution of an already exact grant.
            # A missing original is a recoverable preflight error, before any action submission.
            await self.backend.evidence._access_in_transaction(connection,
                [TaskMemoryCitation.from_metadata(original.original_evidence())], bind=False)
            payload = await tasks.payload(connection, owner, parameters,
                source_payload_ids=(parent["scope_payload_id"], original.payload_id, event["payload_id"]))
            identity = await connection.fetchval("""insert into jarvis_operation_grants
                (owner_id,capability,operation_key,request_event_id,request_revision,
                 scope_revision,scope_payload_id,expires_at)
                values($1,$2,$3,$4,$5,1,$6,$7) returning id""",
                owner, MEMORY_CORRECTION_CAPABILITY, key, event["id"], event["revision"],
                payload, parent["expires_at"])
            await connection.execute("""insert into jarvis_task_grants(owner_id,task_id,grant_id)
                values($1,$2,$3)""", owner, task.id, identity)
            return self._action(await tasks.validate_grant(connection, owner, identity))

    async def invoke(self, name, arguments, *, provider_call_id):
        if name != CORRECTION_TOOL or set(arguments) == {"operation_ref"}:
            return await self.base.invoke(name, arguments, provider_call_id=provider_call_id)
        if (not isinstance(provider_call_id, str) or not 1 <= len(provider_call_id) <= 200
                or not provider_call_id.strip() or "\x00" in provider_call_id):
            raise ValueError("A bounded native tool-call identity is required")
        try:
            parsed = CorrectSourceArguments.model_validate(arguments)
        except ValidationError:
            return failed("invalid_tool_input",
                "Supply source_id, expected_revision and replacement, or only an existing operation_ref.")
        try:
            action = await self.prepare(parsed)
        except ClaimLost:
            raise
        except (TaskError, SourceError) as error:
            message = ("Inspect the current original with explain_memory before formulating the correction."
                       if error.code == "task_memory_evidence_not_read" else error.message)
            return failed(error.code, message + " No new correction was submitted by this invocation.")
        self.base.actions[action.reference] = action
        return await self.base.execute_granted(action, self.base.registry.get(CORRECTION_TOOL),
                                               provider_call_id=provider_call_id)
