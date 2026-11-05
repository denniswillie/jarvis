"""Exact local memory effects through canonical publication and task receipts."""
from __future__ import annotations

from app.knowledge.models import SourceError
from app.knowledge.publication import CanonicalSourcePublication
from app.tasks.context_events import ContextIngress
from app.tasks.context_repository import ContextRepository
from app.tasks.memory_evidence import TaskMemoryCitation
from app.tasks.models import TaskError, digest, json_object
from app.tools.inputs import optional_datetime, required_string, success
from app.tools.invocation import current_tool_invocation

MEMORY_WRITE_CAPABILITY = "memory.remember"


class TaskMemoryWriteBackend:
    def __init__(self, evidence):
        self.evidence = evidence
        self.context, self.sources = evidence.context, evidence.sources
        self.publisher = ContextRepository(
            self.context.tasks,publication=CanonicalSourcePublication(self.sources,evidence.area_id),
        )

    async def remember(self, *, owner_id, session_id, parameters):
        ctx, invocation = self.context, current_tool_invocation()
        task = invocation.task
        if (owner_id != ctx.task.owner_id or session_id != ctx.task.session_id or task is None
                or task.task_id != ctx.task.id or task.task_revision != ctx.claim.task_revision
                or task.attempt_id != ctx.claim.attempt_id or task.action_id is None):
            raise TaskError("memory_write_scope_mismatch", "A current authorized task action is required.",403)
        if set(parameters)-{"memory_type","content","occurred_at"}:
            raise ValueError("Unsupported selected-memory write parameters")
        kind = required_string(parameters,"memory_type",max_length=30)
        content = required_string(parameters,"content",max_length=8000)
        if kind not in {"fact","experience","procedure","goal"} or "\x00" in content:
            raise ValueError("Invalid explicit memory statement")
        occurred_at = optional_datetime(parameters,"occurred_at")
        async with self.sources.transaction(owner_id) as connection:
            await ctx.tasks.lock_owner(connection,owner_id)
            _, action = await ctx.actions._locked_action(connection,ctx.claim,task.action_id)
            grant = await ctx.tasks.validate_grant(connection,owner_id,action["grant_id"])
            if (action["status"] != "submitting" or action["submission_token"] is None
                    or action["operation_key"] != task.operation_key
                    or grant["capability"] != MEMORY_WRITE_CAPABILITY
                    or action["parameters_digest"] != digest(parameters)
                    or digest(json_object(grant["scope"])) != digest(parameters)
                    or invocation.request_event_id != grant["request_event_id"]
                    or invocation.request_revision != grant["request_revision"]):
                raise TaskError("memory_write_not_authorized", "The write does not match its exact submitted grant.",403)
            approved_session = await connection.fetchval(
                "select session_id from jarvis_context_events where owner_id=$1 and id=$2",
                owner_id,grant["request_event_id"],
            )
            event = await self.publisher.append_in_transaction(connection,owner_id,ContextIngress(
                session_id=approved_session,source_namespace="task-explicit-memory",
                source_event_id=str(action["id"]),receipt_id=str(action["id"]),kind="tool_outcome",
                trust="tool_receipt",source_time=occurred_at,payload=parameters,expected_revision=0,
            ),payload_id=action["parameters_payload_id"])
            source_id = await connection.fetchval(
                "select source_id from jarvis_knowledge_context_sources where owner_id=$1 and event_id=$2",
                owner_id,event.id,
            )
            source = await self.sources._read(connection,owner_id,self.evidence.area_id,source_id)
            citation = (await self.evidence.bind_in_transaction(connection,[source.original_evidence()]))[0]
            metadata = source.original_evidence()
            metadata.pop("content")
            result = {
                "memory_status":"stored","memory_id":str(source.source_id),"source":metadata,
                "task_evidence":citation,
                "authorization":{"action_id":str(action["id"]),"request_event_id":str(grant["request_event_id"]),
                                 "request_revision":grant["request_revision"]},
                "meaning":(
                    "Receipt of local storage and queued ingestion at commit, not current semantic-search readiness. "
                    "This is the exact authorized statement, not an independently observed fact. "
                    "Use explain_memory for current availability and original content."
                ),
            }
            await ctx.actions.record_receipt_in_transaction(
                connection,owner_id,ctx.task.id,action["id"],action["submission_token"],"accepted",
                {"tool":"remember","parameters_digest":digest(parameters),"result":success(result)},
            )
        return result

    async def revalidate_receipt(self, action, result):
        if action.capability != MEMORY_WRITE_CAPABILITY or result.get("status") != "success":
            return result
        citation = TaskMemoryCitation.model_validate(result["task_evidence"])
        source = await self.sources.read(
            self.context.task.owner_id,self.evidence.area_id,citation.source_id,
        )
        if TaskMemoryCitation.from_metadata(source.original_evidence()) != citation:
            raise SourceError("memory_receipt_source_changed","The stored statement changed after this receipt.")
        # A resumed native attempt must bind its own source dependencies.
        await self.evidence.bind([source.original_evidence()])
        return result
