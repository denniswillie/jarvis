"""Exact preauthorized task corrections using existing source/action transactions.

This adapter does not issue edit grants or infer user intent. Dynamic frontier
selection and foreground admission are separate integration responsibilities.
"""
from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from app.knowledge.lifecycle import SourceLifecycleService
from app.knowledge.lifecycle_tools import CorrectSourceArguments
from app.knowledge.models import SourceError
from app.tasks.context_repository import ContextRepository
from app.tasks.memory_evidence import TaskMemoryCitation
from app.tasks.models import TaskError, TaskVoiceSource, digest, json_object
from app.tools.inputs import success
from app.tools.invocation import current_tool_invocation
from app.tools.registry import ToolDefinition

MEMORY_CORRECTION_CAPABILITY = "memory.correct_source"


class TaskMemoryCorrectionBackend:
    def __init__(self, evidence):
        self.evidence = evidence
        self.context, self.sources = evidence.context, evidence.sources
        self.lifecycle = SourceLifecycleService(self.sources, self.context.tasks)

    async def _capture_context(self, connection, task, manifest_id, original):
        """Capture source lineage before this task invalidates its own copies."""
        owner = task.owner_id
        old = await connection.fetchrow("""select m.*,p.content,p.availability
            from jarvis_task_context_manifests m join jarvis_context_payloads p
            on p.owner_id=m.owner_id and p.id=m.payload_id
            where m.owner_id=$1 and m.id=$2 and m.task_id=$3 for update of m""",
            owner, manifest_id, task.id)
        if (old is None or old["status"] != "active" or old["availability"] != "available"
                or old["task_revision"] != self.context.claim.task_revision):
            raise TaskError("memory_correction_context_unavailable", "The editing context is no longer valid.")
        bound = await connection.fetch("""select c.*,p.availability,p.content is null as content_removed,
            lock_jarvis_copy_source(c.owner_id,c.payload_id) as copy_status
            from jarvis_task_context_sources c join jarvis_context_payloads p
            on p.owner_id=c.owner_id and p.id=c.payload_id
            where c.owner_id=$1 and c.manifest_id=$2""", owner, manifest_id)
        direct = await connection.fetch("""select area_id,source_id,source_revision,source_generation,
            payload_id,content_digest,source_usage from jarvis_task_memory_sources
            where owner_id=$1 and manifest_id=$2""", owner, manifest_id)
        # Memory tools can read an older derived source outside the selected
        # conversation window. Its lineage matters just as much as a message's.
        roots = {row["payload_id"] for row in bound if row["copy_status"] == "available"}
        roots.update(row["payload_id"] for row in direct)
        graph = await connection.fetch("""with recursive links as (
                select dependent_payload_id as dependent,source_payload_id as source
                from jarvis_context_payload_dependencies where owner_id=$1
                union
                select m.payload_id,c.payload_id from jarvis_task_context_manifests m
                join jarvis_task_context_sources c on c.owner_id=m.owner_id and c.manifest_id=m.id
                where m.owner_id=$1
                union
                select m.payload_id,c.payload_id from jarvis_task_context_manifests m
                join jarvis_task_memory_sources c on c.owner_id=m.owner_id and c.manifest_id=m.id
                where m.owner_id=$1
                union
                select e.payload_id,m.payload_id from jarvis_task_events e
                join jarvis_task_attempts a on a.owner_id=e.owner_id and a.id=e.attempt_id
                join jarvis_task_context_manifests m on m.owner_id=a.owner_id and m.id=a.context_manifest_id
                where e.owner_id=$1 and e.payload_id is not null
                union
                select a.checkpoint_payload_id,m.payload_id from jarvis_task_attempts a
                join jarvis_task_context_manifests m on m.owner_id=a.owner_id and m.id=a.context_manifest_id
                where a.owner_id=$1 and a.checkpoint_payload_id is not null
            ), ancestry(root,payload) as (
                select id,id from unnest($2::uuid[]) id
                union
                select a.root,l.source from ancestry a join links l on l.dependent=a.payload
            ), affected(payload) as (
                select $3::uuid
                union
                select l.dependent from affected a join links l on l.source=a.payload
            )
            select a.payload,bool_or(a.payload<>a.root) as upstream,
                exists(select 1 from affected f where f.payload=a.payload) as target_dependent
            from ancestry a group by a.payload""", owner, list(roots), original.payload_id)
        ancestors = [row["payload"] for row in graph]
        upstream = [row["payload"] for row in graph if row["upstream"]]
        origins = await connection.fetch("""select r.area_id,r.source_id,r.source_revision,r.source_generation,
                r.source_payload_id as payload_id,r.content_digest,r.source_usage
            from jarvis_context_memory_sources r
            where r.owner_id=$1 and r.dependent_payload_id=any($2::uuid[])
            union
            select r.area_id,r.source_id,r.source_revision,r.source_generation,
                r.payload_id,r.content_digest,r.source_usage
            from jarvis_task_context_manifests m join jarvis_task_memory_sources r
                on r.owner_id=m.owner_id and r.manifest_id=m.id
            where m.owner_id=$1 and m.payload_id=any($2::uuid[])
            union
            select v.area_id,v.source_id,v.revision,s.generation,v.payload_id,p.content_digest,
                case when v.revision=s.current_revision then 'current' else 'historical' end
            from jarvis_knowledge_source_versions v join jarvis_knowledge_sources s
                on s.owner_id=v.owner_id and s.id=v.source_id
            join jarvis_context_payloads p on p.owner_id=v.owner_id and p.id=v.payload_id
            where v.owner_id=$1 and v.payload_id=any($3::uuid[])""", owner, ancestors, upstream)
        await connection.fetch("""select id from jarvis_context_payloads where owner_id=$1
            and id=any($2::uuid[]) order by id for share""", owner,
            list({old["payload_id"], *(row["payload_id"] for row in bound),
                  *(row["payload_id"] for row in (*origins, *direct))}))
        memories = {}
        for row in (*origins, *direct):
            key = (row["source_id"], row["source_revision"])
            if key in memories:
                prior = memories[key]
                if any(prior[field] != row[field] for field in (
                        "area_id", "payload_id", "content_digest", "source_usage")):
                    raise TaskError("memory_correction_context_unavailable", "Conflicting original dependencies.")
                continue
            evidence = await self.sources._read(connection, owner, row["area_id"],
                                                row["source_id"], row["source_revision"])
            citation = TaskMemoryCitation.from_metadata(evidence.original_evidence())
            if (evidence.payload_id != row["payload_id"] or evidence.content_digest != row["content_digest"]
                    or citation.source_usage != row["source_usage"]
                    or (row["source_usage"] == "current" and evidence.generation != row["source_generation"])):
                raise TaskError("memory_correction_context_unavailable", "A context original changed before correction.")
            copy_status = await connection.fetchval(
                "select lock_jarvis_copy_source($1,$2)", owner, row["payload_id"])
            memories[key] = {field: row[field] for field in (
                "area_id", "source_id", "source_revision", "payload_id", "content_digest", "source_usage"
            )} | {"source_generation": evidence.generation, "copy_status": copy_status}
        return {"manifest": dict(old), "context_sources": bound, "memories": list(memories.values()),
                "retirable_payloads": {row["payload"] for row in graph if row["target_dependent"]}}

    async def _successor(self, connection, task, captured, original, revised, action_id):
        """Build a new source-safe snapshot; the revoked predecessor stays revoked."""
        ctx, owner = self.context, task.owner_id
        old = captured["manifest"]
        manifest_id = old["id"]
        retired = await connection.fetchrow("""select status,task_revision,payload_id
            from jarvis_task_context_manifests where owner_id=$1 and id=$2 and task_id=$3""",
            owner, manifest_id, task.id)
        if (retired is None or retired["status"] not in {"superseded", "revoked"}
                or retired["task_revision"] != ctx.claim.task_revision or retired["payload_id"] != old["payload_id"]):
            raise TaskError("memory_correction_context_unavailable", "The editing snapshot changed unexpectedly.")
        records = await ContextRepository(ctx.tasks)._records(
            connection, owner, [row["event_id"] for row in captured["context_sources"]])
        retired_events = []
        removed_states = {"unavailable", "deleted", "expired"}
        for prior in captured["context_sources"]:
            current = records[prior["event_id"]]
            if current.payload_id != prior["payload_id"] or current.revision != prior["event_revision"]:
                raise TaskError("memory_correction_context_unavailable", "A context source identity changed.")
            if current.availability != prior["availability"] or current.content_digest != prior["content_digest"]:
                already_gapped = (prior["content_removed"] and prior["availability"] in removed_states
                                  or prior["copy_status"] == "revoked")
                if (current.availability not in removed_states or current.content is not None
                        or not (already_gapped or prior["payload_id"] in captured["retirable_payloads"])):
                    raise TaskError("memory_correction_context_unavailable", "Unrelated context changed during correction.")
                retired_events.append(str(current.id))
            if current.availability == "available":
                copy_status = await connection.fetchval(
                    "select lock_jarvis_copy_source($1,$2)", owner, current.payload_id)
                if copy_status != "available":
                    if (copy_status != "revoked" or (
                            prior["copy_status"] != "revoked"
                            and current.payload_id not in captured["retirable_payloads"])):
                        raise TaskError("memory_correction_context_unavailable",
                                        "An unrelated context source lost copy permission.")
                    # A retained historical result is not permission to copy a
                    # revoked answer. Project an explicit gap without changing
                    # the original event or reviving its delivery.
                    records[current.id] = current.model_copy(
                        update={"availability": "unavailable", "content": None})
                    retired_events.append(str(current.id))
        retained_memories, retired_memories = [], []
        for row in captured["memories"]:
            state = await connection.fetchrow("""select availability,content_digest,
                lock_jarvis_copy_source(owner_id,id) as copy_status
                from jarvis_context_payloads where owner_id=$1 and id=$2""", owner, row["payload_id"])
            if state is not None and state["copy_status"] == "available":
                if state["content_digest"] != row["content_digest"]:
                    raise TaskError("memory_correction_context_unavailable", "A memory original changed unexpectedly.")
                retained_memories.append(row)
            elif (state is not None and state["copy_status"] in {"removed", "revoked"}
                    and row["source_id"] != original.source_id
                    and (row["copy_status"] == "revoked"
                         or row["payload_id"] in captured["retirable_payloads"])):
                retired_memories.append({"source_id": str(row["source_id"]), "revision": row["source_revision"],
                    "availability": state["availability"], "copy_status": state["copy_status"]})
            else:
                raise TaskError("memory_correction_context_unavailable", "An unrelated memory original was removed.")
        content = json_object(old["content"])
        messages = []
        for message in content["messages"]:
            record = records.get(UUID(message["id"]))
            if record is None:
                raise TaskError("memory_correction_context_unavailable", "A copied context message has no tracked source.")
            messages.append(record.as_context() | {"pinned": message.get("pinned", False)})
        content = content | {"messages": messages, "local_memory_correction": {
            "previous_manifest_id": str(manifest_id), "action_id": str(action_id),
            "source_id": str(original.source_id), "prior_revision": original.revision,
            "revision": revised.revision, "retired_context_event_ids": retired_events,
            "retired_memory_evidence": retired_memories,
            "meaning": "This task corrected its own source. Retired copied reads are gaps, not current evidence; "
                       "retained earlier originals are explicitly historical. No retired payload was restored.",
        }}
        parents = {revised.payload_id, *(row["payload_id"] for row in retained_memories),
                   *(record.payload_id for record in records.values() if record.availability == "available")}
        payload = await ctx.tasks.payload(connection, owner, content, source_payload_ids=tuple(parents))
        identity = await connection.fetchval("""insert into jarvis_task_context_manifests
            (owner_id,task_id,task_revision,context_watermark,payload_id)
            values($1,$2,$3,$4,$5) returning id""",
            owner, task.id, task.revision, old["context_watermark"], payload)
        if records:
            await connection.executemany("""insert into jarvis_task_context_sources
                (owner_id,manifest_id,event_id,event_revision,payload_id,content_digest,source_usage)
                values($1,$2,$3,$4,$5,$6,$7)""",
                [(owner, identity, record.id, record.revision, record.payload_id,
                  record.content_digest, prior["source_usage"])
                 for prior in captured["context_sources"] for record in [records[prior["event_id"]]]])
        for row in retained_memories:
            target = row["source_id"] == original.source_id
            await connection.execute("""insert into jarvis_task_memory_sources
                (owner_id,manifest_id,area_id,source_id,source_revision,source_generation,
                 payload_id,content_digest,source_usage) values($1,$2,$3,$4,$5,$6,$7,$8,$9)""",
                owner, identity, row["area_id"], row["source_id"], row["source_revision"],
                revised.generation if target else row["source_generation"],
                row["payload_id"], row["content_digest"], "historical" if target else row["source_usage"])
        await connection.execute("""update jarvis_task_attempts set context_manifest_id=$3
            where owner_id=$1 and id=$2""", owner, ctx.claim.attempt_id, identity)
        await ctx.tasks.emit(connection, task, "context_advanced_after_local_correction",
            payload=content["local_memory_correction"] | {"manifest_id": str(identity)}, attempt_id=ctx.claim.attempt_id)
        historical = []
        for row in retained_memories:
            if row["source_id"] == original.source_id:
                source = await self.sources._read(connection, owner, row["area_id"], original.source_id,
                                                  row["source_revision"])
                historical.append(TaskMemoryCitation.from_metadata(source.original_evidence()).model_dump(mode="json"))
        return identity, historical, retired_memories

    async def correct(self, *, owner_id, session_id, parameters):
        ctx, invocation = self.context, current_tool_invocation()
        scope = invocation.task
        if (owner_id != ctx.task.owner_id or session_id != ctx.task.session_id or scope is None
                or scope.task_id != ctx.task.id or scope.task_revision != ctx.claim.task_revision
                or scope.attempt_id != ctx.claim.attempt_id or scope.action_id is None):
            raise TaskError("memory_correction_scope_mismatch", "A current exact task action is required.", 403)
        arguments = CorrectSourceArguments.model_validate(parameters)
        async with self.sources.transaction(owner_id) as connection:
            await ctx.tasks.lock_owner(connection, owner_id)
            task, action = await ctx.actions._locked_action(connection, ctx.claim, scope.action_id)
            grant = await ctx.tasks.validate_grant(connection, owner_id, action["grant_id"])
            if (action["status"] != "submitting" or action["submission_token"] is None
                    or action["operation_key"] != scope.operation_key
                    or grant["capability"] != MEMORY_CORRECTION_CAPABILITY
                    or action["parameters_digest"] != digest(parameters)
                    or digest(json_object(grant["scope"])) != digest(parameters)
                    or invocation.request_event_id != grant["request_event_id"]
                    or invocation.request_revision != grant["request_revision"]):
                raise TaskError("memory_correction_not_authorized",
                                "The correction does not match its exact submitted grant.", 403)
            approved_session = await connection.fetchval(
                "select session_id from jarvis_context_events where owner_id=$1 and id=$2",
                owner_id, grant["request_event_id"])
            request = TaskVoiceSource(grant["request_event_id"], grant["request_revision"], approved_session)
            original = await self.sources._read(connection, owner_id, self.evidence.area_id, arguments.source_id)
            if original.revision != arguments.expected_revision:
                raise SourceError("knowledge_source_revision_conflict", "The granted source revision changed.")
            # This grant already fixes the exact source, revision and replacement.
            # Verify and bind the actual original atomically; a redundant prior
            # model read supplies no additional authority for this exact action.
            await self.evidence._access_in_transaction(connection,
                [TaskMemoryCitation.from_metadata(original.original_evidence())], bind=True)
            _, previous, _ = await ContextRepository(ctx.tasks)._archive_scope(connection, ctx.claim, None)
            captured = await self._capture_context(connection, task, previous, original)
            state = await self.lifecycle.correct_in_transaction(connection, owner_id, self.evidence.area_id,
                arguments.source_id, expected_revision=arguments.expected_revision,
                replacement=arguments.replacement, request=request, access_check=self.evidence.access_check)
            revised = await self.sources._read(connection, owner_id, self.evidence.area_id, arguments.source_id)
            manifest, historical, retired_memories = await self._successor(connection, task, captured, original, revised, action["id"])
            citation = (await self.evidence.bind_in_transaction(connection, [revised.original_evidence()]))[0]
            metadata = revised.original_evidence()
            metadata.pop("content")
            result = {"source_state": state, "source": metadata, "task_evidence": citation,
                "historical_evidence": historical, "context_manifest_id": str(manifest),
                "retired_memory_evidence": retired_memories,
                "context_transition": "Prior copied reads and retired_memory_evidence are unavailable and must not be cited. "
                    "Use returned current and historical "
                    "evidence, and read_task_context_source/archive for source-safe context; do not rely on retired copies.",
                "authorization": {"action_id": str(action["id"]), "request_event_id": str(request.event_id),
                                  "request_revision": request.revision},
                "meaning": "This exact local correction and its receipt committed together; ingestion is queued. "
                    "Use task_evidence for the corrected source. Prior citations for this source are obsolete; "
                    "historical_evidence explicitly identifies the retained earlier text. External originals were "
                    "not edited. The replacement is user-requested/model-formulated, not independently observed."}
            await ctx.actions.record_receipt_in_transaction(connection, owner_id, task.id, action["id"],
                action["submission_token"], "accepted", {"tool": "correct_memory_source",
                    "parameters_digest": digest(parameters), "result": success(result)})
        return result

    async def revalidate_receipt(self, action, result):
        if action.capability != MEMORY_CORRECTION_CAPABILITY or result.get("status") != "success":
            return result
        citations = [TaskMemoryCitation.model_validate(result["task_evidence"])] + [
            TaskMemoryCitation.model_validate(row) for row in result["historical_evidence"]]
        metadata = []
        for citation in citations:
            source = await self.sources.read(self.context.task.owner_id, self.evidence.area_id,
                citation.source_id, revision=citation.revision if citation.source_usage == "historical" else None)
            current = source.original_evidence()
            if TaskMemoryCitation.from_metadata(current) != citation:
                raise SourceError("memory_receipt_source_changed", "The correction source changed after its receipt.")
            metadata.append(current)
        await self.evidence.bind(metadata)
        return result


@dataclass
class TaskMemoryCorrectionToolModule:
    backend: TaskMemoryCorrectionBackend
    name = "task_memory_correction"

    def build_tools(self, context):
        async def correct(parameters):
            return success(await self.backend.correct(owner_id=context.owner_id,
                session_id=context.session_id, parameters=parameters))
        return (ToolDefinition("correct_memory_source",
            "Execute the exact preauthorized source correction. Its current original and granted revision "
            "are verified atomically; a separate earlier read call is not required. "
            "A retry reuses the accepted action; it cannot change parameters or correct another source. "
            "After success use the returned current and historical evidence references, not obsolete citations. "
            "Ingestion readiness and external-file edits are not implied.",
            CorrectSourceArguments.model_json_schema(), correct, mutates_state=True),)
