"""Optional canonical-ingress publication; no vendor call runs at capture time."""
from contextlib import asynccontextmanager
from dataclasses import dataclass
from uuid import UUID

from app.images import ImageReference
from app.knowledge.indexing_policy import indexing_exclusion
from app.knowledge.models import SourceError, SourcePublication
from app.tasks.models import digest


@dataclass(frozen=True)
class CanonicalSourcePublication:
    sources: object
    area_id: UUID
    selections: object = None

    def __post_init__(self):
        if self.selections is not None and (
            self.selections.sources.pool is not self.sources.pool or self.selections.default_area_id != self.area_id
        ):
            raise ValueError("Canonical publication and selection must share a database and configured default")

    @asynccontextmanager
    async def transaction(self, tasks, owner_id):
        # Use one connection and one commit. Source ownership precedes task
        # ownership so source removal and context capture do not invert them.
        async with self.sources.transaction(owner_id) as connection:
            await tasks.lock_owner(connection, owner_id)
            yield connection

    async def publish(self, connection, owner_id, record, *, requested_area_id=None):
        if record.owner_id != owner_id:
            raise SourceError("knowledge_source_scope_mismatch", "The event belongs to a different owner.", 403)
        if requested_area_id is not None and (
            not isinstance(requested_area_id, UUID)
            or self.selections is None and requested_area_id != self.area_id
        ):
            raise SourceError("knowledge_source_scope_mismatch", "The receipt cannot override this publisher's scope.", 403)
        area_id = self.area_id
        if self.selections is not None and record.is_current:
            if (record.kind == "assistant_output" and record.trust == "assistant_output"
                    and (record.content or {}).get("authored_by") == "frontier_expert"):
                expert_area = await self._expert_area(connection, owner_id, record)
                if expert_area is None:
                    # Preserve canonical speech, including legacy results with
                    # no recorded task area, without guessing a memory scope.
                    return None
                if requested_area_id is not None and requested_area_id != expert_area:
                    raise SourceError("knowledge_context_scope_conflict", "Expert delivery and requested publication scope disagree.")
                requested_area_id = expert_area
            image_area = await self._image_area(connection, owner_id, record)
            if image_area is not None:
                if requested_area_id is not None and requested_area_id != image_area:
                    raise SourceError("knowledge_context_scope_conflict", "Image evidence and requested publication scope disagree.")
                requested_area_id = image_area
            area_id = await self.selections.bind_context(
                connection, owner_id, record, requested_area_id=requested_area_id)
        if record.finality not in {"final", "revised"} or record.availability != "available":
            return None
        if not record.is_current:
            raise SourceError("knowledge_context_superseded", "Only a current canonical revision may be published.")
        status = await connection.fetchval(
            "select status from jarvis_knowledge_areas where owner_id=$1 and id=$2 for share",
            owner_id, area_id,
        )
        # Archive pauses memory publication, not canonical conversation capture.
        # Missing/deleted scopes still fail through ordinary publication checks.
        if status == "archived" or (self.selections is not None and status == "deleted"):
            return None
        namespace = "context:" + digest({"namespace": record.source_namespace})
        current = await connection.fetchrow(
            """select s.current_revision,s.status,e.source_namespace,e.source_event_id
                from jarvis_knowledge_sources s left join jarvis_knowledge_context_sources cs
                on cs.owner_id=s.owner_id and cs.source_id=s.id and cs.source_revision=s.current_revision
                left join jarvis_context_events e on e.owner_id=cs.owner_id and e.id=cs.event_id
                where s.owner_id=$1 and s.area_id=$2 and s.namespace=$3 and s.source_key=$4""",
            owner_id, area_id, namespace, record.source_event_id,
        )
        if current is not None and (
            current["status"] == "forgotten"
            or (current["source_namespace"] is not None and
                (current["source_namespace"], current["source_event_id"]) !=
                (record.source_namespace, record.source_event_id))
        ):
            # Keep capture, but do not let a late ASR revision or importer replay
            # undo explicit forgetting or replace a separately authored correction.
            return None
        revision = current["current_revision"] if current is not None else 0
        source = await self.sources.publish_in_transaction(connection, owner_id, SourcePublication(
            area_id=area_id, namespace=namespace, source_key=record.source_event_id,
            payload_id=record.payload_id, source_type=record.kind, trust=record.trust,
            source_time=record.source_time, session_id=record.session_id,
            time_basis="Canonical event source_time; original payload retains capture and ingress provenance",
            expected_revision=revision,
        ))
        binding_sql = """insert into jarvis_knowledge_context_sources
            (owner_id,source_id,source_revision,event_id,event_revision)
            values($1,$2,$3,$4,$5) on conflict(owner_id,source_id,source_revision) do nothing"""
        exclusion = indexing_exclusion({
            "context_binding_matches": (source.owner_id == owner_id and source.area_id == area_id
                and source.payload_id == record.payload_id and source.content == record.content
                and source.is_current),
            "context_event_id": record.id,
            "source_type": source.source_type, "trust": source.trust,
            "context_kind": record.kind, "context_trust": record.trust,
            "context_source_namespace": record.source_namespace,
            "context_source_event_id": record.source_event_id,
            "source_namespace": namespace, "source_key": record.source_event_id,
        }, record.content)
        if exclusion is None:
            await connection.execute(binding_sql,
                owner_id, source.source_id, source.revision, record.id, record.revision)
        else:
            # Keep canonical evidence and its mapping, but avoid a worker claim
            # merely to discover this existing transport-only indexing policy.
            # The inserted binding is invisible to a same-statement table read:
            # include RETURNING explicitly, as well as an exact existing binding.
            # Never settle a claimed/dispatched/uncertain provider operation here.
            cancelled = await connection.fetchval(
                "with inserted as (" + binding_sql + " returning *) " + """
                update jarvis_memory_outbox o
                set status='cancelled',last_error=$8,updated_at=clock_timestamp()
                from jarvis_knowledge_source_versions v,jarvis_knowledge_sources s
                where o.owner_id=$1 and o.document_id=$6 and o.generation=$7
                    and o.operation='retain' and o.status='pending'
                    and o.dispatch_count=0 and o.worker_id is null and o.lease_until is null
                    and v.owner_id=o.owner_id and v.document_id=o.document_id
                    and v.source_id=$2 and v.revision=$3 and v.generation=o.generation
                    and v.payload_id=$9
                    and s.owner_id=v.owner_id and s.id=v.source_id and s.status='active'
                    and s.current_revision=v.revision and s.generation=v.generation
                    and exists (
                        select 1 from inserted b where b.event_id=$4 and b.event_revision=$5
                        union all
                        select 1 from jarvis_knowledge_context_sources b
                        where b.owner_id=$1 and b.source_id=$2 and b.source_revision=$3
                            and b.event_id=$4 and b.event_revision=$5
                    ) returning true""",
                owner_id, source.source_id, source.revision, record.id, record.revision,
                source.document_id, source.generation, exclusion, record.payload_id,
            )
            if cancelled:
                source = source.model_copy(update={"ingestion_status": "cancelled",
                    "ingestion_reason": exclusion, "semantic_index_exclusion": exclusion})
        return source

    async def _expert_area(self, connection, owner_id, record):
        """Resolve original result ownership, not the receiving session's topic.

        Provenance labels alone are insufficient: the owned voice attempt must
        match the intent, receiving session and actual result dependency. Old
        attempt generations remain historical evidence, not renewed delivery
        authority. Existing source-copy fences still govern payload availability.
        """
        content = record.content or {}
        intent, attempt, generation = (content.get("intent_id"), content.get("attempt_id"),
                                       content.get("delivery_generation"))
        if (not isinstance(intent, str) or not isinstance(attempt, str)
                or type(generation) is not int or generation < 1):
            return None
        try:
            intent_id, attempt_id = UUID(intent), UUID(attempt)
        except ValueError:
            return None
        return await connection.fetchval("""select t.knowledge_area_id
            from jarvis_attention_attempts a join jarvis_attention_intents i
                on i.owner_id=a.owner_id and i.id=a.intent_id
            join jarvis_background_tasks t on t.owner_id=i.owner_id and t.id=i.task_id
            join jarvis_context_payload_dependencies d
                on d.owner_id=i.owner_id and d.source_payload_id=i.source_payload_id
            where a.owner_id=$1 and a.id=$2 and i.id=$3 and a.generation=$4
                and a.channel='voice' and a.session_id=$5 and d.dependent_payload_id=$6
                and i.source_namespace='task_result'""",
            owner_id, attempt_id, intent_id, generation, record.session_id, record.payload_id)

    async def _image_area(self, connection, owner_id, record):
        """Route retained-image receipts by recorded proof, not delivery time.

        These are application image envelopes, not model-selected area labels.
        Each root must be an actual dependency of the owned canonical payload.
        Ordinary camera frames without a retained-source proof keep normal scope.
        """
        if record.availability != "available":
            return None
        if record.kind == "observation" and record.trust in {"sensor", "derived"}:
            images = (record.content or {}).get("images", [])
        elif record.kind == "tool_outcome" and record.trust == "tool_receipt":
            result = (record.content or {}).get("result", {})
            if not isinstance(result, dict) or result.get("status") != "success" or result.get("image_status") != "selected":
                return None
            images = [result.get("image")]
        else:
            return None
        if not isinstance(images, list) or len(images) > 64:
            raise SourceError("knowledge_image_scope_invalid", "The image evidence envelope is invalid.")
        roots = set()
        for metadata in images:
            if not isinstance(metadata, dict) or metadata.get("record_type") != "source_evidence":
                continue
            try:
                reference = ImageReference.from_metadata(metadata, owner_id=owner_id)
            except (KeyError, TypeError, ValueError) as error:
                raise SourceError("knowledge_image_scope_invalid", "The retained image reference is invalid.") from error
            roots.update(reference.source_payload_ids)
        if not roots:
            return None
        rows = await connection.fetch("""select d.source_payload_id as root,m.area_id
            from jarvis_context_payload_dependencies d join jarvis_context_memory_sources m
                on m.owner_id=d.owner_id and m.dependent_payload_id=d.source_payload_id
            where d.owner_id=$1 and d.dependent_payload_id=$2 and d.source_payload_id=any($3::uuid[])
            group by d.source_payload_id,m.area_id""", owner_id, record.payload_id, sorted(roots))
        areas = {row["area_id"] for row in rows}
        if {row["root"] for row in rows} != roots or len(areas) != 1:
            raise SourceError("knowledge_image_scope_unavailable", "Retained image dependencies have no single verified area.")
        return areas.pop()
