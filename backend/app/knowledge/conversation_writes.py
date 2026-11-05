"""Foreground explicit memories through canonical sources and existing provenance.

Semantic intent belongs to the model. Deterministic code enforces the finalized
request, scope, original dependencies and repeat identity, not phrase matching.
"""
from __future__ import annotations

from uuid import UUID

from app.knowledge.models import SourceError
from app.knowledge.publication import CanonicalSourcePublication
from app.knowledge.read_provenance import SourceReadProvenance
from app.tasks.context_events import ContextIngress
from app.tasks.context_repository import ContextRepository
from app.tasks.models import TaskVoiceSource, digest
from app.tools.inputs import optional_datetime, required_string
from app.tools.invocation import current_tool_invocation


NAMESPACE = "conversation-explicit-memory"


class ConversationMemoryWriteBackend:
    capture_after_commit = True

    def __init__(self, sources, requests, *, owner_id, area_id, access_check=None, selections=None):
        if sources.pool is not requests.pool:
            raise ValueError("Memory publication and request admission must share one pool")
        self.sources, self.requests = sources, requests
        self.owner_id, self.area_id, self.access_check = owner_id, area_id, access_check
        self.selections = selections
        self.publisher = ContextRepository(requests, publication=CanonicalSourcePublication(sources, area_id, selections))
        self.provenance = SourceReadProvenance(sources)

    def _authorize(self, owner_id):
        if owner_id != self.owner_id:
            raise SourceError("knowledge_owner_mismatch", "Memory is outside this runtime owner scope.", 403)
        if self.access_check is not None:
            self.access_check(owner_id)

    async def remember(self, *, owner_id, session_id, parameters):
        self._authorize(owner_id)
        invocation = current_tool_invocation()
        if (session_id is None or invocation.task is not None or invocation.provider_call_id is None
                or invocation.request_event_id is None):
            raise SourceError("knowledge_request_not_finalized",
                "A current finalized conversational user request is required; worker actions use their own grants.", 403)
        if set(parameters) - {"memory_type", "content", "occurred_at"}:
            raise ValueError("Unsupported selected-memory write parameters")
        kind = required_string(parameters, "memory_type", max_length=30)
        content = required_string(parameters, "content", max_length=8000)
        if kind not in {"fact", "experience", "procedure", "goal"} or "\x00" in content:
            raise ValueError("Invalid explicit memory statement")
        occurred = optional_datetime(parameters, "occurred_at")
        request = TaskVoiceSource(invocation.request_event_id, invocation.request_revision, session_id)
        payload = {
            "memory_type": kind, "content": content,
            "occurred_at": occurred.isoformat() if occurred is not None else None,
            "attribution": {
                "kind": "model_formulated_from_finalized_user_request",
                "request_event_id": str(request.event_id),
                "request_revision": request.revision,
            },
        }
        # Exact retry under a new provider call ID is the same saved statement.
        # A genuinely later user request is a distinct source, even with the same text.
        identity = digest({"session_id": str(session_id), "statement": payload})
        async with self.sources.pool.acquire() as connection:
            parents = await connection.fetch("""select s.id,s.area_id from jarvis_knowledge_context_sources cs
                join jarvis_knowledge_sources s on s.owner_id=cs.owner_id and s.id=cs.source_id
                where cs.owner_id=$1 and cs.event_id=$2 and cs.event_revision=$3""",
                owner_id, request.event_id, request.revision)
        if len(parents) != 1:
            raise SourceError("knowledge_request_source_unavailable",
                "The request has no unambiguous source-backed original; nothing was saved.", 409)
        parent_area = parents[0]["area_id"] if self.selections is not None else self.area_id
        parent = await self.sources.read(owner_id, parent_area, parents[0]["id"])
        if (parent.context_event_id != request.event_id or parent.context_event_revision != request.revision
                or parent.context_session_id != session_id):
            raise SourceError("knowledge_request_source_changed",
                "The finalized request changed; do not reuse its old memory authorization.", 409)

        async def materialize(connection, originals):
            await self.requests.lock_owner(connection, owner_id)
            await self.requests._voice_source(connection, owner_id, request)
            self._authorize(owner_id)
            prior = await connection.fetchval("""select event_id from jarvis_context_ingress_receipts
                where owner_id=$1 and source_namespace=$2 and source_event_id=$3 and receipt_id=$3""",
                owner_id, NAMESPACE, identity)
            if prior is None:
                area_id = self.area_id
                if self.selections is not None:
                    selected = await self.selections.read_in_transaction(connection, owner_id, session_id)
                    area_id = UUID(selected["area_id"])
                await self.sources._area(connection, owner_id, area_id)
            event = await self.publisher.append_in_transaction(connection, owner_id, ContextIngress(
                session_id=session_id, source_namespace=NAMESPACE, source_event_id=identity,
                receipt_id=identity, kind="tool_outcome", trust="tool_receipt", source_time=occurred,
                payload=payload, expected_revision=0, source_payload_ids=tuple(originals),
            ))
            stored = await connection.fetchrow("""select s.id,s.area_id from jarvis_knowledge_context_sources cs
                join jarvis_knowledge_sources s on s.owner_id=cs.owner_id and s.id=cs.source_id
                where cs.owner_id=$1 and cs.event_id=$2 and cs.event_revision=$3""", owner_id, event.id, event.revision)
            if stored is None:
                raise SourceError("knowledge_publication_unavailable", "The saved statement is not available as a source.")
            area_id = stored["area_id"] if self.selections is not None else self.area_id
            source = await self.sources._read(connection, owner_id, area_id, stored["id"])
            metadata = source.original_evidence()
            metadata.pop("content")
            self._authorize(owner_id)
            return {
                "memory_status": "stored", "memory_id": str(source.source_id), "source": metadata,
                "replayed": prior is not None,
                "meaning": (
                    "Local source storage and queued ingestion, not semantic-search readiness or independent truth. "
                    "The statement was model-formulated from this finalized user request, whose original is preserved. "
                    "This does not schedule a reminder or infer a recurring responsibility. "
                    "Use explain_memory for the stored original; inspect source state after an interrupted acknowledgment."
                ),
            }, event.payload_id

        return await self.provenance.bind(owner_id, parent_area, [parent.original_evidence()], materialize)
