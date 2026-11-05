"""Owned area metadata and reversible visibility, not source or task authority."""
from __future__ import annotations

from uuid import UUID

from app.knowledge.models import KnowledgeAreaDefinition, SourceError
from app.knowledge.sources import SOURCE_VERSION_SQL
from app.tasks.context_events import ContextIngress
from app.tasks.context_repository import ContextRepository
from app.tasks.models import digest, json_object


AREA_OPERATIONS = "knowledge-area-management"


class KnowledgeAreaService:
    def __init__(self, sources, requests):
        if sources.pool is not requests.pool:
            raise ValueError("Area changes and user requests must share one database pool")
        self.sources, self.requests = sources, requests

    @staticmethod
    def _payload(row):
        return {key: (value.isoformat() if key in {"created_at", "updated_at"} else str(value) if key == "id" else value)
                for key, value in dict(row).items() if key != "owner_id"} | {"policy": {
                    "revision": row["revision"], "scope": "private_owner",
                    "ordinary_source_access_enabled": row["status"] == "active",
                    "ordinary_sources": "current_available_originals",
                    "historical_sources": "explicit_available_revision",
                    "cross_area_access": "explicit_owned_scope_not_name_matching",
                    "archive": "hide_from_ordinary_recall_preserve_originals",
                    "forget": "suppress_index_preserve_originals",
                    "delete": "explicit_source_lifecycle",
                    "descriptions_grant_authority": False,
                    "meaning": "Current application access/lifecycle rules, not editable responsibilities, retention schedules or external-account grants.",
                }}

    @staticmethod
    async def _read(connection, owner_id, area_id, *, lock=False):
        row = await connection.fetchrow(
            "select * from jarvis_knowledge_areas where owner_id=$1 and id=$2" + (" for update" if lock else ""),
            owner_id, area_id,
        )
        if row is None or row["status"] == "deleted":
            raise SourceError("knowledge_area_unavailable", "This knowledge area is not available for management.", 404)
        return row

    async def read(self, owner_id, area_id):
        async with self.sources.transaction(owner_id) as connection:
            return self._payload(await self._read(connection, owner_id, area_id))

    async def list(self, owner_id, *, after_id=None, limit=32, include_archived=True):
        if type(limit) is not int or not 1 <= limit <= 64 or type(include_archived) is not bool:
            raise ValueError("Invalid knowledge-area page bounds")
        if after_id is not None and not isinstance(after_id, UUID):
            raise ValueError("An exact area cursor UUID is required")
        async with self.sources.transaction(owner_id) as connection:
            if after_id is not None:
                await self._read(connection, owner_id, after_id)
            rows = await connection.fetch("""select * from jarvis_knowledge_areas
                where owner_id=$1 and status<>'deleted' and ($2::uuid is null or id>$2)
                    and ($3 or status='active') order by id limit $4""",
                owner_id, after_id, include_archived, limit + 1)
        return {"areas": [self._payload(row) for row in rows[:limit]],
                "next_after_id": str(rows[limit - 1]["id"]) if len(rows) > limit else None}

    async def history(self, owner_id, area_id, *, after_revision=0, limit=32):
        if type(after_revision) is not int or after_revision < 0 or type(limit) is not int or not 1 <= limit <= 64:
            raise ValueError("Invalid area history page bounds")
        async with self.sources.transaction(owner_id) as connection:
            area = await self._read(connection, owner_id, area_id)
            first = await connection.fetchval("""select min(revision) from jarvis_knowledge_area_versions
                where owner_id=$1 and area_id=$2""", owner_id, area_id)
            if after_revision and not await connection.fetchval("""select true from jarvis_knowledge_area_versions
                    where owner_id=$1 and area_id=$2 and revision=$3""", owner_id, area_id, after_revision):
                raise SourceError("knowledge_area_history_cursor", "Read a history page before using its revision cursor.", 400)
            rows = await connection.fetch("""select h.revision,h.status,h.origin,h.area_updated_at,h.recorded_at,
                    p.availability as metadata_availability,p.content
                from jarvis_knowledge_area_versions h
                left join jarvis_context_payloads p on p.owner_id=h.owner_id and p.id=h.metadata_payload_id
                where h.owner_id=$1 and h.area_id=$2 and h.revision>$3
                order by h.revision limit $4""", owner_id, area_id, after_revision, limit + 1)
            history = [{key: (value.isoformat() if key in {"area_updated_at", "recorded_at"} else value)
                        for key, value in dict(row).items() if key != "content"} |
                       {"metadata": json_object(row["content"]) if row["metadata_availability"] == "available"
                        and row["content"] is not None else None} for row in rows[:limit]]
        return {"area": self._payload(area), "history": history, "first_recorded_revision": first,
                "earlier_history_unavailable": first is None or first > 1,
                "next_after_revision": rows[limit - 1]["revision"] if len(rows) > limit else None,
                "provider_contacted": False}

    async def memberships(self, owner_id, area_id, *, after_id=None, limit=32):
        if type(limit) is not int or not 1 <= limit <= 64 or (after_id is not None and not isinstance(after_id, UUID)):
            raise ValueError("Invalid source membership page bounds")
        async with self.sources.transaction(owner_id) as connection:
            area = await self._read(connection, owner_id, area_id)
            if after_id is not None and not await connection.fetchval("""select true from jarvis_knowledge_sources
                    where owner_id=$1 and area_id=$2 and id=$3""", owner_id, area_id, after_id):
                raise SourceError("knowledge_area_source_cursor", "The source cursor is outside this area.", 400)
            # Project metadata only: do not copy private original content into this view.
            rows = await connection.fetch("""select source_id,revision,current_generation,source_status,availability,
                    source_type,trust,source_time,time_basis,recorded_at,context_is_current,ingestion_status
                from (""" + SOURCE_VERSION_SQL + """
                    and v.revision=s.current_revision and ($3::uuid is null or v.source_id>$3)
                ) scoped order by source_id limit $4""", owner_id, area_id, after_id, limit + 1)
            items = [{"source_id": str(row["source_id"]), "revision": row["revision"],
                "generation": row["current_generation"], "status": row["source_status"],
                "original_availability": row["availability"], "source_type": row["source_type"],
                "trust": row["trust"], "source_time": row["source_time"].isoformat() if row["source_time"] else None,
                "time_basis": row["time_basis"], "recorded_at": row["recorded_at"].isoformat(),
                "ordinary_content_available": area["status"] == "active" and row["source_status"] == "active"
                    and row["availability"] == "available" and row["context_is_current"],
                "ingestion_status": row["ingestion_status"] or "not_queued"} for row in rows[:limit]]
        return {"area": self._payload(area), "memberships": items,
                "next_after_id": str(rows[limit - 1]["source_id"]) if len(rows) > limit else None,
                "provider_contacted": False}

    @staticmethod
    async def _link_history(connection, owner_id, row, parent):
        # Metadata is a removable copy of the accepted request, not durable audit prose.
        await connection.execute("""insert into jarvis_context_payload_dependencies(
                owner_id,source_payload_id,dependent_payload_id)
            select owner_id,$2,metadata_payload_id from jarvis_knowledge_area_versions
            where owner_id=$1 and area_id=$3 and revision=$4 and metadata_payload_id is not null
            on conflict do nothing""", owner_id, parent["payload_id"], row["id"], row["revision"])

    @staticmethod
    def _operation_identity(request, operation, area_id=None):
        # One creation batch per finalized request. Provider call IDs and a
        # model's changed spelling cannot create another batch after a lost ACK.
        return digest({"request": request.parameters(), "operation": operation,
                       "area_id": str(area_id) if area_id is not None else None})

    async def _receipt(self, connection, owner_id, identity, parameters):
        row = await connection.fetchrow("""select r.event_id,p.availability,p.content
            from jarvis_context_ingress_receipts r
            join jarvis_context_events e on e.owner_id=r.owner_id and e.id=r.event_id
            join jarvis_context_payloads p on p.owner_id=e.owner_id and p.id=e.payload_id
            where r.owner_id=$1 and r.source_namespace=$2 and r.source_event_id=$3 and r.receipt_id=$3""",
            owner_id, AREA_OPERATIONS, identity)
        if row is None:
            return None
        if row["availability"] != "available" or row["content"] is None:
            raise SourceError("knowledge_area_receipt_unavailable",
                "The prior area operation's receipt is unavailable. Inspect current areas; do not repeat its effect.")
        receipt = json_object(row["content"])
        if receipt["arguments_digest"] != digest(parameters):
            raise SourceError("knowledge_area_request_conflict",
                "This request already accepted different area changes. Inspect current areas before a new user request.")
        return receipt | {"event_id": str(row["event_id"])}

    async def _record(self, connection, owner_id, request, parent, identity, parameters, rows, *, changed):
        if changed:
            for row in rows:
                await self._link_history(connection, owner_id, row, parent)
        payload = {"arguments_digest": digest(parameters), "changed": changed,
                   "applied_areas": [{"id": str(row["id"]), "revision": row["revision"]} for row in rows]}
        event = await ContextRepository(self.requests).append_in_transaction(connection, owner_id, ContextIngress(
            session_id=request.session_id, source_namespace=AREA_OPERATIONS, source_event_id=identity,
            receipt_id=identity, kind="tool_outcome", trust="tool_receipt", expected_revision=0,
            payload=payload, source_payload_ids=(parent["payload_id"],),
        ))
        return payload | {"event_id": str(event.id)}

    async def _result(self, connection, owner_id, receipt, *, replayed):
        # A receipt proves the accepted operation, not that its old metadata is
        # still current. Return current owned rows and the applied revisions apart.
        rows = [await self._read(connection, owner_id, UUID(item["id"])) for item in receipt["applied_areas"]]
        return {"areas": [self._payload(row) for row in rows], "replayed": replayed,
                "changed": receipt["changed"], "applied_areas": receipt["applied_areas"],
                "receipt_event_id": receipt["event_id"], "provider_contacted": False,
                "source_access_changed": False, "tasks_created_or_cancelled": False}

    async def create(self, owner_id, definitions, *, request, access_check=None):
        if not isinstance(definitions, (list, tuple)) or not 1 <= len(definitions) <= 8:
            raise ValueError("Create between one and eight explicitly requested areas together")
        definitions = [KnowledgeAreaDefinition.model_validate(item) for item in definitions]
        parameters = {"areas": [item.model_dump(mode="json") for item in definitions]}
        if len({item.model_dump_json() for item in definitions}) != len(definitions):
            raise SourceError("knowledge_area_duplicate_input", "Include each requested area once in the batch.", 422)
        async with self.sources.transaction(owner_id) as connection:
            await self.requests.lock_owner(connection, owner_id)
            parent = await self.requests._voice_source(connection, owner_id, request)
            if access_check is not None:
                access_check(owner_id)
            identity = self._operation_identity(request, "create")
            receipt = await self._receipt(connection, owner_id, identity, parameters)
            replayed = receipt is not None
            if receipt is None:
                rows = [await connection.fetchrow("""insert into jarvis_knowledge_areas(owner_id,name,domain,description)
                    values($1,$2,$3,$4) returning *""", owner_id, item.name, item.domain, item.description)
                    for item in definitions]
                receipt = await self._record(connection, owner_id, request, parent, identity, parameters, rows, changed=True)
            result = await self._result(connection, owner_id, receipt, replayed=replayed)
            if access_check is not None:
                access_check(owner_id)
            return result

    async def update(self, owner_id, area_id, definition, *, expected_revision, request, access_check=None):
        if type(expected_revision) is not int or expected_revision < 1:
            raise ValueError("An exact positive area revision is required")
        definition = KnowledgeAreaDefinition.model_validate(definition)
        parameters = {"area_id": str(area_id), "expected_revision": expected_revision,
                      "definition": definition.model_dump(mode="json")}
        async with self.sources.transaction(owner_id) as connection:
            await self.requests.lock_owner(connection, owner_id)
            parent = await self.requests._voice_source(connection, owner_id, request)
            if access_check is not None:
                access_check(owner_id)
            identity = self._operation_identity(request, "update", area_id)
            receipt = await self._receipt(connection, owner_id, identity, parameters)
            replayed = receipt is not None
            if receipt is None:
                row = await self._read(connection, owner_id, area_id, lock=True)
                if row["revision"] != expected_revision:
                    raise SourceError("knowledge_area_revision_conflict", "Read current area metadata before changing it.")
                changed = any(row[key] != value for key, value in definition.model_dump().items())
                if changed:
                    row = await connection.fetchrow("""update jarvis_knowledge_areas
                        set name=$3,domain=$4,description=$5,revision=revision+1
                        where owner_id=$1 and id=$2 returning *""", owner_id, area_id,
                        definition.name, definition.domain, definition.description)
                receipt = await self._record(connection, owner_id, request, parent, identity, parameters, [row], changed=changed)
            result = await self._result(connection, owner_id, receipt, replayed=replayed)
            if access_check is not None:
                access_check(owner_id)
            result["area"] = result.pop("areas")[0]
            return result

    async def set_status(self, owner_id, area_id, status, *, expected_revision, request, access_check=None):
        if status not in {"active", "archived"}:
            raise ValueError("Only reversible area visibility changes are supported")
        if type(expected_revision) is not int or expected_revision < 1:
            raise ValueError("An exact positive area revision is required")
        async with self.sources.transaction(owner_id) as connection:
            # Match CanonicalSourcePublication: source ownership, then task
            # ownership, then canonical request/stream and area row locks.
            await self.requests.lock_owner(connection, owner_id)
            parent = await self.requests._voice_source(connection, owner_id, request)
            if access_check is not None:
                access_check(owner_id)
            row = await self._read(connection, owner_id, area_id, lock=True)
            if row["revision"] != expected_revision:
                raise SourceError("knowledge_area_revision_conflict",
                    "The area changed. Read its current state; do not replay an old request with a guessed revision.")
            changed = row["status"] != status
            if changed:
                row = await connection.fetchrow("""update jarvis_knowledge_areas
                    set status=$3,revision=revision+1 where owner_id=$1 and id=$2 returning *""",
                    owner_id, area_id, status)
                await self._link_history(connection, owner_id, row, parent)
            if access_check is not None:
                access_check(owner_id)
            return {"area": self._payload(row), "changed": changed,
                    "originals_deleted": False, "tasks_created_or_cancelled": False,
                    "provider_contacted": False}
