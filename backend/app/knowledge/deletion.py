"""Scoped deletion receipts over existing originals and durable cleanup intents.

This is a trusted application seam, not a model authorization boundary. Callers
must authorize the owner/area/source action. No provider I/O, independent queue,
content copy, or claim of universal erasure is introduced here.
"""
from __future__ import annotations

from uuid import UUID

from app.knowledge.models import SourceError
from app.knowledge.legacy_messages import UNRESOLVED, inspect_imported_message_original
from app.knowledge.sources import CURRENT_ORIGINAL_SQL


UNVERIFIED_COPY_CLASSES = (
    "downstream_task_context_and_artifact_copies",
    "unlinked_legacy_records_and_caches",
    "sandbox_and_client_copies",
    "provider_internal_derivatives_logs_and_backups",
    "database_backups",
    "external_sources_and_delivered_messages",
)


class SourceDeletionService:
    def __init__(self, sources):
        self.sources = sources

    @staticmethod
    def _bound(value, *, minimum, maximum=None):
        if type(value) is not int or value < minimum or (maximum is not None and value > maximum):
            raise ValueError("Invalid deletion inspection/retry bound")

    @staticmethod
    async def _source(connection, owner_id, area_id, source_id):
        # Unlike a live read, cleanup inspection remains available after an area
        # is disabled. Scope comes from the trusted caller and exact owned row.
        source = await connection.fetchrow(
            """select id,current_revision,generation,status,updated_at,namespace,source_key from jarvis_knowledge_sources
                where owner_id=$1 and area_id=$2 and id=$3 for update""",
            owner_id, area_id, source_id,
        )
        if source is None:
            raise SourceError("knowledge_source_not_found", "No source exists in the authorized area.", 404)
        return source

    async def request(self, owner_id, area_id, source_id, *, expected_revision):
        await self.sources.delete_source(
            owner_id, area_id, source_id, expected_revision=expected_revision,
        )
        return await self.inspect(owner_id, area_id, source_id)

    async def inspect(self, owner_id, area_id, source_id, *, after_source_id=None, limit=32):
        """Current durable metadata, not a fresh audit of third-party storage."""
        self._bound(limit, minimum=1, maximum=128)
        if after_source_id is not None and not isinstance(after_source_id, UUID):
            raise ValueError("An exact shared-source cursor is required")
        async with self.sources.transaction(owner_id) as connection:
            source = await self._source(connection, owner_id, area_id, source_id)
            local = await connection.fetchrow(
                """select count(*) as versions,count(distinct p.id) as payloads,
                    count(distinct p.id) filter(where p.content is not null) as remaining_content_payloads,
                    count(distinct p.id) filter(where p.availability<>'deleted') as not_deleted_payloads
                    from jarvis_knowledge_source_versions v
                    join jarvis_context_payloads p on p.owner_id=v.owner_id and p.id=v.payload_id
                    where v.owner_id=$1 and v.source_id=$2""",
                owner_id, source_id,
            )
            provider = await connection.fetchrow(
                """select count(*) as documents,
                    count(*) filter(where o.status='done' and o.cleanup_operation_ids is not null) as cleanup_recorded,
                    count(*) filter(where o.status in ('pending','leased','accepted','outcome_unknown')) as pending,
                    count(*) filter(where o.status='failed') as failed,
                    count(*) filter(where o.id is null or o.status in ('searchable','cancelled')
                        or (o.status='done' and o.cleanup_operation_ids is null)) as unverified
                    from jarvis_knowledge_source_versions v
                    left join jarvis_memory_outbox o on o.owner_id=v.owner_id
                        and o.document_id=v.document_id and o.operation='delete'
                    where v.owner_id=$1 and v.source_id=$2""",
                owner_id, source_id,
            )
            shared = await connection.fetch("""select distinct retained.id as source_id,
                    retained.area_id,retained.current_revision as revision,retained.status as state
                from jarvis_knowledge_source_versions target
                join jarvis_context_payloads p on p.owner_id=target.owner_id and p.id=target.payload_id
                join jarvis_knowledge_source_versions other on other.owner_id=p.owner_id
                    and other.payload_id=p.id and other.source_id<>target.source_id
                join jarvis_knowledge_sources retained on retained.owner_id=other.owner_id and retained.id=other.source_id
                where target.owner_id=$1 and target.source_id=$2
                    and p.availability<>'deleted' and retained.status<>'deleted'
                    and ($3::uuid is null or retained.id>$3)
                order by source_id limit $4""", owner_id, source_id, after_source_id, limit + 1)
            shared_items = [{"source_id": str(row["source_id"]), "area_id": str(row["area_id"]),
                             "revision": row["revision"], "state": row["state"]} for row in shared[:limit]]
            legacy_message = await inspect_imported_message_original(connection, owner_id, source)
            observed_at = await connection.fetchval("select clock_timestamp()")
        local, provider = dict(local), dict(provider)
        if source["status"] != "deleted":
            state = "not_requested"
        elif (local["remaining_content_payloads"] or local["not_deleted_payloads"]
              or provider["failed"] or provider["unverified"] or not local["versions"]
              or (legacy_message is not None and legacy_message["state"] in UNRESOLVED)):
            state = "requires_attention"
        elif provider["pending"]:
            state = "pending"
        else:
            state = "scoped_cleanup_recorded"
        return {
            "source_id": str(source["id"]), "revision": source["current_revision"],
            "generation": source["generation"], "read_barrier": source["status"],
            "state": state, "observed_at": observed_at.isoformat(),
            "source_deleted_at": source["updated_at"].isoformat() if source["status"] == "deleted" else None,
            "local_originals": local, "provider": provider,
            "legacy_message_original": legacy_message,
            "shared_original_sources": {"items": shared_items,
                "next_after_source_id": shared_items[-1]["source_id"] if len(shared) > limit else None,
                "meaning": "Other retained sources sharing current or historical originals. Inspection grants no deletion authority."},
            "scope": "source_originals_and_recorded_hindsight_cleanup",
            "full_erasure_verified": False,
            "unverified_copy_classes": list(UNVERIFIED_COPY_CLASSES),
        }

    async def documents(self, owner_id, area_id, source_id, *, after_revision=0, limit=32):
        """Page the immutable original/document manifest without returning text."""
        self._bound(after_revision, minimum=0)
        self._bound(limit, minimum=1, maximum=128)
        async with self.sources.transaction(owner_id) as connection:
            source = await self._source(connection, owner_id, area_id, source_id)
            rows = await connection.fetch(
                """select v.revision,v.document_id,v.payload_id,p.availability as original_state,
                    r.status as retain_status,d.id as cleanup_operation_id,d.status as cleanup_status,
                    d.failure_count,cardinality(d.cleanup_operation_ids) as native_operation_count
                    from jarvis_knowledge_source_versions v
                    join jarvis_context_payloads p on p.owner_id=v.owner_id and p.id=v.payload_id
                    left join jarvis_memory_outbox r on r.owner_id=v.owner_id
                        and r.document_id=v.document_id and r.operation='retain'
                    left join jarvis_memory_outbox d on d.owner_id=v.owner_id
                        and d.document_id=v.document_id and d.operation='delete'
                    where v.owner_id=$1 and v.source_id=$2 and v.revision>$3
                    order by v.revision limit $4""",
                owner_id, source_id, after_revision, limit + 1,
            )
        items = []
        for row in rows[:limit]:
            item = dict(row)
            item["payload_id"] = str(item["payload_id"])
            if item["cleanup_operation_id"] is not None:
                item["cleanup_operation_id"] = str(item["cleanup_operation_id"])
            items.append(item)
        return {
            "source_id": str(source["id"]), "generation": source["generation"],
            "items": items,
            "next_after_revision": items[-1]["revision"] if len(rows) > limit else None,
        }

    async def retry_failed(self, owner_id, area_id, source_id, *, expected_generation, limit=32):
        """Compatibility seam: physical deletion is still required here."""
        self._bound(expected_generation, minimum=1)
        self._bound(limit, minimum=1, maximum=128)
        async with self.sources.transaction(owner_id) as connection:
            source = await self._source(connection, owner_id, area_id, source_id)
            if source["status"] != "deleted":
                raise SourceError("knowledge_source_not_deleted", "A deleted source is required for cleanup retry.")
            return await self._retry_failed(connection, owner_id, source, expected_generation, limit)

    async def retry_failed_cleanup(self, owner_id, area_id, source_id, *, expected_generation, limit=32):
        """Trusted recovery for obsolete, unavailable, forgotten or deleted originals."""
        async with self.sources.transaction(owner_id) as connection:
            return await self.retry_failed_cleanup_in_transaction(
                connection, owner_id, area_id, source_id, expected_generation=expected_generation, limit=limit)

    async def retry_failed_cleanup_in_transaction(
        self, connection, owner_id, area_id, source_id, *, expected_generation, limit=32,
    ):
        """Compose authorized recovery with the caller's source-owner transaction."""
        if not connection.is_in_transaction():
            raise RuntimeError("Source cleanup retry requires an owned transaction")
        self._bound(expected_generation, minimum=1)
        self._bound(limit, minimum=1, maximum=128)
        source = await self._source(connection, owner_id, area_id, source_id)
        return await self._retry_failed(connection, owner_id, source, expected_generation, limit)

    async def _retry_failed(self, connection, owner_id, source, expected_generation, limit):
        if source["generation"] != expected_generation:
            raise SourceError("knowledge_source_generation_conflict",
                              "Inspect the current source generation before retrying cleanup.")
        # Use the worker's original-preservation rule, not recall eligibility:
        # archiving an area does not authorize deletion of its current originals.
        joins = """from jarvis_memory_outbox o
            join jarvis_knowledge_source_versions v on v.owner_id=o.owner_id and v.document_id=o.document_id
            join jarvis_knowledge_sources s on s.owner_id=v.owner_id and s.id=v.source_id
            join jarvis_context_payloads p on p.owner_id=v.owner_id and p.id=v.payload_id
            where v.owner_id=$1 and v.source_id=$2 and o.operation='delete' and o.status='failed'"""
        rows = await connection.fetch(
            f"""with candidates as (
                select o.id {joins} and not({CURRENT_ORIGINAL_SQL})
                order by v.revision limit $3 for update of o skip locked
            ) update jarvis_memory_outbox o set status='pending',generation=$4,fence=o.fence+1,
                failure_count=0,last_error=null,worker_id=null,lease_until=null,
                next_attempt_at=clock_timestamp(),updated_at=clock_timestamp()
                from candidates c where o.id=c.id returning o.id""",
            owner_id, source["id"], limit, source["generation"])
        remaining = await connection.fetchrow(
            f"""select count(*) as remaining_failed,
                count(*) filter(where not({CURRENT_ORIGINAL_SQL})) as remaining_retryable,
                count(*) filter(where {CURRENT_ORIGINAL_SQL}) as blocked_current_originals {joins}""",
            owner_id, source["id"])
        return {
            "source_id": str(source["id"]), "revision": source["current_revision"],
            "generation": source["generation"], "state": source["status"],
            "requeued": len(rows), **dict(remaining), "provider_contacted": False,
            "cleanup_completion_verified": False,
        }
