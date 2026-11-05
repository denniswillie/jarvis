"""Fenced durable memory intents. Provider I/O never runs inside transactions."""
from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from app.knowledge.models import SourceError
from app.knowledge.sources import CURRENT_ORIGINAL_SQL, SourceRepository, SOURCE_VERSION_SQL


RETAINED_VERSION_SQL = SOURCE_VERSION_SQL + " and a.status in ('active','archived')"


class RetentionPaused(SourceError):
    def __init__(self):
        super().__init__("knowledge_area_archived", "New indexing is paused while this area is archived.")


class IngestionClaimLost(SourceError):
    def __init__(self):
        super().__init__("memory_ingestion_claim_lost", "Another attempt or source revision owns this work.")


@dataclass(frozen=True)
class IngestionClaim:
    id: UUID
    owner_id: str
    area_id: UUID
    source_id: UUID
    document_id: str
    bank_name: str
    operation: str
    generation: int
    fence: int


# A lease reserves capacity before network I/O. Unknown dispatched outcomes
# retain their reservation even after retries stop, until cleanup is proven.
# This measures Jarvis admissions, not the native worker's execution slots.
RESERVED_ADMISSIONS_SQL = """
    select count(*) from jarvis_memory_outbox reserved
    where reserved.owner_id=$1 and reserved.operation='retain'
    and (reserved.status in ('leased','accepted','outcome_unknown')
         or (reserved.dispatch_count>0 and reserved.status in ('pending','failed','cancelled')))
    and (reserved.status<>'failed' or reserved.last_error is distinct from 'hindsight_retention_failed')
    and not exists (
        select 1 from jarvis_memory_outbox cleanup
        where cleanup.owner_id=reserved.owner_id and cleanup.document_id=reserved.document_id
          and cleanup.operation='delete' and cleanup.status='done')
"""


class OutboxRepository:
    def __init__(self, sources: SourceRepository, *, lease_seconds=60, retry_seconds=3,
                 max_failures=8, max_in_flight=8):
        if (any(type(v) is not int or v < 1 for v in (lease_seconds, retry_seconds, max_failures, max_in_flight))
                or max_in_flight > 64):
            raise ValueError("Invalid memory worker bounds")
        self.sources = sources
        self.lease_seconds, self.retry_seconds, self.max_failures = lease_seconds, retry_seconds, max_failures
        self.max_in_flight = max_in_flight

    async def diagnostics(self, owner_id):
        """Owner-scoped queue metadata only; no originals or provider secrets."""
        async with self.sources.transaction(owner_id) as connection:
            rows = await connection.fetch(
                """select operation,status,count(*) as count,sum(failure_count) as failures,
                    min(created_at) as oldest_at from jarvis_memory_outbox
                    where owner_id=$1 group by operation,status order by operation,status""", owner_id,
            )
            reserved = await connection.fetchval(RESERVED_ADMISSIONS_SQL, owner_id)
        pending = {"pending", "leased", "accepted", "outcome_unknown"}
        oldest = [row["oldest_at"] for row in rows if row["status"] in pending]
        return {
            "counts": [{"operation": row["operation"], "status": row["status"], "count": row["count"]}
                       for row in rows],
            "pending_operations": sum(row["count"] for row in rows if row["status"] in pending),
            "recorded_failures": sum(row["failures"] for row in rows),
            "oldest_pending_at": min(oldest).isoformat() if oldest else None,
            "native_admissions": {
                "limit": self.max_in_flight, "reserved": reserved,
                "available": max(0, self.max_in_flight-reserved),
                "at_capacity": reserved >= self.max_in_flight,
                "meaning": "Conservative owner-scoped reservations; not native executing operations.",
            },
        }

    async def claim(self, owner_id, worker_id):
        if not isinstance(worker_id, str) or not worker_id.strip() or len(worker_id) > 200:
            raise ValueError("Invalid memory worker identity")
        async with self.sources.transaction(owner_id) as connection:
            # The owner lock serializes this count with every claim/publication.
            reserved = await connection.fetchval(RESERVED_ADMISSIONS_SQL, owner_id)
            row = await connection.fetchrow(
                """select o.*,v.area_id,v.source_id,b.bank_name from jarvis_memory_outbox o
                    join jarvis_knowledge_source_versions v on v.owner_id=o.owner_id and v.document_id=o.document_id
                    join jarvis_memory_banks b on b.owner_id=v.owner_id and b.id=v.bank_id
                    join jarvis_knowledge_areas a on a.owner_id=v.owner_id and a.id=v.area_id
                    where o.owner_id=$1 and o.next_attempt_at<=clock_timestamp()
                    and (o.operation='delete' or a.status='active' or o.dispatch_count>0 or o.status='leased')
                    and (o.status in ('pending','accepted','outcome_unknown')
                         or (o.status='leased' and o.lease_until<=clock_timestamp()))
                    and (o.operation='delete' or o.dispatch_count>0
                         or o.status in ('leased','accepted','outcome_unknown') or $2)
                    order by (o.operation='delete') desc,o.next_attempt_at,o.created_at,o.id
                    limit 1 for update of o skip locked""", owner_id, reserved < self.max_in_flight,
            )
            if row is None:
                return None
            fence = await connection.fetchval(
                """update jarvis_memory_outbox set status='leased',fence=fence+1,worker_id=$2,
                    lease_until=clock_timestamp()+make_interval(secs=>$3),updated_at=clock_timestamp()
                    where id=$1 returning fence""", row["id"], worker_id, self.lease_seconds,
            )
            return IngestionClaim(**{key: row[key] for key in IngestionClaim.__dataclass_fields__ if key != "fence"},
                                  fence=fence)

    async def _owned(self, connection, claim):
        row = await connection.fetchrow(
            """select * from jarvis_memory_outbox where owner_id=$1 and id=$2 and generation=$3
                and fence=$4 and status='leased' and lease_until>clock_timestamp() for update""",
            claim.owner_id, claim.id, claim.generation, claim.fence,
        )
        if row is None:
            raise IngestionClaimLost()
        return row

    async def retention_source(self, claim):
        """Private owned-worker access; archived originals are not public recall."""
        if claim.operation != "retain":
            raise ValueError("Not a retain claim")
        async with self.sources.transaction(claim.owner_id) as connection:
            intent = await self._owned(connection, claim)
            row = await connection.fetchrow(
                RETAINED_VERSION_SQL + " and v.document_id=$3", claim.owner_id, claim.area_id, claim.document_id,
            )
            source = self.sources._version(row)
            if source.generation != claim.generation:
                raise SourceError("knowledge_source_changed", "This retention intent is obsolete.")
            if row["area_status"] == "archived" and intent["dispatch_count"] == 0:
                raise RetentionPaused()
            return source

    async def prepare_retain(self, claim):
        if claim.operation != "retain":
            raise ValueError("Not a retain claim")
        async with self.sources.transaction(claim.owner_id) as connection:
            await self._owned(connection, claim)
            row = await connection.fetchrow(
                RETAINED_VERSION_SQL + " and v.document_id=$3", claim.owner_id, claim.area_id, claim.document_id,
            )
            source = self.sources._version(row)
            if source.generation != claim.generation:
                raise SourceError("knowledge_source_changed", "This retention intent is obsolete.")
            if row["area_status"] == "archived":
                raise RetentionPaused()
            # This is durable before the network write. Cleanup cannot infer
            # 'not sent' after a process exits between this write and the call.
            await connection.execute(
                "update jarvis_memory_outbox set dispatch_count=dispatch_count+1 where id=$1", claim.id,
            )
            return source

    async def skip_indexing(self, claim):
        """Cancel only a proven undispatched intent; never revoke its original."""
        if claim.operation != "retain":
            raise ValueError("Not a retain claim")
        async with self.sources.transaction(claim.owner_id) as connection:
            intent = await self._owned(connection, claim)
            row = await connection.fetchrow(
                RETAINED_VERSION_SQL + " and v.document_id=$3", claim.owner_id, claim.area_id, claim.document_id,
            )
            source = self.sources._version(row)
            if source.generation != claim.generation:
                raise SourceError("knowledge_source_changed", "This retention intent is obsolete.")
            if intent["dispatch_count"] != 0 or source.semantic_index_exclusion is None:
                return False
            await connection.execute(
                """update jarvis_memory_outbox set status='cancelled',last_error=$2,
                    worker_id=null,lease_until=null,updated_at=clock_timestamp() where id=$1""",
                claim.id, source.semantic_index_exclusion,
            )
            return True

    async def retention_state(self, claim):
        async with self.sources.transaction(claim.owner_id) as connection:
            await self._owned(connection, claim)
            row = await connection.fetchrow(
                """select id,status,dispatch_count from jarvis_memory_outbox
                    where owner_id=$1 and document_id=$2 and operation='retain'""",
                claim.owner_id, claim.document_id,
            )
            return dict(row) if row else None

    async def cleanup_operations(self, claim):
        if claim.operation != "delete":
            raise ValueError("Not a delete claim")
        async with self.sources.transaction(claim.owner_id) as connection:
            row = await self._owned(connection, claim)
            value = row["cleanup_operation_ids"]
            return tuple(value) if value is not None else None

    async def prepare_cleanup(self, claim, operation_ids):
        if claim.operation != "delete":
            raise ValueError("Not a delete claim")
        identities = tuple(UUID(str(item)) for item in operation_ids)
        if len(identities) > 64 or len(set(identities)) != len(identities):
            raise ValueError("Invalid memory cleanup inventory")
        async with self.sources.transaction(claim.owner_id) as connection:
            row = await self._owned(connection, claim)
            existing = row["cleanup_operation_ids"]
            if existing is not None:
                if tuple(existing) != identities:
                    raise SourceError("memory_cleanup_inventory_conflict", "Cleanup already has a different inventory.")
                return tuple(existing)
            retained = await connection.fetchrow(
                """select id,dispatch_count from jarvis_memory_outbox
                    where owner_id=$1 and document_id=$2 and operation='retain'""",
                claim.owner_id, claim.document_id,
            )
            dispatched = retained is not None and retained["dispatch_count"] > 0
            if (dispatched and (not identities or identities[0] != retained["id"])) or (not dispatched and identities):
                raise SourceError("memory_cleanup_root_mismatch", "Cleanup must match this source's retention intent.")
            await connection.execute(
                "update jarvis_memory_outbox set cleanup_operation_ids=$2::uuid[] where id=$1",
                claim.id, list(identities),
            )
            return identities

    async def deletion_allowed(self, claim):
        async with self.sources.transaction(claim.owner_id) as connection:
            await self._owned(connection, claim)
            return await connection.fetchval(
            f"""select not({CURRENT_ORIGINAL_SQL})
                    from jarvis_knowledge_source_versions v
                    join jarvis_knowledge_sources s on s.owner_id=v.owner_id and s.id=v.source_id
                    join jarvis_context_payloads p on p.owner_id=v.owner_id and p.id=v.payload_id
                    join jarvis_knowledge_areas a on a.owner_id=v.owner_id and a.id=v.area_id
                    where v.owner_id=$1 and v.document_id=$2""", claim.owner_id, claim.document_id,
            )

    async def settle(self, claim, status, *, reason=None):
        if status not in {"pending", "accepted", "searchable", "done", "outcome_unknown", "failed", "cancelled"}:
            raise ValueError("Invalid ingestion settlement")
        if status == "pending" and claim.operation != "retain":
            raise ValueError("Only a retention admission can be paused")
        async with self.sources.transaction(claim.owner_id) as connection:
            row = await self._owned(connection, claim)
            if status == "pending" and row["dispatch_count"]:
                status = "outcome_unknown"
            if claim.operation == "delete" and status == "done" and row["cleanup_operation_ids"] is None:
                raise SourceError("memory_cleanup_not_prepared", "Remote cleanup requires a durable operation inventory.")
            if claim.operation == "retain":
                row = await connection.fetchrow(
                    RETAINED_VERSION_SQL + " and v.document_id=$3", claim.owner_id, claim.area_id, claim.document_id,
                )
                try:
                    source = self.sources._version(row)
                    valid = source.generation == claim.generation
                except SourceError:
                    valid = False
                if not valid:
                    status, reason = "cancelled", "source_revoked"
                    await self.sources._retire_documents(
                        connection, claim.owner_id, claim.source_id,
                        await connection.fetchval(
                            "select generation from jarvis_knowledge_sources where owner_id=$1 and id=$2",
                            claim.owner_id, claim.source_id,
                        ),
                        # Retire only obsolete versions, never a newer current document.
                        except_document=await connection.fetchval(
                            """select v.document_id from jarvis_knowledge_sources s
                                join jarvis_knowledge_source_versions v on v.owner_id=s.owner_id
                                and v.source_id=s.id and v.revision=s.current_revision
                                where s.owner_id=$1 and s.id=$2 and s.status='active'""",
                            claim.owner_id, claim.source_id,
                        ),
                    )
            await connection.execute(
                """update jarvis_memory_outbox set status=$2,last_error=$3,worker_id=null,lease_until=null,
                    next_attempt_at=clock_timestamp()+make_interval(secs=>$4),
                    updated_at=clock_timestamp() where id=$1""",
                claim.id, status, reason, self.retry_seconds,
            )
            return status

    async def failure(self, claim, code, *, retryable):
        async with self.sources.transaction(claim.owner_id) as connection:
            row = await self._owned(connection, claim)
            failures = row["failure_count"] + 1
            status = "outcome_unknown" if retryable and failures < self.max_failures else "failed"
            await connection.execute(
                """update jarvis_memory_outbox set status=$2,failure_count=$3,last_error=$4,
                    next_attempt_at=clock_timestamp()+make_interval(secs=>$5),worker_id=null,lease_until=null,
                    updated_at=clock_timestamp() where id=$1""",
                claim.id, status, failures, code, min(300, self.retry_seconds * 2 ** min(failures, 6)),
            )
            return status
