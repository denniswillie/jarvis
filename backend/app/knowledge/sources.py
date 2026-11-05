"""Canonical source mapping and transactional ingestion intents.

No model inference, provider I/O or background loop lives here. Original content
remains in the existing removable payload store. Callers supply trusted owner
and area scope; provider bank names and document IDs are server-owned.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
import json
from uuid import UUID

from app.evidence.models import OwnedReference
from app.guard_diagnostics import guard_context, guard_wait_stage
from app.knowledge.models import CanonicalOwnerBusy, SourceError, SourcePublication, SourceVersion
from app.knowledge.indexing_policy import indexing_exclusion, TRANSPORT_RECEIPT_EXCLUDED
from app.knowledge.product_notes import NOTE_NAMESPACE, delete_note_original
from app.knowledge.product_files import FILE_NAMESPACE, delete_file_original
from app.knowledge.legacy_messages import delete_imported_message_original


# Shared by deletion admission and recovery. Area archival alone never permits
# removal of a current retained original.
CURRENT_ORIGINAL_SQL = "s.status='active' and s.current_revision=v.revision and p.availability='available'"


SOURCE_VERSION_SQL = """select v.*,b.bank_name,p.content,p.availability,a.status as area_status,
    s.current_revision,s.generation as current_generation,s.status as source_status,
    o.status as ingestion_status,o.last_error as ingestion_reason,
    s.namespace as source_namespace,s.source_key,
    ce.source_namespace as context_source_namespace,ce.source_event_id as context_source_event_id,
    ce.kind as context_kind,ce.trust as context_trust,
    (ce.payload_id=v.payload_id and cs.event_revision=ce.revision) as context_binding_matches,
    coalesce(ss.session_id,ce.session_id) as session_id,
    cs.event_id as context_event_id,
    cs.event_revision as context_event_revision,ce.recorded_at as context_recorded_at,
    ce.session_id as context_session_id,
    not exists(select 1 from jarvis_context_events newer where ce.id is not null
        and newer.owner_id=ce.owner_id and newer.source_namespace=ce.source_namespace
        and newer.source_event_id=ce.source_event_id and newer.revision>ce.revision) as context_is_current
    from jarvis_knowledge_source_versions v
    join jarvis_knowledge_sources s on s.owner_id=v.owner_id and s.id=v.source_id
    join jarvis_context_payloads p on p.owner_id=v.owner_id and p.id=v.payload_id
    join jarvis_memory_banks b on b.owner_id=v.owner_id and b.id=v.bank_id
    join jarvis_knowledge_areas a on a.owner_id=v.owner_id and a.id=v.area_id
    left join jarvis_knowledge_source_sessions ss on ss.owner_id=v.owner_id
        and ss.source_id=v.source_id and ss.source_revision=v.revision
    left join jarvis_knowledge_context_sources cs on cs.owner_id=v.owner_id
        and cs.source_id=v.source_id and cs.source_revision=v.revision
    left join jarvis_context_events ce on ce.owner_id=cs.owner_id and ce.id=cs.event_id
    left join jarvis_memory_outbox o on o.owner_id=v.owner_id
        and o.document_id=v.document_id and o.operation='retain'
    where v.owner_id=$1 and v.area_id=$2"""

# Ordinary reads and publication never acquire archived originals. Only the
# owned ingestion ledger uses the underlying projection for reconciliation.
VERSION_SQL = SOURCE_VERSION_SQL + " and a.status='active'"


class SourceRepository:
    def __init__(self, pool):
        self.pool = pool

    @asynccontextmanager
    async def transaction(self, owner_id, *, wait_for_owner=True):
        """Exclusive ownership for publication, correction and removal."""
        async with self._owner_transaction(owner_id, wait_for_owner=wait_for_owner, shared=False) as connection:
            yield connection

    @asynccontextmanager
    async def read_guard_transaction(self, owner_id, *, wait_for_owner=True):
        """Shared exclusion from mutations; callers must not upgrade or mutate.

        This is not a PostgreSQL read-only transaction: source guards take row
        SHARE locks. Retain the owner lock through the guarded external writer.
        """
        async with self._owner_transaction(owner_id, wait_for_owner=wait_for_owner, shared=True) as connection:
            yield connection

    @asynccontextmanager
    async def derivation_transaction(self, owner_id, *, wait_for_owner=True):
        """Append fresh derived copies under shared canonical-owner exclusion.

        Only audited payload/provenance materializers may use this mode. They
        must not change originals, owner sequences or state, or upgrade locks.
        Canonical publication/correction/removal must use transaction instead.
        """
        async with self._owner_transaction(owner_id, wait_for_owner=wait_for_owner, shared=True) as connection:
            yield connection

    @asynccontextmanager
    async def _owner_transaction(self, owner_id, *, wait_for_owner, shared):
        OwnedReference.validate_owner(owner_id)
        if type(wait_for_owner) is not bool:
            raise ValueError("Owner lock waiting must be explicit")
        async with guard_context(self.pool.acquire(), "pool_acquire") as connection:
            async with guard_context(connection.transaction(), "transaction_enter"):
                with guard_wait_stage("memory_owner_lock"):
                    suffix = "_shared" if shared else ""
                    if wait_for_owner:
                        await connection.execute(
                            f"select pg_advisory_xact_lock{suffix}(hashtextextended(current_schema() || ':jarvis-memory-owner:' || $1,0))",
                            owner_id,
                        )
                    elif await connection.fetchval(
                        f"select pg_try_advisory_xact_lock{suffix}(hashtextextended(current_schema() || ':jarvis-memory-owner:' || $1,0))",
                        owner_id,
                    ) is not True:
                        raise CanonicalOwnerBusy()
                yield connection

    async def _area(self, connection, owner_id, area_id, *, allow_archived=False):
        status = await connection.fetchval(
            "select status from jarvis_knowledge_areas where owner_id=$1 and id=$2 for share",
            owner_id, area_id,
        )
        if status != "active" and not (allow_archived and status == "archived"):
            raise SourceError("knowledge_area_unavailable", "The authorized knowledge area is unavailable.", 404)

    @staticmethod
    def _version(row, *, historical=False):
        if row is None:
            raise SourceError("knowledge_source_not_found", "No source exists in the authorized area.", 404)
        current = row["revision"] == row["current_revision"] and row["context_is_current"]
        if (row["source_status"] != "active" or row["availability"] != "available"
                or (not historical and (not current or not row["context_is_current"]))):
            raise SourceError("knowledge_source_unavailable", "This source revision is not available in the requested scope.")
        values = {key: row[key] for key in SourceVersion.model_fields
                  if key in row and key not in {"content", "is_current", "generation"}}
        values.update(
            content=json.loads(row["content"]) if isinstance(row["content"], str) else row["content"],
            generation=row["current_generation"], is_current=current,
            ingestion_status=row["ingestion_status"] or "not_queued",
        )
        values["semantic_index_exclusion"] = indexing_exclusion(row, values["content"])
        return SourceVersion.model_validate(values)

    async def _read(self, connection, owner_id, area_id, source_id, revision=None):
        row = await connection.fetchrow(
            VERSION_SQL + " and v.source_id=$3 and v.revision=coalesce($4::bigint,s.current_revision)",
            owner_id, area_id, source_id, revision,
        )
        return self._version(row, historical=revision is not None)

    async def _read_many(self, connection, owner_id, area_id, requests):
        """Resolve ordered current/explicit-history reads in one database trip.

        Keep missing rows so one unavailable source cannot silently disappear
        from a dependency set. The same version validator handles both paths.
        """
        requests = tuple(requests)
        if not requests:
            return []
        rows = await connection.fetch(
            """select versions.*
                from unnest($3::uuid[],$4::bigint[]) with ordinality
                    as requested(source_id,revision,position)
                left join lateral (""" + VERSION_SQL + """
                    and v.source_id=requested.source_id
                    and v.revision=coalesce(requested.revision,s.current_revision)
                ) as versions on true order by requested.position""",
            owner_id, area_id, [source_id for source_id, _ in requests],
            [revision for _, revision in requests],
        )
        return [self._version(row if row["source_id"] is not None else None,
                              historical=revision is not None)
                for row, (_, revision) in zip(rows, requests, strict=True)]

    async def read(self, owner_id, area_id, source_id, *, revision=None):
        """Historical access must be requested explicitly; deletion still wins."""
        if revision is not None and (type(revision) is not int or revision < 1):
            raise ValueError("invalid source revision")
        async with self.transaction(owner_id) as connection:
            return await self._read(connection, owner_id, area_id, source_id, revision)

    async def list_versions(self, owner_id, area_id, source_id, *, limit=8,
                            before_revision=None, expected_generation=None):
        """Bounded original-version discovery; no vendor or inferred history."""
        self._validate_version_window(limit, before_revision, expected_generation)
        async with self.transaction(owner_id) as connection:
            return await self.list_versions_in_transaction(
                connection, owner_id, area_id, source_id, limit=limit,
                before_revision=before_revision, expected_generation=expected_generation,
            )

    @staticmethod
    def _validate_version_window(limit, before_revision, expected_generation):
        if type(limit) is not int or not 1 <= limit <= 16:
            raise ValueError("limit must be between 1 and 16")
        for value in (before_revision, expected_generation):
            if value is not None and (type(value) is not int or value < 1):
                raise ValueError("Version cursors and generations must be positive integers")
        if before_revision is not None and expected_generation is None:
            raise ValueError("A continuation requires the returned expected_generation")

    async def list_versions_in_transaction(self, connection, owner_id, area_id, source_id, *, limit=8,
                                           before_revision=None, expected_generation=None):
        """Read within the caller's owned guard, without acquiring another lock."""
        self._validate_version_window(limit, before_revision, expected_generation)
        current = await self._read(connection, owner_id, area_id, source_id)
        if expected_generation is not None and current.generation != expected_generation:
            raise SourceError("knowledge_source_changed", "The source changed; restart version discovery.")
        rows = await connection.fetch(
            VERSION_SQL + " and v.source_id=$3 and ($4::bigint is null or v.revision<$4)"
            " order by v.revision desc limit $5",
            owner_id, area_id, source_id, before_revision, limit + 1,
        )
        versions, unavailable = [], 0
        for row in rows[:limit]:
            try:
                metadata = self._version(row, historical=True).original_evidence()
            except SourceError:
                unavailable += 1
                continue
            metadata.pop("content")
            versions.append(metadata)
        return {
            "memory_id": str(source_id), "current_revision": current.revision,
            "expected_generation": current.generation, "versions": versions,
            "unavailable_revisions": unavailable,
            "next_before_revision": rows[limit - 1]["revision"] if len(rows) > limit else None,
        }

    async def resolve_document(self, owner_id, area_id, document_id, *, historical=False):
        """Expand a vendor-returned ID only through this trusted canonical map."""
        async with self.transaction(owner_id) as connection:
            return await self.resolve_document_in_transaction(
                connection, owner_id, area_id, document_id, historical=historical,
            )

    async def resolve_document_in_transaction(self, connection, owner_id, area_id, document_id, *, historical=False):
        """Resolve under an existing owned read/write guard; deletion still wins."""
        row = await connection.fetchrow(
            VERSION_SQL + " and v.document_id=$3", owner_id, area_id, document_id,
        )
        return self._version(row, historical=historical)

    async def revalidate(self, evidence: SourceVersion):
        fresh = await self.resolve_document(
            evidence.owner_id, evidence.area_id, evidence.document_id,
            historical=not evidence.is_current,
        )
        if fresh.generation != evidence.generation or fresh.content_digest != evidence.content_digest:
            raise SourceError("knowledge_source_changed", "The source changed while this operation was running.")
        return fresh

    async def publish(self, owner_id: str, request: SourcePublication) -> SourceVersion:
        """Publish existing original evidence and its retain intent atomically."""
        if request.namespace.startswith(("context:", "project:")) or request.namespace in {NOTE_NAMESPACE, FILE_NAMESPACE}:
            raise SourceError("reserved_source_namespace", "Canonical context and product publication require their trusted ingress adapters.", 403)
        async with self.transaction(owner_id) as connection:
            return await self.publish_in_transaction(connection, owner_id, request)

    async def publish_in_transaction(self, connection, owner_id: str, request: SourcePublication) -> SourceVersion:
        """Internal composition seam; caller holds this owner's source transaction."""
        if not connection.is_in_transaction():
            raise RuntimeError("Source publication requires an existing transaction")
        await self._area(connection, owner_id, request.area_id)
        if request.session_id is not None:
            session = await connection.fetchval(
                "select id from jarvis_sessions where owner_id=$1 and id=$2 for key share",
                owner_id, request.session_id,
            )
            if session is None:
                raise SourceError("knowledge_session_unavailable", "The source session is outside the owner scope.", 404)
        # Lock original evidence before its mapping, matching removal-trigger order.
        payload = await connection.fetchrow(
            "select id,content_digest,availability from jarvis_context_payloads where owner_id=$1 and id=$2 for share",
            owner_id, request.payload_id,
        )
        if payload is None or payload["availability"] != "available":
            raise SourceError("knowledge_payload_unavailable", "An available original payload is required.")
        source = await connection.fetchrow(
            """select * from jarvis_knowledge_sources where owner_id=$1 and area_id=$2
                and namespace=$3 and source_key=$4 for update""",
            owner_id, request.area_id, request.namespace, request.source_key,
        )
        if source is not None and source["status"] == "deleted":
            raise SourceError("knowledge_source_deleted", "Deleted source identity cannot be republished.")
        if source is not None and source["status"] == "forgotten":
            raise SourceError("knowledge_source_forgotten", "This source is suppressed; automatic publication cannot unforget it.")
        if source is not None and source["status"] == "active":
            prior = await self._read(connection, owner_id, request.area_id, source["id"])
            if all(getattr(prior, key) == getattr(request, key) for key in
                   ("payload_id", "source_type", "trust", "source_time", "time_basis", "session_id")):
                return prior
        revision = source["current_revision"] if source else 0
        if request.expected_revision != revision:
            raise SourceError("knowledge_source_revision_conflict", "Publish against the current source revision.")
        generation = source["generation"] + 1 if source else 1
        # Read an existing bank without a no-op UPDATE, or use the inserted
        # RETURNING row (not a same-statement reread of the newly inserted row).
        # Publication still owns the same exclusive owner transaction.
        bank_id = await connection.fetchval(
            """with inserted as (
                insert into jarvis_memory_banks(owner_id,area_id) values($1,$2)
                on conflict(owner_id,area_id) do nothing returning id
            ) select id from inserted
            union all select id from jarvis_memory_banks where owner_id=$1 and area_id=$2
            limit 1""", owner_id, request.area_id,
        )
        if bank_id is None:
            raise SourceError("knowledge_bank_unavailable", "The canonical memory bank could not be resolved.")
        if source is None:
            source_id = await connection.fetchval(
                """insert into jarvis_knowledge_sources(owner_id,area_id,namespace,source_key,current_revision,generation)
                    values($1,$2,$3,$4,1,1) returning id""",
                owner_id, request.area_id, request.namespace, request.source_key,
            )
        else:
            source_id = source["id"]
        document_id = await connection.fetchval(
            """insert into jarvis_knowledge_source_versions(owner_id,area_id,source_id,revision,generation,
                payload_id,bank_id,content_digest,source_type,trust,source_time,time_basis)
                values($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12) returning document_id""",
            owner_id, request.area_id, source_id, revision + 1, generation, request.payload_id,
            bank_id, payload["content_digest"], request.source_type, request.trust.value,
            request.source_time, request.time_basis,
        )
        if request.session_id is not None:
            await connection.execute(
                """insert into jarvis_knowledge_source_sessions(owner_id,source_id,source_revision,session_id)
                    values($1,$2,$3,$4)""",
                owner_id, source_id, revision + 1, request.session_id,
            )
        if source is not None:
            await connection.execute(
                """update jarvis_knowledge_sources set current_revision=$3,generation=$4,status='active'
                    where owner_id=$1 and id=$2""", owner_id, source_id, revision + 1, generation,
            )
            await self._retire_documents(connection, owner_id, source_id, generation, except_document=document_id)
        await connection.execute(
            """insert into jarvis_memory_outbox(owner_id,document_id,generation,operation)
                values($1,$2,$3,'retain')""", owner_id, document_id, generation,
        )
        return await self._read(connection, owner_id, request.area_id, source_id)

    async def _retire_documents(self, connection, owner_id, source_id, generation, *, except_document=None,
                                reschedule_existing=True):
        await connection.execute(
            """update jarvis_memory_outbox o set status='cancelled',updated_at=clock_timestamp()
                from jarvis_knowledge_source_versions v where v.owner_id=$1 and v.source_id=$2
                and v.document_id is distinct from $3 and o.owner_id=v.owner_id
                and o.document_id=v.document_id and o.operation='retain' and o.status='pending'""",
            owner_id, source_id, except_document,
        )
        await connection.execute(
            """insert into jarvis_memory_outbox(owner_id,document_id,generation,operation)
                select owner_id,document_id,$3,'delete' from jarvis_knowledge_source_versions
                where owner_id=$1 and source_id=$2 and document_id is distinct from $4
                on conflict(owner_id,document_id,operation) do update
                    set generation=excluded.generation,status='pending',updated_at=clock_timestamp()
                    where $5""",
            owner_id, source_id, generation, except_document, reschedule_existing,
        )

    async def delete_source(self, owner_id, area_id, source_id, *, expected_revision):
        """Forget originals, not just vendor summaries. Only trusted callers authorize this."""
        if type(expected_revision) is not int or expected_revision < 1:
            raise ValueError("invalid source revision")
        async with self.transaction(owner_id) as connection:
            return await self.delete_source_in_transaction(
                connection, owner_id, area_id, source_id, expected_revision=expected_revision,
            )

    async def delete_source_in_transaction(self, connection, owner_id, area_id, source_id, *, expected_revision):
        """Internal seam; caller holds this owner's source transaction and authorizes the scope."""
        if not connection.is_in_transaction():
            raise RuntimeError("Source deletion requires an existing transaction")
        OwnedReference.validate_owner(owner_id)
        if type(expected_revision) is not int or expected_revision < 1:
            raise ValueError("invalid source revision")
        await self._area(connection, owner_id, area_id, allow_archived=True)
        # Payload removal also revokes other views of these same originals.
        await connection.fetch(
            """select p.id from jarvis_context_payloads p where p.owner_id=$1 and exists(
                select 1 from jarvis_knowledge_source_versions v where v.owner_id=p.owner_id
                and v.area_id=$2 and v.source_id=$3 and v.payload_id=p.id) order by p.id for update""",
            owner_id, area_id, source_id,
        )
        source = await connection.fetchrow(
            "select * from jarvis_knowledge_sources where owner_id=$1 and area_id=$2 and id=$3 for update",
            owner_id, area_id, source_id,
        )
        if source is None:
            raise SourceError("knowledge_source_not_found", "No source exists in the authorized area.", 404)
        if source["current_revision"] != expected_revision:
            raise SourceError("knowledge_source_revision_conflict", "Delete against the current source revision.")
        shared = await connection.fetchval(
            """select exists(select 1 from jarvis_knowledge_source_versions target
                join jarvis_context_payloads p on p.owner_id=target.owner_id and p.id=target.payload_id
                join jarvis_knowledge_source_versions other on other.owner_id=p.owner_id
                    and other.payload_id=p.id and other.source_id<>target.source_id
                join jarvis_knowledge_sources retained on retained.owner_id=other.owner_id
                    and retained.id=other.source_id
                where target.owner_id=$1 and target.source_id=$2
                    and p.availability<>'deleted' and retained.status<>'deleted')""",
            owner_id, source_id,
        )
        if shared:
            raise SourceError(
                "knowledge_source_shared_original",
                "Another retained source references this original; single-source deletion would exceed its scope.",
            )
        return await self._delete_source_originals(connection, owner_id, source)

    async def delete_sources_in_transaction(self, connection, owner_id, targets):
        """Delete an explicitly authorized bounded set, without expanding shared scope.

        Targets are trusted (area_id, source_id, expected_revision) tuples. The
        caller owns the source transaction and authorizes the whole set first.
        """
        if not connection.is_in_transaction():
            raise RuntimeError("Source deletion requires an existing transaction")
        OwnedReference.validate_owner(owner_id)
        if not isinstance(targets, (list, tuple)) or not 2 <= len(targets) <= 32:
            raise ValueError("An explicit group of two to 32 sources is required")
        for area_id, source_id, revision in targets:
            if not isinstance(area_id, UUID) or not isinstance(source_id, UUID) or type(revision) is not int or revision < 1:
                raise ValueError("Exact owned source identities and positive revisions are required")
        identities = sorted(source_id for _, source_id, _ in targets)
        if len(set(identities)) != len(identities):
            raise ValueError("Source deletion targets must be unique")
        for area_id in sorted({area_id for area_id, _, _ in targets}):
            await self._area(connection, owner_id, area_id, allow_archived=True)
        # Lock all originals before source rows, matching single-source deletion.
        # Validate the complete group before removing any member's bytes.
        await connection.fetch("""select p.id from jarvis_context_payloads p
            where p.owner_id=$1 and exists(select 1 from jarvis_knowledge_source_versions v
                where v.owner_id=p.owner_id and v.payload_id=p.id and v.source_id=any($2::uuid[]))
            order by p.id for update""", owner_id, identities)
        rows = await connection.fetch("""select * from jarvis_knowledge_sources
            where owner_id=$1 and id=any($2::uuid[]) order by id for update""", owner_id, identities)
        sources = {row["id"]: row for row in rows}
        for area_id, source_id, revision in targets:
            source = sources.get(source_id)
            if source is None or source["area_id"] != area_id:
                raise SourceError("knowledge_source_not_found", "No source exists in the authorized area.", 404)
            if source["current_revision"] != revision:
                raise SourceError("knowledge_source_revision_conflict", "Delete against every current source revision.")
        shared = await connection.fetchval("""select exists(
            select 1 from jarvis_knowledge_source_versions target
            join jarvis_context_payloads p on p.owner_id=target.owner_id and p.id=target.payload_id
            join jarvis_knowledge_source_versions other on other.owner_id=p.owner_id and other.payload_id=p.id
            join jarvis_knowledge_sources retained on retained.owner_id=other.owner_id and retained.id=other.source_id
            where target.owner_id=$1 and target.source_id=any($2::uuid[])
                and not(other.source_id=any($2::uuid[])) and p.availability<>'deleted'
                and retained.status<>'deleted')""", owner_id, identities)
        if shared:
            raise SourceError("knowledge_source_shared_original",
                "A retained source outside the authorized group shares an original. Inspect the shared scope; nothing was deleted.")
        return [await self._delete_source_originals(connection, owner_id, sources[source_id])
                for source_id in identities]

    async def _delete_source_originals(self, connection, owner_id, source):
        """Removal after exact revision and complete shared-scope admission."""
        source_id = source["id"]
        await delete_note_original(connection, owner_id, source)
        legacy_message = await delete_imported_message_original(connection, owner_id, source)
        if source["status"] != "deleted":
            await connection.execute(
                """update jarvis_knowledge_sources set status='deleted',generation=generation+1
                    where owner_id=$1 and id=$2 and status<>'deleted'""", owner_id, source_id,
            )
        # Transition the source first: file-manifest removal also revokes its dependent original.
        await delete_file_original(connection, owner_id, source)
        generation = await connection.fetchval(
            "select generation from jarvis_knowledge_sources where owner_id=$1 and id=$2", owner_id, source_id,
        )
        await connection.execute(
            """update jarvis_context_payloads p set availability='deleted',content=null
                where p.owner_id=$1 and p.availability<>'deleted' and exists(
                select 1 from jarvis_knowledge_source_versions v where v.owner_id=p.owner_id
                and v.source_id=$2 and v.payload_id=p.id)""", owner_id, source_id,
        )
        # A repeated request is not a retry of failed or completed provider
        # work. Late dispatched retains still reschedule through settle().
        await self._retire_documents(connection, owner_id, source_id, generation,
                                     reschedule_existing=False)
        result = {"source_id": str(source_id), "generation": generation,
                  "read_barrier": "deleted", "vendor_cleanup": "pending"}
        if legacy_message is not None:
            result["legacy_message_original"] = legacy_message
        return result

    async def current_snapshot(self, owner_id, area_id, *, limit=8):
        """One statement snapshot; aggregate metadata without copying all payloads."""
        if type(limit) is not int or not 1 <= limit <= 32:
            raise ValueError("Invalid current-source limit")
        async with self.read_guard_transaction(owner_id) as connection:
            await self._area(connection, owner_id, area_id)
            row = await connection.fetchrow(
                f"""with totals as (
                    select count(*) as total_sources,
                        count(*) filter(where s.status='active' and p.availability='available') as available_sources,
                        count(*) filter(where s.status='active' and p.availability='available'
                            and o.status is distinct from 'searchable') as not_searchable_sources,
                        count(*) filter(where s.status='active' and p.availability='available'
                            and o.status='cancelled' and o.last_error='{TRANSPORT_RECEIPT_EXCLUDED}'
                            and o.dispatch_count=0) as policy_excluded_sources,
                        count(*) filter(where s.status='active' and o.status='failed') as failed_ingestion_sources,
                        count(*) filter(where s.status='active' and o.status='outcome_unknown') as uncertain_ingestion_sources
                    from jarvis_knowledge_sources s
                    join jarvis_knowledge_source_versions v on v.owner_id=s.owner_id
                        and v.source_id=s.id and v.revision=s.current_revision
                    join jarvis_context_payloads p on p.owner_id=v.owner_id and p.id=v.payload_id
                    left join jarvis_memory_outbox o on o.owner_id=v.owner_id
                        and o.document_id=v.document_id and o.operation='retain'
                    where s.owner_id=$1 and s.area_id=$2
                ), recent as (""" + VERSION_SQL + """
                    and v.revision=s.current_revision and s.status='active' and p.availability='available'
                    order by v.recorded_at desc,v.source_id limit $3
                ) select totals.*,clock_timestamp() as observed_at,
                    coalesce((select jsonb_agg(to_jsonb(recent)) from recent),'[]'::jsonb) as records from totals""",
                owner_id, area_id, limit,
            )
            records = json.loads(row["records"]) if isinstance(row["records"], str) else row["records"]
            result = dict(row)
            result.pop("records")
            result["sources"] = []
            for record in records:
                try:
                    result["sources"].append(self._version(record))
                except SourceError:
                    # A mapping whose canonical source is superseded cannot
                    # supply current evidence, even before its consumer catches up.
                    pass
            return result
