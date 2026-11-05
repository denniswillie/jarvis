"""Explicit bounded retention over canonical sources and existing deletion intents."""
from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime

from app.knowledge.models import SourceError
from app.tasks.models import digest


TOMBSTONE_RETENTION = {
    "policy": "retain_suppression_identity",
    "expires_at": None,
    "reason": "Suppress re-ingestion and reconcile existing operations across retries and supported restores.",
    "original_content_required": False,
    "automatic_purge": False,
    "release": "Only a deliberate scoped reset/reimport, not elapsed time or a provider delete acknowledgment.",
}


class SourceRetentionService:
    def __init__(self, sources, requests, *, clock: Callable[[], datetime] | None = None):
        if sources.pool is not requests.pool:
            raise ValueError("Retention and canonical requests must share a database")
        if clock is not None and not callable(clock):
            raise ValueError("Retention clock must be callable")
        self.sources, self.requests = sources, requests
        # Internal domain-time injection for calendar replay. The normal path
        # continues to use the database clock; no tool/request can supply this.
        # Authorization, leases, transport deadlines and audit time stay real.
        self._clock = clock

    def _clock_time(self):
        if self._clock is None:
            return None
        value = self._clock()
        if not isinstance(value, datetime):
            raise ValueError("Retention clock must return a timezone-aware datetime")
        return self._deadline(value)

    @staticmethod
    def _positive(value, *, zero=False):
        if type(value) is not int or value < (0 if zero else 1):
            raise ValueError("Exact integer revisions are required")

    @staticmethod
    def _deadline(value):
        if value is None:
            return None
        if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("Retention expiry must include a time zone")
        return value.astimezone(UTC)

    async def _snapshot(self, db, owner_id, area_id, source_id):
        source = await db.fetchrow("""select id,status,current_revision,generation
            from jarvis_knowledge_sources where owner_id=$1 and area_id=$2 and id=$3""",
            owner_id, area_id, source_id)
        if source is None:
            raise SourceError("knowledge_source_not_found", "No source exists in the authorized area.", 404)
        policy = await db.fetchrow("""select policy_revision,source_revision,source_generation,
            expires_at,status,reason,next_check_at,checked_at,updated_at from jarvis_source_retention
            where owner_id=$1 and area_id=$2 and source_id=$3""", owner_id, area_id, source_id)
        values = ({key: value.isoformat() if isinstance(value, datetime) else value
                   for key, value in dict(policy).items()} if policy else {
                       "policy_revision": 0, "status": "not_scheduled", "expires_at": None,
                       "reason": "No automatic source deletion without an explicit policy."})
        return {"source_id": str(source_id), "area_id": str(area_id),
                "source_revision": source["current_revision"], "source_generation": source["generation"],
                "source_state": source["status"], "retention": values,
                "tombstone_retention": dict(TOMBSTONE_RETENTION),
                "full_erasure_verified": False,
                "meaning": "Source retention metadata, not a provider/storage cleanup audit. "
                           "Expiry preserves available dependent evidence and never silently expands to shared originals."}

    async def inspect(self, owner_id, area_id, source_id):
        async with self.sources.transaction(owner_id) as db:
            return await self._snapshot(db, owner_id, area_id, source_id)

    async def set_policy(self, owner_id, area_id, source_id, *, expected_source_revision,
                         expected_source_generation, expected_policy_revision, expires_at,
                         request, access_check=None):
        self._positive(expected_source_revision)
        self._positive(expected_source_generation)
        self._positive(expected_policy_revision, zero=True)
        expires_at = self._deadline(expires_at)
        parameters = digest({"area_id": str(area_id), "source_id": str(source_id),
            "source_revision": expected_source_revision, "source_generation": expected_source_generation,
            "policy_revision": expected_policy_revision,
            "expires_at": expires_at.isoformat() if expires_at else None, "request": request.parameters()})
        async with self.sources.transaction(owner_id) as db:
            await self.requests.lock_owner(db, owner_id)
            await self.requests._voice_source(db, owner_id, request)
            if access_check is not None:
                access_check(owner_id)
            await self.sources._area(db, owner_id, area_id, allow_archived=True)
            replay = await db.fetchrow("""select parameters_digest,accepted_policy_revision
                from jarvis_source_retention_requests where owner_id=$1 and source_id=$2
                and request_event_id=$3 and request_revision=$4""",
                owner_id, source_id, request.event_id, request.revision)
            if replay is not None:
                if replay["parameters_digest"] != parameters:
                    raise SourceError("retention_request_conflict",
                        "This request already changed retention with different parameters. Inspect current policy.")
                return await self._snapshot(db, owner_id, area_id, source_id) | {
                    "replayed": True, "applied_policy_revision": replay["accepted_policy_revision"]}
            state = await self._snapshot(db, owner_id, area_id, source_id)
            if (state["source_revision"] != expected_source_revision
                    or state["source_generation"] != expected_source_generation
                    or state["retention"]["policy_revision"] != expected_policy_revision):
                raise SourceError("retention_revision_conflict", "Read current source and retention revisions first.")
            if state["source_state"] == "deleted":
                raise SourceError("knowledge_source_deleted", "Retention cannot restore a deleted source.")
            revision = expected_policy_revision + 1
            await db.execute("""insert into jarvis_source_retention
                (owner_id,area_id,source_id,policy_revision,source_revision,source_generation,
                 expires_at,request_event_id,request_revision,status)
                values($1,$2,$3,$4,$5,$6,$7,$8,$9,$10)
                on conflict(owner_id,source_id) do update set policy_revision=excluded.policy_revision,
                    source_revision=excluded.source_revision,source_generation=excluded.source_generation,
                    expires_at=excluded.expires_at,request_event_id=excluded.request_event_id,
                    request_revision=excluded.request_revision,status=excluded.status,reason=null,
                    next_check_at=clock_timestamp(),checked_at=null,updated_at=clock_timestamp()""",
                owner_id, area_id, source_id, revision, expected_source_revision, expected_source_generation,
                expires_at, request.event_id, request.revision, "scheduled" if expires_at else "disabled")
            await db.execute("""insert into jarvis_source_retention_requests
                (owner_id,source_id,request_event_id,request_revision,parameters_digest,accepted_policy_revision)
                values($1,$2,$3,$4,$5,$6)""",
                owner_id, source_id, request.event_id, request.revision, parameters, revision)
            if access_check is not None:
                access_check(owner_id)
            return await self._snapshot(db, owner_id, area_id, source_id) | {
                "replayed": False, "applied_policy_revision": revision}

    async def _referenced(self, db, owner_id, source_id):
        # Original locks stop a new dependent copy being admitted between this
        # test and deletion. Internal version links are not independent holds.
        await db.fetch("""select p.id from jarvis_context_payloads p
            where p.owner_id=$1 and exists(select 1 from jarvis_knowledge_source_versions v
                where v.owner_id=p.owner_id and v.source_id=$2 and v.payload_id=p.id)
            order by p.id for update""", owner_id, source_id)
        return await db.fetchval("""select exists(
            select 1 from jarvis_knowledge_source_versions v
            join jarvis_context_payload_dependencies d on d.owner_id=v.owner_id and d.source_payload_id=v.payload_id
            join jarvis_context_payloads p on p.owner_id=d.owner_id and p.id=d.dependent_payload_id
            where v.owner_id=$1 and v.source_id=$2 and p.availability='available'
                and not exists(select 1 from jarvis_knowledge_source_versions own
                    where own.owner_id=p.owner_id and own.source_id=$2 and own.payload_id=p.id)
            union all
            select 1 from jarvis_task_memory_sources ref
            join jarvis_task_context_manifests m on m.owner_id=ref.owner_id and m.id=ref.manifest_id
            join jarvis_context_payloads p on p.owner_id=m.owner_id and p.id=m.payload_id
            where ref.owner_id=$1 and ref.source_id=$2 and m.status='active' and p.availability='available'
            union all
            select 1 from jarvis_messages message
            join jarvis_knowledge_source_versions v on v.owner_id=message.owner_id
                and v.payload_id=message.source_payload_id
            where v.owner_id=$1 and v.source_id=$2 and message.content is not null
        )""", owner_id, source_id)

    async def _expire(self, owner_id, candidate, *, as_of=None):
        as_of = self._clock_time() if as_of is None else self._deadline(as_of)
        async with self.sources.transaction(owner_id) as db:
            await self.requests.lock_owner(db, owner_id)
            policy = await db.fetchrow("""select * from jarvis_source_retention
                where owner_id=$1 and source_id=$2 and policy_revision=$3
                    and status='scheduled' and expires_at<=coalesce($4::timestamptz,clock_timestamp())
                    and next_check_at<=coalesce($4::timestamptz,clock_timestamp()) for update""",
                owner_id, candidate["source_id"], candidate["policy_revision"], as_of)
            if policy is None:
                return None
            source = await db.fetchrow("""select status,current_revision,generation
                from jarvis_knowledge_sources where owner_id=$1 and area_id=$2 and id=$3""",
                owner_id, policy["area_id"], policy["source_id"])
            status, reason = "scheduled", None
            if source["status"] == "deleted":
                status, reason = "expired", "already_deleted"
            elif (source["current_revision"] != policy["source_revision"]
                    or source["generation"] != policy["source_generation"]):
                status, reason = "requires_review", "source_changed"
            elif not await self.requests._source_available(
                    db, owner_id, policy["request_event_id"], policy["request_revision"]):
                status, reason = "requires_review", "request_evidence_unavailable"
            elif await self._referenced(db, owner_id, policy["source_id"]):
                reason = "retained_dependents"
            else:
                try:
                    async with db.transaction():
                        await self.sources.delete_source_in_transaction(
                            db, owner_id, policy["area_id"], policy["source_id"],
                            expected_revision=policy["source_revision"])
                    status, reason = "expired", "originals_deleted_provider_cleanup_queued"
                except SourceError as error:
                    # A shared source is never enrolled or deleted implicitly.
                    reason = error.code
                    if reason not in ("knowledge_source_shared_original", "legacy_original_cleanup_busy"):
                        status = "requires_review"
            await db.execute("""update jarvis_source_retention set status=$3,reason=$4,
                checked_at=clock_timestamp(),updated_at=clock_timestamp(),
                next_check_at=coalesce($5::timestamptz,clock_timestamp())+interval '1 minute'
                where owner_id=$1 and source_id=$2""", owner_id, policy["source_id"], status, reason, as_of)
            return {"source_id": str(policy["source_id"]), "policy_revision": policy["policy_revision"],
                    "status": status, "reason": reason, "full_erasure_verified": False}

    async def process_due(self, owner_id, *, limit=16):
        if type(limit) is not int or not 1 <= limit <= 128:
            raise ValueError("Retention pass must contain one to 128 sources")
        # Freeze one injected instant for selection and all guarded rechecks.
        # None deliberately retains the existing SQL wall-clock behavior.
        as_of = self._clock_time()
        # The transaction also validates owner identity. Each candidate has its
        # own short atomic operation and fresh authority/revision checks.
        async with self.sources.transaction(owner_id) as db:
            candidates = await db.fetch("""select source_id,policy_revision from jarvis_source_retention
                where owner_id=$1 and status='scheduled' and expires_at<=coalesce($3::timestamptz,clock_timestamp())
                    and next_check_at<=coalesce($3::timestamptz,clock_timestamp())
                order by next_check_at,source_id limit $2""", owner_id, limit, as_of)
        results = []
        for candidate in candidates:
            result = await self._expire(owner_id, candidate, as_of=as_of)
            if result is not None:
                results.append(result)
        return results
