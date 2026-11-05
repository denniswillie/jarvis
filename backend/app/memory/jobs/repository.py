from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Mapping, Sequence, cast
from uuid import UUID

import asyncpg

from app.memory.jobs.models import (
    MEMORY_JOB_STATUSES,
    MemoryJob,
    MemoryJobEnqueueResult,
    MemoryJobQueueDiagnostics,
    MemoryJobStatus,
    normalize_job_error,
    normalize_job_json_object,
)


class MemoryJobError(Exception):
    pass


class MemoryJobMissingError(MemoryJobError):
    pass


class MemoryJobSessionMissingError(MemoryJobError):
    pass


class MemoryJobClaimLostError(MemoryJobError):
    pass


class MemoryJobStateError(MemoryJobError):
    pass


class MemoryJobConflictError(MemoryJobError):
    pass


class MemoryJobTransactionRequiredError(MemoryJobError):
    pass


class MemoryJobRepository:
    def __init__(
        self,
        pool: asyncpg.Pool,
        *,
        default_max_attempts: int = 3,
    ) -> None:
        if default_max_attempts < 1:
            raise ValueError("default_max_attempts must be positive")
        self._pool = pool
        self._default_max_attempts = default_max_attempts

    async def enqueue(
        self,
        *,
        owner_id: str,
        session_id: UUID | None,
        job_type: str,
        dedupe_key: str,
        payload: Mapping[str, Any],
        priority: int = 0,
        max_attempts: int | None = None,
        available_at: datetime,
        connection: asyncpg.Connection | None = None,
    ) -> MemoryJobEnqueueResult:
        normalized_owner_id = _require_text(owner_id, "owner_id")
        normalized_job_type = _require_text(job_type, "job_type")
        normalized_dedupe_key = _require_text(dedupe_key, "dedupe_key")
        safe_payload = normalize_job_json_object(payload, label="memory job payload")
        effective_max_attempts = (
            self._default_max_attempts if max_attempts is None else max_attempts
        )
        if effective_max_attempts < 1:
            raise ValueError("max_attempts must be positive")

        if connection is not None:
            if not connection.is_in_transaction():
                raise MemoryJobTransactionRequiredError(
                    "caller-owned enqueue requires an active database transaction"
                )
            return await self._enqueue_with_connection(
                connection,
                owner_id=normalized_owner_id,
                session_id=session_id,
                job_type=normalized_job_type,
                dedupe_key=normalized_dedupe_key,
                payload=safe_payload,
                priority=priority,
                max_attempts=effective_max_attempts,
                available_at=available_at,
            )

        async with self._pool.acquire() as conn:
            async with conn.transaction():
                return await self._enqueue_with_connection(
                    conn,
                    owner_id=normalized_owner_id,
                    session_id=session_id,
                    job_type=normalized_job_type,
                    dedupe_key=normalized_dedupe_key,
                    payload=safe_payload,
                    priority=priority,
                    max_attempts=effective_max_attempts,
                    available_at=available_at,
                )

    async def enqueue_unless_job_type_active(
        self,
        *,
        owner_id: str,
        session_id: UUID | None,
        job_type: str,
        dedupe_key: str,
        payload: Mapping[str, Any],
        priority: int = 0,
        max_attempts: int | None = None,
        available_at: datetime,
    ) -> MemoryJobEnqueueResult:
        """Enqueue once no queued, claimed, or retryable job of this type exists."""

        normalized_owner_id = _require_text(owner_id, "owner_id")
        normalized_job_type = _require_text(job_type, "job_type")
        normalized_dedupe_key = _require_text(dedupe_key, "dedupe_key")
        safe_payload = normalize_job_json_object(payload, label="memory job payload")
        effective_max_attempts = (
            self._default_max_attempts if max_attempts is None else max_attempts
        )
        if effective_max_attempts < 1:
            raise ValueError("max_attempts must be positive")

        async with self._pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    "select pg_advisory_xact_lock(hashtextextended($1, 0))",
                    json.dumps(
                        [normalized_owner_id, normalized_job_type, "active_job_type"],
                        ensure_ascii=True,
                        separators=(",", ":"),
                    ),
                )
                active = await conn.fetchrow(
                    """
                    select *
                    from memory_jobs
                    where owner_id = $1
                      and job_type = $2
                      and status in ('queued', 'claimed', 'failed')
                    order by created_at desc, id desc
                    limit 1
                    for update
                    """,
                    normalized_owner_id,
                    normalized_job_type,
                )
                if active is not None:
                    return MemoryJobEnqueueResult(
                        job=_job_from_row(active),
                        created=False,
                    )
                return await self._enqueue_with_connection(
                    conn,
                    owner_id=normalized_owner_id,
                    session_id=session_id,
                    job_type=normalized_job_type,
                    dedupe_key=normalized_dedupe_key,
                    payload=safe_payload,
                    priority=priority,
                    max_attempts=effective_max_attempts,
                    available_at=available_at,
                )

    async def get_by_dedupe_key(
        self,
        *,
        owner_id: str,
        job_type: str,
        dedupe_key: str,
    ) -> MemoryJob | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                select *
                from memory_jobs
                where owner_id = $1 and job_type = $2 and dedupe_key = $3
                order by created_at desc, id desc
                limit 1
                """,
                owner_id,
                job_type,
                dedupe_key,
            )
        return _job_from_row(row) if row is not None else None

    async def claim_next(
        self,
        *,
        worker_id: str,
        job_types: Sequence[str] | None = None,
    ) -> MemoryJob | None:
        normalized_worker_id = _require_text(worker_id, "worker_id")
        normalized_job_types = _normalize_job_types(job_types)
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                row = await conn.fetchrow(
                    """
                    with candidate as (
                        select id
                        from memory_jobs
                        where status = 'queued'
                          and available_at <= now()
                          and attempt_count < max_attempts
                          and ($2::text[] is null or job_type = any($2::text[]))
                        order by priority desc, available_at asc, created_at asc, id asc
                        for update skip locked
                        limit 1
                    )
                    update memory_jobs jobs
                    set status = 'claimed',
                        attempt_count = jobs.attempt_count + 1,
                        claimed_at = now(),
                        claimed_by = $1,
                        claim_token = gen_random_uuid(),
                        updated_at = now()
                    from candidate
                    where jobs.id = candidate.id
                    returning jobs.*
                    """,
                    normalized_worker_id,
                    list(normalized_job_types) if normalized_job_types is not None else None,
                )
        return _job_from_row(row) if row is not None else None

    async def complete(
        self,
        *,
        owner_id: str,
        job_id: UUID,
        claim_token: UUID,
        completed_at: datetime,
        result: Mapping[str, Any] | None = None,
        connection: asyncpg.Connection | None = None,
    ) -> MemoryJob:
        safe_result = (
            normalize_job_json_object(result, label="memory job result")
            if result is not None
            else None
        )
        if connection is not None:
            _require_active_transaction(connection, operation="completion")
            return await self._complete_with_connection(
                connection,
                owner_id=owner_id,
                job_id=job_id,
                claim_token=claim_token,
                completed_at=completed_at,
                result=safe_result,
            )

        async with self._pool.acquire() as conn:
            async with conn.transaction():
                return await self._complete_with_connection(
                    conn,
                    owner_id=owner_id,
                    job_id=job_id,
                    claim_token=claim_token,
                    completed_at=completed_at,
                    result=safe_result,
                )

    async def lock_claim(
        self,
        *,
        owner_id: str,
        job_id: UUID,
        claim_token: UUID,
        connection: asyncpg.Connection,
    ) -> MemoryJob:
        """Lock and revalidate a claim before a handler writes domain state."""

        _require_active_transaction(connection, operation="claim validation")
        job = await _load_job_for_transition(connection, owner_id, job_id)
        _validate_matching_claim(job, claim_token)
        return job

    async def fail(
        self,
        *,
        owner_id: str,
        job_id: UUID,
        claim_token: UUID,
        error: str,
        retry_at: datetime,
        terminal: bool = False,
    ) -> MemoryJob:
        normalized_error = normalize_job_error(error)
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                row = await conn.fetchrow(
                    """
                    update memory_jobs
                    set status = case
                            when attempt_count >= max_attempts then 'dead'
                            when $6 then 'dead'
                            else 'failed'
                        end,
                        attempt_count = case
                            when $6 then max_attempts
                            else attempt_count
                        end,
                        available_at = $5,
                        claimed_at = null,
                        claimed_by = null,
                        claim_token = null,
                        result = null,
                        last_error = $4,
                        completed_at = null,
                        updated_at = now()
                    where owner_id = $1
                      and id = $2
                      and status = 'claimed'
                      and claim_token = $3
                    returning *
                    """,
                    owner_id,
                    job_id,
                    claim_token,
                    normalized_error,
                    retry_at,
                    terminal,
                )
                if row is not None:
                    return _job_from_row(row)
                existing = await _load_job_for_transition(conn, owner_id, job_id)
                if existing.status in {"failed", "dead"}:
                    if existing.last_error == normalized_error:
                        return existing
                    raise MemoryJobConflictError(
                        "failed job was replayed with a different error"
                    )
                _require_matching_claim(existing, claim_token)
        raise AssertionError("claim validation must raise or return")

    async def renew_claim(
        self,
        *,
        owner_id: str,
        job_id: UUID,
        claim_token: UUID,
    ) -> MemoryJob:
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                row = await conn.fetchrow(
                    """
                    update memory_jobs
                    set claimed_at = now(),
                        updated_at = now()
                    where owner_id = $1
                      and id = $2
                      and status = 'claimed'
                      and claim_token = $3
                    returning *
                    """,
                    owner_id,
                    job_id,
                    claim_token,
                )
                if row is not None:
                    return _job_from_row(row)
                existing = await _load_job_for_transition(conn, owner_id, job_id)
                _require_matching_claim(existing, claim_token)
        raise AssertionError("claim validation must raise")

    async def requeue_due_failed(self, *, limit: int = 100) -> list[MemoryJob]:
        _require_limit(limit)
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                rows = await conn.fetch(
                    """
                    with due as (
                        select id
                        from memory_jobs
                        where status = 'failed'
                          and available_at <= now()
                          and attempt_count < max_attempts
                        order by available_at asc, priority desc, created_at asc, id asc
                        for update skip locked
                        limit $1
                    )
                    update memory_jobs jobs
                    set status = 'queued',
                        last_error = null,
                        updated_at = now()
                    from due
                    where jobs.id = due.id
                    returning jobs.*
                    """,
                    limit,
                )
        return [_job_from_row(row) for row in rows]

    async def recover_stale_claims(
        self,
        *,
        claimed_before: datetime,
        retry_at: datetime,
        error: str = "worker lease expired",
        limit: int = 100,
    ) -> list[MemoryJob]:
        _require_limit(limit)
        normalized_error = normalize_job_error(error)
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                rows = await conn.fetch(
                    """
                    with stale as (
                        select id
                        from memory_jobs
                        where status = 'claimed' and claimed_at <= $1
                        order by claimed_at asc, id asc
                        for update skip locked
                        limit $4
                    )
                    update memory_jobs jobs
                    set status = case
                            when jobs.attempt_count >= jobs.max_attempts then 'dead'
                            else 'failed'
                        end,
                        available_at = $2,
                        claimed_at = null,
                        claimed_by = null,
                        claim_token = null,
                        last_error = $3,
                        updated_at = now()
                    from stale
                    where jobs.id = stale.id
                    returning jobs.*
                    """,
                    claimed_before,
                    retry_at,
                    normalized_error,
                    limit,
                )
        return [_job_from_row(row) for row in rows]

    async def retry_dead(
        self,
        *,
        owner_id: str,
        job_id: UUID,
        available_at: datetime,
    ) -> MemoryJob:
        async with self._pool.acquire() as conn:
            try:
                row = await conn.fetchrow(
                    """
                    update memory_jobs
                    set status = 'queued',
                        attempt_count = 0,
                        available_at = $3,
                        last_error = null,
                        updated_at = now()
                    where owner_id = $1 and id = $2 and status = 'dead'
                    returning *
                    """,
                    owner_id,
                    job_id,
                    available_at,
                )
            except asyncpg.UniqueViolationError as exc:
                raise MemoryJobConflictError(
                    "another active job already uses this dedupe key"
                ) from exc
            if row is not None:
                return _job_from_row(row)
            existing = await conn.fetchrow(
                "select * from memory_jobs where owner_id = $1 and id = $2",
                owner_id,
                job_id,
            )
        if existing is None:
            raise MemoryJobMissingError()
        raise MemoryJobStateError("only dead jobs can be explicitly retried")

    async def delete_dead(self, *, owner_id: str, job_id: UUID) -> bool:
        async with self._pool.acquire() as conn:
            removed = await conn.fetchval(
                """
                delete from memory_jobs
                where owner_id = $1 and id = $2 and status = 'dead'
                returning true
                """,
                owner_id,
                job_id,
            )
        return bool(removed)

    async def cleanup_completed_before(
        self,
        *,
        completed_before: datetime,
        limit: int = 500,
        connection: asyncpg.Connection | None = None,
    ) -> int:
        _require_limit(limit)
        if connection is not None:
            removed = await connection.fetchval(
                """
                with expired as (
                    select id
                    from memory_jobs
                    where status = 'completed' and completed_at < $1
                    order by completed_at asc, id asc
                    limit $2
                ), deleted as (
                    delete from memory_jobs jobs
                    using expired
                    where jobs.id = expired.id
                    returning jobs.id
                )
                select count(*)::integer from deleted
                """,
                completed_before,
                limit,
            )
            return int(removed or 0)
        async with self._pool.acquire() as conn:
            removed = await conn.fetchval(
                """
                with expired as (
                    select id
                    from memory_jobs
                    where status = 'completed' and completed_at < $1
                    order by completed_at asc, id asc
                    limit $2
                ), deleted as (
                    delete from memory_jobs jobs
                    using expired
                    where jobs.id = expired.id
                    returning jobs.id
                )
                select count(*)::integer from deleted
                """,
                completed_before,
                limit,
            )
        return int(removed or 0)

    async def get_diagnostics(self) -> MemoryJobQueueDiagnostics:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                select
                    count(*) filter (where status = 'queued')::integer as queued_count,
                    count(*) filter (where status = 'claimed')::integer as claimed_count,
                    count(*) filter (where status = 'completed')::integer as completed_count,
                    count(*) filter (where status = 'failed')::integer as failed_count,
                    count(*) filter (where status = 'dead')::integer as dead_count,
                    extract(epoch from (now() - min(created_at)
                        filter (where status = 'queued')))::double precision
                        as oldest_queued_age_seconds,
                    coalesce(
                        array_agg(distinct job_type order by job_type)
                            filter (where status = 'claimed'),
                        array[]::text[]
                    ) as active_job_types,
                    max(completed_at) filter (where status = 'completed')
                        as last_completed_at
                from memory_jobs
                """
            )
        if row is None:
            raise MemoryJobStateError("memory job diagnostics query returned no row")
        counts = {
            status: int(row[f"{status}_count"] or 0)
            for status in MEMORY_JOB_STATUSES
        }
        oldest_age = row["oldest_queued_age_seconds"]
        return MemoryJobQueueDiagnostics(
            counts_by_status=counts,
            oldest_queued_age_seconds=(
                max(0.0, float(oldest_age)) if oldest_age is not None else None
            ),
            active_job_types=tuple(row["active_job_types"] or ()),
            last_completed_at=row["last_completed_at"],
        )

    async def _complete_with_connection(
        self,
        conn: asyncpg.Connection,
        *,
        owner_id: str,
        job_id: UUID,
        claim_token: UUID,
        completed_at: datetime,
        result: Mapping[str, Any] | None,
    ) -> MemoryJob:
        row = await conn.fetchrow(
            """
            update memory_jobs
            set status = 'completed',
                claimed_at = null,
                claimed_by = null,
                claim_token = null,
                result = $4::jsonb,
                last_error = null,
                completed_at = $5,
                updated_at = now()
            where owner_id = $1
              and id = $2
              and status = 'claimed'
              and claim_token = $3
            returning *
            """,
            owner_id,
            job_id,
            claim_token,
            _json_dumps(result) if result is not None else None,
            completed_at,
        )
        if row is not None:
            return _job_from_row(row)
        existing = await _load_job_for_transition(conn, owner_id, job_id)
        if existing.status == "completed":
            if existing.result == result:
                return existing
            raise MemoryJobConflictError(
                "completed job was replayed with a different result"
            )
        _require_matching_claim(existing, claim_token)
        raise AssertionError("claim validation must raise")

    async def _enqueue_with_connection(
        self,
        conn: asyncpg.Connection,
        *,
        owner_id: str,
        session_id: UUID | None,
        job_type: str,
        dedupe_key: str,
        payload: Mapping[str, Any],
        priority: int,
        max_attempts: int,
        available_at: datetime,
    ) -> MemoryJobEnqueueResult:
        await conn.execute(
            "select pg_advisory_xact_lock(hashtextextended($1, 0))",
            json.dumps(
                [owner_id, job_type, dedupe_key],
                ensure_ascii=True,
                separators=(",", ":"),
            ),
        )
        existing = await conn.fetchrow(
            """
            select *
            from memory_jobs
            where owner_id = $1 and job_type = $2 and dedupe_key = $3
            order by created_at desc, id desc
            limit 1
            for update
            """,
            owner_id,
            job_type,
            dedupe_key,
        )
        if existing is not None:
            existing_job = _job_from_row(existing)
            _require_matching_enqueue_intent(
                existing_job,
                session_id=session_id,
                payload=payload,
                priority=priority,
                max_attempts=max_attempts,
            )
            return MemoryJobEnqueueResult(job=existing_job, created=False)

        row = await conn.fetchrow(
            """
            insert into memory_jobs (
                owner_id,
                session_id,
                job_type,
                dedupe_key,
                payload,
                priority,
                max_attempts,
                available_at
            )
            select $1, $2, $3, $4, $5::jsonb, $6, $7, $8
            where $2::uuid is null
               or exists (
                    select 1
                    from jarvis_sessions
                    where owner_id = $1 and id = $2
               )
            returning *
            """,
            owner_id,
            session_id,
            job_type,
            dedupe_key,
            _json_dumps(payload),
            priority,
            max_attempts,
            available_at,
        )
        if row is None:
            raise MemoryJobSessionMissingError()
        return MemoryJobEnqueueResult(job=_job_from_row(row), created=True)


async def _load_job_for_transition(
    conn: asyncpg.Connection,
    owner_id: str,
    job_id: UUID,
) -> MemoryJob:
    row = await conn.fetchrow(
        "select * from memory_jobs where owner_id = $1 and id = $2 for update",
        owner_id,
        job_id,
    )
    if row is None:
        raise MemoryJobMissingError()
    return _job_from_row(row)


def _require_matching_claim(job: MemoryJob, claim_token: UUID) -> None:
    _validate_matching_claim(job, claim_token)
    raise MemoryJobConflictError("claimed job could not be updated")


def _validate_matching_claim(job: MemoryJob, claim_token: UUID) -> None:
    if job.status != "claimed":
        raise MemoryJobStateError(f"job is {job.status}, not claimed")
    if job.claim_token != claim_token:
        raise MemoryJobClaimLostError()


def _require_active_transaction(
    connection: asyncpg.Connection,
    *,
    operation: str,
) -> None:
    if not connection.is_in_transaction():
        raise MemoryJobTransactionRequiredError(
            f"caller-owned {operation} requires an active database transaction"
        )


def _require_matching_enqueue_intent(
    job: MemoryJob,
    *,
    session_id: UUID | None,
    payload: Mapping[str, Any],
    priority: int,
    max_attempts: int,
) -> None:
    conflicts: list[str] = []
    if job.session_id != session_id:
        conflicts.append("session_id")
    if dict(job.payload) != dict(payload):
        conflicts.append("payload")
    if job.priority != priority:
        conflicts.append("priority")
    if job.max_attempts != max_attempts:
        conflicts.append("max_attempts")
    if conflicts:
        raise MemoryJobConflictError(
            "dedupe key was reused with different durable intent: " + ", ".join(conflicts)
        )


def _job_from_row(row: Mapping[str, Any]) -> MemoryJob:
    result = row["result"]
    return MemoryJob(
        id=row["id"],
        owner_id=row["owner_id"],
        session_id=row["session_id"],
        job_type=row["job_type"],
        dedupe_key=row["dedupe_key"],
        payload=_json_object(row["payload"]),
        status=cast(MemoryJobStatus, row["status"]),
        priority=row["priority"],
        attempt_count=row["attempt_count"],
        max_attempts=row["max_attempts"],
        available_at=row["available_at"],
        claimed_at=row["claimed_at"],
        claimed_by=row["claimed_by"],
        claim_token=row["claim_token"],
        result=_json_object(result) if result is not None else None,
        last_error=row["last_error"],
        completed_at=row["completed_at"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _json_object(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, Mapping):
        raise ValueError("database JSON value must be an object")
    return dict(value)


def _json_dumps(value: Mapping[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def _normalize_job_types(job_types: Sequence[str] | None) -> tuple[str, ...] | None:
    if job_types is None:
        return None
    normalized = tuple(dict.fromkeys(_require_text(item, "job_type") for item in job_types))
    if not normalized:
        raise ValueError("job_types cannot be empty")
    return normalized


def _require_text(value: str, label: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{label} cannot be blank")
    return normalized


def _require_limit(limit: int) -> None:
    if limit < 1 or limit > 10_000:
        raise ValueError("limit must be between 1 and 10000")
