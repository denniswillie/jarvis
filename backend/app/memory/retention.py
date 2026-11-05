from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import asyncpg

from app.memory.jobs.models import MemoryJob
from app.memory.jobs.repository import MemoryJobRepository


EXPIRE_WORKING_MEMORY_JOB = "memory.expire_working_memory"
COMPACT_OBSERVATIONS_JOB = "memory.compact_observations"
DELETE_OBSERVATIONS_JOB = "memory.delete_observations"
DELETE_VISUAL_FRAME_ROWS_JOB = "memory.delete_visual_frame_rows"
DELETE_SKIPPED_EVENTS_JOB = "memory.delete_skipped_events"
CLEANUP_COMPLETED_JOBS_JOB = "memory.cleanup_completed_jobs"
CLEANUP_RETRIEVAL_AUDITS_JOB = "memory.cleanup_retrieval_audits"

DATABASE_MAINTENANCE_JOB_TYPES = (
    EXPIRE_WORKING_MEMORY_JOB,
    COMPACT_OBSERVATIONS_JOB,
    DELETE_OBSERVATIONS_JOB,
    DELETE_VISUAL_FRAME_ROWS_JOB,
    DELETE_SKIPPED_EVENTS_JOB,
    CLEANUP_COMPLETED_JOBS_JOB,
    CLEANUP_RETRIEVAL_AUDITS_JOB,
)


@dataclass(frozen=True, slots=True)
class MemoryRetentionPolicy:
    observation_detail_days: int = 7
    unreferenced_observation_days: int = 30
    unreferenced_visual_frame_days: int = 30
    skipped_event_days: int = 90
    completed_job_days: int = 14
    retrieval_audit_days: int = 30
    batch_size: int = 500

    def __post_init__(self) -> None:
        values = (
            self.observation_detail_days,
            self.unreferenced_observation_days,
            self.unreferenced_visual_frame_days,
            self.skipped_event_days,
            self.completed_job_days,
            self.retrieval_audit_days,
            self.batch_size,
        )
        if any(isinstance(value, bool) or value < 1 for value in values):
            raise ValueError("memory retention values must be positive integers")

    def observation_detail_cutoff(self, now: datetime) -> datetime:
        return _utc(now) - timedelta(days=self.observation_detail_days)

    def unreferenced_observation_cutoff(self, now: datetime) -> datetime:
        return _utc(now) - timedelta(days=self.unreferenced_observation_days)

    def skipped_event_cutoff(self, now: datetime) -> datetime:
        return _utc(now) - timedelta(days=self.skipped_event_days)

    def unreferenced_visual_frame_cutoff(self, now: datetime) -> datetime:
        return _utc(now) - timedelta(days=self.unreferenced_visual_frame_days)

    def completed_job_cutoff(self, now: datetime) -> datetime:
        return _utc(now) - timedelta(days=self.completed_job_days)

    def retrieval_audit_cutoff(self, now: datetime) -> datetime:
        return _utc(now) - timedelta(days=self.retrieval_audit_days)


class MemoryRetentionRepository:
    """Bounded, transaction-scoped maintenance mutations."""

    async def expire_working_memory(
        self,
        *,
        owner_id: str,
        now: datetime,
        limit: int,
        connection: asyncpg.Connection,
    ) -> Mapping[str, int]:
        _require_limit(limit)
        session_count = await connection.fetchval(
            """
            with expired as (
                select session_id
                from jarvis_session_working_memory
                where owner_id = $1 and expires_at <= $2
                order by expires_at asc, session_id asc
                for update skip locked
                limit $3
            ), deleted as (
                delete from jarvis_session_working_memory working
                using expired
                where working.session_id = expired.session_id
                returning working.session_id
            )
            select count(*)::integer from deleted
            """,
            _require_owner(owner_id),
            _utc(now),
            limit,
        )
        owner_count = await connection.fetchval(
            """
            with expired as (
                select owner_id
                from jarvis_owner_working_memory
                where owner_id = $1
                  and expires_at is not null
                  and expires_at <= $2
                for update skip locked
                limit 1
            ), deleted as (
                delete from jarvis_owner_working_memory working
                using expired
                where working.owner_id = expired.owner_id
                returning working.owner_id
            )
            select count(*)::integer from deleted
            """,
            _require_owner(owner_id),
            _utc(now),
        )
        return {
            "expired_session_working_memory": int(session_count or 0),
            "expired_owner_working_memory": int(owner_count or 0),
        }

    async def compact_observations(
        self,
        *,
        owner_id: str,
        ended_before: datetime,
        compacted_at: datetime,
        limit: int,
        connection: asyncpg.Connection,
    ) -> int:
        _require_limit(limit)
        count = await connection.fetchval(
            """
            with candidates as (
                select observation.id
                from perception_observations observation
                where observation.owner_id = $1
                  and observation.ended_at < $2
                  and observation.compacted_at is null
                  and not exists (
                      select 1
                      from memory_evidence evidence
                      join memory_items memory
                        on memory.id = evidence.memory_item_id
                       and memory.owner_id = evidence.owner_id
                      where evidence.owner_id = observation.owner_id
                        and evidence.evidence_type = 'observation'
                        and evidence.evidence_ref_id = observation.id::text
                        and memory.status = 'active'
                  )
                order by observation.ended_at asc, observation.id asc
                for update skip locked
                limit $4
            ), removed_frame_links as (
                delete from perception_observation_frames link
                using candidates
                where link.observation_id = candidates.id
                returning link.observation_id
            ), compacted as (
                update perception_observations observation
                set primary_visual_frame_id = null,
                    entities = '[]'::jsonb,
                    visible_text = '[]'::jsonb,
                    state_changes = '[]'::jsonb,
                    keyframe_candidates = '[]'::jsonb,
                    interrupt_candidate = null,
                    compacted_at = $3
                from candidates
                where observation.id = candidates.id
                  and (select count(*) from removed_frame_links) >= 0
                returning observation.id
            )
            select count(*)::integer from compacted
            """,
            _require_owner(owner_id),
            _utc(ended_before),
            _utc(compacted_at),
            limit,
        )
        return int(count or 0)

    async def delete_unreferenced_observations(
        self,
        *,
        owner_id: str,
        ended_before: datetime,
        limit: int,
        connection: asyncpg.Connection,
    ) -> Mapping[str, int]:
        _require_limit(limit)
        row = await connection.fetchrow(
            """
            with candidates as materialized (
                select observation.id
                from perception_observations observation
                where observation.owner_id = $1
                  and observation.ended_at < $2
                  and observation.compacted_at is not null
                  and not exists (
                      select 1
                      from memory_evidence evidence
                      join memory_items memory
                        on memory.id = evidence.memory_item_id
                       and memory.owner_id = evidence.owner_id
                      where evidence.owner_id = observation.owner_id
                        and evidence.evidence_type = 'observation'
                        and evidence.evidence_ref_id = observation.id::text
                        and memory.status = 'active'
                  )
                  and not exists (
                      select 1
                      from jarvis_session_working_memory working
                      where working.last_observation_id = observation.id
                  )
                order by observation.ended_at asc, observation.id asc
                for update skip locked
                limit $3
            ), removed_event_evidence as (
                delete from experience_event_evidence evidence
                using candidates
                where evidence.evidence_type = 'observation'
                  and evidence.evidence_ref_id = candidates.id
                returning evidence.event_id
            ), removed_memory_evidence as (
                delete from memory_evidence evidence
                using candidates
                where evidence.owner_id = $1
                  and evidence.evidence_type = 'observation'
                  and evidence.evidence_ref_id = candidates.id::text
                returning evidence.id
            ), deleted_observations as (
                delete from perception_observations observation
                using candidates
                where observation.id = candidates.id
                  and (select count(*) from removed_memory_evidence) >= 0
                returning observation.id
            ), reduced_events as (
                update experience_events event
                set evidence_reduced_at = coalesce(event.evidence_reduced_at, now()),
                    updated_at = now()
                where event.id in (select distinct event_id from removed_event_evidence)
                  and (select count(*) from deleted_observations) >= 0
                returning event.id
            )
            select
                (select count(*)::integer from deleted_observations)
                    as deleted_observations,
                (select count(*)::integer from removed_event_evidence)
                    as deleted_event_evidence,
                (select count(*)::integer from removed_memory_evidence)
                    as deleted_memory_evidence,
                (select count(*)::integer from reduced_events)
                    as reduced_events
            """,
            _require_owner(owner_id),
            _utc(ended_before),
            limit,
        )
        return _integer_mapping(row)

    async def delete_unreferenced_visual_frames(
        self,
        *,
        owner_id: str,
        captured_before: datetime,
        now: datetime,
        limit: int,
        connection: asyncpg.Connection,
    ) -> Mapping[str, int]:
        """Delete expired passive frame rows after durable references release."""

        _require_limit(limit)
        row = await connection.fetchrow(
            """
            with candidates as materialized (
                select frame.id
                from visual_frames frame
                where frame.owner_id = $1
                  and frame.raw_image_status = 'expired'
                  and frame.captured_at < $2
                  and not exists (
                      select 1
                      from visual_frame_retention_reasons reason
                      where reason.owner_id = frame.owner_id
                        and reason.visual_frame_id = frame.id
                        and (reason.expires_at is null or reason.expires_at > $3)
                  )
                  and not exists (
                      select 1 from jarvis_files file
                      where file.owner_id = frame.owner_id
                        and file.source_visual_frame_id = frame.id
                  )
                  and not exists (
                      select 1
                      from perception_observation_frames observation_frame
                      join perception_observations observation
                        on observation.id = observation_frame.observation_id
                      where observation_frame.visual_frame_id = frame.id
                        and observation.compacted_at is null
                  )
                  and not exists (
                      select 1 from perception_observations observation
                      where observation.primary_visual_frame_id = frame.id
                        and observation.compacted_at is null
                  )
                  and not exists (
                      select 1
                      from memory_evidence evidence
                      join memory_items memory
                        on memory.id = evidence.memory_item_id
                       and memory.owner_id = evidence.owner_id
                      where evidence.owner_id = frame.owner_id
                        and evidence.evidence_type = 'frame'
                        and evidence.evidence_ref_id = frame.id::text
                        and memory.status = 'active'
                  )
                order by frame.captured_at asc, frame.id asc
                for update skip locked
                limit $4
            ), removed_observation_frames as (
                delete from perception_observation_frames observation_frame
                using candidates, perception_observations observation
                where observation_frame.visual_frame_id = candidates.id
                  and observation.id = observation_frame.observation_id
                  and observation.compacted_at is not null
                returning observation_frame.observation_id
            ), removed_keyframes as (
                delete from experience_event_keyframes keyframe
                using candidates
                where keyframe.visual_frame_id = candidates.id
                returning keyframe.event_id
            ), removed_memory_evidence as (
                delete from memory_evidence evidence
                using candidates
                where evidence.owner_id = $1
                  and evidence.evidence_type = 'frame'
                  and evidence.evidence_ref_id = candidates.id::text
                returning evidence.id
            ), deleted_frames as (
                delete from visual_frames frame
                using candidates
                where frame.id = candidates.id
                  and (select count(*) from removed_observation_frames) >= 0
                  and (select count(*) from removed_keyframes) >= 0
                  and (select count(*) from removed_memory_evidence) >= 0
                returning frame.id
            ), reduced_events as (
                update experience_events event
                set evidence_reduced_at = coalesce(event.evidence_reduced_at, $3),
                    updated_at = $3
                where event.id in (select distinct event_id from removed_keyframes)
                  and (select count(*) from deleted_frames) >= 0
                returning event.id
            )
            select
                (select count(*)::integer from deleted_frames) as deleted_frames,
                (select count(*)::integer from removed_observation_frames)
                    as deleted_observation_frame_links,
                (select count(*)::integer from removed_keyframes)
                    as deleted_event_keyframes,
                (select count(*)::integer from removed_memory_evidence)
                    as deleted_memory_evidence,
                (select count(*)::integer from reduced_events) as reduced_events
            """,
            _require_owner(owner_id),
            _utc(captured_before),
            _utc(now),
            limit,
        )
        return _integer_mapping(row)

    async def delete_expired_skipped_events(
        self,
        *,
        owner_id: str,
        now: datetime,
        limit: int,
        connection: asyncpg.Connection,
    ) -> Mapping[str, int]:
        _require_limit(limit)
        row = await connection.fetchrow(
            """
            with expired_reasons as materialized (
                select reason.id
                from experience_event_retention_reasons reason
                where reason.owner_id = $1
                  and reason.expires_at is not null
                  and reason.expires_at <= $2
                order by reason.expires_at asc, reason.id asc
                limit $3
            ), removed_expired_reasons as (
                delete from experience_event_retention_reasons reason
                using expired_reasons
                where reason.id = expired_reasons.id
                returning reason.id
            ), candidates as materialized (
                select event.id
                from experience_events event
                where event.owner_id = $1
                  and event.promotion_decision = 'skipped'
                  and event.retention_expires_at is not null
                  and event.retention_expires_at <= $2
                  and not exists (
                      select 1 from memory_items memory
                      where memory.source_experience_event_id = event.id
                  )
                  and not exists (
                      select 1
                      from experience_event_retention_reasons reason
                      where reason.owner_id = event.owner_id
                        and reason.experience_event_id = event.id
                        and (reason.expires_at is null or reason.expires_at > $2)
                  )
                  and not exists (
                      select 1
                      from memory_evidence evidence
                      join memory_items memory
                        on memory.id = evidence.memory_item_id
                       and memory.owner_id = evidence.owner_id
                      where evidence.owner_id = event.owner_id
                        and evidence.evidence_type = 'experience_event'
                        and evidence.evidence_ref_id = event.id::text
                        and memory.status = 'active'
                  )
                  and (select count(*) from removed_expired_reasons) >= 0
                  and not exists (
                      select 1
                      from experience_event_keyframes keyframe
                      join visual_frame_retention_reasons reason
                        on reason.visual_frame_id = keyframe.visual_frame_id
                       and reason.owner_id = event.owner_id
                      where keyframe.event_id = event.id
                        and (reason.expires_at is null or reason.expires_at > $2)
                  )
                order by event.retention_expires_at asc, event.id asc
                for update skip locked
                limit $3
            ), removed_memory_evidence as (
                delete from memory_evidence evidence
                using candidates
                where evidence.owner_id = $1
                  and evidence.evidence_type = 'experience_event'
                  and evidence.evidence_ref_id = candidates.id::text
                returning evidence.id
            ), deleted_events as (
                delete from experience_events event
                using candidates
                where event.id = candidates.id
                  and (select count(*) from removed_memory_evidence) >= 0
                returning event.id
            )
            select
                (select count(*)::integer from deleted_events) as deleted_events,
                (select count(*)::integer from removed_memory_evidence)
                    as deleted_memory_evidence,
                (select count(*)::integer from removed_expired_reasons)
                    as deleted_expired_event_retention_reasons
            """,
            _require_owner(owner_id),
            _utc(now),
            limit,
        )
        return _integer_mapping(row)

    async def delete_retrieval_audits(
        self,
        *,
        owner_id: str,
        created_before: datetime,
        limit: int,
        connection: asyncpg.Connection,
    ) -> int:
        _require_limit(limit)
        count = await connection.fetchval(
            """
            with expired as (
                select id
                from memory_retrieval_runs
                where owner_id = $1 and created_at < $2
                order by created_at asc, id asc
                limit $3
            ), deleted as (
                delete from memory_retrieval_runs run
                using expired
                where run.id = expired.id
                returning run.id
            )
            select count(*)::integer from deleted
            """,
            _require_owner(owner_id),
            _utc(created_before),
            limit,
        )
        return int(count or 0)


class MemoryMaintenanceJobHandler:
    """Dispatch one bounded database maintenance operation through the job runner."""

    def __init__(
        self,
        *,
        job_type: str,
        repository: MemoryRetentionRepository,
        job_repository: MemoryJobRepository,
        policy: MemoryRetentionPolicy,
    ) -> None:
        if job_type not in DATABASE_MAINTENANCE_JOB_TYPES:
            raise ValueError(f"unsupported maintenance job type: {job_type}")
        self.job_type = job_type
        self._repository = repository
        self._job_repository = job_repository
        self._policy = policy

    async def prepare(self, job: MemoryJob) -> None:
        return None

    async def commit(
        self,
        job: MemoryJob,
        prepared_result: None,
        connection: asyncpg.Connection,
    ) -> Mapping[str, Any]:
        now = datetime.now(UTC)
        if self.job_type == EXPIRE_WORKING_MEMORY_JOB:
            return await self._repository.expire_working_memory(
                owner_id=job.owner_id,
                now=now,
                limit=self._policy.batch_size,
                connection=connection,
            )
        if self.job_type == COMPACT_OBSERVATIONS_JOB:
            compacted = await self._repository.compact_observations(
                owner_id=job.owner_id,
                ended_before=self._policy.observation_detail_cutoff(now),
                compacted_at=now,
                limit=self._policy.batch_size,
                connection=connection,
            )
            return {"compacted_observations": compacted}
        if self.job_type == DELETE_OBSERVATIONS_JOB:
            return await self._repository.delete_unreferenced_observations(
                owner_id=job.owner_id,
                ended_before=self._policy.unreferenced_observation_cutoff(now),
                limit=self._policy.batch_size,
                connection=connection,
            )
        if self.job_type == DELETE_VISUAL_FRAME_ROWS_JOB:
            return await self._repository.delete_unreferenced_visual_frames(
                owner_id=job.owner_id,
                captured_before=self._policy.unreferenced_visual_frame_cutoff(now),
                now=now,
                limit=self._policy.batch_size,
                connection=connection,
            )
        if self.job_type == DELETE_SKIPPED_EVENTS_JOB:
            return await self._repository.delete_expired_skipped_events(
                owner_id=job.owner_id,
                now=now,
                limit=self._policy.batch_size,
                connection=connection,
            )
        if self.job_type == CLEANUP_COMPLETED_JOBS_JOB:
            removed = await self._job_repository.cleanup_completed_before(
                completed_before=self._policy.completed_job_cutoff(now),
                limit=self._policy.batch_size,
                connection=connection,
            )
            return {"deleted_completed_jobs": removed}
        if self.job_type == CLEANUP_RETRIEVAL_AUDITS_JOB:
            removed = await self._repository.delete_retrieval_audits(
                owner_id=job.owner_id,
                created_before=self._policy.retrieval_audit_cutoff(now),
                limit=self._policy.batch_size,
                connection=connection,
            )
            return {"deleted_retrieval_audits": removed}
        raise RuntimeError(f"unhandled maintenance job type: {self.job_type}")


def _require_owner(owner_id: str) -> str:
    normalized = owner_id.strip()
    if not normalized:
        raise ValueError("owner_id cannot be blank")
    return normalized


def _require_limit(limit: int) -> None:
    if isinstance(limit, bool) or limit < 1 or limit > 5_000:
        raise ValueError("retention limit must be between 1 and 5000")


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("retention timestamps must be timezone-aware")
    return value.astimezone(UTC)


def _integer_mapping(row: Mapping[str, Any] | None) -> Mapping[str, int]:
    if row is None:
        return {}
    return {key: int(value or 0) for key, value in row.items()}
