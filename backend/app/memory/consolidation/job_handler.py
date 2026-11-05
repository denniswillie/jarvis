from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from typing import Any, Mapping
from uuid import UUID

import asyncpg

from app.experience.repository import (
    ExperienceConflictError,
    ExperienceEventMissingError,
    ExperienceRepository,
)
from app.memory.consolidation.contracts import (
    ConsolidationResult,
    ConsolidationValidationError,
    consolidation_result_payload,
)
from app.memory.consolidation.openai_client import (
    ConsolidationProviderContractError,
    ConsolidationProviderTransientError,
)
from app.memory.consolidation.service import (
    ExperienceConsolidationService,
    PreparedConsolidation,
)
from app.memory.jobs.handlers import (
    MemoryJobTransientError,
    MemoryJobValidationError,
)
from app.memory.jobs.models import MemoryJob
from app.memory.jobs.repository import MemoryJobRepository


CONSOLIDATE_EXPERIENCE_EVENT_JOB = "consolidate_experience_event"


def consolidation_dedupe_key(event_id: UUID) -> str:
    return f"experience-event:{event_id}:consolidate"


class ConsolidateExperienceEventJobHandler:
    job_type = CONSOLIDATE_EXPERIENCE_EVENT_JOB

    def __init__(
        self,
        *,
        repository: ExperienceRepository,
        service: ExperienceConsolidationService,
        candidate_lease_seconds: int,
        promotion_job_repository: MemoryJobRepository | None = None,
    ) -> None:
        if candidate_lease_seconds < 1:
            raise ValueError("candidate lease must be positive")
        self._repository = repository
        self._service = service
        self._candidate_lease_seconds = candidate_lease_seconds
        self._promotion_job_repository = promotion_job_repository

    async def prepare(self, job: MemoryJob) -> PreparedConsolidation:
        event_id = _event_id(job)
        session_id = _session_id(job)
        event = await self._repository.begin_consolidation(
            owner_id=job.owner_id,
            session_id=session_id,
            event_id=event_id,
        )
        if event.status == "consolidated":
            if event.consolidation_result is None:
                raise MemoryJobValidationError(
                    "consolidated event is missing its result"
                )
            try:
                result = ConsolidationResult.model_validate(event.consolidation_result)
            except Exception as error:
                raise MemoryJobValidationError(
                    "stored consolidation result is invalid"
                ) from error
            return PreparedConsolidation(
                event_id=event.id,
                owner_id=event.owner_id,
                session_id=event.session_id,
                result=result,
                provider=event.consolidation_provider or self._service.provider,
                model=event.consolidation_model or self._service.model,
            )

        await self._repository.renew_event_candidate_leases(
            owner_id=job.owner_id,
            event_id=event_id,
            expires_at=datetime.now().astimezone()
            + timedelta(seconds=self._candidate_lease_seconds),
        )
        try:
            return await self._service.consolidate(
                owner_id=job.owner_id,
                session_id=session_id,
                event_id=event_id,
            )
        except asyncio.CancelledError:
            raise
        except (
            ConsolidationProviderContractError,
            ConsolidationValidationError,
            ExperienceEventMissingError,
            ValueError,
        ) as error:
            await self._repository.finish_consolidation_failure(
                owner_id=job.owner_id,
                session_id=session_id,
                event_id=event_id,
                terminal=True,
            )
            message = (
                error.safe_diagnostic()
                if isinstance(error, ConsolidationValidationError)
                else str(error)
            )
            raise MemoryJobValidationError(message) from error
        except ConsolidationProviderTransientError as error:
            await self._repository.finish_consolidation_failure(
                owner_id=job.owner_id,
                session_id=session_id,
                event_id=event_id,
                terminal=job.attempt_count >= job.max_attempts,
            )
            raise MemoryJobTransientError(str(error)) from error
        except Exception:
            await self._repository.finish_consolidation_failure(
                owner_id=job.owner_id,
                session_id=session_id,
                event_id=event_id,
                terminal=job.attempt_count >= job.max_attempts,
            )
            raise

    async def commit(
        self,
        job: MemoryJob,
        prepared_result: PreparedConsolidation,
        connection: asyncpg.Connection,
    ) -> Mapping[str, Any]:
        if prepared_result.owner_id != job.owner_id:
            raise MemoryJobValidationError("prepared consolidation owner changed")
        if prepared_result.session_id != _session_id(job):
            raise MemoryJobValidationError("prepared consolidation session changed")
        if prepared_result.event_id != _event_id(job):
            raise MemoryJobValidationError("prepared consolidation event changed")
        consolidated_at = datetime.now().astimezone()
        event = await self._repository.store_consolidation_result(
            owner_id=job.owner_id,
            session_id=prepared_result.session_id,
            event_id=prepared_result.event_id,
            result=consolidation_result_payload(prepared_result.result),
            provider=prepared_result.provider,
            model=prepared_result.model,
            consolidated_at=consolidated_at,
            connection=connection,
        )
        if self._promotion_job_repository is not None:
            from app.memory.episodic.job_handler import (
                PROMOTE_EXPERIENCE_EVENT_JOB,
                promotion_dedupe_key,
            )

            await self._promotion_job_repository.enqueue(
                owner_id=job.owner_id,
                session_id=prepared_result.session_id,
                job_type=PROMOTE_EXPERIENCE_EVENT_JOB,
                dedupe_key=promotion_dedupe_key(prepared_result.event_id),
                payload={"event_id": str(prepared_result.event_id)},
                available_at=consolidated_at,
                connection=connection,
            )
        return {
            "event_id": str(event.id),
            "event_status": event.status,
            "provider": event.consolidation_provider,
            "model": event.consolidation_model,
        }


class ExperienceConsolidationRecoveryRule:
    name = "experience_event_consolidation"

    def __init__(self, *, candidate_lease_seconds: int, limit: int = 100) -> None:
        if candidate_lease_seconds < 1:
            raise ValueError("candidate lease must be positive")
        if limit < 1:
            raise ValueError("recovery limit must be positive")
        self._candidate_lease_seconds = candidate_lease_seconds
        self._limit = limit

    async def reconcile(
        self,
        *,
        repository: MemoryJobRepository,
        connection: asyncpg.Connection,
        now: datetime,
    ) -> int:
        rows = await connection.fetch(
            """
            select events.*,
                   jobs.status as consolidation_job_status
            from experience_events events
            left join lateral (
                select status
                from memory_jobs
                where owner_id = events.owner_id
                  and job_type = 'consolidate_experience_event'
                  and dedupe_key = 'experience-event:' || events.id::text || ':consolidate'
                order by created_at desc, id desc
                limit 1
            ) jobs on true
            where events.consolidation_ready_at is not null
              and events.status in ('closed', 'consolidating')
            order by events.consolidation_ready_at asc, events.id asc
            for update of events skip locked
            limit $1
            """,
            self._limit,
        )
        changed = 0
        lease_expires_at = now + timedelta(seconds=self._candidate_lease_seconds)
        for row in rows:
            status = row["consolidation_job_status"]
            event_id = row["id"]
            if status == "dead":
                if row["status"] != "failed":
                    await connection.execute(
                        """
                        update experience_events
                        set status = 'failed', updated_at = now()
                        where id = $1 and status in ('closed', 'consolidating')
                        """,
                        event_id,
                    )
                    changed += 1
                continue
            if status == "completed":
                await connection.execute(
                    """
                    update experience_events
                    set status = 'failed', updated_at = now()
                    where id = $1 and status in ('closed', 'consolidating')
                    """,
                    event_id,
                )
                changed += 1
                continue
            if row["status"] == "consolidating" and status != "claimed":
                await connection.execute(
                    """
                    update experience_events
                    set status = 'closed', updated_at = now()
                    where id = $1 and status = 'consolidating'
                    """,
                    event_id,
                )
                changed += 1
            if status is None:
                result = await repository.enqueue(
                    owner_id=row["owner_id"],
                    session_id=row["session_id"],
                    job_type=CONSOLIDATE_EXPERIENCE_EVENT_JOB,
                    dedupe_key=consolidation_dedupe_key(event_id),
                    payload={"event_id": str(event_id)},
                    available_at=now,
                    connection=connection,
                )
                changed += int(result.created)
                status = result.job.status
            if status in {"queued", "claimed", "failed"}:
                await connection.execute(
                    """
                    update visual_frame_retention_reasons
                    set expires_at = greatest(expires_at, $2)
                    where reason_type = 'event_candidate'
                      and reason_ref_id = $1
                      and expires_at is not null
                    """,
                    event_id,
                    lease_expires_at,
                )
        return changed


def _event_id(job: MemoryJob) -> UUID:
    raw = job.payload.get("event_id")
    if not isinstance(raw, str):
        raise MemoryJobValidationError("consolidation job is missing event_id")
    try:
        event_id = UUID(raw)
    except ValueError as error:
        raise MemoryJobValidationError(
            "consolidation job event_id is not a UUID"
        ) from error
    if job.dedupe_key != consolidation_dedupe_key(event_id):
        raise MemoryJobValidationError("consolidation job dedupe key is invalid")
    return event_id


def _session_id(job: MemoryJob) -> UUID:
    if job.session_id is None:
        raise MemoryJobValidationError("consolidation job is missing session_id")
    return job.session_id
