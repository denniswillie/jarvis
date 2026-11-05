from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Mapping
from uuid import UUID

import asyncpg

from app.memory.episodic.repository import (
    EpisodePromotionOutcome,
    EpisodicMemoryRepository,
)
from app.memory.episodic.service import (
    EpisodePromotionContext,
    EpisodePromotionPlan,
    EpisodePromotionService,
)
from app.memory.jobs.handlers import MemoryJobValidationError
from app.memory.jobs.models import MemoryJob
from app.memory.jobs.repository import MemoryJobRepository


if TYPE_CHECKING:
    from app.memory.reflection.service import ReflectionTriggerService


PROMOTE_EXPERIENCE_EVENT_JOB = "promote_experience_event"


def promotion_dedupe_key(event_id: UUID) -> str:
    return f"experience-event:{event_id}:promote"


@dataclass(frozen=True, slots=True)
class PreparedEpisodePromotion:
    context: EpisodePromotionContext
    plan: EpisodePromotionPlan


class PromoteExperienceEventJobHandler:
    job_type = PROMOTE_EXPERIENCE_EVENT_JOB

    def __init__(
        self,
        *,
        repository: EpisodicMemoryRepository,
        service: EpisodePromotionService,
        embedding_job_repository: MemoryJobRepository | None,
        reflection_trigger_service: ReflectionTriggerService | None = None,
        ordinary_keyframe_retention_seconds: int,
        skipped_event_retention_days: int,
    ) -> None:
        if ordinary_keyframe_retention_seconds < 1:
            raise ValueError("ordinary keyframe retention must be positive")
        if skipped_event_retention_days < 1:
            raise ValueError("skipped event retention must be positive")
        self._repository = repository
        self._service = service
        self._embedding_job_repository = embedding_job_repository
        self._reflection_trigger_service = reflection_trigger_service
        self._ordinary_keyframe_retention_seconds = (
            ordinary_keyframe_retention_seconds
        )
        self._skipped_event_retention_days = skipped_event_retention_days

    async def prepare(self, job: MemoryJob) -> PreparedEpisodePromotion:
        event_id = _required_uuid(job.payload, "event_id")
        session_id = _required_session_id(job)
        try:
            context = await self._repository.load_promotion_context(
                owner_id=job.owner_id,
                session_id=session_id,
                event_id=event_id,
            )
            return PreparedEpisodePromotion(
                context=context,
                plan=self._service.plan(context),
            )
        except ValueError as error:
            raise MemoryJobValidationError(str(error)) from error

    async def commit(
        self,
        job: MemoryJob,
        prepared_result: PreparedEpisodePromotion,
        connection: asyncpg.Connection,
    ) -> Mapping[str, Any]:
        plan = prepared_result.plan
        if plan.owner_id != job.owner_id:
            raise MemoryJobValidationError("prepared promotion owner changed")
        if plan.session_id != _required_session_id(job):
            raise MemoryJobValidationError("prepared promotion session changed")
        if plan.event_id != _required_uuid(job.payload, "event_id"):
            raise MemoryJobValidationError("prepared promotion event changed")
        decided_at = datetime.now(UTC)
        outcome = await self._repository.apply_promotion(
            context=prepared_result.context,
            plan=plan,
            decided_at=decided_at,
            ordinary_keyframe_retention_seconds=(
                self._ordinary_keyframe_retention_seconds
            ),
            skipped_event_retention_days=self._skipped_event_retention_days,
            connection=connection,
        )
        if self._embedding_job_repository is not None:
            from app.memory.embeddings.job_handlers import (
                EMBED_EXPERIENCE_EVENT_JOB,
                event_embedding_dedupe_key,
            )

            await self._embedding_job_repository.enqueue(
                owner_id=job.owner_id,
                session_id=plan.session_id,
                job_type=EMBED_EXPERIENCE_EVENT_JOB,
                dedupe_key=event_embedding_dedupe_key(plan.event_id),
                payload={"event_id": str(plan.event_id)},
                available_at=decided_at,
                connection=connection,
            )
        if (
            outcome.decision == "promoted"
            and self._reflection_trigger_service is not None
        ):
            await self._reflection_trigger_service.enqueue_if_eligible(
                owner_id=job.owner_id,
                now=decided_at,
                connection=connection,
            )
        return _outcome_payload(outcome)


def _outcome_payload(outcome: EpisodePromotionOutcome) -> dict[str, Any]:
    return {
        "event_id": str(outcome.event_id),
        "promotion_decision": outcome.decision,
        "promotion_reason": outcome.reason,
        "episode_memory_id": (
            str(outcome.episode_memory_id) if outcome.episode_memory_id else None
        ),
        "selected_keyframe_ids": [
            str(frame_id) for frame_id in outcome.selected_keyframe_ids
        ],
    }


def _required_session_id(job: MemoryJob) -> UUID:
    if job.session_id is None:
        raise MemoryJobValidationError("episode promotion job requires a session")
    return job.session_id


def _required_uuid(payload: Mapping[str, Any], key: str) -> UUID:
    value = payload.get(key)
    if not isinstance(value, str):
        raise MemoryJobValidationError(f"{key} must be a UUID string")
    try:
        return UUID(value)
    except ValueError as error:
        raise MemoryJobValidationError(f"{key} must be a UUID string") from error
