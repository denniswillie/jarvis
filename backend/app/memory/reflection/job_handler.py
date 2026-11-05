from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Mapping
from uuid import UUID

import asyncpg

from app.memory.consolidation.openai_client import (
    ConsolidationProviderContractError,
    ConsolidationProviderTransientError,
)
from app.memory.jobs.handlers import (
    MemoryJobTransientError,
    MemoryJobValidationError,
)
from app.memory.jobs.models import MemoryJob
from app.memory.jobs.repository import MemoryJobRepository
from app.memory.reflection.contracts import ReflectionResult, ReflectionValidationError
from app.memory.reflection.repository import (
    ReflectionApplyOutcome,
    ReflectionEpisodeRange,
    ReflectionRepository,
    reflection_range_key,
)
from app.memory.reflection.service import (
    ReflectionGenerationService,
    ReflectionTriggerService,
)
from app.preferences.repository import PreferenceRepository


GENERATE_REFLECTION_JOB = "generate_reflection"


def reflection_job_dedupe_key(episode_ids: tuple[UUID, ...]) -> str:
    return f"reflection:{reflection_range_key(episode_ids)}"


@dataclass(frozen=True, slots=True)
class PreparedReflection:
    episode_range: ReflectionEpisodeRange
    result: ReflectionResult


class GenerateReflectionJobHandler:
    job_type = GENERATE_REFLECTION_JOB

    def __init__(
        self,
        *,
        repository: ReflectionRepository,
        service: ReflectionGenerationService,
        preference_repository: PreferenceRepository,
        job_repository: MemoryJobRepository,
        minimum_preference_episodes: int,
        minimum_preference_confidence: float,
        minimum_procedure_episodes: int,
    ) -> None:
        self._repository = repository
        self._service = service
        self._preference_repository = preference_repository
        self._job_repository = job_repository
        self._minimum_preference_episodes = minimum_preference_episodes
        self._minimum_preference_confidence = minimum_preference_confidence
        self._minimum_procedure_episodes = minimum_procedure_episodes

    async def prepare(self, job: MemoryJob) -> PreparedReflection:
        episode_ids = _episode_ids(job)
        try:
            episode_range = await self._repository.load_episode_range(
                owner_id=job.owner_id,
                episode_ids=episode_ids,
            )
            if reflection_range_key(episode_range.episode_ids) != _range_key(job):
                raise ValueError("reflection range key does not match its episodes")
            result = await self._service.generate(episode_range)
            return PreparedReflection(episode_range=episode_range, result=result)
        except (ReflectionValidationError, ValueError) as error:
            raise MemoryJobValidationError(str(error)) from error
        except ConsolidationProviderContractError as error:
            raise MemoryJobValidationError(str(error)) from error
        except ConsolidationProviderTransientError as error:
            raise MemoryJobTransientError(str(error)) from error

    async def commit(
        self,
        job: MemoryJob,
        prepared_result: PreparedReflection,
        connection: asyncpg.Connection,
    ) -> Mapping[str, Any]:
        episode_range = await self._repository.load_episode_range(
            owner_id=job.owner_id,
            episode_ids=_episode_ids(job),
            connection=connection,
            lock=True,
        )
        if episode_range.episode_ids != prepared_result.episode_range.episode_ids:
            raise MemoryJobValidationError("prepared reflection episode range changed")
        outcome = await self._repository.apply_reflection(
            connection=connection,
            episode_range=episode_range,
            result=prepared_result.result,
            preference_repository=self._preference_repository,
            job_repository=self._job_repository,
            minimum_preference_episodes=self._minimum_preference_episodes,
            minimum_preference_confidence=self._minimum_preference_confidence,
            minimum_procedure_episodes=self._minimum_procedure_episodes,
            created_at=datetime.now(UTC),
        )
        return _outcome_payload(outcome)


class ReflectionRecoveryRule:
    name = "reflection_ranges"

    def __init__(self, trigger: ReflectionTriggerService) -> None:
        self._trigger = trigger

    async def reconcile(
        self,
        *,
        repository: MemoryJobRepository,
        connection: asyncpg.Connection,
        now: datetime,
    ) -> int:
        del repository
        return await self._trigger.reconcile_all(connection=connection, now=now)


def _episode_ids(job: MemoryJob) -> tuple[UUID, ...]:
    raw_ids = job.payload.get("episode_ids")
    if not isinstance(raw_ids, list) or not raw_ids:
        raise MemoryJobValidationError("episode_ids must be a non-empty UUID list")
    parsed: list[UUID] = []
    for raw_id in raw_ids:
        if not isinstance(raw_id, str):
            raise MemoryJobValidationError("episode_ids must be a UUID list")
        try:
            parsed.append(UUID(raw_id))
        except ValueError as error:
            raise MemoryJobValidationError("episode_ids must be a UUID list") from error
    if len(set(parsed)) != len(parsed):
        raise MemoryJobValidationError("episode_ids cannot contain duplicates")
    return tuple(parsed)


def _range_key(job: MemoryJob) -> str:
    value = job.payload.get("range_key")
    if not isinstance(value, str) or not value.strip():
        raise MemoryJobValidationError("range_key is required")
    return value.strip()


def _outcome_payload(outcome: ReflectionApplyOutcome) -> dict[str, Any]:
    return {
        "reflection_memory_id": str(outcome.reflection_memory_id),
        "created": outcome.created,
        "preference_ids": [str(value) for value in outcome.preference_ids],
        "procedure_memory_ids": [
            str(value) for value in outcome.procedure_memory_ids
        ],
        "dropped_candidates": list(outcome.dropped_candidates),
    }
