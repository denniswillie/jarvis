from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Mapping, Protocol

import asyncpg

from app.memory.jobs.models import MemoryJobEnqueueResult
from app.memory.jobs.repository import MemoryJobRepository
from app.memory.reflection.contracts import ReflectionResult, ReflectionValidationError, parse_reflection_result
from app.memory.reflection.prompts import (
    REFLECTION_INSTRUCTIONS,
    build_reflection_prompt,
)
from app.memory.reflection.repository import (
    ReflectionEpisodeRange,
    ReflectionRepository,
    reflection_range_key,
)


class ReflectionStructuredOutputClient(Protocol):
    async def generate_json(
        self,
        *,
        prompt: str,
        instructions: str,
        schema_name: str,
        schema: Mapping[str, Any],
        owner_id: str,
        max_output_tokens: int,
    ) -> str: ...


class ReflectionGenerationService:
    def __init__(self, client: ReflectionStructuredOutputClient) -> None:
        self._client = client

    async def generate(self, episode_range: ReflectionEpisodeRange) -> ReflectionResult:
        original_prompt = build_reflection_prompt(episode_range.episodes)
        prompt = original_prompt
        for attempt in range(2):
            raw = await self._client.generate_json(
                prompt=prompt,
                instructions=REFLECTION_INSTRUCTIONS,
                schema_name="jarvis_cross_episode_reflection",
                schema=ReflectionResult.model_json_schema(),
                owner_id=episode_range.owner_id,
                max_output_tokens=4_000,
            )
            try:
                return parse_reflection_result(
                    raw, allowed_episode_ids=frozenset(episode_range.episode_ids),
                )
            except ReflectionValidationError as error:
                if attempt:
                    raise
                # Like consolidation, allow one model correction, never guess
                # an identifier or discard validation to admit the result.
                prompt = (
                    original_prompt + "\n\nCorrection required after validation: " + str(error)
                    + ". Return the complete corrected reflection. Copy supporting episode IDs "
                    "verbatim from the supplied episodes; do not invent or splice identifiers. "
                    "Keep unrelated entities and tasks separate, and omit unsupported claims."
                )
        raise AssertionError("reflection generation exhausted without returning or raising")


class ReflectionTriggerService:
    def __init__(
        self,
        *,
        pool: asyncpg.Pool,
        repository: ReflectionRepository,
        job_repository: MemoryJobRepository,
        minimum_episodes: int,
        maximum_episodes: int,
    ) -> None:
        if minimum_episodes < 1 or maximum_episodes < minimum_episodes:
            raise ValueError("invalid reflection trigger thresholds")
        self._pool = pool
        self._repository = repository
        self._job_repository = job_repository
        self._minimum_episodes = minimum_episodes
        self._maximum_episodes = maximum_episodes

    async def enqueue_if_eligible(
        self,
        *,
        owner_id: str,
        now: datetime | None = None,
        connection: asyncpg.Connection | None = None,
    ) -> MemoryJobEnqueueResult | None:
        effective_now = now or datetime.now(UTC)
        if connection is not None:
            return await self._enqueue_with_connection(
                connection,
                owner_id=owner_id,
                now=effective_now,
            )
        async with self._pool.acquire() as acquired:
            async with acquired.transaction():
                return await self._enqueue_with_connection(
                    acquired,
                    owner_id=owner_id,
                    now=effective_now,
                )

    async def reconcile_all(
        self,
        *,
        connection: asyncpg.Connection,
        now: datetime,
    ) -> int:
        created = 0
        for owner_id in await self._repository.list_episode_owners(
            connection=connection
        ):
            result = await self._enqueue_with_connection(
                connection,
                owner_id=owner_id,
                now=now,
            )
            created += int(result is not None and result.created)
        return created

    async def _enqueue_with_connection(
        self,
        connection: asyncpg.Connection,
        *,
        owner_id: str,
        now: datetime,
    ) -> MemoryJobEnqueueResult | None:
        await connection.execute(
            "select pg_advisory_xact_lock(hashtextextended($1, 0))",
            f"reflection-trigger:{owner_id}",
        )
        episode_range = await self._repository.find_eligible_range(
            connection=connection,
            owner_id=owner_id,
            minimum_episodes=self._minimum_episodes,
            maximum_episodes=self._maximum_episodes,
        )
        if episode_range is None:
            return None
        from app.memory.reflection.job_handler import (
            GENERATE_REFLECTION_JOB,
            reflection_job_dedupe_key,
        )

        return await self._job_repository.enqueue(
            owner_id=owner_id,
            session_id=episode_range.session_id,
            job_type=GENERATE_REFLECTION_JOB,
            dedupe_key=reflection_job_dedupe_key(episode_range.episode_ids),
            payload={
                "episode_ids": [str(value) for value in episode_range.episode_ids],
                "range_key": reflection_range_key(episode_range.episode_ids),
            },
            available_at=now,
            connection=connection,
        )
