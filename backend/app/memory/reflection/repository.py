from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from uuid import UUID

import asyncpg

from app.memory.embeddings.job_handlers import (
    EMBED_MEMORY_ITEM_JOB,
    memory_embedding_dedupe_key,
)
from app.memory.forgetting import (
    procedure_identity_fingerprint,
    reflection_identity_fingerprint,
    suppress_automatic_memory_recreation,
)
from app.memory.jobs.repository import MemoryJobRepository
from app.memory.models import MemoryItem
from app.memory.reflection.contracts import (
    InferredPreferenceCandidate,
    ProcedureMemoryCandidate,
    ReflectionResult,
    candidate_episode_ids,
)
from app.preferences.repository import PreferenceRepository


@dataclass(frozen=True, slots=True)
class ReflectionEpisode:
    id: UUID
    owner_id: str
    session_id: UUID | None
    title: str
    summary: str
    content: str
    occurred_at: datetime | None
    created_at: datetime
    is_explicit: bool


@dataclass(frozen=True, slots=True)
class ReflectionEpisodeRange:
    owner_id: str
    episodes: tuple[ReflectionEpisode, ...]

    @property
    def first(self) -> ReflectionEpisode:
        return self.episodes[0]

    @property
    def last(self) -> ReflectionEpisode:
        return self.episodes[-1]

    @property
    def session_id(self) -> UUID | None:
        return self.last.session_id

    @property
    def episode_ids(self) -> tuple[UUID, ...]:
        return tuple(episode.id for episode in self.episodes)


@dataclass(frozen=True, slots=True)
class ReflectionApplyOutcome:
    reflection_memory_id: UUID
    created: bool
    preference_ids: tuple[UUID, ...]
    procedure_memory_ids: tuple[UUID, ...]
    dropped_candidates: tuple[str, ...]


class ReflectionRepository:
    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def list_episode_owners(
        self,
        *,
        connection: asyncpg.Connection,
    ) -> tuple[str, ...]:
        rows = await connection.fetch(
            """
            select distinct owner_id
            from memory_items
            where memory_type = 'episode' and status = 'active'
            order by owner_id
            """
        )
        return tuple(row["owner_id"] for row in rows)

    async def find_eligible_range(
        self,
        *,
        connection: asyncpg.Connection,
        owner_id: str,
        minimum_episodes: int,
        maximum_episodes: int,
    ) -> ReflectionEpisodeRange | None:
        if minimum_episodes < 1 or maximum_episodes < minimum_episodes:
            raise ValueError("invalid reflection episode thresholds")
        cursor = await connection.fetchrow(
            """
            select
                (metadata->>'range_last_episode_created_at')::timestamptz as created_at,
                (metadata->>'range_last_episode_id')::uuid as episode_id
            from memory_items
            where owner_id = $1
              and memory_type = 'reflection'
              and metadata ? 'range_last_episode_created_at'
              and metadata ? 'range_last_episode_id'
            order by
                (metadata->>'range_last_episode_created_at')::timestamptz desc,
                (metadata->>'range_last_episode_id')::uuid desc
            limit 1
            """,
            owner_id,
        )
        cursor_created_at = cursor["created_at"] if cursor is not None else None
        cursor_id = cursor["episode_id"] if cursor is not None else None
        rows = await connection.fetch(
            """
            select *
            from memory_items
            where owner_id = $1
              and memory_type = 'episode'
              and status = 'active'
              and (
                $2::timestamptz is null
                or (created_at, id) > ($2, $3::uuid)
              )
            order by created_at asc, id asc
            limit $4
            """,
            owner_id,
            cursor_created_at,
            cursor_id,
            maximum_episodes,
        )
        if len(rows) < minimum_episodes:
            return None
        return ReflectionEpisodeRange(
            owner_id=owner_id,
            episodes=tuple(_episode_from_row(row) for row in rows),
        )

    async def load_episode_range(
        self,
        *,
        owner_id: str,
        episode_ids: Sequence[UUID],
        connection: asyncpg.Connection | None = None,
        lock: bool = False,
    ) -> ReflectionEpisodeRange:
        if not episode_ids:
            raise ValueError("reflection episode range cannot be empty")
        if connection is None:
            async with self._pool.acquire() as acquired:
                return await self.load_episode_range(
                    owner_id=owner_id,
                    episode_ids=episode_ids,
                    connection=acquired,
                    lock=lock,
                )
        rows = await connection.fetch(
            f"""
            select *
            from memory_items
            where owner_id = $1
              and memory_type = 'episode'
              and status = 'active'
              and id = any($2::uuid[])
            order by created_at asc, id asc
            {"for update" if lock else ""}
            """,
            owner_id,
            list(episode_ids),
        )
        by_id = {row["id"]: row for row in rows}
        if len(by_id) != len(set(episode_ids)):
            raise ValueError("reflection range contains missing or foreign-owner episodes")
        ordered = tuple(_episode_from_row(by_id[episode_id]) for episode_id in episode_ids)
        if tuple(sorted(ordered, key=lambda item: (item.created_at, item.id))) != ordered:
            raise ValueError("reflection episode range is not durably ordered")
        return ReflectionEpisodeRange(owner_id=owner_id, episodes=ordered)

    async def apply_reflection(
        self,
        *,
        connection: asyncpg.Connection,
        episode_range: ReflectionEpisodeRange,
        result: ReflectionResult,
        preference_repository: PreferenceRepository,
        job_repository: MemoryJobRepository,
        minimum_preference_episodes: int,
        minimum_preference_confidence: float,
        minimum_procedure_episodes: int,
        created_at: datetime,
    ) -> ReflectionApplyOutcome:
        owner_id = episode_range.owner_id
        range_key = reflection_range_key(episode_range.episode_ids)
        await connection.execute(
            "select pg_advisory_xact_lock(hashtextextended($1, 0))",
            f"reflection-apply:{owner_id}:{range_key}",
        )
        existing = await connection.fetchrow(
            """
            select * from memory_items
            where owner_id = $1
              and memory_type = 'reflection'
              and metadata->>'range_key' = $2
            order by created_at desc, id desc
            limit 1
            for update
            """,
            owner_id,
            range_key,
        )
        if existing is not None:
            return ReflectionApplyOutcome(
                reflection_memory_id=existing["id"],
                created=False,
                preference_ids=(),
                procedure_memory_ids=(),
                dropped_candidates=(),
            )

        allowed_ids = frozenset(episode_range.episode_ids)
        supporting_ids = candidate_episode_ids(
            result.supporting_episode_ids,
            allowed_episode_ids=allowed_ids,
        )
        if not supporting_ids:
            raise ValueError("reflection has no valid supporting episodes")
        reflection_fingerprint = reflection_identity_fingerprint(range_key)
        suppressed_reflection_id = await suppress_automatic_memory_recreation(
            connection,
            owner_id=owner_id,
            memory_type="reflection",
            identity_fingerprint=reflection_fingerprint,
        )
        if suppressed_reflection_id is not None:
            return ReflectionApplyOutcome(
                reflection_memory_id=suppressed_reflection_id,
                created=False,
                preference_ids=(),
                procedure_memory_ids=(),
                dropped_candidates=("reflection:suppressed_by_forgetting",),
            )
        reflection_id = await connection.fetchval(
            """
            insert into memory_items (
                owner_id, session_id, memory_type, title, content, summary,
                occurred_at, valid_from, importance, confidence, source,
                is_explicit, metadata
            )
            values (
                $1, $2, 'reflection', $3, $4, $5,
                $6, $7, $8, $9, 'reflection_model', false, $10::jsonb
            )
            returning id
            """,
            owner_id,
            episode_range.session_id,
            result.title,
            result.content,
            result.summary,
            episode_range.last.occurred_at or episode_range.last.created_at,
            created_at,
            result.importance,
            result.confidence,
            _json_dumps(
                {
                    "range_key": range_key,
                    "memory_identity_fingerprint": reflection_fingerprint,
                    "range_first_episode_id": str(episode_range.first.id),
                    "range_last_episode_id": str(episode_range.last.id),
                    "range_last_episode_created_at": episode_range.last.created_at.isoformat(),
                    "episode_ids": [str(value) for value in episode_range.episode_ids],
                    "supporting_episode_ids": [str(value) for value in supporting_ids],
                }
            ),
        )
        await self._attach_episode_evidence(
            connection,
            owner_id=owner_id,
            memory_item_id=reflection_id,
            episode_ids=supporting_ids,
            episodes=episode_range.episodes,
            confidence=result.confidence,
        )
        await _enqueue_memory_embedding(
            connection,
            job_repository=job_repository,
            owner_id=owner_id,
            session_id=episode_range.session_id,
            memory_item_id=reflection_id,
            content=result.content,
            available_at=created_at,
        )

        preference_ids: list[UUID] = []
        procedure_ids: list[UUID] = []
        dropped: list[str] = []
        for index, candidate in enumerate(result.preference_candidates):
            episode_ids = candidate_episode_ids(
                candidate.supporting_episode_ids,
                allowed_episode_ids=allowed_ids,
            )
            reason = _preference_rejection_reason(
                candidate,
                episode_ids=episode_ids,
                minimum_episodes=minimum_preference_episodes,
                minimum_confidence=minimum_preference_confidence,
            )
            if reason is not None:
                dropped.append(f"preference[{index}]:{reason}")
                continue
            assert episode_ids is not None
            write = await preference_repository.set_inferred_preference(
                owner_id=owner_id,
                key=candidate.key,
                value=candidate.value,
                confidence=candidate.confidence,
                evidence_episode_ids=episode_ids,
                connection=connection,
            )
            if write.applied:
                preference_ids.append(write.preference.id)
            else:
                dropped.append(f"preference[{index}]:explicit_preference_authoritative")

        for index, candidate in enumerate(result.procedure_candidates):
            episode_ids = candidate_episode_ids(
                candidate.supporting_episode_ids,
                allowed_episode_ids=allowed_ids,
            )
            reason = _procedure_rejection_reason(
                candidate,
                episode_ids=episode_ids,
                episodes=episode_range.episodes,
                minimum_episodes=minimum_procedure_episodes,
            )
            if reason is not None:
                dropped.append(f"procedure[{index}]:{reason}")
                continue
            assert episode_ids is not None
            procedure_id, embedding_required = await self._upsert_procedure(
                connection,
                owner_id=owner_id,
                session_id=episode_range.session_id,
                candidate=candidate,
                episode_ids=episode_ids,
                episodes=episode_range.episodes,
                supporting_reflection_id=reflection_id,
                created_at=created_at,
            )
            if procedure_id is not None:
                procedure_ids.append(procedure_id)
            if procedure_id is not None and embedding_required:
                await _enqueue_memory_embedding(
                    connection,
                    job_repository=job_repository,
                    owner_id=owner_id,
                    session_id=episode_range.session_id,
                    memory_item_id=procedure_id,
                    content=_procedure_content(candidate),
                    available_at=created_at,
                )

        return ReflectionApplyOutcome(
            reflection_memory_id=reflection_id,
            created=True,
            preference_ids=tuple(preference_ids),
            procedure_memory_ids=tuple(procedure_ids),
            dropped_candidates=tuple(dropped),
        )

    async def _upsert_procedure(
        self,
        connection: asyncpg.Connection,
        *,
        owner_id: str,
        session_id: UUID | None,
        candidate: ProcedureMemoryCandidate,
        episode_ids: tuple[UUID, ...],
        episodes: tuple[ReflectionEpisode, ...],
        supporting_reflection_id: UUID,
        created_at: datetime,
    ) -> tuple[UUID | None, bool]:
        procedure_key = normalize_procedure_key(candidate.procedure_key)
        procedure_fingerprint = procedure_identity_fingerprint(procedure_key)
        await connection.execute(
            "select pg_advisory_xact_lock(hashtextextended($1, 0))",
            f"procedure:{owner_id}:{procedure_key}",
        )
        if await suppress_automatic_memory_recreation(
            connection,
            owner_id=owner_id,
            memory_type="procedure",
            identity_fingerprint=procedure_fingerprint,
        ) is not None:
            return None, False
        active = await connection.fetchrow(
            """
            select * from memory_items
            where owner_id = $1
              and memory_type = 'procedure'
              and status = 'active'
              and metadata->>'procedure_key' = $2
            order by created_at desc, id desc
            limit 1
            for update
            """,
            owner_id,
            procedure_key,
        )
        content = _procedure_content(candidate)
        digest = _procedure_digest(candidate)
        if active is not None and _metadata(active)["content_digest"] == digest:
            await connection.execute(
                """
                update memory_items
                set confidence = greatest(confidence, $3), updated_at = now()
                where owner_id = $1 and id = $2 and status = 'active'
                """,
                owner_id,
                active["id"],
                candidate.confidence,
            )
            await self._attach_episode_evidence(
                connection,
                owner_id=owner_id,
                memory_item_id=active["id"],
                episode_ids=episode_ids,
                episodes=episodes,
                confidence=candidate.confidence,
            )
            return active["id"], False

        version = int(_metadata(active).get("version", 0)) + 1 if active else 1
        procedure_id = await connection.fetchval(
            """
            insert into memory_items (
                owner_id, session_id, memory_type, title, content, summary,
                occurred_at, valid_from, importance, confidence, source,
                is_explicit, metadata
            )
            values (
                $1, $2, 'procedure', $3, $4, $5,
                $6, $6, $7, $7, $8, $9, $10::jsonb
            )
            returning id
            """,
            owner_id,
            session_id,
            candidate.title,
            content,
            candidate.summary,
            created_at,
            candidate.confidence,
            "user_instruction" if candidate.is_explicit else "reflection_model",
            candidate.is_explicit,
            _json_dumps(
                {
                    "procedure_key": procedure_key,
                    "memory_identity_fingerprint": procedure_fingerprint,
                    "version": version,
                    "steps": candidate.steps,
                    "content_digest": digest,
                    "supporting_episode_ids": [str(value) for value in episode_ids],
                    "supporting_reflection_id": str(supporting_reflection_id),
                }
            ),
        )
        await self._attach_episode_evidence(
            connection,
            owner_id=owner_id,
            memory_item_id=procedure_id,
            episode_ids=episode_ids,
            episodes=episodes,
            confidence=candidate.confidence,
        )
        if active is not None:
            await connection.execute(
                """
                update memory_items
                set status = 'superseded', valid_to = $3,
                    superseded_by_memory_id = $4, updated_at = now()
                where owner_id = $1 and id = $2 and status = 'active'
                """,
                owner_id,
                active["id"],
                created_at,
                procedure_id,
            )
            await connection.execute(
                """
                insert into memory_links (
                    owner_id, source_memory_id, target_memory_id, relationship,
                    confidence, valid_from, supporting_memory_id, metadata
                )
                values ($1, $2, $3, 'supersedes', $4, $5, $6, $7::jsonb)
                on conflict (
                    owner_id, source_memory_id, target_memory_id, relationship
                ) where valid_to is null do nothing
                """,
                owner_id,
                procedure_id,
                active["id"],
                candidate.confidence,
                created_at,
                supporting_reflection_id,
                _json_dumps({"reason": "material_procedure_change"}),
            )
        return procedure_id, True

    async def _attach_episode_evidence(
        self,
        connection: asyncpg.Connection,
        *,
        owner_id: str,
        memory_item_id: UUID,
        episode_ids: Sequence[UUID],
        episodes: Sequence[ReflectionEpisode],
        confidence: float,
    ) -> None:
        by_id = {episode.id: episode for episode in episodes}
        for episode_id in episode_ids:
            episode = by_id[episode_id]
            await connection.execute(
                """
                insert into memory_evidence (
                    owner_id, memory_item_id, evidence_type, evidence_ref_id,
                    evidence_time, excerpt, confidence, metadata
                )
                values ($1, $2, 'memory', $3, $4, $5, $6, $7::jsonb)
                on conflict (memory_item_id, evidence_type, evidence_ref_id)
                do nothing
                """,
                owner_id,
                memory_item_id,
                str(episode_id),
                episode.occurred_at or episode.created_at,
                episode.summary[:1_000],
                confidence,
                _json_dumps({"memory_type": "episode"}),
            )


def reflection_range_key(episode_ids: Sequence[UUID]) -> str:
    if not episode_ids:
        raise ValueError("reflection range cannot be empty")
    return f"{episode_ids[0]}:{episode_ids[-1]}"


def normalize_procedure_key(value: str) -> str:
    normalized = " ".join(value.casefold().split())
    if not normalized:
        raise ValueError("procedure key cannot be blank")
    return normalized[:200]


def _preference_rejection_reason(
    candidate: InferredPreferenceCandidate,
    *,
    episode_ids: tuple[UUID, ...] | None,
    minimum_episodes: int,
    minimum_confidence: float,
) -> str | None:
    if episode_ids is None:
        return "invalid_evidence"
    if len(episode_ids) < minimum_episodes:
        return "insufficient_distinct_episodes"
    if candidate.confidence < minimum_confidence:
        return "confidence_below_threshold"
    return None


def _procedure_rejection_reason(
    candidate: ProcedureMemoryCandidate,
    *,
    episode_ids: tuple[UUID, ...] | None,
    episodes: Sequence[ReflectionEpisode],
    minimum_episodes: int,
) -> str | None:
    if episode_ids is None:
        return "invalid_evidence"
    by_id = {episode.id: episode for episode in episodes}
    explicitly_supported = candidate.is_explicit and any(
        by_id[episode_id].is_explicit for episode_id in episode_ids
    )
    if len(episode_ids) < minimum_episodes and not explicitly_supported:
        return "insufficient_distinct_episodes"
    if candidate.is_explicit and not explicitly_supported:
        return "explicit_claim_without_explicit_evidence"
    return None


def _procedure_content(candidate: ProcedureMemoryCandidate) -> str:
    steps = "\n".join(
        f"{index}. {step.strip()}" for index, step in enumerate(candidate.steps, start=1)
    )
    return f"{candidate.title.strip()}\n\n{candidate.summary.strip()}\n\n{steps}"


def _procedure_digest(candidate: ProcedureMemoryCandidate) -> str:
    encoded = json.dumps(
        {
            "title": " ".join(candidate.title.casefold().split()),
            "summary": " ".join(candidate.summary.casefold().split()),
            "steps": [" ".join(step.casefold().split()) for step in candidate.steps],
        },
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


async def _enqueue_memory_embedding(
    connection: asyncpg.Connection,
    *,
    job_repository: MemoryJobRepository,
    owner_id: str,
    session_id: UUID | None,
    memory_item_id: UUID,
    content: str,
    available_at: datetime,
) -> None:
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    await job_repository.enqueue(
        owner_id=owner_id,
        session_id=session_id,
        job_type=EMBED_MEMORY_ITEM_JOB,
        dedupe_key=memory_embedding_dedupe_key(
            memory_item_id,
            content_digest=digest,
        ),
        payload={"memory_item_id": str(memory_item_id)},
        available_at=available_at,
        connection=connection,
    )


def _episode_from_row(row: Mapping[str, Any]) -> ReflectionEpisode:
    return ReflectionEpisode(
        id=row["id"],
        owner_id=row["owner_id"],
        session_id=row["session_id"],
        title=row["title"],
        summary=row["summary"],
        content=row["content"],
        occurred_at=row["occurred_at"],
        created_at=row["created_at"],
        is_explicit=row["is_explicit"],
    )


def _metadata(row: Mapping[str, Any] | None) -> dict[str, Any]:
    if row is None:
        return {}
    value = row["metadata"]
    if isinstance(value, str):
        parsed = json.loads(value)
        return parsed if isinstance(parsed, dict) else {}
    return dict(value)


def _json_dumps(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )
