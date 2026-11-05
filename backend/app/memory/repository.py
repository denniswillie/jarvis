from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, cast
from uuid import UUID

import asyncpg

from app.memory.jobs.models import normalize_job_json_object
from app.memory.forgetting import (
    memory_identity_fingerprint,
    record_explicit_memory_override,
    suppress_automatic_memory_recreation,
)
from app.memory.models import (
    EMBEDDING_DIMENSIONS,
    MEMORY_EVIDENCE_TYPES,
    MEMORY_TYPES,
    CalendarEventSnapshot,
    MemoryEmbedding,
    MemoryEvidence,
    MemoryEvidenceDetail,
    MemoryEvidenceType,
    MemoryItem,
    MemoryLink,
    MemoryType,
    ExplicitMemoryEvidenceInput,
    ExplicitMemoryWriteResult,
    ForgetMemoryResult,
    MemoryForgettingTombstone,
    SemanticFactHistoryEntry,
    content_hash,
    json_object_hash,
    normalize_embedding,
)
from app.memory.retrieval.models import (
    RETRIEVAL_RECORD_TYPES,
    RetrievalDocument,
    RetrievalRecordType,
    RetrievalSourceHit,
)
from app.memory.retrieval.query import NormalizedMemoryRetrievalQuery
from app.memory.semantic.contracts import ENTITY_IDENTITY_EMBEDDING_KIND

if TYPE_CHECKING:
    from app.memory.jobs.repository import MemoryJobRepository


class MemoryRepositoryError(Exception):
    pass


class MemoryMissingError(MemoryRepositoryError):
    pass


class MemoryConflictError(MemoryRepositoryError):
    pass


class MemoryScopeError(MemoryRepositoryError):
    pass


class ExperiencePromotionConflictError(MemoryRepositoryError):
    pass


_LOCAL_EVIDENCE_TABLES: dict[str, str] = {
    "frame": "visual_frames",
    "observation": "perception_observations",
    "experience_event": "experience_events",
    "message": "jarvis_messages",
    "tool_execution": "jarvis_tool_executions",
    "reminder": "jarvis_reminders",
    "calendar_event": "calendar_event_snapshots",
    "file": "jarvis_files",
    "memory": "memory_items",
}

_RETRIEVAL_AUDIT_MAX_CANDIDATES = 200
_RETRIEVAL_AUDIT_MAX_RESULTS = 50
_RETRIEVAL_AUDIT_MAX_PROVIDER_SPACES = 16
_RETRIEVAL_AUDIT_MAX_PROVIDER_SPACE_BYTES = 256
_RETRIEVAL_AUDIT_MAX_PROVIDER_SPACES_BYTES = 4096
_RETRIEVAL_AUDIT_PROVIDER_SPACE_PATTERN = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,255}$"
)
_RETRIEVAL_EMBEDDING_KIND_LIMIT = _RETRIEVAL_AUDIT_MAX_PROVIDER_SPACES - 2
_RETRIEVAL_AUDIT_FORBIDDEN_KEY_PARTS = (
    "api_key",
    "authorization",
    "base64",
    "bytes",
    "image",
    "password",
    "secret",
    "signed_url",
    "token",
    "url",
)

_RETRIEVAL_DOCUMENTS_CTE = """
with retrieval_documents as (
    select
        event.id,
        event.owner_id,
        event.session_id,
        'experience_event'::text as record_type,
        event.started_at as occurred_at,
        event.started_at as valid_from,
        event.ended_at as valid_to,
        event.title,
        event.summary,
        -- Tags remain in search_text and embeddings, not in returned evidence content.
        concat_ws(' ', event.scene, event.activity, (event.state - 'tags')::text) as content,
        event.importance,
        event.confidence,
        case when event.status = 'open' then 'active' else 'historical' end
            as record_status,
        false as is_explicit,
        (
            select count(*)::integer
            from experience_event_evidence evidence
            where evidence.event_id = event.id
        ) as evidence_count,
        (
            select count(*)::integer
            from experience_event_keyframes keyframe
            where keyframe.event_id = event.id
        ) as keyframe_count,
        coalesce(
            (
                select array_agg(distinct evidence.evidence_type order by evidence.evidence_type)
                from experience_event_evidence evidence
                where evidence.event_id = event.id
            ),
            '{}'::text[]
        ) as evidence_types,
        (
            select count(*)::integer
            from memory_retrieval_runs retrieval
            where retrieval.owner_id = event.owner_id
              and retrieval.result_ids @> array[event.id]
        ) as retrieval_count,
        lower(concat_ws(' ', event.title, event.summary, event.scene,
            event.activity, event.state::text)) as search_text
    from experience_events event

    union all

    select
        memory.id,
        memory.owner_id,
        memory.session_id,
        memory.memory_type as record_type,
        coalesce(memory.occurred_at, memory.valid_from, memory.created_at) as occurred_at,
        memory.valid_from,
        memory.valid_to,
        memory.title,
        memory.summary,
        memory.content,
        memory.importance,
        memory.confidence,
        memory.status as record_status,
        memory.is_explicit,
        (
            select count(*)::integer
            from memory_evidence evidence
            where evidence.memory_item_id = memory.id
        ) as evidence_count,
        (
            select count(*)::integer
            from memory_evidence evidence
            where evidence.memory_item_id = memory.id
              and evidence.evidence_type = 'frame'
        ) as keyframe_count,
        coalesce(
            (
                select array_agg(distinct evidence.evidence_type order by evidence.evidence_type)
                from memory_evidence evidence
                where evidence.memory_item_id = memory.id
            ),
            '{}'::text[]
        ) as evidence_types,
        (
            select count(*)::integer
            from memory_retrieval_runs retrieval
            where retrieval.owner_id = memory.owner_id
              and retrieval.result_ids @> array[memory.id]
        ) as retrieval_count,
        lower(concat_ws(' ', memory.title, memory.summary, memory.content,
            memory.metadata::text)) as search_text
    from memory_items memory
    where memory.status <> 'forgotten'
)
"""


def _lexical_tokens(value: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(re.findall(r"[^\W_]+", value, flags=re.UNICODE)))[:32]


def _retrieval_lifecycle_predicate(alias: str) -> str:
    if alias not in {"document", "candidate"}:
        raise ValueError("unsupported retrieval SQL alias")
    return f"""
      and (
          {alias}.record_type = 'experience_event'
          or (
              ($6 and {alias}.record_status in ('active', 'superseded', 'retracted'))
              or (
                  not $6
                  and {alias}.record_status = 'active'
                  and ({alias}.valid_from is null or {alias}.valid_from <= now())
                  and ({alias}.record_type = 'episode' or {alias}.valid_to is null or {alias}.valid_to > now())
              )
          )
      )
    """


class MemoryRepository:
    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def get_memory_item(
        self,
        *,
        owner_id: str,
        memory_item_id: UUID,
    ) -> MemoryItem | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "select * from memory_items where owner_id = $1 and id = $2",
                _require_text(owner_id, "owner_id"),
                memory_item_id,
            )
        return _memory_from_row(row) if row is not None else None

    async def list_recent_memory_items(
        self,
        *,
        owner_id: str,
        limit: int,
        memory_types: Sequence[MemoryType] = (),
        session_id: UUID | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        cursor_time: datetime | None = None,
        cursor_id: UUID | None = None,
    ) -> list[MemoryItem]:
        if isinstance(limit, bool) or limit < 1 or limit > 101:
            raise ValueError("limit must be between 1 and 101")
        if (cursor_time is None) != (cursor_id is None):
            raise ValueError("cursor_time and cursor_id must be provided together")
        if start_time is not None and end_time is not None and end_time <= start_time:
            raise ValueError("end_time must be after start_time")
        normalized_types = tuple(
            dict.fromkeys(_normalize_memory_type(value) for value in memory_types)
        )
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                select *
                from memory_items
                where owner_id = $1
                  and status <> 'forgotten'
                  and ($2::text[] is null or memory_type = any($2::text[]))
                  and ($3::uuid is null or session_id = $3)
                  and (
                    $4::timestamptz is null
                    or coalesce(occurred_at, created_at) >= $4
                  )
                  and (
                    $5::timestamptz is null
                    or coalesce(occurred_at, created_at) < $5
                  )
                  and (
                    $6::timestamptz is null
                    or coalesce(occurred_at, created_at) < $6
                    or (
                        coalesce(occurred_at, created_at) = $6
                        and id < $7
                    )
                  )
                order by coalesce(occurred_at, created_at) desc, id desc
                limit $8
                """,
                _require_text(owner_id, "owner_id"),
                list(normalized_types) if normalized_types else None,
                session_id,
                start_time,
                end_time,
                cursor_time,
                cursor_id,
                limit,
            )
        return [_memory_from_row(row) for row in rows]

    async def list_recent_episodes(
        self,
        *,
        owner_id: str,
        limit: int = 50,
    ) -> list[MemoryItem]:
        if limit < 1:
            raise ValueError("limit must be positive")
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                select *
                from memory_items
                where owner_id = $1 and memory_type = 'episode'
                  and status <> 'forgotten'
                order by occurred_at desc nulls last, created_at desc, id desc
                limit $2
                """,
                _require_text(owner_id, "owner_id"),
                limit,
            )
        return [_memory_from_row(row) for row in rows]

    async def list_active_semantic_facts(
        self,
        *,
        owner_id: str,
        limit: int = 50,
    ) -> list[MemoryItem]:
        if limit < 1:
            raise ValueError("limit must be positive")
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                select *
                from memory_items
                where owner_id = $1
                  and memory_type = 'semantic_fact'
                  and status = 'active'
                  and (valid_from is null or valid_from <= now())
                  and (valid_to is null or valid_to > now())
                order by occurred_at desc nulls last, created_at desc, id desc
                limit $2
                """,
                _require_text(owner_id, "owner_id"),
                limit,
            )
        return [_memory_from_row(row) for row in rows]

    async def list_semantic_fact_context(
        self, *, owner_id: str, at: datetime, limit: int = 40, query_text: str = "",
    ) -> list[Mapping[str, Any]]:
        """Bounded owner-only identity/state context, including known successors."""
        if not 1 <= limit <= 81:
            raise ValueError("semantic context limit must be between one and 81")
        if at.tzinfo is None or at.utcoffset() is None:
            raise ValueError("semantic context time must be timezone-aware")
        if len(query_text.encode("utf-8")) > 8192:
            raise ValueError("semantic context query exceeds 8192 bytes")
        normalized_query = query_text.strip().lower()
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                select entity.canonical_name as subject, entity.entity_type,
                       memory.metadata->>'predicate' as predicate,
                       memory.metadata->>'value' as value,
                       coalesce(memory.metadata->>'temporal_scope', 'durable') as temporal_scope,
                       memory.valid_from, memory.valid_to, memory.status,
                       memory.confidence, memory.is_explicit
                from memory_items memory
                join memory_entities entity
                  on entity.owner_id = memory.owner_id
                 and entity.id::text = memory.metadata->>'subject_entity_id'
                cross join lateral (
                    select count(*) as token_hits from unnest($5::text[]) token
                    where position(token in lower(entity.canonical_name)) > 0
                ) relevance
                where memory.owner_id = $1
                  and memory.memory_type = 'semantic_fact'
                  and memory.status in ('active', 'superseded', 'retracted')
                  and jsonb_typeof(memory.metadata->'value') = 'string'
                  and memory.metadata ? 'predicate'
                  and char_length(memory.metadata->>'value') between 1 and 1000
                  and char_length(entity.canonical_name) between 1 and 160
                  and coalesce(memory.metadata->>'temporal_scope', 'durable') <> 'event'
                order by
                  case when $4 <> '' and position(lower(entity.canonical_name) in $4) > 0
                       then 1 else 0 end desc,
                  relevance.token_hits desc,
                  case when (memory.valid_from is null or memory.valid_from <= $2)
                         and (memory.valid_to is null or memory.valid_to > $2)
                       then 0 else 1 end,
                  abs(extract(epoch from (coalesce(memory.valid_from, memory.occurred_at, memory.created_at) - $2))),
                  memory.id
                limit $3
                """,
                _require_text(owner_id, "owner_id"), at, limit,
                normalized_query, list(_lexical_tokens(normalized_query)),
            )
        return [dict(row) for row in rows]

    async def list_recent_reflections(
        self,
        *,
        owner_id: str,
        limit: int = 20,
    ) -> list[MemoryItem]:
        return await self._list_active_memory_type(
            owner_id=owner_id,
            memory_type="reflection",
            limit=limit,
        )

    async def list_active_procedures(
        self,
        *,
        owner_id: str,
        limit: int = 50,
    ) -> list[MemoryItem]:
        return await self._list_active_memory_type(
            owner_id=owner_id,
            memory_type="procedure",
            limit=limit,
        )

    async def list_active_goals(
        self,
        *,
        owner_id: str,
        limit: int = 20,
    ) -> list[MemoryItem]:
        return await self._list_active_memory_type(
            owner_id=owner_id,
            memory_type="goal",
            limit=limit,
        )

    async def _list_active_memory_type(
        self,
        *,
        owner_id: str,
        memory_type: MemoryType,
        limit: int,
    ) -> list[MemoryItem]:
        if memory_type not in {"reflection", "procedure", "goal"}:
            raise ValueError(
                "active memory listing supports reflections, procedures, and goals"
            )
        if limit < 1:
            raise ValueError("limit must be positive")
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                select *
                from memory_items
                where owner_id = $1
                  and memory_type = $2
                  and status = 'active'
                  and (valid_from is null or valid_from <= now())
                  and (valid_to is null or valid_to > now())
                order by occurred_at desc nulls last, created_at desc, id desc
                limit $3
                """,
                _require_text(owner_id, "owner_id"),
                memory_type,
                limit,
            )
        return [_memory_from_row(row) for row in rows]

    async def list_semantic_fact_history(
        self,
        *,
        owner_id: str,
        subject_entity_id: UUID,
        predicate: str,
        limit: int = 50,
    ) -> list[SemanticFactHistoryEntry]:
        if limit < 1:
            raise ValueError("limit must be positive")
        normalized_owner_id = _require_text(owner_id, "owner_id")
        normalized_predicate = _require_text(predicate, "predicate")
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                select *
                from memory_items
                where owner_id = $1
                  and memory_type = 'semantic_fact'
                  and metadata->>'subject_entity_id' = $2
                  and metadata->>'predicate' = $3
                order by valid_from desc nulls last, created_at desc, id desc
                limit $4
                """,
                normalized_owner_id,
                str(subject_entity_id),
                normalized_predicate,
                limit,
            )
            fact_ids = [row["id"] for row in rows]
            evidence_rows = (
                await conn.fetch(
                    """
                    select *
                    from memory_evidence
                    where owner_id = $1
                      and memory_item_id = any($2::uuid[])
                    order by evidence_time asc, created_at asc, id asc
                    """,
                    normalized_owner_id,
                    fact_ids,
                )
                if fact_ids
                else ()
            )
            link_rows = (
                await conn.fetch(
                    """
                    select *
                    from memory_links
                    where owner_id = $1
                      and (
                        source_memory_id = any($2::uuid[])
                        or target_memory_id = any($2::uuid[])
                      )
                    order by valid_from asc, created_at asc, id asc
                    """,
                    normalized_owner_id,
                    fact_ids,
                )
                if fact_ids
                else ()
            )

        evidence_by_fact: dict[UUID, list[MemoryEvidence]] = {
            fact_id: [] for fact_id in fact_ids
        }
        for evidence_row in evidence_rows:
            evidence = _evidence_from_row(evidence_row)
            evidence_by_fact[evidence.memory_item_id].append(evidence)

        links_by_fact: dict[UUID, list[MemoryLink]] = {
            fact_id: [] for fact_id in fact_ids
        }
        for link_row in link_rows:
            link = _link_from_row(link_row)
            if link.source_memory_id in links_by_fact:
                links_by_fact[link.source_memory_id].append(link)
            if (
                link.target_memory_id in links_by_fact
                and link.target_memory_id != link.source_memory_id
            ):
                links_by_fact[link.target_memory_id].append(link)

        return [
            SemanticFactHistoryEntry(
                fact=_memory_from_row(row),
                evidence=tuple(evidence_by_fact[row["id"]]),
                links=tuple(links_by_fact[row["id"]]),
            )
            for row in rows
        ]

    async def list_session_episodes(
        self,
        *,
        owner_id: str,
        session_id: UUID,
    ) -> list[MemoryItem]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                select *
                from memory_items
                where owner_id = $1
                  and session_id = $2
                  and memory_type = 'episode'
                  and status <> 'forgotten'
                order by occurred_at asc nulls last, created_at asc, id asc
                """,
                _require_text(owner_id, "owner_id"),
                session_id,
            )
        return [_memory_from_row(row) for row in rows]

    async def list_episodes_in_range(
        self,
        *,
        owner_id: str,
        start_time: datetime,
        end_time: datetime,
        limit: int = 200,
    ) -> list[MemoryItem]:
        if end_time <= start_time:
            raise ValueError("end_time must be after start_time")
        if limit < 1:
            raise ValueError("limit must be positive")
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                select *
                from memory_items
                where owner_id = $1
                  and memory_type = 'episode'
                  and status <> 'forgotten'
                  and occurred_at >= $2
                  and occurred_at < $3
                order by occurred_at asc, id asc
                limit $4
                """,
                _require_text(owner_id, "owner_id"),
                start_time,
                end_time,
                limit,
            )
        return [_memory_from_row(row) for row in rows]

    async def list_memory_evidence(
        self,
        *,
        owner_id: str,
        memory_item_id: UUID,
    ) -> list[MemoryEvidence]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                select evidence.*
                from memory_evidence evidence
                join memory_items memory on memory.id = evidence.memory_item_id
                where memory.owner_id = $1 and memory.id = $2
                order by evidence.evidence_time asc, evidence.id asc
                """,
                _require_text(owner_id, "owner_id"),
                memory_item_id,
            )
        return [_evidence_from_row(row) for row in rows]

    async def list_memory_evidence_details(
        self,
        *,
        owner_id: str,
        memory_item_id: UUID,
        limit: int = 100,
    ) -> list[MemoryEvidenceDetail]:
        if isinstance(limit, bool) or limit < 1 or limit > 101:
            raise ValueError("limit must be between 1 and 101")
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                select
                    evidence.*,
                    case evidence.evidence_type
                        when 'frame' then exists (
                            select 1 from visual_frames source
                            where source.owner_id = $1
                              and source.id::text = evidence.evidence_ref_id
                        )
                        when 'observation' then exists (
                            select 1 from perception_observations source
                            where source.owner_id = $1
                              and source.id::text = evidence.evidence_ref_id
                        )
                        when 'experience_event' then exists (
                            select 1 from experience_events source
                            where source.owner_id = $1
                              and source.id::text = evidence.evidence_ref_id
                        )
                        when 'message' then exists (
                            select 1 from jarvis_messages source
                            where source.owner_id = $1
                              and source.id::text = evidence.evidence_ref_id
                        )
                        when 'tool_execution' then exists (
                            select 1 from jarvis_tool_executions source
                            where source.owner_id = $1
                              and source.id::text = evidence.evidence_ref_id
                        )
                        when 'reminder' then exists (
                            select 1 from jarvis_reminders source
                            where source.owner_id = $1
                              and source.id::text = evidence.evidence_ref_id
                        )
                        when 'calendar_event' then exists (
                            select 1 from calendar_event_snapshots source
                            where source.owner_id = $1
                              and source.id::text = evidence.evidence_ref_id
                        )
                        when 'file' then exists (
                            select 1 from jarvis_files source
                            where source.owner_id = $1
                              and source.id::text = evidence.evidence_ref_id
                        )
                        when 'memory' then exists (
                            select 1 from memory_items source
                            where source.owner_id = $1
                              and source.id::text = evidence.evidence_ref_id
                        )
                        else false
                    end as source_available,
                    case
                        when evidence.evidence_type <> 'frame' then null
                        else exists (
                            select 1
                            from visual_frames frame
                            where frame.owner_id = $1
                              and frame.id::text = evidence.evidence_ref_id
                              and frame.raw_image_status = 'available'
                              and (
                                frame.raw_image_expires_at > now()
                                or exists (
                                    select 1
                                    from visual_frame_retention_reasons reason
                                    where reason.owner_id = $1
                                      and reason.visual_frame_id = frame.id
                                      and (
                                        reason.expires_at is null
                                        or reason.expires_at > now()
                                      )
                                )
                              )
                        )
                    end as retained_image_available
                from memory_evidence evidence
                join memory_items memory on memory.id = evidence.memory_item_id
                where memory.owner_id = $1 and memory.id = $2
                order by evidence.evidence_time asc, evidence.id asc
                limit $3
                """,
                _require_text(owner_id, "owner_id"),
                memory_item_id,
                limit,
            )
        return [
            MemoryEvidenceDetail(
                evidence=_evidence_from_row(row),
                source_available=bool(row["source_available"]),
                retained_image_available=row["retained_image_available"],
            )
            for row in rows
        ]

    async def list_memory_links(
        self,
        *,
        owner_id: str,
        memory_item_id: UUID,
        limit: int = 50,
    ) -> list[MemoryLink]:
        if isinstance(limit, bool) or limit < 1 or limit > 100:
            raise ValueError("limit must be between 1 and 100")
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                select link.*
                from memory_links link
                join memory_items memory on memory.id = $2
                where memory.owner_id = $1
                  and link.owner_id = $1
                  and (
                    link.source_memory_id = memory.id
                    or link.target_memory_id = memory.id
                  )
                order by link.created_at desc, link.id desc
                limit $3
                """,
                _require_text(owner_id, "owner_id"),
                memory_item_id,
                limit,
            )
        return [_link_from_row(row) for row in rows]

    async def list_memories_superseded_by(
        self,
        *,
        owner_id: str,
        replacement_memory_id: UUID,
        limit: int = 20,
    ) -> list[MemoryItem]:
        if isinstance(limit, bool) or limit < 1 or limit > 50:
            raise ValueError("limit must be between 1 and 50")
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                select *
                from memory_items
                where owner_id = $1 and superseded_by_memory_id = $2
                order by updated_at desc, id desc
                limit $3
                """,
                _require_text(owner_id, "owner_id"),
                replacement_memory_id,
                limit,
            )
        return [_memory_from_row(row) for row in rows]

    async def create_memory_item(
        self,
        *,
        owner_id: str,
        session_id: UUID | None,
        memory_type: MemoryType,
        title: str,
        content: str,
        summary: str = "",
        source_experience_event_id: UUID | None = None,
        occurred_at: datetime | None = None,
        valid_from: datetime | None = None,
        valid_to: datetime | None = None,
        importance: float = 0.5,
        confidence: float = 0.5,
        source: str,
        is_explicit: bool = False,
        metadata: Mapping[str, Any] | None = None,
    ) -> MemoryItem:
        normalized_owner_id = _require_text(owner_id, "owner_id")
        normalized_type = _normalize_memory_type(memory_type)
        if normalized_type == "episode":
            raise ValueError("episode memories must be created with promote_event_to_episode")
        normalized_title = _require_text(title, "title")
        normalized_content = _require_text(content, "content")
        normalized_source = _require_text(source, "source")
        _require_validity(valid_from, valid_to)
        _require_score(importance, "importance")
        _require_score(confidence, "confidence")
        safe_metadata = _normalize_metadata(metadata)
        identity_fingerprint = memory_identity_fingerprint(
            memory_type=normalized_type,
            content=normalized_content,
            metadata=safe_metadata,
            source_experience_event_id=source_experience_event_id,
        )
        safe_metadata["memory_identity_fingerprint"] = identity_fingerprint

        async with self._pool.acquire() as conn:
            async with conn.transaction():
                suppressed_memory_id = (
                    await suppress_automatic_memory_recreation(
                        conn,
                        owner_id=normalized_owner_id,
                        memory_type=normalized_type,
                        identity_fingerprint=identity_fingerprint,
                    )
                    if not is_explicit
                    else None
                )
                if suppressed_memory_id is not None:
                    suppressed_row = await conn.fetchrow(
                        "select * from memory_items where owner_id = $1 and id = $2",
                        normalized_owner_id,
                        suppressed_memory_id,
                    )
                    if suppressed_row is None:
                        raise MemoryConflictError("forgotten memory tombstone is orphaned")
                    return _memory_from_row(suppressed_row)
                row = await conn.fetchrow(
                    """
                    insert into memory_items (
                        owner_id,
                        session_id,
                        memory_type,
                        title,
                        content,
                        summary,
                        source_experience_event_id,
                        occurred_at,
                        valid_from,
                        valid_to,
                        importance,
                        confidence,
                        source,
                        is_explicit,
                        metadata
                    )
                    select $1, $2, $3, $4, $5, $6, $7, $8, $9, $10,
                           $11, $12, $13, $14, $15::jsonb
                    where (
                            $2::uuid is null
                            or exists (
                                select 1 from jarvis_sessions
                                where owner_id = $1 and id = $2
                            )
                        )
                      and (
                            $7::uuid is null
                            or exists (
                                select 1 from experience_events
                                where owner_id = $1
                                  and id = $7
                                  and ($2::uuid is null or session_id = $2)
                            )
                        )
                    returning *
                    """,
                    normalized_owner_id,
                    session_id,
                    normalized_type,
                    normalized_title,
                    normalized_content,
                    summary.strip(),
                    source_experience_event_id,
                    occurred_at,
                    valid_from,
                    valid_to,
                    importance,
                    confidence,
                    normalized_source,
                    is_explicit,
                    _json_dumps(safe_metadata),
                )
                if row is not None and is_explicit:
                    await record_explicit_memory_override(
                        conn,
                        owner_id=normalized_owner_id,
                        memory_type=normalized_type,
                        identity_fingerprint=identity_fingerprint,
                        override_memory_id=row["id"],
                        metadata={"source": normalized_source},
                    )
        if row is None:
            raise MemoryScopeError("session or source event is outside the owner scope")
        return _memory_from_row(row)

    async def create_explicit_memory(
        self,
        *,
        owner_id: str,
        session_id: UUID | None,
        memory_type: MemoryType,
        title: str,
        content: str,
        occurred_at: datetime,
        idempotency_key: str,
        evidence: Sequence[ExplicitMemoryEvidenceInput],
        entity_ids: Sequence[UUID] = (),
        metadata: Mapping[str, Any] | None = None,
        supersedes_memory_id: UUID | None = None,
        job_repository: MemoryJobRepository | None = None,
        warnings: Sequence[str] = (),
    ) -> ExplicitMemoryWriteResult:
        normalized_owner_id = _require_text(owner_id, "owner_id")
        normalized_type = _normalize_memory_type(memory_type)
        if normalized_type == "episode":
            raise ValueError("explicit experiences are stored as reflection memories")
        normalized_title = _require_text(title, "title")
        normalized_content = _require_text(content, "content")
        normalized_key = _require_text(idempotency_key, "idempotency_key")
        safe_metadata = _normalize_metadata(metadata)
        safe_metadata["explicit_idempotency_key"] = normalized_key
        identity_fingerprint = memory_identity_fingerprint(
            memory_type=normalized_type,
            content=normalized_content,
            metadata=safe_metadata,
        )
        safe_metadata["memory_identity_fingerprint"] = identity_fingerprint
        normalized_entity_ids = tuple(dict.fromkeys(entity_ids))
        result_warnings = list(warnings)
        embedding_dedupe_key: str | None = None

        async with self._pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    "select pg_advisory_xact_lock(hashtextextended($1, 0))",
                    f"{normalized_owner_id}:explicit-memory:{normalized_key}",
                )
                existing_row = await conn.fetchrow(
                    """
                    select *
                    from memory_items
                    where owner_id = $1
                      and metadata->>'explicit_idempotency_key' = $2
                    order by created_at desc, id desc
                    limit 1
                    for update
                    """,
                    normalized_owner_id,
                    normalized_key,
                )
                if existing_row is not None:
                    existing = _memory_from_row(existing_row)
                    evidence_rows = await conn.fetch(
                        """
                        select *
                        from memory_evidence
                        where owner_id = $1 and memory_item_id = $2
                        order by evidence_time asc, id asc
                        """,
                        normalized_owner_id,
                        existing.id,
                    )
                    embedding_dedupe_key = _memory_embedding_dedupe_key(existing.id)
                    indexing_status = await conn.fetchval(
                        """
                        select status
                        from memory_jobs
                        where owner_id = $1
                          and job_type = 'embed_memory_item'
                          and dedupe_key = $2
                        order by created_at desc, id desc
                        limit 1
                        """,
                        normalized_owner_id,
                        embedding_dedupe_key,
                    )
                    return ExplicitMemoryWriteResult(
                        memory=existing,
                        evidence=tuple(_evidence_from_row(row) for row in evidence_rows),
                        indexing_status=str(indexing_status or "not_queued"),
                        replayed=True,
                        warnings=tuple(warnings),
                    )

                memory_row = await conn.fetchrow(
                    """
                    insert into memory_items (
                        owner_id, session_id, memory_type, title, content, summary,
                        occurred_at, valid_from, importance, confidence, source,
                        is_explicit, metadata
                    )
                    select $1, $2, $3, $4, $5, $5, $6, $6, 0.85, 1.0,
                           'explicit_agent_memory', true, $7::jsonb
                    where $2::uuid is null
                       or exists (
                            select 1 from jarvis_sessions
                            where owner_id = $1 and id = $2
                       )
                    returning *
                    """,
                    normalized_owner_id,
                    session_id,
                    normalized_type,
                    normalized_title,
                    normalized_content,
                    occurred_at,
                    _json_dumps(safe_metadata),
                )
                if memory_row is None:
                    raise MemoryScopeError("session is outside the owner scope")
                memory = _memory_from_row(memory_row)

                attached: list[MemoryEvidence] = []
                for item in evidence:
                    normalized_evidence_type = _normalize_evidence_type(
                        item.evidence_type
                    )
                    normalized_ref = _require_text(
                        item.evidence_ref_id,
                        "evidence_ref_id",
                    )
                    await _validate_evidence_source(
                        conn,
                        owner_id=normalized_owner_id,
                        evidence_type=normalized_evidence_type,
                        evidence_ref_id=normalized_ref,
                    )
                    evidence_row = await conn.fetchrow(
                        """
                        insert into memory_evidence (
                            owner_id, memory_item_id, evidence_type,
                            evidence_ref_id, evidence_time, excerpt,
                            confidence, metadata
                        )
                        values ($1, $2, $3, $4, $5, $6, $7, $8::jsonb)
                        on conflict (memory_item_id, evidence_type, evidence_ref_id)
                        do nothing
                        returning *
                        """,
                        normalized_owner_id,
                        memory.id,
                        normalized_evidence_type,
                        normalized_ref,
                        item.evidence_time,
                        item.excerpt.strip(),
                        item.confidence,
                        _json_dumps(_normalize_metadata(item.metadata)),
                    )
                    if evidence_row is not None:
                        attached.append(_evidence_from_row(evidence_row))
                    if normalized_evidence_type == "frame":
                        await conn.execute(
                            """
                            insert into visual_frame_retention_reasons (
                                owner_id, visual_frame_id, reason_type,
                                reason_ref_id, expires_at
                            )
                            select $1, frame.id, 'explicit_memory', $3, null
                            from visual_frames frame
                            where frame.owner_id = $1 and frame.id::text = $2
                            on conflict (
                                visual_frame_id, reason_type, reason_ref_id
                            ) do nothing
                            """,
                            normalized_owner_id,
                            normalized_ref,
                            memory.id,
                        )
                    elif normalized_evidence_type == "experience_event":
                        await conn.execute(
                            """
                            insert into experience_event_retention_reasons (
                                owner_id, experience_event_id, reason_type,
                                reason_ref_id, expires_at
                            )
                            select $1, event.id, 'explicit_memory', $3::text, null
                            from experience_events event
                            where event.owner_id = $1 and event.id::text = $2
                            on conflict (
                                experience_event_id, reason_type, reason_ref_id
                            ) do nothing
                            """,
                            normalized_owner_id,
                            normalized_ref,
                            str(memory.id),
                        )

                for entity_id in normalized_entity_ids:
                    owned_entity = await conn.fetchval(
                        """
                        select true
                        from memory_entities
                        where owner_id = $1 and id = $2
                        for key share
                        """,
                        normalized_owner_id,
                        entity_id,
                    )
                    if not owned_entity:
                        result_warnings.append(
                            f"entity_reference_unavailable_during_commit:{entity_id}"
                        )
                        continue
                    await conn.execute(
                        """
                        insert into memory_entity_mentions (
                            owner_id, entity_id, memory_item_id, role, confidence
                        )
                        values ($1, $2, $3, 'explicit_reference', 1.0)
                        on conflict (entity_id, memory_item_id, role) do nothing
                        """,
                        normalized_owner_id,
                        entity_id,
                        memory.id,
                    )

                previous: MemoryItem | None = None
                if supersedes_memory_id is not None:
                    previous = await _load_owned_memory(
                        conn,
                        normalized_owner_id,
                        supersedes_memory_id,
                    )
                    if previous.status not in {"active", "forgotten"}:
                        raise MemoryConflictError(
                            "only an active or forgotten memory can be explicitly corrected"
                        )
                    if previous.status == "active":
                        updated = await conn.fetchrow(
                            """
                            update memory_items
                            set status = 'superseded',
                                valid_to = $3,
                                superseded_by_memory_id = $4,
                                updated_at = now()
                            where owner_id = $1 and id = $2 and status = 'active'
                            returning id
                            """,
                            normalized_owner_id,
                            supersedes_memory_id,
                            occurred_at,
                            memory.id,
                        )
                        if updated is None:
                            raise MemoryConflictError(
                                "memory changed while it was being superseded"
                            )
                        await conn.execute(
                            """
                            insert into memory_links (
                                owner_id, source_memory_id, target_memory_id,
                                relationship, confidence, valid_from, metadata
                            )
                            values ($1, $2, $3, 'supersedes', 1.0, $4, '{}'::jsonb)
                            on conflict (
                                owner_id, source_memory_id, target_memory_id, relationship
                            ) where valid_to is null
                            do nothing
                            """,
                            normalized_owner_id,
                            memory.id,
                            supersedes_memory_id,
                            occurred_at,
                        )

                await record_explicit_memory_override(
                    conn,
                    owner_id=normalized_owner_id,
                    memory_type=normalized_type,
                    identity_fingerprint=identity_fingerprint,
                    override_memory_id=memory.id,
                    source_memory_id=(
                        previous.id
                        if previous is not None and previous.status == "forgotten"
                        else None
                    ),
                    metadata={"explicit_idempotency_key": normalized_key},
                )

                indexing_status = "not_configured"
                if job_repository is not None:
                    from app.memory.embeddings.job_handlers import (
                        EMBED_MEMORY_ITEM_JOB,
                    )

                    embedding_dedupe_key = _memory_embedding_dedupe_key(memory.id)
                    enqueue = await job_repository.enqueue(
                        owner_id=normalized_owner_id,
                        session_id=session_id,
                        job_type=EMBED_MEMORY_ITEM_JOB,
                        dedupe_key=embedding_dedupe_key,
                        payload={"memory_item_id": str(memory.id)},
                        available_at=datetime.now(UTC),
                        connection=conn,
                    )
                    indexing_status = enqueue.job.status

                return ExplicitMemoryWriteResult(
                    memory=memory,
                    evidence=tuple(attached),
                    indexing_status=indexing_status,
                    replayed=False,
                    warnings=tuple(result_warnings),
                )

    async def promote_event_to_episode(
        self,
        *,
        owner_id: str,
        event_id: UUID,
        title: str,
        content: str,
        summary: str = "",
        importance: float,
        confidence: float,
        source: str,
        is_explicit: bool = False,
        metadata: Mapping[str, Any] | None = None,
        promotion_reason: str,
    ) -> MemoryItem:
        normalized_owner_id = _require_text(owner_id, "owner_id")
        normalized_title = _require_text(title, "title")
        normalized_content = _require_text(content, "content")
        normalized_source = _require_text(source, "source")
        normalized_reason = _require_text(promotion_reason, "promotion_reason")
        _require_score(importance, "importance")
        _require_score(confidence, "confidence")
        safe_metadata = _normalize_metadata(metadata)

        async with self._pool.acquire() as conn:  # noqa: SIM117
            async with conn.transaction():
                event = await _lock_event(conn, normalized_owner_id, event_id)
                existing_row = await conn.fetchrow(
                    """
                    select *
                    from memory_items
                    where source_experience_event_id = $1 and memory_type = 'episode'
                    for update
                    """,
                    event_id,
                )
                if existing_row is not None:
                    existing = _memory_from_row(existing_row)
                    _require_matching_episode_intent(
                        existing,
                        owner_id=normalized_owner_id,
                        session_id=event["session_id"],
                        title=normalized_title,
                        content=normalized_content,
                        summary=summary.strip(),
                        importance=importance,
                        confidence=confidence,
                        source=normalized_source,
                        is_explicit=is_explicit,
                        metadata=safe_metadata,
                    )
                    _require_matching_promotion_marker(
                        event,
                        memory_id=existing.id,
                        reason=normalized_reason,
                    )
                    await _mark_event_promoted(
                        conn,
                        owner_id=normalized_owner_id,
                        event_id=event_id,
                        memory_id=existing.id,
                        reason=normalized_reason,
                    )
                    return existing

                if event["promotion_decision"] == "skipped":
                    raise ExperiencePromotionConflictError(
                        "event promotion was already durably skipped"
                    )

                occurred_at = event["ended_at"] or event["started_at"]
                row = await conn.fetchrow(
                    """
                    insert into memory_items (
                        owner_id,
                        session_id,
                        memory_type,
                        title,
                        content,
                        summary,
                        source_experience_event_id,
                        occurred_at,
                        valid_from,
                        importance,
                        confidence,
                        source,
                        is_explicit,
                        metadata
                    )
                    values (
                        $1, $2, 'episode', $3, $4, $5, $6, $7, $7,
                        $8, $9, $10, $11, $12::jsonb
                    )
                    returning *
                    """,
                    normalized_owner_id,
                    event["session_id"],
                    normalized_title,
                    normalized_content,
                    summary.strip(),
                    event_id,
                    occurred_at,
                    importance,
                    confidence,
                    normalized_source,
                    is_explicit,
                    _json_dumps(safe_metadata),
                )
                memory = _memory_from_row(row)
                await _mark_event_promoted(
                    conn,
                    owner_id=normalized_owner_id,
                    event_id=event_id,
                    memory_id=memory.id,
                    reason=normalized_reason,
                )
                return memory

    async def skip_event_promotion(
        self,
        *,
        owner_id: str,
        event_id: UUID,
        reason: str,
    ) -> None:
        normalized_owner_id = _require_text(owner_id, "owner_id")
        normalized_reason = _require_text(reason, "reason")
        async with self._pool.acquire() as conn:  # noqa: SIM117
            async with conn.transaction():
                event = await _lock_event(conn, normalized_owner_id, event_id)
                if event["promotion_decision"] == "promoted":
                    raise ExperiencePromotionConflictError(
                        "a promoted event cannot subsequently be skipped"
                    )
                if event["promotion_decision"] == "skipped":
                    if event["promotion_reason"] != normalized_reason:
                        raise ExperiencePromotionConflictError(
                            "event promotion was skipped with a different reason"
                        )
                    return
                await conn.execute(
                    """
                    update experience_events
                    set promotion_decision = 'skipped',
                        promotion_reason = $3,
                        promoted_episode_memory_id = null,
                        promotion_decided_at = now(),
                        updated_at = now()
                    where owner_id = $1 and id = $2
                    """,
                    normalized_owner_id,
                    event_id,
                    normalized_reason,
                )

    async def mark_semantic_reconciled(
        self,
        *,
        owner_id: str,
        event_id: UUID,
        reconciled_at: datetime,
    ) -> None:
        async with self._pool.acquire() as conn:
            updated = await conn.fetchval(
                """
                update experience_events
                set semantic_reconciled_at = coalesce(semantic_reconciled_at, $3),
                    updated_at = case
                        when semantic_reconciled_at is null then now()
                        else updated_at
                    end
                where owner_id = $1 and id = $2
                returning true
                """,
                _require_text(owner_id, "owner_id"),
                event_id,
                reconciled_at,
            )
        if not updated:
            raise MemoryScopeError("experience event is outside the owner scope")

    async def forget_memory(self, *, owner_id: str, memory_id: UUID) -> MemoryItem:
        result = await self.forget_derived_memory(
            owner_id=owner_id,
            memory_id=memory_id,
            reason="Compatibility forgetting request.",
        )
        return result.memory

    async def forget_derived_memory(
        self,
        *,
        owner_id: str,
        memory_id: UUID,
        reason: str,
    ) -> ForgetMemoryResult:
        normalized_owner_id = _require_text(owner_id, "owner_id")
        normalized_reason = _require_text(reason, "reason")
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    "select pg_advisory_xact_lock(hashtextextended($1, 0))",
                    f"{normalized_owner_id}:forget-memory:{memory_id}",
                )
                source_row = await conn.fetchrow(
                    "select * from memory_items where owner_id = $1 and id = $2 for update",
                    normalized_owner_id,
                    memory_id,
                )
                if source_row is None:
                    raise MemoryMissingError()
                replayed = source_row["status"] == "forgotten"
                memory_type = _normalize_memory_type(source_row["memory_type"])
                fingerprint = memory_identity_fingerprint(
                    memory_type=memory_type,
                    content=source_row["content"],
                    metadata=_json_object(source_row["metadata"]),
                    source_experience_event_id=source_row["source_experience_event_id"],
                )
                await conn.execute(
                    "select pg_advisory_xact_lock(hashtextextended($1, 0))",
                    (
                        f"{normalized_owner_id}:forget-memory-identity:"
                        f"{memory_type}:{fingerprint}"
                    ),
                )
                tombstone_row = await conn.fetchrow(
                    """
                    select * from memory_forgetting_tombstones
                    where owner_id = $1
                      and source_memory_id = $2
                    for update
                    """,
                    normalized_owner_id,
                    memory_id,
                )
                if tombstone_row is None:
                    tombstone_row = await conn.fetchrow(
                        """
                        select * from memory_forgetting_tombstones
                        where owner_id = $1
                          and memory_type = $2
                          and identity_fingerprint = $3
                          and overridden_at is null
                        for update
                        """,
                        normalized_owner_id,
                        memory_type,
                        fingerprint,
                    )
                if tombstone_row is None:
                    tombstone_row = await conn.fetchrow(
                        """
                        insert into memory_forgetting_tombstones (
                            owner_id, memory_type, identity_fingerprint,
                            source_memory_id, reason
                        )
                        values ($1, $2, $3, $4, $5)
                        returning *
                        """,
                        normalized_owner_id,
                        memory_type,
                        fingerprint,
                        memory_id,
                        normalized_reason,
                    )
                row = await conn.fetchrow(
                    """
                    update memory_items
                    set status = 'forgotten',
                        superseded_by_memory_id = null,
                        updated_at = now()
                    where owner_id = $1 and id = $2
                    returning *
                    """,
                    normalized_owner_id,
                    memory_id,
                )
                removed_embeddings = await conn.fetchval(
                    """
                    with removed as (
                        delete from memory_item_embeddings where memory_item_id = $1
                        returning 1
                    ) select count(*)::integer from removed
                    """,
                    memory_id,
                )
                closed_links = await conn.fetchval(
                    """
                    with closed as (
                        update memory_links
                        set valid_to = greatest(valid_from, now()),
                            metadata = metadata || $3::jsonb,
                            updated_at = now()
                        where owner_id = $1
                          and valid_to is null
                          and (
                              source_memory_id = $2
                              or target_memory_id = $2
                              or supporting_memory_id = $2
                          )
                        returning 1
                    ) select count(*)::integer from closed
                    """,
                    normalized_owner_id,
                    memory_id,
                    _json_dumps({"closed_reason": "memory_forgotten"}),
                )
                closed_entity_relations = await conn.fetchval(
                    """
                    with closed as (
                        update memory_entity_relations
                        set status = 'retracted',
                            valid_to = greatest(valid_from, now()),
                            metadata = metadata || $3::jsonb,
                            updated_at = now()
                        where owner_id = $1
                          and supporting_memory_id = $2
                          and status = 'active'
                        returning 1
                    ) select count(*)::integer from closed
                    """,
                    normalized_owner_id,
                    memory_id,
                    _json_dumps({"retracted_reason": "supporting_memory_forgotten"}),
                )
                removed_retention_reasons = await conn.fetchval(
                    """
                    with removed as (
                        delete from visual_frame_retention_reasons
                        where owner_id = $1
                          and reason_type = 'explicit_memory'
                          and reason_ref_id = $2
                        returning 1
                    ) select count(*)::integer from removed
                    """,
                    normalized_owner_id,
                    memory_id,
                )
                removed_event_retention_reasons = await conn.fetchval(
                    """
                    with removed as (
                        delete from experience_event_retention_reasons
                        where owner_id = $1
                          and reason_type = 'explicit_memory'
                          and reason_ref_id = $2::text
                        returning 1
                    ) select count(*)::integer from removed
                    """,
                    normalized_owner_id,
                    str(memory_id),
                )
                cancelled_jobs = await conn.fetchval(
                    """
                    with cancelled as (
                        update memory_jobs
                        set status = 'dead',
                            attempt_count = max_attempts,
                            available_at = now(),
                            claimed_at = null,
                            claimed_by = null,
                            claim_token = null,
                            result = null,
                            last_error = 'memory forgotten before embedding',
                            completed_at = null,
                            updated_at = now()
                        where owner_id = $1
                          and job_type = 'embed_memory_item'
                          and payload->>'memory_item_id' = $2
                          and status in ('queued', 'failed')
                        returning 1
                    ) select count(*)::integer from cancelled
                    """,
                    normalized_owner_id,
                    str(memory_id),
                )
                if row is None or tombstone_row is None:
                    raise RuntimeError("forgetting transaction did not return durable state")
                return ForgetMemoryResult(
                    memory=_memory_from_row(row),
                    tombstone=_forgetting_tombstone_from_row(tombstone_row),
                    replayed=replayed,
                    removed_embedding_count=int(removed_embeddings or 0),
                    closed_memory_link_count=int(closed_links or 0),
                    closed_entity_relation_count=int(closed_entity_relations or 0),
                    removed_retention_reason_count=(
                        int(removed_retention_reasons or 0)
                        + int(removed_event_retention_reasons or 0)
                    ),
                    cancelled_embedding_job_count=int(cancelled_jobs or 0),
                )

    async def supersede_memory(
        self,
        *,
        owner_id: str,
        memory_id: UUID,
        replacement_memory_id: UUID,
    ) -> MemoryItem:
        if memory_id == replacement_memory_id:
            raise ValueError("memory cannot supersede itself")
        normalized_owner_id = _require_text(owner_id, "owner_id")
        async with self._pool.acquire() as conn:  # noqa: SIM117
            async with conn.transaction():
                source = await _load_owned_memory(conn, normalized_owner_id, memory_id, lock=True)
                replacement = await _load_owned_memory(
                    conn,
                    normalized_owner_id,
                    replacement_memory_id,
                    lock=True,
                )
                if replacement.status != "active":
                    raise MemoryConflictError("replacement memory must be active")
                if source.status == "superseded":
                    if source.superseded_by_memory_id == replacement_memory_id:
                        return source
                    raise MemoryConflictError("memory already has a different replacement")
                if source.status != "active":
                    raise MemoryConflictError("only active memory can be superseded")
                row = await conn.fetchrow(
                    """
                    update memory_items
                    set status = 'superseded',
                        superseded_by_memory_id = $3,
                        updated_at = now()
                    where owner_id = $1 and id = $2 and status = 'active'
                    returning *
                    """,
                    normalized_owner_id,
                    memory_id,
                    replacement_memory_id,
                )
                if row is None:
                    raise MemoryConflictError("memory changed while it was being superseded")
                return _memory_from_row(row)

    async def attach_evidence(
        self,
        *,
        owner_id: str,
        memory_item_id: UUID,
        evidence_type: MemoryEvidenceType,
        evidence_ref_id: str | UUID,
        evidence_time: datetime,
        excerpt: str = "",
        confidence: float = 1.0,
        metadata: Mapping[str, Any] | None = None,
    ) -> MemoryEvidence:
        normalized_owner_id = _require_text(owner_id, "owner_id")
        normalized_type = _normalize_evidence_type(evidence_type)
        normalized_ref = _require_text(str(evidence_ref_id), "evidence_ref_id")
        _require_score(confidence, "confidence")
        safe_metadata = _normalize_metadata(metadata)
        async with self._pool.acquire() as conn:  # noqa: SIM117
            async with conn.transaction():
                await _load_owned_memory(conn, normalized_owner_id, memory_item_id)
                await _validate_evidence_source(
                    conn,
                    owner_id=normalized_owner_id,
                    evidence_type=normalized_type,
                    evidence_ref_id=normalized_ref,
                )
                row = await conn.fetchrow(
                    """
                    insert into memory_evidence (
                        owner_id,
                        memory_item_id,
                        evidence_type,
                        evidence_ref_id,
                        evidence_time,
                        excerpt,
                        confidence,
                        metadata
                    )
                    values ($1, $2, $3, $4, $5, $6, $7, $8::jsonb)
                    on conflict (memory_item_id, evidence_type, evidence_ref_id)
                    do nothing
                    returning *
                    """,
                    normalized_owner_id,
                    memory_item_id,
                    normalized_type,
                    normalized_ref,
                    evidence_time,
                    excerpt.strip(),
                    confidence,
                    _json_dumps(safe_metadata),
                )
                if row is not None:
                    return _evidence_from_row(row)
                existing_row = await conn.fetchrow(
                    """
                    select * from memory_evidence
                    where memory_item_id = $1
                      and evidence_type = $2
                      and evidence_ref_id = $3
                    """,
                    memory_item_id,
                    normalized_type,
                    normalized_ref,
                )
                if existing_row is None:
                    raise MemoryConflictError("evidence insert lost without an existing row")
                existing = _evidence_from_row(existing_row)
                if (
                    existing.owner_id != normalized_owner_id
                    or existing.evidence_time != evidence_time
                    or existing.excerpt != excerpt.strip()
                    or existing.confidence != confidence
                    or dict(existing.metadata) != safe_metadata
                ):
                    raise MemoryConflictError(
                        "evidence identity was reused with different immutable data"
                    )
                return existing

    async def record_calendar_event_snapshot(
        self,
        *,
        owner_id: str,
        provider: str,
        calendar_id: str,
        provider_event_id: str,
        event_title: str,
        starts_at: datetime | None,
        ends_at: datetime | None,
        observed_at: datetime,
        snapshot: Mapping[str, Any],
    ) -> CalendarEventSnapshot:
        normalized_owner_id = _require_text(owner_id, "owner_id")
        normalized_provider = _require_text(provider, "provider")
        normalized_calendar_id = _require_text(calendar_id, "calendar_id")
        normalized_event_id = _require_text(provider_event_id, "provider_event_id")
        _require_validity(starts_at, ends_at)
        safe_snapshot = normalize_job_json_object(
            snapshot,
            label="calendar event snapshot",
        )
        digest = json_object_hash(safe_snapshot)

        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                insert into calendar_event_snapshots (
                    owner_id,
                    provider,
                    calendar_id,
                    provider_event_id,
                    event_title,
                    starts_at,
                    ends_at,
                    observed_at,
                    snapshot,
                    content_hash
                )
                values ($1, $2, $3, $4, $5, $6, $7, $8, $9::jsonb, $10)
                on conflict (
                    owner_id,
                    provider,
                    calendar_id,
                    provider_event_id,
                    content_hash
                ) do nothing
                returning *
                """,
                normalized_owner_id,
                normalized_provider,
                normalized_calendar_id,
                normalized_event_id,
                event_title.strip(),
                starts_at,
                ends_at,
                observed_at,
                _json_dumps(safe_snapshot),
                digest,
            )
            if row is None:
                row = await conn.fetchrow(
                    """
                    select * from calendar_event_snapshots
                    where owner_id = $1
                      and provider = $2
                      and calendar_id = $3
                      and provider_event_id = $4
                      and content_hash = $5
                    """,
                    normalized_owner_id,
                    normalized_provider,
                    normalized_calendar_id,
                    normalized_event_id,
                    digest,
                )
        if row is None:
            raise MemoryConflictError(
                "calendar snapshot insert lost without an existing row"
            )
        return _calendar_snapshot_from_row(row)

    async def create_link(
        self,
        *,
        owner_id: str,
        source_memory_id: UUID,
        target_memory_id: UUID,
        relationship: str,
        confidence: float,
        valid_from: datetime,
        supporting_memory_id: UUID | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> MemoryLink:
        if source_memory_id == target_memory_id:
            raise ValueError("memory link cannot point to itself")
        normalized_owner_id = _require_text(owner_id, "owner_id")
        normalized_relationship = _require_text(relationship, "relationship")
        _require_score(confidence, "confidence")
        safe_metadata = _normalize_metadata(metadata)
        async with self._pool.acquire() as conn:  # noqa: SIM117
            async with conn.transaction():
                await _load_owned_memory(conn, normalized_owner_id, source_memory_id)
                await _load_owned_memory(conn, normalized_owner_id, target_memory_id)
                if supporting_memory_id is not None:
                    await _load_owned_memory(conn, normalized_owner_id, supporting_memory_id)
                row = await conn.fetchrow(
                    """
                    insert into memory_links (
                        owner_id,
                        source_memory_id,
                        target_memory_id,
                        relationship,
                        confidence,
                        valid_from,
                        supporting_memory_id,
                        metadata
                    )
                    values ($1, $2, $3, $4, $5, $6, $7, $8::jsonb)
                    on conflict (
                        owner_id,
                        source_memory_id,
                        target_memory_id,
                        relationship
                    ) where valid_to is null
                    do nothing
                    returning *
                    """,
                    normalized_owner_id,
                    source_memory_id,
                    target_memory_id,
                    normalized_relationship,
                    confidence,
                    valid_from,
                    supporting_memory_id,
                    _json_dumps(safe_metadata),
                )
                if row is not None:
                    return _link_from_row(row)
                existing_row = await conn.fetchrow(
                    """
                    select * from memory_links
                    where owner_id = $1
                      and source_memory_id = $2
                      and target_memory_id = $3
                      and relationship = $4
                      and valid_to is null
                    """,
                    normalized_owner_id,
                    source_memory_id,
                    target_memory_id,
                    normalized_relationship,
                )
                if existing_row is None:
                    raise MemoryConflictError("active link insert lost without an existing row")
                existing = _link_from_row(existing_row)
                if (
                    existing.confidence != confidence
                    or existing.valid_from != valid_from
                    or existing.supporting_memory_id != supporting_memory_id
                    or dict(existing.metadata) != safe_metadata
                ):
                    raise MemoryConflictError(
                        "active relationship was reused with different durable data"
                    )
                return existing

    async def close_link(
        self,
        *,
        owner_id: str,
        link_id: UUID,
        valid_to: datetime,
    ) -> MemoryLink:
        normalized_owner_id = _require_text(owner_id, "owner_id")
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                update memory_links
                set valid_to = $3,
                    updated_at = now()
                where owner_id = $1
                  and id = $2
                  and valid_to is null
                  and valid_from <= $3
                returning *
                """,
                normalized_owner_id,
                link_id,
                valid_to,
            )
            if row is not None:
                return _link_from_row(row)
            existing_row = await conn.fetchrow(
                "select * from memory_links where owner_id = $1 and id = $2",
                normalized_owner_id,
                link_id,
            )
        if existing_row is None:
            raise MemoryMissingError()
        existing = _link_from_row(existing_row)
        if existing.valid_to == valid_to:
            return existing
        if existing.valid_to is not None:
            raise MemoryConflictError("memory link was already closed at a different time")
        raise ValueError("memory link cannot end before it begins")

    async def upsert_memory_embedding(
        self,
        *,
        owner_id: str,
        memory_item_id: UUID,
        embedding_kind: str,
        provider: str,
        model: str,
        embedded_content: str,
        embedding: Sequence[float],
    ) -> MemoryEmbedding:
        return await self._upsert_embedding(
            owner_id=owner_id,
            subject_id=memory_item_id,
            subject="memory",
            embedding_kind=embedding_kind,
            provider=provider,
            model=model,
            embedded_content=embedded_content,
            embedding=embedding,
        )

    async def get_memory_embedding(
        self,
        *,
        owner_id: str,
        memory_item_id: UUID,
        embedding_kind: str,
        provider: str,
        model: str,
    ) -> MemoryEmbedding | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                select embedding.*
                from memory_item_embeddings embedding
                join memory_items memory on memory.id = embedding.memory_item_id
                where memory.owner_id = $1
                  and embedding.memory_item_id = $2
                  and embedding.embedding_kind = $3
                  and embedding.provider = $4
                  and embedding.model = $5
                """,
                _require_text(owner_id, "owner_id"),
                memory_item_id,
                _require_text(embedding_kind, "embedding_kind"),
                _require_text(provider, "provider"),
                _require_text(model, "model"),
            )
        return _embedding_from_row(row, "memory_item_id") if row is not None else None

    async def get_event_embedding(
        self,
        *,
        owner_id: str,
        event_id: UUID,
        embedding_kind: str,
        provider: str,
        model: str,
    ) -> MemoryEmbedding | None:
        normalized_owner_id = _require_text(owner_id, "owner_id")
        normalized_kind = _require_text(embedding_kind, "embedding_kind")
        normalized_provider = _require_text(provider, "provider")
        normalized_model = _require_text(model, "model")
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                select embedding.*
                from experience_event_embeddings embedding
                join experience_events event
                  on event.id = embedding.experience_event_id
                where event.owner_id = $1
                  and embedding.experience_event_id = $2
                  and embedding.embedding_kind = $3
                  and embedding.provider = $4
                  and embedding.model = $5
                """,
                normalized_owner_id,
                event_id,
                normalized_kind,
                normalized_provider,
                normalized_model,
            )
        if row is None:
            return None
        return _embedding_from_row(row, "experience_event_id")

    async def upsert_event_embedding(
        self,
        *,
        owner_id: str,
        event_id: UUID,
        embedding_kind: str,
        provider: str,
        model: str,
        embedded_content: str,
        embedding: Sequence[float],
    ) -> MemoryEmbedding:
        return await self._upsert_embedding(
            owner_id=owner_id,
            subject_id=event_id,
            subject="event",
            embedding_kind=embedding_kind,
            provider=provider,
            model=model,
            embedded_content=embedded_content,
            embedding=embedding,
        )

    async def list_time_retrieval_hits(
        self,
        *,
        owner_id: str,
        query: NormalizedMemoryRetrievalQuery,
        limit: int,
    ) -> list[RetrievalSourceHit]:
        _require_retrieval_limit(limit)
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                _RETRIEVAL_DOCUMENTS_CTE
                + """
                select document.*, 0::double precision as match_score,
                       false as exact_match
                from retrieval_documents document
                where document.owner_id = $1
                  and ($2::uuid is null or document.session_id = $2)
                  and document.occurred_at >= $3
                  and document.occurred_at < $4
                  and (
                      cardinality($5::text[]) = 0
                      or document.record_type = any($5::text[])
                  )
                """
                + _retrieval_lifecycle_predicate("document")
                + """
                order by document.occurred_at desc, document.id desc
                limit $7
                """,
                _require_text(owner_id, "owner_id"),
                query.session_id,
                query.start_time,
                query.end_time,
                list(query.memory_types),
                query.include_superseded,
                limit,
            )
        return [
            RetrievalSourceHit(
                document=_retrieval_document_from_row(row),
                source="time",
                raw_score=float(row["match_score"]),
            )
            for row in rows
        ]

    async def list_lexical_retrieval_hits(
        self,
        *,
        owner_id: str,
        query: NormalizedMemoryRetrievalQuery,
        limit: int,
    ) -> list[RetrievalSourceHit]:
        _require_retrieval_limit(limit)
        normalized_query = query.query.strip().lower()
        if not normalized_query:
            return []
        tokens = _lexical_tokens(normalized_query)
        if not tokens:
            return []
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                _RETRIEVAL_DOCUMENTS_CTE
                + """
                select document.*,
                       greatest(
                           case when position($7 in document.search_text) > 0
                               then 1.0 else 0.0 end,
                           lexical.token_score
                       ) as match_score,
                       position($7 in document.search_text) > 0 as exact_match
                from retrieval_documents document
                cross join lateral (
                    select count(*)::double precision
                           / cardinality($8::text[])::double precision as token_score
                    from unnest($8::text[]) token
                    where position(token in document.search_text) > 0
                ) lexical
                where document.owner_id = $1
                  and ($2::uuid is null or document.session_id = $2)
                  and document.occurred_at >= $3
                  and document.occurred_at < $4
                  and (
                      cardinality($5::text[]) = 0
                      or document.record_type = any($5::text[])
                  )
                """
                + _retrieval_lifecycle_predicate("document")
                + """
                  and (
                      position($7 in document.search_text) > 0
                      or lexical.token_score > 0
                  )
                order by exact_match desc, match_score desc,
                         document.occurred_at desc, document.id desc
                limit $9
                """,
                _require_text(owner_id, "owner_id"),
                query.session_id,
                query.start_time,
                query.end_time,
                list(query.memory_types),
                query.include_superseded,
                normalized_query,
                list(tokens),
                limit,
            )
        return [
            RetrievalSourceHit(
                document=_retrieval_document_from_row(row),
                source="lexical",
                raw_score=float(row["match_score"]),
                exact_match=bool(row["exact_match"]),
            )
            for row in rows
        ]

    async def list_embedding_retrieval_hits(
        self,
        *,
        owner_id: str,
        query: NormalizedMemoryRetrievalQuery,
        embedding: Sequence[float],
        provider: str,
        model: str,
        limit: int,
    ) -> list[RetrievalSourceHit]:
        _require_retrieval_limit(limit)
        vector = _vector_literal(normalize_embedding(embedding))
        normalized_owner_id = _require_text(owner_id, "owner_id")
        normalized_provider = _require_text(provider, "provider")
        normalized_model = _require_text(model, "model")
        hits: list[RetrievalSourceHit] = []
        async with self._pool.acquire() as conn:
            kind_rows = await conn.fetch(
                """
                select embedding_kind
                from (
                    select distinct embedding_kind
                    from experience_event_embeddings
                    where provider = $1 and model = $2
                    union
                    select distinct embedding_kind
                    from memory_item_embeddings
                    where provider = $1 and model = $2
                ) kinds
                order by embedding_kind
                limit $3
                """,
                normalized_provider,
                normalized_model,
                _RETRIEVAL_EMBEDDING_KIND_LIMIT,
            )
            for kind_row in kind_rows:
                embedding_kind = kind_row["embedding_kind"]
                source_hits: list[RetrievalSourceHit] = []
                sources = (
                    (
                        "experience_event_embeddings",
                        "experience_event_id",
                        "document.record_type = 'experience_event'",
                    ),
                    (
                        "memory_item_embeddings",
                        "memory_item_id",
                        "document.record_type <> 'experience_event'",
                    ),
                )
                for table, id_column, record_filter in sources:
                    rows = await conn.fetch(
                        _RETRIEVAL_DOCUMENTS_CTE
                        + f"""
                        select document.*,
                               1.0 - (embedding.embedding <=> $7::vector) as match_score,
                               false as exact_match
                        from {table} embedding
                        join retrieval_documents document
                          on document.id = embedding.{id_column}
                         and {record_filter}
                        where document.owner_id = $1
                          and ($2::uuid is null or document.session_id = $2)
                          and document.occurred_at >= $3
                          and document.occurred_at < $4
                          and (
                              cardinality($5::text[]) = 0
                              or document.record_type = any($5::text[])
                          )
                          and embedding.provider = $8
                          and embedding.model = $9
                          and embedding.embedding_kind = $10
                        """
                        + _retrieval_lifecycle_predicate("document")
                        + """
                        order by embedding.embedding <=> $7::vector,
                                 document.occurred_at desc,
                                 document.id desc
                        limit $11
                        """,
                        normalized_owner_id,
                        query.session_id,
                        query.start_time,
                        query.end_time,
                        list(query.memory_types),
                        query.include_superseded,
                        vector,
                        normalized_provider,
                        normalized_model,
                        embedding_kind,
                        limit,
                    )
                    source_hits.extend(
                        RetrievalSourceHit(
                            document=_retrieval_document_from_row(row),
                            source=f"embedding:{embedding_kind}",
                            raw_score=float(row["match_score"]),
                        )
                        for row in rows
                    )
                source_hits.sort(
                    key=lambda hit: (
                        -hit.raw_score,
                        -hit.document.occurred_at.timestamp(),
                        str(hit.document.id),
                    )
                )
                hits.extend(source_hits[:limit])
        return hits

    async def list_adjacent_retrieval_hits(
        self,
        *,
        owner_id: str,
        query: NormalizedMemoryRetrievalQuery,
        anchor_ids: Sequence[UUID],
        limit: int,
    ) -> list[RetrievalSourceHit]:
        _require_retrieval_limit(limit)
        unique_anchor_ids = tuple(dict.fromkeys(anchor_ids))
        if not unique_anchor_ids:
            return []
        if len(unique_anchor_ids) > _RETRIEVAL_AUDIT_MAX_CANDIDATES:
            raise ValueError("too many temporal adjacency anchors")
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                _RETRIEVAL_DOCUMENTS_CTE
                + """
                , anchors as (
                    select *
                    from retrieval_documents
                    where owner_id = $1 and id = any($7::uuid[])
                ), candidate_pairs as (
                    select candidate.*,
                           anchor.id as anchor_id,
                           case
                               when candidate.occurred_at < anchor.occurred_at
                                   then 'before'
                               else 'after'
                           end as direction,
                           abs(extract(epoch from (
                               candidate.occurred_at - anchor.occurred_at
                           ))) as distance_seconds
                    from anchors anchor
                    join retrieval_documents candidate
                      on candidate.owner_id = anchor.owner_id
                     and candidate.session_id = anchor.session_id
                     and candidate.id <> all($7::uuid[])
                     and candidate.occurred_at <> anchor.occurred_at
                    where anchor.session_id is not null
                      and ($2::uuid is null or candidate.session_id = $2)
                      and candidate.occurred_at >= $3
                      and candidate.occurred_at < $4
                      and (
                          cardinality($5::text[]) = 0
                          or candidate.record_type = any($5::text[])
                      )
                    """
                    + _retrieval_lifecycle_predicate("candidate")
                    + """
                ), nearest_per_anchor as (
                    select pairs.*,
                           row_number() over (
                               partition by pairs.anchor_id, pairs.direction
                               order by pairs.distance_seconds,
                                        pairs.occurred_at desc,
                                        pairs.id desc
                           ) as neighbour_rank
                    from candidate_pairs pairs
                ), deduplicated as (
                    select nearest.*,
                           row_number() over (
                               partition by nearest.record_type, nearest.id
                               order by nearest.distance_seconds, nearest.anchor_id
                           ) as candidate_rank
                    from nearest_per_anchor nearest
                    where nearest.neighbour_rank = 1
                )
                select deduplicated.*,
                       1.0 / (1.0 + deduplicated.distance_seconds) as match_score,
                       false as exact_match
                from deduplicated
                where deduplicated.candidate_rank = 1
                order by deduplicated.distance_seconds,
                         deduplicated.occurred_at desc,
                         deduplicated.id desc
                limit $8
                """,
                _require_text(owner_id, "owner_id"),
                query.session_id,
                query.start_time,
                query.end_time,
                list(query.memory_types),
                query.include_superseded,
                list(unique_anchor_ids),
                limit,
            )
        return [
            RetrievalSourceHit(
                document=_retrieval_document_from_row(row),
                source="temporal_adjacency",
                raw_score=float(row["match_score"]),
            )
            for row in rows
        ]

    async def list_exact_entity_retrieval_hits(
        self,
        *,
        owner_id: str,
        query: NormalizedMemoryRetrievalQuery,
        limit: int,
    ) -> list[RetrievalSourceHit]:
        _require_retrieval_limit(limit)
        normalized_query = query.query.strip().lower()
        if not normalized_query:
            return []
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                _RETRIEVAL_DOCUMENTS_CTE
                + """
                , entity_matches as (
                    select entity.id,
                           case
                               when entity.normalized_name = $7 then 1.0
                               when $7 = any(entity.aliases) then 1.0
                               else 0.9
                           end as match_score
                    from memory_entities entity
                    where entity.owner_id = $1
                      and (
                          position(entity.normalized_name in $7) > 0
                          or position($7 in entity.normalized_name) > 0
                          or exists (
                              select 1 from unnest(entity.aliases) alias
                              where position(lower(alias) in $7) > 0
                                 or position($7 in lower(alias)) > 0
                          )
                      )
                ), ranked as (
                    select document.*,
                           max(entity_matches.match_score) as match_score
                    from entity_matches
                    join memory_entity_mentions mention
                      on mention.entity_id = entity_matches.id
                     and mention.owner_id = $1
                    join retrieval_documents document
                      on document.id = mention.memory_item_id
                    where document.owner_id = $1
                      and ($2::uuid is null or document.session_id = $2)
                      and document.occurred_at >= $3
                      and document.occurred_at < $4
                      and (
                          cardinality($5::text[]) = 0
                          or document.record_type = any($5::text[])
                      )
                    """
                    + _retrieval_lifecycle_predicate("document")
                    + """
                    group by document.id, document.owner_id, document.session_id,
                             document.record_type, document.occurred_at,
                             document.valid_from, document.valid_to, document.title,
                             document.summary, document.content, document.importance,
                             document.confidence, document.record_status,
                             document.is_explicit, document.evidence_count,
                             document.keyframe_count, document.evidence_types,
                             document.search_text, document.retrieval_count
                )
                select ranked.*, true as exact_match
                from ranked
                order by match_score desc, occurred_at desc, id desc
                limit $8
                """,
                _require_text(owner_id, "owner_id"),
                query.session_id,
                query.start_time,
                query.end_time,
                list(query.memory_types),
                query.include_superseded,
                normalized_query,
                limit,
            )
        return [
            RetrievalSourceHit(
                document=_retrieval_document_from_row(row),
                source="entity_exact",
                raw_score=float(row["match_score"]),
                exact_match=True,
            )
            for row in rows
        ]

    async def list_entity_embedding_retrieval_hits(
        self,
        *,
        owner_id: str,
        query: NormalizedMemoryRetrievalQuery,
        embedding: Sequence[float],
        provider: str,
        model: str,
        limit: int,
    ) -> list[RetrievalSourceHit]:
        _require_retrieval_limit(limit)
        vector = _vector_literal(normalize_embedding(embedding))
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                _RETRIEVAL_DOCUMENTS_CTE
                + """
                , ranked as (
                    select document.*,
                           max(1.0 - (embedding.embedding <=> $7::vector)) as match_score
                    from memory_entity_mentions mention
                    join memory_entity_embeddings embedding
                      on embedding.entity_id = mention.entity_id
                    join retrieval_documents document
                      on document.id = mention.memory_item_id
                    where document.owner_id = $1
                      and mention.owner_id = $1
                      and ($2::uuid is null or document.session_id = $2)
                      and document.occurred_at >= $3
                      and document.occurred_at < $4
                      and (
                          cardinality($5::text[]) = 0
                          or document.record_type = any($5::text[])
                      )
                      and embedding.embedding_kind = $8
                      and embedding.provider = $9
                      and embedding.model = $10
                    """
                    + _retrieval_lifecycle_predicate("document")
                    + """
                    group by document.id, document.owner_id, document.session_id,
                             document.record_type, document.occurred_at,
                             document.valid_from, document.valid_to, document.title,
                             document.summary, document.content, document.importance,
                             document.confidence, document.record_status,
                             document.is_explicit, document.evidence_count,
                             document.keyframe_count, document.evidence_types,
                             document.search_text, document.retrieval_count
                )
                select ranked.*, false as exact_match
                from ranked
                order by match_score desc, occurred_at desc, id desc
                limit $11
                """,
                _require_text(owner_id, "owner_id"),
                query.session_id,
                query.start_time,
                query.end_time,
                list(query.memory_types),
                query.include_superseded,
                vector,
                ENTITY_IDENTITY_EMBEDDING_KIND,
                _require_text(provider, "provider"),
                _require_text(model, "model"),
                limit,
            )
        return [
            RetrievalSourceHit(
                document=_retrieval_document_from_row(row),
                source="entity_embedding",
                raw_score=float(row["match_score"]),
            )
            for row in rows
        ]

    async def get_retrieval_documents_by_ids(
        self,
        *,
        owner_id: str,
        document_ids: Sequence[UUID],
    ) -> list[RetrievalDocument]:
        unique_ids = tuple(dict.fromkeys(document_ids))
        if not unique_ids:
            return []
        if len(unique_ids) > _RETRIEVAL_AUDIT_MAX_CANDIDATES:
            raise ValueError("too many retrieval document IDs")
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                _RETRIEVAL_DOCUMENTS_CTE
                + """
                select * from retrieval_documents
                where owner_id = $1 and id = any($2::uuid[])
                """,
                _require_text(owner_id, "owner_id"),
                list(unique_ids),
            )
        by_id = {row["id"]: _retrieval_document_from_row(row) for row in rows}
        return [by_id[item_id] for item_id in unique_ids if item_id in by_id]

    async def record_retrieval_run(
        self,
        *,
        owner_id: str,
        session_id: UUID | None,
        query: str,
        constraints: Mapping[str, Any] | None,
        candidate_ids: Sequence[UUID],
        result_ids: Sequence[UUID],
        reinspection_performed: bool,
        provider_spaces: Sequence[str],
        latency_ms: int,
    ) -> UUID:
        normalized_owner_id = _require_text(owner_id, "owner_id")
        normalized_query = query.strip()
        if len(normalized_query.encode("utf-8")) > 8192:
            raise ValueError("retrieval audit query exceeds 8192 bytes")
        safe_constraints = normalize_job_json_object(
            constraints or {},
            label="retrieval constraints",
        )
        _reject_retrieval_audit_secrets(safe_constraints)
        serialized_constraints = _json_dumps(safe_constraints)
        if len(serialized_constraints.encode("utf-8")) > 32768:
            raise ValueError("retrieval audit constraints exceed 32768 bytes")
        normalized_candidates = _normalize_uuid_ids(
            candidate_ids,
            label="candidate_ids",
            maximum=_RETRIEVAL_AUDIT_MAX_CANDIDATES,
        )
        normalized_results = _normalize_uuid_ids(
            result_ids,
            label="result_ids",
            maximum=_RETRIEVAL_AUDIT_MAX_RESULTS,
        )
        if not set(normalized_results).issubset(normalized_candidates):
            raise ValueError("retrieval audit results must be selected from candidates")
        normalized_spaces = _normalize_provider_spaces(provider_spaces)
        if isinstance(latency_ms, bool) or latency_ms < 0:
            raise ValueError("retrieval audit latency_ms must be non-negative")

        async with self._pool.acquire() as conn:
            run_id = await conn.fetchval(
                """
                insert into memory_retrieval_runs (
                    owner_id,
                    session_id,
                    query,
                    constraints,
                    candidate_ids,
                    result_ids,
                    reinspection_performed,
                    provider_spaces,
                    latency_ms
                )
                select $1, $2, $3, $4::jsonb, $5::uuid[], $6::uuid[], $7, $8::text[], $9
                where $2::uuid is null
                   or exists (
                        select 1 from jarvis_sessions
                        where owner_id = $1 and id = $2
                   )
                returning id
                """,
                normalized_owner_id,
                session_id,
                normalized_query,
                serialized_constraints,
                list(normalized_candidates),
                list(normalized_results),
                reinspection_performed,
                list(normalized_spaces),
                latency_ms,
            )
        if run_id is None:
            raise MemoryScopeError("retrieval session is outside the owner scope")
        return cast(UUID, run_id)

    async def _upsert_embedding(
        self,
        *,
        owner_id: str,
        subject_id: UUID,
        subject: str,
        embedding_kind: str,
        provider: str,
        model: str,
        embedded_content: str,
        embedding: Sequence[float],
    ) -> MemoryEmbedding:
        normalized_owner_id = _require_text(owner_id, "owner_id")
        normalized_kind = _require_text(embedding_kind, "embedding_kind")
        normalized_provider = _require_text(provider, "provider")
        normalized_model = _require_text(model, "model")
        normalized_content = _require_text(embedded_content, "embedded_content")
        normalized_embedding = normalize_embedding(embedding)
        digest = content_hash(normalized_content)
        table, subject_column = _embedding_table(subject)

        async with self._pool.acquire() as conn:  # noqa: SIM117
            async with conn.transaction():
                if subject == "memory":
                    await _load_owned_memory(conn, normalized_owner_id, subject_id)
                else:
                    await _load_owned_event(conn, normalized_owner_id, subject_id)
                row = await conn.fetchrow(
                    f"""
                    insert into {table} (
                        {subject_column},
                        embedding_kind,
                        provider,
                        model,
                        dimensions,
                        embedded_content,
                        content_hash,
                        embedding,
                        embedded_at
                    )
                    values ($1, $2, $3, $4, $5, $6, $7, $8::vector, now())
                    on conflict ({subject_column}, embedding_kind, provider, model)
                    do update
                    set dimensions = excluded.dimensions,
                        embedded_content = excluded.embedded_content,
                        content_hash = excluded.content_hash,
                        embedding = excluded.embedding,
                        embedded_at = now(),
                        updated_at = now()
                    where {table}.content_hash is distinct from excluded.content_hash
                    returning *
                    """,
                    subject_id,
                    normalized_kind,
                    normalized_provider,
                    normalized_model,
                    EMBEDDING_DIMENSIONS,
                    normalized_content,
                    digest,
                    _vector_literal(normalized_embedding),
                )
                if row is None:
                    row = await conn.fetchrow(
                        f"""
                        select * from {table}
                        where {subject_column} = $1
                          and embedding_kind = $2
                          and provider = $3
                          and model = $4
                        """,
                        subject_id,
                        normalized_kind,
                        normalized_provider,
                        normalized_model,
                    )
        if row is None:
            raise MemoryConflictError("embedding upsert lost without an existing row")
        return _embedding_from_row(row, subject_column)


async def _lock_event(
    conn: asyncpg.Connection,
    owner_id: str,
    event_id: UUID,
) -> Mapping[str, Any]:
    row = await conn.fetchrow(
        """
        select * from experience_events
        where owner_id = $1 and id = $2
        for update
        """,
        owner_id,
        event_id,
    )
    if row is None:
        raise MemoryScopeError("experience event is outside the owner scope")
    return row


async def _load_owned_event(
    conn: asyncpg.Connection,
    owner_id: str,
    event_id: UUID,
) -> Mapping[str, Any]:
    row = await conn.fetchrow(
        "select * from experience_events where owner_id = $1 and id = $2",
        owner_id,
        event_id,
    )
    if row is None:
        raise MemoryScopeError("experience event is outside the owner scope")
    return row


async def _mark_event_promoted(
    conn: asyncpg.Connection,
    *,
    owner_id: str,
    event_id: UUID,
    memory_id: UUID,
    reason: str,
) -> None:
    updated = await conn.fetchval(
        """
        update experience_events
        set promotion_decision = 'promoted',
            promotion_reason = case
                when promotion_decision = 'pending' then $4
                else promotion_reason
            end,
            promoted_episode_memory_id = $3,
            promotion_decided_at = coalesce(promotion_decided_at, now()),
            updated_at = now()
        where owner_id = $1
          and id = $2
          and (
                promotion_decision = 'pending'
                or (
                    promotion_decision = 'promoted'
                    and promoted_episode_memory_id = $3
                    and promotion_reason is not distinct from $4
                )
              )
        returning true
        """,
        owner_id,
        event_id,
        memory_id,
        reason,
    )
    if not updated:
        raise ExperiencePromotionConflictError(
            "event promotion marker conflicts with the existing durable decision"
        )


async def _load_owned_memory(
    conn: asyncpg.Connection,
    owner_id: str,
    memory_id: UUID,
    *,
    lock: bool = False,
) -> MemoryItem:
    lock_clause = " for update" if lock else ""
    row = await conn.fetchrow(
        f"select * from memory_items where owner_id = $1 and id = $2{lock_clause}",
        owner_id,
        memory_id,
    )
    if row is None:
        raise MemoryScopeError("memory is outside the owner scope")
    return _memory_from_row(row)


async def _validate_evidence_source(
    conn: asyncpg.Connection,
    *,
    owner_id: str,
    evidence_type: MemoryEvidenceType,
    evidence_ref_id: str,
) -> None:
    table = _LOCAL_EVIDENCE_TABLES[evidence_type]
    try:
        source_id = UUID(evidence_ref_id)
    except ValueError as exc:
        raise ValueError(f"{evidence_type} evidence_ref_id must be a UUID") from exc
    exists = await conn.fetchval(
        f"select exists(select 1 from {table} where owner_id = $1 and id = $2)",
        owner_id,
        source_id,
    )
    if not exists:
        raise MemoryScopeError("evidence source is outside the owner scope")


def _require_matching_promotion_marker(
    event: Mapping[str, Any],
    *,
    memory_id: UUID,
    reason: str,
) -> None:
    decision = event["promotion_decision"]
    if decision == "pending":
        return
    if decision != "promoted":
        raise ExperiencePromotionConflictError(
            "event promotion has a conflicting durable decision"
        )
    if event["promoted_episode_memory_id"] != memory_id:
        raise ExperiencePromotionConflictError(
            "event promotion points to a different episode memory"
        )
    if event["promotion_reason"] != reason:
        raise ExperiencePromotionConflictError(
            "event promotion was replayed with a different reason"
        )


def _require_matching_episode_intent(
    memory: MemoryItem,
    *,
    owner_id: str,
    session_id: UUID,
    title: str,
    content: str,
    summary: str,
    importance: float,
    confidence: float,
    source: str,
    is_explicit: bool,
    metadata: Mapping[str, Any],
) -> None:
    expected = {
        "owner_id": owner_id,
        "session_id": session_id,
        "title": title,
        "content": content,
        "summary": summary,
        "importance": importance,
        "confidence": confidence,
        "source": source,
        "is_explicit": is_explicit,
        "metadata": dict(metadata),
    }
    actual = {
        "owner_id": memory.owner_id,
        "session_id": memory.session_id,
        "title": memory.title,
        "content": memory.content,
        "summary": memory.summary,
        "importance": memory.importance,
        "confidence": memory.confidence,
        "source": memory.source,
        "is_explicit": memory.is_explicit,
        "metadata": dict(memory.metadata),
    }
    conflicts = [key for key, value in expected.items() if actual[key] != value]
    if conflicts:
        raise ExperiencePromotionConflictError(
            "event already has an episode with different durable intent: "
            + ", ".join(conflicts)
        )


def _memory_from_row(row: Mapping[str, Any]) -> MemoryItem:
    return MemoryItem(
        id=row["id"],
        owner_id=row["owner_id"],
        session_id=row["session_id"],
        memory_type=cast(MemoryType, row["memory_type"]),
        status=row["status"],
        title=row["title"],
        content=row["content"],
        summary=row["summary"],
        source_experience_event_id=row["source_experience_event_id"],
        occurred_at=row["occurred_at"],
        valid_from=row["valid_from"],
        valid_to=row["valid_to"],
        importance=float(row["importance"]),
        confidence=float(row["confidence"]),
        source=row["source"],
        is_explicit=row["is_explicit"],
        superseded_by_memory_id=row["superseded_by_memory_id"],
        metadata=_json_object(row["metadata"]),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _forgetting_tombstone_from_row(
    row: Mapping[str, Any],
) -> MemoryForgettingTombstone:
    return MemoryForgettingTombstone(
        id=row["id"],
        owner_id=row["owner_id"],
        memory_type=_normalize_memory_type(row["memory_type"]),
        identity_fingerprint=row["identity_fingerprint"],
        source_memory_id=row["source_memory_id"],
        reason=row["reason"],
        forgotten_at=row["forgotten_at"],
        overridden_at=row["overridden_at"],
        override_memory_id=row["override_memory_id"],
        override_metadata=_json_object(row["override_metadata"]),
        suppressed_recreation_count=row["suppressed_recreation_count"],
        last_suppressed_at=row["last_suppressed_at"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _calendar_snapshot_from_row(row: Mapping[str, Any]) -> CalendarEventSnapshot:
    return CalendarEventSnapshot(
        id=row["id"],
        owner_id=row["owner_id"],
        provider=row["provider"],
        calendar_id=row["calendar_id"],
        provider_event_id=row["provider_event_id"],
        event_title=row["event_title"],
        starts_at=row["starts_at"],
        ends_at=row["ends_at"],
        observed_at=row["observed_at"],
        snapshot=_json_object(row["snapshot"]),
        content_hash=row["content_hash"],
        created_at=row["created_at"],
    )


def _evidence_from_row(row: Mapping[str, Any]) -> MemoryEvidence:
    return MemoryEvidence(
        id=row["id"],
        owner_id=row["owner_id"],
        memory_item_id=row["memory_item_id"],
        evidence_type=cast(MemoryEvidenceType, row["evidence_type"]),
        evidence_ref_id=row["evidence_ref_id"],
        evidence_time=row["evidence_time"],
        excerpt=row["excerpt"],
        confidence=float(row["confidence"]),
        metadata=_json_object(row["metadata"]),
        created_at=row["created_at"],
    )


def _link_from_row(row: Mapping[str, Any]) -> MemoryLink:
    return MemoryLink(
        id=row["id"],
        owner_id=row["owner_id"],
        source_memory_id=row["source_memory_id"],
        target_memory_id=row["target_memory_id"],
        relationship=row["relationship"],
        confidence=float(row["confidence"]),
        valid_from=row["valid_from"],
        valid_to=row["valid_to"],
        supporting_memory_id=row["supporting_memory_id"],
        metadata=_json_object(row["metadata"]),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _embedding_from_row(row: Mapping[str, Any], subject_column: str) -> MemoryEmbedding:
    return MemoryEmbedding(
        subject_id=row[subject_column],
        embedding_kind=row["embedding_kind"],
        provider=row["provider"],
        model=row["model"],
        dimensions=row["dimensions"],
        embedded_content=row["embedded_content"],
        content_hash=row["content_hash"],
        embedding=_parse_vector(row["embedding"]),
        embedded_at=row["embedded_at"],
        updated_at=row["updated_at"],
    )


def _retrieval_document_from_row(row: Mapping[str, Any]) -> RetrievalDocument:
    record_type = row["record_type"]
    if record_type not in RETRIEVAL_RECORD_TYPES:
        raise TypeError(f"database returned unsupported retrieval type: {record_type}")
    return RetrievalDocument(
        id=row["id"],
        owner_id=row["owner_id"],
        session_id=row["session_id"],
        record_type=cast(RetrievalRecordType, record_type),
        occurred_at=row["occurred_at"],
        valid_from=row["valid_from"],
        valid_to=row["valid_to"],
        title=row["title"],
        summary=row["summary"],
        content=row["content"],
        importance=float(row["importance"]),
        confidence=float(row["confidence"]),
        record_status=row["record_status"],
        is_explicit=bool(row["is_explicit"]),
        evidence_count=int(row["evidence_count"]),
        keyframe_count=int(row["keyframe_count"]),
        evidence_types=tuple(row["evidence_types"]),
        retrieval_count=int(row["retrieval_count"]),
    )


def _require_retrieval_limit(limit: int) -> None:
    if isinstance(limit, bool) or limit < 1 or limit > _RETRIEVAL_AUDIT_MAX_CANDIDATES:
        raise ValueError(
            f"retrieval limit must be between 1 and {_RETRIEVAL_AUDIT_MAX_CANDIDATES}"
        )


def _normalize_memory_type(memory_type: str) -> MemoryType:
    if memory_type not in MEMORY_TYPES:
        raise ValueError(f"unsupported memory type: {memory_type}")
    return cast(MemoryType, memory_type)


def _normalize_evidence_type(evidence_type: str) -> MemoryEvidenceType:
    if evidence_type not in MEMORY_EVIDENCE_TYPES:
        raise ValueError(f"unsupported evidence type: {evidence_type}")
    return cast(MemoryEvidenceType, evidence_type)


def _embedding_table(subject: str) -> tuple[str, str]:
    if subject == "memory":
        return "memory_item_embeddings", "memory_item_id"
    if subject == "event":
        return "experience_event_embeddings", "experience_event_id"
    raise ValueError(f"unsupported embedding subject: {subject}")


def _normalize_metadata(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    return normalize_job_json_object(metadata or {}, label="memory metadata")


def _json_object(value: Any) -> dict[str, Any]:
    decoded = json.loads(value) if isinstance(value, str) else value
    if not isinstance(decoded, Mapping):
        raise TypeError("database JSON value must be an object")
    return dict(decoded)


def _json_dumps(value: Mapping[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def _memory_embedding_dedupe_key(memory_item_id: UUID) -> str:
    return f"memory-item:{memory_item_id}:embed"


def _vector_literal(values: Sequence[float]) -> str:
    return "[" + ",".join(format(value, ".17g") for value in values) + "]"


def _parse_vector(value: Any) -> tuple[float, ...]:
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped.startswith("[") or not stripped.endswith("]"):
            raise ValueError("database embedding has an invalid vector representation")
        values = () if stripped == "[]" else tuple(float(item) for item in stripped[1:-1].split(","))
        return normalize_embedding(values)
    if isinstance(value, Sequence):
        return normalize_embedding(value)
    raise TypeError("database embedding must be a vector string or numeric sequence")


def _require_text(value: str, label: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{label} cannot be blank")
    return normalized


def _require_score(value: float, label: str) -> None:
    if not 0 <= value <= 1:
        raise ValueError(f"{label} must be between 0 and 1")


def _require_validity(valid_from: datetime | None, valid_to: datetime | None) -> None:
    if valid_from is not None and valid_to is not None and valid_to < valid_from:
        raise ValueError("memory validity cannot end before it begins")


def _normalize_uuid_ids(
    values: Sequence[UUID],
    *,
    label: str,
    maximum: int,
) -> tuple[UUID, ...]:
    if len(values) > maximum:
        raise ValueError(f"{label} cannot contain more than {maximum} values")
    normalized: list[UUID] = []
    seen: set[UUID] = set()
    for value in values:
        if not isinstance(value, UUID):
            raise TypeError(f"{label} must contain UUID values")
        if value not in seen:
            seen.add(value)
            normalized.append(value)
    return tuple(normalized)


def _normalize_provider_spaces(values: Sequence[str]) -> tuple[str, ...]:
    if len(values) > _RETRIEVAL_AUDIT_MAX_PROVIDER_SPACES:
        raise ValueError(
            "provider_spaces cannot contain more than "
            f"{_RETRIEVAL_AUDIT_MAX_PROVIDER_SPACES} values"
        )
    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        space = _require_text(value, "provider space")
        encoded_space = space.encode("utf-8")
        lowered_space = space.casefold()
        if (
            len(encoded_space) > _RETRIEVAL_AUDIT_MAX_PROVIDER_SPACE_BYTES
            or _RETRIEVAL_AUDIT_PROVIDER_SPACE_PATTERN.fullmatch(space) is None
            or "://" in lowered_space
            or lowered_space.startswith(("bearer", "eyj", "sk-"))
        ):
            raise ValueError("provider space must be a bounded non-secret identifier")
        if space not in seen:
            seen.add(space)
            normalized.append(space)
    if (
        len(",".join(normalized).encode("utf-8"))
        > _RETRIEVAL_AUDIT_MAX_PROVIDER_SPACES_BYTES
    ):
        raise ValueError(
            "provider_spaces exceed the 4096-byte retrieval audit limit"
        )
    return tuple(normalized)


def _reject_retrieval_audit_secrets(value: Any, *, path: str = "constraints") -> None:
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            normalized_key = raw_key.casefold().replace("-", "_")
            if any(part in normalized_key for part in _RETRIEVAL_AUDIT_FORBIDDEN_KEY_PARTS):
                raise ValueError(f"retrieval audit cannot store sensitive field: {path}.{raw_key}")
            _reject_retrieval_audit_secrets(child, path=f"{path}.{raw_key}")
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            _reject_retrieval_audit_secrets(child, path=f"{path}[{index}]")
        return
    if isinstance(value, str):
        lowered = value.casefold()
        if "x-amz-signature=" in lowered or "access_token=" in lowered:
            raise ValueError(f"retrieval audit cannot store signed credentials: {path}")
