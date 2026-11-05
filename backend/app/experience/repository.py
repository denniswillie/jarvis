from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from math import isfinite
from typing import Any
from uuid import UUID

import asyncpg

from app.experience.keyframes import EventKeyframeCandidate, ScoredEventKeyframe
from app.experience.models import (
    ExperienceEvent,
    ExperienceEventEvidence,
    ExperienceEventKeyframe,
    ExperienceEventKind,
    ExperienceEvidenceType,
    ExperienceModality,
    KeyframeSelectedBy,
    RetentionReasonType,
    ToolExecution,
    VisualFrameRetentionReason,
    normalize_durable_tool_payload,
    normalize_evidence_metadata,
)
from app.memory.jobs.repository import MemoryJobRepository


class ExperienceSessionMissingError(Exception):
    pass


class ExperienceEventMissingError(Exception):
    pass


class ExperienceEventClosedError(Exception):
    pass


class ExperienceEvidenceScopeError(Exception):
    pass


class ExperienceConflictError(Exception):
    pass


class ToolExecutionMissingError(Exception):
    pass


class ToolExecutionConflictError(Exception):
    pass


class ToolExecutionRepository:
    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def get_executions_by_call_ids(
        self, *, owner_id: str, session_id: UUID, provider_call_ids: Sequence[str],
    ) -> list[ToolExecution]:
        """Read only explicitly known, owner/session-scoped operation identities."""
        if len(provider_call_ids) > 256:
            raise ValueError("Too many execution identities")
        if not provider_call_ids:
            return []
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                select * from jarvis_tool_executions
                where owner_id = $1 and session_id = $2
                  and provider_call_id = any($3::text[])
                order by started_at, id
                """,
                owner_id, session_id, list(provider_call_ids),
            )
        return [_tool_execution_from_row(row) for row in rows]

    async def start_execution(
        self,
        *,
        owner_id: str,
        session_id: UUID,
        provider_call_id: str,
        tool_name: str,
        arguments: Mapping[str, Any],
        mutates_state: bool = False,
        started_at: datetime,
    ) -> ToolExecution:
        execution, _ = await self.start_execution_with_status(
            owner_id=owner_id,
            session_id=session_id,
            provider_call_id=provider_call_id,
            tool_name=tool_name,
            arguments=arguments,
            mutates_state=mutates_state,
            started_at=started_at,
        )
        return execution

    async def start_execution_with_status(
        self,
        *,
        owner_id: str,
        session_id: UUID,
        provider_call_id: str,
        tool_name: str,
        arguments: Mapping[str, Any],
        mutates_state: bool = False,
        started_at: datetime,
    ) -> tuple[ToolExecution, bool]:
        safe_arguments = normalize_durable_tool_payload(arguments, label="tool arguments")
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                row = await conn.fetchrow(
                    """
                    insert into jarvis_tool_executions (
                        owner_id,
                        session_id,
                        provider_call_id,
                        tool_name,
                        mutates_state,
                        arguments,
                        started_at
                    )
                    select owner_id, id, $3, $4, $5, $6::jsonb, $7
                    from jarvis_sessions
                    where owner_id = $1 and id = $2
                    on conflict (session_id, provider_call_id) do nothing
                    returning *
                    """,
                    owner_id,
                    session_id,
                    provider_call_id,
                    tool_name,
                    mutates_state,
                    _json_dumps(safe_arguments),
                    started_at,
                )
                if row is not None:
                    return _tool_execution_from_row(row), True

                existing = await conn.fetchrow(
                    """
                    select *
                    from jarvis_tool_executions
                    where owner_id = $1 and session_id = $2 and provider_call_id = $3
                    """,
                    owner_id,
                    session_id,
                    provider_call_id,
                )
                if existing is None:
                    raise ExperienceSessionMissingError()
                execution = _tool_execution_from_row(existing)
                if (
                    execution.tool_name != tool_name
                    or execution.mutates_state != mutates_state
                    or dict(execution.arguments) != safe_arguments
                ):
                    raise ToolExecutionConflictError("provider call id was reused with different input")
                return execution, False

    async def finalize_execution(
        self,
        *,
        owner_id: str,
        session_id: UUID,
        provider_call_id: str,
        status: str,
        completed_at: datetime,
        result: Mapping[str, Any] | None = None,
        error: Mapping[str, Any] | None = None,
    ) -> ToolExecution:
        if status == "completed":
            if result is None or error is not None:
                raise ValueError("completed execution requires result and no error")
        elif status == "failed":
            if error is None:
                raise ValueError("failed execution requires error")
        else:
            raise ValueError("terminal tool execution status must be completed or failed")

        safe_result = (
            normalize_durable_tool_payload(result, label="tool result")
            if result is not None
            else None
        )
        safe_error = (
            normalize_durable_tool_payload(error, label="tool error", max_bytes=16384)
            if error is not None
            else None
        )
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                row = await conn.fetchrow(
                    """
                    update jarvis_tool_executions
                    set status = $4,
                        result = $5::jsonb,
                        error = $6::jsonb,
                        completed_at = $7
                    where owner_id = $1
                      and session_id = $2
                      and provider_call_id = $3
                      and status = 'started'
                    returning *
                    """,
                    owner_id,
                    session_id,
                    provider_call_id,
                    status,
                    _optional_json_dumps(safe_result),
                    _optional_json_dumps(safe_error),
                    completed_at,
                )
                if row is not None:
                    return _tool_execution_from_row(row)

                existing = await conn.fetchrow(
                    """
                    select *
                    from jarvis_tool_executions
                    where owner_id = $1 and session_id = $2 and provider_call_id = $3
                    """,
                    owner_id,
                    session_id,
                    provider_call_id,
                )
                if existing is None:
                    raise ToolExecutionMissingError()
                execution = _tool_execution_from_row(existing)
                if (
                    execution.status != status
                    or _optional_mapping(execution.result) != safe_result
                    or _optional_mapping(execution.error) != safe_error
                ):
                    raise ToolExecutionConflictError("tool execution already has a different outcome")
                return execution


class ExperienceRepository:
    def __init__(
        self,
        pool: asyncpg.Pool,
        *,
        consolidation_job_repository: MemoryJobRepository | None = None,
    ) -> None:
        self._pool = pool
        self._consolidation_job_repository = consolidation_job_repository

    async def get_open_event(
        self,
        *,
        owner_id: str,
        session_id: UUID,
    ) -> ExperienceEvent | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                select *
                from experience_events
                where owner_id = $1 and session_id = $2 and status = 'open'
                """,
                owner_id,
                session_id,
            )
        return _event_from_row(row) if row else None

    async def get_event(
        self,
        *,
        owner_id: str,
        session_id: UUID,
        event_id: UUID,
    ) -> ExperienceEvent | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                select *
                from experience_events
                where owner_id = $1 and session_id = $2 and id = $3
                """,
                owner_id,
                session_id,
                event_id,
            )
        return _event_from_row(row) if row else None

    async def get_latest_closed_event(
        self,
        *,
        owner_id: str,
        session_id: UUID,
    ) -> ExperienceEvent | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                select *
                from experience_events
                where owner_id = $1
                  and session_id = $2
                  and status <> 'open'
                order by ended_at desc, id desc
                limit 1
                """,
                owner_id,
                session_id,
            )
        return _event_from_row(row) if row else None

    async def list_open_events_due(
        self,
        *,
        idle_cutoff: datetime,
        max_started_cutoff: datetime,
        limit: int,
    ) -> list[ExperienceEvent]:
        if limit < 1:
            raise ValueError("limit must be positive")
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                select *
                from experience_events
                where status = 'open'
                  and (
                    started_at <= $2
                    or coalesce(
                        nullif(state ->> 'last_evidence_at', '')::timestamptz,
                        started_at
                    ) <= $1
                  )
                order by started_at asc, id asc
                limit $3
                """,
                idle_cutoff,
                max_started_cutoff,
                limit,
            )
        return [_event_from_row(row) for row in rows]

    async def get_or_create_open_event(
        self,
        *,
        owner_id: str,
        session_id: UUID,
        event_kind: ExperienceEventKind,
        started_at: datetime,
        modalities: Sequence[ExperienceModality] = (),
    ) -> ExperienceEvent:
        normalized_modalities = _normalize_modalities(modalities)
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    "select pg_advisory_xact_lock(hashtextextended($1, 0))",
                    str(session_id),
                )
                existing = await conn.fetchrow(
                    """
                    select *
                    from experience_events
                    where owner_id = $1 and session_id = $2 and status = 'open'
                    for update
                    """,
                    owner_id,
                    session_id,
                )
                if existing is not None:
                    return _event_from_row(existing)

                row = await conn.fetchrow(
                    """
                    insert into experience_events (
                        owner_id,
                        session_id,
                        status,
                        event_kind,
                        modalities,
                        started_at
                    )
                    select owner_id, id, 'open', $3, $4::text[], $5
                    from jarvis_sessions
                    where owner_id = $1 and id = $2 and status = 'active'
                    returning *
                    """,
                    owner_id,
                    session_id,
                    event_kind,
                    list(normalized_modalities),
                    started_at,
                )
                if row is None:
                    raise ExperienceSessionMissingError()
                return _event_from_row(row)

    async def update_open_event_aggregate(
        self,
        *,
        owner_id: str,
        session_id: UUID,
        event_id: UUID,
        event_kind: ExperienceEventKind,
        modalities: Sequence[ExperienceModality],
        title: str,
        summary: str,
        scene: str | None,
        activity: str | None,
        state: Mapping[str, Any],
        importance: float,
        confidence: float,
        retention_expires_at: datetime | None,
        expected_updated_at: datetime,
    ) -> ExperienceEvent:
        normalized_modalities = _normalize_modalities(modalities)
        _require_score(importance, "importance")
        _require_score(confidence, "confidence")
        async with self._pool.acquire() as conn, conn.transaction():
            event = await _lock_event(
                conn,
                owner_id=owner_id,
                session_id=session_id,
                event_id=event_id,
            )
            if event.status != "open":
                raise ExperienceEventClosedError()
            if event.updated_at != expected_updated_at:
                raise ExperienceConflictError("event aggregate was based on stale state")

            row = await conn.fetchrow(
                """
                update experience_events
                set event_kind = $4,
                    modalities = $5::text[],
                    title = $6,
                    summary = $7,
                    scene = $8,
                    activity = $9,
                    state = $10::jsonb,
                    importance = $11,
                    confidence = $12,
                    retention_expires_at = $13,
                    updated_at = now()
                where owner_id = $1 and session_id = $2 and id = $3 and status = 'open'
                returning *
                """,
                owner_id,
                session_id,
                event_id,
                event_kind,
                list(normalized_modalities),
                title,
                summary,
                scene,
                activity,
                _json_dumps(dict(state)),
                importance,
                confidence,
                retention_expires_at,
            )
            if row is None:
                raise ExperienceConflictError("open event changed during aggregate update")
            return _event_from_row(row)

    async def attach_evidence(
        self,
        *,
        owner_id: str,
        session_id: UUID,
        event_id: UUID,
        evidence_type: ExperienceEvidenceType,
        evidence_ref_id: UUID,
        source_sequence: int | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> ExperienceEventEvidence:
        evidence, _ = await self.attach_evidence_with_status(
            owner_id=owner_id,
            session_id=session_id,
            event_id=event_id,
            evidence_type=evidence_type,
            evidence_ref_id=evidence_ref_id,
            source_sequence=source_sequence,
            metadata=metadata,
        )
        return evidence

    async def attach_evidence_with_status(
        self,
        *,
        owner_id: str,
        session_id: UUID,
        event_id: UUID,
        evidence_type: ExperienceEvidenceType,
        evidence_ref_id: UUID,
        source_sequence: int | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> tuple[ExperienceEventEvidence, bool]:
        safe_metadata = normalize_evidence_metadata(metadata or {})
        if source_sequence is not None and source_sequence < 0:
            raise ValueError("source_sequence cannot be negative")
        async with self._pool.acquire() as conn, conn.transaction():
            event = await _lock_event(
                conn,
                owner_id=owner_id,
                session_id=session_id,
                event_id=event_id,
            )
            existing = await conn.fetchrow(
                """
                select *
                from experience_event_evidence
                where event_id = $1 and evidence_type = $2 and evidence_ref_id = $3
                """,
                event_id,
                evidence_type,
                evidence_ref_id,
            )
            if existing is not None:
                return _evidence_from_row(existing), False
            if event.status != "open":
                raise ExperienceEventClosedError()

            row = await _insert_evidence(
                conn,
                event=event,
                evidence_type=evidence_type,
                evidence_ref_id=evidence_ref_id,
                source_sequence=source_sequence,
                metadata=safe_metadata,
            )
            if row is None:
                raise ExperienceEvidenceScopeError()
            return _evidence_from_row(row), True

    async def attach_evidence_and_update_aggregate(
        self,
        *,
        owner_id: str,
        session_id: UUID,
        event_id: UUID,
        evidence_type: ExperienceEvidenceType,
        evidence_ref_id: UUID,
        source_sequence: int | None,
        metadata: Mapping[str, Any] | None,
        event_kind: ExperienceEventKind,
        modalities: Sequence[ExperienceModality],
        title: str,
        summary: str,
        scene: str | None,
        activity: str | None,
        state: Mapping[str, Any],
        importance: float,
        confidence: float,
        retention_expires_at: datetime | None,
        expected_updated_at: datetime,
        evidence_occurred_at: datetime | None = None,
    ) -> tuple[ExperienceEventEvidence, ExperienceEvent, bool]:
        if evidence_occurred_at is not None and (
                evidence_occurred_at.tzinfo is None or evidence_occurred_at.utcoffset() is None):
            raise ValueError("Evidence occurrence time must be timezone-aware")
        safe_metadata = normalize_evidence_metadata(metadata or {})
        normalized_modalities = _normalize_modalities(modalities)
        if source_sequence is not None and source_sequence < 0:
            raise ValueError("source_sequence cannot be negative")
        _require_score(importance, "importance")
        _require_score(confidence, "confidence")

        async with self._pool.acquire() as conn:
            async with conn.transaction():
                event = await _lock_event(
                    conn,
                    owner_id=owner_id,
                    session_id=session_id,
                    event_id=event_id,
                )
                existing = await conn.fetchrow(
                    """
                    select *
                    from experience_event_evidence
                    where event_id = $1 and evidence_type = $2 and evidence_ref_id = $3
                    """,
                    event_id,
                    evidence_type,
                    evidence_ref_id,
                )
                if existing is not None:
                    return _evidence_from_row(existing), event, False
                if event.status != "open":
                    raise ExperienceEventClosedError()
                if event.updated_at != expected_updated_at:
                    raise ExperienceConflictError(
                        "event aggregate was based on stale state"
                    )

                evidence_row = await _insert_evidence(
                    conn,
                    event=event,
                    evidence_type=evidence_type,
                    evidence_ref_id=evidence_ref_id,
                    source_sequence=source_sequence,
                    metadata=safe_metadata,
                )
                if evidence_row is None:
                    raise ExperienceEvidenceScopeError()

                event_row = await conn.fetchrow(
                    """
                    update experience_events
                    set event_kind = $4,
                        modalities = $5::text[],
                        title = $6,
                        summary = $7,
                        scene = $8,
                        activity = $9,
                        state = $10::jsonb,
                        importance = $11,
                        confidence = $12,
                        retention_expires_at = $13,
                        started_at = least(started_at, coalesce($14::timestamptz, started_at)),
                        updated_at = now()
                    where owner_id = $1
                      and session_id = $2
                      and id = $3
                      and status = 'open'
                    returning *
                    """,
                    owner_id,
                    session_id,
                    event_id,
                    event_kind,
                    list(normalized_modalities),
                    title,
                    summary,
                    scene,
                    activity,
                    _json_dumps(dict(state)),
                    importance,
                    confidence,
                    retention_expires_at,
                    evidence_occurred_at,
                )
                if event_row is None:
                    raise ExperienceConflictError(
                        "open event changed during evidence attachment"
                    )
                return (
                    _evidence_from_row(evidence_row),
                    _event_from_row(event_row),
                    True,
                )

    async def list_event_evidence(
        self,
        *,
        owner_id: str,
        session_id: UUID,
        event_id: UUID,
    ) -> list[ExperienceEventEvidence]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                select evidence.*
                from experience_event_evidence evidence
                join experience_events events on events.id = evidence.event_id
                where events.owner_id = $1 and events.session_id = $2 and events.id = $3
                order by evidence.occurred_at asc,
                         evidence.source_sequence asc nulls last,
                         evidence.evidence_type asc,
                         evidence.evidence_ref_id asc
                """,
                owner_id,
                session_id,
                event_id,
            )
        return [_evidence_from_row(row) for row in rows]

    async def list_nonvisual_source_payloads(
        self,
        *,
        owner_id: str,
        session_id: UUID,
        event_id: UUID,
    ) -> dict[tuple[str, UUID], dict[str, Any]]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                select evidence.evidence_type,
                       evidence.evidence_ref_id,
                       messages.role as message_role,
                       messages.modality as message_modality,
                       messages.content as message_content,
                       messages.metadata as message_metadata,
                       tools.tool_name,
                       tools.mutates_state as tool_mutates_state,
                       tools.arguments as tool_arguments,
                       tools.result as tool_result,
                       tools.status as tool_status,
                       tools.error as tool_error
                from experience_event_evidence evidence
                join experience_events events on events.id = evidence.event_id
                left join jarvis_messages messages
                  on evidence.evidence_type = 'message'
                 and messages.id = evidence.evidence_ref_id
                 and messages.owner_id = events.owner_id
                 and messages.session_id = events.session_id
                left join jarvis_tool_executions tools
                  on evidence.evidence_type = 'tool_execution'
                 and tools.id = evidence.evidence_ref_id
                 and tools.owner_id = events.owner_id
                 and tools.session_id = events.session_id
                where events.owner_id = $1
                  and events.session_id = $2
                  and events.id = $3
                  and evidence.evidence_type in ('message', 'tool_execution')
                order by evidence.occurred_at asc,
                         evidence.source_sequence asc nulls last,
                         evidence.evidence_type asc,
                         evidence.evidence_ref_id asc
                """,
                owner_id,
                session_id,
                event_id,
            )
        payloads: dict[tuple[str, UUID], dict[str, Any]] = {}
        for row in rows:
            evidence_type = row["evidence_type"]
            evidence_ref_id = row["evidence_ref_id"]
            if evidence_type == "message":
                if row["message_role"] is None:
                    continue
                payload = {
                    "role": row["message_role"],
                    "modality": row["message_modality"],
                    "content": row["message_content"],
                    "metadata": _json_object(row["message_metadata"]),
                }
            else:
                if row["tool_name"] is None:
                    continue
                payload = {
                    "tool_name": row["tool_name"],
                    "mutates_state": bool(row["tool_mutates_state"]),
                    "arguments": _json_object(row["tool_arguments"]),
                    "result": _optional_json_object(row["tool_result"]),
                    "status": row["tool_status"],
                    "error": _optional_json_object(row["tool_error"]),
                }
            payloads[(evidence_type, evidence_ref_id)] = payload
        return payloads

    async def list_context_messages(
        self,
        *,
        owner_id: str,
        session_id: UUID,
        started_at: datetime,
        ended_at: datetime,
        limit: int,
    ) -> list[dict[str, Any]]:
        if limit < 1:
            raise ValueError("message context limit must be positive")
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                select id, role, modality, content, metadata, created_at
                from jarvis_messages
                where owner_id = $1
                  and session_id = $2
                  and created_at between $3 and $4
                order by created_at asc, id asc
                limit $5
                """,
                owner_id,
                session_id,
                started_at,
                ended_at,
                limit,
            )
        return [
            {
                "id": row["id"],
                "created_at": row["created_at"],
                "payload": {
                    "role": row["role"],
                    "modality": row["modality"],
                    "content": row["content"],
                    "metadata": _json_object(row["metadata"]),
                },
            }
            for row in rows
        ]

    async def list_event_keyframe_sources(
        self,
        *,
        owner_id: str,
        session_id: UUID,
        event_id: UUID,
        limit: int,
    ) -> list[dict[str, Any]]:
        if limit < 1:
            raise ValueError("keyframe source limit must be positive")
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                select keyframes.event_id,
                       keyframes.visual_frame_id,
                       keyframes.rank,
                       keyframes.reason,
                       keyframes.quality_score,
                       keyframes.selected_by,
                       keyframes.created_at,
                       frames.captured_at,
                       frames.frame_description,
                       frames.storage_bucket,
                       frames.storage_path,
                       frames.content_type,
                       frames.raw_image_status
                from experience_event_keyframes keyframes
                join experience_events events on events.id = keyframes.event_id
                join visual_frames frames
                  on frames.id = keyframes.visual_frame_id
                 and frames.owner_id = events.owner_id
                 and frames.session_id = events.session_id
                where events.owner_id = $1
                  and events.session_id = $2
                  and events.id = $3
                order by keyframes.rank asc, keyframes.visual_frame_id asc
                limit $4
                """,
                owner_id,
                session_id,
                event_id,
                limit,
            )
        return [dict(row) for row in rows]

    async def begin_consolidation(
        self,
        *,
        owner_id: str,
        session_id: UUID,
        event_id: UUID,
    ) -> ExperienceEvent:
        async with self._pool.acquire() as conn, conn.transaction():
            row = await conn.fetchrow(
                """
                update experience_events
                set status = 'consolidating', updated_at = now()
                where owner_id = $1
                  and session_id = $2
                  and id = $3
                  and status = 'closed'
                  and consolidation_ready_at is not null
                returning *
                """,
                owner_id,
                session_id,
                event_id,
            )
            if row is not None:
                return _event_from_row(row)
            existing = await conn.fetchrow(
                """
                select *
                from experience_events
                where owner_id = $1 and session_id = $2 and id = $3
                for update
                """,
                owner_id,
                session_id,
                event_id,
            )
        if existing is None:
            raise ExperienceEventMissingError()
        event = _event_from_row(existing)
        if event.status not in {"consolidating", "consolidated"}:
            raise ExperienceConflictError(
                f"event cannot begin consolidation from {event.status}"
            )
        return event

    async def finish_consolidation_failure(
        self,
        *,
        owner_id: str,
        session_id: UUID,
        event_id: UUID,
        terminal: bool,
    ) -> ExperienceEvent:
        next_status = "failed" if terminal else "closed"
        async with self._pool.acquire() as conn, conn.transaction():
            row = await conn.fetchrow(
                """
                update experience_events
                set status = $4, updated_at = now()
                where owner_id = $1
                  and session_id = $2
                  and id = $3
                  and status = 'consolidating'
                returning *
                """,
                owner_id,
                session_id,
                event_id,
                next_status,
            )
            if row is not None:
                return _event_from_row(row)
            existing = await conn.fetchrow(
                """
                select *
                from experience_events
                where owner_id = $1 and session_id = $2 and id = $3
                """,
                owner_id,
                session_id,
                event_id,
            )
        if existing is None:
            raise ExperienceEventMissingError()
        return _event_from_row(existing)

    async def store_consolidation_result(
        self,
        *,
        owner_id: str,
        session_id: UUID,
        event_id: UUID,
        result: Mapping[str, Any],
        provider: str,
        model: str,
        consolidated_at: datetime,
        connection: asyncpg.Connection,
    ) -> ExperienceEvent:
        if not connection.is_in_transaction():
            raise ValueError("consolidation persistence requires a transaction")
        safe_result = normalize_durable_tool_payload(
            result,
            label="consolidation result",
            max_bytes=262_144,
        )
        normalized_provider = provider.strip()
        normalized_model = model.strip()
        if not normalized_provider or not normalized_model:
            raise ValueError("consolidation provenance cannot be blank")
        row = await connection.fetchrow(
            """
            update experience_events
            set status = 'consolidated',
                title = $4,
                summary = $5,
                scene = coalesce($6, scene),
                importance = $7,
                confidence = $8,
                consolidation_result = $9::jsonb,
                consolidation_provider = $10,
                consolidation_model = $11,
                consolidated_at = $12,
                updated_at = now()
            where owner_id = $1
              and session_id = $2
              and id = $3
              and status in ('closed', 'consolidating')
            returning *
            """,
            owner_id,
            session_id,
            event_id,
            safe_result["event_title"],
            safe_result["event_summary"],
            safe_result.get("location"),
            safe_result["episode_importance"],
            safe_result["confidence"],
            _json_dumps(safe_result),
            normalized_provider,
            normalized_model,
            consolidated_at,
        )
        if row is not None:
            return _event_from_row(row)
        existing_row = await connection.fetchrow(
            """
            select *
            from experience_events
            where owner_id = $1 and session_id = $2 and id = $3
            for update
            """,
            owner_id,
            session_id,
            event_id,
        )
        if existing_row is None:
            raise ExperienceEventMissingError()
        existing = _event_from_row(existing_row)
        if (
            existing.status != "consolidated"
            or dict(existing.consolidation_result or {}) != safe_result
            or existing.consolidation_provider != normalized_provider
            or existing.consolidation_model != normalized_model
        ):
            raise ExperienceConflictError(
                "event already has a different consolidation outcome"
            )
        return existing

    async def renew_event_candidate_leases(
        self,
        *,
        owner_id: str,
        event_id: UUID,
        expires_at: datetime,
    ) -> int:
        async with self._pool.acquire() as conn:
            result = await conn.execute(
                """
                update visual_frame_retention_reasons
                set expires_at = greatest(expires_at, $3)
                where owner_id = $1
                  and reason_type = 'event_candidate'
                  and reason_ref_id = $2
                  and expires_at is not null
                """,
                owner_id,
                event_id,
                expires_at,
            )
        return int(result.rsplit(" ", 1)[-1])

    async def get_session_diagnostics(
        self,
        *,
        owner_id: str,
        session_id: UUID,
    ) -> dict[str, Any]:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                select
                    (
                        select count(*)
                        from experience_events
                        where owner_id = $1 and session_id = $2 and status = 'open'
                    ) as open_event_count,
                    (
                        select count(*)
                        from experience_events
                        where owner_id = $1
                          and session_id = $2
                          and status = 'closed'
                          and consolidation_ready_at is not null
                    ) as awaiting_consolidation_count,
                    (
                        select count(*)
                        from experience_event_evidence evidence
                        join experience_events events on events.id = evidence.event_id
                        where events.owner_id = $1
                          and events.session_id = $2
                          and evidence.evidence_type = 'observation'
                    ) as attached_observation_count,
                    (
                        select max(ended_at)
                        from experience_events
                        where owner_id = $1 and session_id = $2 and status <> 'open'
                    ) as last_closed_at
                """,
                owner_id,
                session_id,
            )
        return dict(row) if row is not None else {}

    async def close_event(
        self,
        *,
        owner_id: str,
        session_id: UUID,
        event_id: UUID,
        ended_at: datetime,
        boundary_reason: str,
        keyframes_finalized_at: datetime,
        consolidation_ready_at: datetime,
    ) -> ExperienceEvent:
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                event = await _lock_event(
                    conn,
                    owner_id=owner_id,
                    session_id=session_id,
                    event_id=event_id,
                )
                if event.status != "open":
                    return event
                if ended_at < event.started_at:
                    raise ValueError("event end cannot be before start")
                row = await conn.fetchrow(
                    """
                    update experience_events
                    set status = 'closed',
                        ended_at = $4,
                        boundary_reason = $5,
                        keyframes_finalized_at = $6,
                        consolidation_ready_at = $7,
                        updated_at = now()
                    where owner_id = $1 and session_id = $2 and id = $3 and status = 'open'
                    returning *
                    """,
                    owner_id,
                    session_id,
                    event_id,
                    ended_at,
                    boundary_reason,
                    keyframes_finalized_at,
                    consolidation_ready_at,
                )
                if row is None:
                    raise ExperienceConflictError("open event changed while closing")
                closed = _event_from_row(row)
                await self._enqueue_consolidation_job(conn, closed)
                return closed

    async def reconcile_event_candidates(
        self,
        *,
        owner_id: str,
        session_id: UUID,
        event_id: UUID,
        observation_id: UUID,
        proposed_frame_ids: Sequence[UUID],
        expires_at: datetime,
        max_count: int,
        selector: Callable[
            [Sequence[EventKeyframeCandidate]], Sequence[ScoredEventKeyframe]
        ],
    ) -> tuple[ScoredEventKeyframe, ...]:
        if max_count < 1:
            raise ValueError("candidate max count must be positive")
        frame_ids = tuple(dict.fromkeys(proposed_frame_ids))
        if not frame_ids:
            return ()
        async with self._pool.acquire() as conn, conn.transaction():
            event = await _lock_event(
                conn,
                owner_id=owner_id,
                session_id=session_id,
                event_id=event_id,
            )
            if event.status != "open":
                raise ExperienceEventClosedError()
            valid_count = await conn.fetchval(
                """
                select count(distinct frames.visual_frame_id)
                from experience_event_evidence evidence
                join perception_observations observations
                  on observations.id = evidence.evidence_ref_id
                join perception_observation_frames frames
                  on frames.observation_id = observations.id
                where evidence.event_id = $1
                  and evidence.evidence_type = 'observation'
                  and observations.owner_id = $2
                  and observations.session_id = $3
                  and observations.id = $4
                  and frames.visual_frame_id = any($5::uuid[])
                  and exists (
                      select 1
                      from jsonb_array_elements(observations.keyframe_candidates) item
                      where item->>'visual_frame_id' = frames.visual_frame_id::text
                  )
                """,
                event_id,
                owner_id,
                session_id,
                observation_id,
                list(frame_ids),
            )
            if int(valid_count or 0) != len(frame_ids):
                raise ExperienceEvidenceScopeError()
            candidates = await _list_event_keyframe_candidates(conn, event_id=event_id)
            selected = tuple(selector(candidates))
            _validate_selected_candidates(selected, candidates, max_count=max_count)
            await _replace_event_candidate_reasons(
                conn,
                event=event,
                selected_frame_ids=tuple(
                    item.candidate.visual_frame_id for item in selected
                ),
                expires_at=expires_at,
            )
            return selected

    async def close_event_with_keyframes(
        self,
        *,
        owner_id: str,
        session_id: UUID,
        event_id: UUID,
        ended_at: datetime,
        boundary_reason: str,
        finalized_at: datetime,
        candidate_expires_at: datetime,
        max_count: int,
        selector: Callable[
            [Sequence[EventKeyframeCandidate]], Sequence[ScoredEventKeyframe]
        ],
    ) -> ExperienceEvent:
        if max_count < 1:
            raise ValueError("keyframe max count must be positive")
        async with self._pool.acquire() as conn, conn.transaction():
            event = await _lock_event(
                conn,
                owner_id=owner_id,
                session_id=session_id,
                event_id=event_id,
            )
            if event.status != "open":
                return event
            if ended_at < event.started_at:
                raise ValueError("event end cannot be before start")

            existing_rows = await conn.fetch(
                """
                select *
                from experience_event_keyframes
                where event_id = $1
                order by rank asc, visual_frame_id asc
                """,
                event_id,
            )
            if len(existing_rows) > max_count:
                raise ExperienceConflictError(
                    "event already has more keyframes than the close policy permits"
                )
            occupied_ranks = {int(row["rank"]) for row in existing_rows}
            existing_frame_ids = tuple(row["visual_frame_id"] for row in existing_rows)
            existing_frame_id_set = set(existing_frame_ids)
            if existing_frame_ids:
                available_existing_frame_ids = await conn.fetch(
                    """
                    select id
                    from visual_frames
                    where owner_id = $1
                      and session_id = $2
                      and id = any($3::uuid[])
                      and raw_image_status = 'available'
                    for update
                    """,
                    owner_id,
                    session_id,
                    list(existing_frame_ids),
                )
                if len(available_existing_frame_ids) != len(existing_frame_ids):
                    raise ExperienceConflictError(
                        "an existing event keyframe no longer has recoverable raw evidence"
                    )
            remaining_count = max_count - len(existing_rows)
            candidates = tuple(
                candidate
                for candidate in await _list_event_keyframe_candidates(
                    conn,
                    event_id=event_id,
                )
                if candidate.visual_frame_id not in existing_frame_id_set
            )
            selected = (
                tuple(selector(candidates))[:remaining_count] if remaining_count else ()
            )
            _validate_selected_candidates(
                selected,
                candidates,
                max_count=remaining_count,
            )
            if selected:
                available_ranks = [
                    rank
                    for rank in range(1, max_count + 1)
                    if rank not in occupied_ranks
                ]
                await conn.execute(
                    """
                    insert into experience_event_keyframes (
                        event_id,
                        visual_frame_id,
                        rank,
                        reason,
                        quality_score,
                        selected_by
                    )
                    select $1, selected.visual_frame_id, selected.rank,
                           selected.reason, selected.quality_score, 'deterministic'
                    from unnest(
                        $2::uuid[],
                        $3::integer[],
                        $4::text[],
                        $5::double precision[]
                    ) as selected(visual_frame_id, rank, reason, quality_score)
                    """,
                    event_id,
                    [item.candidate.visual_frame_id for item in selected],
                    available_ranks[: len(selected)],
                    [item.candidate.reason for item in selected],
                    [item.quality_score for item in selected],
                )
            await _replace_event_candidate_reasons(
                conn,
                event=event,
                selected_frame_ids=(
                    *existing_frame_ids,
                    *(item.candidate.visual_frame_id for item in selected),
                ),
                expires_at=candidate_expires_at,
            )
            row = await conn.fetchrow(
                """
                update experience_events
                set status = 'closed',
                    ended_at = $4,
                    boundary_reason = $5,
                    keyframes_finalized_at = $6,
                    consolidation_ready_at = $6,
                    updated_at = now()
                where owner_id = $1 and session_id = $2 and id = $3 and status = 'open'
                returning *
                """,
                owner_id,
                session_id,
                event_id,
                ended_at,
                boundary_reason,
                finalized_at,
            )
            if row is None:
                raise ExperienceConflictError("open event changed while closing")
            closed = _event_from_row(row)
            await self._enqueue_consolidation_job(conn, closed)
            return closed

    async def _enqueue_consolidation_job(
        self,
        connection: asyncpg.Connection,
        event: ExperienceEvent,
    ) -> None:
        if self._consolidation_job_repository is None:
            return
        from app.memory.consolidation.job_handler import (
            CONSOLIDATE_EXPERIENCE_EVENT_JOB,
            consolidation_dedupe_key,
        )

        await self._consolidation_job_repository.enqueue(
            owner_id=event.owner_id,
            session_id=event.session_id,
            job_type=CONSOLIDATE_EXPERIENCE_EVENT_JOB,
            dedupe_key=consolidation_dedupe_key(event.id),
            payload={"event_id": str(event.id)},
            available_at=event.consolidation_ready_at or event.updated_at,
            connection=connection,
        )

    async def select_keyframe(
        self,
        *,
        owner_id: str,
        session_id: UUID,
        event_id: UUID,
        visual_frame_id: UUID,
        rank: int,
        reason: str,
        quality_score: float,
        selected_by: KeyframeSelectedBy,
    ) -> ExperienceEventKeyframe:
        if rank < 1:
            raise ValueError("keyframe rank must be positive")
        if not reason.strip():
            raise ValueError("keyframe reason cannot be blank")
        _require_score(quality_score, "quality_score")
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                event = await _lock_event(
                    conn,
                    owner_id=owner_id,
                    session_id=session_id,
                    event_id=event_id,
                )
                if event.status != "open":
                    raise ExperienceEventClosedError()
                try:
                    row = await conn.fetchrow(
                        """
                        insert into experience_event_keyframes (
                            event_id,
                            visual_frame_id,
                            rank,
                            reason,
                            quality_score,
                            selected_by
                        )
                        select events.id, frames.id, $5, $6, $7, $8
                        from experience_events events
                        join visual_frames frames
                          on frames.id = $4
                         and frames.owner_id = events.owner_id
                         and frames.session_id = events.session_id
                        where events.owner_id = $1
                          and events.session_id = $2
                          and events.id = $3
                          and events.status = 'open'
                        on conflict (event_id, visual_frame_id) do nothing
                        returning *
                        """,
                        owner_id,
                        session_id,
                        event_id,
                        visual_frame_id,
                        rank,
                        reason,
                        quality_score,
                        selected_by,
                    )
                except asyncpg.UniqueViolationError as exc:
                    raise ExperienceConflictError(
                        "keyframe rank is already selected for this event"
                    ) from exc
                if row is not None:
                    return _keyframe_from_row(row)
                existing = await conn.fetchrow(
                    """
                    select keyframes.*
                    from experience_event_keyframes keyframes
                    join experience_events events on events.id = keyframes.event_id
                    where events.owner_id = $1
                      and events.session_id = $2
                      and keyframes.event_id = $3
                      and keyframes.visual_frame_id = $4
                    """,
                    owner_id,
                    session_id,
                    event_id,
                    visual_frame_id,
                )
        if existing is None:
            raise ExperienceEvidenceScopeError()
        keyframe = _keyframe_from_row(existing)
        if (
            keyframe.rank != rank
            or keyframe.reason != reason
            or keyframe.quality_score != quality_score
            or keyframe.selected_by != selected_by
        ):
            raise ExperienceConflictError("frame already selected with different keyframe data")
        return keyframe

    async def add_event_retention_reason(
        self,
        *,
        owner_id: str,
        event_id: UUID,
        reason_type: str,
        reason_ref_id: str,
        expires_at: datetime | None = None,
    ) -> bool:
        normalized_reason_type = _require_text(reason_type, "reason_type")
        normalized_reason_ref_id = _require_text(reason_ref_id, "reason_ref_id")
        async with self._pool.acquire() as conn:
            row = await conn.fetchval(
                """
                insert into experience_event_retention_reasons (
                    owner_id, experience_event_id, reason_type, reason_ref_id, expires_at
                )
                select event.owner_id, event.id, $3, $4, $5
                from experience_events event
                where event.owner_id = $1 and event.id = $2
                on conflict (experience_event_id, reason_type, reason_ref_id) do update
                set expires_at = case
                    when experience_event_retention_reasons.expires_at is null
                      or excluded.expires_at is null then null
                    else greatest(
                        experience_event_retention_reasons.expires_at,
                        excluded.expires_at
                    )
                end
                returning true
                """,
                owner_id,
                event_id,
                normalized_reason_type,
                normalized_reason_ref_id,
                expires_at,
            )
        if not row:
            raise ExperienceEventMissingError()
        return True

    async def remove_event_retention_reason(
        self,
        *,
        owner_id: str,
        event_id: UUID,
        reason_type: str,
        reason_ref_id: str,
    ) -> bool:
        async with self._pool.acquire() as conn:
            removed = await conn.fetchval(
                """
                delete from experience_event_retention_reasons
                where owner_id = $1
                  and experience_event_id = $2
                  and reason_type = $3
                  and reason_ref_id = $4
                returning true
                """,
                owner_id,
                event_id,
                _require_text(reason_type, "reason_type"),
                _require_text(reason_ref_id, "reason_ref_id"),
            )
        return bool(removed)

    async def add_retention_reason(
        self,
        *,
        owner_id: str,
        visual_frame_id: UUID,
        reason_type: RetentionReasonType,
        reason_ref_id: UUID,
        expires_at: datetime | None,
    ) -> VisualFrameRetentionReason:
        _validate_retention_expiry(reason_type, expires_at)
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                insert into visual_frame_retention_reasons (
                    owner_id,
                    visual_frame_id,
                    reason_type,
                    reason_ref_id,
                    expires_at
                )
                with retainable_frame as (
                    select owner_id, id
                    from visual_frames
                    where owner_id = $1
                      and id = $2
                      and raw_image_status = 'available'
                    for update
                )
                select owner_id, id, $3, $4, $5
                from retainable_frame
                on conflict (visual_frame_id, reason_type, reason_ref_id) do update
                set expires_at = case
                    when visual_frame_retention_reasons.expires_at is null
                      or excluded.expires_at is null then null
                    else greatest(visual_frame_retention_reasons.expires_at, excluded.expires_at)
                end
                returning *
                """,
                owner_id,
                visual_frame_id,
                reason_type,
                reason_ref_id,
                expires_at,
            )
        if row is None:
            raise ExperienceEvidenceScopeError()
        return _retention_reason_from_row(row)

    async def remove_retention_reason(
        self,
        *,
        owner_id: str,
        visual_frame_id: UUID,
        reason_type: RetentionReasonType,
        reason_ref_id: UUID,
    ) -> bool:
        async with self._pool.acquire() as conn:
            removed = await conn.fetchval(
                """
                delete from visual_frame_retention_reasons
                where owner_id = $1
                  and visual_frame_id = $2
                  and reason_type = $3
                  and reason_ref_id = $4
                returning true
                """,
                owner_id,
                visual_frame_id,
                reason_type,
                reason_ref_id,
            )
        return bool(removed)


async def _list_event_keyframe_candidates(
    conn: asyncpg.Connection,
    *,
    event_id: UUID,
) -> tuple[EventKeyframeCandidate, ...]:
    rows = await conn.fetch(
        """
        select
            frames.id as visual_frame_id,
            frames.captured_at,
            candidate.item->>'reason' as reason,
            (candidate.item->>'confidence')::double precision as analyzer_score,
            observations.confidence as observation_confidence,
            jsonb_array_length(observations.visible_text) > 0 as has_readable_text,
            jsonb_array_length(observations.state_changes) > 0 as has_state_change,
            jsonb_array_length(observations.entities) as entity_count
        from experience_event_evidence evidence
        join perception_observations observations
          on observations.id = evidence.evidence_ref_id
        join lateral jsonb_array_elements(observations.keyframe_candidates)
          as candidate(item) on true
        join perception_observation_frames observation_frames
          on observation_frames.observation_id = observations.id
         and observation_frames.visual_frame_id::text = candidate.item->>'visual_frame_id'
        join visual_frames frames
          on frames.id = observation_frames.visual_frame_id
         and frames.owner_id = observations.owner_id
         and frames.session_id = observations.session_id
        where evidence.event_id = $1
          and evidence.evidence_type = 'observation'
          and frames.raw_image_status = 'available'
        order by frames.captured_at asc, frames.id asc, observations.id asc
        for update of frames
        """,
        event_id,
    )
    return tuple(
        EventKeyframeCandidate(
            visual_frame_id=row["visual_frame_id"],
            captured_at=row["captured_at"],
            reason=row["reason"],
            analyzer_score=float(row["analyzer_score"]),
            observation_confidence=float(row["observation_confidence"]),
            has_readable_text=bool(row["has_readable_text"]),
            has_state_change=bool(row["has_state_change"]),
            entity_count=int(row["entity_count"]),
        )
        for row in rows
    )


async def _replace_event_candidate_reasons(
    conn: asyncpg.Connection,
    *,
    event: ExperienceEvent,
    selected_frame_ids: Sequence[UUID],
    expires_at: datetime,
) -> None:
    selected_ids = list(dict.fromkeys(selected_frame_ids))
    if selected_ids:
        await conn.execute(
            """
            with retainable_frames as (
                select frames.id, events.owner_id
                from experience_events events
                join visual_frames frames
                  on frames.id = any($5::uuid[])
                 and frames.owner_id = events.owner_id
                 and frames.session_id = events.session_id
                where events.owner_id = $1
                  and events.session_id = $2
                  and events.id = $3
                  and frames.raw_image_status = 'available'
                for update of frames
            )
            insert into visual_frame_retention_reasons (
                owner_id,
                visual_frame_id,
                reason_type,
                reason_ref_id,
                expires_at
            )
            select owner_id, id, 'event_candidate', $3, $4
            from retainable_frames
            on conflict (visual_frame_id, reason_type, reason_ref_id) do update
            set expires_at = greatest(
                visual_frame_retention_reasons.expires_at,
                excluded.expires_at
            )
            """,
            event.owner_id,
            event.session_id,
            event.id,
            expires_at,
            selected_ids,
        )
    await conn.execute(
        """
        delete from visual_frame_retention_reasons
        where owner_id = $1
          and reason_type = 'event_candidate'
          and reason_ref_id = $2
          and not (visual_frame_id = any($3::uuid[]))
        """,
        event.owner_id,
        event.id,
        selected_ids,
    )


def _validate_selected_candidates(
    selected: Sequence[ScoredEventKeyframe],
    candidates: Sequence[EventKeyframeCandidate],
    *,
    max_count: int,
) -> None:
    if len(selected) > max_count:
        raise ValueError("keyframe selector exceeded its maximum count")
    candidate_ids = {item.visual_frame_id for item in candidates}
    selected_ids = [item.candidate.visual_frame_id for item in selected]
    if len(selected_ids) != len(set(selected_ids)):
        raise ValueError("keyframe selector returned duplicate frames")
    if not set(selected_ids).issubset(candidate_ids):
        raise ValueError("keyframe selector returned a frame outside the event")


async def _lock_event(
    conn: asyncpg.Connection,
    *,
    owner_id: str,
    session_id: UUID,
    event_id: UUID,
) -> ExperienceEvent:
    row = await conn.fetchrow(
        """
        select *
        from experience_events
        where owner_id = $1 and session_id = $2 and id = $3
        for update
        """,
        owner_id,
        session_id,
        event_id,
    )
    if row is None:
        raise ExperienceEventMissingError()
    return _event_from_row(row)


async def _insert_evidence(
    conn: asyncpg.Connection,
    *,
    event: ExperienceEvent,
    evidence_type: ExperienceEvidenceType,
    evidence_ref_id: UUID,
    source_sequence: int | None,
    metadata: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    sources = {
        "observation": ("perception_observations", "source.ended_at"),
        "message": ("jarvis_messages", "source.created_at"),
        "tool_execution": (
            "jarvis_tool_executions",
            "coalesce(source.completed_at, source.started_at)",
        ),
    }
    if evidence_type not in sources:
        raise ValueError(f"unsupported evidence type: {evidence_type}")
    table, timestamp_expression = sources[evidence_type]
    return await conn.fetchrow(
        f"""
        insert into experience_event_evidence (
            event_id,
            evidence_type,
            evidence_ref_id,
            occurred_at,
            source_sequence,
            metadata
        )
        select $1, $2, source.id, {timestamp_expression}, $4, $5::jsonb
        from {table} source
        where source.id = $3
          and source.owner_id = $6
          and source.session_id = $7
        on conflict (event_id, evidence_type, evidence_ref_id) do nothing
        returning *
        """,
        event.id,
        evidence_type,
        evidence_ref_id,
        source_sequence,
        _json_dumps(dict(metadata)),
        event.owner_id,
        event.session_id,
    )


def _tool_execution_from_row(row: Mapping[str, Any]) -> ToolExecution:
    return ToolExecution(
        id=row["id"],
        owner_id=row["owner_id"],
        session_id=row["session_id"],
        provider_call_id=row["provider_call_id"],
        tool_name=row["tool_name"],
        mutates_state=bool(row["mutates_state"]),
        arguments=_json_object(row["arguments"]),
        result=_optional_json_object(row["result"]),
        status=row["status"],
        error=_optional_json_object(row["error"]),
        started_at=row["started_at"],
        completed_at=row["completed_at"],
        created_at=row["created_at"],
    )


def _event_from_row(row: Mapping[str, Any]) -> ExperienceEvent:
    return ExperienceEvent(
        id=row["id"],
        owner_id=row["owner_id"],
        session_id=row["session_id"],
        status=row["status"],
        event_kind=row["event_kind"],
        modalities=tuple(row["modalities"] or ()),
        started_at=row["started_at"],
        ended_at=row["ended_at"],
        title=row["title"],
        summary=row["summary"],
        scene=row["scene"],
        activity=row["activity"],
        state=_json_object(row["state"]),
        boundary_reason=row["boundary_reason"],
        importance=float(row["importance"]),
        confidence=float(row["confidence"]),
        keyframes_finalized_at=row["keyframes_finalized_at"],
        consolidation_ready_at=row["consolidation_ready_at"],
        evidence_reduced_at=row["evidence_reduced_at"],
        retention_expires_at=row["retention_expires_at"],
        consolidation_result=_optional_json_object(row["consolidation_result"]),
        consolidation_provider=row["consolidation_provider"],
        consolidation_model=row["consolidation_model"],
        consolidated_at=row["consolidated_at"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _evidence_from_row(row: Mapping[str, Any]) -> ExperienceEventEvidence:
    return ExperienceEventEvidence(
        id=row["id"],
        event_id=row["event_id"],
        evidence_type=row["evidence_type"],
        evidence_ref_id=row["evidence_ref_id"],
        occurred_at=row["occurred_at"],
        source_sequence=row["source_sequence"],
        ingested_at=row["ingested_at"],
        metadata=_json_object(row["metadata"]),
    )


def _keyframe_from_row(row: Mapping[str, Any]) -> ExperienceEventKeyframe:
    return ExperienceEventKeyframe(
        event_id=row["event_id"],
        visual_frame_id=row["visual_frame_id"],
        rank=row["rank"],
        reason=row["reason"],
        quality_score=float(row["quality_score"]),
        selected_by=row["selected_by"],
        created_at=row["created_at"],
    )


def _retention_reason_from_row(row: Mapping[str, Any]) -> VisualFrameRetentionReason:
    return VisualFrameRetentionReason(
        id=row["id"],
        owner_id=row["owner_id"],
        visual_frame_id=row["visual_frame_id"],
        reason_type=row["reason_type"],
        reason_ref_id=row["reason_ref_id"],
        expires_at=row["expires_at"],
        created_at=row["created_at"],
    )


def _json_object(value: object) -> dict[str, Any]:
    decoded = json.loads(value) if isinstance(value, str) else value
    if not isinstance(decoded, Mapping):
        raise TypeError("expected a JSON object")
    return dict(decoded)


def _optional_json_object(value: object) -> dict[str, Any] | None:
    return None if value is None else _json_object(value)


def _optional_mapping(value: Mapping[str, Any] | None) -> dict[str, Any] | None:
    return dict(value) if value is not None else None


def _json_dumps(value: Mapping[str, Any]) -> str:
    return json.dumps(value, allow_nan=False, separators=(",", ":"))


def _optional_json_dumps(value: Mapping[str, Any] | None) -> str | None:
    return _json_dumps(value) if value is not None else None


def _normalize_modalities(values: Sequence[ExperienceModality]) -> tuple[ExperienceModality, ...]:
    normalized = tuple(dict.fromkeys(values))
    if not set(normalized).issubset({"audio", "visual", "text", "tool"}):
        raise ValueError("event contains an unsupported modality")
    return normalized


def _require_score(value: float, label: str) -> None:
    if not isfinite(value) or value < 0 or value > 1:
        raise ValueError(f"{label} must be between zero and one")


def _validate_retention_expiry(
    reason_type: RetentionReasonType,
    expires_at: datetime | None,
) -> None:
    permanent_types = {"important_event_keyframe", "user_capture", "explicit_memory"}
    if expires_at is None and reason_type not in permanent_types:
        raise ValueError(f"{reason_type} requires a finite expiry")
