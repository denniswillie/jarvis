from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, cast
from uuid import UUID

import asyncpg

from app.experience.models import ExperienceEvent, ExperienceEventKeyframe
from app.experience.repository import ExperienceRepository
from app.memory.consolidation.contracts import ConsolidationResult
from app.memory.embeddings import PreparedEmbeddingWrite
from app.memory.models import (
    EMBEDDING_DIMENSIONS,
    MemoryItem,
    content_hash,
    normalize_embedding,
)
from app.memory.repository import MemoryRepository
from app.memory.episodic.service import (
    EpisodePromotionContext,
    EpisodePromotionPlan,
    ReminderDeliveryEvidence,
    evidence_excerpt,
)


@dataclass(frozen=True, slots=True)
class EpisodePromotionOutcome:
    event_id: UUID
    decision: str
    reason: str
    episode_memory_id: UUID | None
    selected_keyframe_ids: tuple[UUID, ...]


class EpisodicMemoryRepository:
    """Transactional persistence boundary for episode promotion and embeddings."""

    def __init__(
        self,
        pool: asyncpg.Pool,
        *,
        experience_repository: ExperienceRepository,
        memory_repository: MemoryRepository,
    ) -> None:
        self._pool = pool
        self._experience_repository = experience_repository
        self._memory_repository = memory_repository

    async def load_promotion_context(
        self,
        *,
        owner_id: str,
        session_id: UUID,
        event_id: UUID,
    ) -> EpisodePromotionContext:
        event = await self._experience_repository.get_event(
            owner_id=owner_id,
            session_id=session_id,
            event_id=event_id,
        )
        if event is None or event.consolidation_result is None:
            raise ValueError("consolidated event was not found")
        result = ConsolidationResult.model_validate(event.consolidation_result)
        evidence = tuple(
            await self._experience_repository.list_event_evidence(
                owner_id=owner_id,
                session_id=session_id,
                event_id=event_id,
            )
        )
        source_payloads = (
            await self._experience_repository.list_nonvisual_source_payloads(
                owner_id=owner_id,
                session_id=session_id,
                event_id=event_id,
            )
        )
        context_messages: tuple[Mapping[str, Any], ...] = ()
        if event.ended_at is not None:
            context_messages = tuple(
                await self._experience_repository.list_context_messages(
                    owner_id=owner_id,
                    session_id=session_id,
                    started_at=event.started_at - timedelta(seconds=5),
                    ended_at=event.ended_at + timedelta(seconds=5),
                    limit=20,
                )
            )
        keyframe_rows = await self._experience_repository.list_event_keyframe_sources(
            owner_id=owner_id,
            session_id=session_id,
            event_id=event_id,
            limit=6,
        )
        keyframes = tuple(
            ExperienceEventKeyframe(
                event_id=row["event_id"],
                visual_frame_id=row["visual_frame_id"],
                rank=row["rank"],
                reason=row["reason"],
                quality_score=float(row["quality_score"]),
                selected_by=row["selected_by"],
                created_at=row["created_at"],
            )
            for row in keyframe_rows
        )
        keyframe_capture_times = {
            row["visual_frame_id"]: row["captured_at"] for row in keyframe_rows
        }
        async with self._pool.acquire() as connection:
            signal_row = await connection.fetchrow(
                """
                select
                    exists (
                        select 1
                        from experience_event_evidence evidence
                        join jarvis_tool_executions tools
                          on evidence.evidence_type = 'tool_execution'
                         and tools.id = evidence.evidence_ref_id
                        where evidence.event_id = $3
                          and tools.owner_id = $1
                          and tools.session_id = $2
                          and tools.status = 'completed'
                          and tools.tool_name = 'capture_latest_visual_frame'
                    ) or exists (
                        select 1
                        from experience_event_keyframes keyframes
                        join visual_frame_retention_reasons reasons
                          on reasons.visual_frame_id = keyframes.visual_frame_id
                        where keyframes.event_id = $3
                          and reasons.owner_id = $1
                          and reasons.reason_type = 'explicit_memory'
                    ) as has_explicit_memory
                """,
                owner_id,
                session_id,
                event_id,
            )
            reminder_rows = await connection.fetch(
                """
                select id, reminder_id, created_at, message, metadata
                from jarvis_reminder_events
                where owner_id = $1
                  and session_id = $2
                  and event_type = 'delivered_to_session'
                  and created_at >= $3
                  and created_at <= $4
                order by created_at asc, id asc
                """,
                owner_id,
                session_id,
                event.started_at,
                event.ended_at or event.started_at,
            )
        return EpisodePromotionContext(
            event=event,
            result=result,
            evidence=evidence,
            source_payloads=source_payloads,
            context_messages=context_messages,
            provisional_keyframes=keyframes,
            keyframe_capture_times=keyframe_capture_times,
            has_explicit_memory_evidence=bool(signal_row["has_explicit_memory"]),
            reminder_deliveries=tuple(
                ReminderDeliveryEvidence(
                    reminder_event_id=row["id"],
                    reminder_id=row["reminder_id"],
                    occurred_at=row["created_at"],
                    message=row["message"],
                    metadata=_json_object(row["metadata"]),
                )
                for row in reminder_rows
            ),
        )

    async def apply_promotion(
        self,
        *,
        context: EpisodePromotionContext,
        plan: EpisodePromotionPlan,
        decided_at: datetime,
        ordinary_keyframe_retention_seconds: int,
        skipped_event_retention_days: int,
        connection: asyncpg.Connection,
    ) -> EpisodePromotionOutcome:
        if not connection.is_in_transaction():
            raise ValueError("episode promotion requires an active transaction")
        if plan.event_id != context.event.id or plan.owner_id != context.event.owner_id:
            raise ValueError("promotion plan does not match its event")
        row = await connection.fetchrow(
            """
            select *
            from experience_events
            where owner_id = $1 and session_id = $2 and id = $3
            for update
            """,
            plan.owner_id,
            plan.session_id,
            plan.event_id,
        )
        if row is None:
            raise ValueError("promotion event is outside the owner scope")
        if row["status"] != "consolidated" or row["consolidation_result"] is None:
            raise ValueError("promotion event is not consolidated")
        if row["promotion_decision"] != "pending":
            return await self._existing_outcome(connection, row)

        await connection.execute(
            "delete from experience_event_keyframes where event_id = $1",
            plan.event_id,
        )
        for keyframe in plan.final_keyframes:
            await connection.execute(
                """
                insert into experience_event_keyframes (
                    event_id,
                    visual_frame_id,
                    rank,
                    reason,
                    quality_score,
                    selected_by
                )
                select $1, frames.id, $3, $4, $5, $6
                from visual_frames frames
                where frames.owner_id = $2 and frames.id = $7
                """,
                plan.event_id,
                plan.owner_id,
                keyframe.rank,
                keyframe.reason,
                keyframe.quality_score,
                keyframe.selected_by,
                keyframe.visual_frame_id,
            )

        await connection.execute(
            """
            delete from visual_frame_retention_reasons
            where owner_id = $1
              and reason_type = 'event_candidate'
              and reason_ref_id = $2
            """,
            plan.owner_id,
            plan.event_id,
        )
        await self._apply_final_keyframe_retention(
            connection,
            plan=plan,
            decided_at=decided_at,
            ordinary_keyframe_retention_seconds=ordinary_keyframe_retention_seconds,
        )

        episode_id: UUID | None = None
        if plan.promote:
            episode_id = await self._create_episode(
                connection,
                context=context,
                plan=plan,
                decided_at=decided_at,
            )
            await self._attach_episode_evidence(
                connection,
                context=context,
                plan=plan,
                episode_id=episode_id,
            )
            await self._link_adjacent_episode(
                connection,
                context=context,
                episode_id=episode_id,
            )
            await connection.execute(
                """
                update experience_events
                set promotion_decision = 'promoted',
                    promotion_reason = $4,
                    promoted_episode_memory_id = $5,
                    promotion_decided_at = $6,
                    retention_expires_at = null,
                    updated_at = now()
                where owner_id = $1 and session_id = $2 and id = $3
                """,
                plan.owner_id,
                plan.session_id,
                plan.event_id,
                plan.reason,
                episode_id,
                decided_at,
            )
            decision = "promoted"
        else:
            retention_expires_at = decided_at + timedelta(
                days=skipped_event_retention_days
            )
            await connection.execute(
                """
                update experience_events
                set promotion_decision = 'skipped',
                    promotion_reason = $4,
                    promoted_episode_memory_id = null,
                    promotion_decided_at = $5,
                    retention_expires_at = $6,
                    updated_at = now()
                where owner_id = $1 and session_id = $2 and id = $3
                """,
                plan.owner_id,
                plan.session_id,
                plan.event_id,
                plan.reason,
                decided_at,
                retention_expires_at,
            )
            decision = "skipped"

        return EpisodePromotionOutcome(
            event_id=plan.event_id,
            decision=decision,
            reason=plan.reason,
            episode_memory_id=episode_id,
            selected_keyframe_ids=tuple(
                item.visual_frame_id for item in plan.final_keyframes
            ),
        )

    async def get_event_for_embedding(
        self,
        *,
        owner_id: str,
        session_id: UUID,
        event_id: UUID,
    ) -> ExperienceEvent:
        event = await self._experience_repository.get_event(
            owner_id=owner_id,
            session_id=session_id,
            event_id=event_id,
        )
        if event is None or event.consolidation_result is None:
            raise ValueError("event embedding source was not found")
        return event

    async def list_event_keyframe_sources(
        self,
        *,
        owner_id: str,
        session_id: UUID,
        event_id: UUID,
    ) -> list[Mapping[str, Any]]:
        return await self._experience_repository.list_event_keyframe_sources(
            owner_id=owner_id,
            session_id=session_id,
            event_id=event_id,
            limit=3,
        )

    async def get_memory_item(
        self,
        *,
        owner_id: str,
        memory_item_id: UUID,
    ) -> MemoryItem:
        memory = await self._memory_repository.get_memory_item(
            owner_id=owner_id,
            memory_item_id=memory_item_id,
        )
        if memory is None:
            raise ValueError("memory item was not found")
        return memory

    async def promoted_episode_id(
        self,
        *,
        owner_id: str,
        event_id: UUID,
        connection: asyncpg.Connection,
    ) -> UUID | None:
        return cast(
            UUID | None,
            await connection.fetchval(
                """
                select promoted_episode_memory_id
                from experience_events
                where owner_id = $1 and id = $2 and promotion_decision = 'promoted'
                """,
                owner_id,
                event_id,
            ),
        )

    async def store_event_embedding(
        self,
        *,
        owner_id: str,
        write: PreparedEmbeddingWrite,
        connection: asyncpg.Connection,
    ) -> None:
        await self._store_embedding(
            connection,
            owner_id=owner_id,
            subject="event",
            write=write,
        )

    async def store_memory_embedding(
        self,
        *,
        owner_id: str,
        write: PreparedEmbeddingWrite,
        connection: asyncpg.Connection,
    ) -> None:
        await self._store_embedding(
            connection,
            owner_id=owner_id,
            subject="memory",
            write=write,
        )

    async def _apply_final_keyframe_retention(
        self,
        connection: asyncpg.Connection,
        *,
        plan: EpisodePromotionPlan,
        decided_at: datetime,
        ordinary_keyframe_retention_seconds: int,
    ) -> None:
        if ordinary_keyframe_retention_seconds < 1:
            raise ValueError("ordinary keyframe retention must be positive")
        reason_type = (
            "important_event_keyframe"
            if plan.retain_keyframes_permanently
            else "ordinary_event_keyframe"
        )
        expires_at = (
            None
            if plan.retain_keyframes_permanently
            else decided_at + timedelta(seconds=ordinary_keyframe_retention_seconds)
        )
        for keyframe in plan.final_keyframes:
            await connection.execute(
                """
                insert into visual_frame_retention_reasons (
                    owner_id,
                    visual_frame_id,
                    reason_type,
                    reason_ref_id,
                    expires_at
                )
                values ($1, $2, $3, $4, $5)
                on conflict (visual_frame_id, reason_type, reason_ref_id)
                do update set expires_at = excluded.expires_at
                """,
                plan.owner_id,
                keyframe.visual_frame_id,
                reason_type,
                plan.event_id,
                expires_at,
            )
            if not plan.retain_keyframes_permanently:
                await connection.execute(
                    """
                    update visual_frames
                    set raw_image_expires_at = greatest(raw_image_expires_at, $3),
                        updated_at = now()
                    where owner_id = $1 and id = $2
                    """,
                    plan.owner_id,
                    keyframe.visual_frame_id,
                    expires_at,
                )

    async def _create_episode(
        self,
        connection: asyncpg.Connection,
        *,
        context: EpisodePromotionContext,
        plan: EpisodePromotionPlan,
        decided_at: datetime,
    ) -> UUID:
        event = context.event
        result = context.result
        metadata = {
            "source_experience_event_id": str(event.id),
            "event_kind": event.event_kind,
            "modalities": list(event.modalities),
            "scene": event.scene,
            "activity": event.activity,
            "started_at": event.started_at.isoformat(),
            "ended_at": event.ended_at.isoformat() if event.ended_at else None,
            "consolidation_provider": event.consolidation_provider,
            "consolidation_model": event.consolidation_model,
            "promotion_reason": plan.reason,
        }
        episode_id = await connection.fetchval(
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
                metadata,
                created_at,
                updated_at
            )
            values (
                $1, $2, 'episode', $3, $4, $4, $5, $6, $6, $7,
                $8, $9, 'experience_consolidation', $10, $11::jsonb, $12, $12
            )
            on conflict (source_experience_event_id) where memory_type = 'episode'
            do nothing
            returning id
            """,
            plan.owner_id,
            plan.session_id,
            result.event_title,
            result.event_summary,
            event.id,
            event.started_at,
            event.ended_at,
            result.episode_importance,
            result.confidence,
            plan.is_explicit,
            _json_dumps(metadata),
            decided_at,
        )
        if episode_id is None:
            episode_id = await connection.fetchval(
                """
                select id from memory_items
                where owner_id = $1
                  and source_experience_event_id = $2
                  and memory_type = 'episode'
                """,
                plan.owner_id,
                event.id,
            )
        if episode_id is None:
            raise RuntimeError("episode insert did not produce a memory item")
        return cast(UUID, episode_id)

    async def _attach_episode_evidence(
        self,
        connection: asyncpg.Connection,
        *,
        context: EpisodePromotionContext,
        plan: EpisodePromotionPlan,
        episode_id: UUID,
    ) -> None:
        event = context.event
        await self._insert_evidence(
            connection,
            owner_id=plan.owner_id,
            memory_item_id=episode_id,
            evidence_type="experience_event",
            evidence_ref_id=str(event.id),
            evidence_time=event.started_at,
            excerpt=context.result.event_summary,
            confidence=context.result.confidence,
            metadata={"role": "source_event"},
        )
        attached_message_ids: set[str] = set()
        for evidence in context.evidence:
            if evidence.evidence_type == "message":
                attached_message_ids.add(str(evidence.evidence_ref_id))
            await self._insert_evidence(
                connection,
                owner_id=plan.owner_id,
                memory_item_id=episode_id,
                evidence_type=evidence.evidence_type,
                evidence_ref_id=str(evidence.evidence_ref_id),
                evidence_time=evidence.occurred_at,
                excerpt=evidence_excerpt(
                    evidence,
                    context.source_payloads.get(
                        (evidence.evidence_type, evidence.evidence_ref_id)
                    ),
                ),
                confidence=context.result.confidence,
                metadata={
                    "source_sequence": evidence.source_sequence,
                    "attached_to_event": True,
                },
            )
        for message in context.context_messages:
            message_id = str(message["id"])
            if message_id in attached_message_ids:
                continue
            payload = message.get("payload")
            content = payload.get("content", "") if isinstance(payload, Mapping) else ""
            await self._insert_evidence(
                connection,
                owner_id=plan.owner_id,
                memory_item_id=episode_id,
                evidence_type="message",
                evidence_ref_id=message_id,
                evidence_time=message["created_at"],
                excerpt=str(content).strip()[:1_000],
                confidence=context.result.confidence,
                metadata={"overlapping_context": True},
            )
        for reminder in context.reminder_deliveries:
            await self._insert_evidence(
                connection,
                owner_id=plan.owner_id,
                memory_item_id=episode_id,
                evidence_type="reminder",
                evidence_ref_id=str(reminder.reminder_event_id),
                evidence_time=reminder.occurred_at,
                excerpt=reminder.message,
                confidence=context.result.confidence,
                metadata={
                    "reminder_id": str(reminder.reminder_id),
                    "event_type": "delivered_to_session",
                    **dict(reminder.metadata),
                },
            )
        for keyframe in plan.final_keyframes:
            await self._insert_evidence(
                connection,
                owner_id=plan.owner_id,
                memory_item_id=episode_id,
                evidence_type="frame",
                evidence_ref_id=str(keyframe.visual_frame_id),
                evidence_time=keyframe.captured_at,
                excerpt=keyframe.reason,
                confidence=context.result.confidence,
                metadata={"rank": keyframe.rank, "selected_by": keyframe.selected_by},
            )

    async def _insert_evidence(
        self,
        connection: asyncpg.Connection,
        *,
        owner_id: str,
        memory_item_id: UUID,
        evidence_type: str,
        evidence_ref_id: str,
        evidence_time: datetime,
        excerpt: str,
        confidence: float,
        metadata: Mapping[str, Any],
    ) -> None:
        await connection.execute(
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
            on conflict (memory_item_id, evidence_type, evidence_ref_id) do nothing
            """,
            owner_id,
            memory_item_id,
            evidence_type,
            evidence_ref_id,
            evidence_time,
            excerpt.strip()[:1_000],
            confidence,
            _json_dumps(metadata),
        )

    async def _link_adjacent_episode(
        self,
        connection: asyncpg.Connection,
        *,
        context: EpisodePromotionContext,
        episode_id: UUID,
    ) -> None:
        event = context.event
        previous = await connection.fetchrow(
            """
            select *
            from memory_items
            where owner_id = $1
              and memory_type = 'episode'
              and id <> $2
              and occurred_at <= $3
              and occurred_at >= $3 - interval '10 minutes'
            order by occurred_at desc, id desc
            limit 1
            """,
            event.owner_id,
            episode_id,
            event.started_at,
        )
        if previous is None:
            return
        metadata = _json_object(previous["metadata"])
        same_activity = bool(
            event.activity
            and metadata.get("activity")
            and _normalized_text(event.activity)
            == _normalized_text(str(metadata["activity"]))
        )
        same_scene = bool(
            event.scene
            and metadata.get("scene")
            and _normalized_text(event.scene) == _normalized_text(str(metadata["scene"]))
        )
        if not (same_activity or same_scene):
            return
        relationship = "part_of" if same_activity else "temporally_adjacent_to"
        await connection.execute(
            """
            insert into memory_links (
                owner_id,
                source_memory_id,
                target_memory_id,
                relationship,
                confidence,
                valid_from,
                metadata
            )
            values ($1, $2, $3, $4, $5, $6, $7::jsonb)
            on conflict (owner_id, source_memory_id, target_memory_id, relationship)
                where valid_to is null
            do nothing
            """,
            event.owner_id,
            episode_id,
            previous["id"],
            relationship,
            min(context.result.confidence, float(previous["confidence"])),
            event.started_at,
            _json_dumps({"scene_continuity": same_scene, "activity_continuity": same_activity}),
        )

    async def _existing_outcome(
        self,
        connection: asyncpg.Connection,
        event_row: Mapping[str, Any],
    ) -> EpisodePromotionOutcome:
        rows = await connection.fetch(
            """
            select visual_frame_id
            from experience_event_keyframes
            where event_id = $1
            order by rank asc
            """,
            event_row["id"],
        )
        return EpisodePromotionOutcome(
            event_id=event_row["id"],
            decision=event_row["promotion_decision"],
            reason=event_row["promotion_reason"],
            episode_memory_id=event_row["promoted_episode_memory_id"],
            selected_keyframe_ids=tuple(row["visual_frame_id"] for row in rows),
        )

    async def _store_embedding(
        self,
        connection: asyncpg.Connection,
        *,
        owner_id: str,
        subject: str,
        write: PreparedEmbeddingWrite,
    ) -> None:
        normalized = normalize_embedding(write.embedding)
        table = "experience_event_embeddings" if subject == "event" else "memory_item_embeddings"
        column = "experience_event_id" if subject == "event" else "memory_item_id"
        owner_table = "experience_events" if subject == "event" else "memory_items"
        status_column = "'active'" if subject == "event" else "status"
        status = await connection.fetchval(
            f"select {status_column} "
            f"from {owner_table} where owner_id = $1 and id = $2",
            owner_id,
            write.subject_id,
        )
        if status is None:
            raise ValueError("embedding subject is outside the owner scope")
        if subject == "memory" and status == "forgotten":
            return
        await connection.execute(
            f"""
            insert into {table} (
                {column}, embedding_kind, provider, model, dimensions,
                embedded_content, content_hash, embedding, embedded_at
            )
            values ($1, $2, $3, $4, $5, $6, $7, $8::vector, now())
            on conflict ({column}, embedding_kind, provider, model)
            do update set
                dimensions = excluded.dimensions,
                embedded_content = excluded.embedded_content,
                content_hash = excluded.content_hash,
                embedding = excluded.embedding,
                embedded_at = now(),
                updated_at = now()
            where {table}.content_hash is distinct from excluded.content_hash
            """,
            write.subject_id,
            write.embedding_kind,
            write.provider,
            write.model,
            EMBEDDING_DIMENSIONS,
            write.embedded_content,
            content_hash(write.embedded_content),
            _vector_literal(normalized),
        )


def _json_dumps(value: Mapping[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def _json_object(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, str):
        decoded = json.loads(value)
        return dict(decoded) if isinstance(decoded, Mapping) else {}
    return {}


def _normalized_text(value: str) -> str:
    return " ".join(value.casefold().split())


def _vector_literal(values: Sequence[float]) -> str:
    return "[" + ",".join(format(value, ".17g") for value in values) + "]"
