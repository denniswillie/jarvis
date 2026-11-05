from __future__ import annotations

from datetime import datetime
from uuid import UUID

import asyncpg
from app.perception.applicability import effect_connection

from app.knowledge.visual_delivery_sources import bind_visual_delivery_source, lock_visual_delivery_source
from app.visual_memory.models import (
    ActiveVisualReminder,
    BufferedVisualFrame,
    RecentVisualInterrupt,
    UserPreference,
    ValidatedVisualFrame,
    VisualFrameRecord,
    VisualInterruptRecord,
    VisualFrameStorageObject,
)


class VisualFrameDuplicateError(Exception):
    pass


class VisualFrameSessionMissingError(Exception):
    pass


def _record_from_row(row: asyncpg.Record) -> VisualFrameRecord:
    return VisualFrameRecord(
        id=row["id"],
        owner_id=row["owner_id"],
        session_id=row["session_id"],
        audio_session_id=row["audio_session_id"],
        visual_session_id=row["visual_session_id"],
        frame_sequence=row["frame_sequence"],
        source=row["source"],
        captured_at=row["captured_at"],
        server_received_at=row["server_received_at"],
        content_type=row["content_type"],
        width=row["width"],
        height=row["height"],
        storage_bucket=row["storage_bucket"],
        storage_path=row["storage_path"],
        raw_image_status=row["raw_image_status"],
        raw_image_expires_at=row["raw_image_expires_at"],
        analyzer_status=row["analyzer_status"],
        frame_description=row["frame_description"],
        memory_tags=list(row["memory_tags"] or []),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _interrupt_from_row(row: asyncpg.Record) -> VisualInterruptRecord:
    return VisualInterruptRecord(
        id=row["id"],
        owner_id=row["owner_id"],
        session_id=row["session_id"],
        visual_frame_id=row["visual_frame_id"],
        message=row["message"],
        reason=row["reason"],
        confidence=row["confidence"],
        dedupe_bucket=row["dedupe_bucket"],
        spoken_response_id=row["spoken_response_id"],
        created_at=row["created_at"],
    )


def buffered_frame_from_record(record: VisualFrameRecord) -> BufferedVisualFrame:
    return BufferedVisualFrame(
        frame_id=record.id,
        owner_id=record.owner_id,
        session_id=record.session_id,
        audio_session_id=record.audio_session_id,
        visual_session_id=record.visual_session_id,
        frame_sequence=record.frame_sequence,
        source=record.source,
        captured_at=record.captured_at,
        server_received_at=record.server_received_at,
        content_type=record.content_type,
        width=record.width,
        height=record.height,
        storage_bucket=record.storage_bucket,
        storage_path=record.storage_path,
        raw_image_expires_at=record.raw_image_expires_at,
    )


class VisualFrameRepository:
    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def create_pending_frame(
        self,
        *,
        owner_id: str,
        session_id: UUID,
        audio_session_id: str | None,
        visual_session_id: UUID,
        frame: ValidatedVisualFrame,
        server_received_at: datetime,
        storage_bucket: str,
        storage_path: str,
        raw_image_expires_at: datetime,
    ) -> VisualFrameRecord:
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                session_exists = await conn.fetchval(
                    """
                    select exists(
                        select 1
                        from jarvis_sessions
                        where owner_id = $1 and id = $2 and status = 'active'
                    )
                    """,
                    owner_id,
                    session_id,
                )
                if not session_exists:
                    raise VisualFrameSessionMissingError()

                row = await conn.fetchrow(
                    """
                    insert into visual_frames (
                        owner_id,
                        session_id,
                        audio_session_id,
                        visual_session_id,
                        frame_sequence,
                        source,
                        captured_at,
                        server_received_at,
                        content_type,
                        width,
                        height,
                        storage_bucket,
                        storage_path,
                        raw_image_status,
                        raw_image_expires_at,
                        analyzer_status
                    )
                    values (
                        $1, $2, $3, $4, $5, $6, $7, $8,
                        $9, $10, $11, $12, $13, 'failed', $14, 'skipped'
                    )
                    on conflict (session_id, visual_session_id, frame_sequence) do nothing
                    returning *
                    """,
                    owner_id,
                    session_id,
                    audio_session_id,
                    visual_session_id,
                    frame.frame_sequence,
                    frame.source,
                    frame.captured_at,
                    server_received_at,
                    frame.content_type,
                    frame.width,
                    frame.height,
                    storage_bucket,
                    storage_path,
                    raw_image_expires_at,
                )
                if row is None:
                    raise VisualFrameDuplicateError()

                await conn.execute(
                    """
                    update jarvis_sessions
                    set updated_at = now(), last_activity_at = now()
                    where owner_id = $1 and id = $2
                    """,
                    owner_id,
                    session_id,
                )

        return _record_from_row(row)

    async def mark_frame_available(
        self,
        *,
        owner_id: str,
        frame_id: UUID,
        stored_object: VisualFrameStorageObject,
    ) -> VisualFrameRecord | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                update visual_frames
                set
                    storage_bucket = $3,
                    storage_path = $4,
                    raw_image_status = 'available',
                    updated_at = now()
                where owner_id = $1 and id = $2
                returning *
                """,
                owner_id,
                frame_id,
                stored_object.bucket,
                stored_object.path,
            )
        return _record_from_row(row) if row else None

    async def mark_frame_failed(self, *, owner_id: str, frame_id: UUID) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                update visual_frames
                set raw_image_status = 'failed', updated_at = now()
                where owner_id = $1 and id = $2
                """,
                owner_id,
                frame_id,
            )
    async def mark_analysis_pending(self, *, owner_id: str, frame_id: UUID) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                update visual_frames
                set analyzer_status = 'pending', updated_at = now()
                where owner_id = $1 and id = $2
                """,
                owner_id,
                frame_id,
            )

    async def mark_analysis_skipped(self, *, owner_id: str, frame_id: UUID, reason: str) -> None:
        del reason
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                update visual_frames
                set
                    analyzer_status = 'skipped',
                    updated_at = now()
                where owner_id = $1 and id = $2
                """,
                owner_id,
                frame_id,
            )

    async def mark_analysis_failed(self, *, owner_id: str, frame_id: UUID) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                update visual_frames
                set analyzer_status = 'failed', updated_at = now()
                where owner_id = $1 and id = $2
                """,
                owner_id,
                frame_id,
            )

    async def apply_analysis_result(
        self,
        *,
        owner_id: str,
        frame_id: UUID,
        frame_description: str | None,
        memory_tags: list[str],
    ) -> VisualFrameRecord | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                update visual_frames
                set
                    analyzer_status = 'completed',
                    frame_description = $3,
                    memory_tags = $4::text[],
                    updated_at = now()
                where owner_id = $1 and id = $2 and raw_image_status = 'available'
                returning *
                """,
                owner_id,
                frame_id,
                frame_description,
                memory_tags,
            )
        return _record_from_row(row) if row else None

    async def list_recent_visual_interrupts(
        self,
        *,
        owner_id: str,
        session_id: UUID,
        limit: int,
    ) -> list[RecentVisualInterrupt]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                select id, visual_frame_id, message, reason, confidence, dedupe_bucket, created_at
                from visual_interrupts
                where owner_id = $1 and session_id = $2
                order by created_at desc
                limit $3
                """,
                owner_id,
                session_id,
                limit,
            )
        return [
            RecentVisualInterrupt(
                id=row["id"],
                visual_frame_id=row["visual_frame_id"],
                message=row["message"],
                reason=row["reason"],
                confidence=row["confidence"],
                dedupe_bucket=row["dedupe_bucket"],
                created_at=row["created_at"],
            )
            for row in rows
        ]

    async def recent_visual_interrupt_exists(
        self,
        *,
        owner_id: str,
        session_id: UUID,
        dedupe_bucket: str,
        window_seconds: int,
    ) -> bool:
        async with self._pool.acquire() as conn:
            exists = await conn.fetchval(
                """
                select exists(
                    select 1
                    from visual_interrupts
                    where owner_id = $1
                      and session_id = $2
                      and dedupe_bucket = $3
                      and created_at >= now() - make_interval(secs => $4)
                )
                """,
                owner_id,
                session_id,
                dedupe_bucket,
                window_seconds,
            )
        return bool(exists)

    async def create_visual_interrupt(
        self,
        *,
        owner_id: str,
        session_id: UUID,
        visual_frame_id: UUID,
        message: str,
        reason: str,
        confidence: float,
        dedupe_bucket: str,
        source_payload_id: UUID | None = None,
        attention=None,
        max_age_seconds: float | None = None,
        effect_guard=None,
    ) -> VisualInterruptRecord:
        async with effect_connection(self._pool, effect_guard) as conn, conn.transaction():
            await lock_visual_delivery_source(conn, owner_id=owner_id, session_id=session_id,
                payload_id=source_payload_id, frame_ids=(visual_frame_id,))
            if attention is not None:
                if source_payload_id is None:
                    from app.attention.models import AttentionError
                    raise AttentionError("visual_attention_source_required", "A source-backed observation is required.")
                await conn.execute("select pg_advisory_xact_lock(hashtextextended(current_schema() || ':visual-attention:' || $1,0))",
                                   owner_id + ":" + str(source_payload_id))
                previous = await conn.fetchrow("""select v.* from visual_interrupts v
                    join jarvis_attention_intents i on i.owner_id=v.owner_id and i.source_id=v.id
                    where i.owner_id=$1 and i.source_namespace='visual_alert' and i.coalesce_key=$2""",
                    owner_id, "observation:" + str(source_payload_id))
                if previous is not None:
                    return _interrupt_from_row(previous)
            row = await conn.fetchrow(
                """
                insert into visual_interrupts (
                    owner_id,
                    session_id,
                    visual_frame_id,
                    message,
                    reason,
                    confidence,
                    dedupe_bucket
                )
                values ($1, $2, $3, $4, $5, $6, $7)
                returning *
                """,
                owner_id,
                session_id,
                visual_frame_id,
                message,
                reason,
                confidence,
                dedupe_bucket,
            )
            await bind_visual_delivery_source(conn, owner_id=owner_id, session_id=session_id,
                kind="visual", delivery_id=row["id"], payload_id=source_payload_id)
            if attention is not None:
                await attention.enqueue_visual(conn, owner_id=owner_id, kind="visual", event=row,
                    observation_payload_id=source_payload_id, max_age_seconds=max_age_seconds,
                    **({"perception_focus": effect_guard.focus_fence}
                       if effect_guard is not None and effect_guard.focus_fence is not None else {}))
        return _interrupt_from_row(row)

    async def mark_visual_interrupt_spoken(
        self,
        *,
        owner_id: str,
        interrupt_id: UUID,
        spoken_response_id: str,
    ) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                update visual_interrupts
                set spoken_response_id = $3
                where owner_id = $1 and id = $2
                """,
                owner_id,
                interrupt_id,
                spoken_response_id,
            )

    async def list_user_preferences(self, *, owner_id: str, limit: int) -> list[UserPreference]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                select key, value, source, updated_at
                from jarvis_user_preferences
                where owner_id = $1
                order by updated_at desc
                limit $2
                """,
                owner_id,
                limit,
            )
        return [
            UserPreference(
                key=row["key"],
                value=row["value"],
                source=row["source"],
                updated_at=row["updated_at"],
            )
            for row in rows
        ]

    async def list_active_visual_reminders(
        self,
        *,
        owner_id: str,
        limit: int,
    ) -> list[ActiveVisualReminder]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                select id, title, details, remind_at, condition, updated_at
                from jarvis_reminders
                where owner_id = $1
                  and status = 'active'
                  and condition is not null
                order by updated_at desc
                limit $2
                """,
                owner_id,
                limit,
            )
        return [
            ActiveVisualReminder(
                id=row["id"],
                title=row["title"],
                details=row["details"],
                remind_at=row["remind_at"],
                condition=row["condition"],
                updated_at=row["updated_at"],
            )
            for row in rows
        ]

    async def claim_expired_raw_frames(
        self,
        *,
        limit: int,
        stale_claim_seconds: int = 300,
    ) -> list[VisualFrameRecord]:
        if limit < 1:
            raise ValueError("limit must be positive")
        if stale_claim_seconds < 1:
            raise ValueError("stale_claim_seconds must be positive")
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                with candidates as (
                    select frames.id
                    from visual_frames frames
                    where (
                        frames.raw_image_status = 'available'
                        and frames.raw_image_expires_at <= now()
                        and not exists (
                            select 1
                            from visual_frame_retention_reasons reasons
                            where reasons.visual_frame_id = frames.id
                              and reasons.owner_id = frames.owner_id
                              and (
                                  reasons.expires_at is null
                                  or reasons.expires_at > now()
                              )
                        )
                    ) or (
                        frames.raw_image_status = 'expiring'
                        and frames.updated_at
                            <= now() - make_interval(secs => $2)
                    )
                    order by frames.raw_image_expires_at asc, frames.id asc
                    for update of frames skip locked
                    limit $1
                )
                update visual_frames frames
                set raw_image_status = 'expiring', updated_at = now()
                from candidates
                where frames.id = candidates.id
                returning frames.*
                """,
                limit,
                stale_claim_seconds,
            )
        return [_record_from_row(row) for row in rows]

    async def delete_expired_retention_reasons(self, *, limit: int) -> int:
        if limit < 1:
            raise ValueError("limit must be positive")
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                with expired as (
                    select id
                    from visual_frame_retention_reasons
                    where expires_at is not null and expires_at <= now()
                    order by expires_at asc, id asc
                    limit $1
                )
                delete from visual_frame_retention_reasons reasons
                using expired
                where reasons.id = expired.id
                returning reasons.id
                """,
                limit,
            )
        return len(rows)

    async def mark_frame_expired(self, *, owner_id: str, frame_id: UUID) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                update visual_frames
                set raw_image_status = 'expired', updated_at = now()
                where owner_id = $1 and id = $2 and raw_image_status = 'expiring'
                """,
                owner_id,
                frame_id,
            )

    async def release_frame_expiry_claim(self, *, owner_id: str, frame_id: UUID) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                update visual_frames
                set raw_image_status = 'available', updated_at = now()
                where owner_id = $1 and id = $2 and raw_image_status = 'expiring'
                """,
                owner_id,
                frame_id,
            )
