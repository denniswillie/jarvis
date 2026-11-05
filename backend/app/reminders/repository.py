from __future__ import annotations

import json
from datetime import datetime
from typing import Any
from uuid import UUID

import asyncpg

from app.db import json_object
from app.attention.notifications import reminder_text
from app.reminders.models import (
    JarvisReminder, JarvisReminderEvent, ReminderEventType, ReminderStatus, MAX_REMINDER_DELAY_SECONDS,
)


def _metadata(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        return json_object(json.loads(value))
    return json_object(value)


def _reminder_from_record(record: asyncpg.Record) -> JarvisReminder:
    return JarvisReminder(
        id=record["id"],
        session_id=record["session_id"],
        title=record["title"],
        details=record["details"],
        remind_at=record["remind_at"],
        condition=record["condition"],
        status=record["status"],
        created_at=record["created_at"],
        updated_at=record["updated_at"],
        completed_at=record["completed_at"],
        triggered_at=record["triggered_at"],
        trigger_claimed_at=record["trigger_claimed_at"],
        trigger_claim_token=record["trigger_claim_token"],
        trigger_attempt_count=record["trigger_attempt_count"],
        last_trigger_error=record["last_trigger_error"],
        metadata=_metadata(record["metadata"]),
    )


def _event_from_record(record: asyncpg.Record) -> JarvisReminderEvent:
    return JarvisReminderEvent(
        id=record["id"],
        reminder_id=record["reminder_id"],
        session_id=record["session_id"],
        event_type=record["event_type"],
        message=record["message"],
        provider_response_id=record["provider_response_id"],
        created_at=record["created_at"],
        metadata=_metadata(record["metadata"]),
    )


class ReminderRepository:
    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def create_reminder(
        self,
        *,
        owner_id: str,
        session_id: UUID | None,
        title: str,
        details: str,
        remind_at: datetime | None,
        condition: str | None,
        metadata: dict[str, Any] | None = None,
        delay_seconds: int | None = None,
    ) -> JarvisReminder:
        if delay_seconds is not None:
            if (type(delay_seconds) is not int or not 1 <= delay_seconds <= MAX_REMINDER_DELAY_SECONDS
                    or remind_at is not None or condition is not None):
                raise ValueError("invalid_relative_reminder_schedule")
            async with self._pool.acquire() as conn:
                # One database-clock sample AFTER acquisition. The materialized
                # row is reused for every timestamp; no application-clock math,
                # separate clock query, or retry on an uncertain insert.
                record = await conn.fetchrow(
                    """
                    with scheduling as materialized (
                        select clock_timestamp() as scheduled_at
                    )
                    insert into jarvis_reminders (
                        owner_id, session_id, title, details, remind_at, condition,
                        metadata, created_at, updated_at
                    )
                    select $1, $2, $3, $4,
                        scheduled_at + make_interval(secs => $5::double precision),
                        null,
                        $6::jsonb || jsonb_build_object(
                            'scheduled_at', scheduled_at,
                            'delay_seconds', $5::integer,
                            'scheduling_clock', 'postgres_clock_timestamp'
                        ),
                        scheduled_at, scheduled_at
                    from scheduling
                    returning *
                    """,
                    owner_id, session_id, title, details, delay_seconds, json.dumps(metadata or {}),
                )
            return _reminder_from_record(record)
        async with self._pool.acquire() as conn:
            record = await conn.fetchrow(
                """
                insert into jarvis_reminders (
                    owner_id,
                    session_id,
                    title,
                    details,
                    remind_at,
                    condition,
                    metadata
                )
                values ($1, $2, $3, $4, $5, $6, $7::jsonb)
                returning *
                """,
                owner_id,
                session_id,
                title,
                details,
                remind_at,
                condition,
                json.dumps(metadata or {}),
            )
        return _reminder_from_record(record)

    async def list_reminders(
        self,
        *,
        owner_id: str,
        status: ReminderStatus | None,
        limit: int,
    ) -> list[JarvisReminder]:
        async with self._pool.acquire() as conn:
            records = await conn.fetch(
                """
                select *
                from jarvis_reminders
                where owner_id = $1
                  and ($2::text is null or status = $2)
                order by updated_at desc
                limit $3
                """,
                owner_id,
                status,
                limit,
            )
        return [_reminder_from_record(record) for record in records]

    async def list_active_reminders(self, *, owner_id: str) -> list[JarvisReminder]:
        async with self._pool.acquire() as conn:
            records = await conn.fetch(
                """
                select *
                from jarvis_reminders
                where owner_id = $1 and status = 'active'
                  and (not_before is null or not_before <= now())
                  and jarvis_responsibility_notice_time(owner_id,id,now()) <= now()
                order by coalesce(remind_at, updated_at) asc, updated_at desc
                """,
                owner_id,
            )
        return [_reminder_from_record(record) for record in records]

    async def list_events(self, *, owner_id: str, limit: int) -> list[JarvisReminderEvent]:
        async with self._pool.acquire() as conn:
            records = await conn.fetch(
                """
                select *
                from jarvis_reminder_events
                where owner_id = $1
                order by created_at desc
                limit $2
                """,
                owner_id,
                limit,
            )
        return [_event_from_record(record) for record in records]

    async def claim_due_reminders(
        self,
        *,
        owner_id: str,
        claim_timeout_seconds: int,
        max_attempts: int,
        limit: int = 10,
    ) -> list[JarvisReminder]:
        async with self._pool.acquire() as conn:
            records = await conn.fetch(
                """
                with due as (
                    select id
                    from jarvis_reminders
                    where owner_id = $1
                      and status = 'active'
                      and remind_at is not null
                      and remind_at <= now()
                      and (not_before is null or not_before <= now())
                      and jarvis_responsibility_notice_time(owner_id,id,now()) <= now()
                      and triggered_at is null
                      and not exists (
                          select 1 from jarvis_prospective_delivery_events e
                          where e.owner_id=jarvis_reminders.owner_id and e.reminder_id=jarvis_reminders.id
                            and e.event_type in ('generated','dispatched_to_session')
                            and (e.metadata @> '{"attention_owned":true}'::jsonb
                                 or e.metadata @> '{"dispatch_started":true}'::jsonb)
                      )
                      and trigger_attempt_count < $2
                      and (
                          trigger_claimed_at is null
                          or trigger_claimed_at <= now() - make_interval(secs => $3::double precision)
                      )
                    order by remind_at asc
                    for update skip locked
                    limit $4
                )
                update jarvis_reminders reminders
                set
                    trigger_claimed_at = now(),
                    trigger_claim_token = gen_random_uuid(),
                    trigger_attempt_count = reminders.trigger_attempt_count + 1,
                    last_trigger_error = null,
                    updated_at = now()
                from due
                where reminders.id = due.id
                returning reminders.*
                """,
                owner_id,
                max_attempts,
                claim_timeout_seconds,
                limit,
            )
        return [_reminder_from_record(record) for record in records]

    async def finalize_trigger(
        self,
        *,
        owner_id: str,
        reminder_id: UUID,
        claim_token: UUID,
        session_id: UUID | None,
        message: str,
        metadata: dict[str, Any] | None = None,
        attention=None,
    ) -> JarvisReminderEvent:
        async with self._pool.acquire() as conn, conn.transaction():
            reminder = await conn.fetchrow(
                """
                update jarvis_reminders
                set
                    status = 'completed',
                    completed_at = coalesce(completed_at, now()),
                    triggered_at = coalesce(triggered_at, now()),
                    trigger_claimed_at = null,
                    trigger_claim_token = null,
                    last_trigger_error = null,
                    updated_at = now()
                where owner_id = $1
                  and id = $2
                  and trigger_claim_token = $3
                  and triggered_at is null
                  and status = 'active'
                returning *
                """,
                owner_id,
                reminder_id,
                claim_token,
            )
            if reminder is None:
                raise RuntimeError("Reminder trigger could not be finalized.")
            if attention is not None:
                message = reminder_text(reminder)
            event = await conn.fetchrow(
                """
                insert into jarvis_reminder_events (
                    owner_id,
                    reminder_id,
                    session_id,
                    event_type,
                    message,
                    metadata
                )
                values ($1, $2, $3, 'generated', $4, $5::jsonb)
                returning *
                """,
                owner_id,
                reminder_id,
                session_id,
                message,
                json.dumps(metadata or {}),
            )
            if attention is not None:
                await attention.enqueue_reminder(conn, owner_id=owner_id, reminder=reminder, event=event)
        return _event_from_record(event)

    async def mark_event_delivery(
        self,
        *,
        owner_id: str,
        event_id: UUID,
        event_type: ReminderEventType,
        session_id: UUID | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> JarvisReminderEvent | None:
        async with self._pool.acquire() as conn:
            record = await conn.fetchrow(
                """
                update jarvis_reminder_events
                set
                    event_type = $3,
                    session_id = coalesce($4, session_id),
                    metadata = metadata || $5::jsonb
                where owner_id = $1 and id = $2
                returning *
                """,
                owner_id,
                event_id,
                event_type,
                session_id,
                json.dumps(metadata or {}),
            )
        return _event_from_record(record) if record else None

    async def set_event_provider_response_id(
        self,
        *,
        owner_id: str,
        event_id: UUID,
        provider_response_id: str,
    ) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                update jarvis_reminder_events
                set provider_response_id = $3
                where owner_id = $1 and id = $2
                """,
                owner_id,
                event_id,
                provider_response_id,
            )

    async def record_trigger_failure(
        self,
        *,
        owner_id: str,
        reminder: JarvisReminder,
        error_code: str,
        terminal: bool,
    ) -> bool:
        claim_token = reminder.trigger_claim_token
        if claim_token is None:
            return False
        sanitized_error = error_code[:500]
        async with self._pool.acquire() as conn, conn.transaction():
            updated = await conn.fetchrow(
                """
                update jarvis_reminders
                set
                    trigger_claimed_at = null,
                    trigger_claim_token = null,
                    status = case when $4 then 'failed' else status end,
                    last_trigger_error = $3,
                    updated_at = now()
                where owner_id = $1
                  and id = $2
                  and trigger_claim_token = $5
                  and triggered_at is null
                  and status = 'active'
                returning owner_id, id, session_id
                """,
                owner_id,
                reminder.id,
                sanitized_error,
                terminal,
                claim_token,
            )
            if updated is None:
                return False
            if terminal:
                await conn.execute(
                    """
                    insert into jarvis_reminder_events (
                        owner_id,
                        reminder_id,
                        session_id,
                        event_type,
                        message,
                        metadata
                    )
                    values (
                        $1,
                        $2,
                        $3,
                        'failed',
                        'Jarvis could not deliver this reminder.',
                        $4::jsonb
                    )
                    """,
                    updated["owner_id"],
                    updated["id"],
                    updated["session_id"],
                    json.dumps(
                        {
                            "error_code": sanitized_error,
                            "trigger_attempt": reminder.trigger_attempt_count,
                        }
                    ),
                )
        return True

    async def complete_reminder(self, *, owner_id: str, reminder_id: UUID) -> JarvisReminder | None:
        async with self._pool.acquire() as conn:
            record = await conn.fetchrow(
                """
                update jarvis_reminders
                set
                    status = 'completed',
                    completed_at = coalesce(completed_at, now()),
                    trigger_claimed_at = null,
                    trigger_claim_token = null,
                    metadata = metadata || jsonb_build_object('attention_cancelled_at', clock_timestamp()),
                    updated_at = now()
                where owner_id = $1 and id = $2 and status != 'deleted'
                returning *
                """,
                owner_id,
                reminder_id,
            )
        return _reminder_from_record(record) if record else None

    async def delete_reminder(self, *, owner_id: str, reminder_id: UUID) -> JarvisReminder | None:
        async with self._pool.acquire() as conn:
            record = await conn.fetchrow(
                """
                update jarvis_reminders
                set
                    status = 'deleted',
                    trigger_claimed_at = null,
                    trigger_claim_token = null,
                    updated_at = now()
                where owner_id = $1 and id = $2 and status != 'deleted'
                returning *
                """,
                owner_id,
                reminder_id,
            )
        return _reminder_from_record(record) if record else None
