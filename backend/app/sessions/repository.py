from __future__ import annotations

import hashlib
import json
from contextlib import nullcontext
from datetime import datetime
from typing import Any
from uuid import UUID

import asyncpg

from app.db import json_object
from app.sessions.models import JarvisMessage, JarvisSession


def _decode_json_object(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        decoded = json.loads(value)
        return json_object(decoded)
    return json_object(value)


def _session_from_record(record: asyncpg.Record) -> JarvisSession:
    return JarvisSession(
        id=record["id"],
        title=record["title"],
        status=record["status"],
        created_at=record["created_at"],
        updated_at=record["updated_at"],
        closed_at=record["closed_at"],
        last_activity_at=record["last_activity_at"],
        behavior_instructions=record["behavior_instructions"],
        metadata=_decode_json_object(record["metadata"]),
    )


def _message_from_record(record: asyncpg.Record) -> JarvisMessage:
    return JarvisMessage(
        id=record["id"],
        session_id=record["session_id"],
        role=record["role"],
        modality=record["modality"],
        content=record["content"],
        provider_item_id=record["provider_item_id"],
        provider_response_id=record["provider_response_id"],
        source_payload_id=record.get("source_payload_id"),
        created_at=record["created_at"],
        metadata=_decode_json_object(record["metadata"]),
    )


class SessionRepository:
    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def get_latest_active_session(self, owner_id: str) -> JarvisSession | None:
        async with self._pool.acquire() as conn:
            record = await conn.fetchrow(
                """
                select *
                from jarvis_sessions
                where owner_id = $1 and status = 'active'
                order by updated_at desc
                limit 1
                """,
                owner_id,
            )
        return _session_from_record(record) if record else None

    async def create_session(self, owner_id: str, title: str) -> JarvisSession:
        async with self._pool.acquire() as conn:
            record = await conn.fetchrow(
                """
                insert into jarvis_sessions (owner_id, title)
                values ($1, $2)
                returning *
                """,
                owner_id,
                title,
            )
        return _session_from_record(record)

    async def get_session(self, owner_id: str, session_id: UUID) -> JarvisSession | None:
        async with self._pool.acquire() as conn:
            record = await conn.fetchrow(
                """
                select *
                from jarvis_sessions
                where owner_id = $1 and id = $2
                """,
                owner_id,
                session_id,
            )
        return _session_from_record(record) if record else None

    async def get_recent_messages(
        self,
        owner_id: str,
        session_id: UUID,
        *,
        limit: int = 50,
    ) -> list[JarvisMessage]:
        async with self._pool.acquire() as conn:
            records = await conn.fetch(
                """
                select messages.*
                from jarvis_messages messages
                join jarvis_sessions sessions on sessions.id = messages.session_id
                where sessions.owner_id = $1 and messages.owner_id = $1 and sessions.id = $2
                order by messages.created_at desc
                limit $3
                """,
                owner_id,
                session_id,
                limit,
            )
        return [_message_from_record(record) for record in reversed(records)]

    async def create_message(
        self,
        *,
        owner_id: str,
        session_id: UUID,
        role: str,
        modality: str,
        content: str,
        provider_item_id: str | None = None,
        provider_response_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        source_payload_id: UUID | None = None,
        connection: asyncpg.Connection | None = None,
    ) -> JarvisMessage:
        if connection is not None and not connection.is_in_transaction():
            raise RuntimeError("Composed message writes require an active transaction")
        # Ordinary audio remains compatible with a schema where the optional
        # source-bound output migration has not yet been applied.
        source_column = ", source_payload_id" if source_payload_id is not None else ""
        source_value = ", $9" if source_payload_id is not None else ""
        async with self._pool.acquire() if connection is None else nullcontext(connection) as conn:
            record = await conn.fetchrow(
                f"""
                insert into jarvis_messages (
                    owner_id,
                    session_id,
                    role,
                    modality,
                    content,
                    provider_item_id,
                    provider_response_id,
                    metadata{source_column}
                )
                values ($1, $2, $3, $4, $5, $6, $7, $8::jsonb{source_value})
                returning *
                """,
                owner_id,
                session_id,
                role,
                modality,
                content,
                provider_item_id,
                provider_response_id,
                json.dumps(metadata or {}),
                *((source_payload_id,) if source_payload_id is not None else ()),
            )
        await self.touch_session(owner_id=owner_id, session_id=session_id, connection=connection)
        return _message_from_record(record)

    async def touch_session(self, *, owner_id: str, session_id: UUID,
                            connection: asyncpg.Connection | None = None) -> None:
        if connection is not None and not connection.is_in_transaction():
            raise RuntimeError("Composed session writes require an active transaction")
        async with self._pool.acquire() if connection is None else nullcontext(connection) as conn:
            await conn.execute(
                """
                update jarvis_sessions
                set updated_at = now(), last_activity_at = now()
                where owner_id = $1 and id = $2
                """,
                owner_id,
                session_id,
            )

    async def create_or_reuse_gemini_message(
        self,
        *,
        owner_id: str,
        session_id: UUID,
        audio_session_id: str,
        logical_item_id: str,
        role: str,
        content: str,
        occurred_at: datetime,
        provider_item_id: str | None = None,
        provider_response_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> JarvisMessage:
        """Atomically reuse one finalized Gemini item, never matching its text.

        This additive entry point leaves create_message and its GPT callers alone.
        Socket generations deliberately do not participate in durable identity.
        """
        if role not in {"user", "assistant"} or not content.strip():
            raise ValueError("A finalized conversational message is required")
        if not owner_id.strip() or not audio_session_id or not logical_item_id:
            raise ValueError("Gemini message identity is incomplete")
        if occurred_at.tzinfo is None or occurred_at.utcoffset() is None:
            raise ValueError("Message occurrence time must be timezone-aware")
        key = hashlib.sha256(json.dumps(
            [owner_id, str(session_id), audio_session_id, logical_item_id, role],
            separators=(",", ":"),
        ).encode()).hexdigest()
        values = {
            **(metadata or {}), "audio_provider": "gemini",
            "audio_session_id": audio_session_id, "logical_item_id": logical_item_id,
            "gemini_message_key": key,
        }
        async with self._pool.acquire() as conn, conn.transaction():
            await conn.execute(
                "select pg_advisory_xact_lock(hashtextextended($1, 0))",
                "gemini_message:" + key,
            )
            record = await conn.fetchrow(
                """
                select messages.* from jarvis_messages messages
                join jarvis_sessions sessions on sessions.id = messages.session_id
                where messages.owner_id = $1 and sessions.owner_id = $1
                  and messages.session_id = $2
                  and messages.metadata ->> 'gemini_message_key' = $3
                """,
                owner_id, session_id, key,
            )
            if record is not None:
                message = _message_from_record(record)
                if message.role != role or message.modality != "audio" or message.content != content:
                    raise ValueError("Gemini message identity has conflicting content")
                return message
            record = await conn.fetchrow(
                """
                insert into jarvis_messages (
                    owner_id, session_id, role, modality, content,
                    provider_item_id, provider_response_id, metadata, created_at
                )
                select owner_id, id, $3, 'audio', $4, $5, $6, $7::jsonb, $8
                from jarvis_sessions where owner_id = $1 and id = $2
                returning *
                """,
                owner_id, session_id, role, content, provider_item_id,
                provider_response_id, json.dumps(values), occurred_at,
            )
            if record is None:
                raise ValueError("The owned Gemini conversation is unavailable")
            await conn.execute(
                """update jarvis_sessions set updated_at = now(), last_activity_at = now()
                where owner_id = $1 and id = $2""",
                owner_id, session_id,
            )
        return _message_from_record(record)

    async def gemini_message_has_experience_evidence(
        self, *, owner_id: str, session_id: UUID, message_id: UUID,
    ) -> bool:
        """Read across event boundaries when an attachment acknowledgement was lost."""
        async with self._pool.acquire() as conn:
            return bool(await conn.fetchval(
                """
                select exists (
                    select 1 from experience_event_evidence evidence
                    join experience_events events on events.id = evidence.event_id
                    join jarvis_messages messages on messages.id = evidence.evidence_ref_id
                    where events.owner_id = $1 and messages.owner_id = $1
                      and events.session_id = $2 and messages.session_id = $2
                      and evidence.evidence_type = 'message' and messages.id = $3
                      and messages.metadata ->> 'audio_provider' = 'gemini'
                )
                """,
                owner_id, session_id, message_id,
            ))

    async def close_session(self, owner_id: str, session_id: UUID) -> JarvisSession | None:
        async with self._pool.acquire() as conn:
            record = await conn.fetchrow(
                """
                update jarvis_sessions
                set
                    status = 'closed',
                    closed_at = coalesce(closed_at, now()),
                    updated_at = now(),
                    last_activity_at = now()
                where owner_id = $1 and id = $2
                returning *
                """,
                owner_id,
                session_id,
            )
        return _session_from_record(record) if record else None

    async def update_behavior_instructions(
        self,
        owner_id: str,
        session_id: UUID,
        behavior_instructions: str,
    ) -> JarvisSession | None:
        async with self._pool.acquire() as conn:
            record = await conn.fetchrow(
                """
                update jarvis_sessions
                set
                    behavior_instructions = $3,
                    updated_at = now(),
                    last_activity_at = now()
                where owner_id = $1 and id = $2
                returning *
                """,
                owner_id,
                session_id,
                behavior_instructions,
            )
        return _session_from_record(record) if record else None
