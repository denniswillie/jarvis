from __future__ import annotations

import json
from typing import Any
from uuid import UUID

import asyncpg

from app.db import json_object
from app.notes.models import JarvisNote


def _metadata(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        return json_object(json.loads(value))
    return json_object(value)


def _note_from_record(record: asyncpg.Record) -> JarvisNote:
    return JarvisNote(
        id=record["id"],
        session_id=record["session_id"],
        title=record["title"],
        body=record["body"],
        source=record["source"],
        created_at=record["created_at"],
        updated_at=record["updated_at"],
        metadata=_metadata(record["metadata"]),
    )


class NoteRepository:
    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def create_note(
        self,
        *,
        owner_id: str,
        session_id: UUID | None,
        title: str,
        body: str,
        metadata: dict[str, Any] | None = None,
        note_id: UUID | None = None,
    ) -> JarvisNote:
        async with self._pool.acquire() as conn:
            record = await conn.fetchrow(
                """
                insert into jarvis_notes (owner_id, session_id, title, body, metadata, id)
                values ($1, $2, $3, $4, $5::jsonb, coalesce($6::uuid, gen_random_uuid()))
                returning *
                """,
                owner_id,
                session_id,
                title,
                body,
                json.dumps(metadata or {}),
                note_id,
            )
        return _note_from_record(record)

    async def search_notes(self, *, owner_id: str, query: str | None, limit: int) -> list[JarvisNote]:
        pattern = f"%{query}%" if query else None
        async with self._pool.acquire() as conn:
            records = await conn.fetch(
                """
                select *
                from jarvis_notes
                where owner_id = $1
                  and ($2::text is null or title ilike $2 or body ilike $2)
                order by updated_at desc
                limit $3
                """,
                owner_id,
                pattern,
                limit,
            )
        return [_note_from_record(record) for record in records]

    async def update_note(
        self,
        *,
        owner_id: str,
        note_id: UUID,
        title: str | None,
        body: str | None,
    ) -> JarvisNote | None:
        async with self._pool.acquire() as conn:
            record = await conn.fetchrow(
                """
                update jarvis_notes
                set
                    title = coalesce($3, title),
                    body = coalesce($4, body),
                    updated_at = now()
                where owner_id = $1 and id = $2
                returning *
                """,
                owner_id,
                note_id,
                title,
                body,
            )
        return _note_from_record(record) if record else None

    async def delete_note(self, *, owner_id: str, note_id: UUID) -> bool:
        async with self._pool.acquire() as conn:
            result = await conn.execute(
                """
                delete from jarvis_notes
                where owner_id = $1 and id = $2
                """,
                owner_id,
                note_id,
            )
        return result.endswith(" 1")
