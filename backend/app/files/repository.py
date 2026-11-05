from __future__ import annotations

import json
from typing import Any
from uuid import UUID

import asyncpg

from app.db import json_object
from app.files.models import JarvisFile, JarvisFileSource


def _metadata(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        return json_object(json.loads(value))
    return json_object(value)


def _file_from_record(record: asyncpg.Record) -> JarvisFile:
    return JarvisFile(
        id=record["id"],
        session_id=record["session_id"],
        filename=record["filename"],
        content_type=record["content_type"],
        storage_bucket=record["storage_bucket"],
        storage_path=record["storage_path"],
        size_bytes=record["size_bytes"],
        description=record["description"],
        source=record["source"],
        source_visual_frame_id=record["source_visual_frame_id"],
        created_at=record["created_at"],
        metadata=_metadata(record["metadata"]),
    )


class JarvisFileRepository:
    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def create_file(
        self,
        *,
        owner_id: str,
        session_id: UUID | None,
        filename: str,
        content_type: str,
        storage_bucket: str,
        storage_path: str,
        size_bytes: int | None,
        description: str,
        source: JarvisFileSource,
        source_visual_frame_id: UUID | None = None,
        metadata: dict[str, Any] | None = None,
        file_id: UUID | None = None,
    ) -> JarvisFile:
        async with self._pool.acquire() as conn:
            record = await conn.fetchrow(
                """
                insert into jarvis_files (
                    owner_id,
                    session_id,
                    filename,
                    content_type,
                    storage_bucket,
                    storage_path,
                    size_bytes,
                    description,
                    source,
                    source_visual_frame_id,
                    metadata,
                    id
                )
                values ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11::jsonb, coalesce($12::uuid, gen_random_uuid()))
                returning *
                """,
                owner_id,
                session_id,
                filename,
                content_type,
                storage_bucket,
                storage_path,
                size_bytes,
                description,
                source,
                source_visual_frame_id,
                json.dumps(metadata or {}),
                file_id,
            )
        return _file_from_record(record)

    async def list_files(self, *, owner_id: str, query: str | None, limit: int) -> list[JarvisFile]:
        pattern = f"%{query}%" if query else None
        async with self._pool.acquire() as conn:
            records = await conn.fetch(
                """
                select *
                from jarvis_files
                where owner_id = $1
                  and ($2::text is null or filename ilike $2 or description ilike $2)
                order by created_at desc
                limit $3
                """,
                owner_id,
                pattern,
                limit,
            )
        return [_file_from_record(record) for record in records]

    async def get_file(self, *, owner_id: str, file_id: UUID) -> JarvisFile | None:
        async with self._pool.acquire() as conn:
            record = await conn.fetchrow(
                """
                select *
                from jarvis_files
                where owner_id = $1 and id = $2
                """,
                owner_id,
                file_id,
            )
        return _file_from_record(record) if record else None

    async def delete_file(self, *, owner_id: str, file_id: UUID) -> JarvisFile | None:
        async with self._pool.acquire() as conn:
            record = await conn.fetchrow(
                """
                delete from jarvis_files
                where owner_id = $1 and id = $2
                returning *
                """,
                owner_id,
                file_id,
            )
        return _file_from_record(record) if record else None
