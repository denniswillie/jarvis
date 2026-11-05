from __future__ import annotations

from uuid import UUID

import asyncpg

from app.preferences.models import InferredPreferenceWriteResult, JarvisPreference


def _preference_from_record(record: asyncpg.Record) -> JarvisPreference:
    return JarvisPreference(
        id=record["id"],
        key=record["key"],
        value=record["value"],
        source=record["source"],
        confidence=float(record.get("confidence", 1.0)),
        evidence_episode_ids=tuple(record.get("evidence_episode_ids", ())),
        created_at=record["created_at"],
        updated_at=record["updated_at"],
    )


class PreferenceRepository:
    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    def attention_quiet(self):
        """Structured attention policy uses the existing capability ledger."""
        from app.attention.quiet import OwnerQuietRepository

        return OwnerQuietRepository(self._pool)

    async def set_preference(
        self,
        *,
        owner_id: str,
        key: str,
        value: str,
        connection: asyncpg.Connection | None = None,
    ) -> JarvisPreference:
        normalized_key = _required_text(key, "key")
        normalized_value = _required_text(value, "value")

        async def write(conn: asyncpg.Connection) -> JarvisPreference:
            record = await conn.fetchrow(
                """
                insert into jarvis_user_preferences (
                    owner_id, key, value, source, confidence, evidence_episode_ids
                )
                values ($1, $2, $3, 'assistant', 1.0, '{}'::uuid[])
                on conflict (owner_id, key)
                do update set
                    value = excluded.value,
                    source = 'assistant',
                    confidence = 1.0,
                    evidence_episode_ids = '{}'::uuid[],
                    updated_at = now()
                returning *
                """,
                owner_id,
                normalized_key,
                normalized_value,
            )
            return _preference_from_record(record)

        if connection is not None:
            return await write(connection)
        async with self._pool.acquire() as conn:
            return await write(conn)

    async def set_inferred_preference(
        self,
        *,
        owner_id: str,
        key: str,
        value: str,
        confidence: float,
        evidence_episode_ids: tuple[UUID, ...],
        connection: asyncpg.Connection,
    ) -> InferredPreferenceWriteResult:
        normalized_key = _required_text(key, "key")
        normalized_value = _required_text(value, "value")
        if confidence < 0 or confidence > 1:
            raise ValueError("confidence must be between zero and one")
        normalized_evidence = tuple(dict.fromkeys(evidence_episode_ids))
        if not normalized_evidence:
            raise ValueError("inferred preference requires episode evidence")
        record = await connection.fetchrow(
            """
            insert into jarvis_user_preferences (
                owner_id, key, value, source, confidence, evidence_episode_ids
            )
            values ($1, $2, $3, 'inferred_memory', $4, $5::uuid[])
            on conflict (owner_id, key)
            do update set
                value = excluded.value,
                source = 'inferred_memory',
                confidence = excluded.confidence,
                evidence_episode_ids = excluded.evidence_episode_ids,
                updated_at = now()
            where jarvis_user_preferences.source = 'inferred_memory'
            returning *
            """,
            owner_id,
            normalized_key,
            normalized_value,
            confidence,
            list(normalized_evidence),
        )
        if record is not None:
            return InferredPreferenceWriteResult(
                preference=_preference_from_record(record),
                applied=True,
            )
        existing = await connection.fetchrow(
            """
            select * from jarvis_user_preferences
            where owner_id = $1 and key = $2
            """,
            owner_id,
            normalized_key,
        )
        if existing is None:
            raise RuntimeError("preference conflict did not return an existing row")
        return InferredPreferenceWriteResult(
            preference=_preference_from_record(existing),
            applied=False,
        )

    async def search_preferences(self, *, owner_id: str, query: str | None, limit: int) -> list[JarvisPreference]:
        pattern = f"%{query}%" if query else None
        async with self._pool.acquire() as conn:
            records = await conn.fetch(
                """
                select *
                from jarvis_user_preferences
                where owner_id = $1
                  and ($2::text is null or key ilike $2 or value ilike $2)
                order by updated_at desc
                limit $3
                """,
                owner_id,
                pattern,
                limit,
            )
        return [_preference_from_record(record) for record in records]


def _required_text(value: str, label: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{label} cannot be blank")
    return normalized
