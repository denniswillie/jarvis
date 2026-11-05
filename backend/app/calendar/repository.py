from __future__ import annotations

from datetime import datetime

import asyncpg

from app.calendar.models import GoogleOAuthToken


def _token_from_record(record: asyncpg.Record) -> GoogleOAuthToken:
    scope = record["scope"] or ""
    return GoogleOAuthToken(
        owner_id=record["owner_id"],
        access_token=record["access_token"],
        refresh_token=record["refresh_token"],
        token_type=record["token_type"],
        scopes=[value for value in scope.split(" ") if value],
        expires_at=record["expires_at"],
        created_at=record["created_at"],
        updated_at=record["updated_at"],
    )


class GoogleOAuthTokenRepository:
    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def get_token(self, *, owner_id: str) -> GoogleOAuthToken | None:
        async with self._pool.acquire() as conn:
            record = await conn.fetchrow(
                """
                select *
                from google_oauth_tokens
                where owner_id = $1
                """,
                owner_id,
            )
        return _token_from_record(record) if record else None

    async def upsert_token(
        self,
        *,
        owner_id: str,
        access_token: str,
        refresh_token: str | None,
        token_type: str | None,
        scopes: list[str],
        expires_at: datetime | None,
    ) -> GoogleOAuthToken:
        async with self._pool.acquire() as conn:
            record = await conn.fetchrow(
                """
                insert into google_oauth_tokens (
                    owner_id,
                    access_token,
                    refresh_token,
                    token_type,
                    scope,
                    expires_at
                )
                values ($1, $2, $3, $4, $5, $6)
                on conflict (owner_id) do update
                set
                    access_token = excluded.access_token,
                    refresh_token = coalesce(excluded.refresh_token, google_oauth_tokens.refresh_token),
                    token_type = excluded.token_type,
                    scope = excluded.scope,
                    expires_at = excluded.expires_at,
                    updated_at = now()
                returning *
                """,
                owner_id,
                access_token,
                refresh_token,
                token_type,
                " ".join(sorted(set(scopes))),
                expires_at,
            )
        return _token_from_record(record)
