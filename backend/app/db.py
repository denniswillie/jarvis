from __future__ import annotations

from typing import Any

import asyncpg
from fastapi import Request

from app.errors import JarvisHTTPError
from app.settings import JarvisSettings, is_configured_secret


async def create_pool(settings: JarvisSettings) -> asyncpg.Pool:
    if not is_configured_secret(settings.SUPABASE_DB_URL):
        raise JarvisHTTPError(
            503,
            "jarvis_database_not_configured",
            "Jarvis database URL is not configured.",
        )

    return await asyncpg.create_pool(
        dsn=settings.SUPABASE_DB_URL,
        min_size=settings.SUPABASE_DB_POOL_MIN_SIZE,
        max_size=settings.SUPABASE_DB_POOL_MAX_SIZE,
        command_timeout=settings.SUPABASE_DB_COMMAND_TIMEOUT_SECONDS,
        timeout=settings.SUPABASE_DB_CONNECT_TIMEOUT_SECONDS,
    )


def get_pool(request: Request) -> asyncpg.Pool:
    runtime = getattr(request.app.state, "runtime_services", None)
    pool = getattr(runtime, "db_pool", None)
    if pool is None:
        raise JarvisHTTPError(
            503,
            "jarvis_database_not_configured",
            "Jarvis database is not configured for this backend process.",
        )
    return pool


def json_object(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    raise TypeError(f"Expected JSON object, got {type(value).__name__}")
