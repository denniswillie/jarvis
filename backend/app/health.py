from __future__ import annotations

from fastapi import APIRouter, Request

from app.errors import JarvisHTTPError
from app.settings import get_settings


router = APIRouter(tags=["health"])


@router.get("/health")
async def health_check() -> dict[str, str]:
    settings = get_settings()
    return {
        "status": "ok",
        "service": "jarvis-backend",
        "environment": settings.APP_ENV,
    }


@router.get("/ready")
async def readiness_check(request: Request) -> dict[str, str]:
    runtime = getattr(request.app.state, "runtime_services", None)
    pool = getattr(runtime, "db_pool", None)
    if pool is None:
        raise JarvisHTTPError(
            503,
            "jarvis_database_not_ready",
            "Jarvis database pool is not initialized.",
        )
    try:
        async with pool.acquire() as connection:
            await connection.fetchval("select 1")
    except Exception as error:
        raise JarvisHTTPError(
            503,
            "jarvis_database_not_ready",
            "Jarvis database readiness check failed.",
        ) from error
    return {"status": "ready"}
