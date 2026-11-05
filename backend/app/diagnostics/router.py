from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Annotated, Any
from uuid import UUID

import httpx
from fastapi import APIRouter, Depends, Request

from app.auth.static_bearer import JarvisPrincipal, require_static_bearer
from app.errors import JarvisHTTPError
from app.experience.repository import ExperienceRepository
from app.memory.jobs.runner import MemoryJobRunner
from app.perception.frame_window import RecentVisualFrameWindow
from app.perception.services import WorkingMemoryService
from app.realtime.session_manager import RealtimeSessionManager
from app.sessions.repository import SessionRepository
from app.settings import JarvisSettings, get_settings, is_configured_secret

router = APIRouter(prefix="/diagnostics", tags=["diagnostics"])


def _runtime_value(request: Request, name: str) -> object | None:
    runtime = getattr(request.app.state, "runtime_services", None)
    return getattr(runtime, name, None)


STORAGE_PROBE_TIMEOUT_SECONDS = 5.0


def _task_running(task: asyncio.Task[Any] | None) -> bool:
    return task is not None and not task.done()


async def _database_reachable(pool: Any) -> bool:
    if pool is None:
        return False
    try:
        async with pool.acquire() as connection:
            return bool(await connection.fetchval("select 1"))
    except Exception:
        return False


async def _probe_storage_buckets(settings: JarvisSettings) -> dict[str, bool]:
    buckets = {
        "visual_frames": settings.SUPABASE_STORAGE_BUCKET,
        "files": settings.JARVIS_FILES_STORAGE_BUCKET,
    }
    if not (
        is_configured_secret(settings.SUPABASE_URL)
        and is_configured_secret(settings.SUPABASE_SERVICE_ROLE_KEY)
    ):
        return {name: False for name in buckets}

    base_url = settings.SUPABASE_URL.rstrip("/")
    headers = {
        "Authorization": f"Bearer {settings.SUPABASE_SERVICE_ROLE_KEY}",
        "apikey": str(settings.SUPABASE_SERVICE_ROLE_KEY),
    }
    async with httpx.AsyncClient(timeout=STORAGE_PROBE_TIMEOUT_SECONDS) as client:
        async def bucket_exists(bucket: str) -> bool:
            try:
                response = await client.get(
                    f"{base_url}/storage/v1/bucket/{bucket}",
                    headers=headers,
                )
                return response.status_code == 200
            except httpx.HTTPError:
                return False

        results = await asyncio.gather(
            *(bucket_exists(bucket) for bucket in buckets.values())
        )
    return dict(zip(buckets, results, strict=True))


@router.get("/config")
async def get_config_diagnostics(
    request: Request,
    _: Annotated[JarvisPrincipal, Depends(require_static_bearer)],
    settings: Annotated[JarvisSettings, Depends(get_settings)],
) -> dict[str, Any]:
    pool = _runtime_value(request, "db_pool")
    database_reachable, storage_buckets = await asyncio.gather(
        _database_reachable(pool),
        _probe_storage_buckets(settings),
    )

    configured = {
        "static_bearer": is_configured_secret(settings.JARVIS_STATIC_BEARER_TOKEN),
        "database": is_configured_secret(settings.SUPABASE_DB_URL),
        "supabase_storage": (
            is_configured_secret(settings.SUPABASE_URL)
            and is_configured_secret(settings.SUPABASE_SERVICE_ROLE_KEY)
        ),
        "openai_realtime": (
            is_configured_secret(settings.OPENAI_API_KEY)
            and is_configured_secret(settings.OPENAI_REALTIME_MODEL)
        ),
        "openai_reminders": (
            is_configured_secret(settings.OPENAI_API_KEY)
            and is_configured_secret(settings.OPENAI_REMINDER_MODEL)
        ),
        "openai_consolidation": (
            is_configured_secret(settings.OPENAI_API_KEY)
            and is_configured_secret(settings.OPENAI_CONSOLIDATION_MODEL)
        ),
        "openai_web_search": (
            is_configured_secret(settings.OPENAI_API_KEY)
            and is_configured_secret(settings.OPENAI_WEB_SEARCH_MODEL)
        ),
        "gemini_visual": (
            is_configured_secret(settings.GEMINI_API_KEY)
            and is_configured_secret(settings.GEMINI_VISUAL_MODEL)
        ),
        "gemini_memory_embeddings": (
            is_configured_secret(settings.GEMINI_API_KEY)
            and is_configured_secret(settings.GEMINI_MEMORY_EMBEDDING_MODEL)
        ),
        "google_calendar": (
            is_configured_secret(settings.GOOGLE_OAUTH_CLIENT_ID)
            and is_configured_secret(settings.GOOGLE_OAUTH_CLIENT_SECRET)
            and is_configured_secret(settings.GOOGLE_OAUTH_STATE_SECRET)
        ),
    }
    required_checks = {
        "static_bearer": configured["static_bearer"],
        "database": configured["database"] and database_reachable,
        "visual_frames_bucket": storage_buckets["visual_frames"],
        "files_bucket": storage_buckets["files"],
        "openai_realtime": configured["openai_realtime"],
        "gemini_memory_embeddings": configured["gemini_memory_embeddings"],
        "openai_reminders": configured["openai_reminders"],
        "openai_consolidation": configured["openai_consolidation"],
        "openai_web_search": configured["openai_web_search"],
        "gemini_visual": configured["gemini_visual"],
    }
    failed_checks = [name for name, ready in required_checks.items() if not ready]
    maintenance_task = _runtime_value(request, "memory_maintenance_scheduler_task")

    return {
        "status": "ready" if not failed_checks else "degraded",
        "checked_at": datetime.now(UTC),
        "environment": settings.APP_ENV,
        "public_base_url": settings.PUBLIC_BASE_URL,
        "configured": configured,
        "database": {
            "initialized": pool is not None,
            "reachable": database_reachable,
            "pool_min_size": settings.SUPABASE_DB_POOL_MIN_SIZE,
            "pool_max_size": settings.SUPABASE_DB_POOL_MAX_SIZE,
        },
        "storage": {
            "initialized": (
                _runtime_value(request, "visual_frame_storage") is not None
                and _runtime_value(request, "jarvis_file_storage") is not None
            ),
            "buckets": {
                "visual_frames": {
                    "name": settings.SUPABASE_STORAGE_BUCKET,
                    "exists": storage_buckets["visual_frames"],
                },
                "files": {
                    "name": settings.JARVIS_FILES_STORAGE_BUCKET,
                    "exists": storage_buckets["files"],
                },
            },
        },
        "background_jobs": {
            "memory_job_runner": _task_running(
                _runtime_value(request, "memory_job_runner_task")
            ),
            "memory_maintenance_scheduler": _task_running(
                maintenance_task
            ),
            "reminder_scheduler": _task_running(
                _runtime_value(request, "reminder_scheduler_task")
            ),
            "visual_frame_retention": (
                _task_running(
                    _runtime_value(request, "visual_frame_retention_task")
                )
            ),
            "experience_event_sweeper": _task_running(
                _runtime_value(request, "experience_event_sweeper_task")
            ),
        },
        "failed_checks": failed_checks,
    }


async def collect_memory_diagnostics(
    *,
    pool: Any,
    sensory_frame_window: RecentVisualFrameWindow | None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Collect operational memory health without returning memory content or bytes."""

    checked_at = now or datetime.now(UTC)
    async with pool.acquire() as connection:
        summary = await connection.fetchrow(
            """
            select
                (select count(*) from perception_observations) as observation_count,
                (
                    select count(*) from perception_observations
                    where compacted_at is null
                ) as detailed_observation_count,
                (select count(*) from jarvis_session_working_memory) as working_state_count,
                (
                    select count(*) from jarvis_session_working_memory
                    where expires_at <= $1
                ) as expired_working_state_count,
                (
                    select max(updated_at) from jarvis_session_working_memory
                ) as latest_working_state_at,
                (
                    select count(*) from experience_events where status = 'open'
                ) as open_event_count,
                (
                    select count(*) from experience_events where status = 'closed'
                ) as awaiting_consolidation_count,
                (
                    select max(consolidated_at) from experience_events
                    where status = 'consolidated'
                ) as latest_consolidation_at,
                (
                    select count(*)
                    from memory_items items
                    where items.status = 'active'
                      and not exists (
                          select 1 from memory_item_embeddings embeddings
                          where embeddings.memory_item_id = items.id
                      )
                ) as active_memory_items_missing_embeddings,
                (
                    select count(*)
                    from experience_events events
                    where events.status = 'consolidated'
                      and not exists (
                          select 1 from experience_event_embeddings embeddings
                          where embeddings.experience_event_id = events.id
                      )
                ) as consolidated_events_missing_embeddings,
                (
                    select count(*) from memory_jobs where status = 'dead'
                ) as dead_job_count,
                (
                    select min(created_at) from memory_jobs
                    where status in ('queued', 'claimed', 'failed')
                ) as oldest_active_job_at,
                (select count(*) from memory_retrieval_runs) as retrieval_count,
                (
                    select avg(latency_ms)::double precision
                    from memory_retrieval_runs
                ) as retrieval_mean_latency_ms,
                (
                    select percentile_cont(0.95) within group (order by latency_ms)
                    from memory_retrieval_runs
                )::double precision as retrieval_p95_latency_ms
            """,
            checked_at,
        )
        event_status_rows = await connection.fetch(
            """
            select status, count(*)::bigint as count
            from experience_events
            group by status
            order by status
            """
        )
        job_status_rows = await connection.fetch(
            """
            select status, count(*)::bigint as count
            from memory_jobs
            group by status
            order by status
            """
        )
        retention_rows = await connection.fetch(
            """
            select scope, reason_type, count(*)::bigint as count
            from (
                select 'visual_frame'::text as scope, reason_type
                from visual_frame_retention_reasons
                union all
                select 'experience_event'::text as scope, reason_type
                from experience_event_retention_reasons
            ) reasons
            group by scope, reason_type
            order by scope, reason_type
            """
        )

    oldest_active_job_at = summary["oldest_active_job_at"]
    dead_job_count = int(summary["dead_job_count"] or 0)
    missing_embeddings = int(summary["active_memory_items_missing_embeddings"] or 0)
    missing_event_embeddings = int(
        summary["consolidated_events_missing_embeddings"] or 0
    )
    issues: list[str] = []
    if dead_job_count:
        issues.append("dead_memory_jobs")
    if missing_embeddings:
        issues.append("active_memory_embeddings_missing")
    if missing_event_embeddings:
        issues.append("experience_event_embeddings_missing")

    return {
        "status": "ready" if not issues else "degraded",
        "checked_at": checked_at,
        "sensory": (
            sensory_frame_window.diagnostics()
            if sensory_frame_window is not None
            else {
                "session_count": 0,
                "frame_count": 0,
                "max_frames_per_session": None,
                "ttl_seconds": None,
                "oldest_frame_age_seconds": None,
                "newest_frame_age_seconds": None,
            }
        ),
        "perception": {
            "observation_count": int(summary["observation_count"] or 0),
            "detailed_observation_count": int(
                summary["detailed_observation_count"] or 0
            ),
        },
        "working_memory": {
            "state_count": int(summary["working_state_count"] or 0),
            "expired_state_count": int(summary["expired_working_state_count"] or 0),
            "latest_updated_at": summary["latest_working_state_at"],
        },
        "experience_events": {
            "counts_by_status": {
                row["status"]: int(row["count"]) for row in event_status_rows
            },
            "open_count": int(summary["open_event_count"] or 0),
            "awaiting_consolidation_count": int(
                summary["awaiting_consolidation_count"] or 0
            ),
            "latest_successful_consolidation_at": summary[
                "latest_consolidation_at"
            ],
        },
        "jobs": {
            "counts_by_status": {
                row["status"]: int(row["count"]) for row in job_status_rows
            },
            "dead_count": dead_job_count,
            "oldest_active_age_seconds": (
                max(0.0, (checked_at - oldest_active_job_at).total_seconds())
                if oldest_active_job_at is not None
                else None
            ),
        },
        "embeddings": {
            "active_memory_items_missing": missing_embeddings,
            "consolidated_events_missing": missing_event_embeddings,
        },
        "retention_reasons": {
            f"{row['scope']}:{row['reason_type']}": int(row["count"])
            for row in retention_rows
        },
        "retrieval": {
            "run_count": int(summary["retrieval_count"] or 0),
            "mean_latency_ms": summary["retrieval_mean_latency_ms"],
            "p95_latency_ms": summary["retrieval_p95_latency_ms"],
        },
        "issues": issues,
    }


@router.get("/memory")
async def get_memory_diagnostics(
    request: Request,
    _: Annotated[JarvisPrincipal, Depends(require_static_bearer)],
) -> dict[str, Any]:
    pool = _runtime_value(request, "db_pool")
    if pool is None:
        raise JarvisHTTPError(
            503,
            "jarvis_database_not_configured",
            "Jarvis database is not configured for memory diagnostics.",
        )
    sensory_frame_window = _runtime_value(request, "sensory_frame_window")
    return await collect_memory_diagnostics(
        pool=pool,
        sensory_frame_window=(
            sensory_frame_window
            if isinstance(sensory_frame_window, RecentVisualFrameWindow)
            else None
        ),
    )


@router.get("/memory-jobs")
async def get_memory_job_diagnostics(
    request: Request,
    _: Annotated[JarvisPrincipal, Depends(require_static_bearer)],
    settings: Annotated[JarvisSettings, Depends(get_settings)],
) -> dict[str, Any]:
    runner = _runtime_value(request, "memory_job_runner")
    if isinstance(runner, MemoryJobRunner):
        return await runner.diagnostics(enabled=settings.MEMORY_JOB_RUNNER_ENABLED)
    return {
        "enabled": settings.MEMORY_JOB_RUNNER_ENABLED,
        "running": False,
        "worker_id": None,
        "registered_job_types": (),
        "active_job_types": (),
        "active_task_count": 0,
        "counts_by_status": {},
        "oldest_queued_age_seconds": None,
        "last_completion_at": None,
        "reconciliation": None,
    }


@router.get("/session/{session_id}")
async def get_session_diagnostics(
    session_id: UUID,
    request: Request,
    principal: Annotated[JarvisPrincipal, Depends(require_static_bearer)],
    settings: Annotated[JarvisSettings, Depends(get_settings)],
) -> dict[str, Any]:
    pool = _runtime_value(request, "db_pool")
    if pool is None:
        raise JarvisHTTPError(
            503,
            "jarvis_database_not_configured",
            "Jarvis database is not configured for diagnostics.",
        )

    repository = SessionRepository(pool)
    session = await repository.get_session(principal.owner_id, session_id)
    if session is None:
        raise JarvisHTTPError(
            404,
            "jarvis_session_not_found",
            "Jarvis session was not found.",
        )
    recent_messages = await repository.get_recent_messages(
        principal.owner_id,
        session_id,
        limit=20,
    )
    manager = _runtime_value(request, "realtime_session_manager")
    realtime = (
        await manager.get_session_diagnostics(session_id)
        if isinstance(manager, RealtimeSessionManager)
        else {"realtime_connected": False, "audio_session_id": None}
    )
    working_memory_service = _runtime_value(request, "working_memory_service")
    working_memory = (
        await working_memory_service.get_session_working_memory(
            owner_id=principal.owner_id,
            session_id=session_id,
        )
        if isinstance(working_memory_service, WorkingMemoryService)
        else None
    )
    working_state = dict(working_memory.state) if working_memory is not None else {}
    experience_diagnostics = await ExperienceRepository(pool).get_session_diagnostics(
        owner_id=principal.owner_id,
        session_id=session_id,
    )
    experience_sweeper = _runtime_value(request, "experience_event_sweeper")
    scene = working_state.get("scene")
    activity = working_state.get("activity")
    entities = working_state.get("entities")
    return {
        "session_id": session.id,
        "status": session.status,
        "created_at": session.created_at,
        "updated_at": session.updated_at,
        "last_activity_at": session.last_activity_at,
        "closed_at": session.closed_at,
        "recent_message_count": len(recent_messages),
        "latest_message_at": (
            recent_messages[-1].created_at if recent_messages else None
        ),
        "realtime": realtime,
        "working_memory": {
            "available": working_memory is not None,
            "revision": working_memory.revision if working_memory is not None else None,
            "updated_at": working_memory.updated_at if working_memory is not None else None,
            "age_seconds": (
                max(0.0, (datetime.now(UTC) - working_memory.updated_at).total_seconds())
                if working_memory is not None
                else None
            ),
            "scene": scene.get("label") if isinstance(scene, dict) else None,
            "activity": activity.get("label") if isinstance(activity, dict) else None,
            "entity_count": len(entities) if isinstance(entities, dict) else 0,
            "last_observation_id": (
                working_memory.last_observation_id if working_memory is not None else None
            ),
            "latest_experience_event_id": working_state.get(
                "latest_experience_event_id"
            ),
        },
        "experience_events": {
            **experience_diagnostics,
            "keyframe_policy": {
                "final_max_count": settings.EVENT_KEYFRAME_MAX_COUNT,
                "consolidation_candidate_max_count": (
                    settings.EVENT_CONSOLIDATION_KEYFRAME_MAX_COUNT
                ),
                "open_event_candidate_max_count": (
                    settings.EVENT_KEYFRAME_CANDIDATE_MAX_COUNT
                ),
                "candidate_retention_seconds": (
                    settings.EVENT_CANDIDATE_RETENTION_SECONDS
                ),
                "ordinary_retention_seconds": (
                    settings.EVENT_ORDINARY_KEYFRAME_RETENTION_SECONDS
                ),
            },
            "sweeper": (
                experience_sweeper.diagnostics()
                if experience_sweeper is not None
                else None
            ),
        },
    }
