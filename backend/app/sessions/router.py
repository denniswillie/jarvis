from __future__ import annotations

from typing import Annotated
from uuid import UUID

import asyncpg
from fastapi import APIRouter, Depends, Query, Request

from app.auth.static_bearer import JarvisPrincipal, require_static_bearer
from app.db import get_pool
from app.sessions.models import (
    ActiveSessionResponse,
    SessionDetailResponse,
    SessionStartResponse,
    StartSessionRequest,
    UpdateBehaviorInstructionsRequest,
)
from app.sessions.service import SessionService
from app.settings import JarvisSettings, get_settings


router = APIRouter(prefix="/sessions", tags=["sessions"])


def get_session_service(
    request: Request,
    pool: Annotated[asyncpg.Pool, Depends(get_pool)],
    settings: Annotated[JarvisSettings, Depends(get_settings)],
) -> SessionService:
    runtime = getattr(request.app.state, "runtime_services", None)
    return SessionService(
        pool,
        settings,
        sensory_frame_window=getattr(runtime, "sensory_frame_window", None),
        experience_event_service=getattr(runtime, "experience_event_service", None),
        reflection_trigger_service=getattr(
            runtime,
            "reflection_trigger_service",
            None,
        ),
    )


@router.post("/start", response_model=SessionStartResponse)
async def start_session(
    request: StartSessionRequest,
    principal: Annotated[JarvisPrincipal, Depends(require_static_bearer)],
    service: Annotated[SessionService, Depends(get_session_service)],
) -> SessionStartResponse:
    return await service.start_session(principal, request)


@router.get("/active", response_model=ActiveSessionResponse)
async def get_active_session(
    principal: Annotated[JarvisPrincipal, Depends(require_static_bearer)],
    service: Annotated[SessionService, Depends(get_session_service)],
) -> ActiveSessionResponse:
    return await service.get_active_session(principal)


@router.get("/{session_id}", response_model=SessionDetailResponse)
async def get_session(
    session_id: UUID,
    principal: Annotated[JarvisPrincipal, Depends(require_static_bearer)],
    service: Annotated[SessionService, Depends(get_session_service)],
    include_recent_messages: Annotated[bool, Query()] = False,
) -> SessionDetailResponse:
    return await service.get_session(
        principal,
        session_id,
        include_recent_messages=include_recent_messages,
    )


@router.post("/{session_id}/close", response_model=SessionDetailResponse)
async def close_session(
    session_id: UUID,
    principal: Annotated[JarvisPrincipal, Depends(require_static_bearer)],
    service: Annotated[SessionService, Depends(get_session_service)],
) -> SessionDetailResponse:
    return await service.close_session(principal, session_id)


@router.patch("/{session_id}/behavior-instructions", response_model=SessionDetailResponse)
async def update_behavior_instructions(
    session_id: UUID,
    request: UpdateBehaviorInstructionsRequest,
    principal: Annotated[JarvisPrincipal, Depends(require_static_bearer)],
    service: Annotated[SessionService, Depends(get_session_service)],
) -> SessionDetailResponse:
    return await service.update_behavior_instructions(
        principal,
        session_id,
        request.behavior_instructions,
    )
