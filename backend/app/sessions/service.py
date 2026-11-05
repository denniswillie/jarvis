from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from urllib.parse import quote, urlsplit, urlunsplit
from uuid import UUID

import asyncpg

from app.auth.static_bearer import JarvisPrincipal
from app.errors import JarvisHTTPError
from app.experience.service import ExperienceEventService
from app.sessions.models import (
    ActiveSessionResponse,
    RealtimeConfig,
    SessionDetailResponse,
    SessionStartResponse,
    StartSessionRequest,
)
from app.sessions.repository import SessionRepository
from app.settings import JarvisSettings
from app.perception.frame_window import RecentVisualFrameWindow


if TYPE_CHECKING:
    from app.memory.reflection.service import ReflectionTriggerService


DEFAULT_SESSION_TITLE = "Jarvis session"
LOGGER = logging.getLogger(__name__)


def build_realtime_config(settings: JarvisSettings, session_id: UUID) -> RealtimeConfig:
    public_base_url = settings.PUBLIC_BASE_URL.rstrip("/")
    parsed = urlsplit(public_base_url)
    if parsed.scheme == "https":
        websocket_scheme = "wss"
    elif parsed.scheme == "http":
        websocket_scheme = "ws"
    else:
        raise JarvisHTTPError(
            500,
            "jarvis_public_base_url_invalid",
            "PUBLIC_BASE_URL must start with http:// or https://.",
        )

    websocket_base = urlunsplit(
        (websocket_scheme, parsed.netloc, parsed.path.rstrip("/"), "", "")
    )
    encoded_session_id = quote(str(session_id), safe="")
    return RealtimeConfig(
        audio_websocket_url=(
            f"{websocket_base}/realtime/audio/ws?session_id={encoded_session_id}"
        ),
        visual_websocket_url=(
            f"{websocket_base}/realtime/visual/ws?session_id={encoded_session_id}"
        ),
    )


class SessionService:
    def __init__(
        self,
        pool: asyncpg.Pool,
        settings: JarvisSettings,
        sensory_frame_window: RecentVisualFrameWindow | None = None,
        experience_event_service: ExperienceEventService | None = None,
        reflection_trigger_service: ReflectionTriggerService | None = None,
    ) -> None:
        self._repository = SessionRepository(pool)
        self._settings = settings
        self._sensory_frame_window = sensory_frame_window
        self._experience_event_service = experience_event_service
        self._reflection_trigger_service = reflection_trigger_service

    async def start_session(
        self,
        principal: JarvisPrincipal,
        request: StartSessionRequest,
    ) -> SessionStartResponse:
        session = None
        if request.resume_active:
            session = await self._repository.get_latest_active_session(principal.owner_id)

        if session is None:
            title = (request.title or "").strip() or DEFAULT_SESSION_TITLE
            session = await self._repository.create_session(principal.owner_id, title)

        return SessionStartResponse(
            session=session,
            realtime=build_realtime_config(self._settings, session.id),
        )

    async def get_active_session(
        self,
        principal: JarvisPrincipal,
    ) -> ActiveSessionResponse:
        session = await self._repository.get_latest_active_session(principal.owner_id)
        return ActiveSessionResponse(
            session=session,
            realtime=build_realtime_config(self._settings, session.id) if session else None,
        )

    async def get_session(
        self,
        principal: JarvisPrincipal,
        session_id: UUID,
        *,
        include_recent_messages: bool,
    ) -> SessionDetailResponse:
        session = await self._repository.get_session(principal.owner_id, session_id)
        if session is None:
            raise JarvisHTTPError(
                404,
                "jarvis_session_not_found",
                "Jarvis session was not found.",
            )

        recent_messages = None
        if include_recent_messages:
            recent_messages = await self._repository.get_recent_messages(
                principal.owner_id,
                session_id,
            )

        return SessionDetailResponse(
            session=session,
            realtime=build_realtime_config(self._settings, session.id),
            recent_messages=recent_messages,
        )

    async def close_session(
        self,
        principal: JarvisPrincipal,
        session_id: UUID,
    ) -> SessionDetailResponse:
        session = await self._repository.close_session(principal.owner_id, session_id)
        if session is None:
            raise JarvisHTTPError(
                404,
                "jarvis_session_not_found",
                "Jarvis session was not found.",
            )

        if self._sensory_frame_window is not None:
            self._sensory_frame_window.clear(principal.owner_id, session_id)
        if self._experience_event_service is not None:
            self._experience_event_service.schedule_session_close(
                owner_id=principal.owner_id,
                session_id=session_id,
                closed_at=session.closed_at or datetime.now(UTC),
            )
        if self._reflection_trigger_service is not None:
            try:
                await self._reflection_trigger_service.enqueue_if_eligible(
                    owner_id=principal.owner_id,
                    now=session.closed_at or datetime.now(UTC),
                )
            except Exception:
                LOGGER.exception(
                    "reflection scheduling failed after session close owner_id=%s "
                    "session_id=%s",
                    principal.owner_id,
                    session_id,
                )

        return SessionDetailResponse(
            session=session,
            realtime=build_realtime_config(self._settings, session.id),
        )

    async def update_behavior_instructions(
        self,
        principal: JarvisPrincipal,
        session_id: UUID,
        behavior_instructions: str,
    ) -> SessionDetailResponse:
        session = await self._repository.update_behavior_instructions(
            principal.owner_id,
            session_id,
            behavior_instructions,
        )
        if session is None:
            raise JarvisHTTPError(
                404,
                "jarvis_session_not_found",
                "Jarvis session was not found.",
            )

        return SessionDetailResponse(
            session=session,
            realtime=build_realtime_config(self._settings, session.id),
        )
