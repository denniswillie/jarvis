from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import replace
from uuid import UUID, uuid4

from fastapi import APIRouter, WebSocket, status

from app.auth.static_bearer import authenticate_static_bearer
from app.db import get_pool
from app.errors import JarvisHTTPError
from app.experience.repository import ToolExecutionRepository
from app.realtime.models import AudioClientKind, RealtimeSessionContext
from app.realtime.live_selection import select_live_voice_route, require_live_client_offer
from app.realtime.lean_selection import select_lean_voice, require_lean_offer
from app.realtime.provider_selection import AudioProvider, require_gemini_live_enabled, select_audio_provider
from app.realtime.session import RealtimeAudioSession
from app.realtime.session_manager import RealtimeSessionManager
from app.realtime.socket_sender import WebSocketEventSender
from app.reminders.repository import ReminderRepository
from app.runtime import JarvisRuntimeServices
from app.sessions.repository import SessionRepository
from app.settings import JarvisSettings, get_settings, is_configured_secret
from app.visual_memory.services import VisualMemoryServices
from app.tasks.context_capture import SessionContextCapture
from app.tasks.context_repository import ContextRepository
from app.tasks.service import TaskService


router = APIRouter(prefix="/realtime", tags=["realtime"])
LOGGER = logging.getLogger(__name__)
SUPPORTED_AUDIO_CLIENT_KINDS = {"rayban_audio", "phone_audio"}


@router.websocket("/audio/ws")
async def audio_realtime_websocket(websocket: WebSocket) -> None:
    await websocket.accept()
    settings = get_settings()

    try:
        principal = authenticate_static_bearer(
            authorization=websocket.headers.get("authorization"),
            settings=settings,
        )
        jarvis_session_id = _required_uuid_query(websocket, "session_id")
        client_kind = _client_kind(websocket)
        selection = select_audio_provider(
            settings=settings,
            requested_values=websocket.query_params.getlist("audio_provider"),
        )
        live_configuration = select_live_voice_route(settings=settings,
            requested_values=websocket.query_params.getlist("voice_route"),
            provider=selection.provider, transport=websocket.query_params.get("audio_transport", "websocket"))
        lean_selected = select_lean_voice(settings=settings,
            requested_values=websocket.query_params.getlist("conversation_mode"),
            provider=selection.provider, transport=websocket.query_params.get("audio_transport", "websocket"),
            live_selected=live_configuration is not None)
        if live_configuration is not None:
            selection = replace(selection, model=live_configuration.model)
        elif selection.provider is AudioProvider.OPENAI:
            _require_realtime_configured(settings)

        pool = get_pool(websocket)
        repository = SessionRepository(pool)
        jarvis_session = await repository.get_session(principal.owner_id, jarvis_session_id)
        if jarvis_session is None:
            raise JarvisHTTPError(404, "session_not_found", "Jarvis session was not found.")
        if jarvis_session.status != "active":
            raise JarvisHTTPError(409, "session_closed", "Jarvis session is closed.")

        runtime_services = getattr(websocket.app.state, "runtime_services", None)
        if not isinstance(runtime_services, JarvisRuntimeServices):
            raise JarvisHTTPError(
                503,
                "realtime_manager_not_configured",
                "Realtime session manager is not configured.",
            )
        manager = runtime_services.realtime_session_manager
        visual_services = runtime_services.visual_memory_services
        visual_repository = (
            visual_services.repository
            if isinstance(visual_services, VisualMemoryServices)
            else None
        )

        audio_session_id = str(uuid4())
        context = RealtimeSessionContext(
            audio_session_id=audio_session_id,
            owner_id=principal.owner_id,
            jarvis_session_id=jarvis_session.id,
            client_kind=client_kind,
            session=jarvis_session,
        )
        session_class = RealtimeAudioSession
        provider_kwargs = {}
        voice_route_capabilities = None
        transport = websocket.query_params.get("audio_transport", "websocket")
        if transport not in {"websocket", "webrtc"}:
            raise JarvisHTTPError(400, "unsupported_audio_transport", "Unsupported audio transport.")
        if transport == "webrtc":
            if selection.provider is not AudioProvider.OPENAI:
                raise JarvisHTTPError(400, "unsupported_audio_transport", "WebRTC requires OpenAI audio.")
            from app.realtime.direct_session import DirectRealtimeAudioSession
            from app.realtime.webrtc import MAX_SDP_BYTES, validate_offer
            try:
                async with asyncio.timeout(20):
                    raw = await websocket.receive_text()
                if len(raw.encode()) > MAX_SDP_BYTES + 4096:
                    raise ValueError("Offer is too large.")
                offer_payload = json.loads(raw)
                offer, capture_mode = validate_offer(offer_payload)
                voice_route_capabilities = offer_payload.get("voice_route")
            except (ValueError, TimeoutError):
                raise JarvisHTTPError(400, "invalid_webrtc_offer", "A valid audio offer is required.") from None
            session_class = DirectRealtimeAudioSession
            provider_kwargs = {"offer": offer, "capture_mode": capture_mode}
            require_lean_offer(offer_payload, selected=lean_selected, capture_mode=capture_mode)
            if lean_selected:
                from app.realtime.lean_session import LeanRealtimeAudioSession
                session_class = LeanRealtimeAudioSession
                provider_kwargs["runtime_services"] = runtime_services
            if live_configuration is not None:
                require_live_client_offer(capture_mode=capture_mode,
                    client_capabilities=voice_route_capabilities)
        if selection.provider is AudioProvider.GEMINI:
            session_class = _create_gemini_session
            provider_kwargs = {
                "config": selection.gemini_config,
                "connection_admission": lambda: require_gemini_live_enabled(settings),
            }
        context_kwargs = {}
        capture = _create_context_capture(
            runtime_services, owner_id=principal.owner_id, session_id=jarvis_session.id,
            provider=selection.provider.value,
            audio_session_id=audio_session_id,
        )
        if capture is not None:
            context_kwargs["context_capture"] = capture
        if live_configuration is not None:
            session = await _create_live_session(websocket=websocket, context=context,
                settings=settings, repository=repository, runtime_services=runtime_services,
                offer=offer, capture_mode=capture_mode, client_capabilities=voice_route_capabilities,
                context_capture=capture, route_configuration=live_configuration)
        else:
            session = session_class(
                websocket=websocket,
                context=context,
                settings=settings,
                repository=repository,
                tool_modules=runtime_services.tool_modules,
                context_providers=runtime_services.assistant_context_providers,
                visual_repository=visual_repository,
                sensory_frame_window=runtime_services.sensory_frame_window,
                reminder_repository=ReminderRepository(pool),
                tool_execution_repository=ToolExecutionRepository(pool),
                experience_event_service=runtime_services.experience_event_service,
                **context_kwargs,
                **provider_kwargs,
            )
        attachment = None
        if voice_route_capabilities is not None and live_configuration is None:
            from app.realtime.live_state_bridge import install_responsive_voice
            attachment = await install_responsive_voice(
                session, runtime_services, settings, voice_route_capabilities)
        try:
            registered = await manager.register(
                audio_session_id=audio_session_id,
                owner_id=principal.owner_id,
                jarvis_session_id=jarvis_session.id,
                session=session,
            )
        except BaseException:
            if attachment is not None:
                await attachment.close("voice_route_registration_failed")
            if live_configuration is not None:
                await session.close("voice_route_registration_failed")
            raise
        if not registered:
            if attachment is not None:
                await attachment.close("voice_route_registration_rejected")
            if live_configuration is not None:
                await session.close("voice_route_registration_rejected")
            raise JarvisHTTPError(
                409,
                "audio_session_already_active",
                "This Jarvis session already has an active realtime audio connection.",
            )

        try:
            if attachment is not None:
                try:
                    attachment.registered()
                except Exception as error:
                    # Admission succeeded, so the surrounding finally must join
                    # this attachment and unregister the audio owner. Return a
                    # bounded startup error rather than leaking registry details.
                    raise JarvisHTTPError(
                        503,
                        "perception_registration_unavailable",
                        "The session's visual context could not be initialized.",
                    ) from error
            if selection.explicit or selection.provider is AudioProvider.GEMINI or live_configuration is not None:
                # No session tasks are running yet, so this control event precedes
                # the unchanged provider session's session.created notification.
                sender = WebSocketEventSender(websocket)
                try:
                    await sender.send_event(
                        selection.as_event(audio_session_id=audio_session_id)
                    )
                except BaseException:
                    await session.close("provider_selection_failed")
                    raise
                if sender.closed:
                    await session.close("client_disconnect")
                    return
            await session.run()
        finally:
            try:
                if attachment is not None:
                    await attachment.close("voice_session_finished")
                if live_configuration is not None:
                    await session.close("voice_session_finished")
            finally:
                await manager.unregister(
                    audio_session_id=audio_session_id,
                    owner_id=principal.owner_id,
                    jarvis_session_id=jarvis_session.id,
                )
    except JarvisHTTPError as error:
        await _send_ws_error_and_close(websocket, error)


async def _create_live_session(**kwargs):
    # Import only after explicit admission. Merely installing Live never creates
    # a second transport or changes a legacy Realtime session's constructor.
    try:
        from app.realtime.live_session import create_live_session
        return await create_live_session(**kwargs)
    except JarvisHTTPError:
        raise
    except Exception:
        LOGGER.error("Live voice initialization failed")
        raise JarvisHTTPError(503, "live_voice_initialization_failed",
            "The requested Live route could not be initialized; no fallback speaker was started.") from None


def _create_context_capture(runtime_services, *, owner_id, session_id, provider, audio_session_id):
    knowledge = runtime_services.knowledge_runtime
    if knowledge is not None:
        repository = knowledge.context_for(owner_id)
    elif isinstance(runtime_services.background_task_service, TaskService):
        repository = ContextRepository(runtime_services.background_task_service.repository)
    else:
        return None
    return SessionContextCapture(
        repository, owner_id=owner_id, session_id=session_id,
        provider=provider, audio_session_id=audio_session_id,
    )


def _create_gemini_session(**kwargs):
    # Import and initialization failures remain local to the selected branch.
    # The original GPT constructor never gains these arguments or dependencies.
    try:
        from app.realtime.gemini.session import GeminiAudioSession

        return GeminiAudioSession(**kwargs)
    except JarvisHTTPError:
        raise
    except Exception as error:
        LOGGER.error("Gemini audio initialization failed error_type=%s", type(error).__name__)
        raise JarvisHTTPError(
            503, "gemini_live_initialization_failed", "Gemini Live audio could not be initialized.",
            details={"provider": "gemini"},
        ) from None


def _required_uuid_query(websocket: WebSocket, name: str) -> UUID:
    value = websocket.query_params.get(name)
    if value is None:
        raise JarvisHTTPError(400, "missing_session_id", f"{name} is required.")
    try:
        return UUID(value)
    except ValueError as error:
        raise JarvisHTTPError(400, "invalid_session_id", f"{name} must be a UUID.") from error


def _client_kind(websocket: WebSocket) -> AudioClientKind:
    value = websocket.query_params.get("client_kind", "rayban_audio")
    if value not in SUPPORTED_AUDIO_CLIENT_KINDS:
        raise JarvisHTTPError(
            400,
            "unsupported_client_kind",
            "client_kind must be rayban_audio or phone_audio.",
        )
    return value  # type: ignore[return-value]


def _require_realtime_configured(settings: JarvisSettings) -> None:
    if not is_configured_secret(settings.OPENAI_API_KEY) or not is_configured_secret(
        settings.OPENAI_REALTIME_MODEL
    ):
        raise JarvisHTTPError(
            503,
            "realtime_not_configured",
            "OpenAI Realtime is not configured.",
        )


async def _send_ws_error_and_close(websocket: WebSocket, error: JarvisHTTPError) -> None:
    await websocket.send_json(
        {
            "type": "error",
            "code": error.code,
            "message": error.message,
            "recoverable": False,
            "details": error.details,
        }
    )
    await websocket.close(
        code=status.WS_1008_POLICY_VIOLATION,
        reason=error.code[:120],
    )
