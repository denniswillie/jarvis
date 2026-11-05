from __future__ import annotations

import json
import logging
from datetime import UTC, datetime, timedelta
from collections.abc import Awaitable, Callable
from typing import Any
from uuid import UUID

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, status

from app.auth.static_bearer import authenticate_static_bearer
from app.errors import JarvisHTTPError
from app.realtime.socket_sender import WebSocketEventSender
from app.runtime.events import DomainEvent, DomainEventName, VisualFrameAcceptedPayload
from app.sessions.repository import SessionRepository
from app.settings import get_settings
from app.visual_memory.latest_frame_buffer import LatestVisualFrameBuffer
from app.perception.frame_window import (
    RecentVisualFrameWindow, SensoryVisualFrame, VisualStreamLease,
)
from app.visual_memory.repository import (
    VisualFrameDuplicateError,
    VisualFrameRepository,
    VisualFrameSessionMissingError,
    buffered_frame_from_record,
)
from app.visual_memory.services import VisualMemoryServices
from app.visual_memory.storage import VisualFrameStorageClient
from app.visual_memory.validation import VisualFrameValidationError, validate_visual_frame_event


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/realtime/visual", tags=["visual-memory"])


@router.websocket("/ws")
async def visual_realtime_websocket(websocket: WebSocket) -> None:
    await websocket.accept()
    settings = get_settings()
    stream_window: RecentVisualFrameWindow | None = None
    stream_lease: VisualStreamLease | None = None

    def retire_stream() -> None:
        if stream_window is not None and stream_lease is not None:
            stream_window.end_stream(stream_lease)

    sender = WebSocketEventSender(websocket, on_closed=retire_stream)
    send_event = sender.send_event

    try:
        principal = authenticate_static_bearer(
            authorization=websocket.headers.get("authorization"),
            settings=settings,
        )
        session_id = _required_uuid_query(websocket, "session_id")
        visual_session_id = _required_uuid_query(websocket, "visual_session_id")
        audio_session_id = _optional_string_query(websocket, "audio_session_id")

        runtime = getattr(websocket.app.state, "runtime_services", None)
        pool = getattr(runtime, "db_pool", None)
        if pool is None:
            raise JarvisHTTPError(
                503,
                "jarvis_database_not_configured",
                "Jarvis database is not configured.",
            )

        storage = getattr(runtime, "visual_frame_storage", None)
        if not isinstance(storage, VisualFrameStorageClient):
            raise JarvisHTTPError(
                503,
                "storage_not_configured",
                "Visual frame storage is not configured.",
            )

        latest_frame_buffer = getattr(runtime, "latest_visual_frame_buffer", None)
        sensory_frame_window = getattr(runtime, "sensory_frame_window", None)
        if not isinstance(latest_frame_buffer, LatestVisualFrameBuffer) or not isinstance(
            sensory_frame_window,
            RecentVisualFrameWindow,
        ):
            raise JarvisHTTPError(
                503,
                "visual_frame_buffer_not_configured",
                "Visual frame buffer is not configured.",
            )
        visual_memory_services = getattr(runtime, "visual_memory_services", None)

        repository = VisualFrameRepository(pool)
        session_repository = SessionRepository(pool)
        session = await session_repository.get_session(principal.owner_id, session_id)
        if session is None:
            raise JarvisHTTPError(404, "session_not_found", "Jarvis session was not found.")
        if session.status != "active":
            raise JarvisHTTPError(409, "session_closed", "Jarvis session is closed.")

        allowed_sources = {"glasses"}
        if settings.PHONE_CAMERA_DEBUG_ENABLED:
            allowed_sources.add("iphone_camera")

        stream_lease = sensory_frame_window.begin_stream(
            principal.owner_id, session_id, visual_session_id,
        )
        stream_window = sensory_frame_window
        await send_event(
            {
                "type": "visual.session.created",
                "session_id": str(session_id),
                "visual_session_id": str(visual_session_id),
                "audio_session_id": audio_session_id,
                "allowed_sources": sorted(allowed_sources),
            }
        )

        while not sender.closed:
            message = await websocket.receive_text()
            frame_sequence_for_error: int | None = None
            if len(message.encode("utf-8")) > settings.JARVIS_REALTIME_MAX_CLIENT_MESSAGE_BYTES:
                await _send_dropped(send_event, code="message_too_large", message="Visual frame event is too large.")
                continue

            try:
                values = _decode_event(message)
                frame_sequence_for_error = _optional_frame_sequence(values)
                frame = validate_visual_frame_event(
                    values,
                    settings=settings,
                    allowed_sources=allowed_sources,
                )
                now = datetime.now(UTC)
                expires_at = now + timedelta(seconds=settings.VISUAL_RAW_FRAME_TTL_SECONDS)
                storage_path = storage.build_storage_path(
                    owner_id=principal.owner_id,
                    session_id=session_id,
                    visual_session_id=visual_session_id,
                    frame_sequence=frame.frame_sequence,
                )
                record = await repository.create_pending_frame(
                    owner_id=principal.owner_id,
                    session_id=session_id,
                    audio_session_id=audio_session_id,
                    visual_session_id=visual_session_id,
                    frame=frame,
                    server_received_at=now,
                    storage_bucket=storage.bucket,
                    storage_path=storage_path,
                    raw_image_expires_at=expires_at,
                )
                stored_object = await storage.upload_frame(
                    path=storage_path,
                    image_bytes=frame.image_bytes,
                    content_type=frame.content_type,
                )
                try:
                    available_record = await repository.mark_frame_available(
                        owner_id=principal.owner_id,
                        frame_id=record.id,
                        stored_object=stored_object,
                    )
                except Exception:
                    await _cleanup_uploaded_frame(storage=storage, bucket=stored_object.bucket, path=stored_object.path)
                    raise
                if available_record is None:
                    await _cleanup_uploaded_frame(storage=storage, bucket=stored_object.bucket, path=stored_object.path)
                    await repository.mark_frame_failed(owner_id=principal.owner_id, frame_id=record.id)
                    await _send_dropped(
                        send_event,
                        code="visual_frame_missing",
                        message="Visual frame was not found after upload.",
                        frame_sequence=frame.frame_sequence,
                    )
                    continue

                if not sensory_frame_window.owns_stream(stream_lease):
                    await _send_dropped(
                        send_event, code="visual_stream_superseded",
                        message="A newer glasses stream is active.",
                        frame_sequence=frame.frame_sequence,
                    )
                    continue
                buffered = sensory_frame_window.put(
                    SensoryVisualFrame.from_buffered_frame(
                        buffered_frame_from_record(available_record),
                        image_bytes=frame.image_bytes,
                    ),
                    stream_lease=stream_lease,
                )
                if not buffered:
                    continue
                await runtime.domain_event_bus.publish(
                    DomainEvent(
                        name=DomainEventName.VISUAL_FRAME_ACCEPTED,
                        payload=VisualFrameAcceptedPayload(
                            owner_id=principal.owner_id, session_id=session_id,
                            frame_id=available_record.id, frame_sequence=frame.frame_sequence,
                        ),
                    )
                )
            except VisualFrameValidationError as error:
                await _send_dropped(
                    send_event,
                    code=error.code,
                    message=error.message,
                    frame_sequence=frame_sequence_for_error,
                )
                continue
            except VisualFrameDuplicateError:
                await _send_dropped(
                    send_event,
                    code="duplicate_frame",
                    message="Visual frame was already ingested.",
                    frame_sequence=frame_sequence_for_error,
                )
                continue
            except VisualFrameSessionMissingError:
                await _send_dropped(
                    send_event,
                    code="session_not_found",
                    message="Jarvis session is not active.",
                    frame_sequence=frame_sequence_for_error,
                )
                continue
            except JarvisHTTPError as error:
                await _send_dropped(
                    send_event,
                    code=error.code,
                    message=error.message,
                    frame_sequence=frame_sequence_for_error,
                )
                continue
            except Exception:
                logger.exception("visual frame ingestion failed")
                await _send_dropped(
                    send_event,
                    code="visual_frame_create_failed",
                    message="Visual frame could not be stored.",
                    frame_sequence=frame_sequence_for_error,
                )
                continue

            # Storage/acceptance is already committed. A departed client cannot
            # turn acknowledgement failure into an ingestion rollback. Accepted
            # analysis still queues with a sender that becomes a no-op on close.
            await send_event(
                {
                    "type": "visual.frame.accepted",
                    "frame_id": str(available_record.id),
                    "session_id": str(session_id),
                    "visual_session_id": str(visual_session_id),
                    "audio_session_id": audio_session_id,
                    "frame_sequence": frame.frame_sequence,
                    "source": frame.source,
                    "raw_image_expires_at": available_record.raw_image_expires_at.isoformat(),
                }
            )
            if isinstance(visual_memory_services, VisualMemoryServices):
                try:
                    await visual_memory_services.enqueue(
                        frame=available_record,
                        image_bytes=frame.image_bytes,
                        send_diagnostic=send_event,
                    )
                except Exception as error:
                    logger.error(
                        "visual frame analysis enqueue failed frame_id=%s error_type=%s",
                        available_record.id, type(error).__name__,
                    )
                    await send_event(
                        {
                            "type": "visual.frame.analysis_failed",
                            "frame_id": str(available_record.id),
                            "frame_sequence": frame.frame_sequence,
                            "code": "visual_analysis_enqueue_failed",
                        }
                    )
    except WebSocketDisconnect:
        return
    except JarvisHTTPError as error:
        await _send_ws_error_and_close(sender, error)
    finally:
        sender.mark_closed()


def _decode_event(message: str) -> dict[str, Any]:
    try:
        values = json.loads(message)
    except json.JSONDecodeError as error:
        raise VisualFrameValidationError("invalid_json", "Visual frame event must be valid JSON.") from error
    if not isinstance(values, dict):
        raise VisualFrameValidationError("invalid_json", "Visual frame event must be a JSON object.")
    return values


def _required_uuid_query(websocket: WebSocket, name: str) -> UUID:
    value = websocket.query_params.get(name)
    if value is None:
        raise JarvisHTTPError(400, f"missing_{name}", f"{name} is required.")
    try:
        return UUID(value)
    except ValueError as error:
        raise JarvisHTTPError(400, f"invalid_{name}", f"{name} must be a UUID.") from error


def _optional_string_query(websocket: WebSocket, name: str) -> str | None:
    value = websocket.query_params.get(name)
    if value is None:
        return None
    trimmed = value.strip()
    return trimmed or None


def _optional_frame_sequence(values: dict[str, Any]) -> int | None:
    value = values.get("frame_sequence")
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value >= 0:
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


async def _cleanup_uploaded_frame(
    *,
    storage: VisualFrameStorageClient,
    bucket: str,
    path: str,
) -> None:
    try:
        await storage.delete_objects(bucket=bucket, paths=[path])
    except Exception:
        logger.exception("failed to clean up uploaded visual frame object %s/%s", bucket, path)


async def _send_dropped(
    send_event: Callable[[dict[str, object]], Awaitable[None]],
    *,
    code: str,
    message: str,
    frame_sequence: int | None = None,
) -> None:
    payload: dict[str, Any] = {
        "type": "visual.frame.dropped",
        "code": code,
        "message": message,
        "recoverable": True,
    }
    if frame_sequence is not None:
        payload["frame_sequence"] = frame_sequence
    await send_event(payload)


async def _send_ws_error_and_close(
    sender: WebSocketEventSender,
    error: JarvisHTTPError,
) -> None:
    await sender.send_event(
        {
            "type": "error",
            "code": error.code,
            "message": error.message,
            "recoverable": False,
            "details": error.details,
        }
    )
    await sender.close(
        code=status.WS_1008_POLICY_VIOLATION,
        reason=error.code[:120],
    )
