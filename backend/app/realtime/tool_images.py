"""Session-local image delivery, separate from durable domain-tool execution.

A durable ``selected`` result records retrieval, not provider acceptance. Replays
resolve that exact selection again; only a new provider ACK permits ``attached``.
Consequently a persisted success and a session-local delivery failure are not
contradictory. Pixels belong only to the resolver and the shared image writer.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from app.images import ImageReference, ImageResolver, ImageUnavailableError
from app.realtime.visual_context import RequestedImageDeliveryError
from app.realtime.visual_delivery import RealtimeVisualCoordinator


LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class ToolImageRequest:
    connection_epoch: UUID
    provider_call_id: str
    source_response_id: str
    turn_generation: int
    tool_chain_id: str
    started: bool = False
    cancelled: bool = False
    reference: ImageReference | None = None


def image_unavailable(code: str, message: str) -> dict[str, Any]:
    return {
        "status": "failed", "image_status": "unavailable",
        "error": {"code": code, "message": message},
    }


async def resolve_and_deliver_tool_image(
    selection: dict[str, Any], *, resolver: ImageResolver,
    coordinator: RealtimeVisualCoordinator | None, request: ToolImageRequest,
    is_request_active: Callable[[], bool], monotonic: Callable[[], float],
) -> dict[str, Any]:
    """Run outside receive/admission locks, retaining no pixels after return."""
    started = monotonic()
    outcome = "request_cancelled"
    item_id = None
    image = None
    try:
        if selection.get("status") != "success":
            outcome = "selection_failed"
            return {**selection, "image_status": "unavailable"}
        if not is_request_active():
            return image_unavailable("request_cancelled", "The image request has ended.")
        if coordinator is None:
            outcome = "image_delivery_unavailable"
            return image_unavailable(outcome, "Image delivery is unavailable on this connection.")
        # Resolution has its own bounded wait; the coordinator bounds write/ACK.
        try:
            async with asyncio.timeout(coordinator.image_resolution_timeout_seconds):
                image = await resolver(selection)
        except TimeoutError:
            outcome = "image_resolution_timeout"
            return image_unavailable(outcome, "The selected image could not be resolved in time.")
        request.reference = image.reference
        if not is_request_active():
            return image_unavailable("request_cancelled", "The image request has ended.")
        item = await coordinator.deliver_requested_image(
            image, provider_call_id=request.provider_call_id,
            turn_generation=request.turn_generation, is_request_active=is_request_active,
        )
        item_id = item.item_id
        outcome = "attached"
        return {
            **selection, "status": "success", "image_status": "attached",
            "provider_item_id": item_id, "image": image.reference.to_metadata(),
        }
    except (ImageUnavailableError, RequestedImageDeliveryError) as error:
        outcome = error.code
        return image_unavailable(error.code, str(error))
    except asyncio.CancelledError:
        raise
    except Exception:
        # Exception text can include storage paths, URLs, or model/media payloads.
        outcome = "image_resolution_failed"
        return image_unavailable(outcome, "The selected image could not be delivered.")
    finally:
        image = None
        reference = request.reference
        LOGGER.info(
            "tool image delivery call_id=%s response_id=%s generation=%s epoch=%s "
            "frame_id=%s item_id=%s source=%s outcome=%s duration_ms=%.1f",
            request.provider_call_id, request.source_response_id,
            request.turn_generation, request.connection_epoch,
            reference.frame_id if reference else None, item_id,
            reference.source if reference else None, outcome,
            max(0.0, monotonic() - started) * 1000,
        )
