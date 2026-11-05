from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import UUID

from app.images import ImageReference, ImageUnavailableError, ResolvedImage
from app.perception.current_image import CurrentImageService
from app.tools.inputs import failed, success
from app.tools.module import ToolBuildContext
from app.tools.registry import ToolDefinition


@dataclass(frozen=True)
class CurrentImageToolModule:
    service: CurrentImageService

    @property
    def name(self) -> str:
        return "current_image"

    def build_tools(self, context: ToolBuildContext) -> list[ToolDefinition]:
        return [
            create_get_current_camera_view_tool(
                service=self.service,
                owner_id=context.owner_id,
                session_id=context.require_session_id(),
            )
        ]


def create_get_current_camera_view_tool(
    *, service: CurrentImageService, owner_id: str, session_id: UUID
) -> ToolDefinition:
    async def handler(arguments: dict[str, Any]) -> dict[str, Any]:
        if arguments:
            return failed("invalid_tool_input", "get_current_camera_view accepts no arguments.")
        try:
            reference = service.select_current_image(
                owner_id=owner_id, session_id=session_id,
            )
        except ImageUnavailableError as error:
            return {**failed(error.code, error.message), "image_status": "unavailable"}
        # Durable audit records retrieval, never provider attachment or pixels.
        return success({"image_status": "selected", "image": reference.to_metadata()})

    async def resolve_image(selection: dict[str, Any]) -> ResolvedImage:
        if selection.get("status") != "success" or selection.get("image_status") != "selected":
            raise ImageUnavailableError("frame_unavailable", "There is no selected current image.")
        metadata = selection.get("image")
        if not isinstance(metadata, dict):
            raise ImageUnavailableError("frame_unavailable", "The selected image reference is unavailable.")
        try:
            reference = ImageReference.from_metadata(metadata, owner_id=owner_id)
        except (KeyError, TypeError, ValueError) as error:
            raise ImageUnavailableError("frame_unavailable", "The selected image reference is invalid.") from error
        return service.resolve_current_image(
            owner_id=owner_id, session_id=session_id, reference=reference,
        )

    return ToolDefinition(
        name="get_current_camera_view",
        description=(
            "Attach the latest available current camera image for direct inspection. "
            "First use adequate fresh images already acknowledged in this response's live_view "
            "cutoff. Do not call merely because the request is visual or to reconfirm freshness. "
            "Request another view when needed pixels are absent, stale, of uncertain freshness, "
            "or do not show a relevant scene change, or when a new view could supply a missing detail. "
            "A reported change does not require another fetch if the attached image already shows it. "
            "For unreadable details in an otherwise current image, ask for a clearer view rather "
            "than repeatedly fetching an unchanged scene; retrieval does not improve legibility. "
            "Automatic image delivery being disabled does not disable this tool; "
            "it reports when a camera frame is unavailable. "
            "Use for current material missing from the available visual evidence and conversation, "
            "before asking the user to paste or describe it. Never use a present camera "
            "image to answer historical recall; use search_memory and get_memory_image instead. "
            "Returns capture time or explains why no fresh image is available; it does not "
            "analyze the image. To save an image as a file use capture_latest_visual_frame."
        ),
        input_schema={
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
        handler=handler,
        image_resolver=resolve_image,
    )
