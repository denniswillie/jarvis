"""Selected-memory image tool; pixels use the existing realtime transport."""
from dataclasses import dataclass
from typing import ClassVar

from app.errors import JarvisHTTPError
from app.images import ImageReference, ImageUnavailableError
from app.tools.inputs import ToolInputError, failed, optional_uuid, required_uuid, success
from app.tools.provenance import source_read_arguments
from app.tools.registry import ToolDefinition


@dataclass(frozen=True)
class SourceImageToolModule:
    service: object
    owner_id: str
    area_id: object
    name: ClassVar[str] = "source_images"

    def build_tools(self, context):
        if context.owner_id != self.owner_id:
            raise ValueError("Source images are outside this runtime owner scope")
        bind_session = getattr(self.service, "for_session", None)
        service = bind_session(context.session_id) if bind_session is not None else self.service

        async def handler(arguments):
            try:
                if set(arguments) - {"memory_id", "revision", "frame_id"}:
                    raise ToolInputError("invalid_tool_input", "Use memory_id, optional revision and optional frame_id.")
                reference = await service.select_image(
                    owner_id=self.owner_id, area_id=self.area_id, source_id=required_uuid(arguments, "memory_id"),
                    revision=arguments.get("revision"), frame_id=optional_uuid(arguments, "frame_id"),
                )
                return success({"image_status": "selected", "image": reference.to_metadata()})
            except (ToolInputError, ImageUnavailableError, JarvisHTTPError) as error:
                return {**failed(error.code, error.message), "image_status": "unavailable"}
            except ValueError as error:
                return {**failed("invalid_tool_input", str(error)), "image_status": "unavailable"}

        async def resolve(selection):
            try:
                if selection.get("status") != "success" or selection.get("image_status") != "selected":
                    raise ValueError("No selected source image")
                reference = ImageReference.from_metadata(selection["image"], owner_id=self.owner_id)
            except (KeyError, TypeError, ValueError) as error:
                raise service.unavailable() from error
            return await service.resolve_image(owner_id=self.owner_id, reference=reference)

        return [ToolDefinition(
            name="get_memory_image",
            description=(
                "Load one retained historical image from a source-backed memory_id returned by search_memory "
                "or explain_memory, following its recorded evidence links. Omit revision for current source "
                "evidence, or supply a known historical revision. Optionally select an exact frame_id; "
                "an unavailable explicit frame is never substituted. The pixels depict their capture time, "
                "not the present, and may be unreadable. No image analysis, new capture, save or retention extension occurs."
            ),
            input_schema={"type": "object", "additionalProperties": False, "required": ["memory_id"],
                "properties": {
                    "memory_id": {"type": "string", "format": "uuid"},
                    "revision": {"type": ["integer", "null"], "minimum": 1},
                    "frame_id": {"type": ["string", "null"], "format": "uuid"},
                }},
            handler=handler, image_resolver=resolve, audit_arguments=source_read_arguments,
        )]
