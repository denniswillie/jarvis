from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from app.files.repository import JarvisFileRepository
from app.files.storage import JarvisFileStorageClient
from app.tools.module import ToolBuildContext
from app.tools.registry import ToolDefinition
from app.visual_memory.frame_capture_tool import create_capture_latest_visual_frame_tool
from app.visual_memory.latest_frame_buffer import LatestVisualFrameBuffer
from app.visual_memory.repository import VisualFrameRepository


@dataclass(frozen=True)
class VisualCaptureToolModule:
    repository: VisualFrameRepository
    latest_frame_buffer: LatestVisualFrameBuffer | None = None
    file_repository: JarvisFileRepository | None = None
    file_storage: JarvisFileStorageClient | None = None
    name: ClassVar[str] = "visual_capture"

    def build_tools(self, context: ToolBuildContext) -> list[ToolDefinition]:
        if (
            context.session_id is None
            or self.latest_frame_buffer is None
            or self.file_repository is None
            or self.file_storage is None
        ):
            return []
        bind_session = getattr(self.file_repository, "for_session", None)
        files = bind_session(context.session_id) if bind_session is not None else self.file_repository
        return [
            create_capture_latest_visual_frame_tool(
                settings=context.require_settings(),
                latest_frame_buffer=self.latest_frame_buffer,
                visual_repository=self.repository,
                file_repository=files,
                file_storage=self.file_storage,
                owner_id=context.owner_id,
                session_id=context.session_id,
            )
        ]
