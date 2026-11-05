from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any, Awaitable, Callable
from uuid import UUID

from app.images import ImageResolver
from app.settings import JarvisSettings
from app.tools.module import FunctionToolModule, ToolBuildContext, ToolModule
from app.tools.provenance import SourceBoundToolResult, ToolResultCapture
from app.visual_memory.repository import VisualFrameRepository


ToolHandler = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    input_schema: dict[str, Any]
    handler: ToolHandler
    mutates_state: bool = False
    image_resolver: ImageResolver | None = None
    # Optional capability-owned audit projections. Invocation still receives the
    # original arguments/results; persistence need not duplicate source bodies.
    audit_arguments: Callable[[dict[str, Any]], dict[str, Any]] | None = None
    audit_result: Callable[[dict[str, Any]], dict[str, Any]] | None = None
    # Selected read capabilities bind removable originals before persistence/exposure.
    result_capture: Callable[[dict[str, Any], ToolResultCapture], Awaitable[SourceBoundToolResult]] | None = None
    # Explicit opt-in only for a mutation whose durable local effect is replay-safe.
    # Failure to capture its receipt after handler success is not mutation failure.
    capture_after_commit: bool = False
    # Trusted opt-in for an already-scoped, in-memory snapshot reader only.
    # No source capture, effect journal or transcript lookup belongs on this path.
    local_snapshot_read: bool = False


class JarvisToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}

    def register(self, tool: ToolDefinition) -> None:
        if type(tool.local_snapshot_read) is not bool:
            raise ValueError("Local snapshot policy must be an explicit boolean")
        if tool.local_snapshot_read and (
            tool.mutates_state or tool.result_capture is not None
            or tool.image_resolver is not None or tool.capture_after_commit
        ):
            raise ValueError("Local snapshots cannot mutate or require external result materialization")
        if tool.capture_after_commit and (not tool.mutates_state or tool.result_capture is None):
            raise ValueError("After-commit capture requires an explicit mutation and receipt callback")
        if tool.result_capture is not None and (
            tool.image_resolver is not None or (tool.mutates_state and not tool.capture_after_commit)
        ):
            raise ValueError("Source-result capture requires a non-image reader or explicit after-commit receipt")
        if tool.name in self._tools:
            raise ValueError(f"Tool is already registered: {tool.name}")
        self._tools[tool.name] = tool

    def register_many(self, tools: Iterable[ToolDefinition]) -> None:
        for tool in tools:
            self.register(tool)

    def get(self, name: str) -> ToolDefinition | None:
        return self._tools.get(name)

    def list(self) -> list[ToolDefinition]:
        return list(self._tools.values())

    def openai_realtime_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.input_schema,
            }
            for tool in self.list()
        ]


def create_tool_registry(
    *,
    modules: Sequence[ToolModule],
    context: ToolBuildContext,
) -> JarvisToolRegistry:
    """Build a deterministic registry from independently constructed modules."""

    registry = JarvisToolRegistry()
    for module in modules:
        registry.register_many(module.build_tools(context))
    return registry
