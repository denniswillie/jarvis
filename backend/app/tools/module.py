from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Callable, Protocol, TYPE_CHECKING
from uuid import UUID

from app.settings import JarvisSettings

if TYPE_CHECKING:
    from app.tools.registry import ToolDefinition


@dataclass(frozen=True)
class ToolBuildContext:
    """Per-session identity and configuration supplied to every tool module."""

    settings: JarvisSettings | None
    owner_id: str
    session_id: UUID | None

    def require_settings(self) -> JarvisSettings:
        if self.settings is None:
            raise ValueError("Tool module requires Jarvis settings.")
        return self.settings

    def require_session_id(self) -> UUID:
        if self.session_id is None:
            raise ValueError("Tool module requires a Jarvis session id.")
        return self.session_id


class ToolModule(Protocol):
    """Capability-owned provider of model-facing tools."""

    name: str

    def build_tools(self, context: ToolBuildContext) -> Iterable[ToolDefinition]: ...


ToolFactory = Callable[[ToolBuildContext], Iterable["ToolDefinition"]]


@dataclass(frozen=True)
class FunctionToolModule:
    """Small adapter for existing tool factories during the module migration."""

    name: str
    factory: ToolFactory

    def build_tools(self, context: ToolBuildContext) -> Iterable[ToolDefinition]:
        return self.factory(context)
