from __future__ import annotations

from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from app.runtime.services import JarvisRuntimeServices


__all__ = ["JarvisRuntimeServices"]


def __getattr__(name: str) -> Any:
    if name == "JarvisRuntimeServices":
        from app.runtime.services import JarvisRuntimeServices

        return JarvisRuntimeServices
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
