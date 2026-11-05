from __future__ import annotations

from collections.abc import Iterable

from app.memory.jobs.handlers import MemoryJobHandler


class DuplicateMemoryJobHandlerError(ValueError):
    pass


class UnknownMemoryJobHandlerError(LookupError):
    pass


class MemoryJobHandlerRegistry:
    """Explicit process-local mapping from durable job type to implementation."""

    def __init__(self, handlers: Iterable[MemoryJobHandler] = ()) -> None:
        self._handlers: dict[str, MemoryJobHandler] = {}
        for handler in handlers:
            self.register(handler)

    def register(self, handler: MemoryJobHandler) -> None:
        job_type = _normalize_name(handler.job_type, label="job_type")
        if job_type in self._handlers:
            raise DuplicateMemoryJobHandlerError(
                f"memory job handler already registered: {job_type}"
            )
        self._handlers[job_type] = handler

    def get(self, job_type: str) -> MemoryJobHandler | None:
        return self._handlers.get(job_type.strip())

    def require(self, job_type: str) -> MemoryJobHandler:
        handler = self.get(job_type)
        if handler is None:
            raise UnknownMemoryJobHandlerError(
                f"no memory job handler registered for {job_type}"
            )
        return handler

    @property
    def job_types(self) -> tuple[str, ...]:
        return tuple(sorted(self._handlers))


def _normalize_name(value: str, *, label: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{label} cannot be blank")
    return normalized
