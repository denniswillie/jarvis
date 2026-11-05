"""Read-side injection seam for the shared memory tool catalog.

Implementations return tool payloads, not fabricated legacy retrieval records.
Unsupported write/history services must not be mixed into a selected backend.
"""
from typing import Any, Protocol
from uuid import UUID

from app.memory.retrieval.query import MemoryRetrievalQuery


class MemoryWriteBackend(Protocol):
    async def remember(self, *, owner_id: str, session_id: UUID | None,
                       parameters: dict[str, Any]) -> dict[str, Any]: ...


class MemoryReadBackend(Protocol):
    async def search(self, *, owner_id: str, request: MemoryRetrievalQuery) -> dict[str, Any]: ...

    async def explain(
        self, *, owner_id: str, memory_id: UUID, revision: int | None = None,
        offset: int = 0, expected_digest: str | None = None,
    ) -> dict[str, Any]: ...
