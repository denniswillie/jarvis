from __future__ import annotations

from typing import Any, Mapping, Protocol

import asyncpg

from app.memory.jobs.models import MemoryJob


class MemoryJobValidationError(ValueError):
    """A durable job is invalid and must not be retried automatically."""


class MemoryJobTransientError(RuntimeError):
    """A job failed for a reason that may succeed on a later attempt."""


class MemoryJobHandler(Protocol):
    """Two-phase handler that keeps slow work outside its commit transaction."""

    job_type: str

    async def prepare(self, job: MemoryJob) -> Any:
        """Perform provider, storage, parsing, and pure validation work."""

    async def commit(
        self,
        job: MemoryJob,
        prepared_result: Any,
        connection: asyncpg.Connection,
    ) -> Mapping[str, Any] | None:
        """Persist bounded domain changes on the runner-owned connection."""
