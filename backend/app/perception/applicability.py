"""Current interpretation guards, separate from historical observation retention."""
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Callable
from uuid import UUID

from app.knowledge.models import SourceError


class PerceptionEffectUnavailable(SourceError):
    def __init__(self, reason="perception_assignment_unavailable"):
        super().__init__(reason, "The observation is retained, but its assignment is no longer applicable.")


@dataclass(frozen=True)
class PerceptionEffectGuard:
    brief: object
    binding: object
    canonical_guard: Callable | None
    is_live: Callable[[], bool]
    observation_payload_id: UUID

    @property
    def focus_fence(self):
        sources = [source for source in self.brief.dependencies
                   if source.source_namespace == "visual_focus_subject"]
        if not sources:
            return None
        if len(sources) != 1 or sources[0].generation != sources[0].revision:
            raise PerceptionEffectUnavailable("perception_focus_dependency_invalid")
        return {"subject_id": str(sources[0].source_id), "generation": sources[0].generation}

    @asynccontextmanager
    async def __call__(self):
        if self.canonical_guard is None or not self.is_live():
            raise PerceptionEffectUnavailable()
        try:
            # Lock the observation original before taking the focus owner lock.
            # Effect repositories borrow this SAME connection, not another pool
            # lease that could deadlock while we retain the owner's locks.
            async with self.canonical_guard(self.brief, self.binding,
                    effect_payload_ids=(self.observation_payload_id,)) as connection:
                if not self.is_live() or not connection.is_in_transaction():
                    raise PerceptionEffectUnavailable()
                yield connection
                if not self.is_live():
                    raise PerceptionEffectUnavailable()
        except SourceError as error:
            if isinstance(error, PerceptionEffectUnavailable):
                raise
            raise PerceptionEffectUnavailable(error.code) from error


@asynccontextmanager
async def effect_connection(pool, effect_guard=None):
    """Existing repositories keep their transaction, using a guarded lease when supplied."""
    if effect_guard is not None:
        async with effect_guard() as connection:
            yield connection
    else:
        async with pool.acquire() as connection:
            yield connection
