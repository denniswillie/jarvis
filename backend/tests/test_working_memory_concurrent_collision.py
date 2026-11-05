from __future__ import annotations

import asyncio
import unittest
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

from app.perception.models import (
    SessionWorkingMemoryState,
    VisualObservation,
    WorkingMemoryWriteResult,
)
from app.perception.services import WorkingMemoryService
from app.perception.working_memory import WorkingMemoryPolicy


NOW = datetime(2026, 8, 21, 12, 0, tzinfo=UTC)


class _CollidingRepository:
    def __init__(self) -> None:
        self.state: SessionWorkingMemoryState | None = None
        self._initial_reads = 0
        self._both_read = asyncio.Event()
        self._lock = asyncio.Lock()
        self.conflicts = 0

    async def get_session_working_memory(self, *, owner_id, session_id):
        snapshot = self.state
        if snapshot is None and self._initial_reads < 2:
            self._initial_reads += 1
            if self._initial_reads == 2:
                self._both_read.set()
            await self._both_read.wait()
        return snapshot

    async def compare_and_swap_session_working_memory(
        self, *, owner_id, session_id, expected_revision, observation_id,
        state, expires_at, contract_version=1,
    ):
        async with self._lock:
            revision = self.state.revision if self.state else 0
            if revision != expected_revision:
                self.conflicts += 1
                assert self.state is not None
                return WorkingMemoryWriteResult(False, self.state)
            now = datetime.now(UTC)
            self.state = SessionWorkingMemoryState(
                session_id=session_id, owner_id=owner_id, revision=revision + 1,
                last_observation_id=observation_id,
                last_observation_ended_at=datetime.fromisoformat(state["updated_at"]),
                state=state, expires_at=expires_at,
                created_at=self.state.created_at if self.state else now,
                updated_at=now, contract_version=contract_version,
            )
            return WorkingMemoryWriteResult(True, self.state)

    async def get_owner_working_memory(self, *, owner_id):
        return None


class WorkingMemoryCollisionTests(unittest.IsolatedAsyncioTestCase):
    async def test_forced_cas_collision_converges_on_newest_observation(self) -> None:
        repository = _CollidingRepository()
        service = WorkingMemoryService(
            repository=repository,
            policy=WorkingMemoryPolicy(120, 30, 60, 30, 12),
            now=lambda: NOW,
        )
        session_id = uuid4()
        await asyncio.gather(
            service.apply_observation(_observation(session_id, NOW, "older")),
            service.apply_observation(
                _observation(session_id, NOW + timedelta(seconds=1), "newer")
            ),
        )

        self.assertEqual(repository.conflicts, 1)
        assert repository.state is not None
        self.assertEqual(repository.state.state["summary"], "newer")


def _observation(session_id: UUID, ended_at: datetime, summary: str) -> VisualObservation:
    return VisualObservation(
        id=uuid4(), owner_id="owner", session_id=session_id, source="glasses",
        started_at=ended_at, ended_at=ended_at, primary_visual_frame_id=None,
        summary=summary, scene="room", activity="testing", entities=(),
        visible_text=(), state_changes=(), memory_tags=(), keyframe_candidates=(),
        interrupt_candidate=None, starts_new_event=False, boundary_reason=None,
        confidence=0.9, analyzer_provider="test", analyzer_model="test",
        analyzer_contract_version=1, compacted_at=None, created_at=ended_at,
        scene_confidence=0.9, activity_confidence=0.9,
    )
