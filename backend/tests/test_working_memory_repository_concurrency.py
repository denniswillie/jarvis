from __future__ import annotations

import asyncio
import unittest
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

from app.perception.models import SessionWorkingMemoryState, VisualObservation, WorkingMemoryWriteResult
from app.perception.services import WorkingMemoryService
from app.perception.working_memory import WorkingMemoryPolicy
from app.runtime.events import DomainEventBus, DomainEventName


NOW = datetime(2026, 8, 16, 12, 0, tzinfo=UTC)


class _Repository:
    def __init__(self) -> None:
        self.states: dict[tuple[str, UUID], SessionWorkingMemoryState] = {}
        self._lock = asyncio.Lock()

    async def get_session_working_memory(self, *, owner_id, session_id):
        return self.states.get((owner_id, session_id))

    async def compare_and_swap_session_working_memory(
        self, *, owner_id, session_id, expected_revision, observation_id,
        state, expires_at, contract_version=1,
    ):
        async with self._lock:
            key = (owner_id, session_id)
            current = self.states.get(key)
            current_revision = current.revision if current else 0
            if current_revision != expected_revision:
                assert current is not None
                return WorkingMemoryWriteResult(False, current)
            now = datetime.now(UTC)
            stored = SessionWorkingMemoryState(
                session_id=session_id, owner_id=owner_id,
                revision=current_revision + 1, last_observation_id=observation_id,
                last_observation_ended_at=datetime.fromisoformat(state["updated_at"]),
                state=state, expires_at=expires_at,
                created_at=current.created_at if current else now,
                updated_at=now, contract_version=contract_version,
            )
            self.states[key] = stored
            return WorkingMemoryWriteResult(True, stored)

    async def get_owner_working_memory(self, *, owner_id):
        return None


class WorkingMemoryConcurrencyTests(unittest.IsolatedAsyncioTestCase):
    async def test_concurrent_observations_converge_on_newest(self) -> None:
        repository = _Repository()
        bus = DomainEventBus()
        revisions: list[int] = []

        async def record(event):
            revisions.append(event.payload.revision)

        bus.subscribe(DomainEventName.WORKING_MEMORY_UPDATED, record)
        service = _service(repository, bus)
        session_id = uuid4()
        await asyncio.gather(
            service.apply_observation(_observation(session_id, NOW, "older")),
            service.apply_observation(
                _observation(session_id, NOW + timedelta(seconds=1), "newer")
            ),
        )
        stored = await service.get_session_working_memory(
            owner_id="owner", session_id=session_id
        )
        assert stored is not None
        self.assertEqual(stored.state["summary"], "newer")
        self.assertEqual(revisions, sorted(revisions))

    async def test_stale_observation_cannot_rewind_and_sessions_are_isolated(self) -> None:
        repository = _Repository()
        service = _service(repository)
        first_session = uuid4()
        second_session = uuid4()
        await service.apply_observation(
            _observation(first_session, NOW + timedelta(seconds=2), "latest")
        )
        stale = await service.apply_observation(_observation(first_session, NOW, "stale"))
        await service.apply_observation(_observation(second_session, NOW, "other"))
        self.assertFalse(stale.applied)
        self.assertEqual(stale.reason, "stale_observation")
        self.assertEqual(len(repository.states), 2)
        self.assertEqual(repository.states[("owner", first_session)].state["summary"], "latest")


def _service(repository, bus=None):
    return WorkingMemoryService(
        repository=repository,
        policy=WorkingMemoryPolicy(120, 30, 60, 30, 12),
        domain_event_bus=bus,
        now=lambda: NOW,
    )


def _observation(session_id: UUID, at: datetime, summary: str) -> VisualObservation:
    return VisualObservation(
        id=uuid4(), owner_id="owner", session_id=session_id, source="glasses",
        started_at=at, ended_at=at, primary_visual_frame_id=uuid4(), summary=summary,
        scene="room", activity="testing", entities=(), visible_text=(), state_changes=(),
        memory_tags=(), keyframe_candidates=(), interrupt_candidate=None,
        starts_new_event=False, boundary_reason=None, confidence=0.9,
        analyzer_provider="test", analyzer_model="test", analyzer_contract_version=1,
        compacted_at=None, created_at=at,
    )


if __name__ == "__main__":
    unittest.main()
