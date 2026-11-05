from __future__ import annotations

import json
import unittest
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID, uuid4

from app.perception.models import ObservationFrameLink, VisualObservation
from app.perception.repository import PerceptionRepository


class _AsyncContext:
    def __init__(self, value: object) -> None:
        self._value = value

    async def __aenter__(self) -> object:
        return self._value

    async def __aexit__(self, *_: object) -> None:
        return None


class _Connection:
    def __init__(self, fetchrow_results: list[dict[str, Any] | None]) -> None:
        self.fetchrow_results = list(fetchrow_results)
        self.fetchrow_calls: list[tuple[str, tuple[object, ...]]] = []
        self.executemany_calls: list[tuple[str, list[tuple[object, ...]]]] = []

    def transaction(self) -> _AsyncContext:
        return _AsyncContext(self)

    async def fetchrow(self, query: str, *args: object) -> dict[str, Any] | None:
        self.fetchrow_calls.append((query, args))
        return self.fetchrow_results.pop(0)

    async def executemany(
        self,
        query: str,
        args: list[tuple[object, ...]],
    ) -> None:
        self.executemany_calls.append((query, args))


class _Pool:
    def __init__(self, connection: _Connection) -> None:
        self.connection = connection

    def acquire(self) -> _AsyncContext:
        return _AsyncContext(self.connection)


class PerceptionRepositoryTests(unittest.IsolatedAsyncioTestCase):
    async def test_insert_observation_and_frame_links_share_transaction(self) -> None:
        observation = _observation(primary_visual_frame_id=uuid4())
        row = _observation_row(observation)
        connection = _Connection([row])
        repository = PerceptionRepository(_Pool(connection))  # type: ignore[arg-type]
        links = [
            ObservationFrameLink(uuid4(), 0, "context"),
            ObservationFrameLink(observation.primary_visual_frame_id, 1, "primary"),
        ]

        created = await repository.insert_observation(
            observation=observation,
            frame_links=links,
        )

        self.assertEqual(created.id, observation.id)
        self.assertEqual(len(connection.executemany_calls), 1)
        inserted_links = connection.executemany_calls[0][1]
        self.assertEqual([values[2] for values in inserted_links], [0, 1])

    async def test_session_working_memory_applies_newer_observation(self) -> None:
        observation = _observation()
        state_row = _session_state_row(observation, revision=2, scene="office")
        connection = _Connection([state_row])
        repository = PerceptionRepository(_Pool(connection))  # type: ignore[arg-type]

        result = await repository.compare_and_swap_session_working_memory(
            owner_id=observation.owner_id,
            session_id=observation.session_id,
            expected_revision=1,
            observation_id=observation.id,
            state={"scene": "office"},
            expires_at=observation.ended_at + timedelta(minutes=5),
        )

        self.assertTrue(result.applied)
        self.assertEqual(result.state.revision, 2)
        self.assertEqual(result.state.state, {"scene": "office"})
        query = connection.fetchrow_calls[0][0]
        self.assertIn("last_observation_ended_at", query)
        self.assertIn("> jarvis_session_working_memory.last_observation_ended_at", query)
        self.assertIn("from jarvis_session_working_memory current_state", query)

    async def test_session_working_memory_returns_current_state_when_stale(self) -> None:
        observation = _observation()
        current = _session_state_row(observation, revision=4, scene="hallway")
        connection = _Connection([None, current])
        repository = PerceptionRepository(_Pool(connection))  # type: ignore[arg-type]

        result = await repository.compare_and_swap_session_working_memory(
            owner_id=observation.owner_id,
            session_id=observation.session_id,
            expected_revision=3,
            observation_id=observation.id,
            state={"scene": "stale room"},
            expires_at=observation.ended_at + timedelta(minutes=5),
        )

        self.assertFalse(result.applied)
        self.assertEqual(result.state.revision, 4)
        self.assertEqual(result.state.state, {"scene": "hallway"})

    async def test_owner_working_memory_uses_revision_compare_and_swap(self) -> None:
        now = datetime.now(UTC)
        owner_row = {
            "owner_id": "owner",
            "revision": 1,
            "state": json.dumps({"_contract_version": 1, "goal": "leave home"}),
            "expires_at": None,
            "created_at": now,
            "updated_at": now,
        }
        connection = _Connection([owner_row])
        repository = PerceptionRepository(_Pool(connection))  # type: ignore[arg-type]

        result = await repository.compare_and_swap_owner_working_memory(
            owner_id="owner",
            expected_revision=0,
            state={"goal": "leave home"},
        )

        self.assertTrue(result.applied)
        self.assertEqual(result.state.state, {"goal": "leave home"})
        self.assertIn(
            "jarvis_owner_working_memory.revision = $2",
            connection.fetchrow_calls[0][0],
        )
        self.assertIn(
            "from jarvis_owner_working_memory current_state",
            connection.fetchrow_calls[0][0],
        )

    def test_primary_frame_link_must_match_observation(self) -> None:
        observation = _observation(primary_visual_frame_id=uuid4())
        repository = PerceptionRepository(_Pool(_Connection([])))  # type: ignore[arg-type]

        async def insert() -> None:
            await repository.insert_observation(
                observation=observation,
                frame_links=[ObservationFrameLink(uuid4(), 0, "primary")],
            )

        with self.assertRaisesRegex(ValueError, "must match"):
            self.run_async(insert())

    def run_async(self, awaitable: Any) -> Any:
        import asyncio

        return asyncio.run(awaitable)


def _observation(*, primary_visual_frame_id: UUID | None = None) -> VisualObservation:
    now = datetime.now(UTC)
    return VisualObservation(
        id=uuid4(),
        owner_id="owner",
        session_id=uuid4(),
        source="glasses",
        started_at=now,
        ended_at=now,
        primary_visual_frame_id=primary_visual_frame_id,
        summary="A stable scene.",
        scene="room",
        activity=None,
        entities=(),
        visible_text=(),
        state_changes=(),
        memory_tags=(),
        keyframe_candidates=(),
        interrupt_candidate=None,
        starts_new_event=False,
        boundary_reason=None,
        confidence=0.9,
        analyzer_provider="test",
        analyzer_model="test-model",
        analyzer_contract_version=1,
        compacted_at=None,
        created_at=now,
    )


def _observation_row(observation: VisualObservation) -> dict[str, Any]:
    return {
        "id": observation.id,
        "owner_id": observation.owner_id,
        "session_id": observation.session_id,
        "source": observation.source,
        "started_at": observation.started_at,
        "ended_at": observation.ended_at,
        "primary_visual_frame_id": observation.primary_visual_frame_id,
        "summary": observation.summary,
        "scene": observation.scene,
        "activity": observation.activity,
        "entities": "[]",
        "visible_text": "[]",
        "state_changes": "[]",
        "memory_tags": [],
        "keyframe_candidates": "[]",
        "interrupt_candidate": None,
        "starts_new_event": False,
        "boundary_reason": None,
        "confidence": observation.confidence,
        "analyzer_provider": observation.analyzer_provider,
        "analyzer_model": observation.analyzer_model,
        "analyzer_contract_version": 1,
        "compacted_at": None,
        "created_at": observation.created_at,
    }


def _session_state_row(
    observation: VisualObservation,
    *,
    revision: int,
    scene: str,
) -> dict[str, Any]:
    expires_at = observation.ended_at + timedelta(minutes=5)
    return {
        "session_id": observation.session_id,
        "owner_id": observation.owner_id,
        "revision": revision,
        "last_observation_id": observation.id,
        "last_observation_ended_at": observation.ended_at,
        "state": json.dumps({"_contract_version": 1, "scene": scene}),
        "expires_at": expires_at,
        "created_at": observation.created_at,
        "updated_at": observation.created_at,
    }
