from __future__ import annotations

import unittest
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

from app.experience.keyframes import DeterministicKeyframeSelector
from app.experience.repository import ExperienceRepository
from app.visual_memory.repository import VisualFrameRepository
from app.visual_memory.retention import VisualFrameRetentionService


class _RetentionRepository:
    def __init__(self, *, frames: list[Any] | None = None) -> None:
        self.calls: list[str] = []
        self.frames = list(frames or [])

    async def delete_expired_retention_reasons(self, *, limit: int) -> int:
        self.calls.append(f"reasons:{limit}")
        return 2

    async def claim_expired_raw_frames(self, *, limit: int) -> list[Any]:
        self.calls.append(f"frames:{limit}")
        return self.frames

    async def release_frame_expiry_claim(self, *, owner_id: str, frame_id: object) -> None:
        self.calls.append(f"release:{owner_id}:{frame_id}")


class _Storage:
    async def delete_objects(self, *, bucket: str, paths: list[str]) -> None:
        raise AssertionError("storage should not be called without expired frames")


class _AsyncContext:
    def __init__(self, value: object) -> None:
        self.value = value

    async def __aenter__(self) -> object:
        return self.value

    async def __aexit__(self, *_: object) -> None:
        return None


class _Connection:
    def __init__(
        self,
        *,
        fetchrow_results: list[dict[str, Any] | None] | None = None,
        fetch_results: list[list[dict[str, Any]]] | None = None,
        fetchval_results: list[object] | None = None,
        execute_results: list[BaseException | None] | None = None,
    ) -> None:
        self.fetchrow_results = list(fetchrow_results or [])
        self.fetch_results = list(fetch_results or [])
        self.fetchval_results = list(fetchval_results or [])
        self.execute_results = list(execute_results or [])
        self.calls: list[str] = []
        self.call_details: list[tuple[str, tuple[object, ...]]] = []

    def transaction(self) -> _AsyncContext:
        return _AsyncContext(self)

    async def fetchrow(self, query: str, *_: object) -> dict[str, Any] | None:
        self.calls.append(query)
        return self.fetchrow_results.pop(0)

    async def fetch(self, query: str, *_: object) -> list[dict[str, Any]]:
        self.calls.append(query)
        return self.fetch_results.pop(0)

    async def fetchval(self, query: str, *_: object) -> object:
        self.calls.append(query)
        return self.fetchval_results.pop(0)

    async def execute(self, query: str, *args: object) -> str:
        self.calls.append(query)
        self.call_details.append((query, args))
        result = self.execute_results.pop(0) if self.execute_results else None
        if isinstance(result, BaseException):
            raise result
        return "OK"


class _Pool:
    def __init__(self, connection: _Connection) -> None:
        self.connection = connection

    def acquire(self) -> _AsyncContext:
        return _AsyncContext(self.connection)


class KeyframeRetentionTests(unittest.IsolatedAsyncioTestCase):
    async def test_retention_loop_prunes_reasons_before_listing_frames(self) -> None:
        repository = _RetentionRepository()
        service = VisualFrameRetentionService(
            repository=repository,  # type: ignore[arg-type]
            storage=_Storage(),  # type: ignore[arg-type]
            batch_size=25,
        )

        expired = await service.expire_old_raw_frames()

        self.assertEqual(expired, 0)
        self.assertEqual(repository.calls, ["reasons:25", "frames:25"])

    async def test_storage_failure_releases_expiry_claim(self) -> None:
        frame_id = uuid4()
        repository = _RetentionRepository(
            frames=[
                SimpleNamespace(
                    id=frame_id,
                    owner_id="owner-1",
                    storage_bucket="frames",
                    storage_path=f"test/{frame_id}.jpg",
                )
            ]
        )
        service = VisualFrameRetentionService(
            repository=repository,  # type: ignore[arg-type]
            storage=_Storage(),  # type: ignore[arg-type]
            batch_size=25,
        )

        expired = await service.expire_old_raw_frames()

        self.assertEqual(expired, 0)
        self.assertEqual(
            repository.calls,
            ["reasons:25", "frames:25", f"release:owner-1:{frame_id}"],
        )

    async def test_expiry_query_preserves_every_active_retention_reason(self) -> None:
        connection = _Connection(fetch_results=[[]])
        repository = VisualFrameRepository(_Pool(connection))  # type: ignore[arg-type]

        await repository.claim_expired_raw_frames(limit=10)

        query = connection.calls[0]
        self.assertIn("visual_frame_retention_reasons", query)
        self.assertIn("reasons.expires_at is null", query)
        self.assertIn("or reasons.expires_at > now()", query)
        self.assertIn("raw_image_status = 'expiring'", query)
        self.assertIn("for update of frames skip locked", query)

    async def test_expired_reason_cleanup_is_bounded(self) -> None:
        rows = [{"id": uuid4()}, {"id": uuid4()}]
        connection = _Connection(fetch_results=[rows])
        repository = VisualFrameRepository(_Pool(connection))  # type: ignore[arg-type]

        removed = await repository.delete_expired_retention_reasons(limit=2)

        self.assertEqual(removed, 2)
        self.assertIn("limit $1", connection.calls[0].lower())

    async def test_open_event_candidates_are_recomputed_and_capped_at_twelve(self) -> None:
        event = _event_row()
        candidates = _candidate_rows(16)
        connection = _Connection(
            fetchrow_results=[event],
            fetch_results=[candidates],
            fetchval_results=[16],
        )
        repository = ExperienceRepository(_Pool(connection))  # type: ignore[arg-type]
        selector = DeterministicKeyframeSelector()

        retained = await repository.reconcile_event_candidates(
            owner_id=event["owner_id"],
            session_id=event["session_id"],
            event_id=event["id"],
            observation_id=uuid4(),
            proposed_frame_ids=tuple(row["visual_frame_id"] for row in candidates),
            expires_at=datetime.now(UTC) + timedelta(minutes=30),
            max_count=12,
            selector=lambda values: selector.select(values, limit=12),
        )

        self.assertEqual(len(retained), 12)
        self.assertTrue(any("insert into visual_frame_retention_reasons" in q for q in connection.calls))
        self.assertTrue(any("delete from visual_frame_retention_reasons" in q for q in connection.calls))

    async def test_close_selects_six_candidates_before_marking_event_closed(self) -> None:
        event = _event_row()
        closed = _event_row(
            event_id=event["id"],
            session_id=event["session_id"],
            status="closed",
        )
        connection = _Connection(
            fetchrow_results=[event, closed],
            fetch_results=[[], _candidate_rows(10)],
        )
        repository = ExperienceRepository(_Pool(connection))  # type: ignore[arg-type]
        selector = DeterministicKeyframeSelector()

        result = await repository.close_event_with_keyframes(
            owner_id=event["owner_id"],
            session_id=event["session_id"],
            event_id=event["id"],
            ended_at=event["started_at"] + timedelta(seconds=30),
            boundary_reason="idle_gap",
            finalized_at=datetime.now(UTC),
            candidate_expires_at=datetime.now(UTC) + timedelta(minutes=30),
            max_count=6,
            selector=lambda values: selector.select(values, limit=6),
        )

        self.assertEqual(result.status, "closed")
        keyframe_query = next(
            query for query in connection.calls if "insert into experience_event_keyframes" in query
        )
        close_query = next(
            query for query in connection.calls if "update experience_events" in query
        )
        self.assertLess(connection.calls.index(keyframe_query), connection.calls.index(close_query))

    async def test_close_preserves_existing_and_new_candidate_retention(self) -> None:
        event = _event_row()
        existing_frame_id = uuid4()
        closed = _event_row(
            event_id=event["id"],
            session_id=event["session_id"],
            status="closed",
        )
        connection = _Connection(
            fetchrow_results=[event, closed],
            fetch_results=[
                [
                    {
                        "visual_frame_id": existing_frame_id,
                        "rank": 1,
                    }
                ],
                [{"id": existing_frame_id}],
                _candidate_rows(3),
            ],
        )
        repository = ExperienceRepository(_Pool(connection))  # type: ignore[arg-type]
        selector = DeterministicKeyframeSelector()

        await repository.close_event_with_keyframes(
            owner_id=event["owner_id"],
            session_id=event["session_id"],
            event_id=event["id"],
            ended_at=event["started_at"] + timedelta(seconds=30),
            boundary_reason="idle_gap",
            finalized_at=datetime.now(UTC),
            candidate_expires_at=datetime.now(UTC) + timedelta(minutes=30),
            max_count=6,
            selector=lambda values: selector.select(values, limit=6),
        )

        retention_call = next(
            call
            for call in connection.call_details
            if "insert into visual_frame_retention_reasons" in call[0]
        )
        retained_ids = retention_call[1][4]
        self.assertIn(existing_frame_id, retained_ids)

    async def test_keyframe_insert_failure_happens_before_event_close(self) -> None:
        event = _event_row()
        connection = _Connection(
            fetchrow_results=[event],
            fetch_results=[[], _candidate_rows(2)],
            execute_results=[RuntimeError("keyframe insert failed")],
        )
        repository = ExperienceRepository(_Pool(connection))  # type: ignore[arg-type]
        selector = DeterministicKeyframeSelector()

        with self.assertRaisesRegex(RuntimeError, "keyframe insert failed"):
            await repository.close_event_with_keyframes(
                owner_id=event["owner_id"],
                session_id=event["session_id"],
                event_id=event["id"],
                ended_at=event["started_at"] + timedelta(seconds=10),
                boundary_reason="idle_gap",
                finalized_at=datetime.now(UTC),
                candidate_expires_at=datetime.now(UTC) + timedelta(minutes=30),
                max_count=6,
                selector=lambda values: selector.select(values, limit=6),
            )

        self.assertFalse(any("update experience_events" in query for query in connection.calls))


def _candidate_rows(count: int) -> list[dict[str, Any]]:
    started_at = datetime(2026, 8, 21, 12, 0, tzinfo=UTC)
    return [
        {
            "visual_frame_id": uuid4(),
            "captured_at": started_at + timedelta(seconds=index),
            "reason": "readable label" if index % 3 == 0 else "scene state",
            "analyzer_score": 0.75 + (index % 5) * 0.04,
            "observation_confidence": 0.9,
            "has_readable_text": index % 3 == 0,
            "has_state_change": index % 4 == 0,
            "entity_count": index % 5,
        }
        for index in range(count)
    ]


def _event_row(
    *,
    event_id: object | None = None,
    session_id: object | None = None,
    status: str = "open",
) -> dict[str, Any]:
    now = datetime.now(UTC)
    closed = status != "open"
    return {
        "id": event_id or uuid4(),
        "owner_id": "owner",
        "session_id": session_id or uuid4(),
        "status": status,
        "event_kind": "visual",
        "modalities": ["visual"],
        "started_at": now,
        "ended_at": now if closed else None,
        "title": "Event",
        "summary": "Event summary",
        "scene": "desk",
        "activity": "reading",
        "state": "{}",
        "boundary_reason": "idle_gap" if closed else None,
        "importance": 0.5,
        "confidence": 0.8,
        "keyframes_finalized_at": now if closed else None,
        "consolidation_ready_at": now if closed else None,
        "evidence_reduced_at": None,
        "retention_expires_at": None,
        "consolidation_result": None,
        "consolidation_provider": None,
        "consolidation_model": None,
        "consolidated_at": None,
        "created_at": now,
        "updated_at": now,
    }
