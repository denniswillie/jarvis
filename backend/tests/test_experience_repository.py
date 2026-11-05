from __future__ import annotations

import json
import unittest
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid4

import asyncpg

from app.experience.repository import (
    ExperienceConflictError,
    ExperienceEventClosedError,
    ExperienceRepository,
)


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
        fetchrow_results: list[dict[str, Any] | BaseException | None] | None = None,
        fetch_results: list[list[dict[str, Any]]] | None = None,
        fetchval_results: list[object] | None = None,
    ) -> None:
        self.fetchrow_results = list(fetchrow_results or [])
        self.fetch_results = list(fetch_results or [])
        self.fetchval_results = list(fetchval_results or [])
        self.calls: list[tuple[str, str]] = []

    def transaction(self) -> _AsyncContext:
        return _AsyncContext(self)

    async def execute(self, query: str, *_: object) -> str:
        self.calls.append(("execute", query))
        return "SELECT 1"

    async def fetchrow(self, query: str, *_: object) -> dict[str, Any] | None:
        self.calls.append(("fetchrow", query))
        result = self.fetchrow_results.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result

    async def fetch(self, query: str, *_: object) -> list[dict[str, Any]]:
        self.calls.append(("fetch", query))
        return self.fetch_results.pop(0)

    async def fetchval(self, query: str, *_: object) -> object:
        self.calls.append(("fetchval", query))
        return self.fetchval_results.pop(0)


class _Pool:
    def __init__(self, connection: _Connection) -> None:
        self.connection = connection

    def acquire(self) -> _AsyncContext:
        return _AsyncContext(self.connection)


class ExperienceRepositoryTests(unittest.IsolatedAsyncioTestCase):
    async def test_get_or_create_open_event_uses_transaction_lock(self) -> None:
        row = _event_row()
        connection = _Connection(fetchrow_results=[None, row])
        repository = ExperienceRepository(_Pool(connection))  # type: ignore[arg-type]

        created = await repository.get_or_create_open_event(
            owner_id=row["owner_id"],
            session_id=row["session_id"],
            event_kind="visual",
            started_at=row["started_at"],
            modalities=("visual",),
        )

        self.assertEqual(created.id, row["id"])
        self.assertIn("pg_advisory_xact_lock", connection.calls[0][1])
        self.assertIn("for update", connection.calls[1][1].lower())

    async def test_existing_evidence_is_idempotent_after_event_closure(self) -> None:
        event = _event_row(status="closed")
        evidence = _evidence_row(event["id"])
        connection = _Connection(fetchrow_results=[event, evidence])
        repository = ExperienceRepository(_Pool(connection))  # type: ignore[arg-type]

        attached = await repository.attach_evidence(
            owner_id=event["owner_id"],
            session_id=event["session_id"],
            event_id=event["id"],
            evidence_type="message",
            evidence_ref_id=evidence["evidence_ref_id"],
        )

        self.assertEqual(attached.id, evidence["id"])
        self.assertEqual(len(connection.fetchrow_results), 0)

    async def test_aggregate_update_locks_event_and_uses_compare_and_swap(self) -> None:
        event = _event_row()
        updated = dict(event)
        updated["summary"] = "Updated summary"
        updated["updated_at"] = event["updated_at"] + timedelta(milliseconds=1)
        connection = _Connection(fetchrow_results=[event, updated])
        repository = ExperienceRepository(_Pool(connection))  # type: ignore[arg-type]

        result = await repository.update_open_event_aggregate(
            owner_id=event["owner_id"],
            session_id=event["session_id"],
            event_id=event["id"],
            event_kind="visual",
            modalities=("visual",),
            title="Updated event",
            summary="Updated summary",
            scene="desk",
            activity="reading",
            state={"object": "book"},
            importance=0.7,
            confidence=0.9,
            retention_expires_at=None,
            expected_updated_at=event["updated_at"],
        )

        self.assertEqual(result.summary, "Updated summary")
        self.assertIn("for update", connection.calls[0][1].lower())
        self.assertIn("update experience_events", connection.calls[1][1].lower())

    async def test_aggregate_update_rejects_stale_event_state(self) -> None:
        event = _event_row()
        connection = _Connection(fetchrow_results=[event])
        repository = ExperienceRepository(_Pool(connection))  # type: ignore[arg-type]

        with self.assertRaises(ExperienceConflictError):
            await repository.update_open_event_aggregate(
                owner_id=event["owner_id"],
                session_id=event["session_id"],
                event_id=event["id"],
                event_kind="visual",
                modalities=("visual",),
                title="Stale event",
                summary="Stale summary",
                scene=None,
                activity=None,
                state={},
                importance=0.5,
                confidence=0.8,
                retention_expires_at=None,
                expected_updated_at=event["updated_at"] - timedelta(seconds=1),
            )

        self.assertEqual(len(connection.calls), 1)

    async def test_keyframe_selection_rejects_closed_event(self) -> None:
        event = _event_row(status="closed")
        connection = _Connection(fetchrow_results=[event])
        repository = ExperienceRepository(_Pool(connection))  # type: ignore[arg-type]

        with self.assertRaises(ExperienceEventClosedError):
            await repository.select_keyframe(
                owner_id=event["owner_id"],
                session_id=event["session_id"],
                event_id=event["id"],
                visual_frame_id=uuid4(),
                rank=1,
                reason="Late keyframe",
                quality_score=0.8,
                selected_by="deterministic",
            )

        self.assertEqual(len(connection.calls), 1)

    async def test_duplicate_keyframe_rank_is_domain_conflict(self) -> None:
        event = _event_row()
        connection = _Connection(
            fetchrow_results=[event, asyncpg.UniqueViolationError("duplicate rank")]
        )
        repository = ExperienceRepository(_Pool(connection))  # type: ignore[arg-type]

        with self.assertRaises(ExperienceConflictError):
            await repository.select_keyframe(
                owner_id=event["owner_id"],
                session_id=event["session_id"],
                event_id=event["id"],
                visual_frame_id=uuid4(),
                rank=1,
                reason="Competing keyframe",
                quality_score=0.8,
                selected_by="deterministic",
            )

    async def test_evidence_listing_uses_derived_deterministic_order(self) -> None:
        event = _event_row()
        connection = _Connection(fetch_results=[[]])
        repository = ExperienceRepository(_Pool(connection))  # type: ignore[arg-type]

        await repository.list_event_evidence(
            owner_id=event["owner_id"],
            session_id=event["session_id"],
            event_id=event["id"],
        )

        query = connection.calls[0][1]
        self.assertIn("occurred_at asc", query)
        self.assertIn("source_sequence asc nulls last", query)
        self.assertNotIn("ordinal", query)

    async def test_close_event_is_idempotent(self) -> None:
        closed = _event_row(status="closed")
        connection = _Connection(fetchrow_results=[closed])
        repository = ExperienceRepository(_Pool(connection))  # type: ignore[arg-type]

        result = await repository.close_event(
            owner_id=closed["owner_id"],
            session_id=closed["session_id"],
            event_id=closed["id"],
            ended_at=closed["ended_at"],
            boundary_reason="replayed close",
            keyframes_finalized_at=closed["keyframes_finalized_at"],
            consolidation_ready_at=closed["consolidation_ready_at"],
        )

        self.assertEqual(result.status, "closed")
        self.assertEqual(len(connection.fetchrow_results), 0)

    async def test_retention_reason_upsert_preserves_independent_reason(self) -> None:
        now = datetime.now(UTC)
        row = {
            "id": uuid4(),
            "owner_id": "owner",
            "visual_frame_id": uuid4(),
            "reason_type": "event_candidate",
            "reason_ref_id": uuid4(),
            "expires_at": now + timedelta(minutes=30),
            "created_at": now,
        }
        connection = _Connection(fetchrow_results=[row])
        repository = ExperienceRepository(_Pool(connection))  # type: ignore[arg-type]

        result = await repository.add_retention_reason(
            owner_id=row["owner_id"],
            visual_frame_id=row["visual_frame_id"],
            reason_type="event_candidate",
            reason_ref_id=row["reason_ref_id"],
            expires_at=row["expires_at"],
        )

        self.assertEqual(result.id, row["id"])
        self.assertIn(
            "greatest(visual_frame_retention_reasons.expires_at, excluded.expires_at)",
            connection.calls[0][1],
        )


def _event_row(*, status: str = "open") -> dict[str, Any]:
    now = datetime.now(UTC)
    closed = status != "open"
    return {
        "id": uuid4(),
        "owner_id": "owner",
        "session_id": uuid4(),
        "status": status,
        "event_kind": "visual",
        "modalities": ["visual"],
        "started_at": now,
        "ended_at": now if closed else None,
        "title": "",
        "summary": "",
        "scene": None,
        "activity": None,
        "state": "{}",
        "boundary_reason": "boundary" if closed else None,
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


def _evidence_row(event_id: object) -> dict[str, Any]:
    now = datetime.now(UTC)
    return {
        "id": uuid4(),
        "event_id": event_id,
        "evidence_type": "message",
        "evidence_ref_id": uuid4(),
        "occurred_at": now,
        "source_sequence": None,
        "ingested_at": now,
        "metadata": json.dumps({}),
    }
