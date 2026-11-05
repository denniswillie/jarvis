from __future__ import annotations

import unittest
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from app.memory.models import EMBEDDING_DIMENSIONS, content_hash
from app.memory.repository import ExperiencePromotionConflictError, MemoryRepository


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
        fetchval_results: list[object] | None = None,
    ) -> None:
        self.fetchrow_results = list(fetchrow_results or [])
        self.fetchval_results = list(fetchval_results or [])
        self.calls: list[tuple[str, str, tuple[object, ...]]] = []
        self.transaction_count = 0

    def transaction(self) -> _AsyncContext:
        self.transaction_count += 1
        return _AsyncContext(self)

    async def execute(self, query: str, *args: object) -> str:
        self.calls.append(("execute", query, args))
        return "UPDATE 1"

    async def fetchrow(self, query: str, *args: object) -> dict[str, Any] | None:
        self.calls.append(("fetchrow", query, args))
        return self.fetchrow_results.pop(0)

    async def fetchval(self, query: str, *args: object) -> object:
        self.calls.append(("fetchval", query, args))
        return self.fetchval_results.pop(0)


class _Pool:
    def __init__(self, connection: _Connection) -> None:
        self.connection = connection

    def acquire(self) -> _AsyncContext:
        return _AsyncContext(self.connection)


class MemoryRepositoryTests(unittest.IsolatedAsyncioTestCase):
    async def test_episode_promotion_locks_event_and_updates_durable_marker(self) -> None:
        event = _event_row()
        memory = _memory_row(event=event)
        connection = _Connection(fetchrow_results=[event, None, memory], fetchval_results=[True])
        repository = MemoryRepository(_Pool(connection))  # type: ignore[arg-type]

        result = await repository.promote_event_to_episode(
            owner_id="owner",
            event_id=event["id"],
            title=memory["title"],
            content=memory["content"],
            summary=memory["summary"],
            importance=memory["importance"],
            confidence=memory["confidence"],
            source=memory["source"],
            metadata=memory["metadata"],
            promotion_reason="durable event",
        )

        self.assertEqual(result.id, memory["id"])
        self.assertEqual(connection.transaction_count, 1)
        self.assertIn("for update", connection.calls[0][1].lower())
        self.assertIn("promotion_decision = 'promoted'", connection.calls[-1][1])

    async def test_forgotten_episode_is_returned_and_never_recreated(self) -> None:
        event = _event_row(promotion_decision="promoted")
        memory = _memory_row(event=event, status="forgotten")
        event["promoted_episode_memory_id"] = memory["id"]
        event["promotion_reason"] = "durable event"
        connection = _Connection(fetchrow_results=[event, memory], fetchval_results=[True])
        repository = MemoryRepository(_Pool(connection))  # type: ignore[arg-type]

        result = await repository.promote_event_to_episode(
            owner_id="owner",
            event_id=event["id"],
            title=memory["title"],
            content=memory["content"],
            summary=memory["summary"],
            importance=memory["importance"],
            confidence=memory["confidence"],
            source=memory["source"],
            metadata=memory["metadata"],
            promotion_reason="durable event",
        )

        self.assertEqual(result.status, "forgotten")
        self.assertFalse(
            any("insert into memory_items" in query.lower() for _, query, _ in connection.calls)
        )

    async def test_evidence_is_owner_checked_and_idempotent(self) -> None:
        memory = _memory_row()
        evidence = _evidence_row(memory["id"])
        connection = _Connection(
            fetchrow_results=[memory, None, evidence],
            fetchval_results=[True],
        )
        repository = MemoryRepository(_Pool(connection))  # type: ignore[arg-type]

        result = await repository.attach_evidence(
            owner_id="owner",
            memory_item_id=memory["id"],
            evidence_type="message",
            evidence_ref_id=evidence["evidence_ref_id"],
            evidence_time=evidence["evidence_time"],
            excerpt=evidence["excerpt"],
            confidence=evidence["confidence"],
            metadata=evidence["metadata"],
        )

        self.assertEqual(result.id, evidence["id"])
        source_query = next(query for kind, query, _ in connection.calls if kind == "fetchval")
        self.assertIn("jarvis_messages", source_query)
        insert_query = next(
            query for kind, query, _ in connection.calls
            if kind == "fetchrow" and "insert into memory_evidence" in query
        )
        self.assertIn("on conflict", insert_query.lower())

    async def test_active_memory_link_uses_partial_idempotency_key(self) -> None:
        source = _memory_row()
        target = _memory_row()
        link = _link_row(source["id"], target["id"])
        connection = _Connection(fetchrow_results=[source, target, link])
        repository = MemoryRepository(_Pool(connection))  # type: ignore[arg-type]

        result = await repository.create_link(
            owner_id="owner",
            source_memory_id=source["id"],
            target_memory_id=target["id"],
            relationship=link["relationship"],
            confidence=link["confidence"],
            valid_from=link["valid_from"],
        )

        self.assertEqual(result.id, link["id"])
        query = connection.calls[-1][1].lower()
        self.assertIn("where valid_to is null", query)

    async def test_embedding_upsert_is_namespaced_and_hash_guarded(self) -> None:
        event = _event_row()
        vector = tuple(0.01 for _ in range(EMBEDDING_DIMENSIONS))
        embedding = _embedding_row(event["id"], vector)
        connection = _Connection(fetchrow_results=[event, None, embedding])
        repository = MemoryRepository(_Pool(connection))  # type: ignore[arg-type]

        result = await repository.upsert_event_embedding(
            owner_id="owner",
            event_id=event["id"],
            embedding_kind=embedding["embedding_kind"],
            provider=embedding["provider"],
            model=embedding["model"],
            embedded_content=embedding["embedded_content"],
            embedding=vector,
        )

        self.assertEqual(result.content_hash, embedding["content_hash"])
        upsert_query = connection.calls[1][1].lower()
        self.assertIn("embedding_kind, provider, model", upsert_query)
        self.assertIn("content_hash is distinct from excluded.content_hash", upsert_query)
        self.assertIn("embedded_at", upsert_query)

    async def test_event_embedding_lookup_is_owner_scoped(self) -> None:
        event = _event_row()
        vector = tuple(0.01 for _ in range(EMBEDDING_DIMENSIONS))
        embedding = _embedding_row(event["id"], vector)
        connection = _Connection(fetchrow_results=[embedding])
        repository = MemoryRepository(_Pool(connection))  # type: ignore[arg-type]

        result = await repository.get_event_embedding(
            owner_id="owner",
            event_id=event["id"],
            embedding_kind=embedding["embedding_kind"],
            provider=embedding["provider"],
            model=embedding["model"],
        )

        self.assertIsNotNone(result)
        query = connection.calls[0][1].lower()
        self.assertIn("join experience_events", query)
        self.assertIn("event.owner_id = $1", query)

    async def test_missing_event_embedding_returns_none(self) -> None:
        connection = _Connection(fetchrow_results=[None])
        repository = MemoryRepository(_Pool(connection))  # type: ignore[arg-type]

        result = await repository.get_event_embedding(
            owner_id="owner",
            event_id=uuid4(),
            embedding_kind="event_text",
            provider="gemini",
            model="gemini-embedding-2",
        )

        self.assertIsNone(result)

    async def test_calendar_snapshot_is_owner_scoped_before_evidence_attachment(self) -> None:
        snapshot = _calendar_snapshot_row()
        connection = _Connection(fetchrow_results=[snapshot])
        repository = MemoryRepository(_Pool(connection))  # type: ignore[arg-type]

        result = await repository.record_calendar_event_snapshot(
            owner_id="owner",
            provider="google",
            calendar_id="primary",
            provider_event_id="event-1",
            event_title="Dentist",
            starts_at=snapshot["starts_at"],
            ends_at=snapshot["ends_at"],
            observed_at=snapshot["observed_at"],
            snapshot=snapshot["snapshot"],
        )

        self.assertEqual(result.id, snapshot["id"])
        self.assertIn("insert into calendar_event_snapshots", connection.calls[0][1])

    async def test_promotion_replay_rejects_a_changed_reason(self) -> None:
        event = _event_row(promotion_decision="promoted")
        memory = _memory_row(event=event)
        event["promoted_episode_memory_id"] = memory["id"]
        event["promotion_reason"] = "original reason"
        connection = _Connection(fetchrow_results=[event, memory])
        repository = MemoryRepository(_Pool(connection))  # type: ignore[arg-type]

        with self.assertRaisesRegex(
            ExperiencePromotionConflictError,
            "different reason",
        ):
            await repository.promote_event_to_episode(
                owner_id="owner",
                event_id=event["id"],
                title=memory["title"],
                content=memory["content"],
                summary=memory["summary"],
                importance=memory["importance"],
                confidence=memory["confidence"],
                source=memory["source"],
                metadata=memory["metadata"],
                promotion_reason="changed reason",
            )

    async def test_semantic_reconciliation_marker_is_stored_on_event(self) -> None:
        event = _event_row()
        connection = _Connection(fetchval_results=[True])
        repository = MemoryRepository(_Pool(connection))  # type: ignore[arg-type]

        await repository.mark_semantic_reconciled(
            owner_id="owner",
            event_id=event["id"],
            reconciled_at=datetime.now(UTC),
        )

        self.assertIn("semantic_reconciled_at", connection.calls[0][1])


def _event_row(*, promotion_decision: str = "pending") -> dict[str, Any]:
    now = datetime.now(UTC)
    return {
        "id": uuid4(),
        "owner_id": "owner",
        "session_id": uuid4(),
        "status": "closed",
        "started_at": now,
        "ended_at": now,
        "promotion_decision": promotion_decision,
        "promotion_reason": None,
        "promoted_episode_memory_id": None,
    }


def _memory_row(
    *,
    event: dict[str, Any] | None = None,
    status: str = "active",
) -> dict[str, Any]:
    now = datetime.now(UTC)
    event = event or _event_row()
    return {
        "id": uuid4(),
        "owner_id": event["owner_id"],
        "session_id": event["session_id"],
        "memory_type": "episode",
        "status": status,
        "title": "Walk through the park",
        "content": "The user walked through the park.",
        "summary": "Park walk",
        "source_experience_event_id": event["id"],
        "occurred_at": now,
        "valid_from": now,
        "valid_to": None,
        "importance": 0.8,
        "confidence": 0.9,
        "source": "event_consolidation",
        "is_explicit": False,
        "superseded_by_memory_id": None,
        "metadata": {},
        "created_at": now,
        "updated_at": now,
    }


def _evidence_row(memory_id: object) -> dict[str, Any]:
    now = datetime.now(UTC)
    return {
        "id": uuid4(),
        "owner_id": "owner",
        "memory_item_id": memory_id,
        "evidence_type": "message",
        "evidence_ref_id": str(uuid4()),
        "evidence_time": now,
        "excerpt": "A supporting message",
        "confidence": 0.9,
        "metadata": {},
        "created_at": now,
    }


def _link_row(source_id: object, target_id: object) -> dict[str, Any]:
    now = datetime.now(UTC)
    return {
        "id": uuid4(),
        "owner_id": "owner",
        "source_memory_id": source_id,
        "target_memory_id": target_id,
        "relationship": "related_to",
        "confidence": 0.8,
        "valid_from": now,
        "valid_to": None,
        "supporting_memory_id": None,
        "metadata": {},
        "created_at": now,
        "updated_at": now,
    }


def _embedding_row(subject_id: object, vector: tuple[float, ...]) -> dict[str, Any]:
    now = datetime.now(UTC)
    content = "A consolidated experience event"
    return {
        "experience_event_id": subject_id,
        "embedding_kind": "semantic",
        "provider": "openai",
        "model": "text-embedding-3-small",
        "dimensions": EMBEDDING_DIMENSIONS,
        "embedded_content": content,
        "content_hash": content_hash(content),
        "embedding": vector,
        "embedded_at": now,
        "updated_at": now,
    }


def _calendar_snapshot_row() -> dict[str, Any]:
    now = datetime.now(UTC)
    snapshot = {"summary": "Dentist", "status": "confirmed"}
    return {
        "id": uuid4(),
        "owner_id": "owner",
        "provider": "google",
        "calendar_id": "primary",
        "provider_event_id": "event-1",
        "event_title": "Dentist",
        "starts_at": now,
        "ends_at": now,
        "observed_at": now,
        "snapshot": snapshot,
        "content_hash": content_hash(
            '{"status":"confirmed","summary":"Dentist"}'
        ),
        "created_at": now,
    }
