from __future__ import annotations

import unittest
from datetime import UTC, datetime
from uuid import uuid4

from app.memory.models import (
    EMBEDDING_DIMENSIONS,
    CalendarEventSnapshot,
    MemoryEmbedding,
    MemoryItem,
    content_hash,
    json_object_hash,
    normalize_embedding,
)


class MemoryModelTests(unittest.TestCase):
    def test_episode_requires_source_event(self) -> None:
        with self.assertRaisesRegex(ValueError, "source experience event"):
            MemoryItem(**_memory_values(memory_type="episode", source_event_id=None))

    def test_forgotten_episode_preserves_source_identity(self) -> None:
        source_id = uuid4()
        memory = MemoryItem(
            **_memory_values(
                memory_type="episode",
                status="forgotten",
                source_event_id=source_id,
            )
        )

        self.assertEqual(memory.source_experience_event_id, source_id)
        self.assertIsNone(memory.superseded_by_memory_id)

    def test_superseded_memory_requires_a_distinct_replacement(self) -> None:
        values = _memory_values(status="superseded")
        values["superseded_by_memory_id"] = values["id"]

        with self.assertRaisesRegex(ValueError, "supersede itself"):
            MemoryItem(**values)

    def test_superseded_memory_requires_a_validity_end(self) -> None:
        values = _memory_values(status="superseded")
        values["valid_to"] = None

        with self.assertRaisesRegex(ValueError, "validity end"):
            MemoryItem(**values)

    def test_embedding_requires_exact_finite_dimensions_and_matching_hash(self) -> None:
        vector = tuple(0.25 for _ in range(EMBEDDING_DIMENSIONS))
        now = datetime.now(UTC)
        embedding = MemoryEmbedding(
            subject_id=uuid4(),
            embedding_kind="semantic",
            provider="openai",
            model="text-embedding-3-small",
            dimensions=EMBEDDING_DIMENSIONS,
            embedded_content="A remembered fact",
            content_hash=content_hash("A remembered fact"),
            embedding=vector,
            embedded_at=now,
            updated_at=now,
        )

        self.assertEqual(embedding.embedding, vector)
        with self.assertRaisesRegex(ValueError, "1536"):
            normalize_embedding((0.1, 0.2))
        with self.assertRaisesRegex(ValueError, "non-finite"):
            normalize_embedding((0.1,) * (EMBEDDING_DIMENSIONS - 1) + (float("nan"),))

    def test_calendar_snapshot_requires_matching_immutable_content_hash(self) -> None:
        now = datetime.now(UTC)
        snapshot = {"summary": "Dentist", "status": "confirmed"}
        model = CalendarEventSnapshot(
            id=uuid4(),
            owner_id="owner",
            provider="google",
            calendar_id="primary",
            provider_event_id="event-1",
            event_title="Dentist",
            starts_at=now,
            ends_at=now,
            observed_at=now,
            snapshot=snapshot,
            content_hash=json_object_hash(snapshot),
            created_at=now,
        )

        self.assertEqual(model.snapshot, snapshot)


def _memory_values(
    *,
    memory_type: str = "semantic_fact",
    status: str = "active",
    source_event_id: object = "default",
) -> dict[str, object]:
    now = datetime.now(UTC)
    source = uuid4() if source_event_id == "default" and memory_type == "episode" else source_event_id
    replacement = uuid4() if status == "superseded" else None
    return {
        "id": uuid4(),
        "owner_id": "owner",
        "session_id": uuid4(),
        "memory_type": memory_type,
        "status": status,
        "title": "Memory title",
        "content": "Memory content",
        "summary": "Memory summary",
        "source_experience_event_id": source,
        "occurred_at": now,
        "valid_from": now,
        "valid_to": now if status == "superseded" else None,
        "importance": 0.8,
        "confidence": 0.9,
        "source": "test",
        "is_explicit": False,
        "superseded_by_memory_id": replacement,
        "metadata": {},
        "created_at": now,
        "updated_at": now,
    }
