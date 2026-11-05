from __future__ import annotations

import unittest
from datetime import UTC, datetime
from uuid import uuid4

from app.preferences.repository import PreferenceRepository


class _Connection:
    def __init__(self, *, inferred_write, existing=None) -> None:
        self.inferred_write = inferred_write
        self.existing = existing
        self.calls = 0

    async def fetchrow(self, query, *arguments):
        self.calls += 1
        return self.inferred_write if self.calls == 1 else self.existing


class InferredPreferenceReconciliationTests(unittest.IsolatedAsyncioTestCase):
    async def test_inferred_preference_never_overwrites_explicit_value(self) -> None:
        existing = _record(source="assistant", value="coffee")
        connection = _Connection(inferred_write=None, existing=existing)
        repository = PreferenceRepository(pool=None)

        result = await repository.set_inferred_preference(
            owner_id="owner",
            key="preferred_drink",
            value="tea",
            confidence=0.95,
            evidence_episode_ids=(uuid4(), uuid4(), uuid4()),
            connection=connection,
        )

        self.assertFalse(result.applied)
        self.assertEqual(result.preference.value, "coffee")
        self.assertEqual(result.preference.source, "assistant")

    async def test_inferred_preference_can_update_inferred_value(self) -> None:
        written = _record(source="inferred_memory", value="tea")
        connection = _Connection(inferred_write=written)
        repository = PreferenceRepository(pool=None)

        result = await repository.set_inferred_preference(
            owner_id="owner",
            key="preferred_drink",
            value="tea",
            confidence=0.91,
            evidence_episode_ids=written["evidence_episode_ids"],
            connection=connection,
        )

        self.assertTrue(result.applied)
        self.assertEqual(result.preference.value, "tea")
        self.assertEqual(connection.calls, 1)


def _record(*, source: str, value: str) -> dict:
    now = datetime.now(UTC)
    return {
        "id": uuid4(),
        "key": "preferred_drink",
        "value": value,
        "source": source,
        "confidence": 1.0 if source == "assistant" else 0.91,
        "evidence_episode_ids": (
            () if source == "assistant" else (uuid4(), uuid4(), uuid4())
        ),
        "created_at": now,
        "updated_at": now,
    }


if __name__ == "__main__":
    unittest.main()
