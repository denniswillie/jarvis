from __future__ import annotations

import unittest
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from uuid import uuid4

from app.assistant.context_providers import ContextBuildRequest
from app.perception.context import WorkingMemoryContextProvider


NOW = datetime(2026, 8, 16, 12, 0, tzinfo=UTC)


class _Service:
    async def get_session_working_memory(self, *, owner_id, session_id):
        return SimpleNamespace(
            revision=3,
            state={"scene": {"label": "kitchen"}, "entities": {}},
            last_observation_ended_at=NOW - timedelta(seconds=1),
            expires_at=NOW + timedelta(minutes=2),
            updated_at=NOW - timedelta(seconds=1),
        )

    async def get_owner_working_memory(self, *, owner_id):
        return SimpleNamespace(
            revision=2,
            state={"active_goal_ids": ["make-coffee"]},
            expires_at=None,
            updated_at=NOW - timedelta(seconds=2),
        )


class WorkingMemoryContextTests(unittest.IsolatedAsyncioTestCase):
    async def test_provider_returns_revisioned_state(self) -> None:
        request = ContextBuildRequest(
            owner_id="owner", session=SimpleNamespace(id=uuid4()),
            recent_messages=(), requested_at_utc=NOW,
        )
        section = await WorkingMemoryContextProvider(_Service()).build_section(request)
        self.assertEqual(section.key, "working_memory")
        self.assertTrue(section.payload["available"])
        self.assertEqual(section.payload["session"]["revision"], 3)
        self.assertEqual(section.payload["owner"]["revision"], 2)
        self.assertEqual(
            section.payload["session"]["state"]["scene"]["label"],
            "kitchen",
        )
        self.assertEqual(section.freshness, NOW - timedelta(seconds=1))

    async def test_missing_session_returns_empty_fallback(self) -> None:
        request = ContextBuildRequest(
            owner_id="owner", session=None, recent_messages=(), requested_at_utc=NOW
        )
        section = await WorkingMemoryContextProvider(_Service()).build_section(request)
        self.assertEqual(section.size, 0)
        self.assertFalse(section.payload["available"])
        self.assertIsNone(section.payload["session"])


if __name__ == "__main__":
    unittest.main()
