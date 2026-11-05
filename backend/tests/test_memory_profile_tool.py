from __future__ import annotations

import unittest
from datetime import UTC, datetime
from types import SimpleNamespace

from app.memory.projection import MemoryProfile
from app.memory.tools import create_memory_tools


NOW = datetime(2026, 8, 30, 12, 0, tzinfo=UTC)


class _ProjectionService:
    def __init__(self) -> None:
        self.owners: list[str] = []

    async def get_profile(self, *, owner_id: str) -> MemoryProfile:
        self.owners.append(owner_id)
        return MemoryProfile(
            owner_id=owner_id,
            markdown="# Jarvis Memory Profile\n",
            content_hash="b" * 64,
            generated_at=NOW,
            source_revision=4,
            projected_revision=4,
        )


class MemoryProfileToolTests(unittest.IsolatedAsyncioTestCase):
    async def test_tool_uses_shared_projection_service_and_owner_scope(self) -> None:
        service = _ProjectionService()
        tools = create_memory_tools(
            repository=SimpleNamespace(),  # type: ignore[arg-type]
            entity_repository=SimpleNamespace(),  # type: ignore[arg-type]
            owner_id="owner-a",
            projection_service=service,  # type: ignore[arg-type]
        )
        profile_tools = [tool for tool in tools if tool.name == "get_memory_profile"]

        self.assertEqual(len(profile_tools), 1)
        self.assertEqual(profile_tools[0].input_schema["properties"], {})
        result = await profile_tools[0].handler({})

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["profile"]["source_revision"], 4)
        self.assertEqual(service.owners, ["owner-a"])


if __name__ == "__main__":
    unittest.main()

