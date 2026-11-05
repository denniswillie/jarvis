from __future__ import annotations

import unittest
from uuid import uuid4

from app.memory.forgetting import (
    memory_identity_fingerprint,
    semantic_fact_identity_fingerprint,
    suppress_automatic_memory_recreation,
)


class _Connection:
    def __init__(self, source_memory_id=None) -> None:
        self.source_memory_id = source_memory_id
        self.calls = []

    async def fetchrow(self, query, *args):
        self.calls.append((query, args))
        return (
            {"source_memory_id": self.source_memory_id}
            if self.source_memory_id is not None
            else None
        )


class MemoryResurrectionGuardTests(unittest.IsolatedAsyncioTestCase):
    def test_fingerprint_normalizes_equivalent_content(self) -> None:
        first = memory_identity_fingerprint(
            memory_type="goal",
            content="  Buy   milk\nToday ",
        )
        second = memory_identity_fingerprint(
            memory_type="goal",
            content="buy milk today",
        )
        self.assertEqual(first, second)

    def test_semantic_identity_includes_subject_predicate_and_value(self) -> None:
        first = semantic_fact_identity_fingerprint(
            subject_identity_key="name:person:user",
            predicate="prefers",
            value="Quiet cafes",
        )
        second = semantic_fact_identity_fingerprint(
            subject_identity_key="name:person:user",
            predicate="prefers",
            value="Busy cafes",
        )
        self.assertNotEqual(first, second)

    async def test_guard_audits_and_returns_suppressed_source(self) -> None:
        source_id = uuid4()
        connection = _Connection(source_id)
        result = await suppress_automatic_memory_recreation(
            connection,  # type: ignore[arg-type]
            owner_id="owner",
            memory_type="semantic_fact",
            identity_fingerprint="a" * 64,
        )
        self.assertEqual(result, source_id)
        self.assertIn("suppressed_recreation_count", connection.calls[0][0])


if __name__ == "__main__":
    unittest.main()
