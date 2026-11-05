from __future__ import annotations

import unittest
from datetime import UTC, datetime
from types import SimpleNamespace
from uuid import UUID

from app.memory.service import MemoryApplicationService


NOW = datetime(2026, 8, 28, 12, 0, tzinfo=UTC)


class _Repository:
    def __init__(self) -> None:
        self.calls = []

    async def create_explicit_memory(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace()


class _Sessions:
    async def get_recent_messages(self, _owner_id, session_id, *, limit):
        return [
            SimpleNamespace(
                id=UUID(int=3),
                session_id=session_id,
                role="user",
                content="Remember this walk.",
                created_at=NOW,
            )
        ]


class _Frames:
    def __init__(self) -> None:
        self.sequence = 3

    def latest(self, owner_id, session_id, *, max_age_seconds):
        self.sequence += 1
        return SimpleNamespace(
            frame_id=UUID(int=self.sequence),
            owner_id=owner_id,
            session_id=session_id,
            captured_at=NOW,
            source="glasses",
        )


class _Entities:
    async def find_by_name(self, *, owner_id, name, limit):
        if name == "station":
            return [SimpleNamespace(id=UUID(int=5), owner_id=owner_id)]
        if name == "ambiguous":
            return [SimpleNamespace(id=UUID(int=6)), SimpleNamespace(id=UUID(int=7))]
        return []


class MemoryApplicationServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_explicit_experience_attaches_scoped_evidence_and_is_retry_stable(
        self,
    ) -> None:
        repository = _Repository()
        service = MemoryApplicationService(
            repository=repository,  # type: ignore[arg-type]
            retrieval_service=SimpleNamespace(),  # type: ignore[arg-type]
            session_repository=_Sessions(),  # type: ignore[arg-type]
            frame_window=_Frames(),  # type: ignore[arg-type]
            working_memory_service=None,
            embedding_job_repository=None,
            current_frame_max_age_seconds=3,
            entity_repository=_Entities(),  # type: ignore[arg-type]
        )
        arguments = {
            "owner_id": "owner",
            "session_id": UUID(int=2),
            "kind": "experience",
            "content": "I walked past the blue station poster.",
            "occurred_at": None,
            "entity_references": ("station",),
            "attach_triggering_message": True,
            "attach_current_evidence": True,
            "supersedes_memory_id": None,
            "operation_id": "provider-call-1",
        }

        await service.remember(**arguments)
        await service.remember(**arguments)

        first, second = repository.calls
        self.assertEqual(first["memory_type"], "reflection")
        self.assertEqual(
            [item.evidence_type for item in first["evidence"]],
            ["message", "frame"],
        )
        self.assertEqual(first["idempotency_key"], second["idempotency_key"])
        self.assertEqual(first["metadata"]["entity_references"], ["station"])
        self.assertEqual(first["entity_ids"], (UUID(int=5),))

    async def test_fallback_idempotency_ignores_changing_current_frame(self) -> None:
        repository = _Repository()
        service = MemoryApplicationService(
            repository=repository,  # type: ignore[arg-type]
            retrieval_service=SimpleNamespace(),  # type: ignore[arg-type]
            session_repository=_Sessions(),  # type: ignore[arg-type]
            frame_window=_Frames(),  # type: ignore[arg-type]
            working_memory_service=None,
            embedding_job_repository=None,
            current_frame_max_age_seconds=3,
            entity_repository=_Entities(),  # type: ignore[arg-type]
        )
        arguments = {
            "owner_id": "owner",
            "session_id": UUID(int=2),
            "kind": "fact",
            "content": "The station poster is blue.",
            "occurred_at": NOW,
            "entity_references": ("ambiguous", "missing"),
            "attach_triggering_message": True,
            "attach_current_evidence": True,
            "supersedes_memory_id": None,
        }

        await service.remember(**arguments)
        await service.remember(**arguments)

        first, second = repository.calls
        self.assertNotEqual(
            first["evidence"][-1].evidence_ref_id,
            second["evidence"][-1].evidence_ref_id,
        )
        self.assertEqual(first["idempotency_key"], second["idempotency_key"])
        self.assertIn("entity_reference_ambiguous:ambiguous", first["warnings"])
        self.assertIn("entity_reference_unresolved:missing", first["warnings"])


if __name__ == "__main__":
    unittest.main()
