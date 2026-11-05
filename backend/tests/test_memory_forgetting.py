from __future__ import annotations

import unittest
from types import SimpleNamespace
from uuid import uuid4

from app.memory.forgetting import MemoryForgettingService


class _Repository:
    def __init__(self) -> None:
        self.call = None
        self.result = SimpleNamespace(memory=SimpleNamespace(id=uuid4()))

    async def forget_derived_memory(self, **kwargs):
        self.call = kwargs
        return self.result


class MemoryForgettingServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_service_normalizes_reason_and_delegates_once(self) -> None:
        repository = _Repository()
        service = MemoryForgettingService(repository)  # type: ignore[arg-type]
        memory_id = uuid4()

        result = await service.forget(
            owner_id=" owner ",
            memory_id=memory_id,
            reason="  stale belief  ",
        )

        self.assertIs(result, repository.result)
        self.assertEqual(
            repository.call,
            {
                "owner_id": "owner",
                "memory_id": memory_id,
                "reason": "stale belief",
            },
        )

    async def test_default_reason_is_explicit(self) -> None:
        repository = _Repository()
        service = MemoryForgettingService(repository)  # type: ignore[arg-type]
        await service.forget(owner_id="owner", memory_id=uuid4())
        self.assertIn("User requested", repository.call["reason"])


if __name__ == "__main__":
    unittest.main()
