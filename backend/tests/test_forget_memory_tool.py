from __future__ import annotations

import unittest
from types import SimpleNamespace
from uuid import uuid4

from app.memory.repository import MemoryMissingError
from app.memory.tools import MemoryToolModule
from app.tools.module import ToolBuildContext


class _ForgettingService:
    def __init__(self, *, missing: bool = False) -> None:
        self.missing = missing
        self.calls: list[dict[str, object]] = []

    async def forget(self, **kwargs):
        self.calls.append(kwargs)
        if self.missing:
            raise MemoryMissingError()
        return SimpleNamespace(
            memory=SimpleNamespace(id=kwargs["memory_id"]),
            replayed=False,
        )


class ForgetMemoryToolTests(unittest.IsolatedAsyncioTestCase):
    def _tool(self, service: _ForgettingService):
        module = MemoryToolModule(
            repository=SimpleNamespace(),  # type: ignore[arg-type]
            entity_repository=SimpleNamespace(),  # type: ignore[arg-type]
            service=SimpleNamespace(),  # type: ignore[arg-type]
            forgetting_service=service,  # type: ignore[arg-type]
        )
        return next(
            tool
            for tool in module.build_tools(
                ToolBuildContext(settings=None, owner_id="owner", session_id=None)
            )
            if tool.name == "forget_memory"
        )

    async def test_forget_is_owner_scoped_and_explains_preservation(self) -> None:
        service = _ForgettingService()
        tool = self._tool(service)
        memory_id = uuid4()

        result = await tool.handler(
            {"memory_id": str(memory_id), "reason": "No longer useful."}
        )

        self.assertTrue(tool.mutates_state)
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["memory_status"], "forgotten")
        self.assertEqual(service.calls[0]["owner_id"], "owner")
        self.assertEqual(service.calls[0]["memory_id"], memory_id)
        self.assertIn("original messages", result["preserved"])
        self.assertIn("not physically delete", result["privacy_deletion"])

    async def test_cross_owner_and_unknown_ids_share_not_found_result(self) -> None:
        result = await self._tool(_ForgettingService(missing=True)).handler(
            {"memory_id": str(uuid4())}
        )
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["error"]["code"], "memory_not_found")


if __name__ == "__main__":
    unittest.main()
