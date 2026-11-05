from __future__ import annotations

import unittest
from types import SimpleNamespace
from uuid import uuid4

from app.memory.tools import MemoryToolModule
from app.tools.module import ToolBuildContext
from app.tools.registry import create_tool_registry


class MemoryToolRegistryTests(unittest.TestCase):
    def test_memory_module_registers_once_through_generic_registry(self) -> None:
        module = MemoryToolModule(
            repository=SimpleNamespace(),  # type: ignore[arg-type]
            entity_repository=SimpleNamespace(),  # type: ignore[arg-type]
            service=SimpleNamespace(),  # type: ignore[arg-type]
            forgetting_service=SimpleNamespace(),  # type: ignore[arg-type]
            projection_service=SimpleNamespace(),  # type: ignore[arg-type]
            image_service=SimpleNamespace(),  # type: ignore[arg-type]
        )
        registry = create_tool_registry(
            modules=[module],
            context=ToolBuildContext(
                settings=None,
                owner_id="owner",
                session_id=uuid4(),
            ),
        )
        names = [tool.name for tool in registry.list()]
        self.assertEqual(len(names), len(set(names)))
        for tool_name in (
            "search_memory",
            "remember",
            "explain_memory",
            "forget_memory",
            "get_memory_profile",
            "get_memory_image",
        ):
            self.assertEqual(names.count(tool_name), 1)
        self.assertIsNotNone(registry.get("get_memory_image").image_resolver)


if __name__ == "__main__":
    unittest.main()
