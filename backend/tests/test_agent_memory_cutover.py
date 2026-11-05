from __future__ import annotations

import unittest
from types import SimpleNamespace
from uuid import uuid4

from app.memory.tools import MemoryToolModule
from app.reminders.delivery import REMINDER_TOOL_NAMES
from app.tools.module import ToolBuildContext
from app.tools.registry import create_tool_registry
from app.visual_memory.tools import VisualCaptureToolModule


class AgentMemoryCutoverTests(unittest.TestCase):
    def setUp(self) -> None:
        self.context = ToolBuildContext(
            settings=SimpleNamespace(
                VISUAL_LATEST_CAPTURE_MAX_AGE_SECONDS=5,
            ),  # type: ignore[arg-type]
            owner_id="owner",
            session_id=uuid4(),
        )

    def test_visual_capture_module_has_no_caption_search_fallback(self) -> None:
        module = VisualCaptureToolModule(
            repository=SimpleNamespace(),  # type: ignore[arg-type]
        )
        self.assertEqual(list(module.build_tools(self.context)), [])

    def test_active_memory_registry_has_one_authoritative_definition_per_tool(self) -> None:
        modules = [
            MemoryToolModule(
                repository=SimpleNamespace(),  # type: ignore[arg-type]
                entity_repository=SimpleNamespace(),  # type: ignore[arg-type]
                service=SimpleNamespace(),  # type: ignore[arg-type]
                forgetting_service=SimpleNamespace(),  # type: ignore[arg-type]
            ),
            VisualCaptureToolModule(
                repository=SimpleNamespace(),  # type: ignore[arg-type]
            ),
        ]
        registry = create_tool_registry(modules=modules, context=self.context)
        names = [tool.name for tool in registry.list()]

        self.assertNotIn("search_visual_memory", names)
        for tool_name in ("search_memory", "remember", "explain_memory", "forget_memory"):
            self.assertEqual(names.count(tool_name), 1)

    def test_reminder_generation_uses_hierarchical_memory_search(self) -> None:
        self.assertIn("search_memory", REMINDER_TOOL_NAMES)
        self.assertNotIn("search_visual_memory", REMINDER_TOOL_NAMES)


if __name__ == "__main__":
    unittest.main()
