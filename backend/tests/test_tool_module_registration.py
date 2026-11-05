from __future__ import annotations

import asyncio
import unittest
from uuid import uuid4

from app.tools.executor import JarvisToolExecutor
from app.tools.module import FunctionToolModule, ToolBuildContext
from app.tools.registry import (
    JarvisToolRegistry,
    ToolDefinition,
    create_tool_registry,
)


async def _success(_: dict[str, object]) -> dict[str, object]:
    return {"status": "success"}


def _tool(name: str) -> ToolDefinition:
    return ToolDefinition(
        name=name,
        description=f"{name} description",
        input_schema={"type": "object", "properties": {}},
        handler=_success,
    )


class ToolModuleRegistrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.context = ToolBuildContext(
            settings=None,
            owner_id="owner",
            session_id=uuid4(),
        )

    def test_duplicate_tool_names_are_rejected(self) -> None:
        registry = JarvisToolRegistry()
        registry.register(_tool("duplicate"))
        with self.assertRaisesRegex(ValueError, "already registered"):
            registry.register(_tool("duplicate"))

    def test_fake_module_registers_without_central_signature_change(self) -> None:
        module = FunctionToolModule(
            name="fake-capability",
            factory=lambda context: [_tool(f"tool_for_{context.owner_id}")],
        )
        registry = create_tool_registry(modules=[module], context=self.context)
        self.assertEqual([tool.name for tool in registry.list()], ["tool_for_owner"])

    def test_registration_order_and_openai_schema_are_deterministic(self) -> None:
        modules = [
            FunctionToolModule(name="first", factory=lambda _: [_tool("one")]),
            FunctionToolModule(name="second", factory=lambda _: [_tool("two")]),
        ]
        registry = create_tool_registry(modules=modules, context=self.context)
        self.assertEqual(
            registry.openai_realtime_tools(),
            [
                {
                    "type": "function",
                    "name": "one",
                    "description": "one description",
                    "parameters": {"type": "object", "properties": {}},
                },
                {
                    "type": "function",
                    "name": "two",
                    "description": "two description",
                    "parameters": {"type": "object", "properties": {}},
                },
            ],
        )

    def test_executor_can_be_created_from_modules(self) -> None:
        executor = JarvisToolExecutor.from_modules(
            modules=[FunctionToolModule(name="fake", factory=lambda _: [_tool("ping")])],
            context=self.context,
        )
        self.assertEqual(asyncio.run(executor.execute("ping", {})), {"status": "success"})


if __name__ == "__main__":
    unittest.main()
