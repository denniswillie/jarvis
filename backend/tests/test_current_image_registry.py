from __future__ import annotations

import json
import unittest
from datetime import UTC, datetime
from unittest.mock import AsyncMock
from uuid import uuid4

from app.images import ImageReference, ResolvedImage
from app.tools.executor import JarvisToolExecutor
from app.tools.module import FunctionToolModule, ToolBuildContext
from app.tools.registry import JarvisToolRegistry, ToolDefinition, create_tool_registry


async def _ordinary_handler(arguments):
    return {"status": "success", "value": arguments.get("value")}


class ImageToolRegistryTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.reference = ImageReference(
            frame_id=uuid4(), owner_id="owner", session_id=uuid4(),
            source="historical", captured_at=datetime(2026, 9, 9, tzinfo=UTC),
            content_type="image/jpeg", width=48, height=32,
        )
        self.resolver = AsyncMock(return_value=ResolvedImage(
            reference=self.reference, image_bytes=b"private-pixel-payload",
        ))
        self.selected = {
            "status": "success", "image_status": "selected",
            "image": self.reference.to_metadata(),
        }
        self.handler = AsyncMock(return_value=self.selected)
        self.tool = ToolDefinition(
            name="fake_image_capability", description="Select an image.",
            input_schema={"type": "object", "properties": {}, "additionalProperties": False},
            handler=self.handler, image_resolver=self.resolver,
        )
        self.registry = JarvisToolRegistry()
        self.registry.register(self.tool)

    async def test_optional_resolver_remains_server_only(self) -> None:
        self.assertIs(self.registry.get(self.tool.name).image_resolver, self.resolver)
        schema = self.registry.openai_realtime_tools()
        self.assertEqual(schema, [{
            "type": "function", "name": self.tool.name,
            "description": self.tool.description, "parameters": self.tool.input_schema,
        }])
        self.assertNotIn("image_resolver", json.dumps(schema))

    async def test_ordinary_tool_needs_no_resolver_or_schema_change(self) -> None:
        ordinary = ToolDefinition(
            name="ordinary", description="An ordinary tool.",
            input_schema={"type": "object", "properties": {}}, handler=_ordinary_handler,
        )
        self.registry.register(ordinary)
        self.assertIsNone(ordinary.image_resolver)
        result = await JarvisToolExecutor(self.registry).execute("ordinary", {"value": 7})
        self.assertEqual(result, {"status": "success", "value": 7})

    async def test_executor_keeps_image_selection_as_metadata_without_resolving(self) -> None:
        result = await JarvisToolExecutor(self.registry).execute(self.tool.name, {})
        self.assertEqual(result, self.selected)
        self.assertEqual(result["image_status"], "selected")
        self.handler.assert_awaited_once_with({})
        self.resolver.assert_not_awaited()
        self.assertNotIn("private-pixel-payload", json.dumps(result))

    async def test_resolver_can_be_invoked_separately_by_a_transport(self) -> None:
        result = await JarvisToolExecutor(self.registry).execute(self.tool.name, {})
        resolved = await self.registry.get(self.tool.name).image_resolver(result)
        self.assertEqual(resolved.reference, self.reference)
        self.assertEqual(resolved.image_bytes, b"private-pixel-payload")
        self.resolver.assert_awaited_once_with(result)
        self.assertEqual(result["image_status"], "selected")

    async def test_capability_module_registers_without_central_special_cases(self) -> None:
        context = ToolBuildContext(settings=None, owner_id="owner", session_id=uuid4())
        registry = create_tool_registry(
            modules=[FunctionToolModule(name="fake-images", factory=lambda _: [self.tool])],
            context=context,
        )
        self.assertEqual(registry.list(), [self.tool])
        self.assertIs(registry.get(self.tool.name).image_resolver, self.resolver)

    async def test_positional_mutates_state_argument_remains_compatible(self) -> None:
        tool = ToolDefinition("mutation", "An ordinary mutation.", {}, _ordinary_handler, True)
        self.assertTrue(tool.mutates_state)
        self.assertIsNone(tool.image_resolver)


if __name__ == "__main__":
    unittest.main()
