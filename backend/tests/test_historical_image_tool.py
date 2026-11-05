from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

from app.images import ImageUnavailableError
from app.memory.retrieval.models import RETRIEVAL_RECORD_TYPES
from app.memory.tools import MemoryToolModule
from app.tools.module import ToolBuildContext
from test_current_image_service import _jpeg
from test_historical_image_service import _evidence, _service


class HistoricalImageToolTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.evidence = _evidence()
        self.repository = SimpleNamespace(get_image_evidence=AsyncMock(return_value=self.evidence))
        self.loader = SimpleNamespace(download_object=AsyncMock(return_value=_jpeg()))
        module = MemoryToolModule(
            repository=SimpleNamespace(), entity_repository=SimpleNamespace(),
            image_service=_service(self.repository, self.loader),
        )
        self.tool = next(tool for tool in module.build_tools(
            ToolBuildContext(settings=None, owner_id="owner", session_id=uuid4()),
        ) if tool.name == "get_memory_image")
        self.arguments = {
            "record_id": str(self.evidence.reference.record_id),
            "record_type": self.evidence.reference.record_type,
        }

    async def test_schema_is_record_based_and_model_free(self):
        schema = self.tool.input_schema
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(set(schema["properties"]), {"record_id", "record_type", "frame_id"})
        self.assertEqual(set(schema["required"]), {"record_id", "record_type"})
        self.assertEqual(set(schema["properties"]["record_type"]["enum"]), RETRIEVAL_RECORD_TYPES)
        self.assertFalse(self.tool.mutates_state)
        self.assertIsNotNone(self.tool.image_resolver)

    async def test_result_is_selected_not_attached_and_contains_no_pixels_or_analysis(self):
        result = await self.tool.handler(self.arguments)
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["image_status"], "selected")
        self.assertEqual(result["image"], self.evidence.reference.to_metadata())
        self.assertEqual(result["image"]["source"], "historical")
        for key in ("description", "confidence", "provider_item_id", "image_bytes", "owner_id"):
            self.assertNotIn(key, json.dumps(result))
        self.loader.download_object.assert_not_awaited()

    async def test_resolver_restores_exact_durable_selection_with_trusted_owner(self):
        selection = json.loads(json.dumps(await self.tool.handler(self.arguments)))
        selection["image"]["owner_id"] = "untrusted"
        image = await self.tool.image_resolver(selection)
        self.assertEqual(image.reference, self.evidence.reference)
        self.assertEqual(image.image_bytes, _jpeg())
        self.assertEqual(self.repository.get_image_evidence.await_args.kwargs["owner_id"], "owner")
        self.assertEqual(self.repository.get_image_evidence.await_args.kwargs["frame_id"], self.evidence.reference.frame_id)

    async def test_explicit_frame_is_passed_without_silent_substitution(self):
        await self.tool.handler({**self.arguments, "frame_id": str(self.evidence.reference.frame_id)})
        self.assertEqual(self.repository.get_image_evidence.await_args.kwargs["frame_id"], self.evidence.reference.frame_id)
        result = await self.tool.handler({**self.arguments, "frame_id": str(uuid4())})
        self.assertEqual(result["image_status"], "unavailable")
        self.loader.download_object.assert_not_awaited()

    async def test_unknown_arguments_cannot_select_owner_storage_or_model(self):
        for key in ("owner_id", "session_id", "bucket", "path", "url", "question", "reinspect_evidence"):
            with self.subTest(key=key):
                result = await self.tool.handler({**self.arguments, key: "untrusted"})
                self.assertEqual(result["error"]["code"], "invalid_tool_input")
        self.repository.get_image_evidence.assert_not_awaited()

    async def test_malformed_record_and_frame_identifiers_are_rejected(self):
        for arguments in (
            {}, {**self.arguments, "record_id": "not-a-uuid"},
            {**self.arguments, "record_type": "unknown"},
            {**self.arguments, "frame_id": "not-a-uuid"},
        ):
            with self.subTest(arguments=arguments):
                result = await self.tool.handler(arguments)
                self.assertEqual(result["error"]["code"], "invalid_tool_input")
        self.repository.get_image_evidence.assert_not_awaited()

    async def test_unavailable_result_does_not_invent_an_image(self):
        self.repository.get_image_evidence.return_value = None
        result = await self.tool.handler(self.arguments)
        self.assertEqual(result["image_status"], "unavailable")
        self.assertEqual(result["error"]["code"], "frame_unavailable")
        self.assertNotIn("image", result)
        self.loader.download_object.assert_not_awaited()

    async def test_invalid_or_failed_selections_cannot_resolve_pixels(self):
        for selection in (
            {"status": "failed"},
            {"status": "success", "image_status": "selected", "image": {}},
            {"status": "success", "image_status": "attached", "image": self.evidence.reference.to_metadata()},
            {"status": "success", "image_status": "selected", "image": {
                **self.evidence.reference.to_metadata(), "storage_path": "untrusted",
            }},
        ):
            with self.subTest(selection=selection), self.assertRaises(ImageUnavailableError):
                await self.tool.image_resolver(selection)
        self.loader.download_object.assert_not_awaited()
