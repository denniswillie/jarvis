from __future__ import annotations

import json
import unittest
from dataclasses import replace
from uuid import uuid4

import test_current_image_service as image_fixtures

from app.images import ImageReference, ImageUnavailableError
from app.perception.tools import CurrentImageToolModule, create_get_current_camera_view_tool
from app.settings import JarvisSettings
from app.tools.executor import JarvisToolExecutor
from app.tools.module import ToolBuildContext
from app.tools.registry import JarvisToolRegistry


class CurrentImageToolTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.fixture = image_fixtures.CurrentImageServiceTests()
        self.fixture.setUp()
        self.tool = create_get_current_camera_view_tool(
            service=self.fixture.service, owner_id=self.fixture.owner, session_id=self.fixture.session,
        )

    async def test_empty_schema_and_module_registration_replace_the_old_tool(self):
        module = CurrentImageToolModule(self.fixture.service)
        tools = module.build_tools(ToolBuildContext(
            settings=None, owner_id=self.fixture.owner, session_id=self.fixture.session,
        ))
        self.assertEqual(module.name, "current_image")
        self.assertEqual([tool.name for tool in tools], ["get_current_camera_view"])
        self.assertEqual(self.tool.input_schema, {"type": "object", "properties": {}, "additionalProperties": False})
        self.assertFalse(self.tool.mutates_state)
        self.assertIsNotNone(self.tool.image_resolver)
        self.assertIn("search_memory", self.tool.description)
        self.assertIn("capture_latest_visual_frame", self.tool.description)
        self.assertIn(
            "Attach the latest available current camera image for direct inspection",
            self.tool.description,
        )
        self.assertIn("Automatic image delivery being disabled does not disable this tool", self.tool.description)
        self.assertIn("does not analyze", self.tool.description)

    async def test_description_prefers_attached_images_and_preserves_retrieval_fallbacks(self):
        for policy in (
            "First use adequate fresh images already acknowledged in this response's live_view cutoff",
            "Do not call merely because the request is visual or to reconfirm freshness",
            "Request another view when needed pixels are absent, stale, of uncertain freshness",
            "or do not show a relevant scene change, or when a new view could supply a missing detail",
            "A reported change does not require another fetch if the attached image already shows it",
            "before asking the user to paste or describe it",
            "Never use a present camera image to answer historical recall; "
            "use search_memory and get_memory_image instead",
        ):
            with self.subTest(policy=policy):
                self.assertIn(policy, self.tool.description)

    async def test_description_distinguishes_unreadability_from_missing_pixels(self):
        self.assertIn(
            "For unreadable details in an otherwise current image, ask for a clearer view rather "
            "than repeatedly fetching an unchanged scene; retrieval does not improve legibility",
            self.tool.description,
        )

    async def test_selection_is_metadata_only_and_resolver_returns_actual_pixels(self):
        frame = self.fixture._put()
        selection = await self.tool.handler({})
        self.assertEqual(selection["status"], "success")
        self.assertEqual(selection["image_status"], "selected")
        self.assertEqual(selection["image"]["frame_id"], str(frame.frame_id))
        self.assertEqual(selection["image"]["captured_at"], frame.captured_at.isoformat())
        serialized = json.dumps(selection)
        for forbidden in ("image_bytes", "storage_path", "private/owner", "signed_url", "owner_id", "description", "visible_text", "provider_item_id"):
            self.assertNotIn(forbidden, serialized)
        resolved = await self.tool.image_resolver(json.loads(serialized))
        self.assertEqual(resolved.image_bytes, frame.image_bytes)
        self.assertEqual(resolved.reference, self.fixture._select())

    async def test_replay_with_new_module_resolves_exact_reference_not_latest(self):
        first = self.fixture._put()
        selection = json.loads(json.dumps(await self.tool.handler({})))
        self.fixture.clock.advance(1)
        self.fixture._put(self.fixture._frame(frame_sequence=2))
        reconnected_tool = create_get_current_camera_view_tool(
            service=self.fixture.service, owner_id=self.fixture.owner, session_id=self.fixture.session,
        )
        resolved = await reconnected_tool.image_resolver(selection)
        self.assertEqual(resolved.reference.frame_id, first.frame_id)
        self.assertEqual(resolved.image_bytes, first.image_bytes)

    async def test_extra_arguments_cannot_supply_questions_or_override_scope(self):
        self.fixture._put()
        for arguments in ({"question": "Read this."}, {"owner_id": "another-owner"}, {"session_id": str(uuid4())}, {"url": "https://example.com/image.jpg"}):
            with self.subTest(argument=next(iter(arguments))):
                result = await self.tool.handler(arguments)
                self.assertEqual(result["status"], "failed")
                self.assertEqual(result["error"]["code"], "invalid_tool_input")

    async def test_missing_stream_and_missing_frame_have_no_summary_fallback(self):
        result = await self.tool.handler({})
        self.assertEqual(result["error"]["code"], "no_frame")
        self.fixture.window.end_stream(self.fixture.lease)
        result = await self.tool.handler({})
        self.assertEqual(result["error"]["code"], "no_active_stream")
        self.assertEqual(result["image_status"], "unavailable")
        self.assertNotIn("description", result)

    async def test_stale_selection_returns_unavailable_instead_of_interpreted_evidence(self):
        self.fixture._put()
        self.fixture.clock.advance(11)
        result = await self.tool.handler({})
        self.assertEqual(result["error"]["code"], "stale_frame")
        self.assertEqual(result["image_status"], "unavailable")
        self.assertNotIn("image", result)

    async def test_selected_image_is_revalidated_after_expiry(self):
        self.fixture._put()
        selection = await self.tool.handler({})
        self.fixture.clock.advance(11)
        with self.assertRaises(ImageUnavailableError) as raised:
            await self.tool.image_resolver(selection)
        self.assertEqual(raised.exception.code, "stale_frame")

    async def test_same_stream_identifier_reconnect_does_not_replay_old_pixels(self):
        self.fixture._put()
        selection = await self.tool.handler({})
        self.fixture.window.begin_stream(self.fixture.owner, self.fixture.session, self.fixture.lease.visual_session_id)
        with self.assertRaises(ImageUnavailableError) as raised:
            await self.tool.image_resolver(selection)
        self.assertEqual(raised.exception.code, "source_changed")

    async def test_resolver_remains_bound_to_its_session(self):
        self.fixture._put()
        selection = await self.tool.handler({})
        other = create_get_current_camera_view_tool(
            service=self.fixture.service, owner_id=self.fixture.owner, session_id=uuid4(),
        )
        with self.assertRaises(ImageUnavailableError) as raised:
            await other.image_resolver(selection)
        self.assertEqual(raised.exception.code, "frame_unavailable")

    async def test_invalid_or_wrong_kind_selection_is_not_trusted(self):
        self.fixture._put()
        selected = await self.tool.handler({})
        cases = (
            {}, {**selected, "image_status": "attached"}, {**selected, "image": None},
            {**selected, "image": {**selected["image"], "frame_id": "not-a-uuid"}},
            {**selected, "image": {**selected["image"], "captured_at": "not-a-time"}},
            {**selected, "image": {**selected["image"], "source": "historical"}},
        )
        for index, selection in enumerate(cases):
            with self.subTest(case=index), self.assertRaises(ImageUnavailableError):
                await self.tool.image_resolver(selection)

    async def test_metadata_round_trip_preserves_original_timing_and_trusted_owner(self):
        self.fixture._put()
        reference = self.fixture._select()
        decoded = ImageReference.from_metadata(
            json.loads(json.dumps(reference.to_metadata())), owner_id=self.fixture.owner,
        )
        self.assertEqual(decoded, reference)
        metadata = {**reference.to_metadata(), "owner_id": "untrusted"}
        self.assertEqual(ImageReference.from_metadata(metadata, owner_id=self.fixture.owner).owner_id, self.fixture.owner)
        historical = replace(reference, source="historical", record_type="experience", record_id=uuid4())
        self.assertEqual(ImageReference.from_metadata(historical.to_metadata(), owner_id=self.fixture.owner), historical)

    async def test_disabled_background_and_passive_delivery_do_not_disable_retrieval(self):
        self.fixture._put()
        settings = JarvisSettings(
            _env_file=None, GEMINI_API_KEY=None, VISUAL_FAST_PERCEPTION_ENABLED=False, VISUAL_REALTIME_ENABLED=False,
        )
        tool = CurrentImageToolModule(self.fixture.service).build_tools(ToolBuildContext(
            settings=settings, owner_id=self.fixture.owner, session_id=self.fixture.session,
        ))[0]
        result = await tool.handler({})
        self.assertEqual((await tool.image_resolver(result)).reference.frame_id, self.fixture._select().frame_id)

    async def test_late_obsolete_tool_call_uses_existing_unknown_tool_error(self):
        registry = JarvisToolRegistry()
        registry.register(self.tool)
        result = await JarvisToolExecutor(registry).execute("get_current_visual_context", {"question": "What is here?"})
        self.assertEqual(result["error"]["code"], "tool_not_found")

    async def test_module_requires_session_context(self):
        with self.assertRaises(ValueError):
            CurrentImageToolModule(self.fixture.service).build_tools(ToolBuildContext(
                settings=None, owner_id=self.fixture.owner, session_id=None,
            ))
