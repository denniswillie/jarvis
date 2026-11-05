from __future__ import annotations

from datetime import timedelta
import json
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock

import httpx

from app.model_providers.gemini_fast_visual import GeminiFastVisualModel
from app.perception.multimodal import (
    EventBatch, MultimodalEventAnalyzer, PerceptionContext, SituationalReminder,
    build_multimodal_prompt, build_multimodal_schema,
)
from app.visual_memory.gemini_client import (
    GeminiVisualAnalyzerClient, GeminiVisualAnalyzerConfig, GeminiVisualAnalyzerError,
)
from test_multimodal_perception import NOW, dialogue, fact, frame, parse, request, response


REMINDER = SituationalReminder(id="canonical-reminder", revision=1, condition="Leaving")


class SchemaTests(unittest.TestCase):
    def test_current_view_is_required_for_newest_image_and_null_without_images(self):
        req = request(frames=(frame(5), frame(1)))
        schema = build_multimodal_schema(req)
        self.assertIn("current_view", schema["required"])
        self.assertEqual(schema["properties"]["current_view"], {"$ref": "#/$defs/Finding"})
        sources = schema["$defs"]["Finding"]["properties"]["sources"]
        self.assertEqual(sources["items"]["enum"], ["frame:0"])
        self.assertEqual((sources["minItems"], sources["maxItems"]), (1, 1))
        text_schema = build_multimodal_schema(request(frames=(), speech=(dialogue(),)))
        self.assertEqual(text_schema["properties"]["current_view"], {"type": "null"})
        self.assertIn('"current_view_source":"frame:0"', build_multimodal_prompt(req))

    def test_only_new_finding_references_have_per_pass_enums(self):
        req = request(speech=(dialogue(), dialogue(caption_id="assistant", speaker="assistant")),
                      reminders=(REMINDER,))
        schema = build_multimodal_schema(req)
        definitions = schema["$defs"]
        match = definitions["ReminderFinding"]["properties"]
        delivery = definitions["DeliveryFinding"]["properties"]
        self.assertEqual(match["reminder_id"]["enum"], ["reminder:0"])
        self.assertEqual(match["sources"]["items"]["enum"], ["frame:0", "dialogue:0", "dialogue:1"])
        self.assertEqual(delivery["sources"]["items"]["enum"], ["dialogue:1"])
        self.assertEqual(delivery["sources"]["minItems"], 1)
        self.assertEqual(delivery["sources"]["maxItems"], 1)
        for name in ("HumanFact", "HumanActivity", "StatedIntention", "MemoryFinding"):
            self.assertNotIn("enum", definitions[name]["properties"]["sources"]["items"])

    def test_empty_reminders_force_empty_finding_arrays_without_empty_enum(self):
        schema = build_multimodal_schema(request())
        for name in ("triggered_reminders", "observed_reminder_deliveries"):
            self.assertEqual(schema["properties"][name]["maxItems"], 0)
        self.assertNotIn('"enum": []', json.dumps(schema))

    def test_native_schema_omits_large_nested_bounds_but_keeps_reference_bounds(self):
        schema = build_multimodal_schema(request(speech=(dialogue(speaker="assistant"),), reminders=(REMINDER,)))
        definitions = schema["$defs"]
        for name in ("HumanFact", "HumanActivity", "StatedIntention", "MemoryFinding", "ReminderFinding"):
            self.assertNotIn("maxItems", definitions[name]["properties"]["sources"])
            self.assertEqual(definitions[name]["properties"]["sources"]["minItems"], 1)
        for name in ("activities", "stated_intentions", "relevant_details", "uncertainties"):
            self.assertNotIn("maxItems", definitions["HumanStateContent"]["properties"][name])
        self.assertEqual(definitions["DeliveryFinding"]["properties"]["sources"]["maxItems"], 1)

    def test_parser_still_rejects_oversized_human_and_source_lists(self):
        req = request(reminders=(REMINDER,))
        result = parse(response({"activities": [{**fact(), "status": "active"}] * 9},
            reminders=[{"reminder_id": "reminder:0", "description": "Leaving", "sources": ["frame:0"] * 9}],
            memory={"kind": "placement", "description": "Keys", "sources": ["frame:0"] * 9}), req)
        self.assertIsNone(result.updated_human_state)
        self.assertEqual(result.triggered_reminders, ())
        self.assertIsNone(result.memory_worthy_observation)
        self.assertIn("human:invalid_or_stale_snapshot", result.rejected)
        self.assertIn("reminders.0:invalid_shape:sources:too_long", result.rejected)
        self.assertIn("memory:invalid_reference_or_shape", result.rejected)

    def test_no_new_assistant_text_forces_empty_delivery_array(self):
        for speech in ((), (dialogue(),), (dialogue(speaker="assistant", text="Old", new_text_start=3),)):
            with self.subTest(speech=speech):
                schema = build_multimodal_schema(request(speech=speech, reminders=(REMINDER,)))
                self.assertEqual(schema["properties"]["observed_reminder_deliveries"]["maxItems"], 0)
                self.assertNotIn("maxItems", schema["properties"]["triggered_reminders"])

    def test_each_pass_has_independent_handles(self):
        first = build_multimodal_schema(request(speech=(dialogue(speaker="assistant"),), reminders=(REMINDER,)))
        second = build_multimodal_schema(request())
        self.assertEqual(first["$defs"]["DeliveryFinding"]["properties"]["sources"]["items"]["enum"], ["dialogue:0"])
        self.assertNotIn("enum", second["$defs"]["DeliveryFinding"]["properties"]["sources"]["items"])

    def test_reobserved_same_fact_refreshes_evidence_not_other_facts_or_memory(self):
        previous = parse(response({"setting": fact("A card displays 47"),
            "stated_intentions": [fact("Plans to cook", "dialogue:0", "user_report")]}),
            request(speech=(dialogue(text="I plan to cook."),))).updated_human_state
        intention = previous.stated_intentions[0]
        for second in (5, 6):
            content = previous.content().model_dump(mode="json")
            content["setting"]["sources"] = ["frame:0"]
            # The supplied source, not the model's copied timestamp, dates the fact.
            req = request(frames=(frame(second),), previous=previous)
            result = parse(response(content), req)
            self.assertEqual(result.rejected, ())
            self.assertEqual(result.updated_human_state.setting.description, previous.setting.description)
            self.assertEqual(result.updated_human_state.setting.evidence_at, NOW + timedelta(seconds=second))
            self.assertEqual(result.updated_human_state.setting.sources, (f"frame:{req.batch.frames[0].frame_id}",))
            self.assertEqual(result.updated_human_state.stated_intentions, (intention,))
            self.assertIsNone(result.memory_worthy_observation)
            previous = result.updated_human_state

    def test_carried_fact_does_not_refresh_just_because_another_frame_arrives(self):
        previous = parse(response({"setting": fact("A card displays 47")}), request()).updated_human_state
        result = parse(response(previous.content().model_dump(mode="json")),
                       request(frames=(frame(5),), previous=previous))
        self.assertEqual(result.rejected, ())
        self.assertIsNone(result.updated_human_state)
        self.assertIsNone(result.memory_worthy_observation)

    def test_prompt_separates_reobservation_from_carried_evidence_and_memory(self):
        prompt = build_multimodal_prompt(request())
        self.assertIn("even when descriptions are unchanged", prompt)
        self.assertIn("carried prior facts retain original canonical sources and evidence_at", prompt)
        self.assertIn("Compare memory_worthy_observation against prior human state", prompt)
        self.assertIn("Reconfirming a known fact refreshes live evidence, not memory; return null", prompt)

    def test_genuinely_new_observation_still_reaches_memory_output(self):
        previous = parse(response({"setting": fact("A card displays 47")}), request()).updated_human_state
        req = request(frames=(frame(5),), previous=previous)
        result = parse(response(memory={"kind": "placement", "description": "Keys placed beside the card",
                                        "sources": ["frame:0"]}), req)
        self.assertEqual(result.rejected, ())
        self.assertEqual(result.memory_worthy_observation.kind, "placement")
        self.assertEqual(result.memory_worthy_observation.evidence_at, NOW + timedelta(seconds=5))


class SchemaPlumbingTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.requests = []
        self.status = 200

        async def handler(req):
            self.requests.append(json.loads(req.content))
            return httpx.Response(self.status, json={"candidates": [{"content": {"parts": [{"text": response()}]}}]})

        self.client = GeminiVisualAnalyzerClient(GeminiVisualAnalyzerConfig("synthetic", "fixture"))
        await self.client._client.aclose()
        self.client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        self.model = GeminiFastVisualModel(self.client)

    async def asyncTearDown(self):
        await self.client.close()

    async def test_runner_passes_schema_through_all_layers_without_extra_call(self):
        context = PerceptionContext(reminders=(REMINDER,))
        batch = EventBatch("owner", dialogue=(dialogue(speaker="assistant"),))
        publish = AsyncMock()
        analyzer = MultimodalEventAnalyzer(owner_id="owner", model=self.model,
            read_context=AsyncMock(return_value=context), publish=publish, clock=lambda: NOW)
        try:
            analyzer.submit(batch)
            await analyzer.drain()
            self.assertIsNone(analyzer.last_error)
            publish.assert_awaited_once()
            self.assertEqual(len(self.requests), 1)
            config = self.requests[0]["generationConfig"]
            self.assertEqual(config["responseMimeType"], "application/json")
            self.assertEqual(config["responseJsonSchema"]["$defs"]["DeliveryFinding"]["properties"]
                             ["sources"]["items"]["enum"], ["dialogue:0"])
            self.assertEqual(len(self.requests[0]["contents"][0]["parts"]), 1)
        finally:
            await analyzer.close()

    async def test_legacy_requests_do_not_acquire_schema(self):
        await self.model.generate_temporal_json(prompt="existing", images=())
        await self.model.generate_historical_evidence_json(prompt="history", images=())
        await self.client.generate_json(prompt="single", image_bytes=b"image", content_type="image/jpeg")
        for req in self.requests:
            self.assertEqual(req["generationConfig"], {"temperature": 0.2, "responseMimeType": "application/json"})

    async def test_legacy_adapter_does_not_pass_new_keyword_to_old_client(self):
        calls = []

        async def old_client(*, prompt, images):
            calls.append(prompt)
            return "{}"

        model = GeminiFastVisualModel(SimpleNamespace(generate_multiframe_json=old_client))
        self.assertEqual(await model.generate_temporal_json(prompt="existing", images=()), "{}")
        self.assertEqual(calls, ["existing"])

    async def test_provider_schema_rejection_has_no_unschematized_retry(self):
        self.status = 400
        with self.assertRaises(GeminiVisualAnalyzerError):
            await self.model.generate_temporal_json(prompt="test", images=(), response_json_schema=build_multimodal_schema(request()))
        self.assertEqual(len(self.requests), 1)
