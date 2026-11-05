from __future__ import annotations

import json
import unittest
from dataclasses import replace
from uuid import uuid4

from app.memory.consolidation.prompts import (
    CONSOLIDATION_INSTRUCTIONS,
    build_consolidation_prompt,
    build_consolidation_repair_feedback,
    MAX_QUOTATION_CHARACTERS_PER_REQUEST,
    MAX_QUOTATION_CHARACTERS_PER_SOURCE,
    MAX_SOURCE_CONTEXT_CHARACTERS_PER_REQUEST,
    MAX_SOURCE_CONTEXT_CHARACTERS_PER_SOURCE,
    MAX_SOURCE_CONTEXT_STRING_CHARACTERS,
    MAX_SOURCE_CONTEXT_VALUES,
)
from app.memory.consolidation.contracts import (
    ConsolidationValidationError, EvidenceCitation, parse_consolidation_result,
)
from app.memory.consolidation.evidence import build_citation_catalog
from test_consolidation_contracts import _cited_output, _request


class ConsolidationPromptTests(unittest.TestCase):
    def test_prompt_carries_bounded_event_evidence_and_keyframe_identity(self) -> None:
        request = _request()
        prompt = build_consolidation_prompt(request)

        self.assertIn(str(request.event.id), prompt)
        self.assertIn(str(request.evidence[0].evidence_ref_id), prompt)
        self.assertIn(str(request.keyframes[0].visual_frame_id), prompt)
        self.assertIn("active_visual_focus", prompt)
        self.assertIn("Take the keys", prompt)
        self.assertNotIn("image_bytes", prompt)

    def test_instructions_forbid_unsupported_memory_mutations(self) -> None:
        self.assertIn("Do not invent intention", CONSOLIDATION_INSTRUCTIONS)
        self.assertIn("may not be cited", CONSOLIDATION_INSTRUCTIONS)
        self.assertIn("deletion", CONSOLIDATION_INSTRUCTIONS)


    def test_catalog_and_example_use_real_refs_and_literal_source_strings(self):
        request = _request()
        source = replace(request.evidence[0], payload={
            "role": "user", "modality": "text", "content": '  Caf\u00e9:\n\tKeep "HOME".',
            "metadata": {"label": "not citable"},
        })
        request = replace(request, evidence=(source,))
        payload = json.loads(build_consolidation_prompt(request).split("\n\n", 1)[1])
        entry = payload["attached_evidence"][0]
        self.assertNotIn("evidence_id", entry)
        self.assertNotIn("payload", entry)
        self.assertEqual(entry["evidence_ref_id"], str(source.evidence_ref_id))
        self.assertEqual(entry["quotation_material"][0]["text"], source.payload["content"])
        self.assertNotIn("not citable", json.dumps(entry))
        example = EvidenceCitation.model_validate(payload["citation_example"])
        catalog = build_citation_catalog(request)
        self.assertTrue(catalog[("message", source.evidence_ref_id)].contains_quote(
            example.supporting_text,
        ))

    def test_context_and_passive_images_stay_out_of_citation_catalog(self):
        request = _request()
        passive = replace(
            request.evidence[0], evidence_id=uuid4(), evidence_ref_id=uuid4(),
            payload={"role": "user", "modality": "visual", "content": "image observation"},
        )
        context = replace(passive, evidence_id=None, citation_allowed=False)
        request = replace(
            request, evidence=(*request.evidence, passive), context_messages=(context,),
        )
        payload = json.loads(build_consolidation_prompt(request).split("\n\n", 1)[1])
        self.assertEqual(len(payload["attached_evidence"]), 1)
        self.assertFalse(payload["uncitable_evidence"][0]["citation_allowed"])
        self.assertFalse(payload["context_only_messages"][0]["citation_allowed"])
        self.assertNotIn(str(passive.evidence_ref_id), json.dumps(payload["attached_evidence"]))

    def test_catalog_presentation_is_bounded_but_authority_is_not_truncated(self):
        request = _request()
        content = "x" * (MAX_QUOTATION_CHARACTERS_PER_SOURCE + 100) + "tail evidence"
        sources = tuple(replace(
            request.evidence[0], evidence_id=uuid4(), evidence_ref_id=uuid4(),
            payload={"role": "user", "modality": "text", "content": content},
        ) for _ in range(20))
        request = replace(request, evidence=sources)
        payload = json.loads(build_consolidation_prompt(request).split("\n\n", 1)[1])
        entries = payload["attached_evidence"]
        self.assertTrue(all(entry["quotation_material_truncated"] for entry in entries))
        self.assertLessEqual(
            sum(len(item["text"]) for entry in entries for item in entry["quotation_material"]),
            MAX_QUOTATION_CHARACTERS_PER_REQUEST,
        )
        catalog = build_citation_catalog(request)
        self.assertTrue(all(entry.contains_quote("tail evidence") for entry in catalog.values()))

    def test_repair_feedback_prioritizes_the_right_source_without_echoing_bad_text(self):
        request = _request()
        error = ConsolidationValidationError(
            "private raw model text", code="invalid_reference",
            location="entity_candidates[0].evidence[0]",
            evidence_type="message", evidence_ref_id=request.evidence[0].evidence_id,
        )
        feedback = build_consolidation_repair_feedback(error, request)
        self.assertIn("invalid_reference", feedback)
        self.assertIn(error.location, feedback)
        self.assertIn(str(request.evidence[0].evidence_ref_id), feedback)
        self.assertNotIn(str(request.evidence[0].evidence_id), feedback)
        self.assertNotIn("private raw model text", feedback)
        self.assertNotIn(request.evidence[0].payload["content"], feedback)
        self.assertIn("copied verbatim", feedback)


    def test_tool_business_state_survives_without_becoming_a_serialized_quote(self):
        request = _request()
        source = replace(request.evidence[0], evidence_type="tool_execution", payload={
            "tool_name": "update_reminder", "status": "completed", "mutates_state": True,
            "arguments": {"title": "Buy milk"},
            "result": {"reminder": {
                "title": "Buy milk", "status": "cancelled", "enabled": False,
                "trigger_count": 2, "remaining": 0, "next_trigger": None,
                "matches": [], "options": {}, "id": "transport-identifier",
                "metadata": {"secret_label": "not quotation material"},
            }},
        })
        request = replace(request, evidence=(source,))
        payload = json.loads(build_consolidation_prompt(request).split("\n\n", 1)[1])
        entry = payload["attached_evidence"][0]
        context = {item["source_path"]: item["value"] for item in entry["source_context"]}
        self.assertEqual(entry["source_status"], "completed")
        self.assertEqual(context["result.reminder.status"], "cancelled")
        self.assertIs(context["result.reminder.enabled"], False)
        self.assertEqual(context["result.reminder.trigger_count"], 2)
        self.assertEqual(context["result.reminder.remaining"], 0)
        self.assertIsNone(context["result.reminder.next_trigger"])
        self.assertEqual(context["result.reminder.matches"], [])
        self.assertEqual(context["result.reminder.options"], {})
        self.assertIs(context["mutates_state"], True)
        self.assertFalse(entry["source_context_truncated"])
        self.assertNotIn("transport-identifier", json.dumps(entry))
        self.assertNotIn("not quotation material", json.dumps(entry))
        parse_consolidation_result(_cited_output(request, text="cancelled"), request=request)
        for quote in ("2", "false", "null", '"status":"cancelled"', "completed"):
            with self.subTest(quote=quote), self.assertRaises(ConsolidationValidationError):
                parse_consolidation_result(_cited_output(request, text=quote), request=request)

    def test_observation_model_confidence_and_boolean_change_are_preserved(self):
        request = _request()
        source = replace(request.evidence[0], evidence_type="observation", payload={
            "summary": "A device is visible",
            "entities": [{"name": "device", "attributes": {
                "model": "Aurora 7", "status": "off", "provider": "Local Co",
            }}],
            "state_changes": [{
                "subject": "device", "attribute": "power", "before": True, "after": False,
            }],
            "confidence": 0.4, "compacted": False,
        })
        request = replace(request, evidence=(source,))
        payload = json.loads(build_consolidation_prompt(request).split("\n\n", 1)[1])
        entry = payload["attached_evidence"][0]
        context = {item["source_path"]: item["value"] for item in entry["source_context"]}
        self.assertEqual(context["entities[0].attributes.model"], "Aurora 7")
        self.assertEqual(context["entities[0].attributes.status"], "off")
        self.assertEqual(context["entities[0].attributes.provider"], "Local Co")
        self.assertIs(context["state_changes[0].before"], True)
        self.assertIs(context["state_changes[0].after"], False)
        self.assertEqual(context["confidence"], 0.4)
        self.assertIs(context["compacted"], False)
        catalog = build_citation_catalog(request)
        self.assertTrue(catalog[("observation", source.evidence_ref_id)].contains_quote("Aurora 7"))
        self.assertFalse(catalog[("observation", source.evidence_ref_id)].contains_quote("0.4"))

    def test_interpretation_context_is_bounded_and_explicit_about_omissions(self):
        request = _request()
        body = "x" * (MAX_SOURCE_CONTEXT_STRING_CHARACTERS + 100)
        sources = tuple(replace(
            request.evidence[0], evidence_id=uuid4(), evidence_ref_id=uuid4(),
            evidence_type="tool_execution", payload={
                "result": {f"field_{index}": body for index in range(MAX_SOURCE_CONTEXT_VALUES + 5)},
            },
        ) for _ in range(20))
        request = replace(request, evidence=sources)
        payload = json.loads(build_consolidation_prompt(request).split("\n\n", 1)[1])
        total = 0
        for entry in payload["attached_evidence"]:
            self.assertTrue(entry["source_context_truncated"])
            self.assertLessEqual(len(entry["source_context"]), MAX_SOURCE_CONTEXT_VALUES)
            size = sum(len(json.dumps(
                item, ensure_ascii=False, separators=(",", ":"),
            )) for item in entry["source_context"])
            self.assertLessEqual(size, MAX_SOURCE_CONTEXT_CHARACTERS_PER_SOURCE)
            total += size
            for item in entry["source_context"]:
                self.assertLessEqual(len(item["value"]), MAX_SOURCE_CONTEXT_STRING_CHARACTERS)
        self.assertLessEqual(total, MAX_SOURCE_CONTEXT_CHARACTERS_PER_REQUEST)
        self.assertIn("not quotation material", CONSOLIDATION_INSTRUCTIONS)
        self.assertIn("do not infer absence", CONSOLIDATION_INSTRUCTIONS)
