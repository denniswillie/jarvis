from __future__ import annotations

from dataclasses import replace
from datetime import timedelta
import json
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock
from uuid import uuid4

from app.perception.contracts import FastPerceptionOutputError, parse_fast_perception_output
from app.perception.fast_analyzer import FastPerceptionAnalyzer
from app.perception.prompts import build_fast_perception_prompt
from app.perception.references import PerceptionReferences
from app.perception.result_validation import parse_referenced_perception_output
from test_fast_perception_contract import _request, _payload
from test_fast_perception_pipeline import _Harness, NOW


def focused_request():
    request = _request()
    return replace(request, active_focus={"id": str(uuid4()), "target_description": "toilet roll"})


def candidate(subject="focus:0", frame="frame:1", kind="visual_focus"):
    return {"subject_type": kind, "subject_id": subject, "supporting_frame_ids": [frame],
            "confidence": .96, "observed_condition": "A toilet roll is visible.", "suggested_message": "I see the toilet roll."}


def handle_payload(request):
    payload = _payload(request)
    payload["entities"][0]["existing_entity_id"] = "entity:0"
    payload["entities"][0]["last_visible_frame_id"] = "frame:1"
    payload["state_changes"][0]["subject_local_key"] = "entity:0"
    payload["keyframe_candidates"][0]["frame_id"] = "frame:1"
    payload["prospective_matches"] = [candidate()]
    return payload


def parse(payload, request):
    return parse_referenced_perception_output(json.dumps(payload), request=request, references=PerceptionReferences(request))


class PerceptionPartialResultTests(unittest.TestCase):
    def test_short_references_round_trip_to_existing_storage_contract(self):
        request = focused_request()
        result = parse(handle_payload(request), request)
        self.assertEqual(result.rejected, ())
        self.assertEqual(result.output.entities[0].existing_entity_id, "mug-1")
        self.assertEqual(result.output.state_changes[0].subject_local_key, "mug-1")
        self.assertEqual(result.output.keyframe_candidates[0].frame_id, request.primary_frame.frame_id)
        self.assertEqual(str(result.output.prospective_matches[0].subject_id), request.active_focus["id"])
        legacy = parse_fast_perception_output(result.output.model_dump_json(), request=request)
        self.assertEqual(legacy, result.output)
        self.assertNotIn("frame:1", result.output.model_dump_json())
        self.assertNotIn("entity:0", result.output.model_dump_json())

    def test_v2_accepts_existing_uuid_output_without_changing_legacy_validation(self):
        request = focused_request()
        payload = _payload(request)
        payload["prospective_matches"] = [candidate(request.active_focus["id"], str(request.primary_frame.frame_id))]
        self.assertEqual(parse(payload, request).output, parse_fast_perception_output(json.dumps(payload), request=request))
        payload["prospective_matches"][0]["subject_id"] = str(uuid4())
        with self.assertRaises(FastPerceptionOutputError):
            parse_fast_perception_output(json.dumps(payload), request=request)
        self.assertEqual(parse(payload, request).output.prospective_matches, [])

    def test_invalid_optional_candidates_do_not_discard_valid_scene_or_matches(self):
        request = focused_request()
        for invalid in (candidate(str(uuid4())), candidate(frame=str(uuid4())), candidate(kind="unknown"), None, {"private": "SECRET"}):
            with self.subTest(invalid=invalid):
                payload = handle_payload(request)
                payload["prospective_matches"] = [invalid, candidate()]
                payload["interrupt_candidate"] = {"message": "SECRET", "confidence": 5}
                result = parse(payload, request)
                self.assertEqual(result.output.summary, payload["summary"])
                self.assertEqual(len(result.output.prospective_matches), 1)
                self.assertIsNone(result.output.interrupt_candidate)
                self.assertEqual(len(result.rejected), 2)
                self.assertNotIn("SECRET", str(result.rejected))

    def test_bad_dependent_state_and_keyframes_are_rejected_not_persisted(self):
        request = focused_request()
        payload = handle_payload(request)
        payload["state_changes"] += [{"subject_local_key": "missing", "change": "SECRET", "confidence": .9}]
        payload["keyframe_candidates"] += [{"frame_id": str(uuid4()), "reason": "SECRET", "score": .9}]
        result = parse(payload, request)
        self.assertEqual(len(result.output.state_changes), 1)
        self.assertEqual(len(result.output.keyframe_candidates), 1)
        self.assertEqual({r.reason for r in result.rejected}, {"unknown_output_entity", "unknown_frame_reference"})

    def test_invalid_core_is_not_salvaged(self):
        request = focused_request()
        for field, value in (("entities", "bad"), ("summary", None), ("unknown", "SECRET"), ("confidence", 5)):
            payload = handle_payload(request)
            payload[field] = value
            with self.subTest(field=field), self.assertRaises(FastPerceptionOutputError) as error:
                parse(payload, request)
            self.assertNotIn("SECRET", str(error.exception))
        for field, value in (("existing_entity_id", "entity:99"), ("last_visible_frame_id", str(uuid4()))):
            payload = handle_payload(request)
            payload["entities"][0][field] = value
            with self.assertRaises(FastPerceptionOutputError):
                parse(payload, request)
        payload = handle_payload(request)
        payload["visible_text"] = [{"text": "SECRET", "frame_id": "frame:99", "confidence": .9}]
        with self.assertRaises(FastPerceptionOutputError):
            parse(payload, request)

    def test_malformed_json_duplicate_keys_and_nan_are_rejected(self):
        request = focused_request()
        for raw in ('{"summary":"a","summary":"b"}', "[]", "not JSON", '{"confidence":NaN}'):
            with self.subTest(raw=raw), self.assertRaises(FastPerceptionOutputError):
                parse_referenced_perception_output(raw, request=request, references=PerceptionReferences(request))

    def test_references_cannot_cross_owner_session_window_or_changed_target(self):
        request = focused_request()
        references = PerceptionReferences(request)
        for other in (replace(request, owner_id="foreign"), replace(request, session_id=uuid4()),
                      replace(request, window_id=uuid4()), replace(request, active_focus={"id": str(uuid4())})):
            with self.assertRaisesRegex(FastPerceptionOutputError, "scope"):
                parse_referenced_perception_output(json.dumps(handle_payload(request)), request=other, references=references)
        for hint in ({"owner_id": "foreign"}, {"session_id": str(uuid4())}):
            with self.assertRaises(FastPerceptionOutputError):
                PerceptionReferences(replace(request, active_focus={**request.active_focus, **hint}))

    def test_duplicate_targets_and_ambiguous_handle_shaped_entity_ids_fail_closed(self):
        request = focused_request()
        reminder = {"id": str(uuid4())}
        with self.assertRaisesRegex(FastPerceptionOutputError, "duplicate"):
            PerceptionReferences(replace(request, active_reminders=(reminder, reminder)))
        with self.assertRaisesRegex(FastPerceptionOutputError, "ambiguous"):
            PerceptionReferences(replace(request, existing_entity_ids=frozenset({"entity:1", "other"})))
        with self.assertRaisesRegex(ValueError, "unique"):
            replace(request, frames=(request.frames[0], request.frames[0]))

    def test_frame_from_an_earlier_window_cannot_support_a_current_match(self):
        original = focused_request()
        request = replace(original, window_id=uuid4(), frames=(original.primary_frame,))
        payload = _payload(request)
        payload["prospective_matches"] = [candidate(request.active_focus["id"], str(original.frames[0].frame_id))]
        result = parse(payload, request)
        self.assertEqual(result.output.prospective_matches, [])
        self.assertEqual(result.rejected[0].reason, "unknown_frame_reference")
        self.assertEqual(result.output.summary, payload["summary"])

    def test_oversized_or_invalid_optional_collection_does_not_bypass_limits(self):
        request = focused_request()
        for value in (None, "SECRET", [candidate()] * 9):
            payload = handle_payload(request)
            payload["prospective_matches"] = value
            result = parse(payload, request)
            self.assertEqual(result.output.prospective_matches, [])
            self.assertEqual(result.rejected[0].reason, "invalid_candidate_collection")

    def test_prompt_uses_v2_reference_ids_without_exposing_scope_ids(self):
        request = focused_request()
        prompt = build_fast_perception_prompt(request, references=PerceptionReferences(request))
        for durable in (str(request.session_id), str(request.window_id), request.active_focus["id"], str(request.primary_frame.frame_id)):
            self.assertNotIn(durable, prompt)
        self.assertIn('"contract_version": 3', prompt)
        self.assertIn('"id": "focus:0"', prompt)
        self.assertIn('"entity_id": "entity:0"', prompt)
        self.assertIn("viewpoint uncertainty", prompt)

    def test_conditional_reminder_handles_cannot_be_used_as_focus_ids(self):
        request = replace(focused_request(), active_reminders=({"id": str(uuid4())},))
        payload = handle_payload(request)
        payload["prospective_matches"] = [candidate("reminder:0", kind="conditional_reminder"), candidate("reminder:0")]
        result = parse(payload, request)
        self.assertEqual(len(result.output.prospective_matches), 1)
        self.assertEqual(str(result.output.prospective_matches[0].subject_id), request.active_reminders[0]["id"])
        self.assertEqual(result.rejected[0].reason, "unknown_subject_reference")


class PerceptionPartialPipelineTests(unittest.IsolatedAsyncioTestCase):
    async def test_foreign_owner_or_session_frames_do_not_enter_the_request(self):
        harness = _Harness(static=True)
        own = harness.frame(sequence=3, captured_at=NOW + timedelta(seconds=3))
        foreign_owner = replace(harness.frame(sequence=1, captured_at=NOW), owner_id="another-owner")
        foreign_session = replace(harness.frame(sequence=2, captured_at=NOW + timedelta(seconds=1)), session_id=uuid4())
        try:
            for frame in (foreign_owner, foreign_session, own):
                harness.put(frame, b"fixture")
            await harness.pipeline.enqueue(frame=own, image_bytes=b"fixture")
            await harness.pipeline.drain()
            request = harness.fast_analyzer.requests[0]
            self.assertEqual(request.allowed_frame_ids, frozenset({own.id}))
        finally:
            await harness.pipeline.close()

    async def test_real_analyzer_and_pipeline_keep_scene_then_accept_next_search_match(self):
        harness = _Harness(static=True)
        focus_id = uuid4()
        captures = []

        async def generate(*, prompt, images):
            captures.append((prompt, images))
            return json.dumps({"summary": "A toilet roll is visible.", "entities": [], "visible_text": [],
                               "starts_new_event": False, "confidence": .96,
                               "prospective_matches": [candidate("focus:99" if len(captures) == 1 else "focus:0", "frame:0")]})

        model = SimpleNamespace(provider="fixture", model="fixture", generate_temporal_json=generate)
        harness.pipeline._fast_analyzer = FastPerceptionAnalyzer(model)
        original_build = harness.pipeline._context_builder.build

        async def build(frame):
            context = await original_build(frame)
            return replace(context, active_focus={"id": str(focus_id), "target_description": "toilet roll"})

        harness.pipeline._context_builder.build = build
        matches = AsyncMock()
        harness.pipeline._prospective_service = SimpleNamespace(process_matches=matches)
        try:
            for index in (1, 2):
                frame = harness.frame(sequence=index, captured_at=NOW + timedelta(seconds=index))
                harness.put(frame, b"fixture")
                await harness.pipeline.enqueue(frame=frame, image_bytes=b"fixture")
                await harness.pipeline.drain()
            self.assertEqual(harness.visual_repository.failed, [])
            self.assertEqual(len(harness.perception_repository.inserted), 2)
            self.assertEqual(len(harness.working_memory_service.applied), 2)
            matches.assert_awaited_once()
            self.assertEqual(matches.await_args.kwargs["matches"][0].subject_id, focus_id)
            self.assertEqual(harness.interrupt_service.calls, [])
            self.assertTrue(all("frame:" in images[0].label for _, images in captures))
        finally:
            await harness.pipeline.close()
