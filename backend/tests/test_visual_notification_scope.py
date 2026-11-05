"""Unattributed visual alerts cannot silently bypass active search bookkeeping."""
from datetime import UTC, datetime
import json
import unittest
from uuid import uuid4

from app.perception.contracts import FastPerceptionFrame, FastPerceptionRequest
from app.perception.references import PerceptionReferences, REFERENCE_CONTRACT_VERSION
from app.perception.prompts import build_fast_perception_prompt
from app.perception.result_validation import parse_referenced_perception_output


class NotificationScopeTests(unittest.TestCase):
    def setUp(self):
        now = datetime.now(UTC)
        self.focus_id = uuid4()
        self.frame = FastPerceptionFrame(uuid4(), 1, now, "glasses", "image/jpeg", b"parser-fixture")
        self.request_values = dict(
            owner_id="fixture-owner", session_id=uuid4(), window_id=uuid4(), frames=(self.frame,),
            previous_working_state={}, existing_entity_ids=frozenset(), recent_audio_transcripts=(),
            behavior_instructions="Report useful observations.",
            active_focus={"id": str(self.focus_id), "target_description": "Find the white roll", "mode": "find_once"},
            active_reminders=(), recent_interrupts=(), requested_at=now,
        )
        self.payload = {"summary": "A white roll is on the desk.", "starts_new_event": False, "confidence": .9,
                        "interrupt_candidate": {"message": "The roll is on the desk.",
                                                "reason": "The active search target is visible.", "confidence": .9}}

    def parse(self):
        request = FastPerceptionRequest(**self.request_values)
        return parse_referenced_perception_output(json.dumps(self.payload), request=request,
                                                  references=PerceptionReferences(request))

    def test_generic_only_watch_finding_is_rejected_without_losing_scene(self):
        result = self.parse()
        self.assertEqual(result.output.summary, self.payload["summary"])
        self.assertIsNone(result.output.interrupt_candidate)
        self.assertEqual(result.output.prospective_matches, [])
        self.assertEqual(result.rejected[0].reason, "unattributed_interrupt_with_active_watch")

    def test_explicit_unrelated_general_alert_remains_available(self):
        self.payload["interrupt_candidate"].update(scope="general", message="Water is spilling onto the floor.",
                                                    reason="New visible spill, unrelated to the roll search.")
        result = self.parse()
        self.assertEqual(result.output.interrupt_candidate.scope, "general")
        self.assertEqual(result.rejected, ())

    def test_valid_tracked_match_survives_unattributed_generic_candidate(self):
        self.payload["prospective_matches"] = [{
            "subject_type": "visual_focus", "subject_id": "focus:0", "confidence": .9,
            "observed_condition": "White roll is visible on the desk.",
            "supporting_frame_ids": ["frame:0"], "suggested_message": "The roll is on the desk.",
        }]
        result = self.parse()
        self.assertIsNone(result.output.interrupt_candidate)
        self.assertEqual(result.output.prospective_matches[0].subject_id, self.focus_id)
        self.assertEqual(result.output.prospective_matches[0].supporting_frame_ids, [self.frame.frame_id])

    def test_conditional_reminder_also_requires_explicit_generic_scope(self):
        self.request_values.update(active_focus=None, active_reminders=({"id": str(uuid4()), "title": "Find the roll"},))
        self.assertIsNone(self.parse().output.interrupt_candidate)

    def test_no_watch_preserves_legacy_generic_candidate(self):
        self.request_values["active_focus"] = None
        self.assertIsNotNone(self.parse().output.interrupt_candidate)

    def test_native_shape_teaches_the_explicit_scope_marker(self):
        request = FastPerceptionRequest(**self.request_values)
        prompt = build_fast_perception_prompt(request, references=PerceptionReferences(request))
        self.assertIn('"interrupt_candidate": null | {"scope": "general",', prompt)

    def match(self, subject_id="focus:0", **fields):
        self.payload["interrupt_candidate"] = None
        self.payload["prospective_matches"] = [{
            "subject_id": subject_id, "confidence": .9, "observed_condition": "The target is visible.",
            "supporting_frame_ids": ["frame:0"], "suggested_message": "The target is on the desk.", **fields,
        }]
        return self.parse()

    def test_native_focus_reference_supplies_internal_subject_type(self):
        result = self.match()
        self.assertEqual(result.rejected, ())
        match = result.output.prospective_matches[0]
        self.assertEqual(match.subject_type, "visual_focus")
        self.assertEqual(match.subject_id, self.focus_id)

    def test_native_reminder_reference_supplies_internal_subject_type(self):
        identifier = uuid4()
        self.request_values["active_reminders"] = ({"id": str(identifier), "title": "A separate watch"},)
        match = self.match("reminder:0").output.prospective_matches[0]
        self.assertEqual((match.subject_type, match.subject_id), ("conditional_reminder", identifier))

    def test_unique_durable_id_is_compatible_but_foreign_ids_are_not(self):
        self.assertEqual(self.match(str(self.focus_id)).output.prospective_matches[0].subject_id, self.focus_id)
        for value in (str(uuid4()), "focus:1", "frame:0", "the white roll", None):
            with self.subTest(value=value):
                result = self.match(value)
                self.assertEqual(result.output.prospective_matches, [])
                self.assertEqual(result.output.summary, self.payload["summary"])
                self.assertEqual(result.rejected[0].reason, "unknown_subject_reference")

    def test_ambiguous_durable_id_needs_an_explicit_bound_handle(self):
        self.request_values["active_reminders"] = ({"id": str(self.focus_id), "title": "Separate registry"},)
        self.assertEqual(self.match(str(self.focus_id)).rejected[0].reason, "ambiguous_subject_reference")
        self.assertEqual(self.match("focus:0").output.prospective_matches[0].subject_type, "visual_focus")
        self.assertEqual(self.match("reminder:0").output.prospective_matches[0].subject_type, "conditional_reminder")

    def test_explicit_contradictory_legacy_type_is_still_rejected(self):
        result = self.match(subject_type="conditional_reminder")
        self.assertEqual(result.output.prospective_matches, [])
        self.assertEqual(result.rejected[0].reason, "unknown_subject_reference")
        self.assertEqual(self.match(subject_type=None).rejected[0].reason, "unknown_subject_type")
        self.assertEqual(self.match(subject_type="visual_focus").output.prospective_matches[0].subject_id, self.focus_id)

    def test_removed_focus_is_not_resolved_from_old_handle_or_prose(self):
        self.request_values["active_focus"] = None
        self.assertEqual(self.match().output.prospective_matches, [])

    def test_native_shape_has_one_subject_identity_but_legacy_keeps_type(self):
        request = FastPerceptionRequest(**self.request_values)
        native = build_fast_perception_prompt(request, references=PerceptionReferences(request))
        shape = native.split("Required shape:", 1)[1].split("Context JSON:", 1)[0]
        self.assertIn('"prospective_matches": [{"subject_id": string,', shape)
        self.assertNotIn('"subject_type":', shape)
        self.assertIn(f"reference contract version {REFERENCE_CONTRACT_VERSION}", native)
        legacy = build_fast_perception_prompt(request)
        self.assertIn('"subject_type": "visual_focus"|"conditional_reminder"', legacy)

    def test_invalid_scope_is_only_an_optional_candidate_failure(self):
        for scope in ("visual_focus", True, "focus:0"):
            with self.subTest(scope=scope):
                self.payload["interrupt_candidate"]["scope"] = scope
                result = self.parse()
                self.assertEqual(result.output.summary, self.payload["summary"])
                self.assertIsNone(result.output.interrupt_candidate)
                self.assertEqual(result.rejected[0].reason, "invalid_candidate_shape")
