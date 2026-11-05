from __future__ import annotations

import unittest
from datetime import UTC, datetime
from uuid import uuid4

from app.assistant.base_prompt import build_base_prompt
from app.assistant.context import AssistantContext
from app.assistant.context_providers import BehaviorInstructions
from app.realtime.models import RealtimeResponseState
from app.realtime.output_policy import RealtimeOutputItemState
from app.sessions.models import JarvisMessage


class OutputPolicyTests(unittest.TestCase):
    def test_commentary_suppresses_media_without_retaining_its_text(self):
        # The default per-item policy still suppresses commentary. This is not
        # a claim that every selected native voice route suppresses it.
        item = RealtimeOutputItemState("commentary")
        self.assertIsNone(item.observe_phase("commentary"))
        self.assertFalse(item.record_generated())
        self.assertEqual(item.delivery_disposition, "suppressed")
        self.assertFalse(item.has_forwarded_answer)
        self.assertEqual(item.generated_media_events, 1)
        self.assertEqual(item.suppressed_media_events, 1)
        self.assertEqual(item.forwarded_media_events, 0)
        self.assertEqual(item.transcript, "")

    def test_final_and_unknown_phase_preserve_streaming(self):
        for phase in ("final_answer", None, "future_phase"):
            with self.subTest(phase=phase):
                item = RealtimeOutputItemState("answer")
                notice = item.observe_phase(phase)
                self.assertEqual(notice, "unknown_phase" if phase == "future_phase" else None)
                self.assertTrue(item.record_generated())
                item.record_forwarded(audio_chunk=True)
                self.assertTrue(item.has_forwarded_answer)
                self.assertEqual(item.forwarded_media_events, 1)
                self.assertEqual(item.suppressed_media_events, 0)

    def test_late_commentary_does_not_cut_forwarded_utterance(self):
        item = RealtimeOutputItemState("late")
        item.record_forwarded(audio_chunk=True)
        self.assertEqual(item.observe_phase("commentary"), "late_commentary_phase")
        self.assertTrue(item.allows_output)
        self.assertTrue(item.phase_anomaly)
        self.assertEqual(item.delivery_disposition, "forwarded")
        # It was forwarded, but is not silently relabelled a successful final answer.
        self.assertFalse(item.has_forwarded_answer)

    def test_suppression_cannot_be_undone_by_conflicting_metadata(self):
        item = RealtimeOutputItemState("suppressed")
        item.observe_phase("commentary")
        self.assertEqual(item.observe_phase("final_answer"), "conflicting_phase")
        self.assertFalse(item.allows_output)
        self.assertEqual(item.phase, "commentary")
        self.assertTrue(item.phase_anomaly)

    def test_final_phase_is_stable_across_conflicting_or_missing_metadata(self):
        item = RealtimeOutputItemState("final")
        item.observe_phase("final_answer")
        item.record_forwarded(audio_chunk=True)
        self.assertIsNone(item.observe_phase(None))
        self.assertEqual(item.observe_phase("commentary"), "conflicting_phase")
        self.assertEqual(item.phase, "final_answer")
        self.assertTrue(item.has_forwarded_answer)

    def test_done_envelope_without_content_is_not_a_forwarded_answer(self):
        item = RealtimeOutputItemState("empty")
        item.observe_phase("final_answer")
        item.record_forwarded()
        self.assertFalse(item.has_forwarded_answer)

    def test_order_and_phase_are_per_item_not_per_response(self):
        state = RealtimeResponseState(response_id="response", kind="normal_user_turn")
        for item_id, index, phase in (("second", 2, "final_answer"), ("intro", 0, "commentary"), ("first", 1, "final_answer")):
            item = RealtimeOutputItemState(item_id, output_index=index)
            item.observe_phase(phase)
            state.output_items[item_id] = item
        self.assertEqual([item.item_id for item in state.ordered_output_items], ["intro", "first", "second"])
        self.assertFalse(state.output_items["intro"].allows_output)
        self.assertTrue(state.output_items["first"].allows_output)

    def test_metadata_excludes_raw_transcript_and_playback_claims(self):
        item = RealtimeOutputItemState("item")
        item.transcript = "PRIVATE_OUTPUT_TEXT"
        item.record_forwarded()
        metadata = item.as_metadata()
        self.assertEqual(metadata["delivery_disposition"], "forwarded")
        self.assertNotIn("PRIVATE_OUTPUT_TEXT", str(metadata))
        self.assertNotIn("heard", metadata)
        self.assertNotIn("played", metadata)


class OutputHistoryTests(unittest.TestCase):
    def message(self, content, *, phase=None, disposition=None, role="assistant"):
        metadata = {}
        if phase is not None or disposition is not None:
            metadata["realtime_output"] = {
                "phase": phase, "delivery_disposition": disposition,
            }
        return JarvisMessage(
            id=uuid4(), session_id=uuid4(), role=role, modality="audio",
            content=content, provider_item_id=None, provider_response_id=None,
            created_at=datetime.now(UTC), metadata=metadata,
        )

    def test_prompt_excludes_suppressed_output_but_keeps_forwarded_and_legacy_messages(self):
        messages = [
            self.message("hidden", phase="commentary", disposition="suppressed"),
            self.message("result", phase="final_answer", disposition="forwarded"),
            self.message("legacy"),
            self.message("late phase but forwarded", phase="commentary", disposition="forwarded"),
            self.message("user request", phase="commentary", disposition="suppressed", role="user"),
        ]
        context = AssistantContext(
            current_utc_time=datetime.now(UTC), session=None,
            behavior=BehaviorInstructions("", ""), recent_messages=messages,
            active_reminders=[], preferences=[], recent_visual_interrupts=[],
        )
        payload = context.as_prompt_payload()["recent_messages"]
        self.assertEqual([item["content"] for item in payload], [
            "result", "legacy", "late phase but forwarded", "user request",
        ])
        self.assertEqual(payload[0]["realtime_output"], {
            "phase": "final_answer", "delivery_disposition": "forwarded",
        })
        self.assertNotIn("realtime_output", payload[1])
        self.assertNotIn("realtime_output", payload[3])

    def test_prompt_keeps_required_information_in_final_answers(self):
        prompt = build_base_prompt()
        self.assertIn("Commentary may be audible on the active voice route", prompt)
        self.assertIn("Do not rely on commentary to convey essential content", prompt)
        self.assertIn("tool results or failures, necessary qualifications, and safety detail", prompt)
        self.assertIn("in final_answer messages", prompt)
        self.assertIn("avoid routine preparation narration and do not duplicate the final answer", prompt)
        self.assertIn("not confirmed heard", prompt)
        self.assertNotIn("suppresses commentary-phase speech", prompt)
