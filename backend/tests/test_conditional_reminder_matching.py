from __future__ import annotations

import json
import unittest
from datetime import UTC, datetime
from uuid import uuid4

from app.perception.contracts import FastPerceptionFrame, FastPerceptionRequest, parse_fast_perception_output


class ConditionalReminderMatchingTests(unittest.TestCase):
    def test_conditional_reminder_id_is_grounded_in_request(self) -> None:
        now = datetime.now(UTC)
        frame_id, reminder_id = uuid4(), uuid4()
        request = FastPerceptionRequest(
            owner_id="owner", session_id=uuid4(), window_id=uuid4(),
            frames=(FastPerceptionFrame(frame_id, 1, now, "glasses", "image/jpeg", b"x"),),
            previous_working_state={}, existing_entity_ids=frozenset(),
            recent_audio_transcripts=(), behavior_instructions="",
            active_reminders=({"id": str(reminder_id), "condition": "when brushing"},),
            active_focus=None, recent_interrupts=(), requested_at=now,
        )
        payload = {
            "summary": "User brushes teeth.", "scene": None, "activity": None,
            "entities": [], "visible_text": [], "state_changes": [], "memory_tags": [],
            "keyframe_candidates": [], "starts_new_event": False, "boundary_reason": None,
            "interrupt_candidate": None,
            "prospective_matches": [{
                "subject_type": "conditional_reminder", "subject_id": str(reminder_id),
                "confidence": 0.95, "observed_condition": "brushing teeth",
                "supporting_frame_ids": [str(frame_id)], "suggested_message": "Remember to floss.",
            }],
            "confidence": 0.95,
        }
        self.assertEqual(
            parse_fast_perception_output(json.dumps(payload), request=request)
            .prospective_matches[0]
            .subject_id,
            reminder_id,
        )
