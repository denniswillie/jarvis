from __future__ import annotations

import json
import unittest
from datetime import UTC, datetime
from uuid import uuid4

from app.perception.contracts import (
    FastPerceptionFrame,
    FastPerceptionOutputError,
    FastPerceptionRequest,
    parse_fast_perception_output,
)


class VisualFocusMatchingTests(unittest.TestCase):
    def test_candidate_must_reference_supplied_focus_and_frame(self) -> None:
        frame_id = uuid4()
        focus_id = uuid4()
        request = _request(frame_id, focus_id)
        output = parse_fast_perception_output(
            json.dumps(_payload(frame_id, focus_id)),
            request=request,
        )
        self.assertEqual(output.prospective_matches[0].subject_id, focus_id)

        payload = _payload(frame_id, uuid4())
        with self.assertRaises(FastPerceptionOutputError):
            parse_fast_perception_output(json.dumps(payload), request=request)


def _request(frame_id, focus_id):
    now = datetime.now(UTC)
    return FastPerceptionRequest(
        owner_id="owner", session_id=uuid4(), window_id=uuid4(),
        frames=(FastPerceptionFrame(frame_id, 1, now, "glasses", "image/jpeg", b"x"),),
        previous_working_state={}, existing_entity_ids=frozenset(),
        recent_audio_transcripts=(), behavior_instructions="", active_reminders=(),
        active_focus={"id": str(focus_id)}, recent_interrupts=(), requested_at=now,
    )


def _payload(frame_id, focus_id):
    return {
        "summary": "Keys are visible.", "scene": None, "activity": None,
        "entities": [], "visible_text": [], "state_changes": [], "memory_tags": [],
        "keyframe_candidates": [], "starts_new_event": False, "boundary_reason": None,
        "interrupt_candidate": None,
        "prospective_matches": [{
            "subject_type": "visual_focus", "subject_id": str(focus_id),
            "confidence": 0.95, "observed_condition": "keys visible",
            "supporting_frame_ids": [str(frame_id)], "suggested_message": "Your keys are here.",
        }],
        "confidence": 0.95,
    }
