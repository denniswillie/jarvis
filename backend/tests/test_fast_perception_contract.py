from __future__ import annotations

import json
import unittest
from datetime import UTC, datetime, timedelta
from uuid import uuid4

from app.perception.contracts import (
    FastPerceptionFrame,
    FastPerceptionOutputError,
    FastPerceptionRequest,
    parse_fast_perception_output,
)


NOW = datetime(2026, 8, 16, 12, 0, tzinfo=UTC)


class FastPerceptionContractTests(unittest.TestCase):
    def test_valid_temporal_output_is_accepted(self) -> None:
        request = _request()
        output = parse_fast_perception_output(
            json.dumps(_payload(request)),
            request=request,
        )

        self.assertEqual(output.activity.label, "lifting a mug")
        self.assertEqual(output.memory_tags, ["mug", "desk"])

    def test_unknown_frame_reference_is_rejected(self) -> None:
        request = _request()
        payload = _payload(request)
        payload["keyframe_candidates"][0]["frame_id"] = str(uuid4())

        with self.assertRaisesRegex(FastPerceptionOutputError, "outside"):
            parse_fast_perception_output(json.dumps(payload), request=request)

    def test_unknown_existing_entity_is_rejected(self) -> None:
        request = _request()
        payload = _payload(request)
        payload["entities"][0]["existing_entity_id"] = "unknown"

        with self.assertRaisesRegex(FastPerceptionOutputError, "unknown existing"):
            parse_fast_perception_output(json.dumps(payload), request=request)

    def test_unknown_output_field_is_rejected(self) -> None:
        request = _request()
        payload = _payload(request)
        payload["invented"] = True

        with self.assertRaisesRegex(FastPerceptionOutputError, "contract validation"):
            parse_fast_perception_output(json.dumps(payload), request=request)


def _request() -> FastPerceptionRequest:
    first = FastPerceptionFrame(uuid4(), 1, NOW, "glasses", "image/jpeg", b"one")
    second = FastPerceptionFrame(
        uuid4(),
        2,
        NOW + timedelta(seconds=1),
        "glasses",
        "image/jpeg",
        b"two",
    )
    return FastPerceptionRequest(
        owner_id="owner",
        session_id=uuid4(),
        window_id=uuid4(),
        frames=(first, second),
        previous_working_state={"entities": [{"entity_id": "mug-1"}]},
        existing_entity_ids=frozenset({"mug-1"}),
        recent_audio_transcripts=(),
        behavior_instructions="Be concise.",
        active_reminders=(),
        active_focus=None,
        recent_interrupts=(),
        requested_at=NOW,
    )


def _payload(request: FastPerceptionRequest) -> dict:
    return {
        "summary": "The user lifts a mug from the desk.",
        "scene": {"label": "office desk", "confidence": 0.94},
        "activity": {"label": "lifting a mug", "confidence": 0.91},
        "entities": [
            {
                "existing_entity_id": "mug-1",
                "new_local_key": None,
                "type": "object",
                "name": "mug",
                "state": "lifted",
                "attributes": {"color": "blue"},
                "last_visible_frame_id": str(request.primary_frame.frame_id),
                "confidence": 0.95,
            }
        ],
        "visible_text": [],
        "state_changes": [
            {
                "subject_local_key": "mug-1",
                "change": "moved from desk to hand",
                "confidence": 0.9,
            }
        ],
        "memory_tags": ["Mug", "desk", "mug"],
        "keyframe_candidates": [
            {
                "frame_id": str(request.primary_frame.frame_id),
                "reason": "clearest final state",
                "score": 0.88,
            }
        ],
        "starts_new_event": False,
        "boundary_reason": None,
        "interrupt_candidate": None,
        "confidence": 0.92,
    }
