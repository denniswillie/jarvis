from __future__ import annotations

import unittest
from datetime import UTC, datetime
from uuid import uuid4

from app.perception.models import (
    KeyframeCandidate,
    ObservationEntity,
    ObservationStateChange,
    VisibleTextObservation,
    VisualObservation,
    deserialize_entities,
    deserialize_keyframe_candidates,
    deserialize_state_changes,
    deserialize_visible_text,
    deserialize_working_memory_state,
    serialize_entities,
    serialize_keyframe_candidates,
    serialize_state_changes,
    serialize_visible_text,
    serialize_working_memory_state,
)


class PerceptionModelTests(unittest.TestCase):
    def test_structured_observation_fields_round_trip(self) -> None:
        frame_id = uuid4()
        entities = (
            ObservationEntity("coffee mug", "object", 0.91, {"color": "blue"}),
        )
        visible_text = (VisibleTextObservation("EXIT", "wall sign", 0.99),)
        state_changes = (
            ObservationStateChange("door", "state", "closed", "open", 0.82),
        )
        keyframes = (KeyframeCandidate(frame_id, "clearest view", 0.88),)

        self.assertEqual(deserialize_entities(serialize_entities(entities)), entities)
        self.assertEqual(
            deserialize_visible_text(serialize_visible_text(visible_text)),
            visible_text,
        )
        self.assertEqual(
            deserialize_state_changes(serialize_state_changes(state_changes)),
            state_changes,
        )
        self.assertEqual(
            deserialize_keyframe_candidates(serialize_keyframe_candidates(keyframes)),
            keyframes,
        )

    def test_working_memory_state_carries_contract_version(self) -> None:
        encoded = serialize_working_memory_state({"scene": "kitchen"}, contract_version=2)
        version, decoded = deserialize_working_memory_state(encoded)

        self.assertEqual(version, 2)
        self.assertEqual(decoded, {"scene": "kitchen"})

    def test_empty_detailed_observation_is_not_compacted(self) -> None:
        observation = _observation(compacted_at=None)
        self.assertFalse(observation.is_compacted)

    def test_compacted_observation_rejects_retained_bulky_detail(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot retain"):
            _observation(
                compacted_at=datetime.now(UTC),
                entities=(ObservationEntity("mug", "object", 0.9, {}),),
            )

    def test_observation_rejects_invalid_time_range(self) -> None:
        now = datetime.now(UTC)
        with self.assertRaisesRegex(ValueError, "cannot be before"):
            _observation(started_at=now, ended_at=now.replace(year=now.year - 1))

    def test_observation_rejects_non_finite_confidence(self) -> None:
        with self.assertRaisesRegex(ValueError, "between zero and one"):
            _observation(confidence=float("nan"))


def _observation(**overrides: object) -> VisualObservation:
    now = datetime.now(UTC)
    values: dict[str, object] = {
        "id": uuid4(),
        "owner_id": "owner",
        "session_id": uuid4(),
        "source": "glasses",
        "started_at": now,
        "ended_at": now,
        "primary_visual_frame_id": None,
        "summary": "A quiet room.",
        "scene": "room",
        "activity": None,
        "entities": (),
        "visible_text": (),
        "state_changes": (),
        "memory_tags": (),
        "keyframe_candidates": (),
        "interrupt_candidate": None,
        "starts_new_event": False,
        "boundary_reason": None,
        "confidence": 0.9,
        "analyzer_provider": "test",
        "analyzer_model": "test-model",
        "analyzer_contract_version": 1,
        "compacted_at": None,
        "created_at": now,
    }
    values.update(overrides)
    return VisualObservation(**values)  # type: ignore[arg-type]
