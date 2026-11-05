from __future__ import annotations

import unittest
from datetime import UTC, datetime, timedelta
from uuid import uuid4

from app.perception.models import VisualObservation
from app.perception.working_memory import WorkingMemoryPolicy, reduce_session_working_memory


NOW = datetime(2026, 8, 21, 12, 0, tzinfo=UTC)
POLICY = WorkingMemoryPolicy(120, 30, 60, 30, 12)


class WorkingMemoryRegressionTests(unittest.TestCase):
    def test_low_confidence_labels_do_not_corrupt_compatibility_aliases(self) -> None:
        previous = reduce_session_working_memory(
            {},
            _observation("kitchen", "making tea", 0.95, 0.95, NOW),
            policy=POLICY,
        )
        reduced = reduce_session_working_memory(
            previous,
            _observation(
                "garage", "driving", 0.2, 0.2, NOW + timedelta(seconds=1)
            ),
            policy=POLICY,
        )

        self.assertEqual(reduced["scene"]["label"], "kitchen")
        self.assertEqual(reduced["activity"]["label"], "making tea")
        self.assertEqual(reduced["location"], "kitchen")
        self.assertEqual(reduced["current_activity"], "making tea")

    def test_scene_and_activity_use_their_own_confidence(self) -> None:
        previous = reduce_session_working_memory(
            {},
            _observation("kitchen", "making tea", 0.95, 0.95, NOW),
            policy=POLICY,
        )
        reduced = reduce_session_working_memory(
            previous,
            _observation(
                "office", "typing", 0.9, 0.3, NOW + timedelta(seconds=1)
            ),
            policy=POLICY,
        )

        self.assertEqual(reduced["scene"]["label"], "office")
        self.assertEqual(reduced["activity"]["label"], "making tea")


def _observation(
    scene: str,
    activity: str,
    scene_confidence: float,
    activity_confidence: float,
    ended_at: datetime,
) -> VisualObservation:
    return VisualObservation(
        id=uuid4(), owner_id="owner", session_id=uuid4(), source="glasses",
        started_at=ended_at, ended_at=ended_at, primary_visual_frame_id=None,
        summary="Observed a scene.", scene=scene, activity=activity, entities=(),
        visible_text=(), state_changes=(), memory_tags=(), keyframe_candidates=(),
        interrupt_candidate=None, starts_new_event=False, boundary_reason=None,
        confidence=0.99, analyzer_provider="test", analyzer_model="test",
        analyzer_contract_version=1, compacted_at=None, created_at=ended_at,
        scene_confidence=scene_confidence, activity_confidence=activity_confidence,
    )
