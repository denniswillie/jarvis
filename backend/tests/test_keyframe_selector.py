from __future__ import annotations

import unittest
from datetime import UTC, datetime, timedelta
from uuid import UUID

from app.experience.keyframes import (
    DeterministicKeyframeSelector,
    EventKeyframeCandidate,
    KeyframeRetentionPolicy,
)


class DeterministicKeyframeSelectorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.selector = DeterministicKeyframeSelector()
        self.started_at = datetime(2026, 8, 21, 12, 0, tzinfo=UTC)

    def test_selection_is_deterministic_and_bounded(self) -> None:
        candidates = tuple(self._candidate(index) for index in range(20))

        first = self.selector.select(candidates, limit=6)
        second = self.selector.select(tuple(reversed(candidates)), limit=6)

        self.assertEqual(len(first), 6)
        self.assertEqual(
            [item.candidate.visual_frame_id for item in first],
            [item.candidate.visual_frame_id for item in second],
        )

    def test_readable_text_state_and_entities_raise_selection_value(self) -> None:
        plain = self._candidate(1, analyzer_score=0.7)
        rich = self._candidate(
            2,
            analyzer_score=0.7,
            has_readable_text=True,
            has_state_change=True,
            entity_count=4,
        )

        selected = self.selector.select((plain, rich), limit=1)

        self.assertEqual(selected[0].candidate.visual_frame_id, rich.visual_frame_id)

    def test_repetition_penalty_and_temporal_diversity_avoid_near_duplicates(self) -> None:
        candidates = (
            self._candidate(1, reason="label visible", analyzer_score=0.98),
            self._candidate(2, reason="label visible", analyzer_score=0.97),
            self._candidate(10, reason="outcome visible", analyzer_score=0.86),
        )

        selected = self.selector.select(candidates, limit=2)

        self.assertIn(
            candidates[2].visual_frame_id,
            {item.candidate.visual_frame_id for item in selected},
        )

    def test_duplicate_frame_is_returned_once(self) -> None:
        frame = self._candidate(1, analyzer_score=0.5)
        stronger = EventKeyframeCandidate(
            visual_frame_id=frame.visual_frame_id,
            captured_at=frame.captured_at,
            reason="clear text",
            analyzer_score=0.95,
            observation_confidence=frame.observation_confidence,
            has_readable_text=True,
            has_state_change=False,
            entity_count=0,
        )

        selected = self.selector.select((frame, stronger), limit=3)

        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0].candidate.reason, "clear text")

    def test_policy_rejects_inverted_limits(self) -> None:
        with self.assertRaises(ValueError):
            KeyframeRetentionPolicy(
                final_keyframe_max_count=4,
                consolidation_candidate_max_count=3,
            )

    def _candidate(
        self,
        index: int,
        *,
        reason: str = "scene context",
        analyzer_score: float = 0.8,
        has_readable_text: bool = False,
        has_state_change: bool = False,
        entity_count: int = 0,
    ) -> EventKeyframeCandidate:
        return EventKeyframeCandidate(
            visual_frame_id=UUID(int=index + 1),
            captured_at=self.started_at + timedelta(seconds=index),
            reason=reason,
            analyzer_score=analyzer_score,
            observation_confidence=0.85,
            has_readable_text=has_readable_text,
            has_state_change=has_state_change,
            entity_count=entity_count,
        )
