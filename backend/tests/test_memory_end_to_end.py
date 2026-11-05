from __future__ import annotations

import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

from evals.perception_memory.new_architecture_runner import (
    DEFAULT_BASELINE_PATH,
    run_architecture_evaluation,
)
from app.perception.frame_window import RecentVisualFrameWindow, SensoryVisualFrame


class MemoryArchitectureEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.evaluation = run_architecture_evaluation()

    def test_post_cutover_suite_passes_explicit_gates(self) -> None:
        self.assertTrue(self.evaluation.assessment.passed)
        self.assertEqual(
            self.evaluation.report.aggregate.passed_count,
            self.evaluation.report.aggregate.scenario_count,
        )

    def test_post_cutover_suite_does_not_regress_frozen_baseline(self) -> None:
        self.assertTrue(DEFAULT_BASELINE_PATH.exists())
        self.assertIsNotNone(self.evaluation.baseline_comparison)
        self.assertTrue(self.evaluation.baseline_comparison.passed)

    def test_report_marks_legacy_caption_search_as_disabled(self) -> None:
        self.assertFalse(
            self.evaluation.report.metadata["raw_frame_caption_search_enabled"]
        )


class SensoryDiagnosticsTests(unittest.TestCase):
    def test_diagnostics_are_bounded_and_do_not_expose_frame_bytes(self) -> None:
        clock = [10.0]
        window = RecentVisualFrameWindow(
            max_frames_per_session=2,
            ttl_seconds=5,
            monotonic=lambda: clock[0],
        )
        owner_id = "diagnostic-owner"
        session_id = uuid4()
        now = datetime.now(UTC)
        for sequence in range(3):
            window.put(
                SensoryVisualFrame(
                    frame_id=uuid4(),
                    owner_id=owner_id,
                    session_id=session_id,
                    audio_session_id=None,
                    visual_session_id=uuid4(),
                    frame_sequence=sequence,
                    source="glasses",
                    captured_at=now + timedelta(seconds=sequence),
                    server_received_at=now + timedelta(seconds=sequence),
                    content_type="image/jpeg",
                    width=2,
                    height=2,
                    storage_bucket="test",
                    storage_path=f"frames/{sequence}.jpg",
                    raw_image_expires_at=now + timedelta(minutes=5),
                    image_bytes=b"secret-frame-bytes",
                )
            )

        diagnostics = window.diagnostics()
        self.assertEqual(diagnostics["session_count"], 1)
        self.assertEqual(diagnostics["frame_count"], 2)
        self.assertNotIn("image_bytes", diagnostics)
        self.assertNotIn("storage_path", diagnostics)

        clock[0] = 16.0
        self.assertEqual(window.diagnostics()["frame_count"], 0)
