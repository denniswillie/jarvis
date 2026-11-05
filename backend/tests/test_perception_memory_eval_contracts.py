from __future__ import annotations

import hashlib
import importlib.resources
import unittest
from pathlib import Path

from evals.perception_memory.baseline_runner import (
    DEFAULT_MANIFEST_PATH,
    DEFAULT_SCENARIOS_PATH,
    load_manifest,
    load_suite,
    run_baseline,
)
from evals.perception_memory.deterministic_pipeline import evaluate_deterministic_scenario


class PerceptionMemoryEvalContractTests(unittest.TestCase):
    def test_suite_covers_required_baseline_behaviors(self) -> None:
        suite = load_suite(DEFAULT_SCENARIOS_PATH)
        scenario_ids = {scenario.id for scenario in suite.scenarios}
        self.assertTrue(
            {
                "static_scene_current_view",
                "small_text_reading",
                "three_frame_action",
                "scene_change",
                "last_seen_object",
                "recall_one_minute",
                "recall_five_minutes",
                "recall_thirty_minutes",
                "recall_after_jpeg_expiry",
                "omitted_caption_detail",
                "conditional_floss_reminder",
                "duplicate_interruption_suppression",
                "explicit_visual_focus",
                "conflicting_object_location",
                "audio_only_session",
                "voice_only_memory",
                "tool_mutation_recall",
                "mixed_speech_and_visual",
                "capture_latest_frame",
            }.issubset(scenario_ids)
        )

    def test_fixture_references_and_hashes_are_stable(self) -> None:
        suite = load_suite(DEFAULT_SCENARIOS_PATH)
        manifest = load_manifest(DEFAULT_MANIFEST_PATH)
        entries = {entry.id: entry for entry in manifest.entries}
        for scenario in suite.scenarios:
            for fixture_id in scenario.input_frame_fixture_ids:
                self.assertIn(fixture_id, entries)
        for entry in manifest.entries:
            fixture_path = DEFAULT_MANIFEST_PATH.parent / entry.path
            self.assertTrue(fixture_path.is_file())
            self.assertEqual(
                hashlib.sha256(fixture_path.read_bytes()).hexdigest(),
                entry.sha256,
            )

    def test_deterministic_baseline_preserves_known_failures(self) -> None:
        report = run_baseline()
        self.assertEqual(report.aggregate.scenario_count, len(report.scenarios))
        self.assertEqual(
            {result.scenario_id for result in report.scenarios if not result.passed},
            {"omitted_caption_detail", "voice_only_memory"},
        )
        self.assertEqual(report.metadata["deterministic_runs_mutate_production_state"], False)
        self.assertIsNotNone(report.aggregate.current_view_accuracy)
        self.assertIsNotNone(report.aggregate.temporal_accuracy)
        self.assertIsNotNone(report.aggregate.recall_accuracy)
        self.assertIsNotNone(report.aggregate.frame_analysis_p50_lag_ms)
        self.assertIsNotNone(report.aggregate.frame_analysis_p95_lag_ms)
        self.assertIsNotNone(report.aggregate.memory_level_accuracy)
        self.assertGreater(report.aggregate.peak_visual_backlog, 0)
        self.assertIn("gemini", report.aggregate.provider_calls_per_minute)

    def test_eval_data_is_packaged(self) -> None:
        package = importlib.resources.files("evals.perception_memory")
        self.assertTrue(package.joinpath("scenarios.json").is_file())
        self.assertTrue(package.joinpath("fixtures/manifest.json").is_file())
        self.assertTrue(package.joinpath("fixtures/images/desk_static.jpg").is_file())


class PerceptionMemoryStatefulEvalTests(unittest.IsolatedAsyncioTestCase):
    async def test_companion_current_image_trial_records_delivery_not_fake_analysis(self) -> None:
        suite = load_suite(Path(__file__).resolve().parents[1] / "evals/perception_memory/companion_quality_scenarios.json")
        scenario = next(item for item in suite.scenarios if item.id == "shared_view_precision_escalation")
        outcome = await evaluate_deterministic_scenario(
            scenario=scenario, manifest=load_manifest(DEFAULT_MANIFEST_PATH),
            manifest_path=DEFAULT_MANIFEST_PATH, simulate_legacy_gaps=False,
        )
        trace = next(event for event in outcome.diagnostic_events if event["type"] == "shared_view.response_cutoff")
        self.assertEqual(outcome.observation.tool_calls, ["get_current_camera_view"])
        self.assertEqual(trace["requested_image_calls"], 1)
        self.assertEqual(trace["image_delivery_status"], "attached")
        self.assertEqual(trace["secondary_vision_calls"], 0)
        self.assertTrue(trace["cutoff"]["requested_images"])
        self.assertTrue(trace["answer_is_fixture"])
        self.assertFalse(trace["model_reasoning_exercised"])

    async def test_pipeline_executes_stateful_regressions(self) -> None:
        suite = load_suite(DEFAULT_SCENARIOS_PATH)
        scenarios = {scenario.id: scenario for scenario in suite.scenarios}
        manifest = load_manifest(DEFAULT_MANIFEST_PATH)

        expiry = await evaluate_deterministic_scenario(
            scenario=scenarios["recall_after_jpeg_expiry"],
            manifest=manifest,
            manifest_path=DEFAULT_MANIFEST_PATH,
        )
        self.assertTrue(expiry.observation.raw_image_expired)
        self.assertGreater(expiry.observation.search_candidate_count, 0)
        self.assertGreater(expiry.observation.search_returned_description_count, 0)
        self.assertTrue(any(event["type"] == "visual.frame.analyzed" for event in expiry.diagnostic_events))

        dedupe = await evaluate_deterministic_scenario(
            scenario=scenarios["duplicate_interruption_suppression"],
            manifest=manifest,
            manifest_path=DEFAULT_MANIFEST_PATH,
        )
        self.assertEqual(dedupe.observation.interrupts, ["Remember to floss."])

        omitted = await evaluate_deterministic_scenario(
            scenario=scenarios["omitted_caption_detail"],
            manifest=manifest,
            manifest_path=DEFAULT_MANIFEST_PATH,
        )
        self.assertNotIn("12 ml", " ".join(omitted.observation.frame_descriptions).casefold())
        self.assertNotIn("12 ml", omitted.observation.answer.casefold())

        restart = await evaluate_deterministic_scenario(
            scenario=scenarios["voice_only_memory"],
            manifest=manifest,
            manifest_path=DEFAULT_MANIFEST_PATH,
        )
        self.assertEqual(restart.observation.session_restart_count, 1)
        self.assertNotIn("4812", restart.observation.answer)


if __name__ == "__main__":
    unittest.main()
