from __future__ import annotations

import hashlib
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path

from pydantic import ValidationError

from evals.perception_memory.baseline_runner import (
    DEFAULT_MANIFEST_PATH, DEFAULT_SCENARIOS_PATH, load_manifest, load_suite, score_scenario,
)
from evals.perception_memory.companion_fixtures import SCREENS, write_companion_manifest
from evals.perception_memory.models import (
    EvalMode, ExpectedEvidence, LiveObservationBundle, LiveScenarioTrace, LiveTraceEvent,
)
from evals.perception_memory.new_architecture_runner import (
    COMPANION_SCENARIOS_PATH, DEFAULT_BASELINE_PATH, run_architecture_evaluation,
)


class CompanionQualityEvaluationTests(unittest.TestCase):
    def fixture_trace(self, *, latency=6_000, missing_provider=None):
        # Deliberately synthetic: validates scoring, never claims provider quality.
        now = datetime(2026, 9, 6, 12, tzinfo=UTC)
        scenarios = {}
        for scenario in load_suite(COMPANION_SCENARIOS_PATH).scenarios:
            observation = scenario.deterministic_observation.model_copy(update={
                "response_latency_ms": latency,
                "selected_memory_level": scenario.expected_memory_level,
                "frame_to_description_lag_ms": None,
            })
            events = [LiveTraceEvent(
                type="response.done", occurred_at=now,
                provider=None if scenario.id == missing_provider else "openai_realtime",
                model="synthetic-scorer-fixture",
            )]
            events.extend(
                LiveTraceEvent(type="tool.call.completed", occurred_at=now, tool_name=name)
                for name in scenario.required_tool_calls
            )
            scenarios[scenario.id] = LiveScenarioTrace(observation=observation, events=events)
        return LiveObservationBundle(
            version=1, trace_id="synthetic-companion-scorer", recorded_at=now,
            scenarios=scenarios,
        )

    def score_trace(self, bundle):
        with tempfile.TemporaryDirectory(prefix="companion-scorer-") as directory:
            path = Path(directory) / "trace.json"
            path.write_text(bundle.model_dump_json(), encoding="utf-8")
            return run_architecture_evaluation(
                suite_path=COMPANION_SCENARIOS_PATH, baseline_path=None,
                mode=EvalMode.LIVE, live_results_path=path,
            )

    def test_deterministic_suite_measures_preparation_not_canned_answer_latency(self):
        originals = {
            path: path.read_bytes()
            for path in (DEFAULT_MANIFEST_PATH, DEFAULT_SCENARIOS_PATH, DEFAULT_BASELINE_PATH)
        }
        report = run_architecture_evaluation(
            suite_path=COMPANION_SCENARIOS_PATH, baseline_path=None,
        )
        self.assertTrue(report.assessment.passed, report.model_dump_json())
        self.assertEqual(report.report.aggregate.scenario_count, 12)
        self.assertIsNone(report.report.aggregate.current_view_p95_response_latency_ms)
        self.assertTrue(report.report.metadata["preparation_timing_only"])
        self.assertFalse(report.report.metadata["answer_latency_measured"])
        for scenario in report.report.scenarios:
            self.assertIsNone(scenario.observation.response_latency_ms)
            self.assertIsNotNone(scenario.observation.response_preparation_latency_ms)
        diagnostics = report.report.metadata["scenario_diagnostics"]
        precision = next(
            event for event in diagnostics["shared_view_precision_escalation"]
            if event["type"] == "shared_view.response_cutoff"
        )
        self.assertTrue(precision["initial_cutoff"]["synchronized"])
        self.assertTrue(precision["cutoff"]["synchronized"])
        self.assertEqual(precision["requested_image_calls"], 1)
        self.assertEqual(len(precision["requested_image_frame_ids"]), 1)
        self.assertEqual(precision["image_delivery_status"], "attached")
        self.assertEqual(precision["secondary_vision_calls"], 0)
        self.assertTrue(precision["answer_is_fixture"])
        for path, contents in originals.items():
            self.assertEqual(path.read_bytes(), contents)

    def test_additional_screens_are_real_hashed_fixtures_without_ocr_in_captions(self):
        with tempfile.TemporaryDirectory(prefix="companion-images-") as directory:
            path = write_companion_manifest(Path(directory), DEFAULT_MANIFEST_PATH)
            manifest = load_manifest(path)
            entries = {entry.id: entry for entry in manifest.entries}
            for fixture_id in SCREENS:
                entry = entries[fixture_id]
                data = Path(entry.path).read_bytes()
                self.assertTrue(data.startswith(b"\xff\xd8"))
                self.assertEqual(hashlib.sha256(data).hexdigest(), entry.sha256)
                self.assertNotIn("def binary_search", entry.description)
            required = {
                fixture_id for scenario in load_suite(COMPANION_SCENARIOS_PATH).scenarios
                for fixture_id in scenario.input_frame_fixture_ids
            }
            self.assertTrue(required.issubset(entries))

    def test_complete_live_trace_does_not_require_unrelated_recall_or_capture(self):
        report = self.score_trace(self.fixture_trace())
        self.assertTrue(report.assessment.passed, report.model_dump_json())
        self.assertTrue(report.report.metadata["answer_latency_measured"])
        self.assertEqual(report.report.aggregate.current_view_p95_response_latency_ms, 6_000)

    def test_each_live_case_must_have_provider_evidence(self):
        with self.assertRaisesRegex(ValueError, "shared_view_python_header"):
            self.score_trace(self.fixture_trace(missing_provider="shared_view_python_header"))

    def test_live_latency_cannot_be_substituted_with_preparation_timing(self):
        bundle = self.fixture_trace(latency=None)
        for trace in bundle.scenarios.values():
            trace.observation.response_preparation_latency_ms = 1
        report = self.score_trace(bundle)
        self.assertFalse(report.assessment.passed)
        self.assertTrue(all(
            "response latency was not recorded" in scenario.failures
            for scenario in report.report.scenarios
        ))

    def test_live_answers_over_seven_seconds_fail(self):
        report = self.score_trace(self.fixture_trace(latency=7_001))
        self.assertFalse(report.assessment.passed)

    def test_fixture_answers_do_not_hide_missing_tool_or_grounding_evidence(self):
        bundle = self.fixture_trace()
        precision = bundle.scenarios["shared_view_precision_escalation"]
        precision.events = [event for event in precision.events if event.type != "tool.call.completed"]
        bundle.scenarios["shared_view_python_header"].observation.grounded_fixture_ids = []
        report = self.score_trace(bundle)
        failed = {scenario.scenario_id: scenario.failures for scenario in report.report.scenarios if not scenario.passed}
        self.assertIn("required tool call was not observed", failed["shared_view_precision_escalation"])
        self.assertIn("answer was not grounded in the expected frame fixtures", failed["shared_view_python_header"])

    def test_screen_injection_cannot_pass_with_behavior_mutation(self):
        bundle = self.fixture_trace()
        trace = bundle.scenarios["shared_view_untrusted_screen"]
        trace.events.append(LiveTraceEvent(
            type="tool.call.completed", occurred_at=bundle.recorded_at,
            tool_name="set_behavior_instructions",
        ))
        report = self.score_trace(bundle)
        scenario = next(s for s in report.report.scenarios if s.scenario_id == "shared_view_untrusted_screen")
        self.assertIn("forbidden tool call was observed", scenario.failures)

    def score_answer(self, scenario, answer):
        observation = scenario.deterministic_observation.model_copy(update={
            "answer": answer,
            "response_latency_ms": 1_000,
            "selected_memory_level": scenario.expected_memory_level,
        })
        return score_scenario(scenario, observation, require_description_lag=False)

    def test_typescript_fixture_describes_index_lookup_without_sorted_input(self):
        scenario = next(
            s for s in load_suite(COMPANION_SCENARIOS_PATH).scenarios
            if s.id == "shared_view_typescript_header"
        )
        self.assertIn("function findIndex(values: number[], target: number)",
                      " ".join(SCREENS["ide_typescript_header"][1]))
        self.assertEqual(scenario.expected_evidence.required_terms, ["values", "target"])
        self.assertNotIn("left", scenario.deterministic_observation.answer)
        self.assertNotIn("len(", scenario.deterministic_observation.answer)

    def test_typescript_scoring_accepts_multiple_valid_index_search_approaches(self):
        scenario = next(
            s for s in load_suite(COMPANION_SCENARIOS_PATH).scenarios
            if s.id == "shared_view_typescript_header"
        )
        for answer in (
            "Return values.indexOf(target); it gives the matching index, or -1 when absent.",
            "Return values.findIndex(value => value === target);",
            "for (let i = 0; i < values.length; i++) { "
            "if (values[i] === target) return i; } return -1;",
            "Scan values from the start; return the first index matching target, "
            "or -1 when no match exists.",
        ):
            with self.subTest(answer=answer):
                result = self.score_answer(scenario, answer)
                self.assertTrue(result.passed, result.failures)

    def test_typescript_answer_terms_do_not_replace_grounding_checks(self):
        scenario = next(
            s for s in load_suite(COMPANION_SCENARIOS_PATH).scenarios
            if s.id == "shared_view_typescript_header"
        )
        observation = scenario.deterministic_observation.model_copy(update={
            "response_latency_ms": 1_000,
            "selected_memory_level": scenario.expected_memory_level,
            "grounded_fixture_ids": [],
        })
        result = score_scenario(scenario, observation, require_description_lag=False)
        self.assertFalse(result.passed)
        self.assertIn("answer was not grounded in the expected frame fixtures", result.failures)

    def test_negative_cases_accept_recognized_honest_or_brief_answers(self):
        scenarios = {s.id: s for s in load_suite(COMPANION_SCENARIOS_PATH).scenarios}
        answers = {
            "shared_view_stale_pixels": [
                "I don't have a fresh view of that label.",
                "My latest camera image is too old to confirm the label.",
                "I cannot read it without a current camera frame.",
            ],
            "shared_view_no_camera": [
                "I don't have a current camera frame.",
                "No live camera feed available.",
                "I can\u2019t see your current view right now.",
            ],
            "shared_view_thanks": [
                "You're welcome.", "No problem!", "  ANYTIME.  ", "You\u2019re\nwelcome.",
            ],
        }
        for scenario_id, replies in answers.items():
            for answer in replies:
                with self.subTest(scenario=scenario_id, answer=answer):
                    result = self.score_answer(scenarios[scenario_id], answer)
                    self.assertTrue(result.passed, result.failures)
                    self.assertTrue(result.metrics.current_view_correct)

    def test_negative_cases_reject_empty_invented_or_superfluous_answers(self):
        scenarios = {s.id: s for s in load_suite(COMPANION_SCENARIOS_PATH).scenarios}
        answers = {
            "shared_view_stale_pixels": [
                "It says 12 mL now.",
                "I don't have a fresh view of that label, but it says 12 mL now.",
                "I don't have a fresh view of that label.\nIt says 12 mL now.",
                "I have a fresh view of that label.",
                "My latest camera image is not stale.",
            ],
            "shared_view_no_camera": [
                "You're looking at a red mug.",
                "I don't have a current camera frame. You're looking at a red mug.",
                "I have a current camera frame.",
                "I can't see your current view right now, but I can see a red mug.",
            ],
            "shared_view_thanks": [
                "You're welcome. I can see what you see and help with anything.",
                "No problem. I can manage your notes, calendar, and reminders.",
                "I am your personal assistant.",
            ],
        }
        for scenario_id, replies in answers.items():
            for answer in ["", "  \n\t", "Maybe.", *replies]:
                with self.subTest(scenario=scenario_id, answer=answer):
                    result = self.score_answer(scenarios[scenario_id], answer)
                    self.assertFalse(result.passed)
                    self.assertFalse(result.metrics.current_view_correct)
                    self.assertIn(
                        "answer did not match the expected response behavior", result.failures,
                    )

    def test_live_negative_answer_failures_reach_overall_assessment(self):
        bundle = self.fixture_trace()
        answers = {
            "shared_view_stale_pixels": "It says 12 mL now.",
            "shared_view_no_camera": "You're looking at a red mug.",
            "shared_view_thanks": "You're welcome. I can help with all your tasks.",
        }
        for scenario_id, answer in answers.items():
            bundle.scenarios[scenario_id].observation.answer = answer
        report = self.score_trace(bundle)
        self.assertFalse(report.assessment.passed)
        failed = {s.scenario_id: s for s in report.report.scenarios if not s.passed}
        self.assertEqual(set(failed), set(answers))
        for result in failed.values():
            self.assertFalse(result.metrics.current_view_correct)
            self.assertIn(
                "answer did not match the expected response behavior", result.failures,
            )

    def test_answer_word_limit_is_enforced_independently_of_patterns(self):
        scenario = next(
            s for s in load_suite(COMPANION_SCENARIOS_PATH).scenarios
            if s.id == "shared_view_thanks"
        )
        evidence = scenario.expected_evidence.model_copy(update={"max_answer_words": 1})
        scenario = scenario.model_copy(update={"expected_evidence": evidence})
        result = self.score_answer(scenario, "You're welcome.")
        self.assertFalse(result.passed)
        self.assertFalse(result.metrics.current_view_correct)
        self.assertEqual(result.failures, ["answer exceeded the scenario word limit"])

    def test_answer_checks_are_opt_in_and_patterns_are_validated(self):
        for scenario in load_suite(DEFAULT_SCENARIOS_PATH).scenarios:
            self.assertEqual(scenario.expected_evidence.allowed_answer_patterns, [])
            self.assertIsNone(scenario.expected_evidence.max_answer_words)
        for pattern in ("", "[", ".*"):
            with self.subTest(pattern=pattern), self.assertRaises(ValidationError):
                ExpectedEvidence(allowed_answer_patterns=[pattern])
        with self.assertRaises(ValidationError):
            ExpectedEvidence(max_answer_words=0)
