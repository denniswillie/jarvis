"""Frozen task-aware input compilation only; no native/runtime qualification."""
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
import unittest
from unittest.mock import patch

from evals.responsive_voice.native_interruption_scenarios import (
    ScenarioError, _compile_legacy_scenario, main, write_inputs,
)
from evals.responsive_voice import test_native_interruption_scenarios as scenario_fixtures


class TaskAwareCompilationTests(unittest.TestCase):
    setUp = scenario_fixtures.ScenarioTests.setUp
    compile = scenario_fixtures.ScenarioTests.compile

    def test_one_two_three_request_first_and_exact_frozen_triggers(self):
        for cuts in (1, 2, 3):
            with self.subTest(cuts=cuts):
                result = self.compile(cuts=cuts, task_aware=True)
                inputs = result.manifest["utterances"]
                rows = result.expectations["occurrences"]
                self.assertEqual(len(inputs), 2 + 2 * cuts)
                self.assertEqual(rows[0]["source_clip"], "setup/background_task_setup")
                self.assertNotIn("baseline/current_view", [item["source_clip"] for item in rows])
                self.assertEqual(inputs[0]["trigger"], dict(kind="at_time", at_seconds=7.0))
                self.assertEqual(inputs[1]["trigger"], dict(kind="after_task_claim", after_input_ordinal=0, timeout_seconds=30.0))
                for ordinal in range(2, len(inputs), 2):
                    self.assertEqual(inputs[ordinal]["trigger"], dict(kind="response_output_start",
                        after_input_ordinal=ordinal - 1, delay_seconds=0.3, timeout_seconds=10.0,
                        max_pcm_age_seconds=0.05, require_task_claim=True))
                    self.assertEqual(inputs[ordinal + 1]["trigger"], dict(kind="after_input_complete",
                        after_input_ordinal=ordinal, delay_seconds=1.0, timeout_seconds=10.0))
                    self.assertAlmostEqual(rows[ordinal + 1]["latest_nominal_start_seconds"],
                        rows[ordinal]["latest_nominal_start_seconds"] + rows[ordinal]["input_duration_seconds"] + 1.0)
                self.assertEqual([row["ordinal"] for row in rows], list(range(len(inputs))))
                self.assertEqual(len({row["occurrence_id"] for row in rows}), len(inputs))
                self.assertEqual(result.manifest["task_barrier"], dict(version=1, scope_file="task-scope.json", request_input_ordinal=0))
                self.assertIsNone(result.expectations["actual_cuts"])

    def test_default_compilation_is_byte_equivalent_to_legacy(self):
        for case, cuts in (("baseline", 0), ("pause_continue", 3), ("quiet_direct_resume", 1)):
            with self.subTest(case=case):
                old = _compile_legacy_scenario(case, cuts=cuts, binding=self.binding, frame_plan=self.frames,
                                               frame_root=self.root, clips=self.clips)
                new = self.compile(case, cuts)
                self.assertEqual(old, new)

    def test_scope_is_missing_not_fabricated_and_control_paths_not_in_manifest(self):
        result = self.compile(task_aware=True)
        output = self.root / "prepared"
        write_inputs(result, output)
        self.assertFalse((output / "task-scope.json").exists())
        self.assertFalse((output / "token.txt").exists())
        self.assertNotIn("../controls", json.dumps(result.manifest))
        for field in ("worker_id", "runtime_host_id", "attempt_id", "session_id"):
            self.assertNotIn(field, json.dumps(result.manifest))
        self.assertFalse(result.expectations["task_scope_staged"])
        self.assertEqual(result.expectations["task_scope_status"], "missing_until_actual_worker_readiness")
        self.assertEqual(result.expectations["qualification"], "unmeasured")

    def test_prerequisite_binds_real_setup_and_status_ordinals_not_old_view(self):
        result = self.compile(task_aware=True)
        prerequisite = result.expectations["prerequisites"][0]
        self.assertEqual(prerequisite["setup_occurrence_id"], result.expectations["occurrences"][0]["occurrence_id"])
        self.assertEqual(prerequisite["must_precede_occurrence_id"], result.expectations["occurrences"][1]["occurrence_id"])
        self.assertTrue(prerequisite["runtime_barrier_supported"])
        self.assertEqual(prerequisite["verdict"], "unmeasured")

    def test_shared_clip_repetitions_still_have_distinct_occurrences(self):
        rows = self.compile(cuts=3, task_aware=True).expectations["occurrences"]
        self.assertEqual(len({rows[index]["clip_id"] for index in (2, 4, 6)}), 1)
        self.assertEqual(len({rows[index]["occurrence_id"] for index in (2, 4, 6)}), 3)

    def test_unchanged_frame_clock_and_no_unplayed_current_view_asset(self):
        result = self.compile(task_aware=True)
        self.assertEqual([row["at_seconds"] for row in result.manifest["frames"]], [0, 30])
        used = {row["clip_id"] for row in result.manifest["utterances"]}
        self.assertEqual({clip["id"] for clip in result.speech["clips"]}, used)
        self.assertEqual(len([name for name in result.assets if name.endswith(".wav")]), len(used))

    def test_wrong_case_route_and_unbounded_schedule_rejected(self):
        for override in (dict(case="parallel_keep_working"), dict(cuts=0), dict(cuts=4),
                         dict(binding={**self.binding, "voice_route": "live"}), dict(duration=20), dict(task_aware=1)):
            with self.subTest(override=override):
                values = dict(task_aware=True)
                values.update(override)
                with self.assertRaises(ScenarioError):
                    self.compile(**values)

    def test_cli_task_opt_in_uses_frozen_sequence_and_leaves_scope_missing(self):
        binding, frames, output = self.root / "binding.json", self.root / "frames.json", self.root / "cli"
        binding.write_text(json.dumps(self.binding))
        frames.write_text(json.dumps(self.frames))
        with patch("evals.responsive_voice.native_interruption_scenarios.load_bundles", return_value=self.clips), redirect_stdout(io.StringIO()):
            code = main(["--case", "pause_continue", "--cuts", "3", "--task-aware", "--binding", str(binding),
                         "--frames", str(frames), "--output", str(output)])
        self.assertEqual(code, 0)
        manifest = json.loads((output / "manifest.json").read_text())
        self.assertEqual(manifest["duration_seconds"], 120)
        self.assertEqual(manifest["utterances"][0]["trigger"], dict(kind="at_time", at_seconds=7.0))
        self.assertFalse((output / "task-scope.json").exists())

    def test_explicit_three_cut_scene150_has_new_revision_without_retiming(self):
        original = self.compile(cuts=3, task_aware=True, distinct_provider_turns=True)
        extended = self.compile(cuts=3, task_aware=True, distinct_provider_turns=True, duration=150)
        self.assertEqual(original.expectations["case_revision"], 5)
        self.assertEqual(original.manifest["duration_seconds"], 120)
        self.assertEqual(extended.expectations["case_revision"], 6)
        self.assertEqual(extended.expectations["declared_task_scene_seconds"], 150)
        self.assertEqual(extended.expectations["scene_extension_scope"], "three_cuts_only")
        self.assertEqual(extended.manifest["duration_seconds"], 150)
        for key in ("frames", "utterances", "task_barrier"):
            self.assertEqual(extended.manifest[key], original.manifest[key])
        self.assertEqual(extended.speech, original.speech)
        self.assertEqual(extended.assets, original.assets)
        self.assertIsNone(extended.expectations["actual_cuts"])
        self.assertFalse(extended.expectations["task_scope_staged"])

    def test_scene_extension_is_not_a_global_or_legacy_duration_increase(self):
        cases = [dict(cuts=cuts, duration=150) for cuts in (1, 2)]
        cases += [dict(cuts=3, duration=duration) for duration in (120.001, 149, 150.001, 151, 180)]
        cases += [dict(cuts=3, duration=150, distinct_provider_turns=False)]
        for override in cases:
            with self.subTest(override=override):
                values = dict(task_aware=True, distinct_provider_turns=True)
                values.update(override)
                with self.assertRaises(ScenarioError):
                    self.compile(**values)
        for cuts in (1, 2):
            with self.subTest(unchanged=cuts):
                default = self.compile(cuts=cuts, task_aware=True, distinct_provider_turns=True)
                self.assertEqual(default, self.compile(cuts=cuts, task_aware=True,
                                                     distinct_provider_turns=True, duration=120))

    def test_cli_explicit_three_cut150_keeps_missing_runtime_scope(self):
        binding, frames, output = self.root / "binding.json", self.root / "frames.json", self.root / "scene150"
        binding.write_text(json.dumps(self.binding))
        frames.write_text(json.dumps(self.frames))
        with patch("evals.responsive_voice.native_interruption_scenarios.load_bundles", return_value=self.clips), redirect_stdout(io.StringIO()):
            code = main(["--case", "pause_continue", "--cuts", "3", "--task-aware", "--distinct-provider-turns",
                         "--duration", "150", "--binding", str(binding), "--frames", str(frames), "--output", str(output)])
        self.assertEqual(code, 0)
        manifest = json.loads((output / "manifest.json").read_text())
        expected = json.loads((output / "expectations.json").read_text())
        self.assertEqual(manifest["duration_seconds"], 150)
        self.assertEqual(expected["case_revision"], 6)
        self.assertFalse((output / "task-scope.json").exists())
