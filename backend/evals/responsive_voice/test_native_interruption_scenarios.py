"""Offline compiler regressions; synthetic fixtures never launch native media."""
from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import replace
import io
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch
from uuid import uuid4
import wave

from evals.responsive_voice.native_capture_evidence import RATE
from evals.responsive_voice.test_native_capture_evidence import fixture
from evals.responsive_voice.native_interruption_scenarios import (
    AUTHORITY, CASES, DEFAULT_BUNDLES, TASK_CASES, Clip, ScenarioError, account_execution,
    compile_scenario, digest, load_bundles, main, write_inputs,
)


KEYS = {
    "baseline": ("current_view",),
    "conversation": ("speech_only_pause", "continue_answer", "parallel_question", "keep_working", "correction_left",
                     "correction_right", "correction_label", "explicit_cancel", "repeat_cancel", "ambiguous_stop"),
    "state": ("quiet_until_resume", "quiet_direct_question", "resume_summary", "remember_notebook", "correct_notebook",
              "forget_notebook", "recall_notebook", "operation_status", "cancel_if_pending"),
    "setup": ("background_task_setup", "memory_setup", "reminder_setup"),
}


def wav_bytes():
    out = io.BytesIO()
    with wave.open(out, "wb") as wav:
        wav.setparams((1, 2, RATE, 4, "NONE", "not compressed"))
        wav.writeframes(struct.pack("<4h", 0, 128, 128, 0))
    return out.getvalue()


class ScenarioTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.binding = dict(run_id=uuid4().hex, backend_base_url="http://127.0.0.1:8765", owner_id="isolated-fixture-owner",
                            area_id="fixture-area", memory_bank_id="fixture-bank", voice_route="realtime")
        # Signature-only image fixture; no claim of decoded image/native acceptance.
        image = b"\xff\xd8fixture\xff\xd9"
        (self.root / "source.jpg").write_bytes(image)
        self.frames = [dict(at_seconds=0, relative_path="source.jpg", sha256=digest(image)),
                       dict(at_seconds=30, relative_path="source.jpg", sha256=digest(image))]
        data = wav_bytes()
        self.clips = {bundle + "/" + key: Clip(bundle, key, data, digest(data), digest(struct.pack("<4h", 0, 128, 128, 0)),
                                               4, 1, 3, 128) for bundle, keys in KEYS.items() for key in keys}

    def compile(self, case="pause_continue", cuts=1, **kwargs):
        values = dict(binding=self.binding, frame_plan=self.frames, frame_root=self.root, clips=self.clips)
        values.update(kwargs)
        return compile_scenario(case, cuts=cuts, **values)

    def capture(self, compiled):
        capture, pcm, _ = fixture(len(compiled.manifest["utterances"]))
        capture["run_id"] = str(uuid4())
        capture["run_id"] = str(__import__("uuid").UUID(self.binding["run_id"]))
        starts = []
        for ordinal, plan in enumerate(compiled.expectations["occurrences"]):
            actual = capture["utterances"][ordinal]
            actual.update(occurrence_id=plan["occurrence_id"], trigger_id=plan["trigger_id"],
                          clip_id=plan["clip_id"], trigger=plan["trigger"], pcm16_sha256=plan["pcm16_sha256"],
                          triggered_uptime_seconds=100 + ordinal * 2 - 0.1)
            if plan["planned_cut"]:
                starts.append(dict(local_cursor=len(starts) + 1, uptime=100 + ordinal * 2 - 1))
        capture["output_starts"] = starts
        return capture, pcm

    def test_all_families_have_one_two_three_cut_inputs_and_frozen_authority(self):
        for case in CASES:
            for cuts in ((0,) if case == "baseline" else (1, 2, 3)):
                with self.subTest(case=case, cuts=cuts):
                    compiled = self.compile(case, cuts)
                    self.assertEqual(compiled.expectations["planned_cuts"], cuts)
                    self.assertEqual(sum(item["planned_cut"] for item in compiled.expectations["occurrences"]), cuts)
                    self.assertEqual(compiled.expectations["case_ids"], list(CASES[case]))
                    self.assertEqual(compiled.expectations["authority_sha256"], digest(AUTHORITY.read_bytes()))
                    self.assertIsNone(compiled.expectations["actual_cuts"])
                    self.assertEqual(compiled.expectations["qualification"], "unmeasured")

    def test_baseline_remains_only_simple_current_view(self):
        compiled = self.compile("baseline", 0)
        self.assertEqual(len(compiled.manifest["utterances"]), 1)
        self.assertEqual(compiled.expectations["occurrences"][0]["source_clip"], "baseline/current_view")
        self.assertEqual(compiled.manifest["utterances"][0]["trigger"], {"kind": "at_time", "at_seconds": 5.0})

    def test_pause_always_has_explicit_continue_without_waiting_for_pause_reply(self):
        compiled = self.compile(cuts=3)
        occurrences = compiled.expectations["occurrences"]
        self.assertEqual(len(occurrences), 9)
        for index in (3, 5, 7):
            self.assertTrue(occurrences[index]["silence_after_input_allowed"])
            self.assertEqual(occurrences[index + 1]["intent"], "continue_answer")
            self.assertEqual(occurrences[index + 1]["trigger"]["kind"], "at_time")
        self.assertEqual(len({item["occurrence_id"] for item in occurrences}), 9)
        self.assertEqual(occurrences[3]["clip_id"], occurrences[5]["clip_id"])

    def test_cursors_increase_and_timeouts_are_bounded(self):
        compiled = self.compile("correction_chain", 3)
        triggers = [item["trigger"] for item in compiled.manifest["utterances"] if item["trigger"]["kind"] == "output_start"]
        self.assertEqual([item["after_cursor"] for item in triggers], [0, 1, 2])
        for trigger in triggers:
            self.assertGreater(trigger["delay_seconds"], 0)
            self.assertLess(trigger["delay_seconds"], trigger["timeout_seconds"])
            self.assertLessEqual(trigger["timeout_seconds"], compiled.manifest["duration_seconds"])

    def test_response_mode_emits_frozen_trigger_for_one_two_three_variants(self):
        for case in CASES.keys() - {"baseline"}:
            for cuts in (1, 2, 3):
                with self.subTest(case=case, cuts=cuts):
                    compiled = self.compile(case, cuts, duration=120, output_trigger="response_correlated")
                    planned = [p for p in compiled.expectations["occurrences"] if p["planned_cut"]]
                    self.assertEqual(len(planned), cuts)
                    for plan in planned:
                        trigger = plan["trigger"]
                        self.assertEqual(trigger, dict(kind="response_output_start",
                            after_input_ordinal=plan["ordinal"] - 1, delay_seconds=0.3,
                            timeout_seconds=10.0, max_pcm_age_seconds=0.05))
                        before = compiled.expectations["occurrences"][plan["ordinal"] - 1]
                        self.assertAlmostEqual(plan["latest_nominal_start_seconds"],
                            before["latest_nominal_start_seconds"] + before["input_duration_seconds"] + 10)
                    self.assertIsNone(compiled.expectations["actual_cuts"])
                    self.assertEqual(compiled.expectations["qualification"], "unmeasured")
                    self.assertIn("semantic_output_attribution", compiled.expectations["unresolved_acceptance_barriers"])

    def test_response_mode_has_new_revision_and_keeps_unavailable_boundaries_explicit(self):
        original = self.compile("explicit_cancel", 1)
        compiled = self.compile("explicit_cancel", 1, duration=120, output_trigger="response_correlated")
        self.assertEqual(original.expectations["output_trigger"], "energy_cursor")
        self.assertEqual(compiled.expectations["case_revision"], original.expectations["case_revision"] + 1)
        self.assertEqual(compiled.expectations["variant"], "explicit_cancel/native_response_control_with_spoken_setup")
        self.assertIn("playback_drained", compiled.expectations["unsupported_runtime_controls"])
        self.assertIn("request_response_causality", compiled.expectations["unsupported_runtime_controls"])
        self.assertIn("per_packet_response_attribution", compiled.expectations["unsupported_runtime_controls"])
        self.assertNotIn("response_id", json.dumps(compiled.manifest))
        self.assertEqual(compiled.expectations["authority_sha256"], original.expectations["authority_sha256"])

    def test_response_mode_rejects_unsupported_route_missing_cuts_and_invalid_bounds(self):
        for override in (dict(output_trigger="unknown"), dict(response_delay=True),
                         dict(response_delay=float("nan")), dict(response_delay=10),
                         dict(max_pcm_age=0), dict(max_pcm_age=11),
                         dict(binding={**self.binding, "voice_route": "live"})):
            with self.subTest(fields=tuple(override)):
                values = dict(output_trigger="response_correlated", duration=120)
                values.update(override)
                with self.assertRaises(ScenarioError):
                    self.compile(**values)
        with self.assertRaisesRegex(ScenarioError, "response_trigger_requires_cut_variant"):
            self.compile("baseline", 0, output_trigger="response_correlated")

    def test_response_accounting_uses_reader_verdict_but_never_claims_actual_cuts(self):
        # This tests the compiler/reader interface, not native receipt validation.
        # The real reader owns sample, identity, interruption and gate validation.
        for cuts in (1, 2, 3):
            with self.subTest(cuts=cuts):
                compiled = self.compile("correction_chain", cuts, duration=120,
                                        output_trigger="response_correlated")
                capture, pcm = self.capture(compiled)
                rows = [dict(ordinal=p["ordinal"], occurrence_id=p["occurrence_id"],
                             input_verdict="sdk_completed_active_input",
                             response_trigger_verdict="validated_control_binding" if p["planned_cut"] else "not_applicable",
                             response_control_stop=None) for p in compiled.expectations["occurrences"]]
                with patch("evals.responsive_voice.native_interruption_scenarios.analyze_capture",
                           return_value=dict(occurrences=rows, timing_blockers=[])):
                    result = account_execution(compiled, capture, pcm)
                self.assertEqual(result["completed_response_triggered_inputs"], cuts)
                self.assertEqual(result["completed_cursor_triggered_inputs"], 0)
                self.assertEqual(result["input_execution"], "complete")
                self.assertIsNone(result["actual_cuts"])
                self.assertEqual(result["response_control_stop_measurements"], [])
                self.assertEqual(result["response_semantic_causality"], "unknown")
                self.assertEqual(result["provider_response_attribution"], "unavailable")
                self.assertFalse(result["hearing_measured"])

    def test_response_accounting_missing_reader_proof_does_not_count_input_as_a_cut(self):
        compiled = self.compile("correction_chain", 1, output_trigger="response_correlated")
        capture, pcm = self.capture(compiled)
        rows = [dict(ordinal=p["ordinal"], occurrence_id=p["occurrence_id"],
                     input_verdict="sdk_completed_active_input") for p in compiled.expectations["occurrences"]]
        capture["utterances"][1]["response_trigger"] = {"enqueue_revalidated": True}
        with patch("evals.responsive_voice.native_interruption_scenarios.analyze_capture",
                   return_value=dict(occurrences=rows, timing_blockers=[])):
            result = account_execution(compiled, capture, pcm)
        self.assertEqual(result["completed_response_triggered_inputs"], 0)
        self.assertEqual(result["input_execution"], "incomplete")
        self.assertIsNone(result["actual_cuts"])

    def test_followups_cover_nominal_prior_input_even_at_deadline(self):
        clip = self.clips["conversation/speech_only_pause"]
        clips = dict(self.clips, **{"conversation/speech_only_pause": replace(clip, sample_count=2 * RATE)})
        compiled = self.compile(cuts=3, clips=clips)
        for before, after in zip(compiled.expectations["occurrences"], compiled.expectations["occurrences"][1:]):
            if after["trigger"]["kind"] == "at_time":
                self.assertGreater(after["trigger"]["at_seconds"], before["latest_nominal_start_seconds"] + before["input_duration_seconds"])

    def test_runtime_payload_excludes_authority_labels_scripts_and_answers(self):
        compiled = self.compile("quiet_direct_resume", 3)
        text = json.dumps([compiled.manifest, compiled.speech])
        for forbidden in ("E2E-", "predicate", "intent", "first useful", "quiet_until_resume", "expected_answer", "transcript", '"text"'):
            self.assertNotIn(forbidden, text)
        self.assertTrue(all(set(item) == {"clip_id", "trigger"} for item in compiled.manifest["utterances"]))
        self.assertTrue(all(item["relative_path"].startswith("assets/clip_") for item in compiled.speech["clips"]))

    def test_unsupported_state_drain_reconnect_and_longitudinal_barriers_stay_visible(self):
        compiled = self.compile("memory_correction_deletion", 3)
        self.assertIn("playback_drained", compiled.expectations["unsupported_runtime_controls"])
        self.assertIn("after_state", compiled.expectations["unsupported_runtime_controls"])
        self.assertIn("fresh_session_restart_and_days_to_months_clock", compiled.expectations["unresolved_acceptance_barriers"])
        self.assertTrue(all(item["verdict"] == "unmeasured" for item in compiled.expectations["assertions"]))
        self.assertTrue(all(item["trigger"]["kind"] in {"at_time", "output_start"} for item in compiled.manifest["utterances"]))

    def test_task_setup_is_spoken_but_canonical_admission_is_not_fabricated(self):
        for case in TASK_CASES:
            with self.subTest(case=case):
                result = self.compile(case, 3)
                self.assertEqual(result.expectations["occurrences"][1]["source_clip"], "setup/background_task_setup")
                self.assertIn("canonical_task_creation_and_running_state", result.expectations["unresolved_acceptance_barriers"])
                self.assertNotIn("spoken_read_only_task_admission_not_in_supplied_clips", result.expectations["unresolved_acceptance_barriers"])
                prerequisite = result.expectations["prerequisites"][0]
                self.assertEqual(prerequisite["kind"], "canonical_task_creation")
                self.assertEqual(prerequisite["setup_occurrence_id"], result.expectations["occurrences"][1]["occurrence_id"])
                self.assertEqual(prerequisite["must_precede_occurrence_id"], result.expectations["occurrences"][2]["occurrence_id"])
                self.assertFalse(prerequisite["runtime_barrier_supported"])
                self.assertNotIn("seed_task", result.manifest)

    def test_memory_setup_explicitly_precedes_correction_and_requires_source_receipt(self):
        result = self.compile("memory_correction_deletion", 3)
        occurrences = result.expectations["occurrences"]
        self.assertEqual(occurrences[1]["source_clip"], "setup/memory_setup")
        self.assertEqual(occurrences[1]["intent"], "remember_explicit_user_report_not_visual_evidence")
        self.assertEqual(occurrences[2]["source_clip"], "state/correct_notebook")
        prerequisite = result.expectations["prerequisites"][0]
        self.assertEqual(prerequisite["kind"], "canonical_memory_creation")
        self.assertIn("publication_receipt_or_pending_state", prerequisite["required_canonical_evidence"])
        self.assertEqual(prerequisite["verdict"], "unmeasured")

    def test_quiet_setup_includes_task_reminder_and_due_opportunity_not_state_barrier(self):
        result = self.compile("quiet_direct_resume", 1)
        occurrences = result.expectations["occurrences"]
        reminder = next(item for item in occurrences if item["source_clip"] == "setup/reminder_setup")
        quiet = next(item for item in occurrences if item["source_clip"] == "state/quiet_until_resume")
        resume = occurrences[-1]
        self.assertLess(reminder["ordinal"], quiet["ordinal"])
        self.assertGreaterEqual(resume["trigger"]["at_seconds"], reminder["latest_nominal_start_seconds"] + reminder["input_duration_seconds"] + 35)
        self.assertEqual({item["kind"] for item in result.expectations["prerequisites"]}, {"canonical_task_creation", "canonical_reminder_creation"})
        self.assertIn("canonical_reminder_creation_and_due_order", result.expectations["unresolved_acceptance_barriers"])
        self.assertTrue(all(item["trigger"]["kind"] in {"at_time", "output_start"} for item in result.manifest["utterances"]))

    def test_completed_setup_and_assistant_claims_do_not_satisfy_canonical_prerequisites(self):
        for case in ("parallel_keep_working", "memory_correction_deletion", "quiet_direct_resume"):
            with self.subTest(case=case):
                compiled = self.compile(case, 2)
                capture, pcm = self.capture(compiled)
                capture["final_ui_status"] = "The task, memory and reminder were created successfully."
                result = account_execution(compiled, capture, pcm)
                self.assertEqual(result["input_execution"], "complete")
                self.assertEqual(result["semantic_execution"], "inconclusive")
                self.assertTrue(all(item["verdict"] == "inconclusive" for item in result["prerequisites"]))
                self.assertTrue(all(not item["canonical_creation_confirmed"] and item["canonical_receipt"] is None
                                    for item in result["prerequisites"]))
                self.assertFalse(result["semantic_assertions_evaluated"])

    def test_unstarted_setup_or_dependent_input_is_not_reached(self):
        for ordinal in (1, 2):
            with self.subTest(ordinal=ordinal):
                compiled = self.compile("explicit_cancel", 1)
                capture, pcm = self.capture(compiled)
                capture["utterances"][ordinal].update(status="not_started", callbacks=[], attempted_clip_samples=0,
                    sdk_accepted_clip_samples=0, first_sdk_callback_start_uptime_seconds=None,
                    last_sdk_callback_end_uptime_seconds=None, first_successful_activity_uptime_seconds=None,
                    last_successful_activity_uptime_seconds=None, triggered_uptime_seconds=None)
                result = account_execution(compiled, capture, pcm)
                self.assertEqual(result["semantic_execution"], "not_reached")
                self.assertEqual(result["prerequisites"][0]["verdict"], "not_reached")
                self.assertFalse(result["prerequisites"][0]["canonical_creation_confirmed"])

    def test_missing_setup_asset_blocks_task_case_but_not_baseline(self):
        clips = {key: value for key, value in self.clips.items() if not key.startswith("setup/")}
        with self.assertRaisesRegex(ScenarioError, "required_speech_clip_missing"):
            self.compile("explicit_cancel", 1, clips=clips)
        baseline = self.compile("baseline", 0, clips=clips)
        self.assertEqual(len(baseline.manifest["utterances"]), 1)
        self.assertEqual(baseline.expectations["prerequisites"], [])
        self.assertEqual(baseline.expectations["case_revision"], 1)

    def test_spoken_setup_variants_have_new_revision_without_rewriting_authority(self):
        result = self.compile("quiet_direct_resume", 3)
        self.assertEqual(result.expectations["case_revision"], 2)
        self.assertEqual(result.expectations["variant"], "quiet_direct_resume/native_onset_with_spoken_setup")
        self.assertEqual(result.expectations["authority_sha256"], digest(AUTHORITY.read_bytes()))

    def test_actual_cut_count_never_inferred_from_cursor_or_complete_inputs(self):
        for cuts in (1, 2, 3):
            with self.subTest(cuts=cuts):
                compiled = self.compile(cuts=cuts)
                capture, pcm = self.capture(compiled)
                result = account_execution(compiled, capture, pcm)
                self.assertEqual(result["completed_cursor_triggered_inputs"], cuts)
                self.assertEqual(result["input_execution"], "complete")
                self.assertIsNone(result["actual_cuts"])
                self.assertFalse(result["semantic_assertions_evaluated"])
                self.assertFalse(result["hearing_measured"])

    def test_not_started_and_missing_occurrences_never_count_as_pass(self):
        compiled = self.compile(cuts=3)
        capture, pcm = self.capture(compiled)
        for item in capture["utterances"][1:]:
            item.update(status="not_started", callbacks=[], attempted_clip_samples=0, sdk_accepted_clip_samples=0,
                        first_sdk_callback_start_uptime_seconds=None, last_sdk_callback_end_uptime_seconds=None,
                        first_successful_activity_uptime_seconds=None, last_successful_activity_uptime_seconds=None,
                        triggered_uptime_seconds=None)
        capture["utterances"] = capture["utterances"][:4]
        result = account_execution(compiled, capture, pcm)
        self.assertEqual(len(result["occurrences"]), 9)
        self.assertEqual(result["occurrences"][1]["input_verdict"], "not_started")
        self.assertEqual(result["occurrences"][-1]["input_verdict"], "not_recorded")
        self.assertEqual(result["input_execution"], "incomplete")
        self.assertEqual(result["completed_cursor_triggered_inputs"], 0)

    def test_stale_reused_cursor_missing_start_and_expired_wait_fail_accounting(self):
        for mode in ("stale", "missing", "expired", "reused"):
            with self.subTest(mode=mode):
                compiled = self.compile("correction_chain", 2)
                capture, pcm = self.capture(compiled)
                if mode == "stale":
                    capture["output_starts"][0]["uptime"] = 99
                elif mode == "missing":
                    capture["output_starts"] = []
                elif mode == "expired":
                    capture["utterances"][0]["triggered_uptime_seconds"] = 90
                else:
                    capture["output_starts"][1]["local_cursor"] = 1
                    with self.assertRaisesRegex(ScenarioError, "nonmonotonic_output_cursor"):
                        account_execution(compiled, capture, pcm)
                    continue
                self.assertEqual(account_execution(compiled, capture, pcm)["input_execution"], "incomplete")

    def test_foreign_run_clip_and_trigger_are_rejected(self):
        for field in ("run", "clip_id", "trigger", "pcm16_sha256"):
            with self.subTest(field=field):
                compiled = self.compile()
                capture, pcm = self.capture(compiled)
                if field == "run":
                    capture["run_id"] = str(uuid4())
                elif field == "trigger":
                    capture["utterances"][1][field] = dict(kind="at_time", at_seconds=1)
                else:
                    capture["utterances"][1][field] = "foreign"
                with self.assertRaises(ScenarioError):
                    account_execution(compiled, capture, pcm)

    def test_invalid_counts_deadlines_sources_frames_and_private_binding_fail_closed(self):
        for kwargs in (dict(cuts=0), dict(cuts=4), dict(cuts=True), dict(timeout=0), dict(timeout=float("nan")),
                       dict(duration=121), dict(duration=10), dict(timeout=0.1), dict(clips={}),
                       dict(binding={**self.binding, "api_key": "not-to-be-read"}),
                       dict(binding={**self.binding, "backend_base_url": "https://example.com"}),
                       dict(frame_plan=list(reversed(self.frames)))):
            with self.subTest(fields=tuple(kwargs)):
                with self.assertRaises(ScenarioError):
                    self.compile(**kwargs)

    def test_frame_path_escape_and_hash_mismatch_are_rejected(self):
        for field, value in (("relative_path", "../secret.jpg"), ("sha256", "0" * 64)):
            frames = deepcopy(self.frames)
            frames[0][field] = value
            with self.assertRaises(ScenarioError):
                self.compile(frame_plan=frames)

    def test_write_is_fresh_only_preserves_sources_and_never_stages_token(self):
        compiled = self.compile(cuts=3)
        before = (self.root / "source.jpg").read_bytes()
        output = self.root / "compiled"
        write_inputs(compiled, output)
        self.assertEqual(json.loads((output / "manifest.json").read_text()), compiled.manifest)
        self.assertEqual((self.root / "source.jpg").read_bytes(), before)
        self.assertFalse((output / "token.txt").exists())
        with self.assertRaises(FileExistsError):
            write_inputs(compiled, output)

    def prepare_bundles(self):
        paths = {}
        for bundle, keys in KEYS.items():
            directory = self.root / bundle
            directory.mkdir()
            clips = []
            for key in keys:
                clip = self.clips[bundle + "/" + key]
                (directory / (key + ".wav")).write_bytes(clip.wav)
                clips.append(dict(id=key, text="THIS SCRIPT MUST NOT ENTER THE MODEL", relative_path=key + ".wav", sha256=clip.sha256,
                                  sample_rate_hz=RATE, channels=1, sample_width_bytes=2, sample_count=4,
                                  first_active_sample=1, last_active_sample_exclusive=3, energy_threshold_abs_pcm16=128))
            paths[bundle] = directory / "speech.json"
            paths[bundle].write_text(json.dumps(dict(format="jarvis.speech_assets.v1", clips=clips)))
        return paths

    def test_cli_compiles_both_bundles_without_text_or_token(self):
        paths = self.prepare_bundles()
        (self.root / "binding.json").write_text(json.dumps(self.binding))
        (self.root / "frames.json").write_text(json.dumps(self.frames))
        output = self.root / "cli"
        args = ["--case", "quiet_direct_resume", "--cuts", "3", "--binding", str(self.root / "binding.json"),
                "--frames", str(self.root / "frames.json"), "--output", str(output)]
        for bundle, path in paths.items():
            args += ["--" + bundle + "-speech", str(path)]
        log = io.StringIO()
        with redirect_stdout(log):
            self.assertEqual(main(args), 0)
        self.assertEqual(json.loads(log.getvalue())["status"], "inputs_prepared_not_executed")
        self.assertNotIn("SCRIPT", (output / "speech.json").read_text())
        self.assertFalse((output / "token.txt").exists())

    def test_wav_hash_format_and_activity_metadata_are_validated(self):
        paths = self.prepare_bundles()
        self.assertEqual(len(load_bundles(paths)), 23)
        raw = json.loads(paths["conversation"].read_text())
        raw["clips"][0]["first_active_sample"] = 0
        paths["conversation"].write_text(json.dumps(raw))
        with self.assertRaisesRegex(ScenarioError, "activity_metadata"):
            load_bundles(paths)
        raw["clips"][0]["sha256"] = "0" * 64
        paths["conversation"].write_text(json.dumps(raw))
        with self.assertRaisesRegex(ScenarioError, "hash_mismatch"):
            load_bundles(paths)

    def test_cli_response_mode_is_explicit_and_does_not_write_oracle_to_runtime(self):
        paths = self.prepare_bundles()
        (self.root / "binding.json").write_text(json.dumps(self.binding))
        (self.root / "frames.json").write_text(json.dumps(self.frames))
        output = self.root / "response-cli"
        args = ["--case", "correction_chain", "--cuts", "3", "--duration", "120",
                "--output-trigger", "response_correlated", "--binding", str(self.root / "binding.json"),
                "--frames", str(self.root / "frames.json"), "--output", str(output)]
        for bundle, path in paths.items():
            args += ["--" + bundle + "-speech", str(path)]
        with redirect_stdout(io.StringIO()):
            self.assertEqual(main(args), 0)
        manifest = json.loads((output / "manifest.json").read_text())
        self.assertEqual([u["trigger"]["kind"] for u in manifest["utterances"]],
                         ["at_time", "response_output_start", "response_output_start", "response_output_start"])
        self.assertNotIn("semantic", json.dumps(manifest))
        self.assertFalse((output / "token.txt").exists())

    @unittest.skipUnless(all(path.exists() for path in DEFAULT_BUNDLES.values()), "private prepared speech bundles not installed")
    def test_actual_prepared_speech_assets_validate_and_all_variants_fit(self):
        clips = load_bundles(DEFAULT_BUNDLES)
        self.assertEqual({clip.source_id for clip in clips.values() if clip.bundle == "setup"},
                         {"background_task_setup", "memory_setup", "reminder_setup"})
        self.assertEqual(sum(clip.sample_count for clip in clips.values() if clip.bundle == "conversation"), 1083090)
        self.assertEqual(sum(clip.sample_count for clip in clips.values() if clip.bundle == "state"), 1425365)
        for case in CASES:
            for cuts in ((0,) if case == "baseline" else (1, 2, 3)):
                with self.subTest(case=case, cuts=cuts):
                    compiled = self.compile(case, cuts, clips=clips)
                    self.assertLessEqual(compiled.expectations["synthetic_input_seconds"], 60)
                    self.assertLessEqual(len(compiled.manifest["utterances"]), 20)
                    self.assertEqual(compiled.expectations["planned_cuts"], cuts)


if __name__ == "__main__":
    unittest.main()
