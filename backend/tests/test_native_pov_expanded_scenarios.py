"""Offline frozen-clock POV preparation; fixture images do not test decoding."""
from decimal import Decimal, ROUND_CEILING
import io
import json
from pathlib import Path
import struct
import tempfile
import unittest
import wave
from unittest.mock import patch

from evals.responsive_voice import native_pov_scenarios as pov


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value if isinstance(value, bytes) else pov.encoded(value))


def event(variant, ordinal, at, clip):
    return dict(occurrence_id=f"{variant}-{ordinal:02d}", at_ms=at, utterance_id=clip)


class ExpandedPOVTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.pack, self.output = self.root / "pack", self.root / "prepared"
        for target in ("socket.socket.connect", "subprocess.Popen", "asyncpg.connect", "asyncpg.create_pool"):
            guard = patch(target, side_effect=AssertionError("external execution forbidden"))
            guard.start()
            self.addCleanup(guard.stop)
        self.assets = {}
        wipe_pts = [0, 2, 4.001667, 6.003333, 8.003333, 10.005, 12.006667, 14.006667,
                    16.008333, 18.01, 20.01, 22.011667, 24.011667, 26.013333, 28.015]
        for case_id, points in (("c01", list(range(0, 32, 2))), ("c02", list(range(0, 32, 2))), ("c05", wipe_pts)):
            frames = []
            for index, pts in enumerate(points):
                raw = b"\xff\xd8offline-fixture-" + case_id.encode() + bytes([index]) + b"\xff\xd9"
                relative = f"frames/{index + 1:06d}.jpg"
                save(self.pack / "inputs" / case_id / relative, raw)
                frames.append(dict(offset_ms=int((Decimal(str(pts)) * 1000).to_integral_value(rounding=ROUND_CEILING)),
                                   relative_path=relative, sha256=pov.base.digest(raw)))
            self.assets[case_id] = dict(format="jarvis.video_assets.v1", source_audio_included=False,
                source_sha256=pov.base.digest(case_id.encode()), source_first_pts_seconds=0,
                source_pts_seconds=points, source_frame_indices=[i * 60 for i in range(len(points))],
                inputs=dict(initial_context=[], frames=frames, events=[]))
        c01 = [dict(variant_id=v, prior_context_utterance_ids=[], events=[event(v, 1, 18000, clip)])
               for v, clip in pov.VARIANTS.items()]
        c01.append(dict(variant_id="c01-c", prior_context_utterance_ids=[], events=[]))
        c02 = [dict(variant_id=v, prior_context_utterance_ids=["u03"], events=[event(v, 1, 40000, "u04")])
               for v in ("c02-a", "c02-b")]
        c02[1]["frame_admission"] = {"only_release_offsets_less_than_ms": 18000}
        c05a = dict(variant_id="c05-a", prior_context_utterance_ids=["u10"],
                    events=[event("c05-a", i + 1, at, clip) for i, (at, clip) in enumerate(
                        ((0, "u09"), (8000, "u11"), (96000, "u12"), (110000, "u13")))])
        c05b = dict(variant_id="c05-b", prior_context_utterance_ids=[], scenario_end_ms=45000,
                    events=[event("c05-b", 1, 0, "u09"), event("c05-b", 2, 8000, "u11")])
        self.timeline = {"cases": [
            dict(case_id="c01", assets="c01/assets.json", media_end_ms=32000, scenario_end_ms=40000, variants=c01),
            dict(case_id="c02", assets="c02/assets.json", media_end_ms=32000, scenario_end_ms=60000, variants=c02),
            dict(case_id="c05", assets="c05/assets.json", media_end_ms=29984, scenario_end_ms=130000, variants=[c05a, c05b]),
        ]}
        texts = dict(u01="What am I doing right now?", u02="Exactly how many pieces are completely hidden under my hand?",
            u03="Please remember where I put the pepper core.", u04="Where did I last leave the pepper core?",
            u09=pov.TIMER_SCRIPT, u10=pov.QUIET_SCRIPT, u11="What am I holding right now?",
            u12="Can you tell whether the entire panel is clean now?", u13=pov.RESUME_SCRIPT)
        clips = []
        for index, (identity, text) in enumerate(texts.items()):
            stream = io.BytesIO()
            with wave.open(stream, "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(48000)
                wav.writeframes(struct.pack("<hhhh", 0, 128 + index, -256, 0))
            raw = stream.getvalue()
            save(self.pack / "inputs/speech" / (identity + ".wav"), raw)
            clips.append(dict(id=identity, text=text, relative_path=identity + ".wav", sha256=pov.base.digest(raw),
                sample_rate_hz=48000, channels=1, sample_width_bytes=2, sample_count=4,
                first_active_sample=1, last_active_sample_exclusive=3, energy_threshold_abs_pcm16=128,
                original_environment_audio=False, annotation="ORACLE_ONLY_SENTINEL"))
        self.speech = dict(format="jarvis.speech_assets.v1", clips=clips)
        self.oracle = {"cases": []}
        for case in self.timeline["cases"]:
            variants = [dict(id=v["variant_id"], required=["ORACLE_ONLY_SENTINEL"]) for v in case["variants"]]
            self.oracle["cases"].append(dict(case_id=case["case_id"], source="fixture-" + case["case_id"],
                evidence=[dict(observed="ORACLE_ONLY_SENTINEL")], variants=variants))
        self.oracle["cases"][2]["variants"][0].update(quiet_window_ms=[0, 110000],
            direct_question_windows_ms=[[8000, 25000], [96000, 108000]], minimum_quiet_ms=90000)
        self.flush()

    def flush(self):
        for case_id, assets in self.assets.items():
            save(self.pack / "inputs" / case_id / "assets.json", assets)
        save(self.pack / "inputs/timelines.json", self.timeline)
        save(self.pack / "inputs/speech/speech.json", self.speech)
        save(self.pack / "evaluator/expectations.json", self.oracle)

    def prepare(self, *variants, validate=False):
        return pov.prepare(self.pack, self.output, variants=variants or tuple(pov.EXPANDED_VARIANTS),
                           validate_staging=validate)

    def read(self, variant, relative):
        return json.loads((self.output / variant / relative).read_bytes())

    def test_defaults_keep_only_legacy_c01_a_b(self):
        report = pov.prepare(self.pack, self.output)
        self.assertEqual(list(pov.VARIANTS), ["c01-a", "c01-b"])
        self.assertEqual([c["variant_id"] for c in report["cases"]], ["c01-a", "c01-b"])
        self.assertEqual([self.read(v, "runtime/manifest.json")["duration_seconds"] for v in pov.VARIANTS], [45, 45])

    def test_c02_exact_clock_prelude_and_withheld_selection(self):
        self.prepare("c02-a", "c02-b")
        for variant, count in (("c02-a", 16), ("c02-b", 9)):
            manifest = self.read(variant, "runtime/manifest.json")
            self.assertEqual(manifest["duration_seconds"], 95)
            self.assertEqual([f["at_seconds"] for f in manifest["frames"]], list(range(35, 35 + count * 2, 2)))
            self.assertEqual(manifest["utterances"], [
                dict(clip_id="u03", trigger=dict(kind="at_time", at_seconds=5)),
                dict(clip_id="u04", trigger=dict(kind="at_time", at_seconds=75))])
            qualifier = self.read(variant, "evaluator/qualification.json")
            self.assertEqual(qualifier["prelude"]["reservation_end_seconds"], 35)
            self.assertEqual(qualifier["prelude"]["maximum_setup_seconds"], 30)
            self.assertFalse(qualifier["prelude"]["correlated_setup_proven"])
            self.assertFalse(qualifier["prelude"]["decoded_drain_proven"])
            self.assertFalse(qualifier["scope"]["baseline_memory_reuse_allowed"])
            self.assertFalse(qualifier["scope"]["multisession"])
        joins = self.read("c02-b", "evaluator/joins.json")
        self.assertEqual(joins["withheld_at_or_after_source_ms"], 18000)
        self.assertEqual([f["source_offset_ms"] for f in joins["frames"]], list(range(0, 18000, 2000)))
        self.assertFalse((self.output / "c02-b/runtime/frames/000010.jpg").exists())

    def test_c05_quiet_semantics_165_scene_and_actual_timer_basis(self):
        self.prepare("c05-a")
        manifest = self.read("c05-a", "runtime/manifest.json")
        self.assertEqual(manifest["duration_seconds"], 165)
        self.assertEqual([u["trigger"]["at_seconds"] for u in manifest["utterances"]], [5, 35, 43, 131, 145])
        qualifier = self.read("c05-a", "evaluator/qualification.json")
        quiet = qualifier["quiet"]
        self.assertEqual(quiet["script_check"], "matches_until_explicit_resume")
        self.assertFalse(quiet["instruction_has_fixed_expiry"])
        self.assertEqual(quiet["measured_native_window_seconds"], [35, 145])
        self.assertEqual((quiet["measured_seconds"], quiet["minimum_seconds"]), (110, 90))
        self.assertEqual((quiet["required_reconciliation_cycles"], quiet["drain_guard_seconds"]), (2, 5))
        self.assertFalse(quiet["actual_quiet_acceptance_proven"])
        self.assertIsNone(qualifier["timer"]["nominal_due_seconds"])
        self.assertTrue(qualifier["timer"]["script_matches_frozen_ten_seconds"])
        release = self.read("c05-a", "RELEASE_TEMPLATE.json")
        self.assertEqual(release["workload"], "reasoning-reminder")
        self.assertEqual(release["candidate_limits_not_independently_enforced_by_this_template"]["host_lease_seconds"], 525)
        self.assertFalse(release["allow_one_launch"])
        self.assertIsNone(release["authorization_id"])

    def test_c05_source_spacing_is_not_replaced_by_uniform_two_seconds(self):
        self.prepare("c05-a", "c05-b")
        for variant, offset in (("c05-a", 35), ("c05-b", 5)):
            frames = self.read(variant, "runtime/manifest.json")["frames"]
            self.assertEqual([Decimal(str(f["at_seconds"])) * 1000 for f in frames],
                             [offset * 1000 + f["offset_ms"] for f in self.assets["c05"]["inputs"]["frames"]])
        self.assertEqual(self.read("c05-a", "runtime/manifest.json")["frames"][-1]["at_seconds"], 63.015)
        self.assertEqual(self.read("c05-b", "runtime/manifest.json")["duration_seconds"], 50)
        self.assertEqual([u["trigger"]["at_seconds"] for u in self.read("c05-b", "runtime/manifest.json")["utterances"]], [5, 13])

    def test_c05_explicit_zero_tail_survives_actual_stager_and_capture_budget(self):
        from evals.responsive_voice import native_replay_run as runner
        from evals.responsive_voice.frontend_host_qualification import QualificationFailure

        self.prepare("c05-b", "c05-a")
        for variant, duration, count, wait, lease in (
            ("c05-b", 50, 17, 80, 410), ("c05-a", 165, 20, 195, 525)
        ):
            with self.subTest(variant=variant):
                runtime = self.output / variant / "runtime"
                manifest = self.read(variant, "runtime/manifest.json")
                self.assertEqual(manifest["post_media_capture_seconds"], 0)
                plan, staged_count = pov._stage(runtime,
                    pov.base.digest((runtime / "manifest.json").read_bytes()),
                    workload=pov.EXPANDED_VARIANTS[variant]["workload"])
                self.assertEqual(staged_count, count)
                self.assertEqual(plan["duration_seconds"], duration)
                self.assertEqual(plan["post_media_capture_seconds"], 0)
                self.assertEqual(plan["utterances"], manifest["utterances"])
                self.assertEqual([(f["at_seconds"], f["sha256"]) for f in plan["frames"]],
                                 [(f["at_seconds"], f["sha256"]) for f in manifest["frames"]])
                tail = runner.effective_capture_tail(plan)
                self.assertEqual(tail, 0)
                limits = self.read(variant, "RELEASE_TEMPLATE.json")[
                    "candidate_limits_not_independently_enforced_by_this_template"]
                self.assertEqual((limits["capture_wait_seconds"], limits["host_lease_seconds"]),
                                 (wait, lease))
                budget = runner.capture_budget(plan["duration_seconds"], limits["host_lease_seconds"],
                    workload=pov.EXPANDED_VARIANTS[variant]["workload"], post_media_capture_seconds=tail)
                self.assertEqual((budget["capture_wait_seconds"], budget["minimum_lease_seconds"],
                                  budget["lease_seconds"]), (wait, lease, lease))
                omitted = dict(plan)
                del omitted["post_media_capture_seconds"]
                default_tail = runner.effective_capture_tail(omitted)
                self.assertEqual(default_tail, min(20, runner.MAX_DURATION - duration))
                self.assertGreater(default_tail, 0)
                with self.assertRaisesRegex(QualificationFailure, "^lease_too_short_for_capture_and_export$"):
                    runner.capture_budget(duration, lease,
                        workload=pov.EXPANDED_VARIANTS[variant]["workload"],
                        post_media_capture_seconds=default_tail)

    def test_all_silence_timer_controls_require_enabled_actual_machinery(self):
        self.prepare("c05-a", "c05-b", "c01-c")
        for variant in ("c05-a", "c05-b", "c01-c"):
            profile = self.read(variant, "evaluator/qualification.json")["required_actual_profile"]
            self.assertTrue(profile["attention_voice_enabled"])
            self.assertTrue(profile["reminder_scheduler_enabled"])
            self.assertTrue(profile["worker_readiness_required"])
            self.assertFalse(profile["silence_is_sufficient"])
            self.assertFalse(profile["disabled_machinery_silence_can_pass"])
            self.assertEqual(profile["record_output_channel_eligibility"], ["speech", "UI", "push"])

    def test_c01_c_has_no_dummy_speech(self):
        self.prepare("c01-c", validate=True)
        self.assertEqual(self.read("c01-c", "runtime/manifest.json")["utterances"], [])
        self.assertEqual(self.read("c01-c", "runtime/speech/speech.json")["clips"], [])
        self.assertEqual(list((self.output / "c01-c/runtime").rglob("*.wav")), [])

    def test_all_five_existing_prepared_staging_and_runtime_whitelist(self):
        from evals.responsive_voice.native_replay_run import stage_prepared_media

        with patch("evals.responsive_voice.native_replay_run.stage_prepared_media",
                   wraps=stage_prepared_media) as stage:
            report = self.prepare(validate=True)
        self.assertEqual([call.args[0].workload for call in stage.call_args_list],
                         [pov.EXPANDED_VARIANTS[c["variant_id"]]["workload"] for c in report["cases"]])
        self.assertEqual([call.args[0].workload for call in stage.call_args_list],
                         ["none", "none", "reasoning-reminder", "reasoning-reminder", "reasoning-reminder"])
        self.assertEqual([call.args[0].result_release_policy for call in stage.call_args_list],
                         ["all_sdk_inputs"] * 5)
        self.assertEqual([c["offline_prepared_staging"] for c in report["cases"]], ["passed"] * 5)
        self.assertEqual(report["copied_jpeg_files"], 71)
        self.assertEqual(report["copied_wav_files"], 11)
        self.assertEqual((report["native_runs"], report["provider_calls"], report["sql_calls"]), (0, 0, 0))
        for case in report["cases"]:
            variant = case["variant_id"]
            manifest = self.read(variant, "runtime/manifest.json")
            fields = {"version", "run_id", "backend_base_url", "owner_id", "area_id",
                "memory_bank_id", "voice_route", "bearer_token_file", "duration_seconds", "speech_manifest", "frames", "utterances"}
            if variant in {"c05-a", "c05-b"}:
                fields.add("post_media_capture_seconds")
                self.assertEqual(manifest["post_media_capture_seconds"], 0)
            else:
                self.assertNotIn("post_media_capture_seconds", manifest)
            self.assertEqual(set(manifest), fields)
            for clip in self.read(variant, "runtime/speech/speech.json")["clips"]:
                self.assertEqual(set(clip), set(pov.SPEECH_FIELDS))
            for path in (self.output / variant / "runtime").rglob("*"):
                if path.is_file():
                    self.assertNotIn(b"ORACLE_ONLY_SENTINEL", path.read_bytes())
                    self.assertNotIn(pov.QUIET_SCRIPT.encode(), path.read_bytes())
            for item in self.read(variant, "evaluator/joins.json")["frames"]:
                raw = (self.output / variant / "runtime" / item["runtime_relative_path"]).read_bytes()
                self.assertEqual(raw, Path(item["source_path"]).read_bytes())
                self.assertEqual(pov.base.digest(raw), item["sha256"])

    def test_staging_requires_explicit_case_workload(self):
        with self.assertRaisesRegex(TypeError, "workload"):
            pov._stage(self.output, "0" * 64)

    def test_timed_quiet_mismatch_is_flagged_without_rewriting_oracle_or_wav(self):
        next(c for c in self.speech["clips"] if c["id"] == "u10")["text"] = "Please do not disturb me for ninety seconds."
        self.flush()
        oracle_before = (self.pack / "evaluator/expectations.json").read_bytes()
        wav_before = (self.pack / "inputs/speech/u10.wav").read_bytes()
        self.prepare("c05-a")
        quiet = self.read("c05-a", "evaluator/qualification.json")["quiet"]
        self.assertEqual(quiet["script_check"], "mismatch_or_unrecognized")
        self.assertIsNone(quiet["instruction_has_fixed_expiry"])
        self.assertEqual(self.read("c05-a", "evaluator/expectations.json")["original_variant"],
                         self.oracle["cases"][2]["variants"][0])
        self.assertEqual((self.pack / "evaluator/expectations.json").read_bytes(), oracle_before)
        self.assertEqual((self.output / "c05-a/runtime/speech/u10.wav").read_bytes(), wav_before)

    def test_future_frame_or_changed_bytes_rejected(self):
        self.assets["c05"]["inputs"]["frames"][2]["offset_ms"] = 4001
        self.flush()
        with self.assertRaisesRegex(pov.base.ScenarioError, "source_frame_timing_changed_or_future"):
            self.prepare("c05-a")
        self.assets["c05"]["inputs"]["frames"][2]["offset_ms"] = 4002
        self.flush()
        (self.pack / "inputs/c02/frames/000001.jpg").write_bytes(b"changed")
        with self.assertRaisesRegex(pov.base.ScenarioError, "frame_hash_mismatch"):
            self.prepare("c02-a")

    def test_multisession_and_duplicate_selection_remain_out_of_scope(self):
        for variants in (("c02-c",), ("c05-a", "c05-a")):
            with self.subTest(variants=variants), self.assertRaisesRegex(pov.base.ScenarioError, "unsupported_or_duplicate_variant"):
                self.prepare(*variants)

    def test_existing_output_never_overwritten(self):
        self.output.mkdir()
        marker = self.output / "main-owned"
        marker.write_bytes(b"keep")
        with self.assertRaises(FileExistsError):
            self.prepare("c02-a")
        self.assertEqual(marker.read_bytes(), b"keep")


if __name__ == "__main__":
    unittest.main()
