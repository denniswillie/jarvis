"""Offline prepared-scene limits, independent timeouts and zero-input controls.

Synthetic JPEG markers exercise staging, not image decoding or visual quality.
No runtime host, simulator, provider, SQL or authored POV media is used.
"""
from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
import struct
import tempfile
import unittest
import wave
from unittest.mock import patch

from evals.responsive_voice import native_interruption_scenarios as compiler
from evals.responsive_voice import native_replay_run as runner


def save(path, value):
    path.write_text(json.dumps(value) + "\n")


class PreparedBoundsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.number = 0
        for target in ("socket.socket.connect", "subprocess.Popen", "asyncpg.connect", "asyncpg.create_pool"):
            guard = patch(target, side_effect=AssertionError("external execution forbidden"))
            guard.start()
            self.addCleanup(guard.stop)

    def options(self, *, duration=45, empty=False, task_aware=False, scenario="baseline", cuts=0,
                input_mode=runner.PREPARED, utterances=None, frame_at=0):
        self.number += 1
        root = self.root / str(self.number)
        media, artifact = root / "media", root / "artifact"
        media.mkdir(parents=True)
        artifact.mkdir()
        jpeg = b"\xff\xd8offline-contract-only\xff\xd9"
        (media / "frame.jpg").write_bytes(jpeg)
        clips = []
        if not empty:
            stream = io.BytesIO()
            with wave.open(stream, "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(48000)
                wav.writeframes(struct.pack("<hhhh", 0, 128, -256, 0))
            data = stream.getvalue()
            (media / "clip.wav").write_bytes(data)
            clips = [{"id": "question", "relative_path": "clip.wav", "sha256": hashlib.sha256(data).hexdigest(),
                      "sample_rate_hz": 48000, "channels": 1, "sample_width_bytes": 2,
                      "sample_count": 4, "first_active_sample": 1, "last_active_sample_exclusive": 3,
                      "energy_threshold_abs_pcm16": 128}]
        save(media / "speech.json", {"format": "jarvis.speech_assets.v1", "clips": clips})
        manifest = {
            "version": 1, "run_id": "compiler-placeholder", "backend_base_url": "http://127.0.0.1:1",
            "owner_id": "offline-owner", "area_id": "offline-area", "memory_bank_id": "offline-bank",
            "voice_route": "realtime", "bearer_token_file": "absent-token.txt", "duration_seconds": duration,
            "speech_manifest": "speech.json",
            "frames": [{"at_seconds": frame_at, "relative_path": "frame.jpg", "sha256": hashlib.sha256(jpeg).hexdigest()}],
            "utterances": utterances if utterances is not None else ([] if empty else [self.at_time(5)]),
        }
        if task_aware:
            manifest["task_barrier"] = {"version": 1, "scope_file": "task-scope.json", "request_input_ordinal": 0}
        save(media / "manifest.json", manifest)
        return runner.RunOptions(
            input_manifest=media / "manifest.json", media_root=media, speech_manifest=media / "speech.json",
            artifact_dir=artifact, ledger_dir=root / "unused-ledger", risk_authorization=root / "absent-release.json",
            canonical_export_command=root / "absent-exporter.json", hindsight_url="http://127.0.0.1:1",
            hindsight_api_key_file=root / "absent-key", input_mode=input_mode, scenario=scenario, cuts=cuts,
            task_aware=task_aware,
        )

    @staticmethod
    def at_time(at):
        return {"clip_id": "question", "trigger": {"kind": "at_time", "at_seconds": at}}

    @staticmethod
    def stage(options):
        digest = hashlib.sha256(options.input_manifest.read_bytes()).hexdigest()
        return runner.stage_media(options, expected_manifest_sha256=digest)

    @staticmethod
    def change_manifest(options, **changes):
        manifest = json.loads(options.input_manifest.read_text())
        manifest.update(changes)
        save(options.input_manifest, manifest)

    def task_sequence(self, kind, timeout):
        sequence = [self.at_time(1), {"clip_id": "question", "trigger": {
            "kind": "after_task_claim", "after_input_ordinal": 0,
            "timeout_seconds": timeout if kind == "after_task_claim" else 1}}]
        if kind == "after_input_complete":
            sequence.append({"clip_id": "question", "trigger": {"kind": kind, "after_input_ordinal": 1,
                            "delay_seconds": 0, "timeout_seconds": timeout}})
        if kind == "output_start":
            cut = {"kind": kind, "after_cursor": 0, "delay_seconds": .1, "timeout_seconds": timeout}
        else:
            cut = {"kind": "response_output_start", "after_input_ordinal": len(sequence) - 1,
                   "delay_seconds": .1, "timeout_seconds": timeout if kind == "response_output_start" else 1,
                   "max_pcm_age_seconds": .05, "require_task_claim": True}
        return sequence + [{"clip_id": "question", "trigger": cut}]

    def test_scene_180_accepts_late_frames_and_speech(self):
        options = self.options(duration=180, frame_at=179, utterances=[self.at_time(174)])
        plan, records = self.stage(options)
        self.assertEqual(plan["duration_seconds"], 180)
        self.assertEqual(plan["frames"][0]["at_seconds"], 179)
        self.assertEqual(plan["utterances"], [self.at_time(174)])
        self.assertEqual(len(records), 2)

    def test_scene_duration_rejects_outside_finite_bounds(self):
        for duration in (180.001, 181, 0, -1, True):
            with self.subTest(duration=duration), self.assertRaisesRegex(compiler.ScenarioError, "invalid_duration"):
                self.stage(self.options(duration=duration))
        for duration in (float("nan"), float("inf")):
            with self.subTest(duration=duration), self.assertRaisesRegex(ValueError, "nonfinite_json"):
                self.stage(self.options(duration=duration))

    def test_pcm_ceiling_and_long_workload_lease(self):
        self.assertEqual(runner.MAX_CAPTURE_PCM_BYTES, 20_160_000)
        budget = runner.capture_budget(165, None, workload=runner.REASONING_REMINDER)
        self.assertEqual(budget["minimum_lease_seconds"], 525)
        self.assertEqual(budget["lease_seconds"], 525)
        self.assertEqual(budget["capture_wait_seconds"], 195)
        self.assertEqual(budget["decoded_pcm_byte_limit"], 18_720_000)
        self.assertEqual(runner.capture_budget(180, None)["lease_seconds"], 420)
        self.assertEqual(runner.capture_budget(45, None)["lease_seconds"], 300)
        self.assertEqual(runner.capture_budget(45, 285)["lease_seconds"], 285)
        with self.assertRaisesRegex(Exception, "lease_too_short_for_capture_and_export"):
            runner.capture_budget(165, 524, workload=runner.REASONING_REMINDER)

    def test_retained_pcm_copy_exact_ceiling_and_one_byte_over(self):
        source = self.root / "large-offline.pcm"
        with source.open("wb") as handle:
            handle.truncate(20_160_000)
        record = runner.copy_bounded(source, self.root / "retained.pcm", runner.MAX_CAPTURE_PCM_BYTES)
        self.assertEqual(record["bytes"], 20_160_000)
        with source.open("r+b") as handle:
            handle.truncate(20_160_001)
        with self.assertRaisesRegex(Exception, "artifact_byte_bound"):
            runner.copy_bounded(source, self.root / "must-not-exist.pcm", runner.MAX_CAPTURE_PCM_BYTES)
        self.assertFalse((self.root / "must-not-exist.pcm").exists())

    def test_all_task_aware_timeouts_keep_independent_120_ceiling(self):
        self.assertEqual(runner.MAX_TRIGGER_TIMEOUT, 120)
        for kind in ("output_start", "response_output_start", "after_task_claim", "after_input_complete"):
            with self.subTest(kind=kind, timeout=120):
                self.stage(self.options(duration=180, task_aware=True, scenario="pause_continue", cuts=1,
                           utterances=self.task_sequence(kind, 120)))
            with self.subTest(kind=kind, timeout=120.001), self.assertRaisesRegex(
                    compiler.ScenarioError, "invalid_(task_)?trigger_timeout"):
                self.stage(self.options(duration=180, task_aware=True, scenario="pause_continue", cuts=1,
                           utterances=self.task_sequence(kind, 120.001)))

    def test_timeout_still_cannot_exceed_short_scene(self):
        with self.assertRaisesRegex(compiler.ScenarioError, "invalid_task_trigger_timeout"):
            self.stage(self.options(duration=45, task_aware=True, scenario="pause_continue", cuts=1,
                       utterances=self.task_sequence("after_task_claim", 46)))

    def test_non_task_output_timeout_also_stays_120(self):
        for kind in ("output_start", "response_output_start"):
            sequence = self.task_sequence(kind, 121)
            cut = sequence[-1]["trigger"]
            if kind == "response_output_start":
                cut.pop("require_task_claim")
                cut["after_input_ordinal"] = 0
            with self.subTest(kind=kind), self.assertRaisesRegex(compiler.ScenarioError, "invalid_trigger_timeout"):
                self.stage(self.options(duration=180, scenario="pause_continue", cuts=1,
                           utterances=[self.at_time(1), sequence[-1]]))

    def test_zero_input_prepared_baseline_has_no_dummy_wav(self):
        options = self.options(duration=180, empty=True)
        plan, records = self.stage(options)
        self.assertEqual(plan["utterances"], [])
        self.assertEqual(len(records), 1)  # One visual fixture, no speech asset.
        self.assertEqual(json.loads((options.artifact_dir / "inputs/speech/speech.json").read_text()),
                         {"format": "jarvis.speech_assets.v1", "clips": []})
        self.assertEqual(list(options.artifact_dir.rglob("*.wav")), [])

    def test_zero_input_exception_does_not_admit_other_modes(self):
        for scope in ({"scenario": "pause_continue"}, {"scenario": "parallel_keep_working", "cuts": 1},
                      {"scenario": runner.SCENARIO}, {"task_aware": True}, {"cuts": 1},
                      {"cuts": False}, {"input_mode": "baseline"}):
            options = self.options(empty=True, **scope)
            digest = hashlib.sha256(options.input_manifest.read_bytes()).hexdigest()
            with self.subTest(scope=scope), self.assertRaisesRegex(Exception, "invalid_utterances"):
                runner.stage_prepared_media(options, digest)

    def test_empty_utterances_and_clips_must_agree(self):
        options = self.options()
        self.change_manifest(options, utterances=[])
        with self.assertRaisesRegex(Exception, "empty_prepared_input_mismatch"):
            self.stage(options)
        options = self.options(empty=True)
        self.change_manifest(options, utterances=[self.at_time(5)])
        with self.assertRaisesRegex(Exception, "empty_prepared_input_mismatch"):
            self.stage(options)

    def test_legacy_default_input_mode_remains_fixed_45(self):
        options = self.options(input_mode="baseline", scenario=runner.SCENARIO, cuts=None)
        manifest = json.loads(options.input_manifest.read_text())
        frame = manifest["frames"][0]
        manifest["frames"] = [dict(frame, at_seconds=at) for at in (20, 22, 24)]
        manifest["utterances"] = [{"clip_id": "current_activity", "trigger": {"kind": "at_time", "at_seconds": 25}}]
        save(options.input_manifest, manifest)
        speech = json.loads(options.speech_manifest.read_text())
        speech["clips"][0]["id"] = "current_activity"
        save(options.speech_manifest, speech)
        plan, _ = self.stage(options)
        self.assertEqual(plan["duration_seconds"], 45)
        self.assertEqual(runner.RunOptions.__dataclass_fields__["input_mode"].default, "baseline")
        for duration in (120, 165, 180):
            self.change_manifest(options, duration_seconds=duration)
            with self.subTest(duration=duration), self.assertRaisesRegex(Exception, "only_reviewed_baseline_supported"):
                self.stage(replace(options, artifact_dir=self.root / "never-staged"))

    def test_duration_change_invalidates_exact_manifest_authorization_hash(self):
        options = self.options()
        old_hash = hashlib.sha256(options.input_manifest.read_bytes()).hexdigest()
        self.change_manifest(options, duration_seconds=165)
        with self.assertRaisesRegex(Exception, "authorized_manifest_hash_mismatch"):
            runner.stage_prepared_media(options, old_hash)

    def test_input_pcm_budget_did_not_expand(self):
        options = self.options(duration=180)
        speech = json.loads(options.speech_manifest.read_text())
        speech["clips"][0]["sample_count"] = 60 * 48000 + 1
        save(options.speech_manifest, speech)
        with self.assertRaisesRegex(compiler.ScenarioError, "invalid_speech_sample_count"):
            self.stage(options)

    def test_task_interruption_compiler_default_and_cap_remain_120(self):
        class ReachedLegacyCompiler(Exception):
            pass

        with patch.object(compiler, "_compile_legacy_scenario", side_effect=ReachedLegacyCompiler) as original:
            with self.assertRaises(ReachedLegacyCompiler):
                compiler.compile_scenario("pause_continue", cuts=1, task_aware=True,
                                          binding={"voice_route": "realtime"})
            self.assertEqual(original.call_args.kwargs["duration"], 120)
        with self.assertRaisesRegex(compiler.ScenarioError, "invalid_duration"):
            compiler.compile_scenario("pause_continue", cuts=1, task_aware=True, duration=120.001,
                                      binding={"voice_route": "realtime"})


if __name__ == "__main__":
    unittest.main()
