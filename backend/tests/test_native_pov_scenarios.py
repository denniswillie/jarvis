"""Offline contract tests. Fixture JPEG markers test staging, not image decoding."""
import io
import json
from pathlib import Path
import struct
import tempfile
import unittest
import wave
from unittest.mock import patch

from evals.responsive_voice import native_pov_scenarios as pov


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value if isinstance(value, bytes) else pov.encoded(value))


class NativePOVTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.pack, self.output = self.root / "pack", self.root / "prepared"
        frames = []
        for index in range(16):
            raw = b"\xff\xd8offline-contract-fixture-" + bytes([index]) + b"\xff\xd9"
            relative = f"frames/{index + 1:06d}.jpg"
            put(self.pack / "inputs/c01" / relative, raw)
            frames.append({"offset_ms": index * 2000, "relative_path": relative, "sha256": pov.base.digest(raw)})
        self.assets = {
            "format": "jarvis.video_assets.v1", "source_audio_included": False,
            "source_sha256": "f" * 64, "source_first_pts_seconds": 0,
            "source_pts_seconds": list(range(0, 32, 2)), "source_frame_indices": list(range(0, 1920, 120)),
            "inputs": {"frames": frames, "initial_context": [], "events": []},
            "permission": {"hosted_inference": False},
        }
        self.timeline = {"case_id": "c01", "assets": "c01/assets.json", "media_end_ms": 32000,
                         "scenario_end_ms": 40000, "variants": [
                             {"variant_id": variant, "prior_context_utterance_ids": [],
                              "events": [{"occurrence_id": variant + "-01", "at_ms": 18000, "utterance_id": clip}]}
                             for variant, clip in pov.VARIANTS.items()]}
        self.speech = {"format": "jarvis.speech_assets.v1", "clips": []}
        for clip in pov.VARIANTS.values():
            buffer = io.BytesIO()
            with wave.open(buffer, "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(48000)
                wav.writeframes(struct.pack("<hhhh", 0, 128, -129, 0))
            raw = buffer.getvalue()
            put(self.pack / "inputs/speech" / (clip + ".wav"), raw)
            self.speech["clips"].append({
                "id": clip, "relative_path": clip + ".wav", "sha256": pov.base.digest(raw),
                "sample_rate_hz": 48000, "channels": 1, "sample_width_bytes": 2, "sample_count": 4,
                "first_active_sample": 1, "last_active_sample_exclusive": 3,
                "energy_threshold_abs_pcm16": 128, "original_environment_audio": False,
                "text": "SCRIPT_ONLY_SENTINEL", "unexpected_answer": "ORACLE_ONLY_SENTINEL",
            })
        self.oracle = {"cases": [{"case_id": "c01", "source": "fixture-source",
                                  "evidence": [{"observed": "ORACLE_ONLY_SENTINEL"}],
                                  "variants": [{"id": variant, "required": ["ORACLE_ONLY_SENTINEL"]}
                                               for variant in pov.VARIANTS]}]}
        self.flush_metadata()

    def flush_metadata(self):
        put(self.pack / "inputs/c01/assets.json", self.assets)
        put(self.pack / "inputs/timelines.json", {"cases": [self.timeline]})
        put(self.pack / "inputs/speech/speech.json", self.speech)
        put(self.pack / "evaluator/expectations.json", self.oracle)

    def prepare(self, **kwargs):
        return pov.prepare(self.pack, self.output, **kwargs)

    def test_exact_wrappers_and_no_annotation_leakage(self):
        report = self.prepare()
        self.assertEqual(report["copied_jpeg_files"], 32)
        self.assertEqual(report["copied_wav_files"], 2)
        for variant, clip in pov.VARIANTS.items():
            runtime = self.output / variant / "runtime"
            manifest = json.loads((runtime / "manifest.json").read_bytes())
            speech = json.loads((runtime / "speech/speech.json").read_bytes())
            self.assertEqual([f["at_seconds"] for f in manifest["frames"]], list(range(5, 37, 2)))
            self.assertEqual(manifest["duration_seconds"], 45)
            self.assertEqual(manifest["utterances"], [{"clip_id": clip, "trigger": {"kind": "at_time", "at_seconds": 23}}])
            self.assertEqual(len(speech["clips"]), 1)
            self.assertEqual(set(speech["clips"][0]), set(pov.SPEECH_FIELDS))
            self.assertEqual(speech["clips"][0]["id"], clip)
            self.assertFalse((runtime / "token.txt").exists())
            for frame in manifest["frames"]:
                self.assertEqual((runtime / frame["relative_path"]).read_bytes(),
                                 (self.pack / "inputs/c01" / frame["relative_path"]).read_bytes())
            self.assertEqual((runtime / "speech" / (clip + ".wav")).read_bytes(),
                             (self.pack / "inputs/speech" / (clip + ".wav")).read_bytes())
            for path in runtime.rglob("*"):
                if path.is_file():
                    self.assertNotIn(b"SCRIPT_ONLY_SENTINEL", path.read_bytes())
                    self.assertNotIn(b"ORACLE_ONLY_SENTINEL", path.read_bytes())
            joins = json.loads((self.output / variant / "evaluator/joins.json").read_bytes())
            self.assertEqual([f["source_pts_seconds"] for f in joins["frames"]], list(range(0, 32, 2)))
            self.assertEqual(joins["utterance"]["sidecar_occurrence_id"], variant + "-01")
            self.assertEqual(joins["utterance"]["native_input_ordinal"], 0)
            release = json.loads((self.output / variant / "RELEASE_TEMPLATE.json").read_bytes())
            self.assertFalse(release["allow_one_launch"])
            self.assertIsNone(release["authorization_id"])
            self.assertEqual(release["input_manifest_sha256"], pov.base.digest((runtime / "manifest.json").read_bytes()))
        self.assertFalse(json.loads((self.pack / "inputs/c01/assets.json").read_bytes())["permission"]["hosted_inference"])

    def test_existing_prepared_staging_both_cases_without_external_io(self):
        from evals.responsive_voice.native_replay_run import stage_prepared_media

        with patch("socket.socket.connect", side_effect=AssertionError("network forbidden")), \
             patch("subprocess.Popen", side_effect=AssertionError("native/process forbidden")), \
             patch("asyncpg.connect", side_effect=AssertionError("SQL forbidden")), \
             patch("asyncpg.create_pool", side_effect=AssertionError("SQL forbidden")), \
             patch("evals.responsive_voice.native_replay_run.stage_prepared_media",
                   wraps=stage_prepared_media) as stage:
            report = self.prepare(validate_staging=True)
        self.assertEqual([call.args[0].workload for call in stage.call_args_list], ["none", "none"])
        self.assertEqual([call.args[0].result_release_policy for call in stage.call_args_list],
                         ["all_sdk_inputs", "all_sdk_inputs"])
        self.assertEqual([c["offline_prepared_staging"] for c in report["cases"]], ["passed", "passed"])
        self.assertEqual([c["temporary_staged_media_files"] for c in report["cases"]], [17, 17])
        self.assertEqual(report["native_runs"], 0)
        self.assertFalse(report["temporary_staging_copies_retained"])

    def test_rejects_changed_frame_bytes(self):
        put(self.pack / "inputs/c01/frames/000001.jpg", b"changed")
        with self.assertRaisesRegex(pov.base.ScenarioError, "frame_hash_mismatch"):
            self.prepare()
        self.assertFalse(self.output.exists())

    def test_rejects_future_or_retimed_source_frame(self):
        self.assets["source_pts_seconds"][1] = 2.001
        self.flush_metadata()
        with self.assertRaisesRegex(pov.base.ScenarioError, "source_frame_timing_changed_or_future"):
            self.prepare()

    def test_rejects_escaped_frame_path(self):
        self.assets["inputs"]["frames"][0]["relative_path"] = "../outside.jpg"
        self.flush_metadata()
        with self.assertRaisesRegex(pov.base.ScenarioError, "asset_path_escape"):
            self.prepare()

    def test_rejects_changed_speech_bytes(self):
        put(self.pack / "inputs/speech/u02.wav", b"changed")
        with self.assertRaisesRegex(pov.base.ScenarioError, "speech_hash_mismatch"):
            self.prepare()
        self.assertFalse(self.output.exists())

    def test_rejects_prelude_instead_of_silently_dropping_it(self):
        self.timeline["variants"][0]["prior_context_utterance_ids"] = ["u03"]
        self.flush_metadata()
        with self.assertRaisesRegex(pov.base.ScenarioError, "unsupported_c01_variant_timeline"):
            self.prepare()

    def test_no_overwrite_and_no_unsupported_variant(self):
        self.output.mkdir()
        owner = self.output / "owned-by-main"
        owner.write_bytes(b"leave unchanged")
        with self.assertRaises(FileExistsError):
            self.prepare()
        self.assertEqual(owner.read_bytes(), b"leave unchanged")
        with self.assertRaisesRegex(pov.base.ScenarioError, "unsupported_or_duplicate_variant"):
            self.prepare(variants=("c02-c",))


if __name__ == "__main__":
    unittest.main()
