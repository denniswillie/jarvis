"""Offline byte-lineage tests; no native renderer, network or provider calls."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from evals.responsive_voice import native_pov_scenarios as pov
from evals.responsive_voice import native_replay_run as runner


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()


class FrameLineageTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.pack, self.runtime = self.root / "pack", self.root / "prepared/runtime"
        self.variant = "c01-c"
        self.output = self.root / "receipts/lineage.json"
        self.asset_root = self.pack / "inputs/c01"
        (self.asset_root / "frames").mkdir(parents=True)
        (self.runtime / "frames").mkdir(parents=True)
        (self.runtime / "speech").mkdir()
        frames = []
        for ordinal, offset in enumerate((0, 2000, 4002)):
            # The shared staging validator checks JPEG signatures only. These
            # fixtures assert byte identity, not successful image decoding.
            data = b"\xff\xd8" + bytes([ordinal]) + b"\xff\xd9"
            relative = f"frames/{ordinal}.jpg"
            (self.asset_root / relative).write_bytes(data)
            (self.runtime / relative).write_bytes(data)
            frames.append({"offset_ms": offset, "relative_path": relative,
                           "sha256": hashlib.sha256(data).hexdigest(), "width": 1920, "height": 1440})
        self.assets = {"source_sha256": "a" * 64, "source_first_pts_seconds": 0,
                       "source_frame_indices": [0, 60, 120], "source_pts_seconds": [0, 2, 4.001667],
                       "inputs": {"frames": frames}}
        self.manifest = {"version": 1, "run_id": "00000000-0000-0000-0000-000000000000",
            "backend_base_url": "http://127.0.0.1:1", "owner_id": "00000000-0000-0000-0000-000000000000",
            "area_id": "00000000-0000-0000-0000-000000000000",
            "memory_bank_id": "00000000-0000-0000-0000-000000000000", "voice_route": "realtime",
            "bearer_token_file": "compiler-placeholder", "duration_seconds": 45,
            "speech_manifest": "speech/speech.json", "utterances": [],
            "frames": [{"relative_path": f["relative_path"], "sha256": f["sha256"],
                        "at_seconds": (5000 + f["offset_ms"]) / 1000} for f in frames]}
        (self.runtime / "speech/speech.json").write_bytes(encoded({"format": "jarvis.speech_assets.v1", "clips": []}))
        self.save()
        for name in ("socket.socket", "subprocess.Popen"):
            guard = patch(name, side_effect=AssertionError("offline only"))
            guard.start()
            self.addCleanup(guard.stop)

    def save(self):
        (self.asset_root / "assets.json").write_bytes(encoded(self.assets))
        self.manifest_bytes = encoded(self.manifest)
        (self.runtime / "manifest.json").write_bytes(self.manifest_bytes)
        self.manifest_hash = hashlib.sha256(self.manifest_bytes).hexdigest()

    def run_receipt(self, **overrides):
        arguments = dict(pack_root=self.pack, variant=self.variant, runtime=self.runtime,
                         output=self.output, expected_manifest_sha256=self.manifest_hash)
        arguments.update(overrides)
        return pov.write_frame_lineage_receipt(**arguments)

    def test_real_stage_identity_clock_and_no_runtime_changes(self):
        before = {str(p.relative_to(self.runtime)): p.read_bytes() for p in self.runtime.rglob("*") if p.is_file()}
        with patch.object(runner, "stage_prepared_media", wraps=runner.stage_prepared_media) as stage:
            receipt = self.run_receipt()
        stage.assert_called_once()
        self.assertEqual(stage.call_args.args[0].workload, "reasoning-reminder")
        self.assertEqual(stage.call_args.args[0].result_release_policy, "all_sdk_inputs")
        after = {str(p.relative_to(self.runtime)): p.read_bytes() for p in self.runtime.rglob("*") if p.is_file()}
        self.assertEqual(before, after)
        self.assertEqual(receipt["runtime_manifest"]["sha256"], self.manifest_hash)
        self.assertEqual(receipt["preparation"]["frame_count"], 3)
        self.assertEqual(receipt["preparation"]["jpeg_bytes"], 15)
        self.assertFalse(receipt["preparation"]["resize_applied"])
        self.assertIsNone(receipt["preparation"]["resize_parameters"])
        self.assertEqual(receipt["frames"][-1]["staged"]["at_seconds"], 9.002)
        for i, frame in enumerate(receipt["frames"]):
            self.assertEqual(frame["source"]["sha256"], frame["staged"]["sha256"])
            self.assertEqual(frame["staged"]["relative_path"], f"frames/frame-{i}.jpg")
        self.assertEqual(self.output.stat().st_mode & 0o777, 0o600)

    def test_native_contract_is_not_an_observation_or_output_hash(self):
        receipt = self.run_receipt()
        native = receipt["native_transform"]
        self.assertEqual((native["max_dimension"], native["jpeg_compression_quality"]), (768, 0.7))
        self.assertFalse(native["executed_here"])
        self.assertFalse(native["native_input_output_hash_join_proven"])
        self.assertIsNone(native["native_build_binding"])
        for frame in receipt["frames"]:
            self.assertIsNone(frame["native"]["output_sha256"])
            self.assertIsNone(frame["native"]["frame_sequence"])
            self.assertEqual(frame["native"]["conditional_output_dimensions"], {"width": 768, "height": 576})

    def test_reject_wrong_manifest_pin(self):
        with self.assertRaisesRegex(ValueError, "manifest_hash_mismatch"):
            self.run_receipt(expected_manifest_sha256="f" * 64)
        self.assertFalse(self.output.exists())

    def test_reject_modified_original(self):
        (self.asset_root / "frames/0.jpg").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "source_hash_mismatch"):
            self.run_receipt()

    def test_reject_modified_prepared_bytes(self):
        (self.runtime / "frames/0.jpg").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "prepared_bytes_mismatch"):
            self.run_receipt()

    def test_reject_stage_byte_mutation(self):
        real_stage = runner.stage_prepared_media
        def corrupt(options, pin):
            plan, records = real_stage(options, pin)
            (options.artifact_dir / "inputs" / plan["frames"][0]["relative_path"]).write_bytes(b"changed")
            return plan, records
        with patch.object(runner, "stage_prepared_media", side_effect=corrupt):
            with self.assertRaisesRegex(ValueError, "staged_bytes_mismatch"):
                self.run_receipt()
        self.assertFalse(self.output.exists())

    def test_reject_stage_clock_mutation(self):
        real_stage = runner.stage_prepared_media
        def corrupt(options, pin):
            plan, records = real_stage(options, pin)
            plan["frames"][0]["at_seconds"] += 1
            return plan, records
        with patch.object(runner, "stage_prepared_media", side_effect=corrupt):
            with self.assertRaisesRegex(ValueError, "staged_clock_mismatch"):
                self.run_receipt()

    def test_reject_future_source_clock_and_retiming(self):
        self.assets["inputs"]["frames"][2]["offset_ms"] = 4001
        self.save()
        with self.assertRaisesRegex(ValueError, "source_clock_mismatch"):
            self.run_receipt()
        self.assets["inputs"]["frames"][2]["offset_ms"] = 4002
        self.manifest["frames"][2]["at_seconds"] = 9.001
        self.save()
        with self.assertRaisesRegex(ValueError, "prepared_clock_mismatch"):
            self.run_receipt()

    def test_withheld_selection_has_no_excluded_frame_join(self):
        self.variant = "c02-b"
        new_root = self.pack / "inputs/c02"
        self.asset_root.rename(new_root)
        self.asset_root = new_root
        self.assets["source_pts_seconds"] = [0, 16, 18]
        for frame, offset in zip(self.assets["inputs"]["frames"], (0, 16000, 18000)):
            frame["offset_ms"] = offset
        withheld_hash = self.assets["inputs"]["frames"][2]["sha256"]
        self.manifest["duration_seconds"] = 95
        self.manifest["frames"] = self.manifest["frames"][:2]
        for frame, at in zip(self.manifest["frames"], (35, 51)):
            frame["at_seconds"] = at
        self.save()
        with patch.object(runner, "stage_prepared_media", wraps=runner.stage_prepared_media) as stage:
            receipt = self.run_receipt()
        stage.assert_called_once()
        self.assertEqual(stage.call_args.args[0].workload, "none")
        self.assertEqual(stage.call_args.args[0].result_release_policy, "all_sdk_inputs")
        self.assertEqual(len(receipt["frames"]), 2)
        self.assertNotIn(withheld_hash, json.dumps(receipt))

    def test_outside_runtime_and_inputs_and_no_overwrite(self):
        for output in (self.runtime / "lineage.json", self.pack / "inputs/lineage.json"):
            with self.assertRaisesRegex(ValueError, "outside_model_inputs"):
                self.run_receipt(output=output)
        self.run_receipt()
        before = self.output.read_bytes()
        with self.assertRaisesRegex(ValueError, "output_exists"):
            self.run_receipt()
        self.assertEqual(self.output.read_bytes(), before)

    def test_no_silent_multisession_support_or_count_mismatch(self):
        with self.assertRaisesRegex(ValueError, "unsupported_lineage_variant"):
            self.run_receipt(variant="c02-c")
        self.manifest["frames"].pop()
        self.save()
        with self.assertRaisesRegex(ValueError, "selected_count_mismatch"):
            self.run_receipt()


if __name__ == "__main__":
    unittest.main()
