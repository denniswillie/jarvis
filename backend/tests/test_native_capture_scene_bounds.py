"""Offline reader scene bounds; no changes to semantic, null or claim rules."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from evals.responsive_voice import native_capture_evidence as reader
from evals.responsive_voice import native_replay_run as runner
from evals.responsive_voice.test_native_capture_evidence import fixture


class CaptureSceneBoundsTests(unittest.TestCase):
    def setUp(self):
        for target in ("socket.socket.connect", "subprocess.Popen", "asyncpg.connect", "asyncpg.create_pool"):
            guard = patch(target, side_effect=AssertionError("external execution forbidden"))
            guard.start()
            self.addCleanup(guard.stop)

    @staticmethod
    def long_silent_capture(startup_seconds=0):
        capture, _, _ = fixture()
        duration = startup_seconds + 180
        started = 100 - startup_seconds
        blocks = duration * 100
        pcm_bytes = duration * 48_000 * 2
        capture.update(utterances=[], input_blocks=0, decoded_blocks=blocks, output_starts=[])
        capture["decoded_pcm"].update(sample_count=duration * 48_000, byte_count=pcm_bytes)
        capture["decoded_sample_map"] = [
            {"sample_offset": index * 480, "sample_count": 480,
             "callback_start_uptime_seconds": started + index / 100,
             "callback_end_uptime_seconds": started + (index + 1) / 100, "local_output_epoch": 1}
            for index in range(blocks)
        ]
        opened = deepcopy(capture["output_gates"][0])
        opened["requested_uptime_seconds"] = min(opened["requested_uptime_seconds"], started)
        closed = dict(opened, event_id=capture["media_epoch_id"] + "/output-gate/1",
                      enabled=False, requested_uptime_seconds=280.1)
        capture["output_gates"] = [opened, closed]
        return capture, bytes(pcm_bytes)

    def test_reader_and_runner_bounds_agree(self):
        self.assertEqual(reader.MAX_SCENE_DURATION, runner.MAX_DURATION)
        self.assertEqual(reader.MAX_SCENE_DURATION, 180)
        self.assertEqual(reader.MAX_PCM_BYTES, runner.MAX_CAPTURE_PCM_BYTES)
        self.assertEqual(reader.MAX_PCM_BYTES, 20_160_000)
        self.assertEqual(reader.MAX_DECODED_BLOCKS, 21_000)

    def test_exact_180_second_pcm_and_map_read_without_invented_answer(self):
        capture, pcm = self.long_silent_capture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "capture.json").write_text(json.dumps(capture))
            (root / "decoded.pcm").write_bytes(pcm)
            result = reader.read_capture(root / "capture.json")
        self.assertEqual(result["capture_verdict"], "structurally_valid_evidence")
        self.assertEqual(result["occurrences"], [])
        self.assertIsNone(result["first_decoded_onset"]["sample"])
        self.assertFalse(result["hearing_measured"])
        self.assertFalse(result["answer_quality_judged"])
        self.assertFalse(result["first_decoded_onset"]["useful_speech_established"])

    def test_startup_plus_max_scene_allocation_read_without_invented_answer(self):
        capture, pcm = self.long_silent_capture(startup_seconds=30)
        self.assertEqual(len(pcm), 20_160_000)
        self.assertEqual(len(capture["decoded_sample_map"]), 21_000)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "capture.json").write_text(json.dumps(capture))
            (root / "decoded.pcm").write_bytes(pcm)
            result = reader.read_capture(root / "capture.json")
        self.assertEqual(result["capture_verdict"], "structurally_valid_evidence")
        self.assertEqual(result["occurrences"], [])
        self.assertIsNone(result["first_decoded_onset"]["sample"])
        self.assertFalse(result["hearing_measured"])
        self.assertFalse(result["answer_quality_judged"])
        self.assertFalse(result["first_decoded_onset"]["useful_speech_established"])

    def test_pcm_above_210_second_allocation_rejected(self):
        capture, _, _ = fixture()
        with self.assertRaisesRegex(reader.EvidenceError, "invalid_pcm_length"):
            reader.analyze_capture(capture, bytes(20_160_002))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "capture.json").write_text(json.dumps(capture))
            with (root / "decoded.pcm").open("wb") as stream:
                stream.truncate(20_160_001)
            with self.assertRaises(reader.EvidenceError):
                reader.read_capture(root / "capture.json")

    def test_decoded_map_above_21000_blocks_rejected(self):
        capture, _, _ = fixture()
        capture["decoded_pcm"].update(sample_count=0, byte_count=0)
        capture["decoded_sample_map"] = [{}] * 21_001
        with self.assertRaisesRegex(reader.EvidenceError, "invalid_decoded_sample_map"):
            reader.analyze_capture(capture, b"")

    def test_absolute_task_times_allow_180_but_not_later_or_nonfinite(self):
        for at in (0, 45, 120, 165, 180):
            trigger = {"kind": "at_time", "at_seconds": at}
            with self.subTest(at=at):
                self.assertEqual(reader._task_trigger_semantics(trigger, 0), trigger)
        for at in (-1, 180.001, float("nan"), float("inf"), True):
            with self.subTest(at=at), self.assertRaises(reader.EvidenceError):
                reader._task_trigger_semantics({"kind": "at_time", "at_seconds": at}, 0)

    def test_monotonic_domain_keeps_30_second_margin_around_long_scene(self):
        anchor = 100.0
        self.assertEqual(reader._time(70, anchor, "bad_time"), 70)
        self.assertEqual(reader._time(310, anchor, "bad_time"), 310)
        for value in (69.999, 310.001):
            with self.subTest(value=value), self.assertRaisesRegex(
                    reader.EvidenceError, "timestamp_outside_supported_monotonic_seconds_domain"):
                reader._time(value, anchor, "bad_time")

    def test_relative_task_timeouts_remain_120(self):
        triggers = [
            {"kind": "after_task_claim", "after_input_ordinal": 0, "timeout_seconds": 120},
            {"kind": "after_input_complete", "after_input_ordinal": 0, "timeout_seconds": 120, "delay_seconds": 1},
            {"kind": "response_output_start", "after_input_ordinal": 0, "timeout_seconds": 120,
             "delay_seconds": 1, "max_pcm_age_seconds": .05, "require_task_claim": True},
        ]
        for trigger in triggers:
            with self.subTest(kind=trigger["kind"]):
                self.assertEqual(reader._task_trigger_semantics(trigger, 1), trigger)
                with self.assertRaisesRegex(reader.EvidenceError, "invalid_task_trigger_bounds"):
                    reader._task_trigger_semantics(dict(trigger, timeout_seconds=120.001), 1)


if __name__ == "__main__":
    unittest.main()
