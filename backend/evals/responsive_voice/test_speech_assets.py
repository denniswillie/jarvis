import array
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import wave

from evals.responsive_voice.speech_assets import inspect_pcm, prepare_speech_bundle
from evals.responsive_voice.video_assets import sha256_file


class SpeechAssetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def write_wave(self, name="test.wav", samples=(0, 0, 129, -130, 0, 0), channels=1):
        path = self.root / name
        with wave.open(str(path), "wb") as output:
            output.setnchannels(channels)
            output.setsampwidth(2)
            output.setframerate(48000)
            output.writeframes(array.array("h", samples).tobytes())
        return path

    def test_signal_end_is_separate_from_file_end(self):
        result = inspect_pcm(self.write_wave())
        self.assertEqual(result["sample_count"], 6)
        self.assertEqual(result["first_active_sample"], 2)
        self.assertEqual(result["last_active_sample_exclusive"], 4)
        self.assertAlmostEqual(result["trailing_below_threshold_ms"], 2 * 1000 / 48000)
        self.assertIn("not_semantic", result["boundary_kind"])

    def test_silent_and_wrong_format_inputs_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "no_signal"):
            inspect_pcm(self.write_wave(samples=(0, 0, 0)))
        with self.assertRaisesRegex(ValueError, "pcm16_mono"):
            inspect_pcm(self.write_wave("stereo.wav", channels=2))

    def test_invalid_scripts_fail_before_synthesis(self):
        with patch("evals.responsive_voice.speech_assets.make_speech") as speech:
            for rows in ((), (("../escape", "Question"),), (("same", "One"), ("same", "Two")),
                         (("good", ""),), (("good", "x" * 801),)):
                with self.subTest(rows=rows), self.assertRaises(ValueError):
                    prepare_speech_bundle(self.root / "out", rows)
            speech.assert_not_called()

    def test_existing_output_is_not_overwritten(self):
        with patch("evals.responsive_voice.speech_assets.make_speech") as speech:
            with self.assertRaises(FileExistsError):
                prepare_speech_bundle(self.root, (("question", "What is visible?"),))
            speech.assert_not_called()

    @unittest.skipUnless(Path("/usr/bin/say").exists() and Path("/usr/bin/afconvert").exists(),
                         "macOS local speech tools unavailable")
    def test_real_local_synthesis_produces_hashed_pcm_fixture_without_provider(self):
        manifest = prepare_speech_bundle(self.root / "out", (("question", "What is in front of me?"),))
        clip, = manifest["clips"]
        self.assertEqual(clip["sample_rate_hz"], 48000)
        self.assertGreater(clip["last_active_sample_exclusive"], clip["first_active_sample"])
        self.assertEqual(clip["sha256"], sha256_file(self.root / "out" / clip["relative_path"]))
        self.assertEqual(manifest["provider_calls"], 0)
        self.assertFalse(manifest["frontend_replay_executed"])
        self.assertFalse(manifest["microphone_used"])


if __name__ == "__main__":
    unittest.main()
