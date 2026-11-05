import asyncio
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from app.files.local_file import snapshot_file
from app.recordings.timing import RecordingTimingOptions, correct_recording_timing


class RecordingTimingOptionsTests(unittest.TestCase):
    def test_invalid_values_are_rejected(self):
        for values in ({"video_delay_ms": -1}, {"video_delay_ms": True},
                       {"video_delay_ms": 2001}, {"max_bytes": 0},
                       {"timeout_seconds": float("nan")}, {"timeout_seconds": 0},
                       {"ffmpeg": ""}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                RecordingTimingOptions(**values)


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "Requires local FFmpeg")
class RecordingTimingTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory(prefix="jarvis-timing-tests-")
        cls.fixture = Path(cls.directory.name).resolve() / "original.mp4"
        subprocess.run([
            "ffmpeg", "-v", "error", "-nostdin", "-n", "-f", "lavfi", "-i",
            "testsrc2=size=160x120:rate=24:duration=2", "-f", "lavfi", "-i",
            "sine=frequency=440:sample_rate=48000:duration=2",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-bf", "0",
            "-video_track_timescale", "600", "-c:a", "aac", str(cls.fixture)], check=True)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    async def asyncSetUp(self):
        self.original = await snapshot_file(self.fixture, max_bytes=1_000_000)
        self.options = RecordingTimingOptions(video_delay_ms=1000, max_bytes=1_000_000,
                                              max_duration_ms=10_000, timeout_seconds=30)
        self.outputs = []

    async def asyncTearDown(self):
        for output in self.outputs:
            output.close()
        self.original.close()

    async def check(self, options):
        result = await correct_recording_timing(self.original, options)
        if result.corrected is not None:
            self.outputs.append(result.corrected)
        self.original._check()
        self.assertEqual(hashlib.sha256(self.original.path.read_bytes()).hexdigest(),
                         self.original.content_sha256)
        return result

    async def test_disabled_does_not_start_media_process(self):
        with patch("app.recordings.timing._run", side_effect=AssertionError("Must not execute")):
            result = await self.check(replace(self.options, video_delay_ms=0))
        self.assertEqual(result.status, "disabled")
        self.assertIsNone(result.corrected)

    async def test_corrected_copy_keeps_original_and_identical_audio(self):
        result = await self.check(self.options)
        self.assertEqual(result.status, "corrected", result.reason)
        self.assertNotEqual(result.corrected.path, self.original.path)
        self.assertFalse(result.metadata["audio_reencoded"])
        self.assertEqual(result.metadata["original_sha256"], self.original.content_sha256)
        pcm = lambda path: subprocess.check_output([
            "ffmpeg", "-v", "error", "-i", str(path), "-map", "0:a:0", "-f", "f32le", "-"])
        self.assertEqual(pcm(self.original.path), pcm(result.corrected.path))
        def video_pts(path):
            raw = subprocess.check_output([
                "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_packets",
                "-show_entries", "packet=pts_time", "-of", "json", str(path)])
            return [float(p["pts_time"]) for p in json.loads(raw)["packets"]]
        before, after = video_pts(self.original.path), video_pts(result.corrected.path)
        self.assertEqual(after[0], 0)
        self.assertEqual(len(after), len(before) + 1)
        for a, b in zip(before, after[1:]):
            self.assertAlmostEqual(b - a, 1.0, places=5)
        self.assertAlmostEqual(result.metadata["duration_seconds"], 3.0, delta=0.05)

    async def test_closing_corrected_copy_does_not_close_original(self):
        result = await self.check(self.options)
        self.assertEqual(result.status, "corrected", result.reason)
        result.corrected.close()
        self.original._check()
        self.assertTrue(self.original.path.exists())

    async def test_missing_executable_falls_back_to_original(self):
        result = await self.check(replace(self.options, ffmpeg="/nonexistent/jarvis-ffmpeg"))
        self.assertEqual((result.status, result.reason), ("failed", "executable_unavailable"))
        self.assertIsNone(result.corrected)

    async def test_input_size_limit(self):
        result = await self.check(replace(self.options, max_bytes=1))
        self.assertEqual((result.status, result.reason), ("skipped", "input_size_limit"))

    async def test_duration_limit_does_not_trim_original_to_fit(self):
        result = await self.check(replace(self.options, max_duration_ms=2500))
        self.assertEqual((result.status, result.reason), ("skipped", "corrected_duration_limit"))
        self.assertIsNone(result.corrected)

    async def test_timeout_and_cancellation_keep_original(self):
        async def blocked(*args):
            await asyncio.Event().wait()
        with patch("app.recordings.timing._run", side_effect=blocked):
            result = await self.check(replace(self.options, timeout_seconds=0.01))
            self.assertEqual((result.status, result.reason), ("failed", "timeout"))
            task = asyncio.create_task(correct_recording_timing(self.original, self.options))
            await asyncio.sleep(0)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.original._check()

    async def test_changed_audio_is_rejected(self):
        with patch("app.recordings.timing._audio_hash", side_effect=["a" * 64, "b" * 64]):
            result = await self.check(self.options)
        self.assertEqual((result.status, result.reason), ("failed", "audio_changed"))
        self.assertIsNone(result.corrected)

    async def test_failed_render_never_returns_a_partial_file(self):
        with patch("app.recordings.timing._run", side_effect=RuntimeError("synthetic failure")):
            result = await self.check(self.options)
        self.assertEqual(result.status, "failed")
        self.assertIsNone(result.corrected)


if __name__ == "__main__":
    unittest.main()
