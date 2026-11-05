"""Regression coverage for misleading recording duration metadata."""

import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

from app.files.local_file import snapshot_file
from app.recordings import timing


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg required")
class RecordingTimingFrameBoundaryTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="recording-timing-boundary-")
        self.addCleanup(self.temporary.cleanup)
        source = Path(self.temporary.name).resolve() / "offset-recording.mp4"
        subprocess.run([
            "ffmpeg", "-v", "error", "-nostdin", "-n", "-copyts",
            "-f", "lavfi", "-i", "testsrc2=size=160x120:rate=24:duration=2",
            "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=2",
            "-vf", "setpts=PTS+2/(24*TB)", "-c:v", "libx264", "-bf", "0",
            "-pix_fmt", "yuv420p", "-fps_mode:v", "passthrough",
            "-enc_time_base:v", "1/600", "-video_track_timescale", "600",
            "-c:a", "aac", "-movflags", "+faststart", str(source),
        ], check=True, timeout=20)
        self.original = await snapshot_file(source, max_bytes=1_000_000)
        self.addCleanup(self.original.close)
        self.options = timing.RecordingTimingOptions(
            video_delay_ms=1000, max_bytes=1_000_000,
            max_duration_ms=10_000, timeout_seconds=30)

    async def test_nonzero_start_and_misleading_duration_preserve_actual_frames(self):
        real_probe = timing._probe

        async def misleading_header(path, options):
            data = await real_probe(path, options)
            if path == self.original.path:
                video, _ = timing._tracks(data)
                self.assertGreater(float(video["start_time"]), 0)
                video["duration"] = str(float(video["duration"]) + 0.5)
            return data

        with patch.object(timing, "_probe", side_effect=misleading_header):
            result = await timing.correct_recording_timing(self.original, self.options)
        self.assertEqual((result.status, result.reason), ("corrected", "validated"))
        self.addCleanup(result.corrected.close)

        def timestamps(path):
            data = json.loads(subprocess.check_output([
                "ffprobe", "-v", "error", "-select_streams", "v:0",
                "-show_entries", "frame=best_effort_timestamp_time", "-of", "json", str(path),
            ], timeout=20))
            return [float(frame["best_effort_timestamp_time"]) for frame in data["frames"]]

        before = timestamps(self.original.path)
        after = timestamps(result.corrected.path)
        self.assertEqual(len(after), len(before) + 1)
        self.assertEqual(after[0], 0)
        for original, corrected in zip(before, after[1:]):
            self.assertAlmostEqual(corrected - original, 1.0, places=5)
        self.original._check()

    async def test_incorrect_final_frame_is_still_rejected(self):
        with patch.object(timing, "_last_video_timestamp", new=AsyncMock(side_effect=[2.0, 3.25])):
            result = await timing.correct_recording_timing(self.original, self.options)
        self.assertEqual((result.status, result.reason), ("failed", "video_timeline_changed"))
        self.assertIsNone(result.corrected)
        self.original._check()

    async def test_missing_tail_frames_are_not_treated_as_success(self):
        with patch.object(timing, "_run", new=AsyncMock(return_value=b'{"frames": []}')):
            with self.assertRaisesRegex(timing._CorrectionError, "video_frame_timestamp_unavailable"):
                await timing._last_video_timestamp(self.original.path, 2.0, self.options)
