"""Actual ffmpeg extraction of generated test footage; no external dataset."""
import asyncio
from datetime import datetime, timezone
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from evals.responsive_voice.video_assets import extract_clip, sha256_file
from evals.responsive_voice.video_replay import PermissionRecord, ReplayCase, ReplayInputs, run_replay
from evals.responsive_voice.test_video_replay import Port


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "local video decoder unavailable")
class VideoAssetTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / "synthetic.mp4"
        subprocess.run([shutil.which("ffmpeg"), "-nostdin", "-v", "error", "-f", "lavfi",
            "-i", "testsrc=size=32x24:rate=5", "-t", "3", "-c:v", "mpeg4", str(self.source)],
            check=True, timeout=15, stdin=subprocess.DEVNULL, capture_output=True)
        self.permission = PermissionRecord(evidence="Locally generated synthetic test pattern",
            local_evaluation=True, retain_derived_results=True)

    def extract(self, **overrides):
        arguments = dict(video=self.source, output=self.root / "extracted", permission=self.permission,
                         expected_sha256=sha256_file(self.source))
        arguments.update(overrides)
        return extract_clip(**arguments)

    async def test_real_decoder_preserves_timestamps_then_replay_consumes_frames(self):
        manifest = self.extract()
        self.assertEqual(manifest["source_frame_indices"], [0, 5, 10])
        self.assertEqual(manifest["source_pts_seconds"], [0, 1, 2])
        self.assertEqual([f["offset_ms"] for f in manifest["inputs"]["frames"]], [0, 1000, 2000])
        self.assertFalse(manifest["source_audio_included"])
        self.assertFalse(manifest["quality_qualified"])
        port = Port()
        case = ReplayCase(case_id="synthetic", permission=self.permission,
                          inputs=ReplayInputs.model_validate(manifest["inputs"]))
        report = await run_replay(case, asset_root=self.root / "extracted", port=port,
            origin=datetime.now(timezone.utc), realtime=False)
        self.assertEqual(report["execution_status"], "completed")
        self.assertEqual(len(port.frames), 3)
        self.assertTrue(all(f.image_bytes.startswith(b"\xff\xd8") for f in port.frames))

    def test_permission_rejected_before_decoder(self):
        denied = self.permission.model_copy(update={"local_evaluation": False})
        with patch("evals.responsive_voice.video_assets._execute") as execute:
            with self.assertRaisesRegex(ValueError, "permission_unresolved"):
                self.extract(permission=denied)
            execute.assert_not_called()

    def test_source_checksum_and_existing_output_fail_without_decoder(self):
        with patch("evals.responsive_voice.video_assets._execute") as execute:
            with self.assertRaisesRegex(ValueError, "checksum_mismatch"):
                self.extract(expected_sha256="0" * 64)
            existing = self.root / "existing"
            existing.mkdir()
            with self.assertRaisesRegex(ValueError, "already_exists"):
                self.extract(output=existing)
            execute.assert_not_called()

    def test_sampling_interval_is_bounded(self):
        for interval in (0, 1, 5001, True):
            with self.subTest(interval=interval), self.assertRaisesRegex(ValueError, "sampling_interval"):
                self.extract(sample_interval_ms=interval)


if __name__ == "__main__":
    unittest.main()
