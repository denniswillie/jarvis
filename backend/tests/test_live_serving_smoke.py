"""Small checks for chronological frame replay in the existing voice smoke."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import live_serving_smoke as smoke


class VisualReplayTests(unittest.IsolatedAsyncioTestCase):
    def test_frame_sequence_is_explicit_and_validated(self):
        with patch.object(smoke, "validate_visual_frame_event") as validate:
            first = smoke.visual_event(b"fixture", settings=object())
            second = smoke.visual_event(b"fixture", settings=object(), frame_sequence=2)
        self.assertEqual(first["frame_sequence"], 1)
        self.assertEqual(second["frame_sequence"], 2)
        self.assertLessEqual(first["captured_at"], second["captured_at"])
        self.assertEqual(validate.call_count, 2)

    async def test_invalid_replay_is_rejected_before_loading_a_bundle(self):
        for image, interval in ((None, 1), (Path("fixture.jpg"), 0),
                                (Path("fixture.jpg"), 6), (Path("fixture.jpg"), float("nan"))):
            with self.subTest(image=image, interval=interval), self.assertRaises(ValueError):
                await smoke.run(Path("unused"), seconds=30, execute_paid=True,
                                image_path=image, image_interval=interval)
