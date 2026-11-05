"""Offline input checks for the serving harness's optional synthetic camera frame."""
import base64
import unittest

from app.settings import JarvisSettings
from app.visual_memory.validation import VisualFrameValidationError
from live_serving_smoke import visual_event


class VisualInputTests(unittest.TestCase):
    def setUp(self):
        self.settings = JarvisSettings()
        # Minimal JPEG header for the ingest metadata parser, not a decoder fixture.
        self.header = bytes.fromhex("ffd8ffc00011080050006403011100021100031100ffd9")

    def test_uses_actual_visual_wire_schema_and_original_bytes(self):
        event = visual_event(self.header, settings=self.settings, source="iphone_camera")
        self.assertEqual(event["type"], "visual.frame.capture")
        self.assertEqual(event["source"], "iphone_camera")
        self.assertEqual(event["frame_sequence"], 1)
        self.assertEqual(base64.b64decode(event["image_base64"]), self.header)
        self.assertTrue(event["captured_at"].endswith("+00:00"))

    def test_rejects_non_jpeg_before_session_creation(self):
        with self.assertRaises(VisualFrameValidationError):
            visual_event(b"not a jpeg", settings=self.settings)

    def test_enforces_serving_frame_size_limit(self):
        with self.assertRaises(VisualFrameValidationError):
            visual_event(self.header + b"x" * self.settings.VISUAL_FRAME_MAX_BYTES,
                         settings=self.settings)
