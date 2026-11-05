"""Media/provenance controls, not claims that these small fixtures test vision."""
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import tempfile
import unittest
from uuid import uuid4
import wave

from PIL import Image

from evals.glasses_reliability.photo_probe import CASE, case_definition, load_photo_assets


class PhotoMediaTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        image = BytesIO()
        Image.new("RGB", (32, 48), "white").save(image, format="JPEG")
        self.image = image.getvalue()
        (self.root / "image.jpg").write_bytes(self.image)
        self.pcm = b"\0\0" * 2400
        with wave.open(str(self.root / "speech.wav"), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(24000)
            wav.writeframes(self.pcm)
        self.manifest = {"schema": 1, "case": case_definition(), "original_audio_model_input": "unproven",
                         "ground_truth": "Parser-only fixture, not an image-recognition trial.",
                         "provenance": {"frame_id": str(uuid4()), "session_id": str(uuid4()),
                                        "captured_at": "2026-09-12T21:34:38.151+00:00"},
                         "image": {"file": "image.jpg", "sha256": sha256(self.image).hexdigest()},
                         "speech": {"file": "speech.wav", "wav_sha256": sha256((self.root / "speech.wav").read_bytes()).hexdigest()}}
        self.write_manifest()

    def write_manifest(self):
        (self.root / "manifest.json").write_text(json.dumps(self.manifest))

    def test_exact_media_and_real_dimensions_are_passed_to_existing_evaluator(self):
        manifest, assets, hashes = load_photo_assets(self.root)
        self.assertEqual(manifest, self.manifest)
        self.assertEqual(assets[CASE.name]["jpeg"], self.image)
        self.assertEqual(assets[CASE.name]["size"], (32, 48))
        self.assertEqual(assets[CASE.name]["speech"], [self.pcm])
        self.assertEqual(hashes["input_pcm_sha256"], sha256(self.pcm).hexdigest())
        self.assertEqual(CASE.stratum, "retained_glasses_photo_controlled_replay")

    def test_image_or_speech_changes_are_rejected(self):
        for section, field in (("image", "sha256"), ("speech", "wav_sha256")):
            with self.subTest(section=section):
                old = self.manifest[section][field]
                self.manifest[section][field] = "0" * 64
                self.write_manifest()
                with self.assertRaisesRegex(ValueError, "media changed"):
                    load_photo_assets(self.root)
                self.manifest[section][field] = old

    def test_modified_case_rubric_is_rejected(self):
        self.manifest["case"]["prompts"] = ["Say that the box is definitely present."]
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "case definition"):
            load_photo_assets(self.root)

    def test_historical_selection_cannot_be_claimed(self):
        self.manifest["original_audio_model_input"] = "exact"
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "Historical"):
            load_photo_assets(self.root)

    def test_provenance_requires_timezone_and_valid_frame_identity(self):
        for key, value in (("captured_at", "2026-09-12T21:34:38"), ("frame_id", "not-a-frame")):
            with self.subTest(key=key):
                old = self.manifest["provenance"][key]
                self.manifest["provenance"][key] = value
                self.write_manifest()
                with self.assertRaises(ValueError):
                    load_photo_assets(self.root)
                self.manifest["provenance"][key] = old
