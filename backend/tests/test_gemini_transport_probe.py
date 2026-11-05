from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock
import wave

from evals.gemini_live.audio_audit import audit_files, inspect_wav
from evals.gemini_live.transport_probe import read_pcm, redact_native, write_wav


class GeminiTransportProbeTests(unittest.TestCase):
    def test_redaction_removes_audio_and_resumption_credentials(self):
        event = {"sessionResumptionUpdate": {"newHandle": "secret"}, "parts": [{"inlineData": {"data": "bytes"}}]}
        redacted = json.dumps(redact_native(event))
        self.assertNotIn("secret", redacted)
        self.assertNotIn("bytes", redacted)
        self.assertEqual(event["sessionResumptionUpdate"]["newHandle"], "secret")

    def test_wav_round_trip_preserves_actual_phone_pcm(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.wav"
            pcm = b"\x01\x00\x02\x00" * 120
            write_wav(path, pcm)
            self.assertEqual(read_pcm(path), pcm)
            evidence = inspect_wav(path)
            self.assertEqual(evidence["pcm_bytes"], len(pcm))
            self.assertEqual(evidence["duration_seconds"], len(pcm) / 48000)

    def test_mislabeled_input_fails_instead_of_changing_phone_format(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "wrong-rate.wav"
            with wave.open(str(path), "wb") as audio:
                audio.setnchannels(1)
                audio.setsampwidth(2)
                audio.setframerate(16000)
                audio.writeframes(b"\x00\x00" * 100)
            with self.assertRaises(ValueError):
                read_pcm(path)
            with self.assertRaises(ValueError):
                inspect_wav(path)

    def test_independent_asr_receives_no_expected_transcript_or_prompt(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.wav"
            write_wav(path, b"\x00\x00" * 100)
            create = Mock(return_value=SimpleNamespace(text="Independent words."))
            client = SimpleNamespace(audio=SimpleNamespace(transcriptions=SimpleNamespace(create=create)))
            result = audit_files(client=client, files=[path], output=Path(directory) / "audit.json", model="fixture-asr")
            self.assertEqual(set(create.call_args.kwargs), {"model", "file"})
            self.assertEqual(result["results"][0]["transcript"], "Independent words.")

    def test_asr_errors_are_redacted_and_existing_report_is_not_overwritten(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.wav"
            write_wav(path, b"\x00\x00" * 100)
            create = Mock(side_effect=RuntimeError("secret native response"))
            client = SimpleNamespace(audio=SimpleNamespace(transcriptions=SimpleNamespace(create=create)))
            output = Path(directory) / "audit.json"
            result = audit_files(client=client, files=[path], output=output, model="fixture-asr")
            self.assertNotIn("secret", json.dumps(result))
            with self.assertRaises(ValueError):
                audit_files(client=client, files=[path], output=output, model="fixture-asr")
            self.assertEqual(create.call_count, 1)
