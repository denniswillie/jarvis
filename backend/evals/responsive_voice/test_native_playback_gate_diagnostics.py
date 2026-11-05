"""Host-only Swift regression and compatibility with the unchanged PCM reader."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from evals.responsive_voice.native_capture_evidence import analyze_capture
from evals.responsive_voice.test_native_capture_evidence import fixture, response_fixture


ROOT = Path(__file__).resolve().parents[3]


class PlaybackGateDiagnosticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="jarvis-playback-diagnostics-")
        cls.addClassCleanup(cls.temporary.cleanup)
        binary = Path(cls.temporary.name) / "gate-regressions"
        sources = [
            "ios/Jarvis/Jarvis/Services/RealtimeDiagnostics.swift",
            "ios/Jarvis/Jarvis/Services/JarvisReplaySession.swift",
            "ios/Jarvis/Jarvis/Services/JarvisReplayEvidence.swift",
            "ios/Jarvis/OfflineTests/PlaybackGateDiagnostics.swift",
        ]
        # Compile the exact production policy declaration, without introducing
        # transport-model stubs or a second copy of the policy for offline tests.
        model = (ROOT / "ios/Jarvis/Jarvis/Models/JarvisAudioProvider.swift").read_text()
        start = "enum JarvisDirectOutputPolicy {"
        end = "\nenum JarvisAudioProviderError:"
        if model.count(start) != 1 or model.count(end) != 1:
            raise AssertionError("production policy declaration boundaries changed")
        declaration = model.split(start, 1)[1].split(end, 1)[0]
        policy = Path(cls.temporary.name) / "DirectOutputPolicy.swift"
        policy.write_text("import Foundation\n\n" + start + declaration)
        command = ["/usr/bin/swiftc", "-D", "DEBUG", str(policy),
                   *[str(ROOT / p) for p in sources], "-o", str(binary)]
        compiled = subprocess.run(command, capture_output=True, text=True, timeout=90)
        if compiled.returncode:
            raise AssertionError("offline Swift compilation failed:\n" + compiled.stderr)
        executed = subprocess.run([str(binary)], capture_output=True, text=True, timeout=15)
        if executed.returncode:
            raise AssertionError("offline Swift regression failed:\n" + executed.stderr)
        cls.swift = json.loads(executed.stdout)
        print("offline_swift_checks=" + str(cls.swift["checks"]))

    def with_diagnostics(self, capture):
        result = deepcopy(capture)
        examples = self.swift["diagnostics"]
        for index, gate in enumerate(result["output_gates"]):
            gate["playback_diagnostics"] = examples[index % len(examples)]
        return result

    def test_actual_swift_core_and_privacy_regressions(self):
        self.assertGreaterEqual(self.swift["checks"], 64)
        self.assertTrue(all("playback_diagnostics" in row for row in self.swift["gate_records"]))
        self.assertNotIn("DO_NOT_RETAIN_RAW_ITEM_TRANSCRIPT_AUDIO_OR_SECRET", json.dumps(self.swift))

    def test_commentary_binding_policy_and_real_pcm_state(self):
        self.assertGreaterEqual(self.swift["policy_binding_checks"], 45)
        gates = self.swift["commentary_gate_records"]
        self.assertEqual(max(row["local_output_epoch"] for row in gates), 1)
        tool_gates = [row for row in gates
                      if row["playback_diagnostics"]["item_type_category"] == "function_call"]
        self.assertGreaterEqual(len(tool_gates), 2)
        self.assertFalse(tool_gates[0]["playback_diagnostics"]["predicate_allows_playback"])
        self.assertTrue(tool_gates[-1]["playback_diagnostics"]["predicate_allows_playback"])
        self.assertTrue(all(not row["playback_diagnostics"]["item_policy_allows_playback"] for row in tool_gates))

    def test_legacy_capture_reader_is_unchanged(self):
        capture, pcm, _ = fixture()
        self.assertEqual(analyze_capture(self.with_diagnostics(capture), pcm), analyze_capture(capture, pcm))

    def test_response_correlated_reader_is_unchanged(self):
        capture, pcm = response_fixture()
        self.assertEqual(analyze_capture(self.with_diagnostics(capture), pcm), analyze_capture(capture, pcm))

    def test_diagnostics_cannot_upgrade_missing_trigger_evidence(self):
        capture, pcm = response_fixture()
        capture["utterances"][1]["response_trigger"] = None
        before = analyze_capture(capture, pcm)
        after = analyze_capture(self.with_diagnostics(capture), pcm)
        self.assertEqual(after, before)
        self.assertNotEqual(after["occurrences"][1]["response_trigger_verdict"], "validated_control_binding")
        self.assertIsNone(after["actual_cuts"])


if __name__ == "__main__":
    unittest.main()
