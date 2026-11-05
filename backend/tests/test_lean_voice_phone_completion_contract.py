"""Compile the actual response.done branch in isolation, not the iOS app."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


@unittest.skipUnless(os.environ.get("JARVIS_IOS_CONTRACT_TESTS") == "1",
                     "Opt-in host Swift compiler contract test")
class LeanPhoneCompletionContractTests(unittest.TestCase):
    def test_delayed_backend_completion_cannot_retire_native_response(self):
        root = Path(__file__).resolve().parents[2]
        source = (root / "ios/Jarvis/Jarvis/ViewModels/JarvisAudioSessionViewModel.swift").read_text()
        branch = source.split('case "response.done":', 1)[1].split('case "reminder.started":', 1)[0]
        program = """
        enum Turn { case idle, speaking }
        struct Playback { var isPlaying = false }
        final class Harness {
            var usesLeanVoice = false
            var hasAssistantResponseInProgress = true
            var playbackService = Playback()
            var directAudioClient = Playback()
            var turnState = Turn.speaking
            func backendDone() {
        """ + branch + """
            }
        }
        let native = Harness()
        native.usesLeanVoice = true
        native.backendDone()
        precondition(native.hasAssistantResponseInProgress)
        precondition(native.turnState == .speaking)
        let legacy = Harness()
        legacy.backendDone()
        precondition(!legacy.hasAssistantResponseInProgress)
        precondition(legacy.turnState == .idle)
        let draining = Harness()
        draining.directAudioClient.isPlaying = true
        draining.backendDone()
        precondition(!draining.hasAssistantResponseInProgress)
        precondition(draining.turnState == .speaking)
        """
        with tempfile.TemporaryDirectory(prefix="jarvis-lean-completion-") as folder:
            swift, executable = Path(folder) / "main.swift", Path(folder) / "contract"
            swift.write_text(program)
            subprocess.run(["xcrun", "swiftc", str(swift), "-o", str(executable)],
                           check=True, capture_output=True, text=True, timeout=60)
            subprocess.run([str(executable)], check=True, capture_output=True,
                           text=True, timeout=10)
