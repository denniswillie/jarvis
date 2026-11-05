"""Manager imports must not depend on another test's import order."""
import subprocess
import sys
import unittest


class RealtimeManagerImportTests(unittest.TestCase):
    def test_manager_can_be_imported_first_in_a_fresh_interpreter(self):
        result = subprocess.run(
            [sys.executable, "-B", "-c", "from app.realtime.session_manager import RealtimeSessionManager; RealtimeSessionManager()"],
            capture_output=True, text=True, timeout=10,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
