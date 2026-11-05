"""Deterministic protocol tests; native receipt checks live in the probes."""
import json
from pathlib import Path
import tempfile
import unittest

from contracts import assert_tool_receipts, fixture_effect, tool_result


class ContractsTest(unittest.TestCase):
    def test_dictionary_becomes_json_text(self):
        payload = {"owner": "fixture-a", "unknown": None, "complete": False}
        wire = tool_result(payload)
        self.assertIsInstance(wire, str)
        self.assertEqual(json.loads(wire), payload)

    def test_text_is_preserved(self):
        self.assertEqual(tool_result("already serialized"), "already serialized")

    def test_non_json_numbers_fail(self):
        with self.assertRaises(ValueError):
            tool_result({"invalid": float("nan")})

    def test_rejected_native_receipt_fails(self):
        with self.assertRaises(AssertionError):
            assert_tool_receipts({"messages": [{"role": "tool", "content": '{"error_type":"tool_result_contract"}'}]})

    def test_effect_replay_is_idempotent(self):
        with tempfile.TemporaryDirectory(prefix="jarvis-runtime-reuse.unit-") as directory:
            database = Path(directory) / "effects.sqlite"
            first = fixture_effect(database, "operation-1", {"value": "synthetic"})
            second = fixture_effect(database, "operation-1", {"value": "synthetic"})
            self.assertTrue(first["created"])
            self.assertFalse(second["created"])
            self.assertEqual(first["receipt"], second["receipt"])

    def test_conflicting_effect_identity_fails(self):
        with tempfile.TemporaryDirectory(prefix="jarvis-runtime-reuse.unit-") as directory:
            database = Path(directory) / "effects.sqlite"
            fixture_effect(database, "operation-1", {"value": "first"})
            with self.assertRaises(ValueError):
                fixture_effect(database, "operation-1", {"value": "second"})


if __name__ == "__main__":
    unittest.main()
