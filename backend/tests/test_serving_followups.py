"""Bounded repeated speech inputs reuse the normal serving harness."""
from pathlib import Path
import unittest
from unittest.mock import patch

from live_serving_smoke import followup_inputs


class FollowupInputTests(unittest.TestCase):
    @patch("live_serving_smoke.validate_bundle", return_value=(None, None, [{"relative_path": "speech.wav"}]))
    def test_existing_single_input_and_two_ordered_inputs(self, validate):
        first, second = Path("first"), Path("second")
        self.assertEqual(followup_inputs(40, None, 8, None, 16), [])
        single = followup_inputs(40, first, 8, None, 16)
        both = followup_inputs(40, first, 8, second, 16)
        self.assertEqual(single, both[:1])
        self.assertEqual([item["name"] for item in both], ["followup", "second_followup"])
        self.assertEqual([item["after"] for item in both], [8, 16])

    @patch("live_serving_smoke.validate_bundle", return_value=(None, None, [{}]))
    def test_invalid_order_or_missing_observation_window(self, validate):
        for first, second_time in ((None, 16), (Path("first"), 8), (Path("first"), 35)):
            with self.assertRaises(ValueError):
                followup_inputs(40, first, 8, Path("second"), second_time)
