from __future__ import annotations

import unittest

from pydantic import ValidationError

from app.settings import JarvisSettings


class ConsolidationSettingsTests(unittest.TestCase):
    def test_keyframe_input_is_capped_at_six(self) -> None:
        with self.assertRaises(ValidationError):
            JarvisSettings(
                _env_file=None,
                MEMORY_CONSOLIDATION_MAX_KEYFRAMES=7,
            )


if __name__ == "__main__":
    unittest.main()
