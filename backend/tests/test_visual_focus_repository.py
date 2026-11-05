from __future__ import annotations

import unittest
from pathlib import Path


class VisualFocusRepositoryContractTests(unittest.TestCase):
    def test_schema_enforces_one_open_focus(self) -> None:
        schema = (Path(__file__).parents[1] / "app" / "schema.sql").read_text()
        self.assertIn("jarvis_visual_focuses_owner_open_unique_idx", schema)
        self.assertIn("where completed_at is null and cleared_at is null", schema)
