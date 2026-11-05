"""Real isolated PostgreSQL + controlled memory HTTP, not native/model acceptance."""
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch
from uuid import uuid4

from app.runtime.feature_config import KnowledgeConfiguration
from evals.responsive_voice.memory_lifecycle_qualification import run_qualification
from evals.responsive_voice.memory_live_state_qualification import BOUNDS, MemoryLiveStateCase, STAGES
from test_responsive_memory_lifecycle_qualification import ControlledHindsightHTTP


@unittest.skipUnless(os.getenv("JARVIS_MEMORY_LIVE_STATE_SQL_TEST") == "1",
                     "Explicit disposable PostgreSQL qualification required")
class MemoryLiveStateSQLTests(unittest.IsolatedAsyncioTestCase):
    async def test_fresh_corrected_delivery_deletion_guards_and_no_resurrection(self):
        service = ControlledHindsightHTTP()
        directory = (Path(os.environ["JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"])
                     / ("memory-live-state-controlled-" + uuid4().hex))
        configuration = KnowledgeConfiguration(base_url="http://127.0.0.1:19999",
            api_key="controlled-offline-key", area_id=uuid4())
        with patch("app.knowledge.runtime.HindsightClient", side_effect=service.client), \
             patch("evals.responsive_voice.memory_lifecycle_qualification.HindsightClient", side_effect=service.client):
            report = await run_qualification(os.environ["JARVIS_SECOND_BRAIN_TEST_DB_URL"],
                configuration, directory, mode="controlled_hindsight_real_postgres",
                bounds=BOUNDS, case_factory=MemoryLiveStateCase)
        print("memory_live_state_controlled_report=" + str(directory / "qualification.json"))
        self.assertTrue(report["accepted"], json.dumps(report, indent=2))
        self.assertFalse(report["native_semantic_quality_qualified"])
        self.assertEqual(set(report["stages"]), set(STAGES))
        self.assertTrue(all(stage["status"] == "passed" for stage in report["stages"].values()))
        self.assertEqual([entry["label"] for entry in report["live_state_checks"]],
                         ["original", "corrected", "after_deletion"])
        self.assertEqual(report["live_state_checks"][1]["selected_sources"][0]["revision"], 2)
        self.assertEqual(len(report["live_state_checks"][2]["selected_sources"]), 1)
        self.assertEqual({entry["label"] for entry in report["invalidation"]}, {"correction", "deletion"})
        self.assertTrue(all(entry["canonical_binding_rejected"] and entry["no_new_response_create"]
                            for entry in report["invalidation"]))
        self.assertEqual(len(set(report["fresh_live_state_epochs"])), 3)
        self.assertTrue(all(not item["before_close"]["subscription_needs_resync"]
                            for item in report["voice_fixture_runs"]))
        self.assertLessEqual(report["call_usage"]["actual_recall_queries"], 8)
        self.assertEqual(report["cleanup"]["database"]["remaining_schemas"], 0)
        self.assertEqual(report["cleanup"]["database"]["remaining_public_owner_rows"], 0)
        self.assertTrue(report["cleanup"]["memory"]["native_bank_absent"])
        self.assertFalse(service.banks)

