"""Bounded regression coverage for focused native-evidence harness failures."""
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from evals.second_brain.native_expert_delivery import drain_seed_sources
from test_project_connector_native import cites_progress_correction


class NativeEvidenceTests(unittest.IsolatedAsyncioTestCase):
    async def test_spoken_record_setup_does_not_wait_for_nonexistent_seed(self):
        with patch("evals.second_brain.native_expert_delivery.drain", new_callable=AsyncMock) as drain:
            await drain_seed_sources("fixture", "worker", {}, {})
            drain.assert_not_awaited()
            await drain_seed_sources("fixture", "worker", {}, {"seed": object()})
            drain.assert_awaited_once_with("fixture", "worker", {}, label="delegated-memory-seed")

    def test_exact_context_or_source_evidence_matches_actual_correction(self):
        event = SimpleNamespace(id=uuid4(), revision=2)
        self.assertTrue(cites_progress_correction(
            [{"event_id": str(event.id), "revision": 2}], [], event))
        self.assertTrue(cites_progress_correction([], [
            {"context_event_id": str(event.id), "context_event_revision": 2}], event))
        for identity, revision in ((uuid4(), 2), (event.id, 1)):
            with self.subTest(identity=identity, revision=revision):
                self.assertFalse(cites_progress_correction(
                    [{"event_id": str(identity), "revision": revision}], [], event))
                self.assertFalse(cites_progress_correction([], [
                    {"context_event_id": str(identity), "context_event_revision": revision}], event))
