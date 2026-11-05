"""Native acceptance prerequisites cannot be bypassed by direct replay."""
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from evals.second_brain.hindsight_trace import exercise


class NativeIngestionTraceTests(unittest.IsolatedAsyncioTestCase):
    async def run_trace(self, statuses):
        source = SimpleNamespace(document_id="fixture-document")
        receipt = {"id": uuid4(), "status": statuses[0], "last_error": "fixture_error",
                   "failure_count": 1 if statuses[0] == "failed" else 0}
        connection = SimpleNamespace(fetchrow=AsyncMock(return_value=receipt))
        @asynccontextmanager
        async def acquire():
            yield connection
        fixture = SimpleNamespace(
            publish=AsyncMock(return_value=source), owner="fixture-owner", area=uuid4(),
            pool=SimpleNamespace(acquire=acquire), sources=SimpleNamespace(delete_source=AsyncMock()),
        )
        cleaned = []
        @asynccontextmanager
        async def owned_fixture(*args, **kwargs):
            try:
                yield fixture
            finally:
                cleaned.append(True)
        client = SimpleNamespace(
            close=AsyncMock(), retain=AsyncMock(return_value={"operation_id": str(receipt["id"])}),
        )
        worker = SimpleNamespace(step=AsyncMock(side_effect=statuses))
        with patch("evals.second_brain.hindsight_trace.source_fixture", owned_fixture), \
             patch("evals.second_brain.hindsight_trace.HindsightClient", return_value=client), \
             patch("evals.second_brain.hindsight_trace.OutboxRepository"), \
             patch("evals.second_brain.hindsight_trace.IngestionWorker", return_value=worker), \
             patch("evals.second_brain.hindsight_trace.SourceMemoryRetrieval") as retrieval, \
             patch("evals.second_brain.hindsight_trace.asyncio.sleep", new_callable=AsyncMock), \
             patch("evals.second_brain.hindsight_trace.save"):
            report = await exercise(Path("/fixture"), {"url": "fixture", "token": "fixture"},
                                    SimpleNamespace(SUPABASE_DB_URL="fixture"))
        self.assertEqual(cleaned, [True])
        client.close.assert_awaited_once()
        fixture.sources.delete_source.assert_not_awaited()
        retrieval.assert_not_called()
        return report, client

    async def test_failed_first_step_preserves_error_without_direct_replay(self):
        report, client = await self.run_trace(["failed"])
        self.assertFalse(report["checks"]["async_accepted_not_searchable"])
        self.assertEqual(report["error"]["code"], "async_accepted_not_searchable")
        self.assertEqual(report["outbox_receipt"]["last_error"], "fixture_error")
        client.retain.assert_not_awaited()
        self.assertNotIn("source_backed_recall", report["checks"])

    async def test_failed_completion_never_claims_recall_or_deletion_acceptance(self):
        report, client = await self.run_trace(["accepted", "failed"])
        self.assertTrue(report["checks"]["async_accepted_not_searchable"])
        self.assertFalse(report["checks"]["native_ingestion_complete"])
        self.assertEqual(report["error"]["code"], "native_ingestion_complete")
        client.retain.assert_awaited_once()
        self.assertNotIn("canonical_read_barrier", report["checks"])
