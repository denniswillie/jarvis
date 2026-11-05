"""Offline classification: deleted candidate != unavailable canonical store."""
import asyncio
import json
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from app.live_state.canonical import memory_dependency
from app.live_state.historical_enrichment import HistoricalEnrichment
from app.live_state.models import SessionScope
from app.live_state.refresh import CanonicalRefreshWorker
from app.runtime.events import DomainEventBus, DomainEventName
from evals.responsive_voice.memory_live_state_diagnostics import CanonicalRefreshDiagnostics
import test_live_state_historical_enrichment as fixtures


class CanonicalRefreshDiagnosticTests(unittest.IsolatedAsyncioTestCase):
    async def fixture(self):
        fixture = fixtures.HistoricalEnrichmentCompositionTests(methodName="runTest")
        await fixture.asyncSetUp()
        self.addAsyncCleanup(fixture.asyncTearDown)
        return fixture

    async def start(self, fixture, diagnostics):
        def create(**kwargs):
            enrichment = HistoricalEnrichment(**kwargs)
            diagnostics.observe(enrichment)
            return enrichment
        with patch("app.live_state.runtime.HistoricalEnrichment", side_effect=create):
            return await fixture.start()

    async def test_deleted_candidate_is_omitted_without_losing_independent_original(self):
        fixture = await self.fixture()
        deleted = fixture.new_source("Deleted synthetic input")
        independent = fixture.new_source("Independent synthetic input")
        candidates = [source.original_evidence() | {"content_included": True}
                      for source in (deleted, independent)]
        fixture.f.db.remove(deleted.payload_id)
        fixture.f.originals.pop(deleted.source_id)
        fixture.retrieval.recall.side_effect = None
        fixture.retrieval.recall.return_value = {"status": "available", "sources": candidates}
        diagnostics = CanonicalRefreshDiagnostics()
        live = await self.start(fixture, diagnostics)
        await fixture.until(lambda: live.enrichment.status == "available" and bool(fixture.memories()))
        selected, = fixture.memories()
        self.assertIn(memory_dependency(independent), selected.sources)
        self.assertNotIn(memory_dependency(deleted), selected.sources)
        self.assertTrue(await live.binder.available(selected.guard_binding))
        self.assertGreater(live.enrichment.omitted_sources, 0)
        self.assertEqual(live.enrichment.worker.failure_count, 0)
        self.assertFalse(diagnostics.records)

    async def test_store_failure_is_not_misreported_as_deleted_candidate(self):
        fixture = await self.fixture()
        fixture.f.sources.resolve_document.side_effect = RuntimeError("private-error-text-not-for-report")
        diagnostics = CanonicalRefreshDiagnostics()
        live = await self.start(fixture, diagnostics)
        await fixture.until(lambda: live.enrichment.status == "unavailable")
        self.assertEqual(live.enrichment.last_error, "RuntimeError")
        self.assertFalse(fixture.memories())
        self.assertTrue(any(record["phase"] == "lookup_source_resolution"
                            and record["exception_type"] == "RuntimeError" for record in diagnostics.records))
        self.assertNotIn("private-error-text-not-for-report", json.dumps(diagnostics.document()))

    async def test_canonical_deadline_records_underlying_cancellation_and_timeout(self):
        scope = SessionScope(owner_id="fixture", session_id=uuid4())
        worker = CanonicalRefreshWorker(scope=scope, bus=DomainEventBus(),
            event_names=(DomainEventName.MEMORY_ITEM_CHANGED,), read=asyncio.Event().wait,
            apply=AsyncMock(), invalidate=AsyncMock(), interval_seconds=10, timeout_seconds=.01)
        enrichment = SimpleNamespace(worker=worker, scope=scope, _closed=False,
                                     _selected=(), _lookup=AsyncMock())
        diagnostics = CanonicalRefreshDiagnostics()
        diagnostics.observe(enrichment)
        worker.start()
        self.addAsyncCleanup(worker.close)
        async with asyncio.timeout(1):
            while worker.failure_count == 0:
                await asyncio.sleep(.002)
        self.assertEqual(worker.last_error, "TimeoutError")
        self.assertTrue(any(record.get("exception_type") == "CancelledError"
                            and not record["shutdown_cancellation"] for record in diagnostics.records))
        self.assertTrue(any(record.get("canonical_exception_type") == "TimeoutError"
                            for record in diagnostics.records))

    async def test_diagnostic_inventory_is_bounded(self):
        diagnostics = CanonicalRefreshDiagnostics(limit=2)
        for index in range(5):
            diagnostics.append({"sequence": index})
        self.assertEqual(len(diagnostics.records), 2)
        self.assertEqual(diagnostics.omitted, 3)
