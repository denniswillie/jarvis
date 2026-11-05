"""Offline seam/oracle tests, never native memory or frontend acceptance."""
import asyncio
from datetime import UTC, datetime
import json
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock
from uuid import uuid4

from app.knowledge.models import SourceVersion
from app.live_state.canonical import context_dependency, memory_dependency
from app.live_state.models import GuardedPayloadBinding, SelectedMemoryEntry, SessionScope
from evals.responsive_voice.memory_lifecycle_qualification import QualificationFailure
from evals.responsive_voice.memory_live_state_qualification import (
    BOUNDS, CallBudget, ControlledProviderWire, ObservedRetrieval, verify_selected,
)


class LiveMemoryBudgetTests(unittest.IsolatedAsyncioTestCase):
    async def test_actual_result_is_held_not_replaced(self):
        result = {"status": "available", "sources": [{"source_id": "opaque", "revision": 3}]}
        original = SimpleNamespace(recall=AsyncMock(return_value=result))
        budget, trace = CallBudget(), []
        observed = ObservedRetrieval(original, budget, trace, hold=True)
        task = asyncio.create_task(observed.recall("owner", "area", "query"))
        await asyncio.wait_for(observed.entered.wait(), 1)
        self.assertFalse(task.done())
        self.assertEqual(budget.recalls, 1)
        original.recall.assert_awaited_once_with("owner", "area", "query")
        observed.release.set()
        self.assertIs(await task, result)
        self.assertEqual(trace[0]["source_versions"][0]["revision"], 3)

    async def test_query_cap_precedes_network_call(self):
        budget = CallBudget()
        budget.recalls = 8
        original = SimpleNamespace(recall=AsyncMock())
        with self.assertRaises(QualificationFailure):
            await ObservedRetrieval(original, budget, []).recall("owner", "area", "query")
        original.recall.assert_not_awaited()

    async def test_cleanup_has_separate_bounded_http_reserve(self):
        budget = CallBudget()
        client = SimpleNamespace(_call=AsyncMock(return_value={"ok": True}))
        original = client._call
        budget.install(client)
        budget.main_http = 400
        with self.assertRaises(QualificationFailure):
            await client._call("GET", "/fixture")
        original.assert_not_awaited()
        budget.cleaning = True
        self.assertEqual(await client._call("DELETE", "/fixture"), {"ok": True})
        budget.cleanup_http = 160
        with self.assertRaises(QualificationFailure):
            await client._call("GET", "/fixture")
        self.assertEqual(original.await_count, 1)
        self.assertEqual(BOUNDS.lifecycle_seconds, 240)

    async def test_controlled_wire_does_not_invent_completed_audio(self):
        wire = ControlledProviderWire()
        await wire.send(json.dumps({"type": "response.create", "response": {"metadata": {"request": "test"}}}))
        event = await wire.incoming.get()
        self.assertEqual(event["type"], "response.created")
        self.assertTrue(wire.incoming.empty())
        await wire.send(json.dumps({"type": "response.cancel", "response_id": event["response"]["id"]}))
        cancelled = await wire.incoming.get()
        self.assertEqual(cancelled["response"]["status"], "cancelled")
        self.assertFalse(cancelled["response"]["output"])


class LiveMemorySelectionOracleTests(unittest.TestCase):
    def setUp(self):
        self.scope = SessionScope(owner_id="fixture", session_id=uuid4(), area_id=uuid4())
        timestamp = datetime(2026, 1, 5, tzinfo=UTC)
        self.source = SourceVersion(owner_id="fixture", area_id=self.scope.area_id, source_id=uuid4(),
            revision=2, generation=2, payload_id=uuid4(), document_id="fixture", bank_name="fixture",
            content_digest="a" * 64, source_type="user_transcript", trust="derived",
            source_time=timestamp, time_basis="fixture", recorded_at=timestamp,
            content={"text": "Synthetic source only"}, is_current=True, ingestion_status="searchable")
        self.request = SimpleNamespace(id=uuid4(), revision=1)
        dependencies = (memory_dependency(self.source), context_dependency(self.request))
        self.entry = SelectedMemoryEntry(entry_id="historical:fixture", scope=self.scope,
            summary=json.dumps(self.source.content, sort_keys=True), evidence_kind="canonical_record",
            received_at=timestamp, observed_at=timestamp, sources=dependencies,
            guard_binding=GuardedPayloadBinding(scope=self.scope, payload_root_ids=(uuid4(),), dependencies=dependencies))

    def test_exact_source_intent_provenance_and_time_required(self):
        snapshot = SimpleNamespace(entries=(self.entry,))
        self.assertEqual(verify_selected(snapshot, (self.source,), self.request), [self.entry])
        variants = ({"observed_at": None}, {"summary": "Made up answer"},
                    {"evidence_kind": "observed"}, {"guard_binding": None})
        for change in variants:
            with self.subTest(change=change), self.assertRaises(QualificationFailure):
                verify_selected(SimpleNamespace(entries=(self.entry.model_copy(update=change),)),
                                (self.source,), self.request)

    def test_other_intent_old_revision_or_withdrawn_source_cannot_pass(self):
        snapshot = SimpleNamespace(entries=(self.entry,))
        with self.assertRaises(QualificationFailure):
            verify_selected(snapshot, (self.source,), SimpleNamespace(id=uuid4(), revision=1))
        with self.assertRaises(QualificationFailure):
            verify_selected(snapshot, (self.source.model_copy(update={"revision": 1}),), self.request)
        with self.assertRaises(QualificationFailure):
            verify_selected(snapshot, (), self.request, absent=(self.source.source_id,))

