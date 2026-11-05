"""Actual session/state/binder/brief composition, controlled offline retrieval.

No provider, database, services or hardware are contacted. SQL storage is the
existing bounded canonical fixture; guards are production implementations.
"""
import asyncio
from copy import deepcopy
from datetime import timedelta
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock
from uuid import uuid4

from app.knowledge.models import SourceError
from app.live_state.canonical import context_dependency, memory_dependency
from app.live_state.models import ObservationEntry
from app.live_state.perception_adapter import observation_from_record
from app.live_state.producers import CanonicalStateProducer
from app.live_state.projection import select_voice_context
from app.live_state.runtime import create_session_live_state
from app.tasks.models import digest
from tests.test_live_state_session_runtime import RuntimeTests, SessionConnection
from tests.test_live_state_perception_adapter import fixture as visual_fixture


class EnrichmentConnection(SessionConnection):
    async def fetchval(self, query, *args):
        if " ".join(query.lower().split()) == (
                "select content from jarvis_context_payloads where owner_id=$1 and id=$2"):
            self.query(query, args)
            payload = self.db.payloads.get(args[1])
            return payload["content"] if payload and payload["owner_id"] == args[0] else None
        return await super().fetchval(query, *args)

    async def fetch(self, query, *args):
        if "from jarvis_visual_focuses f" in " ".join(query.lower().split()):
            return []
        return await super().fetch(query, *args)


class HistoricalEnrichmentCompositionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.f = RuntimeTests()
        self.f.setUp()
        self.f.connection = EnrichmentConnection(self.f.db)
        self.monotonic = 0.0
        self.calls, self.active, self.max_active = [], 0, 0
        self.entered, self.release = asyncio.Event(), asyncio.Event()
        self.release.set()
        self.block_first = False
        self.response_source = self.f.source

        async def resolve(owner, area, document):
            self.assertEqual((owner, area), (self.f.scope.owner_id, self.f.scope.area_id))
            source = next((s for s in self.f.originals.values() if s.document_id == document), None)
            if source is None or not self.f.db.available(owner, source.payload_id):
                raise SourceError("knowledge_source_unavailable", "Fixture original removed")
            return source

        async def revalidate(source):
            fresh = await resolve(source.owner_id, source.area_id, source.document_id)
            if fresh != source:
                raise SourceError("knowledge_source_changed", "Fixture original changed")
            return fresh

        self.f.sources.resolve_document = AsyncMock(side_effect=resolve)
        self.f.sources.revalidate = AsyncMock(side_effect=revalidate)

        async def recall(owner, area, query):
            self.assertEqual((owner, area), (self.f.scope.owner_id, self.f.scope.area_id))
            self.assertFalse(self.f.connection.active, "Retrieval must run outside source transactions")
            self.calls.append(query)
            selected = self.response_source
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            try:
                if self.block_first and len(self.calls) == 1:
                    self.entered.set()
                    await self.release.wait()
                return {"status": "available", "sources": [selected.original_evidence() | {"content_included": True}]}
            finally:
                self.active -= 1

        self.retrieval = SimpleNamespace(recall=AsyncMock(side_effect=recall))
        self.services = self.f.services(background_task_service=None)
        self.services.knowledge_runtime.retrieval = self.retrieval
        self.live = None

    async def asyncTearDown(self):
        await self.f.asyncTearDown()

    async def until(self, predicate):
        async with asyncio.timeout(2):
            while not predicate():
                await asyncio.sleep(.002)

    def new_source(self, text):
        content = {"text": text}
        source = self.f.source.model_copy(update={"source_id": uuid4(), "payload_id": uuid4(),
            "document_id": uuid4().hex, "content": content, "content_digest": digest(content)})
        self.f.originals[source.source_id] = source
        self.f.db.payloads[source.payload_id] = dict(owner_id=source.owner_id, content=content,
            content_digest=source.content_digest, availability="available")
        return source

    def intent(self, text):
        old = self.f.record
        content = {"text": text}
        record = old.model_copy(update={"id": uuid4(), "payload_id": uuid4(),
            "source_event_id": uuid4().hex, "sequence": old.sequence + 1,
            "origin_sequence": old.origin_sequence + 1, "content": content, "content_digest": digest(content)})
        self.f.record = record
        self.f.db.events[record.id] = record.model_dump()
        self.f.db.payloads[record.payload_id] = dict(owner_id=record.owner_id, content=content,
            content_digest=record.content_digest, availability="available")
        self.f.context.snapshot.return_value = {"recent": [record], "pinned": [], "watermark": record.sequence}
        return record

    async def start(self, **enrichment_options):
        self.live = create_session_live_state(self.services, scope=self.f.scope, enabled=True,
            perception_brief_enabled=True, clock=lambda: self.f.now, interval_seconds=.01, timeout_seconds=.5,
            enrichment_options={"monotonic": lambda: self.monotonic, "debounce_seconds": 0,
                                **enrichment_options})
        self.f.runtimes.append(self.live)
        await self.live.start()
        await self.until(lambda: self.live.service.snapshot().readiness == "ready")
        return self.live

    def memories(self):
        return [e for e in self.live.service.snapshot().entries if e.kind == "selected_memory"]

    def brief_text(self):
        composition = self.live.perception_brief_bridge.composition
        return " ".join(p.finding for p in composition.brief.prior_evidence) if composition else ""

    async def test_blocked_A_then_B_stale_A_never_publishes_and_B_reaches_bound_brief(self):
        self.intent("Inspect experiment A")
        a = self.response_source = self.new_source("A requires 41.5 units, not 415.")
        self.block_first = True
        self.release.clear()
        live = await self.start()
        await asyncio.wait_for(self.entered.wait(), 1)
        await self.until(lambda: live.perception_brief_bridge.composition is not None)
        self.assertEqual(live.service.snapshot().readiness, "ready")
        self.assertFalse(self.memories())
        b = self.response_source = self.new_source("B requires 12.75 units, not 1275.")
        request_b = self.intent("Inspect experiment B")
        await self.until(lambda: live.enrichment._wanted is not None
                         and "experiment B" in live.enrichment._wanted.query)
        self.release.set()
        await self.until(lambda: "12.75" in self.brief_text())
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(self.max_active, 1)
        self.assertEqual(live.enrichment.stale_results, 1)
        self.assertNotIn("41.5", select_voice_context(live.service.snapshot(), now=self.f.now).text)
        entry, = self.memories()
        self.assertIn(memory_dependency(b), entry.sources)
        self.assertIn(context_dependency(request_b), entry.sources)
        self.assertNotIn(memory_dependency(a), entry.sources)
        self.assertEqual(entry.observed_at, b.source_time)
        self.assertTrue(await live.binder.available(entry.guard_binding))
        composition = live.perception_brief_bridge.composition
        self.assertEqual(composition.brief.intent_ref, context_dependency(request_b))
        async with live.perception_runtime.binder.guard(composition.brief, live.perception_brief_bridge._binding):
            pass
        self.f.sources.current_snapshot.assert_not_awaited()
        self.assertIs(live.enrichment.retrieval, self.retrieval)

    async def test_stored_brief_fixture_read_is_exact_and_owner_scoped(self):
        root = self.f.record.payload_id
        sql = "select content from jarvis_context_payloads where owner_id=$1 and id=$2"
        self.assertEqual(await self.f.connection.fetchval(sql, self.f.scope.owner_id, root),
                         self.f.db.payloads[root]["content"])
        self.assertIsNone(await self.f.connection.fetchval(sql, "another-owner", root))
        self.assertIsNone(await self.f.connection.fetchval(sql, self.f.scope.owner_id, uuid4()))

    async def test_newest_pending_context_and_two_per_minute_budget(self):
        self.intent("Inspect A")
        self.block_first = True
        self.release.clear()
        live = await self.start()
        await asyncio.wait_for(self.entered.wait(), 1)
        for text in ("Inspect B", "Inspect C", "Inspect D"):
            self.intent(text)
            await self.until(lambda text=text: live.enrichment._wanted is not None
                             and text in live.enrichment._wanted.query)
        self.release.set()
        await self.until(lambda: len(self.calls) == 2 and self.active == 0)
        self.assertIn("Inspect D", self.calls[1])
        self.assertEqual(self.max_active, 1)
        self.intent("Inspect E")
        await self.until(lambda: live.enrichment.status == "budget_wait")
        self.assertEqual(len(self.calls), 2)
        self.intent("Inspect F")
        await self.until(lambda: live.enrichment._wanted is not None and "Inspect F" in live.enrichment._wanted.query)
        self.monotonic = 61
        live.enrichment._wake.set()
        await self.until(lambda: len(self.calls) == 3)
        self.assertIn("Inspect F", self.calls[2])

    async def test_equal_intent_and_projection_renewal_do_not_repeat_lookup(self):
        self.intent("Inspect the display")
        live = await self.start()
        await self.until(lambda: bool(self.memories()))
        before = live.enrichment.worker.refresh_count
        self.intent("  INSPECT   the display  ")
        await self.until(lambda: live.enrichment.worker.refresh_count > before + 3)
        self.assertEqual(len(self.calls), 1)
        await self.until(lambda: context_dependency(self.f.record) in self.memories()[0].sources)
        self.assertEqual(live.enrichment.status, "available")

    async def test_withdrawal_during_lookup_rejects_original_without_blocking_readiness(self):
        self.block_first = True
        self.release.clear()
        live = await self.start()
        await asyncio.wait_for(self.entered.wait(), 1)
        self.f.db.remove(self.response_source.payload_id)
        self.f.originals.clear()
        self.release.set()
        await self.until(lambda: self.active == 0 and live.enrichment.status == "available")
        self.assertFalse(self.memories())
        self.assertEqual(live.service.snapshot().readiness, "ready")
        self.assertGreater(live.enrichment.omitted_sources, 0)

    async def test_withdrawn_or_new_generation_sources_retract_and_old_binding_fails(self):
        live = await self.start()
        await self.until(lambda: bool(self.memories()))
        entry, = self.memories()
        binding = await live.bind(select_voice_context(live.service.snapshot(), now=self.f.now))
        self.f.originals[self.f.source.source_id] = self.f.source.model_copy(update={"generation": 2})
        with self.assertRaises(SourceError):
            async with live.guard(binding):
                self.fail("An old lifecycle generation authorized speech")
        await self.until(lambda: not self.memories())
        self.assertFalse(await live.binder.available(entry.guard_binding))
        self.assertEqual(len(self.calls), 1)

    async def test_timeout_and_shutdown_join_do_not_close_borrowed_retrieval(self):
        self.block_first = True
        self.release.clear()
        live = await self.start(deadline_seconds=.03)
        await self.until(lambda: live.enrichment.status == "unavailable")
        self.assertEqual(live.service.snapshot().readiness, "ready")
        self.assertEqual(self.active, 0)
        await live.close()
        self.assertTrue(live.enrichment._task.done())
        self.assertTrue(live.enrichment.worker._task.done())
        self.assertEqual(len(self.calls), 1)

    async def test_actual_activity_transition_coalesces_same_label_frames(self):
        live = await self.start()
        await self.until(lambda: bool(self.memories()))
        producer = CanonicalStateProducer(live.service, producer_id="fixture-observation", clock=lambda: self.f.now)
        producer.start()

        async def activity(label):
            _, _, template, _ = visual_fixture()
            content = deepcopy(template.content)
            raw = content["record"]
            raw.update(owner_id=self.f.scope.owner_id, session_id=str(self.f.scope.session_id), activity=label,
                activity_confidence=.95, started_at=self.f.now.isoformat(), ended_at=self.f.now.isoformat(),
                created_at=self.f.now.isoformat())
            content["images"][0].update(session_id=str(self.f.scope.session_id), captured_at=self.f.now.isoformat())
            record = template.model_copy(update={"owner_id": self.f.scope.owner_id, "session_id": self.f.scope.session_id,
                "content": content, "content_digest": digest(content), "source_time": self.f.now, "recorded_at": self.f.now})
            self.f.db.events[record.id] = record.model_dump()
            self.f.db.payloads[record.payload_id] = dict(owner_id=record.owner_id, content=content,
                content_digest=record.content_digest, availability="available")
            _, frames = observation_from_record(record)
            binding = await live.binder.bind_record(record, "Activity: " + label)
            producer.publish(ObservationEntry(entry_id="working_observation", scope=self.f.scope,
                summary="Activity: " + label, frames=frames, confidence=.95, sources=binding.dependencies,
                guard_binding=binding, evidence_kind="observed", observed_at=self.f.now, received_at=self.f.now,
                fresh_until=self.f.now + timedelta(seconds=10), expires_at=self.f.now + timedelta(seconds=10)))
            await live.service.wait_idle()

        await activity("cooking")
        await self.until(lambda: len(self.calls) == 2 and self.active == 0)
        await activity("cooking")
        count = live.enrichment.worker.refresh_count
        await self.until(lambda: live.enrichment.worker.refresh_count > count + 2)
        self.assertEqual(len(self.calls), 2)
        await activity("leaving")
        await self.until(lambda: live.enrichment.status == "budget_wait")
        self.monotonic = 61
        live.enrichment._wake.set()
        await self.until(lambda: len(self.calls) == 3)
        self.assertIn("leaving", self.calls[2])


if __name__ == "__main__":
    unittest.main()
