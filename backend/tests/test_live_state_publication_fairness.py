"""Deterministic regression for 12-event historical publication / 8-slot voice queue."""
import asyncio
from datetime import UTC, datetime
from types import SimpleNamespace
import unittest
from uuid import uuid4

from app.live_state.models import (
    GuardedPayloadBinding, SelectedMemoryEntry, SessionScope, SourceDependency, SourceWatermark,
)
from app.live_state.producers import CanonicalStateProducer
from app.live_state.reducer import dependency_available
from app.live_state.service import LiveStateService
from app.realtime.live_state_bridge import SessionLiveStateBridge
from app.voice_routes.contracts import RouteIdentity


def clock():
    return datetime.now(UTC)


class HistoricalPublicationFairnessTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.scope = SessionScope(owner_id="deterministic-fixture", session_id=uuid4(), area_id=uuid4())
        self.service = LiveStateService(self.scope, clock=clock)
        await self.service.start()
        self.addAsyncCleanup(self.service.close)
        self.producer = CanonicalStateProducer(self.service, producer_id="historical-enrichment", clock=clock)
        self.producer.start()
        self.service.mark_ready()
        self.intent = SourceDependency(source_namespace="context_event", source_id=uuid4(), revision=1)

    def entry(self, index):
        memory = SourceDependency(source_namespace="knowledge_source", source_id=uuid4(), revision=2, generation=2)
        dependencies = (self.intent, memory)
        return SelectedMemoryEntry(entry_id=f"historical:fixture-{index}", scope=self.scope,
            summary="Synthetic corrected source", evidence_kind="canonical_record",
            received_at=clock(), observed_at=clock(), sources=dependencies,
            guard_binding=GuardedPayloadBinding(scope=self.scope, payload_root_ids=(uuid4(),), dependencies=dependencies))

    async def collector(self, capacity=8):
        subscription = self.service.subscribe(capacity=capacity)
        initial = await subscription.receive()
        self.assertFalse(initial.resync_required)
        notices = []

        async def consume():
            while (notice := await subscription.receive()) is not None:
                notices.append(notice)
        task = asyncio.create_task(consume())

        async def close():
            subscription.close()
            await task
        self.addAsyncCleanup(close)
        await asyncio.sleep(0)
        return subscription, notices

    async def test_four_corrected_memories_do_not_spuriously_close_normal_voice_attachment(self):
        closed = asyncio.Event()
        reasons = []

        async def close(reason):
            reasons.append(reason)
            closed.set()
        live = SimpleNamespace(service=self.service, scope=self.scope,
            requires_reconstruction=False, close=self.service.close)
        session = SimpleNamespace(_closed=closed, close=close)
        route_identity = RouteIdentity(scope=self.scope, route_id=uuid4(), route_generation=1)
        route = SimpleNamespace(ready=False, request=SimpleNamespace(route=route_identity))
        bridge = SessionLiveStateBridge(session, live, route, clock=clock)
        bridge.start()
        self.addAsyncCleanup(bridge.close)
        _, notices = await self.collector()
        entries = [self.entry(index) for index in range(4)]
        for entry in entries:
            self.assertTrue(self.producer.publish(entry).queued)
        await asyncio.wait_for(self.service.wait_idle(), 1)
        await asyncio.sleep(0)
        self.assertEqual(self.producer._sequence, 12)
        self.assertEqual(len(notices), 12)
        self.assertFalse(any(notice.resync_required for notice in notices))
        self.assertFalse(bridge._subscription._needs_resync)
        self.assertFalse(self.service.needs_resync)
        self.assertIsNone(bridge.failure)
        self.assertFalse(closed.is_set(), reasons)
        self.assertEqual(set(self.service.snapshot().entries), set(entries))

    async def test_capacity_one_ready_consumer_receives_every_notice_in_large_burst(self):
        _, notices = await self.collector(capacity=1)
        for index in range(16):
            self.assertTrue(self.producer.publish(self.entry(index)).queued)
        await asyncio.wait_for(self.service.wait_idle(), 1)
        await asyncio.sleep(0)
        self.assertEqual(len(notices), 48)
        self.assertFalse(any(notice.resync_required for notice in notices))
        self.assertEqual(len({notice.snapshot.cursor.revision for notice in notices}), 48)

    async def test_genuinely_blocked_subscriber_still_requires_resynchronization(self):
        subscription = self.service.subscribe(capacity=8)
        async def close_subscription():
            subscription.close()
        self.addAsyncCleanup(close_subscription)
        await subscription.receive()
        for index in range(4):
            self.assertTrue(self.producer.publish(self.entry(index)).queued)
        await asyncio.wait_for(self.service.wait_idle(), 1)
        notice = await subscription.receive()
        self.assertTrue(notice.resync_required)
        self.assertTrue(notice.critical)
        self.assertEqual(notice.reason, "subscriber_resynchronization_required")
        self.assertFalse(self.service.needs_resync)

    async def test_withdrawal_is_delivered_and_stale_publication_cannot_resurrect(self):
        _, notices = await self.collector()
        entry = self.entry(0)
        self.producer.publish(entry)
        await self.service.wait_idle()
        dependency = entry.sources[1]
        withdrawn = dependency.model_copy(update={"generation": 3})
        self.producer.source_changed(SourceWatermark(source=withdrawn, availability="retracted", recorded_at=clock()))
        self.producer.publish(entry)
        await self.service.wait_idle()
        await asyncio.sleep(0)
        self.assertFalse(self.service.snapshot().entries)
        self.assertFalse(dependency_available(self.service._state, dependency))
        self.assertTrue(any(notice.critical and not notice.snapshot.entries for notice in notices))
        self.assertFalse(any(notice.resync_required for notice in notices))

