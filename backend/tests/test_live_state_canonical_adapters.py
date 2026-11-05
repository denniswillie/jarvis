"""Canonical binding, projection ordering and nonblocking refresh regressions."""
import asyncio
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock
from uuid import uuid4

from app.knowledge.models import SourceError, SourceVersion
from app.live_state.canonical import CanonicalContextBinder, context_dependency, memory_dependency
from app.live_state.models import GuardedPayloadBinding, SessionScope, StateCursor
from app.live_state.producers import CanonicalStateProducer, selected_memory_entry
from app.live_state.projection import VoiceSelection
from app.live_state.refresh import CanonicalRefreshWorker
from app.runtime.events import DomainEvent, DomainEventBus, DomainEventName, WorkingMemoryUpdatedPayload
from app.tasks.context_events import ContextRecord


NOW = datetime(2026, 10, 30, 18, tzinfo=UTC)


def memory(scope, **changes):
    return SourceVersion(**(dict(owner_id=scope.owner_id, area_id=scope.area_id,
        source_id=uuid4(), revision=1, generation=1, payload_id=uuid4(),
        document_id="synthetic-document", bank_name="synthetic-bank", content_digest="a" * 64,
        source_type="note", trust="user_input", source_time=NOW - timedelta(days=1),
        time_basis="original user report", recorded_at=NOW, content={"text": "A synthetic fact"},
        is_current=True, ingestion_status="searchable") | changes))


def record(scope):
    return ContextRecord(id=uuid4(), owner_id=scope.owner_id, session_id=scope.session_id,
        sequence=1, origin_sequence=1, source_namespace="voice:openai", source_event_id="item-1",
        revision=1, supersedes_event_id=None, kind="user_transcript", trust="user_input",
        finality="final", source_time=NOW, recorded_at=NOW, payload_id=uuid4(),
        availability="available", content_digest="b" * 64, content={"text": "A synthetic request"},
        is_current=True)


class CanonicalBindingTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.scope = SessionScope(owner_id="live-state-fixture", session_id=uuid4(), area_id=uuid4())
        self.connection = SimpleNamespace(is_in_transaction=lambda: True,
            fetchval=AsyncMock(return_value=True), fetch=AsyncMock(return_value=[]),
            execute=AsyncMock(), executemany=AsyncMock())
        @asynccontextmanager
        async def transaction(owner):
            self.assertEqual(owner, self.scope.owner_id)
            yield self.connection
        self.sources = SimpleNamespace(pool=None, transaction=transaction,
            _area=AsyncMock(), _read_many=AsyncMock(return_value=[]))
        self.binder = CanonicalContextBinder(self.sources, self.scope, clock=lambda: NOW)
        self.binder.tasks.lock_owner = AsyncMock()
        self.binder.tasks.payload = AsyncMock(return_value=uuid4())

    def selection(self, dependencies=(), roots=None, **changes):
        values = dict(scope=self.scope, cursor=StateCursor(epoch=uuid4(), revision=3),
            text="Exact combined context", dependencies=dependencies,
            parent_payload_root_ids=roots or (uuid4(),), selected_entry_ids=("entry",),
            omitted_entry_ids=(), expires_at=NOW + timedelta(seconds=10))
        return VoiceSelection(**(values | changes))

    async def test_memory_binding_keeps_original_time_and_exact_lineage(self):
        value = memory(self.scope)
        self.sources._read_many.return_value = [value]
        binding = await self.binder.bind_memory(value, "A synthetic fact")
        self.assertEqual(binding.dependencies, (memory_dependency(value),))
        self.assertEqual(self.sources._read_many.await_count, 2)
        document = self.binder.tasks.payload.call_args.args[2]
        self.assertEqual(document["text"], "A synthetic fact")
        entry = selected_memory_entry(value, "A synthetic fact", binding=binding,
            received_at=NOW, expires_at=NOW + timedelta(seconds=20))
        self.assertEqual(entry.observed_at, value.source_time)
        self.assertNotEqual(entry.observed_at, entry.received_at)

    async def test_record_binding_rechecks_original_and_records_usage(self):
        value = record(self.scope)
        self.binder.context._record = AsyncMock(return_value=value)
        binding = await self.binder.bind_record(value, value.content["text"])
        self.assertEqual(binding.dependencies, (context_dependency(value),))
        self.assertEqual(self.connection.execute.call_args.args[-1], "current")

    async def test_changed_record_never_materializes(self):
        value = record(self.scope)
        self.binder.context._record = AsyncMock(return_value=value.model_copy(update={"is_current": False}))
        with self.assertRaises(SourceError):
            await self.binder.bind_record(value, "old text")
        self.binder.tasks.payload.assert_not_awaited()

    async def test_combined_text_gets_its_own_removable_root(self):
        value = memory(self.scope)
        self.sources._read_many.return_value = [value]
        selection = self.selection((memory_dependency(value),), roots=(uuid4(), uuid4()))
        binding = await self.binder(selection)
        self.assertEqual(binding.payload_root_ids, (self.binder.tasks.payload.return_value,))
        self.assertEqual(binding.dependencies, selection.dependencies)
        self.assertEqual(self.binder.tasks.payload.call_args.args[2]["text"], selection.text)
        parents = {row[1] for row in self.connection.executemany.call_args.args[1]}
        self.assertEqual(parents, set(selection.parent_payload_root_ids))

    async def test_retired_parent_prevents_combination(self):
        self.connection.fetchval.side_effect = [True, False]
        with self.assertRaises(SourceError):
            await self.binder(self.selection())
        self.binder.tasks.payload.assert_not_awaited()

    async def test_changed_generation_prevents_combination(self):
        value = memory(self.scope)
        self.sources._read_many.return_value = [value.model_copy(update={"generation": 2})]
        with self.assertRaises(SourceError):
            await self.binder(self.selection((memory_dependency(value),)))
        self.binder.tasks.payload.assert_not_awaited()

    async def test_expired_selection_does_not_touch_storage(self):
        with self.assertRaises(SourceError):
            await self.binder(self.selection(expires_at=NOW))
        self.connection.fetchval.assert_not_awaited()

    async def test_wrong_scope_rejected_before_storage(self):
        other = self.scope.model_copy(update={"session_id": uuid4()})
        with self.assertRaises(SourceError):
            await self.binder(self.selection(scope=other))
        self.connection.fetchval.assert_not_awaited()

    async def test_output_guard_rechecks_after_successful_binding(self):
        value = memory(self.scope)
        self.sources._read_many.return_value = [value]
        binding = await self.binder.bind_memory(value, "A synthetic fact")
        self.sources._read_many.return_value = [value.model_copy(update={"generation": 2})]
        self.assertFalse(await self.binder.available(binding))


class ProducerTests(unittest.TestCase):
    def test_local_projection_order_does_not_replace_canonical_versions(self):
        scope = SessionScope(owner_id="producer-fixture", session_id=uuid4(), area_id=uuid4())
        value = memory(scope, revision=9, generation=11)
        binding = GuardedPayloadBinding(scope=scope, payload_root_ids=(uuid4(),),
            dependencies=(memory_dependency(value),))
        entry = selected_memory_entry(value, "A fact", binding=binding,
            received_at=NOW, expires_at=NOW + timedelta(seconds=10))
        events, registrations = [], []
        def submit(event):
            events.append(event)
            return SimpleNamespace(queued=True)
        sink = SimpleNamespace(snapshot=lambda: SimpleNamespace(scope=scope), submit=submit,
            register_producer=lambda *args: registrations.append(args))
        producer = CanonicalStateProducer(sink, producer_id="selected-memory", clock=lambda: NOW)
        producer.start()
        producer.publish(entry)
        self.assertEqual(len(registrations), 1)
        self.assertEqual([event.producer.aggregate.revision for event in events], [1, 2])
        self.assertEqual(events[-1].entry.sources[0].revision, 9)
        self.assertEqual(events[-1].entry.sources[0].generation, 11)
        with self.assertRaises(RuntimeError):
            producer.start()


class RefreshTests(unittest.IsolatedAsyncioTestCase):
    async def test_bus_publish_does_not_wait_for_read_and_close_unsubscribes(self):
        bus = DomainEventBus()
        scope = SessionScope(owner_id="refresh-fixture", session_id=uuid4())
        started, release, applied = asyncio.Event(), asyncio.Event(), asyncio.Event()
        async def read():
            started.set()
            await release.wait()
            return 7
        async def apply(value):
            self.assertEqual(value, 7)
            applied.set()
        worker = CanonicalRefreshWorker(scope=scope, bus=bus,
            event_names=(DomainEventName.WORKING_MEMORY_UPDATED,), read=read,
            apply=apply, invalidate=AsyncMock(), interval_seconds=0.02)
        worker.start()
        try:
            await asyncio.wait_for(started.wait(), 1)
            result = await asyncio.wait_for(bus.publish(DomainEvent(
                name=DomainEventName.WORKING_MEMORY_UPDATED,
                payload=WorkingMemoryUpdatedPayload(scope.owner_id, scope.session_id, 2))), 0.2)
            self.assertEqual(result.delivered_count, 1)
            self.assertFalse(applied.is_set())
            release.set()
            await asyncio.wait_for(applied.wait(), 1)
        finally:
            await worker.close()
        self.assertFalse(bus._handlers)

    async def test_periodic_reread_recovers_without_an_event(self):
        bus = DomainEventBus()
        scope = SessionScope(owner_id="refresh-fixture", session_id=uuid4())
        recovered = asyncio.Event()
        read = AsyncMock(side_effect=[RuntimeError("fixture unavailable"), "fresh"])
        async def apply(value):
            self.assertEqual(value, "fresh")
            recovered.set()
        invalidate = AsyncMock()
        worker = CanonicalRefreshWorker(scope=scope, bus=bus,
            event_names=(DomainEventName.WORKING_MEMORY_UPDATED,), read=read,
            apply=apply, invalidate=invalidate, interval_seconds=0.01)
        worker.start()
        try:
            await asyncio.wait_for(recovered.wait(), 1)
            self.assertGreaterEqual(read.await_count, 2)
            invalidate.assert_awaited_once_with("canonical_refresh_unavailable")
        finally:
            await worker.close()

    async def test_unsubscribe_preserves_another_subscriber(self):
        bus = DomainEventBus()
        first, second = AsyncMock(), AsyncMock()
        name = DomainEventName.WORKING_MEMORY_UPDATED
        bus.subscribe(name, first)
        bus.subscribe(name, second)
        self.assertTrue(bus.unsubscribe(name, first))
        self.assertFalse(bus.unsubscribe(name, first))
        event = DomainEvent(name=name, payload=WorkingMemoryUpdatedPayload("fixture", uuid4(), 1))
        result = await bus.publish(event)
        self.assertEqual(result.delivered_count, 1)
        first.assert_not_awaited()
        second.assert_awaited_once_with(event)
