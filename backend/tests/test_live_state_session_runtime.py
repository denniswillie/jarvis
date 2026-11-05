"""Offline composition tests: real state/binder/producers/workers, bounded IO fakes."""
import asyncio
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock
from uuid import uuid4

from app.knowledge.models import SourceError, SourceVersion
from app.live_state.models import SessionScope
from app.live_state.projection import select_voice_context
from app.live_state.runtime import LiveStateRuntime, create_session_live_state
from app.runtime.events import DomainEvent, DomainEventBus, DomainEventName, MemoryItemChangedPayload
from app.runtime.feature_config import MemoryAuthority, TaskLimits
from app.tasks.context_events import ContextRecord
from app.tasks.context_repository import RECORD_SQL
from app.tasks.repository import TaskRepository
from tests.test_live_state_task_context import Connection, Database


NOW = datetime(2026, 10, 30, 20, tzinfo=UTC)


class SessionConnection(Connection):
    """Extend the bounded SQL store for this mixed memory/context/task fixture.

    Task reads, source-copy edges and revocation remain the same persistence
    operations used by the focused task tests. No approved binding is invented
    here and no canonical repository method is replaced with an optimistic read.
    """

    async def execute(self, sql, *args):
        normalized = " ".join(sql.lower().split())
        if normalized.startswith("insert into jarvis_context_memory_sources"):
            self.query(sql, args)
            owner, dependent, area, source, revision, generation, original, content_digest, usage = args
            assert self.active and (owner, area) == (self.db.scope.owner_id, self.db.scope.area_id)
            assert self.db.available(owner, original)
            assert self.db.payloads[dependent]["owner_id"] == owner
            assert revision > 0 and generation > 0 and usage in {"current", "historical"}
            key = (owner, dependent, source, revision)
            # Mirror the actual insert's unique key and conflict-do-nothing
            # semantics, retaining all provenance fields for assertions.
            self.db.memory_sources.setdefault(key, args)
            return "INSERT 1"
        return await super().execute(sql, *args)

    async def fetch(self, sql, *args):
        normalized = " ".join(sql.lower().split())
        if normalized.startswith("select id from jarvis_knowledge_sources"):
            self.query(sql, args)
            assert self.active
            assert (args[0], args[1]) == (self.db.scope.owner_id, self.db.scope.area_id)
            return [{"id": identity} for identity in args[2]]
        if normalized.startswith("select e.*,p.availability,p.content_digest,p.content"):
            assert "e.id=any($2::uuid[])" in normalized
            self.query(sql, args)
            rows = [await self.fetchrow(RECORD_SQL + " where e.owner_id=$1 and e.id=$2", args[0], identity)
                    for identity in args[1]]
            return [row for row in rows if row is not None]
        return await super().fetch(sql, *args)


class SessionDatabase(Database):
    def __init__(self, scope):
        super().__init__(scope)
        self.memory_sources = {}

    @asynccontextmanager
    async def acquire(self):
        yield SessionConnection(self)


class RuntimeTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.scope = SessionScope(owner_id="integration-fixture", session_id=uuid4(), area_id=uuid4())
        self.now = NOW
        self.record = ContextRecord(id=uuid4(), owner_id=self.scope.owner_id,
            session_id=self.scope.session_id, sequence=1, origin_sequence=1,
            source_namespace="voice:test", source_event_id="input-1", revision=1,
            supersedes_event_id=None, kind="user_transcript", trust="user_input", finality="final",
            source_time=NOW, recorded_at=NOW, payload_id=uuid4(), availability="available",
            content_digest="b" * 64, content={"text": "Inspect the display"}, is_current=True)
        self.source = SourceVersion(owner_id=self.scope.owner_id, area_id=self.scope.area_id,
            source_id=uuid4(), revision=1, generation=1, payload_id=uuid4(), document_id="original",
            bank_name="offline", content_digest="a" * 64, source_type="note", trust="user_input",
            source_time=NOW - timedelta(days=1), time_basis="user report", recorded_at=NOW,
            content={"text": "The reference value is 14"}, is_current=True, ingestion_status="not_queued")
        self.originals = {self.source.source_id: self.source}
        self.records = {self.record.id: self.record}
        self.selection = {"area_id": str(self.scope.area_id), "area_status": "active",
                          "selection_revision": 0, "selection_event_id": None}
        self.db = SessionDatabase(self.scope)
        self.connection = SessionConnection(self.db)
        for identity, content in ((self.source.payload_id, self.source.content),
                                  (self.record.payload_id, self.record.content)):
            self.db.payloads[identity] = {"owner_id": self.scope.owner_id, "content": content,
                                         "availability": "available", "created_at": NOW}
        self.db.events[self.record.id] = self.record.model_dump()

        @asynccontextmanager
        async def transaction(owner):
            self.assertEqual(owner, self.scope.owner_id)
            async with self.connection.transaction():
                yield self.connection

        async def read_many(connection, owner, area, requests):
            self.assertEqual((owner, area), (self.scope.owner_id, self.scope.area_id))
            try:
                return [self.originals[identity] for identity, revision in requests]
            except KeyError:
                raise SourceError("missing_original", "Original removed") from None

        async def current_snapshot(owner, area, *, limit):
            self.assertEqual((owner, area), (self.scope.owner_id, self.scope.area_id))
            return {"sources": list(self.originals.values())[:limit]}

        self.sources = SimpleNamespace(pool=self.db,
            transaction=transaction, _area=AsyncMock(), _read_many=read_many,
            current_snapshot=AsyncMock(side_effect=current_snapshot))
        self.context = SimpleNamespace(snapshot=AsyncMock(return_value={
            "recent": [self.record], "pinned": [], "watermark": 1}))
        self.areas = SimpleNamespace(read_in_transaction=AsyncMock(side_effect=lambda *args: dict(self.selection)))
        self.task_answer = "The reference value is 14.25, not 1425."
        identity = self.db.task(status="succeeded", answer=self.task_answer, area=self.scope.area_id)
        self.task = self.db.tasks[identity]
        self.task.update(created_at=NOW - timedelta(hours=1), updated_at=NOW - timedelta(minutes=2))
        self.task_request = self.db.events[self.task["request_event_id"]]
        self.task_request.update(source_time=NOW - timedelta(hours=1), recorded_at=NOW - timedelta(hours=1))
        self.task_receipt = self.db.receipts[identity]
        self.task_receipt["result_recorded_at"] = NOW - timedelta(minutes=3)
        self.tasks = TaskRepository(self.db, TaskLimits())
        self.bus = DomainEventBus()
        self.runtimes = []

    def runtime(self, **options):
        values = dict(scope=self.scope, sources=self.sources, context=self.context,
            area_selections=self.areas, bus=self.bus, access_check=lambda owner: None,
            tasks=self.tasks, clock=lambda: self.now, interval_seconds=0.01,
            timeout_seconds=0.2)
        runtime = LiveStateRuntime(**(values | options))
        # Spies delegate to production implementations; actual payloads and
        # source edges are persisted in the bounded offline store.
        runtime.binder.tasks.payload = AsyncMock(wraps=runtime.binder.tasks.payload)
        runtime.binder.context._records = AsyncMock(wraps=runtime.binder.context._records)
        self.runtimes.append(runtime)
        return runtime

    async def asyncTearDown(self):
        for runtime in self.runtimes:
            await runtime.close()

    async def until(self, predicate):
        async with asyncio.timeout(1):
            while not predicate():
                await asyncio.sleep(0.002)

    async def ready(self, runtime):
        await runtime.start()
        await self.until(lambda: runtime.service.snapshot().readiness == "ready")

    def event(self, *, owner=None, session=None):
        return DomainEvent(DomainEventName.MEMORY_ITEM_CHANGED, MemoryItemChangedPayload(
            owner_id=owner or self.scope.owner_id, memory_item_id=self.source.source_id,
            change_type="changed", session_id=session or self.scope.session_id))

    def services(self, **overrides):
        def context_for(owner):
            if owner != self.scope.owner_id:
                raise SourceError("knowledge_owner_mismatch", "Wrong authenticated owner")
            return self.context
        knowledge = SimpleNamespace(sources=self.sources, area_selections=self.areas, context_for=context_for)
        return SimpleNamespace(**(dict(memory_authority=MemoryAuthority.HINDSIGHT,
            knowledge_runtime=knowledge, background_task_service=SimpleNamespace(repository=self.tasks),
            working_memory_service=None, domain_event_bus=self.bus, _closed=False) | overrides))

    def test_factory_is_explicit_and_rejects_fallback_or_wrong_owner(self):
        self.assertIsNone(create_session_live_state(object(), scope=self.scope))
        with self.assertRaises(SourceError):
            create_session_live_state(self.services(memory_authority=MemoryAuthority.LEGACY),
                                      scope=self.scope, enabled=True)
        with self.assertRaises(SourceError):
            create_session_live_state(self.services(),
                scope=self.scope.model_copy(update={"owner_id": "other"}), enabled=True)
        runtime = create_session_live_state(self.services(), scope=self.scope, enabled=True)
        self.assertIs(runtime.context, self.context)
        self.assertIs(runtime.tasks, self.tasks)
        self.sources.current_snapshot.assert_not_awaited()

    async def test_real_components_publish_guarded_originals_and_worker_result(self):
        runtime = self.runtime()
        await self.ready(runtime)
        snapshot = runtime.service.snapshot()
        self.assertEqual(snapshot.conversation.turn_id, self.record.id)
        self.assertEqual({entry.kind for entry in snapshot.entries}, {"focus", "selected_memory", "task"})
        selection = select_voice_context(snapshot, now=self.now)
        self.assertIn("reference value is 14", selection.text)
        task = next(entry for entry in snapshot.entries if entry.kind == "task")
        self.assertIn(task.entry_id, selection.selected_entry_ids)
        self.assertNotIn(task.entry_id, selection.omitted_entry_ids)
        self.assertIn(self.task_answer, selection.text)
        self.assertEqual(task.status, "succeeded")
        self.assertEqual(task.task.revision, self.task["revision"])
        self.assertEqual(task.sources[0].revision, self.task["last_event_sequence"])
        self.assertIsNone(task.sources[0].generation)
        self.assertEqual(task.observed_at, self.task["updated_at"])
        self.assertNotEqual(task.observed_at, task.received_at)
        self.assertIn(self.task_receipt["result_recorded_at"].isoformat(), task.summary)
        self.assertIn("delivery/heard state is unknown", task.summary)
        self.assertTrue(await runtime.binder.available(task.guard_binding))
        binding = await runtime.bind(selection)
        async with runtime.guard(binding):
            self.assertEqual(binding.scope, self.scope)
        memory = next(entry for entry in snapshot.entries if entry.kind == "selected_memory")
        self.assertEqual(memory.observed_at, self.source.source_time)
        self.assertEqual(memory.sources[0].generation, 1)
        self.assertIn((self.scope.owner_id, memory.guard_binding.payload_root_ids[0],
            self.scope.area_id, self.source.source_id, self.source.revision,
            self.source.generation, self.source.payload_id, self.source.content_digest, "current"),
            self.db.memory_sources.values())

    async def test_current_pending_question_is_source_backed_not_a_claim_of_completion(self):
        self.task["status"] = "waiting_for_input"
        self.task_receipt["event_type"] = "waiting_for_input"
        question = "Which value should I compare: 14.25 or 1425?"
        self.db.payloads[self.task["result_payload_id"]]["content"] = {"answer": question}
        runtime = self.runtime(interval_seconds=5)
        await self.ready(runtime)
        selection = select_voice_context(runtime.service.snapshot(), now=self.now)
        self.assertIn(question, selection.text)
        self.assertIn("Pending question", selection.text)
        task = next(entry for entry in runtime.service.snapshot().entries if entry.kind == "task")
        self.assertEqual(task.status, "waiting_for_input")
        self.assertIn("not authorization", task.summary)
        binding = await runtime.bind(selection)
        async with runtime.guard(binding):
            self.assertEqual(binding.scope, self.scope)
        # Same intent and sequence, different execution owner: old question is
        # immediately unusable, without waiting for a bus event or next poll.
        self.task.update(status="running", current_attempt_id=uuid4(), fence=self.task["fence"] + 1,
                         result_payload_id=None)
        with self.assertRaises(SourceError):
            async with runtime.guard(binding):
                self.fail("An old pending question survived the task state change")

    async def test_task_change_between_candidate_read_and_apply_never_publishes_old_answer(self):
        runtime = self.runtime(interval_seconds=5)
        await self.ready(runtime)
        previous = await runtime.bind(select_voice_context(runtime.service.snapshot(), now=self.now))
        value = await runtime._read()
        revision, sequence = self.task["revision"], self.task["last_event_sequence"]
        self.task.update(status="running", current_attempt_id=uuid4(), fence=self.task["fence"] + 1,
                         result_payload_id=None, updated_at=NOW - timedelta(seconds=1))
        await runtime._apply(value)
        snapshot = runtime.service.snapshot()
        task = next(entry for entry in snapshot.entries if entry.kind == "task")
        self.assertEqual((task.task.revision, task.sources[0].revision), (revision, sequence))
        self.assertEqual(task.status, "running")
        self.assertEqual(task.observed_at, self.task["updated_at"])
        self.assertNotIn(self.task_answer, select_voice_context(snapshot, now=self.now).text)
        with self.assertRaises(SourceError):
            async with runtime.guard(previous):
                self.fail("The previously bound result remained current")
        current = await runtime.bind(select_voice_context(snapshot, now=self.now))
        async with runtime.guard(current):
            pass

    async def test_request_correction_between_read_and_apply_omits_old_task(self):
        runtime = self.runtime(interval_seconds=5)
        await self.ready(runtime)
        previous = await runtime.bind(select_voice_context(runtime.service.snapshot(), now=self.now))
        value = await runtime._read()
        self.task_request["is_current"] = False
        await runtime._apply(value)
        snapshot = runtime.service.snapshot()
        self.assertFalse(any(entry.kind == "task" for entry in snapshot.entries))
        self.assertNotIn(self.task_answer, select_voice_context(snapshot, now=self.now).text)
        self.assertFalse(runtime.task_context._cache)
        with self.assertRaises(SourceError):
            async with runtime.guard(previous):
                self.fail("Corrected request evidence authorized an old answer")

    async def test_removed_worker_result_revokes_copy_without_erasing_valid_task_status(self):
        runtime = self.runtime(interval_seconds=5)
        await self.ready(runtime)
        previous = await runtime.bind(select_voice_context(runtime.service.snapshot(), now=self.now))
        self.db.remove(self.task["result_payload_id"])
        with self.assertRaises(SourceError):
            async with runtime.guard(previous):
                self.fail("Removed result reached the provider-write boundary")
        await runtime._apply(await runtime._read())
        task = next(entry for entry in runtime.service.snapshot().entries if entry.kind == "task")
        self.assertEqual(task.status, "succeeded")
        self.assertNotIn(self.task_answer, task.summary)
        self.assertIn("unavailable", task.summary)
        self.assertTrue(await runtime.binder.available(task.guard_binding))

    async def test_invalidated_manifest_cannot_keep_result_current(self):
        runtime = self.runtime(interval_seconds=5)
        await self.ready(runtime)
        previous = await runtime.bind(select_voice_context(runtime.service.snapshot(), now=self.now))
        self.task_receipt["manifest_status"] = "revoked"
        with self.assertRaises(SourceError):
            async with runtime.guard(previous):
                self.fail("An invalidated manifest still authorized its old answer")
        await runtime._apply(await runtime._read())
        selection = select_voice_context(runtime.service.snapshot(), now=self.now)
        self.assertNotIn(self.task_answer, selection.text)
        self.assertIn("Worker text is unavailable", selection.text)

    async def test_start_and_bus_do_not_wait_for_slow_canonical_io(self):
        blocked = asyncio.Event()
        self.sources.current_snapshot.side_effect = lambda *args, **kwargs: None

        async def slow(*args, **kwargs):
            await blocked.wait()
            return {"sources": []}

        self.sources.current_snapshot.side_effect = slow
        runtime = self.runtime(timeout_seconds=5)
        await runtime.start()
        await self.until(lambda: self.sources.current_snapshot.await_count > 0)
        self.assertEqual(runtime.service.snapshot().readiness, "warming")
        async with asyncio.timeout(0.2):
            result = await self.bus.publish(self.event())
            await runtime.close()
        self.assertEqual(result.delivered_count, 1)
        self.assertTrue(runtime.worker._task.done())
        self.assertFalse(blocked.is_set())

    async def test_polling_rereads_authorities_without_bus_delivery(self):
        runtime = self.runtime()
        await self.ready(runtime)
        self.originals.clear()
        self.db.tasks.clear()
        await self.until(lambda: (
            {entry.kind for entry in runtime.service.snapshot().entries} == {"focus"}
            and runtime.worker.refresh_count >= 2
        ))
        self.assertGreaterEqual(runtime.worker.refresh_count, 2)

    async def test_reuses_bound_payload_but_revalidates_actual_sources(self):
        runtime = self.runtime()
        await self.ready(runtime)
        count = runtime.binder.tasks.payload.await_count
        await self.until(lambda: runtime.worker.refresh_count >= 3)
        self.assertEqual(runtime.binder.tasks.payload.await_count, count)
        self.assertGreater(runtime.binder.context._records.await_count, 0)

    async def test_hint_filters_other_owner_and_session(self):
        runtime = self.runtime(interval_seconds=5)
        await self.ready(runtime)
        count = runtime.worker.refresh_count
        await self.bus.publish(self.event(owner="another-owner"))
        await self.bus.publish(self.event(session=uuid4()))
        await asyncio.sleep(0.01)
        self.assertEqual(runtime.worker.refresh_count, count)
        await self.bus.publish(self.event())
        await self.until(lambda: runtime.worker.refresh_count > count)

    async def test_output_guard_rejects_removed_original_after_binding(self):
        runtime = self.runtime(interval_seconds=5)
        await self.ready(runtime)
        selection = select_voice_context(runtime.service.snapshot(), now=self.now)
        binding = await runtime.bind(selection)
        self.originals.clear()
        with self.assertRaises(SourceError):
            async with runtime.guard(binding):
                self.fail("Removed original reached provider boundary")

    async def test_area_revision_change_latches_reconstruction(self):
        runtime = self.runtime(interval_seconds=5)
        await self.ready(runtime)
        binding = await runtime.bind(select_voice_context(runtime.service.snapshot(), now=self.now))
        self.selection["selection_revision"] = 1
        with self.assertRaises(SourceError):
            async with runtime.guard(binding):
                self.fail("Changed session selection reached output")
        self.selection["selection_revision"] = 0
        self.assertTrue(runtime.requires_reconstruction)
        with self.assertRaises(SourceError):
            await runtime.binder.validate_scope()

    async def test_transient_read_failure_withdraws_then_rereads_not_hint_replay(self):
        runtime = self.runtime()
        await self.ready(runtime)
        original_reader = self.context.snapshot.side_effect
        self.context.snapshot.side_effect = RuntimeError("offline read failure")
        await self.until(lambda: runtime.worker.failure_count > 0)
        await runtime.service.wait_idle()
        self.assertEqual(runtime.service.snapshot().entries, ())
        with self.assertRaises(SourceError):
            await runtime.bind(select_voice_context(runtime.service.snapshot(), now=self.now))
        self.context.snapshot.side_effect = original_reader
        await self.until(lambda: runtime.worker.last_error is None and bool(runtime.service.snapshot().entries))
        self.assertFalse(runtime.requires_reconstruction)

    async def test_scope_leak_and_oversized_original_are_not_projected(self):
        self.originals[self.source.source_id] = self.source.model_copy(update={"content": {"text": "x" * 2001}})
        self.task["session_id"] = uuid4()
        runtime = self.runtime()
        await self.ready(runtime)
        self.assertEqual({entry.kind for entry in runtime.service.snapshot().entries}, {"focus"})

    async def test_state_overflow_never_silently_reconstructs_epoch(self):
        runtime = self.runtime(queue_capacity=1)
        initial = runtime.service.snapshot().cursor.epoch
        await runtime.start()
        await self.until(lambda: runtime.requires_reconstruction)
        await runtime.close()
        self.assertEqual(runtime.service.snapshot().cursor.epoch, initial)
        self.assertNotEqual(runtime.service.snapshot().readiness, "ready")

    async def test_close_drains_and_unsubscribes_without_closing_borrowed_owners(self):
        runtime = self.runtime()
        await self.ready(runtime)
        subscription = runtime.service.subscribe()
        await runtime.close()
        await runtime.close()
        self.assertIsNone(await subscription.receive())
        self.assertEqual((await self.bus.publish(self.event())).delivered_count, 0)
        self.assertTrue(runtime.worker._task.done())
        self.assertTrue(runtime._monitor.done())

    async def test_absent_visual_evidence_keeps_context_runtime_healthy(self):
        working = SimpleNamespace(get_session_working_memory=AsyncMock(return_value=None))
        runtime = self.runtime(working_memory=working)
        await self.ready(runtime)
        self.assertIsNone(runtime.failure)
        self.assertFalse(any(entry.kind == "observation" for entry in runtime.service.snapshot().entries))
        self.assertIsNone(runtime.perception.worker.last_error)
        await runtime.close()
        self.assertTrue(runtime.perception.worker._task.done())
        self.assertTrue(runtime.worker._task.done())

    async def test_unexpected_perception_stop_is_reported_after_all_cleanup(self):
        working = SimpleNamespace(get_session_working_memory=AsyncMock(return_value=None))
        runtime = self.runtime(working_memory=working)
        await self.ready(runtime)
        runtime.perception.worker._task.cancel()
        await self.until(lambda: runtime.failure is not None)
        self.assertIsInstance(runtime.failure, RuntimeError)
        with self.assertRaises(ExceptionGroup) as raised:
            await runtime.close()
        self.assertTrue(any(isinstance(error, RuntimeError) for error in raised.exception.exceptions))
        self.assertTrue(runtime.perception.worker._task.done())
        self.assertTrue(runtime.worker._task.done())
        self.assertFalse(self.bus._handlers)
        self.assertEqual(runtime.service.snapshot().readiness, "degraded")
        self.runtimes.remove(runtime)


if __name__ == "__main__":
    unittest.main()
