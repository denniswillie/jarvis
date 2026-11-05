"""Real task repository/binder/runtime with bounded offline persistence.

The fixture implements only the SQL persistence boundary. No tool, model,
service, live database, or test-only canonical adapter is involved. This is not
PostgreSQL trigger/locking qualification; those remain database integration work.
"""
import asyncio
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
import json
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

from app.knowledge.models import SourceError
from app.knowledge.sources import SourceRepository
from app.live_state.canonical import CanonicalContextBinder
from app.live_state.models import LiveSnapshot, SessionScope, StateCursor
from app.live_state.projection import select_voice_context
from app.live_state.runtime import LiveStateRuntime
from app.live_state.task_context import TaskContextProjector, TASK_SNAPSHOT
from app.runtime.events import DomainEvent, DomainEventBus, DomainEventName, MemoryItemChangedPayload
from app.runtime.feature_config import TaskLimits
from app.tasks.context_events import ContextRecord
from app.tasks.models import TaskView
from app.tasks.repository import TaskRepository


NOW = datetime(2026, 10, 30, 23, tzinfo=UTC)


class Database:
    """Bounded SQL fixture: immutable source rows plus mutable canonical tasks."""
    def __init__(self, scope):
        self.scope = scope
        self.tasks, self.payloads, self.events, self.receipts = {}, {}, {}, {}
        self.parents, self.event_dependencies = {}, {}
        self.lock = asyncio.Lock()
        self.queries = []
        self.fail_ids = False
        self.block_ids = None
        self.ids_started = asyncio.Event()
        self.closed = False

    def payload(self, content, owner=None):
        if len(self.payloads) >= 256:
            raise AssertionError("Fixture payload budget exceeded")
        identity = uuid4()
        self.payloads[identity] = {"owner_id": owner or self.scope.owner_id,
            "content": content, "availability": "available", "created_at": NOW - timedelta(minutes=2)}
        return identity

    def task(self, *, status="running", answer=None, owner=None, session=None, area=None):
        owner = owner or self.scope.owner_id
        session = session or self.scope.session_id
        identity = uuid4()
        request_payload = self.payload({"text": "Calculate the reference value"}, owner)
        request = ContextRecord(id=uuid4(), owner_id=owner, session_id=session,
            sequence=1, origin_sequence=1, source_namespace="voice:test", source_event_id=str(identity),
            revision=1, supersedes_event_id=None, kind="user_transcript", trust="user_input",
            finality="final", source_time=NOW - timedelta(hours=1), recorded_at=NOW - timedelta(hours=1),
            payload_id=request_payload, availability="available", content_digest="a" * 64,
            content={"text": "Calculate the reference value"}, is_current=True)
        self.events[request.id] = request.model_dump()
        description = self.payload({"title": "Reference calculation", "requested_outcome": "Calculate it",
                                    "mode": "background"}, owner)
        task = TaskView(id=identity, owner_id=owner, session_id=session, knowledge_area_id=area,
            backend_kind="frontier", model="offline-fixture", status=status, revision=3,
            context_watermark=1, title="Reference calculation", mode="background", requested_outcome="Calculate it",
            description_payload_id=description, request_event_id=request.id, request_revision=1,
            current_attempt_id=None, fence=4, attempt_count=2, active_seconds=1,
            budget=TaskLimits(), result_payload_id=None, reason=None, pending_status=None,
            created_at=NOW - timedelta(hours=1), updated_at=NOW - timedelta(minutes=2))
        self.tasks[identity] = task.model_dump() | {"last_event_sequence": 8}
        if answer is not None:
            result = self.payload({"answer": answer}, owner)
            manifest = self.payload({"request": "Canonical worker context"}, owner)
            self.tasks[identity]["result_payload_id"] = result
            self.receipts[identity] = {
                "event_id": uuid4(), "result_recorded_at": NOW - timedelta(minutes=3),
                "result_attempt_id": uuid4(), "manifest_id": uuid4(), "manifest_status": "active",
                "manifest_payload_id": manifest, "task_id": identity, "revision": 3,
                "event_type": status, "payload_id": result,
            }
        return identity

    def available(self, owner, identity, seen=None):
        seen = set() if seen is None else seen
        if identity in seen:
            return True
        seen.add(identity)
        row = self.payloads.get(identity)
        if not row or row["owner_id"] != owner or row["availability"] != "available":
            return False
        for event_id, revision in self.event_dependencies.get(identity, ()):
            event = self.events[event_id]
            if not event["is_current"] or event["revision"] != revision:
                return False
        return all(self.available(owner, parent, seen) for parent in self.parents.get(identity, ()))

    @asynccontextmanager
    async def acquire(self):
        yield Connection(self)

    def remove(self, identity):
        self.payloads[identity].update(availability="deleted", content=None)


class Connection:
    def __init__(self, db):
        self.db = db
        self.active = False

    def is_in_transaction(self):
        return self.active

    @asynccontextmanager
    async def transaction(self, **kwargs):
        async with self.db.lock:
            self.active = True
            try:
                yield self
            finally:
                self.active = False

    def query(self, sql, args):
        normalized = " ".join(sql.lower().split())
        self.db.queries.append((normalized, args))
        return normalized

    async def execute(self, sql, *args):
        sql = self.query(sql, args)
        if "pg_advisory_xact_lock" in sql:
            assert self.active
            return "SELECT 1"
        if "insert into jarvis_context_payload_dependencies" in sql:
            owner, parent, child = args
            assert self.active and self.db.available(owner, parent)
            self.db.parents.setdefault(child, set()).add(parent)
            return "INSERT 1"
        if "insert into jarvis_context_event_dependencies" in sql:
            owner, child, event, revision, original, usage = args
            assert usage == "current" and self.db.events[event]["payload_id"] == original
            self.db.event_dependencies.setdefault(child, []).append((event, revision))
            return "INSERT 1"
        raise AssertionError(sql)

    async def executemany(self, sql, args):
        for row in args:
            await self.execute(sql, *row)

    async def fetchval(self, sql, *args):
        sql = self.query(sql, args)
        if "pg_try_advisory_xact_lock" in sql:
            assert self.active and args[0] == self.db.scope.owner_id
            return True  # Controlled result; real lock behavior has SQL coverage.
        if "insert into jarvis_context_payloads" in sql:
            assert self.active
            return self.db.payload(json.loads(args[1]), args[0])
        if "lock_jarvis_copy_sources" in sql:
            assert self.active
            return all(self.db.available(args[0], identity) for identity in args[1])
        if "from jarvis_sessions" in sql:
            return (args[0], args[1]) == (self.db.scope.owner_id, self.db.scope.session_id)
        if "from jarvis_context_streams" in sql:
            return 1
        if "from jarvis_knowledge_areas" in sql:
            return True if "status='active'" in sql else "active"
        raise AssertionError(sql)

    async def fetchrow(self, sql, *args):
        sql = self.query(sql, args)
        if sql.startswith("select t.*,p.content as description"):
            task = self.db.tasks.get(args[1])
            if task is None or task["owner_id"] != args[0]:
                return None
            description = self.db.payloads[task["description_payload_id"]]["content"]
            return dict(task) | {"description": description}
        if sql.startswith("select t.last_event_sequence,"):
            # Model the scope/revision predicates of the real SQL, rather than
            # hand a pre-approved answer to the binder.
            for predicate in ("a.task_revision=e.task_revision", "m.task_revision=a.task_revision",
                              "m.task_id=a.task_id", "e.task_revision=t.revision",
                              "e.payload_id=t.result_payload_id", "e.event_type=t.status",
                              "receipt.manifest_status='active'", "p.availability='available'"):
                assert predicate in sql
            task = self.db.tasks[args[1]]
            result = {name: None for name in ("event_id", "result_recorded_at", "result_attempt_id",
                "manifest_id", "manifest_status", "manifest_payload_id", "result")}
            result["last_event_sequence"] = task["last_event_sequence"]
            receipt = self.db.receipts.get(args[1])
            if receipt and (receipt["task_id"], receipt["revision"], receipt["event_type"], receipt["payload_id"]) == (
                    task["id"], task["revision"], task["status"], task["result_payload_id"]):
                result.update({name: receipt[name] for name in result if name in receipt})
                payload = self.db.payloads.get(task["result_payload_id"])
                if receipt["manifest_status"] == "active" and payload["availability"] == "available":
                    result["result"] = payload["content"]
            return result
        if sql.startswith("select e.id,p.availability,e.finality,e.trust,e.kind"):
            assert "newer.revision>e.revision" in sql
            event = self.db.events.get(args[1])
            if not event or (event["owner_id"], event["revision"], event["is_current"]) != (args[0], args[2], True):
                return None
            return dict(event) | {"availability": self.db.payloads[event["payload_id"]]["availability"]}
        if sql.startswith("select e.*,p.availability,p.content_digest,p.content"):
            event = self.db.events.get(args[1])
            if event is None or event["owner_id"] != args[0]:
                return None
            payload = self.db.payloads[event["payload_id"]]
            return dict(event) | {"availability": payload["availability"], "content": payload["content"]}
        if sql.startswith("select content from jarvis_context_payloads"):
            payload = self.db.payloads.get(args[1])
            if payload and payload["owner_id"] == args[0] and payload["availability"] == "available":
                return {"content": payload["content"]}
            return None
        if sql.startswith("select m.status from jarvis_task_attempts"):
            return {"status": "active"}
        raise AssertionError(sql)

    async def fetch(self, sql, *args):
        sql = self.query(sql, args)
        if "select grant_id from jarvis_task_grants" in sql:
            return []
        if sql.startswith("select id from jarvis_background_tasks"):
            assert "session_id=$2" in sql and "knowledge_area_id is not distinct from $3::uuid" in sql
            self.db.ids_started.set()
            if self.db.fail_ids:
                raise RuntimeError("Synthetic database read outage")
            if self.db.block_ids is not None:
                await self.db.block_ids.wait()
            tasks = [task for task in self.db.tasks.values() if (
                task["owner_id"], task["session_id"], task["knowledge_area_id"]) == args[:3]]
            tasks.sort(key=lambda task: (task["status"] in {"succeeded", "failed", "cancelled"},
                                         -task["updated_at"].timestamp(), task["id"]))
            return [{"id": task["id"]} for task in tasks[:args[3]]]
        raise AssertionError(sql)


class TaskContextTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.scope = SessionScope(owner_id="task-context-fixture", session_id=uuid4())
        self.now = NOW
        self.db = Database(self.scope)
        self.sources = SourceRepository(self.db)
        self.repository = TaskRepository(self.db, TaskLimits())
        self.binder = CanonicalContextBinder(self.sources, self.scope, clock=lambda: self.now)
        self.projector = TaskContextProjector(self.binder, self.repository)

    async def project(self, ids=None, projector=None):
        projector = projector or self.projector
        return await projector.project(await projector.read_ids() if ids is None else ids,
            received_at=self.now, expires_at=self.now + timedelta(seconds=10))

    def selection(self, entries):
        return select_voice_context(LiveSnapshot(scope=self.scope,
            cursor=StateCursor(epoch=uuid4(), revision=1), as_of=self.now,
            readiness="ready", entries=entries), now=self.now)

    async def test_real_repository_and_binder_publish_all_current_statuses(self):
        identity = self.db.task()
        for status in ("queued", "running", "waiting_for_input", "paused", "succeeded", "failed",
                       "cancel_requested", "cancelled", "outcome_unknown"):
            with self.subTest(status=status):
                self.db.tasks[identity]["status"] = status
                entries = await self.project()
                self.assertEqual(entries[0].status, status)
                self.assertEqual(entries[0].task.revision, 3)
                self.assertEqual(entries[0].sources[0].revision, 8)
                self.assertIsNone(entries[0].sources[0].generation)
                self.assertIn(status, self.selection(entries).text)
                self.assertTrue(await self.binder.available(entries[0].guard_binding))

    async def test_worker_text_and_original_times_reach_real_combined_voice_copy(self):
        answer = "Use (low + high) // 2.\nThe value is 14.25, not 1425."
        identity = self.db.task(status="succeeded", answer=answer)
        entries = await self.project()
        entry = entries[0]
        self.assertEqual(entry.observed_at, self.db.tasks[identity]["updated_at"])
        self.assertNotEqual(entry.observed_at, entry.received_at)
        self.assertIn(json.dumps(answer), entry.summary)
        self.assertIn(self.db.receipts[identity]["result_recorded_at"].isoformat(), entry.summary)
        self.assertIn("delivery/heard state is unknown", entry.summary)
        selection = self.selection(entries)
        self.assertEqual(selection.selected_entry_ids, (entry.entry_id,))
        combined = await self.binder(selection)
        self.assertTrue(await self.binder.available(combined))
        root = entry.guard_binding.payload_root_ids[0]
        task = self.db.tasks[identity]
        request = self.db.events[task["request_event_id"]]
        self.assertEqual(self.db.parents[root], {task["description_payload_id"], request["payload_id"],
            task["result_payload_id"], self.db.receipts[identity]["manifest_payload_id"]})
        self.assertEqual(self.db.parents[combined.payload_root_ids[0]], {root})
        self.assertEqual(self.db.event_dependencies[root], [(request["id"], request["revision"])])
        self.assertFalse(any(sql.startswith("update jarvis_background_tasks") for sql, _ in self.db.queries))

    async def test_waiting_question_is_exact_and_not_a_completion_or_action(self):
        self.db.task(status="waiting_for_input", answer="Which account: A or B?")
        entry, = await self.project()
        self.assertIn('Pending question', entry.summary)
        self.assertIn('"Which account: A or B?"', entry.summary)
        self.assertEqual(entry.status, "waiting_for_input")
        self.assertIn("not authorization", entry.summary)

    async def test_guard_rechecks_status_attempt_and_fence_without_intent_or_event_change(self):
        identity = self.db.task()
        for field, value in (("status", "paused"), ("current_attempt_id", uuid4()), ("fence", 9),
                             ("updated_at", NOW - timedelta(seconds=30)), ("pending_status", "cancelled"),
                             ("result_payload_id", uuid4()), ("revision", 4)):
            with self.subTest(field=field):
                entry, = await self.project()
                binding = await self.binder(self.selection((entry,)))
                previous = self.db.tasks[identity][field]
                self.db.tasks[identity][field] = value
                self.assertFalse(await self.binder.available(entry.guard_binding))
                self.assertFalse(await self.binder.available(binding))
                self.db.tasks[identity][field] = previous

    async def test_same_intent_new_state_uses_real_sequence_and_new_source_not_fake_revision(self):
        identity = self.db.task()
        old, = await self.project()
        self.db.tasks[identity].update(status="paused", last_event_sequence=9)
        new, = await self.project()
        self.assertEqual((old.task.revision, new.task.revision), (3, 3))
        self.assertEqual((old.sources[0].revision, new.sources[0].revision), (8, 9))
        self.assertNotEqual(old.sources[0].source_id, new.sources[0].source_id)
        self.assertFalse(await self.binder.available(old.guard_binding))

    async def test_unchanged_cache_revalidates_without_renewing_observed_or_snapshot_time(self):
        identity = self.db.task(status="succeeded", answer="42")
        first, = await self.project()
        count = len(self.db.payloads)
        self.now += timedelta(seconds=30)
        second, = await self.project()
        self.assertEqual(first.summary, second.summary)
        self.assertEqual(first.observed_at, second.observed_at)
        self.assertEqual(first.guard_binding, second.guard_binding)
        self.assertEqual(len(self.db.payloads), count)
        self.assertGreater(second.fresh_until, first.fresh_until)
        self.db.tasks[identity]["status"] = "paused"
        self.assertFalse(await self.binder.available(second.guard_binding))

    async def test_removed_originals_revoke_combined_copy_and_do_not_republish_answer(self):
        for original in ("request", "description", "result", "manifest"):
            with self.subTest(original=original):
                identity = self.db.task(status="succeeded", answer="Sensitive original")
                entry, = await self.project((identity,))
                combined = await self.binder(self.selection((entry,)))
                task = self.db.tasks[identity]
                root = {"request": self.db.events[task["request_event_id"]]["payload_id"],
                    "description": task["description_payload_id"], "result": task["result_payload_id"],
                    "manifest": self.db.receipts[identity]["manifest_payload_id"]}[original]
                self.db.remove(root)
                self.assertFalse(await self.binder.available(combined))
                entries = await self.project((identity,))
                self.assertFalse(any("Sensitive original" in item.summary for item in entries))

    async def test_request_correction_blocks_old_status_and_result(self):
        identity = self.db.task(status="succeeded", answer="Old instruction result")
        entry, = await self.project()
        self.db.events[self.db.tasks[identity]["request_event_id"]]["is_current"] = False
        self.assertFalse(await self.binder.available(entry.guard_binding))
        self.assertEqual(await self.project(), ())

    async def test_manifest_invalidation_revokes_previous_answer_but_can_report_status_only(self):
        identity = self.db.task(status="succeeded", answer="Do not resurrect")
        entry, = await self.project()
        self.db.receipts[identity]["manifest_status"] = "revoked"
        self.assertFalse(await self.binder.available(entry.guard_binding))
        current, = await self.project()
        self.assertEqual(current.status, "succeeded")
        self.assertIn("unavailable", current.summary)
        self.assertNotIn("Do not resurrect", current.summary)

    async def test_missing_stale_or_wrong_task_result_receipt_never_approves_worker_text(self):
        for change in ("missing", "revision", "task_id", "event_type"):
            with self.subTest(change=change):
                identity = self.db.task(status="succeeded", answer="Unproved answer")
                if change == "missing":
                    del self.db.receipts[identity]
                else:
                    self.db.receipts[identity][change] = {"revision": 2, "task_id": uuid4(),
                                                         "event_type": "stale_attempt_outcome"}[change]
                entry, = await self.project((identity,))
                self.assertNotIn("Unproved answer", entry.summary)
                self.assertIn("unavailable", entry.summary)

    async def test_scope_filter_applies_before_limit_and_stale_cross_scope_candidates_are_rejected(self):
        owned = self.db.task()
        others = [self.db.task(owner="other-owner"), self.db.task(session=uuid4()), self.db.task(area=uuid4())]
        projector = TaskContextProjector(self.binder, self.repository, limit=1)
        self.assertEqual(await projector.read_ids(), (owned,))
        for identity in others:
            self.assertEqual(await self.project((identity,), projector), ())
        entry, = await self.project((owned,))
        other = CanonicalContextBinder(self.sources, self.scope.model_copy(update={"session_id": uuid4()}),
                                       clock=lambda: self.now)
        with self.assertRaises(SourceError):
            async with other.guard(entry.guard_binding):
                self.fail("Cross-session binding admitted")

    async def test_oversized_answer_is_omitted_whole_without_truncating_facts(self):
        self.db.task(status="succeeded", answer="Exact answer: " + "1234567890" * 300)
        entry, = await self.project()
        self.assertLessEqual(len(entry.summary), 2000)
        self.assertNotIn("Exact answer:", entry.summary)
        self.assertIn("omitted", entry.summary)

    async def test_candidate_read_is_not_a_status_snapshot(self):
        identity = self.db.task()
        ids = await self.projector.read_ids()
        self.db.tasks[identity].update(status="cancel_requested", pending_status="cancelled", fence=7)
        entry, = await self.project(ids)
        self.assertEqual(entry.status, "cancel_requested")
        self.assertIn("cancelled; not confirmed", entry.summary)
        self.assertIn("fence=7", entry.summary)

    async def test_bounds_cache_eviction_missing_task_and_clear(self):
        identities = tuple(self.db.task() for _ in range(4))
        projector = TaskContextProjector(self.binder, self.repository, limit=2)
        self.assertEqual(len(await self.project(identities, projector)), 2)
        self.assertEqual(len(projector._cache), 2)
        self.assertEqual(len(await self.project(identities[2:], projector)), 2)
        self.assertEqual(set(projector._cache), set(identities[2:]))
        del self.db.tasks[identities[2]]
        self.assertEqual(await self.project((identities[2],), projector), ())
        self.assertFalse(projector._cache)
        await self.project((identities[3],), projector)
        projector.clear()
        self.assertFalse(projector._cache)
        for limit in (0, 17, True):
            with self.assertRaises(ValueError):
                TaskContextProjector(self.binder, self.repository, limit=limit)

    async def test_future_times_and_uncommitted_sequence_are_not_projected(self):
        identity = self.db.task()
        self.db.tasks[identity]["updated_at"] = NOW + timedelta(seconds=1)
        self.assertEqual(await self.project(), ())
        self.db.tasks[identity].update(updated_at=NOW, last_event_sequence=0)
        self.assertEqual(await self.project(), ())

    async def test_forged_generation_sequence_and_format_are_rejected(self):
        self.db.task()
        entry, = await self.project()
        dependency = entry.sources[0]
        for changes in ({"generation": 1}, {"revision": 999}):
            binding = entry.guard_binding.model_copy(update={
                "dependencies": (dependency.model_copy(update=changes),)})
            self.assertFalse(await self.binder.available(binding))
        self.db.payloads[dependency.source_id]["content"]["format"] = "not-a-task-snapshot"
        self.assertFalse(await self.binder.available(entry.guard_binding))

    async def test_task_snapshot_source_is_not_confused_with_intent_revision(self):
        self.db.task()
        entry, = await self.project()
        self.assertEqual(entry.sources[0].source_namespace, TASK_SNAPSHOT)
        self.assertEqual(entry.sources[0].source_id, entry.guard_binding.payload_root_ids[0])
        self.assertNotEqual(entry.sources[0].source_id, entry.task.aggregate_id)


class TaskRuntimeTests(TaskContextTests):
    # Only persistence input for the unrelated conversation window is stubbed;
    # task repository, sources, binder, projector, producer, reducer and refresh
    # lifecycle are the actual production implementations.
    async def runtime(self, **options):
        runtime = LiveStateRuntime(scope=self.scope, sources=self.sources,
            context=SimpleNamespace(snapshot=AsyncMock(return_value={"recent": [], "pinned": []})),
            area_selections=None, bus=DomainEventBus(), access_check=lambda owner: None,
            tasks=self.repository, clock=lambda: self.now, interval_seconds=0.01,
            timeout_seconds=0.2, **options)
        self.addAsyncCleanup(runtime.close)
        await runtime.start()
        return runtime

    async def until(self, predicate):
        async with asyncio.timeout(2):
            while not predicate():
                await asyncio.sleep(0.002)

    async def test_runtime_polls_real_status_and_result_without_bus_or_fast_lane_io(self):
        identity = self.db.task(status="waiting_for_input", answer="Which version?")
        runtime = await self.runtime()
        await self.until(lambda: runtime.service.snapshot().readiness == "ready")
        count = len(self.db.queries)
        selection = select_voice_context(runtime.service.snapshot(), now=self.now)
        self.assertIn("Which version?", selection.text)
        self.assertEqual(len(self.db.queries), count)
        old = await runtime.bind(selection)
        self.db.tasks[identity].update(status="running", current_attempt_id=uuid4(), fence=5,
                                       result_payload_id=None, last_event_sequence=9)
        with self.assertRaises(SourceError):
            async with runtime.guard(old):
                self.fail("Stale question was admitted")
        await self.until(lambda: any(entry.status == "running" for entry in runtime.service.snapshot().entries))
        self.assertNotIn("Which version?", select_voice_context(runtime.service.snapshot(), now=self.now).text)
        self.assertGreaterEqual(runtime.worker.refresh_count, 2)

    async def test_runtime_failure_withdrawal_poll_recovery_and_owned_cache_cleanup(self):
        self.db.task()
        runtime = await self.runtime()
        await self.until(lambda: runtime.service.snapshot().readiness == "ready")
        self.db.fail_ids = True
        await self.until(lambda: runtime.worker.failure_count > 0)
        await runtime.service.wait_idle()
        self.assertFalse(runtime.service.snapshot().entries)
        self.assertFalse(runtime.task_context._cache)
        self.db.fail_ids = False
        await self.until(lambda: runtime.worker.last_error is None and bool(runtime.service.snapshot().entries))
        await runtime.close()
        self.assertTrue(runtime.worker._task.done())
        self.assertFalse(runtime.task_context._cache)
        self.assertFalse(runtime.worker.bus._handlers)
        self.assertFalse(self.db.closed)

    async def test_start_hint_and_close_do_not_wait_for_blocked_task_storage(self):
        self.db.task()
        self.db.block_ids = asyncio.Event()
        runtime = await self.runtime()
        await asyncio.wait_for(self.db.ids_started.wait(), 1)
        self.assertEqual(runtime.service.snapshot().readiness, "warming")
        event = DomainEvent(DomainEventName.MEMORY_ITEM_CHANGED, MemoryItemChangedPayload(
            owner_id=self.scope.owner_id, memory_item_id=uuid4(), change_type="changed",
            session_id=self.scope.session_id))
        async with asyncio.timeout(0.2):
            await runtime.worker.bus.publish(event)
            await runtime.close()
        self.assertFalse(self.db.block_ids.is_set())
        self.assertTrue(runtime.worker._task.done())


if __name__ == "__main__":
    unittest.main()
