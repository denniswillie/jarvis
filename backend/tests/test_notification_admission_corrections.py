"""Offline canonical correction proof and the shared owner-mutation handoff."""
import asyncio
from contextlib import asynccontextmanager
from datetime import UTC, datetime
import json
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from app.attention.notification_ledger.admission import TaskAdmissionAdapter, create_or_correct
from app.live_state.models import SessionScope
from app.runtime.feature_config import TaskLimits
from app.tasks.context_events import ContextRecord
from app.tasks.models import TaskCreate, TaskError, TaskStatus, TaskVoiceSource, digest
from app.tasks.repository import TaskRepository
from app.tasks.voice_tools import BackgroundTaskToolModule, StartArguments
from app.tools.invocation import CanonicalRequestIdentity, bind_tool_invocation
from app.tools.module import ToolBuildContext
from app.tools.registry import create_tool_registry


class CorrectionFixture:
    """Only storage is simulated; production admission/proof code executes."""
    _create_parameters = staticmethod(TaskRepository._create_parameters)
    _mutation_parameters = staticmethod(TaskRepository._mutation_parameters)

    def __init__(self):
        self.scope = SessionScope(owner_id="asr-fixture", session_id=uuid4())
        self.records, self.receipts, self.tasks, self.accepted = {}, {}, {}, {}
        self.repository, self.pool = self, None
        self.lock, self.before_mutate = asyncio.Lock(), None
        self.submitted = False
        self.sources = SimpleNamespace(get=self.get_source, _record=self.record)
        self.adapter = TaskAdmissionAdapter(self, sources=self.sources)
        self.args = StartArguments(title="Find the red book", requested_outcome="Research the red book.")
        self.first = self.source(1)

    def source(self, revision, **changes):
        now = datetime.now(UTC)
        for identity, old in tuple(self.records.items()):
            self.records[identity] = old.model_copy(update={"is_current": False})
        previous = max(self.records.values(), key=lambda record: record.revision, default=None)
        record = ContextRecord(id=uuid4(), owner_id=self.scope.owner_id, session_id=self.scope.session_id,
            sequence=revision, origin_sequence=1, source_namespace="voice:fixture", source_event_id="utterance-1",
            revision=revision, supersedes_event_id=previous.id if previous else None, kind="user_transcript",
            trust="user_input", finality="final" if revision == 1 else "revised", source_time=now,
            recorded_at=now, payload_id=uuid4(), availability="available", content_digest="a" * 64,
            content={"text": "Synthetic authorized text."}, is_current=True).model_copy(update=changes)
        self.records[record.id] = record
        return record

    async def record(self, connection, owner, event_id):
        record = self.records[event_id]
        if record.owner_id != owner:
            raise TaskError("context_event_not_found", "Wrong owner")
        return record

    async def get_source(self, owner, event_id, *, session_id):
        record = await self.record(self, owner, event_id)
        if record.session_id != session_id:
            raise TaskError("context_event_not_found", "Wrong session")
        return record

    @asynccontextmanager
    async def read_transaction(self):
        yield self

    async def fetchrow(self, query, *args):
        if "from jarvis_task_requests" in query:
            return self.receipts.get(args[1])
        if "from jarvis_task_events e" in query:
            return self.accepted[args[1]]
        raise AssertionError(query)

    async def fetchval(self, query, *args):
        if "lock_jarvis_copy_source" in query:
            return "available"
        if "from jarvis_task_actions" in query:
            return self.submitted
        raise AssertionError(query)

    async def load(self, connection, owner, task_id):
        return self.tasks[task_id]

    def save_receipt(self, request_id, task, operation, parameters):
        self.receipts[request_id] = {"task_id": task.id, "operation": operation,
            "parameters_digest": digest(parameters), "accepted_revision": task.revision}

    async def create(self, owner, request, *, source, mode):
        async with self.lock:
            params = self._create_parameters(request, source=source, mode=mode)
            receipt = self.receipts.get(request.request_id)
            if receipt:
                if receipt["parameters_digest"] != digest(params):
                    raise TaskError("task_request_conflict", "Different proposal")
                return self.tasks[receipt["task_id"]]
            task = SimpleNamespace(id=uuid4(), owner_id=owner, session_id=source.session_id,
                knowledge_area_id=None, revision=1, fence=1, title=request.title, mode=mode,
                requested_outcome=request.requested_outcome, status=TaskStatus.QUEUED, reason=None,
                pending_status=None, updated_at=datetime.now(UTC), budget=TaskLimits(),
                request_event_id=source.event_id, request_revision=source.revision)
            self.tasks[task.id] = task
            self.accepted[task.id] = {"id": uuid4(), "availability": "available", "content": json.dumps({
                "title": task.title, "requested_outcome": task.requested_outcome, "mode": mode,
                "instruction_request": source.parameters()})}
            self.save_receipt(request.request_id, task, "create", params)
            return task

    async def mutate(self, owner, task_id, operation, request, *, source, admission):
        async with self.lock:
            if self.before_mutate:
                self.before_mutate()
                self.before_mutate = None
            record = self.records[source.event_id]
            if not record.authorizes_request or record.revision != source.revision:
                raise TaskError("task_request_evidence_missing", "Source changed")
            params = self._mutation_parameters(request, source)
            receipt = self.receipts.get(request.request_id)
            task = self.tasks[task_id]
            if receipt:
                if receipt["parameters_digest"] != digest(params):
                    raise TaskError("task_request_conflict", "Different correction")
                return task
            if request.expected_revision != task.revision:
                raise TaskError("task_revision_conflict", "Task changed")
            await admission(self, task)
            task = SimpleNamespace(**(vars(task) | {"revision": task.revision + 1, "fence": task.fence + 1,
                "requested_outcome": request.message, "request_event_id": source.event_id,
                "request_revision": source.revision}))
            self.tasks[task_id] = task
            self.save_receipt(request.request_id, task, operation, params)
            return task

    async def admit(self, record, args=None):
        return await self.adapter.create(self.scope, record.id, record.revision, args or self.args)


class AdmissionCorrectionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.fixture = CorrectionFixture()
        self.original = await self.fixture.admit(self.fixture.first)
        self.new = self.fixture.source(2)
        self.args = self.fixture.args.model_copy(update={"title": "Find the blue book",
            "requested_outcome": "Research the blue book."})

    async def test_newer_canonical_transcript_updates_same_task_and_fence_once(self):
        f = self.fixture
        first, second = await asyncio.gather(f.admit(self.new, self.args), f.admit(self.new, self.args))
        self.assertEqual((first.id, second.id), (self.original.id, self.original.id))
        self.assertEqual((first.revision, second.revision, first.fence), (2, 2, 2))
        self.assertEqual(first.requested_outcome, self.args.requested_outcome)
        self.assertEqual(first.title, self.original.title)
        self.assertEqual(first.request_event_id, self.new.id)
        self.assertEqual(len(f.receipts), 2)
        self.assertEqual(len(f.tasks), 1)

    async def test_replay_after_restart_uses_receipt_not_an_in_memory_cache(self):
        f = self.fixture
        await f.admit(self.new, self.args)
        restarted = TaskAdmissionAdapter(f, sources=f.sources)
        replay = await restarted.create(f.scope, self.new.id, 2, self.args)
        self.assertEqual(replay.revision, 2)
        self.assertEqual(len(f.receipts), 2)

    async def test_native_voice_and_router_share_actual_correction_path(self):
        f = self.fixture
        f.reference = TaskRepository.reference
        with patch("app.tasks.voice_tools.ContextRepository", return_value=f.sources):
            registry = create_tool_registry(modules=(BackgroundTaskToolModule(f),),
                context=ToolBuildContext(None, f.scope.owner_id, f.scope.session_id))
        with bind_tool_invocation(provider_call_id="new-transport-call", request_event_id=self.new.id,
                request_revision=2, request_identity=CanonicalRequestIdentity.from_record(self.new)):
            voice = await registry.get("start_background_task").handler(self.args.model_dump(mode="json"))
        routed = await f.admit(self.new, self.args)
        self.assertEqual(voice["status"], "success", voice)
        self.assertEqual(voice["task"]["revision"], routed.revision)
        self.assertEqual(routed.revision, 2)

    async def test_same_revision_different_objective_conflicts(self):
        f = self.fixture
        await f.admit(self.new, self.args)
        with self.assertRaises(TaskError) as caught:
            await f.admit(self.new, self.args.model_copy(update={"requested_outcome": "Unrelated request"}))
        self.assertEqual(caught.exception.code, "task_request_conflict")
        self.assertEqual(f.tasks[self.original.id].revision, 2)

    async def test_incompatible_capabilities_mode_area_do_not_gain_authority(self):
        for change in ({"requested_capabilities": ["sandbox_network"]}, {"mode": "consult"},
                       {"knowledge_area_id": uuid4()}, {"requested_capabilities": ["memory_edit"]}):
            with self.subTest(change=change), self.assertRaises(TaskError):
                await self.fixture.admit(self.new, self.args.model_copy(update=change))
        self.assertEqual(self.fixture.tasks[self.original.id].revision, 1)

    async def test_provisional_stale_unavailable_or_unrelated_source_cannot_correct(self):
        f = self.fixture
        for change in ({"finality": "provisional"}, {"availability": "deleted", "content": None},
                       {"is_current": False}, {"origin_sequence": 99}):
            with self.subTest(change=change):
                f.records[self.new.id] = self.new.model_copy(update=change)
                with self.assertRaises(TaskError):
                    await f.admit(self.new, self.args)
        self.assertEqual(f.tasks[self.original.id].revision, 1)

    async def test_unrelated_current_task_source_is_not_overwritten(self):
        f = self.fixture
        foreign = f.source(3, source_event_id="other-utterance", origin_sequence=3)
        f.records[self.new.id] = self.new
        f.tasks[self.original.id].request_event_id = foreign.id
        f.tasks[self.original.id].request_revision = foreign.revision
        with self.assertRaises(TaskError):
            await f.admit(self.new, self.args)

    async def test_cancelled_task_is_not_silently_reopened(self):
        for status in (TaskStatus.CANCELLED, TaskStatus.CANCEL_REQUESTED):
            self.fixture.tasks[self.original.id].status = status
            with self.subTest(status=status), self.assertRaises(TaskError):
                await self.fixture.admit(self.new, self.args)

    async def test_withdrawal_rechecked_inside_mutation_not_only_read_snapshot(self):
        f = self.fixture
        f.before_mutate = lambda: f.accepted[self.original.id].update(availability="deleted", content=None)
        with self.assertRaises(TaskError):
            await f.admit(self.new, self.args)
        self.assertEqual(f.tasks[self.original.id].revision, 1)
        self.assertEqual(len(f.receipts), 1)

    async def test_revision_race_is_not_blindly_retried(self):
        f = self.fixture
        f.before_mutate = lambda: setattr(f.tasks[self.original.id], "revision", 7)
        with self.assertRaises(TaskError) as caught:
            await f.admit(self.new, self.args)
        self.assertEqual(caught.exception.code, "task_revision_conflict")
        self.assertEqual(len(f.receipts), 1)

    async def test_settled_submitted_action_requires_explicit_review(self):
        f = self.fixture
        f.submitted = True
        with self.assertRaises(TaskError) as caught:
            await f.admit(self.new, self.args)
        self.assertEqual(caught.exception.code, "task_reconciliation_required")
        self.assertEqual(f.tasks[self.original.id].revision, 1)

    async def test_non_conflict_service_failures_do_not_enter_correction(self):
        service = SimpleNamespace(repository=self.fixture,
            create=AsyncMock(side_effect=TaskError("task_worker_not_ready", "Unavailable")))
        request = TaskCreate(request_id=uuid4(), title="Fixture", requested_outcome="Fixture")
        with self.assertRaises(TaskError) as caught:
            await create_or_correct(service, self.fixture.scope.owner_id, request,
                source=TaskVoiceSource(self.new.id, 2, self.fixture.scope.session_id), mode="background")
        self.assertEqual(caught.exception.code, "task_worker_not_ready")
