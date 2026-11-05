"""Opt-in isolated SQL qualification of committed task-control refresh proof.

Run via evals.second_brain.run_tests --live-db --manifest-dir PRIVATE_DIR
--pattern test_task_control_context_sql.py. No worker, provider, storage or bank.
Negative query probes substitute arguments, not canonical SQL business state.
"""

from contextlib import AsyncExitStack
from dataclasses import replace
from datetime import UTC, datetime, timedelta
import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from uuid import uuid4

from app.knowledge.models import SourceError
from app.live_state.canonical import CanonicalContextBinder
from app.live_state.models import SessionScope
from app.live_state.task_context import (
    TaskContextProjector, TaskSnapshotControlChange, _committed_task_control, _read,
)
from app.runtime.feature_config import TaskLimits
from app.sessions.repository import SessionRepository
from app.tasks.context_events import ContextIngress
from app.tasks.context_repository import ContextRepository
from app.tasks.models import TaskCreate, TaskError, TaskMutation, TaskStatus, TaskVoiceSource
from app.tasks.repository import TaskRepository
from evals.second_brain.source_fixture import source_fixture


class ProofQuery:
    """Observe execution of the production SQL, never replace its result."""

    def __init__(self, connection):
        self.connection = connection
        self.calls = 0

    async def fetchval(self, query, *arguments):
        if "/* task_control_refresh */" not in query:
            raise AssertionError("Expected the actual committed-control proof query")
        self.calls += 1
        return await self.connection.fetchval(query, *arguments)


@unittest.skipUnless(os.environ.get("JARVIS_SECOND_BRAIN_TEST_DB_URL"),
                     "requires isolated --live-db wrapper")
class TaskControlContextSQLTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.directory = Path(os.environ["JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"]) / self._testMethodName
        self.evidence = dict(test=self.id(), classification="real_SQL_no_worker_or_provider",
                             storage_used=False, hindsight_bank_used=False, checks=[])
        self.addAsyncCleanup(self.write_evidence)
        self.stack = AsyncExitStack()
        self.addAsyncCleanup(self.stack.aclose)
        self.fixture = await self.stack.enter_async_context(source_fixture(
            os.environ["JARVIS_SECOND_BRAIN_TEST_DB_URL"], self.directory, test_name=self.id()))
        self.owner = self.fixture.owner
        self.repo = TaskRepository(self.fixture.pool, TaskLimits())
        self.context = ContextRepository(self.repo)
        self.session = await SessionRepository(self.fixture.pool).create_session(self.owner, "Task control SQL")
        self.other_session = await SessionRepository(self.fixture.pool).create_session(
            self.fixture.other, "Foreign fixture session")
        self.scope = SessionScope(owner_id=self.owner, session_id=self.session.id, area_id=self.fixture.area)
        # SQL network time is not the application freshness/deadline oracle here.
        self.binder = CanonicalContextBinder(self.fixture.sources, self.scope,
            clock=lambda: datetime.now(UTC), timeout_seconds=30)
        self.projector = TaskContextProjector(self.binder, self.repo)
        self.evidence.update(owner_id=self.owner, other_owner_id=self.fixture.other,
            schema=self.fixture.fixture.manifest["schemas"][0], manifest=str(self.fixture.fixture.path))

    async def write_evidence(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        fixture = getattr(self, "fixture", None)
        if fixture is not None:
            self.evidence.update(cleanup=fixture.fixture.manifest.get("cleanup"),
                                 fixture_status=fixture.fixture.manifest["status"])
        with (self.directory / ("checks-" + uuid4().hex + ".json")).open("x") as output:
            json.dump(self.evidence, output, indent=2, default=str)

    def plan(self, *names):
        self.evidence["checks"] = [dict(name=name, status="unreached") for name in names]

    def check(self, name, condition, **detail):
        row = next(row for row in self.evidence["checks"] if row["name"] == name)
        row.update(status="passed" if condition else "failed", **detail)
        self.assertTrue(condition, name)

    async def source(self, text, *, owner=None, session=None, key=None,
                     expected_revision=0, finality=None):
        return await self.context.append(owner or self.owner, ContextIngress(
            session_id=session or self.session.id, source_namespace="fixture:task-control-sql",
            source_event_id=key or uuid4().hex, receipt_id=uuid4().hex,
            kind="user_transcript", trust="user_input",
            finality=finality or ("revised" if expected_revision else "final"),
            payload={"text": text}, expected_revision=expected_revision))

    @staticmethod
    def voice(record):
        return TaskVoiceSource(record.id, record.revision, record.session_id)

    async def create(self):
        source = await self.source("Create a reasoning-only comparison of three desk layouts.")
        return await self.repo.create(self.owner, TaskCreate(request_id=uuid4(),
            title="Fixture desk comparison", requested_outcome="Compare three layouts; no external actions.",
            session_id=self.session.id, knowledge_area_id=self.fixture.area),
            model="controlled-no-provider", source=self.voice(source), mode="background")

    async def snapshot(self, task):
        now = datetime.now(UTC)
        entries = await self.projector.project((task.id,), received_at=now, expires_at=now + timedelta(minutes=5))
        self.assertEqual(len(entries), 1, "A real current canonical projection is required")
        async with self.binder._transaction() as connection:
            snapshot = await _read(self.binder, connection, task.id)
        return entries[0], snapshot

    async def proof(self, old_entry, old, current, *, binder=None):
        async with self.fixture.pool.acquire() as connection:
            query = ProofQuery(connection)
            result = await _committed_task_control(binder or self.binder, query,
                old.state, current, old_entry.sources[0])
            self.assertEqual(query.calls, 1, "Negative must execute SQL, not only fail a Python precondition")
        return result

    async def guard_error(self, entry):
        with self.assertRaises(SourceError) as caught:
            async with self.binder.guard(entry.guard_binding):
                self.fail("An obsolete projection must never authorize output")
        return caught.exception

    async def test_committed_cancel_and_update_link_exact_finalized_source(self):
        names = ("proof", "stale_rejected_as_control", "fresh_authorized", "deduplicated", "linkage")
        self.plan(*(operation + "_" + name for operation in ("cancel", "update") for name in names))
        for operation in ("cancel", "update"):
            with self.subTest(operation=operation):
                task = await self.create()
                old_entry, old = await self.snapshot(task)
                source = await self.source("Cancel the task." if operation == "cancel"
                                           else "Replace the task objective: compare two desk layouts only.")
                request = TaskMutation(request_id=uuid4(), expected_revision=task.revision,
                    message="Cancel the task." if operation == "cancel" else "Compare two layouts; no external actions.")
                current = await self.repo.mutate(self.owner, task.id, operation, request, source=self.voice(source))
                fresh_entry, fresh = await self.snapshot(current)
                self.check(operation + "_proof", await self.proof(old_entry, old, fresh),
                           task_id=str(task.id), request_id=str(request.request_id), source_id=str(source.id))
                error = await self.guard_error(old_entry)
                self.check(operation + "_stale_rejected_as_control", isinstance(error, TaskSnapshotControlChange))
                async with self.binder.guard(fresh_entry.guard_binding):
                    self.check(operation + "_fresh_authorized", current.status == (
                        TaskStatus.CANCELLED if operation == "cancel" else TaskStatus.QUEUED)
                        and current.request_event_id == source.id and current.request_revision == source.revision)
                replay = await self.repo.mutate(self.owner, task.id, operation, request, source=self.voice(source))
                async with self.fixture.pool.acquire() as connection:
                    count = await connection.fetchval("select count(*) from jarvis_task_requests where owner_id=$1 and request_id=$2",
                                                      self.owner, request.request_id)
                    events = await connection.fetchval("select count(*) from jarvis_task_events where owner_id=$1 and task_id=$2 and event_type=$3",
                                                       self.owner, task.id, operation + "_requested")
                    linked = await connection.fetchval("""select count(*) from jarvis_task_events e
                        join jarvis_context_payload_dependencies d on d.owner_id=e.owner_id and d.dependent_payload_id=e.payload_id
                        where e.owner_id=$1 and e.task_id=$2 and e.task_revision=$3
                            and e.event_type=$4 and d.source_payload_id=$5""", self.owner, task.id,
                        current.revision, operation + "_requested", source.payload_id)
                self.check(operation + "_deduplicated", replay.revision == current.revision and count == events == 1)
                self.check(operation + "_linkage", linked == 1 and current.context_watermark == source.sequence
                           and current.fence > task.fence)

    async def test_exact_proof_query_rejects_foreign_source_owner_version_and_resume(self):
        self.plan("baseline_control", "wrong_owner", "wrong_source", "wrong_version", "resume_not_control", "resume_stale_hard_fence")
        task = await self.create()
        old_entry, old = await self.snapshot(task)
        source = await self.source("Cancel this task.")
        cancelled = await self.repo.mutate(self.owner, task.id, "cancel",
            TaskMutation(request_id=uuid4(), expected_revision=task.revision), source=self.voice(source))
        cancelled_entry, fresh = await self.snapshot(cancelled)
        self.check("baseline_control", await self.proof(old_entry, old, fresh))
        other_binder = SimpleNamespace(scope=SimpleNamespace(owner_id=self.fixture.other))
        self.check("wrong_owner", not await self.proof(old_entry, old, fresh, binder=other_binder))
        unrelated = await self.source("A separate finalized instruction, not this cancellation.")
        wrong_source = replace(fresh, request=SimpleNamespace(sequence=fresh.request.sequence, payload_id=unrelated.payload_id))
        self.check("wrong_source", not await self.proof(old_entry, old, wrong_source))
        wrong_version = replace(fresh, task=fresh.task.model_copy(update={"revision": fresh.task.revision + 1}))
        self.check("wrong_version", not await self.proof(old_entry, old, wrong_version))
        resume_source = await self.source("Explicitly resume the cancelled reasoning-only task.")
        resumed = await self.repo.mutate(self.owner, task.id, "resume",
            TaskMutation(request_id=uuid4(), expected_revision=cancelled.revision), source=self.voice(resume_source))
        _, resumed_snapshot = await self.snapshot(resumed)
        self.check("resume_not_control", not await self.proof(cancelled_entry, fresh, resumed_snapshot))
        error = await self.guard_error(cancelled_entry)
        self.check("resume_stale_hard_fence", not isinstance(error, TaskSnapshotControlChange)
                   and error.code == "live_task_snapshot_changed", code=error.code)

    async def test_source_authorization_and_later_correction_remain_hard_fences(self):
        self.plan("foreign_source", "wrong_source_revision", "pending_source", "rejections_do_not_mutate",
                  "valid_update", "source_correction_hard_fence")
        task = await self.create()
        old_entry, _ = await self.snapshot(task)
        foreign = await self.source("Change that task.", owner=self.fixture.other, session=self.other_session.id)
        valid_key = uuid4().hex
        valid = await self.source("Compare two desk layouts only.", key=valid_key)
        pending = await self.source("Unfinished replacement instruction", finality="provisional")
        for name, source in (("foreign_source", self.voice(foreign)),
                             ("wrong_source_revision", TaskVoiceSource(valid.id, valid.revision + 1, valid.session_id)),
                             ("pending_source", self.voice(pending))):
            with self.assertRaises(TaskError) as caught:
                await self.repo.mutate(self.owner, task.id, "update",
                    TaskMutation(request_id=uuid4(), expected_revision=task.revision, message="Compare two layouts."),
                    source=source)
            self.check(name, caught.exception.code == "task_request_evidence_missing", code=caught.exception.code)
        same = await self.repo.get(self.owner, task.id)
        async with self.fixture.pool.acquire() as connection:
            count = await connection.fetchval("select count(*) from jarvis_task_requests where owner_id=$1 and task_id=$2",
                                               self.owner, task.id)
        self.check("rejections_do_not_mutate", same.revision == task.revision and same.fence == task.fence
                   and same.request_event_id == task.request_event_id and count == 1)
        updated = await self.repo.mutate(self.owner, task.id, "update",
            TaskMutation(request_id=uuid4(), expected_revision=task.revision, message="Compare two layouts."),
            source=self.voice(valid))
        fresh_entry, _ = await self.snapshot(updated)
        self.check("valid_update", isinstance(await self.guard_error(old_entry), TaskSnapshotControlChange))
        await self.source("Correction: that replacement instruction was not authorized.",
                          key=valid_key, expected_revision=valid.revision)
        error = await self.guard_error(fresh_entry)
        self.check("source_correction_hard_fence", not isinstance(error, TaskSnapshotControlChange), code=error.code)
