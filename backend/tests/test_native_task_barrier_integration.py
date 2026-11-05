"""Opt-in real SQL qualification; no LLM, worker service or native app.

Run through evals.second_brain.run_tests --live-db with a private manifest dir.
Actual registered foreground handlers, finalized canonical sources, task claims,
registry leases and runtime-start/finish APIs are used. The runtime handle is
explicitly controlled; provider_state=live does not mean LLM computation.
No direct SQL task/attempt state updates or external storage/banks are used.
"""

from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack
from datetime import UTC, datetime, timedelta
import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from uuid import UUID, uuid4

from app.runtime.feature_config import TaskLimits
from app.sessions.repository import SessionRepository
from app.tasks.catalog import ConfiguredTaskProvider, TaskWorkerRegistry
from app.tasks.context_events import ContextIngress
from app.tasks.context_repository import ContextRepository
from app.tasks.models import TaskStatus, WorkerOutcome, digest
from app.tasks.repository import TaskRepository
from app.tasks.service import TaskService
from app.tasks.voice_tools import BackgroundTaskToolModule
from app.tools.invocation import CanonicalRequestIdentity, bind_tool_invocation, current_tool_invocation
from app.tools.module import ToolBuildContext
from app.tools.registry import JarvisToolRegistry
from evals.responsive_voice.native_task_barrier import (
    BarrierError, CanonicalTaskObserver, FixtureScope, ReceiptConsumer,
    decode_receipt, witness_from_observation,
)
from evals.responsive_voice.native_task_barrier_host import NativeTaskBarrierHost
from evals.second_brain.source_fixture import source_fixture


MODEL = "controlled-sql-frontier"


class ObservedGateRepository(TaskRepository):
    """Observe the real insert and optionally hold its still-open transaction."""

    def __init__(self, *args):
        super().__init__(*args)
        self.notes = []
        self.pause_create = False
        self.inserted = asyncio.Event()
        self.allow_commit = asyncio.Event()

    async def _save_request(self, connection, task, request_id, operation, parameters):
        await super()._save_request(connection, task, request_id, operation, parameters)
        if operation != "create":
            return
        invocation = current_tool_invocation()
        self.notes.append(dict(owner_id=task.owner_id, request_id=str(request_id), task_id=str(task.id),
            operation=operation, parameters_digest=digest(parameters), accepted_revision=task.revision,
            call_id=invocation.provider_call_id, request_event_id=str(invocation.request_event_id),
            request_revision=invocation.request_revision, voice_source=parameters["voice_source"],
            committed=False))
        self.inserted.set()
        if self.pause_create:
            await self.allow_commit.wait()


@unittest.skipUnless(os.environ.get("JARVIS_SECOND_BRAIN_TEST_DB_URL"), "requires isolated --live-db wrapper")
class NativeTaskBarrierIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.directory = Path(os.environ["JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"]) / self._testMethodName
        self.evidence = dict(classification="real_SQL_controlled_runtime_not_native_or_LLM",
            test=self._testMethodName, checks=[], storage_used=False, hindsight_bank_used=False)
        self.addAsyncCleanup(self._write_evidence)
        self.stack = AsyncExitStack()
        self.addAsyncCleanup(self.stack.aclose)
        self.fixture = await self.stack.enter_async_context(source_fixture(
            os.environ["JARVIS_SECOND_BRAIN_TEST_DB_URL"], self.directory, test_name=self.id()))
        self.owner = self.fixture.owner
        self.run_id = UUID(self.fixture.fixture.run_id)
        self.schema = self.fixture.fixture.manifest["schemas"][0]
        self.repo = ObservedGateRepository(self.fixture.pool, TaskLimits(lease_seconds=8,
            active_seconds=120, owner_concurrency=2, global_concurrency=4))
        self.sources = ContextRepository(self.repo)
        self.session = await SessionRepository(self.fixture.pool).create_session(self.owner, "SQL barrier fixture")
        self.worker_id = "controlled-sql-" + self.run_id.hex
        self.runtime_host_id = "controlled-host-" + self.run_id.hex
        self.providers = {"frontier": ConfiguredTaskProvider(MODEL)}
        self.registry = TaskWorkerRegistry(self.fixture.pool)
        await self.registry.advertise(worker_id=self.worker_id, owner_id=self.owner,
            host_id=self.runtime_host_id, providers=self.providers, ttl_seconds=120)
        self.addAsyncCleanup(self.registry.withdraw, self.owner, self.worker_id)
        self.host_expires_at = datetime.now(UTC) + timedelta(seconds=120)
        self.service = TaskService(self.repo, self.providers, registry=self.registry)
        self.runtime = SimpleNamespace(db_pool=self.fixture.pool, background_task_service=self.service,
            tool_modules=(BackgroundTaskToolModule(self.service),))
        self.calls = []
        self.evidence.update(run_id=self.run_id.hex, owner_id=self.owner, schema=self.schema,
            session_id=str(self.session.id), worker_id=self.worker_id, runtime_host_id=self.runtime_host_id,
            manifest=str(self.fixture.fixture.path))

    async def _write_evidence(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        fixture = getattr(self, "fixture", None)
        if fixture is not None:
            self.evidence["cleanup"] = fixture.fixture.manifest.get("cleanup")
            self.evidence["fixture_status"] = fixture.fixture.manifest["status"]
        identity = getattr(self, "run_id", uuid4()).hex
        path = self.directory / ("checks-" + identity + ".json")
        with path.open("x") as output:
            json.dump(self.evidence, output, indent=2, default=str)

    def plan(self, *names):
        self.evidence["checks"] = [dict(name=name, status="unreached") for name in names]

    def check(self, name, condition, **detail):
        record = next(row for row in self.evidence["checks"] if row["name"] == name)
        record.update(status="passed" if condition else "failed", **detail)
        self.assertTrue(condition, name)

    async def source(self, text, *, key=None, expected_revision=0):
        return await self.sources.append(self.owner, ContextIngress(session_id=self.session.id,
            source_namespace="fixture:foreground-task-barrier", source_event_id=key or uuid4().hex,
            receipt_id=uuid4().hex, kind="user_transcript", trust="user_input",
            finality="final" if expected_revision == 0 else "revised", payload={"text": text},
            expected_revision=expected_revision))

    async def call(self, source, name="start_background_task", arguments=None):
        tools = JarvisToolRegistry()
        context = ToolBuildContext(None, self.owner, self.session.id)
        for module in self.runtime.tool_modules:
            tools.register_many(module.build_tools(context))
        call_id = "controlled-foreground-" + uuid4().hex
        self.calls.append(dict(tool=name, call_id=call_id))
        with bind_tool_invocation(provider_call_id=call_id, request_event_id=source.id,
            request_revision=source.revision, request_identity=CanonicalRequestIdentity.from_record(source)):
            return await tools.get(name).handler(arguments or dict(title="WorkbenchPlan",
                requested_outcome="Compare three layouts and give five organizing steps. Reason only. No external actions.",
                mode="background", knowledge_area_id=str(self.fixture.area),
                requested_capabilities=["reasoning", "session_context"]))

    def scope(self, *, barrier_id=None):
        return FixtureScope(run_id=self.run_id, barrier_id=barrier_id or uuid4(), owner_id=self.owner,
            session_id=self.session.id, database_schema=self.schema, worker_id=self.worker_id,
            runtime_host_id=self.runtime_host_id, model=MODEL, host_expires_at=self.host_expires_at)

    def observer(self, note=None, *, scope=None):
        scope = scope or self.scope()
        witness = witness_from_observation(scope, note or self.repo.notes[-1], self.calls)
        return CanonicalTaskObserver(self.fixture.pool, scope, witness, ttl_seconds=2.0)

    async def claim(self, task_id):
        claim = await self.repo.claim_next(self.worker_id, ("frontier",), owner_id=self.owner,
            provider_models={"frontier": MODEL}, worker_capabilities=())
        self.assertIsNotNone(claim, "real canonical claim missing")
        self.assertEqual(claim.task.id, task_id)
        await self.repo.start_runtime(claim, "controlled-no-LLM-" + claim.attempt_id.hex,
            {"classification": "controlled_SQL_runtime_no_provider"}, host_id=self.runtime_host_id)
        return claim

    async def create_and_claim(self):
        source = await self.source("Start one reasoning-only WorkbenchPlan background task.")
        result = await self.call(source)
        self.assertTrue(self.repo.notes, "registered handler did not reach real canonical create")
        note = self.repo.notes[-1]
        task_id = UUID(note["task_id"])
        claim = await self.claim(task_id)
        return self.observer(note), claim, result

    async def test_uncommitted_admission_is_invisible_then_real_claim_is_eligible(self):
        self.plan("registered_handler_reached_uncommitted_insert", "independent_read_rejects_uncommitted",
                  "committed_queued_task_not_active", "real_registry_and_claim_eligible", "heartbeat_retains_exact_claim")
        source = await self.source("Start the controlled reasoning-only background task.")
        self.repo.pause_create = True
        job = asyncio.create_task(self.call(source))
        try:
            async with asyncio.timeout(20):
                await self.repo.inserted.wait()
            self.check("registered_handler_reached_uncommitted_insert", not job.done() and len(self.repo.notes) == 1)
            observer = self.observer()
            async with asyncio.timeout(5):
                hidden = await observer.observe()
            self.check("independent_read_rejects_uncommitted", hidden.status == "blocked"
                and hidden.reason == "admission_not_committed", reason=hidden.reason)
            self.repo.allow_commit.set()
            await asyncio.wait_for(job, 20)
            queued = await observer.observe()
            self.check("committed_queued_task_not_active", queued.status == "blocked"
                and queued.reason == "task_not_active", reason=queued.reason)
            claim = await self.claim(observer.admission.task_id)
            active = await observer.observe()
            ready = await self.registry.ready(self.owner, "frontier", MODEL, worker_id=self.worker_id)
            self.check("real_registry_and_claim_eligible", ready and active.status == "eligible"
                and active.claim.attempt_id == claim.attempt_id and active.claim.task_revision == claim.task_revision,
                task_id=str(claim.task.id), attempt_id=str(claim.attempt_id), fence=claim.fence)
            renewed = await self.repo.renew(claim)
            repeated = await observer.observe()
            self.check("heartbeat_retains_exact_claim", renewed and repeated.status == "eligible"
                and repeated.claim.identity() == active.claim.identity() and repeated.sequence > active.sequence)
        finally:
            self.repo.allow_commit.set()
            if not job.done():
                job.cancel()
            await asyncio.gather(job, return_exceptions=True)

    async def test_host_correction_withdraws_newer_receipt_and_joins(self):
        self.plan("real_handler_host_claim_eligible", "canonical_correction_changes_revision_and_source",
                  "newer_blocked_receipt_rejects_old_eligible", "old_claim_cannot_publish_result",
                  "host_closes_without_self_wakeup")
        host = NativeTaskBarrierHost.attach(self.runtime, run_id=self.run_id, barrier_id=uuid4(),
            owner_id=self.owner, database_schema=self.schema, model=MODEL, host_expires_at=self.host_expires_at,
            poll_interval_seconds=0.2, query_timeout_seconds=3, close_timeout_seconds=3, ttl_seconds=3)
        self.addAsyncCleanup(host.close)
        controls = self.directory / ("controls-" + self.run_id.hex)
        controls.mkdir(mode=0o700)
        ready = await self.registry.ready(self.owner, "frontier", MODEL, worker_id=self.worker_id)
        await host.configure(worker_readiness=dict(status="ready", registry_ready=ready,
            run_id=self.run_id.hex, owner_id=self.owner, database_schema=self.schema, model=MODEL,
            worker_id=self.worker_id, host_id=self.runtime_host_id,
            host_expires_at=self.host_expires_at.isoformat()), control_root=controls)
        source = await self.source("Start one WorkbenchPlan reasoning-only background task.")
        await self.call(source)
        claim = await self.claim(UUID(self.repo.notes[-1]["task_id"]))
        async with asyncio.timeout(12):
            while host.reason != "active_claim_observed":
                await asyncio.sleep(0.05)
        old_bytes = host.exchange.read()
        old = decode_receipt(old_bytes)
        consumer = ReceiptConsumer(old.scope)
        consumer.consume(old_bytes, now=datetime.now(UTC))
        self.check("real_handler_host_claim_eligible", old.claim.attempt_id == claim.attempt_id,
                   task_id=str(claim.task.id), attempt_id=str(claim.attempt_id), sequence=old.sequence)
        correction = await self.source("Correct that task: compare two layouts, retaining reasoning-only restrictions.")
        await self.call(correction, "update_background_task", dict(task_ref=self.repo.reference(claim.task.id),
            expected_revision=claim.task_revision, message="Compare two layouts. Reason only. No external actions."))
        current = await self.repo.get(self.owner, claim.task.id)
        self.check("canonical_correction_changes_revision_and_source", current.revision > claim.task_revision
            and current.request_event_id == correction.id, current_revision=current.revision)
        async with asyncio.timeout(12):
            while True:
                new_bytes = host.exchange.read()
                new = decode_receipt(new_bytes)
                if new.status == "blocked" and new.sequence > old.sequence:
                    break
                await asyncio.sleep(0.05)
        with self.assertRaises(BarrierError) as blocked:
            consumer.consume(new_bytes, now=datetime.now(UTC))
        with self.assertRaisesRegex(BarrierError, "receipt_replayed"):
            consumer.consume(old_bytes, now=datetime.now(UTC))
        self.check("newer_blocked_receipt_rejects_old_eligible", blocked.exception.code == new.reason
            and new.reason == "source_ineligible", reason=new.reason, sequence=new.sequence)
        accepted = await self.repo.finish(claim, WorkerOutcome(status=TaskStatus.SUCCEEDED,
            payload={"text": "Controlled stale result must not become current."}))
        latest = await self.repo.get(self.owner, claim.task.id)
        self.check("old_claim_cannot_publish_result", not accepted and latest.result_payload_id is None)
        closed = await host.close()
        sequence = host.sequence
        await asyncio.sleep(0.3)
        self.check("host_closes_without_self_wakeup", closed["joined"] and closed["hooks_restored"]
            and host.poll_task.done() and host.sequence == sequence, close=closed)

    async def test_terminal_cancel_and_natural_lease_expiry_fence_real_claims(self):
        self.plan("successful_terminal_result_fences", "registered_cancel_fences",
                  "natural_attempt_lease_expiry_fences", "three_distinct_real_tasks_and_attempts")
        identities = []
        observer, claim, _ = await self.create_and_claim()
        self.assertEqual((await observer.observe()).status, "eligible")
        accepted = await self.repo.finish(claim, WorkerOutcome(status=TaskStatus.SUCCEEDED,
            payload={"text": "Controlled SQL result, not model-generated."}))
        terminal = await observer.observe()
        current = await self.repo.get(self.owner, claim.task.id)
        self.check("successful_terminal_result_fences", accepted and current.status == TaskStatus.SUCCEEDED
            and current.result_payload_id is not None and terminal.reason == "task_not_active")
        identities.append((claim.task.id, claim.attempt_id))
        observer, claim, _ = await self.create_and_claim()
        self.assertEqual((await observer.observe()).status, "eligible")
        cancellation = await self.source("Cancel that controlled task, not unrelated work.")
        await self.call(cancellation, "cancel_background_task", dict(task_ref=self.repo.reference(claim.task.id),
            expected_revision=claim.task_revision, message="Explicitly cancel this task."))
        cancelled = await observer.observe()
        current = await self.repo.get(self.owner, claim.task.id)
        self.check("registered_cancel_fences", current.status == TaskStatus.CANCEL_REQUESTED
            and cancelled.status == "blocked" and cancelled.reason == "source_ineligible")
        identities.append((claim.task.id, claim.attempt_id))
        observer, claim, _ = await self.create_and_claim()
        live = await observer.observe()
        self.assertEqual(live.status, "eligible")
        await asyncio.sleep(max(0, (live.claim.lease_until - datetime.now(UTC)).total_seconds()) + 0.15)
        expired = await observer.observe()
        async with self.fixture.pool.acquire() as connection:
            actual_expired = await connection.fetchval(
                "select lease_until<=clock_timestamp() from jarvis_task_attempts where owner_id=$1 and id=$2",
                self.owner, claim.attempt_id)
        self.check("natural_attempt_lease_expiry_fences", actual_expired and expired.status == "blocked"
            and expired.reason == "lease_expired", lease_until=live.claim.lease_until)
        identities.append((claim.task.id, claim.attempt_id))
        async with self.fixture.pool.acquire() as connection:
            tasks = await connection.fetchval("select count(*) from jarvis_background_tasks where owner_id=$1", self.owner)
            attempts = await connection.fetchval("select count(*) from jarvis_task_attempts where owner_id=$1", self.owner)
        self.check("three_distinct_real_tasks_and_attempts", len(set(identities)) == tasks == attempts == 3,
                   identities=[(str(t), str(a)) for t, a in identities])
