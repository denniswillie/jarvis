"""Real Jarvis handlers and PostgreSQL; no native-audio quality claim."""
import asyncio
import json
import os
import time
from pathlib import Path
import unittest
from uuid import uuid4

import asyncpg

from app.assistant.tool_limits import ToolCallBudget
from app.experience.repository import ToolExecutionRepository
from app.migrations.runner import apply_migrations, quote_schema, read_schema_readiness
from app.realtime.gemini.operations import GeminiOperations
from app.runtime.feature_config import TaskLimits
from app.tasks.catalog import ConfiguredTaskProvider, TaskWorkerRegistry
from app.tasks.context import TaskContextBuilder
from app.tasks.context_capture import SessionContextCapture
from app.tasks.context_events import ContextIngress
from app.tasks.context_repository import ContextRepository
from app.tasks.models import TaskCreate, TaskStatus, WorkerOutcome
from app.tasks.repository import TaskRepository
from app.tasks.service import TaskService
from app.tasks.voice_tools import BackgroundTaskToolModule
from app.tools.executor import JarvisToolExecutor
from app.tools.module import ToolBuildContext
from app.tools.registry import create_tool_registry
from evals.second_brain.fixtures import DatabaseFixture


@unittest.skipUnless(os.getenv("JARVIS_SECOND_BRAIN_TEST_DB_URL")
    and os.getenv("JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"), "Explicit isolated DB fixture configuration required")
class VoiceTaskDatabaseTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        dsn = os.environ["JARVIS_SECOND_BRAIN_TEST_DB_URL"]
        self.fixture = DatabaseFixture(dsn, Path(os.environ["JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"]), test_name=self.id())
        self.addAsyncCleanup(self.fixture.cleanup)
        connection = await asyncpg.connect(dsn, timeout=10, command_timeout=90)
        try:
            self.schema = await self.fixture.create_schema(connection)
            await apply_migrations(connection, schema=self.schema, fresh=True)
        finally:
            await connection.close()
        async def setup(connection):
            await connection.execute(f"set search_path to {quote_schema(self.schema)},pg_catalog,public,extensions")
        self.pool = await asyncpg.create_pool(dsn, min_size=1, max_size=3, timeout=10, command_timeout=30, setup=setup)
        self.addAsyncCleanup(self.pool.close)
        self.owner, self.other = self.fixture.owners
        self.repository = TaskRepository(self.pool, TaskLimits())
        self.service = TaskService(self.repository, {"frontier": ConfiguredTaskProvider("fixture-only")})
        self.sources = ContextRepository(self.repository)
        self.session = await self.new_session()
        self.adapters = {}

    async def new_session(self, owner=None):
        async with self.pool.acquire() as connection:
            return await connection.fetchval("insert into jarvis_sessions(owner_id) values($1) returning id", owner or self.owner)

    def adapter(self, provider, session, owner):
        key = (provider, session, owner)
        if key not in self.adapters:
            capture = SessionContextCapture(self.sources, owner_id=owner, session_id=session,
                provider=provider, audio_session_id="synthetic-" + uuid4().hex)
            self.addAsyncCleanup(capture.close)
            registry = create_tool_registry(modules=(BackgroundTaskToolModule(self.service),),
                context=ToolBuildContext(None, owner, session))
            executions = ToolExecutionRepository(self.pool)
            executor = (GeminiOperations(registry=registry, repository=executions, owner_id=owner,
                session_id=session, audio_session_id=capture.audio_session_id, context_capture=capture)
                if provider == "gemini" else JarvisToolExecutor(registry, execution_repository=executions,
                    owner_id=owner, session_id=session, context_capture=capture))
            self.adapters[key] = (capture, registry, executor)
        return self.adapters[key]

    async def invoke(self, name, args, *, provider="openai", session=None, owner=None,
                     input_id=None, text="Synthetic user request for this isolated test.", finalized=True):
        session, owner = session or self.session, owner or self.owner
        capture, registry, executor = self.adapter(provider, session, owner)
        input_id = input_id or uuid4().hex
        capture.input_admitted(input_id)
        if finalized:
            capture.input_final(input_id, text)
        call_id = uuid4().hex
        started = time.monotonic()
        if provider == "gemini":
            call, _ = executor.receive({"id": call_id, "name": name, "args": args}, namespace=1, turn_number=1)
            capture.bind_call(call.execution_id, input_id)
            result = await executor.execute(call, still_current=lambda: True,
                budget=ToolCallBudget(max_total_calls=20, max_repeated_calls=10))
        else:
            capture.bind_call(call_id, input_id)
            result = await executor.execute(name, args, provider_call_id=call_id)
        self.fixture.manifest.setdefault("voice_timings", []).append({
            "provider": provider, "tool": name, "duration_ms": (time.monotonic() - started) * 1000,
            "status": result.get("status"), "error_code": (result.get("error") or {}).get("code"),
            "capture_issue": capture.issue,
        })
        self.fixture.persist()
        return result

    async def start(self, **kwargs):
        args = kwargs.pop("args", {"title": "Synthetic code", "requested_outcome": "Check the synthetic midpoint.", "mode": "consult"})
        result = await self.invoke("start_background_task", args, **kwargs)
        self.assertEqual(result["status"], "success", result)
        return result["task"]

    async def task(self, reference):
        identity = await self.repository.resolve_reference(self.owner, reference)
        return await self.repository.get(self.owner, identity)

    async def test_both_provider_executors_replay_after_cancelled_native_delivery(self):
        for provider in ("openai", "gemini"):
            with self.subTest(provider=provider):
                first = await self.start(provider=provider, input_id="same-input")
                if provider == "gemini":
                    self.adapter(provider, self.session, self.owner)[2].cancel(namespace=1)
                replay = await self.start(provider=provider, input_id="same-input")
                self.assertEqual(first["task_ref"], replay["task_ref"])
                self.assertEqual((await self.task(first["task_ref"])).status, TaskStatus.QUEUED)
        self.assertEqual(len(await self.repository.list(self.owner)), 2)

    async def test_distinct_identical_requests_and_explicit_independent_slots(self):
        first = await self.start(input_id="first")
        second = await self.start(input_id="second")
        self.assertNotEqual(first["task_ref"], second["task_ref"])
        args = {"title": "Synthetic independent work", "requested_outcome": "Read another synthetic source.",
                "mode": "background", "request_slot": 2}
        third = await self.start(input_id="first", args=args)
        self.assertNotIn(third["task_ref"], {first["task_ref"], second["task_ref"]})
        conflict = await self.invoke("start_background_task", args | {"request_slot": 1}, input_id="first")
        self.assertEqual(conflict["error"]["code"], "task_request_conflict")
        self.assertEqual(len(await self.repository.list(self.owner)), 3)

    async def test_provisional_input_and_wrong_owner_cannot_mutate(self):
        blocked = await self.invoke("start_background_task", {"title": "Synthetic", "requested_outcome": "Test"},
                                    finalized=False)
        self.assertEqual(blocked["error"]["code"], "task_request_not_finalized")
        first = await self.start()
        foreign_session = await self.new_session(self.other)
        denied = await self.invoke("cancel_background_task", {"task_ref": first["task_ref"], "expected_revision": 1},
                                  owner=self.other, session=foreign_session)
        self.assertEqual(denied["error"]["code"], "task_reference_unavailable")
        self.assertEqual((await self.task(first["task_ref"])).revision, 1)

    async def test_two_tasks_targeted_update_cancel_and_stale_revision(self):
        first, second = await self.start(), await self.start()
        claims = [await self.repository.claim_next("fixture-worker", ("frontier",)) for _ in range(2)]
        self.assertTrue(all(claim is not None for claim in claims))
        state = await self.invoke("list_background_tasks", {})
        self.assertEqual({item["state"] for item in state["tasks"]}, {"running"})
        args = {"task_ref": first["task_ref"], "expected_revision": 1,
                "message": "Check the midpoint using the newly corrected synthetic source; keep prior constraints."}
        updated = await self.invoke("update_background_task", args, input_id="correction")
        replay = await self.invoke("update_background_task", args, input_id="correction")
        self.assertEqual(updated["task"]["revision"], 2)
        self.assertEqual(replay["task"]["revision"], 2)
        stale = await self.invoke("update_background_task", args)
        self.assertEqual(stale["error"]["code"], "task_revision_conflict")
        cancelled = await self.invoke("cancel_background_task", {"task_ref": second["task_ref"], "expected_revision": 1})
        self.assertEqual(cancelled["task"]["state"], "cancel_requested")
        self.assertEqual((await self.task(first["task_ref"])).pending_status, "queued")
        for claim in claims:
            self.assertFalse(await self.repository.finish(claim, WorkerOutcome(TaskStatus.SUCCEEDED, {"answer": "Stale"})))

    async def test_new_session_context_is_attached_but_unrelated_session_is_not(self):
        first = await self.start(text="Original synthetic project constraint: use ascending input.")
        second_session, unrelated = await self.new_session(), await self.new_session()
        await self.sources.append(self.owner, ContextIngress(session_id=unrelated, source_namespace="fixture",
            source_event_id="unrelated", receipt_id="once", kind="user_transcript", trust="user_input",
            payload={"text": "Unrelated private session must not reach this worker."}))
        await self.invoke("update_background_task", {"task_ref": first["task_ref"], "expected_revision": 1,
            "message": "Check the midpoint and preserve ascending input."}, session=second_session,
            text="New synthetic correction: use parentheses around low plus high.", input_id="new-session-correction")
        claim = await self.repository.claim_next("fixture-worker", ("frontier",))
        manifest = await TaskContextBuilder(self.sources).build(claim)
        serialized = json.dumps(manifest.content)
        self.assertIn("ascending input", serialized)
        self.assertIn("parentheses around low plus high", serialized)
        self.assertNotIn("Unrelated private session", serialized)
        archive = await self.sources.archive_for_claim(claim)
        self.assertEqual({source.session_id for source in archive}, {self.session, second_session})
        for source in archive:
            self.assertEqual((await self.sources.source_for_claim(claim, source.id)).id, source.id)
        async with self.pool.acquire() as connection:
            self.assertEqual(await connection.fetchval("select count(*) from jarvis_task_session_context where owner_id=$1",
                                                       self.owner), 2)

    async def test_status_snapshot_keeps_result_and_matching_milestone(self):
        first = await self.start()
        task, detail = await self.repository.voice_detail(self.owner, (await self.task(first["task_ref"])).id)
        self.assertIsNone(detail["result"])
        self.assertEqual(detail["last_milestone"]["event_type"], "accepted")
        first_sequence = detail["last_milestone"]["sequence"]
        claim = await self.repository.claim_next("fixture-worker", ("frontier",))
        await TaskContextBuilder(self.sources).build(claim)
        answer = {"format": "jarvis.hermes.task_result.v1", "answer": "Synthetic completed answer."}
        self.assertTrue(await self.repository.finish(claim, WorkerOutcome(TaskStatus.SUCCEEDED, answer)))
        task, detail = await self.repository.voice_detail(self.owner, task.id)
        self.assertEqual(detail["result"], answer)
        self.assertEqual(detail["last_milestone"]["event_type"], "succeeded")
        self.assertEqual(detail["last_milestone"]["task_revision"], task.revision)
        self.assertGreater(detail["last_milestone"]["sequence"], first_sequence)
        async with self.pool.acquire() as connection:
            await connection.execute(
                "update jarvis_context_payloads set availability='deleted',content=null where owner_id=$1 and id=$2",
                self.owner, task.result_payload_id)
        _, detail = await self.repository.voice_detail(self.owner, task.id)
        self.assertIsNone(detail["result"])
        self.assertEqual(detail["last_milestone"]["event_type"], "succeeded")

    async def test_foreground_snapshot_retains_terminal_states_bounds_and_owner_privacy(self):
        cancelled = await self.start(args={"title": "Cancelled fixture", "requested_outcome": "Synthetic only"})
        await self.invoke("cancel_background_task",
            {"task_ref": cancelled["task_ref"], "expected_revision": 1})
        succeeded = await self.start(args={"title": "Completed private title", "requested_outcome": "Synthetic only"})
        claim = await self.repository.claim_next("fixture-worker", ("frontier",))
        await TaskContextBuilder(self.sources).build(claim)
        self.assertTrue(await self.repository.finish(claim, WorkerOutcome(TaskStatus.SUCCEEDED,
            {"format": "jarvis.hermes.task_result.v1", "answer": "PRIVATE-ANSWER-NOT-FOR-SNAPSHOT"})))
        failed = await self.start(args={"title": "Failed fixture", "requested_outcome": "Synthetic only"})
        claim = await self.repository.claim_next("fixture-worker", ("frontier",))
        await TaskContextBuilder(self.sources).build(claim)
        self.assertTrue(await self.repository.finish(claim, WorkerOutcome(TaskStatus.FAILED,
            {"error": "synthetic-only"})))
        active = await self.start(args={"title": "Still active", "requested_outcome": "Synthetic only"})
        foreign_session = await self.new_session(self.other)
        foreign = await self.start(owner=self.other, session=foreign_session,
            args={"title": "FOREIGN-PRIVATE-TITLE", "requested_outcome": "Synthetic only"})
        async with self.repository.transaction(self.owner):
            summary = await asyncio.wait_for(self.repository.context_summaries(self.owner), timeout=2)
        self.assertEqual(summary["total"], 1)
        self.assertEqual(summary["terminal_total"], 3)
        self.assertEqual([item["task_ref"] for item in summary["tasks"]], [active["task_ref"]])
        self.assertEqual({item["task_ref"]: item["status"] for item in summary["recent_terminal_tasks"]},
            {cancelled["task_ref"]: "cancelled", succeeded["task_ref"]: "succeeded", failed["task_ref"]: "failed"})
        self.assertEqual(summary["recent_terminal_tasks"][0]["task_ref"], failed["task_ref"])
        self.assertNotIn("PRIVATE-ANSWER", json.dumps(summary))
        self.assertNotIn("FOREIGN-PRIVATE", json.dumps(summary))
        self.assertNotIn(foreign["task_ref"], json.dumps(summary))
        bounded = await self.repository.context_summaries(self.owner, limit=1)
        self.assertEqual((bounded["total"], bounded["terminal_total"]), (1, 3))
        self.assertEqual(len(bounded["tasks"]), 1)
        self.assertEqual([item["task_ref"] for item in bounded["recent_terminal_tasks"]], [failed["task_ref"]])
        completed = await self.task(succeeded["task_ref"])
        async with self.pool.acquire() as connection:
            await connection.execute(
                "update jarvis_context_payloads set availability='deleted',content=null where owner_id=$1 and id=$2",
                self.owner, completed.description_payload_id)
        after = await self.repository.context_summaries(self.owner)
        self.assertNotIn("Completed private title", json.dumps(after))
        self.assertEqual(next(item["status"] for item in after["recent_terminal_tasks"]
                              if item["task_ref"] == succeeded["task_ref"]), "succeeded")
        other = await self.repository.context_summaries(self.other)
        self.assertEqual(other["terminal_total"], 0)
        self.assertEqual([item["task_ref"] for item in other["tasks"]], [foreign["task_ref"]])

    async def test_pending_question_and_invalidated_expert_result(self):
        first = await self.start()
        claim = await self.repository.claim_next("fixture-worker", ("frontier",))
        await TaskContextBuilder(self.sources).build(claim)
        answer = {"format": "jarvis.hermes.task_result.v1", "status": "waiting_for_input",
                  "answer": "Is this synthetic input sorted?", "evidence": [], "completed_operation_refs": [],
                  "uncertainties": ["Input order is unavailable."]}
        self.assertTrue(await self.repository.finish(claim, WorkerOutcome(TaskStatus.WAITING, answer)))
        summaries = await self.repository.context_summaries(self.owner)
        self.assertEqual(summaries["tasks"][0]["pending_question"], answer["answer"])
        for provider in ("openai", "gemini"):
            with self.subTest(provider=provider):
                result = await self.invoke("get_background_task", {"task_ref": first["task_ref"]}, provider=provider)
                self.assertTrue(result["expert_output_available"])
                self.assertEqual(result["stored_output_kind"], "legacy_expert")
                self.assertFalse(result["expert_speech_prepared"])
                self.assertEqual(result["expert_output_unavailable_reason"], "expert_speech_not_prepared")
                self.assertNotIn("expert_result", result)
                self.assertNotIn(answer["answer"], json.dumps(result))
                self.assertNotIn(answer["uncertainties"][0], json.dumps(result))
                self.assertEqual(result["answer_delivery"]["reason"], "attention_coordinator_unavailable")
        _, canonical = await self.repository.voice_detail(self.owner, claim.task.id)
        self.assertEqual(canonical["result"]["answer"], answer["answer"])
        self.assertEqual(canonical["result"]["uncertainties"], answer["uncertainties"])
        original = await self.sources.get(self.owner, claim.task.request_event_id)
        await self.sources.append(self.owner, ContextIngress(session_id=self.session,
            source_namespace=original.source_namespace, source_event_id=original.source_event_id,
            receipt_id="corrected", kind="user_transcript", trust="user_input", finality="revised",
            payload={"text": "Corrected synthetic objective."}, expected_revision=original.revision))
        result = await self.invoke("get_background_task", {"task_ref": first["task_ref"]})
        self.assertNotIn("expert_result", result)
        self.assertFalse(result["expert_output_available"])
        self.assertIsNotNone(result["result_unavailable_reason"])
        summaries = await self.repository.context_summaries(self.owner)
        self.assertIsNone(summaries["tasks"][0]["pending_question"])
        self.assertFalse(summaries["tasks"][0]["pending_question_available"])

    async def test_removed_attached_source_revokes_copied_manifest(self):
        first = await self.start()
        later = await self.new_session()
        await self.invoke("update_background_task", {"task_ref": first["task_ref"], "expected_revision": 1,
            "message": "Use the new synthetic constraint."}, session=later)
        claim = await self.repository.claim_next("fixture-worker", ("frontier",))
        manifest = await TaskContextBuilder(self.sources).build(claim)
        source = await self.sources.get(self.owner, claim.task.request_event_id)
        async with self.pool.acquire() as connection:
            await connection.execute("update jarvis_context_payloads set availability='deleted',content=null where owner_id=$1 and id=$2",
                                     self.owner, source.payload_id)
            row = await connection.fetchrow("""select m.status,p.availability from jarvis_task_context_manifests m
                join jarvis_context_payloads p on p.owner_id=m.owner_id and p.id=m.payload_id where m.id=$1""", manifest.id)
        self.assertEqual((row["status"], row["availability"]), ("revoked", "deleted"))

    async def test_task_audits_do_not_persist_duplicate_private_arguments_or_answers(self):
        first = await self.start(args={"title": "PRIVATE-SYNTHETIC-TITLE",
            "requested_outcome": "PRIVATE-SYNTHETIC-OBJECTIVE"})
        claim = await self.repository.claim_next("fixture-worker", ("frontier",))
        await TaskContextBuilder(self.sources).build(claim)
        await self.repository.finish(claim, WorkerOutcome(TaskStatus.SUCCEEDED,
            {"answer": "PRIVATE-SYNTHETIC-EXPERT", "format": "jarvis.hermes.task_result.v1",
             "status": "succeeded", "evidence": [], "completed_operation_refs": [], "uncertainties": []}))
        for provider in ("openai", "gemini"):
            with self.subTest(provider=provider):
                result = await self.invoke("get_background_task", {"task_ref": first["task_ref"]}, provider=provider)
                self.assertTrue(result["expert_output_available"])
                self.assertEqual(result["stored_output_kind"], "legacy_expert")
                self.assertFalse(result["expert_speech_prepared"])
                self.assertNotIn("PRIVATE-SYNTHETIC-EXPERT", json.dumps(result))
                self.assertNotIn("expert_result", result)
                self.assertEqual(result["answer_delivery"]["state"], "unavailable")
            capture = self.adapter(provider, self.session, self.owner)[0]
            await capture.flush()
        _, canonical = await self.repository.voice_detail(self.owner, claim.task.id)
        self.assertEqual(canonical["result"]["answer"], "PRIVATE-SYNTHETIC-EXPERT")
        async with self.pool.acquire() as connection:
            executions = await connection.fetch(
                "select arguments,result from jarvis_tool_executions where owner_id=$1", self.owner)
            tool_sources = await connection.fetch("""select p.content from jarvis_context_events e
                join jarvis_context_payloads p on p.owner_id=e.owner_id and p.id=e.payload_id
                where e.owner_id=$1 and e.kind='tool_outcome'""", self.owner)
        serialized = json.dumps([dict(row) for row in (*executions, *tool_sources)], default=str)
        self.assertNotIn("PRIVATE-SYNTHETIC", serialized)
        self.assertIn(first["task_ref"], serialized)

    async def test_incomplete_legacy_result_is_not_verified_or_exposed_for_retelling(self):
        first = await self.start()
        claim = await self.repository.claim_next("fixture-worker", ("frontier",))
        await TaskContextBuilder(self.sources).build(claim)
        marker = "PRIVATE-INCOMPLETE-LEGACY-ANSWER"
        await self.repository.finish(claim, WorkerOutcome(TaskStatus.SUCCEEDED,
            {"format": "jarvis.hermes.task_result.v1", "answer": marker}))
        for provider in ("openai", "gemini"):
            with self.subTest(provider=provider):
                result = await self.invoke("get_background_task", {"task_ref": first["task_ref"]}, provider=provider)
                self.assertFalse(result["expert_output_available"])
                self.assertEqual(result["stored_output_kind"], "unverified")
                self.assertFalse(result["expert_speech_prepared"])
                self.assertEqual(result["expert_output_unavailable_reason"], "stored_output_authorship_unverified")
                self.assertNotIn(marker, json.dumps(result))
                self.assertNotIn("expert_result", result)

    async def test_status_read_does_not_wait_for_owner_write_and_then_sees_source_correction(self):
        first = await self.start()
        task = await self.task(first["task_ref"])
        original = await self.sources.get(self.owner, task.request_event_id)
        async with self.repository.transaction(self.owner) as connection:
            await self.sources.append_in_transaction(connection, self.owner, ContextIngress(
                session_id=self.session, source_namespace=original.source_namespace,
                source_event_id=original.source_event_id, receipt_id="corrected-status",
                kind="user_transcript", trust="user_input", finality="revised",
                payload={"text": "Corrected request"}, expected_revision=original.revision))
            observed, details = await asyncio.wait_for(
                self.repository.voice_detail(self.owner, task.id), timeout=2)
            self.assertEqual(observed.revision, task.revision)
            self.assertIsNone(details["result_unavailable_reason"])
        _, details = await self.repository.voice_detail(self.owner, task.id)
        self.assertEqual(details["result_unavailable_reason"], "request_evidence_unavailable")

    async def test_read_only_status_still_checks_area_and_grant_revocation(self):
        async with self.repository.transaction(self.owner) as connection:
            area = await connection.fetchval(
                "insert into jarvis_knowledge_areas(owner_id,name,domain) values($1,'Fixture','test') returning id",
                self.owner)
            event = await self.repository._request_event(connection, self.owner, uuid4(),
                {"literal_request": "Synthetic authorization only; perform no external action."})
            payload = await self.repository.payload(connection, self.owner, {"scope": "synthetic-only"})
            grant = await connection.fetchval("""insert into jarvis_operation_grants(owner_id,capability,
                operation_key,request_event_id,request_revision,scope_revision,scope_payload_id,expires_at)
                values($1,'fixture.read','fixture-operation',$2,1,1,$3,
                clock_timestamp()+interval '1 hour') returning id""", self.owner, event["id"], payload)
        task = await self.repository.create(self.owner, TaskCreate(request_id=uuid4(), title="Scoped read",
            requested_outcome="Inspect synthetic state", session_id=self.session,
            knowledge_area_id=area, grant_ids=[grant]), model="fixture")
        _, details = await self.repository.voice_detail(self.owner, task.id)
        self.assertIsNone(details["result_unavailable_reason"])
        async with self.pool.acquire() as connection:
            await connection.execute("update jarvis_operation_grants set status='revoked' where owner_id=$1 and id=$2",
                                     self.owner, grant)
        _, details = await self.repository.voice_detail(self.owner, task.id)
        self.assertEqual(details["result_unavailable_reason"], "task_grant_unavailable")
        async with self.pool.acquire() as connection:
            await connection.execute("""update jarvis_knowledge_areas set status='deleted',revision=revision+1
                where owner_id=$1 and id=$2""", self.owner, area)
        _, details = await self.repository.voice_detail(self.owner, task.id)
        self.assertEqual(details["result_unavailable_reason"], "knowledge_area_unavailable")

    async def test_readiness_runs_inside_admission_and_replay_does_not_reopen_it(self):
        registry = TaskWorkerRegistry(self.pool)
        self.service.registry = registry
        checks = []
        original = registry.require
        async def require(owner, kind, model, *, connection=None):
            self.assertIsNotNone(connection)
            self.assertTrue(connection.is_in_transaction())
            checks.append((owner, kind, model))
            await original(owner, kind, model, connection=connection)
        registry.require = require
        args = {"title": "Synthetic task", "requested_outcome": "Check this source."}
        missing = await self.invoke("start_background_task", args, input_id="worker-case")
        self.assertEqual(missing["error"]["code"], "task_worker_not_ready")
        self.assertEqual(await self.repository.list(self.owner), [])
        await registry.advertise(worker_id="fixture-ready", owner_id=self.owner, host_id="fixture",
            providers=self.service.providers, ttl_seconds=120)
        accepted = await self.start(args=args, input_id="worker-case")
        await registry.withdraw(self.owner, "fixture-ready")
        replay = await self.start(args=args, input_id="worker-case")
        self.assertEqual(replay["task_ref"], accepted["task_ref"])
        self.assertEqual(len(checks), 2)  # Missing/new, but no readiness check on replay.
        self.service.providers = {}
        replay = await self.start(args=args, input_id="worker-case")
        self.assertEqual(replay["task_ref"], accepted["task_ref"])

    async def test_resume_readiness_is_atomic_and_replay_survives_worker_loss(self):
        task = await self.start()
        cancelled = await self.invoke("cancel_background_task",
            {"task_ref": task["task_ref"], "expected_revision": 1})
        self.assertEqual(cancelled["task"]["state"], "cancelled")
        registry = TaskWorkerRegistry(self.pool)
        self.service.registry = registry
        args = {"task_ref": task["task_ref"], "expected_revision": 2}
        missing = await self.invoke("resume_background_task", args, input_id="resume-case")
        self.assertEqual(missing["error"]["code"], "task_worker_not_ready")
        self.assertEqual((await self.task(task["task_ref"])).revision, 2)
        await registry.advertise(worker_id="fixture-ready", owner_id=self.owner, host_id="fixture",
            providers=self.service.providers, ttl_seconds=120)
        resumed = await self.invoke("resume_background_task", args, input_id="resume-case")
        self.assertEqual(resumed["task"]["revision"], 3)
        await registry.withdraw(self.owner, "fixture-ready")
        replay = await self.invoke("resume_background_task", args, input_id="resume-case")
        self.assertEqual(replay["task"]["revision"], 3)

    async def test_unknown_knowledge_area_is_not_silently_downgraded(self):
        result = await self.invoke("start_background_task", {"title": "Synthetic scoped task",
            "requested_outcome": "Check this area's original source", "knowledge_area_id": str(uuid4())})
        self.assertEqual(result["error"]["code"], "task_area_unavailable")
        self.assertEqual(await self.repository.list(self.owner), [])

    async def test_readiness_checks_the_voice_context_table(self):
        async with self.pool.acquire() as connection:
            self.assertTrue((await read_schema_readiness(connection, schema=self.schema)).ready)
            await connection.execute("drop table jarvis_task_session_context")
            readiness = await read_schema_readiness(connection, schema=self.schema)
            self.assertFalse(readiness.ready)
            self.assertIn("missing_table:jarvis_task_session_context", readiness.issues)

    async def test_cancelled_queued_task_does_not_attach_cancel_session(self):
        first = await self.start()
        later = await self.new_session()
        result = await self.invoke("cancel_background_task", {"task_ref": first["task_ref"], "expected_revision": 1}, session=later)
        self.assertEqual(result["task"]["state"], "cancelled")
        async with self.pool.acquire() as connection:
            self.assertEqual(await connection.fetchval("select count(*) from jarvis_task_session_context where owner_id=$1",
                                                       self.owner), 1)
