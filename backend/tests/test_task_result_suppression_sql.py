"""Five opt-in real-SQL cases using the existing manifested isolated fixture.

No provider/model/native transports run. The maintained fixture owns fresh schemas,
two unique owners, cleanup, and its independent SQL audit. Run only with the explicit
test DB URL and manifest directory while holding the parent's exclusive SQL slot.
"""
import asyncio
import os
from types import SimpleNamespace
import unittest
from uuid import uuid4

from app.attention.models import AttentionError
from app.attention.service import AttentionService
from app.experience.repository import ToolExecutionRepository
from app.tasks.context_capture import SessionContextCapture
from app.tasks.context_repository import ContextRepository
from app.tasks.hermes_result import HermesTaskResult
from app.tasks.models import TaskError, TaskMutation, TaskStatus, TaskVoiceSource
from app.tasks.runner import TaskRunner
from app.tasks.service import TaskService
from app.tasks.voice_tools import BackgroundTaskToolModule
from app.tools.executor import JarvisToolExecutor
from app.tools.module import ToolBuildContext
from app.tools.registry import create_tool_registry
import test_attention_inbox as attention_fixture


@unittest.skipUnless(
    os.getenv("JARVIS_SECOND_BRAIN_TEST_DB_URL")
    and os.getenv("JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"),
    "Explicit manifested database and exclusive SQL ownership required",
)
class TaskResultSuppressionSQLTests(unittest.IsolatedAsyncioTestCase):
    complete = attention_fixture.AttentionDatabaseTests.complete
    claim = attention_fixture.AttentionDatabaseTests.claim

    async def asyncSetUp(self):
        await attention_fixture.AttentionDatabaseTests.asyncSetUp(self)
        self.service = TaskService(self.tasks, {})
        self.service.attention_service = AttentionService(self.inbox)
        self.addAsyncCleanup(self.service.attention_service.close)
        self.capture = SessionContextCapture(ContextRepository(self.tasks),
            owner_id=self.f.owner, session_id=self.session, provider="openai",
            audio_session_id="sql-fixture-" + uuid4().hex)
        self.addAsyncCleanup(self.capture.close)
        self.executions = ToolExecutionRepository(self.f.pool)
        registry = create_tool_registry(modules=[BackgroundTaskToolModule(self.service)],
            context=ToolBuildContext(settings=None, owner_id=self.f.owner, session_id=self.session))
        self.executor = JarvisToolExecutor(registry, execution_repository=self.executions,
            owner_id=self.f.owner, session_id=self.session, context_capture=self.capture)

    async def user_request(self, text="Cancel that task and do not announce its result."):
        identity, call = "fixture-input:" + uuid4().hex, "fixture-call:" + uuid4().hex
        self.capture.input_admitted(identity)
        self.capture.input_final(identity, text)
        self.capture.bind_call(call, identity)
        record = await self.capture.request_for_call(call)
        self.assertIsNotNone(record)
        self.assertTrue(record.authorizes_request)
        return identity, call, TaskVoiceSource(record.id, record.revision, self.session)

    async def invoke(self, task, identity, *, call=None, message=""):
        call = call or "fixture-call:" + uuid4().hex
        self.capture.bind_call(call, identity)
        result = await self.executor.execute("cancel_background_task", {
            "task_ref": self.tasks.reference(task.id), "expected_revision": task.revision,
            "message": message, "suppress_result_announcement": True,
        }, provider_call_id=call)
        self.assertTrue(await self.capture.flush())
        return call, result

    async def intent(self, identity):
        async with self.f.pool.acquire() as connection:
            return await connection.fetchrow("""select * from jarvis_attention_intents
                where owner_id=$1 and id=$2""", self.f.owner, identity)

    async def task_original(self, task):
        async with self.f.pool.acquire() as connection:
            row = await connection.fetchrow("""select t.status,t.revision,t.result_payload_id,
                p.content::text as content,p.availability from jarvis_background_tasks t
                left join jarvis_context_payloads p on p.owner_id=t.owner_id and p.id=t.result_payload_id
                where t.owner_id=$1 and t.id=$2""", self.f.owner, task.id)
        return dict(row)

    async def test_completed_exact_dismissal_preserves_result_and_durable_tool_audit(self):
        task, intent = await self.complete()
        other_task, other_intent = await self.complete()
        before = await self.task_original(task)
        identity, call, _source = await self.user_request()
        _, result = await self.invoke(task, identity, call=call)
        self.assertEqual(result["status"], "success")
        self.assertTrue(result["accepted"])
        self.assertEqual(result["work"]["status"], "already_completed")
        self.assertFalse(result["work"]["completed_work_cancelled"])
        self.assertTrue(result["delivery"]["durable_suppression"])
        self.assertEqual(result["delivery"]["intent_id"], str(intent))
        self.assertEqual(await self.task_original(task), before)
        self.assertEqual((await self.intent(intent))["state"], "dismissed")
        self.assertEqual((await self.intent(other_intent))["state"], "pending")
        self.assertEqual((await self.tasks.get(self.f.owner, other_task.id)).status, TaskStatus.SUCCEEDED)
        executions = await self.executions.get_executions_by_call_ids(
            owner_id=self.f.owner, session_id=self.session, provider_call_ids=[call])
        self.assertEqual(len(executions), 1)
        self.assertEqual(executions[0].status, "completed")
        self.assertEqual(executions[0].result["work"], result["work"])
        self.assertEqual(executions[0].result["delivery"], result["delivery"])
        async with self.f.pool.acquire() as connection:
            rows = await connection.fetch("""select operation,accepted_generation from jarvis_attention_requests
                where owner_id=$1 and intent_id=$2""", self.f.owner, intent)
            cancelled = await connection.fetchval("""select count(*) from jarvis_task_requests
                where owner_id=$1 and task_id=$2 and operation='cancel'""", self.f.owner, task.id)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["operation"], "dismiss")
        self.assertEqual(rows[0]["accepted_generation"], result["delivery"]["generation"])
        self.assertEqual(cancelled, 0)

    async def test_wrong_owner_stale_revision_and_replaced_result_do_not_dismiss(self):
        task, old_intent = await self.complete()
        _identity, _call, source = await self.user_request()
        request = TaskMutation(request_id=uuid4(), expected_revision=task.revision)
        with self.assertRaises(TaskError):
            await self.inbox.dismiss_task_result(self.f.other, task.id, request, source=source)
        with self.assertRaises(TaskError) as stale:
            await self.inbox.dismiss_task_result(self.f.owner, task.id,
                request.model_copy(update={"expected_revision": task.revision + 1}), source=source)
        self.assertEqual(stale.exception.code, "task_revision_conflict")
        self.assertEqual((await self.intent(old_intent))["state"], "pending")

        updated = await self.tasks.mutate(self.f.owner, task.id, "update", TaskMutation(
            request_id=uuid4(), expected_revision=task.revision, message="A separate revised fixture objective."))
        claim = await self.tasks.claim_next("fixture-worker", ("frontier",))
        self.assertEqual(claim.task.id, task.id)
        outcome = await HermesTaskResult(status="succeeded", answer="Revised synthetic result.",
            speech=[{"text": "Revised synthetic result."}], evidence=[],
            completed_operation_refs=[], uncertainties=[]).verified(
                TaskRunner(self.tasks, {}).context(claim), SimpleNamespace())
        self.assertTrue(await self.tasks.finish(claim, outcome))
        current = await self.tasks.get(self.f.owner, task.id)
        self.assertEqual(current.revision, updated.revision)
        self.assertNotEqual(current.result_payload_id, task.result_payload_id)
        before = await self.task_original(current)
        with self.assertRaises(TaskError) as old_request:
            await self.inbox.dismiss_task_result(self.f.owner, task.id, request, source=source)
        self.assertEqual(old_request.exception.code, "task_revision_conflict")
        self.assertEqual(await self.task_original(current), before)
        async with self.f.pool.acquire() as connection:
            rows = await connection.fetch("""select id,state from jarvis_attention_intents
                where owner_id=$1 and task_id=$2 and task_revision=$3 and source_payload_id=$4""",
                self.f.owner, current.id, current.revision, current.result_payload_id)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["state"], "pending")
        self.assertEqual((await self.intent(old_intent))["state"], "revoked")

    async def test_duplicate_conflicting_and_corrected_source_requests_are_fenced(self):
        task, intent = await self.complete()
        identity, call, source = await self.user_request()
        _, first = await self.invoke(task, identity, call=call)
        _, duplicate = await self.invoke(task, identity)
        self.assertTrue(duplicate["delivery"]["replayed_request"])
        self.assertEqual(first["delivery"]["generation"], duplicate["delivery"]["generation"])
        _, conflict = await self.invoke(task, identity, message="Different parameters under the same request")
        self.assertEqual(conflict["status"], "failed")
        self.assertEqual(conflict["error"]["code"], "attention_request_conflict")
        self.capture.input_final(identity, "Corrected transcript: do not change that task.")
        self.assertTrue(await self.capture.flush())
        with self.assertRaises(TaskError) as corrected:
            await self.inbox.dismiss_task_result(self.f.owner, task.id,
                TaskMutation(request_id=uuid4(), expected_revision=task.revision), source=source)
        self.assertEqual(corrected.exception.code, "task_request_evidence_missing")
        self.assertEqual((await self.intent(intent))["generation"], first["delivery"]["generation"])
        async with self.f.pool.acquire() as connection:
            count = await connection.fetchval("""select count(*) from jarvis_attention_requests
                where owner_id=$1 and intent_id=$2""", self.f.owner, intent)
        self.assertEqual(count, 1)

    async def test_real_claim_write_guard_serializes_dismissal_and_preserves_forwarding(self):
        task, intent = await self.complete()
        identity, call, _source = await self.user_request()
        claim = await self.claim()
        self.assertEqual(claim.intent_id, intent)
        await self.inbox.prepare(claim, self.tts)  # Local request preparation, never TTS network I/O.
        await self.inbox.submitted(claim)
        await self.inbox.forwarded(claim, samples=2400)
        entered, release = asyncio.Event(), asyncio.Event()

        async def guarded_writer():
            async with self.inbox.guard(claim):
                entered.set()
                await release.wait()

        writing = asyncio.create_task(guarded_writer())
        suppressing = None
        try:
            await asyncio.wait_for(entered.wait(), 10)
            suppressing = asyncio.create_task(self.invoke(task, identity, call=call))
            with self.assertRaises(TimeoutError):
                await asyncio.wait_for(asyncio.shield(suppressing), .15)
        finally:
            release.set()
            await asyncio.wait_for(asyncio.gather(writing), 15)
            if suppressing is not None:
                _call, result = await asyncio.wait_for(suppressing, 30)
        self.assertTrue(result["delivery"]["durable_suppression"])
        with self.assertRaises(AttentionError) as fenced:
            async with self.inbox.guard(claim):
                self.fail("Dismissed generation reached a subsequent writer")
        self.assertEqual(fenced.exception.code, "attention_claim_lost")
        detail = await self.inbox.detail(self.f.owner, intent)
        self.assertEqual(detail["state"], "dismissed")
        self.assertEqual(len(detail["attempts"]), 1)
        self.assertEqual(detail["attempts"][0]["phase"], "revoked")
        self.assertEqual(detail["attempts"][0]["forwarded_samples"], 2400)
        self.assertIsNone(detail["attempts"][0]["played_samples"])
        self.assertEqual(result["delivery"]["hearing"], "not_established")

    async def test_real_publication_trigger_exact_missing_match_is_not_future_mute(self):
        task, intent = await self.complete()
        other_task, other_intent = await self.complete()
        async with self.f.pool.acquire() as connection:
            # Re-firing the actual publication trigger cannot duplicate the exact intent.
            await connection.execute("""update jarvis_background_tasks set result_payload_id=result_payload_id
                where owner_id=$1 and id=$2""", self.f.owner, task.id)
            count = await connection.fetchval("""select count(*) from jarvis_attention_intents
                where owner_id=$1 and task_id=$2 and task_revision=$3 and source_payload_id=$4""",
                self.f.owner, task.id, task.revision, task.result_payload_id)
            self.assertEqual(count, 1)
            # Explicit owned-fixture fault injection: remove only the unclaimed target.
            # This is not fabricated native evidence or a production cleanup operation.
            await connection.execute("delete from jarvis_attention_intents where owner_id=$1 and id=$2",
                                     self.f.owner, intent)
        before = await self.task_original(task)
        identity, call, _source = await self.user_request()
        _, result = await self.invoke(task, identity, call=call)
        self.assertFalse(result["accepted"])
        self.assertFalse(result["delivery"]["durable_suppression"])
        self.assertEqual(result["delivery"]["reason"], "exact_result_intent_missing")
        self.assertEqual(await self.task_original(task), before)
        self.assertEqual((await self.intent(other_intent))["state"], "pending")
        async with self.f.pool.acquire() as connection:
            await connection.execute("""update jarvis_background_tasks set result_payload_id=result_payload_id
                where owner_id=$1 and id=$2""", self.f.owner, task.id)
            rows = await connection.fetch("""select id,state from jarvis_attention_intents
                where owner_id=$1 and task_id=$2 and task_revision=$3 and source_payload_id=$4""",
                self.f.owner, task.id, task.revision, task.result_payload_id)
            requests = await connection.fetchval("""select count(*) from jarvis_attention_requests
                where owner_id=$1 and intent_id=$2""", self.f.owner, rows[0]["id"])
        self.assertEqual(len(rows), 1)
        self.assertNotEqual(rows[0]["id"], intent)
        self.assertEqual(rows[0]["state"], "pending")
        self.assertEqual(requests, 0)
        self.assertFalse(result["delivery"]["durable_suppression"])
        self.assertEqual((await self.tasks.get(self.f.owner, other_task.id)).status, TaskStatus.SUCCEEDED)

