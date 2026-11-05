"""Actual source-backed foreground saves; no production owner or provider calls."""
import asyncio
from datetime import UTC, datetime
from dataclasses import replace
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch
from uuid import UUID, uuid4

from app.experience.repository import ToolExecutionRepository
from app.memory.tools import MemoryToolModule
from app.tools.executor import JarvisToolExecutor
from app.tools.module import ToolBuildContext
from app.tools.registry import create_tool_registry, JarvisToolRegistry
from app.knowledge.conversation_writes import ConversationMemoryWriteBackend, NAMESPACE
from app.knowledge.lifecycle import SourceLifecycleService
from app.knowledge.memory_backend import SourceMemoryReadBackend
from app.knowledge.models import SourceError
from app.knowledge.publication import CanonicalSourcePublication
from app.realtime.router import _create_context_capture
from app.runtime.feature_config import TaskLimits
from app.tasks.context_events import ContextIngress
from app.tasks.context_repository import ContextRepository
from app.tasks.models import TaskError, TaskVoiceSource
from app.tasks.repository import TaskRepository
from app.tools.invocation import TaskToolInvocation, bind_tool_invocation
from app.tools.provenance import ToolResultCapture
from evals.second_brain.source_fixture import source_fixture


@unittest.skipUnless(os.getenv("JARVIS_SECOND_BRAIN_TEST_DB_URL") and
                     os.getenv("JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"),
                     "Explicit manifested source fixtures required")
class ConversationMemoryWriteTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.f = await self.enterAsyncContext(source_fixture(
            os.environ["JARVIS_SECOND_BRAIN_TEST_DB_URL"],
            Path(os.environ["JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"]),test_name=self.id()))
        self.requests = TaskRepository(self.f.pool,TaskLimits())
        self.history = ContextRepository(self.requests,publication=CanonicalSourcePublication(self.f.sources,self.f.area))
        self.writer = ConversationMemoryWriteBackend(
            self.f.sources,self.requests,owner_id=self.f.owner,area_id=self.f.area)
        self.lifecycle = SourceLifecycleService(self.f.sources,self.requests)
        async with self.f.pool.acquire() as c:
            self.session = await c.fetchval("insert into jarvis_sessions(owner_id) values($1) returning id",self.f.owner)
        self.parameters = {"memory_type":"goal","content":"Finish the synthetic robot calibration."}

    async def request(self, text="Remember that I want to finish calibration.", **changes):
        return await self.history.append(self.f.owner,ContextIngress(**(dict(
            session_id=self.session,source_namespace="voice:foreground-memory",
            source_event_id=str(uuid4()),receipt_id=str(uuid4()),kind="user_transcript",trust="user_input",
            source_time=datetime.now(UTC),payload={"text":text},
        ) | changes)))

    async def save(self, request, *, parameters=None, call_id="fixture-call", owner=None, session=None, task=None):
        with bind_tool_invocation(provider_call_id=call_id,request_event_id=request.id,
                                  request_revision=request.revision,task=task):
            return await self.writer.remember(
                owner_id=owner or self.f.owner,session_id=session or self.session,
                parameters=self.parameters if parameters is None else parameters)

    async def counts(self):
        async with self.f.pool.acquire() as c:
            return {
                "statements":await c.fetchval("select count(*) from jarvis_context_events where source_namespace=$1",NAMESPACE),
                "sources":await c.fetchval("select count(*) from jarvis_knowledge_sources"),
                "operations":await c.fetchval("select count(*) from jarvis_memory_outbox"),
            }

    async def test_exact_retry_new_provider_id_and_concurrency_preserve_one_statement(self):
        request = await self.request()
        first,second = await asyncio.gather(self.save(request,call_id="first"),self.save(request,call_id="second"))
        self.assertEqual(first["memory_id"],second["memory_id"])
        self.assertEqual({first["replayed"],second["replayed"]},{False,True})
        source = await self.f.sources.read(self.f.owner,self.f.area,UUID(first["memory_id"]))
        self.assertEqual(source.trust.value,"tool_receipt")
        self.assertEqual(source.content["attribution"]["kind"],"model_formulated_from_finalized_user_request")
        self.assertEqual(source.content["attribution"]["request_event_id"],str(request.id))
        self.assertIsNone(source.source_time)
        self.assertEqual(source.ingestion_status,"pending")
        self.assertNotIn("content",first["source"])
        self.assertEqual(await self.counts(),{"statements":1,"sources":2,"operations":2})

    async def test_distinct_requests_and_distinct_statements_are_not_text_deduplicated(self):
        request = await self.request()
        first = await self.save(request)
        second = await self.save(request,parameters=self.parameters | {"content":"Buy a replacement motor."},call_id="second")
        later = await self.request()
        third = await self.save(later,call_id="third")
        self.assertEqual(len({first["memory_id"],second["memory_id"],third["memory_id"]}),3)

    async def test_lost_commit_response_reuses_durable_statement(self):
        request = await self.request()
        native = self.writer.provenance.bind
        saved = []
        async def lost(*args,**kwargs):
            result = await native(*args,**kwargs)
            saved.append(result)
            raise RuntimeError("Synthetic lost local response")
        with patch.object(self.writer.provenance,"bind",side_effect=lost):
            with self.assertRaises(RuntimeError):
                await self.save(request)
        recovered = await self.save(request,call_id="new-provider-id")
        self.assertEqual(recovered["memory_id"],saved[0]["memory_id"])
        self.assertTrue(recovered["replayed"])
        self.assertEqual((await self.counts())["statements"],1)

    async def test_failure_before_commit_rolls_back_the_statement_and_outbox(self):
        request = await self.request()
        before = await self.counts()
        native = self.f.sources.publish_in_transaction
        async def fail(*args,**kwargs):
            await native(*args,**kwargs)
            raise RuntimeError("Synthetic failure before local commit")
        with patch.object(self.f.sources,"publish_in_transaction",side_effect=fail):
            with self.assertRaises(RuntimeError):
                await self.save(request)
        self.assertEqual(await self.counts(),before)

    async def test_owner_session_missing_request_and_worker_scopes_fail_closed(self):
        request = await self.request()
        before = await self.counts()
        with self.assertRaises(SourceError):
            await self.save(request,owner=self.f.other)
        with self.assertRaises(SourceError):
            await self.save(request,session=uuid4())
        with self.assertRaises(SourceError):
            await self.save(request,call_id=None)
        with self.assertRaises(SourceError):
            await self.save(request,task=TaskToolInvocation(uuid4(),1,uuid4()))
        with self.assertRaises(SourceError):
            await self.writer.remember(owner_id=self.f.owner,session_id=self.session,parameters=self.parameters)
        self.assertEqual(await self.counts(),before)

    async def test_provisional_or_unpublished_request_cannot_authorize_a_save(self):
        provisional = await self.request(finality="provisional")
        with self.assertRaises(SourceError):
            await self.save(provisional)
        unindexed = await ContextRepository(self.requests).append(self.f.owner,ContextIngress(
            session_id=self.session,source_namespace="voice:unpublished",source_event_id=str(uuid4()),
            receipt_id=str(uuid4()),kind="user_transcript",trust="user_input",payload={"text":"Unindexed request"},
        ))
        with self.assertRaises(SourceError):
            await self.save(unindexed)
        self.assertEqual((await self.counts())["statements"],0)

    async def test_request_revision_invalidates_saved_derivative_and_stale_admission(self):
        request = await self.request()
        saved = await self.save(request)
        revised = await self.request("Corrected request: calibration is unfinished.",
            source_namespace=request.source_namespace,source_event_id=request.source_event_id,
            expected_revision=request.revision,finality="revised")
        with self.assertRaises(SourceError):
            await self.f.sources.read(self.f.owner,self.f.area,UUID(saved["memory_id"]))
        with self.assertRaises((SourceError,TaskError)):
            await self.save(request,call_id="stale-call")
        current = await self.save(revised,call_id="revised-call")
        self.assertNotEqual(current["memory_id"],saved["memory_id"])

    async def test_forgotten_saved_statement_is_not_resurrected_by_a_retry(self):
        request = await self.request()
        saved = await self.save(request)
        deletion_request = await self.request("Stop recalling that saved calibration goal.")
        await self.lifecycle.forget(self.f.owner,self.f.area,UUID(saved["memory_id"]),expected_revision=1,
            request=TaskVoiceSource(deletion_request.id,deletion_request.revision,self.session))
        before = await self.counts()
        with self.assertRaises((SourceError,TaskError)):
            await self.save(request,call_id="late-retry")
        self.assertEqual(await self.counts(),before)

    async def test_archived_area_does_not_accept_explicit_memory(self):
        request = await self.request()
        async with self.f.pool.acquire() as c:
            await c.execute("update jarvis_knowledge_areas set status='archived',revision=revision+1 where owner_id=$1 and id=$2",
                            self.f.owner,self.f.area)
        with self.assertRaises(SourceError):
            await self.save(request)
        self.assertEqual((await self.counts())["statements"],0)

    async def test_remember_receipt_uses_actual_realtime_source_bound_capture(self):
        request = await self.request()
        saved = await self.save(request)
        runtime = SimpleNamespace(context_for=lambda owner:self.history)
        services = SimpleNamespace(knowledge_runtime=runtime,background_task_service=None)
        capture = _create_context_capture(services,owner_id=self.f.owner,session_id=self.session,
                                          provider="openai",audio_session_id=str(uuid4()))
        reader = SourceMemoryReadBackend(self.f.sources,SimpleNamespace(sources=self.f.sources),
                                        owner_id=self.f.owner,area_id=self.f.area)
        bound = await reader.capture_tool_result(
            {"status":"success",**saved},ToolResultCapture(capture,"fixture-call","remember"))
        self.assertNotIn("memory_id",bound.audit_receipt())
        deletion_request = await self.request("Stop recalling the saved calibration goal.")
        await self.lifecycle.forget(self.f.owner,self.f.area,UUID(saved["memory_id"]),expected_revision=1,
            request=TaskVoiceSource(deletion_request.id,deletion_request.revision,self.session))
        record = await self.history.get(self.f.owner,bound.event_id)
        self.assertNotEqual(record.availability,"available")

    async def test_actual_executor_records_mutating_source_receipts_and_honest_capture_failure(self):
        reader = SourceMemoryReadBackend(self.f.sources,SimpleNamespace(sources=self.f.sources),
                                        owner_id=self.f.owner,area_id=self.f.area)
        module = MemoryToolModule(read_backend=reader,write_backend=self.writer)
        registry = create_tool_registry(modules=(module,),context=ToolBuildContext(None,self.f.owner,self.session))
        tool = registry.get("remember")
        self.assertTrue(tool.mutates_state)
        self.assertTrue(tool.capture_after_commit)
        runtime = SimpleNamespace(context_for=lambda owner:self.history)
        capture = _create_context_capture(SimpleNamespace(knowledge_runtime=runtime,background_task_service=None),
            owner_id=self.f.owner,session_id=self.session,provider="openai",audio_session_id=str(uuid4()))
        self.addAsyncCleanup(capture.close)
        capture.admit_openai_commit("commit-one",1)
        capture.openai_committed("item-one")
        capture.openai_transcript("item-one","Remember this synthetic goal.")
        self.assertTrue(await capture.flush())
        capture.bind_openai_response("response-one",1)
        def executor(selected):
            return JarvisToolExecutor(selected,execution_repository=ToolExecutionRepository(self.f.pool),
                owner_id=self.f.owner,session_id=self.session,context_capture=capture,require_source_tracking=True)
        first_executor = executor(registry)
        capture.bind_openai_call("first-durable","response-one")
        first = await first_executor.execute("remember",self.parameters,provider_call_id="first-durable")
        self.assertEqual(first["status"],"success")
        same_call = await first_executor.execute("remember",self.parameters,provider_call_id="first-durable")
        self.assertEqual(same_call["memory_id"],first["memory_id"])
        failed_registry = JarvisToolRegistry()
        failed_registry.register(replace(tool,result_capture=AsyncMock(side_effect=RuntimeError("Synthetic capture failure"))))
        second_statement = self.parameters | {"content":"PRIVATE_FIXTURE_RECEIPT_BODY: buy a motor controller."}
        capture.bind_openai_call("capture-fails","response-one")
        failure = await executor(failed_registry).execute("remember",second_statement,provider_call_id="capture-fails")
        self.assertEqual(failure["error"]["code"],"tool_receipt_unavailable_after_commit")
        self.assertTrue(failure["error"]["local_effect_reported"])
        failure_replay = await executor(failed_registry).execute(
            "remember",second_statement,provider_call_id="capture-fails")
        self.assertEqual(failure_replay["error"],failure["error"])
        capture.bind_openai_call("recover-receipt","response-one")
        recovered = await first_executor.execute("remember",second_statement,provider_call_id="recover-receipt")
        self.assertEqual(recovered["status"],"success")
        self.assertTrue(recovered["replayed"])
        self.assertEqual((await self.counts())["statements"],2)
        async with self.f.pool.acquire() as c:
            self.assertFalse(await c.fetchval("""select exists(select 1 from jarvis_tool_executions
                where arguments::text like '%PRIVATE_FIXTURE_RECEIPT_BODY%'
                   or result::text like '%PRIVATE_FIXTURE_RECEIPT_BODY%')"""))
