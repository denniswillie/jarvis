"""Selected-memory writes use real canonical publication and durable action receipts."""
import asyncio
import json
import os
from pathlib import Path
import signal
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch
from uuid import UUID, uuid4

from app.knowledge.runtime import KnowledgeRuntime
from app.memory.tools import create_memory_tools
from app.runtime.feature_config import FrontierConfiguration, KnowledgeConfiguration, TaskLimits
from app.settings import JarvisSettings
from app.tasks.hermes import HermesRuntimeConfiguration, HermesTaskProvider
from app.tasks.memory_writes import MEMORY_WRITE_CAPABILITY, TaskMemoryWriteBackend
from app.tasks.models import ClaimLost, TaskCreate, TaskError, TaskMutation, TaskStatus
from app.tasks.repository import TaskRepository
from app.tasks.runner import TaskRunner
from app.tasks.worker import WorkerContextFactory
from evals.second_brain.source_fixture import source_fixture


class SelectedWriterCatalogTests(unittest.IsolatedAsyncioTestCase):
    def tools(self, **changes):
        self.writer = SimpleNamespace(remember=AsyncMock(return_value={"memory_status":"stored"}))
        return {tool.name:tool for tool in create_memory_tools(
            owner_id="fixture",read_backend=object(),write_backend=self.writer,**changes,
        )}

    async def test_selected_catalog_uses_existing_remember_tool_with_narrow_scope(self):
        tools = self.tools()
        self.assertEqual(set(tools),{"search_memory","explain_memory","remember"})
        tool = tools["remember"]
        self.assertTrue(tool.mutates_state)
        self.assertEqual(set(tool.input_schema["properties"]),{"memory_type","content","occurred_at"})
        parameters = {"memory_type":"fact","content":"Exact authorized text"}
        result = await tool.handler(parameters)
        self.assertEqual(result["memory_status"],"stored")
        self.writer.remember.assert_awaited_once_with(owner_id="fixture",session_id=None,parameters=parameters)

    async def test_unsupported_write_semantics_are_not_silently_ignored(self):
        tools = self.tools()
        for key in ("supersedes_memory_id","attach_current_evidence","entity_references"):
            result = await tools["remember"].handler({"memory_type":"fact","content":"Fixture",key:True})
            self.assertEqual(result["status"],"failed")
        self.writer.remember.assert_not_awaited()

    def test_selected_writer_cannot_activate_with_legacy_or_no_selected_reader(self):
        with self.assertRaises(ValueError):
            create_memory_tools(owner_id="fixture",write_backend=object())
        with self.assertRaises(ValueError):
            self.tools(service=object())

    async def test_read_only_selected_catalog_remains_read_only(self):
        tools = create_memory_tools(owner_id="fixture",read_backend=object())
        self.assertEqual({tool.name for tool in tools},{"search_memory","explain_memory"})


@unittest.skipUnless(os.getenv("JARVIS_SECOND_BRAIN_TEST_DB_URL") and
                     os.getenv("JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"),
                     "Explicit manifested database required")
class TaskMemoryWriteTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.root = Path(os.environ["JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"])
        self.f = await self.enterAsyncContext(source_fixture(
            os.environ["JARVIS_SECOND_BRAIN_TEST_DB_URL"],self.root,test_name=self.id(),
        ))
        self.tasks = TaskRepository(self.f.pool,TaskLimits(
            lease_seconds=300,request_output_tokens=4096,task_output_tokens=20000,task_input_tokens=100000,
        ))
        self.parameters = {"memory_type":"fact","content":"My fictional calibration marker is "+uuid4().hex,
                           "occurred_at":"2026-09-14T12:00:00+00:00"}
        async with self.f.pool.acquire() as connection:
            self.session = await connection.fetchval(
                "insert into jarvis_sessions(owner_id) values($1) returning id",self.f.owner)
            payload = await self.tasks.payload(connection,self.f.owner,{"text":"Remember this exact approved statement."})
            self.event = await connection.fetchval(
                "insert into jarvis_context_events(owner_id,session_id,source_namespace,source_event_id,kind,finality,trust,payload_id)"
                " values($1,$2,'fixture-memory-approval',$3,'user_request','final','user_input',$4) returning id",
                self.f.owner,self.session,str(uuid4()),payload)
            self.scope = await self.tasks.payload(connection,self.f.owner,self.parameters)
            self.grant = await connection.fetchval(
                "insert into jarvis_operation_grants(owner_id,capability,operation_key,request_event_id,request_revision,"
                "scope_revision,scope_payload_id,expires_at) values($1,$2,$3,$4,1,1,$5,"
                "clock_timestamp()+interval '1 hour') returning id",
                self.f.owner,MEMORY_WRITE_CAPABILITY,str(uuid4()),self.event,self.scope)
        self.knowledge = SimpleNamespace(
            sources=self.f.sources,configuration=SimpleNamespace(area_id=self.f.area),
            retrieval=SimpleNamespace(sources=self.f.sources),
            context_for=lambda owner:self.assertEqual(owner,self.f.owner),
        )

    async def prepare(self, *, knowledge=None, existing=None):
        if existing is None:
            task = await self.tasks.create(self.f.owner,TaskCreate(
                request_id=uuid4(),title="Synthetic explicit memory",
                requested_outcome="Remember the exact authorized statement, then explain_memory to read it. "
                    "Include the actual saved memory_id and completed operation reference in your answer. "
                    "A local save does not establish semantic-search readiness.",
                session_id=self.session,knowledge_area_id=self.f.area,grant_ids=(self.grant,),
            ),model=os.getenv("JARVIS_S04_MODEL","fixture-model"))
        else:
            task = existing
        claim = await self.tasks.claim_next("memory-write-fixture",("frontier",))
        self.assertEqual(claim.task.id,task.id)
        ctx = TaskRunner(self.tasks,{}).context(claim)
        p = await WorkerContextFactory(self.f.pool,knowledge=knowledge or self.knowledge)(ctx)
        return ctx,p

    async def call(self, prepared, *, identity=None):
        return await prepared.tools.invoke("remember",{"operation_ref":"op_"+self.grant.hex},
                                           provider_call_id=identity or "native-"+uuid4().hex)

    async def assert_single(self):
        async with self.f.pool.acquire() as connection:
            for table in ("jarvis_knowledge_sources","jarvis_knowledge_source_versions","jarvis_memory_outbox","jarvis_task_actions"):
                self.assertEqual(await connection.fetchval("select count(*) from "+table),1)
            self.assertEqual(await connection.fetchval(
                "select count(*) from jarvis_context_events where source_namespace='task-explicit-memory'"),1)
            row = await connection.fetchrow(
                "select a.parameters_payload_id,a.status,v.payload_id from jarvis_task_actions a "
                "join jarvis_knowledge_source_versions v on v.owner_id=a.owner_id")
            self.assertEqual(row["status"],"accepted")
            self.assertEqual(row["payload_id"],self.scope)
            self.assertEqual(row["parameters_payload_id"],self.scope)

    async def test_save_publishes_one_original_and_local_receipt_atomically(self):
        ctx,p = await self.prepare()
        result = await self.call(p)
        self.assertEqual(result["status"],"success")
        self.assertEqual(result["memory_status"],"stored")
        self.assertEqual(result["source"]["ingestion_status"],"pending")
        self.assertEqual(result["source"]["trust"],"tool_receipt")
        self.assertEqual(result["authorization"]["request_event_id"],str(self.event))
        source = await self.f.sources.read(self.f.owner,self.f.area,UUID(result["memory_id"]))
        self.assertEqual(source.content,self.parameters)
        self.assertEqual(source.context_session_id,self.session)
        self.assertEqual(source.source_time.isoformat(),self.parameters["occurred_at"])
        await self.assert_single()
        read = await p.tools.invoke("explain_memory",{"memory_id":result["memory_id"]},provider_call_id="read-original")
        self.assertEqual(read["status"],"success")
        self.assertIn(self.parameters["content"],read["content_page"]["text"])
        await p.tools.verified_operation("op_"+self.grant.hex)

    async def test_new_call_ids_replay_one_logical_save_without_another_tool_step(self):
        ctx,p = await self.prepare()
        first = await self.call(p)
        second = await self.call(p)
        self.assertEqual(first["memory_id"],second["memory_id"])
        self.assertTrue(second["task_operation"]["replayed"])
        self.assertEqual((await ctx.budgets.usage(self.f.owner,ctx.task.id))["steps"],1)
        await self.assert_single()

    async def test_failed_receipt_transaction_rolls_back_source_event_and_ingestion(self):
        ctx,p = await self.prepare()
        original = ctx.actions.record_receipt_in_transaction
        async def fail_accept(connection,owner,task,action,token,status,receipt):
            if status=="accepted":
                raise RuntimeError("synthetic commit-boundary failure")
            return await original(connection,owner,task,action,token,status,receipt)
        with patch.object(ctx.actions,"record_receipt_in_transaction",fail_accept):
            with self.assertRaises(RuntimeError):
                await self.call(p)
        async with self.f.pool.acquire() as connection:
            self.assertEqual(await connection.fetchval("select count(*) from jarvis_knowledge_sources"),0)
            self.assertEqual(await connection.fetchval(
                "select count(*) from jarvis_context_events where source_namespace='task-explicit-memory'"),0)
            self.assertEqual(await connection.fetchval("select count(*) from jarvis_memory_outbox"),0)
        self.assertEqual((await self.call(p))["status"],"outcome_unknown")

    async def test_cancel_after_commit_and_concurrent_reissue_preserve_the_accepted_receipt(self):
        ctx,p = await self.prepare()
        committed,release = asyncio.Event(),asyncio.Event()
        original = TaskMemoryWriteBackend.remember
        async def paused(backend,**arguments):
            result = await original(backend,**arguments)
            committed.set()
            await release.wait()
            return result
        with patch.object(TaskMemoryWriteBackend,"remember",paused):
            first = asyncio.create_task(self.call(p))
            try:
                await asyncio.wait_for(committed.wait(),20)
                concurrent = await self.call(p)
                self.assertEqual(concurrent["status"],"success")
                first.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await first
                replay = await self.call(p)
                self.assertEqual(replay["memory_id"],concurrent["memory_id"])
                self.assertTrue(replay["task_operation"]["replayed"])
                await self.assert_single()
            finally:
                release.set()
                await asyncio.gather(first,return_exceptions=True)

    async def test_direct_handler_and_changed_scope_cannot_create_an_ungranted_memory(self):
        _,p = await self.prepare()
        result = await p.tools.registry.get("remember").handler(self.parameters)
        self.assertEqual(result["status"],"failed")
        denied = await p.tools.invoke("remember",{"operation_ref":"op_"+self.grant.hex,"content":"different"},
                                      provider_call_id="changed")
        self.assertEqual(denied["status"],"failed")
        async with self.f.pool.acquire() as connection:
            self.assertEqual(await connection.fetchval("select count(*) from jarvis_knowledge_sources"),0)

    async def test_deleting_saved_original_revokes_its_grant_and_prevents_recreation(self):
        ctx,p = await self.prepare()
        result = await self.call(p)
        await self.f.sources.delete_source(self.f.owner,self.f.area,UUID(result["memory_id"]),expected_revision=1)
        with self.assertRaises(ClaimLost):
            await self.call(p)
        async with self.f.pool.acquire() as connection:
            self.assertEqual(await connection.fetchval("select count(*) from jarvis_knowledge_sources"),1)
            self.assertEqual(await connection.fetchval("select status from jarvis_knowledge_sources"),"deleted")
            self.assertIsNone(await connection.fetchval("select content from jarvis_context_payloads where id=$1",self.scope))

    async def native_knowledge(self):
        knowledge = await KnowledgeRuntime.create(
            pool=self.f.pool,owner_id=self.f.owner,tasks=self.tasks,
            configuration=KnowledgeConfiguration(
                base_url=os.environ["JARVIS_S04_MEMORY_URL"],api_key=os.environ["JARVIS_S04_MEMORY_TOKEN"],
                area_id=self.f.area,poll_seconds=.2,
            ),
        )
        self.addAsyncCleanup(knowledge.close)
        knowledge.start()
        return knowledge

    def provider(self, knowledge):
        return HermesTaskProvider(
            FrontierConfiguration(model=os.environ["JARVIS_S04_MODEL"],
                                  reasoning_effort=os.getenv("JARVIS_S04_REASONING_EFFORT","medium"),
                                  api_key=JarvisSettings().OPENAI_API_KEY),
            HermesRuntimeConfiguration(Path(os.environ["JARVIS_HERMES_PYTHON"]),self.root/"native-state"),
            prepare=WorkerContextFactory(self.f.pool,knowledge=knowledge),
        )

    async def execute(self, provider, ctx, suffix):
        outcome = await asyncio.wait_for(provider.execute(ctx),150)
        await self.tasks.finish(ctx.claim,outcome)
        (self.root/(suffix+"-"+str(ctx.task.id)+".json")).write_text(json.dumps({
            "status":outcome.status,"payload":outcome.payload,
            "usage":await ctx.budgets.usage(self.f.owner,ctx.task.id),
        },indent=2))
        return outcome

    async def wait_searchable(self, knowledge, memory_id):
        deadline = asyncio.get_running_loop().time()+240
        while asyncio.get_running_loop().time()<deadline:
            source = await self.f.sources.read(self.f.owner,self.f.area,UUID(memory_id))
            if source.ingestion_status=="searchable":
                recalled = await knowledge.retrieval.recall(self.f.owner,self.f.area,"What is my fictional calibration marker?")
                self.assertTrue(any(item["source_id"]==memory_id for item in recalled["sources"]))
                return
            self.assertNotIn(source.ingestion_status,{"failed","cancelled"})
            await asyncio.sleep(.5)
        self.fail("Native memory ingestion did not reach searchable")

    @unittest.skipUnless(os.getenv("JARVIS_HERMES_PYTHON") and os.getenv("JARVIS_S04_MEMORY_URL"),
                         "Actual Hermes and isolated Hindsight required")
    async def test_native_memory_save_receipt_original_and_hindsight_retrieval(self):
        knowledge = await self.native_knowledge()
        ctx,_ = await self.prepare(knowledge=knowledge)
        outcome = await self.execute(self.provider(knowledge),ctx,"native-memory-write")
        self.assertEqual(outcome.status,TaskStatus.SUCCEEDED)
        async with self.f.pool.acquire() as connection:
            memory_id = str(await connection.fetchval("select id from jarvis_knowledge_sources"))
        self.assertIn(memory_id,outcome.payload["answer"])
        self.assertEqual(outcome.payload["completed_operation_refs"],["op_"+self.grant.hex])
        self.assertIn("remember",[item["name"] for item in outcome.payload["tool_receipts"]])
        self.assertIn("explain_memory",[item["name"] for item in outcome.payload["tool_receipts"]])
        await self.assert_single()
        await self.wait_searchable(knowledge,memory_id)

    @unittest.skipUnless(os.getenv("JARVIS_HERMES_PYTHON") and os.getenv("JARVIS_S04_MEMORY_URL"),
                         "Actual Hermes and isolated Hindsight required")
    async def test_native_kill_after_memory_commit_then_resume_reuses_one_receipt(self):
        knowledge = await self.native_knowledge()
        ctx,_ = await self.prepare(knowledge=knowledge)
        provider = self.provider(knowledge)
        committed,release = asyncio.Event(),asyncio.Event()
        original = TaskMemoryWriteBackend.remember
        async def paused(backend,**arguments):
            result = await original(backend,**arguments)
            committed.set()
            await release.wait()
            return result
        with patch.object(TaskMemoryWriteBackend,"remember",paused):
            job = asyncio.create_task(provider.execute(ctx))
            try:
                await asyncio.wait_for(committed.wait(),120)
                run = provider._runs[ctx.claim.attempt_id]
                os.killpg(run.process.pid,signal.SIGKILL)
                with self.assertRaises(TaskError):
                    await asyncio.wait_for(job,20)
            finally:
                release.set()
                await asyncio.gather(job,return_exceptions=True)
        await self.tasks.finish(ctx.claim,provider.incomplete("fixture_native_kill_after_commit"))
        paused_task = await self.tasks.get(self.f.owner,ctx.task.id)
        self.assertEqual(paused_task.status,TaskStatus.PAUSED)
        resumed = await self.tasks.mutate(self.f.owner,ctx.task.id,"resume",
            TaskMutation(request_id=uuid4(),expected_revision=paused_task.revision))
        second,_ = await self.prepare(knowledge=knowledge,existing=resumed)
        outcome = await self.execute(provider,second,"native-memory-write-resumed")
        self.assertEqual(outcome.status,TaskStatus.SUCCEEDED)
        await self.assert_single()
        self.assertEqual(outcome.payload["completed_operation_refs"],["op_"+self.grant.hex])
