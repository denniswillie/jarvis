"""Manifested real-ledger checks; native inference is separately opt-in."""
import asyncio
import json
import os
from pathlib import Path
import unittest
from uuid import uuid4

from app.runtime.feature_config import TaskLimits
from app.settings import JarvisSettings
from app.tasks.model_gateway import ModelEvent, ModelGatewayError, OpenAIResponsesHTTP, TaskModelGateway
from app.tasks.models import BudgetExceeded, ClaimLost, TaskCreate, TaskMutation
from app.tasks.repository import TaskRepository
from app.tasks.runner import TaskRunner
from evals.second_brain.source_fixture import source_fixture


class LedgerUpstream:
    def __init__(self):
        self.calls = 0
        self.counted = 11
        self.complete = True
        self.before_count = None

    async def count(self,payload):
        if self.before_count:
            await self.before_count()
        return self.counted

    async def events(self,payload):
        self.calls += 1
        identifier = "resp_fixture_" + str(self.calls)
        values = [{"type":"response.created","response":{"id":identifier,"status":"in_progress"}}]
        if self.complete:
            values.append({"type":"response.completed","response":{"id":identifier,"status":"completed",
                "usage":{"input_tokens":11,"output_tokens":7}}})
        for value in values:
            yield ModelEvent(value,("data: "+json.dumps(value)+"\n\n").encode())


@unittest.skipUnless(
    os.getenv("JARVIS_SECOND_BRAIN_TEST_DB_URL") and os.getenv("JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"),
    "Explicit manifested second-brain database fixtures required",
)
class ModelGatewayDatabaseTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.root = Path(os.environ["JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"])
        self.f = await self.enterAsyncContext(source_fixture(
            os.environ["JARVIS_SECOND_BRAIN_TEST_DB_URL"],self.root,test_name=self.id(),
        ))
        self.tasks = TaskRepository(self.f.pool,TaskLimits(
            lease_seconds=300,request_output_tokens=2048,task_output_tokens=8192,task_input_tokens=4096,
        ))
        self.task = await self.tasks.create(self.f.owner,TaskCreate(
            request_id=uuid4(),title="Synthetic model transport check",
            requested_outcome="Check only the supplied synthetic code; no side effects.",knowledge_area_id=self.f.area,
        ),model=os.getenv("JARVIS_S04_MODEL","fixture-model"))
        self.claim = await self.tasks.claim_next("fixture-model-gateway",("frontier",))
        self.context = TaskRunner(self.tasks,{}).context(self.claim)
        self.upstream = LedgerUpstream()
        self.gateway = TaskModelGateway(self.context,self.upstream)
        self.payload = {"model":self.task.model,"input":"Synthetic request","max_output_tokens":100}

    async def collect(self):
        return b"".join([chunk async for chunk in self.gateway.stream(self.payload)])

    async def test_distinct_requests_settle_real_ledger_and_preserve_native_handle(self):
        await self.collect()
        await self.collect()
        usage = await self.context.budgets.usage(self.f.owner,self.task.id)
        self.assertEqual(usage,{"input":22,"output":14,"steps":0})
        async with self.f.pool.acquire() as connection:
            row = await connection.fetchrow(
                "select provider_state,provider_handle from jarvis_task_attempts where owner_id=$1 and id=$2",
                self.f.owner,self.claim.attempt_id,
            )
            count = await connection.fetchval("select count(*) from jarvis_task_reservations where owner_id=$1 and status='settled'",
                                               self.f.owner)
        self.assertEqual(row["provider_state"],"terminal")
        self.assertEqual(row["provider_handle"],"openai_response:resp_fixture_2")
        self.assertEqual(count,2)

    async def test_real_task_input_budget_rejects_before_provider_dispatch(self):
        self.upstream.counted = 5000
        with self.assertRaises(BudgetExceeded):
            await self.collect()
        self.assertEqual(self.upstream.calls,0)
        self.assertEqual(await self.context.budgets.usage(self.f.owner,self.task.id),{"input":0,"output":0,"steps":0})

    async def test_unknown_stream_stays_charged_and_cannot_be_reissued(self):
        self.upstream.complete = False
        with self.assertRaises(ModelGatewayError):
            await self.collect()
        with self.assertRaises(ModelGatewayError):
            await self.collect()
        self.assertEqual(self.upstream.calls,1)
        await self.tasks.abandon(self.claim,reason="synthetic_stream_lost")
        self.assertEqual(await self.context.budgets.usage(self.f.owner,self.task.id),{"input":11,"output":100,"steps":0})
        async with self.f.pool.acquire() as connection:
            row = await connection.fetchrow(
                "select status,provider_state from jarvis_task_attempts where owner_id=$1 and id=$2",
                self.f.owner,self.claim.attempt_id,
            )
        self.assertEqual(row["status"],"outcome_unknown")
        self.assertEqual(row["provider_state"],"live")

    async def test_real_cancellation_during_count_prevents_inference(self):
        async def cancel():
            await self.tasks.mutate(self.f.owner,self.task.id,"cancel",TaskMutation(
                request_id=uuid4(),expected_revision=self.task.revision,
            ))
        self.upstream.before_count = cancel
        with self.assertRaises(ClaimLost):
            await self.collect()
        self.assertEqual(self.upstream.calls,0)

    @unittest.skipUnless(os.getenv("JARVIS_S04_LIVE_MODEL_TEST")=="1","Explicit native provider opt-in required")
    async def test_native_responses_counter_and_stream_use_real_task_ledger(self):
        upstream = OpenAIResponsesHTTP(JarvisSettings().OPENAI_API_KEY,timeout_seconds=90)
        self.gateway = TaskModelGateway(self.context,upstream,max_requests=1)
        self.payload = {
            "model":self.task.model,"max_output_tokens":2048,"reasoning":{"effort":"low"},
            "input":"An unfinished Python binary search contains 'mid = low + high // 2' and then 'ret'. "
                    "Identify the two concrete problems briefly. Do not invent any other lines or say the function is finished.",
        }
        try:
            result = await asyncio.wait_for(self.collect(),120)
        finally:
            await upstream.aclose()
        events = [json.loads(line[6:]) for line in result.decode().splitlines() if line.startswith("data: ")]
        terminal = next(value["response"] for value in events if value["type"]=="response.completed")
        answer = "\n".join(part.get("text","") for item in terminal.get("output",[])
                           for part in item.get("content",[]) if part.get("type")=="output_text")
        usage = await self.context.budgets.usage(self.f.owner,self.task.id)
        report = {
            "model":self.task.model,"response_id":terminal["id"],"answer":answer,
            "native_inference":True,"native_hermes":False,"ledger_usage":usage,
            "provider_usage":terminal["usage"],"source":"synthetic inline code only",
        }
        (self.root/"native-model-gateway.json").write_text(json.dumps(report,indent=2)+"\n")
        self.assertGreater(usage["input"],0)
        self.assertGreater(usage["output"],0)
        self.assertEqual(usage["input"],terminal["usage"]["input_tokens"])
        self.assertEqual(usage["output"],terminal["usage"]["output_tokens"])
        self.assertIn("(low + high)",answer)
        self.assertIn("ret",answer.lower())
