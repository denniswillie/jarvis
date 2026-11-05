import asyncio
from dataclasses import replace
import json
import os
from pathlib import Path
import signal
import sys
import unittest
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from uuid import uuid4

import asyncpg

from app.runtime.feature_config import TaskLimits
from app.settings import JarvisSettings
from app.tasks.catalog import ConfiguredTaskProvider, TaskWorkerRegistry
from app.tasks.models import TaskCreate, TaskError, TaskStatus, WorkerOutcome
from app.tasks.repository import TaskRepository
from app.tasks.runner import TaskRunner
from app.tasks.service import TaskService
from app.tasks.worker import TaskWorkerRole, WorkerRoleConfiguration, worker_health
from evals.second_brain.source_fixture import source_fixture


@unittest.skipUnless(os.getenv("JARVIS_SECOND_BRAIN_TEST_DB_URL") and
                     os.getenv("JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"),"Manifested fixtures required")
class WorkerRoleLedgerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.root = Path(os.environ["JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"])
        self.f = await self.enterAsyncContext(source_fixture(
            os.environ["JARVIS_SECOND_BRAIN_TEST_DB_URL"],self.root,test_name=self.id()))
        self.tasks = TaskRepository(self.f.pool,TaskLimits(lease_seconds=30,poll_seconds=.1))
        self.registry = TaskWorkerRegistry(self.f.pool)
        self.providers = {"frontier":ConfiguredTaskProvider("fixture-model")}

    async def advertise(self, *, owner=None, host="host-a", worker="fixture-worker"):
        await self.registry.advertise(worker_id=worker,owner_id=owner or self.f.owner,host_id=host,
                                      providers=self.providers,ttl_seconds=10)

    async def create(self, *, owner=None, model="fixture-model"):
        return await self.tasks.create(owner or self.f.owner,TaskCreate(
            request_id=uuid4(),title="Synthetic role task",requested_outcome="Calculate a synthetic answer."),model=model)

    async def test_readiness_is_owner_model_lifetime_and_drain_scoped(self):
        self.assertFalse(await self.registry.ready(self.f.owner,"frontier","fixture-model"))
        await self.advertise()
        self.assertTrue(await self.registry.ready(self.f.owner,"frontier","fixture-model"))
        self.assertFalse(await self.registry.ready(self.f.other,"frontier","fixture-model"))
        self.assertFalse(await self.registry.ready(self.f.owner,"frontier","different-model"))
        with self.assertRaises(TaskError):
            await self.advertise(owner=self.f.other)
        async with self.f.pool.acquire() as connection:
            await connection.execute("update jarvis_task_workers set expires_at=clock_timestamp()-interval '1 second'")
        self.assertFalse(await self.registry.ready(self.f.owner,"frontier","fixture-model"))
        await self.advertise()
        await self.registry.withdraw(self.f.owner,"fixture-worker")
        self.assertFalse(await self.registry.ready(self.f.owner,"frontier","fixture-model"))

    async def test_api_replay_survives_worker_loss_but_new_intake_does_not(self):
        service = TaskService(self.tasks,self.providers,registry=self.registry)
        request = TaskCreate(request_id=uuid4(),title="Fixture",requested_outcome="Synthetic")
        with self.assertRaises(TaskError): await service.create(self.f.owner,request)
        await self.advertise()
        accepted = await service.create(self.f.owner,request)
        await self.registry.withdraw(self.f.owner,"fixture-worker")
        replay = await service.create(self.f.owner,request)
        self.assertEqual(replay.id,accepted.id)
        with self.assertRaises(TaskError):
            await service.create(self.f.owner,request.model_copy(update={"request_id":uuid4()}))

    async def test_claim_filters_prevent_cross_owner_or_model_execution(self):
        foreign = await self.create(owner=self.f.other)
        old_model = await self.create(model="different-model")
        own = await self.create()
        claim = await self.tasks.claim_next("owner-worker",("frontier",),owner_id=self.f.owner,
                                           provider_models={"frontier":"fixture-model"})
        self.assertEqual(claim.task.id,own.id)
        self.assertEqual((await self.tasks.get(self.f.other,foreign.id)).status,TaskStatus.QUEUED)
        self.assertEqual((await self.tasks.get(self.f.owner,old_model.id)).status,TaskStatus.QUEUED)

    async def test_recovery_stays_on_the_native_host_and_cannot_reassign_it(self):
        await self.create()
        claim = await self.tasks.claim_next("original",("frontier",))
        await self.tasks.start_runtime(claim,"hermes-run:"+uuid4().hex,{},host_id="host-a")
        await self.tasks.abandon(claim)
        self.assertIsNone(await self.tasks.claim_recovery("wrong-host",owner_id=self.f.owner,host_id="host-b"))
        self.assertIsNone(await self.tasks.claim_recovery("wrong-owner",owner_id=self.f.other,host_id="host-a"))
        self.assertIsNone(await self.tasks.claim_recovery("wrong-kind",host_id="host-a",backend_kinds=("sandbox",)))
        recovered = await self.tasks.claim_recovery("right-host",owner_id=self.f.owner,host_id="host-a",
                                                   backend_kinds=("frontier",))
        self.assertEqual(recovered.attempt_id,claim.attempt_id)
        async with self.f.pool.acquire() as connection:
            with self.assertRaises(asyncpg.CheckViolationError):
                await connection.execute("update jarvis_task_attempts set runtime_host_id='host-b' where id=$1",
                                         claim.attempt_id)

    async def test_worker_role_advertises_runs_and_withdraws_without_voice_services(self):
        class SyntheticProvider:
            model = "fixture-model"
            async def execute(inner,context):
                return WorkerOutcome(TaskStatus.SUCCEEDED,{"answer":"Synthetic deterministic worker receipt"})
        configuration = WorkerRoleConfiguration(python="/bin/python",work_root=self.root/uuid4().hex,
                                                 host_id="host-a",heartbeat_seconds=1)
        role = TaskWorkerRole(self.tasks,{"frontier":SyntheticProvider()},owner_id=self.f.owner,
                              configuration=configuration)
        stop = asyncio.Event()
        running = asyncio.create_task(role.run(stop))
        try:
            for _ in range(100):
                if await self.registry.ready(self.f.owner,"frontier","fixture-model"): break
                if running.done(): await running
                await asyncio.sleep(.05)
            self.assertTrue(await worker_health(self.f.pool,self.f.owner,configuration))
            task = await self.create()
            for _ in range(100):
                current = await self.tasks.get(self.f.owner,task.id)
                if current.status==TaskStatus.SUCCEEDED: break
                await asyncio.sleep(.05)
            self.assertEqual(current.status,TaskStatus.SUCCEEDED)
        finally:
            stop.set()
            await asyncio.wait_for(running,10)
        self.assertFalse(await self.registry.ready(self.f.owner,"frontier","fixture-model"))
        self.assertFalse((configuration.work_root/"worker.json").exists())

    @unittest.skipUnless(os.getenv("JARVIS_HERMES_PYTHON"),"Explicit native Hermes interpreter required")
    async def test_real_separate_worker_cli_accepts_and_finishes_a_frontier_task(self):
        model = os.environ["JARVIS_S04_MODEL"]
        state = self.root/("cli-"+uuid4().hex)
        state.mkdir(mode=0o700)
        async with self.f.pool.acquire() as connection:
            schema = await connection.fetchval("select current_schema()")
        url = urlsplit(os.environ["JARVIS_SECOND_BRAIN_TEST_DB_URL"])
        query = dict(parse_qsl(url.query))
        query["search_path"] = schema+",pg_catalog,public,extensions"
        dsn = urlunsplit((url.scheme,url.netloc,url.path,urlencode(query),url.fragment))
        environment = {
            "PATH":"/usr/bin:/bin","PYTHONPATH":str(Path(__file__).resolve().parents[1]),
            "PYTHONUNBUFFERED":"1","JARVIS_OWNER_ID":self.f.owner,"SUPABASE_DB_URL":dsn,
            "OPENAI_API_KEY":JarvisSettings().OPENAI_API_KEY,
            "SECOND_BRAIN_ENABLED":"true","SECOND_BRAIN_TASKS_ENABLED":"true",
            "SECOND_BRAIN_FRONTIER_ENABLED":"true","SECOND_BRAIN_FRONTIER_MODEL":model,
            "SECOND_BRAIN_FRONTIER_REASONING_EFFORT":"low","SECOND_BRAIN_HERMES_PYTHON":os.environ["JARVIS_HERMES_PYTHON"],
            "SECOND_BRAIN_HERMES_WORK_ROOT":str(state/"native-state"),"SECOND_BRAIN_WORKER_HOST_ID":"fixture-cli",
            "SECOND_BRAIN_WORKER_HEARTBEAT_SECONDS":"1","SECOND_BRAIN_POLL_SECONDS":"0.2",
            "SECOND_BRAIN_REQUEST_OUTPUT_TOKENS":"4096","SECOND_BRAIN_KNOWLEDGE_ENABLED":"false",
        }
        process = await asyncio.create_subprocess_exec(sys.executable,"-m","app.tasks.worker","run",
            cwd=state,env=environment,stdout=asyncio.subprocess.PIPE,stderr=asyncio.subprocess.PIPE,start_new_session=True)
        drain = asyncio.create_task(process.communicate())
        resource = {"format":"jarvis.worker_role_fixture.v1","pid":process.pid,"state":str(state),"phase":"running"}
        receipt = state/"worker-resources.json"
        receipt.write_text(json.dumps(resource))
        try:
            for _ in range(180):
                if await self.registry.ready(self.f.owner,"frontier",model): break
                if process.returncode is not None:
                    self.fail("Worker stopped before readiness: "+repr((await drain)[0].decode()[-1500:]))
                await asyncio.sleep(.25)
            self.assertTrue(await self.registry.ready(self.f.owner,"frontier",model))
            service = TaskService(self.tasks,{"frontier":ConfiguredTaskProvider(model)},registry=self.registry)
            task = await service.create(self.f.owner,TaskCreate(request_id=uuid4(),title="Synthetic CLI reasoning",
                requested_outcome="Compute 17 times 19 directly. Give the numerical answer briefly."))
            for _ in range(300):
                current = await self.tasks.get(self.f.owner,task.id)
                if current.status in (TaskStatus.SUCCEEDED,TaskStatus.PAUSED,TaskStatus.FAILED,TaskStatus.WAITING): break
                await asyncio.sleep(.25)
            resource["task_state"] = current.model_dump(mode="json")
            async with self.f.pool.acquire() as connection:
                events = await connection.fetch("select event_type from jarvis_task_events where owner_id=$1 and task_id=$2 order by sequence",self.f.owner,task.id)
                resource["task_event_kinds"] = [event["event_type"] for event in events]
            self.assertEqual(current.status,TaskStatus.SUCCEEDED,current.reason)
            async with self.f.pool.acquire() as connection:
                result = json.loads(await connection.fetchval(
                    "select content::text from jarvis_context_payloads where owner_id=$1 and id=$2",
                    self.f.owner,current.result_payload_id))
            self.assertIn("323",result["answer"])
            check = await asyncio.create_subprocess_exec(sys.executable,"-m","app.tasks.worker","check",
                cwd=state,env=environment,stdout=asyncio.subprocess.PIPE,stderr=asyncio.subprocess.PIPE)
            stdout,_ = await asyncio.wait_for(check.communicate(),10)
            self.assertEqual(check.returncode,0,stdout.decode())
            self.assertTrue(json.loads(stdout)["ready"])
            resource["task_id"],resource["answer"] = str(task.id),result["answer"]
        finally:
            if process.returncode is None:
                process.send_signal(signal.SIGTERM)
                try: await asyncio.wait_for(asyncio.shield(drain),12)
                except TimeoutError:
                    os.killpg(process.pid,signal.SIGKILL)
            stdout,stderr = await drain
            def sanitized(value):
                text = value.decode(errors="replace")[-12000:]
                for secret in (environment["OPENAI_API_KEY"],environment["SUPABASE_DB_URL"],os.environ["JARVIS_SECOND_BRAIN_TEST_DB_URL"]):
                    if secret: text = text.replace(secret,"[redacted]")
                return text
            resource["stdout"],resource["stderr"] = sanitized(stdout),sanitized(stderr)
            resource.update(phase="stopped",exit_code=process.returncode)
            receipt.write_text(json.dumps(resource))
        self.assertEqual(process.returncode,0)
        self.assertFalse(await self.registry.ready(self.f.owner,"frontier",model))
        self.assertFalse((state/"native-state"/"worker.json").exists())
