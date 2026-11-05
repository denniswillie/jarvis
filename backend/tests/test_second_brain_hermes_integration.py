"""Real task ledger and production Hermes adapter; native cases require opt-in."""
import asyncio
from datetime import UTC, datetime
from io import BytesIO
import json
import os
import signal
import shutil
from pathlib import Path
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
from uuid import uuid4

from app.images import ImageReference, ResolvedImage
from app.notes.repository import NoteRepository
from legacy_note_tool_fixture import NoteToolModule
from app.runtime.feature_config import FrontierConfiguration, TaskLimits
from app.settings import JarvisSettings
from app.tasks.context_events import ContextIngress
from app.tasks.context_images import TaskContextImages
from app.tasks.context_repository import ContextRepository
from app.tasks.hermes import HermesContextFactory, HermesRuntimeConfiguration, HermesTaskProvider, _NativeRun
from app.tasks.model_gateway import ModelGatewayError, TaskModelGateway
from app.tasks.models import ClaimLost, TaskCreate, TaskError, TaskMutation, TaskStatus, WorkerOutcome
from app.tasks.repository import TaskRepository
from app.tasks.runner import TaskRunner
from evals.second_brain.source_fixture import source_fixture
from test_second_brain_model_gateway import Upstream


@unittest.skipUnless(
    os.getenv("JARVIS_SECOND_BRAIN_TEST_DB_URL") and os.getenv("JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"),
    "Explicit manifested second-brain fixtures required",
)
class HermesLedgerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.root = Path(os.environ["JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"])
        self.f = await self.enterAsyncContext(source_fixture(
            os.environ["JARVIS_SECOND_BRAIN_TEST_DB_URL"],self.root,test_name=self.id(),
        ))
        self.tasks = TaskRepository(self.f.pool,TaskLimits(
            lease_seconds=300,request_output_tokens=4096,task_output_tokens=20000,task_input_tokens=100000,
        ))
        async with self.f.pool.acquire() as connection:
            self.session = await connection.fetchval("insert into jarvis_sessions(owner_id) values($1) returning id",self.f.owner)

    async def claim(self, outcome="Check a synthetic task.", *, grant_ids=()):
        task = await self.tasks.create(self.f.owner,TaskCreate(
            request_id=uuid4(),title="Synthetic Hermes worker",requested_outcome=outcome,
            session_id=self.session,knowledge_area_id=self.f.area,grant_ids=grant_ids,
        ),model=os.getenv("JARVIS_S04_MODEL","fixture-model"))
        claim = await self.tasks.claim_next("hermes-fixture-worker",("frontier",))
        self.assertEqual(claim.task.id,task.id)
        return TaskRunner(self.tasks,{}).context(claim)

    async def runtime(self, context):
        handle = "hermes-run:"+uuid4().hex
        await self.tasks.start_runtime(context.claim,handle,{"fixture":True})
        return handle

    async def test_nested_model_settlement_does_not_end_native_runtime(self):
        ctx = await self.claim()
        handle = await self.runtime(ctx)
        gateway = TaskModelGateway(ctx,Upstream(),run_id=handle,managed_runtime=True)
        payload = {"model":ctx.task.model,"input":"Synthetic","max_output_tokens":100}
        for _ in range(2):
            [value async for value in gateway.stream(payload)]
        async with self.f.pool.acquire() as connection:
            row = await connection.fetchrow("select provider_state,provider_handle from jarvis_task_attempts where id=$1",
                                            ctx.claim.attempt_id)
        self.assertEqual(row["provider_state"],"live")
        self.assertEqual(row["provider_handle"],handle)
        self.assertEqual(await ctx.budgets.usage(self.f.owner,ctx.task.id),{"input":22,"output":14,"steps":0})
        await self.tasks.finish(ctx.claim,WorkerOutcome(TaskStatus.SUCCEEDED,{"synthetic":"done"}))

    async def test_nested_request_cannot_use_another_runtime_or_overlap_unknown_usage(self):
        ctx = await self.claim()
        handle = await self.runtime(ctx)
        reservation = await ctx.reserve("first-native-model",input_tokens=10,output_tokens=20)
        with self.assertRaises(ClaimLost):
            await ctx.budgets.submit(ctx.claim,reservation["id"],parent_provider_handle="hermes-run:"+uuid4().hex)
        await ctx.budgets.submit(ctx.claim,reservation["id"],parent_provider_handle=handle)
        second = await ctx.reserve("second-native-model",input_tokens=10,output_tokens=20)
        with self.assertRaises(TaskError):
            await ctx.budgets.submit(ctx.claim,second["id"],parent_provider_handle=handle)

    async def test_ended_native_run_keeps_unknown_model_usage_charged(self):
        ctx = await self.claim()
        handle = await self.runtime(ctx)
        reservation = await ctx.reserve("uncertain-model",input_tokens=31,output_tokens=100)
        await ctx.budgets.submit(ctx.claim,reservation["id"],parent_provider_handle=handle)
        await self.tasks.finish(ctx.claim,HermesTaskProvider.incomplete("model_usage_unknown"))
        self.assertEqual(await ctx.budgets.usage(self.f.owner,ctx.task.id),{"input":31,"output":100,"steps":0})
        task = await self.tasks.get(self.f.owner,ctx.task.id)
        self.assertEqual(task.status,TaskStatus.PAUSED)

    async def cleanup_runtime(self, ctx):
        configuration = HermesRuntimeConfiguration(
            Path("/usr/bin/python3"), self.root / ("cleanup-state-" + uuid4().hex),
        )
        provider = HermesTaskProvider(SimpleNamespace(model=ctx.task.model), configuration)
        run = _NativeRun(configuration, ctx, SimpleNamespace(
            run_id="hermes-run:" + uuid4().hex, close=Mock(),
        ))
        self.addCleanup(shutil.rmtree, configuration.work_root)
        (run.home / "synthetic-private-copy.txt").write_text("Synthetic source material.")
        await self.tasks.start_runtime(ctx.claim, run.run_id, {"fixture":True})
        await ctx.checkpoint({
            "format":"jarvis.hermes.run.v1", "run_id":run.run_id, "phase":"terminal",
            "outcome":{"status":"succeeded", "payload":{"answer":"Synthetic result."}, "reason":None},
        })
        return provider, run

    async def test_private_cleanup_failure_reconciles_same_run_without_resetting_usage(self):
        ctx = await self.claim()
        provider, run = await self.cleanup_runtime(ctx)
        reservation = await ctx.reserve("synthetic-unknown-call", input_tokens=31, output_tokens=100)
        await ctx.budgets.submit(ctx.claim, reservation["id"], parent_provider_handle=run.run_id)
        with patch("app.tasks.hermes.shutil.rmtree", side_effect=PermissionError("synthetic")):
            with self.assertRaises(TaskError):
                await run.stop()
        await self.tasks.abandon(ctx.claim, reason="synthetic_cleanup_failure")
        recovery = await self.tasks.claim_recovery("cleanup-recovery")
        self.assertEqual(recovery.attempt_id, ctx.claim.attempt_id)
        recovered = TaskRunner(self.tasks, {}).context(recovery)
        with patch("app.tasks.hermes.shutil.rmtree", side_effect=PermissionError("synthetic")):
            result = await provider.reconcile(recovered, cancel=False)
        self.assertEqual(result.state, "unknown")
        await self.tasks.resolve_recovery(recovery, result)
        self.assertTrue(run.home.exists())
        self.assertIsNone((await self.tasks.get(self.f.owner, ctx.task.id)).result_payload_id)
        result = await provider.reconcile(recovered, cancel=False)
        self.assertEqual(result.state, "terminal")
        self.assertEqual(result.outcome.payload, {"answer":"Synthetic result."})
        self.assertFalse(run.home.exists())
        await self.tasks.resolve_recovery(recovery, result)
        self.assertEqual(await ctx.budgets.usage(self.f.owner, ctx.task.id), {"input":31, "output":100, "steps":0})
        async with self.f.pool.acquire() as connection:
            rows = await connection.fetch("select id,provider_handle from jarvis_task_attempts where owner_id=$1 and task_id=$2",
                                          self.f.owner, ctx.task.id)
        self.assertEqual([(row["id"], row["provider_handle"]) for row in rows], [(ctx.claim.attempt_id, run.run_id)])
        self.assertEqual(json.loads((run.directory / "run.json").read_text())["private_home_cleanup"], {"state":"complete"})

    async def test_legacy_exited_receipt_does_not_bypass_private_copy_cleanup(self):
        ctx = await self.claim()
        provider, run = await self.cleanup_runtime(ctx)
        run.marker.update(phase="exited", exit_code=0)
        del run.marker["private_home_cleanup"]
        run.write_marker()
        await self.tasks.abandon(ctx.claim, reason="synthetic_legacy_exit")
        recovery = await self.tasks.claim_recovery("legacy-cleanup-recovery")
        result = await provider.reconcile(TaskRunner(self.tasks, {}).context(recovery), cancel=False)
        self.assertEqual(result.state, "terminal")
        self.assertEqual(result.outcome.payload, {"answer":"Synthetic result."})
        self.assertFalse(run.home.exists())
        self.assertEqual(json.loads((run.directory / "run.json").read_text())["private_home_cleanup"], {"state":"complete"})

    def provider(self, *, prepare=None):
        return HermesTaskProvider(
            FrontierConfiguration(model=os.environ["JARVIS_S04_MODEL"],reasoning_effort="low",
                                  api_key=JarvisSettings().OPENAI_API_KEY),
            HermesRuntimeConfiguration(Path(os.environ["JARVIS_HERMES_PYTHON"]),self.root/"native-state"),
            prepare=prepare,
        )

    async def execute(self, provider, ctx):
        outcome = await asyncio.wait_for(provider.execute(ctx),150)
        self.assertTrue(await self.tasks.finish(ctx.claim,outcome))
        (self.root/("native-outcome-"+str(ctx.task.id)+".json")).write_text(json.dumps({
            "task_id":str(ctx.task.id),"status":outcome.status,"payload":outcome.payload,
            "usage":await ctx.budgets.usage(self.f.owner,ctx.task.id),
        },indent=2)+"\n")
        return outcome

    async def grant_note(self, parameters):
        async with self.f.pool.acquire() as connection:
            request = await self.tasks.payload(connection,self.f.owner,{"text":"Create this exact synthetic note."})
            event = await connection.fetchval(
                "insert into jarvis_context_events(owner_id,session_id,source_namespace,source_event_id,kind,finality,trust,payload_id)"
                " values($1,$2,'fixture-hermes',$3,'user_request','final','user_input',$4) returning id",
                self.f.owner,self.session,str(uuid4()),request)
            scope = await self.tasks.payload(connection,self.f.owner,parameters)
            grant = await connection.fetchval(
                "insert into jarvis_operation_grants(owner_id,capability,operation_key,request_event_id,request_revision,"
                "scope_revision,scope_payload_id,expires_at) values($1,'notes.create',$2,$3,1,1,$4,"
                "clock_timestamp()+interval '1 hour') returning id",self.f.owner,str(uuid4()),event,scope)
        return grant

    @unittest.skipUnless(os.getenv("JARVIS_HERMES_PYTHON"),"Explicit native Hermes interpreter required")
    async def test_native_exact_note_write_and_read_receipt(self):
        parameters = {"title":"Synthetic delegated note","body":"Fixture marker "+uuid4().hex}
        grant = await self.grant_note(parameters)
        ctx = await self.claim("Create the exact authorized note, then use search_notes to confirm it exists. "
                               "Include the actual saved note ID in your final answer and report its completed operation reference.",
                               grant_ids=(grant,))
        notes = NoteRepository(self.f.pool)
        provider = self.provider(prepare=HermesContextFactory(
            modules=(NoteToolModule(notes),),read_tools=("search_notes",),capability_tools={"notes.create":"create_note"},
        ))
        outcome = await self.execute(provider,ctx)
        self.assertEqual(outcome.status,TaskStatus.SUCCEEDED)
        async with self.f.pool.acquire() as connection:
            rows = await connection.fetch("select id,body from jarvis_notes where owner_id=$1",self.f.owner)
        self.assertEqual(len(rows),1)
        self.assertEqual(rows[0]["body"],parameters["body"])
        self.assertIn(str(rows[0]["id"]),outcome.payload["answer"])
        self.assertEqual(outcome.payload["completed_operation_refs"],["op_"+grant.hex])
        names = [receipt["name"] for receipt in outcome.payload["tool_receipts"]]
        self.assertIn("create_note",names)
        self.assertIn("search_notes",names)
        self.assertTrue(all(receipt["call_id"] for receipt in outcome.payload["tool_receipts"]))

    @unittest.skipUnless(os.getenv("JARVIS_HERMES_PYTHON"),"Explicit native Hermes interpreter required")
    async def test_native_reads_pixels_not_label_in_metadata(self):
        from PIL import Image, ImageDraw, ImageFont
        label = "LANTERN " + str(uuid4().int % 90000 + 10000)
        image = Image.new("RGB",(550,220),"white")
        ImageDraw.Draw(image).text((24,76),label,font=ImageFont.load_default(size=42),fill="black")
        buffer = BytesIO()
        image.save(buffer,format="JPEG",quality=95)
        pixels = buffer.getvalue()
        reference = ImageReference(frame_id=uuid4(),owner_id=self.f.owner,session_id=self.session,
            source="historical",captured_at=datetime.now(UTC),content_type="image/jpeg",width=550,height=220)
        source = await ContextRepository(self.tasks).append(self.f.owner,ContextIngress(
            session_id=self.session,source_namespace="fixture-hermes-image",source_event_id=str(uuid4()),
            receipt_id=str(uuid4()),kind="observation",trust="sensor",payload={"images":[reference.to_metadata()]},
        ))
        ctx = await self.claim("Read the short label in the original image referenced by the session observation. "
                              "You must retrieve the original image; the label is not in the metadata. Cite that observation.")
        class FixturePixels:
            async def resolve_image(inner, *, owner_id, reference):
                self.assertEqual(owner_id,self.f.owner)
                async def revalidate():
                    row = await ContextRepository(self.tasks).source_for_claim(ctx.claim,source.id)
                    self.assertEqual(row.revision,source.revision)
                    self.assertEqual(row.availability,"available")
                return ResolvedImage(reference=reference,image_bytes=pixels,revalidate=revalidate)
        async def images(context, repository):
            return TaskContextImages(repository,historical=FixturePixels())
        outcome = await self.execute(self.provider(prepare=HermesContextFactory(images_factory=images)),ctx)
        self.assertEqual(outcome.status,TaskStatus.SUCCEEDED)
        self.assertIn(label,outcome.payload["answer"])
        async with self.f.pool.acquire() as connection:
            images_sent = await connection.fetchval(
                "select exists(select 1 from jarvis_context_payloads where owner_id=$1 "
                "and content->>'format'='jarvis.hermes.model_request.v1' "
                "and content->'images' @> $2::jsonb)",self.f.owner,json.dumps([{"frame_id":str(reference.frame_id)}]))
        self.assertTrue(images_sent,"Native provider requests must contain the actual registered original image")
        self.assertIn(str(source.id),[item["event_id"] for item in outcome.payload["evidence"]])
        self.assertIn("read_task_context_image",[item["name"] for item in outcome.payload["tool_receipts"]])

    @unittest.skipUnless(os.getenv("JARVIS_HERMES_PYTHON"),"Explicit native Hermes interpreter required")
    async def test_native_missing_image_asks_for_input_without_fabricating(self):
        ctx = await self.claim("Read the code in my camera image. No image has actually been provided to this task. "
                              "Ask me for that missing image before attempting the task.")
        outcome = await self.execute(self.provider(),ctx)
        self.assertEqual(outcome.status,TaskStatus.WAITING)
        self.assertTrue(outcome.payload["uncertainties"])
        self.assertFalse(outcome.payload["completed_operation_refs"])

    @unittest.skipUnless(os.getenv("JARVIS_HERMES_PYTHON"),"Explicit native Hermes interpreter required")
    async def test_two_native_tasks_overlap_while_status_read_remains_available(self):
        first = await self.claim("Compute 17 times 19 and answer briefly.")
        second = await self.claim("Compute 23 times 29 and answer briefly.")
        provider = self.provider()
        jobs = [asyncio.create_task(self.execute(provider,ctx)) for ctx in (first,second)]
        maximum = 0
        try:
            for _ in range(2000):
                maximum = max(maximum,sum(run.process is not None and run.process.returncode is None
                                          for run in provider._runs.values()))
                if maximum==2:
                    status = await asyncio.wait_for(self.tasks.get(self.f.owner,first.task.id),3)
                    self.assertEqual(status.status,TaskStatus.RUNNING)
                    break
                if all(job.done() for job in jobs):
                    break
                await asyncio.sleep(.025)
            outcomes = await asyncio.gather(*jobs)
        finally:
            for job in jobs:
                if not job.done(): job.cancel()
            await asyncio.gather(*jobs,return_exceptions=True)
        self.assertEqual(maximum,2)
        self.assertIn("323",outcomes[0].payload["answer"])
        self.assertIn("667",outcomes[1].payload["answer"])

    async def interrupt_native_note(self, *, after_effect):
        parameters = {"title":"Synthetic crash boundary","body":"Crash marker "+uuid4().hex}
        grant = await self.grant_note(parameters)
        ctx = await self.claim(
            "Create the exact authorized note once. Return its accepted receipt and actual note ID. "
            "If its earlier submission was reconciled, use that outcome without submitting it again.",
            grant_ids=(grant,),
        )
        class PausingNotes(NoteRepository):
            def __init__(inner, pool):
                super().__init__(pool)
                inner.entered, inner.release = asyncio.Event(), asyncio.Event()
            async def create_note(inner, **arguments):
                if after_effect:
                    note = await super().create_note(**arguments)
                inner.entered.set()
                await inner.release.wait()
                return note if after_effect else await super().create_note(**arguments)
        notes = PausingNotes(self.f.pool)
        provider = self.provider(prepare=HermesContextFactory(
            modules=(NoteToolModule(notes),),read_tools=("search_notes",),
            capability_tools={"notes.create":"create_note"},
        ))
        job = asyncio.create_task(provider.execute(ctx))
        entered = asyncio.create_task(notes.entered.wait())
        try:
            done, _ = await asyncio.wait((entered,job),timeout=100,return_when=asyncio.FIRST_COMPLETED)
            if job in done:
                self.fail("Native task ended before the intended effect boundary: "+repr(await job))
            self.assertIn(entered,done,"Native task did not reach the requested effect boundary")
            run = provider._runs[ctx.claim.attempt_id]
            self.assertIsNone(run.process.returncode)
            os.killpg(run.process.pid,signal.SIGKILL)
            with self.assertRaises(TaskError):
                await asyncio.wait_for(job,15)
            self.assertFalse(run.home.exists())
        finally:
            if not job.done(): job.cancel()
            entered.cancel()
            await asyncio.gather(job,entered,return_exceptions=True)
            notes.release.set()

        await self.tasks.abandon(ctx.claim,reason="synthetic_native_crash")
        recovery = await self.tasks.claim_recovery("replacement-native-worker")
        self.assertEqual(recovery.attempt_id,ctx.claim.attempt_id)
        # A new provider instance must recover from durable metadata, not _runs.
        replacement = self.provider(prepare=HermesContextFactory(
            modules=(NoteToolModule(NoteRepository(self.f.pool)),),read_tools=("search_notes",),
            capability_tools={"notes.create":"create_note"},
        ))
        result = await replacement.reconcile(TaskRunner(self.tasks,{}).context(recovery),cancel=False)
        self.assertEqual(result.state,"terminal")
        await self.tasks.resolve_recovery(recovery,result)
        task = await self.tasks.get(self.f.owner,ctx.task.id)
        self.assertEqual(task.status,TaskStatus.UNKNOWN)
        with self.assertRaises(TaskError):
            await self.tasks.mutate(self.f.owner,task.id,"resume",
                                   TaskMutation(request_id=uuid4(),expected_revision=task.revision))
        async with self.f.pool.acquire() as connection:
            action = await connection.fetchrow(
                "select * from jarvis_task_actions where owner_id=$1 and task_id=$2",self.f.owner,task.id)
            rows = await connection.fetch("select id,body from jarvis_notes where owner_id=$1",self.f.owner)
        self.assertEqual(action["status"],"outcome_unknown")
        self.assertEqual(len(rows),1 if after_effect else 0)
        proof = {"method":"Independent synthetic handler boundary and database read",
                 "note_id":str(rows[0]["id"]) if rows else None}
        await ctx.actions.reconcile(self.f.owner,task.id,action["id"],
                                    outcome="accepted" if after_effect else "not_submitted",proof=proof)
        current = await self.tasks.get(self.f.owner,task.id)
        self.assertEqual(current.status,TaskStatus.PAUSED)
        await self.tasks.mutate(self.f.owner,current.id,"resume",
                                TaskMutation(request_id=uuid4(),expected_revision=current.revision))
        resumed = await self.tasks.claim_next("resumed-native-worker",("frontier",))
        self.assertEqual(resumed.task.id,task.id)
        self.assertNotEqual(resumed.attempt_id,ctx.claim.attempt_id)
        final = await self.execute(replacement,TaskRunner(self.tasks,{}).context(resumed))
        async with self.f.pool.acquire() as connection:
            count = await connection.fetchval("select count(*) from jarvis_notes where owner_id=$1",self.f.owner)
        self.assertEqual(count,len(rows),"Restart must not create another effect")
        if after_effect:
            self.assertEqual(final.status,TaskStatus.SUCCEEDED)
            self.assertIn(str(rows[0]["id"]),final.payload["answer"])
            self.assertEqual(final.payload["completed_operation_refs"],["op_"+grant.hex])
        else:
            self.assertIn(final.status,(TaskStatus.PAUSED,TaskStatus.WAITING,TaskStatus.FAILED))
            self.assertFalse(final.payload["completed_operation_refs"])
        (self.root/("native-recovery-"+str(task.id)+".json")).write_text(json.dumps({
            "task_id":str(task.id),"after_effect":after_effect,"action_id":str(action["id"]),
            "reconciled_outcome":"accepted" if after_effect else "not_submitted",
            "original_attempt":str(ctx.claim.attempt_id),"resumed_attempt":str(resumed.attempt_id),
            "note_count":count,"final_status":final.status,
        },indent=2)+"\n")

    @unittest.skipUnless(os.getenv("JARVIS_HERMES_PYTHON"),"Explicit native Hermes interpreter required")
    async def test_native_crash_after_effect_before_receipt_does_not_duplicate_note(self):
        await self.interrupt_native_note(after_effect=True)

    @unittest.skipUnless(os.getenv("JARVIS_HERMES_PYTHON"),"Explicit native Hermes interpreter required")
    async def test_native_crash_before_effect_does_not_fabricate_or_resubmit_completion(self):
        await self.interrupt_native_note(after_effect=False)

    async def change_native_task(self, operation):
        class PausingReads(NoteRepository):
            def __init__(inner, pool):
                super().__init__(pool)
                inner.entered = asyncio.Event()
                inner.release = asyncio.Event()
            async def search_notes(inner, **arguments):
                inner.entered.set()
                await inner.release.wait()
                return await super().search_notes(**arguments)
        notes = PausingReads(self.f.pool)
        provider = self.provider(prepare=HermesContextFactory(
            modules=(NoteToolModule(notes),),read_tools=("search_notes",),
        ))
        ctx = await self.claim("Use search_notes to list my saved notes, then report what you found.")
        job = asyncio.create_task(provider.execute(ctx))
        entered = asyncio.create_task(notes.entered.wait())
        try:
            done, _ = await asyncio.wait((entered,job),timeout=100,return_when=asyncio.FIRST_COMPLETED)
            if job in done:
                self.fail("Native task ended before the requested read: "+repr(await job))
            self.assertIn(entered,done)
            old_run = provider._runs[ctx.claim.attempt_id]
            new_message = "Replace the earlier note-listing assignment. Compute 23 times 29 directly and answer briefly."
            changed = await self.tasks.mutate(self.f.owner,ctx.task.id,operation,TaskMutation(
                request_id=uuid4(),expected_revision=ctx.task.revision,
                message=new_message if operation=="update" else "Stop this task.",
            ))
            recovery = await self.tasks.claim_recovery("revision-controller")
            self.assertEqual(recovery.task.revision,changed.revision)
            self.assertEqual(recovery.attempt_id,ctx.claim.attempt_id)
            result = await provider.reconcile(TaskRunner(self.tasks,{}).context(recovery),cancel=True)
            self.assertEqual(result.state,"terminal")
            await self.tasks.resolve_recovery(recovery,result)
            await asyncio.wait_for(asyncio.gather(job,return_exceptions=True),10)
            self.assertTrue(old_run.finished.is_set())
            self.assertIsNotNone(old_run.process.returncode)
            self.assertFalse(old_run.home.exists())
        finally:
            if not job.done(): job.cancel()
            entered.cancel()
            await asyncio.gather(job,entered,return_exceptions=True)
            notes.release.set()
        current = await self.tasks.get(self.f.owner,ctx.task.id)
        self.assertIsNone(current.result_payload_id,"The obsolete native answer must not become a current result")
        if operation=="cancel":
            self.assertEqual(current.status,TaskStatus.CANCELLED)
            self.assertIsNone(await self.tasks.claim_next("should-not-restart",("frontier",)))
        else:
            self.assertEqual(current.status,TaskStatus.QUEUED)
            next_claim = await self.tasks.claim_next("updated-native-worker",("frontier",))
            self.assertEqual(next_claim.task.requested_outcome,new_message)
            self.assertNotEqual(next_claim.attempt_id,ctx.claim.attempt_id)
            outcome = await self.execute(self.provider(),TaskRunner(self.tasks,{}).context(next_claim))
            self.assertEqual(outcome.status,TaskStatus.SUCCEEDED)
            self.assertIn("667",outcome.payload["answer"])
        (self.root/("native-"+operation+"-"+str(ctx.task.id)+".json")).write_text(json.dumps({
            "task_id":str(ctx.task.id),"operation":operation,"old_revision":ctx.task.revision,
            "new_revision":changed.revision,"obsolete_process_exit_code":old_run.process.returncode,
            "final_status":(await self.tasks.get(self.f.owner,ctx.task.id)).status,
        },indent=2)+"\n")

    @unittest.skipUnless(os.getenv("JARVIS_HERMES_PYTHON"),"Explicit native Hermes interpreter required")
    async def test_native_cancellation_stops_the_owned_process_and_cannot_publish(self):
        await self.change_native_task("cancel")

    @unittest.skipUnless(os.getenv("JARVIS_HERMES_PYTHON"),"Explicit native Hermes interpreter required")
    async def test_native_update_fences_old_work_and_continues_from_the_new_manifest(self):
        await self.change_native_task("update")
