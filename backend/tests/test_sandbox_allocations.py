"""Real PostgreSQL ownership/submission boundaries; no E2B API calls."""
import asyncio
from dataclasses import replace
from datetime import UTC, datetime, timedelta
import os
from pathlib import Path
import unittest
from uuid import uuid4

from app.runtime.feature_config import TaskLimits
from app.sandbox.allocations import SandboxAllocationRepository
from app.sandbox.models import SandboxInfo, SandboxPolicy
from app.tasks.models import ClaimLost, TaskCreate, TaskError, TaskMutation, json_text
from app.tasks.repository import TaskRepository
from evals.second_brain.source_fixture import source_fixture


@unittest.skipUnless(os.getenv("JARVIS_SECOND_BRAIN_TEST_DB_URL") and
    os.getenv("JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"), "Explicit manifested database fixtures required")
class SandboxAllocationDatabaseTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.f = await self.enterAsyncContext(source_fixture(os.environ["JARVIS_SECOND_BRAIN_TEST_DB_URL"],
            Path(os.environ["JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"]), test_name=self.id()))
        self.tasks = TaskRepository(self.f.pool, TaskLimits(lease_seconds=300))
        self.allocations = SandboxAllocationRepository(self.tasks)
        self.policy = SandboxPolicy()
        async with self.f.pool.acquire() as c:
            payload = await self.tasks.payload(c,self.f.owner,{"text":"Run a synthetic task in an offline sandbox."})
            self.event = await c.fetchval("""insert into jarvis_context_events
                (owner_id,source_namespace,source_event_id,kind,finality,trust,payload_id)
                values($1,'fixture-sandbox',$2,'user_request','final','user_input',$3) returning id""",
                self.f.owner,str(uuid4()),payload)
            scope = await self.tasks.payload(c,self.f.owner,self.policy.model_dump(mode="json"))
            self.grant = await c.fetchval("""insert into jarvis_operation_grants
                (owner_id,capability,operation_key,request_event_id,request_revision,scope_revision,scope_payload_id,expires_at)
                values($1,'sandbox.execute',$2,$3,1,1,$4,clock_timestamp()+interval '1 hour') returning id""",
                self.f.owner,str(uuid4()),self.event,scope)
        self.task = await self.tasks.create(self.f.owner,TaskCreate(request_id=uuid4(),title="Synthetic sandbox",
            requested_outcome="Compute synthetic output offline.",knowledge_area_id=self.f.area,
            grant_ids=(self.grant,)),model="fixture-no-model")
        self.claim = await self.tasks.claim_next("fixture-sandbox-worker",("frontier",))

    async def reserved(self):
        return await self.allocations.reserve(self.claim,self.grant)

    def receipt(self, row, **changes):
        return SandboxInfo(sandbox_id="fixture-sandbox-"+row["id"].hex, scope=self.allocations.scope(row),
            policy=self.policy,state="running",expires_at=datetime.now(UTC)+timedelta(minutes=5),
            template_id="fixture-base").model_copy(update=changes)

    async def submitted(self):
        row = await self.reserved()
        token = await self.allocations.begin_creation(self.claim,row["id"])
        return row,token

    async def test_concurrent_reservations_share_one_allocation(self):
        rows = await asyncio.gather(self.reserved(),self.reserved())
        self.assertEqual(rows[0]["id"],rows[1]["id"])
        self.assertIsNone(rows[0]["submission_token"])
        self.assertEqual(rows[0]["phase"],"reserved")

    async def test_only_one_concurrent_submission_gets_authority(self):
        row = await self.reserved()
        results = await asyncio.gather(*[self.allocations.begin_creation(self.claim,row["id"]) for _ in range(2)],
                                      return_exceptions=True)
        self.assertEqual(sum(isinstance(value,TaskError) for value in results),1)
        error = next(value for value in results if isinstance(value,TaskError))
        self.assertEqual(error.code,"sandbox_creation_already_submitted")

    async def test_receipt_preparation_and_current_scope(self):
        row,token = await self.submitted()
        result = await self.allocations.record_created(self.f.owner,row["id"],token,self.receipt(row))
        self.assertEqual(result["phase"],"preparing")
        ready = await self.allocations.ready(self.claim,row["id"])
        self.assertEqual(ready["phase"],"ready")
        self.assertEqual((await self.allocations.current(self.claim,row["id"]))["sandbox_id"],ready["sandbox_id"])
        await self.allocations.creation_uncertain(self.f.owner,row["id"],token)
        self.assertEqual((await self.allocations.current(self.claim,row["id"]))["phase"],"ready")

    async def test_uncertain_creation_cannot_be_retried(self):
        row,token = await self.submitted()
        await self.allocations.creation_uncertain(self.f.owner,row["id"],token)
        with self.assertRaises(TaskError):
            await self.allocations.begin_creation(self.claim,row["id"])
        self.assertEqual((await self.reserved())["phase"],"uncertain")

    async def test_wrong_owner_scope_and_submission_tokens_are_rejected(self):
        row,token = await self.submitted()
        for owner,identity,handle in ((self.f.other,token,self.receipt(row)),
            (self.f.owner,uuid4(),self.receipt(row)),
            (self.f.owner,token,self.receipt(row,scope=self.allocations.scope(row).model_copy(update={"attempt_id":uuid4()})))):
            with self.subTest(owner=owner,identity=identity):
                with self.assertRaises(TaskError):
                    await self.allocations.record_created(owner,row["id"],identity,handle)

    async def test_second_native_vm_cannot_replace_the_recorded_receipt(self):
        row,token = await self.submitted()
        handle = self.receipt(row)
        await self.allocations.record_created(self.f.owner,row["id"],token,handle)
        with self.assertRaises(TaskError) as caught:
            await self.allocations.record_created(self.f.owner,row["id"],token,handle.model_copy(update={"sandbox_id":"another-vm"}))
        self.assertEqual(caught.exception.code,"sandbox_receipt_conflict")

    async def test_cancelled_task_late_receipt_requires_cleanup_not_reactivation(self):
        row,token = await self.submitted()
        await self.tasks.mutate(self.f.owner,self.task.id,"cancel",TaskMutation(request_id=uuid4(),expected_revision=1))
        result = await self.allocations.record_created(self.f.owner,row["id"],token,self.receipt(row))
        self.assertEqual(result["phase"],"cleanup_requested")
        with self.assertRaises(ClaimLost):
            await self.allocations.ready(self.claim,row["id"])

    async def test_cleanup_before_late_receipt_never_restores_ready(self):
        row,token = await self.submitted()
        await self.allocations.request_cleanup(self.f.owner,row["id"])
        result = await self.allocations.record_created(self.f.owner,row["id"],token,self.receipt(row))
        self.assertEqual(result["phase"],"cleanup_requested")
        with self.assertRaises(TaskError):
            await self.allocations.ready(self.claim,row["id"])

    async def test_unsubmitted_intent_can_be_cleaned_without_provider_claims(self):
        row = await self.reserved()
        result = await self.allocations.request_cleanup(self.f.owner,row["id"])
        self.assertEqual(result["phase"],"cleaned")
        with self.assertRaises(TaskError):
            await self.allocations.begin_creation(self.claim,row["id"])

    async def test_stale_fence_and_expired_deadline_cannot_submit(self):
        row = await self.reserved()
        with self.assertRaises(ClaimLost):
            await self.allocations.begin_creation(replace(self.claim,fence=self.claim.fence+1),row["id"])
        async with self.f.pool.acquire() as c:
            await c.execute("update jarvis_sandbox_allocations set deadline=clock_timestamp()-interval '1 second' where id=$1",row["id"])
        with self.assertRaises(TaskError):
            await self.allocations.begin_creation(self.claim,row["id"])

    async def test_revoked_grant_blocks_access_but_keeps_cleanup_identity(self):
        row,token = await self.submitted()
        async with self.f.pool.acquire() as c:
            await c.execute("update jarvis_operation_grants set status='revoked' where id=$1",self.grant)
        result = await self.allocations.record_created(self.f.owner,row["id"],token,self.receipt(row))
        self.assertEqual(result["phase"],"cleanup_requested")
        with self.assertRaises(ClaimLost):
            await self.allocations.ready(self.claim,row["id"])
        self.assertEqual(self.allocations.scope(result),self.allocations.scope(row))
