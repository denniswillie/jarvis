"""Real durable cleanup races with a controlled provider, not fake DB acceptance."""
import asyncio
from datetime import UTC, datetime, timedelta
import os
from pathlib import Path
import unittest
from uuid import uuid4

import asyncpg

from app.sandbox.cleanup import SandboxCleaner, SandboxCleanupRepository
from app.sandbox.models import E2BConfiguration, SandboxError, SandboxInfo
from app.tasks.models import TaskError, TaskMutation
from evals.second_brain.sandbox_fixture import sandbox_task_fixture


class Provider:
    configuration = E2BConfiguration(api_key="synthetic-test-key",request_timeout_seconds=1)

    def __init__(self):
        self.resources,self.kills = {},[]
        self.list_error = False
        self.entered,self.release = asyncio.Event(),None

    async def discover(self,scope,policy):
        if self.list_error:
            raise SandboxError("synthetic_discovery_uncertain")
        return [key for key,value in self.resources.items() if value.scope==scope and value.policy==policy]

    async def info(self,identity,scope,policy):
        if identity not in self.resources:
            raise SandboxError("sandbox_not_found")
        value = self.resources[identity]
        if value.scope!=scope or value.policy!=policy:
            raise SandboxError("sandbox_scope_mismatch")
        return value

    async def kill(self,identity,scope,policy):
        await self.info(identity,scope,policy)
        self.kills.append(identity)
        self.entered.set()
        if self.release is not None:
            await self.release.wait()
        del self.resources[identity]
        return True


@unittest.skipUnless(os.getenv("JARVIS_SECOND_BRAIN_TEST_DB_URL") and
    os.getenv("JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"),"Explicit manifested database fixtures required")
class SandboxCleanupDatabaseTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.f = await self.enterAsyncContext(sandbox_task_fixture(os.environ["JARVIS_SECOND_BRAIN_TEST_DB_URL"],
            Path(os.environ["JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"]),test_name=self.id()))
        self.owner = self.f.source.owner
        self.repository = SandboxCleanupRepository(self.f.allocations,retry_seconds=1)
        self.provider = Provider()
        self.cleaner = SandboxCleaner(self.repository,self.provider)

    async def allocation(self,*,receipt=True,exists=True):
        row = await self.f.allocations.reserve(self.f.claim,self.f.grant)
        token = await self.f.allocations.begin_creation(self.f.claim,row["id"])
        handle = SandboxInfo(sandbox_id="fixture-"+row["id"].hex,scope=self.f.allocations.scope(row),
            policy=self.f.policy,state="running",expires_at=datetime.now(UTC)+timedelta(minutes=5),template_id="fixture-base")
        if exists:
            self.provider.resources[handle.sandbox_id] = handle
        if receipt:
            await self.f.allocations.record_created(self.owner,row["id"],token,handle)
            await self.f.allocations.ready(self.f.claim,row["id"])
        else:
            await self.f.allocations.creation_uncertain(self.owner,row["id"],token)
        return row,token,handle

    async def row(self,identity):
        async with self.f.source.pool.acquire() as connection:
            return dict(await connection.fetchrow("select * from jarvis_sandbox_allocations where id=$1",identity))

    async def due(self,identity,*,expire_lease=False):
        async with self.f.source.pool.acquire() as connection:
            await connection.execute("update jarvis_sandbox_allocations set next_cleanup_at=clock_timestamp() where id=$1",identity)
            if expire_lease:
                await connection.execute("update jarvis_sandbox_allocations set cleanup_lease_until=clock_timestamp()-interval '1 second' where id=$1",identity)

    async def cancel(self):
        await self.f.tasks.mutate(self.owner,self.f.task.id,"cancel",TaskMutation(request_id=uuid4(),expected_revision=1))

    async def test_cancelled_task_cleans_native_resource_without_explicit_cleanup_request(self):
        row,_,handle = await self.allocation()
        await self.cancel()
        result = await self.cleaner.run_once(self.owner,"cleanup-one")
        self.assertEqual(result["state"],"cleaned")
        self.assertEqual(self.provider.kills,[handle.sandbox_id])
        self.assertEqual((await self.row(row["id"]))["cleanup_observed_ids"],[handle.sandbox_id])

    async def test_deadline_cleans_paused_vm_without_resuming_it(self):
        row,_,handle = await self.allocation()
        self.provider.resources[handle.sandbox_id] = handle.model_copy(update={"state":"paused"})
        async with self.f.source.pool.acquire() as c:
            await c.execute("update jarvis_sandbox_allocations set deadline=clock_timestamp()-interval '1 second' where id=$1",row["id"])
        self.assertEqual((await self.cleaner.run_once(self.owner,"cleanup-one"))["state"],"cleaned")

    async def test_revocation_cleans_only_exact_owner_allocation(self):
        row,_,handle = await self.allocation()
        foreign = handle.model_copy(update={"sandbox_id":"foreign-vm",
            "scope":handle.scope.model_copy(update={"owner_id":self.f.source.other})})
        self.provider.resources[foreign.sandbox_id] = foreign
        async with self.f.source.pool.acquire() as c:
            await c.execute("update jarvis_operation_grants set status='revoked' where id=$1",self.f.grant)
        self.assertEqual((await self.cleaner.run_once(self.owner,"cleanup-one"))["state"],"cleaned")
        self.assertEqual(list(self.provider.resources),["foreign-vm"])

    async def test_lost_create_receipt_is_discovered_persisted_and_cleaned(self):
        row,_,handle = await self.allocation(receipt=False)
        await self.f.allocations.request_cleanup(self.owner,row["id"])
        self.assertEqual((await self.cleaner.run_once(self.owner,"cleanup-one"))["state"],"cleaned")
        saved = await self.row(row["id"])
        self.assertIsNone(saved["sandbox_id"])
        self.assertEqual(saved["cleanup_observed_ids"],[handle.sandbox_id])

    async def test_empty_discovery_remains_uncertain_and_late_vm_is_cleaned_later(self):
        row,_,handle = await self.allocation(receipt=False,exists=False)
        await self.f.allocations.request_cleanup(self.owner,row["id"])
        result = await self.cleaner.run_once(self.owner,"cleanup-one")
        self.assertEqual(result["error"],"sandbox_creation_outcome_unresolved")
        with self.assertRaises(TaskError):
            await self.f.allocations.begin_creation(self.f.claim,row["id"])
        self.provider.resources[handle.sandbox_id] = handle
        await self.due(row["id"])
        self.assertEqual((await self.cleaner.run_once(self.owner,"cleanup-two"))["state"],"cleaned")

    async def test_discovery_failure_still_cleans_known_handle_but_does_not_claim_completion(self):
        row,_,handle = await self.allocation()
        await self.cancel()
        self.provider.list_error = True
        first = await self.cleaner.run_once(self.owner,"cleanup-one")
        self.assertEqual(first["state"],"cleanup_requested")
        self.assertEqual(self.provider.kills,[handle.sandbox_id])
        self.provider.list_error = False
        await self.due(row["id"])
        self.assertEqual((await self.cleaner.run_once(self.owner,"cleanup-two"))["state"],"cleaned")

    async def test_concurrent_cleaners_do_not_duplicate_the_live_claim(self):
        row,_,handle = await self.allocation()
        await self.cancel()
        results = await asyncio.gather(self.cleaner.run_once(self.owner,"cleanup-one"),
                                       self.cleaner.run_once(self.owner,"cleanup-two"))
        self.assertEqual(sum(result is not None for result in results),1)
        self.assertEqual(self.provider.kills,[handle.sandbox_id])

    async def test_replaced_cleanup_fence_cannot_authorize_or_finalize(self):
        row,_,handle = await self.allocation()
        await self.cancel()
        old = await self.repository.claim_next(self.owner,"old")
        await self.due(row["id"],expire_lease=True)
        new = await self.repository.claim_next(self.owner,"new")
        self.assertGreater(new.generation,old.generation)
        for operation in (self.repository.authorize(old),
            self.repository.finish(old,absent_ids=[handle.sandbox_id],discovery_empty=True)):
            with self.assertRaises(SandboxError) as caught:
                await operation
            self.assertEqual(caught.exception.code,"sandbox_cleanup_claim_lost")
        self.assertEqual((await self.cleaner.clean(new))["state"],"cleaned")

    async def test_task_lease_loss_preserves_bounded_native_recovery_window(self):
        row,_,handle = await self.allocation()
        async with self.f.source.pool.acquire() as c:
            await c.execute("update jarvis_task_attempts set lease_until=clock_timestamp()-interval '1 second' where id=$1",self.f.claim.attempt_id)
        self.assertIsNone(await self.cleaner.run_once(self.owner,"cleanup-one"))
        self.assertIn(handle.sandbox_id,self.provider.resources)
        self.assertEqual((await self.row(row["id"]))["phase"],"ready")

    async def test_cleanup_worker_interruption_preserves_discovered_handles_for_restart(self):
        row,_,handle = await self.allocation(receipt=False)
        await self.cancel()
        self.provider.release = asyncio.Event()
        pending = asyncio.create_task(self.cleaner.run_once(self.owner,"old-worker"))
        try:
            await asyncio.wait_for(self.provider.entered.wait(),5)
            pending.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await pending
            self.assertEqual((await self.row(row["id"]))["cleanup_observed_ids"],[handle.sandbox_id])
            await self.due(row["id"],expire_lease=True)
            self.provider.release.set()
            self.assertEqual((await self.cleaner.run_once(self.owner,"new-worker"))["state"],"cleaned")
        finally:
            self.provider.release.set()
            if not pending.done():
                pending.cancel()
            await asyncio.gather(pending,return_exceptions=True)

    async def test_late_receipt_after_cleanup_snapshot_prevents_false_absence(self):
        row,token,handle = await self.allocation(receipt=False)
        await self.cancel()
        claim = await self.repository.claim_next(self.owner,"cleanup-one")
        await self.f.allocations.record_created(self.owner,row["id"],token,handle)
        result = await self.repository.finish(claim,absent_ids=(),discovery_empty=True)
        self.assertEqual(result["state"],"cleanup_requested")
        await self.due(row["id"])
        self.assertEqual((await self.cleaner.run_once(self.owner,"cleanup-two"))["state"],"cleaned")

    async def test_sql_cannot_reassign_native_identity_or_restore_execution_after_cleanup(self):
        row,_,_ = await self.allocation()
        async with self.f.source.pool.acquire() as c:
            for sql in ("set sandbox_id='another-vm'","set submission_token=gen_random_uuid()",
                        "set deadline=deadline+interval '1 hour'","set policy='{}'::jsonb"):
                with self.subTest(sql=sql),self.assertRaises(asyncpg.CheckViolationError):
                    await c.execute("update jarvis_sandbox_allocations "+sql+" where id=$1",row["id"])
        await self.cancel()
        await self.cleaner.run_once(self.owner,"cleanup-one")
        async with self.f.source.pool.acquire() as c:
            with self.assertRaises(asyncpg.CheckViolationError):
                await c.execute("update jarvis_sandbox_allocations set phase='ready' where id=$1",row["id"])
        self.assertEqual((await self.f.allocations.request_cleanup(self.owner,row["id"]))["phase"],"cleaned")

    async def test_cancelled_unsubmitted_intent_needs_no_provider_mutation(self):
        row = await self.f.allocations.reserve(self.f.claim,self.f.grant)
        await self.cancel()
        self.assertIsNone(await self.cleaner.run_once(self.owner,"cleanup-one"))
        self.assertEqual((await self.row(row["id"]))["phase"],"cleaned")
        self.assertEqual(self.provider.kills,[])
