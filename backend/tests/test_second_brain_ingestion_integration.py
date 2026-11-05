"""Real Jarvis DB intents and recovery, with explicitly simulated provider I/O."""
import asyncio
from contextlib import AsyncExitStack
import os
from pathlib import Path
import unittest
from uuid import uuid4

import asyncpg

from app.knowledge.hindsight import HindsightClient, HindsightError, original_document, document_metadata
from app.knowledge.ingestion import IngestionWorker
from app.knowledge.models import SourceError
from app.knowledge.outbox import OutboxRepository, IngestionClaimLost
from app.knowledge.retrieval import SourceMemoryRetrieval
from evals.second_brain.source_fixture import source_fixture


class Provider:
    verify_document = staticmethod(HindsightClient.verify_document)
    child_operation_ids = staticmethod(HindsightClient.child_operation_ids)
    terminal_operations = HindsightClient.terminal_operations
    erase_operations = HindsightClient.erase_operations
    def __init__(self):
        self.operations, self.documents, self.sent = {}, {}, []
        self.lose_ack = False
        self.before_document = None
        self.gate = None
        self.deleted_operations = []
        self.lose_delete_ack_for = None
        self.cancel_after_delete_for = None
    async def ensure_bank(self, bank):
        if self.gate is not None: await self.gate.wait()
    async def operation(self, bank, operation_id):
        return self.operations.get(str(operation_id))
    async def retain(self, source, operation_id):
        key = str(operation_id)
        self.sent.append(key)
        self.operations.setdefault(key, {"operation_id":key,"status":"processing"})
        self.documents[source.document_id] = dict(
            id=source.document_id,bank_id=source.bank_name,original_text=original_document(source),
            document_metadata=document_metadata(source))
        if self.lose_ack:
            self.lose_ack = False
            raise HindsightError("lost_fixture_ack",retryable=True)
        return {"async":True,"operation_id":key}
    async def document(self, bank, document_id):
        if self.before_document is not None:
            hook,self.before_document=self.before_document,None
            await hook()
        return self.documents.get(document_id)
    async def delete_document(self, bank, document_id):
        self.documents.pop(document_id,None)
    async def delete_terminal_operation(self, bank, operation_id):
        identity = str(operation_id)
        operation = self.operations.get(identity)
        if operation is None:
            return False
        if operation["status"] not in {"completed", "failed", "cancelled"}:
            raise HindsightError("fixture_operation_not_terminal")
        self.operations.pop(identity)
        self.deleted_operations.append(identity)
        if self.lose_delete_ack_for == identity:
            self.lose_delete_ack_for = None
            raise HindsightError("fixture_lost_operation_delete_ack", retryable=True)
        if self.cancel_after_delete_for == identity:
            self.cancel_after_delete_for = None
            raise asyncio.CancelledError()
        return True
    async def recall(self, bank, query):
        return {"results":[{"document_id":doc,"text":"Untrusted vendor interpretation"}
                           for doc in [*self.documents,"unknown-source"]]}
    def complete(self):
        for operation in self.operations.values(): operation["status"]="completed"


@unittest.skipUnless(os.getenv("JARVIS_SECOND_BRAIN_TEST_DB_URL") and
                     os.getenv("JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"),
                     "Explicit second-brain DB fixtures required")
class IngestionDatabaseTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        stack=AsyncExitStack()
        self.addAsyncCleanup(stack.aclose)
        self.f=await stack.enter_async_context(source_fixture(
            os.environ["JARVIS_SECOND_BRAIN_TEST_DB_URL"],
            Path(os.environ["JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"]),test_name=self.id()))
        self.provider=Provider()
        self.outbox=OutboxRepository(self.f.sources,retry_seconds=1)
        self.worker=IngestionWorker(self.outbox,self.provider)
    async def step(self):
        await self.f.ready_jobs()
        return await self.worker.step(self.f.owner,"fixture-worker")

    async def admissions(self):
        return (await self.outbox.diagnostics(self.f.owner))["native_admissions"]

    def admission_limit(self, limit):
        self.outbox = OutboxRepository(self.f.sources, retry_seconds=3600, max_in_flight=limit)
        self.worker = IngestionWorker(self.outbox, self.provider)

    async def due(self, claim):
        async with self.f.pool.acquire() as connection:
            await connection.execute(
                "update jarvis_memory_outbox set next_attempt_at=clock_timestamp() where id=$1", claim.id,
            )

    async def test_admission_cap_preserves_originals_and_allows_existing_reconciliation(self):
        self.admission_limit(2)
        sources = [await self.f.publish(source_key="bounded-"+str(i)) for i in range(3)]
        admitted = []
        for _ in range(2):
            claim = await self.outbox.claim(self.f.owner, "bounded")
            admitted.append(claim)
            await self.outbox.prepare_retain(claim)
            await self.outbox.settle(claim, "accepted")
        self.assertIsNone(await self.outbox.claim(self.f.owner, "bounded"))
        diagnostics = await self.admissions()
        self.assertEqual(diagnostics["reserved"], 2)
        self.assertTrue(diagnostics["at_capacity"])
        self.assertEqual((await self.outbox.diagnostics(self.f.other))["native_admissions"]["reserved"], 0)
        pending = await self.f.sources.read(self.f.owner, self.f.area, sources[-1].source_id)
        self.assertEqual(pending.ingestion_status, "pending")
        self.assertEqual(pending.content, {"text": "Synthetic user report"})
        await self.due(admitted[0])
        resumed = await self.outbox.claim(self.f.owner, "reconcile")
        self.assertEqual(resumed.id, admitted[0].id)
        await self.outbox.settle(resumed, "searchable")
        next_claim = await self.outbox.claim(self.f.owner, "bounded")
        self.assertEqual(next_claim.document_id, sources[-1].document_id)
        self.assertEqual((await self.admissions())["reserved"], 2)

    async def test_concurrent_claims_share_one_owner_admission_limit(self):
        self.admission_limit(1)
        await self.f.publish(source_key="concurrent-one")
        await self.f.publish(source_key="concurrent-two")
        claims = await asyncio.gather(
            self.outbox.claim(self.f.owner, "first"), self.outbox.claim(self.f.owner, "second"),
        )
        self.assertEqual(sum(claim is not None for claim in claims), 1)
        self.assertEqual((await self.admissions())["reserved"], 1)

    async def test_expired_pre_dispatch_lease_is_reserved_and_can_recover_at_capacity(self):
        self.admission_limit(1)
        await self.f.publish(source_key="leased")
        await self.f.publish(source_key="waiting")
        first = await self.outbox.claim(self.f.owner, "first")
        self.assertIsNone(await self.outbox.claim(self.f.owner, "second"))
        async with self.f.pool.acquire() as connection:
            await connection.execute(
                "update jarvis_memory_outbox set lease_until=clock_timestamp()-interval '1 second' where id=$1",
                first.id,
            )
        recovered = await self.outbox.claim(self.f.owner, "recovery")
        self.assertEqual(recovered.id, first.id)
        self.assertGreater(recovered.fence, first.fence)
        self.assertEqual((await self.admissions())["reserved"], 1)

    async def test_native_cleanup_bypasses_full_cap_and_releases_cancelled_retention(self):
        self.admission_limit(1)
        source = await self.f.publish(source_key="delete-admitted")
        self.assertEqual(await self.worker.step(self.f.owner, "worker"), "accepted")
        waiting = await self.f.publish(source_key="waiting")
        self.assertIsNone(await self.outbox.claim(self.f.owner, "worker"))
        await self.f.sources.delete_source(self.f.owner, self.f.area, source.source_id, expected_revision=1)
        self.assertEqual((await self.admissions())["reserved"], 1)
        self.provider.complete()
        self.assertEqual(await self.worker.step(self.f.owner, "worker"), "done")
        self.assertEqual((await self.admissions())["reserved"], 0)
        self.assertNotIn(source.document_id, self.provider.documents)
        self.assertEqual((await self.outbox.claim(self.f.owner, "worker")).document_id, waiting.document_id)

    async def test_unknown_terminal_failure_does_not_release_dispatched_reservation(self):
        self.admission_limit(1)
        await self.f.publish(source_key="unknown")
        claim = await self.outbox.claim(self.f.owner, "worker")
        await self.outbox.prepare_retain(claim)
        await self.outbox.failure(claim, "lost_native_ack", retryable=False)
        await self.f.publish(source_key="waiting")
        self.assertIsNone(await self.outbox.claim(self.f.owner, "worker"))
        self.assertEqual((await self.admissions())["reserved"], 1)

    async def test_unclassified_failed_dispatch_still_reserves_capacity(self):
        self.admission_limit(1)
        await self.f.publish(source_key="unclassified")
        claim = await self.outbox.claim(self.f.owner, "worker")
        await self.outbox.prepare_retain(claim)
        await self.outbox.settle(claim, "failed")
        await self.f.publish(source_key="waiting")
        self.assertEqual((await self.admissions())["reserved"], 1)
        self.assertIsNone(await self.outbox.claim(self.f.owner, "worker"))

    async def test_known_native_terminal_failure_releases_capacity(self):
        self.admission_limit(1)
        await self.f.publish(source_key="native-failed")
        claim = await self.outbox.claim(self.f.owner, "worker")
        await self.outbox.prepare_retain(claim)
        await self.outbox.settle(claim, "failed", reason="hindsight_retention_failed")
        waiting = await self.f.publish(source_key="waiting")
        self.assertEqual((await self.admissions())["reserved"], 0)
        self.assertEqual((await self.outbox.claim(self.f.owner, "worker")).document_id, waiting.document_id)

    async def test_policy_rejection_before_dispatch_does_not_consume_capacity(self):
        self.admission_limit(1)
        await self.f.publish(source_key="policy-rejected")
        claim = await self.outbox.claim(self.f.owner, "worker")
        await self.outbox.failure(claim, "hindsight_bank_policy_not_satisfied", retryable=False)
        waiting = await self.f.publish(source_key="waiting")
        self.assertEqual((await self.admissions())["reserved"], 0)
        self.assertEqual((await self.outbox.claim(self.f.owner, "worker")).document_id, waiting.document_id)

    async def test_lost_ack_recovers_original_operation_and_does_not_resubmit(self):
        source=await self.f.publish()
        self.provider.lose_ack=True
        self.assertEqual(await self.step(),"outcome_unknown")
        self.assertEqual((await self.f.sources.read(self.f.owner,self.f.area,source.source_id)).ingestion_status,
                         "outcome_unknown")
        self.worker=IngestionWorker(OutboxRepository(self.f.sources,retry_seconds=1),self.provider)
        self.assertEqual(await self.step(),"accepted")
        self.provider.complete()
        self.assertEqual(await self.step(),"searchable")
        self.assertEqual(len(self.provider.sent),1)

    async def test_expired_worker_cannot_settle_success_under_new_fence(self):
        await self.f.publish()
        old=await self.outbox.claim(self.f.owner,"old")
        async with self.f.pool.acquire() as connection:
            await connection.execute("update jarvis_memory_outbox set lease_until=clock_timestamp()-interval '1 second' where id=$1",old.id)
        fresh=await self.outbox.claim(self.f.owner,"new")
        self.assertEqual(old.id,fresh.id)
        self.assertGreater(fresh.fence,old.fence)
        with self.assertRaises(IngestionClaimLost):
            await self.outbox.settle(old,"searchable")

    async def test_delete_before_dispatch_never_submits_original(self):
        source=await self.f.publish()
        await self.f.sources.delete_source(self.f.owner,self.f.area,source.source_id,expected_revision=1)
        self.assertEqual(await self.step(),"done")
        self.assertFalse(self.provider.sent)
        self.assertIsNone(await self.step())

    async def test_late_acceptance_after_deletion_is_not_publishable_and_cleanup_waits(self):
        source=await self.f.publish()
        claim=await self.outbox.claim(self.f.owner,"old")
        admitted=await self.outbox.prepare_retain(claim)
        await self.f.sources.delete_source(self.f.owner,self.f.area,source.source_id,expected_revision=1)
        await self.provider.retain(admitted,claim.id)
        self.assertEqual(await self.outbox.settle(claim,"accepted"),"cancelled")
        self.assertEqual(await self.step(),"accepted")
        self.assertIn(source.document_id,self.provider.documents)
        self.provider.complete()
        self.assertEqual(await self.step(),"done")
        self.assertNotIn(source.document_id,self.provider.documents)
        with self.assertRaises(SourceError):
            await self.f.sources.revalidate(source)

    async def test_uncertain_dispatch_is_not_falsely_reported_cleaned(self):
        source=await self.f.publish()
        claim=await self.outbox.claim(self.f.owner,"old")
        await self.outbox.prepare_retain(claim)
        await self.f.sources.delete_source(self.f.owner,self.f.area,source.source_id,expected_revision=1)
        self.assertEqual(await self.step(),"outcome_unknown")

    async def prepare_native_cleanup(self):
        source = await self.f.publish()
        self.assertEqual(await self.step(), "accepted")
        root = self.provider.sent[-1]
        child = str(uuid4())
        self.provider.operations[root]["child_operations"] = [{"operation_id": child, "status": "completed"}]
        self.provider.operations[child] = {"operation_id": child, "status": "completed"}
        self.provider.complete()
        self.assertEqual(await self.step(), "searchable")
        await self.f.sources.delete_source(self.f.owner, self.f.area, source.source_id, expected_revision=1)
        return source, root, child

    async def test_operation_inventory_is_durable_immutable_and_preserves_unrelated_records(self):
        source, root, child = await self.prepare_native_cleanup()
        unrelated = str(uuid4())
        self.provider.operations[unrelated] = {"operation_id": unrelated, "status": "completed"}
        self.assertEqual(await self.step(), "done")
        self.assertEqual(self.provider.deleted_operations, [child, root])
        self.assertIn(unrelated, self.provider.operations)
        async with self.f.pool.acquire() as connection:
            row = await connection.fetchrow(
                "select id,cleanup_operation_ids from jarvis_memory_outbox where document_id=$1 and operation='delete'",
                source.document_id,
            )
            self.assertEqual([str(item) for item in row["cleanup_operation_ids"]], [root, child])
            with self.assertRaises(asyncpg.CheckViolationError):
                await connection.execute(
                    "update jarvis_memory_outbox set cleanup_operation_ids=null where id=$1", row["id"],
                )

    async def test_lost_child_delete_ack_recovers_with_persisted_inventory(self):
        _, root, child = await self.prepare_native_cleanup()
        self.provider.lose_delete_ack_for = child
        self.assertEqual(await self.step(), "outcome_unknown")
        self.assertNotIn(child, self.provider.operations)
        self.assertIn(root, self.provider.operations)
        self.worker = IngestionWorker(OutboxRepository(self.f.sources, retry_seconds=1), self.provider)
        self.assertEqual(await self.step(), "done")
        self.assertNotIn(root, self.provider.operations)
        self.assertEqual(self.provider.deleted_operations, [child, root])

    async def test_lost_root_delete_ack_is_not_mistaken_for_unknown_retention(self):
        _, root, child = await self.prepare_native_cleanup()
        self.provider.lose_delete_ack_for = root
        self.assertEqual(await self.step(), "outcome_unknown")
        self.assertNotIn(root, self.provider.operations)
        self.worker = IngestionWorker(OutboxRepository(self.f.sources, retry_seconds=1), self.provider)
        self.assertEqual(await self.step(), "done")
        self.assertEqual(self.provider.deleted_operations, [child, root])

    async def test_cancelled_cleanup_resumes_after_lease_expiry_without_resurrecting_original(self):
        source, root, _ = await self.prepare_native_cleanup()
        self.provider.cancel_after_delete_for = root
        with self.assertRaises(asyncio.CancelledError):
            await self.step()
        async with self.f.pool.acquire() as connection:
            await connection.execute(
                "update jarvis_memory_outbox set lease_until=clock_timestamp()-interval '1 second' "
                "where document_id=$1 and operation='delete'", source.document_id,
            )
        self.worker = IngestionWorker(OutboxRepository(self.f.sources, retry_seconds=1), self.provider)
        self.assertEqual(await self.step(), "done")
        self.assertNotIn(source.document_id, self.provider.documents)
        self.assertEqual(len(self.provider.sent), 1)

    async def test_active_child_prevents_cleanup_preparation_and_remote_deletion(self):
        source, _, child = await self.prepare_native_cleanup()
        self.provider.operations[child]["status"] = "processing"
        self.assertEqual(await self.step(), "accepted")
        self.assertIn(source.document_id, self.provider.documents)
        self.assertFalse(self.provider.deleted_operations)
        async with self.f.pool.acquire() as connection:
            self.assertIsNone(await connection.fetchval(
                "select cleanup_operation_ids from jarvis_memory_outbox where document_id=$1 and operation='delete'",
                source.document_id,
            ))
        self.provider.complete()
        self.assertEqual(await self.step(), "done")

    async def test_cleanup_preparation_rejects_foreign_root_and_expired_fence(self):
        _, root, child = await self.prepare_native_cleanup()
        old = await self.outbox.claim(self.f.owner, "old-cleaner")
        with self.assertRaises(SourceError):
            await self.outbox.prepare_cleanup(old, (uuid4(),))
        async with self.f.pool.acquire() as connection:
            await connection.execute(
                "update jarvis_memory_outbox set lease_until=clock_timestamp()-interval '1 second' where id=$1", old.id,
            )
        fresh = await self.outbox.claim(self.f.owner, "new-cleaner")
        with self.assertRaises(IngestionClaimLost):
            await self.outbox.prepare_cleanup(old, (root, child))
        with self.assertRaises(SourceError):
            await self.outbox.settle(fresh, "done")
        self.assertEqual([str(item) for item in await self.outbox.prepare_cleanup(fresh, (root, child))], [root, child])

    async def test_source_expansion_preserves_originals_and_rechecks_after_network_wait(self):
        source=await self.f.publish("Reported keys in bag, not observed there")
        await self.step()
        self.provider.complete()
        await self.step()
        retrieval=SourceMemoryRetrieval(self.f.sources,self.provider,timeout_seconds=15)
        result=await retrieval.recall(self.f.owner,self.f.area,"Where are the keys?")
        self.assertEqual(result["sources"][0]["trust"],"user_input")
        self.assertEqual(result["sources"][0]["content"]["text"],"Reported keys in bag, not observed there")
        self.assertEqual(result["rejected_candidates"],1)
        self.assertNotIn("Untrusted vendor interpretation",str(result))
        self.provider.before_document=lambda: self.f.sources.delete_source(
            self.f.owner,self.f.area,source.source_id,expected_revision=1)
        result=await retrieval.recall(self.f.owner,self.f.area,"Where are the keys?")
        self.assertFalse(result["sources"])

    async def test_pending_native_processing_does_not_hold_database_transaction(self):
        source=await self.f.publish()
        self.provider.gate=asyncio.Event()
        running=asyncio.create_task(self.worker.step(self.f.owner,"waiting"))
        try:
            await asyncio.sleep(0)
            original=await asyncio.wait_for(self.f.sources.read(self.f.owner,self.f.area,source.source_id),timeout=3)
            self.assertEqual(original.content,source.content)
        finally:
            running.cancel()
            await asyncio.gather(running,return_exceptions=True)
