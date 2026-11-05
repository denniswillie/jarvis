"""Recovery preparation: real source/outbox state, no native I/O in these tests."""
from contextlib import AsyncExitStack
import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from app.knowledge.lifecycle_journal import JournalCheckpoint, SourceLifecycleJournal
from app.knowledge.hindsight import HindsightClient, HindsightError
from app.knowledge.ingestion import IngestionWorker
from app.knowledge.models import SourceError
from app.knowledge.outbox import IngestionClaimLost, OutboxRepository
from app.knowledge.provider_recovery import ProviderRecoveryService
from evals.second_brain.source_fixture import SourceFixture, source_fixture
from evals.second_brain.hindsight_restore_trace import RecallTraceClient


class RecallObservationTests(unittest.IsolatedAsyncioTestCase):
    async def test_trace_observes_native_candidates_without_filtering_or_supplementing(self):
        client = RecallTraceClient("http://127.0.0.1:1", "fixture")
        self.addAsyncCleanup(client.close)
        original = {"results": [{"document_id": "old"}, {"document_id": "current"}, {"text": "uncited"}]}
        with patch.object(HindsightClient, "recall", AsyncMock(return_value=original)) as recall:
            result = await client.recall("fixture-bank", "question")
        self.assertIs(result, original)
        self.assertEqual(client.last_recall_document_ids, {"old", "current"})
        recall.assert_awaited_once_with("fixture-bank", "question", max_tokens=2400)


class ProviderRecoveryBoundsTests(unittest.TestCase):
    def test_invalid_scope_and_bounds_are_refused(self):
        checkpoint = JournalCheckpoint(owner_id="fixture", epoch=uuid4(), through_sequence=0)
        for owner, cursor, limit in (
            ("other", None, 1), ("fixture", "", 1), ("fixture", 3, 1),
            ("fixture", "x" * 201, 1), ("fixture", None, True),
            ("fixture", None, 0), ("fixture", None, 129),
        ):
            with self.subTest(owner=owner, cursor=cursor, limit=limit), self.assertRaises((ValueError, SourceError)):
                ProviderRecoveryService.validate_page(owner, checkpoint, cursor, limit)


@unittest.skipUnless(os.getenv("JARVIS_SECOND_BRAIN_TEST_DB_URL") and
                     os.getenv("JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"),
                     "Explicit isolated database fixture required")
class ProviderRecoveryDatabaseTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.stack = AsyncExitStack()
        self.addAsyncCleanup(self.stack.aclose)
        self.f = await self.stack.enter_async_context(source_fixture(
            os.environ["JARVIS_SECOND_BRAIN_TEST_DB_URL"],
            Path(os.environ["JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"]), test_name=self.id()))
        self.recovery = ProviderRecoveryService(self.f.sources)
        self.journal = SourceLifecycleJournal(self.f.pool)
        self.outbox = OutboxRepository(self.f.sources)

    async def pin(self):
        return await self.journal.checkpoint(self.f.owner)

    async def prepare(self, **changes):
        return await self.recovery.prepare_page(
            self.f.owner, self.f.area, **({"checkpoint": await self.pin()} | changes))

    async def rows(self):
        async with self.f.pool.acquire() as connection:
            return [dict(row) for row in await connection.fetch(
                "select * from jarvis_memory_outbox order by id")]

    async def test_existing_receipts_and_terminal_cleanup_inventory_survive(self):
        source = await self.f.publish("Private original")
        claim = await self.outbox.claim(self.f.owner, "fixture")
        await self.outbox.prepare_retain(claim)
        await self.outbox.settle(claim, "searchable")
        await self.f.sources.delete_source(self.f.owner, self.f.area, source.source_id, expected_revision=1)
        delete = await self.outbox.claim(self.f.owner, "fixture")
        inventory = (claim.id, uuid4())
        # Deliberately simulated provider receipt for this database-only test.
        await self.outbox.prepare_cleanup(delete, inventory)
        await self.outbox.settle(delete, "done")
        current = await self.f.publish("New current original", source_key="current")
        current_claim = await self.outbox.claim(self.f.owner, "fixture")
        self.assertEqual(current_claim.document_id, current.document_id)
        await self.outbox.prepare_retain(current_claim)
        await self.outbox.settle(current_claim, "searchable")
        before = {row["id"]: row for row in await self.rows()}
        result = await self.prepare()
        after = {row["id"]: row for row in await self.rows()}
        self.assertEqual(set(before), set(after))
        self.assertEqual(after[delete.id]["status"], "pending")
        self.assertEqual(tuple(after[delete.id]["cleanup_operation_ids"]), inventory)
        self.assertEqual(after[current_claim.id]["status"], "outcome_unknown")
        self.assertEqual(after[current_claim.id]["dispatch_count"], 1)
        self.assertEqual(result["changed"], {"retains": 1, "deletions": 1, "obsolete_retains_cancelled": 0})
        self.assertFalse(result["provider_contacted"])
        self.assertFalse(result["application_release_authorized"])
        self.assertNotIn("Private original", json.dumps(result))
        repeated = await self.prepare()
        self.assertEqual(await self.rows(), list(after.values()))
        self.assertEqual(sum(repeated["changed"].values()), 0)

    async def test_live_lease_refused_and_expired_claim_cannot_settle(self):
        await self.f.publish()
        claim = await self.outbox.claim(self.f.owner, "fixture")
        before = await self.rows()
        with self.assertRaises(SourceError) as error:
            await self.prepare()
        self.assertEqual(error.exception.code, "memory_recovery_live_lease")
        self.assertEqual(await self.rows(), before)
        async with self.f.pool.acquire() as connection:
            await connection.execute(
                "update jarvis_memory_outbox set lease_until=clock_timestamp()-interval '1 second' where id=$1", claim.id)
        await self.prepare()
        with self.assertRaises(IngestionClaimLost):
            await self.outbox.settle(claim, "searchable")
        row = (await self.rows())[0]
        self.assertEqual(row["id"], claim.id)
        self.assertGreater(row["fence"], claim.fence)

    async def test_checkpoint_epoch_head_and_cursor_scope_fail_without_mutation(self):
        pin = await self.pin()
        await self.f.publish()
        before = await self.rows()
        for options in (
            {"checkpoint": pin},
            {"checkpoint": (await self.pin()).model_copy(update={"epoch": uuid4()})},
            {"after_document_id": "missing-document"},
        ):
            with self.subTest(options=options), self.assertRaises(SourceError):
                await self.prepare(**options)
            self.assertEqual(await self.rows(), before)
        with self.assertRaises(SourceError):
            await self.recovery.prepare_page(self.f.other, self.f.area, checkpoint=await self.pin())

    async def test_bounded_preparation_covers_each_immutable_version(self):
        await self.f.publish("First")
        await self.f.publish("Corrected", expected_revision=1)
        await self.f.publish("Another", source_key="another")
        pin, cursor, considered = await self.pin(), None, 0
        for _ in range(3):
            result = await self.prepare(checkpoint=pin, after_document_id=cursor, limit=1)
            considered += result["documents_considered"]
            pin = JournalCheckpoint.model_validate(result["checkpoint"])
            cursor = result["next_after_document_id"]
        self.assertEqual(considered, 3)
        self.assertTrue(result["complete"])
        self.assertIsNone(cursor)
        self.assertEqual(await self.pin(), pin)

    async def test_missing_authoritative_receipt_rolls_back_whole_page(self):
        first = await self.f.publish("Keep original")
        async with self.f.pool.acquire() as connection:
            await connection.execute("delete from jarvis_memory_outbox where document_id=$1", first.document_id)
        before = await self.rows()
        with self.assertRaises(SourceError) as error:
            await self.prepare()
        self.assertEqual(error.exception.code, "memory_recovery_receipt_missing")
        self.assertEqual(await self.rows(), before)
        self.assertEqual((await self.f.sources.read(self.f.owner, self.f.area, first.source_id)).content, first.content)

    async def test_inactive_area_is_not_silently_physically_forgotten(self):
        await self.f.publish("Keep archived original")
        async with self.f.pool.acquire() as connection:
            await connection.execute(
                "update jarvis_knowledge_areas set status='archived',revision=revision+1 where owner_id=$1 and id=$2",
                self.f.owner, self.f.area)
        before = await self.rows()
        with self.assertRaises(SourceError) as error:
            await self.prepare()
        self.assertEqual(error.exception.code, "knowledge_area_unavailable")
        self.assertEqual(await self.rows(), before)

    async def test_populated_foreign_owner_is_unchanged(self):
        await self.f.publish("Primary owner")
        async with self.f.pool.acquire() as connection:
            other_area = await connection.fetchval(
                "insert into jarvis_knowledge_areas(owner_id,name,domain) values($1,'Other','test') returning id",
                self.f.other)
        other = SourceFixture(self.f.fixture, self.f.pool, self.f.sources,
                              self.f.other, self.f.owner, other_area)
        control = await other.publish("Private other-owner control")
        for owner in (self.f.owner, self.f.other):
            claim = await self.outbox.claim(owner, "fixture")
            await self.outbox.prepare_retain(claim)
            await self.outbox.settle(claim, "searchable")
        before = [row for row in await self.rows() if row["owner_id"] == self.f.other]
        result = await self.prepare()
        self.assertEqual(result["documents_considered"], 1)
        self.assertEqual([row for row in await self.rows() if row["owner_id"] == self.f.other], before)
        self.assertEqual((await self.f.sources.read(self.f.other, other_area, control.source_id)).content,
                         control.content)

    async def test_conflicting_current_cleanup_refused_without_queue_mutation(self):
        source = await self.f.publish()
        async with self.f.pool.acquire() as connection:
            await connection.execute(
                "insert into jarvis_memory_outbox(owner_id,document_id,generation,operation) values($1,$2,1,'delete')",
                self.f.owner, source.document_id)
        before = await self.rows()
        with self.assertRaises(SourceError) as error:
            await self.prepare()
        self.assertEqual(error.exception.code, "memory_recovery_conflicting_cleanup")
        self.assertEqual(await self.rows(), before)

    async def test_missing_native_receipt_does_not_turn_unknown_dispatch_into_erasure(self):
        source = await self.f.publish()
        claim = await self.outbox.claim(self.f.owner, "fixture")
        await self.outbox.prepare_retain(claim)
        await self.outbox.failure(claim, "simulated_lost_ack", retryable=True)
        await self.f.sources.delete_source(self.f.owner, self.f.area, source.source_id, expected_revision=1)
        before_ids = {row["id"] for row in await self.rows()}
        await self.prepare()
        provider = SimpleNamespace(
            terminal_operations=AsyncMock(side_effect=HindsightError("hindsight_retention_outcome_unknown", retryable=True)),
            delete_document=AsyncMock(), erase_operations=AsyncMock(), retain=AsyncMock())
        worker = IngestionWorker(self.outbox, provider)
        self.assertEqual(await worker.step(self.f.owner, "fixture"), "outcome_unknown")
        after = await self.rows()
        self.assertEqual({row["id"] for row in after}, before_ids)
        retain = next(row for row in after if row["operation"] == "retain")
        cleanup = next(row for row in after if row["operation"] == "delete")
        self.assertEqual((retain["id"], retain["dispatch_count"]), (claim.id, 1))
        self.assertIsNone(cleanup["cleanup_operation_ids"])
        self.assertEqual(cleanup["status"], "outcome_unknown")
        provider.delete_document.assert_not_awaited()
        provider.erase_operations.assert_not_awaited()
        provider.retain.assert_not_awaited()
