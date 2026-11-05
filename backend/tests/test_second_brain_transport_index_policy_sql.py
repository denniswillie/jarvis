"""Isolated real SQL and canonical capture; provider/storage are controlled only."""
import os
from types import SimpleNamespace
import unittest
from uuid import uuid4

from app.knowledge.hindsight import HindsightError
from app.knowledge.indexing_policy import TRANSPORT_RECEIPT_EXCLUDED
from app.knowledge.ingestion import IngestionWorker
from app.knowledge.models import SourceError
from app.knowledge.observations import ObservationPublication
from app.knowledge.outbox import OutboxRepository
from app.knowledge.retrieval import SourceMemoryRetrieval
from app.perception.models import ObservationFrameLink
from app.perception.repository import PerceptionRepository
from app.tasks.context_capture import SessionContextCapture
from app.tasks.context_events import ContextIngress
import test_second_brain_image_scope as image_scope
import test_second_brain_ingestion_integration as ingestion
import test_second_brain_observation_publication as observations


@unittest.skipUnless(os.getenv("JARVIS_SECOND_BRAIN_TEST_DB_URL") and
                     os.getenv("JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"), "Explicit manifested database required")
class TransportPolicyDatabaseTests(unittest.IsolatedAsyncioTestCase):
    select = image_scope.ImageScopeDatabaseTests.select
    saved_reference = image_scope.ImageScopeDatabaseTests.saved_reference

    async def asyncSetUp(self):
        await image_scope.ImageScopeDatabaseTests.asyncSetUp(self)
        self.provider = ingestion.Provider()
        self.outbox = OutboxRepository(self.f.sources, retry_seconds=1)
        self.worker = IngestionWorker(self.outbox, self.provider)

    async def capture_receipts(self, reference, *, tool=False):
        capture = SessionContextCapture(self.history, owner_id=self.f.owner, session_id=self.session,
            provider="openai", audio_session_id="transport-policy", write_timeout=10)
        self.addAsyncCleanup(capture.close)
        if tool:
            capture.tool_outcome("audit-image", "get_memory_image", {"status":"success",
                "image_status":"selected", "image":reference.to_metadata()})
        capture.image_delivery(reference, "socket_written", "fixture-image")
        self.assertTrue(await capture.flush(), capture.issue)
        capture.image_delivery(reference, "provider_acknowledged", "fixture-image")
        self.assertTrue(await capture.flush(), capture.issue)
        async with self.f.pool.acquire() as connection:
            rows = await connection.fetch("""select s.id,s.area_id from jarvis_knowledge_sources s
                join jarvis_knowledge_context_sources cs on cs.owner_id=s.owner_id and cs.source_id=s.id
                join jarvis_context_events e on e.owner_id=cs.owner_id and e.id=cs.event_id
                where e.owner_id=$1 and e.source_namespace=$2 order by e.sequence""", self.f.owner, capture.namespace)
        return [await self.f.sources.read(self.f.owner, r["area_id"], r["id"]) for r in rows]

    async def drain(self):
        for _ in range(80):
            self.provider.complete()
            await self.f.ready_jobs()
            if await self.worker.step(self.f.owner, "controlled-policy-worker") is None:
                return
        self.fail("Bounded controlled ingestion did not settle")

    async def claim_document(self, document):
        for _ in range(40):
            claim = await self.outbox.claim(self.f.owner, "controlled-legacy-worker")
            self.assertIsNotNone(claim)
            if claim.document_id == document:
                return claim
            await self.worker._retain(claim)
        self.fail("Target claim not reached")

    async def test_skip_preserves_original_area_copies_audit_pixels_and_deletion(self):
        await self.select(self.project)
        file, reference = await self.saved_reference("policy-project")
        await self.select(self.f.area)
        copies = await self.capture_receipts(reference, tool=True)
        self.assertEqual(len(copies), 3)
        self.assertEqual({s.area_id for s in copies}, {self.project})
        receipts = [s for s in copies if s.source_type == "observation"]
        self.assertEqual(len(receipts), 2)
        self.assertTrue(all(s.semantic_index_exclusion == TRANSPORT_RECEIPT_EXCLUDED for s in receipts))
        self.assertTrue(all(s.source_time == self.frame.captured_at for s in receipts))
        await self.drain()
        for old in receipts:
            fresh = await self.f.sources.read(self.f.owner, self.project, old.source_id)
            self.assertEqual(fresh.content, old.content)
            self.assertEqual(fresh.ingestion_status, "cancelled")
            self.assertEqual(fresh.ingestion_reason, TRANSPORT_RECEIPT_EXCLUDED)
            self.assertNotIn(fresh.document_id, self.provider.documents)
            with self.assertRaises(SourceError):
                await self.f.sources.read(self.f.other, self.project, old.source_id)
            with self.assertRaises(SourceError):
                await self.f.sources.read(self.f.owner, self.f.area, old.source_id)
        snapshot = await self.f.sources.current_snapshot(self.f.owner, self.project)
        self.assertEqual(snapshot["policy_excluded_sources"], 2)
        self.assertGreaterEqual(snapshot["not_searchable_sources"], 2)
        resolved = await self.scoped_images.resolve_image(owner_id=self.f.owner, reference=reference)
        self.assertEqual(resolved.image_bytes, self.raw.image)
        self.assertTrue((await self.files.delete_source_file(owner_id=self.f.owner, file_id=file.id))["deleted"])
        for old in receipts:
            with self.assertRaises(SourceError):
                await self.f.sources.read(self.f.owner, self.project, old.source_id, revision=old.revision)
        self.assertFalse(self.storage.objects)

    async def test_dispatched_lost_ack_reconciles_but_semantic_recall_excludes_receipts(self):
        _, reference = await self.saved_reference("policy-legacy")
        copies = await self.capture_receipts(reference)
        item = copies[0]
        claim = await self.claim_document(item.document_id)
        # Controlled pre-policy dispatch through the genuine durable intent API.
        original = await self.outbox.prepare_retain(claim)
        self.provider.lose_ack = True
        with self.assertRaises(HindsightError):
            await self.provider.retain(original, claim.id)
        self.assertEqual(await self.outbox.failure(claim, "fixture_lost_ack", retryable=True), "outcome_unknown")
        await self.drain()
        fresh = await self.f.sources.read(self.f.owner, self.f.area, item.source_id)
        self.assertEqual(fresh.ingestion_status, "searchable")  # Actual controlled document reconciliation.
        self.assertEqual(fresh.semantic_index_exclusion, TRANSPORT_RECEIPT_EXCLUDED)
        self.assertEqual(self.provider.sent.count(str(claim.id)), 1)
        recalled = await SourceMemoryRetrieval(self.f.sources, self.provider).recall(self.f.owner, self.f.area, "image")
        self.assertGreaterEqual(recalled["policy_excluded_candidates"], 1)
        self.assertNotIn(item.document_id, [s["document_id"] for s in recalled["sources"]])
        self.assertEqual(fresh.content, item.content)
        async with self.f.pool.acquire() as connection:
            self.assertEqual(await connection.fetchval("select dispatch_count from jarvis_memory_outbox where id=$1", claim.id), 1)
        self.assertFalse(self.storage.objects == {})  # Retained storage was not deindexed/deleted.

    async def test_real_analyzer_positive_negative_and_forged_user_payload_remain_eligible(self):
        _, reference = await self.saved_reference("policy-findings")
        receipt = (await self.capture_receipts(reference))[0]
        forged = await self.history.append(self.f.owner, ContextIngress(session_id=self.session,
            source_namespace="voice:fixture-user", source_event_id="user-request", receipt_id="user-one",
            kind="user_transcript", trust="user_input", payload=receipt.content | {
                "text":"Synthetic user report quoting image transport metadata."}))
        unbound = await self.f.publish_payload(receipt.content, source_key="unbound-receipt-lookalike",
                                               source_type="observation", trust="sensor")
        self.assertIsNone(unbound.semantic_index_exclusion)
        publisher = ObservationPublication(self.history, owner_id=self.f.owner)
        repository = PerceptionRepository(self.f.pool, observation_publication=publisher)
        helper = SimpleNamespace(f=self.f, session=self.session, frame=self.frame.id, when=self.frame.captured_at)
        ids = []
        for summary in ("A core is on the board, position uncertain.", "No core is visible in this captured frame."):
            observation = observations.ObservationPublicationDatabaseTests.observation(helper, summary=summary)
            await repository.insert_observation(observation=observation,
                frame_links=(ObservationFrameLink(self.frame.id, 0, "primary"),))
            async with self.f.pool.acquire() as connection:
                identity = await connection.fetchval("""select v.source_id from jarvis_observation_sources o
                    join jarvis_knowledge_source_versions v on v.owner_id=o.owner_id and v.payload_id=o.payload_id
                    where o.owner_id=$1 and o.observation_id=$2""", self.f.owner, observation.id)
            ids.append(identity)
            self.assertIsNone((await self.f.sources.read(self.f.owner, self.f.area, identity)).semantic_index_exclusion)
        await self.drain()
        recalled = await SourceMemoryRetrieval(self.f.sources, self.provider, max_sources=32).recall(
            self.f.owner, self.f.area, "core")
        returned = {s["source_id"] for s in recalled["sources"]}
        self.assertTrue({str(identity) for identity in ids} <= returned)
        self.assertIn(str(unbound.source_id), returned)
        async with self.f.pool.acquire() as connection:
            user_id = await connection.fetchval("select source_id from jarvis_knowledge_context_sources where owner_id=$1 and event_id=$2",
                                               self.f.owner, forged.id)
        old = await self.f.sources.read(self.f.owner, self.f.area, user_id)
        self.assertIsNone(old.semantic_index_exclusion)
        self.assertIn(str(user_id), returned)
        await self.history.append(self.f.owner, ContextIngress(session_id=self.session,
            source_namespace="voice:fixture-user", source_event_id="user-request", receipt_id="user-two",
            kind="user_transcript", trust="user_input", finality="revised", expected_revision=forged.revision,
            payload={"text":"Corrected synthetic user report."}))
        with self.assertRaises(SourceError):
            await self.f.sources.revalidate(old)
        fresh = await self.f.sources.read(self.f.owner, self.f.area, user_id)
        self.assertGreater(fresh.revision, old.revision)
        await self.f.sources.delete_source(self.f.owner, self.f.area, user_id, expected_revision=fresh.revision)
        with self.assertRaises(SourceError):
            await self.f.sources.read(self.f.owner, self.f.area, user_id, revision=old.revision)
