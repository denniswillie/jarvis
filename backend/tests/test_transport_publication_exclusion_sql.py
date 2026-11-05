"""Actual manifested SQL: early exclusion preserves canonical evidence and fences."""
from app.knowledge.indexing_policy import TRANSPORT_RECEIPT_EXCLUDED
from app.tasks.context_capture import SessionContextCapture
from test_second_brain_publication_integration import SourcePublicationDatabaseTests


class TransportPublicationSQLTests(SourcePublicationDatabaseTests):
    def image_receipt(self, state="socket_written", delivery="fixture-delivery"):
        capture = SessionContextCapture(object(), owner_id=self.f.owner, session_id=self.session,
            provider="openai", audio_session_id="fixture-audio")
        emitted = []
        capture._put = lambda source, **kw: emitted.append(capture._event(source, **kw))
        capture.image_delivery(None, state, delivery)
        return emitted[0]

    async def outbox(self, source):
        async with self.f.pool.acquire() as connection:
            return dict(await connection.fetchrow("""select status,last_error,dispatch_count,fence
                from jarvis_memory_outbox where owner_id=$1 and document_id=$2
                and operation='retain'""", self.f.owner, source.document_id))

    async def test_receipts_are_immediately_excluded_without_losing_originals(self):
        for state in ("socket_written", "provider_acknowledged", "write_outcome_unknown",
                      "not_written", "provider_rejected"):
            event = self.image_receipt(state)
            record = await self.context.append(self.f.owner, event)
            item = await self.mapped(record)
            self.assertEqual(await self.outbox(item), {"status": "cancelled",
                "last_error": TRANSPORT_RECEIPT_EXCLUDED, "dispatch_count": 0, "fence": 0})
            self.assertEqual(item.content, record.content)
            self.assertEqual(item.context_event_id, record.id)
            self.assertEqual(item.payload_id, record.payload_id)
            self.assertEqual(item.semantic_index_exclusion, TRANSPORT_RECEIPT_EXCLUDED)
            before = await self.counts()
            self.assertEqual((await self.context.append(self.f.owner, event)).id, record.id)
            self.assertEqual(await self.counts(), before)
        for event in (self.ingress(key="user", payload=dict(event.payload) | {"text": "User-supplied receipt-shaped data"}),
                      self.ingress(key="finding", kind="observation", trust="derived",
                          payload={"summary": "A visible fixture finding"})):
            ordinary = await self.mapped(await self.context.append(self.f.owner, event))
            self.assertEqual((await self.outbox(ordinary))["status"], "pending")
            self.assertIsNone(ordinary.semantic_index_exclusion)

    async def test_dispatch_and_existing_claim_are_never_settled_by_publication(self):
        record = await self.context.append(self.f.owner, self.image_receipt())
        item = await self.mapped(record)
        for status, dispatches, claimed in (("pending", 1, False), ("accepted", 1, False),
                                          ("leased", 0, True)):
            async with self.f.pool.acquire() as connection:
                await connection.execute("""update jarvis_memory_outbox set status=$3,
                    dispatch_count=$4,last_error='prior-operation',
                    worker_id=case when $5 then 'fixture-worker' else null end,
                    lease_until=case when $5 then clock_timestamp()+interval '1 minute' else null end
                    where owner_id=$1 and document_id=$2 and operation='retain'""",
                    self.f.owner, item.document_id, status, dispatches, claimed)
            async with self.publisher.transaction(self.tasks, self.f.owner) as connection:
                await self.publisher.publish(connection, self.f.owner, record)
            result = await self.outbox(item)
            self.assertEqual((result["status"], result["dispatch_count"], result["last_error"]),
                             (status, dispatches, "prior-operation"))

    async def test_correction_is_indexable_and_original_removal_still_propagates(self):
        event = self.image_receipt()
        record = await self.context.append(self.f.owner, event)
        old = await self.mapped(record)
        corrected = await self.context.append(self.f.owner, event.model_copy(update={
            "receipt_id": "fixture-correction", "expected_revision": 1, "finality": "revised",
            "payload": dict(event.payload) | {"summary": "New substantive visual evidence"}}))
        current = await self.mapped(corrected)
        self.assertEqual(current.source_id, old.source_id)
        self.assertEqual(current.revision, 2)
        self.assertEqual((await self.outbox(current))["status"], "pending")
        self.assertIsNone(current.semantic_index_exclusion)
        await self.f.sources.delete_source(self.f.owner, self.f.area, current.source_id,
                                           expected_revision=current.revision)
        async with self.f.pool.acquire() as connection:
            available = await connection.fetchval("""select count(*) from jarvis_context_payloads
                where owner_id=$1 and id=any($2::uuid[]) and availability='available'""",
                self.f.owner, [old.payload_id, current.payload_id])
        self.assertEqual(available, 0)
