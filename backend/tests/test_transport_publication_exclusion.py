"""Publication policy seam; mocks do not qualify SQL atomicity or latency."""
from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock
from uuid import uuid4

from app.knowledge.indexing_policy import TRANSPORT_RECEIPT_EXCLUDED
from app.knowledge.publication import CanonicalSourcePublication
from test_second_brain_transport_index_policy import receipt, source


class TransportPublicationTests(unittest.IsolatedAsyncioTestCase):
    def setup_publication(self, **changes):
        row, content = receipt()
        item = source(source_type="observation", trust="sensor", content=content)
        record = SimpleNamespace(owner_id=item.owner_id, payload_id=item.payload_id,
            content=content, source_namespace=row["context_source_namespace"],
            source_event_id=row["context_source_event_id"], id=row["context_event_id"],
            revision=1, kind="observation", trust="sensor", is_current=True,
            finality="final", availability="available", source_time=item.source_time,
            session_id=None)
        for key, value in changes.items():
            setattr(record, key, value)
        connection = SimpleNamespace(fetchval=AsyncMock(side_effect=["active", True]),
            fetchrow=AsyncMock(return_value=None), execute=AsyncMock())
        sources = SimpleNamespace(publish_in_transaction=AsyncMock(return_value=item))
        return CanonicalSourcePublication(sources, item.area_id), connection, record, item

    async def test_exact_receipt_binds_and_skips_in_one_statement(self):
        publisher, connection, record, item = self.setup_publication()
        result = await publisher.publish(connection, item.owner_id, record)
        self.assertEqual(result.ingestion_status, "cancelled")
        self.assertEqual(result.semantic_index_exclusion, TRANSPORT_RECEIPT_EXCLUDED)
        connection.execute.assert_not_awaited()
        args = connection.fetchval.call_args.args
        self.assertEqual(args[1:], (item.owner_id, item.source_id, item.revision,
            record.id, record.revision, item.document_id, item.generation,
            TRANSPORT_RECEIPT_EXCLUDED, record.payload_id))
        for guard in ("returning *", "from inserted", "o.dispatch_count=0",
                      "o.status='pending'", "o.worker_id is null", "o.lease_until is null",
                      "s.current_revision=v.revision", "b.event_revision=$5"):
            self.assertIn(guard, args[0])

    async def test_already_claimed_or_dispatched_row_is_not_reported_cancelled(self):
        publisher, connection, record, item = self.setup_publication()
        connection.fetchval.side_effect = ["active", None]
        self.assertIs(await publisher.publish(connection, item.owner_id, record), item)

    async def test_non_receipt_and_mismatched_source_keep_ordinary_publication(self):
        for change in ({"trust": "user_input", "kind": "user_transcript"},
                       {"source_event_id": "forged"}, {"payload_id": uuid4()},
                       {"content": {"summary": "A genuine visual finding"}}):
            with self.subTest(change=change):
                publisher, connection, record, item = self.setup_publication(**change)
                self.assertIs(await publisher.publish(connection, item.owner_id, record), item)
                connection.execute.assert_awaited_once()
                self.assertEqual(connection.fetchval.await_count, 1)

