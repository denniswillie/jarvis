"""Offline source-policy tests; controlled I/O is not SQL or vendor evidence."""
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock
from uuid import uuid4

from app.knowledge.current_context import CurrentSourceContext
from app.knowledge.indexing_policy import indexing_exclusion, TRANSPORT_RECEIPT_EXCLUDED
from app.knowledge.ingestion import IngestionWorker
from app.knowledge.models import SourceError, SourceVersion
from app.knowledge.outbox import IngestionClaimLost, OutboxRepository
from app.knowledge.retrieval import SourceMemoryRetrieval
from app.tasks.context_capture import SessionContextCapture
from app.tasks.models import digest


def receipt(state="socket_written"):
    capture = SessionContextCapture(object(), owner_id="fixture", session_id=uuid4(),
                                    provider="openai", audio_session_id="fixture-audio")
    emitted = []
    capture._put = lambda source, **kw: emitted.append(capture._event(source, **kw))
    capture.image_delivery(None, state, "fixture-delivery")
    event = emitted[0]
    row = {"context_binding_matches": True, "context_event_id": uuid4(),
           "source_type": event.kind, "trust": event.trust,
           "context_kind": event.kind, "context_trust": event.trust,
           "context_source_namespace": event.source_namespace,
           "context_source_event_id": event.source_event_id,
           "source_namespace": "context:" + digest({"namespace": event.source_namespace}),
           "source_key": event.source_event_id}
    return row, dict(event.payload)


def source(**changes):
    return SourceVersion(**(dict(owner_id="fixture", area_id=uuid4(), source_id=uuid4(),
        revision=1, generation=1, payload_id=uuid4(), document_id="document", bank_name="bank",
        content_digest="a"*64, source_type="observation", trust="derived", source_time=datetime.now(UTC),
        time_basis="synthetic", recorded_at=datetime.now(UTC), content={"summary":"synthetic finding"},
        is_current=True, ingestion_status="pending") | changes))


class TransportPolicyTests(unittest.IsolatedAsyncioTestCase):
    def test_real_capture_envelopes_and_forged_user_payloads(self):
        for state in ("socket_written", "provider_acknowledged", "write_outcome_unknown", "not_written", "provider_rejected"):
            with self.subTest(state=state):
                row, content = receipt(state)
                self.assertEqual(indexing_exclusion(row, content), TRANSPORT_RECEIPT_EXCLUDED)
        row, content = receipt()
        for key, value in {"context_binding_matches": False, "context_event_id": None,
                           "source_type":"user_transcript", "trust":"user_input",
                           "context_kind":"tool_outcome", "context_trust":"derived",
                           "context_source_namespace":"voice:forged", "context_source_event_id":"image:forged",
                           "source_namespace":"fixture", "source_key":"forged"}.items():
            with self.subTest(broken_authority=key):
                self.assertIsNone(indexing_exclusion(row | {key:value}, content))
        self.assertIsNone(indexing_exclusion(row, content | {"summary":"a genuine additional finding"}))
        self.assertIsNone(indexing_exclusion(row, content | {"delivery_id":"other"}))
        for finding in ("The core is on the board.", "No core is visible in this frame."):
            with self.subTest(finding=finding):
                self.assertIsNone(indexing_exclusion(row | {"context_source_namespace":"perception:analyzer:v1",
                    "context_trust":"derived", "trust":"derived"}, {"record":{"summary":finding}}))

    async def test_never_dispatched_skip_has_no_provider_io_or_searchable_claim(self):
        item = source(semantic_index_exclusion=TRANSPORT_RECEIPT_EXCLUDED)
        repo = SimpleNamespace(lease_seconds=60, retention_source=AsyncMock(return_value=item),
                               skip_indexing=AsyncMock(return_value=True), settle=AsyncMock())
        provider = SimpleNamespace(ensure_bank=AsyncMock(), operation=AsyncMock(), retain=AsyncMock())
        claim = SimpleNamespace(generation=1, bank_name="bank", id=uuid4(), document_id=item.document_id)
        self.assertEqual(await IngestionWorker(repo, provider)._retain(claim), "cancelled")
        repo.settle.assert_not_awaited()
        provider.ensure_bank.assert_not_awaited()
        provider.operation.assert_not_awaited()
        provider.retain.assert_not_awaited()

    async def test_dispatched_and_unknown_operations_keep_existing_reconciliation(self):
        for operation in (None, {"status":"processing"}, {"status":"completed"}):
            with self.subTest(operation=operation):
                item = source(semantic_index_exclusion=TRANSPORT_RECEIPT_EXCLUDED)
                repo = SimpleNamespace(lease_seconds=60, retention_source=AsyncMock(return_value=item),
                    skip_indexing=AsyncMock(return_value=False), prepare_retain=AsyncMock(return_value=item),
                    settle=AsyncMock(side_effect=lambda claim,status,**kw:status))
                provider = SimpleNamespace(ensure_bank=AsyncMock(), operation=AsyncMock(return_value=operation),
                    retain=AsyncMock(), document=AsyncMock(return_value={}), verify_document=lambda *args:None)
                claim = SimpleNamespace(generation=1, bank_name="bank", id=uuid4(), document_id=item.document_id)
                actual = await IngestionWorker(repo, provider)._retain(claim)
                self.assertEqual(actual, "searchable" if operation and operation["status"]=="completed" else "accepted")
                self.assertEqual(provider.retain.await_count, int(operation is None))

    async def test_atomic_skip_requires_owned_undispatched_current_source(self):
        connection = SimpleNamespace(fetchrow=AsyncMock(return_value={}), execute=AsyncMock())
        @asynccontextmanager
        async def transaction(owner):
            yield connection
        item = source(semantic_index_exclusion=TRANSPORT_RECEIPT_EXCLUDED)
        sources = SimpleNamespace(transaction=transaction, _version=lambda row:item)
        repo = OutboxRepository(sources)
        claim = SimpleNamespace(operation="retain", owner_id="fixture", area_id=item.area_id,
                                document_id=item.document_id, generation=1, id=uuid4())
        repo._owned = AsyncMock(return_value={"dispatch_count":1})
        self.assertFalse(await repo.skip_indexing(claim))
        connection.execute.assert_not_awaited()
        repo._owned.return_value = {"dispatch_count":0}
        self.assertTrue(await repo.skip_indexing(claim))
        self.assertEqual(connection.execute.call_args.args[2], TRANSPORT_RECEIPT_EXCLUDED)
        self.assertIn("status='cancelled'", connection.execute.call_args.args[0])
        connection.execute.reset_mock()
        sources._version = lambda row:item.model_copy(update={"generation":2})
        with self.assertRaises(SourceError):
            await repo.skip_indexing(claim)
        sources._version = lambda row:item
        repo._owned.side_effect = IngestionClaimLost()
        with self.assertRaises(IngestionClaimLost):
            await repo.skip_indexing(claim)
        connection.execute.assert_not_awaited()

    async def test_recall_filters_before_budget_without_extra_policy_queries_and_revalidates(self):
        receipt_source = source(document_id="receipt", semantic_index_exclusion=TRANSPORT_RECEIPT_EXCLUDED)
        positive = source(document_id="positive")
        negative = source(document_id="negative", content={"summary":"No core visible in this frame."})
        connection = SimpleNamespace(fetchval=AsyncMock(return_value="bank"))
        @asynccontextmanager
        async def transaction(owner):
            yield connection
        originals = {s.document_id:s for s in (receipt_source, positive, negative)}
        sources = SimpleNamespace(transaction=transaction, _area=AsyncMock(),
            resolve_document=AsyncMock(side_effect=lambda owner,area,doc:originals[doc]),
            revalidate=AsyncMock(side_effect=lambda item:item))
        provider = SimpleNamespace(recall=AsyncMock(return_value={"results":[{"document_id":d} for d in originals]}),
                                   document=AsyncMock(return_value={}), verify_document=lambda *args:None)
        service = SourceMemoryRetrieval(sources, provider, max_sources=2)
        result = await service.recall("fixture", positive.area_id, "core")
        self.assertEqual([s["document_id"] for s in result["sources"]], ["positive","negative"])
        self.assertEqual(result["policy_excluded_candidates"], 1)
        self.assertEqual(result["omitted_sources"], 0)
        self.assertEqual(sources.resolve_document.await_count, 3)
        self.assertEqual(provider.document.await_count, 2)
        self.assertEqual(sources.revalidate.await_count, 2)
        sources.revalidate.side_effect = SourceError("knowledge_source_changed", "synthetic correction")
        changed = await service.recall("fixture", positive.area_id, "core")
        self.assertEqual(changed["sources"], [])
        self.assertEqual(changed["rejected_candidates"], 2)
        sources.resolve_document.side_effect = SourceError("knowledge_source_unavailable", "synthetic deletion")
        deleted = await service.recall("fixture", positive.area_id, "core")
        self.assertEqual(deleted["policy_excluded_candidates"], 0)
        self.assertEqual(deleted["rejected_candidates"], 3)

    async def test_readiness_exclusion_is_not_success_or_pending(self):
        item = source(ingestion_status="cancelled", ingestion_reason=TRANSPORT_RECEIPT_EXCLUDED,
                      semantic_index_exclusion=TRANSPORT_RECEIPT_EXCLUDED)
        sources = SimpleNamespace(current_snapshot=AsyncMock(return_value={
            "observed_at":datetime.now(UTC), "total_sources":1, "available_sources":1,
            "not_searchable_sources":1, "policy_excluded_sources":1,
            "failed_ingestion_sources":0, "uncertain_ingestion_sources":0, "sources":[item]}))
        result = await CurrentSourceContext(sources).describe("fixture", item.area_id)
        self.assertEqual(result["not_searchable_sources"], 1)
        self.assertEqual(result["policy_excluded_sources"], 1)
        self.assertEqual(result["sources"][0]["ingestion_status"], "cancelled")
        self.assertEqual(result["sources"][0]["ingestion_reason"], TRANSPORT_RECEIPT_EXCLUDED)
