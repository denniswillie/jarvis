"""M02 source substrate tests. No Hindsight service or personal ingestion."""
import asyncio
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import unittest
from uuid import uuid4

import asyncpg

from app.knowledge.models import SourceError, SourcePublication
from app.knowledge.deletion import SourceDeletionService
from app.knowledge.sources import SourceRepository
from app.migrations.runner import apply_migrations, quote_schema
from evals.second_brain.fixtures import DatabaseFixture


@unittest.skipUnless(os.getenv("JARVIS_SECOND_BRAIN_TEST_DB_URL") and
                     os.getenv("JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"),
                     "Explicit second-brain database fixtures required")
class MemorySourceDatabaseTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.dsn = os.environ["JARVIS_SECOND_BRAIN_TEST_DB_URL"]
        self.fixture = DatabaseFixture(self.dsn, Path(os.environ["JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"]),
                                       test_name=self.id())
        self.addAsyncCleanup(self.fixture.cleanup)
        connection = await asyncpg.connect(self.dsn, timeout=10, command_timeout=90)
        try:
            self.schema = await self.fixture.create_schema(connection)
            await apply_migrations(connection, schema=self.schema, fresh=True)
        finally:
            await connection.close()

        async def setup(connection):
            await connection.execute(f"set search_path to {quote_schema(self.schema)},pg_catalog,public,extensions")
        self.pool = await asyncpg.create_pool(self.dsn, min_size=1, max_size=2, timeout=10,
                                             command_timeout=30, setup=setup)
        self.addAsyncCleanup(self.pool.close)
        self.repo = SourceRepository(self.pool)
        self.owner, self.other = self.fixture.owners
        async with self.pool.acquire() as connection:
            self.area = await connection.fetchval(
                "insert into jarvis_knowledge_areas(owner_id,name,domain) values($1,'Fixture','test') returning id",
                self.owner,
            )
        self.when = datetime(2026, 9, 14, 12, tzinfo=UTC)

    async def publication(self, text="Synthetic original", **changes):
        async with self.pool.acquire() as connection:
            payload = await connection.fetchval(
                "insert into jarvis_context_payloads(owner_id,availability,content) values($1,'available',$2::jsonb) returning id",
                self.owner, json.dumps({"text": text}),
            )
        return SourcePublication(**(dict(
            area_id=self.area, namespace="fixture", source_key="source-one", payload_id=payload,
            source_type="experience", trust="user_input", source_time=self.when,
            time_basis="original user report", expected_revision=0,
        ) | changes))

    async def test_atomic_publication_concurrent_replay_and_original_expansion(self):
        request = await self.publication()
        async with self.pool.acquire() as connection:
            # Area history has its own removable original. Concurrent source
            # publication must preserve all existing payloads without new copies.
            original_payloads = {row["id"] for row in await connection.fetch(
                "select id from jarvis_context_payloads where owner_id=$1", self.owner)}
        first, replay = await asyncio.gather(self.repo.publish(self.owner, request),
                                            self.repo.publish(self.owner, request))
        self.assertEqual(first, replay)
        self.assertEqual(first.ingestion_status, "pending")
        self.assertEqual(first.source_time, self.when)
        self.assertEqual(first.payload_id, request.payload_id)
        self.assertNotIn("bank_name", first.original_evidence())
        self.assertEqual(await self.repo.resolve_document(self.owner, self.area, first.document_id), first)
        async with self.pool.acquire() as connection:
            for table in ("jarvis_knowledge_sources", "jarvis_knowledge_source_versions",
                          "jarvis_memory_banks", "jarvis_memory_outbox"):
                self.assertEqual(await connection.fetchval(f"select count(*) from {table}"), 1)
            self.assertEqual({row["id"] for row in await connection.fetch(
                "select id from jarvis_context_payloads where owner_id=$1", self.owner)}, original_payloads)

    async def test_correction_rejects_stale_current_view_but_allows_explicit_history(self):
        old_request = await self.publication("Stopped coding, not completed")
        old = await self.repo.publish(self.owner, old_request)
        new = await self.repo.publish(self.owner, await self.publication("Corrected progress", expected_revision=1))
        self.assertEqual((new.source_id, new.revision, new.generation), (old.source_id, 2, 2))
        self.assertNotEqual(old.document_id, new.document_id)
        for operation in (self.repo.revalidate(old), self.repo.resolve_document(self.owner, self.area, old.document_id),
                          self.repo.publish(self.owner, old_request)):
            with self.assertRaises(SourceError):
                await operation
        historical = await self.repo.read(self.owner, self.area, old.source_id, revision=1)
        self.assertFalse(historical.is_current)
        self.assertEqual(historical.content["text"], "Stopped coding, not completed")
        async with self.pool.acquire() as connection:
            states = await connection.fetch("select operation,status from jarvis_memory_outbox where document_id=$1",
                                            old.document_id)
        self.assertEqual({(r["operation"], r["status"]) for r in states},
                         {("retain", "cancelled"), ("delete", "pending")})

    async def test_owner_area_and_unknown_document_are_denied_without_namespace_guessing(self):
        original = await self.repo.publish(self.owner, await self.publication())
        for owner, area, document in ((self.other, self.area, original.document_id),
                                     (self.owner, uuid4(), original.document_id),
                                     (self.owner, self.area, "unknown-vendor-document")):
            with self.subTest(owner=owner, area=area), self.assertRaises(SourceError):
                await self.repo.resolve_document(owner, area, document)
        with self.assertRaises(SourceError):
            await self.repo.publish(self.other, await self.publication())

    async def test_forgetting_removes_originals_and_blocks_replay_and_history(self):
        request = await self.publication("Private fixture-only detail")
        original = await self.repo.publish(self.owner, request)
        receipt = await self.repo.delete_source(self.owner, self.area, original.source_id, expected_revision=1)
        self.assertEqual(receipt["vendor_cleanup"], "pending")
        self.assertEqual(receipt["read_barrier"], "deleted")
        with self.assertRaises(SourceError):
            await self.repo.read(self.owner, self.area, original.source_id, revision=1)
        replacement = await self.publication("Late replay of deleted source")
        with self.assertRaises(SourceError):
            await self.repo.publish(self.owner, replacement)
        again = await self.repo.delete_source(self.owner, self.area, original.source_id, expected_revision=1)
        self.assertEqual(receipt, again)
        async with self.pool.acquire() as connection:
            payload = await connection.fetchrow("select content,availability from jarvis_context_payloads where id=$1",
                                                 original.payload_id)
        self.assertEqual(dict(payload), {"content": None, "availability": "deleted"})

    async def test_upstream_payload_removal_invalidates_source_without_repository_delete(self):
        original = await self.repo.publish(self.owner, await self.publication())
        async with self.pool.acquire() as connection:
            await connection.execute("update jarvis_context_payloads set content=null,availability='deleted' where id=$1",
                                     original.payload_id)
        with self.assertRaises(SourceError):
            await self.repo.revalidate(original)
        async with self.pool.acquire() as connection:
            row = await connection.fetchrow("select status,generation from jarvis_knowledge_sources where id=$1",
                                            original.source_id)
            cleanup = await connection.fetchval(
                "select status from jarvis_memory_outbox where document_id=$1 and operation='delete'",
                original.document_id,
            )
        self.assertEqual(dict(row), {"status": "deleted", "generation": 2})
        self.assertEqual(cleanup, "pending")

    async def test_original_and_version_metadata_cannot_be_rewritten(self):
        source = await self.repo.publish(self.owner, await self.publication())
        async with self.pool.acquire() as connection:
            with self.assertRaises(asyncpg.CheckViolationError):
                await connection.execute("update jarvis_knowledge_source_versions set trust='sensor' where document_id=$1",
                                         source.document_id)
            with self.assertRaises(asyncpg.CheckViolationError):
                await connection.execute("update jarvis_knowledge_sources set source_key='other',generation=2 where id=$1",
                                         source.source_id)

    async def test_deletion_receipt_distinguishes_originals_provider_and_unverified_copies(self):
        service = SourceDeletionService(self.repo)
        source = await self.repo.publish(self.owner, await self.publication("Private receipt fixture"))
        initial = await service.inspect(self.owner, self.area, source.source_id)
        self.assertEqual(initial["state"], "not_requested")
        self.assertEqual(initial["local_originals"]["remaining_content_payloads"], 1)
        pending = await service.request(self.owner, self.area, source.source_id, expected_revision=1)
        self.assertEqual(pending["state"], "pending")
        self.assertEqual(pending["read_barrier"], "deleted")
        self.assertEqual(pending["local_originals"]["remaining_content_payloads"], 0)
        self.assertEqual(pending["provider"]["pending"], 1)
        self.assertFalse(pending["full_erasure_verified"])
        self.assertIn("database_backups", pending["unverified_copy_classes"])
        self.assertNotIn("Private receipt fixture", json.dumps(pending))
        # A durable worker receipt is the subject of this test, not native
        # erasure. Native provider cleanup is exercised in its opt-in suite.
        async with self.pool.acquire() as connection:
            await connection.execute("""update jarvis_memory_outbox set status='done',cleanup_operation_ids='{}'
                where owner_id=$1 and document_id=$2 and operation='delete'""", self.owner, source.document_id)
        recorded = await service.inspect(self.owner, self.area, source.source_id)
        self.assertEqual(recorded["state"], "scoped_cleanup_recorded")
        self.assertEqual(recorded["provider"]["cleanup_recorded"], 1)
        self.assertFalse(recorded["full_erasure_verified"])
        repeated = await service.request(self.owner, self.area, source.source_id, expected_revision=1)
        self.assertEqual(repeated["state"], "scoped_cleanup_recorded")
        self.assertEqual(repeated["source_deleted_at"], pending["source_deleted_at"])

    async def test_deletion_retry_is_explicit_bounded_and_preserves_cleanup_identity(self):
        service = SourceDeletionService(self.repo)
        first = await self.repo.publish(self.owner, await self.publication())
        await self.repo.publish(self.owner, await self.publication("Revision two", expected_revision=1))
        await self.repo.publish(self.owner, await self.publication("Revision three", expected_revision=2))
        receipt = await service.request(self.owner, self.area, first.source_id, expected_revision=3)
        async with self.pool.acquire() as connection:
            await connection.execute("""update jarvis_memory_outbox set status='failed',failure_count=8,
                last_error='fixture_failure',cleanup_operation_ids='{}' where owner_id=$1 and operation='delete'""",
                self.owner)
        before = await service.documents(self.owner, self.area, first.source_id)
        repeated = await service.request(self.owner, self.area, first.source_id, expected_revision=3)
        self.assertEqual(repeated["state"], "requires_attention")
        self.assertEqual(repeated["provider"]["failed"], 3)
        with self.assertRaises(SourceError):
            await service.retry_failed(self.owner, self.area, first.source_id,
                                       expected_generation=receipt["generation"] + 1)
        retry = await service.retry_failed(self.owner, self.area, first.source_id,
                                           expected_generation=receipt["generation"], limit=1)
        self.assertEqual((retry["requeued"], retry["remaining_failed"]), (1, 2))
        after = await service.documents(self.owner, self.area, first.source_id)
        self.assertEqual([item["cleanup_operation_id"] for item in before["items"]],
                         [item["cleanup_operation_id"] for item in after["items"]])
        self.assertEqual([item["native_operation_count"] for item in after["items"]], [0, 0, 0])
        self.assertEqual([item["cleanup_status"] for item in after["items"]], ["pending", "failed", "failed"])
        self.assertEqual(after["items"][0]["failure_count"], 0)
        await service.retry_failed(self.owner, self.area, first.source_id,
                                   expected_generation=receipt["generation"])
        empty = await service.retry_failed(self.owner, self.area, first.source_id,
                                           expected_generation=receipt["generation"])
        self.assertEqual((empty["requeued"], empty["remaining_failed"]), (0, 0))

    async def test_deletion_manifest_is_paginated_metadata_with_strict_scope_and_bounds(self):
        service = SourceDeletionService(self.repo)
        first = await self.repo.publish(self.owner, await self.publication("Do not copy this original"))
        for revision in range(1, 4):
            await self.repo.publish(self.owner, await self.publication(
                f"Private revision {revision + 1}", expected_revision=revision))
        page = await service.documents(self.owner, self.area, first.source_id, limit=2)
        self.assertEqual([item["revision"] for item in page["items"]], [1, 2])
        self.assertEqual(page["next_after_revision"], 2)
        second = await service.documents(self.owner, self.area, first.source_id,
                                         after_revision=page["next_after_revision"], limit=2)
        self.assertEqual([item["revision"] for item in second["items"]], [3, 4])
        self.assertIsNone(second["next_after_revision"])
        self.assertNotIn("Do not copy", json.dumps(page))
        self.assertNotIn("bank_name", json.dumps(page))
        for owner, area, source_id in ((self.other, self.area, first.source_id),
                                       (self.owner, uuid4(), first.source_id),
                                       (self.owner, self.area, uuid4())):
            with self.subTest(owner=owner, area=area), self.assertRaises(SourceError):
                await service.inspect(owner, area, source_id)
            with self.assertRaises(SourceError):
                await service.documents(owner, area, source_id)
            with self.assertRaises(SourceError):
                await service.retry_failed(owner, area, source_id, expected_generation=1)
        for options in ({"limit": 0}, {"limit": 129}, {"limit": True}, {"after_revision": -1},
                        {"after_revision": 1.5}, {"after_revision": False}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                await service.documents(self.owner, self.area, first.source_id, **options)
        with self.assertRaises(SourceError):
            await service.retry_failed(self.owner, self.area, first.source_id, expected_generation=4)

    async def test_deletion_receipt_does_not_hide_missing_provider_intents(self):
        service = SourceDeletionService(self.repo)
        source = await self.repo.publish(self.owner, await self.publication())
        await service.request(self.owner, self.area, source.source_id, expected_revision=1)
        async with self.pool.acquire() as connection:
            await connection.execute("""delete from jarvis_memory_outbox
                where owner_id=$1 and document_id=$2 and operation='delete'""", self.owner, source.document_id)
        missing = await service.inspect(self.owner, self.area, source.source_id)
        self.assertEqual(missing["state"], "requires_attention")
        self.assertEqual(missing["provider"]["unverified"], 1)
        repaired = await service.request(self.owner, self.area, source.source_id, expected_revision=1)
        self.assertEqual(repaired["state"], "pending")
        self.assertEqual(repaired["provider"]["unverified"], 0)

    async def test_upstream_deletion_receipt_reports_retained_historical_originals(self):
        service = SourceDeletionService(self.repo)
        first = await self.repo.publish(self.owner, await self.publication())
        current = await self.repo.publish(self.owner, await self.publication("Correction", expected_revision=1))
        async with self.pool.acquire() as connection:
            await connection.execute("""update jarvis_context_payloads set availability='deleted',content=null
                where owner_id=$1 and id=$2""", self.owner, current.payload_id)
        partial = await service.inspect(self.owner, self.area, first.source_id)
        self.assertEqual(partial["read_barrier"], "deleted")
        self.assertEqual(partial["state"], "requires_attention")
        self.assertEqual(partial["local_originals"]["remaining_content_payloads"], 1)
        requested = await service.request(self.owner, self.area, first.source_id, expected_revision=2)
        self.assertEqual(requested["local_originals"]["versions"], 2)
        self.assertEqual(requested["local_originals"]["remaining_content_payloads"], 0)
        self.assertEqual(requested["local_originals"]["not_deleted_payloads"], 0)

    async def test_deletion_retry_does_not_touch_other_sources_or_live_leases(self):
        service = SourceDeletionService(self.repo)
        source = await self.repo.publish(self.owner, await self.publication())
        other = await self.repo.publish(self.owner, await self.publication(source_key="another-source"))
        deleted = await service.request(self.owner, self.area, source.source_id, expected_revision=1)
        await service.request(self.owner, self.area, other.source_id, expected_revision=1)
        async with self.pool.acquire() as connection:
            await connection.execute("""update jarvis_memory_outbox set status='failed'
                where owner_id=$1 and document_id=$2 and operation='delete'""", self.owner, other.document_id)
            await connection.execute("""update jarvis_memory_outbox set status='leased',worker_id='fixture',
                lease_until=clock_timestamp()+interval '1 minute'
                where owner_id=$1 and document_id=$2 and operation='delete'""", self.owner, source.document_id)
        retried = await service.retry_failed(self.owner, self.area, source.source_id,
                                             expected_generation=deleted["generation"])
        self.assertEqual(retried["requeued"], 0)
        self.assertEqual((await service.inspect(self.owner, self.area, source.source_id))["state"], "pending")
        self.assertEqual((await service.inspect(self.owner, self.area, other.source_id))["provider"]["failed"], 1)
        for value in (0, -1, True, 1.5):
            with self.subTest(value=value), self.assertRaises(ValueError):
                await service.retry_failed(self.owner, self.area, source.source_id, expected_generation=value)

    async def test_single_source_deletion_refuses_shared_originals_without_partial_writes(self):
        service = SourceDeletionService(self.repo)
        request = await self.publication("Shared original must remain")
        source = await self.repo.publish(self.owner, request)
        sibling = await self.repo.publish(self.owner, request.model_copy(update={"source_key": "shared-sibling"}))
        async with self.pool.acquire() as connection:
            other_area = await connection.fetchval(
                "insert into jarvis_knowledge_areas(owner_id,name,domain) values($1,'Other fixture','test') returning id",
                self.owner)
        cross_area = await self.repo.publish(self.owner, request.model_copy(update={
            "area_id": other_area, "source_key": "cross-area-sibling",
        }))
        with self.assertRaises(SourceError) as caught:
            await service.request(self.owner, self.area, source.source_id, expected_revision=1)
        self.assertEqual(caught.exception.code, "knowledge_source_shared_original")
        for original in (source, sibling, cross_area):
            restored = await self.repo.read(self.owner, original.area_id, original.source_id)
            self.assertEqual(restored.content["text"], "Shared original must remain")
            self.assertEqual(restored.generation, 1)
        async with self.pool.acquire() as connection:
            self.assertEqual(await connection.fetchval(
                "select count(*) from jarvis_memory_outbox where owner_id=$1 and operation='delete'", self.owner), 0)

    async def test_shared_historical_original_is_also_protected_from_single_source_erasure(self):
        service = SourceDeletionService(self.repo)
        request = await self.publication("Retained historical original")
        source = await self.repo.publish(self.owner, request)
        sibling = await self.repo.publish(self.owner, request.model_copy(update={"source_key": "sibling"}))
        await self.repo.publish(self.owner, await self.publication(
            "New current sibling", source_key="sibling", expected_revision=1))
        with self.assertRaises(SourceError):
            await service.request(self.owner, self.area, source.source_id, expected_revision=1)
        historical = await self.repo.read(self.owner, self.area, sibling.source_id, revision=1)
        self.assertEqual(historical.content["text"], "Retained historical original")
        self.assertFalse(historical.is_current)
