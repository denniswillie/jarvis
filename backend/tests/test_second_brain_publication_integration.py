"""Real canonical-source transactions in manifested synthetic database schemas."""
from datetime import UTC, datetime
import os
from pathlib import Path
import unittest
from uuid import uuid4

import asyncpg

from app.knowledge.current_context import CurrentSourceContext
from app.knowledge.models import SourceError, SourcePublication
from app.knowledge.publication import CanonicalSourcePublication
from app.runtime.feature_config import TaskLimits
from app.tasks.context_capture import SessionContextCapture
from app.tasks.context_events import ContextIngress
from app.tasks.context_repository import ContextRepository
from app.tasks.repository import TaskRepository
from evals.second_brain.source_fixture import source_fixture


@unittest.skipUnless(
    os.getenv("JARVIS_SECOND_BRAIN_TEST_DB_URL") and os.getenv("JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"),
    "Explicit manifested second-brain database fixtures required",
)
class SourcePublicationDatabaseTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.f = await self.enterAsyncContext(source_fixture(
            os.environ["JARVIS_SECOND_BRAIN_TEST_DB_URL"],
            Path(os.environ["JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"]), test_name=self.id(),
        ))
        self.tasks = TaskRepository(self.f.pool, TaskLimits(lease_seconds=60))
        self.plain = ContextRepository(self.tasks)
        self.publisher = CanonicalSourcePublication(self.f.sources, self.f.area)
        self.context = ContextRepository(self.tasks, publication=self.publisher)
        self.when = datetime(2026, 9, 14, 12, tzinfo=UTC)
        async with self.f.pool.acquire() as connection:
            self.session = await connection.fetchval(
                "insert into jarvis_sessions(owner_id) values($1) returning id", self.f.owner,
            )

    def ingress(self, *, key="utterance", receipt=None, text="Synthetic original", **changes):
        return ContextIngress(**(dict(
            session_id=self.session, source_namespace="publication-fixture", source_event_id=key,
            receipt_id=receipt or str(uuid4()), kind="user_transcript", trust="user_input",
            source_time=self.when, payload={"text": text},
        ) | changes))

    async def mapped(self, event):
        async with self.f.pool.acquire() as connection:
            identity = await connection.fetchval(
                "select source_id from jarvis_knowledge_context_sources where owner_id=$1 and event_id=$2",
                self.f.owner, event.id,
            )
        self.assertIsNotNone(identity)
        return await self.f.sources.read(self.f.owner, self.f.area, identity)

    async def counts(self):
        async with self.f.pool.acquire() as connection:
            return {table: await connection.fetchval("select count(*) from " + table) for table in (
                "jarvis_context_payloads", "jarvis_context_events", "jarvis_context_ingress_receipts",
                "jarvis_knowledge_sources", "jarvis_knowledge_source_versions",
                "jarvis_knowledge_context_sources", "jarvis_memory_outbox",
            )}

    async def test_disabled_publication_has_no_memory_side_effects_or_implicit_backfill(self):
        request = self.ingress(receipt="legacy")
        record = await self.plain.append(self.f.owner, request)
        replay = await self.context.append(self.f.owner, request)
        self.assertEqual(replay.id, record.id)
        counts = await self.counts()
        self.assertEqual(counts["jarvis_context_events"], 1)
        self.assertEqual(counts["jarvis_knowledge_sources"], 0)
        self.assertEqual(counts["jarvis_memory_outbox"], 0)

    async def test_publication_links_exact_original_and_replay_is_idempotent(self):
        initial = await self.counts()
        request = self.ingress(receipt="exact")
        record = await self.context.append(self.f.owner, request)
        source = await self.mapped(record)
        self.assertEqual((source.payload_id, source.context_event_id, source.context_event_revision),
                         (record.payload_id, record.id, record.revision))
        self.assertEqual((source.content, source.trust, source.source_time),
                         (record.content, record.trust, record.source_time))
        self.assertEqual(source.context_recorded_at, record.recorded_at)
        self.assertEqual(source.ingestion_status, "pending")
        before = await self.counts()
        # Area metadata already owns a payload in a fresh current-schema fixture.
        # Publication adds exactly one row to each of these tables; replay adds none.
        self.assertEqual({table: count - initial[table] for table, count in before.items()},
                         {table: 1 for table in initial})
        self.assertEqual((await self.context.append(self.f.owner, request)).id, record.id)
        self.assertEqual(await self.counts(), before)

    async def test_provisional_and_unknown_inputs_publish_only_after_finalization(self):
        await self.context.append(self.f.owner, self.ingress(key="draft", finality="provisional"))
        await self.context.append(self.f.owner, self.ingress(
            key="unknown", finality="unavailable", payload={"text": None},
        ))
        self.assertEqual((await self.counts())["jarvis_memory_outbox"], 0)
        record = await self.context.append(self.f.owner, self.ingress(
            key="draft", finality="revised", expected_revision=1, text="Actual finalized report",
        ))
        source = await self.mapped(record)
        self.assertEqual((record.revision, source.revision, source.context_event_revision), (2, 1, 2))
        self.assertEqual((await self.counts())["jarvis_knowledge_sources"], 1)

    async def test_distinct_publications_reuse_one_bank_without_rewriting_it(self):
        first = await self.mapped(await self.context.append(self.f.owner, self.ingress(key="bank-first")))
        async with self.f.pool.acquire() as connection:
            before = dict(await connection.fetchrow(
                "select *,xmin::text as row_version from jarvis_memory_banks where owner_id=$1 and area_id=$2",
                self.f.owner, self.f.area))
        second = await self.mapped(await self.context.append(self.f.owner, self.ingress(key="bank-second")))
        async with self.f.pool.acquire() as connection:
            after = dict(await connection.fetchrow(
                "select *,xmin::text as row_version from jarvis_memory_banks where owner_id=$1 and area_id=$2",
                self.f.owner, self.f.area))
            banks = await connection.fetchval(
                "select count(*) from jarvis_memory_banks where owner_id=$1 and area_id=$2",
                self.f.owner, self.f.area)
        self.assertEqual(after, before)
        self.assertEqual(banks, 1)
        self.assertNotEqual(first.document_id, second.document_id)
        self.assertEqual((first.ingestion_status, second.ingestion_status), ("pending", "pending"))

    async def test_correction_invalidates_old_current_evidence_and_publishes_new_revision(self):
        old_record = await self.context.append(self.f.owner, self.ingress(text="Stopped coding"))
        old = await self.mapped(old_record)
        new_record = await self.context.append(self.f.owner, self.ingress(
            text="Stopped coding, function not completed", finality="revised", expected_revision=1,
        ))
        new = await self.mapped(new_record)
        self.assertEqual((new.source_id, new.revision), (old.source_id, 2))
        self.assertGreater(new.generation, old.generation)
        with self.assertRaises(SourceError):
            await self.f.sources.revalidate(old)
        history = await self.f.sources.read(self.f.owner, self.f.area, old.source_id, revision=1)
        self.assertFalse(history.is_current)
        self.assertEqual(history.content["text"], "Stopped coding")
        async with self.f.pool.acquire() as connection:
            rows = await connection.fetch(
                "select operation,status from jarvis_memory_outbox where document_id=$1", old.document_id,
            )
        self.assertEqual({(row["operation"], row["status"]) for row in rows},
                         {("retain", "cancelled"), ("delete", "pending")})
        snapshot = await self.f.sources.current_snapshot(self.f.owner, self.f.area)
        self.assertEqual(snapshot["available_sources"], 1)
        self.assertEqual(snapshot["sources"][0].context_event_id, new_record.id)

    async def test_correction_without_publisher_still_closes_stale_read_barrier(self):
        event = await self.context.append(self.f.owner, self.ingress())
        old = await self.mapped(event)
        await self.plain.append(self.f.owner, self.ingress(
            text="Corrected while publication is disabled", finality="revised", expected_revision=1,
        ))
        with self.assertRaises(SourceError):
            await self.f.sources.resolve_document(self.f.owner, self.f.area, old.document_id)
        snapshot = await self.f.sources.current_snapshot(self.f.owner, self.f.area)
        self.assertEqual((snapshot["total_sources"], snapshot["available_sources"]), (1, 0))
        self.assertEqual(snapshot["sources"], [])
        async with self.f.pool.acquire() as connection:
            self.assertEqual(await connection.fetchval(
                "select status from jarvis_memory_outbox where document_id=$1 and operation='delete'",
                old.document_id,
            ), "pending")

    async def test_invalid_area_rolls_back_canonical_event_payload_receipt_and_intent(self):
        before = await self.counts()
        bad = ContextRepository(self.tasks, publication=CanonicalSourcePublication(self.f.sources, uuid4()))
        with self.assertRaises(SourceError):
            await bad.append(self.f.owner, self.ingress())
        self.assertEqual(await self.counts(), before)

    async def test_reserved_identity_and_forged_lineage_are_rejected(self):
        event = await self.plain.append(self.f.owner, self.ingress())
        independent = await self.f.publish("Different original payload")
        with self.assertRaises(SourceError):
            await self.f.sources.publish(self.f.owner, SourcePublication(
                area_id=self.f.area, namespace="context:forged", source_key="fixture",
                payload_id=event.payload_id, source_type=event.kind, trust=event.trust,
                source_time=event.source_time, time_basis="fixture", expected_revision=0,
            ))
        async with self.f.pool.acquire() as connection:
            with self.assertRaises(asyncpg.CheckViolationError):
                await connection.execute(
                    """insert into jarvis_knowledge_context_sources
                    (owner_id,source_id,source_revision,event_id,event_revision) values($1,$2,$3,$4,$5)""",
                    self.f.owner, independent.source_id, independent.revision, event.id, event.revision,
                )

    async def test_local_pending_evidence_survives_failed_ingestion_and_denies_foreign_scope(self):
        first = await self.f.publish("Pending synthetic note")
        second = await self.f.publish("Failed synthetic note", source_key="second")
        async with self.f.pool.acquire() as connection:
            await connection.execute(
                "update jarvis_memory_outbox set status='failed' where document_id=$1", second.document_id,
            )
        result = await CurrentSourceContext(self.f.sources, timeout_seconds=10).describe(
            self.f.owner, self.f.area, include_content=True,
        )
        self.assertEqual((result["available_sources"], result["not_searchable_sources"]), (2, 2))
        self.assertEqual(result["failed_ingestion_sources"], 1)
        self.assertEqual({item["content"]["text"] for item in result["sources"]},
                         {"Pending synthetic note", "Failed synthetic note"})
        self.assertFalse(result["provider_contacted"])
        with self.assertRaises(SourceError):
            await self.f.sources.current_snapshot(self.f.other, self.f.area)
        async with self.f.pool.acquire() as connection:
            await connection.execute(
                "update jarvis_context_payloads set content=null,availability='deleted' where id=$1",
                first.payload_id,
            )
        snapshot = await self.f.sources.current_snapshot(self.f.owner, self.f.area)
        self.assertEqual(snapshot["available_sources"], 1)
        self.assertEqual(snapshot["sources"][0].source_id, second.source_id)

    async def test_both_audio_capture_adapters_publish_finalized_originals(self):
        for provider in ("openai", "gemini"):
            capture = SessionContextCapture(
                self.context, owner_id=self.f.owner, session_id=self.session,
                provider=provider, audio_session_id="synthetic-" + provider, write_timeout=15,
            )
            self.addAsyncCleanup(capture.close)
            capture.input_admitted("voice-input")
            self.assertTrue(await capture.flush())
            capture.input_final("voice-input", "Fixture report from " + provider)
            self.assertTrue(await capture.flush())
            await capture.close()
        snapshot = await self.f.sources.current_snapshot(self.f.owner, self.f.area)
        self.assertEqual(len(snapshot["sources"]), 2)
        self.assertEqual({source.content["audio_provider"] for source in snapshot["sources"]},
                         {"openai", "gemini"})
        self.assertTrue(all(source.trust == "user_input" for source in snapshot["sources"]))
        self.assertTrue(all(source.context_event_revision == 2 for source in snapshot["sources"]))
