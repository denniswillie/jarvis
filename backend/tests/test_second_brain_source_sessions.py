"""Trusted source sessions across publication, migration, recall and correction."""
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import unittest
from uuid import UUID, uuid4

import asyncpg

from app.knowledge.backfill import LegacyBackfill, LegacyRow, NAMESPACE, fingerprint
from app.knowledge.browser import SourceMemoryBrowser
from app.knowledge.hindsight import original_document
from app.knowledge.lifecycle import SourceLifecycleService
from app.knowledge.memory_backend import SourceMemoryReadBackend
from app.knowledge.models import SourceError, SourcePublication, SourceVersion
from app.knowledge.publication import CanonicalSourcePublication
from app.knowledge.retrieval import SourceMemoryRetrieval, SourceRecallScope
from app.migrations.runner import apply_migrations, load_migrations, read_schema_readiness
from app.runtime.feature_config import TaskLimits
from app.tasks.context_events import ContextIngress
from app.tasks.context_repository import ContextRepository
from app.tasks.models import TaskVoiceSource
from app.tasks.repository import TaskRepository
from evals.second_brain.source_fixture import source_fixture


def fixture_context(name, **kwargs):
    return source_fixture(os.environ["JARVIS_SECOND_BRAIN_TEST_DB_URL"],
        Path(os.environ["JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"]), test_name=name, **kwargs)


@unittest.skipUnless(os.getenv("JARVIS_SECOND_BRAIN_TEST_DB_URL") and
                     os.getenv("JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"),
                     "Explicit manifested source fixtures required")
class SourceSessionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.f = await self.enterAsyncContext(fixture_context(self.id()))
        async with self.f.pool.acquire() as c:
            self.first = await c.fetchval("insert into jarvis_sessions(owner_id) values($1) returning id", self.f.owner)
            self.second = await c.fetchval("insert into jarvis_sessions(owner_id) values($1) returning id", self.f.owner)
            self.foreign = await c.fetchval("insert into jarvis_sessions(owner_id) values($1) returning id", self.f.other)
        self.tasks = TaskRepository(self.f.pool, TaskLimits())
        self.history = ContextRepository(self.tasks, publication=CanonicalSourcePublication(self.f.sources, self.f.area))

    async def event(self, session, text="Synthetic reported progress"):
        return await self.history.append(self.f.owner, ContextIngress(session_id=session,
            source_namespace="fixture:session", source_event_id=str(uuid4()), receipt_id=str(uuid4()),
            kind="user_transcript", trust="user_input", source_time=datetime.now(UTC), payload={"text": text}))

    async def test_supplied_session_is_verified_not_inferred_from_body(self):
        first = await self.f.publish_payload({"text": "Synthetic note", "session_id": str(self.foreign)},
                                            session_id=self.first)
        self.assertEqual(first.session_id, self.first)
        self.assertIsNone(first.context_event_id)
        self.assertIsNone(first.context_session_id)
        unknown = await self.f.publish_payload({"session_id": str(self.first)}, source_key="unknown")
        self.assertIsNone(unknown.session_id)
        self.assertTrue(SourceRecallScope(session_id=self.first).matches(first))
        self.assertFalse(SourceRecallScope(session_id=self.first).matches(unknown))
        self.assertFalse(SourceRecallScope(session_id=self.foreign).matches(first))

    async def test_source_and_database_reject_foreign_session(self):
        with self.assertRaises(SourceError) as caught:
            await self.f.publish(session_id=self.foreign)
        self.assertEqual(caught.exception.code, "knowledge_session_unavailable")
        original = await self.f.publish()
        async with self.f.pool.acquire() as c:
            with self.assertRaises(asyncpg.ForeignKeyViolationError):
                await c.execute("""insert into jarvis_knowledge_source_sessions
                    (owner_id,source_id,source_revision,session_id) values($1,$2,1,$3)""",
                    self.f.owner, original.source_id, self.foreign)

    async def test_replay_and_session_change_have_distinct_revision_semantics(self):
        source = await self.f.publish(session_id=self.first)
        request = SourcePublication(area_id=self.f.area, namespace="fixture", source_key="source-one",
            payload_id=source.payload_id, source_type=source.source_type, trust=source.trust,
            source_time=source.source_time, time_basis=source.time_basis,
            session_id=self.first, expected_revision=0)
        replay = await self.f.sources.publish(self.f.owner, request)
        self.assertEqual((replay.source_id, replay.revision), (source.source_id, 1))
        moved = await self.f.sources.publish(self.f.owner, request.model_copy(
            update={"session_id": self.second, "expected_revision": 1}))
        self.assertEqual(moved.revision, 2)
        self.assertEqual(moved.session_id, self.second)
        prior = await self.f.sources.read(self.f.owner, self.f.area, source.source_id, revision=1)
        self.assertEqual(prior.session_id, self.first)
        self.assertFalse(SourceRecallScope(session_id=self.first).matches(moved))

    async def test_canonical_capture_and_correction_keep_each_versions_authored_session(self):
        event = await self.event(self.first)
        async with self.f.pool.acquire() as c:
            identity = await c.fetchval("select source_id from jarvis_knowledge_context_sources where event_id=$1", event.id)
        first = await self.f.sources.read(self.f.owner, self.f.area, identity)
        self.assertEqual((first.session_id, first.context_session_id), (self.first, self.first))
        request = await self.event(self.second, "Correct the earlier statement")
        await SourceLifecycleService(self.f.sources, self.tasks).correct(
            self.f.owner, self.f.area, identity, expected_revision=1,
            replacement="Calibration is not finished.", request=TaskVoiceSource(request.id, request.revision, request.session_id))
        current = await self.f.sources.read(self.f.owner, self.f.area, identity)
        old = await self.f.sources.read(self.f.owner, self.f.area, identity, revision=1)
        self.assertEqual(current.session_id, self.second)
        self.assertEqual(old.session_id, self.first)
        self.assertFalse(old.is_current)

    async def test_imported_session_is_queryable_without_fabricating_context_event(self):
        async with self.f.pool.acquire() as c:
            for session in (self.first, self.second):
                await c.execute("""insert into jarvis_messages(owner_id,session_id,role,modality,content)
                    values($1,$2,'user','audio','Synthetic imported progress')""", self.f.owner, session)
        result = await LegacyBackfill(self.f.sources).run(self.f.owner, self.f.area)
        self.assertTrue(result["local_manifest_complete"])
        self.assertEqual(result["session_association_issues"], 0)
        reader = SourceMemoryReadBackend(self.f.sources, SourceMemoryRetrieval(self.f.sources, None),
                                        owner_id=self.f.owner, area_id=self.f.area)
        browser = SourceMemoryBrowser(self.f.sources, reader, owner_id=self.f.owner, area_id=self.f.area)
        page = await browser.recent(owner_id=self.f.owner, session_id=self.first)
        self.assertEqual(len(page["items"]), 1)
        self.assertEqual(page["items"][0]["session_id"], str(self.first))
        original = await self.f.sources.read(self.f.owner, self.f.area, UUID(page["items"][0]["id"]))
        self.assertIsNone(original.context_event_id)
        self.assertIsNone(original.context_session_id)
        self.assertTrue(SourceRecallScope(session_id=self.first).matches(original))
        self.assertFalse(SourceRecallScope(session_id=self.second).matches(original))

    async def test_incomplete_session_accounting_blocks_completed_import_receipt(self):
        async with self.f.pool.acquire() as c:
            await c.execute("""insert into jarvis_messages(owner_id,session_id,role,modality,content)
                values($1,$2,'user','audio','Synthetic imported statement')""", self.f.owner, self.first)
        importer = LegacyBackfill(self.f.sources)
        run_id = uuid4()
        report = await importer.run(self.f.owner, self.f.area, run_id=run_id)
        self.assertTrue(report["local_manifest_complete"])
        async with self.f.pool.acquire() as c:
            await c.execute("delete from jarvis_knowledge_source_sessions where owner_id=$1", self.f.owner)
        replay = await importer.run(self.f.owner, self.f.area, run_id=run_id)
        self.assertEqual(replay["status"], "requires_attention")
        self.assertEqual(replay["session_association_issues"], 1)
        self.assertFalse(replay["local_manifest_complete"])

    async def test_importer_quarantines_unverified_session_instead_of_binding_model_text(self):
        async with self.f.pool.acquire() as c:
            message = await c.fetchval("""insert into jarvis_messages(owner_id,session_id,role,modality,content)
                values($1,$2,'user','audio','Synthetic fixture') returning id""", self.f.owner, self.first)
        importer = LegacyBackfill(self.f.sources)
        async with importer.snapshot(self.f.owner) as snapshot:
            row = snapshot.rows[("jarvis_messages", str(message))]
            value = row.value | {"session_id": str(self.foreign)}
            invalid = LegacyRow(row.table, row.key, value, fingerprint(value))
            outcome = await importer._import(self.f.owner, self.f.area, uuid4(), invalid, snapshot, {})
        self.assertEqual(outcome["disposition"], "quarantined")
        self.assertEqual(outcome["reason"], "original_session_outside_owner_snapshot")
        self.assertIsNone(outcome["source_id"])


@unittest.skipUnless(os.getenv("JARVIS_SECOND_BRAIN_TEST_DB_URL") and
                     os.getenv("JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"),
                     "Explicit manifested migration fixture required")
class SourceSessionMigrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_v16_to_v17_upgrade_preserves_original_documents_and_outbox(self):
        previous = tuple(m for m in load_migrations() if m.version <= 16)
        through_session = tuple(m for m in load_migrations() if m.version <= 17)
        async with fixture_context(self.id(), migrations=previous) as f:
            tasks = TaskRepository(f.pool, TaskLimits())
            history = ContextRepository(tasks)
            async with f.pool.acquire() as c:
                session = await c.fetchval("insert into jarvis_sessions(owner_id) values($1) returning id", f.owner)
                message = await c.fetchval("""insert into jarvis_messages(owner_id,session_id,role,modality,content)
                    values($1,$2,'user','audio','Synthetic historical session record') returning id""", f.owner, session)
            event = await history.append(f.owner, ContextIngress(session_id=session,
                source_namespace="fixture:v16", source_event_id=str(uuid4()), receipt_id=str(uuid4()),
                kind="user_transcript", trust="user_input", source_time=datetime.now(UTC), payload={"text": "Canonical original"}))
            importer = LegacyBackfill(f.sources)
            async with importer.snapshot(f.owner) as snapshot:
                legacy = snapshot.rows[("jarvis_messages", str(message))]
                original = snapshot.original(legacy)
                counts, input_digest = snapshot.counts, snapshot.input_digest

            # Fixture rows implement the documented v16 persisted shape, not a copied old engine.
            async def seed(payload_id, namespace, key, kind, trust, stamp):
                async with f.sources.transaction(f.owner) as c:
                    await c.execute("insert into jarvis_memory_banks(owner_id,area_id) values($1,$2) on conflict do nothing", f.owner, f.area)
                    bank = await c.fetchval("select id from jarvis_memory_banks where owner_id=$1 and area_id=$2", f.owner, f.area)
                    identity = await c.fetchval("""insert into jarvis_knowledge_sources
                        (owner_id,area_id,namespace,source_key,current_revision,generation) values($1,$2,$3,$4,1,1) returning id""",
                        f.owner, f.area, namespace, key)
                    document = await c.fetchval("""insert into jarvis_knowledge_source_versions(
                        owner_id,area_id,source_id,revision,generation,payload_id,bank_id,content_digest,source_type,trust,source_time,time_basis)
                        select $1,$2,$3,1,1,p.id,$4,p.content_digest,$5,$6,$7,'Original v16 fixture clock'
                        from jarvis_context_payloads p where p.owner_id=$1 and p.id=$8 returning document_id""",
                        f.owner, f.area, identity, bank, kind, trust, stamp, payload_id)
                    await c.execute("insert into jarvis_memory_outbox(owner_id,document_id,generation,operation) values($1,$2,1,'retain')",
                                    f.owner, document)
                    return identity
            canonical = await seed(event.payload_id, "fixture:v16", "canonical", "user_transcript", "user_input", event.source_time)
            async with f.pool.acquire() as c:
                await c.execute("""insert into jarvis_knowledge_context_sources
                    (owner_id,source_id,source_revision,event_id,event_revision) values($1,$2,1,$3,1)""", f.owner, canonical, event.id)
                payload = await c.fetchval("insert into jarvis_context_payloads(owner_id,availability,content) values($1,'available',$2::jsonb) returning id",
                                          f.owner, json.dumps(original))
            imported = await seed(payload, NAMESPACE, legacy.source_key, "legacy:jarvis_messages", "user_input", datetime.now(UTC))
            async with f.pool.acquire() as c:
                run = await c.fetchval("""insert into jarvis_legacy_import_runs
                    (owner_id,area_id,mapping_version,input_digest,population_counts,status)
                    values($1,$2,1,$3,$4::jsonb,'complete') returning id""", f.owner, f.area, input_digest, json.dumps(counts))
                await c.execute("""insert into jarvis_legacy_import_records
                    (owner_id,run_id,population,origin_key,input_digest,disposition,source_id,source_revision,source_generation)
                    values($1,$2,'jarvis_messages',$3,$4,'imported',$5,1,1)""",
                    f.owner, run, str(message), legacy.input_digest, imported)
                rows = await c.fetch("""select v.*,b.bank_name,p.content from jarvis_knowledge_source_versions v
                    join jarvis_memory_banks b on b.id=v.bank_id
                    join jarvis_context_payloads p on p.id=v.payload_id""")
                before = {}
                for row in rows:
                    values = {key: row[key] for key in SourceVersion.model_fields if key in row}
                    values.update(content=json.loads(row["content"]), is_current=True, ingestion_status="pending")
                    value = SourceVersion.model_validate(values)
                    before[value.source_id] = original_document(value)
                operations = [dict(r) for r in await c.fetch("select * from jarvis_memory_outbox order by id")]
                schema = await c.fetchval("select current_schema()")
                readiness = await read_schema_readiness(c, schema=schema, migrations=through_session)
                self.assertEqual(readiness.pending, (17,))
                self.assertEqual(await apply_migrations(c, schema=schema, migrations=through_session), (17,))
                self.assertEqual(await apply_migrations(c, schema=schema, migrations=through_session), ())
                self.assertEqual(operations, [dict(r) for r in await c.fetch("select * from jarvis_memory_outbox order by id")])
                self.assertTrue((await read_schema_readiness(c, schema=schema, migrations=through_session)).ready)
            for identity in (canonical, imported):
                source = await f.sources.read(f.owner, f.area, identity)
                self.assertEqual(source.session_id, session)
                self.assertEqual(original_document(source), before[identity])
            legacy_source = await f.sources.read(f.owner, f.area, imported)
            self.assertIsNone(legacy_source.context_event_id)
            async with f.pool.acquire() as c:
                await c.execute("drop table jarvis_knowledge_source_sessions")
                readiness = await read_schema_readiness(c, schema=schema, migrations=through_session)
                self.assertIn("missing_table:jarvis_knowledge_source_sessions", readiness.issues)
