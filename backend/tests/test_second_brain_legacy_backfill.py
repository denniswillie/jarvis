"""Maintenance-window import with real legacy/source tables and isolated fixtures."""
import asyncio
from datetime import UTC, datetime, timedelta
import json
import os
from pathlib import Path
import unittest
from unittest.mock import AsyncMock, patch
from uuid import UUID, uuid4

import asyncpg

from app.knowledge.backfill import LegacyBackfill, NAMESPACE
from app.knowledge.deletion import SourceDeletionService
from app.knowledge.retention import SourceRetentionService
from app.knowledge.legacy_populations import BY_TABLE
from app.knowledge.lifecycle import SourceLifecycleService
from app.knowledge.models import SourceError
from app.runtime.feature_config import TaskLimits
from app.tasks.context_events import ContextIngress
from app.tasks.context_repository import ContextRepository
from app.tasks.models import TaskVoiceSource
from app.tasks.repository import TaskRepository
from evals.second_brain.source_fixture import source_fixture


class LegacyPopulationTests(unittest.TestCase):
    def test_credentials_and_embedding_spaces_have_explicit_dispositions(self):
        self.assertEqual(BY_TABLE["google_oauth_tokens"].fields, ("owner_id",))
        for name in ("memory_item_embeddings", "memory_entity_embeddings", "experience_event_embeddings"):
            self.assertEqual(BY_TABLE[name].mode, "retained_rebuildable")
            self.assertEqual(BY_TABLE[name].keys[-3:], ("embedding_kind", "provider", "model"))
            self.assertNotIn("embedding", BY_TABLE[name].fields)
            self.assertNotIn("embedded_content", BY_TABLE[name].fields)
        self.assertTrue(BY_TABLE["calendar_event_snapshots"].optional)


@unittest.skipUnless(
    os.getenv("JARVIS_SECOND_BRAIN_TEST_DB_URL") and os.getenv("JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"),
    "Explicit manifested second-brain fixtures required",
)
class LegacyBackfillDatabaseTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.f = await self.enterAsyncContext(source_fixture(
            os.environ["JARVIS_SECOND_BRAIN_TEST_DB_URL"],
            Path(os.environ["JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"]), test_name=self.id()))
        self.importer = LegacyBackfill(self.f.sources)
        self.tasks = TaskRepository(self.f.pool, TaskLimits())
        self.history = ContextRepository(self.tasks)
        self.lifecycle = SourceLifecycleService(self.f.sources, self.tasks)
        self.sessions = {}
        async with self.f.pool.acquire() as c:
            for owner in (self.f.owner, self.f.other):
                self.sessions[owner] = await c.fetchval(
                    "insert into jarvis_sessions(owner_id) values($1) returning id", owner)

    async def message(self, text="The robot still needs calibration.", *, role="user", owner=None):
        owner = owner or self.f.owner
        async with self.f.pool.acquire() as c:
            return await c.fetchval("""insert into jarvis_messages(owner_id,session_id,role,content,modality)
                values($1,$2,$3,$4,'audio') returning id""", owner, self.sessions[owner], role, text)

    async def memory(self, text="A legacy model says the robot is complete.", *, evidence=None, owner=None):
        owner = owner or self.f.owner
        async with self.f.pool.acquire() as c:
            memory = await c.fetchval("""insert into memory_items(
                owner_id,memory_type,source,content,title,is_explicit,status,importance,confidence)
                values($1,'semantic_fact','user_statement',$2,'Fixture claim',true,'active',0.5,0.5) returning id""", owner, text)
            if evidence:
                await c.execute("""insert into memory_evidence(
                    owner_id,memory_item_id,evidence_type,evidence_ref_id,excerpt,evidence_time,confidence)
                    values($1,$2,$3,$4,'Synthetic legacy excerpt',clock_timestamp(),0.7)""", owner, memory, evidence[0], str(evidence[1]))
            return memory

    async def source(self, table, key):
        async with self.f.pool.acquire() as c:
            identity = await c.fetchval("""select id from jarvis_knowledge_sources
                where owner_id=$1 and area_id=$2 and namespace=$3 and source_key=$4""",
                self.f.owner, self.f.area, NAMESPACE, table + "/" + str(key))
        return await self.f.sources.read(self.f.owner, self.f.area, identity)

    async def finish(self, *, run_id=None, batch_size=64):
        run_id = run_id or uuid4()
        for _ in range(12):
            result = await self.importer.run(self.f.owner, self.f.area, run_id=run_id, batch_size=batch_size)
            if result["status"] != "pending":
                return result
        self.fail("Bounded synthetic import did not finish")

    async def request(self):
        record = await self.history.append(self.f.owner, ContextIngress(
            session_id=self.sessions[self.f.owner], source_namespace="voice:backfill-test",
            source_event_id=str(uuid4()), receipt_id=str(uuid4()), kind="user_transcript", trust="user_input",
            source_time=datetime.now(UTC), payload={"text": "Apply this explicit fixture lifecycle operation."}))
        return TaskVoiceSource(record.id, record.revision, record.session_id)

    async def test_bounded_resume_and_new_run_keep_originals_and_operation_ids(self):
        user = await self.message()
        assistant = await self.message("I am not sure yet.", role="assistant")
        claim = await self.memory(evidence=("message", user))
        await self.message("Foreign original must not be copied.", owner=self.f.other)
        run_id = uuid4()
        first = await self.importer.run(self.f.owner, self.f.area, run_id=run_id, batch_size=1)
        self.assertEqual(first["status"], "pending")
        final = await self.finish(run_id=run_id, batch_size=1)
        self.assertTrue(final["local_manifest_complete"])
        self.assertFalse(final["ready_for_cutover"])
        self.assertFalse(final["provider_contacted"])
        self.assertEqual(final["dispositions"]["imported"], 3)
        self.assertEqual(final["dispositions"]["attached"], 1)
        original = await self.source("jarvis_messages", user)
        output = await self.source("jarvis_messages", assistant)
        derived = await self.source("memory_items", claim)
        self.assertEqual((original.trust.value, output.trust.value, derived.trust.value),
                         ("user_input", "assistant_output", "derived"))
        self.assertTrue(derived.content["record"]["is_explicit"])
        self.assertEqual(derived.content["references"][0]["state"], "source_backed")
        async with self.f.pool.acquire() as c:
            before = [dict(r) for r in await c.fetch("select id,document_id,operation from jarvis_memory_outbox order by id")]
            self.assertFalse(await c.fetchval("""select exists(select 1 from jarvis_context_payloads
                where content::text like '%Foreign original%')"""))
        replay = await self.finish(run_id=run_id)
        self.assertEqual(replay["sources_processed_in_batch"], 0)
        new_run = await self.finish()
        self.assertEqual(new_run["dispositions"]["already_present"], 3)
        async with self.f.pool.acquire() as c:
            self.assertEqual(before, [dict(r) for r in await c.fetch(
                "select id,document_id,operation from jarvis_memory_outbox order by id")])

    async def test_changed_input_cannot_resume_an_old_cursor(self):
        message = await self.message()
        await self.message("Second input")
        run_id = uuid4()
        await self.importer.run(self.f.owner, self.f.area, run_id=run_id, batch_size=1)
        async with self.f.pool.acquire() as c:
            await c.execute("update jarvis_messages set content='Changed during maintenance' where id=$1", message)
        with self.assertRaises(SourceError) as error:
            await self.importer.run(self.f.owner, self.f.area, run_id=run_id)
        self.assertEqual(error.exception.code, "legacy_snapshot_changed")

    async def test_committed_receipt_survives_lost_batch_response(self):
        message = await self.message()
        run_id = uuid4()
        original = self.importer._import
        async def lost(*args, **kwargs):
            await original(*args, **kwargs)
            raise RuntimeError("Fixture lost response after local commit")
        with patch.object(self.importer, "_import", side_effect=lost):
            with self.assertRaises(RuntimeError):
                await self.importer.run(self.f.owner, self.f.area, run_id=run_id)
        saved = await self.source("jarvis_messages", message)
        final = await self.finish(run_id=run_id)
        self.assertTrue(final["local_manifest_complete"])
        self.assertEqual((await self.source("jarvis_messages", message)).document_id, saved.document_id)
        async with self.f.pool.acquire() as c:
            self.assertEqual(await c.fetchval("select count(*) from jarvis_memory_outbox where operation='retain'"), 1)

    async def test_source_correction_is_not_overwritten_by_a_new_import_run(self):
        message = await self.message()
        await self.finish()
        source = await self.source("jarvis_messages", message)
        await self.lifecycle.correct(self.f.owner, self.f.area, source.source_id, expected_revision=1,
                                     replacement="Explicit current correction.", request=await self.request())
        result = await self.finish()
        self.assertEqual(result["status"], "requires_attention")
        self.assertEqual(result["dispositions"]["quarantined"], 1)
        current = await self.f.sources.read(self.f.owner, self.f.area, source.source_id)
        self.assertEqual(current.content["text"], "Explicit current correction.")

    async def test_forgetting_an_imported_parent_revokes_bound_legacy_claim(self):
        message = await self.message()
        claim = await self.memory(evidence=("message", message))
        result = await self.finish()
        parent = await self.source("jarvis_messages", message)
        child = await self.source("memory_items", claim)
        await self.lifecycle.forget(self.f.owner, self.f.area, parent.source_id, expected_revision=1,
                                    request=await self.request())
        with self.assertRaises(SourceError):
            await self.f.sources.read(self.f.owner, self.f.area, child.source_id)
        repeated = await self.finish(run_id=UUID(result["run_id"]))
        self.assertEqual(repeated["status"], "requires_attention")
        self.assertGreater(repeated["changed_source_receipts"], 0)
        new_run = await self.finish()
        self.assertEqual(new_run["dispositions"]["suppressed"], 3)
        self.assertTrue(new_run["local_manifest_complete"])

    async def test_legacy_forgetting_barrier_prevents_any_retain_dispatch(self):
        memory = await self.memory()
        async with self.f.pool.acquire() as c:
            await c.execute("""insert into memory_forgetting_tombstones(
                owner_id,source_memory_id,memory_type,identity_fingerprint,reason)
                values($1,$2,'semantic_fact',$3,'Fixture explicit forgetting')""", self.f.owner, memory, "a" * 64)
        report = await self.finish()
        self.assertEqual(report["dispositions"]["suppressed"], 1)
        async with self.f.pool.acquire() as c:
            row = await c.fetchrow("select status,generation from jarvis_knowledge_sources where owner_id=$1", self.f.owner)
            retain = await c.fetchrow("select status,dispatch_count from jarvis_memory_outbox where operation='retain'")
        self.assertEqual((row["status"], row["generation"]), ("forgotten", 2))
        self.assertEqual((retain["status"], retain["dispatch_count"]), ("cancelled", 0))

    async def test_credentials_and_foreign_child_rows_are_never_copied(self):
        claim = await self.memory()
        async with self.f.pool.acquire() as c:
            await c.execute("""insert into google_oauth_tokens(owner_id,access_token,refresh_token)
                values($1,'PRIVATE_ACCESS_FIXTURE','PRIVATE_REFRESH_FIXTURE')""", self.f.owner)
            await c.execute("""insert into memory_evidence(
                owner_id,memory_item_id,evidence_type,evidence_ref_id,excerpt,evidence_time,confidence)
                values($1,$2,'message',$3,'PRIVATE_FOREIGN_EXCERPT',clock_timestamp(),0.7)""", self.f.other, claim, str(uuid4()))
        report = await self.finish()
        self.assertEqual(report["dispositions"]["excluded_credentials"], 1)
        self.assertEqual(report["population_counts"]["memory_evidence"], 0)
        async with self.f.pool.acquire() as c:
            self.assertFalse(await c.fetchval("""select exists(select 1 from jarvis_context_payloads
                where content::text like '%PRIVATE_%')"""))
        self.assertEqual((await self.source("memory_items", claim)).content["attachments"], [])

    async def test_missing_evidence_and_exact_decimal_remain_honest_original_data(self):
        claim = await self.memory(evidence=("message", uuid4()))
        async with self.f.pool.acquire() as c:
            await c.execute("""update memory_items set metadata=
                '{"weight_kg": 123.000000000000000001, "repetitions": 8}'::jsonb where id=$1""", claim)
        await self.finish()
        source = await self.source("memory_items", claim)
        self.assertEqual(source.content["record"]["metadata"]["weight_kg"], "123.000000000000000001")
        self.assertEqual(source.content["record"]["metadata"]["repetitions"], 8)
        self.assertEqual(source.content["references"][0]["state"], "missing_or_outside_owner_scope")
        self.assertIsNone(source.source_time)

    async def test_cycle_is_quarantined_instead_of_inventing_evidence_order(self):
        first, second = await self.memory(), await self.memory()
        async with self.f.pool.acquire() as c:
            for source, target in ((first, second), (second, first)):
                await c.execute("""insert into memory_evidence(owner_id,memory_item_id,evidence_type,evidence_ref_id,evidence_time,confidence)
                    values($1,$2,'memory',$3,clock_timestamp(),0.7)""", self.f.owner, source, str(target))
        report = await self.finish()
        self.assertEqual(report["status"], "requires_attention")
        self.assertEqual(report["dispositions"]["quarantined"], 4)
        async with self.f.pool.acquire() as c:
            self.assertEqual(await c.fetchval("select count(*) from jarvis_knowledge_sources"), 0)

    async def test_missing_previous_origin_prevents_new_snapshot_readiness(self):
        message = await self.message()
        await self.finish()
        async with self.f.pool.acquire() as c:
            await c.execute("delete from jarvis_messages where id=$1", message)
        report = await self.finish()
        self.assertEqual((report["status"], report["previous_origins_missing"]), ("requires_attention", 1))

    async def test_source_table_locks_freeze_the_input_until_batch_release(self):
        message = await self.message()
        async def change():
            async with self.f.pool.acquire() as c:
                await c.execute("update jarvis_messages set content='After the snapshot' where id=$1", message)
        task = None
        try:
            async with self.importer.snapshot(self.f.owner) as snapshot:
                self.assertEqual(snapshot.rows[("jarvis_messages", str(message))].value["content"],
                                 "The robot still needs calibration.")
                task = asyncio.create_task(change())
                with self.assertRaises(asyncio.TimeoutError):
                    await asyncio.wait_for(asyncio.shield(task), 0.1)
            await asyncio.wait_for(task, 5)
        finally:
            if task is not None and not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

    async def test_bounds_and_foreign_scope_cannot_create_an_import(self):
        await self.message()
        with self.assertRaises(SourceError) as error:
            await self.importer.run(self.f.owner, self.f.area, max_rows=1)
        self.assertEqual(error.exception.code, "legacy_snapshot_too_large")
        with self.assertRaises(SourceError):
            await self.importer.run(self.f.other, self.f.area)
        async with self.f.pool.acquire() as c:
            self.assertEqual(await c.fetchval("select count(*) from jarvis_legacy_import_runs"), 0)

    async def test_owned_attachment_with_foreign_parent_is_quarantined(self):
        foreign = await self.memory(owner=self.f.other)
        async with self.f.pool.acquire() as c:
            await c.execute("""insert into memory_evidence(
                owner_id,memory_item_id,evidence_type,evidence_ref_id,evidence_time,confidence)
                values($1,$2,'message',$3,clock_timestamp(),0.7)""", self.f.owner, foreign, str(uuid4()))
        report = await self.finish()
        self.assertEqual(report["status"], "requires_attention")
        self.assertEqual(report["dispositions"]["quarantined"], 1)
        self.assertEqual(report["recorded_records"], report["expected_records"])

    async def test_writer_committed_during_lock_wait_is_in_the_snapshot(self):
        message = await self.message("Before writer")
        snapshot_task = None
        async def observe():
            async with self.importer.snapshot(self.f.owner) as snapshot:
                return snapshot.rows[("jarvis_messages", str(message))].value["content"]
        async with self.f.pool.acquire() as c:
            async with c.transaction():
                await c.execute("update jarvis_messages set content='Committed writer value' where id=$1", message)
                snapshot_task = asyncio.create_task(observe())
                with self.assertRaises(asyncio.TimeoutError):
                    await asyncio.wait_for(asyncio.shield(snapshot_task), 0.1)
        self.assertEqual(await asyncio.wait_for(snapshot_task, 5), "Committed writer value")

    async def test_unmapped_source_column_refuses_silent_partial_copy(self):
        await self.message()
        async with self.f.pool.acquire() as c:
            await c.execute("alter table jarvis_messages add column fixture_new_original_field text")
        with self.assertRaises(SourceError) as error:
            await self.importer.run(self.f.owner, self.f.area)
        self.assertEqual(error.exception.code, "legacy_source_schema_changed")

    async def shared_message_original(self):
        message = await self.message("One exact original imported into separate areas.")
        await self.finish()
        first = await self.source("jarvis_messages", message)
        async with self.f.pool.acquire() as c:
            other_area = await c.fetchval("""insert into jarvis_knowledge_areas
                select (jsonb_populate_record(null::jarvis_knowledge_areas,
                    to_jsonb(a)||jsonb_build_object('id',$2::uuid,'name','Other imported area'))).*
                from jarvis_knowledge_areas a where a.owner_id=$1 and a.id=$3 returning id""",
                self.f.owner, uuid4(), self.f.area)
        report = await self.importer.run(self.f.owner, other_area)
        self.assertEqual(report["status"], "complete")
        async with self.f.pool.acquire() as c:
            identity = await c.fetchval("""select id from jarvis_knowledge_sources
                where owner_id=$1 and area_id=$2 and namespace=$3 and source_key=$4""",
                self.f.owner, other_area, NAMESPACE, "jarvis_messages/" + str(message))
        second = await self.f.sources.read(self.f.owner, other_area, identity)
        return message, first, second

    async def test_cross_area_import_keeps_shared_original_and_reports_retained_owner(self):
        message, first, second = await self.shared_message_original()
        for _ in range(2):
            result = await self.f.sources.delete_source(
                self.f.owner, first.area_id, first.source_id, expected_revision=first.revision)
            detail = result["legacy_message_original"]
            self.assertEqual(detail["state"], "shared_original_retained")
            self.assertEqual(detail["shared_original_sources"]["items"], [{
                "source_id": str(second.source_id), "area_id": str(second.area_id),
                "revision": second.revision, "state": "active",
            }])
            self.assertFalse(detail["shared_original_sources"]["has_more"])
        receipt = await SourceDeletionService(self.f.sources).inspect(
            self.f.owner, first.area_id, first.source_id)
        self.assertEqual(receipt["state"], "requires_attention")
        self.assertEqual(receipt["legacy_message_original"]["state"], "shared_original_retained")
        self.assertEqual((await self.f.sources.read(self.f.owner, second.area_id, second.source_id)).source_id,
                         second.source_id)
        async with self.f.pool.acquire() as c:
            row = await c.fetchrow("select content,source_payload_id from jarvis_messages where id=$1", message)
        self.assertEqual(row["content"], "One exact original imported into separate areas.")
        self.assertIsNone(row["source_payload_id"])

    async def test_last_retained_import_removes_the_shared_raw_original(self):
        message, first, second = await self.shared_message_original()
        await self.f.sources.delete_source(
            self.f.owner, first.area_id, first.source_id, expected_revision=first.revision)
        result = await self.f.sources.delete_source(
            self.f.owner, second.area_id, second.source_id, expected_revision=second.revision)
        self.assertEqual(result["legacy_message_original"]["state"], "source_bound_redacted")
        async with self.f.pool.acquire() as c:
            row = await c.fetchrow("select content,source_payload_id from jarvis_messages where id=$1", message)
        self.assertEqual(row["content"], "")
        self.assertEqual(row["source_payload_id"], second.payload_id)
        for area in (first.area_id, second.area_id):
            report = await self.importer.run(self.f.owner, area)
            self.assertEqual(report["dispositions"].get("suppressed"), 1)
        async with self.f.pool.acquire() as c:
            self.assertFalse(await c.fetchval("""select exists(select 1 from jarvis_memory_outbox o
                join jarvis_knowledge_source_versions v on v.owner_id=o.owner_id and v.document_id=o.document_id
                where v.owner_id=$1 and v.source_id=any($2::uuid[])
                    and o.operation='retain' and o.status='pending')""",
                self.f.owner, [first.source_id, second.source_id]))

    async def test_explicit_group_removes_raw_original_only_after_its_retained_imports(self):
        message, first, second = await self.shared_message_original()
        async with self.f.sources.transaction(self.f.owner) as c:
            await self.f.sources.delete_sources_in_transaction(c, self.f.owner, [
                (s.area_id, s.source_id, s.revision) for s in (first, second)])
        async with self.f.pool.acquire() as c:
            self.assertEqual(await c.fetchval("select content from jarvis_messages where id=$1", message), "")
            self.assertEqual(await c.fetchval("""select count(*) from jarvis_knowledge_sources
                where owner_id=$1 and id=any($2::uuid[]) and status='deleted'""",
                self.f.owner, [first.source_id, second.source_id]), 2)

    async def test_partial_group_does_not_erase_a_third_retained_import(self):
        message, first, second = await self.shared_message_original()
        async with self.f.pool.acquire() as c:
            third_area = await c.fetchval("""insert into jarvis_knowledge_areas
                select (jsonb_populate_record(null::jarvis_knowledge_areas,
                    to_jsonb(a)||jsonb_build_object('id',$2::uuid,'name','Third imported area'))).*
                from jarvis_knowledge_areas a where a.owner_id=$1 and a.id=$3 returning id""",
                self.f.owner, uuid4(), self.f.area)
        self.assertEqual((await self.importer.run(self.f.owner, third_area))["status"], "complete")
        async with self.f.sources.transaction(self.f.owner) as c:
            results = await self.f.sources.delete_sources_in_transaction(c, self.f.owner, [
                (s.area_id, s.source_id, s.revision) for s in (first, second)])
        self.assertTrue(all(r["legacy_message_original"]["state"] == "shared_original_retained" for r in results))
        async with self.f.pool.acquire() as c:
            self.assertEqual(await c.fetchval("select content from jarvis_messages where id=$1", message),
                             "One exact original imported into separate areas.")
            self.assertIsNone(await c.fetchval("select source_payload_id from jarvis_messages where id=$1", message))

    async def test_deleted_import_removes_exact_raw_message_and_preserves_controls(self):
        message = await self.message("Exact imported original.")
        control = await self.message("Independent original.", role="assistant")
        foreign = await self.message("Foreign original.", owner=self.f.other)
        await self.finish()
        source = await self.source("jarvis_messages", message)
        result = await self.f.sources.delete_source(
            self.f.owner, self.f.area, source.source_id, expected_revision=1)
        self.assertEqual(result["legacy_message_original"]["state"], "source_bound_redacted")
        async with self.f.pool.acquire() as c:
            target = await c.fetchrow("select * from jarvis_messages where id=$1", message)
            self.assertEqual(target["content"], "")
            self.assertEqual(target["source_payload_id"], source.payload_id)
            self.assertNotIn("Exact imported original.", target["metadata"])
            self.assertEqual(await c.fetchval("select content from jarvis_messages where id=$1", control),
                             "Independent original.")
            self.assertEqual(await c.fetchval("select content from jarvis_messages where id=$1", foreign),
                             "Foreign original.")
        replay = await self.finish()
        self.assertTrue(replay["local_manifest_complete"])
        async with self.f.pool.acquire() as c:
            self.assertFalse(await c.fetchval("""select exists(select 1 from jarvis_memory_outbox o
                join jarvis_knowledge_source_versions v on v.owner_id=o.owner_id and v.document_id=o.document_id
                where v.owner_id=$1 and v.source_id=$2 and o.operation='retain' and o.status='pending')""",
                self.f.owner, source.source_id))
        receipt = await SourceDeletionService(self.f.sources).inspect(self.f.owner, self.f.area, source.source_id)
        self.assertEqual(receipt["legacy_message_original"]["state"], "source_bound_redacted")

    async def test_repeated_deletion_cannot_restore_raw_text_or_detach_it(self):
        message = await self.message()
        await self.finish()
        source = await self.source("jarvis_messages", message)
        for _ in range(2):
            await self.f.sources.delete_source(self.f.owner, self.f.area, source.source_id, expected_revision=1)
        async with self.f.pool.acquire() as c:
            for assignment in ("content='Restored text'", "source_payload_id=null"):
                with self.subTest(assignment=assignment), self.assertRaises(asyncpg.CheckViolationError):
                    async with c.transaction():
                        await c.execute("update jarvis_messages set " + assignment + " where id=$1", message)
            self.assertEqual(await c.fetchval("select content from jarvis_messages where id=$1", message), "")

    async def test_changed_message_text_is_preserved_and_reported(self):
        message = await self.message()
        await self.finish()
        source = await self.source("jarvis_messages", message)
        async with self.f.pool.acquire() as c:
            await c.execute("update jarvis_messages set content='A newer independent revision' where id=$1", message)
        result = await self.f.sources.delete_source(self.f.owner, self.f.area, source.source_id, expected_revision=1)
        self.assertEqual(result["legacy_message_original"]["state"], "current_origin_changed")
        receipt = await SourceDeletionService(self.f.sources).inspect(self.f.owner, self.f.area, source.source_id)
        self.assertEqual(receipt["state"], "requires_attention")
        self.assertEqual(receipt["legacy_message_original"]["state"], "current_origin_changed")
        async with self.f.pool.acquire() as c:
            self.assertEqual(await c.fetchval("select content from jarvis_messages where id=$1", message),
                             "A newer independent revision")
            self.assertIsNone(await c.fetchval("select source_payload_id from jarvis_messages where id=$1", message))

    async def test_changed_metadata_is_not_erased_even_when_text_matches(self):
        message = await self.message()
        await self.finish()
        source = await self.source("jarvis_messages", message)
        async with self.f.pool.acquire() as c:
            await c.execute("""update jarvis_messages set metadata='{"new":"independent metadata"}' where id=$1""", message)
        result = await self.f.sources.delete_source(self.f.owner, self.f.area, source.source_id, expected_revision=1)
        self.assertEqual(result["legacy_message_original"]["state"], "current_origin_changed")
        async with self.f.pool.acquire() as c:
            self.assertEqual(json.loads(await c.fetchval("select metadata::text from jarvis_messages where id=$1", message)),
                             {"new": "independent metadata"})
            self.assertEqual(await c.fetchval("select content from jarvis_messages where id=$1", message),
                             "The robot still needs calibration.")

    async def test_deleting_imported_copy_preserves_its_independent_bound_original(self):
        message, parent = await self.linked_message()
        await self.finish()
        source = await self.source("jarvis_messages", message)
        result = await self.f.sources.delete_source(self.f.owner, self.f.area, source.source_id, expected_revision=1)
        self.assertEqual(result["legacy_message_original"]["state"], "separate_source_retained")
        async with self.f.pool.acquire() as c:
            self.assertEqual(await c.fetchval("select source_payload_id from jarvis_messages where id=$1", message), parent)
            self.assertEqual(await c.fetchval("select content from jarvis_messages where id=$1", message),
                             "The robot still needs calibration.")
            self.assertEqual(await c.fetchval("select availability from jarvis_context_payloads where id=$1", parent),
                             "available")

    async def test_repeated_delete_repairs_an_original_left_by_the_old_writer(self):
        message = await self.message()
        await self.finish()
        source = await self.source("jarvis_messages", message)
        # Reproduce the previous writer without modifying a migration or ledger.
        with patch("app.knowledge.sources.delete_imported_message_original", new=AsyncMock(return_value=None)):
            await self.f.sources.delete_source(self.f.owner, self.f.area, source.source_id, expected_revision=1)
        async with self.f.pool.acquire() as c:
            self.assertEqual(await c.fetchval("select content from jarvis_messages where id=$1", message),
                             "The robot still needs calibration.")
            self.assertIsNone(await c.fetchval("select content from jarvis_context_payloads where id=$1", source.payload_id))
        await self.f.sources.delete_source(self.f.owner, self.f.area, source.source_id, expected_revision=1)
        async with self.f.pool.acquire() as c:
            self.assertEqual(await c.fetchval("select content from jarvis_messages where id=$1", message), "")
            self.assertEqual(await c.fetchval("select source_payload_id from jarvis_messages where id=$1", message),
                             source.payload_id)

    async def test_removed_payload_can_be_bound_only_as_content_free_exact_repair(self):
        message = await self.message()
        unrelated = await self.message("Independent row")
        await self.finish()
        source = await self.source("jarvis_messages", message)
        async with self.f.pool.acquire() as c:
            await c.execute("update jarvis_context_payloads set availability='expired',content=null where id=$1",
                            source.payload_id)
            metadata = json.dumps({"response_kind":"source_bound_unavailable",
                                   "source_status":"expired","source_payload_id":str(source.payload_id)})
            for target, content, body in ((message, "Not empty", metadata),
                                           (unrelated, "", metadata), (message, "", "{}")):
                with self.subTest(target=target, content=content, metadata=body):
                    with self.assertRaises(asyncpg.CheckViolationError):
                        async with c.transaction():
                            await c.execute("""update jarvis_messages set source_payload_id=$2,
                                content=$3,metadata=$4::jsonb where id=$1""", target, source.payload_id, content, body)
        await self.f.sources.delete_source(self.f.owner, self.f.area, source.source_id, expected_revision=1)
        async with self.f.pool.acquire() as c:
            self.assertEqual(await c.fetchval("select content from jarvis_messages where id=$1", message), "")
            self.assertEqual(await c.fetchval("select content from jarvis_messages where id=$1", unrelated), "Independent row")

    async def test_digest_preserves_exact_decimal_metadata_and_utc_timestamp(self):
        message = await self.message("Synthetic unicode: \\u00e9")
        async with self.f.pool.acquire() as c:
            await c.execute("""update jarvis_messages set
                metadata='{"weight":123.000000000000000001,"count":8,"nested":[0.0000000001]}'::jsonb,
                created_at='2026-09-10 12:34:56.123456+05:30'::timestamptz where id=$1""", message)
        await self.finish()
        source = await self.source("jarvis_messages", message)
        async with self.f.sources.transaction(self.f.owner) as c:
            await c.execute("set local timezone='Pacific/Auckland'")
            result = await self.f.sources.delete_source_in_transaction(
                c, self.f.owner, self.f.area, source.source_id, expected_revision=1)
        self.assertEqual(result["legacy_message_original"]["state"], "source_bound_redacted")

    async def test_same_owner_import_lock_refuses_cleanup_without_deadlock(self):
        message = await self.message()
        await self.finish()
        source = await self.source("jarvis_messages", message)
        async with self.importer.snapshot(self.f.owner):
            with self.assertRaises(SourceError) as error:
                await asyncio.wait_for(self.f.sources.delete_source(
                    self.f.owner, self.f.area, source.source_id, expected_revision=1), 3)
            self.assertEqual(error.exception.code, "legacy_original_cleanup_busy")
        self.assertEqual((await self.source("jarvis_messages", message)).source_id, source.source_id)
        await self.f.sources.delete_source(self.f.owner, self.f.area, source.source_id, expected_revision=1)

    async def test_other_owner_frozen_table_lock_refuses_cleanup_and_can_retry(self):
        message = await self.message()
        await self.finish()
        source = await self.source("jarvis_messages", message)
        async with self.importer.snapshot(self.f.other):
            with self.assertRaises(SourceError) as error:
                await asyncio.wait_for(self.f.sources.delete_source(
                    self.f.owner, self.f.area, source.source_id, expected_revision=1), 3)
            self.assertEqual(error.exception.code, "legacy_original_cleanup_busy")
        await self.f.sources.delete_source(self.f.owner, self.f.area, source.source_id, expected_revision=1)

    async def test_deletion_failure_rolls_back_message_binding_and_redaction(self):
        message = await self.message()
        await self.finish()
        source = await self.source("jarvis_messages", message)
        with patch.object(self.f.sources, "_retire_documents", side_effect=RuntimeError("Synthetic rollback")):
            with self.assertRaises(RuntimeError):
                await self.f.sources.delete_source(self.f.owner, self.f.area, source.source_id, expected_revision=1)
        self.assertEqual((await self.source("jarvis_messages", message)).source_id, source.source_id)
        async with self.f.pool.acquire() as c:
            self.assertIsNone(await c.fetchval("select source_payload_id from jarvis_messages where id=$1", message))
            self.assertEqual(await c.fetchval("select content from jarvis_messages where id=$1", message),
                             "The robot still needs calibration.")

    async def test_corrected_source_still_removes_its_exact_historical_message(self):
        message = await self.message()
        await self.finish()
        source = await self.source("jarvis_messages", message)
        await self.lifecycle.correct(self.f.owner, self.f.area, source.source_id, expected_revision=1,
                                     replacement="Explicit correction", request=await self.request())
        await self.f.sources.delete_source(self.f.owner, self.f.area, source.source_id, expected_revision=2)
        async with self.f.pool.acquire() as c:
            self.assertEqual(await c.fetchval("select content from jarvis_messages where id=$1", message), "")
            self.assertEqual(await c.fetchval("select source_payload_id from jarvis_messages where id=$1", message),
                             source.payload_id)

    async def test_scheduled_expiry_uses_the_same_original_cleanup(self):
        message = await self.message()
        await self.finish()
        source = await self.source("jarvis_messages", message)
        retention = SourceRetentionService(self.f.sources, self.tasks)
        await retention.set_policy(self.f.owner, self.f.area, source.source_id,
            expected_source_revision=source.revision, expected_source_generation=source.generation,
            expected_policy_revision=0, expires_at=datetime.now(UTC)-timedelta(seconds=1),
            request=await self.request())
        results = await retention.process_due(self.f.owner)
        self.assertEqual(results[0]["status"], "expired")
        async with self.f.pool.acquire() as c:
            self.assertEqual(await c.fetchval("select content from jarvis_messages where id=$1", message), "")

    async def test_unmapped_original_column_refuses_deletion_without_partial_effects(self):
        message = await self.message()
        await self.finish()
        source = await self.source("jarvis_messages", message)
        async with self.f.pool.acquire() as c:
            await c.execute("alter table jarvis_messages add column fixture_new_original text")
        with self.assertRaises(SourceError) as error:
            await self.f.sources.delete_source(self.f.owner, self.f.area, source.source_id, expected_revision=1)
        self.assertEqual(error.exception.code, "legacy_source_schema_changed")
        self.assertEqual((await self.source("jarvis_messages", message)).source_id, source.source_id)

    async def linked_message(self):
        message = await self.message(role="assistant")
        async with self.f.pool.acquire() as c:
            payload = await c.fetchval("""insert into jarvis_context_payloads(owner_id,availability,content)
                values($1,'available','{"text":"Exact synthetic original"}'::jsonb) returning id""", self.f.owner)
            await c.execute("update jarvis_messages set source_payload_id=$2 where id=$1", message, payload)
        return message, payload

    async def test_imported_message_keeps_its_existing_payload_dependency(self):
        message, payload = await self.linked_message()
        await self.finish()
        source = await self.source("jarvis_messages", message)
        async with self.f.pool.acquire() as c:
            self.assertTrue(await c.fetchval("""select exists(select 1 from jarvis_context_payload_dependencies
                where owner_id=$1 and source_payload_id=$2 and dependent_payload_id=$3)""",
                self.f.owner, payload, source.payload_id))
            await c.execute("""update jarvis_context_payloads set availability='deleted',content=null
                where owner_id=$1 and id=$2""", self.f.owner, payload)
        with self.assertRaises(SourceError):
            await self.f.sources.read(self.f.owner, self.f.area, source.source_id)
        replay = await self.finish()
        self.assertEqual(replay["dispositions"].get("suppressed"), 1)

    async def test_removed_message_original_is_not_imported(self):
        _, payload = await self.linked_message()
        async with self.f.pool.acquire() as c:
            await c.execute("""update jarvis_context_payloads set availability='deleted',content=null
                where owner_id=$1 and id=$2""", self.f.owner, payload)
        result = await self.finish()
        self.assertTrue(result["local_manifest_complete"])
        self.assertEqual(result["dispositions"].get("suppressed"), 1)
        async with self.f.pool.acquire() as c:
            self.assertEqual(await c.fetchval("select count(*) from jarvis_knowledge_sources"), 0)

    async def test_ddl_committed_while_waiting_is_validated_after_locks(self):
        await self.message()
        async def observe():
            async with self.importer.snapshot(self.f.owner):
                self.fail("Unmapped column was silently accepted")
        async with self.f.pool.acquire() as c:
            async with c.transaction():
                await c.execute("alter table jarvis_messages add column fixture_concurrent_field text")
                task = asyncio.create_task(observe())
                with self.assertRaises(asyncio.TimeoutError):
                    await asyncio.wait_for(asyncio.shield(task), 0.1)
        with self.assertRaises(SourceError) as error:
            await asyncio.wait_for(task, 5)
        self.assertEqual(error.exception.code, "legacy_source_schema_changed")

    async def test_mixed_populations_preserve_historical_and_visual_evidence(self):
        stamp = datetime(2026, 9, 10, 12, tzinfo=UTC)
        user = await self.message()
        claim = await self.memory(evidence=("message", user))
        historical = await self.memory("Earlier calibration estimate, now retracted.")
        async with self.f.pool.acquire() as c:
            note = await c.fetchval("""insert into jarvis_notes(owner_id,title,body,updated_at)
                values($1,'Robot notebook','Calibration remains unfinished.',$2) returning id""", self.f.owner, stamp)
            frame = await c.fetchval("""insert into visual_frames(
                owner_id,session_id,visual_session_id,frame_sequence,source,captured_at,content_type,
                width,height,storage_bucket,storage_path,raw_image_expires_at,raw_image_status)
                values($1,$2,$3,1,'glasses',$4,'image/jpeg',432,768,'fixture-only',
                       'PRIVATE_UNUSED_IMAGE_PATH',$4,'expired') returning id""",
                self.f.owner,self.sessions[self.f.owner],uuid4(),stamp)
            observation = uuid4()
            await c.execute("""insert into perception_observations(
                id,owner_id,session_id,source,started_at,ended_at,summary,confidence,
                analyzer_provider,analyzer_model,analyzer_contract_version,primary_visual_frame_id,visible_text)
                values($1,$2,$3,'glasses',$4,$4,'Robot appears partly assembled.',0.6,
                       'fixture','synthetic-only',1,$5,'["Calibration pending"]'::jsonb)""",
                observation,self.f.owner,self.sessions[self.f.owner],stamp,frame)
            await c.execute("""insert into perception_observation_frames(observation_id,visual_frame_id,ordinal,role)
                values($1,$2,0,'primary')""", observation,frame)
            event = await c.fetchval("""insert into experience_events(
                owner_id,session_id,event_kind,started_at,summary)
                values($1,$2,'mixed',$3,'Legacy activity summary, not completion proof.') returning id""",
                self.f.owner,self.sessions[self.f.owner],stamp)
            await c.execute("""insert into experience_event_evidence(event_id,evidence_type,evidence_ref_id,occurred_at)
                values($1,'observation',$2,$3)""", event,observation,stamp)
            await c.execute("""insert into experience_event_keyframes(
                event_id,visual_frame_id,rank,reason,quality_score,selected_by)
                values($1,$2,1,'Synthetic prior selection',0.5,'deterministic')""", event,frame)
            await c.execute("""update memory_items set memory_type='episode',source_experience_event_id=$2,
                occurred_at=$3 where id=$1""", claim,event,stamp)
            await c.execute("""update memory_items set status='retracted',valid_from=$2,valid_to=$2
                where id=$1""", historical,stamp)
            await c.execute("""insert into memory_links(
                owner_id,source_memory_id,target_memory_id,relationship,confidence)
                values($1,$2,$3,'earlier_context',0.5)""", self.f.owner,claim,historical)
            entity = await c.fetchval("""insert into memory_entities(
                owner_id,entity_type,canonical_name,normalized_name,confidence,updated_at)
                values($1,'project','Fixture robot','fixture robot',0.8,$2) returning id""", self.f.owner,stamp)
            await c.execute("""insert into memory_entity_mentions(owner_id,entity_id,memory_item_id,role,confidence)
                values($1,$2,$3,'subject',0.8)""", self.f.owner,entity,claim)
            await c.execute("""insert into memory_entity_relations(
                owner_id,subject_entity_id,predicate,object_value,confidence,valid_from,supporting_memory_id)
                values($1,$2,'has_status','"unverified legacy description"'::jsonb,0.5,$3,$4)""",
                self.f.owner,entity,stamp,claim)
        result = await self.finish()
        self.assertTrue(result["local_manifest_complete"])
        self.assertEqual(result["dispositions"]["imported"], 7)
        self.assertEqual(result["dispositions"]["attached"], 7)
        observed = await self.source("perception_observations", observation)
        experienced = await self.source("experience_events", event)
        named = await self.source("memory_entities", entity)
        noted = await self.source("jarvis_notes", note)
        old = await self.source("memory_items", historical)
        self.assertEqual((observed.trust.value,experienced.trust.value,named.trust.value,noted.trust.value),
                         ("derived","derived","derived","external_content"))
        for original in (observed,experienced,named,noted):
            self.assertEqual(original.source_time,stamp)
        self.assertEqual(observed.content["record"]["visible_text"],["Calibration pending"])
        image = next(r for r in observed.content["references"] if r["table"] == "visual_frames")
        self.assertEqual(image["external_metadata"]["raw_image_status"],"expired")
        self.assertFalse(image["bytes_verified"])
        self.assertEqual(old.content["record"]["status"],"retracted")
        self.assertEqual(old.content["record"]["valid_from"],old.content["record"]["valid_to"])
        async with self.f.pool.acquire() as c:
            self.assertFalse(await c.fetchval("""select exists(select 1 from jarvis_context_payloads
                where content::text like '%PRIVATE_UNUSED_IMAGE_PATH%')"""))
        original = await self.source("memory_items",claim)
        await self.lifecycle.forget(self.f.owner,self.f.area,original.source_id,expected_revision=1,
                                    request=await self.request())
        with self.assertRaises(SourceError):
            await self.f.sources.read(self.f.owner,self.f.area,named.source_id)
        self.assertEqual((await self.f.sources.read(
            self.f.owner,self.f.area,observed.source_id)).source_id,observed.source_id)
