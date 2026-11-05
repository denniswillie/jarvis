"""Large frozen legacy evidence sets preserve every dependency and atomicity."""
from pathlib import Path
import os
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from app.knowledge.backfill import LegacyBackfill, NAMESPACE
from app.knowledge.models import SourceError
from app.knowledge.read_provenance import SourceReadProvenance
from evals.second_brain.source_fixture import source_fixture


class InteractiveEvidenceBoundTests(unittest.IsolatedAsyncioTestCase):
    async def test_normal_interactive_limit_is_unchanged(self):
        provenance = SourceReadProvenance(SimpleNamespace(pool=None))
        self.assertEqual(provenance.reference_limit, 32)
        with self.assertRaises(SourceError) as caught:
            await provenance.bind_in_transaction(
                SimpleNamespace(is_in_transaction=lambda: True), "fixture", uuid4(),
                [{}] * 33, AsyncMock())
        self.assertEqual(caught.exception.code, "knowledge_read_evidence_limit")


@unittest.skipUnless(os.getenv("JARVIS_SECOND_BRAIN_TEST_DB_URL") and
                     os.getenv("JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"),
                     "Manifested database fixture required")
class BackfillEvidenceFanoutTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.f = await self.enterAsyncContext(source_fixture(
            os.environ["JARVIS_SECOND_BRAIN_TEST_DB_URL"],
            Path(os.environ["JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"]), test_name=self.id()))
        self.importer = LegacyBackfill(self.f.sources)
        self.run_id = uuid4()
        self.parents = sorted((uuid4() for _ in range(56)), key=str)
        async with self.f.pool.acquire() as connection:
            session = await connection.fetchval(
                "insert into jarvis_sessions(owner_id) values($1) returning id", self.f.owner)
            await connection.executemany(
                "insert into jarvis_messages(id,owner_id,session_id,role,content,modality) "
                "values($1,$2,$3,'user','Synthetic supporting report','audio')",
                [(identity, self.f.owner, session) for identity in self.parents])
            self.claim = await connection.fetchval("""insert into memory_items(
                owner_id,memory_type,source,content,title,is_explicit,status,importance,confidence)
                values($1,'semantic_fact','user_statement','Synthetic aggregate','Fixture',true,'active',0.5,0.5)
                returning id""", self.f.owner)
            await connection.executemany("""insert into memory_evidence(
                owner_id,memory_item_id,evidence_type,evidence_ref_id,excerpt,evidence_time,confidence)
                values($1,$2,'message',$3,'Synthetic evidence',clock_timestamp(),0.7)""",
                [(self.f.owner, self.claim, str(identity)) for identity in self.parents])

    async def imported(self, table, identity):
        async with self.f.pool.acquire() as connection:
            source = await connection.fetchval(
                "select id from jarvis_knowledge_sources where owner_id=$1 and area_id=$2 "
                "and namespace=$3 and source_key=$4", self.f.owner, self.f.area, NAMESPACE,
                table + "/" + str(identity))
        return await self.f.sources.read(self.f.owner, self.f.area, source)

    async def test_full_fanout_replay_and_late_parent_forgetting(self):
        result = await self.importer.run(self.f.owner, self.f.area, run_id=self.run_id)
        self.assertTrue(result["local_manifest_complete"])
        self.assertEqual(result["dispositions"]["imported"], 57)
        source = await self.imported("memory_items", self.claim)
        async with self.f.pool.acquire() as connection:
            dependencies = await connection.fetchval(
                "select count(*) from jarvis_context_payload_dependencies where owner_id=$1 "
                "and dependent_payload_id=$2", self.f.owner, source.payload_id)
            citations = await connection.fetchval(
                "select count(*) from jarvis_context_memory_sources where owner_id=$1 "
                "and dependent_payload_id=$2", self.f.owner, source.payload_id)
        self.assertEqual((dependencies, citations), (56, 56))
        replay = await self.importer.run(self.f.owner, self.f.area, run_id=self.run_id)
        self.assertTrue(replay["local_manifest_complete"])
        self.assertEqual(replay["sources_processed_in_batch"], 0)
        later_parent = await self.imported("jarvis_messages", self.parents[-1])
        await self.f.sources.delete_source(self.f.owner, self.f.area, later_parent.source_id,
                                           expected_revision=later_parent.revision)
        with self.assertRaises(SourceError):
            await self.f.sources.read(self.f.owner, self.f.area, source.source_id)

    async def test_second_group_failure_rolls_back_original_receipt_and_outbox(self):
        original = self.importer.provenance.bind_in_transaction
        groups = 0

        async def fail_later(*args, **kwargs):
            nonlocal groups
            groups += 1
            if groups == 2:
                raise SourceError("fixture_late_evidence_failure", "Synthetic late-group failure")
            return await original(*args, **kwargs)

        with patch.object(self.importer.provenance, "bind_in_transaction", side_effect=fail_later):
            with self.assertRaises(SourceError):
                await self.importer.run(self.f.owner, self.f.area, run_id=self.run_id)
        self.assertEqual(groups, 2)
        async with self.f.pool.acquire() as connection:
            self.assertEqual(await connection.fetchval(
                "select count(*) from jarvis_knowledge_sources where owner_id=$1 and namespace=$2 and source_key=$3",
                self.f.owner, NAMESPACE, "memory_items/" + str(self.claim)), 0)
            self.assertEqual(await connection.fetchval(
                "select count(*) from jarvis_legacy_import_records where owner_id=$1 and run_id=$2 and population='memory_items'",
                self.f.owner, self.run_id), 0)
            self.assertEqual(await connection.fetchval(
                "select count(*) from jarvis_memory_outbox where owner_id=$1", self.f.owner), 56)
            self.assertEqual(await connection.fetchval(
                "select count(*) from jarvis_context_payloads where owner_id=$1 "
                "and content#>>'{record,id}'=$2", self.f.owner, str(self.claim)), 0)
        retry = await self.importer.run(self.f.owner, self.f.area, run_id=self.run_id)
        self.assertTrue(retry["local_manifest_complete"])
        self.assertEqual(retry["dispositions"]["imported"], 57)
