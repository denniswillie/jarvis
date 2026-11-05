"""Opt-in isolated SQL regressions; no provider, model or native execution.

Requires the established JARVIS_SECOND_BRAIN_TEST_DB_URL and
JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR gates. Main owns actual execution.
The legacy case explicitly constructs the old current binding; it does not
replace the fixed production capture or claim a native watcher root match.
"""
import asyncio
from contextlib import asynccontextmanager
import json
import os
from pathlib import Path
import unittest
from uuid import uuid4

import asyncpg

from app.knowledge.models import SourceError
from app.knowledge.observation_inputs import ObservationInputCapture
from app.knowledge.publication import CanonicalSourcePublication
from app.knowledge.sources import SourceRepository
from app.migrations.runner import apply_migrations, load_migrations, quote_schema
from app.runtime.feature_config import TaskLimits
from app.tasks.context_events import ContextIngress
from app.tasks.context_history import canonical_history_payload
from app.tasks.context_repository import ContextRepository
from app.tasks.repository import TaskRepository
from evals.second_brain.fixtures import DatabaseFixture
from test_notification_ledger_database import cleanup_fixture_resources
from test_perception_pending_transcript import capture_context, capture_request


@unittest.skipUnless(os.getenv("JARVIS_SECOND_BRAIN_TEST_DB_URL") and
                     os.getenv("JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"),
                     "Explicit isolated database and fixture manifest directory required")
class PendingTranscriptSQLTests(unittest.IsolatedAsyncioTestCase):
    @asynccontextmanager
    async def fixture(self):
        dsn = os.environ["JARVIS_SECOND_BRAIN_TEST_DB_URL"]
        fixture = DatabaseFixture(dsn, Path(os.environ["JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"]),
                                  test_name=self.id())
        connection = pool = None
        report = {"status": "running", "cleanup": None}
        try:
            connection = await asyncpg.connect(dsn, timeout=10, command_timeout=90)
            schema = await fixture.create_schema(connection)
            migrations = tuple(item for item in load_migrations() if item.version <= 39)
            self.assertEqual(migrations[-1].version, 39)
            await asyncio.wait_for(apply_migrations(connection, schema=schema, fresh=True,
                                                   migrations=migrations), timeout=240)
            path_sql = f"set search_path to {quote_schema(schema)},pg_catalog,extensions"

            async def setup(c):
                await c.execute(path_sql)
                await c.execute("set lock_timeout='8s'; set statement_timeout='20s'")

            await setup(connection)
            pool = await asyncpg.create_pool(dsn, min_size=1, max_size=1, timeout=10,
                                            command_timeout=25, setup=setup)
            self.connection, self.owner, self.other_owner = connection, *fixture.owners
            self.area = await connection.fetchval("insert into jarvis_knowledge_areas(owner_id,name,domain) "
                "values($1,'Pending transcript fixture','test') returning id", self.owner)
            self.session = await connection.fetchval(
                "insert into jarvis_sessions(owner_id) values($1) returning id", self.owner)
            self.sources = SourceRepository(pool)
            self.tasks = TaskRepository(pool, TaskLimits(lease_seconds=300))
            self.context = ContextRepository(self.tasks,
                publication=CanonicalSourcePublication(self.sources, self.area))
            self.capture = ObservationInputCapture(self.context)
            self.ingress_id = "input:" + uuid4().hex
            async with asyncio.timeout(90):
                yield
            report["status"] = "passed"
        except BaseException:
            report["status"] = "failed"
            raise
        finally:
            if fixture.manifest["schemas"]:
                await cleanup_fixture_resources(fixture, report, lambda: None,
                                                pool=pool, connection=connection)
            elif connection is not None:
                await connection.close(timeout=5)
            print(json.dumps({"test": self.id(), "fixture": str(fixture.path), **report}))

    async def append(self, text, *, revision=0):
        return await self.context.append(self.owner, ContextIngress(session_id=self.session,
            source_namespace="voice:pending-transcript-sql", source_event_id=self.ingress_id,
            receipt_id=str(uuid4()), expected_revision=revision, kind="user_transcript", trust="user_input",
            finality="unavailable" if text is None else "revised" if revision else "final",
            payload={"text": text, "transcript_status": "pending" if text is None else "finalized",
                     "input_id": self.ingress_id, "audio_provider": "controlled-fixture"}))

    async def prepare(self):
        when = await self.connection.fetchval("select clock_timestamp()")
        return await self.capture.prepare(capture_request(self.owner, self.session, when),
                                          capture_context(self.owner, self.session))

    async def locked(self, *roots, owner=None):
        async with self.connection.transaction():
            return await self.connection.fetchval("select lock_jarvis_copy_sources($1,$2::uuid[])",
                                                  owner or self.owner, list(roots))

    async def descendant(self, parent):
        async with self.tasks.transaction(self.owner) as connection:
            return await self.tasks.payload(connection, self.owner, {"controlled_result": "synthetic"},
                                            source_payload_ids=(parent,))

    async def assert_current_binding(self, snapshot, record):
        rows = await self.connection.fetch("select event_id,event_revision,source_payload_id,source_usage "
            "from jarvis_context_event_dependencies where owner_id=$1 and dependent_payload_id=$2",
            self.owner, snapshot.payload_id)
        self.assertEqual([tuple(row) for row in rows], [(record.id, record.revision, record.payload_id, "current")])
        self.assertTrue(await self.connection.fetchval("select exists(select 1 "
            "from jarvis_context_payload_dependencies where owner_id=$1 and source_payload_id=$2 "
            "and dependent_payload_id=$3)", self.owner, record.payload_id, snapshot.payload_id))

    async def assert_removed(self, prepared, snapshot, child):
        self.assertFalse(await self.locked(snapshot.payload_id))
        self.assertFalse(await self.locked(child))
        rows = await self.connection.fetch("select availability,content from jarvis_context_payloads "
            "where owner_id=$1 and id=any($2::uuid[])", self.owner, [snapshot.payload_id, child])
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row["availability"] != "available" and row["content"] is None for row in rows))
        with self.assertRaises(SourceError) as raised:
            await self.capture.validate(prepared, snapshot)
        self.assertEqual(raised.exception.code, "observation_input_removed")

    async def test_pending_finalization_revokes_legacy_binding_not_fixed_capture(self):
        async with self.fixture():
            pending = await self.append(None)
            self.assertEqual((pending.finality, pending.availability), ("unavailable", "available"))
            old_voice, roots, _ = canonical_history_payload({"records": [pending], "total": 1})
            self.assertEqual(roots, (pending.payload_id,))
            self.assertEqual(pending.dependency_usage, "current")
            # Reproduce the old selected binding with the real SQL dependency guards.
            async with self.tasks.transaction(self.owner) as connection:
                old = await self.tasks.payload(connection, self.owner,
                    {"format": "fixture.legacy_visual_selection", "voice_context": old_voice}, source_payload_ids=roots)
                await connection.execute("insert into jarvis_context_event_dependencies "
                    "(owner_id,dependent_payload_id,event_id,event_revision,source_payload_id,source_usage) "
                    "values($1,$2,$3,$4,$5,'current')", self.owner, old, pending.id, pending.revision, pending.payload_id)
            old_child = await self.descendant(old)
            old_grandchild = await self.descendant(old_child)
            prepared, fixed = await self.prepare()
            fixed_child = await self.descendant(fixed.payload_id)
            self.assertEqual(prepared.voice_context["records"], [])
            self.assertTrue(any(gap["reason"] == "transcript_not_finalized" for gap in prepared.input_context_gaps))
            self.assertEqual(await self.connection.fetchval("select count(*) from jarvis_context_event_dependencies "
                "where owner_id=$1 and dependent_payload_id=$2", self.owner, fixed.payload_id), 0)
            self.assertEqual(await self.connection.fetchval("select count(*) from jarvis_context_payload_dependencies "
                "where owner_id=$1 and dependent_payload_id=$2", self.owner, fixed.payload_id), 0)
            self.assertTrue(await self.locked(old, old_child, old_grandchild, fixed.payload_id, fixed_child))
            finalized = await self.append("The synthetic marker is blue.", revision=1)
            self.assertEqual(finalized.revision, 2)
            for root in (old, old_child, old_grandchild):
                self.assertFalse(await self.locked(root))
            self.assertTrue(await self.locked(pending.payload_id, finalized.payload_id, fixed.payload_id, fixed_child))
            await self.capture.validate(prepared, fixed)

    async def test_finalized_consumed_text_correction_still_revokes_input_and_result(self):
        async with self.fixture():
            original = await self.append("The synthetic marker is blue.")
            prepared, snapshot = await self.prepare()
            self.assertEqual(prepared.voice_context["records"][0]["content"]["text"], original.content["text"])
            await self.assert_current_binding(snapshot, original)
            child = await self.descendant(snapshot.payload_id)
            self.assertTrue(await self.locked(snapshot.payload_id, child))
            corrected = await self.append("Correction: the synthetic marker is red.", revision=1)
            await self.assert_removed(prepared, snapshot, child)
            self.assertTrue(await self.locked(original.payload_id))
            fresh, fresh_snapshot = await self.prepare()
            self.assertEqual(fresh.voice_context["records"][0]["content"]["text"], corrected.content["text"])
            await self.assert_current_binding(fresh_snapshot, corrected)
            await self.capture.validate(fresh, fresh_snapshot)

    async def test_finalized_source_owner_guard_and_deletion_still_revoke(self):
        async with self.fixture():
            original = await self.append("Retain only while its original is available.")
            prepared, snapshot = await self.prepare()
            await self.assert_current_binding(snapshot, original)
            child = await self.descendant(snapshot.payload_id)
            self.assertFalse(await self.locked(original.payload_id, owner=self.other_owner))
            self.assertFalse(await self.locked(snapshot.payload_id, owner=self.other_owner))
            self.assertTrue(await self.locked(original.payload_id, snapshot.payload_id, child))
            source = await self.connection.fetchval("select source_id from jarvis_knowledge_source_versions "
                "where owner_id=$1 and payload_id=$2", self.owner, original.payload_id)
            self.assertIsNotNone(source)
            await self.sources.delete_source(self.owner, self.area, source, expected_revision=1)
            self.assertFalse(await self.locked(original.payload_id))
            await self.assert_removed(prepared, snapshot, child)


if __name__ == "__main__":
    unittest.main()
