"""Native PostgreSQL integrity tests in manifested disposable schemas.

These do not touch production owner rows or activate background processing.
Run through evals.second_brain.run_tests with --live-db and --manifest-dir.
"""
import asyncio
from importlib.resources import files
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from uuid import uuid4

import asyncpg

from app.migrations.runner import (
    Migration, MigrationError, apply_migrations, baseline_sql, load_migrations,
    quote_schema, read_schema_readiness, render_fresh_schema,
)
from evals.second_brain.fixtures import DatabaseFixture

MIGRATION_VERSIONS = tuple(item.version for item in load_migrations())


class BaselineRenderingTests(unittest.TestCase):
    def test_render_uses_independent_baseline_even_without_a_generated_schema(self):
        with tempfile.TemporaryDirectory(prefix="jarvis-baseline-unit-") as temporary:
            root = Path(temporary)
            (root / "baseline.sql").write_text("create table baseline_fixture(id integer);\n", encoding="utf-8")

            def resources(package):
                self.assertEqual(package, "app.migrations")
                return root

            with patch("app.migrations.runner.files", side_effect=resources):
                rendered = render_fresh_schema((Migration(1, "fixture", "select 1;"),))
            self.assertTrue(rendered.startswith("create table baseline_fixture(id integer);"))
            self.assertIn("-- BEGIN GENERATED VERSIONED MIGRATIONS", rendered)
            self.assertIn("select 1;", rendered)


@unittest.skipUnless(os.getenv("JARVIS_SECOND_BRAIN_TEST_DB_URL") and os.getenv("JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"),
                     "Explicit second-brain DB test configuration and manifest directory required")
class FoundationDatabaseTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.dsn = os.environ["JARVIS_SECOND_BRAIN_TEST_DB_URL"]
        self.fixture = DatabaseFixture(self.dsn, Path(os.environ["JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"]), test_name=self.id())
        self.addAsyncCleanup(self.fixture.cleanup)
        self.connection = await asyncpg.connect(self.dsn, timeout=10, command_timeout=90)
        self.addAsyncCleanup(self.connection.close)
        self.schema = await self.fixture.create_schema(self.connection)
        self.owner, self.other_owner = self.fixture.owners
        await self.connection.execute(f"set search_path to {quote_schema(self.schema)}, pg_catalog, public, extensions")

    async def initialize(self):
        self.assertEqual(await apply_migrations(self.connection, schema=self.schema, fresh=True), MIGRATION_VERSIONS)

    async def payload(self, *, owner=None, availability="available"):
        return await self.connection.fetchval("""insert into jarvis_context_payloads(owner_id,availability,content)
            values($1,$2,$3::jsonb) returning id""", owner or self.owner, availability,
            json.dumps({"text": "Synthetic fixture, not a real instruction."}) if availability in ("available", "not_finalized") else None)

    async def event(self, *, owner=None, source_id=None, revision=1, supersedes=None, finality="final",
                    trust="user_input", payload_id=None, session_id=None):
        owner = owner or self.owner
        payload_id = payload_id or await self.payload(owner=owner)
        return await self.connection.fetchrow("""insert into jarvis_context_events(
            owner_id,source_namespace,source_event_id,revision,supersedes_event_id,kind,finality,trust,payload_id,session_id)
            values($1,'fixture',$2,$3,$4,'user_request',$5,$6,$7,$8) returning *""",
            owner, source_id or str(uuid4()), revision, supersedes, finality, trust, payload_id, session_id)

    async def grant(self, event, *, owner=None, payload_id=None):
        payload_id = payload_id or await self.payload(owner=owner)
        return await self.connection.fetchval("""insert into jarvis_operation_grants(
            owner_id,capability,operation_key,request_event_id,request_revision,scope_revision,scope_payload_id,expires_at)
            values($1,'fixture_only','fixture-report',$2,$3,1,$4,clock_timestamp()+interval '1 hour') returning id""",
            owner or self.owner, event["id"], event["revision"], payload_id)

    async def snapshot(self, schema):
        statements = (
            """select c.relname,a.attname,format_type(a.atttypid,a.atttypmod),a.attnotnull,
                 pg_get_expr(d.adbin,d.adrelid) from pg_class c join pg_namespace n on n.oid=c.relnamespace
                 join pg_attribute a on a.attrelid=c.oid left join pg_attrdef d on d.adrelid=c.oid and d.adnum=a.attnum
                 where n.nspname=$1 and c.relkind='r' and a.attnum>0 and not a.attisdropped order by c.relname,a.attnum""",
            """select c.relname,k.conname,pg_get_constraintdef(k.oid) from pg_constraint k
                 join pg_class c on c.oid=k.conrelid join pg_namespace n on n.oid=c.relnamespace
                 where n.nspname=$1 order by c.relname,k.conname""",
            "select indexname,indexdef from pg_indexes where schemaname=$1 order by indexname",
            """select p.proname,pg_get_functiondef(p.oid) from pg_proc p join pg_namespace n on n.oid=p.pronamespace
                 where n.nspname=$1 order by p.proname""",
            """select t.tgname,pg_get_triggerdef(t.oid) from pg_trigger t join pg_class c on c.oid=t.tgrelid
                 join pg_namespace n on n.oid=c.relnamespace where n.nspname=$1 and not t.tgisinternal order by t.tgname""",
        )
        result = []
        for sql in statements:
            rows = await self.connection.fetch(sql, schema)
            result.append(tuple(tuple(str(value).replace('"' + schema + '"', "SCHEMA").replace(schema, "SCHEMA")
                                      for value in row) for row in rows))
        return tuple(result)

    async def test_fresh_upgrade_and_canonical_sql_have_schema_parity(self):
        await self.initialize()
        fresh = await self.snapshot(self.schema)
        upgrade = await self.fixture.create_schema(self.connection)
        await self.connection.execute(f"set search_path to {quote_schema(upgrade)}, pg_catalog, public, extensions")
        await self.connection.execute(baseline_sql())
        self.assertEqual(await apply_migrations(self.connection, schema=upgrade), MIGRATION_VERSIONS)
        self.assertEqual(fresh, await self.snapshot(upgrade))
        canonical = await self.fixture.create_schema(self.connection)
        await self.connection.execute(f"set search_path to {quote_schema(canonical)}, pg_catalog, public, extensions")
        await self.connection.execute(files("app").joinpath("schema.sql").read_text(encoding="utf-8"))
        self.assertEqual(fresh, await self.snapshot(canonical))
        self.assertTrue((await read_schema_readiness(self.connection, schema=canonical)).ready)

    async def test_readiness_does_not_create_ledger_and_fresh_refuses_existing_schema(self):
        before = await read_schema_readiness(self.connection, schema=self.schema)
        self.assertFalse(before.ready)
        self.assertIsNone(await self.connection.fetchval("select to_regclass($1)", f"{self.schema}.jarvis_schema_migrations"))
        with self.assertRaises(MigrationError):
            await apply_migrations(self.connection, schema=self.schema)
        await self.initialize()
        with self.assertRaises(MigrationError):
            await apply_migrations(self.connection, schema=self.schema, fresh=True)
        self.assertTrue((await read_schema_readiness(self.connection, schema=self.schema)).ready)

    async def test_repeated_and_concurrent_migrators_apply_once(self):
        await self.connection.execute(baseline_sql())
        other = await asyncpg.connect(self.dsn, timeout=10, command_timeout=90)
        try:
            results = await asyncio.gather(apply_migrations(self.connection, schema=self.schema),
                                           apply_migrations(other, schema=self.schema))
        finally:
            await other.close()
        self.assertEqual(sorted(results, key=len), [(), MIGRATION_VERSIONS])
        self.assertEqual(await apply_migrations(self.connection, schema=self.schema), ())
        self.assertEqual(await self.connection.fetchval("select count(*) from jarvis_schema_migrations"), len(MIGRATION_VERSIONS))

    async def test_checksum_conflict_unknown_version_and_missing_table_are_visible(self):
        await self.initialize()
        original = load_migrations()[0]
        changed = (Migration(1, original.name, original.sql + "\n-- changed"),)
        with self.assertRaises(MigrationError):
            await apply_migrations(self.connection, schema=self.schema, migrations=changed)
        self.assertFalse((await read_schema_readiness(self.connection, schema=self.schema, migrations=changed)).ready)
        await self.connection.execute("insert into jarvis_schema_migrations(version,name,checksum) values(99,'unknown',repeat('0',64))")
        self.assertIn("unknown_version:99", (await read_schema_readiness(self.connection, schema=self.schema)).issues)
        await self.connection.execute("delete from jarvis_schema_migrations where version=99; drop table jarvis_operation_grants cascade")
        self.assertIn("missing_table:jarvis_operation_grants", (await read_schema_readiness(self.connection, schema=self.schema)).issues)

    async def test_failed_pending_migration_rolls_back_ddl_and_ledger(self):
        await self.connection.execute(baseline_sql())
        failing = load_migrations() + (Migration(len(MIGRATION_VERSIONS) + 1, "fixture_failure", "create table fixture_partial(id integer); select 1/0;"),)
        with self.assertRaises(asyncpg.DivisionByZeroError):
            await apply_migrations(self.connection, schema=self.schema, migrations=failing)
        for table in ("jarvis_schema_migrations", "jarvis_knowledge_areas", "fixture_partial"):
            self.assertIsNone(await self.connection.fetchval("select to_regclass($1)", f"{self.schema}.{table}"))
        self.assertEqual(await apply_migrations(self.connection, schema=self.schema), MIGRATION_VERSIONS)

    async def test_context_owner_isolation_immutable_identity_and_revision_chain(self):
        await self.initialize()
        original = await self.event(source_id="utterance-one")
        revised = await self.event(source_id="utterance-one", revision=2, supersedes=original["id"], finality="revised")
        self.assertGreater(revised["sequence"], original["sequence"])
        with self.assertRaises(asyncpg.UniqueViolationError):
            await self.event(source_id="utterance-one")
        with self.assertRaises(asyncpg.CheckViolationError):
            await self.event(source_id="wrong-source", revision=2, supersedes=original["id"])
        with self.assertRaises(asyncpg.CheckViolationError):
            await self.connection.execute("update jarvis_context_events set source_event_id='rewritten' where id=$1", original["id"])
        foreign_payload = await self.payload(owner=self.other_owner)
        with self.assertRaises(asyncpg.ForeignKeyViolationError):
            await self.event(payload_id=foreign_payload)
        other = await self.event(owner=self.other_owner, source_id="utterance-one")
        self.assertEqual(other["sequence"], 1)
        self.assertNotEqual(original["id"], other["id"])

    async def test_session_owner_reference_and_server_assigned_sequence(self):
        await self.initialize()
        session = await self.connection.fetchval("insert into jarvis_sessions(owner_id) values($1) returning id", self.other_owner)
        with self.assertRaises(asyncpg.ForeignKeyViolationError):
            await self.event(session_id=session)
        event = await self.event(owner=self.other_owner, session_id=session)
        await self.connection.execute("update jarvis_sessions set status='closed' where id=$1", session)
        self.assertEqual(await self.connection.fetchval("select session_id from jarvis_context_events where id=$1", event["id"]), session)
        with self.assertRaises(asyncpg.CheckViolationError):
            await self.connection.execute("""insert into jarvis_context_events(owner_id,sequence,source_namespace,
                source_event_id,kind,finality,trust,payload_id) values($1,999,'fixture','injected','user_request','final','user_input',$2)""",
                self.other_owner, event["payload_id"])

    async def test_context_sequence_serializes_concurrent_commits_for_one_owner(self):
        await self.initialize()
        payload_id = await self.payload()
        other = await asyncpg.connect(self.dsn, timeout=10, command_timeout=90)
        pending = None
        try:
            await other.execute(f"set search_path to {quote_schema(self.schema)}, pg_catalog, public, extensions")
            async with self.connection.transaction():
                first = await self.event()
                pending = asyncio.create_task(other.fetchrow("""insert into jarvis_context_events(
                    owner_id,source_namespace,source_event_id,kind,finality,trust,payload_id)
                    values($1,'fixture','concurrent','user_request','final','user_input',$2) returning *""", self.owner, payload_id))
                with self.assertRaises(TimeoutError):
                    await asyncio.wait_for(asyncio.shield(pending), timeout=0.1)
            second = await pending
            self.assertGreater(second["sequence"], first["sequence"])
            self.assertGreaterEqual(second["recorded_at"], first["recorded_at"])
        finally:
            if pending is not None and not pending.done():
                pending.cancel()
                await asyncio.gather(pending, return_exceptions=True)
            await other.close()

    async def test_payload_removal_preserves_event_but_erases_content_and_digest(self):
        await self.initialize()
        event = await self.event()
        self.assertIsNotNone(await self.connection.fetchval("select content_digest from jarvis_context_payloads where id=$1", event["payload_id"]))
        with self.assertRaises(asyncpg.CheckViolationError):
            await self.connection.execute("update jarvis_context_payloads set content='{}' where id=$1", event["payload_id"])
        await self.connection.execute("update jarvis_context_payloads set content=null,availability='deleted' where id=$1", event["payload_id"])
        erased = await self.connection.fetchrow("select * from jarvis_context_payloads where id=$1", event["payload_id"])
        self.assertIsNone(erased["content"])
        self.assertIsNone(erased["content_digest"])
        self.assertIsNotNone(erased["removed_at"])
        self.assertEqual(await self.connection.fetchval("select sequence from jarvis_context_events where id=$1", event["id"]), event["sequence"])
        with self.assertRaises(asyncpg.CheckViolationError):
            await self.connection.execute("update jarvis_context_payloads set content='{}',availability='available',removed_at=null where id=$1", event["payload_id"])
        with self.assertRaises(asyncpg.CheckViolationError):
            await self.grant(event)

    async def test_grants_require_final_explicit_current_same_owner_request_and_scope(self):
        await self.initialize()
        event = await self.event(source_id="request-one")
        grant = await self.grant(event)
        with self.assertRaises(asyncpg.CheckViolationError):
            await self.connection.execute("""insert into jarvis_operation_grants(
                owner_id,capability,operation_key,request_event_id,request_revision,scope_revision,scope_payload_id,expires_at)
                values($1,'fixture_only','infinite-grant',$2,1,1,$3,'infinity')""", self.owner, event["id"], event["payload_id"])
        for bad_event in (await self.event(finality="provisional"), await self.event(trust="assistant_output")):
            with self.assertRaises(asyncpg.CheckViolationError):
                await self.grant(bad_event)
        with self.assertRaises(asyncpg.CheckViolationError):
            await self.grant(event, owner=self.other_owner)
        foreign_payload = await self.payload(owner=self.other_owner)
        with self.assertRaises(asyncpg.CheckViolationError):
            await self.grant(event, payload_id=foreign_payload)
        with self.assertRaises(asyncpg.CheckViolationError):
            await self.connection.execute("update jarvis_operation_grants set operation_key='broadened' where id=$1", grant)
        await self.connection.execute("update jarvis_operation_grants set status='revoked' where id=$1", grant)
        with self.assertRaises(asyncpg.CheckViolationError):
            await self.connection.execute("update jarvis_operation_grants set status='active',revoked_at=null where id=$1", grant)
        await self.event(source_id="request-one", revision=2, supersedes=event["id"], finality="revised")
        with self.assertRaises(asyncpg.CheckViolationError):
            await self.grant(event)

    async def test_capability_preferences_are_scoped_versioned_and_append_only(self):
        await self.initialize()
        event = await self.event()
        policy = await self.payload()
        sql = """insert into jarvis_capability_preferences(owner_id,capability,revision,request_event_id,request_revision,
                 policy_payload_id,knowledge_area_id) values($1,'fixture_only',$2,$3,1,$4,$5) returning id"""
        first = await self.connection.fetchval(sql, self.owner, 1, event["id"], policy, None)
        with self.assertRaises(asyncpg.UniqueViolationError):
            await self.connection.fetchval(sql, self.owner, 1, event["id"], policy, None)
        await self.connection.fetchval(sql, self.owner, 2, event["id"], policy, None)
        foreign_area = await self.connection.fetchval("insert into jarvis_knowledge_areas(owner_id,name,domain) values($1,'fixture','test') returning id", self.other_owner)
        with self.assertRaises(asyncpg.ForeignKeyViolationError):
            await self.connection.fetchval(sql, self.owner, 1, event["id"], policy, foreign_area)
        with self.assertRaises(asyncpg.CheckViolationError):
            await self.connection.execute("update jarvis_capability_preferences set revision=9 where id=$1", first)

    async def test_knowledge_area_identity_and_revision_cannot_be_reassigned(self):
        await self.initialize()
        area = await self.connection.fetchval("insert into jarvis_knowledge_areas(owner_id,name,domain) values($1,'fixture','test') returning id", self.owner)
        with self.assertRaises(asyncpg.CheckViolationError):
            await self.connection.execute("update jarvis_knowledge_areas set owner_id=$2,revision=2 where id=$1", area, self.other_owner)
        with self.assertRaises(asyncpg.CheckViolationError):
            await self.connection.execute("update jarvis_knowledge_areas set name='new name' where id=$1", area)
        await self.connection.execute("update jarvis_knowledge_areas set name='new name',revision=2 where id=$1", area)
        self.assertEqual(await self.connection.fetchval("select owner_id from jarvis_knowledge_areas where id=$1", area), self.owner)

    async def test_cleanup_refuses_nonfixture_owner_and_preserves_other_manifest_schema(self):
        await self.initialize()
        # The cleanup guard applies to any owner-scoped table. Keep this
        # deliberate foreign-row fault separate from immutable source journals.
        await self.connection.execute("create table fixture_foreign_owner(owner_id text not null); insert into fixture_foreign_owner values('nonfixture-owner')")
        with self.assertRaisesRegex(RuntimeError, "Non-fixture owner"):
            await self.fixture.cleanup()
        await self.connection.execute("delete from fixture_foreign_owner where owner_id='nonfixture-owner'")
        sibling = DatabaseFixture(self.dsn, self.fixture.path.parent, test_name=self.id() + ":sentinel")
        self.addAsyncCleanup(sibling.cleanup)
        sibling_schema = await sibling.create_schema(self.connection)
        await self.connection.execute(f"create view {quote_schema(sibling_schema)}.external_view as select id from {quote_schema(self.schema)}.jarvis_knowledge_areas")
        with self.assertRaisesRegex(RuntimeError, "External schema depends"):
            await self.fixture.cleanup()
        await self.connection.execute(f"drop view {quote_schema(sibling_schema)}.external_view")
        result = await self.fixture.cleanup()
        self.assertEqual(result["remaining_schemas"], 0)
        self.assertTrue(await self.connection.fetchval("select exists(select 1 from pg_namespace where nspname=$1)", sibling_schema))
