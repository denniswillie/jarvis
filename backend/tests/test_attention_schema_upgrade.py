"""Upgrade existing task storage without automatically replaying old answers."""
import os
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from uuid import uuid4

import asyncpg

from app.migrations.runner import apply_migrations, load_migrations, quote_schema, read_schema_readiness
from app.runtime.feature_config import TaskLimits
from app.tasks.hermes_result import HermesTaskResult
from app.tasks.models import TaskCreate
from app.tasks.repository import TaskRepository
from app.tasks.runner import TaskRunner
from evals.second_brain.fixtures import DatabaseFixture


@unittest.skipUnless(os.getenv("JARVIS_SECOND_BRAIN_TEST_DB_URL") and
                     os.getenv("JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"),
                     "Explicit manifested database required")
class AttentionUpgradeTests(unittest.IsolatedAsyncioTestCase):
    async def test_upgrade_preserves_old_results_and_atomically_materializes_new_results(self):
        dsn = os.environ["JARVIS_SECOND_BRAIN_TEST_DB_URL"]
        fixture = DatabaseFixture(dsn, Path(os.environ["JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"]), test_name=self.id())
        self.addAsyncCleanup(fixture.cleanup)
        connection = await asyncpg.connect(dsn, timeout=10, command_timeout=90)
        self.addAsyncCleanup(connection.close)
        schema = await fixture.create_schema(connection)
        migrations = load_migrations()
        await apply_migrations(connection, schema=schema, fresh=True,
                               migrations=tuple(m for m in migrations if m.version<8))
        async def setup(c):
            await c.execute(f"set search_path to {quote_schema(schema)},pg_catalog,extensions")
        pool = await asyncpg.create_pool(dsn, min_size=1, max_size=2, timeout=10, command_timeout=30, setup=setup)
        self.addAsyncCleanup(pool.close)
        tasks = TaskRepository(pool, TaskLimits(lease_seconds=300))

        async def publish():
            task = await tasks.create(fixture.owners[0], TaskCreate(request_id=uuid4(),
                title="Synthetic upgrade", requested_outcome="Reason only."), model="fixture-model")
            claim = await tasks.claim_next("upgrade-fixture", ("frontier",))
            ctx = TaskRunner(tasks, {}).context(claim)
            outcome = await HermesTaskResult(status="succeeded", answer="Synthetic expert answer.",
                speech=[{"text":"Synthetic expert answer."}], evidence=[],
                completed_operation_refs=[], uncertainties=[]).verified(ctx, SimpleNamespace())
            await tasks.finish(claim, outcome)
            return await tasks.get(fixture.owners[0], task.id)

        # Seed the historical schema using its own contract. Today's repository
        # requires later source/dependency tables and must not be run on v7.
        # The fixture pool excludes public so a missing table cannot fall through.
        async with pool.acquire() as c, c.transaction():
            owner = fixture.owners[0]
            original = await c.fetchval("""insert into jarvis_context_payloads(owner_id,availability,content)
                values($1,'available','{"text":"Synthetic historical request"}') returning id""", owner)
            request = await c.fetchval("""insert into jarvis_context_events
                (owner_id,source_namespace,source_event_id,kind,finality,trust,payload_id)
                values($1,'fixture-upgrade','old-request','user_request','final','user_input',$2) returning id""",
                owner, original)
            description = await c.fetchval("""insert into jarvis_context_payloads(owner_id,availability,content)
                values($1,'available','{"title":"Synthetic historical task","requested_outcome":"Reason only."}')
                returning id""", owner)
            old_result = json.dumps({"answer":"Synthetic historical expert answer.", "evidence":[],
                                    "completed_operation_refs":[], "uncertainties":[]})
            payload = await c.fetchval("""insert into jarvis_context_payloads(owner_id,availability,content)
                values($1,'available',$2::jsonb) returning id""", owner, old_result)
            identity = await c.fetchval("""insert into jarvis_background_tasks
                (owner_id,backend_kind,model,status,description_payload_id,request_event_id,
                 request_revision,budget,result_payload_id)
                values($1,'frontier','fixture-model','succeeded',$2,$3,1,'{}'::jsonb,$4) returning id""",
                owner, description, request, payload)
            old = SimpleNamespace(id=identity, result_payload_id=payload)
        self.assertIsNotNone(old.result_payload_id)
        self.assertEqual(await apply_migrations(connection, schema=schema), tuple(m.version for m in migrations if m.version >= 8))
        self.assertTrue((await read_schema_readiness(connection, schema=schema)).ready)
        async with pool.acquire() as c:
            self.assertEqual(await c.fetchval("select count(*) from jarvis_attention_intents"), 0)
        async with pool.acquire() as c:
            retained = await c.fetchrow("""select t.result_payload_id,p.content::text as content
                from jarvis_background_tasks t join jarvis_context_payloads p on p.owner_id=t.owner_id
                and p.id=t.result_payload_id where t.owner_id=$1 and t.id=$2""", fixture.owners[0], old.id)
        self.assertEqual(retained["result_payload_id"], old.result_payload_id)
        self.assertEqual(json.loads(retained["content"]), json.loads(old_result))
        new = await publish()
        async with pool.acquire() as c:
            self.assertEqual(await c.fetchval("select task_id from jarvis_attention_intents"), new.id)
            # A failed inbox insertion must roll back its result publication,
            # rather than leave a completed answer with no logical intent.
            await c.execute("alter table jarvis_attention_intents add constraint fixture_reject_next_intent check(task_id='"+str(new.id)+"'::uuid)")
        with self.assertRaises(asyncpg.CheckViolationError):
            await publish()
        async with pool.acquire() as c:
            self.assertEqual(await c.fetchval("select count(*) from jarvis_attention_intents"), 1)
            pending = await c.fetchrow("select status,result_payload_id from jarvis_background_tasks where id<>$1 and id<>$2", old.id, new.id)
        self.assertEqual(pending["status"], "running")
        self.assertIsNone(pending["result_payload_id"])
