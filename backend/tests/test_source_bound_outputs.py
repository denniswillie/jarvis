"""Source-bound copies: manifested real storage, no native model or real owner."""
import asyncio
from datetime import UTC, datetime
import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from uuid import uuid4

import asyncpg

from app.attention.repository import AttentionRepository
from app.knowledge.publication import CanonicalSourcePublication
from app.migrations.runner import apply_migrations, load_migrations, quote_schema, read_schema_readiness
from app.runtime.feature_config import TaskLimits
from app.sessions.repository import SessionRepository
from app.tasks.context_capture import SessionContextCapture
from app.tasks.context_events import ContextIngress
from app.tasks.context_repository import ContextRepository
from app.tasks.hermes_result import HermesTaskResult
from app.tasks.models import TaskCreate, TaskError
from app.tasks.repository import TaskRepository
from app.tasks.runner import TaskRunner
from evals.second_brain.fixtures import DatabaseFixture
from evals.second_brain.source_fixture import source_fixture


ENABLED = bool(os.getenv("JARVIS_SECOND_BRAIN_TEST_DB_URL") and
               os.getenv("JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"))


@unittest.skipUnless(ENABLED, "Explicit manifested database required")
class SourceBoundOutputTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.f = await self.enterAsyncContext(source_fixture(
            os.environ["JARVIS_SECOND_BRAIN_TEST_DB_URL"],
            Path(os.environ["JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"]), test_name=self.id()))
        self.tasks = TaskRepository(self.f.pool, TaskLimits(lease_seconds=300))
        self.messages = SessionRepository(self.f.pool)
        self.session = await self.messages.create_session(self.f.owner, "Synthetic source lineage")
        self.context = ContextRepository(self.tasks,
            publication=CanonicalSourcePublication(self.f.sources, self.f.area))

    def ingress(self, parent, key, **changes):
        return ContextIngress(**(dict(session_id=self.session.id, source_namespace="fixture-bound",
            source_event_id=key, receipt_id=key, kind="assistant_output", trust="assistant_output",
            source_time=datetime.now(UTC), payload={"text":"Synthetic copied expert content."},
            source_payload_ids=(parent,)) | changes))

    async def copy(self, parent, key):
        return await self.context.append(self.f.owner, self.ingress(parent, key))

    async def message(self, parent, **metadata):
        return await self.messages.create_message(owner_id=self.f.owner, session_id=self.session.id,
            role="assistant", modality="audio", content="Synthetic copied expert content.",
            source_payload_id=parent, metadata=metadata)

    async def expert(self, *, manifest=False):
        task = await self.tasks.create(self.f.owner, TaskCreate(request_id=uuid4(), session_id=self.session.id,
            title="Synthetic expert", requested_outcome="Reason about synthetic data only."), model="fixture-expert")
        claim = await self.tasks.claim_next("fixture-expert", ("frontier",))
        original = await self.context.get(self.f.owner, task.request_event_id)
        if manifest:
            await self.context.store_manifest(claim, watermark=original.sequence, records=(original,),
                content={"fixture":"Original synthetic request snapshot."})
        context = TaskRunner(self.tasks, {}).context(claim)
        outcome = await HermesTaskResult(status="succeeded", answer="Synthetic expert answer.",
            speech=[{"text":"Synthetic expert answer."}], evidence=[], completed_operation_refs=[],
            uncertainties=[]).verified(context, SimpleNamespace())
        await self.tasks.finish(claim, outcome)
        task = await self.tasks.get(self.f.owner, task.id)
        inbox = AttentionRepository(self.tasks)
        delivery = await inbox.claim_next(self.f.owner, self.session.id, "fixture-delivery")
        self.assertEqual(delivery.source_payload_id, task.result_payload_id)
        return task, delivery, original

    async def test_original_deletion_purges_transitive_copies_and_queues_hindsight_deletion(self):
        original = await self.f.publish("Synthetic original that can be forgotten.")
        first = await self.copy(original.payload_id, "first")
        second = await self.copy(first.payload_id, "second")
        message = await self.message(second.payload_id)
        await self.f.sources.delete_source(self.f.owner, self.f.area, original.source_id,
                                          expected_revision=original.revision)
        for record in (first, second):
            removed = await self.context.get(self.f.owner, record.id)
            self.assertEqual(removed.availability, "deleted")
            self.assertIsNone(removed.content)
        rows = await self.messages.get_recent_messages(self.f.owner, self.session.id)
        self.assertEqual(rows[0].id, message.id)
        self.assertEqual(rows[0].content, "")
        self.assertEqual(rows[0].metadata["source_status"], "deleted")
        async with self.f.pool.acquire() as connection:
            self.assertEqual(await connection.fetchval(
                "select count(*) from jarvis_memory_outbox where owner_id=$1 and operation='delete'",
                self.f.owner), 3)

    async def test_late_copy_and_message_cannot_resurrect_a_removed_original(self):
        original = await self.f.publish()
        await self.f.sources.delete_source(self.f.owner, self.f.area, original.source_id,
                                          expected_revision=original.revision)
        with self.assertRaises(TaskError) as raised:
            await self.copy(original.payload_id, "late")
        self.assertEqual(raised.exception.code, "context_source_removed")
        with self.assertRaises(asyncpg.CheckViolationError):
            await self.message(original.payload_id)
        self.assertEqual(await self.messages.get_recent_messages(self.f.owner, self.session.id), [])

    async def test_foreign_parent_is_not_borrowed_and_cycles_are_rejected(self):
        async with self.f.pool.acquire() as connection:
            foreign = await connection.fetchval("""insert into jarvis_context_payloads(owner_id,availability,content)
                values($1,'available','{"text":"Other fixture owner."}') returning id""", self.f.other)
        with self.assertRaises(TaskError) as raised:
            await self.copy(foreign, "foreign")
        self.assertEqual(raised.exception.code, "context_dependency_unavailable")
        original = await self.f.publish()
        child = await self.copy(original.payload_id, "child")
        async with self.f.pool.acquire() as connection:
            with self.assertRaises(asyncpg.CheckViolationError):
                await connection.execute("""insert into jarvis_context_payload_dependencies
                    (owner_id,source_payload_id,dependent_payload_id) values($1,$2,$3)""",
                    self.f.owner, child.payload_id, original.payload_id)

    async def test_expert_copies_cannot_omit_or_mislabel_their_source(self):
        task, delivery, _ = await self.expert()
        event = ContextIngress(session_id=self.session.id, source_namespace="voice:fixture-unbound",
            source_event_id="output", receipt_id="output", kind="assistant_output", trust="assistant_output",
            payload={"text":"Synthetic expert answer.","authored_by":"frontier_expert",
                     "attempt_id":str(delivery.attempt_id)})
        with self.assertRaises(asyncpg.CheckViolationError):
            await self.context.append(self.f.owner, event)
        unrelated = await self.f.publish(source_key="unrelated-original")
        with self.assertRaises(asyncpg.CheckViolationError):
            await self.context.append(self.f.owner, event.model_copy(update={"source_payload_ids":(unrelated.payload_id,)}))
        with self.assertRaises(asyncpg.CheckViolationError):
            await self.message(unrelated.payload_id, response_kind="expert_tts", intent_id=str(delivery.intent_id))
        self.assertEqual(await self.messages.get_recent_messages(self.f.owner, self.session.id), [])

    async def test_ingress_replay_binds_the_same_original_not_just_equal_text(self):
        original = await self.f.publish()
        other = await self.f.publish(source_key="different")
        event = self.ingress(original.payload_id, "replay")
        first = await self.context.append(self.f.owner, event)
        replay = await self.context.append(self.f.owner, event)
        self.assertEqual(first.id, replay.id)
        with self.assertRaises(TaskError) as raised:
            await self.context.append(self.f.owner, event.model_copy(update={"source_payload_ids":(other.payload_id,)}))
        self.assertEqual(raised.exception.code, "context_receipt_conflict")
        async with self.f.pool.acquire() as connection:
            self.assertEqual(await connection.fetchval("""select count(*) from jarvis_context_payload_dependencies
                where owner_id=$1 and dependent_payload_id=$2""", self.f.owner, first.payload_id), 1)

    async def test_bound_message_cannot_detach_or_replace_its_text(self):
        original = await self.f.publish()
        message = await self.message(original.payload_id)
        async with self.f.pool.acquire() as connection:
            with self.assertRaises(asyncpg.CheckViolationError):
                await connection.execute("update jarvis_messages set source_payload_id=null where owner_id=$1 and id=$2",
                                         self.f.owner, message.id)
            with self.assertRaises(asyncpg.CheckViolationError):
                await connection.execute("update jarvis_messages set content='Different synthetic text' where owner_id=$1 and id=$2",
                                         self.f.owner, message.id)

    async def test_correction_retires_copies_but_retains_the_original_historical_result(self):
        task, delivery, original = await self.expert(manifest=True)
        copied = await self.copy(task.result_payload_id, "expert-copy")
        message = await self.message(task.result_payload_id, response_kind="expert_tts", intent_id=str(delivery.intent_id))
        await self.context.append(self.f.owner, ContextIngress(session_id=self.session.id,
            source_namespace=original.source_namespace, source_event_id=original.source_event_id,
            receipt_id="fixture-correction", kind=original.kind, trust=original.trust, finality="revised",
            expected_revision=original.revision, payload={"text":"Corrected synthetic request."}))
        self.assertIsNone((await self.tasks.get(self.f.owner, task.id)).result_payload_id)
        self.assertIsNone((await self.context.get(self.f.owner, copied.id)).content)
        self.assertEqual((await self.messages.get_recent_messages(self.f.owner, self.session.id))[0].content, "")
        async with self.f.pool.acquire() as connection:
            self.assertEqual(await connection.fetchval(
                "select availability from jarvis_context_payloads where owner_id=$1 and id=$2",
                self.f.owner, task.result_payload_id), "available")
        with self.assertRaises(TaskError) as raised:
            await self.copy(task.result_payload_id, "stale-after-correction")
        self.assertEqual(raised.exception.code, "context_source_removed")

    async def test_input_removal_purges_completed_result_and_its_copies(self):
        task, _, original = await self.expert(manifest=True)
        copied = await self.copy(task.result_payload_id, "expert-copy")
        await self.message(task.result_payload_id)
        async with self.f.pool.acquire() as connection:
            await connection.execute("""update jarvis_context_payloads set availability='deleted',content=null
                where owner_id=$1 and id=$2""", self.f.owner, original.payload_id)
            for identity in (task.result_payload_id, copied.payload_id):
                self.assertIsNone(await connection.fetchval(
                    "select content from jarvis_context_payloads where owner_id=$1 and id=$2",
                    self.f.owner, identity))
        self.assertEqual((await self.messages.get_recent_messages(self.f.owner, self.session.id))[0].content, "")

    async def test_queued_expert_capture_is_discarded_after_intentional_source_removal(self):
        task, delivery, _ = await self.expert()
        capture = SessionContextCapture(self.context, owner_id=self.f.owner, session_id=self.session.id,
            provider="openai", audio_session_id="fixture-audio")
        self.addAsyncCleanup(capture.close)
        provenance = dict(attempt_id=str(delivery.attempt_id), intent_id=str(delivery.intent_id), chunk_index=0)
        capture.expert_speech(response_id="r", item_id="i", text="Synthetic expert answer.", samples=100,
            complete=True, interrupted=False, provenance=provenance, source_payload_id=task.result_payload_id)
        self.assertTrue(await capture.flush())
        async with self.f.pool.acquire() as connection:
            await connection.execute("""update jarvis_context_payloads set availability='deleted',content=null
                where owner_id=$1 and id=$2""", self.f.owner, task.result_payload_id)
        capture.expert_speech(response_id="r", item_id="i", text="Synthetic expert answer.", samples=100,
            complete=True, interrupted=True, provenance=provenance, source_payload_id=task.result_payload_id)
        self.assertTrue(await capture.flush())
        self.assertIsNone(capture.issue)
        async with self.f.pool.acquire() as connection:
            self.assertFalse(await connection.fetchval("""select exists(select 1 from jarvis_context_payloads p
                join jarvis_context_events e on e.owner_id=p.owner_id and e.payload_id=p.id
                where e.owner_id=$1 and e.source_namespace=$2 and e.kind='assistant_output' and p.content is not null)""",
                self.f.owner, capture.namespace))

    async def test_playback_progress_does_not_invalidate_expert_text_but_source_removal_does(self):
        expert, delivery, _ = await self.expert()
        capture = SessionContextCapture(self.context, owner_id=self.f.owner, session_id=self.session.id,
            provider="openai", audio_session_id="fixture-audio")
        self.addAsyncCleanup(capture.close)
        capture.expert_speech(response_id="r", item_id="i", text="Synthetic expert answer.", samples=100,
            complete=True, interrupted=False, provenance={"attempt_id":str(delivery.attempt_id)},
            source_payload_id=expert.result_payload_id)
        self.assertTrue(await capture.flush())
        records = await self.context.archive(self.f.owner, self.session.id)
        output = next(record for record in records if record.kind == "assistant_output")
        self.assertEqual(output.dependency_usage, "historical")
        task = await self.tasks.create(self.f.owner, TaskCreate(request_id=uuid4(), session_id=self.session.id,
            title="Synthetic follow-up", requested_outcome="Use the fixed expert script."), model="fixture")
        claim = await self.tasks.claim_next("fixture", ("frontier",))
        manifest = await self.context.store_manifest(claim, watermark=output.sequence, records=(output,),
            content={"fixture":"Synthetic fixed-script dependency."})
        self.assertTrue(capture.playback_receipt("i", response_id="r", samples=100, state="completed"))
        self.assertTrue(await capture.flush())
        self.assertEqual((await self.context.read_manifest(self.f.owner, task.id, manifest.id)).status, "active")
        async with self.f.pool.acquire() as connection:
            await connection.execute("""update jarvis_context_payloads set availability='deleted',content=null
                where owner_id=$1 and id=$2""", self.f.owner, expert.result_payload_id)
            self.assertEqual(await connection.fetchval("select status from jarvis_task_context_manifests where id=$1", manifest.id), "revoked")

    async def test_delete_wins_while_a_new_copy_waits_on_the_source_lock(self):
        original = await self.f.publish()
        plain = ContextRepository(self.tasks)
        pending = None
        async with self.f.pool.acquire() as remover, self.f.pool.acquire() as reader:
            pid = await reader.fetchval("select pg_backend_pid()")
            async def append():
                async with reader.transaction():
                    await self.tasks.lock_owner(reader, self.f.owner)
                    return await plain.append_in_transaction(reader, self.f.owner, self.ingress(original.payload_id, "racing"))
            try:
                async with remover.transaction():
                    await remover.execute("select id from jarvis_context_payloads where id=$1 for update", original.payload_id)
                    pending = asyncio.create_task(append())
                    async with asyncio.timeout(5):
                        while not await remover.fetchval("select exists(select 1 from pg_locks where pid=$1 and not granted)", pid):
                            await asyncio.sleep(.02)
                    await remover.execute("""update jarvis_context_payloads set availability='deleted',content=null
                        where owner_id=$1 and id=$2""", self.f.owner, original.payload_id)
                with self.assertRaises(TaskError) as raised:
                    await asyncio.wait_for(pending, 5)
                self.assertEqual(raised.exception.code, "context_source_removed")
            finally:
                if pending is not None and not pending.done():
                    pending.cancel()
                    await asyncio.gather(pending, return_exceptions=True)

    async def test_committed_copy_is_included_when_deletion_waited_for_its_source_lock(self):
        original = await self.f.publish()
        plain = ContextRepository(self.tasks)
        pending = None
        async with self.f.pool.acquire() as writer, self.f.pool.acquire() as remover:
            pid = await remover.fetchval("select pg_backend_pid()")
            async def remove():
                await remover.execute("""update jarvis_context_payloads set availability='deleted',content=null
                    where owner_id=$1 and id=$2""", self.f.owner, original.payload_id)
            try:
                async with writer.transaction():
                    await self.tasks.lock_owner(writer, self.f.owner)
                    copied = await plain.append_in_transaction(writer, self.f.owner, self.ingress(original.payload_id, "racing"))
                    pending = asyncio.create_task(remove())
                    async with asyncio.timeout(5):
                        while not await writer.fetchval("select exists(select 1 from pg_locks where pid=$1 and not granted)", pid):
                            await asyncio.sleep(.02)
                await asyncio.wait_for(pending, 5)
                self.assertIsNone((await plain.get(self.f.owner, copied.id)).content)
            finally:
                if pending is not None and not pending.done():
                    pending.cancel()
                    await asyncio.gather(pending, return_exceptions=True)


@unittest.skipUnless(ENABLED, "Explicit manifested database required")
class SourceBoundUpgradeTests(unittest.IsolatedAsyncioTestCase):
    async def legacy_schema(self):
        dsn = os.environ["JARVIS_SECOND_BRAIN_TEST_DB_URL"]
        fixture = DatabaseFixture(dsn, Path(os.environ["JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"]), test_name=self.id())
        self.addAsyncCleanup(fixture.cleanup)
        connection = await asyncpg.connect(dsn, timeout=10, command_timeout=90)
        self.addAsyncCleanup(connection.close)
        schema = await fixture.create_schema(connection)
        migrations = load_migrations()
        await apply_migrations(connection, schema=schema, fresh=True,
                               migrations=tuple(m for m in migrations if m.version<10))
        async def setup(c):
            await c.execute(f"set search_path to {quote_schema(schema)},pg_catalog,public,extensions")
        pool = await asyncpg.create_pool(dsn, min_size=1, max_size=2, timeout=10, command_timeout=30, setup=setup)
        self.addAsyncCleanup(pool.close)
        owner = fixture.owners[0]
        return connection, pool, owner, schema

    async def test_upgrade_binds_existing_expert_messages_and_context_copies(self):
        connection, pool, owner, schema = await self.legacy_schema()
        messages = SessionRepository(pool)
        session = await messages.create_session(owner, "Synthetic upgrade")
        ordinary = await messages.create_message(owner_id=owner, session_id=session.id,
            role="user", modality="audio", content="Synthetic ordinary audio before migration.")
        self.assertEqual(ordinary.content, "Synthetic ordinary audio before migration.")
        # Seed the actual pre-10 shape, not today's repository API (which
        # correctly requires the source-dependency table introduced by 10).
        tasks = TaskRepository(pool, TaskLimits(lease_seconds=300))
        async with pool.acquire() as c:
            original_payload = await c.fetchval("""insert into jarvis_context_payloads
                (owner_id,availability,content) values($1,'available',$2::jsonb) returning id""",
                owner, json.dumps({"text": "Synthetic answer."}))
            intent_id = await c.fetchval("""insert into jarvis_attention_intents
                (owner_id,source_namespace,source_id,source_revision,source_payload_id,
                 preferred_session_id,state,attempt_count)
                values($1,'fixture_legacy_expert',$2,1,$2,$3,'played',1) returning id""",
                owner, original_payload, session.id)
            attempt_id = await c.fetchval("""insert into jarvis_attention_attempts
                (owner_id,intent_id,generation,channel,session_id,worker_id,phase,
                 forwarded_samples,played_samples,lease_until,ended_at)
                values($1,$2,1,'voice',$3,'fixture-delivery','played',100,100,
                       clock_timestamp()+interval '1 minute',clock_timestamp()) returning id""",
                owner, intent_id, session.id)
            message_id = await c.fetchval("""insert into jarvis_messages(owner_id,session_id,role,modality,content,metadata)
                values($1,$2,'assistant','audio','Synthetic answer.',$3::jsonb) returning id""",
                owner, session.id, json.dumps({"response_kind": "expert_tts", "intent_id": str(intent_id)}))
        context = ContextRepository(tasks)
        copied = await context.append(owner, ContextIngress(session_id=session.id, source_namespace="voice:fixture-upgrade",
            source_event_id="output", receipt_id="output", kind="assistant_output", trust="assistant_output",
            payload={"text":"Synthetic answer.","authored_by":"frontier_expert","attempt_id":str(attempt_id)}))
        self.assertEqual(await apply_migrations(connection, schema=schema),
                         tuple(m.version for m in load_migrations() if m.version >= 10))
        self.assertTrue((await read_schema_readiness(connection, schema=schema)).ready)
        async with pool.acquire() as c:
            self.assertEqual(await c.fetchval("select source_payload_id from jarvis_messages where id=$1", message_id), original_payload)
            self.assertEqual(await c.fetchval("""select source_payload_id from jarvis_context_payload_dependencies
                where owner_id=$1 and dependent_payload_id=$2""", owner, copied.payload_id), original_payload)
            await c.execute("update jarvis_context_payloads set availability='deleted',content=null where id=$1", original_payload)
            self.assertEqual(await c.fetchval("select content from jarvis_messages where id=$1", message_id), "")
            self.assertIsNone(await c.fetchval("select content from jarvis_context_payloads where id=$1", copied.payload_id))

    async def test_unmapped_history_and_context_abort_upgrade_without_erasing_evidence(self):
        connection, pool, owner, schema = await self.legacy_schema()
        session = await SessionRepository(pool).create_session(owner, "Synthetic unmapped upgrade")
        async with pool.acquire() as c:
            identity = await c.fetchval("""insert into jarvis_messages(owner_id,session_id,role,modality,content,metadata)
                values($1,$2,'assistant','audio','Synthetic unmapped answer.',
                '{"response_kind":"expert_tts","intent_id":"unknown-fixture"}') returning id""", owner, session.id)
        with self.assertRaises(asyncpg.CheckViolationError):
            await apply_migrations(connection, schema=schema)
        self.assertFalse((await read_schema_readiness(connection, schema=schema)).ready)
        async with pool.acquire() as c:
            self.assertEqual(await c.fetchval("select content from jarvis_messages where id=$1", identity), "Synthetic unmapped answer.")
            self.assertFalse(await c.fetchval("""select exists(select 1 from pg_attribute
                where attrelid='jarvis_messages'::regclass and attname='source_payload_id' and not attisdropped)"""))
            await c.execute("delete from jarvis_messages where owner_id=$1 and id=$2", owner, identity)
        tasks = TaskRepository(pool, TaskLimits())
        context = ContextRepository(tasks)
        copied = await context.append(owner, ContextIngress(session_id=session.id, source_namespace="voice:fixture-unmapped",
            source_event_id="output", receipt_id="output", kind="assistant_output", trust="assistant_output",
            payload={"text":"Synthetic orphaned context.","authored_by":"frontier_expert","attempt_id":str(uuid4())}))
        with self.assertRaises(asyncpg.CheckViolationError):
            await apply_migrations(connection, schema=schema)
        self.assertEqual((await context.get(owner, copied.id)).content["text"], "Synthetic orphaned context.")


class SourceDependencyContractTests(unittest.TestCase):
    def test_dependency_identity_is_normalized_without_changing_legacy_defaults(self):
        first, second = uuid4(), uuid4()
        fields = dict(session_id=None, source_namespace="fixture", source_event_id="fixture", receipt_id="fixture",
            kind="assistant_output", trust="assistant_output", payload={"text":"Synthetic fixture."})
        self.assertEqual(ContextIngress(**fields).source_payload_ids, ())
        event = ContextIngress(**fields, source_payload_ids=(second, first, second))
        self.assertEqual(event.source_payload_ids, tuple(sorted((first, second))))
