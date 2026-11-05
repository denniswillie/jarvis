"""Real-database graph depth, shared descendants and concurrent-copy removal."""
import asyncio
import os
from pathlib import Path
import unittest

import asyncpg

from app.runtime.feature_config import TaskLimits
from app.sessions.repository import SessionRepository
from app.tasks.repository import TaskRepository
from evals.second_brain.source_fixture import source_fixture


ENABLED = bool(os.getenv("JARVIS_SECOND_BRAIN_TEST_DB_URL") and
               os.getenv("JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"))


@unittest.skipUnless(ENABLED, "Explicit manifested database required")
class SourceCopyGraphTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.f = await self.enterAsyncContext(source_fixture(
            os.environ["JARVIS_SECOND_BRAIN_TEST_DB_URL"],
            Path(os.environ["JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"]), test_name=self.id()))
        self.tasks = TaskRepository(self.f.pool, TaskLimits())
        self.messages = SessionRepository(self.f.pool)
        self.session = await self.messages.create_session(self.f.owner, "Synthetic copy graph")
        self.original = await self.f.publish("Synthetic graph original.", source_key="graph-target")
        self.control = await self.f.publish("Synthetic unrelated original.", source_key="graph-control")

    async def graph(self, count, *, root=None, fanout=False):
        original = root or self.original
        copies = []
        async with self.tasks.transaction(self.f.owner) as db:
            parent = original.payload_id
            for number in range(count):
                payload = await self.tasks.payload(db, self.f.owner,
                    {"text": "Synthetic dependent", "sequence": number},
                    source_payload_ids=(original.payload_id if fanout else parent,))
                copies.append(payload)
                parent = payload
        return copies

    async def delete(self, original=None):
        original = original or self.original
        async with self.f.sources.transaction(self.f.owner) as db:
            await db.execute("set local statement_timeout='10s'")
            return await self.f.sources.delete_source_in_transaction(
                db, self.f.owner, self.f.area, original.source_id, expected_revision=original.revision)

    async def assert_removed(self, identities, state="deleted"):
        async with self.f.pool.acquire() as db:
            row = await db.fetchrow("""select count(*)::int as total,
                count(*) filter(where availability=$3 and content is null)::int as removed
                from jarvis_context_payloads where owner_id=$1 and id=any($2::uuid[])""",
                self.f.owner, identities, state)
        self.assertEqual(row["total"], len(identities))
        self.assertEqual(row["removed"], len(identities))
        self.assertEqual((await self.f.sources.read(
            self.f.owner, self.f.area, self.control.source_id)).source_id, self.control.source_id)

    async def test_deep_chain_deletes_without_raising_the_database_stack_limit(self):
        copies = await self.graph(384)
        async with self.f.pool.acquire() as db:
            before = await db.fetchval("show max_stack_depth")
        await self.delete()
        await self.assert_removed(copies)
        async with self.f.pool.acquire() as db:
            self.assertEqual(await db.fetchval("show max_stack_depth"), before)
            self.assertEqual(await db.fetchval(
                "select status from jarvis_knowledge_sources where owner_id=$1 and id=$2",
                self.f.owner, self.original.source_id), "deleted")

    async def test_wide_shared_descendant_scrubs_message_but_preserves_other_original(self):
        copies = await self.graph(1000, fanout=True)
        async with self.tasks.transaction(self.f.owner) as db:
            merged = await self.tasks.payload(db, self.f.owner, {"text": "Synthetic merged evidence"},
                source_payload_ids=(*copies, self.control.payload_id))
        message = await self.messages.create_message(owner_id=self.f.owner, session_id=self.session.id,
            role="assistant", modality="audio", content="Synthetic copied conclusion.", source_payload_id=merged)
        await self.delete()
        await self.assert_removed([*copies, merged])
        rows = await self.messages.get_recent_messages(self.f.owner, self.session.id)
        self.assertEqual(rows[0].id, message.id)
        self.assertEqual(rows[0].content, "")
        self.assertEqual(rows[0].metadata["source_status"], "deleted")
        await self.delete()
        await self.assert_removed([*copies, merged])

    async def test_unavailable_and_expired_chains_can_be_escalated_to_deleted(self):
        for state in ("unavailable", "expired"):
            with self.subTest(state=state):
                original = await self.f.publish("Synthetic state-transition original.", source_key=state)
                copies = await self.graph(192, root=original)
                async with self.f.sources.transaction(self.f.owner) as db:
                    await db.execute("set local statement_timeout='10s'")
                    await db.execute("""update jarvis_context_payloads set availability=$3,content=null
                        where owner_id=$1 and id=$2""", self.f.owner, original.payload_id, state)
                await self.assert_removed(copies, state)
                await self.delete(original)
                await self.assert_removed(copies)

    async def test_existing_order_and_owner_guards_still_reject_invalid_edges(self):
        copies = await self.graph(3)
        async with self.f.pool.acquire() as db:
            foreign = await db.fetchval("""insert into jarvis_context_payloads(owner_id,availability,content)
                values($1,'available','{"text":"Other synthetic owner"}') returning id""", self.f.other)
            for parent, child in ((copies[-1], copies[0]), (copies[0], copies[0]), (foreign, copies[-1])):
                with self.subTest(parent=parent, child=child), self.assertRaises(asyncpg.CheckViolationError):
                    await db.execute("""insert into jarvis_context_payload_dependencies
                        (owner_id,source_payload_id,dependent_payload_id) values($1,$2,$3)""",
                        self.f.owner, parent, child)
        await self.delete()
        await self.assert_removed(copies)

    async def test_late_copy_and_message_cannot_use_a_removed_deep_leaf(self):
        copies = await self.graph(256)
        await self.delete()
        with self.assertRaises(asyncpg.CheckViolationError):
            async with self.tasks.transaction(self.f.owner) as db:
                await self.tasks.payload(db, self.f.owner, {"text": "Late synthetic copy"},
                                         source_payload_ids=(copies[-1],))
        with self.assertRaises(asyncpg.CheckViolationError):
            await self.messages.create_message(owner_id=self.f.owner, session_id=self.session.id,
                role="assistant", modality="audio", content="Late synthetic message.", source_payload_id=copies[-1])
        await self.assert_removed(copies)

    async def test_copy_committed_after_traversal_snapshot_is_still_removed(self):
        copies = await self.graph(96)
        pid_ready = asyncio.get_running_loop().create_future()
        pending = None
        async def remove():
            async with self.f.sources.transaction(self.f.owner) as db:
                pid_ready.set_result(await db.fetchval("select pg_backend_pid()"))
                await db.execute("set local statement_timeout='10s'")
                await self.f.sources.delete_source_in_transaction(db, self.f.owner, self.f.area,
                    self.original.source_id, expected_revision=1)
        try:
            async with self.f.pool.acquire() as writer:
                async with writer.transaction():
                    late = await self.tasks.payload(writer, self.f.owner, {"text": "Concurrent synthetic copy"},
                                                    source_payload_ids=(copies[-1],))
                    pending = asyncio.create_task(remove())
                    pid = await asyncio.wait_for(pid_ready, 5)
                    async with asyncio.timeout(5):
                        while not await writer.fetchval(
                                "select exists(select 1 from pg_locks where pid=$1 and not granted)", pid):
                            await asyncio.sleep(0.01)
                await asyncio.wait_for(pending, 10)
            await self.assert_removed([*copies, late])
        finally:
            if pending is not None and not pending.done():
                pending.cancel()
                await asyncio.gather(pending, return_exceptions=True)
