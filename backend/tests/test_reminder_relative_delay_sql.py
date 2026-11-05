"""Real database timing and durable replay, through manifested disposable schemas.

No scheduler, provider, Hindsight, or native session is started. A controlled
post-insert exception models a lost caller return, not a missing database effect.
"""
import asyncio
from datetime import datetime, timedelta
import json
import os
from pathlib import Path
import unittest
from uuid import uuid4

import asyncpg

from app.experience.repository import ToolExecutionRepository
from app.migrations.runner import quote_schema
from app.reminders.repository import ReminderRepository
from app.reminders.tools import create_reminder_tools
from app.tools.executor import JarvisToolExecutor
from app.tools.registry import JarvisToolRegistry
from evals.second_brain.source_fixture import close_fixture_pool, source_fixture


class ObservedReminderRepository(ReminderRepository):
    """Count real repository invocations; optionally lose only the committed return."""

    def __init__(self, pool, *, lose_return=False):
        super().__init__(pool)
        self.calls = 0
        self.committed = []
        self.lose_return = lose_return

    async def create_reminder(self, **arguments):
        self.calls += 1
        reminder = await super().create_reminder(**arguments)
        self.committed.append(reminder)
        if self.lose_return:
            raise RuntimeError("controlled_return_lost_after_real_insert")
        return reminder


@unittest.skipUnless(
    os.getenv("JARVIS_SECOND_BRAIN_TEST_DB_URL")
    and os.getenv("JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"),
    "requires explicitly configured manifested SQL qualification",
)
class RelativeReminderDatabaseTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.dsn = os.environ["JARVIS_SECOND_BRAIN_TEST_DB_URL"]
        self.directory = Path(os.environ["JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"])
        self.fixture = await self.enterAsyncContext(source_fixture(
            self.dsn, self.directory, test_name=self.id()))
        self.pool, self.owner = self.fixture.pool, self.fixture.owner
        self.schema = self.fixture.fixture.manifest["schemas"][0]
        async with self.pool.acquire() as connection:
            self.assertEqual(await connection.fetchval("select current_schema()"), self.schema)
            self.session = await connection.fetchval(
                "insert into jarvis_sessions(owner_id,title) values($1,'Relative reminder SQL') returning id",
                self.owner,
            )
        self.repository = ReminderRepository(self.pool)
        self.executions = ToolExecutionRepository(self.pool)

    def executor(self, repository):
        registry = JarvisToolRegistry()
        registry.register_many(create_reminder_tools(
            repository=repository, owner_id=self.owner, session_id=self.session))
        return JarvisToolExecutor(registry, execution_repository=self.executions,
            owner_id=self.owner, session_id=self.session)

    async def create(self, repository=None, **changes):
        return await (repository or self.repository).create_reminder(**{
            "owner_id": self.owner, "session_id": self.session,
            "title": "Check the task", "details": "Isolated relative schedule",
            "remind_at": None, "condition": None, "delay_seconds": 10,
            **changes,
        })

    async def stored(self, reminder_id):
        async with self.pool.acquire() as connection:
            row = await connection.fetchrow("""select *,
                remind_at-(metadata->>'scheduled_at')::timestamptz as persisted_delay,
                clock_timestamp() as inspected_at
                from jarvis_reminders where owner_id=$1 and session_id=$2 and id=$3""",
                self.owner, self.session, reminder_id)
        self.assertIsNotNone(row)
        value = dict(row)
        if isinstance(value["metadata"], str):
            value["metadata"] = json.loads(value["metadata"])
        return value

    async def reminder_count(self):
        async with self.pool.acquire() as connection:
            return await connection.fetchval(
                "select count(*) from jarvis_reminders where owner_id=$1", self.owner)

    def assert_anchor(self, reminder, row):
        anchor = datetime.fromisoformat(row["metadata"]["scheduled_at"])
        self.assertIsNotNone(anchor.utcoffset())
        self.assertEqual(row["owner_id"], self.owner)
        self.assertEqual(row["session_id"], self.session)
        self.assertEqual(row["id"], reminder.id)
        self.assertEqual(row["persisted_delay"], timedelta(seconds=10))
        self.assertEqual(row["remind_at"] - anchor, timedelta(seconds=10))
        self.assertEqual(row["created_at"], anchor)
        self.assertEqual(row["updated_at"], anchor)
        self.assertEqual(reminder.remind_at, row["remind_at"])
        self.assertEqual(reminder.created_at, anchor)
        self.assertEqual(reminder.updated_at, anchor)
        self.assertEqual(reminder.metadata, row["metadata"])
        self.assertEqual(row["metadata"]["delay_seconds"], 10)
        self.assertEqual(row["metadata"]["scheduling_clock"], "postgres_clock_timestamp")
        self.assertLessEqual(anchor, row["inspected_at"])
        return anchor

    def record(self, **facts):
        directory = self.directory / "relative-reminder-evidence"
        directory.mkdir(mode=0o700, exist_ok=True)
        path = directory / (self.fixture.fixture.run_id + ".json")
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w") as stream:
            json.dump({"test": self.id(), "fixture_manifest": str(self.fixture.fixture.path),
                "owner_id": self.owner, "session_id": str(self.session), "schema": self.schema,
                "application_clock_injected": False, "native_or_provider_acceptance": False,
                **facts}, stream, indent=2, default=str)
            stream.write("\n")

    async def test_persisted_ten_seconds_share_one_database_anchor_and_scope(self):
        async with self.pool.acquire() as connection:
            before = await connection.fetchval("select clock_timestamp()")
        reminder = await self.create(metadata={"scheduled_at": "not-authority",
            "delay_seconds": 999, "scheduling_clock": "not-authority", "unrelated": "preserved"})
        row = await self.stored(reminder.id)
        anchor = self.assert_anchor(reminder, row)
        self.assertGreaterEqual(anchor, before)
        self.assertEqual(row["metadata"]["unrelated"], "preserved")
        self.assertEqual(await self.reminder_count(), 1)
        self.assertEqual(await self.repository.list_reminders(
            owner_id=self.fixture.other, status=None, limit=10), [])
        self.record(reminder_id=reminder.id, database_before=before, scheduled_at=anchor,
            remind_at=row["remind_at"], persisted_delay_seconds=row["persisted_delay"].total_seconds(),
            created_at=row["created_at"], updated_at=row["updated_at"], reserved_metadata_overridden=True)

    async def test_pool_wait_anchors_after_database_marker_before_release(self):
        async def setup(connection):
            await connection.execute(
                f"set search_path to {quote_schema(self.schema)},pg_catalog,public,extensions")

        pool = await asyncpg.create_pool(self.dsn, min_size=1, max_size=1, timeout=5,
            command_timeout=10, setup=setup)
        self.addAsyncCleanup(close_fixture_pool, pool)
        repository = ReminderRepository(pool)
        started = asyncio.Event()
        task = None

        async def create_while_held():
            started.set()
            return await self.create(repository)

        try:
            async with pool.acquire() as held:
                self.assertEqual(await held.fetchval("select current_schema()"), self.schema)
                task = asyncio.create_task(create_while_held())
                await asyncio.wait_for(started.wait(), 2)
                self.assertEqual(pool.get_size(), 1)
                self.assertEqual(pool.get_idle_size(), 0)
                self.assertFalse(task.done())
                # The task yielded in real pool acquisition. This is a DB clock
                # marker immediately before releasing that one held connection.
                marker = await held.fetchval("select clock_timestamp()")
            reminder = await asyncio.wait_for(task, 15)
            row = await self.stored(reminder.id)
            anchor = self.assert_anchor(reminder, row)
            self.assertGreaterEqual(anchor, marker)
            self.assertEqual(await self.reminder_count(), 1)
            self.record(reminder_id=reminder.id, held_database_marker=marker, scheduled_at=anchor,
                remind_at=row["remind_at"], max_pool_size=1, pending_while_held=True,
                anchor_after_marker_seconds=(anchor-marker).total_seconds())
        finally:
            if task is not None:
                if not task.done():
                    task.cancel()
                await asyncio.gather(task, return_exceptions=True)

    async def test_completed_provider_call_replays_original_schedule_once(self):
        repository = ObservedReminderRepository(self.pool)
        executor = self.executor(repository)
        call_id = "relative-completed-" + uuid4().hex
        arguments = {"title": "Check the task", "delay_seconds": 10}
        first = await executor.execute("create_reminder", arguments, provider_call_id=call_id)
        self.assertEqual(first["status"], "success")
        reminder = repository.committed[0]
        before = await self.stored(reminder.id)
        anchor = self.assert_anchor(reminder, before)
        async with self.pool.acquire() as connection:
            replay_marker = await connection.fetchval("select clock_timestamp()")
        second = await executor.execute("create_reminder", arguments, provider_call_id=call_id)
        self.assertEqual(second, first)
        self.assertEqual(repository.calls, 1)
        self.assertEqual(await self.reminder_count(), 1)
        after = await self.stored(reminder.id)
        self.assertEqual(after["metadata"], before["metadata"])
        self.assertEqual(after["remind_at"], before["remind_at"])
        self.assertEqual(after["created_at"], before["created_at"])
        self.assertEqual(after["updated_at"], before["updated_at"])
        rows = await self.executions.get_executions_by_call_ids(
            owner_id=self.owner, session_id=self.session, provider_call_ids=[call_id])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].status, "completed")
        self.assertEqual(rows[0].result["reminder"], first["reminder"])
        self.assertEqual(rows[0].owner_id, self.owner)
        self.assertEqual(rows[0].session_id, self.session)
        self.record(provider_call_id=call_id, reminder_id=reminder.id, scheduled_at=anchor,
            remind_at=after["remind_at"], database_replay_marker=replay_marker,
            real_repository_invocations=repository.calls, reminder_rows=1, execution_status="completed",
            replay_result_identical=True)

    async def test_lost_return_after_real_insert_replay_does_not_duplicate(self):
        repository = ObservedReminderRepository(self.pool, lose_return=True)
        executor = self.executor(repository)
        call_id = "relative-lost-return-" + uuid4().hex
        arguments = {"title": "Check the task", "delay_seconds": 10}
        first = await executor.execute("create_reminder", arguments, provider_call_id=call_id)
        self.assertEqual(first["status"], "failed")
        self.assertEqual(repository.calls, 1)
        reminder = repository.committed[0]
        before = await self.stored(reminder.id)
        anchor = self.assert_anchor(reminder, before)
        repository.lose_return = False
        second = await executor.execute("create_reminder", arguments, provider_call_id=call_id)
        self.assertEqual(second, first)
        self.assertEqual(repository.calls, 1)
        self.assertEqual(await self.reminder_count(), 1)
        after = await self.stored(reminder.id)
        self.assertEqual(after["metadata"], before["metadata"])
        self.assertEqual(after["remind_at"], before["remind_at"])
        rows = await self.executions.get_executions_by_call_ids(
            owner_id=self.owner, session_id=self.session, provider_call_ids=[call_id])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].status, "failed")
        self.record(provider_call_id=call_id, reminder_id=reminder.id, scheduled_at=anchor,
            remind_at=after["remind_at"], real_repository_invocations=repository.calls, reminder_rows=1,
            execution_status="failed", controlled_loss="after_real_committed_insert_return",
            failed_result_does_not_prove_no_effect=True, replay_result_identical=True)
