"""Offline relative scheduling contracts; PostgreSQL clock execution is separate.

Real tool/repository/executor with controlled connection and typed ledger I/O.
The fake database clock qualifies query shape/order and returned values, not an
actual PostgreSQL timestamp, scheduler delivery, provider behavior or hearing.
"""
import asyncio
from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json
import unittest
from uuid import uuid4

from app.experience.models import ToolExecution
from app.reminders.models import MAX_REMINDER_DELAY_SECONDS
from app.reminders.repository import ReminderRepository
from app.reminders.tools import create_reminder_tools
from app.tools.executor import JarvisToolExecutor
from app.tools.registry import JarvisToolRegistry


class DatabaseIO:
    def __init__(self):
        self.now = datetime(2026, 11, 1, 2, 9, 46, 288765, tzinfo=timezone.utc)
        self.acquiring, self.release = asyncio.Event(), asyncio.Event()
        self.release.set()
        self.depth = self.acquisitions = self.clock_samples = 0
        self.calls, self.rows = [], []
        self.lose_insert_response = False

    @asynccontextmanager
    async def acquire(self):
        self.acquisitions += 1
        self.acquiring.set()
        await self.release.wait()
        self.depth += 1
        try:
            yield self
        finally:
            self.depth -= 1

    async def fetchrow(self, query, *args):
        assert self.depth == 1
        self.calls.append((query, args))
        relative = "with scheduling as materialized" in query
        if relative:
            assert query.count("clock_timestamp()") == 1
            owner, session, title, details, delay, raw_metadata = args
            self.clock_samples += 1
            scheduled = self.now
            due, condition = scheduled + timedelta(seconds=delay), None
            metadata = json.loads(raw_metadata) | {"scheduled_at": scheduled.isoformat(),
                "delay_seconds": delay, "scheduling_clock": "postgres_clock_timestamp"}
        else:
            owner, session, title, details, due, condition, raw_metadata = args
            scheduled, metadata = self.now, json.loads(raw_metadata)
        row = dict(id=uuid4(), owner_id=owner, session_id=session, title=title, details=details,
            remind_at=due, condition=condition, status="active", created_at=scheduled,
            updated_at=scheduled, completed_at=None, triggered_at=None, trigger_claimed_at=None,
            trigger_claim_token=None, trigger_attempt_count=0, last_trigger_error=None,
            metadata=json.dumps(metadata))
        self.rows.append(row)
        if self.lose_insert_response:
            raise TimeoutError("offline_insert_response_lost")
        return row


class ExecutionIO:
    def __init__(self):
        self.rows = {}
        self.fail_finalization = False

    async def start_execution_with_status(self, **fields):
        key = fields["provider_call_id"]
        if key in self.rows:
            prior = self.rows[key]
            assert (prior.owner_id, prior.session_id, prior.arguments, prior.tool_name) == (
                fields["owner_id"], fields["session_id"], fields["arguments"], fields["tool_name"])
            return prior, False
        row = ToolExecution(id=uuid4(), **fields, status="started", result=None, error=None,
                            completed_at=None, created_at=fields["started_at"])
        self.rows[key] = row
        return row, True

    async def finalize_execution(self, *, owner_id, session_id, provider_call_id, **fields):
        old = self.rows[provider_call_id]
        assert (old.owner_id, old.session_id) == (owner_id, session_id)
        if self.fail_finalization:
            raise TimeoutError("offline_terminal_receipt_lost")
        self.rows[provider_call_id] = row = replace(old, **fields)
        return row


class RelativeReminderTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.db, self.ledger = DatabaseIO(), ExecutionIO()
        self.owner, self.session = "offline-relative-owner", uuid4()
        self.repository = ReminderRepository(self.db)
        self.registry = JarvisToolRegistry()
        self.registry.register_many(create_reminder_tools(repository=self.repository,
            owner_id=self.owner, session_id=self.session))
        self.tool = self.registry.get("create_reminder")
        self.executor = JarvisToolExecutor(self.registry, execution_repository=self.ledger,
            owner_id=self.owner, session_id=self.session)

    def tearDown(self):
        self.assertEqual(self.db.depth, 0)

    async def create(self, **arguments):
        return await self.tool.handler({"title": "Check the task", **arguments})

    async def test_exact_ten_seconds_and_persisted_returned_clock_receipt(self):
        result = await self.create(delay_seconds=10)
        self.assertEqual(result["status"], "success")
        reminder = result["reminder"]
        scheduled = datetime.fromisoformat(reminder["metadata"]["scheduled_at"])
        self.assertEqual(datetime.fromisoformat(reminder["remind_at"]) - scheduled, timedelta(seconds=10))
        self.assertEqual(datetime.fromisoformat(reminder["created_at"]), scheduled)
        self.assertEqual(datetime.fromisoformat(reminder["updated_at"]), scheduled)
        self.assertEqual(reminder["metadata"]["delay_seconds"], 10)
        self.assertEqual(reminder["metadata"]["scheduling_clock"], "postgres_clock_timestamp")
        self.assertEqual((self.db.acquisitions, self.db.clock_samples, len(self.db.rows)), (1, 1, 1))
        query, args = self.db.calls[0]
        self.assertEqual(query.count("clock_timestamp()"), 1)
        self.assertIn("with scheduling as materialized", query)
        self.assertIn("scheduled_at + make_interval(secs => $5::double precision)", query)
        self.assertEqual(args[4], 10)
        self.assertFalse(any(isinstance(value, datetime) for value in args))

    async def test_pool_wait_does_not_start_relative_timer_before_acquisition(self):
        self.db.release.clear()
        before = self.db.now
        task = asyncio.create_task(self.create(delay_seconds=10))
        try:
            await asyncio.wait_for(self.db.acquiring.wait(), 1)
            self.assertEqual((self.db.clock_samples, self.db.calls), (0, []))
            self.db.now += timedelta(seconds=37)
            self.db.release.set()
            result = await asyncio.wait_for(task, 1)
            reminder = result["reminder"]
            self.assertEqual(datetime.fromisoformat(reminder["metadata"]["scheduled_at"]), before + timedelta(seconds=37))
            self.assertEqual(datetime.fromisoformat(reminder["remind_at"]), before + timedelta(seconds=47))
        finally:
            self.db.release.set()
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def test_relative_metadata_cannot_override_database_derived_fields(self):
        reminder = await self.repository.create_reminder(owner_id=self.owner, session_id=self.session,
            title="Check", details="", remind_at=None, condition=None, delay_seconds=10,
            metadata={"scheduled_at": "invented", "delay_seconds": 1, "scheduling_clock": "invented",
                      "unrelated": "preserved"})
        self.assertEqual(reminder.metadata, {"scheduled_at": self.db.now.isoformat(), "delay_seconds": 10,
            "scheduling_clock": "postgres_clock_timestamp", "unrelated": "preserved"})
        self.assertIn("$6::jsonb || jsonb_build_object", self.db.calls[0][0])

    async def test_invalid_relative_types_and_bounds_reject_before_database(self):
        for delay in (True, False, 0, -1, MAX_REMINDER_DELAY_SECONDS + 1, 10.0, 0.5,
                      "10", float("nan"), float("inf"), [], {}):
            with self.subTest(type=type(delay).__name__, value=repr(delay)):
                result = await self.create(delay_seconds=delay)
                self.assertEqual(result["status"], "failed")
                self.assertEqual(result["error"]["code"], "invalid_tool_input")
        self.assertEqual((self.db.acquisitions, self.db.calls), (0, []))

    async def test_mixed_relative_inputs_have_explicit_error_before_database(self):
        for extra in ({"remind_at": "2026-11-02T10:00:00Z"}, {"condition": "when the door opens"},
                      {"remind_at": "invalid", "condition": "when the door opens"}, {"condition": ""}):
            with self.subTest(fields=tuple(extra)):
                result = await self.create(delay_seconds=10, **extra)
                self.assertEqual(result["error"]["code"], "invalid_tool_input")
                self.assertIn("cannot be combined", result["error"]["message"])
        self.assertEqual(self.db.acquisitions, 0)

    async def test_repository_itself_rejects_invalid_relative_schedule(self):
        for delay, due, condition in ((True, None, None), (0, None, None), (10.0, None, None),
            (MAX_REMINDER_DELAY_SECONDS + 1, None, None), (10, self.db.now, None), (10, None, "visual")):
            with self.subTest(delay=delay, absolute=due is not None, condition=condition is not None):
                with self.assertRaisesRegex(ValueError, "invalid_relative_reminder_schedule"):
                    await self.repository.create_reminder(owner_id=self.owner, session_id=self.session,
                        title="Check", details="", remind_at=due, condition=condition, delay_seconds=delay)
        self.assertEqual(self.db.acquisitions, 0)

    async def test_relative_endpoints_and_explicit_nulls(self):
        for delay in (1, MAX_REMINDER_DELAY_SECONDS):
            with self.subTest(delay=delay):
                result = await self.create(delay_seconds=delay, remind_at=None, condition=None)
                self.assertEqual(result["status"], "success")
                self.assertEqual(datetime.fromisoformat(result["reminder"]["remind_at"]),
                                 self.db.now + timedelta(seconds=delay))

    async def test_existing_absolute_condition_and_combined_paths_unchanged(self):
        due = "2026-11-02T10:00:00+00:00"
        for arguments in ({"remind_at": due}, {"condition": "when the door opens"},
                          {"remind_at": due, "condition": "when the door opens", "delay_seconds": None}):
            with self.subTest(fields=tuple(arguments)):
                result = await self.create(**arguments)
                self.assertEqual(result["status"], "success")
                self.assertEqual(result["reminder"]["remind_at"], arguments.get("remind_at"))
                self.assertEqual(result["reminder"]["condition"], arguments.get("condition"))
                self.assertEqual(result["reminder"]["metadata"], {})
                self.assertNotIn("clock_timestamp()", self.db.calls[-1][0])
        self.assertEqual(self.db.clock_samples, 0)
        absent = await self.create()
        self.assertEqual(absent["error"]["code"], "invalid_tool_input")
        self.assertEqual(len(self.db.rows), 3)

    async def test_completed_same_call_replays_original_deadline_without_new_insert(self):
        arguments = {"title": "Check the task", "delay_seconds": 10}
        first = await self.executor.execute("create_reminder", arguments, provider_call_id="same-call")
        self.db.now += timedelta(minutes=5)
        second = await self.executor.execute("create_reminder", arguments, provider_call_id="same-call")
        self.assertEqual(first["status"], "success")
        self.assertEqual(second, first)
        self.assertEqual((len(self.db.rows), self.db.clock_samples, len(self.db.calls)), (1, 1, 1))
        self.assertEqual(self.ledger.rows["same-call"].status, "completed")

    async def test_ambiguous_insert_response_is_not_automatically_retried(self):
        self.db.lose_insert_response = True
        arguments = {"title": "Check the task", "delay_seconds": 10}
        first = await self.executor.execute("create_reminder", arguments, provider_call_id="lost-response")
        self.assertEqual(first["status"], "failed")
        self.assertEqual(len(self.db.rows), 1)  # Failure is NOT proof that no reminder exists.
        self.db.now += timedelta(minutes=5)
        second = await self.executor.execute("create_reminder", arguments, provider_call_id="lost-response")
        self.assertEqual(second, first)
        self.assertEqual((self.db.clock_samples, len(self.db.calls)), (1, 1))

    async def test_missing_terminal_ledger_receipt_does_not_repeat_insert(self):
        self.ledger.fail_finalization = True
        arguments = {"title": "Check the task", "delay_seconds": 10}
        first = await self.executor.execute("create_reminder", arguments, provider_call_id="lost-ledger")
        self.assertEqual(first["error"]["code"], "tool_execution_persistence_failed")
        second = await self.executor.execute("create_reminder", arguments, provider_call_id="lost-ledger")
        self.assertEqual(second["error"]["code"], "tool_execution_in_progress")
        self.assertEqual((len(self.db.rows), self.db.clock_samples, len(self.db.calls)), (1, 1, 1))

    async def test_cancel_while_waiting_for_pool_never_samples_or_inserts(self):
        self.db.release.clear()
        task = asyncio.create_task(self.create(delay_seconds=10))
        try:
            await asyncio.wait_for(self.db.acquiring.wait(), 1)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertEqual((self.db.clock_samples, self.db.calls, self.db.rows), (0, [], []))
        finally:
            self.db.release.set()
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    def test_normal_tool_schema_advertises_relative_delay_and_mutation(self):
        advertised = next(t for t in self.registry.openai_realtime_tools() if t["name"] == "create_reminder")
        field = advertised["parameters"]["properties"]["delay_seconds"]
        self.assertEqual(field["type"], ["integer", "null"])
        self.assertEqual((field["minimum"], field["maximum"]), (1, MAX_REMINDER_DELAY_SECONDS))
        self.assertIn("Do not calculate an absolute time for a relative delay", advertised["description"])
        self.assertTrue(self.tool.mutates_state)
