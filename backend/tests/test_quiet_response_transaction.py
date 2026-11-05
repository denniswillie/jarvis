"""Offline exact-task quiet/source transaction ownership, not SQL acceptance."""
import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch
from uuid import UUID, uuid4

from app.attention.quiet import OWNER_LOCK
from app.attention.quiet_delivery import QuietDeliverySuppressed, quiet_response_guard
from app.realtime.source_context import LiveContextError, LiveResponseSources


class Connection:
    def __init__(self, name, trace):
        self.name, self.trace = name, trace
        self.depth = self.entries = self.exits = 0
        self.available = self.valid = True
        self.lock_error = None

    @asynccontextmanager
    async def transaction(self):
        self.depth += 1
        self.entries += 1
        try:
            yield self
        finally:
            self.depth -= 1
            self.exits += 1

    def is_in_transaction(self):
        return self.depth > 0 and self.valid

    async def execute(self, query, owner):
        if query != OWNER_LOCK or not self.is_in_transaction():
            raise AssertionError("Expected transaction-bound owner lock")
        self.trace.append((self.name, "owner", owner))
        if self.lock_error is not None:
            raise self.lock_error
        return "SELECT 1"

    async def fetchval(self, query, owner, roots):
        if query != "select lock_jarvis_copy_sources($1,$2::uuid[])" or not self.is_in_transaction():
            raise AssertionError("Expected transaction-bound source check")
        self.trace.append((self.name, "sources", tuple(roots)))
        return self.available


class Pool:
    def __init__(self, trace):
        self.trace = trace
        self.connections = []
        self.released = 0

    @asynccontextmanager
    async def acquire(self):
        connection = Connection("pool-" + str(len(self.connections)), self.trace)
        self.connections.append(connection)
        try:
            yield connection
        finally:
            self.released += 1


class QuietResponseTransactionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.trace = []
        self.pool = Pool(self.trace)
        self.parent = Connection("canonical", self.trace)
        self.sources = LiveResponseSources(self.pool, owner_id="fixture-owner", session_id=uuid4(),
                                          audio_session_id="fixture-audio")
        self.sources.observe((UUID(int=1),))
        self.decision = AsyncMock(side_effect=self.allowed)
        patcher = patch("app.attention.quiet_delivery.quiet_delivery_decision", self.decision)
        patcher.start()
        self.addCleanup(patcher.stop)

    async def allowed(self, connection, owner, *, kind, channel):
        self.assertTrue(connection.is_in_transaction())
        self.assertIs(self.sources._current_connection(), connection)
        self.assertEqual((owner, kind, channel), ("fixture-owner", "unsolicited", "voice"))
        self.trace.append((connection.name, "quiet", owner))
        return SimpleNamespace(allowed=True)

    def guard(self, source_guard=None):
        return quiet_response_guard(self.sources, origin="prospective",
                                    source_guard=source_guard or self.sources.context_guard)

    def assert_released(self):
        self.assertEqual(len(self.pool.connections), self.pool.released)
        for connection in self.pool.connections:
            self.assertEqual(connection.depth, 0)
            self.assertEqual(connection.entries, connection.exits)
        self.assertIsNone(self.sources._current_connection())

    async def test_borrowed_owner_transaction_keeps_quiet_sources_and_writer_on_one_connection(self):
        async with self.parent.transaction(), self.sources.use_transaction(self.parent):
            async with self.guard() as connection:
                self.assertIs(connection, self.parent)
                self.assertIs(self.sources._current_connection(), self.parent)
                self.assertEqual(self.parent.depth, 1)
                self.assertEqual(self.parent.entries, 1)
                self.assertEqual(self.pool.connections, [])
                self.trace.append((connection.name, "writer", None))
            self.assertIs(self.sources._current_connection(), self.parent)
            self.assertTrue(self.parent.is_in_transaction())
        self.assertEqual([row[1] for row in self.trace], ["owner", "quiet", "sources", "writer"])
        self.assertEqual(self.parent.entries, self.parent.exits)
        self.assert_released()

    async def test_without_borrowed_transaction_acquires_and_releases_exactly_one(self):
        async with self.guard() as connection:
            self.assertIs(connection, self.pool.connections[0])
            self.assertEqual(connection.depth, 1)
            self.assertEqual(self.pool.released, 0)
            self.trace.append((connection.name, "writer", None))
        self.assertEqual(len(self.pool.connections), 1)
        self.assertEqual([row[1] for row in self.trace], ["owner", "quiet", "sources", "writer"])
        self.assert_released()

    async def test_child_does_not_borrow_inherited_parent_transaction(self):
        async def child():
            self.assertIsNone(self.sources._current_connection())
            async with self.guard() as connection:
                self.assertIsNot(connection, self.parent)
                self.assertIs(self.sources._current_connection(), connection)
            self.assertIsNone(self.sources._current_connection())
        async with self.parent.transaction(), self.sources.use_transaction(self.parent):
            await asyncio.create_task(child())
            self.assertIs(self.sources._current_connection(), self.parent)
            self.assertTrue(self.parent.is_in_transaction())
            self.assertEqual(self.parent.entries, 1)
            self.assertEqual(len(self.pool.connections), 1)
        self.assertFalse(any(row[0] == "canonical" for row in self.trace))
        self.assert_released()

    async def test_inactive_borrowed_transaction_rejects_before_sql_without_fallback(self):
        async with self.parent.transaction(), self.sources.use_transaction(self.parent):
            self.parent.valid = False
            with self.assertRaises(LiveContextError) as caught:
                async with self.guard():
                    self.fail("Inactive transaction cannot authorize a write")
            self.assertEqual(caught.exception.code, "live_context_transaction_required")
            self.assertIs(self.sources._current_connection(), self.parent)
        self.assertEqual(self.trace, [])
        self.assertEqual(self.pool.connections, [])
        self.decision.assert_not_awaited()
        self.assert_released()

    async def test_quiet_policy_rejects_with_valid_borrowed_source_authority(self):
        self.decision.side_effect = None
        self.decision.return_value = SimpleNamespace(allowed=False, reason="owner_quiet_active")
        async with self.parent.transaction(), self.sources.use_transaction(self.parent):
            with self.assertRaises(QuietDeliverySuppressed) as caught:
                async with self.guard():
                    self.fail("Canonical source authority does not bypass quiet")
            self.assertEqual(caught.exception.code, "owner_quiet_active")
            self.assertEqual(self.parent.depth, 1)
            self.assertIs(self.sources._current_connection(), self.parent)
        self.assertEqual([row[1] for row in self.trace], ["owner"])
        self.assertEqual(self.pool.connections, [])
        self.assert_released()

    async def test_failed_quiet_read_is_sanitized_and_never_enters_source_or_writer(self):
        self.decision.side_effect = RuntimeError("PRIVATE fixture content")
        async with self.parent.transaction(), self.sources.use_transaction(self.parent):
            with self.assertRaises(QuietDeliverySuppressed) as caught:
                async with self.guard():
                    self.fail("Failed quiet reads do not permit output")
            self.assertEqual(caught.exception.code, "owner_quiet_unavailable")
            self.assertNotIn("PRIVATE", str(caught.exception))
            self.assertIs(self.sources._current_connection(), self.parent)
        self.assertEqual([row[1] for row in self.trace], ["owner"])
        self.assertEqual(self.pool.connections, [])
        self.assert_released()

    async def test_source_withdrawal_after_allowed_quiet_still_rejects(self):
        self.parent.available = False
        async with self.parent.transaction(), self.sources.use_transaction(self.parent):
            with self.assertRaises(LiveContextError) as caught:
                async with self.guard():
                    self.fail("Allowed quiet policy does not authorize revoked sources")
            self.assertEqual(caught.exception.code, "live_context_revoked")
            self.assertIs(self.sources._current_connection(), self.parent)
        self.assertEqual([row[1] for row in self.trace], ["owner", "quiet", "sources"])
        self.assertEqual(self.pool.connections, [])
        self.assert_released()

    async def test_owner_lock_error_and_cancellation_propagate_without_policy_or_sources(self):
        for error in (RuntimeError("fixture failure"), asyncio.CancelledError()):
            with self.subTest(error=type(error).__name__):
                self.parent.lock_error = error
                async with self.parent.transaction(), self.sources.use_transaction(self.parent):
                    with self.assertRaises(type(error)):
                        async with self.guard():
                            self.fail("Owner-lock failure must stop the write")
                    self.assertIs(self.sources._current_connection(), self.parent)
        self.decision.assert_not_awaited()
        self.assertEqual(self.pool.connections, [])
        self.assertTrue(all(row[1] == "owner" for row in self.trace))
        self.assert_released()

    async def test_cancelled_quiet_wait_restores_borrowed_context_and_joins_transaction(self):
        entered, joined = asyncio.Event(), asyncio.Event()
        async def wait(*args, **kwargs):
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                joined.set()
        self.decision.side_effect = wait
        restored = []
        async def writer():
            async with self.parent.transaction(), self.sources.use_transaction(self.parent):
                try:
                    async with self.guard():
                        self.fail("Cancelled policy check cannot reach a writer")
                finally:
                    restored.append(self.sources._current_connection() is self.parent)
        task = asyncio.create_task(writer())
        try:
            await asyncio.wait_for(entered.wait(), 0.5)
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        self.assertTrue(task.cancelled())
        self.assertTrue(joined.is_set())
        self.assertEqual(restored, [True])
        self.assertEqual(self.parent.depth, 0)
        self.assertEqual(self.parent.entries, self.parent.exits)
        self.assertEqual(self.pool.connections, [])
        self.assert_released()

    async def test_writer_exception_restores_parent_without_ending_its_transaction(self):
        async with self.parent.transaction(), self.sources.use_transaction(self.parent):
            with self.assertRaises(RuntimeError):
                async with self.guard():
                    raise RuntimeError("fixture unknown write")
            self.assertTrue(self.parent.is_in_transaction())
            self.assertIs(self.sources._current_connection(), self.parent)
            self.assertEqual(self.parent.exits, 0)
        self.assertEqual(self.parent.entries, self.parent.exits)
        self.assertEqual(self.pool.connections, [])
        self.assert_released()

    async def test_no_source_guard_still_requires_locked_allowed_quiet(self):
        async with self.parent.transaction(), self.sources.use_transaction(self.parent):
            async with quiet_response_guard(self.sources, origin="prospective") as connection:
                self.assertIs(connection, self.parent)
                self.assertEqual(self.parent.depth, 1)
        self.assertEqual([row[1] for row in self.trace], ["owner", "quiet"])
        self.assertEqual(self.pool.connections, [])
        self.assert_released()
