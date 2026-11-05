"""Shared validation ownership; mutations and materialization stay exclusive."""
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from app.attention.quiet import OWNER_LOCK, OWNER_SHARED_LOCK
from app.attention.quiet_delivery import quiet_response_guard
from app.knowledge.models import CanonicalOwnerBusy
from app.knowledge.sources import SourceRepository
from app.live_state.canonical import CanonicalContextBinder
from app.live_state.models import GuardedPayloadBinding, SessionScope
from app.realtime.source_context import LiveResponseSources
from app.runtime.feature_config import TaskLimits
from app.tasks.models import TaskError
from app.tasks.repository import TaskRepository


class Connection:
    def __init__(self):
        self.calls, self.lock_available = [], True

    @asynccontextmanager
    async def transaction(self):
        yield

    def is_in_transaction(self):
        return True

    async def execute(self, query, *args):
        self.calls.append(query)

    async def fetchval(self, query, *args):
        self.calls.append(query)
        return self.lock_available if "pg_try_advisory" in query else True

    async def fetchrow(self, query, *args):
        self.calls.append(query)
        return None


class Pool:
    def __init__(self):
        self.connection = Connection()

    @asynccontextmanager
    async def acquire(self):
        yield self.connection


class CanonicalSharedGuardTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.pool = Pool()
        self.sources = SourceRepository(self.pool)
        self.scope = SessionScope(owner_id="shared-guard-owner", session_id=uuid4())
        self.binder = CanonicalContextBinder(self.sources, self.scope, clock=lambda: datetime.now(UTC))

    async def test_materialization_transaction_remains_exclusive_in_existing_order(self):
        async with self.binder._transaction():
            pass
        locks = [q for q in self.pool.connection.calls if "pg_advisory" in q]
        self.assertEqual(len(locks), 2)
        self.assertIn(":jarvis-memory-owner:", locks[0])
        self.assertIn(":jarvis-task-owner:", locks[1])
        self.assertTrue(all("_shared" not in q for q in locks))

    async def test_guard_holds_shared_memory_and_task_keys(self):
        binding = GuardedPayloadBinding(scope=self.scope, payload_root_ids=(uuid4(),), dependencies=())
        async with self.binder.guard(binding):
            pass
        locks = [q for q in self.pool.connection.calls if "pg_advisory" in q]
        self.assertEqual(len(locks), 2)
        self.assertTrue(all("pg_advisory_xact_lock_shared(" in q for q in locks))
        self.assertIn(":jarvis-memory-owner:", locks[0])
        self.assertIn(":jarvis-task-owner:", locks[1])
        self.assertTrue(any("lock_jarvis_copy_sources" in q for q in self.pool.connection.calls))

    async def test_nonwaiting_reader_rejects_exclusive_owner(self):
        self.pool.connection.lock_available = False
        with self.assertRaises(CanonicalOwnerBusy):
            async with self.binder._transaction(wait_for_owner=False, read_only_guard=True):
                self.fail("Busy owner entered")
        self.assertIn("pg_try_advisory_xact_lock_shared(", self.pool.connection.calls[0])
        self.assertEqual(len(self.pool.connection.calls), 1)

    async def test_legacy_source_adapter_stays_exclusive(self):
        legacy = SimpleNamespace(pool=self.pool, transaction=self.sources.transaction)
        binder = CanonicalContextBinder(legacy, self.scope, clock=lambda: datetime.now(UTC))
        async with binder._transaction(read_only_guard=True):
            pass
        self.assertFalse(any("_shared(" in q for q in self.pool.connection.calls))

    async def test_borrowed_quiet_guard_does_not_upgrade(self):
        response_sources = LiveResponseSources(self.pool, owner_id=self.scope.owner_id,
            session_id=self.scope.session_id, audio_session_id="fixture")
        with patch("app.attention.quiet_delivery.assert_unsolicited_allowed", new=AsyncMock()):
            async with response_sources.use_transaction(self.pool.connection):
                async with quiet_response_guard(response_sources, origin="visual_interrupt"):
                    pass
        self.assertIn(OWNER_SHARED_LOCK, self.pool.connection.calls)
        self.assertNotIn(OWNER_LOCK, self.pool.connection.calls)

    async def test_task_source_reader_uses_shared_row_lock(self):
        tasks = TaskRepository(self.pool, TaskLimits())
        tasks.load = AsyncMock(return_value=object())
        tasks.context_problem = AsyncMock(return_value=None)
        connection = SimpleNamespace(is_in_transaction=lambda: True,
            fetchrow=AsyncMock(return_value={"last_event_sequence": 1, "result": None}))
        identity = uuid4()
        await tasks.live_context_in_transaction(connection, self.scope.owner_id, identity)
        tasks.load.assert_awaited_once_with(connection, self.scope.owner_id, identity, lock="share")

    async def test_task_load_keeps_mutation_lock_and_supports_shared_read(self):
        tasks = TaskRepository(self.pool, TaskLimits())
        for mode, suffix in ((True, " for update of t"), ("share", " for share of t")):
            with self.subTest(mode=mode), self.assertRaises(TaskError):
                await tasks.load(self.pool.connection, self.scope.owner_id, uuid4(), lock=mode)
            self.assertTrue(self.pool.connection.calls[-1].endswith(suffix))
