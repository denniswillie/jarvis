"""Actual publication/coordinator/source guards; fake SQL and explicit fake ACKs."""
import asyncio
from contextlib import asynccontextmanager
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from uuid import UUID

from app.realtime.source_context import LiveContextError, LiveResponseSources
from app.realtime.voice_route_runtime import RealtimeVoiceRouteRuntime
from app.voice_routes.coordinator import CoordinationPhase, VoiceRouteCoordinator
from tests import test_realtime_voice_route as route_fixtures
from tests import test_response_preparation_source_race as source_fixtures


class BorrowedConnection(source_fixtures.Connection):
    def is_in_transaction(self):
        return self.entered > self.exited


class NativeContextAdditionRaceTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.connection = BorrowedConnection()
        self.pool = source_fixtures.Pool(self.connection)
        self.sources = LiveResponseSources(self.pool, owner_id="fixture", session_id=UUID(int=1),
                                          audio_session_id=str(UUID(int=2)))
        self.runtime = object.__new__(RealtimeVoiceRouteRuntime)
        self.runtime.session = SimpleNamespace(_source_context=self.sources, _closed=asyncio.Event(),
            _context_revoked=False, context=SimpleNamespace(audio_session_id=str(UUID(int=2))))
        self.runtime._installed = True
        self.runtime._closing = self.runtime._closed = False
        self.runtime.channel = SimpleNamespace(ready=True)
        self.runtime._payload_roots = lambda _: (UUID(int=1),)
        @asynccontextmanager
        async def atomic(_):
            async with self.pool.acquire() as connection, connection.transaction():
                yield connection
        self.runtime._atomic_guard = atomic
        self.control = route_fixtures.FakeControl()
        self.control.source_guard = self.runtime._source_guard
        self.route = route_fixtures.make_adapter(self.control)
        self.request = route_fixtures.make_start()
        self.runtime.request = self.request
        async def valid(_):
            return True
        self.owner = VoiceRouteCoordinator(self.route.identity.scope, validity=valid,
            control_timeout=1, clock=lambda:route_fixtures.NOW)
        self.runtime.coordinator = self.owner
        logging = patch("app.realtime.voice_route_runtime._READINESS_LOGGER.warning")
        self.log = logging.start()
        self.addCleanup(logging.stop)
        await self.owner.start(self.route, self.request)
        self.assertTrue(self.runtime.ready)
        self.initial_acquisitions = self.pool.entered

    async def asyncTearDown(self):
        self.connection.on_lock = None
        await self.owner.close()

    def rows(self, stage):
        return [row for call in self.log.call_args_list
                if (row := json.loads(call.args[1]))["stage"] == stage]

    async def publish(self):
        update = route_fixtures.make_context(revision=1)
        self.owner.offer_context(update)
        async with asyncio.timeout(1):
            while (self.owner.status.phase == CoordinationPhase.ACTIVE
                   and self.owner._bridge.status.last_receipt is None):
                await asyncio.sleep(0)
        return update

    def counts(self):
        return tuple(sum(item["type"] == kind for item in self.control.writes)
                     for kind in ("conversation.item.create", "conversation.item.delete"))

    async def test_one_or_two_unrelated_additions_refresh_before_one_create_without_replaying_delete(self):
        # Two additions during consecutive lock waits, both availability-valid.
        additions = [UUID(int=2), UUID(int=3)]
        async def add(_):
            if additions:
                self.sources.observe((additions.pop(0),))
        self.connection.on_lock = add
        update = await self.publish()
        self.assertTrue(self.runtime.ready)
        self.assertEqual(self.owner._bridge.status.last_receipt.status, "written")
        self.assertEqual(self.counts(), (2, 1))
        self.assertEqual(self.pool.entered - self.initial_acquisitions, 1)
        self.assertEqual(self.connection.locks[-1], (UUID(int=1), UUID(int=2), UUID(int=3)))
        self.assertEqual(self.connection.rejected, set())
        rows = self.rows("source_guard_prewrite_refresh")
        self.assertEqual([row["refresh_attempt"] for row in rows], [1, 2])
        self.assertTrue(all(row["update_id"] == str(update.update_id) and not row["guard_yielded"] for row in rows))
        self.assertEqual(self.rows("source_guard_rejected"), [])

    async def test_sustained_additions_exhaust_bounded_guard_without_provider_replay(self):
        count = 0
        async def add(_):
            nonlocal count
            count += 1
            self.sources.observe((UUID(int=100 + count),))
        self.connection.on_lock = add
        await self.publish()
        self.assertEqual(count, 3)
        self.assertFalse(self.runtime.ready)
        self.assertEqual(self.owner.status.reason, "context_receipt_unconfirmed")
        self.assertEqual(self.counts(), (1, 1))
        self.assertEqual(len(self.rows("source_guard_prewrite_refresh")), 2)
        self.assertEqual(self.rows("source_guard_rejected")[-1]["error_code"], "live_context_changed")

    async def test_actual_withdrawal_on_refreshed_lock_still_fences(self):
        count = 0
        async def mutate(_):
            nonlocal count
            count += 1
            if count == 1:
                self.sources.observe((UUID(int=2),))
            else:
                self.connection.rejected.add(UUID(int=1))
        self.connection.on_lock = mutate
        await self.publish()
        self.assertEqual(count, 2)
        self.assertFalse(self.runtime.ready)
        self.assertEqual(self.counts(), (1, 1))
        self.assertEqual(len(self.rows("source_guard_prewrite_refresh")), 1)
        self.assertEqual(self.rows("source_guard_rejected")[-1]["error_code"], "live_context_revoked")

    async def test_same_code_after_guard_yield_is_unknown_write_not_retryable(self):
        original = self.sources.context_guard
        @asynccontextmanager
        async def fail_after_yield(**kwargs):
            async with original(**kwargs):
                yield
                raise LiveContextError("live_context_changed")
        self.sources.context_guard = fail_after_yield
        await self.publish()
        self.assertFalse(self.runtime.ready)
        self.assertEqual(self.counts(), (2, 1))
        self.assertEqual(self.rows("source_guard_prewrite_refresh"), [])
        self.assertTrue(self.rows("source_guard_rejected")[-1]["guard_yielded"])

    async def test_unrelated_snapshot_revision_coalesces_without_source_invalidation(self):
        entered, release = asyncio.Event(), asyncio.Event()
        original = self.runtime._atomic_guard
        @asynccontextmanager
        async def blocked(payload):
            if payload.cursor.revision == 1:
                entered.set()
                await release.wait()
            async with original(payload) as connection:
                yield connection
        self.runtime._atomic_guard = blocked
        self.owner.offer_context(route_fixtures.make_context(revision=1))
        try:
            await asyncio.wait_for(entered.wait(), 0.5)
            latest = route_fixtures.make_context(revision=2)
            self.owner.offer_context(latest)
            release.set()
            async with asyncio.timeout(1):
                while (self.owner._bridge.status.last_receipt is None
                       or self.owner._bridge.status.last_receipt.cursor != latest.cursor):
                    await asyncio.sleep(0)
            self.assertTrue(self.runtime.ready)
            self.assertEqual(self.rows("source_guard_rejected"), [])
        finally:
            release.set()

    async def test_unknown_guard_error_is_redacted_and_never_retried(self):
        @asynccontextmanager
        async def private_error(_):
            raise ValueError("PRIVATE_SOURCE_CONTENT")
            yield
        self.runtime._atomic_guard = private_error
        await self.publish()
        self.assertFalse(self.runtime.ready)
        self.assertEqual(self.counts(), (1, 1))
        self.assertEqual(self.rows("source_guard_prewrite_refresh"), [])
        rows = self.rows("source_guard_rejected")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["error_code"], "unrecognized_code")
        self.assertNotIn("PRIVATE", repr(self.log.call_args_list))
