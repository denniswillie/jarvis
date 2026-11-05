"""Independent source-boundary regressions, with controlled storage and no I/O.

The accepted-output tests exercise the pinned-source contract, not provider
acceptance, playback, or SQL correction/permission predicates. Existing
test_native_context_addition_race and test_response_preparation_source_race
cover coordinator fencing and no replay after possible provider writes.
"""

import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace
import unittest
from unittest.mock import Mock
from uuid import UUID

from app.realtime.source_context import LiveContextError, LiveResponseSources
from app.realtime.voice_route_runtime import RealtimeVoiceRouteRuntime
from tests import test_native_context_addition_race as addition_fixtures
from tests import test_realtime_voice_route as route_fixtures
from tests import test_response_preparation_source_race as source_fixtures


class VoiceContextCausalBoundaryTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.original_root, self.candidate_root = UUID(int=10), UUID(int=20)
        self.connection = addition_fixtures.BorrowedConnection()
        self.pool = source_fixtures.Pool(self.connection)
        self.sources = LiveResponseSources(
            self.pool, owner_id="offline-causal-boundary", session_id=UUID(int=1),
            audio_session_id=str(UUID(int=2)),
        )
        self.sources.observe((self.original_root,))
        self.original = await self.sources.prepare("accepted-response")
        self.runtime = object.__new__(RealtimeVoiceRouteRuntime)
        self.runtime.session = SimpleNamespace(_source_context=self.sources)
        self.runtime._payload_roots = lambda _: (self.candidate_root,)
        self.runtime._readiness_diagnostic = Mock()
        self.candidate = route_fixtures.make_context(revision=1)

    async def asyncTearDown(self):
        self.assertEqual(self.pool.entered, self.pool.exited)
        self.assertEqual(self.connection.entered, self.connection.exited)

    async def assert_candidate_unexposed(self):
        self.assertIs(await self.sources.prepare("accepted-response"), self.original)
        following = await self.sources.prepare("following-response")
        self.assertEqual(following.sources, (self.original_root,))
        self.connection.rejected.add(self.candidate_root)
        # A later withdrawal of a never-exposed candidate cannot poison the
        # established context or force a new original response identity.
        async with self.sources.context_guard():
            pass
        async with self.sources.guard(self.original):
            pass
        self.assertEqual(self.sources.source_for("accepted-response"), self.original.payload_id)

    async def test_accepted_output_pins_original_roots_but_new_dispatch_stays_strict(self):
        pinned = self.sources.snapshot_for("accepted-response")
        self.assertIs(pinned, self.original)
        self.sources.observe((self.candidate_root,))

        async with self.sources.output_guard(pinned):
            self.assertEqual(
                self.connection.locks[-1], (self.original_root, self.original.payload_id)
            )
        self.assertIs(self.sources.snapshot_for("accepted-response"), pinned)
        self.assertEqual(pinned.sources, (self.original_root,))
        self.assertEqual(len(self.connection.payloads), 1)
        self.assertIsNone(self.sources.snapshot_for("unknown-request"))

        # The output-only exception must not authorize a fresh response.create
        # or action against the old version of a genuinely changed context.
        with self.assertRaises(LiveContextError) as caught:
            async with self.sources.guard(pinned):
                self.fail("A stale new-dispatch snapshot was authorized")
        self.assertEqual(caught.exception.code, "live_context_changed")

    async def test_accepted_output_still_fences_unavailable_original_or_payload(self):
        self.sources.observe((self.candidate_root,))
        # These are controlled canonical-availability failures, not three
        # separately executed SQL deletion/correction/permission workflows.
        for unavailable in (self.original_root, self.original.payload_id):
            with self.subTest(unavailable=unavailable):
                self.connection.rejected = {unavailable}
                with self.assertRaises(LiveContextError) as caught:
                    async with self.sources.output_guard(self.original):
                        self.fail("Unavailable accepted evidence was authorized")
                self.assertEqual(caught.exception.code, "live_context_revoked")
                self.assertEqual(
                    self.connection.locks[-1], (self.original_root, self.original.payload_id)
                )
                self.assertEqual(
                    self.sources.source_for("accepted-response"), self.original.payload_id
                )

    async def test_rejected_prewrite_candidate_never_becomes_exposed_dependency(self):
        @asynccontextmanager
        async def reject_before_write(_):
            async with self.pool.acquire() as connection, connection.transaction():
                raise LiveContextError("live_context_changed")
                yield connection

        self.runtime._atomic_guard = reject_before_write
        with self.assertRaises(LiveContextError) as caught:
            async with self.runtime._source_guard(self.candidate):
                self.fail("The rejected preparation reached its writer")
        self.assertEqual(caught.exception.code, "live_context_changed")
        await self.assert_candidate_unexposed()

    async def test_cancelled_joined_preparation_never_becomes_exposed_dependency(self):
        entered, joined, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
        writer_reached = False

        @asynccontextmanager
        async def wait_before_write(_):
            try:
                async with self.pool.acquire() as connection, connection.transaction():
                    entered.set()
                    await release.wait()
                    yield connection
            finally:
                joined.set()

        self.runtime._atomic_guard = wait_before_write

        async def prepare():
            nonlocal writer_reached
            async with self.runtime._source_guard(self.candidate):
                writer_reached = True

        owned = asyncio.create_task(prepare())
        try:
            await asyncio.wait_for(entered.wait(), timeout=0.5)
            owned.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await asyncio.wait_for(owned, timeout=0.5)
            self.assertTrue(owned.done())
            self.assertTrue(joined.is_set())
            self.assertFalse(writer_reached)
            await self.assert_candidate_unexposed()
        finally:
            release.set()
            if not owned.done():
                owned.cancel()
            await asyncio.gather(owned, return_exceptions=True)

    async def test_possible_handoff_error_retains_candidate_without_reentering_writer(self):
        @asynccontextmanager
        async def valid(_):
            async with self.pool.acquire() as connection, connection.transaction():
                yield connection

        self.runtime._atomic_guard = valid
        writer_entries = 0
        uncertain = RuntimeError("controlled unknown handoff outcome")
        with self.assertRaises(RuntimeError) as caught:
            async with self.runtime._source_guard(self.candidate):
                writer_entries += 1
                self.runtime._source_handoff(self.candidate)
                # Exposure is possible, not an assertion of wire submission,
                # provider acceptance, or hearing. It cannot be rolled back.
                raise uncertain
        self.assertIs(caught.exception, uncertain)
        self.assertEqual(writer_entries, 1)
        following = await self.sources.prepare("following-response")
        self.assertEqual(following.sources, (self.original_root, self.candidate_root))
        self.connection.rejected.add(self.candidate_root)
        with self.assertRaises(LiveContextError) as revoked:
            async with self.sources.context_guard():
                self.fail("Possibly exposed, withdrawn evidence was forgotten")
        self.assertEqual(revoked.exception.code, "live_context_revoked")
