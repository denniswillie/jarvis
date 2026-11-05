"""Actual SQL reader overlap, exclusive mutation fences and correction rejection."""
import asyncio
from datetime import UTC, datetime
import os
import unittest
from uuid import uuid4

from app.attention.quiet_delivery import quiet_response_guard
from app.knowledge.models import CanonicalOwnerBusy, SourceError
from app.knowledge.sources import SourceRepository
from app.live_state.canonical import CanonicalContextBinder
from app.live_state.models import SessionScope
from app.realtime.source_context import LiveResponseSources
from app.tasks.models import TaskCreate, TaskVoiceSource
import test_second_brain_context_integration as fixture


@unittest.skipUnless(os.getenv("JARVIS_SECOND_BRAIN_TEST_DB_URL")
                     and os.getenv("JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"),
                     "Explicit manifested SQL fixture required")
class CanonicalSharedGuardSQLTests(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = fixture.CanonicalContextDatabaseTests.asyncSetUp
    ingress = fixture.CanonicalContextDatabaseTests.ingress

    async def test_shared_guards_overlap_but_mutations_and_old_sources_remain_fenced(self):
        sources = SourceRepository(self.pool)
        binder = CanonicalContextBinder(sources,
            SessionScope(owner_id=self.owner, session_id=self.session), clock=lambda: datetime.now(UTC))
        record = await self.context.append(self.owner, self.ingress(finality="final"))
        binding = await binder.bind_record(record, "Synthetic source-backed context")
        # Warm both fixture leases; connection establishment is not a guard test.
        async with self.pool.acquire(), self.pool.acquire():
            pass

        async with binder.guard(binding):
            async with binder.guard(binding, wait_for_owner=False):
                pass  # Same owner, same roots, separate actual connections.
        async with binder.guard(binding):
            with self.assertRaises(CanonicalOwnerBusy):
                async with sources.transaction(self.owner, wait_for_owner=False):
                    self.fail("Exclusive publication entered during a read guard")
        async with binder.guard(binding):
            async with self.tasks.transaction() as connection:
                self.assertFalse(await self.tasks.try_lock_owner(connection, self.owner))
        async with sources.transaction(self.owner):
            with self.assertRaises(CanonicalOwnerBusy):
                async with binder._transaction(wait_for_owner=False, read_only_guard=True):
                    self.fail("Reader entered an exclusive source mutation")
        async with self.tasks.transaction(self.owner):
            with self.assertRaises(CanonicalOwnerBusy):
                async with binder._transaction(wait_for_owner=False, read_only_guard=True):
                    self.fail("Reader entered an exclusive task mutation")
            # Failed task-lock acquisition must release its earlier memory lock.
            async with sources.transaction(self.owner, wait_for_owner=False):
                pass
        async with sources.read_guard_transaction(self.owner):
            async with sources.transaction(self.other, wait_for_owner=False):
                pass

        # A borrowed quiet check must not upgrade its shared task-owner lock.
        left = LiveResponseSources(self.pool, owner_id=self.owner, session_id=self.session, audio_session_id="left")
        right = LiveResponseSources(self.pool, owner_id=self.owner, session_id=self.session, audio_session_id="right")
        async with binder._transaction(read_only_guard=True) as first:
            async with left.use_transaction(first):
                async with quiet_response_guard(left, origin="visual_interrupt"):
                    async with self.tasks.transaction() as second:
                        self.assertFalse(await self.tasks.try_lock_owner(second, self.owner))
                        await self.tasks.lock_owner_shared(second, self.owner)
                        async with right.use_transaction(second):
                            async with quiet_response_guard(right, origin="visual_interrupt"):
                                pass

        task = await self.tasks.create(self.owner, TaskCreate(request_id=uuid4(), session_id=self.session,
            title="Synthetic task", requested_outcome="Read-only context fixture"), model="fixture",
            source=TaskVoiceSource(record.id, record.revision, self.session), mode="background")
        # Multiple task-backed readers must not take opposing exclusive row locks.
        async with self.tasks.transaction() as first:
            await self.tasks.live_context_in_transaction(first, self.owner, task.id)
            async with self.tasks.transaction() as second:
                async with asyncio.timeout(2):
                    other, _ = await self.tasks.live_context_in_transaction(second, self.owner, task.id)
                self.assertEqual(other.id, task.id)

        corrected = await self.context.append(self.owner,
            self.ingress(finality="revised", expected_revision=record.revision, text="Corrected source"))
        self.assertGreater(corrected.revision, record.revision)
        with self.assertRaises(SourceError):
            async with binder.guard(binding):
                self.fail("Shared mode authorized an obsolete canonical binding")
