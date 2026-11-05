"""Real canonical guards with controlled storage/barriers, not native acceptance."""
import asyncio
import unittest
from unittest.mock import AsyncMock, patch

from app.live_state.canonical import context_dependency, memory_dependency
from app.live_state.historical_enrichment import HistoricalEnrichment
from app.live_state.projection import select_voice_context
import test_live_state_historical_enrichment as fixtures


class HistoricalPublicationStageTests(unittest.IsolatedAsyncioTestCase):
    async def fixture(self):
        f = fixtures.HistoricalEnrichmentCompositionTests(methodName="runTest")
        await f.asyncSetUp()
        self.addAsyncCleanup(f.asyncTearDown)
        return f

    def candidates(self, f, sources):
        f.retrieval.recall.side_effect = None
        f.retrieval.recall.return_value = {"status": "available", "sources": [
            source.original_evidence() | {"content_included": True} for source in sources]}

    async def test_four_sources_publish_once_despite_many_same_context_refreshes(self):
        f = await self.fixture()
        sources = [f.new_source(f"Independent original {index}: {index}.25 units") for index in range(4)]
        self.candidates(f, sources)
        entered, release = asyncio.Event(), asyncio.Event()
        original = f.f.sources.revalidate

        async def held(source):
            if not entered.is_set():
                entered.set()
                await release.wait()
            return await original(source)

        f.f.sources.revalidate = AsyncMock(side_effect=held)
        live = await f.start(timeout_seconds=.03, publication_deadline_seconds=1)
        self.addCleanup(release.set)
        await asyncio.wait_for(entered.wait(), 1)
        bind = live.binder.bind_memory
        live.binder.bind_memory = AsyncMock(wraps=bind)
        observed = live.enrichment.worker.refresh_count
        first_context = live.enrichment._wanted
        await f.until(lambda: live.enrichment.worker.refresh_count >= observed + 5)
        self.assertIsNot(live.enrichment._wanted, first_context)
        self.assertEqual(live.enrichment._wanted.fingerprint, first_context.fingerprint)
        self.assertEqual(live.enrichment.worker.failure_count, 0)
        self.assertFalse(f.memories())
        self.assertEqual(live.service.snapshot().readiness, "ready")
        release.set()
        await f.until(lambda: len(f.memories()) == 4 and live.enrichment.status == "available")
        for source in sources:
            entry, = [item for item in f.memories() if memory_dependency(source) in item.sources]
            self.assertEqual(entry.observed_at, source.source_time)
            self.assertTrue(await live.binder.available(entry.guard_binding))
        observed = live.enrichment.worker.refresh_count
        await f.until(lambda: live.enrichment.worker.refresh_count >= observed + 3)
        self.assertEqual(live.binder.bind_memory.await_count, 4)
        self.assertEqual(f.retrieval.recall.await_count, 1)
        self.assertEqual(live.enrichment.publication_failures, 0)
        self.assertEqual(live.enrichment.worker.failure_count, 0)

    async def test_new_intent_wins_while_old_publication_is_blocked(self):
        f = await self.fixture()
        a = f.response_source = f.new_source("Old A is 41.5")
        f.intent("Inspect A")
        entered, release = asyncio.Event(), asyncio.Event()
        original = f.f.sources.revalidate

        async def held(source):
            if source.source_id == a.source_id and not entered.is_set():
                entered.set()
                await release.wait()
            return await original(source)

        f.f.sources.revalidate = AsyncMock(side_effect=held)
        live = await f.start(publication_deadline_seconds=1)
        self.addCleanup(release.set)
        await asyncio.wait_for(entered.wait(), 1)
        b = f.response_source = f.new_source("Current B is 12.75")
        request = f.intent("Inspect B")
        await f.until(lambda: len(f.calls) == 2 and live.enrichment._selected == (b,))
        self.assertFalse(f.memories())
        release.set()
        await f.until(lambda: bool(f.memories()) and live.enrichment.status == "available")
        entry, = f.memories()
        self.assertIn(memory_dependency(b), entry.sources)
        self.assertIn(context_dependency(request), entry.sources)
        self.assertNotIn(memory_dependency(a), entry.sources)
        self.assertTrue(await live.binder.available(entry.guard_binding))
        self.assertEqual(live.enrichment.publication_failures, 0)

    async def test_deletion_after_first_binding_is_caught_before_batch_publication(self):
        f = await self.fixture()
        deleted, independent = f.new_source("Deleted original"), f.new_source("Independent original 19.25")
        self.candidates(f, [deleted, independent])
        entered, release = asyncio.Event(), asyncio.Event()
        original = f.f.sources.revalidate

        async def held(source):
            if source.source_id == independent.source_id and not entered.is_set():
                entered.set()
                await release.wait()
            return await original(source)

        f.f.sources.revalidate = AsyncMock(side_effect=held)
        live = await f.start(publication_deadline_seconds=1)
        self.addCleanup(release.set)
        await asyncio.wait_for(entered.wait(), 1)
        f.f.db.remove(deleted.payload_id)
        f.f.originals.pop(deleted.source_id)
        release.set()
        await f.until(lambda: bool(f.memories()) and live.enrichment.status == "available")
        entry, = f.memories()
        self.assertIn(memory_dependency(independent), entry.sources)
        self.assertNotIn(memory_dependency(deleted), entry.sources)
        self.assertTrue(await live.binder.available(entry.guard_binding))
        self.assertEqual(f.retrieval.recall.await_count, 1)
        self.assertEqual(live.enrichment.worker.failure_count, 0)
        self.assertGreaterEqual(live.enrichment.publication_failures, 1)

    async def test_publication_timeout_has_its_own_bound_and_keeps_refresh_alive(self):
        f = await self.fixture()
        entered, release = asyncio.Event(), asyncio.Event()

        async def held(source):
            entered.set()
            await release.wait()
            return source

        f.f.sources.revalidate = AsyncMock(side_effect=held)
        live = await f.start(timeout_seconds=.03, publication_deadline_seconds=.04)
        await asyncio.wait_for(entered.wait(), 1)
        await f.until(lambda: live.enrichment.publication_failures > 0)
        self.assertEqual(live.enrichment.last_error, "historical_publication_timeout")
        self.assertEqual(live.enrichment.worker.failure_count, 0)
        self.assertFalse(f.memories())
        self.assertEqual(live.service.snapshot().readiness, "ready")
        self.assertEqual(f.retrieval.recall.await_count, 1)
        await live.close()
        self.assertTrue(live.enrichment._publication_task.done())
        self.assertTrue(live.enrichment._task.done())
        self.assertTrue(live.enrichment.worker._task.done())

    async def test_close_cancels_and_joins_inflight_publication_without_releasing_barrier(self):
        f = await self.fixture()
        entered, cancelled, release = asyncio.Event(), asyncio.Event(), asyncio.Event()

        async def held(source):
            entered.set()
            try:
                await release.wait()
            finally:
                cancelled.set()
            return source

        f.f.sources.revalidate = AsyncMock(side_effect=held)
        live = await f.start()
        await asyncio.wait_for(entered.wait(), 1)
        await asyncio.wait_for(live.close(), 1)
        self.assertTrue(cancelled.is_set())
        self.assertFalse(release.is_set())
        self.assertTrue(live.enrichment._publication_task.done())
        self.assertTrue(live.enrichment._task.done())
        self.assertFalse(live.service.snapshot().entries)

    async def test_epoch_retirement_cannot_publish_held_old_selection(self):
        f = await self.fixture()
        marker = "RETIRED_EPOCH_ORIGINAL_41_5"
        f.response_source = f.new_source(marker)
        entered, release, revalidated, completed = (asyncio.Event() for _ in range(4))
        outcomes, guards = [], []
        original = f.f.sources.revalidate

        async def held(source):
            entered.set()
            await release.wait()
            result = await original(source)
            revalidated.set()
            return result

        def create(**kwargs):
            enrichment = HistoricalEnrichment(**kwargs)
            publish, current = enrichment._publish, enrichment._publication_current

            def observed_guard(context, generation):
                accepted = current(context, generation)
                guards.append((enrichment.service.snapshot().cursor.epoch, accepted))
                return accepted

            async def observed_publication():
                try:
                    outcome = await publish()
                    outcomes.append(outcome)
                    return outcome
                finally:
                    completed.set()

            enrichment._publication_current = observed_guard
            enrichment._publish = observed_publication
            return enrichment

        f.f.sources.revalidate = AsyncMock(side_effect=held)
        with patch("app.live_state.runtime.HistoricalEnrichment", side_effect=create):
            live = await f.start()
        self.addCleanup(release.set)
        await asyncio.wait_for(entered.wait(), 1)
        subscription = live.service.subscribe(capacity=32)
        snapshots = []

        async def collect():
            while (notice := await subscription.receive()) is not None:
                snapshots.append(notice.snapshot)

        collector = asyncio.create_task(collect())

        async def stop_collector():
            subscription.close()
            await collector

        self.addAsyncCleanup(stop_collector)
        old_epoch = live.service.snapshot().cursor.epoch
        live.service.reset_for_reconciliation()
        new_epoch = live.service.snapshot().cursor.epoch
        release.set()
        await asyncio.wait_for(completed.wait(), 1)
        await live.service.wait_idle()
        await asyncio.sleep(0)
        # This is an actual rejected completion, not cancellation by teardown.
        self.assertTrue(revalidated.is_set())
        self.assertEqual(outcomes, [False])
        self.assertIn((new_epoch, False), guards)
        self.assertFalse(live._closing)
        self.assertFalse(live.enrichment._closed)
        self.assertNotEqual(new_epoch, old_epoch)
        self.assertTrue(snapshots)
        self.assertTrue(any(snapshot.cursor.epoch == new_epoch for snapshot in snapshots))
        for snapshot in snapshots:
            self.assertFalse(any(entry.kind == "selected_memory" for entry in snapshot.entries))
            self.assertNotIn(marker, select_voice_context(snapshot, now=f.f.now).text)
        self.assertFalse(f.memories())
        self.assertEqual(live.enrichment.publication_completed, 0)
        await asyncio.wait_for(live.close(), 1)
        await collector
        self.assertTrue(live.enrichment._publication_task.done())
        self.assertTrue(live.enrichment._task.done())
        self.assertTrue(live.enrichment.worker._task.done())
