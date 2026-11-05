"""Bounded ranked reserves with real canonical guards and controlled storage."""
import asyncio
import json
import unittest
from unittest.mock import AsyncMock
from uuid import uuid4

from app.live_state.canonical import memory_dependency
from app.live_state.historical_enrichment import CANDIDATE_LIMIT, MEMORY_TEXT_LIMIT
from app.live_state.projection import select_voice_context
import test_live_state_historical_enrichment as fixtures


class HistoricalCandidateAdmissionTests(unittest.IsolatedAsyncioTestCase):
    async def fixture(self):
        f = fixtures.HistoricalEnrichmentCompositionTests(methodName="runTest")
        await f.asyncSetUp()
        self.addAsyncCleanup(f.asyncTearDown)
        return f

    def candidates(self, f, sources):
        f.retrieval.recall.side_effect = None
        f.retrieval.recall.return_value = {"status": "available", "sources": [
            source.original_evidence() | {"content_included": True} for source in sources]}

    def selected_ids(self, f):
        return {source.source_id for entry in f.memories() for source in entry.sources
                if source.source_namespace == "knowledge_source"}

    async def test_oversized_second_candidate_does_not_displace_valid_fifth(self):
        f = await self.fixture()
        sources = [f.new_source(f"Useful original {index}: {index}.25 units") for index in range(4)]
        oversized = f.new_source("PRIVATE_OVERSIZED_MEMORY_" + "x" * 1100)
        self.candidates(f, [sources[0], oversized, *sources[1:]])
        live = await f.start()
        await f.until(lambda: len(f.memories()) == 4)
        self.assertEqual(self.selected_ids(f), {source.source_id for source in sources})
        self.assertEqual(len(live.enrichment._candidates), 5)
        self.assertEqual(len(live.enrichment._selected), 4)
        for source in sources:
            entry, = [entry for entry in f.memories() if memory_dependency(source) in entry.sources]
            self.assertEqual(entry.observed_at, source.source_time)
            self.assertLessEqual(len(entry.summary), MEMORY_TEXT_LIMIT)
            self.assertTrue(await live.binder.available(entry.guard_binding))
        selection = select_voice_context(live.service.snapshot(), now=f.f.now)
        for source in sources:
            self.assertIn(memory_dependency(source), selection.dependencies)
        binding = await live.bind(selection)
        async with live.guard(binding):
            pass
        diagnostics = live.enrichment.candidate_diagnostics()
        self.assertTrue(any(item["source_id"] == str(oversized.source_id)
            and item["reason"] == "text_budget_exceeded" for item in diagnostics["recent_omissions"]))
        self.assertNotIn("PRIVATE_OVERSIZED_MEMORY", json.dumps(diagnostics))
        self.assertEqual(f.retrieval.recall.await_count, 1)

    async def test_first_four_publishable_sources_preserve_rank_and_eight_candidate_bound(self):
        f = await self.fixture()
        sources = [f.new_source(f"Ranked original {index}") for index in range(9)]
        self.candidates(f, sources)
        live = await f.start()
        await f.until(lambda: len(f.memories()) == 4)
        self.assertEqual(live.enrichment._candidates, tuple(sources[:CANDIDATE_LIMIT]))
        self.assertEqual(live.enrichment._selected, tuple(sources[:4]))
        self.assertEqual(self.selected_ids(f), {source.source_id for source in sources[:4]})
        documents = [call.args[2] for call in f.f.sources.resolve_document.await_args_list]
        self.assertEqual(documents, [source.document_id for source in sources[:8]])
        self.assertNotIn(sources[8].document_id, documents)
        self.assertEqual(f.retrieval.recall.await_count, 1)
        self.assertLessEqual(len(live.enrichment._selected), live.enrichment.max_sources)

    async def test_deleted_candidate_during_first_publication_backfills_without_another_lookup(self):
        f = await self.fixture()
        sources = [f.new_source(f"Original {index}") for index in range(5)]
        self.candidates(f, sources)
        entered, release = asyncio.Event(), asyncio.Event()
        revalidate = f.f.sources.revalidate

        async def held(source):
            if source.source_id == sources[0].source_id and not entered.is_set():
                entered.set()
                await release.wait()
            return await revalidate(source)

        f.f.sources.revalidate = AsyncMock(side_effect=held)
        live = await f.start()
        self.addCleanup(release.set)
        await asyncio.wait_for(entered.wait(), 1)
        f.f.db.remove(sources[0].payload_id)
        release.set()
        await f.until(lambda: self.selected_ids(f) == {source.source_id for source in sources[1:]})
        self.assertEqual(f.retrieval.recall.await_count, 1)
        self.assertEqual(live.enrichment.publication_failures, 0)
        self.assertNotIn(memory_dependency(sources[0]),
                         select_voice_context(live.service.snapshot(), now=f.f.now).dependencies)

    async def assert_later_invalidation_backfills(self, *, correction):
        f = await self.fixture()
        sources = [f.new_source(f"Independent source {index}") for index in range(5)]
        self.candidates(f, sources)
        live = await f.start()
        await f.until(lambda: len(f.memories()) == 4)
        old_entry, = [entry for entry in f.memories() if memory_dependency(sources[0]) in entry.sources]
        if correction:
            f.f.originals[sources[0].source_id] = sources[0].model_copy(update={"generation": 2})
        else:
            f.f.db.remove(sources[0].payload_id)
        self.assertFalse(await live.binder.available(old_entry.guard_binding))
        await f.until(lambda: self.selected_ids(f) == {source.source_id for source in sources[1:]})
        self.assertEqual(live.enrichment._selected, tuple(sources[1:]))
        self.assertEqual(f.retrieval.recall.await_count, 1)
        for entry in f.memories():
            self.assertTrue(await live.binder.available(entry.guard_binding))
        diagnostics = live.enrichment.candidate_diagnostics()
        self.assertTrue(any(item["source_id"] == str(sources[0].source_id)
            and item["reason"] == "canonical_source_or_binding_unavailable"
            for item in diagnostics["recent_omissions"]))

    async def test_deleted_published_source_does_not_resurrect_from_reserve(self):
        await self.assert_later_invalidation_backfills(correction=False)

    async def test_changed_source_generation_does_not_reuse_old_selected_revision(self):
        await self.assert_later_invalidation_backfills(correction=True)

    async def test_replacement_intent_discards_old_reserve_while_publication_is_held(self):
        f = await self.fixture()
        old_sources = [f.new_source(f"Obsolete A {index}") for index in range(5)]
        current = f.new_source("Current B is 12.75")
        self.candidates(f, old_sources)
        f.intent("Inspect A")
        entered, release = asyncio.Event(), asyncio.Event()
        revalidate = f.f.sources.revalidate

        async def held(source):
            if source.source_id == old_sources[0].source_id and not entered.is_set():
                entered.set()
                await release.wait()
            return await revalidate(source)

        f.f.sources.revalidate = AsyncMock(side_effect=held)
        live = await f.start()
        self.addCleanup(release.set)
        await asyncio.wait_for(entered.wait(), 1)
        self.candidates(f, [current])
        f.intent("Inspect B")
        await f.until(lambda: live.enrichment._candidates == (current,))
        release.set()
        await f.until(lambda: self.selected_ids(f) == {current.source_id})
        self.assertEqual(live.enrichment._selected, (current,))
        self.assertEqual(f.retrieval.recall.await_count, 2)
        self.assertNotIn("Obsolete A", select_voice_context(live.service.snapshot(), now=f.f.now).text)

    async def test_reasons_are_bounded_and_do_not_retain_untrusted_ids_or_content(self):
        f = await self.fixture()
        first, later = f.new_source("First valid fact"), f.new_source("Later valid fact")
        evidence = first.original_evidence() | {"content_included": True}
        f.retrieval.recall.side_effect = None
        f.retrieval.recall.return_value = {"status": "available", "sources": [
            {"content_included": False, "source_id": "PRIVATE_RAW_MEMORY_TEXT"},
            evidence, evidence, evidence | {"content_digest": "forged"},
            later.original_evidence() | {"content_included": True}]}
        live = await f.start()
        await f.until(lambda: len(f.memories()) == 2)
        reasons = {item["reason"] for item in live.enrichment.candidate_diagnostics()["recent_omissions"]}
        self.assertTrue({"content_not_included", "duplicate_source", "source_version_mismatch"} <= reasons)
        for index in range(40):
            live.enrichment._omit_candidate("content_not_included", phase="lookup", rank=1,
                candidate={"source_id": str(uuid4()), "revision": index + 1})
        diagnostics = live.enrichment.candidate_diagnostics()
        self.assertEqual(len(diagnostics["recent_omissions"]), 32)
        self.assertFalse(diagnostics["raw_content_included"])
        self.assertNotIn("PRIVATE_RAW_MEMORY_TEXT", json.dumps(diagnostics))
        self.assertNotIn("First valid fact", json.dumps(diagnostics))
        self.assertEqual(self.selected_ids(f), {first.source_id, later.source_id})
