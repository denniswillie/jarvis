"""Controlled 1/2/3 interruption bursts, not ASR, native media or voice E2E.

Actual session handlers, enrichment, canonical guards and provider writer run.
Source mutations are explicit canonical-fixture changes, NOT production owner
tool acceptance; the isolated SQL/native lifecycle separately exercises that.
"""
import asyncio
from contextlib import asynccontextmanager
import unittest
from uuid import uuid4

from app.live_state.canonical import context_dependency, memory_dependency
from app.tasks.models import digest
from responsive_joined_enrichment_fixtures import A_MARKER, B_MARKER, JoinedEnrichmentFixture


class MemoryInterruptionBurstTests(unittest.IsolatedAsyncioTestCase):
    @asynccontextmanager
    async def fixture(self):
        fixture = JoinedEnrichmentFixture(self)
        try:
            await fixture.setup()
            yield fixture
        finally:
            await fixture.close()

    def replace_source(self, fixture, revision):
        old = fixture.a
        content = {"text": f"CURRENT_A_REVISION_{revision}: exact value {revision}.25"}
        current = old.model_copy(update={"revision": revision, "generation": revision,
            "payload_id": uuid4(), "document_id": uuid4().hex,
            "content": content, "content_digest": digest(content)})
        fixture.db.originals[current.source_id] = current
        fixture.db.copies[current.payload_id] = {"owner": current.owner_id, "content": content, "parents": set()}
        fixture.a = current
        return current

    async def test_speech_cutoff_bursts_do_not_cancel_pending_memory_work(self):
        for count in (1, 2, 3):
            with self.subTest(cutoffs=count):
                async with self.fixture() as f:
                    originals = dict(f.db.originals)
                    for index in range(count):
                        await f.ask(f"controlled-cutoff-{index}")
                        await f.until(lambda: bool(f.session._active_responses))
                        await f.session._handle_client_event({"type": "response.cancel"})
                        await f.until(lambda: not f.session._active_responses)
                        self.assertFalse(f.release_a.is_set())
                        self.assertEqual(f.active, 1)
                        self.assertFalse(f.live.enrichment._task.done())
                    f.release_a.set()
                    await f.until(lambda: A_MARKER in f.voice_text())
                    self.assertEqual(len(f.calls), 1)
                    self.assertEqual(f.db.originals, originals)
                    self.assertTrue(f.db.available(f.scope.owner_id, f.b.payload_id))

    async def test_latest_of_multiple_corrections_wins_before_old_recall_returns(self):
        for count in (1, 2, 3):
            with self.subTest(replacements=count):
                async with self.fixture() as f:
                    independent = f.b
                    for index in range(count):
                        current = self.replace_source(f, index + 2)
                        request = f.intent(f"Help with experiment A correction {index + 2}")
                        await f.until(lambda: f.live.enrichment._wanted is not None
                            and f"correction {index + 2}" in f.live.enrichment._wanted.query)
                    self.assertEqual(len(f.calls), 1)
                    f.release_a.set()
                    await f.until(lambda: current.content["text"] in f.voice_text())
                    entries = [entry for entry in f.live.service.snapshot().entries if entry.kind == "selected_memory"]
                    self.assertEqual(len(entries), 1)
                    self.assertIn(memory_dependency(current), entries[0].sources)
                    self.assertIn(context_dependency(request), entries[0].sources)
                    self.assertTrue(await f.live.binder.available(entries[0].guard_binding))
                    self.assertNotIn(A_MARKER, f.voice_text())
                    self.assertEqual(len(f.calls), 2)
                    self.assertEqual(f.live.enrichment.stale_results, 1)
                    self.assertEqual(len(f.db.originals), 2)
                    self.assertEqual(f.db.originals[independent.source_id], independent)

    async def test_deletion_and_replacement_bursts_during_recall_never_resurrect(self):
        for count in (1, 2, 3):
            with self.subTest(replacements=count):
                async with self.fixture() as f:
                    deleted = f.a
                    f.db.copies[deleted.payload_id]["content"] = None
                    f.db.originals.pop(deleted.source_id)
                    for index in range(count):
                        request = f.intent(f"Help with experiment B after deletion, request {index}")
                        await f.until(lambda: f.live.enrichment._wanted is not None
                            and f"request {index}" in f.live.enrichment._wanted.query)
                    f.release_a.set()
                    await f.until(lambda: B_MARKER in f.voice_text())
                    entry, = [item for item in f.live.service.snapshot().entries if item.kind == "selected_memory"]
                    self.assertIn(memory_dependency(f.b), entry.sources)
                    self.assertIn(context_dependency(request), entry.sources)
                    self.assertNotIn(A_MARKER, f.voice_text())
                    self.assertNotIn(deleted.source_id, f.db.originals)
                    self.assertEqual(len(f.db.originals), 1)
                    self.assertEqual(len(f.calls), 2)
                    self.assertEqual(f.max_active, 1)

    async def test_cutoff_bursts_then_deletion_while_context_publication_is_pending(self):
        for count in (1, 2, 3):
            with self.subTest(cutoffs=count):
                async with self.fixture() as f:
                    entered, release = asyncio.Event(), asyncio.Event()
                    publish = f.route.adapter.publish_context

                    async def pending(update):
                        if A_MARKER in update.text and not entered.is_set():
                            entered.set()
                            await release.wait()
                        return await publish(update)
                    f.route.adapter.publish_context = pending
                    try:
                        await f.ask("controlled-output-before-publication")
                        await f.until(lambda: bool(f.session._active_responses))
                        f.release_a.set()
                        await asyncio.wait_for(entered.wait(), 1)
                        for _ in range(count):
                            await f.session._handle_client_event({"type": "response.cancel"})
                        await f.until(lambda: not f.session._active_responses)
                        self.assertFalse(f.live.enrichment._task.done())
                        self.assertFalse(release.is_set())
                        f.db.copies[f.a.payload_id]["content"] = None
                        f.db.originals.pop(f.a.source_id)
                        request = f.intent("Help with experiment B after pending-context deletion")
                        await f.until(lambda: any(item.kind == "selected_memory"
                            and memory_dependency(f.b) in item.sources for item in f.live.service.snapshot().entries))
                        release.set()
                        await f.until(lambda: not f.route.ready)
                        self.assertNotIn(A_MARKER, f.voice_text())
                        entry, = [item for item in f.live.service.snapshot().entries if item.kind == "selected_memory"]
                        self.assertIn(memory_dependency(f.b), entry.sources)
                        self.assertIn(context_dependency(request), entry.sources)
                        self.assertTrue(await f.live.binder.available(entry.guard_binding))
                        self.assertEqual(len(f.db.originals), 1)
                        self.assertEqual(len(f.calls), 2)
                        # Revoked pending publication requires route reconstruction.
                        # Do not invent successful delivery of B on the fenced route.
                    finally:
                        release.set()
