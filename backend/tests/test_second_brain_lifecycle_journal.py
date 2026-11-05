"""Source recovery metadata, using the maintained manifested source fixture."""
import asyncio
from contextlib import AsyncExitStack
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from uuid import uuid4

import asyncpg

from app.knowledge.lifecycle_journal import JournalCheckpoint, SourceLifecycleJournal
from app.knowledge.models import SourceError
from app.knowledge.outbox import OutboxRepository
from evals.second_brain.source_fixture import source_fixture


class ExportJournal(SourceLifecycleJournal):
    def __init__(self, *, fail_after=None):
        super().__init__(None)
        self.pin = JournalCheckpoint(owner_id="fixture", epoch=uuid4(), through_sequence=7)
        self.fail_after = fail_after

    async def checkpoint(self, owner_id):
        return self.pin

    async def page(self, owner_id, checkpoint, *, after_sequence=0, limit=256):
        if self.fail_after is not None and after_sequence >= self.fail_after:
            raise SourceError("source_journal_gap", "Injected fixture gap")
        cursor = min(checkpoint.through_sequence, after_sequence + min(limit, 2))
        events = [{"sequence": value, "detail": {"status": "deleted"}} for value in range(after_sequence + 1, cursor + 1)]
        return {"events": events, "next_after_sequence": cursor, "complete": cursor == checkpoint.through_sequence}


class JournalExportTests(unittest.IsolatedAsyncioTestCase):
    async def test_export_segments_are_private_bounded_checksummed_and_not_restore_authority(self):
        journal = ExportJournal()
        with tempfile.TemporaryDirectory(prefix="jarvis-journal-unit-") as temporary:
            root = Path(temporary) / "first"
            first = await journal.export("fixture", root, max_events=3)
            self.assertEqual((first["event_count"], first["next_after_sequence"], first["complete"]), (3, 3, False))
            self.assertEqual(first["status"], "sealed")
            self.assertFalse(first["restore_authorized"])
            raw = (root / "events.ndjson").read_bytes()
            self.assertEqual(first["sha256"], hashlib.sha256(raw).hexdigest())
            self.assertEqual([json.loads(line)["sequence"] for line in raw.splitlines()], [1, 2, 3])
            self.assertEqual(root.stat().st_mode & 0o777, 0o700)
            self.assertEqual((root / "events.ndjson").stat().st_mode & 0o777, 0o600)
            second = await journal.export("fixture", Path(temporary) / "second",
                                           checkpoint=journal.pin, after_sequence=3, max_events=4)
            self.assertEqual((second["event_count"], second["next_after_sequence"], second["complete"]), (4, 7, True))
            self.assertEqual(second["checkpoint"], first["checkpoint"])
            with self.assertRaises(FileExistsError):
                await journal.export("fixture", root)
            self.assertEqual((root / "events.ndjson").read_bytes(), raw)

    async def test_failed_export_cannot_be_mistaken_for_a_sealed_segment(self):
        journal = ExportJournal(fail_after=2)
        with tempfile.TemporaryDirectory(prefix="jarvis-journal-unit-") as temporary:
            root = Path(temporary) / "failed"
            with self.assertRaises(SourceError):
                await journal.export("fixture", root)
            manifest = json.loads((root / "manifest.json").read_text())
            self.assertEqual(manifest["status"], "incomplete")
            self.assertNotIn("sha256", manifest)
            self.assertFalse(manifest["restore_authorized"])

    async def test_export_rejects_scope_and_bounds_before_allocating_a_directory(self):
        journal = ExportJournal()
        with tempfile.TemporaryDirectory(prefix="jarvis-journal-unit-") as temporary:
            root = Path(temporary) / "invalid"
            for kwargs in ({"max_events": 0}, {"max_events": True}, {"max_events": 100001},
                           {"after_sequence": -1}, {"after_sequence": 8}):
                with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                    await journal.export("fixture", root, **kwargs)
            with self.assertRaises(ValueError):
                await journal.export("other", root)
            self.assertFalse(root.exists())


@unittest.skipUnless(os.getenv("JARVIS_SECOND_BRAIN_TEST_DB_URL") and
                     os.getenv("JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"),
                     "Explicit second-brain database fixtures required")
class SourceJournalDatabaseTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.stack = AsyncExitStack()
        self.addAsyncCleanup(self.stack.aclose)
        self.fixture = await self.stack.enter_async_context(source_fixture(
            os.environ["JARVIS_SECOND_BRAIN_TEST_DB_URL"],
            Path(os.environ["JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"]), test_name=self.id()))
        self.journal = SourceLifecycleJournal(self.fixture.pool)
        self.owner = self.fixture.owner

    async def events(self, checkpoint=None):
        checkpoint = checkpoint or await self.journal.checkpoint(self.owner)
        return (await self.journal.page(self.owner, checkpoint, limit=512))["events"]

    async def test_correction_and_deletion_are_transactional_metadata_without_original_bodies(self):
        source = await self.fixture.publish("Private original body", source_key="private-repository-path")
        await self.fixture.publish("Private corrected body", source_key="private-repository-path", expected_revision=1)
        await self.fixture.sources.delete_source(self.owner, self.fixture.area, source.source_id, expected_revision=2)
        rows = await self.events()
        source_rows = [r for r in rows if r["kind"] == "source" and r["object_id"] == str(source.source_id)]
        self.assertEqual([r["detail"]["status"] for r in source_rows], ["active", "active", "deleted"])
        self.assertEqual([r["detail"]["revision"] for r in source_rows], [1, 2, 2])
        self.assertEqual(len({r["detail"]["identity_digest"] for r in source_rows}), 1)
        raw = json.dumps(rows)
        for private in ("Private original body", "Private corrected body", "private-repository-path"):
            self.assertNotIn(private, raw)
        removed = [r for r in rows if r["kind"] == "payload" and r["detail"]["availability"] == "deleted"]
        self.assertEqual(len(removed), 2)
        self.assertTrue(all(r["detail"]["prior_content_digest"] for r in removed))
        self.assertEqual([r["sequence"] for r in rows], list(range(1, len(rows) + 1)))
        before = await self.journal.checkpoint(self.owner)
        await self.fixture.sources.delete_source(self.owner, self.fixture.area, source.source_id, expected_revision=2)
        self.assertEqual(await self.journal.checkpoint(self.owner), before)

    async def test_rollback_does_not_publish_an_event_or_consume_a_cursor(self):
        before = await self.journal.checkpoint(self.owner)
        with self.assertRaises(RuntimeError):
            async with self.fixture.pool.acquire() as connection:
                async with connection.transaction():
                    await connection.execute("""insert into jarvis_context_payloads(owner_id,availability,content)
                        values($1,'available','{"text":"rolled back"}')""", self.owner)
                    raise RuntimeError("Rollback this fixture write")
        self.assertEqual(await self.journal.checkpoint(self.owner), before)
        self.assertEqual(len(await self.events()), before.through_sequence)

    async def test_owner_sequence_waits_for_commit_not_wall_clock_or_allocation(self):
        before = await self.journal.checkpoint(self.owner)
        pending = None
        async with self.fixture.pool.acquire() as first, self.fixture.pool.acquire() as second:
            try:
                async with first.transaction():
                    await first.execute("""insert into jarvis_context_payloads(owner_id,availability,content)
                        values($1,'available','{"text":"first"}')""", self.owner)
                    pending = asyncio.create_task(second.execute(
                        """insert into jarvis_context_payloads(owner_id,availability,content)
                            values($1,'available','{"text":"second"}')""", self.owner))
                    with self.assertRaises(TimeoutError):
                        await asyncio.wait_for(asyncio.shield(pending), timeout=0.1)
                await pending
            finally:
                if pending is not None and not pending.done():
                    pending.cancel()
                    await asyncio.gather(pending, return_exceptions=True)
        after = await self.journal.checkpoint(self.owner)
        self.assertEqual(after.through_sequence, before.through_sequence + 2)
        rows = await self.events(after)
        self.assertEqual([r["sequence"] for r in rows], list(range(1, len(rows) + 1)))

    async def test_frozen_checkpoint_excludes_later_changes_and_pagination_is_exact(self):
        await self.fixture.publish("At checkpoint")
        frozen = await self.journal.checkpoint(self.owner)
        await self.fixture.publish("Later source", source_key="later")
        cursor, sequences = 0, []
        while cursor < frozen.through_sequence:
            page = await self.journal.page(self.owner, frozen, after_sequence=cursor, limit=1)
            sequences += [item["sequence"] for item in page["events"]]
            cursor = page["next_after_sequence"]
        self.assertEqual(sequences, list(range(1, frozen.through_sequence + 1)))
        self.assertTrue(page["complete"])
        self.assertGreater((await self.journal.checkpoint(self.owner)).through_sequence, frozen.through_sequence)

    async def test_foreign_scope_epoch_change_and_missing_rows_fail_closed(self):
        source = await self.fixture.publish()
        pin = await self.journal.checkpoint(self.owner)
        with self.assertRaises(SourceError):
            await self.journal.page(self.fixture.other, pin)
        with self.assertRaises(SourceError):
            await self.journal.page(self.owner, pin.model_copy(update={"epoch": uuid4()}))
        async with self.fixture.pool.acquire() as connection:
            for sql in ("update jarvis_source_journal_events set kind='area' where owner_id=$1",
                        "delete from jarvis_source_journal_events where owner_id=$1"):
                with self.assertRaises(asyncpg.CheckViolationError):
                    await connection.execute(sql, self.owner)
            # Isolated corruption of a cursor, not removal of a real tombstone.
            await connection.execute("""update jarvis_source_journal_streams
                set last_sequence=last_sequence+1 where owner_id=$1""", self.owner)
        broken = await self.journal.checkpoint(self.owner)
        with self.assertRaises(SourceError) as caught:
            await self.journal.page(self.owner, broken)
        self.assertEqual(caught.exception.code, "source_journal_gap")
        self.assertEqual((await self.fixture.sources.read(self.owner, self.fixture.area, source.source_id)).content,
                         source.content)

    async def test_cleanup_intent_and_terminal_inventory_are_preserved_without_poll_noise(self):
        source = await self.fixture.publish()
        outbox = OutboxRepository(self.fixture.sources, retry_seconds=1)
        retain = await outbox.claim(self.owner, "fixture")
        await outbox.prepare_retain(retain)
        await self.fixture.sources.delete_source(self.owner, self.fixture.area, source.source_id, expected_revision=1)
        delete = await outbox.claim(self.owner, "fixture")
        before = await self.journal.checkpoint(self.owner)
        await outbox.prepare_cleanup(delete, [retain.id])
        await outbox.settle(delete, "done")
        rows = await self.events()
        cleanup = [r for r in rows if r["kind"] == "cleanup" and r["object_id"] == str(delete.id)]
        self.assertEqual(cleanup[0]["detail"]["status"], "pending")
        self.assertEqual(cleanup[-1]["detail"]["status"], "done")
        self.assertEqual(cleanup[-1]["detail"]["native_operation_ids"], [str(retain.id)])
        self.assertEqual(cleanup[-1]["detail"]["retain_dispatch_count"], 1)
        self.assertEqual((await self.journal.checkpoint(self.owner)).through_sequence, before.through_sequence + 2)
        # A late local retain settlement conservatively reopens cleanup. The
        # recovery journal must not leave the old terminal receipt as latest.
        self.assertEqual(await outbox.settle(retain, "searchable"), "cancelled")
        reopened = [r for r in await self.events() if r["kind"] == "cleanup"]
        self.assertEqual(reopened[-1]["object_id"], str(delete.id))
        self.assertEqual(reopened[-1]["detail"]["status"], "pending")
        # This tests recorded outbox metadata, not physical native erasure.

    async def test_real_database_export_pins_owner_epoch_and_detects_completed_segment(self):
        await self.fixture.publish("Never exported as text")
        pin = await self.journal.checkpoint(self.owner)
        with tempfile.TemporaryDirectory(prefix="jarvis-journal-db-export-") as temporary:
            root = Path(temporary) / "segment"
            result = await self.journal.export(self.owner, root, checkpoint=pin)
            self.assertTrue(result["complete"])
            self.assertEqual(result["event_count"], pin.through_sequence)
            raw = (root / "events.ndjson").read_bytes()
            self.assertNotIn(b"Never exported as text", raw)
            self.assertEqual(result["sha256"], hashlib.sha256(raw).hexdigest())
        for kwargs in ({"limit": 0}, {"limit": 513}, {"after_sequence": -1}, {"limit": True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                await self.journal.page(self.owner, pin, **kwargs)
