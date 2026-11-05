"""Offline canonical-caption intake, restart and forget interleavings."""
import asyncio
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock

from app.shared_memory import BankScope, ForgetResult, ForgetScope, MemoryRequestError
from app.shared_memory.ingestion import CanonicalCaption, ObservationDelta
from app.shared_memory.intake import CaptionPage, LiveDialogueIntake


NOW = datetime(2026, 11, 4, 12, tzinfo=timezone.utc)


def caption(sequence=1, *, key="a", revision=1, text="The whole caption, not one received word.", at=NOW):
    return CanonicalCaption(
        sequence, "session", "voice:connection", "live-caption:" + key, revision,
        "user_transcript", at - timedelta(seconds=3), at, "not_finalized",
        {"voice_route": "live", "caption_id": key, "caption_revision": revision,
         "speaker": "user", "text": text, "fragments": [
             {"provider_event_id": "f1", "start_ms": 100, "end_ms": 450, "delta": "The "},
             {"provider_event_id": "f2", "start_ms": 451, "end_ms": 900, "delta": text[4:]},
         ]},
    )


class Source:
    owner_id = "default"

    def __init__(self):
        self.rows = []
        self.head = 0

    def add(self, row):
        self.rows.append(row)
        self.head = max(self.head, row.sequence)

    async def high_watermark(self):
        return self.head

    async def read_after(self, cursor, *, through=None, limit=128):
        ceiling = self.head if through is None else through
        latest = {}
        for row in self.rows:
            if cursor < row.sequence <= ceiling:
                latest[(row.source_namespace, row.source_event_id)] = row
        rows = sorted(latest.values(), key=lambda row: row.sequence)[:limit]
        return CaptionPage(tuple(rows), rows[-1].sequence if len(rows) == limit else ceiling)


class Memory:
    def __init__(self):
        self.scope = BankScope.for_owner("default", generation="test")
        self.operations, self.documents = {}, {}
        self.posts, self.deletes = [], []
        self.post_started = asyncio.Event()
        self.release_operation = None
        self.lose_ack = False
        self.operation_missing = False
        self.fail_delete = False

    async def _request(self, method, path, *, body=None, missing_ok=False):
        if method == "POST":
            operation = body["operation_id"]
            self.posts.append(body)
            self.operations.setdefault(operation, body["items"])
            self.post_started.set()
            if self.lose_ack:
                self.lose_ack = False
                raise MemoryRequestError("hindsight_transport_error")
            return {"success": True, "operation_id": operation}
        operation = path.rsplit("/", 1)[-1]
        if self.operation_missing or operation not in self.operations:
            return None
        if self.release_operation is not None:
            await self.release_operation.wait()
        for document in self.operations[operation]:
            self.documents[document["document_id"]] = document
        return {"status": "completed"}

    async def forget(self, ids, *, invalidate):
        self.deletes.append(tuple(ids))
        scope = ForgetScope(self.scope.owner_id, self.scope.bank_id, tuple(ids))
        error = None
        try:
            await invalidate(scope)
        except Exception:
            error = "local_invalidation_failed"
        if self.fail_delete:
            return ForgetResult(scope, (), {identity: "delete_failed" for identity in ids}, error is None, error)
        for identity in ids:
            self.documents.pop(identity, None)
        return ForgetResult(scope, tuple(ids), {}, error is None, error)


class IntakeTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "cursor.sqlite"
        self.source, self.memory = Source(), Memory()
        self.now = NOW
        self.invalidator = AsyncMock()
        self.instances = []
        self.intake = self.make_intake()

    def make_intake(self, **kwargs):
        intake = LiveDialogueIntake(self.memory, self.source, self.path,
                                   invalidate=self.invalidator, now=lambda: self.now,
                                   poll_seconds=0.001, operation_timeout=0.05, **kwargs)
        self.instances.append(intake)
        return intake

    async def asyncTearDown(self):
        for intake in self.instances:
            intake._release()
        self.temp.cleanup()

    async def test_first_enable_starts_at_current_head_no_backfill(self):
        self.source.add(caption(10, at=NOW - timedelta(days=30)))
        self.assertEqual(await self.intake.initialize(), 10)
        await self.intake.tick()
        self.assertFalse(self.memory.posts)

    async def test_settle_and_whole_latest_caption(self):
        await self.intake.initialize()
        self.source.add(caption(1, text="The"))
        self.source.add(caption(2, revision=2))
        self.now = NOW + timedelta(seconds=17)
        self.assertEqual((await self.intake.tick())["retained"], 0)
        self.now += timedelta(seconds=1)
        self.assertEqual((await self.intake.tick())["retained"], 1)
        item = self.memory.posts[0]["items"][0]
        self.assertIn("The whole caption", item["content"])
        self.assertEqual(item["metadata"]["caption_revision"], "2")
        self.assertEqual(self.intake.cursor, 2)

    async def test_preserves_provisional_and_approximate_timing(self):
        item = caption().document().payload()
        self.assertEqual(item["metadata"]["speech_finality"], "provisional")
        self.assertEqual(item["metadata"]["settlement"], "application_settled_not_provider_final")
        self.assertEqual(item["timestamp"], (NOW - timedelta(seconds=3)).isoformat())
        self.assertEqual(json.loads(item["metadata"]["fragment_offsets"])[1]["end_ms"], 900)
        self.assertNotIn("delta", item["metadata"]["fragment_offsets"])

    async def test_observation_fragments_are_not_caption_input(self):
        await self.intake.initialize()
        self.source.add(replace(caption(), kind="observation"))
        self.now += timedelta(seconds=20)
        await self.intake.tick()
        self.assertFalse(self.memory.posts)
        self.assertEqual(self.intake.cursor, 1)

    async def test_new_revision_replaces_same_document(self):
        await self.intake.initialize()
        self.source.add(caption())
        self.now += timedelta(seconds=20)
        await self.intake.tick()
        self.source.add(caption(2, revision=2, text="Corrected caption"))
        await self.intake.tick()
        self.assertEqual(len(self.memory.documents), 1)
        self.assertIn("Corrected caption", next(iter(self.memory.documents.values()))["content"])
        self.assertEqual(self.memory.posts[0]["items"][0]["document_id"], self.memory.posts[1]["items"][0]["document_id"])

    async def test_cursor_survives_restart_and_no_transcript_is_saved(self):
        await self.intake.initialize()
        self.source.add(caption())
        self.now += timedelta(seconds=20)
        await self.intake.tick()
        self.intake._release()
        restarted = self.make_intake()
        self.assertEqual(await restarted.initialize(), 1)
        await restarted.tick()
        self.assertEqual(len(self.memory.posts), 1)
        self.assertNotIn(b"whole caption", self.path.read_bytes())
        self.assertEqual(self.path.stat().st_mode & 0o777, 0o600)

    async def test_session_close_flushes_before_silence(self):
        await self.intake.initialize()
        self.source.add(caption())
        await self.intake.close(session_id="session")
        self.assertEqual(len(self.memory.documents), 1)
        self.assertIsNotNone(self.intake._db)

    async def test_durable_session_close_flushes_without_transport_callback(self):
        await self.intake.initialize()
        self.source.add(replace(caption(), session_closed_at=NOW))
        await self.intake.tick()
        self.assertEqual(len(self.memory.documents), 1)

    async def test_transport_reconnect_does_not_end_settle_period(self):
        await self.intake.initialize()
        self.source.add(caption())
        await self.intake.tick()
        self.source.add(replace(caption(2, key="reconnected"), source_namespace="voice:new-connection"))
        await self.intake.tick()
        self.assertFalse(self.memory.posts)

    async def test_final_close_flushes_and_releases(self):
        await self.intake.initialize()
        self.source.add(caption())
        await self.intake.close()
        self.assertEqual(len(self.memory.documents), 1)
        self.assertIsNone(self.intake._db)

    async def test_forget_waits_for_current_extraction_and_later_revision_stays_forgotten(self):
        await self.intake.initialize()
        self.source.add(caption())
        self.now += timedelta(seconds=20)
        self.memory.release_operation = asyncio.Event()
        writer = asyncio.create_task(self.intake.tick())
        await self.memory.post_started.wait()
        deletion = asyncio.create_task(self.intake.forget([caption().document_id]))
        await asyncio.sleep(0)
        self.assertFalse(self.memory.deletes)
        self.memory.release_operation.set()
        await writer
        result = await deletion
        self.assertTrue(result.requested_deletions_applied)
        self.assertFalse(self.memory.documents)
        self.source.add(caption(2, revision=2, text="Do not restore"))
        await self.intake.tick()
        self.assertEqual(len(self.memory.posts), 1)

    async def test_forget_before_settle_suppresses_pending_caption(self):
        await self.intake.initialize()
        self.source.add(caption())
        await self.intake.forget([caption().document_id])
        await self.intake.close()
        self.assertFalse(self.memory.posts)

    async def test_failed_delete_retried_once_on_startup(self):
        await self.intake.initialize()
        self.memory.fail_delete = True
        result = await self.intake.forget([caption().document_id])
        self.assertFalse(result.requested_deletions_applied)
        self.intake._release()
        self.memory.fail_delete = False
        restarted = self.make_intake()
        await restarted.initialize()
        await restarted.initialize()
        self.assertEqual(len(self.memory.deletes), 2)
        self.assertTrue(restarted.startup_forget_result.requested_deletions_applied)
        self.source.add(caption())
        self.now += timedelta(seconds=20)
        await restarted.tick()
        self.assertFalse(self.memory.posts)

    async def test_local_invalidation_failure_remains_pending(self):
        await self.intake.initialize()
        failed = AsyncMock(side_effect=RuntimeError("offline"))
        result = await self.intake.forget([caption().document_id], invalidate=failed)
        self.assertFalse(result.requested_deletions_applied)
        self.assertEqual(self.intake._pending_forgets(), (caption().document_id,))

    async def test_lost_ack_recovers_native_operation_without_duplicate_submission(self):
        await self.intake.initialize()
        self.source.add(caption())
        self.now += timedelta(seconds=20)
        self.memory.lose_ack = True
        with self.assertRaises(MemoryRequestError):
            await self.intake.tick()
        self.intake._release()
        restarted = self.make_intake()
        await restarted.initialize()
        await restarted.tick()
        self.assertEqual(len(self.memory.posts), 1)
        self.assertEqual(restarted.cursor, 1)

    async def test_unknown_operation_cannot_report_successful_forget(self):
        await self.intake.initialize()
        self.source.add(caption())
        self.now += timedelta(seconds=20)
        self.memory.lose_ack = self.memory.operation_missing = True
        with self.assertRaises(MemoryRequestError):
            await self.intake.tick()
        result = await self.intake.forget([caption().document_id])
        self.assertFalse(result.requested_deletions_applied)
        self.assertTrue(result.local_invalidated)
        self.assertFalse(self.memory.deletes)
        self.assertEqual(len(self.memory.posts), 1)

    async def test_multiple_pages_do_not_skip_captions(self):
        intake = self.intake
        intake.batch_size = 1
        await intake.initialize()
        self.source.add(caption(1))
        self.source.add(caption(2, key="b"))
        await intake.close()
        self.assertEqual(len(self.memory.documents), 2)

    async def test_second_owner_cannot_use_checkpoint(self):
        await self.intake.initialize()
        self.intake._release()
        self.memory.scope = BankScope.for_owner("other", generation="test")
        self.source.owner_id = "other"
        with self.assertRaises(ValueError):
            await self.make_intake().initialize()

    async def test_second_instance_is_excluded(self):
        await self.intake.initialize()
        with self.assertRaises(BlockingIOError):
            await self.make_intake().initialize()

    async def test_stopped_runner_does_not_ingest(self):
        await self.intake.initialize()
        self.source.add(caption())
        stop = asyncio.Event()
        stop.set()
        await self.intake.run(stop)
        self.assertFalse(self.memory.posts)

    def observation(self, text="Keys were placed on the counter."):
        return ObservationDelta("visual-frame-one", "session", text, NOW, NOW, NOW,
                                frame_ids=("frame-one",)).document()

    async def test_visual_acceptance_uses_pending_slot_without_advancing_caption_cursor(self):
        receipt = await self.intake.retain_observation(self.observation())
        self.assertEqual(receipt["status"], "accepted")
        self.assertEqual(self.intake.cursor, 0)
        row = self.intake._db.execute("SELECT * FROM checkpoint").fetchone()
        self.assertEqual(row["pending_cursor"], row["sequence"])
        self.assertEqual(json.loads(row["pending_documents"]), [self.observation().document_id])
        self.assertNotIn(b"Keys were placed", self.path.read_bytes())
        await self.intake.tick()
        self.assertIn(self.observation().document_id, self.memory.documents)
        self.assertEqual(self.intake.cursor, 0)

    async def test_visual_forget_waits_for_native_completion_then_blocks_later_delta(self):
        document = self.observation()
        await self.intake.retain_observation(document)
        self.memory.release_operation = asyncio.Event()
        deletion = asyncio.create_task(self.intake.forget([document.document_id]))
        await asyncio.sleep(0)
        self.assertFalse(self.memory.deletes)
        self.memory.release_operation.set()
        result = await deletion
        self.assertTrue(result.requested_deletions_applied)
        self.assertNotIn(document.document_id, self.memory.documents)
        suppressed = await self.intake.retain_observation(self.observation("Do not restore this correction"))
        self.assertEqual(suppressed["status"], "suppressed_forgotten")
        self.assertEqual(len(self.memory.posts), 1)

    async def test_visual_queued_behind_forget_is_suppressed_under_same_lock(self):
        entered, release = asyncio.Event(), asyncio.Event()
        forget = self.memory.forget

        async def paused_forget(*args, **kwargs):
            entered.set()
            await release.wait()
            return await forget(*args, **kwargs)

        self.memory.forget = paused_forget
        deletion = asyncio.create_task(self.intake.forget([self.observation().document_id]))
        await entered.wait()
        writer = asyncio.create_task(self.intake.retain_observation(self.observation()))
        await asyncio.sleep(0)
        self.assertFalse(self.memory.posts)
        release.set()
        await deletion
        self.assertFalse((await writer)["submitted"])
        self.assertFalse(self.memory.posts)

    async def test_visual_lost_ack_recovers_after_restart_without_resubmission(self):
        self.memory.lose_ack = True
        with self.assertRaises(MemoryRequestError):
            await self.intake.retain_observation(self.observation())
        self.intake._release()
        restarted = self.make_intake()
        await restarted.tick()
        self.assertEqual(len(self.memory.posts), 1)
        self.assertIn(self.observation().document_id, self.memory.documents)
        self.assertEqual(restarted.cursor, 0)

    async def test_unknown_visual_receipt_never_replays_as_caption_or_claims_forgotten(self):
        self.memory.lose_ack = self.memory.operation_missing = True
        with self.assertRaises(MemoryRequestError):
            await self.intake.retain_observation(self.observation())
        with self.assertRaisesRegex(MemoryRequestError, "observation_retention_outcome_unknown"):
            await self.intake.tick()
        result = await self.intake.forget([self.observation().document_id])
        self.assertFalse(result.requested_deletions_applied)
        self.assertTrue(result.local_invalidated)
        self.assertFalse(self.memory.deletes)
        self.assertEqual(len(self.memory.posts), 1)
        self.assertEqual(self.intake._pending_forgets(), (self.observation().document_id,))
        self.intake._release()
        self.memory.operation_missing = False
        restarted = self.make_intake()
        await restarted.initialize()
        self.assertTrue(restarted.startup_forget_result.requested_deletions_applied)
        self.assertNotIn(self.observation().document_id, self.memory.documents)

    async def test_visual_generation_check_runs_inside_serialized_writer(self):
        receipt = await self.intake.retain_observation(self.observation(), is_current=lambda: False)
        self.assertEqual(receipt["status"], "suppressed_stale_generation")
        self.assertFalse(self.memory.posts)

    async def test_caption_writer_settles_visual_receipt_before_its_own_operation(self):
        await self.intake.retain_observation(self.observation())
        self.source.add(caption())
        self.now += timedelta(seconds=20)
        await self.intake.tick()
        self.assertEqual(len(self.memory.posts), 2)
        self.assertEqual(len(self.memory.documents), 2)
        self.assertEqual(self.intake.cursor, 1)

    async def test_visual_writer_rejects_dialogue_document(self):
        with self.assertRaises(ValueError):
            await self.intake.retain_observation(caption().document())


if __name__ == "__main__":
    unittest.main()
