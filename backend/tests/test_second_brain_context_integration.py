import asyncio
from contextlib import asynccontextmanager
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import unittest
from uuid import UUID, uuid4

import asyncpg

from app.migrations.runner import apply_migrations, quote_schema
from app.runtime.feature_config import TaskLimits
from app.tasks.context import TaskContextBuilder
from app.tasks.context_capture import SessionContextCapture
from app.tasks.context_events import ContextIngress, ContextWindow
from app.assistant.context_providers import ContextBuildRequest
from app.tasks.session_context import SessionTaskContextService
from types import SimpleNamespace
from app.tasks.context_repository import ContextRepository
from app.tasks.models import TaskCreate, TaskError, TaskMutation, TaskStatus, WorkerOutcome
from app.tasks.repository import TaskRepository
from evals.second_brain.fixtures import DatabaseFixture


@unittest.skipUnless(os.getenv("JARVIS_SECOND_BRAIN_TEST_DB_URL") and os.getenv("JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"),
                     "Explicit second-brain database fixture configuration required")
class CanonicalContextDatabaseTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        dsn = os.environ["JARVIS_SECOND_BRAIN_TEST_DB_URL"]
        self.fixture = DatabaseFixture(dsn, Path(os.environ["JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"]), test_name=self.id())
        self.addAsyncCleanup(self.fixture.cleanup)
        connection = await asyncpg.connect(dsn, timeout=10, command_timeout=90)
        try:
            self.schema = await self.fixture.create_schema(connection)
            await apply_migrations(connection, schema=self.schema, fresh=True)
        finally:
            await connection.close()
        async def setup(connection):
            await connection.execute(f'set search_path to {quote_schema(self.schema)},pg_catalog,public,extensions')
        self.pool = await asyncpg.create_pool(dsn, min_size=1, max_size=2, timeout=10, command_timeout=30, setup=setup)
        self.addAsyncCleanup(self.pool.close)
        self.owner, self.other = self.fixture.owners
        self.tasks = TaskRepository(self.pool, TaskLimits(lease_seconds=60))
        self.context = ContextRepository(self.tasks)
        async with self.pool.acquire() as connection:
            self.session = await connection.fetchval("insert into jarvis_sessions(owner_id) values($1) returning id", self.owner)

    def ingress(self, *, identity="utterance", receipt=None, text="Synthetic source", **changes):
        return ContextIngress(**({"session_id": self.session, "source_namespace": "fixture", "source_event_id": identity,
            "receipt_id": receipt or str(uuid4()), "kind": "user_transcript", "trust": "user_input", "payload": {"text": text},
            "source_time": datetime(2026, 9, 13, 12, tzinfo=UTC)} | changes))

    async def claim(self):
        task = await self.tasks.create(self.owner, TaskCreate(request_id=uuid4(), title="Fixture context",
            requested_outcome="Read the exact synthetic context", session_id=self.session), model="fixture")
        claim = await self.tasks.claim_next("context-fixture", ("frontier",))
        self.assertEqual(claim.task.id, task.id)
        return claim

    async def test_duplicate_ingress_and_late_replay_preserve_one_revised_utterance(self):
        provisional = self.ingress(receipt="partial", finality="provisional", text="Old draft")
        a, b = await asyncio.gather(self.context.append(self.owner, provisional), self.context.append(self.owner, provisional))
        self.assertEqual(a.id, b.id)
        self.assertFalse(a.authorizes_request)
        with self.assertRaises(TaskError):
            await self.context.request_evidence(self.owner, self.session, source_namespace="fixture", source_event_id="utterance")
        final = await self.context.append(self.owner, self.ingress(receipt="final", text="Final words", finality="revised", expected_revision=1))
        self.assertTrue(final.authorizes_request)
        self.assertEqual((await self.context.request_evidence(self.owner, self.session,
            source_namespace="fixture", source_event_id="utterance")).id, final.id)
        self.assertEqual(final.origin_sequence, a.sequence)
        self.assertEqual(final.supersedes_event_id, a.id)
        replay = await self.context.append(self.owner, provisional)
        self.assertEqual(replay.id, a.id)
        self.assertFalse(replay.is_current)
        self.assertEqual(len((await self.context.snapshot(self.owner, self.session, max_messages=10, max_pins=10))["recent"]), 1)

    async def test_repeated_new_utterance_and_foreign_scope_are_distinct(self):
        first = await self.context.append(self.owner, self.ingress(identity="one"))
        second = await self.context.append(self.owner, self.ingress(identity="two"))
        self.assertNotEqual(first.id, second.id)
        with self.assertRaises(TaskError):
            await self.context.get(self.other, first.id)
        with self.assertRaises(TaskError):
            await self.context.append(self.other, self.ingress())

    async def test_voice_capture_is_manifest_backed_for_both_provider_namespaces(self):
        for provider in ("openai", "gemini"):
            capture = SessionContextCapture(self.context, owner_id=self.owner, session_id=self.session,
                provider=provider, audio_session_id=str(uuid4()))
            self.addAsyncCleanup(capture.close)
            capture.input_admitted("missing")
            capture.input_final("request", "Remember the original synthetic decision")
            capture.bind_call("call", "request")
            request = await capture.request_for_call("call")
            self.assertIsNotNone(request)
            capture.observe_output({"type": "transcript.delta", "item_id": "partial", "response_id": "response",
                "text": "Unfinished synthetic response"}, forwarded=False)
            capture.tool_outcome("call", "fixture_read", {"status": "success", "value": "actual synthetic result"})
            await capture.close()
            self.assertIsNone(capture.issue)
            manifest = await TaskContextBuilder(self.context).build(await self.claim())
            messages = [m for m in manifest.content["messages"] if m["source_namespace"] == capture.namespace]
            self.assertEqual(len(messages), 4)
            self.assertIn(str(request.id), [m["id"] for m in messages])
            self.assertIn("session_closed_without_transcript", json.dumps(messages))
            self.assertIn("actual synthetic result", json.dumps(messages))
            self.assertEqual(next(m for m in messages if m["kind"] == "assistant_output")["heard_state"], "unknown")

    async def test_unknown_input_and_unheard_partial_output_are_preserved_honestly(self):
        gap = await self.context.append(self.owner, self.ingress(text=None, finality="unavailable"))
        output = await self.context.append(self.owner, self.ingress(identity="reply", kind="assistant_output", trust="assistant_output",
            finality="provisional", payload={"text": "Partial spoken draft", "response_id": "fixture-response", "interrupted": True}))
        self.assertFalse(gap.authorizes_request)
        self.assertEqual(output.as_context()["heard_state"], "unknown")
        manifest = await TaskContextBuilder(self.context).build(await self.claim())
        contents = json.dumps(manifest.content)
        self.assertIn("Partial spoken draft", contents)
        self.assertIn('"finality": "unavailable"', contents)

    async def test_compaction_retains_pin_and_archive_can_retrieve_old_decision(self):
        original = await self.context.append(self.owner, self.ingress(identity="decision", text="Keep the original fixture decision"))
        await self.context.pin(self.owner, self.session, original.id, reason="explicit decision")
        for i in range(6):
            await self.context.append(self.owner, self.ingress(identity=f"new-{i}", text=f"Later fixture message {i}"))
        claim = await self.claim()
        builder = TaskContextBuilder(self.context, window=ContextWindow(max_messages=2))
        manifest = await builder.build(claim)
        self.assertGreater(manifest.content["history"]["omitted_count"], 0)
        self.assertIn(str(original.id), [m["id"] for m in manifest.content["messages"]])
        archive = await builder.read_archive(claim, through=manifest.context_watermark)
        self.assertIn("Keep the original fixture decision", json.dumps(archive))

    async def test_archive_reads_become_manifest_dependencies_before_returning(self):
        old = await self.context.append(self.owner, self.ingress(identity="old", text="Archived private detail"))
        await self.context.append(self.owner, self.ingress(identity="new", text="Recent source"))
        claim = await self.claim()
        builder = TaskContextBuilder(self.context, window=ContextWindow(max_messages=1))
        manifest = await builder.build(claim)
        self.assertNotIn(str(old.id), [m["id"] for m in manifest.content["messages"]])
        archive = await builder.read_archive(claim, through=manifest.context_watermark)
        self.assertIn("Archived private detail", json.dumps(archive))
        async with self.pool.acquire() as connection:
            await connection.execute("update jarvis_context_payloads set availability='deleted',content=null where id=$1", old.payload_id)
        with self.assertRaises(TaskError):
            await self.context.read_manifest(self.owner, claim.task.id, manifest.id)

    async def test_correction_invalidates_manifest_and_fences_old_publication(self):
        original = await self.context.append(self.owner, self.ingress(text="Original"))
        claim = await self.claim()
        manifest = await TaskContextBuilder(self.context).build(claim)
        await self.context.append(self.owner, self.ingress(text="Corrected", finality="revised", expected_revision=original.revision))
        with self.assertRaises(TaskError):
            await self.context.read_manifest(self.owner, claim.task.id, manifest.id)
        await self.tasks.finish(claim, WorkerOutcome(TaskStatus.SUCCEEDED, {"answer": "obsolete answer"}))
        current = await self.tasks.get(self.owner, claim.task.id)
        self.assertEqual(current.status, TaskStatus.PAUSED)
        self.assertIsNone(current.result_payload_id)
        self.assertNotIn("obsolete answer", json.dumps(await self.tasks.events(self.owner, claim.task.id), default=str))

    async def test_payload_deletion_revokes_and_erases_derived_manifest(self):
        source = await self.context.append(self.owner, self.ingress(text="Private fixture-only content"))
        claim = await self.claim()
        manifest = await TaskContextBuilder(self.context).build(claim)
        async with self.pool.acquire() as connection:
            await connection.execute("update jarvis_context_payloads set availability='deleted',content=null where owner_id=$1 and id=$2", self.owner, source.payload_id)
            row = await connection.fetchrow("select m.status,p.content from jarvis_task_context_manifests m join jarvis_context_payloads p on p.id=m.payload_id where m.id=$1", manifest.id)
        self.assertEqual(row["status"], "revoked")
        self.assertIsNone(row["content"])
        with self.assertRaises(TaskError):
            await self.context.append(self.owner, self.ingress(text="A late restoration", finality="revised"))

    async def test_update_invalidates_only_the_exact_task_manifest(self):
        await self.context.append(self.owner, self.ingress())
        first = await self.claim()
        second = await self.claim()
        builder = TaskContextBuilder(self.context)
        a, b = await builder.build(first), await builder.build(second)
        await self.tasks.mutate(self.owner, first.task.id, "update", TaskMutation(request_id=uuid4(), expected_revision=1, message="Changed only this task"))
        with self.assertRaises(TaskError):
            await self.context.read_manifest(self.owner, first.task.id, a.id)
        self.assertEqual((await self.context.read_manifest(self.owner, second.task.id, b.id)).id, b.id)

    async def test_source_changed_between_snapshot_and_manifest_commit_is_rejected(self):
        original = await self.context.append(self.owner, self.ingress())
        claim = await self.claim()
        snapshot = await self.context.snapshot(self.owner, self.session, max_messages=10, max_pins=10)
        async with self.pool.acquire() as connection:
            await connection.execute("update jarvis_context_payloads set availability='deleted',content=null where id=$1", original.payload_id)
        with self.assertRaises(TaskError):
            await self.context.store_manifest(claim, watermark=snapshot["watermark"], records=snapshot["recent"], content={"must_not_persist": True})

    async def test_recent_window_counts_inherited_pins_outside_dedicated_pin_window(self):
        claim = await self.claim()
        for i in range(5):
            record = await self.context.append(self.owner, self.ingress(identity=f"recent-pin-{i}"))
            await self.context.pin(self.owner, self.session, record.id, reason="fixture commitment")
        revised = await self.context.append(
            self.owner, self.ingress(identity="recent-pin-0", text="Corrected oldest pin", finality="revised"),
        )
        builder = TaskContextBuilder(self.context, window=ContextWindow(max_messages=5, max_pins=2))
        manifest = await builder.build(claim)
        history = manifest.content["history"]
        self.assertEqual(history["included_pinned"], 5)
        self.assertEqual(history["omitted_pinned_count"], 0)
        self.assertEqual(history["pin_window_omitted_count"], 3)
        self.assertTrue(all(message["pinned"] for message in manifest.content["messages"]))
        self.assertIn(str(revised.id), [message["id"] for message in manifest.content["messages"]])

    async def test_pinned_overflow_is_counted_and_every_pin_remains_retrievable(self):
        pins = []
        for i in range(5):
            record = await self.context.append(self.owner, self.ingress(identity=f"pin-{i}", text=f"Original pin {i}"))
            await self.context.pin(self.owner, self.session, record.id, reason="fixture commitment")
            pins.append(record)
        revised = await self.context.append(self.owner, self.ingress(identity="pin-0", text="Corrected pin zero", finality="revised"))
        claim = await self.claim()
        builder = TaskContextBuilder(self.context, window=ContextWindow(max_messages=1, max_pins=2))
        manifest = await builder.build(claim)
        self.assertEqual(manifest.content["history"]["total_pinned_sources"], 5)
        self.assertEqual(manifest.content["history"]["pin_window_omitted_count"], 3)
        after, found = 0, []
        while True:
            page = await builder.read_archive(claim, after=after, limit=2, index_only=True, pinned_only=True)
            found.extend(page["events"])
            after = page["next_after"]
            if not page["has_more"]:
                break
        self.assertEqual(len(found), 5)
        self.assertIn(str(revised.id), [event["id"] for event in found])
        self.assertNotIn(str(pins[0].id), [event["id"] for event in found])
        chunk = await builder.read_source(claim, pins[2].id)
        self.assertIn("Original pin 2", chunk["content_json"])
        async with self.pool.acquire() as connection:
            await connection.execute("update jarvis_context_payloads set availability='deleted',content=null where id=$1", pins[2].payload_id)
        with self.assertRaises(TaskError):
            await self.context.read_manifest(self.owner, claim.task.id, manifest.id)

    async def test_archive_and_source_reads_cannot_cross_manifest_watermark(self):
        original = await self.context.append(self.owner, self.ingress(identity="admitted"))
        claim = await self.claim()
        builder = TaskContextBuilder(self.context)
        manifest = await builder.build(claim)
        later = await self.context.append(self.owner, self.ingress(identity="future", text="Not admitted to this snapshot"))
        page = await builder.read_archive(claim)
        self.assertIn(str(original.id), [event["id"] for event in page["events"]])
        self.assertNotIn(str(later.id), [event["id"] for event in page["events"]])
        with self.assertRaises(TaskError):
            await builder.read_source(claim, later.id)
        with self.assertRaises(TaskError):
            await builder.read_archive(claim, through=later.sequence)
        async with self.pool.acquire() as connection:
            other_session = await connection.fetchval("insert into jarvis_sessions(owner_id) values($1) returning id", self.owner)
        outside = await self.context.append(self.owner, self.ingress(identity="other-session", session_id=other_session))
        with self.assertRaises(TaskError):
            await builder.read_source(claim, outside.id)
        self.assertEqual(manifest.context_watermark, page["events"][-1]["sequence"])

    async def test_large_source_is_losslessly_readable_and_late_correction_fences_reads(self):
        text = ('Fixture quote: " and newline: \\n ' * 400)
        original = await self.context.append(self.owner, self.ingress(identity="oversized", text=text))
        claim = await self.claim()
        builder = TaskContextBuilder(self.context, window=ContextWindow(max_characters=4000))
        await builder.build(claim)
        offset, chunks = 0, []
        while offset is not None:
            page = await builder.read_source(claim, original.id, offset=offset)
            self.assertLessEqual(len(json.dumps(page, ensure_ascii=True)), 4000)
            chunks.append(page["content_json"])
            offset = page["next_offset"]
        self.assertEqual(json.loads("".join(chunks))["text"], text)
        await self.context.append(self.owner, self.ingress(identity="oversized", text="Corrected", finality="revised"))
        with self.assertRaises(TaskError):
            await builder.read_source(claim, original.id)

    async def test_foreground_context_summary_is_owner_scoped_and_does_not_replay_commands(self):
        await self.context.append(self.owner, self.ingress(identity="old-command", text="An old private command"))
        claim = await self.claim()
        other_task = await self.tasks.create(
            self.other, TaskCreate(request_id=uuid4(), title="Other owner's task", requested_outcome="Unrelated fixture"),
            model="fixture",
        )
        service = SessionTaskContextService(self.context, timeout_seconds=10)
        request = ContextBuildRequest(self.owner, SimpleNamespace(id=self.session), (), datetime.now(UTC))
        payload = await service.describe(request)
        self.assertEqual(payload["active_task_count"], 1)
        self.assertEqual(payload["active_tasks"][0]["id"], str(claim.task.id))
        self.assertNotIn(str(other_task.id), json.dumps(payload))
        self.assertNotIn("An old private command", json.dumps(payload))
        self.assertGreater(payload["canonical_history"]["current_source_count"], 0)
        with self.assertRaises(TaskError):
            await service.describe(ContextBuildRequest(self.other, SimpleNamespace(id=self.session), (), datetime.now(UTC)))

    async def test_streaming_assistant_snapshot_survives_progress_but_not_deletion(self):
        capture = SessionContextCapture(self.context, owner_id=self.owner, session_id=self.session,
            provider="gemini", audio_session_id=str(uuid4()))
        self.addAsyncCleanup(capture.close)
        output = SimpleNamespace(item_id="streaming-reply", response_id="streaming-response",
            transcript="An unfinished acknowledgment", provider_audio_bytes=24000,
            forwarded_audio_bytes=24000, interrupted=False, completed=False)
        capture.gemini_output(output)
        self.assertTrue(await capture.flush())
        claim = await self.claim()
        manifest = await TaskContextBuilder(self.context).build(claim)
        partial = next(message for message in manifest.content["messages"]
                       if message["kind"] == "assistant_output")
        self.assertEqual(partial["finality"], "provisional")
        self.assertEqual(partial["heard_state"], "unknown")
        original = await self.context.get(self.owner, UUID(partial["id"]))
        output.transcript = "The acknowledgment has finished, while the expert continues."
        output.completed = True
        output.provider_audio_bytes = output.forwarded_audio_bytes = 96000
        capture.gemini_output(output)
        self.assertTrue(await capture.flush())
        current = await self.context.read_manifest(self.owner, claim.task.id, manifest.id)
        self.assertEqual(current.status, "active")
        # Historical partial text is still a real privacy dependency.
        async with self.pool.acquire() as connection:
            usage = await connection.fetchval(
                "select source_usage from jarvis_task_context_sources where manifest_id=$1 and event_id=$2",
                manifest.id, original.id)
            self.assertEqual(usage, "historical")
            await connection.execute(
                "update jarvis_context_payloads set availability='deleted',content=null where owner_id=$1 and id=$2",
                self.owner, original.payload_id)
        with self.assertRaises(TaskError):
            await self.context.read_manifest(self.owner, claim.task.id, manifest.id)

    async def test_streaming_snapshot_race_and_final_assistant_correction_are_distinct(self):
        await self.context.append(self.owner, self.ingress(identity="reply",
            kind="assistant_output", trust="assistant_output", finality="provisional", text="Partial"))
        claim = await self.claim()
        snapshot = await self.context.snapshot(self.owner, self.session, max_messages=10, max_pins=10)
        await self.context.append(self.owner, self.ingress(identity="reply",
            kind="assistant_output", trust="assistant_output", finality="final", text="Finished"))
        historical = await self.context.store_manifest(claim, watermark=snapshot["watermark"],
            records=snapshot["recent"], content={"messages": [record.as_context() for record in snapshot["recent"]]})
        self.assertEqual((await self.context.read_manifest(self.owner, claim.task.id, historical.id)).status, "active")
        current = await TaskContextBuilder(self.context).build(claim)
        await self.context.append(self.owner, self.ingress(identity="reply", kind="assistant_output",
            trust="assistant_output", finality="revised", text="Corrected finished explanation"))
        with self.assertRaises(TaskError):
            await self.context.read_manifest(self.owner, claim.task.id, current.id)

    async def test_context_snapshot_does_not_wait_for_uncommitted_owner_correction(self):
        original = await self.context.append(self.owner, self.ingress(text="Original request"))
        async with self.tasks.transaction(self.owner) as connection:
            revised = await self.context.append_in_transaction(
                connection, self.owner, self.ingress(text="Corrected request", finality="revised"))
            snapshot = await asyncio.wait_for(self.context.snapshot(
                self.owner, self.session, max_messages=10, max_pins=10), timeout=2)
            self.assertEqual([record.id for record in snapshot["recent"]], [original.id])
            self.assertLess(snapshot["watermark"], revised.sequence)
        current = await self.context.snapshot(self.owner, self.session, max_messages=10, max_pins=10)
        self.assertEqual([record.id for record in current["recent"]], [revised.id])

    async def test_read_snapshot_is_repeatable_across_committed_correction(self):
        original = await self.context.append(self.owner, self.ingress(text="Original request"))
        async with self.tasks.read_transaction() as connection:
            self.assertEqual(await connection.fetchval("show transaction_read_only"), "on")
            self.assertEqual(await connection.fetchval("show transaction_isolation"), "repeatable read")
            before = await self.context._record(connection, self.owner, original.id)
            revised = await self.context.append(self.owner, self.ingress(text="Corrected request", finality="revised"))
            after = await self.context._record(connection, self.owner, original.id)
            self.assertTrue(before.is_current)
            self.assertTrue(after.is_current)
            self.assertEqual(after.content, before.content)
        latest = await self.context.get(self.owner, original.id)
        self.assertFalse(latest.is_current)
        self.assertNotEqual(original.id, revised.id)

    async def test_exact_standalone_instruction_source_is_readable_without_other_owner_history(self):
        first = await self.tasks.create(self.owner, TaskCreate(request_id=uuid4(), title="Standalone",
            requested_outcome="Do not make changes."), model="fixture")
        other = await self.tasks.create(self.owner, TaskCreate(request_id=uuid4(), title="Unrelated",
            requested_outcome="Unrelated standalone source"), model="fixture")
        claim = await self.tasks.claim_next("context-fixture", ("frontier",))
        self.assertEqual(claim.task.id, first.id)
        builder = TaskContextBuilder(self.context)
        manifest = await builder.build(claim)
        self.assertEqual(manifest.content["task"]["request_source"]["event_id"], str(first.request_event_id))
        page = await builder.read_source(claim, first.request_event_id)
        self.assertEqual(json.loads(page["content_json"])["requested_outcome"], "Do not make changes.")
        with self.assertRaises(TaskError) as caught:
            await builder.read_source(claim, other.request_event_id)
        self.assertEqual(caught.exception.code, "context_event_not_found")

    async def test_request_lookup_does_not_wait_for_owner_write_lock_and_revalidates_mutation(self):
        original = await self.context.append(self.owner, self.ingress(text="Original request"))
        async with self.tasks.transaction(self.owner) as connection:
            revised = await self.context.append_in_transaction(
                connection, self.owner, self.ingress(text="Corrected request", finality="revised"),
            )
            # Read the committed statement snapshot without waiting for unrelated
            # owner work. The uncommitted correction must not grant authority.
            observed = await asyncio.wait_for(self.context.request_evidence(
                self.owner, self.session, source_namespace="fixture", source_event_id="utterance",
            ), timeout=1)
            self.assertEqual(observed.id, original.id)
        latest = await self.context.request_evidence(
            self.owner, self.session, source_namespace="fixture", source_event_id="utterance",
        )
        self.assertEqual(latest.id, revised.id)
        with self.assertRaises(TaskError) as caught:
            await self.tasks.create(self.owner, TaskCreate(
                request_id=uuid4(), title="Stale admission", requested_outcome="Must not run",
                session_id=self.session), model="fixture", source_event_id=observed.id)
        self.assertEqual(caught.exception.code, "task_request_evidence_missing")
        self.assertEqual(await self.tasks.list(self.owner), [])
