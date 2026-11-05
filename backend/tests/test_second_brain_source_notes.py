"""Selected note producers: atomic originals, lineage, imports and worker reads."""
import asyncio
import asyncpg
from contextlib import asynccontextmanager
import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch
from uuid import UUID, uuid4

from app.knowledge.backfill import LegacyBackfill, NAMESPACE
from app.knowledge.lifecycle import SourceLifecycleService
from app.knowledge.models import SourceError
from app.knowledge.notes import SourceNoteRepository
from app.knowledge.product_notes import NOTE_NAMESPACE, note_identity
from app.knowledge.selection import configure_selected_memory
from app.notes.repository import NoteRepository
from legacy_note_tool_fixture import NoteToolModule, _note_payload
from app.runtime.feature_config import MemoryAuthority, TaskLimits
from app.settings import JarvisSettings
from app.tasks.context import TaskContextBuilder
from app.tasks.context_capture import SessionContextCapture
from app.tasks.context_events import ContextIngress
from app.tasks.context_repository import ContextRepository
from app.tasks.memory_evidence import TaskMemoryEvidence
from app.tasks.models import TaskCreate, TaskError, TaskVoiceSource
from app.tasks.repository import TaskRepository
from app.tasks.runner import TaskRunner
from app.tools.executor import JarvisToolExecutor
from app.tools.invocation import bind_tool_invocation
from app.tools.module import ToolBuildContext
from app.tools.registry import JarvisToolRegistry
from evals.second_brain.source_fixture import source_fixture


class NoteSourceIdentityTests(unittest.TestCase):
    def test_only_exact_trusted_product_identities_match(self):
        identity = uuid4()
        self.assertEqual(note_identity(NOTE_NAMESPACE, str(identity)), identity)
        self.assertEqual(note_identity(NAMESPACE, "jarvis_notes/" + str(identity)), identity)
        for namespace, key in (("fixture", str(identity)), (NOTE_NAMESPACE, identity.hex),
                               (NAMESPACE, "memory_items/" + str(identity)), (NOTE_NAMESPACE, "bad")):
            self.assertIsNone(note_identity(namespace, key))


@unittest.skipUnless(os.getenv("JARVIS_SECOND_BRAIN_TEST_DB_URL") and
                     os.getenv("JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"),
                     "Explicit manifested database required")
class SourceNoteDatabaseTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.f = await self.enterAsyncContext(source_fixture(
            os.environ["JARVIS_SECOND_BRAIN_TEST_DB_URL"],
            Path(os.environ["JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"]), test_name=self.id()))
        self.notes = SourceNoteRepository(self.f.sources, owner_id=self.f.owner, area_id=self.f.area)
        self.legacy = NoteRepository(self.f.pool)
        self.tasks = TaskRepository(self.f.pool, TaskLimits())
        self.history = ContextRepository(self.tasks)
        async with self.f.pool.acquire() as connection:
            self.session = await connection.fetchval(
                "insert into jarvis_sessions(owner_id) values($1) returning id", self.f.owner)
            self.foreign = await connection.fetchval(
                "insert into jarvis_sessions(owner_id) values($1) returning id", self.f.other)

    async def create(self, **changes):
        return await self.notes.create_note(**(dict(owner_id=self.f.owner, session_id=self.session,
            title="Robot calibration", body="Calibration remains unfinished.") | changes))

    async def original(self, note, **kwargs):
        return await self.f.sources.read(
            self.f.owner, self.f.area, UUID(note.memory_source["source_id"]), **kwargs)

    async def request(self):
        record = await self.history.append(self.f.owner, ContextIngress(
            session_id=self.session, source_namespace="note-test", source_event_id=str(uuid4()),
            receipt_id=str(uuid4()), kind="user_transcript", trust="user_input",
            payload={"text": "Apply this synthetic note lifecycle operation."}))
        return TaskVoiceSource(record.id, record.revision, self.session)

    async def test_atomic_create_edit_history_and_delete_original(self):
        note = await self.create()
        source = await self.original(note)
        self.assertEqual(source.content["record"]["body"], note.body)
        self.assertEqual(source.trust.value, "external_content")
        self.assertEqual(source.session_id, self.session)
        self.assertEqual(source.source_time, note.updated_at)
        changed = await self.notes.update_note(
            owner_id=self.f.owner, note_id=note.id, title=None, body="Calibration is still pending; measured drift.")
        current = await self.original(changed)
        self.assertEqual(current.source_id, source.source_id)
        self.assertEqual(current.revision, 2)
        self.assertEqual((await self.original(note, revision=1)).content["record"]["body"], note.body)
        same = await self.notes.update_note(
            owner_id=self.f.owner, note_id=note.id, title=None, body=changed.body)
        self.assertEqual(same.memory_source["revision"], 2)
        async with self.f.pool.acquire() as connection:
            jobs = await connection.fetch(
                "select operation,status from jarvis_memory_outbox where owner_id=$1", self.f.owner)
        self.assertIn(("retain", "pending"), [(row["operation"], row["status"]) for row in jobs])
        self.assertIn(("delete", "pending"), [(row["operation"], row["status"]) for row in jobs])
        self.assertTrue(await self.notes.delete_note(owner_id=self.f.owner, note_id=note.id))
        self.assertEqual(await self.legacy.search_notes(owner_id=self.f.owner, query=None, limit=8), [])
        with self.assertRaises(SourceError):
            await self.original(note, revision=1)

    async def test_publication_failure_rolls_back_note_and_edit(self):
        with patch.object(self.f.sources, "publish_in_transaction", side_effect=RuntimeError("fixture failure")):
            with self.assertRaises(RuntimeError):
                await self.create()
        self.assertEqual(await self.legacy.search_notes(owner_id=self.f.owner, query=None, limit=8), [])
        note = await self.create()
        with patch.object(self.f.sources, "publish_in_transaction", side_effect=RuntimeError("fixture failure")):
            with self.assertRaises(RuntimeError):
                await self.notes.update_note(owner_id=self.f.owner, note_id=note.id, title="Changed", body=None)
        self.assertEqual((await self.legacy.search_notes(owner_id=self.f.owner, query=None, limit=8))[0].title, note.title)

    async def test_repeated_save_identity_never_recreates_deleted_original(self):
        with bind_tool_invocation(provider_call_id="stable-note-save"):
            first = await self.create()
            second = await self.create()
            self.assertEqual(first.id, second.id)
            with self.assertRaises(SourceError):
                await self.create(body="A different note under the same operation.")
            await self.notes.delete_note(owner_id=self.f.owner, note_id=first.id)
            with self.assertRaises(SourceError):
                await self.create()
        self.assertEqual(await self.legacy.search_notes(owner_id=self.f.owner, query=None, limit=8), [])

    async def test_foreign_owner_and_session_cannot_write(self):
        for changes in ({"owner_id": self.f.other}, {"session_id": self.foreign}):
            with self.subTest(changes=changes), self.assertRaises((SourceError, asyncpg.ForeignKeyViolationError)):
                await self.create(**changes)
        self.assertEqual(await self.legacy.search_notes(owner_id=self.f.owner, query=None, limit=8), [])

    async def test_imported_note_keeps_its_source_identity_when_edited(self):
        note = await self.legacy.create_note(owner_id=self.f.owner, session_id=self.session,
                                             title="Imported notebook", body="Not finished.")
        importer = LegacyBackfill(self.f.sources)
        run_id = uuid4()
        for _ in range(8):
            result = await importer.run(self.f.owner, self.f.area, run_id=run_id, batch_size=64)
            if result["status"] != "pending":
                break
        self.assertTrue(result["local_manifest_complete"], result)
        before = (await self.notes.search_notes(owner_id=self.f.owner, query="Imported", limit=8))[0]
        changed = await self.notes.update_note(owner_id=self.f.owner, note_id=note.id, title=None, body="Paused until tomorrow.")
        self.assertEqual(before.memory_source["source_id"], changed.memory_source["source_id"])
        self.assertEqual(changed.memory_source["revision"], 2)
        async with self.f.pool.acquire() as connection:
            rows = await connection.fetch(
                "select namespace from jarvis_knowledge_sources where owner_id=$1", self.f.owner)
        self.assertEqual([row["namespace"] for row in rows], [NAMESPACE])

    async def test_unimported_and_out_of_band_changes_are_explicit_failures(self):
        note = await self.legacy.create_note(owner_id=self.f.owner, session_id=self.session,
                                             title="Unimported", body="Must not disappear.")
        with self.assertRaises(SourceError) as caught:
            await self.notes.search_notes(owner_id=self.f.owner, query="Unimported", limit=8)
        self.assertEqual(caught.exception.code, "note_import_incomplete")
        await self.legacy.delete_note(owner_id=self.f.owner, note_id=note.id)
        note = await self.create()
        await self.legacy.update_note(owner_id=self.f.owner, note_id=note.id, title=None, body="Untracked change")
        with self.assertRaises(SourceError) as caught:
            await self.notes.search_notes(owner_id=self.f.owner, query="Robot", limit=8)
        self.assertEqual(caught.exception.code, "note_original_changed")

    async def test_source_delete_erases_original_even_after_area_archival(self):
        note = await self.create()
        source = await self.original(note)
        async with self.f.pool.acquire() as connection:
            await connection.execute("update jarvis_knowledge_areas set status='archived',revision=revision+1 where id=$1", self.f.area)
        await self.f.sources.delete_source(self.f.owner, self.f.area, source.source_id, expected_revision=1)
        self.assertEqual(await self.legacy.search_notes(owner_id=self.f.owner, query=None, limit=8), [])

    async def test_generic_correction_cannot_fork_an_editable_note(self):
        note = await self.create()
        with self.assertRaises(SourceError) as caught:
            await SourceLifecycleService(self.f.sources, self.tasks).correct(
                self.f.owner, self.f.area, UUID(note.memory_source["source_id"]), expected_revision=1,
                replacement="Pretend this is a different note.", request=await self.request())
        self.assertEqual(caught.exception.code, "note_requires_original_edit")
        self.assertEqual((await self.original(note)).revision, 1)

    async def test_forgetting_prevents_note_read_and_automatic_unforget(self):
        note = await self.create()
        await SourceLifecycleService(self.f.sources, self.tasks).forget(
            self.f.owner, self.f.area, UUID(note.memory_source["source_id"]),
            expected_revision=1, request=await self.request())
        self.assertEqual(await self.notes.search_notes(owner_id=self.f.owner, query=None, limit=8), [])
        with self.assertRaises(SourceError):
            await self.notes.update_note(owner_id=self.f.owner, note_id=note.id, title=None, body="Changed")
        self.assertTrue(await self.notes.delete_note(owner_id=self.f.owner, note_id=note.id))

    async def test_preview_is_bounded_without_truncating_stored_original(self):
        body = "Synthetic long note. " * 1000
        note = await self.create(body=body)
        payload = _note_payload(note)
        self.assertFalse(payload["body_complete"])
        self.assertEqual(payload["body_characters"], len(body))
        self.assertLess(len(json.dumps(payload)), 8000)
        self.assertEqual((await self.original(note)).content["record"]["body"], body)

    async def test_one_connection_pool_and_source_bound_executor(self):
        permit = asyncio.Semaphore(1)
        @asynccontextmanager
        async def acquire():
            async with permit, self.f.pool.acquire() as connection:
                yield connection
        from app.knowledge.sources import SourceRepository
        repo = SourceNoteRepository(SourceRepository(SimpleNamespace(acquire=acquire)),
                                    owner_id=self.f.owner, area_id=self.f.area)
        async with asyncio.timeout(3):
            note = await repo.create_note(owner_id=self.f.owner, session_id=self.session, title="One slot", body="Original")
        settings = JarvisSettings(_env_file=None)
        registry = JarvisToolRegistry()
        registry.register_many(NoteToolModule(self.notes).build_tools(
            ToolBuildContext(settings, self.f.owner, self.session)))
        capture = SessionContextCapture(self.history, owner_id=self.f.owner, session_id=self.session,
                                        provider="openai", audio_session_id="source-note-test")
        self.addAsyncCleanup(capture.close)
        executor = JarvisToolExecutor(registry, owner_id=self.f.owner, session_id=self.session, context_capture=capture)
        result = await executor.execute("search_notes", {"query": "One slot"}, provider_call_id="note-search")
        self.assertEqual(result["status"], "success", result)
        self.assertTrue(await capture.flush())
        async with self.f.pool.acquire() as connection:
            dependent = await connection.fetchval(
                "select dependent_payload_id from jarvis_context_memory_sources where owner_id=$1", self.f.owner)
        self.assertIsNotNone(dependent)
        await self.notes.update_note(owner_id=self.f.owner, note_id=note.id, title=None, body="Revised")
        async with self.f.pool.acquire() as connection:
            self.assertNotEqual(await connection.fetchval(
                "select availability from jarvis_context_payloads where id=$1", dependent), "available")

    async def test_task_note_reads_bind_exact_source_and_invalidate_after_edit(self):
        note = await self.create()
        await self.tasks.create(self.f.owner, TaskCreate(request_id=uuid4(), title="Read notebook",
            requested_outcome="Read the saved robot note.", session_id=self.session), model="fixture-model")
        claim = await self.tasks.claim_next("source-note-worker", ("frontier",))
        context = TaskRunner(self.tasks, {}).context(claim)
        await TaskContextBuilder(self.history).build(claim)
        evidence = TaskMemoryEvidence(context, self.f.sources, area_id=self.f.area)
        worker_notes = SourceNoteRepository(self.f.sources, owner_id=self.f.owner,
                                            area_id=self.f.area, task_evidence=evidence)
        result = await worker_notes.search_notes(owner_id=self.f.owner, query="Robot", limit=8)
        self.assertEqual(result[0].task_evidence["source_id"], note.memory_source["source_id"])
        await self.notes.update_note(owner_id=self.f.owner, note_id=note.id, title=None, body="Changed")
        from app.tasks.memory_evidence import TaskMemoryCitation
        with self.assertRaises((TaskError, SourceError)):
            await evidence.verify(TaskMemoryCitation.model_validate(result[0].task_evidence))

    async def test_competing_area_mappings_cannot_delete_another_original(self):
        from app.knowledge.models import SourcePublication
        note = await self.create()
        source = await self.original(note)
        async with self.f.sources.transaction(self.f.owner) as connection:
            other_area = await connection.fetchval(
                "insert into jarvis_knowledge_areas(owner_id,name,domain) values($1,'Other','test') returning id",
                self.f.owner)
            await self.f.sources.publish_in_transaction(connection, self.f.owner, SourcePublication(
                area_id=other_area, namespace=NAMESPACE, source_key="jarvis_notes/" + str(note.id),
                payload_id=source.payload_id, source_type=source.source_type, trust=source.trust,
                source_time=source.source_time, time_basis=source.time_basis,
                session_id=source.session_id, expected_revision=0))
        with self.assertRaises(SourceError):
            await self.notes.delete_note(owner_id=self.f.owner, note_id=note.id)
        with self.assertRaises(SourceError):
            await self.f.sources.delete_source(self.f.owner, self.f.area, source.source_id, expected_revision=1)
        self.assertEqual(len(await self.legacy.search_notes(owner_id=self.f.owner, query=None, limit=8)), 1)


import test_second_brain_task_tool_bridge_integration as bridge_tests
from app.tasks.tool_bridge import TaskToolBridge


@unittest.skipUnless(os.getenv("JARVIS_SECOND_BRAIN_TEST_DB_URL") and
                     os.getenv("JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"),
                     "Explicit manifested database required")
class SourceNoteTaskActionTests(unittest.IsolatedAsyncioTestCase):
    make_bridge = bridge_tests.TaskToolBridgeDatabaseTests.make_bridge
    call = bridge_tests.TaskToolBridgeDatabaseTests.call
    count = bridge_tests.TaskToolBridgeDatabaseTests.count

    async def asyncSetUp(self):
        await bridge_tests.TaskToolBridgeDatabaseTests.asyncSetUp(self)
        self.history = ContextRepository(self.tasks)
        await TaskContextBuilder(self.history).build(self.claim)
        self.evidence = TaskMemoryEvidence(self.context, self.f.sources, area_id=self.f.area)
        self.notes = SourceNoteRepository(self.f.sources, owner_id=self.f.owner,
                                         area_id=self.f.area, task_evidence=self.evidence)
        self.bridge = await TaskToolBridge.from_modules(
            self.context, (NoteToolModule(self.notes),), read_tools=("search_notes",),
            capability_tools={"notes.create": "create_note"}, memory_evidence=self.evidence,
            receipt_revalidator=self.notes.revalidate_receipt)

    async def test_atomic_source_receipt_replays_once_and_is_revoked_on_delete(self):
        first = await self.call()
        second = await self.call(identity="another-provider-id")
        self.assertEqual(first["status"], "success", first)
        self.assertEqual(first["note"]["id"], second["note"]["id"])
        self.assertTrue(second["task_operation"]["replayed"])
        self.assertEqual(await self.count(), 1)
        async with self.f.pool.acquire() as connection:
            receipt = await connection.fetchval(
                "select receipt_payload_id from jarvis_task_actions where owner_id=$1 and id=$2",
                self.f.owner, UUID(first["task_operation"]["action_id"]))
            references = await connection.fetchval(
                "select count(*) from jarvis_context_memory_sources where owner_id=$1 and dependent_payload_id=$2",
                self.f.owner, receipt)
        self.assertEqual(references, 1)
        ordinary = SourceNoteRepository(self.f.sources, owner_id=self.f.owner, area_id=self.f.area)
        await ordinary.delete_note(owner_id=self.f.owner, note_id=UUID(first["note"]["id"]))
        async with self.f.pool.acquire() as connection:
            row = await connection.fetchrow(
                "select availability,content from jarvis_context_payloads where id=$1", receipt)
        self.assertNotEqual(row["availability"], "available")
        self.assertIsNone(row["content"])
        with self.assertRaises((TaskError, SourceError)):
            await self.call(identity="after-delete")
        self.assertEqual(await self.count(), 0)

    async def test_lost_outer_acknowledgement_keeps_atomic_accepted_receipt(self):
        with patch.object(self.context.actions, "record_receipt", new=AsyncMock(
                side_effect=RuntimeError("Synthetic lost acknowledgement"))):
            with self.assertRaises(RuntimeError):
                await self.call()
        replay = await self.call(identity="after-lost-ack")
        self.assertEqual(replay["status"], "success", replay)
        self.assertTrue(replay["task_operation"]["replayed"])
        self.assertEqual(await self.count(), 1)

    async def test_source_failure_rolls_back_local_effect_without_duplicate_retry(self):
        with patch.object(self.f.sources, "publish_in_transaction", side_effect=RuntimeError("Synthetic publication loss")):
            with self.assertRaises(RuntimeError):
                await self.call()
        self.assertEqual(await self.count(), 0)
        replay = await self.call(identity="after-source-failure")
        self.assertEqual(replay["status"], "outcome_unknown")
        self.assertEqual(await self.count(), 0)

    async def test_missing_authorized_task_invocation_cannot_write(self):
        with self.assertRaises(TaskError):
            await self.notes.create_note(owner_id=self.f.owner, session_id=self.session, **self.parameters)
        self.assertEqual(await self.count(), 0)
