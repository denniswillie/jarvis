"""Complete file originals, storage lifecycle and existing foreground tool capture."""
import asyncio
from contextlib import asynccontextmanager
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from uuid import UUID, uuid4

import asyncpg

from app.files.repository import JarvisFileRepository
from app.files.tools import FileToolModule, _file_payload
from app.knowledge.files import SourceFileRepository
from app.knowledge.lifecycle import SourceLifecycleService
from app.knowledge.models import SourceError, SourcePublication
from app.knowledge.product_files import FILE_NAMESPACE, FILE_ORIGINAL_FORMAT, file_identity
from app.knowledge.sources import SourceRepository
from app.runtime.feature_config import TaskLimits
from app.settings import JarvisSettings
from app.tasks.context_capture import SessionContextCapture
from app.tasks.context_events import ContextIngress
from app.tasks.context_repository import ContextRepository
from app.tasks.models import TaskVoiceSource
from app.tasks.repository import TaskRepository
from app.tools.executor import JarvisToolExecutor
from app.tools.invocation import TaskToolInvocation, bind_tool_invocation
from app.tools.module import ToolBuildContext
from app.tools.registry import JarvisToolRegistry
from evals.second_brain.file_storage_fixture import MemoryStorage
from evals.second_brain.source_fixture import source_fixture


class FileSourceIdentityTests(unittest.TestCase):
    def test_identity_is_exact_and_reserved(self):
        identity = uuid4()
        self.assertEqual(file_identity(FILE_NAMESPACE, str(identity)), identity)
        for namespace, key in ((FILE_NAMESPACE, identity.hex), ("fixture", str(identity)),
                               (FILE_NAMESPACE, "bad"), (FILE_NAMESPACE, str(identity).upper())):
            self.assertIsNone(file_identity(namespace, key))


@unittest.skipUnless(os.getenv("JARVIS_SECOND_BRAIN_TEST_DB_URL") and
                     os.getenv("JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"),
                     "Explicit manifested database required")
class SourceFileDatabaseTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.f = await self.enterAsyncContext(source_fixture(
            os.environ["JARVIS_SECOND_BRAIN_TEST_DB_URL"],
            Path(os.environ["JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"]), test_name=self.id()))
        self.storage = MemoryStorage()
        self.files = SourceFileRepository(self.f.sources, self.storage, owner_id=self.f.owner, area_id=self.f.area)
        self.legacy = JarvisFileRepository(self.f.pool)
        self.tasks = TaskRepository(self.f.pool, TaskLimits())
        self.history = ContextRepository(self.tasks)
        async with self.f.pool.acquire() as connection:
            self.session = await connection.fetchval(
                "insert into jarvis_sessions(owner_id) values($1) returning id", self.f.owner)
            self.foreign = await connection.fetchval(
                "insert into jarvis_sessions(owner_id) values($1) returning id", self.f.other)

    async def create(self, **changes):
        return await self.files.create_text_file(**(dict(owner_id=self.f.owner, session_id=self.session,
            filename="robot-progress.txt", content="Calibration is not complete.\n",
            description="Synthetic robot work", operation_key="source-file-test") | changes))

    async def source(self, file, **changes):
        return await self.f.sources.read(self.f.owner, self.f.area,
            UUID(file.memory_source["source_id"]), **changes)

    async def operation(self):
        async with self.f.pool.acquire() as connection:
            return await connection.fetchrow("select * from jarvis_file_publications where owner_id=$1", self.f.owner)

    async def request(self):
        record = await self.history.append(self.f.owner, ContextIngress(
            session_id=self.session, source_namespace="file-test", source_event_id=str(uuid4()),
            receipt_id=str(uuid4()), kind="user_transcript", trust="user_input",
            payload={"text": "Apply this synthetic file lifecycle operation."}))
        return TaskVoiceSource(record.id, record.revision, self.session)

    async def test_verified_original_is_complete_and_replay_is_one_source(self):
        file = await self.create()
        source = await self.source(file)
        self.assertEqual(source.content["text"], "Calibration is not complete.\n")
        self.assertEqual(source.content["record"], file.model_dump(mode="json"))
        self.assertEqual(source.content["content_sha256"], hashlib.sha256(source.content["text"].encode()).hexdigest())
        self.assertEqual(source.source_time, file.created_at)
        self.assertEqual(source.session_id, self.session)
        self.assertEqual(source.trust.value, "assistant_output")
        self.assertEqual(source.ingestion_status, "pending")
        self.assertEqual((await self.create()).id, file.id)
        self.assertEqual(self.storage.uploads, 1)
        async with self.f.pool.acquire() as connection:
            self.assertEqual(await connection.fetchval(
                "select count(*) from jarvis_knowledge_sources where owner_id=$1", self.f.owner), 1)
        self.assertEqual(await self.files.read_content(owner_id=self.f.owner, file_id=file.id),
                         source.content["text"].encode())

    async def test_full_200000_character_multibyte_and_escaped_originals(self):
        for index, content in enumerate(("\U0001f7e2" * 200000, "\x01" * 199999 + "x")):
            with self.subTest(kind=index):
                file = await self.create(content=content, operation_key=f"large-file-{index}")
                original = await self.source(file)
                self.assertEqual(original.content["text"], content)
                self.assertEqual(file.size_bytes, len(content.encode()))
                self.assertEqual(await self.files.read_content(owner_id=self.f.owner, file_id=file.id), content.encode())
                self.assertLess(len(json.dumps(_file_payload(file))), 8000)
        self.assertEqual(self.storage.uploads, 2)

    async def test_ordinary_and_malformed_payloads_cannot_use_the_large_original_exception(self):
        text = "\U0001f7e2" * 70000
        for value in ({"text": text}, {"format": FILE_ORIGINAL_FORMAT, "text": text},
                      {"format": FILE_ORIGINAL_FORMAT, "record": {}, "text": text, "content_sha256": "bad"},
                      {"format": FILE_ORIGINAL_FORMAT, "record": {}, "text": text,
                       "content_sha256": "0" * 64, "unexpected": True}):
            with self.subTest(fields=tuple(value)), self.assertRaises(asyncpg.CheckViolationError):
                async with self.f.pool.acquire() as connection:
                    await connection.execute(
                        "insert into jarvis_context_payloads(owner_id,availability,content) values($1,'available',$2::jsonb)",
                        self.f.owner, json.dumps(value))

    async def test_invalid_text_fails_before_any_external_effect(self):
        for content in ("", " ", "a" * 200001, "a\x00b", "\ud800"):
            with self.subTest(length=len(content)), self.assertRaises((ValueError, UnicodeError)):
                await self.create(content=content)
        self.assertEqual(self.storage.uploads, 0)

    async def test_adoption_failure_rolls_back_file_source_and_reuses_upload(self):
        with patch.object(self.f.sources, "publish_in_transaction", side_effect=RuntimeError("synthetic failed adopt")):
            with self.assertRaises(RuntimeError):
                await self.create()
        self.assertEqual(await self.legacy.list_files(owner_id=self.f.owner, query=None, limit=8), [])
        async with self.f.pool.acquire() as connection:
            self.assertEqual(await connection.fetchval(
                "select count(*) from jarvis_knowledge_sources where owner_id=$1", self.f.owner), 0)
        self.assertEqual((await self.operation())["phase"], "uploaded")
        self.assertIsNotNone((await self.create()).memory_source)
        self.assertEqual(self.storage.uploads, 1)

    async def test_lost_ack_recovers_original_without_duplicate_upload(self):
        self.storage.mode = "fail_after"
        with self.assertRaises(RuntimeError):
            await self.create()
        file = await self.create()
        self.assertEqual((await self.source(file)).revision, 1)
        self.assertEqual(self.storage.uploads, 1)

    async def test_delete_original_revokes_memory_and_verifies_remote_cleanup(self):
        file = await self.create()
        original = await self.source(file)
        result = await self.files.delete_source_file(owner_id=self.f.owner, file_id=file.id)
        self.assertTrue(result["deleted"], result)
        self.assertFalse(result["full_erasure_verified"])
        self.assertEqual(result["storage_cleanup"], "deleted")
        self.assertIsNone(await self.legacy.get_file(owner_id=self.f.owner, file_id=file.id))
        self.assertFalse(self.storage.objects)
        with self.assertRaises(SourceError):
            await self.source(file)
        with self.assertRaises(SourceError):
            await self.create()
        self.assertEqual(self.storage.uploads, 1)
        async with self.f.pool.acquire() as connection:
            jobs = await connection.fetch("select operation,status from jarvis_memory_outbox where owner_id=$1",
                                          self.f.owner)
        self.assertIn(("delete", "pending"), [(row["operation"], row["status"]) for row in jobs])
        self.assertTrue((await self.files.delete_source_file(owner_id=self.f.owner, file_id=file.id))["deleted"])

    async def test_source_deletion_enqueues_file_cleanup_even_after_area_archive(self):
        file = await self.create()
        source = await self.source(file)
        async with self.f.pool.acquire() as connection:
            await connection.execute(
                "update jarvis_knowledge_areas set status='archived',revision=revision+1 where id=$1", self.f.area)
        await self.f.sources.delete_source(self.f.owner, self.f.area, source.source_id, expected_revision=1)
        self.assertIsNone(await self.legacy.get_file(owner_id=self.f.owner, file_id=file.id))
        self.assertEqual((await self.operation())["phase"], "delete_pending")
        outcomes = await self.files.publications.reconcile_pending(owner_id=self.f.owner)
        self.assertEqual(outcomes[0]["phase"], "deleted", outcomes)
        self.assertFalse(self.storage.objects)

    async def test_forget_preserves_original_but_blocks_reads_replay_and_tool_copies(self):
        file = await self.create()
        await SourceLifecycleService(self.f.sources, self.tasks).forget(
            self.f.owner, self.f.area, UUID(file.memory_source["source_id"]),
            expected_revision=1, request=await self.request())
        self.assertEqual(await self.files.list_files(owner_id=self.f.owner, query=None, limit=8), [])
        self.assertIsNotNone(await self.legacy.get_file(owner_id=self.f.owner, file_id=file.id))
        self.assertTrue(self.storage.objects)
        for read in (self.files.get_file, self.files.read_content):
            with self.assertRaises(SourceError):
                await read(owner_id=self.f.owner, file_id=file.id)
        with self.assertRaises(SourceError):
            await self.create()
        self.assertTrue((await self.files.delete_source_file(owner_id=self.f.owner, file_id=file.id))["deleted"])

    async def test_source_correction_cannot_change_immutable_file_behind_original(self):
        file = await self.create()
        with self.assertRaises(SourceError) as caught:
            await SourceLifecycleService(self.f.sources, self.tasks).correct(
                self.f.owner, self.f.area, UUID(file.memory_source["source_id"]), expected_revision=1,
                replacement="Pretend the file says the opposite.", request=await self.request())
        self.assertEqual(caught.exception.code, "file_requires_new_original")
        self.assertEqual((await self.source(file)).revision, 1)

    async def test_request_replay_with_new_provider_id_has_stable_file_identity(self):
        request = await self.request()
        with bind_tool_invocation(provider_call_id="first", request_event_id=request.event_id,
                                  request_revision=request.revision):
            first = await self.create(operation_key=None)
        with bind_tool_invocation(provider_call_id="second", request_event_id=request.event_id,
                                  request_revision=request.revision):
            second = await self.create(operation_key=None)
        self.assertEqual(first.id, second.id)
        self.assertEqual(self.storage.uploads, 1)

    async def test_foreign_owner_session_and_worker_path_fail_before_upload(self):
        for changes in ({"owner_id": self.f.other}, {"session_id": self.foreign}):
            with self.subTest(changes=changes), self.assertRaises(SourceError):
                await self.create(**changes)
        with bind_tool_invocation(provider_call_id="task", task=TaskToolInvocation(uuid4(), 1, uuid4())):
            with self.assertRaises(SourceError) as caught:
                await self.create()
        self.assertEqual(caught.exception.code, "file_task_grant_required")
        self.assertEqual(self.storage.uploads, 0)

    async def test_delete_during_pending_upload_does_not_adopt_late_original(self):
        self.storage.mode = "pause_after"
        work = asyncio.create_task(self.create())
        try:
            await asyncio.wait_for(self.storage.entered.wait(), 3)
            operation = await self.operation()
            result = await self.files.delete_source_file(owner_id=self.f.owner, file_id=operation["file_id"])
            self.assertTrue(result["deleted"], result)
            self.storage.release.set()
            with self.assertRaises(SourceError):
                await work
        finally:
            self.storage.release.set()
            await asyncio.gather(work, return_exceptions=True)
        self.assertEqual(await self.legacy.list_files(owner_id=self.f.owner, query=None, limit=8), [])
        async with self.f.pool.acquire() as connection:
            self.assertEqual(await connection.fetchval(
                "select count(*) from jarvis_knowledge_sources where owner_id=$1", self.f.owner), 0)
        self.assertFalse(self.storage.objects)

    async def test_one_connection_and_foreground_capture_have_source_fences(self):
        permit = asyncio.Semaphore(1)
        @asynccontextmanager
        async def acquire():
            async with permit, self.f.pool.acquire() as connection:
                yield connection
        files = SourceFileRepository(SourceRepository(SimpleNamespace(acquire=acquire)), self.storage,
                                     owner_id=self.f.owner, area_id=self.f.area)
        async with asyncio.timeout(4):
            file = await files.create_text_file(owner_id=self.f.owner, session_id=self.session,
                filename="one-slot.txt", content="Original", description="One slot", operation_key="single")
        settings = JarvisSettings(_env_file=None)
        registry = JarvisToolRegistry()
        registry.register_many(FileToolModule(self.files, self.storage).build_tools(
            ToolBuildContext(settings, self.f.owner, self.session)))
        capture = SessionContextCapture(self.history, owner_id=self.f.owner, session_id=self.session,
                                        provider="openai", audio_session_id="source-file-test")
        self.addAsyncCleanup(capture.close)
        executor = JarvisToolExecutor(registry, owner_id=self.f.owner, session_id=self.session, context_capture=capture)
        result = await executor.execute("get_file_metadata", {"file_id": str(file.id)}, provider_call_id="file-read")
        self.assertEqual(result["status"], "success", result)
        self.assertNotIn("storage_path", result["file"])
        self.assertTrue(await capture.flush())
        async with self.f.pool.acquire() as connection:
            dependent = await connection.fetchval(
                "select dependent_payload_id from jarvis_context_memory_sources where owner_id=$1", self.f.owner)
        self.assertIsNotNone(dependent)
        await self.files.delete_source_file(owner_id=self.f.owner, file_id=file.id)
        async with self.f.pool.acquire() as connection:
            self.assertNotEqual(await connection.fetchval(
                "select availability from jarvis_context_payloads where id=$1", dependent), "available")

    async def test_unpublished_and_out_of_band_metadata_are_not_silent_fallback(self):
        file = await self.create()
        async with self.f.pool.acquire() as connection:
            await connection.execute("update jarvis_files set filename='untracked.txt' where owner_id=$1 and id=$2",
                                     self.f.owner, file.id)
        with self.assertRaises(SourceError) as caught:
            await self.files.get_file(owner_id=self.f.owner, file_id=file.id)
        self.assertEqual(caught.exception.code, "file_original_changed")
        await self.legacy.create_file(owner_id=self.f.owner, session_id=self.session, filename="legacy.txt",
            content_type="text/plain", storage_bucket="fixture", storage_path="fixture-legacy", size_bytes=0,
            description="Unpublished", source="assistant")
        with self.assertRaises(SourceError) as caught:
            await self.files.list_files(owner_id=self.f.owner, query="Unpublished", limit=8)
        self.assertEqual(caught.exception.code, "file_import_incomplete")

    async def test_lost_delete_ack_can_be_reconciled_without_resurrecting_source(self):
        file = await self.create()
        self.storage.mode = "delete_ack_lost"
        with self.assertRaises(RuntimeError):
            await self.files.delete_source_file(owner_id=self.f.owner, file_id=file.id)
        self.assertIsNone(await self.legacy.get_file(owner_id=self.f.owner, file_id=file.id))
        self.assertTrue((await self.files.delete_source_file(owner_id=self.f.owner, file_id=file.id))["deleted"])
        self.assertFalse(self.storage.objects)

    async def test_parent_invalidation_quarantines_original_without_destroying_retained_file(self):
        parent = await self.f.publish("Initial report")
        file = await self.create(memory_sources=[parent.original_evidence()])
        await self.f.publish("Corrected report", expected_revision=1)
        with self.assertRaises(SourceError):
            await self.files.get_file(owner_id=self.f.owner, file_id=file.id)
        self.assertIsNotNone(await self.legacy.get_file(owner_id=self.f.owner, file_id=file.id))
        self.assertTrue(self.storage.objects)
        self.assertEqual((await self.operation())["phase"], "quarantined")
        self.assertTrue((await self.files.delete_source_file(owner_id=self.f.owner, file_id=file.id))["deleted"])

    async def test_reserved_namespace_and_untracked_metadata_writes_are_rejected(self):
        original = await self.f.publish()
        with self.assertRaises(SourceError):
            await self.f.sources.publish(self.f.owner, SourcePublication(
                area_id=self.f.area, namespace=FILE_NAMESPACE, source_key=str(uuid4()), payload_id=original.payload_id,
                source_type="file", trust="assistant_output", source_time=None, expected_revision=0, time_basis="fixture"))
        for method in (self.files.create_file, self.files.delete_file):
            with self.assertRaises(SourceError):
                await method(owner_id=self.f.owner)

    async def test_parent_privacy_deletion_removes_derived_files_but_not_unrelated_files(self):
        parent = await self.f.publish("Private source report")
        derived = await self.create(memory_sources=[parent.original_evidence()])
        second = await self.create(operation_key="second-level",
            memory_sources=[(await self.source(derived)).original_evidence()])
        retained = await self.create(operation_key="unrelated")
        await self.f.sources.delete_source(self.f.owner, self.f.area, parent.source_id, expected_revision=1)
        for file in (derived, second):
            self.assertIsNone(await self.legacy.get_file(owner_id=self.f.owner, file_id=file.id))
            with self.assertRaises(SourceError):
                await self.source(file)
        self.assertIsNotNone(await self.files.get_file(owner_id=self.f.owner, file_id=retained.id))
        outcomes = await self.files.publications.reconcile_pending(owner_id=self.f.owner)
        self.assertEqual(len(outcomes), 2)
        self.assertTrue(all(item["phase"] == "deleted" for item in outcomes), outcomes)
        self.assertEqual(len(self.storage.objects), 1)
        self.assertEqual(await self.files.read_content(owner_id=self.f.owner, file_id=retained.id),
                         b"Calibration is not complete.\n")
