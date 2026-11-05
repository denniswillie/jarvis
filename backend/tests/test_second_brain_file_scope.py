"""Conversation scope routes new files; persisted operations own retries and recovery."""
import asyncio
from contextlib import asynccontextmanager
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch
from uuid import UUID, uuid4

from app.files.publication import FilePublicationService
from app.files.tools import FileToolModule, _file_payload, create_file_tools
from app.knowledge.files import SourceFileRepository
from app.knowledge.models import SourceError
from app.knowledge.publication import CanonicalSourcePublication
from app.knowledge.session_areas import SessionKnowledgeAreas
from app.runtime.feature_config import TaskLimits
from app.tasks.context_capture import SessionContextCapture
from app.tasks.context_events import ContextIngress
from app.tasks.context_repository import ContextRepository
from app.tasks.models import TaskVoiceSource
from app.tasks.repository import TaskRepository
from app.tools.invocation import bind_tool_invocation
from app.tools.module import ToolBuildContext
from app.tools.provenance import ToolResultCapture
from app.visual_memory.tools import VisualCaptureToolModule
from evals.second_brain.file_storage_fixture import MemoryStorage
from evals.second_brain.source_fixture import source_fixture


class FileScopeContractTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.owner, self.area, self.project, self.session = "file-scope-fixture", uuid4(), uuid4(), uuid4()
        self.connection = SimpleNamespace(fetch=AsyncMock(return_value=[]), fetchrow=AsyncMock(),
                                          fetchval=AsyncMock(return_value=None))
        @asynccontextmanager
        async def transaction(owner):
            yield self.connection
        self.sources = SimpleNamespace(pool=object(), transaction=transaction, _area=AsyncMock())
        self.selections = SimpleNamespace(sources=self.sources, default_area_id=self.area, _session=AsyncMock(),
            read_in_transaction=AsyncMock(return_value={"area_id": str(self.project)}))
        self.root = SourceFileRepository(self.sources, MemoryStorage(), owner_id=self.owner,
                                         area_id=self.area, selections=self.selections)
        self.files = self.root.for_session(self.session)

    async def test_existing_publication_and_source_area_win_without_default_lookup(self):
        self.connection.fetchval.return_value = self.area
        with patch("app.knowledge.files.file_sources", new_callable=AsyncMock,
                   return_value=[{"area_id": self.area}]):
            self.assertEqual(await self.files._area_for(self.connection, uuid4()), self.area)
        self.selections.read_in_transaction.assert_not_awaited()
        self.selections._session.assert_awaited_once_with(self.connection, self.owner, self.session)
        self.assertIsNone(self.root.session_id)

    async def test_conflicting_mapping_is_not_guessed(self):
        self.connection.fetchval.return_value = self.project
        with patch("app.knowledge.files.file_sources", new_callable=AsyncMock,
                   return_value=[{"area_id": self.area}]), self.assertRaises(SourceError) as caught:
            await self.files._area_for(self.connection, uuid4())
        self.assertEqual(caught.exception.code, "file_source_conflict")

    async def test_empty_search_keeps_scope_through_tool_conversion_and_switch(self):
        tools = {t.name: t for t in FileToolModule(self.root, self.root.storage).build_tools(
            ToolBuildContext(None, self.owner, self.session))}
        result = await tools["list_files"].handler({})
        self.assertEqual(result["files"], [])
        self.assertEqual(result["knowledge_scope"]["area_id"], str(self.project))
        self.selections.read_in_transaction.return_value = {"area_id": str(self.area)}
        later = await tools["list_files"].handler({})
        self.assertEqual(later["knowledge_scope"]["area_id"], str(self.area))
        self.assertEqual(result["knowledge_scope"]["area_id"], str(self.project))

    async def test_result_capture_keeps_actual_area_and_rejects_foreign_or_mixed_scope(self):
        context = SimpleNamespace(capture=SimpleNamespace(owner_id=self.owner, session_id=self.session,
            repository=SimpleNamespace(pool=self.sources.pool)), call_id="result", tool_name="list_files")
        result = {"files": [], "knowledge_scope": {"area_id": str(self.project)}}
        with patch.object(self.files.provenance, "bind", new_callable=AsyncMock) as bind:
            await self.files.capture_tool_result(result, context)
            self.assertEqual(bind.call_args.args[:3], (self.owner, self.project, []))
        self.selections.read_in_transaction.assert_not_awaited()
        context.capture.session_id = uuid4()
        with self.assertRaises(SourceError):
            await self.files.capture_tool_result(result, context)
        context.capture.session_id = self.session
        result["files"] = [{"memory_source": {"area_id": str(self.area)}}]
        with self.assertRaises(SourceError) as caught:
            await self.files.capture_tool_result(result, context)
        self.assertEqual(caught.exception.code, "knowledge_result_scope_mismatch")

    async def test_owner_and_creation_session_rejected_before_storage(self):
        for owner, session in (("foreign", self.session), (self.owner, uuid4())):
            with self.assertRaises(SourceError):
                await self.files.create_text_file(owner_id=owner, session_id=session,
                    filename="no.txt", content="No write", description="Synthetic")
            with self.assertRaises(SourceError):
                await self.files.capture_current_frame(owner_id=owner, session_id=session,
                    filename="no.jpg", description="Synthetic")
        self.assertEqual(self.root.storage.uploads, 0)
        self.connection.fetchrow.assert_not_awaited()

    async def test_fixed_publications_stay_restricted_and_active_check_uses_recorded_area(self):
        fixed = SourceFileRepository(self.sources, self.root.storage, owner_id=self.owner, area_id=self.area)
        self.assertIs(fixed.for_session(self.session), fixed)
        self.assertEqual(fixed.publications.area_filter, self.area)
        self.assertIsNone(self.files.publications.area_filter)
        self.connection.fetchval.return_value = "available"
        await self.files.publications._active(self.connection, {"area_id": self.project,
            "delete_requested": False, "availability": "available", "metadata_payload_id": uuid4()})
        self.sources._area.assert_awaited_once_with(self.connection, self.owner, self.project)

    def test_capture_module_binds_the_same_trusted_session(self):
        context = SimpleNamespace(session_id=self.session, owner_id=self.owner, require_settings=lambda: object())
        with patch("app.visual_memory.tools.create_capture_latest_visual_frame_tool") as build:
            VisualCaptureToolModule(None, object(), self.root, self.root.storage).build_tools(context)
        selected = build.call_args.kwargs["file_repository"]
        self.assertEqual(selected.session_id, self.session)
        self.assertIs(selected.selections, self.selections)
        self.assertIsNone(self.root.session_id)

    def test_incompatible_scopes_and_worker_rebinding_are_rejected(self):
        with self.assertRaises(ValueError):
            self.root.for_session("not-a-trusted-uuid")
        with self.assertRaises(ValueError):
            SourceFileRepository(self.sources, self.root.storage, owner_id=self.owner, area_id=self.area,
                                 selections=self.selections, task_evidence=object())
        wrong = SimpleNamespace(sources=self.sources, default_area_id=uuid4())
        with self.assertRaises(ValueError):
            FilePublicationService(self.sources, self.root.storage, owner_id=self.owner, area_id=self.area,
                                   selections=wrong)


@unittest.skipUnless(os.getenv("JARVIS_SECOND_BRAIN_TEST_DB_URL") and
                     os.getenv("JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"),
                     "Explicit manifested database required")
class FileScopeDatabaseTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.f = await self.enterAsyncContext(source_fixture(os.environ["JARVIS_SECOND_BRAIN_TEST_DB_URL"],
            Path(os.environ["JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"]), test_name=self.id()))
        self.tasks = TaskRepository(self.f.pool, TaskLimits())
        self.selections = SessionKnowledgeAreas(self.f.sources, self.tasks, self.f.area)
        self.history = ContextRepository(self.tasks, publication=CanonicalSourcePublication(
            self.f.sources, self.f.area, self.selections))
        self.storage = MemoryStorage()
        self.root = self.repository()
        async with self.f.pool.acquire() as connection:
            self.session = await connection.fetchval(
                "insert into jarvis_sessions(owner_id) values($1) returning id", self.f.owner)
            self.other_session = await connection.fetchval(
                "insert into jarvis_sessions(owner_id) values($1) returning id", self.f.owner)
            self.project = await connection.fetchval(
                "insert into jarvis_knowledge_areas(owner_id,name,domain) values($1,'Bench','robotics') returning id", self.f.owner)
            self.fitness = await connection.fetchval(
                "insert into jarvis_knowledge_areas(owner_id,name,domain) values($1,'Training','fitness') returning id", self.f.owner)
        self.files = self.root.for_session(self.session)

    def repository(self):
        return SourceFileRepository(self.f.sources, self.storage, owner_id=self.f.owner,
                                     area_id=self.f.area, selections=self.selections)

    async def request(self, text):
        return await self.history.append(self.f.owner, ContextIngress(session_id=self.session,
            source_namespace="voice:file-scope", source_event_id=str(uuid4()), receipt_id=str(uuid4()),
            kind="user_transcript", trust="user_input", payload={"text": text}))

    async def select(self, area):
        event = await self.request("Select a synthetic knowledge area.")
        current = await self.selections.read(self.f.owner, self.session)
        await self.selections.select(self.f.owner, area,
            request=TaskVoiceSource(event.id, event.revision, self.session),
            expected_revision=current["selection_revision"])

    async def create(self, key="scoped-file"):
        return await self.files.create_text_file(owner_id=self.f.owner, session_id=self.session,
            filename="bench.txt", content="Calibration paused, not completed.\n",
            description="Synthetic progress", operation_key=key)

    async def test_selected_tools_retry_and_late_receipts_preserve_original_scope(self):
        await self.select(self.project)
        tools = {t.name: t for t in create_file_tools(repository=self.root, storage=self.storage,
                                                      owner_id=self.f.owner, session_id=self.session)}
        request = await self.request("Save the synthetic progress file.")
        arguments = {"filename": "bench.txt", "content": "Calibration paused, not completed.\n"}
        with bind_tool_invocation(provider_call_id="create", request_event_id=request.id, request_revision=request.revision):
            original = await tools["create_text_file"].handler(arguments)
        file_id = UUID(original["file"]["id"])
        self.assertEqual(original["file"]["memory_source"]["area_id"], str(self.project))
        found = await tools["list_files"].handler({})
        empty = await tools["list_files"].handler({"query": "not present"})
        self.assertEqual(await self.root.for_session(self.other_session).list_files(
            owner_id=self.f.owner, query=None, limit=8), [])
        await self.select(self.fitness)
        with bind_tool_invocation(provider_call_id="new-provider-id", request_event_id=request.id, request_revision=request.revision):
            replay = await tools["create_text_file"].handler(arguments)
        self.assertEqual(replay["file"]["id"], str(file_id))
        self.assertEqual(self.storage.uploads, 1)
        self.assertEqual((await tools["list_files"].handler({}))["files"], [])
        self.assertEqual((await tools["get_file_metadata"].handler({"file_id": str(file_id)}))["file"]["memory_source"]["area_id"],
                         str(self.project))
        self.assertEqual(await self.files.read_content(owner_id=self.f.owner, file_id=file_id), arguments["content"].encode())
        capture = SessionContextCapture(self.history, owner_id=self.f.owner, session_id=self.session,
                                         provider="openai", audio_session_id="file-scope")
        self.addAsyncCleanup(capture.close)
        receipts = []
        for call_id, name, result in (("found", "list_files", found), ("empty", "list_files", empty),
                                      ("created", "create_text_file", original)):
            receipts.append(await self.files.capture_tool_result(result, ToolResultCapture(capture, call_id, name)))
        async with self.f.pool.acquire() as connection:
            for receipt in receipts:
                self.assertEqual(await connection.fetchval("""select s.area_id from jarvis_knowledge_context_sources cs
                    join jarvis_knowledge_sources s on s.owner_id=cs.owner_id and s.id=cs.source_id
                    where cs.owner_id=$1 and cs.event_id=$2""", self.f.owner, receipt.event_id), self.project)
        self.assertTrue((await self.files.delete_source_file(owner_id=self.f.owner, file_id=file_id))["deleted"])
        async with self.f.pool.acquire() as connection:
            self.assertNotEqual(await connection.fetchval("select availability from jarvis_context_payloads where owner_id=$1 and id=$2",
                self.f.owner, receipts[0].payload_id), "available")
        with bind_tool_invocation(provider_call_id="after-delete", request_event_id=request.id, request_revision=request.revision):
            with self.assertRaises(SourceError):
                await tools["create_text_file"].handler(arguments)
        self.assertEqual(self.storage.uploads, 1)

    async def test_concurrent_retry_during_upload_does_not_follow_later_selection(self):
        await self.select(self.project)
        self.storage.mode = "pause_after"
        work = asyncio.create_task(self.create())
        try:
            await asyncio.wait_for(self.storage.entered.wait(), 10)
            await self.select(self.fitness)
            second = await self.create()
            self.storage.release.set()
            first = await work
            self.assertEqual(first.id, second.id)
            self.assertEqual(first.memory_source["area_id"], str(self.project))
            self.assertEqual(second.memory_source["area_id"], str(self.project))
            self.assertEqual(self.storage.uploads, 1)
            fixed = SourceFileRepository(self.f.sources, self.storage, owner_id=self.f.owner, area_id=self.f.area)
            with self.assertRaises(SourceError):
                await fixed.publications.read_content(owner_id=self.f.owner, file_id=first.id)
        finally:
            self.storage.release.set()
            await asyncio.gather(work, return_exceptions=True)

    async def test_restart_recovery_processes_operations_in_all_owned_areas(self):
        self.storage.mode = "fail_after"
        for area, key in ((self.project, "robotics-upload"), (self.fitness, "fitness-upload")):
            await self.select(area)
            with self.assertRaises(RuntimeError):
                await self.create(key)
        restarted = self.repository()
        first = await restarted.publications.reconcile_pending(owner_id=self.f.owner, limit=1)
        second = await restarted.publications.reconcile_pending(owner_id=self.f.owner, limit=1)
        self.assertEqual([first[0]["phase"], second[0]["phase"]], ["available", "available"])
        self.assertNotEqual(first[0]["operation_id"], second[0]["operation_id"])
        self.assertEqual(self.storage.uploads, 2)
        async with self.f.pool.acquire() as connection:
            rows = await connection.fetch("select area_id,file_id from jarvis_file_publications where owner_id=$1", self.f.owner)
        self.assertEqual({r["area_id"] for r in rows}, {self.project, self.fitness})
        for row in rows:
            file = await restarted.get_file(owner_id=self.f.owner, file_id=row["file_id"])
            self.assertEqual(file.memory_source["area_id"], str(row["area_id"]))
        with self.assertRaises(SourceError):
            await restarted.publications.reconcile_pending(owner_id=self.f.other)

    async def test_unknown_upload_archived_cleanup_and_tombstone_keep_recorded_scope(self):
        await self.select(self.project)
        self.storage.mode = "fail_before"
        with self.assertRaises(RuntimeError):
            await self.create()
        async with self.f.pool.acquire() as connection:
            row = await connection.fetchrow("select * from jarvis_file_publications where owner_id=$1", self.f.owner)
        await self.select(self.fitness)
        async with self.f.pool.acquire() as connection:
            await connection.execute("update jarvis_knowledge_areas set status='archived',revision=revision+1 where owner_id=$1 and id=$2",
                                     self.f.owner, self.project)
        pending = await self.files.delete_source_file(owner_id=self.f.owner, file_id=row["file_id"])
        self.assertEqual(pending["storage_cleanup"], "delete_pending")
        self.assertFalse(pending["deleted"])
        self.storage.objects[(row["storage_bucket"], row["storage_path"])] = b"Calibration paused, not completed.\n"
        recovered = await self.repository().publications.reconcile_pending(owner_id=self.f.owner)
        self.assertEqual([r["phase"] for r in recovered], ["deleted"])
        self.assertFalse(self.storage.objects)
        with self.assertRaises(SourceError):
            await self.create()
        self.assertEqual(self.storage.uploads, 1)
        self.assertEqual((await self.selections.read(self.f.owner, self.session))["area_id"], str(self.fitness))


@unittest.skipUnless(os.getenv("JARVIS_SECOND_BRAIN_TEST_DB_URL") and
                     os.getenv("JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"),
                     "Explicit manifested database required")
class SavedCameraScopeDatabaseTests(unittest.IsolatedAsyncioTestCase):
    request = FileScopeDatabaseTests.request
    select = FileScopeDatabaseTests.select

    async def test_saved_camera_retry_after_switch_and_disconnect_preserves_pixels_and_scope(self):
        from test_second_brain_saved_files import SavedFileDatabaseTests
        await SavedFileDatabaseTests.asyncSetUp(self)
        self.selections = SessionKnowledgeAreas(self.f.sources, self.tasks, self.f.area)
        self.history = ContextRepository(self.tasks, publication=CanonicalSourcePublication(
            self.f.sources, self.f.area, self.selections))
        async with self.f.pool.acquire() as connection:
            project = await connection.fetchval(
                "insert into jarvis_knowledge_areas(owner_id,name,domain) values($1,'View','robotics') returning id", self.f.owner)
        root = SourceFileRepository(self.f.sources, self.storage, owner_id=self.f.owner,
            area_id=self.f.area, selections=self.selections, current_images=self.current)
        self.files = root.for_session(self.session)
        await self.select(project)
        first = await SavedFileDatabaseTests.save(self)
        await SavedFileDatabaseTests.expire_raw(self)
        await self.select(self.f.area)
        second = await SavedFileDatabaseTests.save(self)
        self.assertEqual(first.id, second.id)
        self.assertEqual(second.memory_source["area_id"], str(project))
        self.assertEqual(self.storage.uploads, 1)
        self.assertEqual(await self.files.read_content(owner_id=self.f.owner, file_id=first.id), self.raw.image)
        original = await self.f.sources.read(self.f.owner, project, UUID(first.memory_source["source_id"]))
        self.assertEqual(original.source_time, self.frame.captured_at)
        capture = SessionContextCapture(self.history, owner_id=self.f.owner, session_id=self.session,
                                         provider="openai", audio_session_id="saved-scope")
        self.addAsyncCleanup(capture.close)
        receipt = await self.files.capture_tool_result({"file": _file_payload(first)},
            ToolResultCapture(capture, "saved", "capture_latest_visual_frame"))
        async with self.f.pool.acquire() as connection:
            self.assertEqual(await connection.fetchval("""select s.area_id from jarvis_knowledge_context_sources cs
                join jarvis_knowledge_sources s on s.owner_id=cs.owner_id and s.id=cs.source_id
                where cs.owner_id=$1 and cs.event_id=$2""", self.f.owner, receipt.event_id), project)
