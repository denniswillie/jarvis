"""Saved-camera files across selected tools, source images, HTTP and runtime recovery."""
import asyncio
from dataclasses import replace
from datetime import UTC, datetime, timedelta
import json
import os
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch
from uuid import UUID, uuid4

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import httpx

from app.auth.static_bearer import require_static_bearer
from app.errors import JarvisHTTPError
from app.files.router import router
from app.files.tools import FileToolModule
from app.images import ImageReference, ImageUnavailableError
from app.knowledge.files import SourceFileRepository
from app.knowledge.images import SourceImageService
from app.knowledge.lifecycle import SourceLifecycleService
from app.knowledge.models import SourceError
from app.knowledge.publication import CanonicalSourcePublication
from app.knowledge.runtime import KnowledgeRuntime
from app.knowledge.session_areas import SessionKnowledgeAreas
from app.knowledge.selection import configure_selected_memory, UnavailableSelectedFiles
from app.perception.current_image import CurrentImageService
from app.perception.frame_window import RecentVisualFrameWindow, SensoryVisualFrame
from app.runtime.feature_config import MemoryAuthority
from app.settings import JarvisSettings, get_settings
from app.tasks.context import TaskContextBuilder
from app.tasks.context_capture import SessionContextCapture
from app.tasks.context_events import ContextIngress
from app.tasks.context_repository import ContextRepository
from app.tasks.models import TaskCreate, TaskVoiceSource
from app.tasks.stored_images import StoredFrameImageService, StoredTaskImageFactory
from app.tools.executor import JarvisToolExecutor
from app.tools.invocation import bind_tool_invocation
from app.tools.module import ToolBuildContext
from app.tools.registry import JarvisToolRegistry
from app.visual_memory.latest_frame_buffer import LatestVisualFrameBuffer
from app.visual_memory.tools import VisualCaptureToolModule
from evals.second_brain.file_storage_fixture import MemoryStorage
import test_second_brain_stored_images as stored_tests


def window_for(owner, session, frame, pixels):
    window = RecentVisualFrameWindow(max_frames_per_session=4, ttl_seconds=120)
    lease = window.begin_stream(owner, session, frame.visual_session_id)
    window.put(SensoryVisualFrame(frame_id=frame.id, owner_id=owner, session_id=session,
        audio_session_id=None, visual_session_id=frame.visual_session_id, frame_sequence=frame.frame_sequence,
        source="glasses", captured_at=frame.captured_at, server_received_at=frame.server_received_at,
        content_type=frame.content_type, width=frame.width, height=frame.height,
        storage_bucket=frame.storage_bucket, storage_path=frame.storage_path,
        raw_image_expires_at=frame.raw_image_expires_at, image_bytes=pixels), stream_lease=lease)
    current = CurrentImageService(frame_window=window, max_age_seconds=60,
        max_image_bytes=512000, max_width=1024, max_height=1024)
    return window, lease, current


@unittest.skipUnless(os.getenv("JARVIS_SECOND_BRAIN_TEST_DB_URL") and
                     os.getenv("JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"),
                     "Explicit manifested database required")
class SavedFileDatabaseTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        await stored_tests.StoredImageDatabaseTests.asyncSetUp(self)
        self.raw = self.storage
        self.storage = MemoryStorage()
        self.window, self.lease, self.current = window_for(self.f.owner, self.session, self.frame, self.raw.image)
        self.files = SourceFileRepository(self.f.sources, self.storage, owner_id=self.f.owner,
                                         area_id=self.f.area, current_images=self.current)
        original_download = self.raw.download_object
        async def download(**kwargs):
            if kwargs["bucket"] == self.storage.bucket:
                value = await self.storage.read_object(**kwargs)
                if value is None:
                    raise ImageUnavailableError("frame_unavailable", "Synthetic saved object missing")
                return value
            return await original_download(**kwargs)
        self.raw.download_object = download
        self.images = SourceImageService(self.f.sources, self.service, owner_id=self.f.owner, area_id=self.f.area)
        self.settings = JarvisSettings(_env_file=None, JARVIS_OWNER_ID=self.f.owner,
            JARVIS_STATIC_BEARER_TOKEN="synthetic-file-test-only-secret",
            VISUAL_LATEST_CAPTURE_MAX_AGE_SECONDS=60, VISUAL_FRAME_MAX_BYTES=512000,
            VISUAL_FRAME_MAX_WIDTH=1024, VISUAL_FRAME_MAX_HEIGHT=1024)
        self.area_selections = SessionKnowledgeAreas(self.f.sources, self.tasks, self.f.area)
        self.history = ContextRepository(self.tasks, publication=CanonicalSourcePublication(
            self.f.sources, self.f.area, self.area_selections))

    async def save(self, **changes):
        return await self.files.capture_current_frame(**(dict(owner_id=self.f.owner, session_id=self.session,
            filename="saved-view.jpg", description="A saved view, not proof of task completion",
            operation_key="saved-view") | changes))

    async def source(self, file):
        return await self.f.sources.read(self.f.owner, self.f.area, UUID(file.memory_source["source_id"]))

    async def selected(self, file):
        source = await self.source(file)
        return await self.images.select_image(owner_id=self.f.owner, area_id=self.f.area, source_id=source.source_id)

    async def expire_raw(self):
        self.window.end_stream(self.lease)
        async with self.f.pool.acquire() as connection:
            await connection.execute(
                """update visual_frames set raw_image_status='expired',
                    raw_image_expires_at=clock_timestamp()-interval '1 second' where id=$1""", self.frame.id)

    async def request(self):
        record = await self.history.append(self.f.owner, ContextIngress(
            session_id=self.session, source_namespace="saved-file-test", source_event_id=str(uuid4()),
            receipt_id=str(uuid4()), kind="user_transcript", trust="user_input", payload={"text": "Apply test deletion."}))
        return TaskVoiceSource(record.id, record.revision, self.session)

    def services(self, runtime=True):
        value = SimpleNamespace(sources=self.f.sources, context_repository=self.history,
            area_selections=self.area_selections,
            configuration=SimpleNamespace(area_id=self.f.area), context_for=lambda owner: self.history,
            memory_read_backend_for=lambda owner: SimpleNamespace(), set_file_publications=lambda service: None)
        value.area_tool_module_for = lambda owner: KnowledgeRuntime.area_tool_module_for(value, owner)
        return SimpleNamespace(memory_authority=MemoryAuthority.HINDSIGHT, knowledge_runtime=value if runtime else None,
            settings=self.settings, visual_frame_storage=self.raw, jarvis_file_storage=self.storage,
            latest_visual_frame_buffer=LatestVisualFrameBuffer(ttl_seconds=120, frame_window=self.window),
            tool_modules=(FileToolModule(self.files.legacy_repository, self.storage),
                VisualCaptureToolModule(None, LatestVisualFrameBuffer(ttl_seconds=120, frame_window=self.window),
                                        self.files.legacy_repository, self.storage)),
            assistant_context_providers=())

    def application(self, services=None):
        services = services or self.services()
        configure_selected_memory(services)
        app = FastAPI()
        app.include_router(router)
        app.state.runtime_services = services
        app.dependency_overrides[get_settings] = lambda: self.settings
        @app.exception_handler(JarvisHTTPError)
        async def error_handler(request: Request, error: JarvisHTTPError):
            return JSONResponse({"error": {"code": error.code, "message": error.message}}, status_code=error.status_code)
        return app

    async def test_saved_pixels_survive_raw_expiry_without_refreshing_capture_time(self):
        file = await self.save()
        source = await self.source(file)
        self.assertEqual(source.source_time, self.frame.captured_at)
        self.assertEqual(source.trust.value, "tool_receipt")
        self.assertEqual(source.content["pixel_origin"], "sensor")
        self.assertEqual(source.content["description_origin"], "assistant_output")
        self.assertEqual(source.content["image"]["retained_file_id"], str(file.id))
        await self.expire_raw()
        reference = await self.selected(file)
        self.assertEqual(reference.captured_at, self.frame.captured_at)
        self.assertIsNone(reference.raw_image_expires_at)
        self.assertEqual(reference.retained_file_id, file.id)
        self.assertEqual(reference.source, "historical")
        resolved = await self.images.resolve_image(owner_id=self.f.owner, reference=reference)
        self.assertEqual(resolved.image_bytes, self.raw.image)
        await resolved.revalidate()
        with self.assertRaises(ImageUnavailableError):
            await self.service.resolve_image(owner_id=self.f.owner, reference=self.images._snapshot(reference))

    async def test_replay_after_camera_disconnect_uses_the_saved_frame(self):
        first = await self.save()
        await self.expire_raw()
        replay = await self.save()
        self.assertEqual(first.id, replay.id)
        self.assertEqual(self.storage.uploads, 1)
        with self.assertRaises(SourceError):
            await self.save(description="Different request under the same identity")
        await self.files.delete_source_file(owner_id=self.f.owner, file_id=first.id)
        with self.assertRaises(SourceError):
            await self.save()
        self.assertEqual(self.storage.uploads, 1)

    async def test_lost_upload_ack_recovers_after_camera_disconnect(self):
        self.storage.mode = "fail_after"
        with self.assertRaises(RuntimeError):
            await self.save()
        await self.expire_raw()
        self.assertIsNotNone((await self.save()).memory_source)
        self.assertEqual(self.storage.uploads, 1)

    async def test_no_stream_stale_or_foreign_capture_does_not_upload(self):
        self.window.end_stream(self.lease)
        with self.assertRaises(ImageUnavailableError):
            await self.save()
        self.window, self.lease, current = window_for(self.f.owner, self.session, self.frame, self.raw.image)
        self.files.current_images = CurrentImageService(frame_window=self.window, max_age_seconds=0.000001,
            max_image_bytes=512000, max_width=1024, max_height=1024)
        with self.assertRaises(ImageUnavailableError):
            await self.save()
        with self.assertRaises(SourceError):
            await self.save(owner_id=self.f.other)
        self.assertEqual(self.storage.uploads, 0)

    async def test_reconnect_during_selection_cannot_save_old_stream_pixels(self):
        original = self.current.select_current_image
        def select(**kwargs):
            reference = original(**kwargs)
            self.window.begin_stream(self.f.owner, self.session, self.frame.visual_session_id)
            return reference
        with patch.object(self.current, "select_current_image", side_effect=select):
            with self.assertRaises(ImageUnavailableError):
                await self.save()
        self.assertEqual(self.storage.uploads, 0)

    async def test_deleted_source_invalidates_queued_pixels_and_reconciles_object(self):
        file = await self.save()
        reference = await self.selected(file)
        resolved = await self.images.resolve_image(owner_id=self.f.owner, reference=reference)
        await self.f.sources.delete_source(self.f.owner, self.f.area, UUID(file.memory_source["source_id"]), expected_revision=1)
        with self.assertRaises(ImageUnavailableError):
            await resolved.revalidate()
        self.assertEqual([r["phase"] for r in await self.files.publications.reconcile_pending(owner_id=self.f.owner)], ["deleted"])
        self.assertFalse(self.storage.objects)

    async def test_forged_file_or_source_reference_never_selects_other_pixels(self):
        file = await self.save()
        reference = await self.selected(file)
        for changed in (replace(reference, retained_file_id=uuid4()), replace(reference, frame_id=uuid4()),
                        replace(reference, record_id=uuid4()), replace(reference, owner_id=self.f.other)):
            with self.subTest(ref=changed), self.assertRaises(ImageUnavailableError):
                await self.images.resolve_image(owner_id=self.f.owner, reference=changed)
        original = await self.source(file)
        forged = await self.f.publish_payload(original.content, source_type="saved_visual_frame", trust="tool_receipt")
        with self.assertRaises(ImageUnavailableError):
            await self.images.select_image(owner_id=self.f.owner, area_id=self.f.area, source_id=forged.source_id)

    async def test_corrupt_saved_object_is_not_served(self):
        file = await self.save()
        reference = await self.selected(file)
        self.storage.objects[(file.storage_bucket, file.storage_path)] = stored_tests.pixels("Different")
        with self.assertRaises((ImageUnavailableError, SourceError)):
            await self.images.resolve_image(owner_id=self.f.owner, reference=reference)
        with self.assertRaises(SourceError):
            await self.files.read_content(owner_id=self.f.owner, file_id=file.id)

    async def test_selected_file_and_capture_modules_share_source_adapter_and_restore_legacy(self):
        services = self.services()
        configure_selected_memory(services)
        configure_selected_memory(services)
        file_module = next(m for m in services.tool_modules if isinstance(m, FileToolModule))
        capture_module = next(m for m in services.tool_modules if isinstance(m, VisualCaptureToolModule))
        self.assertIsInstance(file_module.repository, SourceFileRepository)
        self.assertIs(file_module.repository, capture_module.file_repository)
        registry = JarvisToolRegistry()
        registry.register_many(capture_module.build_tools(ToolBuildContext(self.settings, self.f.owner, self.session)))
        capture = SessionContextCapture(self.history, owner_id=self.f.owner, session_id=self.session,
                                        provider="openai", audio_session_id="saved-file")
        self.addAsyncCleanup(capture.close)
        executor = JarvisToolExecutor(registry, owner_id=self.f.owner, session_id=self.session, context_capture=capture)
        result = await executor.execute("capture_latest_visual_frame", {"description": "Saved test view"},
                                        provider_call_id="save-current")
        self.assertEqual(result["status"], "success", result)
        self.assertIn("memory_source", result["file"])
        self.assertTrue(await capture.flush())
        async with self.f.pool.acquire() as connection:
            self.assertGreater(await connection.fetchval(
                "select count(*) from jarvis_context_memory_sources where owner_id=$1", self.f.owner), 0)
        services.memory_authority = MemoryAuthority.LEGACY
        configure_selected_memory(services)
        self.assertIs(next(m.repository for m in services.tool_modules if isinstance(m, FileToolModule)),
                      self.files.legacy_repository)

    async def test_unavailable_selected_runtime_never_falls_back_to_old_save(self):
        services = self.services(runtime=False)
        configure_selected_memory(services)
        adapter = next(m.repository for m in services.tool_modules if isinstance(m, FileToolModule))
        self.assertIsInstance(adapter, UnavailableSelectedFiles)
        with self.assertRaises(SourceError):
            await adapter.capture_current_frame(owner_id=self.f.owner)
        self.assertEqual(self.storage.uploads, 0)

    async def test_http_requires_real_bearer_and_returns_safe_literal_filename(self):
        file = await self.save(filename="saved?view#name\r\n.jpg")
        app = self.application()
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://fixture") as client:
            path = f"/files/{file.id}/content"
            self.assertEqual((await client.get(path)).status_code, 401)
            self.assertEqual((await client.get(path, headers={"Authorization": "Bearer wrong"})).status_code, 403)
            auth = {"Authorization": "Bearer " + self.settings.JARVIS_STATIC_BEARER_TOKEN}
            response = await client.get(path, headers=auth)
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(response.content, self.raw.image)
            self.assertIn("no-store", response.headers["cache-control"])
            self.assertEqual(response.headers["vary"], "Authorization")
            self.assertIn("%0D%0A", response.headers["content-disposition"])
            self.assertEqual(response.headers["x-content-type-options"], "nosniff")
            metadata = await client.get(f"/files/{file.id}", headers=auth)
            self.assertNotIn("storage_path", metadata.json()["file"])
            self.assertEqual((await client.get(f"/files/{uuid4()}/content", headers=auth)).status_code, 404)
            deleted = await client.delete(f"/files/{file.id}", headers=auth)
            self.assertEqual(deleted.status_code, 200, deleted.text)
            self.assertTrue(deleted.json()["deleted"])
            self.assertEqual((await client.get(path, headers=auth)).status_code, 404)

    async def test_http_read_is_revoked_if_source_deleted_during_storage_fetch(self):
        file = await self.save()
        original_read = self.storage.read_object
        async def read(**kwargs):
            value = await original_read(**kwargs)
            await self.f.sources.delete_source(self.f.owner, self.f.area, UUID(file.memory_source["source_id"]), expected_revision=1)
            return value
        self.storage.read_object = read
        app = self.application()
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://fixture") as client:
            response = await client.get(f"/files/{file.id}/content",
                headers={"Authorization": "Bearer " + self.settings.JARVIS_STATIC_BEARER_TOKEN})
        self.assertGreaterEqual(response.status_code, 400)
        self.assertNotEqual(response.content, self.raw.image)

    async def test_worker_image_factory_reads_saved_source_after_raw_expiry(self):
        file = await self.save()
        reference = await self.selected(file)
        await self.expire_raw()
        factory = StoredTaskImageFactory(self.f.pool, self.raw, self.settings)
        event = await self.history.append(self.f.owner, ContextIngress(
            session_id=self.session, source_namespace="file-image-selection", source_event_id=str(uuid4()),
            receipt_id=str(uuid4()), kind="tool_outcome", trust="tool_receipt", source_time=reference.captured_at,
            source_payload_ids=reference.source_payload_ids,
            payload={"result": {"status": "success", "image_status": "selected", "image": reference.to_metadata()}}))
        await self.tasks.create(self.f.owner, TaskCreate(request_id=uuid4(), title="Read saved image",
            requested_outcome="Read this historical saved picture", session_id=self.session), model="fixture-model")
        claim = await self.tasks.claim_next("saved-file-worker", ("frontier",))
        await TaskContextBuilder(self.history).build(claim)
        worker_images = await factory(None, self.history)
        selection = await worker_images.select(claim, event.id)
        resolved = await worker_images.resolve(claim, selection)
        self.assertEqual(resolved.image_bytes, self.raw.image)
        self.assertFalse(selection["current_view_verified"])



    async def test_selected_save_returns_specific_camera_unavailability(self):
        self.window.end_stream(self.lease)
        services = self.services()
        configure_selected_memory(services)
        module = next(m for m in services.tool_modules if isinstance(m, VisualCaptureToolModule))
        registry = JarvisToolRegistry()
        registry.register_many(module.build_tools(ToolBuildContext(self.settings, self.f.owner, self.session)))
        executor = JarvisToolExecutor(registry, owner_id=self.f.owner, session_id=self.session)
        result = await executor.execute("capture_latest_visual_frame", {"description": "Save unavailable view"},
                                        provider_call_id="unavailable-save")
        self.assertNotEqual(result["status"], "success", result)
        self.assertIn("no_active_stream", json.dumps(result))
        self.assertEqual(self.storage.uploads, 0)


class FileRecoveryRuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_reconciliation_failure_is_visible_and_runtime_shutdown_waits_for_it(self):
        sources = SimpleNamespace(pool=object())
        config = SimpleNamespace(area_id=uuid4(), worker_lease_seconds=10, native_inflight_limit=1,
                                 worker_deadline_seconds=10, poll_seconds=0.01, shutdown_seconds=1)
        # Exercise the actual runtime loop without running a model or database worker.
        runtime = KnowledgeRuntime.__new__(KnowledgeRuntime)
        runtime.owner_id, runtime.sources, runtime.configuration = "synthetic-owner", sources, config
        runtime.stop, runtime._closed = asyncio.Event(), False
        runtime.file_issue, runtime.file_task, runtime.task = None, None, None
        runtime.retention_task = None
        runtime.client = SimpleNamespace(close=AsyncMock())
        ran = asyncio.Event()
        async def reconcile(**kwargs):
            ran.set()
            raise RuntimeError("Synthetic storage failure")
        service = SimpleNamespace(sources=sources, owner_id=runtime.owner_id, area_id=config.area_id,
                                  reconcile_pending=reconcile)
        runtime.set_file_publications(service)
        runtime.file_task = asyncio.create_task(runtime._reconcile_files())
        await asyncio.wait_for(ran.wait(), 1)
        await asyncio.sleep(0)
        self.assertEqual(runtime.file_issue, "file_reconciliation_failed")
        await runtime.close()
        self.assertTrue(runtime.file_task.done())
        runtime.client.close.assert_awaited_once()
        with self.assertRaises(ValueError):
            runtime.set_file_publications(service)


@unittest.skipUnless(os.getenv("JARVIS_FILE_PUBLICATION_LIVE") == "1" and
                     os.getenv("JARVIS_S04_STORAGE_LIVE") == "1" and os.getenv("JARVIS_HERMES_PYTHON") and
                     os.getenv("JARVIS_SECOND_BRAIN_TEST_DB_URL") and os.getenv("JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"),
                     "Explicit native saved-file storage and Hermes opt-in required")
class NativeSavedFileTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        from test_second_brain_file_publication import NativeFilePublicationTests
        from app.runtime.feature_config import TaskLimits
        from app.tasks.repository import TaskRepository
        from pathlib import Path
        await NativeFilePublicationTests.asyncSetUp(self)
        self.root = Path(os.environ["JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"])
        self.tasks = TaskRepository(self.f.pool, TaskLimits(lease_seconds=300))
        self.history = ContextRepository(self.tasks)

    async def test_real_hermes_reads_saved_files_after_raw_objects_are_removed(self):
        from app.storage.supabase_config import get_supabase_storage_config
        from app.visual_memory.storage import VisualFrameStorageClient
        from evals.second_brain.stored_image_fixture import stored_image_fixture
        settings = JarvisSettings()
        labels = ["HARBOR " + str(uuid4().int % 90000 + 10000), "CANYON " + str(uuid4().int % 90000 + 10000)]
        contents = [stored_tests.pixels(label) for label in labels]
        files = []
        repository = SourceFileRepository(self.f.sources, self.storage, owner_id=self.f.owner, area_id=self.f.area)
        async with stored_image_fixture(self.f, self.root, settings, session_id=self.session, images=contents) as records:
            for index, frame in enumerate(records):
                _, _, repository.current_images = window_for(self.f.owner, self.session, frame, contents[index])
                file = await repository.capture_current_frame(owner_id=self.f.owner, session_id=self.session,
                    filename=f"saved-native-{index}.jpg", description="Synthetic saved picture; no label supplied in text.",
                    operation_key=f"native-saved-{index}")
                files.append(file)
        # The raw fixture has already deleted and independently verified absence
        # of its objects. Subsequent reads must use the separately retained files.
        async with self.f.pool.acquire() as connection:
            states = await connection.fetch("select raw_image_status from visual_frames where id=any($1::uuid[])",
                                            [frame.id for frame in records])
        self.assertTrue(all(row["raw_image_status"] != "available" for row in states))
        storage = VisualFrameStorageClient(get_supabase_storage_config(settings))
        self.addAsyncCleanup(storage.close)
        raw = StoredFrameImageService(self.f.pool, storage, max_bytes=512000, max_width=1024,
                                      max_height=1024, timeout_seconds=10)
        images = SourceImageService(self.f.sources, raw, owner_id=self.f.owner, area_id=self.f.area)
        events = []
        for index, file in enumerate(files):
            reference = await images.select_image(owner_id=self.f.owner, area_id=self.f.area,
                                                   source_id=UUID(file.memory_source["source_id"]))
            self.assertEqual(reference.retained_file_id, file.id)
            self.assertEqual((await images.resolve_image(owner_id=self.f.owner, reference=reference)).image_bytes, contents[index])
            events.append(await self.history.append(self.f.owner, ContextIngress(
                session_id=self.session, source_namespace="fixture-native-saved-file", source_event_id=str(uuid4()),
                receipt_id=str(uuid4()), kind="tool_outcome", trust="tool_receipt", source_time=reference.captured_at,
                source_payload_ids=reference.source_payload_ids, payload={
                    "result": {"status": "success", "image_status": "selected", "image": reference.to_metadata()},
                    "meaning": "Synthetic saved image. The temporary original no longer exists; no live view is asserted."})))
        await stored_tests.StoredImageDatabaseTests.run_cli(self, labels, records, events, settings, reasoning_effort="medium")
        self.assertEqual(len(self.manifest["objects"]), 2)
        for file in files:
            self.assertTrue((await repository.delete_source_file(owner_id=self.f.owner, file_id=file.id))["deleted"])
