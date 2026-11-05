"""Source-backed images through real source, capture, storage and worker boundaries."""
import asyncio
from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import UTC, datetime
import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch
from uuid import UUID, uuid4

from app.images import ImageReference, ImageUnavailableError
from app.knowledge.backfill import LegacyBackfill, NAMESPACE
from app.knowledge.images import SourceImageService
from app.knowledge.image_tools import SourceImageToolModule
from app.knowledge.lifecycle import SourceLifecycleService
from app.knowledge.models import SourcePublication
from app.knowledge.publication import CanonicalSourcePublication
from app.knowledge.read_provenance import SourceReadProvenance
from app.knowledge.selection import configure_selected_memory
from app.realtime.session import RealtimeAudioSession
from app.realtime.source_context import LiveResponseSources
from app.realtime.tool_images import ToolImageRequest
from app.runtime.feature_config import MemoryAuthority
from app.settings import JarvisSettings
from app.tasks.context import TaskContextBuilder
from app.tasks.context_capture import SessionContextCapture
from app.tasks.context_events import ContextIngress
from app.tasks.context_repository import ContextRepository
from app.tasks.models import TaskCreate, TaskError, TaskVoiceSource
from app.tasks.stored_images import StoredFrameImageService, StoredTaskImageFactory
from app.tools.executor import JarvisToolExecutor
from app.tools.module import ToolBuildContext
from app.tools.registry import JarvisToolRegistry
from app.visual_memory.storage import VisualFrameStorageClient
from evals.second_brain.stored_image_fixture import stored_image_fixture
import test_second_brain_stored_images as stored_tests


@unittest.skipUnless(os.getenv("JARVIS_SECOND_BRAIN_TEST_DB_URL") and
                     os.getenv("JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"),
                     "Explicit manifested database required")
class SourceImageDatabaseTests(unittest.IsolatedAsyncioTestCase):
    reference = stored_tests.StoredImageDatabaseTests.reference

    async def asyncSetUp(self):
        await stored_tests.StoredImageDatabaseTests.asyncSetUp(self)
        self.history = ContextRepository(self.tasks, publication=CanonicalSourcePublication(self.f.sources, self.f.area))
        self.images = SourceImageService(self.f.sources, self.service, owner_id=self.f.owner, area_id=self.f.area)
        self.settings = JarvisSettings(_env_file=None, VISUAL_FRAME_MAX_BYTES=512000,
                                      VISUAL_FRAME_MAX_WIDTH=1024, VISUAL_FRAME_MAX_HEIGHT=1024)

    async def original(self, reference=None):
        reference = reference or self.reference(current=True)
        event = await self.history.append(self.f.owner, ContextIngress(
            session_id=self.session, source_namespace="fixture-source-image", source_event_id=str(uuid4()),
            receipt_id=str(uuid4()), kind="observation", trust="sensor", source_time=reference.captured_at,
            payload={"images": [reference.to_metadata()]}))
        async with self.f.pool.acquire() as connection:
            identity = await connection.fetchval(
                "select source_id from jarvis_knowledge_context_sources where owner_id=$1 and event_id=$2",
                self.f.owner, event.id)
        return await self.f.sources.read(self.f.owner, self.f.area, identity)

    async def select(self, source, **kwargs):
        return await self.images.select_image(owner_id=self.f.owner, area_id=self.f.area,
                                              source_id=source.source_id, **kwargs)

    async def corrected(self, source):
        request = await self.history.append(self.f.owner, ContextIngress(
            session_id=self.session, source_namespace="fixture-image-correction", source_event_id=str(uuid4()),
            receipt_id=str(uuid4()), kind="user_transcript", trust="user_input",
            payload={"text": "Correct the earlier observation; do not treat it as current evidence."}))
        await SourceLifecycleService(self.f.sources, self.tasks).correct(
            self.f.owner, self.f.area, source.source_id, expected_revision=source.revision,
            replacement="The previous picture does not establish the present scene.",
            request=TaskVoiceSource(request.id, request.revision, request.session_id))

    async def test_one_connection_pool_does_not_deadlock_image_selection(self):
        from app.knowledge.sources import SourceRepository
        original = await self.original()
        permit = asyncio.Semaphore(1)
        @asynccontextmanager
        async def acquire():
            async with permit, self.f.pool.acquire() as connection:
                yield connection
        pool = SimpleNamespace(acquire=acquire)
        stored = StoredFrameImageService(pool, self.storage, max_bytes=512000, max_width=1024,
                                         max_height=1024, timeout_seconds=1)
        service = SourceImageService(SourceRepository(pool), stored, owner_id=self.f.owner, area_id=self.f.area)
        async with asyncio.timeout(2):
            reference = await service.select_image(owner_id=self.f.owner, area_id=self.f.area,
                                                    source_id=original.source_id)
            resolved = await service.resolve_image(owner_id=self.f.owner, reference=reference)
            await resolved.revalidate()
        self.assertEqual(resolved.image_bytes, self.storage.image)

    async def test_exact_pixels_and_capture_clock_without_legacy_lookup(self):
        source = await self.original()
        with patch("app.memory.retrieval.image_evidence.HistoricalImageService.resolve_image",
                   side_effect=AssertionError("Legacy image authority used")):
            reference = await self.select(source)
            result = await self.images.resolve_image(owner_id=self.f.owner, reference=reference)
        self.assertEqual(result.image_bytes, self.storage.image)
        self.assertEqual(reference.source, "historical")
        self.assertEqual(reference.captured_at, self.frame.captured_at)
        self.assertEqual(reference.record_type, "source_evidence")
        self.assertEqual(reference.source_payload_ids, (reference.record_id,))
        self.assertNotEqual(reference.record_id, source.source_id)
        self.assertNotIn("stream_connection_id", reference.to_metadata())
        self.assertNotIn("storage_path", reference.to_metadata())
        await result.revalidate()

    async def test_explicit_missing_or_foreign_frame_is_not_substituted(self):
        source = await self.original()
        with self.assertRaises(ImageUnavailableError):
            await self.select(source, frame_id=uuid4())
        reference = await self.select(source)
        for changed in (replace(reference, frame_id=uuid4()), replace(reference, width=600),
                        replace(reference, record_id=uuid4()), replace(reference, owner_id=self.f.other)):
            with self.subTest(changed=changed), self.assertRaises(ImageUnavailableError):
                await self.images.resolve_image(owner_id=self.f.owner, reference=changed)

    async def test_arbitrary_user_json_does_not_grant_image_access(self):
        fake = await self.f.publish_payload({"images": [self.reference().to_metadata()]})
        with self.assertRaises(ImageUnavailableError):
            await self.select(fake)

    async def test_correction_blocks_queued_copy_but_explicit_history_can_be_selected(self):
        source = await self.original()
        reference = await self.select(source)
        resolved = await self.images.resolve_image(owner_id=self.f.owner, reference=reference)
        await self.corrected(source)
        with self.assertRaises(ImageUnavailableError):
            await resolved.revalidate()
        with self.assertRaises(ImageUnavailableError):
            await self.select(source)
        historical = await self.select(source, revision=1)
        self.assertNotEqual(historical.record_id, reference.record_id)
        result = await self.images.resolve_image(owner_id=self.f.owner, reference=historical)
        self.assertEqual(result.image_bytes, self.storage.image)
        await self.f.sources.delete_source(self.f.owner, self.f.area, source.source_id, expected_revision=2)
        with self.assertRaises(ImageUnavailableError):
            await result.revalidate()

    async def test_retention_and_storage_changes_still_block_source_authorized_pixels(self):
        reference = await self.select(await self.original())
        resolved = await self.images.resolve_image(owner_id=self.f.owner, reference=reference)
        async with self.f.pool.acquire() as connection:
            await connection.execute("update visual_frames set raw_image_status='deleted' where id=$1", self.frame.id)
        with self.assertRaises(ImageUnavailableError):
            await resolved.revalidate()

    async def test_capture_and_tool_metadata_depend_on_same_source_root(self):
        source = await self.original()
        reference = await self.select(source)
        capture = SessionContextCapture(self.history, owner_id=self.f.owner, session_id=self.session,
                                        provider="openai", audio_session_id="source-image-capture")
        self.addAsyncCleanup(capture.close)
        capture.image_delivery(reference, "provider_acknowledged", "selected-image")
        capture.tool_outcome("image-call", "get_memory_image",
                             {"status": "success", "image_status": "selected", "image": reference.to_metadata()})
        self.assertTrue(await capture.flush())
        async with self.f.pool.acquire() as connection:
            copies = await connection.fetch(
                """select d.dependent_payload_id from jarvis_context_payload_dependencies d
                    join jarvis_context_events e on e.owner_id=d.owner_id and e.payload_id=d.dependent_payload_id
                    where d.owner_id=$1 and d.source_payload_id=$2""", self.f.owner, reference.record_id)
        self.assertEqual(len(copies), 2)
        await self.f.sources.delete_source(self.f.owner, self.f.area, source.source_id, expected_revision=1)
        async with self.f.pool.acquire() as connection:
            for row in copies:
                self.assertNotEqual(await connection.fetchval("select lock_jarvis_copy_source($1,$2)",
                                     self.f.owner, row["dependent_payload_id"]), "available")

    async def test_recorded_parent_revisions_supply_images_without_inferred_relationships(self):
        original = await self.original()
        async def materialize(connection, parents):
            payload = await connection.fetchval(
                """insert into jarvis_context_payloads(owner_id,availability,content)
                    values($1,'available','{"text":"A derived account with recorded image evidence."}'::jsonb) returning id""",
                self.f.owner)
            await connection.executemany("""insert into jarvis_context_payload_dependencies
                (owner_id,source_payload_id,dependent_payload_id) values($1,$2,$3)""",
                [(self.f.owner, parent, payload) for parent in parents])
            child = await self.f.sources.publish_in_transaction(connection, self.f.owner, SourcePublication(
                area_id=self.f.area, namespace="fixture-derived-image", source_key=str(uuid4()),
                payload_id=payload, source_type="derived_account", trust="derived", source_time=None,
                time_basis="No independent event clock", expected_revision=0))
            return child, payload
        child = await SourceReadProvenance(self.f.sources).bind(
            self.f.owner, self.f.area, [original.original_evidence()], materialize)
        reference = await self.select(child)
        self.assertEqual(reference.frame_id, self.frame.id)
        resolved = await self.images.resolve_image(owner_id=self.f.owner, reference=reference)
        await self.corrected(original)
        with self.assertRaises(ImageUnavailableError):
            await resolved.revalidate()

    async def test_actual_imported_frame_survives_retirement_of_legacy_memory_rows(self):
        async with self.f.pool.acquire() as connection:
            memory = await connection.fetchval("""insert into memory_items(
                owner_id,memory_type,source,content,title,is_explicit,status,importance,confidence)
                values($1,'semantic_fact','inferred_observation','Synthetic image claim','Image fixture',false,'active',0.5,0.5)
                returning id""", self.f.owner)
            await connection.execute("""insert into memory_evidence(
                owner_id,memory_item_id,evidence_type,evidence_ref_id,evidence_time,confidence)
                values($1,$2,'frame',$3,clock_timestamp(),0.5)""", self.f.owner, memory, str(self.frame.id))
        report = await LegacyBackfill(self.f.sources).run(self.f.owner, self.f.area)
        self.assertTrue(report["local_manifest_complete"])
        async with self.f.pool.acquire() as connection:
            identity = await connection.fetchval(
                "select id from jarvis_knowledge_sources where owner_id=$1 and namespace=$2 and source_key=$3",
                self.f.owner, NAMESPACE, "memory_items/" + str(memory))
            await connection.execute("delete from memory_evidence where owner_id=$1 and memory_item_id=$2", self.f.owner, memory)
            await connection.execute("delete from memory_items where owner_id=$1 and id=$2", self.f.owner, memory)
        source = await self.f.sources.read(self.f.owner, self.f.area, identity)
        reference = await self.select(source)
        resolved = await self.images.resolve_image(owner_id=self.f.owner, reference=reference)
        self.assertEqual(resolved.image_bytes, self.storage.image)

    async def test_selected_catalog_executor_and_worker_factory_use_source_images(self):
        from app.knowledge.runtime import KnowledgeRuntime
        from app.knowledge.session_areas import SessionKnowledgeAreas
        selections = SessionKnowledgeAreas(self.f.sources, self.tasks, self.f.area)
        self.history = ContextRepository(self.tasks,
            publication=CanonicalSourcePublication(self.f.sources, self.f.area, selections))
        source = await self.original()
        async with self.f.pool.acquire() as connection:
            project = await connection.fetchval(
                "insert into jarvis_knowledge_areas(owner_id,name,domain) values($1,'Another project','robotics') returning id",
                self.f.owner)
        request = await self.history.append(self.f.owner, ContextIngress(session_id=self.session,
            source_namespace="fixture-image-area", source_event_id=str(uuid4()), receipt_id=str(uuid4()),
            kind="user_transcript", trust="user_input", payload={"text": "Switch to another synthetic project."}))
        await selections.select(self.f.owner, project,
            request=TaskVoiceSource(request.id, request.revision, self.session), expected_revision=0)
        runtime = SimpleNamespace(sources=self.f.sources, context_repository=self.history,
            configuration=SimpleNamespace(area_id=self.f.area), context_for=lambda owner: self.history,
            memory_read_backend_for=lambda owner: SimpleNamespace(), area_selections=selections)
        runtime.area_tool_module_for = lambda owner: KnowledgeRuntime.area_tool_module_for(runtime, owner)
        self.settings.JARVIS_OWNER_ID = self.f.owner
        services = SimpleNamespace(memory_authority=MemoryAuthority.HINDSIGHT, knowledge_runtime=runtime,
            settings=self.settings, visual_frame_storage=self.storage, tool_modules=(), assistant_context_providers=())
        configure_selected_memory(services)
        configure_selected_memory(services)
        modules = [module for module in services.tool_modules if isinstance(module, SourceImageToolModule)]
        self.assertEqual(len(modules), 1)
        self.assertIs(modules[0].service.selections, selections)
        registry = JarvisToolRegistry()
        registry.register_many(modules[0].build_tools(ToolBuildContext(self.settings, self.f.owner, self.session)))
        capture = SessionContextCapture(self.history, owner_id=self.f.owner, session_id=self.session,
                                        provider="openai", audio_session_id="source-image-executor")
        self.addAsyncCleanup(capture.close)
        executor = JarvisToolExecutor(registry, owner_id=self.f.owner, session_id=self.session, context_capture=capture)
        result = await executor.execute("get_memory_image", {"memory_id": str(source.source_id)}, provider_call_id="image-tool")
        self.assertEqual(result["status"], "success", result)
        self.assertTrue(await capture.flush())
        resolved = await registry.get("get_memory_image").image_resolver(result)
        capture.image_delivery(resolved.reference, "socket_written", "image-write")
        self.assertTrue(await capture.flush())
        async with self.f.pool.acquire() as connection:
            event_id = await connection.fetchval(
                """select e.id from jarvis_context_events e join jarvis_context_payloads p on p.id=e.payload_id
                    where e.owner_id=$1 and p.content->>'delivery_id'='image-write'""", self.f.owner)
            area = await connection.fetchval("""select s.area_id from jarvis_knowledge_context_sources cs
                join jarvis_knowledge_sources s on s.owner_id=cs.owner_id and s.id=cs.source_id
                where cs.owner_id=$1 and cs.event_id=$2""", self.f.owner, event_id)
        self.assertEqual(area, self.f.area)
        self.assertNotEqual(area, project)
        await self.tasks.create(self.f.owner, TaskCreate(request_id=uuid4(), title="Inspect historical image",
                                requested_outcome="Read the authorized image.", session_id=self.session), model="fixture-model")
        claim = await self.tasks.claim_next("source-image-worker", ("frontier",))
        await TaskContextBuilder(self.history).build(claim)
        factory = StoredTaskImageFactory(self.f.pool, self.storage, self.settings)
        worker_images = await factory(None, self.history)
        selection = await worker_images.select(claim, event_id)
        image = await worker_images.resolve(claim, selection)
        self.assertEqual(image.image_bytes, self.storage.image)
        await self.f.sources.delete_source(self.f.owner, self.f.area, source.source_id, expected_revision=1)
        with self.assertRaises((ImageUnavailableError, TaskError)):
            await image.revalidate()

    async def test_actual_realtime_metadata_write_is_source_fenced(self):
        source = await self.original()
        reference = await self.select(source)
        session = object.__new__(RealtimeAudioSession)
        session.context = SimpleNamespace(owner_id=self.f.owner)
        session._source_context = LiveResponseSources(self.f.pool, owner_id=self.f.owner,
            session_id=self.session, audio_session_id="fixture-image-voice")
        session._context_capture = None
        session._closed = asyncio.Event()
        session._begin_context_rebuild = Mock()
        sent = []
        async def send_event(factory, *, write_guard=None):
            if write_guard is not None:
                async with write_guard():
                    value = factory()
            else:
                value = factory()
            if value is not None:
                sent.append(value)
            return value is not None
        session.openai = SimpleNamespace(send_event=send_event)
        request = ToolImageRequest(connection_epoch=uuid4(), provider_call_id="fixture-image-call",
            source_response_id="response", turn_generation=1, tool_chain_id="chain")
        result = {"status": "success", "image_status": "attached", "image": reference.to_metadata()}
        session._capture_image_delivery(reference, "socket_written", "image")
        self.assertTrue(session._source_context.has_sources)
        self.assertIsNotNone(await session._write_image_tool_output(request, result, lambda: True))
        self.assertEqual(len(sent), 1)
        await self.f.sources.delete_source(self.f.owner, self.f.area, source.source_id, expected_revision=1)
        self.assertIsNone(await session._write_image_tool_output(request, result, lambda: True))
        self.assertEqual(len(sent), 1)
        session._begin_context_rebuild.assert_called_once()

    @unittest.skipUnless(os.getenv("JARVIS_HERMES_PYTHON") and os.getenv("JARVIS_S04_STORAGE_LIVE") == "1",
                         "Explicit actual storage and native worker opt-in required")
    async def test_separate_hermes_worker_reads_two_source_bound_images(self):
        from app.storage.supabase_config import get_supabase_storage_config
        settings = JarvisSettings()
        labels = ["HARBOR " + str(uuid4().int % 90000 + 10000),
                  "CANYON " + str(uuid4().int % 90000 + 10000)]
        async with stored_image_fixture(self.f, self.root, settings, session_id=self.session,
                images=[stored_tests.pixels(label) for label in labels]) as records:
            memory_ids = []
            async with self.f.pool.acquire() as connection:
                for frame in records:
                    memory = await connection.fetchval("""insert into memory_items(
                        owner_id,memory_type,source,content,title,is_explicit,status,importance,confidence)
                        values($1,'semantic_fact','inferred_observation','Synthetic retained image source',
                               'Native image fixture',false,'active',0.5,0.5) returning id""", self.f.owner)
                    memory_ids.append(memory)
                    await connection.execute("""insert into memory_evidence(
                        owner_id,memory_item_id,evidence_type,evidence_ref_id,evidence_time,confidence)
                        values($1,$2,'frame',$3,clock_timestamp(),0.5)""", self.f.owner, memory, str(frame.id))
            report = await LegacyBackfill(self.f.sources).run(self.f.owner, self.f.area)
            self.assertTrue(report["local_manifest_complete"])
            storage = VisualFrameStorageClient(get_supabase_storage_config(settings))
            try:
                raw = StoredFrameImageService(self.f.pool, storage, max_bytes=512000, max_width=1024,
                                               max_height=1024, timeout_seconds=10)
                images = SourceImageService(self.f.sources, raw, owner_id=self.f.owner, area_id=self.f.area)
                events = []
                for memory in memory_ids:
                    async with self.f.pool.acquire() as connection:
                        identity = await connection.fetchval(
                            "select id from jarvis_knowledge_sources where owner_id=$1 and namespace=$2 and source_key=$3",
                            self.f.owner, NAMESPACE, "memory_items/" + str(memory))
                    reference = await images.select_image(owner_id=self.f.owner, area_id=self.f.area, source_id=identity)
                    events.append(await self.history.append(self.f.owner, ContextIngress(
                        session_id=self.session, source_namespace="fixture-native-source-image", source_event_id=str(uuid4()),
                        receipt_id=str(uuid4()), kind="observation", trust="sensor", source_time=reference.captured_at,
                        source_payload_ids=reference.source_payload_ids,
                        payload={"images": [reference.to_metadata()],
                                 "meaning": "Synthetic retained image selection; prior model delivery is not asserted."})))
                await stored_tests.StoredImageDatabaseTests.run_cli(
                    self, labels, records, events, settings, reasoning_effort="medium")
            finally:
                await storage.close()

    @unittest.skipUnless(os.getenv("JARVIS_S04_STORAGE_LIVE") == "1", "Explicit actual storage opt-in required")
    async def test_real_stored_object_through_source_backed_resolution(self):
        from app.storage.supabase_config import get_supabase_storage_config
        settings = JarvisSettings()
        async with stored_image_fixture(self.f, self.root, settings, session_id=self.session,
                images=[stored_tests.pixels("SOURCE " + uuid4().hex[:8])]) as records:
            storage = VisualFrameStorageClient(get_supabase_storage_config(settings))
            try:
                raw = StoredFrameImageService(self.f.pool, storage, max_bytes=512000, max_width=1024,
                                               max_height=1024, timeout_seconds=10)
                images = SourceImageService(self.f.sources, raw, owner_id=self.f.owner, area_id=self.f.area)
                source = await self.original(self.reference(records[0], current=True))
                reference = await images.select_image(owner_id=self.f.owner, area_id=self.f.area, source_id=source.source_id)
                resolved = await images.resolve_image(owner_id=self.f.owner, reference=reference)
                self.assertTrue(resolved.image_bytes.startswith(b"\xff\xd8"))
                await resolved.revalidate()
                await self.f.sources.delete_source(self.f.owner, self.f.area, source.source_id, expected_revision=1)
                with self.assertRaises(ImageUnavailableError):
                    await resolved.revalidate()
            finally:
                await storage.close()


class SourceImageCaptureUnitTests(unittest.IsolatedAsyncioTestCase):
    async def test_malformed_image_receipt_records_gap_without_breaking_later_speech_capture(self):
        from test_second_brain_capture import CaptureRepository
        repository = CaptureRepository()
        capture = SessionContextCapture(repository, owner_id="fixture", session_id=uuid4(),
                                        provider="openai", audio_session_id="malformed-image-fixture")
        self.addAsyncCleanup(capture.close)
        capture.tool_outcome("bad-image", "get_memory_image", {"image": {"record_type": "source_evidence"}})
        self.assertEqual(capture.issue, "context_tool_payload_invalid")
        capture.input_final("later-input", "Synthetic later speech")
        self.assertFalse(await capture.flush())  # The earlier evidence gap remains honestly incomplete.
        self.assertEqual(len(repository.of_kind("context_gap")), 1)
        self.assertEqual(len(repository.of_kind("tool_outcome")), 0)
        self.assertEqual(len(repository.of_kind("user_transcript")), 1)
