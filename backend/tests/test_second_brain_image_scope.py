"""Retained-image scope survives topic changes, queued capture and provider ACKs."""
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
import os
import unittest
from unittest.mock import AsyncMock
from uuid import uuid4

from app.images import ImageReference, ImageUnavailableError
from app.knowledge.files import SourceFileRepository
from app.knowledge.image_tools import SourceImageToolModule
from app.knowledge.images import SourceImageService
from app.knowledge.models import SourceError
from app.knowledge.publication import CanonicalSourcePublication
from app.tasks.context_capture import SessionContextCapture
from app.tasks.context_events import ContextIngress
from app.tools.module import ToolBuildContext


class ImageScopeContractTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.owner, self.area, self.project, self.session, self.root = "image-scope-fixture", uuid4(), uuid4(), uuid4(), uuid4()
        self.sources = SimpleNamespace(pool=object(), read=AsyncMock(side_effect=SourceError("fixture_stop", "Synthetic read boundary")))
        self.selections = SimpleNamespace(sources=self.sources, default_area_id=self.area,
            source_area=AsyncMock(return_value=self.project), bind_context=AsyncMock())
        self.service = SourceImageService(self.sources, SimpleNamespace(timeout_seconds=1),
            owner_id=self.owner, area_id=self.area, selections=self.selections)
        now = datetime.now(UTC)
        self.reference = ImageReference(owner_id=self.owner, session_id=self.session, frame_id=uuid4(),
            source="historical", captured_at=now, content_type="image/jpeg", width=8, height=8,
            raw_image_expires_at=now+timedelta(hours=1), record_type="source_evidence", record_id=self.root)
        self.connection = SimpleNamespace(fetch=AsyncMock(return_value=[{"root": self.root, "area_id": self.project}]))
        self.publisher = CanonicalSourcePublication(self.sources, self.area, self.selections)

    def record(self, **changes):
        return SimpleNamespace(**(dict(owner_id=self.owner, is_current=True, availability="available",
            kind="observation", trust="sensor", payload_id=uuid4(),
            content={"images": [self.reference.to_metadata()]}) | changes))

    async def test_observation_and_tool_result_use_the_recorded_dependency_area(self):
        observation = self.record()
        tool = self.record(kind="tool_outcome", trust="tool_receipt", content={"result": {
            "status": "success", "image_status": "selected", "image": self.reference.to_metadata()}})
        for record in (observation, tool):
            self.assertEqual(await self.publisher._image_area(self.connection, self.owner, record), self.project)
            self.assertEqual(self.connection.fetch.call_args.args[1:], (self.owner, record.payload_id, [self.root]))

    async def test_missing_and_mixed_recorded_dependencies_are_rejected(self):
        for rows in ([], [{"root": uuid4(), "area_id": self.project}],
                     [{"root": self.root, "area_id": self.project}, {"root": self.root, "area_id": self.area}]):
            self.connection.fetch.return_value = rows
            with self.assertRaises(SourceError) as caught:
                await self.publisher._image_area(self.connection, self.owner, self.record())
            self.assertEqual(caught.exception.code, "knowledge_image_scope_unavailable")

    async def test_user_content_and_raw_camera_metadata_do_not_choose_scope(self):
        raw = self.reference.to_metadata() | {"record_type": None, "record_id": None}
        records = [self.record(kind="user_transcript", trust="user_input"),
                   self.record(content={"images": [raw]}), self.record(availability="deleted"),
                   self.record(kind="tool_outcome", trust="tool_receipt", content={"result": {"status": "failed"}})]
        for record in records:
            self.assertIsNone(await self.publisher._image_area(self.connection, self.owner, record))
        self.connection.fetch.assert_not_awaited()

    async def test_malformed_retained_reference_is_not_used_as_scope(self):
        with self.assertRaises(SourceError) as caught:
            await self.publisher._image_area(self.connection, self.owner,
                self.record(content={"images": [{"record_type": "source_evidence", "record_id": "not-a-uuid"}]}))
        self.assertEqual(caught.exception.code, "knowledge_image_scope_invalid")

    async def test_explicit_scope_cannot_override_an_image_proof(self):
        with self.assertRaises(SourceError) as caught:
            await self.publisher.publish(self.connection, self.owner, self.record(), requested_area_id=self.area)
        self.assertEqual(caught.exception.code, "knowledge_context_scope_conflict")
        self.selections.bind_context.assert_not_awaited()

    async def test_tool_binds_trusted_session_and_reads_exact_original_area(self):
        identity = uuid4()
        tool = SourceImageToolModule(self.service, self.owner, self.area).build_tools(
            ToolBuildContext(None, self.owner, self.session))[0]
        result = await tool.handler({"memory_id": str(identity)})
        self.assertEqual(result["image_status"], "unavailable")
        self.selections.source_area.assert_awaited_once_with(self.owner, self.session, identity)
        self.sources.read.assert_awaited_once_with(self.owner, self.project, identity, revision=None)
        self.assertIsNone(self.service.session_id)

    def test_fixed_scopes_and_owner_fences_remain(self):
        fixed = SourceImageService(self.sources, object(), owner_id=self.owner, area_id=self.area)
        self.assertIs(fixed.for_session(self.session), fixed)
        with self.assertRaises(ImageUnavailableError):
            fixed._authorize(self.owner, self.project)
        with self.assertRaises(ImageUnavailableError):
            self.service._authorize("foreign", self.project)
        with self.assertRaises(ValueError):
            self.service.for_session("not-a-trusted-uuid")
        with self.assertRaises(ValueError):
            SourceImageService(self.sources, object(), owner_id=self.owner, area_id=uuid4(), selections=self.selections)


@unittest.skipUnless(os.getenv("JARVIS_SECOND_BRAIN_TEST_DB_URL") and
                     os.getenv("JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"),
                     "Explicit manifested database required")
class ImageScopeDatabaseTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        from test_second_brain_saved_files import SavedFileDatabaseTests
        await SavedFileDatabaseTests.asyncSetUp(self)
        self.selections = self.area_selections
        async with self.f.pool.acquire() as connection:
            self.project = await connection.fetchval(
                "insert into jarvis_knowledge_areas(owner_id,name,domain) values($1,'Image bench','robotics') returning id", self.f.owner)
        self.root_files = SourceFileRepository(self.f.sources, self.storage, owner_id=self.f.owner,
            area_id=self.f.area, selections=self.selections, current_images=self.current)
        self.files = self.root_files.for_session(self.session)
        self.root_images = SourceImageService(self.f.sources, self.service, owner_id=self.f.owner,
            area_id=self.f.area, selections=self.selections)
        self.scoped_images = self.root_images.for_session(self.session)

    async def select(self, area):
        from app.tasks.models import TaskVoiceSource
        event = await self.history.append(self.f.owner, ContextIngress(session_id=self.session,
            source_namespace="voice:image-scope", source_event_id=str(uuid4()), receipt_id=str(uuid4()),
            kind="user_transcript", trust="user_input", payload={"text": "Select a synthetic knowledge area."}))
        before = await self.selections.read(self.f.owner, self.session)
        await self.selections.select(self.f.owner, area, request=TaskVoiceSource(event.id, event.revision, self.session),
                                     expected_revision=before["selection_revision"])

    async def saved_reference(self, key):
        from uuid import UUID
        file = await self.files.capture_current_frame(owner_id=self.f.owner, session_id=self.session,
            filename=key+".jpg", description="Synthetic historical evidence", operation_key=key)
        reference = await self.scoped_images.select_image(owner_id=self.f.owner, area_id=self.f.area,
                                                          source_id=UUID(file.memory_source["source_id"]))
        return file, reference

    async def test_selection_pixels_delayed_delivery_and_deletion_keep_original_area(self):
        await self.select(self.project)
        file, reference = await self.saved_reference("project-view")
        await self.select(self.f.area)
        tool = SourceImageToolModule(self.root_images, self.f.owner, self.f.area).build_tools(
            ToolBuildContext(self.settings, self.f.owner, self.session))[0]
        result = await tool.handler({"memory_id": file.memory_source["source_id"]})
        self.assertEqual(result["image_status"], "selected", result)
        resolved = await tool.image_resolver(result)
        self.assertEqual(resolved.image_bytes, self.raw.image)
        self.assertEqual(resolved.reference.captured_at, self.frame.captured_at)
        await resolved.revalidate()
        capture = SessionContextCapture(self.history, owner_id=self.f.owner, session_id=self.session,
            provider="openai", audio_session_id="image-scope", write_timeout=10)
        self.addAsyncCleanup(capture.close)
        capture.tool_outcome("read-image", "get_memory_image", result)
        capture.image_delivery(resolved.reference, "socket_written", "provider-image")
        self.assertTrue(await capture.flush(), capture.issue)
        capture.image_delivery(resolved.reference, "provider_acknowledged", "provider-image")
        self.assertTrue(await capture.flush(), capture.issue)
        async with self.f.pool.acquire() as connection:
            rows = await connection.fetch("""select s.area_id,e.kind,e.source_time from jarvis_context_events e
                join jarvis_knowledge_context_sources cs on cs.owner_id=e.owner_id and cs.event_id=e.id
                join jarvis_knowledge_sources s on s.owner_id=cs.owner_id and s.id=cs.source_id
                where e.owner_id=$1 and e.source_namespace=$2 order by e.sequence""", self.f.owner, capture.namespace)
        self.assertEqual(len(rows), 3)
        self.assertEqual({r["area_id"] for r in rows}, {self.project})
        self.assertTrue(all(r["source_time"] == self.frame.captured_at for r in rows if r["kind"] == "observation"))
        self.assertTrue((await self.files.delete_source_file(owner_id=self.f.owner, file_id=file.id))["deleted"])
        with self.assertRaises(ImageUnavailableError):
            await resolved.revalidate()
        self.assertEqual((await tool.handler({"memory_id": file.memory_source["source_id"]}))["image_status"], "unavailable")
        capture.image_delivery(reference, "provider_acknowledged", "late-deleted-image")
        self.assertTrue(await capture.flush(), capture.issue)
        async with self.f.pool.acquire() as connection:
            self.assertEqual(await connection.fetchval("select count(*) from jarvis_context_events where owner_id=$1 and source_namespace=$2",
                self.f.owner, capture.namespace), 3)
        self.assertFalse(self.storage.objects)

    async def test_unlinked_mixed_and_overridden_image_proofs_cannot_choose_scope(self):
        await self.select(self.project)
        _, project = await self.saved_reference("project-proof")
        await self.select(self.f.area)
        _, default = await self.saved_reference("default-proof")
        def event(images, roots=()):
            return ContextIngress(session_id=self.session, source_namespace="fixture:image-proof",
                source_event_id=str(uuid4()), receipt_id=str(uuid4()), kind="observation", trust="sensor",
                payload={"images": [r.to_metadata() for r in images]}, source_payload_ids=roots)
        with self.assertRaises(SourceError):
            await self.history.append(self.f.owner, event([project]))
        with self.assertRaises(SourceError):
            await self.history.append(self.f.owner, event([project, default],
                (*project.source_payload_ids, *default.source_payload_ids)))
        with self.assertRaises(SourceError):
            async with self.history.publication.transaction(self.tasks, self.f.owner) as connection:
                await self.history.append_in_transaction(connection, self.f.owner,
                    event([project], project.source_payload_ids), publication_area_id=self.f.area)
        async with self.f.pool.acquire() as connection:
            self.assertEqual(await connection.fetchval("select count(*) from jarvis_context_events where owner_id=$1 and source_namespace=$2",
                self.f.owner, "fixture:image-proof"), 0)
