"""Durable file effects using real source/task SQL and adversarial storage."""
import asyncio
from contextlib import asynccontextmanager
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from uuid import uuid4

import asyncpg
import httpx

from app.errors import JarvisHTTPError
from app.files.models import StoredFileObject
from app.files.publication import FilePublicationService, FilePublicationSpec, MAX_FILE_BYTES
from app.files.repository import JarvisFileRepository
from app.files.storage import JarvisFileStorageClient
from app.knowledge.models import SourceError, SourcePublication
from app.knowledge.sources import SourceRepository
from evals.second_brain.source_fixture import source_fixture


from evals.second_brain.file_storage_fixture import MemoryStorage


class FileStorageTransportTests(unittest.IsolatedAsyncioTestCase):
    async def client(self, handler):
        config = SimpleNamespace(supabase_url="https://fixture.invalid", service_role_key="fixture-key", files_bucket="files")
        storage = JarvisFileStorageClient(config)
        await storage.close()
        storage._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        self.addAsyncCleanup(storage.close)
        return storage

    async def test_upload_and_read_encode_literal_reserved_filename_characters(self):
        seen = []
        def handler(request):
            seen.append(request)
            return httpx.Response(200, content=b"literal")
        storage = await self.client(handler)
        path = "owners/test/a?b#c%name.txt"
        await storage.upload_bytes(path=path, content=b"literal", content_type="text/plain")
        self.assertEqual(await storage.read_object(bucket="files", path=path, max_bytes=7), b"literal")
        for request in seen:
            if request.method == "POST":
                self.assertFalse(request.url.query)
            else:
                self.assertEqual(set(request.url.params), {"jarvis_file_read"})
            self.assertFalse(request.url.fragment)
            self.assertIn(b"a%3Fb%23c%25name.txt", request.url.raw_path)

    async def test_each_availability_probe_has_a_fresh_cache_identity(self):
        identities = []
        def handler(request):
            identities.append(request.url.params.get("jarvis_file_read"))
            return httpx.Response(404, json={"code": "NoSuchKey", "statusCode": "404"})
        storage = await self.client(handler)
        for _ in range(2):
            self.assertIsNone(await storage.read_object(bucket="files", path="same-object", max_bytes=0))
        self.assertTrue(all(identities))
        self.assertNotEqual(identities[0], identities[1])

    async def test_only_exact_missing_object_is_absent(self):
        for status, body, absent in (
            (400, {"code": "NoSuchKey", "statusCode": "404"}, True),
            (404, {"code": "NoSuchKey", "statusCode": 404}, True),
            (404, {"code": "NoSuchBucket", "statusCode": 404}, False),
            (403, {"code": "NoSuchKey", "statusCode": 404}, False),
            (404, {"message": "not found"}, False),
        ):
            with self.subTest(status=status, body=body):
                storage = await self.client(lambda request: httpx.Response(status, json=body))
                if absent:
                    self.assertIsNone(await storage.read_object(bucket="files", path="test", max_bytes=10))
                else:
                    with self.assertRaises(JarvisHTTPError):
                        await storage.read_object(bucket="files", path="test", max_bytes=10)

    async def test_redirects_and_async_success_are_not_completed_storage_receipts(self):
        for status in (202, 204, 206, 302, 307):
            with self.subTest(status=status):
                storage = await self.client(lambda request: httpx.Response(status, content=b""))
                with self.assertRaises(JarvisHTTPError):
                    await storage.upload_bytes(path="file", content=b"", content_type="text/plain")
                with self.assertRaises(JarvisHTTPError):
                    await storage.read_object(bucket="files", path="file", max_bytes=0)

    async def test_read_bound_is_enforced_and_unknown_errors_are_not_absence(self):
        storage = await self.client(lambda request: httpx.Response(200, content=b"x" * 20))
        with self.assertRaises(JarvisHTTPError):
            await storage.read_object(bucket="files", path="test", max_bytes=10)
        storage = await self.client(lambda request: httpx.Response(404, content=b"x" * 3000))
        with self.assertRaises(JarvisHTTPError):
            await storage.read_object(bucket="files", path="test", max_bytes=10)


@unittest.skipUnless(os.getenv("JARVIS_SECOND_BRAIN_TEST_DB_URL") and
                     os.getenv("JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"),
                     "Explicit manifested database required")
class FilePublicationDatabaseTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.f = await self.enterAsyncContext(source_fixture(
            os.environ["JARVIS_SECOND_BRAIN_TEST_DB_URL"],
            Path(os.environ["JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"]), test_name=self.id()))
        async with self.f.pool.acquire() as connection:
            self.session = await connection.fetchval(
                "insert into jarvis_sessions(owner_id) values($1) returning id", self.f.owner)
            self.foreign = await connection.fetchval(
                "insert into jarvis_sessions(owner_id) values($1) returning id", self.f.other)
        self.storage = MemoryStorage()
        self.service = FilePublicationService(self.f.sources, self.storage, owner_id=self.f.owner, area_id=self.f.area)
        self.repository = JarvisFileRepository(self.f.pool)
        self.spec = FilePublicationSpec(session_id=self.session, filename="unfinished-robot.txt",
            content_type="text/plain; charset=utf-8", description="Synthetic unfinished work", source="assistant")
        self.content = b"Calibration is not complete.\n"

    async def publish(self, **changes):
        return await self.service.publish(**(dict(owner_id=self.f.owner, operation_key="same-file",
            spec=self.spec, content=self.content) | changes))

    async def operation(self, key="same-file"):
        async with self.f.pool.acquire() as connection:
            return await connection.fetchrow(
                "select * from jarvis_file_publications where owner_id=$1 and operation_key=$2", self.f.owner, key)

    async def test_stable_adoption_replay_read_and_deletion(self):
        first = await self.publish()
        second = await self.publish()
        self.assertEqual((first.phase, second.phase), ("available", "available"))
        self.assertEqual(first.file_id, second.file_id)
        self.assertEqual(first.file.id, first.file_id)
        self.assertIn(str(first.file_id), first.file.storage_path)
        self.assertEqual(self.storage.uploads, 1)
        self.assertEqual(await self.service.read_content(owner_id=self.f.owner, file_id=first.file_id), self.content)
        deleted = await self.service.delete(owner_id=self.f.owner, operation_id=first.operation_id)
        self.assertEqual(deleted.phase, "deleted")
        self.assertIsNone(await self.repository.get_file(owner_id=self.f.owner, file_id=first.file_id))
        self.assertFalse(self.storage.objects)
        self.assertIsNone((await self.operation())["storage_path"])
        self.assertEqual((await self.publish()).phase, "deleted")
        self.assertEqual(self.storage.uploads, 1)

    async def test_adoption_callback_and_metadata_roll_back_together(self):
        fail = True
        async def adopt(connection, file, spec, content, payload):
            await self.f.sources.publish_in_transaction(connection, self.f.owner, SourcePublication(
                area_id=self.f.area, namespace="fixture-artifacts", source_key=str(file.id), payload_id=payload,
                source_type="file", trust="assistant_output", source_time=file.created_at,
                time_basis="Synthetic publication receipt", expected_revision=0, session_id=self.session))
            if fail:
                raise RuntimeError("Synthetic adoption failure after source publication")
        self.service.adopt = adopt
        with self.assertRaises(RuntimeError):
            await self.publish()
        self.assertEqual(await self.repository.list_files(owner_id=self.f.owner, query=None, limit=8), [])
        async with self.f.pool.acquire() as connection:
            self.assertEqual(await connection.fetchval(
                "select count(*) from jarvis_knowledge_sources where owner_id=$1", self.f.owner), 0)
        self.assertEqual((await self.operation())["phase"], "uploaded")
        fail = False
        self.assertEqual((await self.publish()).phase, "available")
        self.assertEqual(self.storage.uploads, 1)

    async def test_lost_upload_ack_is_reconciled_without_resubmission(self):
        self.storage.mode = "fail_after"
        with self.assertRaises(RuntimeError):
            await self.publish()
        self.assertEqual((await self.operation())["phase"], "outcome_unknown")
        self.assertEqual((await self.publish()).phase, "available")
        self.assertEqual(self.storage.uploads, 1)

    async def test_absent_unknown_upload_stays_pending_until_late_bytes_arrive(self):
        self.storage.mode = "fail_before"
        with self.assertRaises(RuntimeError):
            await self.publish()
        operation = await self.operation()
        self.assertEqual((await self.publish()).phase, "outcome_unknown")
        pending = await self.service.delete(owner_id=self.f.owner, operation_id=operation["id"])
        self.assertEqual(pending.phase, "delete_pending")
        self.assertFalse(pending.upload_settled)
        self.storage.objects[(operation["storage_bucket"], operation["storage_path"])] = self.content
        completed = await self.service.reconcile(owner_id=self.f.owner, operation_id=operation["id"])
        self.assertEqual(completed.phase, "deleted")
        self.assertEqual(self.storage.uploads, 1)
        self.assertFalse(self.storage.objects)
        self.assertEqual(await self.repository.list_files(owner_id=self.f.owner, query=None, limit=8), [])

    async def test_cancelled_upload_cannot_be_resubmitted(self):
        self.storage.mode = "pause_before"
        work = asyncio.create_task(self.publish())
        try:
            await asyncio.wait_for(self.storage.entered.wait(), 3)
            work.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await work
            self.assertEqual((await self.publish()).phase, "outcome_unknown")
            self.assertEqual(self.storage.uploads, 1)
        finally:
            self.storage.release.set()
            await asyncio.gather(work, return_exceptions=True)

    async def test_concurrent_reader_can_recover_ack_but_not_upload_again(self):
        self.storage.mode = "pause_after"
        work = asyncio.create_task(self.publish())
        try:
            await asyncio.wait_for(self.storage.entered.wait(), 3)
            second = await self.publish()
            self.assertEqual(second.phase, "available")
            self.storage.release.set()
            first = await work
            self.assertEqual(first.file_id, second.file_id)
            self.assertEqual(self.storage.uploads, 1)
        finally:
            self.storage.release.set()
            await asyncio.gather(work, return_exceptions=True)

    async def test_deletion_during_pending_upload_cannot_adopt_late_file(self):
        self.storage.mode = "pause_before"
        work = asyncio.create_task(self.publish())
        try:
            await asyncio.wait_for(self.storage.entered.wait(), 3)
            operation = await self.operation()
            pending = await self.service.delete(owner_id=self.f.owner, operation_id=operation["id"])
            self.assertEqual(pending.phase, "delete_pending")
            self.storage.release.set()
            self.assertEqual((await work).phase, "deleted")
            self.assertFalse(self.storage.objects)
            self.assertEqual(await self.repository.list_files(owner_id=self.f.owner, query=None, limit=8), [])
        finally:
            self.storage.release.set()
            await asyncio.gather(work, return_exceptions=True)

    async def test_lost_delete_ack_is_reconciled_and_never_revives_file(self):
        created = await self.publish()
        self.storage.mode = "delete_ack_lost"
        with self.assertRaises(RuntimeError):
            await self.service.delete(owner_id=self.f.owner, operation_id=created.operation_id)
        self.assertIsNone(await self.repository.get_file(owner_id=self.f.owner, file_id=created.file_id))
        self.assertEqual((await self.service.reconcile(
            owner_id=self.f.owner, operation_id=created.operation_id)).phase, "deleted")

    async def test_conflict_foreign_scope_and_size_fail_before_upload(self):
        await self.publish()
        with self.assertRaises(SourceError):
            await self.publish(content=b"Changed bytes")
        with self.assertRaises(SourceError):
            await self.publish(owner_id=self.f.other)
        with self.assertRaises(SourceError):
            await self.publish(operation_key="foreign-session", spec=self.spec.model_copy(update={"session_id": self.foreign}))
        with self.assertRaises(ValueError):
            await self.publish(operation_key="too-large", content=b"x" * (MAX_FILE_BYTES + 1))
        self.assertEqual(self.storage.uploads, 1)

    async def test_corrupt_object_is_not_adopted_or_exposed(self):
        self.storage.mode = "fail_after"
        with self.assertRaises(RuntimeError):
            await self.publish()
        operation = await self.operation()
        self.storage.objects[(operation["storage_bucket"], operation["storage_path"])] = b"z" * len(self.content)
        with self.assertRaises(SourceError):
            await self.service.reconcile(owner_id=self.f.owner, operation_id=operation["id"])
        self.assertEqual(await self.repository.list_files(owner_id=self.f.owner, query=None, limit=8), [])

    async def test_source_correction_quarantines_without_destroying_original(self):
        original = await self.f.publish("Synthetic unfinished task")
        result = await self.publish(memory_sources=[original.original_evidence()])
        await self.f.publish("Synthetic corrected task", expected_revision=1)
        self.assertIsNotNone(await self.repository.get_file(owner_id=self.f.owner, file_id=result.file_id))
        with self.assertRaises(SourceError):
            await self.service.read_content(owner_id=self.f.owner, file_id=result.file_id)
        current = await self.service.reconcile(owner_id=self.f.owner, operation_id=result.operation_id)
        self.assertEqual(current.phase, "quarantined")
        self.assertFalse(current.delete_requested)
        self.assertEqual(await self.service.reconcile_pending(owner_id=self.f.owner), [])
        self.assertTrue(self.storage.objects)
        self.assertEqual((await self.service.delete(
            owner_id=self.f.owner, operation_id=result.operation_id)).phase, "deleted")
        self.assertFalse(self.storage.objects)

    async def test_cleanup_after_archival_and_prepared_cancellation_need_no_upload(self):
        pending = await self.service.prepare(owner_id=self.f.owner, operation_key="unsubmitted",
                                             spec=self.spec, content=self.content)
        async with self.f.pool.acquire() as connection:
            await connection.execute(
                "update jarvis_knowledge_areas set status='archived',revision=revision+1 where id=$1", self.f.area)
        deleted = await self.service.delete(owner_id=self.f.owner, operation_id=pending.operation_id)
        self.assertEqual(deleted.phase, "deleted")
        self.assertEqual(self.storage.uploads, 0)

    async def test_one_connection_pool_does_not_wrap_storage_io_in_transaction(self):
        permit = asyncio.Semaphore(1)
        active = False
        @asynccontextmanager
        async def acquire():
            nonlocal active
            async with permit, self.f.pool.acquire() as connection:
                active = True
                try:
                    yield connection
                finally:
                    active = False
        class CheckedStorage(MemoryStorage):
            async def upload_bytes(storage, **arguments):
                self.assertFalse(active)
                return await super().upload_bytes(**arguments)
            async def read_object(storage, **arguments):
                self.assertFalse(active)
                return await super().read_object(**arguments)
        service = FilePublicationService(SourceRepository(SimpleNamespace(acquire=acquire)), CheckedStorage(),
                                         owner_id=self.f.owner, area_id=self.f.area)
        async with asyncio.timeout(3):
            result = await service.publish(owner_id=self.f.owner, operation_key="one-slot",
                                            spec=self.spec, content=self.content)
        self.assertEqual(result.phase, "available")

    async def test_database_refuses_locator_substitution_and_token_replacement(self):
        created = await self.publish()
        async with self.f.pool.acquire() as connection:
            for column, value in (("storage_path", "different-object"), ("upload_token", uuid4())):
                with self.subTest(column=column), self.assertRaises(asyncpg.CheckViolationError):
                    await connection.execute(
                        f"update jarvis_file_publications set {column}=$2 where id=$1", created.operation_id, value)

    async def test_full_existing_text_tool_character_limit_preserves_utf8_bytes(self):
        content = ("\U0001f642" * 200000).encode("utf-8")
        result = await self.publish(content=content)
        self.assertEqual(result.file.size_bytes, 800000)
        self.assertEqual(await self.service.read_content(owner_id=self.f.owner, file_id=result.file_id), content)

    async def test_pending_reconciliation_rotates_unresolved_work(self):
        self.storage.mode = "fail_before"
        for key in ("first-unknown", "second-unknown"):
            with self.assertRaises(RuntimeError):
                await self.publish(operation_key=key)
        first = await self.service.reconcile_pending(owner_id=self.f.owner, limit=1)
        second = await self.service.reconcile_pending(owner_id=self.f.owner, limit=1)
        self.assertNotEqual(first[0]["operation_id"], second[0]["operation_id"])


@unittest.skipUnless(os.getenv("JARVIS_FILE_PUBLICATION_LIVE") == "1" and
                     os.getenv("JARVIS_SECOND_BRAIN_TEST_DB_URL") and
                     os.getenv("JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"),
                     "Explicit live storage and manifested database opt-in required")
class NativeFilePublicationTests(unittest.IsolatedAsyncioTestCase):
    operation = FilePublicationDatabaseTests.operation

    async def asyncSetUp(self):
        await FilePublicationDatabaseTests.asyncSetUp(self)
        from app.settings import JarvisSettings
        from app.storage.supabase_config import get_supabase_storage_config
        from evals.second_brain.stored_image_fixture import require_absent_object
        configuration = get_supabase_storage_config(JarvisSettings())
        if configuration is None:
            self.fail("Native storage was requested but credentials are unavailable")
        directory = Path(os.environ["JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"]).parent / "file-manifests"
        directory.mkdir(mode=0o700, exist_ok=True)
        self.manifest_path = directory / ("file-publication-" + uuid4().hex + ".json")
        self.manifest = {"format": "jarvis.file_publication_fixture.v1", "owner": self.f.owner,
                         "status": "running", "objects": []}

        def save():
            temporary = self.manifest_path.with_name(self.manifest_path.name + "." + uuid4().hex + ".tmp")
            descriptor = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            with os.fdopen(descriptor, "w") as output:
                json.dump(self.manifest, output, sort_keys=True)
            os.replace(temporary, self.manifest_path)

        outer = self
        class RecordedStorage(JarvisFileStorageClient):
            async def upload_bytes(storage, *, path, content, content_type):
                item = {"bucket": storage.bucket, "path": path, "size": len(content),
                        "sha256": hashlib.sha256(content).hexdigest(), "upload_settled": False}
                outer.manifest["objects"].append(item)
                save()  # Manifest the exact object before the external effect.
                result = await super().upload_bytes(path=path, content=content, content_type=content_type)
                item["upload_settled"] = True
                save()
                return result

        self.storage = RecordedStorage(configuration)
        self.addAsyncCleanup(self.storage.close)
        self.service = FilePublicationService(self.f.sources, self.storage, owner_id=self.f.owner, area_id=self.f.area)
        save()
        async def cleanup():
            failures = []
            for item in self.manifest["objects"]:
                try:
                    content = await self.storage.read_object(
                        bucket=item["bucket"], path=item["path"], max_bytes=MAX_FILE_BYTES)
                    if not item["upload_settled"]:
                        if content is None or hashlib.sha256(content).hexdigest() != item["sha256"]:
                            raise RuntimeError("Native upload outcome remains unknown; retain its cleanup manifest")
                        item["upload_settled"] = True
                    if content is not None:
                        await self.storage.delete_object(bucket=item["bucket"], path=item["path"])
                    async with httpx.AsyncClient(timeout=30) as client:
                        await require_absent_object(client, configuration, bucket=item["bucket"], path=item["path"])
                    item["status"] = "cleaned"
                except Exception as error:
                    item["status"] = "cleanup_unverified"
                    item["error_type"] = type(error).__name__
                    failures.append(item["error_type"])
                save()
            self.manifest["status"] = "cleaned" if not failures else "cleanup_unverified"
            save()
            if failures:
                raise RuntimeError("Native file cleanup is unverified: " + ", ".join(failures))
        self.addAsyncCleanup(cleanup)

    async def test_real_storage_publish_replay_read_and_delete(self):
        content = ("Synthetic file publication " + uuid4().hex + "\n").encode("utf-8")
        spec = self.spec.model_copy(update={"filename": "native-fixture?literal#name%25.txt"})
        first = await self.service.publish(owner_id=self.f.owner, operation_key="native-file",
                                            spec=spec, content=content)
        again = await self.service.publish(owner_id=self.f.owner, operation_key="native-file",
                                            spec=spec, content=content)
        self.assertEqual(first.phase, "available")
        self.assertEqual(first.file.filename, spec.filename)
        self.assertEqual(first.file_id, again.file_id)
        self.assertEqual(len(self.manifest["objects"]), 1)
        self.assertEqual(await self.service.read_content(owner_id=self.f.owner, file_id=first.file_id), content)
        self.assertEqual((await self.service.delete(
            owner_id=self.f.owner, operation_id=first.operation_id)).phase, "deleted")

    async def test_real_complete_source_original_and_source_requested_storage_cleanup(self):
        from app.knowledge.files import SourceFileRepository
        from uuid import UUID
        files = SourceFileRepository(self.f.sources, self.storage, owner_id=self.f.owner, area_id=self.f.area)
        prefix = "Synthetic retained text, not a completed physical action: " + uuid4().hex + "\n"
        text = prefix + "\U0001f642" * (200000 - len(prefix))
        file = await files.create_text_file(owner_id=self.f.owner, session_id=self.session,
            filename="native-source?full#text.txt", content=text, description="Native synthetic full original",
            operation_key="native-source-file")
        original = await self.f.sources.read(self.f.owner, self.f.area, UUID(file.memory_source["source_id"]))
        self.assertEqual(original.content["text"], text)
        self.assertEqual(original.ingestion_status, "pending")  # No Hindsight acceptance is implied.
        self.assertEqual(await files.read_content(owner_id=self.f.owner, file_id=file.id), text.encode("utf-8"))
        self.assertEqual(len(self.manifest["objects"]), 1)
        await self.f.sources.delete_source(self.f.owner, self.f.area, original.source_id, expected_revision=1)
        self.assertIsNone(await self.repository.get_file(owner_id=self.f.owner, file_id=file.id))
        results = await files.publications.reconcile_pending(owner_id=self.f.owner)
        self.assertEqual([result["phase"] for result in results], ["deleted"])
