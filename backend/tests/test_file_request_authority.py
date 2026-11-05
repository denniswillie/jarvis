"""Offline request fences around the real publication lifecycle; no SQL/provider I/O."""

from contextlib import asynccontextmanager
import hashlib
import json
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

from app.files.publication import FilePublicationResult, FilePublicationService, FilePublicationSpec
from app.files.request_authority import FileRequestAuthority, REQUEST_AUTHORITY_KEY
from app.knowledge.files import SourceFileRepository
from app.knowledge.models import SourceError
from app.tasks.models import TaskError
from app.tools.invocation import TaskToolInvocation, bind_tool_invocation


class FileRequestAuthorityTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.pool, self.owner = object(), "owner"
        self.session, self.event, self.area, self.request_payload = (uuid4() for _ in range(4))
        self.in_transaction = self.task_locked = False
        self.current = True
        self.prepared = self.blob_exists = False
        self.invalidate_upload = self.invalidate_read = False
        self.content = b"An explicitly requested text file.\n"
        self.file = None
        self.dependencies = []
        self.connection = SimpleNamespace(
            is_in_transaction=lambda: self.in_transaction,
            execute=AsyncMock(side_effect=self.execute),
            fetchrow=AsyncMock(side_effect=self.fetchrow),
            fetchval=AsyncMock(side_effect=self.fetchval),
            executemany=AsyncMock(side_effect=self.executemany),
        )

        @asynccontextmanager
        async def transaction(owner):
            self.assertEqual(owner, self.owner)
            self.assertFalse(self.in_transaction)
            self.in_transaction, self.task_locked = True, False
            try:
                yield self.connection
            finally:
                self.in_transaction, self.task_locked = False, False

        self.sources = SimpleNamespace(pool=self.pool, transaction=transaction, _area=AsyncMock())
        self.tasks = SimpleNamespace(pool=self.pool, lock_owner=AsyncMock(side_effect=self.lock_owner),
                                     _voice_source=AsyncMock(side_effect=self.original))
        self.authority = FileRequestAuthority(self.tasks)
        self.base_spec = FilePublicationSpec(session_id=self.session, filename="requested.txt",
            content_type="text/plain; charset=utf-8", description="Requested fixture", source="assistant")
        with self.invocation():
            self.spec = self.authority.bind(owner_id=self.owner, spec=self.base_spec)
        self.row = dict(id=uuid4(), file_id=uuid4(), phase="prepared", upload_token=None,
            upload_settled=False, delete_requested=False, availability="available",
            storage_bucket="files", storage_path="fixed", size_bytes=len(self.content),
            content_sha256=hashlib.sha256(self.content).hexdigest(), area_id=self.area,
            metadata_payload_id=uuid4(), specification={"spec": self.spec.model_dump(mode="json")})
        self.storage = SimpleNamespace(bucket="files", build_file_path=Mock(return_value="fixed"),
            upload_bytes=AsyncMock(side_effect=self.upload), read_object=AsyncMock(side_effect=self.read),
            upload_file=AsyncMock(), delete_object=AsyncMock())
        self.adopt = AsyncMock(side_effect=self.adoption)
        self.repository = SimpleNamespace(get_file=AsyncMock(side_effect=lambda **kwargs: self.file),
                                         create_file=AsyncMock(side_effect=self.create_file))
        patcher = patch("app.files.publication.JarvisFileRepository", return_value=self.repository)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.service = self.service_for(self.authority)

    def invocation(self, **changes):
        return bind_tool_invocation(**({"provider_call_id": "native-call", "request_event_id": self.event,
                                        "request_revision": 1} | changes))

    def service_for(self, authority):
        service = FilePublicationService(self.sources, self.storage, owner_id=self.owner, area_id=self.area,
                                          adopt=self.adopt, request_authority=authority)
        service._row = AsyncMock(side_effect=lambda *args: dict(self.row))
        service._result = AsyncMock(side_effect=lambda connection, row: self.result(row))
        async def materialize(connection, owner, area, originals, callback):
            return (await callback(connection, ()))[0]
        service.provenance.bind_in_transaction = AsyncMock(side_effect=materialize)
        return service

    def source_files(self, *, guarded=True, **kwargs):
        return SourceFileRepository(self.sources, self.storage, owner_id=self.owner, area_id=self.area,
            request_authority=self.authority if guarded else None, **kwargs)

    def result(self, row=None):
        row = self.row if row is None else row
        return FilePublicationResult(row["id"], row["file_id"], row["phase"], self.file,
                                     row["upload_settled"], row["delete_requested"])

    async def lock_owner(self, connection, owner):
        self.assertTrue(self.in_transaction)
        self.assertEqual(owner, self.owner)
        self.task_locked = True

    async def original(self, connection, owner, reference):
        self.assertTrue(self.in_transaction and self.task_locked)
        if not self.current or (owner, reference.session_id, reference.event_id, reference.revision) != (
                self.owner, self.session, self.event, 1):
            raise TaskError("task_request_evidence_missing", "Fixture original unavailable", 403)
        return {"payload_id": self.request_payload}

    async def execute(self, sql, *args):
        self.assertTrue(self.in_transaction)
        if "pg_advisory_xact_lock" in sql:
            self.task_locked = True
        elif "set phase='uploading',upload_token" in sql:
            self.row.update(phase="uploading", upload_token=args[2])
        elif "set upload_settled=$3,phase=$4" in sql:
            self.row.update(upload_settled=args[2], phase=args[3])
        elif "set upload_settled=true,phase=$3" in sql:
            self.row.update(upload_settled=True, phase=args[2])
        elif "set phase='available'" in sql:
            self.row["phase"] = "available"
        elif "set phase='outcome_unknown'" in sql:
            self.row["phase"] = "outcome_unknown"
        else:
            raise AssertionError(sql)

    async def fetchrow(self, sql, *args):
        if "select id,area_id,parameters_digest from jarvis_file_publications" in sql:
            return dict(self.row) if self.prepared else None
        raise AssertionError(sql)

    async def fetchval(self, sql, *args):
        if "select exists(select 1 from jarvis_sessions" in sql:
            return args == (self.owner, self.session)
        if "lock_jarvis_copy_sources(" in sql:
            return True
        if "lock_jarvis_copy_source(" in sql:
            return "available"
        if "insert into jarvis_context_payloads" in sql:
            self.row["specification"] = json.loads(args[1])
            return self.row["metadata_payload_id"]
        if "insert into jarvis_file_publications" in sql:
            self.prepared = True
            self.row.update(parameters_digest=args[3], file_id=args[4], metadata_payload_id=args[5],
                            content_sha256=args[6], size_bytes=args[7])
            return self.row["id"]
        raise AssertionError(sql)

    async def executemany(self, sql, rows):
        self.assertIn("jarvis_context_payload_dependencies", sql)
        self.dependencies.extend(rows)

    async def upload(self, *, path, content, content_type):
        self.assertFalse(self.in_transaction)
        self.blob_exists = True
        if self.invalidate_upload:
            self.current = False
        return SimpleNamespace(bucket="files", path=path, size_bytes=len(content))

    async def read(self, **kwargs):
        self.assertFalse(self.in_transaction)
        if self.invalidate_read:
            self.current = False
        return self.content if self.blob_exists else None

    async def create_file(self, **kwargs):
        self.file = SimpleNamespace(id=kwargs["file_id"])
        return self.file

    async def adoption(self, *args):
        self.assertTrue(self.in_transaction)
        if REQUEST_AUTHORITY_KEY in self.row["specification"]["spec"]["metadata"]:
            self.assertTrue(self.task_locked and self.current)

    async def prepare(self, spec=None):
        return await self.service.prepare(owner_id=self.owner, operation_key="exact-file",
                                          spec=spec or self.spec, content=self.content)

    async def submit(self, service=None, spec=None):
        return await (service or self.service)._publish_prepared(
            self.owner, self.result(), spec or self.spec, content=self.content)

    async def test_manifest_persists_exact_reference_and_original_dependency(self):
        await self.prepare()
        stored = self.row["specification"]["spec"]["metadata"][REQUEST_AUTHORITY_KEY]
        self.assertEqual(stored, {"version": 1, "owner_id": self.owner, "event_id": str(self.event),
                                  "revision": 1, "session_id": str(self.session)})
        self.assertIn((self.owner, self.row["metadata_payload_id"], self.request_payload), self.dependencies)
        self.assertNotIn(self.content.decode(), json.dumps(stored))

    async def test_new_provider_call_retry_retains_same_manifest_identity(self):
        await self.prepare()
        with self.invocation(provider_call_id="another-native-call"):
            same = self.authority.bind(owner_id=self.owner, spec=self.base_spec)
        self.assertEqual(same, self.spec)
        await self.prepare(same)
        inserts = [call for call in self.connection.fetchval.call_args_list
                   if "insert into jarvis_file_publications" in call.args[0]]
        self.assertEqual(len(inserts), 1)

    async def test_changed_request_rejected_before_manifest_or_upload(self):
        self.current = False
        with self.assertRaises(SourceError):
            await self.prepare()
        self.assertFalse(self.prepared)
        with self.assertRaises(SourceError):
            await self.submit()
        self.storage.upload_bytes.assert_not_awaited()

    async def test_current_request_rechecked_before_upload_and_adoption(self):
        await self.prepare()
        result = await self.submit()
        self.assertEqual(result.phase, "available")
        self.assertEqual(self.tasks._voice_source.await_count, 4)
        self.storage.upload_bytes.assert_awaited_once()
        self.adopt.assert_awaited_once()

    async def test_changed_request_after_upload_blocks_adoption_not_claiming_rollback(self):
        self.invalidate_upload = True
        with self.assertRaises(SourceError):
            await self.submit()
        self.assertTrue(self.blob_exists)
        self.assertEqual(self.row["phase"], "uploaded")
        self.repository.create_file.assert_not_awaited()
        self.adopt.assert_not_awaited()
        with self.assertRaises(SourceError):
            await self.service.reconcile(owner_id=self.owner, operation_id=self.row["id"])
        self.storage.upload_bytes.assert_awaited_once()

    async def test_change_during_storage_read_is_rechecked_in_adoption_transaction(self):
        self.invalidate_read = True
        with self.assertRaises(SourceError):
            await self.submit()
        self.repository.create_file.assert_not_awaited()
        self.adopt.assert_not_awaited()

    async def recover(self):
        self.row.update(phase="uploaded", upload_token=uuid4(), upload_settled=True)
        self.blob_exists = True
        restored = self.service_for(FileRequestAuthority(self.tasks))
        result = await restored.reconcile(owner_id=self.owner, operation_id=self.row["id"])
        self.assertEqual(result.phase, "available")
        self.storage.upload_bytes.assert_not_awaited()
        self.adopt.assert_awaited_once()
        self.assertTrue(all(call.args[2].event_id == self.event for call in self.tasks._voice_source.call_args_list))

    async def test_restart_recovery_needs_no_ambient_original(self):
        await self.recover()

    async def test_restart_recovery_ignores_another_ambient_user_request(self):
        with self.invocation(request_event_id=uuid4(), request_revision=12):
            await self.recover()

    async def test_missing_guard_cannot_adopt_a_marked_publication(self):
        with self.assertRaises(SourceError):
            await self.submit(self.service_for(None))
        self.row.update(phase="uploaded", upload_token=uuid4())
        self.blob_exists = True
        with self.assertRaises(SourceError):
            await self.service_for(None).reconcile(owner_id=self.owner, operation_id=self.row["id"])
        self.storage.upload_bytes.assert_not_awaited()
        self.adopt.assert_not_awaited()

    async def test_guard_does_not_infer_authority_for_old_unmarked_manifest(self):
        self.row["specification"]["spec"] = self.base_spec.model_dump(mode="json")
        with self.assertRaises(SourceError):
            await self.submit(spec=self.base_spec)
        self.storage.upload_bytes.assert_not_awaited()

    async def test_foreign_owner_session_and_event_fail_closed(self):
        for field, value in (("owner_id", "foreign"), ("session_id", str(uuid4())), ("event_id", str(uuid4()))):
            raw = {**self.spec.metadata[REQUEST_AUTHORITY_KEY], field: value}
            spec = self.spec.model_copy(update={"metadata": {REQUEST_AUTHORITY_KEY: raw}})
            with self.subTest(field=field), self.assertRaises(SourceError):
                await self.prepare(spec)
        self.assertFalse(self.prepared)
        self.storage.upload_bytes.assert_not_awaited()

    async def test_missing_request_and_worker_invocation_cannot_bind(self):
        for context in (
            bind_tool_invocation(provider_call_id="completion"),
            self.invocation(provider_call_id=None),
            self.invocation(task=TaskToolInvocation(uuid4(), 1, uuid4())),
        ):
            with context, self.assertRaises(SourceError):
                self.authority.bind(owner_id=self.owner, spec=self.base_spec)

    async def test_source_repository_rejects_completion_and_identity_overrides(self):
        files = self.source_files()
        files.publications.publish = AsyncMock()
        parameters = dict(owner_id=self.owner, session_id=self.session, filename="requested.txt",
                          content="Requested", description="Fixture")
        with bind_tool_invocation(provider_call_id="completion"), self.assertRaises(SourceError):
            await files.create_text_file(**parameters)
        with self.invocation():
            for override in ({"operation_key": "model-key"}, {"source_payload_ids": (uuid4(),)},
                             {"memory_sources": ({},)}):
                with self.assertRaises(SourceError):
                    await files.create_text_file(**parameters, **override)
        files.publications.publish.assert_not_awaited()

    async def test_source_repository_captures_original_before_publication(self):
        files = self.source_files()
        files.publications.publish = AsyncMock(return_value=SimpleNamespace(phase="available", file_id=uuid4()))
        files.get_file = AsyncMock(return_value="stored fixture")
        with self.invocation():
            await files.create_text_file(owner_id=self.owner, session_id=self.session, filename="requested.txt",
                                         content="Requested", description="Fixture")
        kwargs = files.publications.publish.call_args.kwargs
        self.assertEqual(kwargs["spec"].metadata, self.spec.metadata)
        self.assertTrue(kwargs["operation_key"].startswith("text-file:"))

    async def test_unsupported_mutations_do_not_reach_effects(self):
        files = self.source_files()
        for invoke in (
            lambda: files.delete_source_file(owner_id=self.owner, file_id=uuid4()),
            lambda: files.publish_generated_snapshot(owner_id=self.owner, snapshot={}),
            lambda: files.capture_current_frame(owner_id=self.owner, session_id=self.session,
                                                filename="camera.jpg", description="Fixture"),
            lambda: self.service.delete(owner_id=self.owner, operation_id=self.row["id"]),
            lambda: self.service.publish_file(owner_id=self.owner, operation_key="file", spec=self.spec, local_file=None),
        ):
            with self.assertRaises(SourceError):
                await invoke()
        self.storage.upload_bytes.assert_not_awaited()
        self.storage.upload_file.assert_not_awaited()
        self.storage.delete_object.assert_not_awaited()

    async def test_same_pool_required_and_session_copy_preserves_guard(self):
        foreign = FileRequestAuthority(SimpleNamespace(pool=object()))
        for create in (
            lambda: FilePublicationService(self.sources, self.storage, owner_id=self.owner, area_id=self.area,
                                            request_authority=foreign),
            lambda: self.source_files(task_evidence=object()),
            lambda: self.source_files(task_recovery=object()),
        ):
            with self.assertRaises(ValueError):
                create()
        selections = SimpleNamespace(sources=self.sources, default_area_id=self.area)
        files = self.source_files(selections=selections).for_session(self.session)
        self.assertIs(files.request_authority, self.authority)
        self.assertIs(files.publications.request_authority, self.authority)

    async def test_old_unguarded_publication_path_unchanged(self):
        self.row["specification"]["spec"] = self.base_spec.model_dump(mode="json")
        result = await self.submit(self.service_for(None), self.base_spec)
        self.assertEqual(result.phase, "available")
        self.tasks._voice_source.assert_not_awaited()
        self.storage.upload_bytes.assert_awaited_once()
        self.adopt.assert_awaited_once()

    async def test_old_unguarded_source_writer_keeps_direct_operation_keys(self):
        files = self.source_files(guarded=False)
        files.publications.publish = AsyncMock(return_value=SimpleNamespace(phase="available", file_id=uuid4()))
        files.get_file = AsyncMock(return_value="legacy fixture")
        await files.create_text_file(owner_id=self.owner, session_id=self.session, filename="old.txt",
                                     content="Legacy", description="Fixture", operation_key="old-key")
        kwargs = files.publications.publish.call_args.kwargs
        self.assertEqual(kwargs["operation_key"], "old-key")
        self.assertEqual(kwargs["spec"].metadata, {})
