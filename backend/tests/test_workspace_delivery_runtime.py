import hashlib
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

from app.errors import JarvisHTTPError
from app.files.tools import FileToolModule
from app.files.workspace_access import WorkspaceFileAccess, file_version
from app.knowledge.files import SourceFileRepository
from app.runtime.feature_config import MemoryAuthority
from app.runtime.workspace_delivery import (
    WorkspaceContentExporter, WorkspaceContentResolver, initialize_workspace_presentations,
)


class FileAccessTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.owner, self.session, self.file_id = "fixture-owner", uuid4(), uuid4()
        self.file = SimpleNamespace(id=self.file_id, memory_source={"document_id": "exact-file-v1"}, size_bytes=3)
        self.connection = SimpleNamespace(fetchval=AsyncMock(return_value=True))
        manager = MagicMock()
        manager.__aenter__.return_value = self.connection
        self.pool = SimpleNamespace(acquire=MagicMock(return_value=manager))
        self.repository = SimpleNamespace(_authorize=MagicMock(), selections=object(),
            sources=SimpleNamespace(pool=self.pool), get_file=AsyncMock(return_value=self.file),
            read_content=AsyncMock(return_value=b"abc"), for_session=MagicMock())
        self.repository.for_session.return_value = self.repository
        self.access = WorkspaceFileAccess(self.repository)

    async def test_owned_session_and_exact_bytes(self):
        file, content, checksum = await self.access.read(self.owner, self.session, self.file_id, "exact-file-v1")
        self.assertIs(file, self.file)
        self.assertEqual(content, b"abc")
        self.assertEqual(checksum, hashlib.sha256(content).hexdigest())
        self.repository.for_session.assert_called_once_with(self.session)
        self.assertEqual(self.connection.fetchval.call_args.args[1:], (self.owner, self.session))

    async def test_foreign_session_precedes_content(self):
        self.connection.fetchval.return_value = False
        with self.assertRaises(JarvisHTTPError):
            await self.access.get(self.owner, self.session, self.file_id)
        self.repository.get_file.assert_not_awaited()

    async def test_selected_area_requires_session(self):
        with self.assertRaises(JarvisHTTPError):
            await self.access.get(self.owner, None, self.file_id)
        self.repository.get_file.assert_not_awaited()

    async def test_existing_fixed_area_path_does_not_invent_session(self):
        self.repository.selections = None
        self.assertIs(await self.access.get(self.owner, None, self.file_id), self.file)
        self.pool.acquire.assert_not_called()

    async def test_exact_version_never_substituted(self):
        with self.assertRaises(JarvisHTTPError) as error:
            await self.access.read(self.owner, self.session, self.file_id, "old-version")
        self.assertEqual(error.exception.code, "file_version_unavailable")
        self.repository.read_content.assert_not_awaited()

    async def test_source_changed_during_download(self):
        self.repository.get_file.side_effect = [self.file,
            SimpleNamespace(memory_source={"document_id": "replacement"}, size_bytes=3)]
        with self.assertRaises(JarvisHTTPError):
            await self.access.read(self.owner, self.session, self.file_id)

    async def test_source_removed_during_download(self):
        self.repository.get_file.side_effect = [self.file, None]
        with self.assertRaises(JarvisHTTPError):
            await self.access.read(self.owner, self.session, self.file_id)

    async def test_mismatched_size_not_delivered(self):
        self.repository.read_content.return_value = b"changed"
        with self.assertRaises(JarvisHTTPError) as error:
            await self.access.read(self.owner, self.session, self.file_id)
        self.assertEqual(error.exception.code, "file_content_mismatch")

    def test_missing_version_not_inferred(self):
        with self.assertRaises(JarvisHTTPError):
            file_version(SimpleNamespace(memory_source=None))


class DeliveryBindingTests(unittest.IsolatedAsyncioTestCase):
    def services(self, *, files=True):
        pool = object()
        sources = SimpleNamespace(pool=pool)
        repository = MagicMock(spec=SourceFileRepository)
        repository.owner_id, repository.sources, repository.task_evidence = "owner", sources, None
        module = MagicMock(spec=FileToolModule)
        module.repository = repository
        services = SimpleNamespace(memory_authority=MemoryAuthority.HINDSIGHT,
            knowledge_runtime=SimpleNamespace(sources=sources), db_pool=pool,
            settings=SimpleNamespace(JARVIS_OWNER_ID="owner"), tool_modules=[module] if files else [])
        return services, repository

    def test_binding_has_no_io_and_does_not_create_another_store(self):
        services, repository = self.services()
        initialize_workspace_presentations(services)
        self.assertIs(services.workspace_presentations.repository.pool, services.db_pool)
        self.assertIsNotNone(services.workspace_presentations.exporter)
        repository.get_file.assert_not_called()

    def test_wrong_owner_pool_or_worker_files_not_bound(self):
        for field, value in [("owner_id", "other"), ("sources", SimpleNamespace(pool=object())), ("task_evidence", object())]:
            services, repository = self.services()
            setattr(repository, field, value)
            initialize_workspace_presentations(services)
            self.assertIsNone(services.workspace_presentations.exporter)

    def test_legacy_does_not_gain_workspace_authority(self):
        services, _ = self.services()
        services.memory_authority = MemoryAuthority.LEGACY
        initialize_workspace_presentations(services)
        self.assertIsNone(services.workspace_presentations)

    async def test_file_export_reuses_retained_identity(self):
        file_id = uuid4()
        access = SimpleNamespace(get=AsyncMock(return_value=SimpleNamespace(content_type="text/plain")),
            read=AsyncMock(return_value=(SimpleNamespace(id=file_id), b"abc", hashlib.sha256(b"abc").hexdigest())))
        exporter = WorkspaceContentExporter(access)
        ctx = SimpleNamespace(owner_id="owner", session_id=uuid4())
        request = SimpleNamespace(artifact_kind="file", action="download", artifact_id=file_id, version_id="file-v1")
        prepared = await exporter(ctx, request)
        self.assertEqual(prepared.file_id, file_id)
        self.assertEqual(prepared.byte_count, 3)
        access.read.assert_awaited_once_with("owner", ctx.session_id, file_id, "file-v1")

    async def test_large_file_not_reported_ready(self):
        access = SimpleNamespace(get=AsyncMock(return_value=SimpleNamespace(content_type="text/plain")),
            read=AsyncMock(return_value=(SimpleNamespace(id=uuid4()), b"x" * (8 * 1024 * 1024 + 1), "a" * 64)))
        exporter = WorkspaceContentExporter(access)
        with self.assertRaises(JarvisHTTPError) as error:
            await exporter(SimpleNamespace(owner_id="owner", session_id=uuid4()),
                SimpleNamespace(artifact_kind="file", action="download", artifact_id=uuid4(), version_id="v1"))
        self.assertFalse(error.exception.details["effect_committed"])
