"""Offline retained authority/transfer tests; no provider, database or decoder calls."""
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

from app.files.local_file import snapshot_file
from app.files.publication import FilePublicationService
from app.files.transfer import FileManifest, FileTransferPolicy, MP4Inspection
from app.hermes_sessions.bindings import BindingError, SessionBindings
from app.hermes_sessions.domain_scope import DomainInputScope
from app.hermes_sessions.domain_scopes import DomainScopeAssociations
from app.hermes_sessions.sandbox_bindings import SessionSandboxBindings
from app.hermes_sessions.sandboxes import RetainedSandboxService
from app.hermes_sessions.source_files import RetainedFileTools
from app.knowledge.models import SourceError
from app.sandbox.file_transfer import INPUT_ROOT
from app.tools.invocation import bind_tool_invocation
from app.tools.module import ToolBuildContext
from test_sandboxes import Provider


class RetainedFileTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.bindings = SessionBindings(self.root / "bindings.sqlite", owner_id="owner", profile_home=self.root)
        master = self.bindings.reserve(request_id="master", parent_id=None, label="Master", launch={})
        self.master = self.bindings.bind(master.logical_id, "native-master")
        child = self.bindings.reserve(request_id="child", parent_id=master.logical_id, label="Child", launch={})
        self.child = self.bindings.bind(child.logical_id, "native-child")
        self.scopes = DomainScopeAssociations(self.root / "scopes.sqlite", self.bindings)
        self.session = uuid4()
        self.allowed = frozenset({"list_files", "get_file_metadata", "create_text_file", "sandbox_stage_file", "sandbox_publish_file"})
        self.scope = self.scopes.bind_initial(DomainInputScope("owner", self.master.logical_id, "input-one", self.session, self.allowed))
        self.service = SimpleNamespace(bindings=self.bindings)
        self.data = b"exact binary fixture\x00\xff"
        self.file_id, self.area = uuid4(), uuid4()
        self.source = {"source_id": str(uuid4()), "revision": 1, "generation": 1,
                       "content_digest": "a" * 64, "is_current": True}
        self.file = SimpleNamespace(id=self.file_id, session_id=self.session, filename="input.pdf",
            content_type="application/pdf", storage_bucket="fixture", storage_path="private",
            size_bytes=len(self.data), description="fixture", source="assistant", source_visual_frame_id=None,
            created_at=datetime.now(timezone.utc), metadata={}, memory_source=self.source, task_evidence=None)
        self.files = SimpleNamespace(owner_id="owner", area_id=self.area, sources=SimpleNamespace(pool=object()),
            storage=SimpleNamespace(bucket="fixture"), access_check=Mock(), selections=None,
            media_policy=FileTransferPolicy("mp4_capture_v1", 256 * 1024**2),
            media_inspector=SimpleNamespace(inspect_retained_input=AsyncMock(side_effect=self.inspect_media)),
            get_file=AsyncMock(return_value=self.file), read_content=AsyncMock(return_value=self.data),
            read_file=AsyncMock(), capture_tool_result=AsyncMock(), list_files=AsyncMock(return_value=[]),
            publications=SimpleNamespace(read_manifest=AsyncMock(return_value=(
                FileManifest(size_bytes=len(self.data), content_sha256=hashlib.sha256(self.data).hexdigest()), FileTransferPolicy()))))
        self.files.for_session = lambda session: self.files
        self.resources = SessionSandboxBindings(self.root / "resources.sqlite", owner_id="owner", profile_id=str(self.root))
        self.addCleanup(self.resources.close)
        self.provider = Provider()
        self.provider.stage_input = AsyncMock(side_effect=lambda handle, key, data: {
            "path": INPUT_ROOT + "/" + key, "sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)})
        self.provider.stage_input_file = AsyncMock(side_effect=lambda handle, key, local, **kw: {
            "path": INPUT_ROOT + "/" + key, "sha256": local.content_sha256, "size_bytes": local.size_bytes})
        self.provider.read_workspace_file = AsyncMock(return_value=self.data)
        self.provider.read_workspace_file_snapshot = AsyncMock()
        self.sandboxes = RetainedSandboxService(self.resources, self.provider)
        self.tools = RetainedFileTools(service=self.service, scope_for_input=self.scopes.scope_for_input,
            files=self.files, sandbox_service=lambda: self.sandboxes)

    async def inspect_media(self, local, policy):
        return MP4Inspection(content_sha256=local.content_sha256, size_bytes=local.size_bytes,
            duration_ms=1000, width=640, height=480, video_codec="h264", audio_codec=None)

    def definitions(self, scope=None):
        scope = scope or self.scope
        return {tool.name: tool for tool in self.tools.module(scope).build_tools(
            ToolBuildContext(settings=None, owner_id="owner", session_id=self.session))}

    async def local(self):
        path = self.root / "output.mp4"
        path.write_bytes(self.data)
        local = await snapshot_file(path, max_bytes=1024)
        self.addCleanup(local.close)
        return local

    async def test_binary_staging_preserves_bytes_and_exact_session(self):
        result = await self.definitions()["sandbox_stage_file"].handler({"file_id": str(self.file_id)})
        self.assertEqual(result["status"], "success")
        args = self.provider.stage_input.call_args.args
        self.assertEqual(args[0].scope.logical_session_id, self.master.logical_id)
        self.assertEqual(args[2], self.data)
        self.assertEqual(result["sandbox_input"]["sha256"], hashlib.sha256(self.data).hexdigest())
        self.assertIsNone(self.resources.get(self.child.logical_id))

    async def test_mp4_staging_uses_verified_local_stream_and_closes_it(self):
        local = await self.local()
        self.files.read_file.return_value = local
        self.files.publications.read_manifest.return_value = (
            FileManifest.from_file(local), FileTransferPolicy("mp4_capture_v1", 256 * 1024**2))
        await self.definitions()["sandbox_stage_file"].handler({"file_id": str(self.file_id)})
        self.provider.stage_input.assert_not_awaited()
        self.provider.stage_input_file.assert_awaited_once()
        self.assertTrue(local._closed)

    async def test_rejected_original_does_not_allocate(self):
        self.files.get_file.return_value = None
        with self.assertRaises(SourceError):
            await self.tools.stage(self.scope, self.file_id)
        self.assertEqual(self.provider.creates, 0)

    async def test_revocation_after_transfer_pauses_without_deleting(self):
        self.files.get_file.side_effect = [self.file, self.file, None]
        with self.assertRaises(SourceError):
            await self.tools.stage(self.scope, self.file_id)
        self.assertEqual(self.provider.pauses, 1)
        self.assertEqual(self.provider.creates, 1)
        self.assertEqual((await self.sandboxes.inspect(self.master.logical_id)).state, "paused")

    async def test_real_publication_owner_and_media_hooks_are_reused(self):
        publication = self.tools.publication(self.session)
        self.assertIsInstance(publication, FilePublicationService)
        self.assertIs(publication.storage, self.files.storage)
        self.assertIs(publication.sources, self.files.sources)
        self.assertIs(publication.adopt.__self__, self.tools)
        self.assertIs(publication.adopt_file.__self__, self.tools)
        self.assertEqual(publication.transfer_policy.policy_id, "mp4_output_v1")

    async def test_text_and_async_parent_writes_do_not_forge_foreground_requests(self):
        child = self.scopes.derive(self.scope, self.child.logical_id, "child-input", kind="work")
        completion = self.scopes.derive(child, self.master.logical_id, "completion", kind="completion")
        parent = self.scopes.derive(child, self.master.logical_id, "parent-message", kind="parent_message")
        callback = self.scopes.bind_initial(replace(child, input_id="authenticated-callback"))
        for scope in (self.scope, child, completion, parent, callback):
            with self.subTest(input=scope.input_id):
                publisher = SimpleNamespace(publish=AsyncMock(return_value=SimpleNamespace(
                    phase="available", file_id=self.file_id)))
                self.tools.publication = Mock(return_value=publisher)
                with bind_tool_invocation(provider_call_id="native-call"):
                    result = await self.definitions(scope)["create_text_file"].handler({"filename": "result.txt", "content": "Saved"})
                self.assertEqual(result["status"], "success")
                spec = publisher.publish.call_args.kwargs["spec"]
                self.assertNotIn("file_request_authority", spec.metadata)
                self.assertIsNone(scope.request_event_id)
                connection = SimpleNamespace(fetchval=AsyncMock(return_value=True))
                self.assertEqual(await self.tools.authorize_publication(connection, spec), scope)
                invalid = spec.model_copy(update={"session_id": uuid4()})
                with self.assertRaises(SourceError):
                    await self.tools.authorize_publication(connection, invalid)

    async def test_binary_publish_reads_only_current_retained_workspace(self):
        await self.sandboxes.ensure(self.master.logical_id)
        publisher = SimpleNamespace(publish=AsyncMock(return_value=SimpleNamespace(phase="available", file_id=self.file_id)))
        self.tools.publication = Mock(return_value=publisher)
        with bind_tool_invocation(provider_call_id="publish-one"):
            result = await self.definitions()["sandbox_publish_file"].handler(
                {"path": "output/report.pdf", "filename": "report.pdf", "format": "pdf"})
        self.assertEqual(result["status"], "success")
        self.assertEqual(self.provider.read_workspace_file.call_args.args[0].scope.logical_session_id, self.master.logical_id)
        self.assertEqual(publisher.publish.call_args.kwargs["content"], self.data)
        self.assertEqual(publisher.publish.call_args.kwargs["spec"].content_type, "application/pdf")

    async def test_mp4_publish_uses_file_backed_existing_publication(self):
        await self.sandboxes.ensure(self.master.logical_id)
        local = await self.local()
        self.provider.read_workspace_file_snapshot.return_value = local
        publisher = SimpleNamespace(publish_file=AsyncMock(return_value=SimpleNamespace(phase="available", file_id=self.file_id)))
        self.tools.publication = Mock(return_value=publisher)
        with bind_tool_invocation(provider_call_id="publish-mp4"):
            result = await self.definitions()["sandbox_publish_file"].handler(
                {"path": "edited.mp4", "filename": "edited.mp4", "format": "mp4"})
        self.assertEqual(result["status"], "success")
        self.assertIs(publisher.publish_file.call_args.kwargs["local_file"], local)
        spec = publisher.publish_file.call_args.kwargs["spec"]
        self.assertEqual(spec.metadata["inspection"]["media_inspection"]["duration_ms"], 1000)
        self.assertEqual(spec.metadata["media_inspection"], spec.metadata["inspection"]["media_inspection"])
        self.assertTrue(local._closed)

    async def test_host_paths_and_extra_identity_fields_are_rejected(self):
        for arguments in ({"path": "/etc/passwd", "filename": "report.txt", "format": "text"},
                          {"path": "../elsewhere", "filename": "report.txt", "format": "text"},
                          {"path": "report.txt", "filename": "report.txt", "format": "text", "owner_id": "other"}):
            with self.assertRaises(ValueError), bind_tool_invocation(provider_call_id="invalid"):
                await self.definitions()["sandbox_publish_file"].handler(arguments)
        self.assertEqual(self.provider.creates, 0)

    async def test_foreign_scope_is_rejected_before_files_or_provider(self):
        with self.assertRaises(BindingError):
            await self.tools.stage(replace(self.scope, owner_id="other"), self.file_id)
        self.files.get_file.assert_not_awaited()
        self.assertEqual(self.provider.creates, 0)
