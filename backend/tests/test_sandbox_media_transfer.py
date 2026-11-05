"""Bounded media transport, fixed-helper semantics, and publication ownership.

No live VM, identity switch, provider, SQL, or source-policy enablement occurs.
"""
import asyncio
import hashlib
import json
import os
from pathlib import Path
import shlex
import stat
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

import httpx

from app.files.generated import ArtifactPolicy
from app.files.local_file import CHUNK_BYTES, _snapshot_chunks
from app.files.transfer import FileTransferPolicy, MP4_OUTPUT_POLICY
from app.sandbox.artifacts import SandboxArtifactTools
from app.sandbox.e2b_provider import E2BProvider
from app.sandbox.file_transfer import FREEZE_OUTPUT, OUTPUT_ROOT, REMOVE_OUTPUT
from app.sandbox.models import E2BConfiguration, SandboxError
from app.tasks.models import TaskError


class NativeFreezeTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.workspace = self.root / "workspace"
        self.workspace.mkdir()
        self.scope = {"__name__": "fixture"}
        exec(FREEZE_OUTPUT, self.scope)
        # Execute the actual helper against an isolated temporary tree. Only
        # identity switching and root-directory ownership are substituted.
        def fstat(fd):
            value = os.fstat(fd)
            if stat.S_ISDIR(value.st_mode):
                return SimpleNamespace(st_uid=0, st_mode=value.st_mode)
            return value
        proxy = SimpleNamespace(**{name: getattr(os, name) for name in (
            "open", "close", "mkdir", "unlink", "read", "write", "fsync", "fchmod",
            "O_RDONLY", "O_WRONLY", "O_CREAT", "O_EXCL", "O_DIRECTORY", "O_NOFOLLOW", "O_NONBLOCK")})
        proxy.fstat = fstat
        proxy.setgroups, proxy.setegid, proxy.seteuid = Mock(), Mock(), Mock()
        self.scope.update(os=proxy, PARENT=str(self.root), WORKSPACE=str(self.workspace),
            pwd=SimpleNamespace(getpwnam=Mock(return_value=SimpleNamespace(pw_uid=1234, pw_gid=1234))))
        self.proxy = proxy
        self.request = {"stage_id": "a" * 32, "path": "clip.mp4", "limit": 1000}

    def test_freeze_reads_unprivileged_and_creates_exact_private_immutable_copy(self):
        (self.workspace / "clip.mp4").write_bytes(b"synthetic MP4 bytes")
        receipt = self.scope["freeze"](self.request)
        target = self.root / "outgoing" / self.request["stage_id"]
        self.assertEqual(target.read_bytes(), b"synthetic MP4 bytes")
        self.assertEqual(receipt["sha256"], hashlib.sha256(target.read_bytes()).hexdigest())
        self.assertEqual(stat.S_IMODE(target.stat().st_mode), 0o400)
        self.assertEqual(self.proxy.seteuid.call_args_list, [unittest.mock.call(1234), unittest.mock.call(0)])
        cleanup = {"__name__": "fixture"}
        exec(REMOVE_OUTPUT, cleanup)
        cleanup.update(os=self.proxy, PARENT=str(self.root))
        self.assertTrue(cleanup["remove"](self.request)["removed"])
        self.assertFalse(target.exists())

    def test_symlink_parent_leaf_directory_and_empty_are_rejected(self):
        (self.workspace / "real").mkdir()
        (self.workspace / "real" / "file").write_bytes(b"x")
        (self.workspace / "link").symlink_to(self.workspace / "real", target_is_directory=True)
        (self.workspace / "leaf").symlink_to(self.workspace / "real" / "file")
        (self.workspace / "empty").write_bytes(b"")
        for path in ("link/file", "leaf", "real", "empty", "../outside"):
            with self.subTest(path=path), self.assertRaises((OSError, ValueError)):
                self.scope["freeze"](self.request | {"path": path})
            self.assertFalse((self.root / "outgoing" / self.request["stage_id"]).exists())

    def test_mutation_or_oversize_removes_incomplete_stage(self):
        source = self.workspace / "clip.mp4"
        source.write_bytes(b"before")
        original = self.proxy.read
        def mutate(fd, count):
            value = original(fd, count)
            if value:
                source.write_bytes(b"after!")
            return value
        self.proxy.read = mutate
        with self.assertRaisesRegex(ValueError, "file changed"):
            self.scope["freeze"](self.request)
        self.assertFalse((self.root / "outgoing" / self.request["stage_id"]).exists())
        self.proxy.read = original
        with self.assertRaisesRegex(ValueError, "size"):
            self.scope["freeze"](self.request | {"limit": 1})


class StreamingBody(httpx.AsyncByteStream):
    def __init__(self, content, error=None):
        self.content, self.error, self.reads, self.closed = content, error, 0, False

    async def __aiter__(self):
        for offset in range(0, len(self.content), CHUNK_BYTES):
            self.reads += 1
            yield self.content[offset:offset + CHUNK_BYTES]
            if self.error:
                raise self.error

    async def aclose(self):
        self.closed = True


class MediaTransferTests(unittest.IsolatedAsyncioTestCase):
    def setup_transfer(self, content=b"synthetic", *, status=200, headers=None, body_error=None,
                       bad_digest=False, cleanup_error=False):
        provider = E2BProvider(E2BConfiguration(api_key="fixture"), sdk=SimpleNamespace())
        body = StreamingBody(content, body_error)
        transport = httpx.MockTransport(lambda request: httpx.Response(status, headers=headers,
                                                                         stream=body))
        client = httpx.AsyncClient(transport=transport, follow_redirects=False)
        self.client_patch = patch("app.sandbox.e2b_provider.httpx.AsyncClient", return_value=client)
        self.client_factory = self.client_patch.start()
        self.addCleanup(self.client_patch.stop)
        async def command(script, **kwargs):
            outer = shlex.split(script)
            self.assertEqual(outer[-4:-1], ["/bin/sh", "-eu", "-c"])
            inner = shlex.split(outer[-1])
            self.assertEqual(inner[:5], ["exec", "/usr/bin/python3", "-I", "-S", "-c"])
            request = json.loads(inner[-1])
            if "limit" not in request:
                if cleanup_error:
                    raise RuntimeError("private provider URL")
                return SimpleNamespace(stdout=json.dumps({"stage_id": request["stage_id"], "removed": True}))
            return SimpleNamespace(stdout=json.dumps({"stage_id": request["stage_id"],
                "size_bytes": len(content), "sha256": "0" * 64 if bad_digest else hashlib.sha256(content).hexdigest()}))
        box = SimpleNamespace(commands=SimpleNamespace(run=AsyncMock(side_effect=command)),
            download_url=Mock(return_value="https://provider.invalid/files?signature=private"))
        provider._connect = AsyncMock(return_value=box)
        provider.info = AsyncMock(return_value=SimpleNamespace(state="running"))
        handle = SimpleNamespace(sandbox_id="existing", scope=object(), policy=object())
        handle.provider_scope = handle.scope
        self.provider, self.box, self.body, self.handle = provider, box, body, handle
        return provider

    async def transfer(self, *, policy=None, max_bytes=None):
        policy = policy or FileTransferPolicy(MP4_OUTPUT_POLICY, 20 * 1024 * 1024)
        return await self.provider.read_workspace_file_snapshot(self.handle, "output/movie.mp4",
            policy=policy, max_bytes=policy.max_bytes if max_bytes is None else max_bytes)

    async def test_above_document_limit_streams_verified_chunks_and_cleans_remote_stage(self):
        content = b"v" * (9 * 1024 * 1024)
        self.setup_transfer(content)
        result = await self.transfer()
        try:
            self.assertEqual(result.size_bytes, len(content))
            self.assertEqual(result.content_sha256, hashlib.sha256(content).hexdigest())
            count = 0
            async for block in result.iter_chunks():
                self.assertLessEqual(len(block), CHUNK_BYTES)
                count += len(block)
            self.assertEqual(count, len(content))
            self.assertTrue(self.body.closed)
            self.assertEqual(self.box.commands.run.await_count, 2)
            self.assertTrue(self.box.download_url.call_args.kwargs["path"].startswith(OUTPUT_ROOT + "/"))
            self.provider._connect.assert_awaited_once_with(self.handle)
            self.provider.info.assert_awaited_once_with("existing", self.handle.scope, self.handle.policy)
        finally:
            path = result.path
            result.close()
        self.assertFalse(path.exists())

    async def test_above_32mib_export_streams_exact_chunks_and_cleans_remote_stage(self):
        content = b"v" * (33 * 1024 * 1024)
        self.setup_transfer(content)
        result = await self.transfer(policy=FileTransferPolicy(MP4_OUTPUT_POLICY, 500_000_000))
        self.addCleanup(result.close)
        digest, size = hashlib.sha256(), 0
        async for block in result.iter_chunks():
            self.assertLessEqual(len(block), CHUNK_BYTES)
            digest.update(block)
            size += len(block)
        self.assertEqual(size, len(content))
        self.assertEqual(digest.hexdigest(), hashlib.sha256(content).hexdigest())
        self.assertEqual(result.content_sha256, digest.hexdigest())
        self.assertGreater(self.body.reads, 512)
        self.assertTrue(self.body.closed)
        self.assertEqual(self.box.commands.run.await_count, 2)
        self.assertEqual(self.box.commands.run.await_args_list[0].kwargs["timeout"], 60)
        self.assertEqual(self.box.commands.run.await_args_list[1].kwargs["timeout"], 5)

    async def test_download_budget_and_signature_are_separate_from_control_timeout(self):
        self.setup_transfer()
        self.provider.configuration = E2BConfiguration(api_key="fixture",
            request_timeout_seconds=7, transfer_timeout_seconds=90.5)
        with patch("app.sandbox.e2b_provider.asyncio.timeout", wraps=asyncio.timeout) as timeout:
            result = await self.transfer()
        self.addCleanup(result.close)
        self.assertIn(unittest.mock.call(90.5), timeout.call_args_list)
        budget = self.client_factory.call_args.kwargs["timeout"]
        self.assertEqual((budget.connect, budget.pool, budget.read, budget.write), (7, 7, 30, 90.5))
        self.assertEqual(self.box.download_url.call_args.kwargs["use_signature_expiration"], 91)
        self.assertEqual(self.provider._options()["request_timeout"], 7)
        freeze = self.box.commands.run.await_args_list[0].kwargs
        self.assertEqual((freeze["timeout"], freeze["request_timeout"]), (60, 7))

    async def test_error_redirect_and_encoding_are_rejected_without_reading_body(self):
        for status, headers in ((500, {}), (302, {"location": "https://elsewhere.invalid"}),
                                (200, {"content-encoding": "gzip"}), (200, {"content-length": "999"})):
            with self.subTest(status=status, headers=headers):
                self.setup_transfer(status=status, headers=headers)
                try:
                    with self.assertRaises(SandboxError):
                        await self.transfer()
                    self.assertEqual(self.body.reads, 0)
                    self.assertTrue(self.body.closed)
                    self.assertEqual(self.box.commands.run.await_count, 2)
                finally:
                    self.client_patch.stop()

    async def test_digest_mismatch_and_disconnect_are_sanitized(self):
        for bad_digest, error in ((True, None), (False, httpx.ReadError("private signed URL"))):
            with self.subTest(bad_digest=bad_digest):
                self.setup_transfer(bad_digest=bad_digest, body_error=error)
                try:
                    with self.assertRaises(SandboxError) as caught:
                        await self.transfer()
                    self.assertNotIn("private", str(caught.exception))
                    self.assertTrue(self.body.closed)
                    self.assertEqual(self.box.commands.run.await_count, 2)
                finally:
                    self.client_patch.stop()

    async def test_cancel_preserves_cancellation_and_attempts_exact_cleanup(self):
        self.setup_transfer(body_error=asyncio.CancelledError())
        with self.assertRaises(asyncio.CancelledError):
            await self.transfer()
        self.assertTrue(self.body.closed)
        self.assertEqual(self.box.commands.run.await_count, 2)

    async def test_unknown_cleanup_never_returns_a_successful_local_snapshot(self):
        self.setup_transfer(cleanup_error=True)
        with self.assertRaisesRegex(SandboxError, "sandbox_file_cleanup_uncertain"):
            await self.transfer()

    async def test_document_policy_and_unverified_limits_do_not_connect(self):
        self.setup_transfer()
        for policy, limit in ((FileTransferPolicy(), 100),
                (FileTransferPolicy(MP4_OUTPUT_POLICY, 100), 101)):
            with self.subTest(policy=policy), self.assertRaises(ValueError):
                await self.provider.read_workspace_file_snapshot(self.handle, "x.mp4", policy=policy, max_bytes=limit)
        self.provider._connect.assert_not_awaited()


class ArtifactSnapshotOwnershipTests(unittest.IsolatedAsyncioTestCase):
    async def fixture(self):
        async def chunks():
            yield b"synthetic"
        local = await _snapshot_chunks(chunks(), max_bytes=100)
        self.addCleanup(local.close)
        service = object.__new__(SandboxArtifactTools)
        service.publisher = SimpleNamespace(files=SimpleNamespace(
            media_policy=FileTransferPolicy(MP4_OUTPUT_POLICY, 100), media_inspector=Mock()),
            publish=AsyncMock(side_effect=[TaskError("artifact_original_required", "original"), {"saved": True}]))
        service.policy = ArtifactPolicy(formats=("mp4",), transfer_policy=MP4_OUTPUT_POLICY, max_file_bytes=100)
        service.context = SimpleNamespace(claim=object(), task=SimpleNamespace(owner_id="fixture"))
        service.bridge = SimpleNamespace(_check=AsyncMock())
        service.broker = SimpleNamespace(provider=SimpleNamespace(read_workspace_file_snapshot=AsyncMock(return_value=local)),
            allocations=SimpleNamespace(handle=Mock(return_value=object()), current=AsyncMock(), request_cleanup=AsyncMock()))
        service.ready_allocation = AsyncMock(return_value={"id": "existing"})
        return service, local, {"path": "movie.mp4", "filename": "movie.mp4", "format": "mp4",
                               "operation_key": "movie", "description": "Synthetic movie"}

    async def test_success_closes_local_snapshot_after_existing_publisher(self):
        service, local, arguments = await self.fixture()
        self.assertEqual(await service.publish(arguments, "call"), {"saved": True})
        self.assertFalse(local.path.exists())
        self.assertIs(service.publisher.publish.call_args.args[1], local)

    async def test_replay_does_not_read_or_allocate_vm(self):
        service, _, arguments = await self.fixture()
        service.publisher.publish.side_effect = None
        service.publisher.publish.return_value = {"saved": True}
        await service.publish(arguments, "replay")
        service.ready_allocation.assert_not_awaited()
        service.broker.provider.read_workspace_file_snapshot.assert_not_awaited()

    async def test_revocation_closes_bytes_and_requests_existing_allocation_cleanup(self):
        service, local, arguments = await self.fixture()
        service.bridge._check.side_effect = TaskError("source_revoked", "revoked")
        with self.assertRaises(TaskError):
            await service.publish(arguments, "call")
        self.assertFalse(local.path.exists())
        service.broker.allocations.request_cleanup.assert_awaited_once()
        self.assertEqual(service.publisher.publish.await_count, 1)
