"""Offline input-only transport/installer checks; no VM, SQL or media renderer."""
import asyncio
from email.parser import BytesParser
from email.policy import default
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

from app.files.local_file import CHUNK_BYTES, LocalFileError, _snapshot_chunks
from app.files.transfer import FileTransferPolicy, MP4_CAPTURE_POLICY, MP4_OUTPUT_POLICY
from app.sandbox.e2b_provider import E2BProvider
from app.sandbox.file_transfer import INPUT_ROOT, INSTALL_INPUT_FILE
from app.sandbox.models import E2BConfiguration, SandboxError


class InstallerTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.incoming, self.inputs = self.root / "incoming", self.root / "inputs"
        self.incoming.mkdir(mode=0o700)
        self.inputs.mkdir(mode=0o755)
        self.scope = {"__name__": "fixture"}
        exec(INSTALL_INPUT_FILE, self.scope)
        self.read_sizes = []

        def fstat(fd):
            info = os.fstat(fd)
            values = {key: getattr(info, key) for key in (
                "st_dev", "st_ino", "st_mode", "st_uid", "st_size", "st_mtime_ns", "st_ctime_ns")}
            values["st_uid"] = 0
            return SimpleNamespace(**values)

        def read(fd, count):
            self.read_sizes.append(count)
            return os.read(fd, count)

        def entry_stat(*args, **kwargs):
            info = os.stat(*args, **kwargs)
            return SimpleNamespace(**{key: 0 if key == "st_uid" else getattr(info, key) for key in (
                "st_dev", "st_ino", "st_mode", "st_uid", "st_size", "st_mtime_ns", "st_ctime_ns")})

        self.proxy = SimpleNamespace(**{name: getattr(os, name) for name in (
            "close", "fchmod", "link", "unlink", "O_RDONLY", "O_DIRECTORY", "O_NOFOLLOW", "O_NONBLOCK")})
        self.proxy.open = lambda path, *args, **kwargs: os.open(self.root if path == "/" else path, *args, **kwargs)
        self.proxy.fstat, self.proxy.stat, self.proxy.read = fstat, entry_stat, read
        self.scope.update(os=self.proxy, INCOMING="/incoming", INPUTS="/inputs")
        self.request = {"identity": "a" * 64, "stage_id": "b" * 32, "policy": MP4_CAPTURE_POLICY,
                        "limit": 64 * 1024 * 1024}

    def create(self, content=b"exact retained media", *, stage_id=None):
        request = self.request | {"size": len(content), "sha256": hashlib.sha256(content).hexdigest()}
        if stage_id:
            request["stage_id"] = stage_id
        source = self.incoming / request["stage_id"]
        source.write_bytes(content)
        source.chmod(0o600)
        return request, source

    def test_large_chunked_immutable_install_and_identical_existing_target(self):
        content = b"v" * (9 * 1024 * 1024)
        request, source = self.create(content)
        receipt = self.scope["install"](request)
        target = self.inputs / request["identity"]
        self.assertEqual(receipt, {"path": "/inputs/" + request["identity"],
                                  "sha256": hashlib.sha256(content).hexdigest(), "size_bytes": len(content)})
        self.assertFalse(source.exists())
        self.assertEqual(target.read_bytes(), content)
        self.assertEqual(stat.S_IMODE(target.stat().st_mode), 0o444)
        inode = target.stat().st_ino
        repeat, _ = self.create(content, stage_id="c" * 32)
        self.assertEqual(self.scope["install"](repeat), receipt)
        self.assertEqual(target.stat().st_ino, inode)
        self.assertTrue(self.read_sizes)
        self.assertEqual(set(self.read_sizes), {CHUNK_BYTES})

    def test_conflict_never_overwrites_existing_target_and_remains_unclean(self):
        request, source = self.create()
        target = self.inputs / request["identity"]
        target.write_bytes(b"old version")
        target.chmod(0o444)
        with self.assertRaises(ValueError):
            self.scope["install"](request)
        self.assertEqual(target.read_bytes(), b"old version")
        self.assertTrue(source.exists())  # owner must retire uncertain allocation

    def test_wrong_hash_mutation_and_policy_bounds_fail(self):
        request, source = self.create()
        with self.assertRaisesRegex(ValueError, "digest"):
            self.scope["install"](request | {"sha256": "0" * 64})
        for patch_value in ({"policy": "document_v1"}, {"limit": True}, {"size": 0},
                            {"limit": 1}, {"policy": MP4_OUTPUT_POLICY, "limit": 500_000_001},
                            {"identity": "../escape"}, {"stage_id": "../escape"}):
            with self.subTest(patch_value=patch_value), self.assertRaises(ValueError):
                self.scope["install"](request | patch_value)
        original = self.proxy.read
        def changed(fd, count):
            block = original(fd, count)
            if block:
                source.write_bytes(b"changed" + b"!" * 12)
            return block
        self.proxy.read = changed
        with self.assertRaisesRegex(ValueError, "changed|size"):
            self.scope["install"](request)
        self.assertFalse((self.inputs / request["identity"]).exists())

    def test_symlinks_nonregular_and_unsafe_permissions_fail(self):
        request, source = self.create()
        source.unlink()
        source.symlink_to(self.inputs)
        with self.assertRaises(OSError):
            self.scope["install"](request)
        source.unlink()
        source.mkdir()
        with self.assertRaisesRegex(ValueError, "protection"):
            self.scope["install"](request)
        source.rmdir()
        request, source = self.create()
        source.chmod(0o666)
        with self.assertRaisesRegex(ValueError, "protection"):
            self.scope["install"](request)
        source.chmod(0o600)
        self.inputs.chmod(0o777)
        with self.assertRaisesRegex(ValueError, "protection"):
            self.scope["install"](request)
        self.inputs.chmod(0o755)
        self.inputs.rename(self.root / "real-inputs")
        self.inputs.symlink_to(self.root / "real-inputs", target_is_directory=True)
        with self.assertRaises(OSError):
            self.scope["install"](request)


class RequestStreamingTransport(httpx.AsyncBaseTransport):
    """Unlike MockTransport, do not read/buffer the whole body before handler."""
    def __init__(self, handler):
        self.handler, self.closed = handler, False

    async def handle_async_request(self, request):
        return await self.handler(request)

    async def aclose(self):
        self.closed = True


class ReplyBody(httpx.AsyncByteStream):
    def __init__(self, size):
        self.size = size

    async def __aiter__(self):
        for offset in range(0, self.size, 2048):
            yield b"x" * min(2048, self.size - offset)


class InputTransferTests(unittest.IsolatedAsyncioTestCase):
    async def setup_transfer(self, content=None, *, status=200, url=None, receipt_bad=False,
                             upload_failure=None, during_upload=None, after_install=None,
                             response_size=2):
        content = content if content is not None else b"synthetic recording"
        async def chunks():
            for offset in range(0, len(content), CHUNK_BYTES):
                yield content[offset:offset + CHUNK_BYTES]
        self.local = await _snapshot_chunks(chunks(), max_bytes=len(content))
        self.addCleanup(self.local.close)
        self.provider = E2BProvider(E2BConfiguration(api_key="fixture"), sdk=SimpleNamespace())
        self.received, self.chunk_sizes, self.requests = [], [], []
        async def upload(request):
            self.requests.append(request)
            async for chunk in request.stream:
                self.chunk_sizes.append(len(chunk))
                self.received.append(chunk)
                if during_upload:
                    during_upload()
                if upload_failure:
                    raise upload_failure
            return httpx.Response(status, stream=ReplyBody(response_size))
        self.transport = RequestStreamingTransport(upload)
        client = httpx.AsyncClient(transport=self.transport, follow_redirects=False)
        client_patch = patch("app.sandbox.e2b_provider.httpx.AsyncClient", return_value=client)
        self.client_factory = client_patch.start()
        self.addCleanup(client_patch.stop)
        async def install(script, **kwargs):
            outer = shlex.split(script)
            self.assertEqual(outer[:3], ["exec", "/usr/bin/env", "-i"])
            self.assertEqual(outer[-4:-1], ["/bin/sh", "-eu", "-c"])
            inner = shlex.split(outer[-1])
            self.assertEqual(inner[:5], ["exec", "/usr/bin/python3", "-I", "-S", "-c"])
            self.assertEqual(inner[5], INSTALL_INPUT_FILE)
            self.install_request = json.loads(inner[6])
            value = {"path": INPUT_ROOT + "/" + self.install_request["identity"],
                     "sha256": self.install_request["sha256"], "size_bytes": self.install_request["size"]}
            if after_install:
                after_install()
            return SimpleNamespace(stdout=json.dumps(value | ({"size_bytes": 0} if receipt_bad else {})))
        self.box = SimpleNamespace(upload_url=Mock(return_value=url or "https://provider.invalid/files?signature=private"),
                                   commands=SimpleNamespace(run=AsyncMock(side_effect=install)))
        self.provider._connect = AsyncMock(return_value=self.box)
        self.provider.info = AsyncMock(return_value=SimpleNamespace(state="running"))
        self.handle = SimpleNamespace(sandbox_id="existing", scope=object(), policy=object())
        self.handle.provider_scope = self.handle.scope

    async def transfer(self, policy=None):
        return await self.provider.stage_input_file(self.handle, "a" * 64, self.local,
            policy=policy or FileTransferPolicy(MP4_CAPTURE_POLICY, 16 * 1024 * 1024))

    def assert_local_owned(self):
        self.assertTrue(self.local.path.exists())
        self.assertFalse(self.local._closed)

    async def test_above_8mib_streams_exact_multipart_without_base64_or_movie_in_command(self):
        content = b"v" * (9 * 1024 * 1024)
        await self.setup_transfer(content)
        result = await self.transfer()
        self.assertEqual(result["size_bytes"], len(content))
        self.assertEqual(result["sha256"], hashlib.sha256(content).hexdigest())
        request = self.requests[0]
        envelope = b"".join(self.received)  # test parser only, never production
        parsed = BytesParser(policy=default).parsebytes(
            b"Content-Type: " + request.headers["Content-Type"].encode() + b"\r\n\r\n" + envelope)
        parts = list(parsed.iter_parts())
        self.assertEqual(len(parts), 1)
        self.assertEqual(parts[0].get_payload(decode=True), content)
        self.assertEqual(len(envelope), int(request.headers["Content-Length"]))
        self.assertEqual(parts[0].get_param("name", header="content-disposition"), "file")
        self.assertLessEqual(max(self.chunk_sizes), CHUNK_BYTES)
        self.assertEqual(self.install_request["limit"], 16 * 1024 * 1024)
        self.assertEqual(self.install_request["policy"], MP4_CAPTURE_POLICY)
        self.assertNotIn(self.local.path.as_posix(), self.box.commands.run.call_args.args[0])
        self.assertLess(len(self.box.commands.run.call_args.args[0]), 10000)
        self.assertTrue(self.transport.closed)
        self.box.commands.run.assert_awaited_once()
        self.assertEqual(self.box.upload_url.call_args.kwargs["user"], "root")
        self.assertEqual(self.box.upload_url.call_args.kwargs["use_signature_expiration"], 300)
        self.assert_local_owned()

    async def test_above_32mib_output_policy_upload_remains_chunked_and_exact(self):
        content = b"v" * (33 * 1024 * 1024)
        await self.setup_transfer(content)
        result = await self.transfer(FileTransferPolicy(MP4_OUTPUT_POLICY, 500_000_000))
        self.assertEqual(result["size_bytes"], len(content))
        self.assertEqual(result["sha256"], hashlib.sha256(content).hexdigest())
        self.assertLessEqual(max(self.chunk_sizes), CHUNK_BYTES)
        self.assertGreater(len(self.chunk_sizes), 512)
        self.assertEqual(sum(self.chunk_sizes), int(self.requests[0].headers["Content-Length"]))
        self.assertEqual(self.install_request["size"], len(content))
        self.assertEqual(self.install_request["limit"], 500_000_000)
        self.assertEqual(self.install_request["policy"], MP4_OUTPUT_POLICY)
        self.assertEqual(len(self.requests), 1)
        self.box.commands.run.assert_awaited_once()
        self.assertLess(len(self.box.commands.run.call_args.args[0]), 10000)
        self.assertTrue(self.transport.closed)
        self.assert_local_owned()

    async def test_transfer_budget_is_separate_from_control_rpc_budget(self):
        await self.setup_transfer()
        with patch("app.sandbox.e2b_provider.asyncio.timeout", wraps=asyncio.timeout) as timeout:
            await self.transfer()
        self.assertIn(unittest.mock.call(300), timeout.call_args_list)
        upload_timeout = self.client_factory.call_args.kwargs["timeout"]
        self.assertEqual(upload_timeout.write, 300)
        self.assertEqual(upload_timeout.connect, 20)
        self.assertEqual(upload_timeout.pool, 20)
        self.assertEqual(upload_timeout.read, 30)
        self.assertEqual(self.provider._options()["request_timeout"], 20)
        self.assertEqual(self.box.commands.run.call_args.kwargs["request_timeout"], 20)
        self.assertEqual(self.box.commands.run.call_args.kwargs["timeout"], 60)

    async def test_custom_transfer_budget_and_signature_are_independent(self):
        await self.setup_transfer()
        self.provider.configuration = E2BConfiguration(api_key="fixture",
            request_timeout_seconds=7, transfer_timeout_seconds=90.5)
        with patch("app.sandbox.e2b_provider.asyncio.timeout", wraps=asyncio.timeout) as timeout:
            await self.transfer()
        self.assertIn(unittest.mock.call(90.5), timeout.call_args_list)
        self.assertEqual(self.client_factory.call_args.kwargs["timeout"].write, 90.5)
        self.assertEqual(self.box.upload_url.call_args.kwargs["use_signature_expiration"], 91)
        self.assertEqual(self.provider._options()["request_timeout"], 7)
        self.assertEqual(self.box.commands.run.call_args.kwargs["request_timeout"], 7)

    def test_transfer_budget_is_finite_and_bounded(self):
        for value in (0, -1, 301, float("inf"), float("nan")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                E2BConfiguration(api_key="fixture", transfer_timeout_seconds=value)

    async def test_transfer_timeout_never_retries_or_installs(self):
        await self.setup_transfer(upload_failure=httpx.WriteTimeout("fixture"))
        with self.assertRaisesRegex(SandboxError, "^sandbox_input_cleanup_uncertain$"):
            await self.transfer()
        self.assertEqual(len(self.requests), 1)
        self.box.commands.run.assert_not_awaited()
        self.assertTrue(self.transport.closed)
        self.assert_local_owned()

    async def test_output_policy_retains_narrowed_limit(self):
        await self.setup_transfer()
        await self.transfer(FileTransferPolicy(MP4_OUTPUT_POLICY, 100))
        self.assertEqual(self.install_request["limit"], 100)
        self.assertEqual(self.install_request["policy"], MP4_OUTPUT_POLICY)

    async def test_document_or_over_limit_rejected_before_connection(self):
        await self.setup_transfer()
        for policy in (FileTransferPolicy(), FileTransferPolicy(MP4_CAPTURE_POLICY, 1)):
            with self.subTest(policy=policy), self.assertRaises(ValueError):
                await self.transfer(policy)
        self.provider._connect.assert_not_awaited()
        self.assert_local_owned()

    async def test_invalid_url_no_upload_or_install(self):
        await self.setup_transfer(url="http://provider.invalid/files?signature=private")
        for url in ("http://provider.invalid/files", "https://user:secret@provider.invalid/files",
                    "https://provider.invalid/wrong", "https://provider.invalid/files#fragment"):
            self.box.upload_url.return_value = url
            with self.subTest(url=url), self.assertRaisesRegex(SandboxError, "sandbox_input_upload_invalid"):
                await self.transfer()
        self.assertEqual(self.requests, [])
        self.box.commands.run.assert_not_awaited()
        self.assert_local_owned()

    async def test_modified_local_before_write_has_no_remote_effect(self):
        await self.setup_transfer()
        self.local.path.chmod(0o600)
        self.local.path.write_bytes(b"changed")
        with self.assertRaises(LocalFileError):
            await self.transfer()
        self.provider._connect.assert_not_awaited()
        self.assert_local_owned()

    async def test_midstream_mutation_is_uncertain_and_never_installs(self):
        def mutate():
            self.local.path.chmod(0o600)
        await self.setup_transfer(during_upload=mutate)
        with self.assertRaisesRegex(SandboxError, "sandbox_input_cleanup_uncertain"):
            await self.transfer()
        self.assertEqual(len(self.requests), 1)
        self.box.commands.run.assert_not_awaited()
        self.assertTrue(self.transport.closed)
        self.assert_local_owned()

    async def test_lost_upload_or_redirect_does_not_retry_or_install(self):
        await self.setup_transfer(upload_failure=RuntimeError("private signature"))
        with self.assertRaisesRegex(SandboxError, "^sandbox_input_cleanup_uncertain$"):
            await self.transfer()
        self.assertEqual(len(self.requests), 1)
        self.box.commands.run.assert_not_awaited()
        self.assert_local_owned()

    async def test_redirect_rejected_without_follow_or_install(self):
        await self.setup_transfer(status=307)
        with self.assertRaisesRegex(SandboxError, "sandbox_input_cleanup_uncertain"):
            await self.transfer()
        self.assertEqual(len(self.requests), 1)
        self.box.commands.run.assert_not_awaited()

    async def test_bad_remote_receipt_has_no_second_install(self):
        await self.setup_transfer(receipt_bad=True)
        with self.assertRaisesRegex(SandboxError, "sandbox_input_cleanup_uncertain"):
            await self.transfer()
        self.box.commands.run.assert_awaited_once()
        self.assert_local_owned()

    async def test_bool_size_receipt_cannot_equal_one_byte_integer(self):
        await self.setup_transfer(b"x")
        self.box.commands.run.side_effect = None
        self.box.commands.run.return_value = SimpleNamespace(stdout=json.dumps({
            "path": INPUT_ROOT + "/" + "a" * 64, "sha256": self.local.content_sha256, "size_bytes": True}))
        with self.assertRaisesRegex(SandboxError, "sandbox_input_cleanup_uncertain"):
            await self.transfer()
        self.box.commands.run.assert_awaited_once()

    async def test_oversize_upload_response_rejected(self):
        await self.setup_transfer(response_size=4097)
        with self.assertRaisesRegex(SandboxError, "sandbox_input_cleanup_uncertain"):
            await self.transfer()
        self.box.commands.run.assert_not_awaited()

    async def test_after_install_mutation_requires_cleanup(self):
        await self.setup_transfer(after_install=lambda: self.local.path.chmod(0o600))
        with self.assertRaisesRegex(SandboxError, "sandbox_input_cleanup_uncertain"):
            await self.transfer()
        self.box.commands.run.assert_awaited_once()
        self.assert_local_owned()

    async def test_cancel_propagates_closes_transport_but_not_caller_file(self):
        await self.setup_transfer(upload_failure=asyncio.CancelledError())
        with self.assertRaises(asyncio.CancelledError):
            await self.transfer()
        self.box.commands.run.assert_not_awaited()
        self.assertTrue(self.transport.closed)
        self.assert_local_owned()

    async def test_allocation_lost_after_install_is_not_success(self):
        await self.setup_transfer()
        self.provider.info.return_value = SimpleNamespace(state="paused")
        with self.assertRaisesRegex(SandboxError, "sandbox_input_cleanup_uncertain"):
            await self.transfer()
        self.box.commands.run.assert_awaited_once()
        self.assert_local_owned()


if __name__ == "__main__":
    unittest.main()
