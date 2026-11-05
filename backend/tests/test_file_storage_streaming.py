"""Offline streaming/ownership checks; no storage service or database calls."""
import asyncio
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile
import unittest
from unittest.mock import patch

import httpx

from app.errors import JarvisHTTPError
from app.files.local_file import CHUNK_BYTES, LocalFileError, snapshot_file
from app.files.storage import JarvisFileStorageClient
from app.storage.supabase_config import SupabaseStorageConfig


class ResponseStream(httpx.AsyncByteStream):
    def __init__(self, chunks, *, wait_after_first=False, fail=None):
        self.chunks = chunks
        self.wait_after_first, self.fail = wait_after_first, fail
        self.entered = asyncio.Event()
        self.reads = 0
        self.closed = False

    async def __aiter__(self):
        for block in self.chunks:
            self.reads += 1
            yield block
            self.entered.set()
            if self.wait_after_first:
                await asyncio.Event().wait()
        if self.fail:
            raise self.fail

    async def aclose(self):
        self.closed = True


class StreamingTransport(httpx.AsyncBaseTransport):
    """Unlike MockTransport, never pre-buffer the outbound request body."""
    def __init__(self):
        self.calls = []
        self.upload_sizes = []
        self.upload_hash = hashlib.sha256()
        self.on_upload_chunk = None
        self.status = 200
        self.headers = {}
        self.response_stream = ResponseStream([])
        self.error = None

    async def handle_async_request(self, request):
        self.calls.append(request)
        if request.method == "POST":
            async for block in request.stream:
                self.upload_sizes.append(len(block))
                self.upload_hash.update(block)
                if self.on_upload_chunk:
                    await self.on_upload_chunk(len(self.upload_sizes))
        if self.error:
            raise self.error
        return httpx.Response(self.status, headers=self.headers, stream=self.response_stream)


class FileStorageStreamingTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="jarvis-storage-test-")
        self.root = Path(self.directory.name).resolve()
        self.created = []
        temporary = tempfile.TemporaryDirectory

        def owned_directory(**kwargs):
            directory = temporary(dir=self.root, **kwargs)
            self.created.append(Path(directory.name))
            return directory

        self.temp_patch = patch("app.files.local_file.tempfile.TemporaryDirectory", side_effect=owned_directory)
        self.temp_patch.start()
        self.storage = JarvisFileStorageClient(SupabaseStorageConfig(
            supabase_url="https://storage.invalid", service_role_key="fixture-secret",
            visual_frame_bucket="frames", files_bucket="files"))
        await self.storage.close()
        self.transport = StreamingTransport()
        self.storage._client = httpx.AsyncClient(transport=self.transport)
        self.handles = []

    async def asyncTearDown(self):
        await self.storage.close()
        for handle in self.handles:
            handle.close()
        self.temp_patch.stop()
        self.directory.cleanup()

    async def local(self, content=b"verified content"):
        original = self.root / "original"
        original.write_bytes(content)
        handle = await snapshot_file(original, max_bytes=len(content), expected_size=len(content),
            expected_sha256=hashlib.sha256(content).hexdigest())
        self.handles.append(handle)
        return handle

    async def upload(self, handle):
        return await self.storage.upload_file(path="owned/exact", local_file=handle, content_type="video/mp4")

    async def download(self, limit=1024):
        result = await self.storage.read_object_file(bucket="private", path="owned/exact", max_bytes=limit)
        if result:
            self.handles.append(result)
        return result

    def assert_no_partial(self):
        self.assertTrue(all(not directory.exists() for directory in self.created))

    async def test_private_snapshot_is_independent_and_cleanup_is_owned(self):
        handle = await self.local()
        self.assertEqual(stat.S_IMODE(handle.path.parent.stat().st_mode), 0o700)
        self.assertEqual(stat.S_IMODE(handle.path.stat().st_mode), 0o400)
        (self.root / "original").write_bytes(b"later original content")
        self.assertEqual(b"".join([block async for block in handle.iter_chunks()]), b"verified content")
        handle.close(); handle.close()
        self.assertFalse(handle.path.exists())
        with self.assertRaises(LocalFileError) as error:
            await anext(handle.iter_chunks())
        self.assertEqual(error.exception.code, "local_file_closed")

    async def test_symlink_leaf_parent_directory_and_fifo_are_rejected(self):
        original = self.root / "original"
        original.write_bytes(b"x")
        symlink = self.root / "link"
        symlink.symlink_to(original)
        linked_parent = self.root / "linked-parent"
        linked_parent.symlink_to(self.root, target_is_directory=True)
        fifo = self.root / "pipe"
        os.mkfifo(fifo)
        for path in (symlink, linked_parent / "original", self.root, fifo):
            with self.subTest(path=path.name), self.assertRaises(LocalFileError):
                await snapshot_file(path, max_bytes=10)
        self.assert_no_partial()

    async def test_wrong_snapshot_length_hash_and_limit_leave_no_files(self):
        original = self.root / "original"
        original.write_bytes(b"abc")
        for kwargs, code in (({"max_bytes": 2}, "local_file_size_exceeded"),
                             ({"max_bytes": 4, "expected_size": 4}, "local_file_length_mismatch"),
                             ({"max_bytes": 3, "expected_sha256": "0" * 64}, "local_file_digest_mismatch")):
            with self.subTest(code=code), self.assertRaises(LocalFileError) as caught:
                await snapshot_file(original, **kwargs)
            self.assertEqual(caught.exception.code, code)
            self.assert_no_partial()

    async def test_source_mutation_during_snapshot_is_rejected(self):
        original = self.root / "original"
        original.write_bytes(b"a" * (CHUNK_BYTES * 2))
        read = os.read
        changed = False

        def mutate(descriptor, length):
            nonlocal changed
            result = read(descriptor, length)
            if not changed:
                changed = True
                original.write_bytes(b"b" * (CHUNK_BYTES * 2))
            return result

        with patch("app.files.local_file.os.read", side_effect=mutate), self.assertRaises(LocalFileError) as caught:
            await snapshot_file(original, max_bytes=CHUNK_BYTES * 2)
        self.assertEqual(caught.exception.code, "local_file_changed")
        self.assert_no_partial()

    async def test_more_than_eight_mib_upload_is_chunked_exact_and_non_upserting(self):
        raw = b"x" * (8 * 1024 * 1024 + 37)
        handle = await self.local(raw)
        self.transport.status = 201
        self.transport.response_stream = ResponseStream([b"secret-provider-response"])
        result = await self.upload(handle)
        self.assertEqual((result.path, result.size_bytes), ("owned/exact", len(raw)))
        self.assertEqual(sum(self.transport.upload_sizes), len(raw))
        self.assertLessEqual(max(self.transport.upload_sizes), CHUNK_BYTES)
        self.assertGreater(len(self.transport.upload_sizes), 128)
        self.assertEqual(self.transport.upload_hash.hexdigest(), handle.content_sha256)
        request = self.transport.calls[0]
        self.assertEqual(request.headers["Content-Length"], str(len(raw)))
        self.assertEqual(request.headers["x-upsert"], "false")
        self.assertNotIn("Transfer-Encoding", request.headers)
        self.assertEqual(self.transport.response_stream.reads, 0)
        self.assertTrue(self.transport.response_stream.closed)
        self.assertTrue(handle.path.exists())  # Caller still owns this snapshot.

    async def test_changed_private_snapshot_is_rejected_before_request(self):
        handle = await self.local()
        handle.path.chmod(0o600)
        handle.path.write_bytes(b"different bytes")
        with self.assertRaises(LocalFileError):
            await self.upload(handle)
        self.assertEqual(self.transport.calls, [])

    async def test_chunk_hash_detects_change_between_stat_check_and_read(self):
        handle = await self.local(b"a" * CHUNK_BYTES)
        with patch("app.files.local_file.os.pread", return_value=b"b" * CHUNK_BYTES):
            with self.assertRaises(LocalFileError) as caught:
                await self.upload(handle)
        self.assertEqual(caught.exception.code, "local_file_changed")
        self.assertEqual(self.transport.upload_sizes, [])
        self.assertEqual(len(self.transport.calls), 1)

    async def test_mutation_after_first_upload_chunk_never_yields_changed_bytes_or_retries(self):
        handle = await self.local(b"a" * (CHUNK_BYTES * 3))

        async def mutate(index):
            if index == 1:
                handle.path.chmod(0o600)
                handle.path.write_bytes(b"b" * handle.size_bytes)

        self.transport.on_upload_chunk = mutate
        with self.assertRaises(LocalFileError):
            await self.upload(handle)
        self.assertEqual(self.transport.upload_sizes, [CHUNK_BYTES])
        self.assertEqual(len(self.transport.calls), 1)

    async def test_upload_transport_error_or_rejection_is_sanitized_without_resubmit(self):
        handle = await self.local()
        for failure in (httpx.ReadError("SECRET URL TOKEN"), None):
            with self.subTest(transport_error=failure is not None):
                self.transport.calls.clear()
                self.transport.error = failure
                self.transport.status = 409
                self.transport.response_stream = ResponseStream([b"SECRET BODY"])
                with self.assertRaises(JarvisHTTPError) as caught:
                    await self.upload(handle)
                self.assertEqual(len(self.transport.calls), 1)
                self.assertNotIn("SECRET", str(caught.exception) + str(caught.exception.details))
                self.assertEqual(self.transport.response_stream.reads, 0)

    async def test_more_than_eight_mib_download_returns_exact_owned_file(self):
        count, tail = 129, b"tail"
        block = b"d" * CHUNK_BYTES
        size = len(block) * count + len(tail)
        self.transport.headers = {"Content-Length": str(size)}
        self.transport.response_stream = ResponseStream([block] * count + [tail])
        handle = await self.download(size)
        self.assertEqual(handle.size_bytes, size)
        checksum, lengths = hashlib.sha256(), []
        async for part in handle.iter_chunks():
            lengths.append(len(part)); checksum.update(part)
        expected = hashlib.sha256()
        for _ in range(count): expected.update(block)
        expected.update(tail)
        self.assertEqual(handle.content_sha256, expected.hexdigest())
        self.assertEqual(checksum.hexdigest(), expected.hexdigest())
        self.assertLessEqual(max(lengths), CHUNK_BYTES)
        self.assertTrue(self.transport.response_stream.closed)
        self.assertEqual(self.transport.calls[0].headers["Accept-Encoding"], "identity")

    async def test_declared_overflow_rejected_before_body(self):
        self.transport.headers = {"Content-Length": "10000000"}
        self.transport.response_stream = ResponseStream([b"unused"])
        with self.assertRaises(JarvisHTTPError) as caught:
            await self.download(5)
        self.assertEqual(caught.exception.code, "jarvis_file_size_exceeded")
        self.assertEqual(self.transport.response_stream.reads, 0)
        self.assertTrue(self.transport.response_stream.closed)
        self.assert_no_partial()

    async def test_undeclared_overflow_wrong_length_or_encoding_cleans_partial(self):
        for headers, code in (({}, "jarvis_file_size_exceeded"),
                ({"Content-Length": "5"}, "jarvis_file_length_mismatch"),
                ({"Content-Length": "10"}, "jarvis_file_length_mismatch"),
                ({"Content-Length": "invalid"}, "jarvis_file_length_mismatch"),
                ({"Content-Encoding": "gzip"}, "jarvis_file_read_failed")):
            with self.subTest(headers=headers):
                self.transport.headers = headers
                self.transport.response_stream = ResponseStream([b"123", b"456"])
                with self.assertRaises(JarvisHTTPError) as caught:
                    await self.download(5 if not headers else 20)
                self.assertEqual(caught.exception.code, code)
                self.assertTrue(self.transport.response_stream.closed)
                self.assert_no_partial()

    async def test_only_exact_missing_receipt_is_absent(self):
        for status, body, absent in ((400, {"code": "NoSuchKey", "statusCode": "404"}, True),
                (404, {"code": "NoSuchKey", "statusCode": 404}, True),
                (404, {"code": "AccessDenied", "statusCode": 404}, False),
                (403, {"code": "NoSuchKey", "statusCode": 404}, False),
                (500, {"code": "NoSuchKey", "statusCode": 404}, False),
                (404, {"code": "NoSuchKey", "statusCode": 403}, False),
                (404, {"message": "SECRET TOKEN"}, False)):
            with self.subTest(status=status, body=body):
                self.transport.status = status
                self.transport.response_stream = ResponseStream([json.dumps(body).encode()])
                if absent:
                    self.assertIsNone(await self.download())
                else:
                    with self.assertRaises(JarvisHTTPError) as caught:
                        await self.download()
                    self.assertEqual(caught.exception.details, {"status_code": status})
                self.assertTrue(self.transport.response_stream.closed)
                self.assert_no_partial()

    async def test_error_body_is_bounded_and_never_echoed(self):
        self.transport.status = 404
        self.transport.response_stream = ResponseStream([b"SECRET" * 400, b"unread"])
        with self.assertRaises(JarvisHTTPError) as caught:
            await self.download()
        self.assertEqual(self.transport.response_stream.reads, 1)
        self.assertEqual(caught.exception.details, {"status_code": 404})
        self.assert_no_partial()

    async def test_partial_download_cancellation_closes_response_and_removes_file(self):
        stream = ResponseStream([b"p" * CHUNK_BYTES], wait_after_first=True)
        self.transport.response_stream = stream
        pending = asyncio.create_task(self.download(CHUNK_BYTES * 2))
        await asyncio.wait_for(stream.entered.wait(), 1)
        self.assertTrue(any(directory.exists() for directory in self.created))
        pending.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await pending
        self.assertTrue(stream.closed)
        self.assert_no_partial()

    async def test_cancelled_snapshot_removes_its_partial_file(self):
        original = self.root / "original"
        original.write_bytes(b"s" * (CHUNK_BYTES * 3))
        entered = asyncio.Event()
        write = os.write

        def observe(descriptor, block):
            result = write(descriptor, block)
            entered.set()
            return result

        # A Mock retains memoryview call arguments and would artificially keep
        # the writer's bytearray exported after the real write has returned.
        with patch("app.files.local_file.os.write", new=observe):
            pending = asyncio.create_task(snapshot_file(original, max_bytes=CHUNK_BYTES * 3))
            await entered.wait()
            pending.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await pending
        self.assert_no_partial()

    async def test_partial_download_transport_failure_is_unknown_and_cleans(self):
        self.transport.response_stream = ResponseStream([b"p" * CHUNK_BYTES], fail=httpx.ReadError("SECRET"))
        with self.assertRaises(JarvisHTTPError) as caught:
            await self.download(CHUNK_BYTES * 2)
        self.assertEqual(caught.exception.code, "jarvis_file_read_failed")
        self.assertNotIn("SECRET", str(caught.exception))
        self.assertEqual(len(self.transport.calls), 1)
        self.assertTrue(self.transport.response_stream.closed)
        self.assert_no_partial()

    async def test_zero_length_and_invalid_bounds(self):
        self.transport.headers = {"Content-Length": "0"}
        handle = await self.download(0)
        self.assertEqual(handle.size_bytes, 0)
        self.assertEqual(handle.content_sha256, hashlib.sha256(b"").hexdigest())
        self.assertEqual([part async for part in handle.iter_chunks()], [])
        for bound in (-1, False, 1.5):
            with self.subTest(bound=bound), self.assertRaises(ValueError):
                await self.download(bound)

    async def test_existing_byte_methods_keep_their_return_shapes_and_missing_classification(self):
        uploaded = await self.storage.upload_bytes(path="old/bytes", content=b"old", content_type="text/plain")
        self.assertEqual((uploaded.bucket, uploaded.path, uploaded.size_bytes), ("files", "old/bytes", 3))
        self.transport.response_stream = ResponseStream([b"old"])
        self.assertEqual(await self.storage.read_object(bucket="files", path="old/bytes", max_bytes=3), b"old")
        self.transport.status = 400
        self.transport.response_stream = ResponseStream([b'{"code":"NoSuchKey","statusCode":404}'])
        self.assertIsNone(await self.storage.read_object(bucket="files", path="missing", max_bytes=3))


if __name__ == "__main__":
    unittest.main()
