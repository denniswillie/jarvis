from __future__ import annotations

import asyncio
import unittest

import httpx

from app.errors import JarvisHTTPError
from app.storage.supabase_config import SupabaseStorageConfig
from app.visual_memory.storage import VisualFrameStorageClient


class _Stream(httpx.AsyncByteStream):
    def __init__(self, chunks, *, block=False):
        self.chunks = chunks
        self.block = block
        self.entered = asyncio.Event()
        self.closed = False
        self.reads = 0

    async def __aiter__(self):
        self.entered.set()
        if self.block:
            await asyncio.Event().wait()
        for chunk in self.chunks:
            self.reads += 1
            yield chunk

    async def aclose(self):
        self.closed = True


class VisualFrameStorageBoundedTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.storage = VisualFrameStorageClient(SupabaseStorageConfig(
            supabase_url="https://storage.invalid", service_role_key="fixture-secret",
            visual_frame_bucket="frames", files_bucket="files",
        ))
        await self.storage._client.aclose()
        self.status = 200
        self.headers = {}
        self.stream = _Stream([b"123", b"456"])

        def respond(request):
            self.request = request
            return httpx.Response(self.status, headers=self.headers, stream=self.stream)

        self.storage._client = httpx.AsyncClient(transport=httpx.MockTransport(respond))

    async def asyncTearDown(self):
        await self.storage.close()

    async def download(self, max_bytes=8):
        return await self.storage.download_object(bucket="private", path="owner/frame.jpg", max_bytes=max_bytes)

    async def test_bounded_download_returns_pixels_and_closes_response(self):
        self.assertEqual(await self.download(), b"123456")
        self.assertTrue(self.stream.closed)
        self.assertEqual(self.request.headers["Authorization"], "Bearer fixture-secret")

    async def test_large_declared_size_is_rejected_without_reading_body(self):
        self.headers = {"Content-Length": "1000000"}
        with self.assertRaises(JarvisHTTPError) as caught:
            await self.download()
        self.assertEqual(caught.exception.code, "visual_frame_storage_download_too_large")
        self.assertEqual(self.stream.reads, 0)
        self.assertTrue(self.stream.closed)

    async def test_missing_or_false_content_length_cannot_bypass_stream_bound(self):
        for headers in ({}, {"Content-Length": "1"}, {"Content-Length": "invalid"}):
            with self.subTest(headers=headers):
                self.headers = headers
                self.stream = _Stream([b"12345", b"67890", b"never needed"])
                with self.assertRaises(JarvisHTTPError) as caught:
                    await self.download()
                self.assertEqual(caught.exception.code, "visual_frame_storage_download_too_large")
                self.assertEqual(self.stream.reads, 2)
                self.assertTrue(self.stream.closed)

    async def test_error_body_is_not_loaded_or_exposed(self):
        self.status = 404
        self.stream = _Stream([b"SECRET_STORAGE_ERROR"])
        with self.assertRaises(JarvisHTTPError) as caught:
            await self.download()
        self.assertEqual(caught.exception.details, {"status_code": 404})
        self.assertEqual(self.stream.reads, 0)
        self.assertTrue(self.stream.closed)

    async def test_cancellation_closes_stream(self):
        self.stream = _Stream([], block=True)
        task = asyncio.create_task(self.download())
        await asyncio.wait_for(self.stream.entered.wait(), 1)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertTrue(self.stream.closed)

    async def test_timeout_closes_stream(self):
        self.stream = _Stream([], block=True)
        with self.assertRaises(TimeoutError):
            async with asyncio.timeout(0.02):
                await self.download()
        self.assertTrue(self.stream.closed)

    async def test_legacy_call_without_limit_keeps_existing_behavior(self):
        self.stream = _Stream([b"12345", b"67890"])
        self.assertEqual(await self.download(max_bytes=None), b"1234567890")

    async def test_nonpositive_limit_is_rejected_before_request(self):
        for limit in (0, -1, False):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                await self.download(max_bytes=limit)
        self.assertEqual(self.stream.reads, 0)
