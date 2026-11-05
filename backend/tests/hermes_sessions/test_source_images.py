import base64
from contextlib import asynccontextmanager
from dataclasses import replace
from io import BytesIO
from types import SimpleNamespace as NS
import unittest
from unittest.mock import AsyncMock
from uuid import uuid4

from PIL import Image
from pydantic import ValidationError

from app.hermes_sessions.domain_scope import DomainInputScope
from app.hermes_sessions.domain_transport import _native_image_result, MAX_RESPONSE
from app.hermes_sessions.source_images import RetainedSandboxImages
from app.images import ImageUnavailableError


class SourceImageTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        data = BytesIO()
        Image.new("RGB", (32, 16), "blue").save(data, format="PNG")
        self.content = data.getvalue()
        self.check = AsyncMock()
        self.provider = NS(read_workspace_file=AsyncMock(return_value=self.content))
        self.entered = []
        @asynccontextmanager
        async def use(session):
            self.entered.append(session)
            yield NS(sandbox_id="same-sandbox")
        self.sandboxes = NS(provider=self.provider, inspect=AsyncMock(), use=use)
        scope = DomainInputScope("owner", "master", "input", uuid4(), frozenset({"sandbox_view_image"}))
        self.images = RetainedSandboxImages(scope, lambda: self.sandboxes, self.check)

    async def test_pixels_scoped_to_same_sandbox_and_native_envelope(self):
        result = await self.images.select({"path": "screen.png"})
        resolved = await self.images.resolve(result)
        envelope = await _native_image_result(NS(payload=result, image=resolved))
        self.assertTrue(envelope["_multimodal"])
        url = envelope["content"][1]["image_url"]["url"]
        self.assertEqual(base64.b64decode(url.split(",", 1)[1]), self.content)
        self.assertNotIn("base64", str(result))
        self.assertEqual(self.entered, ["master"])
        self.assertEqual(result["image"]["sandbox_id"], "same-sandbox")
        self.assertNotIn("task_id", result["image"])
        self.assertLess(len(str(envelope)), MAX_RESPONSE)

    async def test_paths_and_permissions_fail_before_file_read(self):
        for path in ("/etc/passwd", "../private.png", "https://example.com/image.png"):
            with self.subTest(path=path), self.assertRaises((ValueError, ValidationError)):
                await self.images.select({"path": path})
        self.check.side_effect = ValueError("access revoked")
        with self.assertRaises(ValueError):
            await self.images.select({"path": "screen.png"})
        self.provider.read_workspace_file.assert_not_awaited()

    async def test_invalid_oversized_or_unmatched_pixels_rejected(self):
        for content in (b"not an image", b"x" * (self.images.MAX_BYTES + 1)):
            self.provider.read_workspace_file.return_value = content
            with self.assertRaises(ImageUnavailableError):
                await self.images.select({"path": "screen.png"})
        self.provider.read_workspace_file.return_value = self.content
        result = await self.images.select({"path": "screen.png"})
        with self.assertRaises(ImageUnavailableError):
            await self.images.resolve({**result, "selection_id": "wrong"})
        self.check.side_effect = ValueError("revoked before send")
        with self.assertRaises(ValueError):
            await _native_image_result(NS(payload=result, image=self.images.image))

    async def test_transport_rejects_unsupported_and_oversized_images(self):
        result = await self.images.select({"path": "screen.png"})
        for image in (object(), replace(self.images.image, image_bytes=b"x" * (512 * 1024 + 1))):
            envelope = await _native_image_result(NS(payload=result, image=image))
            self.assertEqual(envelope["error"]["code"], "domain_image_transport_unavailable")
