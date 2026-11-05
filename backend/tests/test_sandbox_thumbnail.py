"""Bounded screenshot decoding only; no live provider or VM."""
import base64
from io import BytesIO
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from PIL import Image

from app.sandbox.models import SandboxError
from app.sandbox.view_provider import CAPTURE_THUMBNAIL, E2BDesktopViewer, decode_thumbnail


def encoded_image(size=(320, 180), format="JPEG"):
    output = BytesIO()
    Image.new("RGB", size, "teal").save(output, format=format)
    return base64.b64encode(output.getvalue()).decode("ascii")


class ThumbnailTests(unittest.IsolatedAsyncioTestCase):
    def test_valid_small_jpeg(self):
        self.assertTrue(decode_thumbnail(encoded_image()).startswith(b"\xff\xd8"))

    def test_invalid_image_and_bounds(self):
        for value in ["not-base64", encoded_image((321, 180)), encoded_image(format="PNG"), "x" * 45001,
                      base64.b64encode(b"x" * 32769).decode("ascii")]:
            with self.assertRaises(SandboxError):
                decode_thumbnail(value)

    async def test_capture_uses_nonresuming_attach_and_existing_privileged_launcher(self):
        provider = SimpleNamespace(_run_privileged=AsyncMock(return_value=SimpleNamespace(stdout=encoded_image())))
        viewer = E2BDesktopViewer(provider, object())
        handle, box = object(), object()
        viewer._attach = AsyncMock(return_value=(box, object()))
        data = await viewer.thumbnail(handle)
        self.assertTrue(data.startswith(b"\xff\xd8"))
        viewer._attach.assert_awaited_once_with(handle)
        provider._run_privileged.assert_awaited_once_with(box, CAPTURE_THUMBNAIL, timeout=8)
