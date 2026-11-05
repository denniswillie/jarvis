"""Offline HTTP signing boundary checks; no hosted Storage requests."""
import base64
from datetime import datetime, timezone
import json
import unittest

import httpx

from app.errors import JarvisHTTPError
from app.storage.supabase_config import SupabaseStorageConfig

from app.files import storage as candidate


class SignedTargetTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.storage = candidate.JarvisFileStorageClient(SupabaseStorageConfig(
            "https://fixture.supabase.co", "fixture-service-secret", "frames", "files"))
        await self.storage.close()
        self.requests = []
        self.status = 200
        self.claims = {"url": "files/owners/test/content", "upsert": False,
                       "exp": int(datetime.now(timezone.utc).timestamp()) + 3600}
        self.reply_mutator = lambda reply: reply

        def response(request):
            self.requests.append(request)
            payload = base64.urlsafe_b64encode(json.dumps(self.claims).encode()).decode().rstrip("=")
            token = "fixture." + payload + ".signature"
            body = self.reply_mutator({"url": "/object/upload/sign/files/owners/test/content?token=" + token,
                                      "token": token})
            return httpx.Response(self.status, json=body,
                headers={"Location": "https://untrusted.invalid/"})
        self.storage._client = httpx.AsyncClient(transport=httpx.MockTransport(response))

    async def asyncTearDown(self):
        await self.storage.close()

    async def test_exact_non_upsert_target_and_credential_kept_out_of_repr(self):
        target = await self.storage.sign_recording_upload(path="owners/test/content")
        request = self.requests[0]
        self.assertEqual(request.method, "POST")
        self.assertEqual(str(request.url),
                         "https://fixture.supabase.co/storage/v1/object/upload/sign/files/owners/test/content")
        self.assertEqual(request.headers["x-upsert"], "false")
        self.assertEqual(request.headers["authorization"], "Bearer fixture-service-secret")
        self.assertEqual(request.headers["content-type"], "application/json")
        self.assertEqual(request.content, b"{}")
        self.assertEqual(target.resumable_url,
                         "https://fixture.storage.supabase.co/storage/v1/upload/resumable/sign")
        self.assertEqual((target.bucket, target.path), ("files", "owners/test/content"))
        self.assertNotIn(target.token, repr(target))
        self.assertNotIn("fixture-service-secret", repr(target))

    async def test_redirect_or_provider_failure_is_unknown_without_secret_echo(self):
        for status in (302, 400, 500):
            self.status = status
            with self.subTest(status=status), self.assertRaises(JarvisHTTPError) as caught:
                await self.storage.sign_recording_upload(path="owners/test/content")
            self.assertNotIn("fixture-service-secret", str(caught.exception))
        self.assertEqual(len(self.requests), 3)

    async def test_wrong_object_upsert_or_expiry_is_rejected(self):
        original = dict(self.claims)
        for changes in ({"url": "files/other/content"}, {"upsert": True},
                        {"exp": 1}, {"exp": original["exp"] + 14400}, {"exp": True}):
            self.claims = original | changes
            with self.subTest(changes=changes), self.assertRaises(JarvisHTTPError):
                await self.storage.sign_recording_upload(path="owners/test/content")

    async def test_response_host_or_path_cannot_redirect_capability(self):
        for url in ("https://untrusted.invalid/object?token=abc", "/object/upload/sign/files/other?token=abc"):
            self.reply_mutator = lambda value, replacement=url: value | {"url": replacement}
            with self.subTest(url=url), self.assertRaises(JarvisHTTPError):
                await self.storage.sign_recording_upload(path="owners/test/content")

    async def test_invalid_relative_path_never_calls_storage(self):
        for path in ("/owners/test", "owners/../test", "owners//test", "owners/\\test", ""):
            with self.subTest(path=path), self.assertRaises(ValueError):
                await self.storage.sign_recording_upload(path=path)
        self.assertEqual(self.requests, [])


if __name__ == "__main__":
    unittest.main()
