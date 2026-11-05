import asyncio
import json
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from fastapi import FastAPI
from fastapi.responses import JSONResponse
import httpx

from app.errors import JarvisHTTPError
from app.files.recording_ingress import RecordingIngress
from app.files.recording_router import router
from app.settings import JarvisSettings, get_settings


class RecordingRouteTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.recording, self.session = uuid4(), uuid4()
        self.ingress = object.__new__(RecordingIngress)
        self.ingress.admissions = SimpleNamespace(_authorize=lambda owner: None)
        self.ingress.finalize = AsyncMock(side_effect=self.consume)
        self.ingress.reconcile = AsyncMock(return_value={"outcome": "publication_unresolved", "phase": "outcome_unknown"})
        self.app = FastAPI(); self.app.include_router(router)
        self.app.state.runtime_services = SimpleNamespace(recording_ingress=self.ingress)
        self.app.dependency_overrides[get_settings] = lambda: JarvisSettings(_env_file=None,
            JARVIS_STATIC_BEARER_TOKEN="recording-fixture-token", JARVIS_OWNER_ID="recording-route-owner")

        @self.app.exception_handler(JarvisHTTPError)
        async def error(request, exc):
            return JSONResponse({"code": exc.code}, status_code=exc.status_code)
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app), base_url="http://fixture.invalid")
        self.addAsyncCleanup(self.client.aclose)
        self.manifest = {"filename": "original.mp4", "description": "Client label", "size_bytes": 3,
            "content_sha256": "a" * 64, "client_report": {"generation_id": str(uuid4()),
                "timeline_end_ms": 1000, "video_frames": 24, "source_audio_frames": 48000,
                "padded_audio_frames": 0}}
        self.headers = {"Authorization": "Bearer recording-fixture-token", "Content-Type": "video/mp4",
            "X-Jarvis-Recording-Manifest": json.dumps(self.manifest)}
        self.path = f"/recordings/{self.recording}/content?session_id={self.session}"

    async def consume(self, **kwargs):
        self.assertEqual(kwargs["owner_id"], "recording-route-owner")
        self.assertEqual(kwargs["session_id"], self.session)
        self.assertEqual(kwargs["recording_id"], self.recording)
        self.assertEqual(b"".join([part async for part in kwargs["chunks"]]), b"abc")
        return {"outcome": "saved", "recording_id": str(self.recording)}

    async def test_authenticated_raw_stream_and_private_saved_result(self):
        response = await self.client.put(self.path, content=b"abc", headers=self.headers)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["cache-control"], "private, no-store")
        self.ingress.finalize.assert_awaited_once()

    async def test_missing_wrong_bearer_or_session_never_calls_retention(self):
        for headers, path, expected in (({}, self.path, 401),
                (self.headers | {"Authorization": "Bearer wrong"}, self.path, 403),
                (self.headers, self.path.split("?", 1)[0], 422)):
            response = await self.client.put(path, content=b"abc", headers=headers)
            self.assertEqual(response.status_code, expected)
        self.ingress.finalize.assert_not_awaited()

    async def test_bad_descriptor_mime_or_declared_length_rejected_before_retention(self):
        for changed in ({"X-Jarvis-Recording-Manifest": "x" * 8193},
                        {"X-Jarvis-Recording-Manifest": "{"}, {"Content-Type": "application/json"},
                        {"Content-Length": "4"},
                        {"X-Jarvis-Recording-Manifest": json.dumps(self.manifest | {"owner_id": "foreign"})}):
            with self.subTest(fields=list(changed)):
                response = await self.client.put(self.path, content=b"abc", headers=self.headers | changed)
                self.assertEqual(response.status_code, 422)
        self.ingress.finalize.assert_not_awaited()

    async def test_unconfigured_runtime_does_not_enable_recording_from_auth_alone(self):
        self.app.state.runtime_services = SimpleNamespace()
        response = await self.client.put(self.path, content=b"abc", headers=self.headers)
        self.assertEqual(response.status_code, 503)
        self.ingress.finalize.assert_not_awaited()

    async def test_uncertain_result_and_reconciliation_never_claim_saved(self):
        self.ingress.finalize.side_effect = None
        self.ingress.finalize.return_value = {"outcome": "publication_unresolved", "phase": "outcome_unknown"}
        response = await self.client.put(self.path, content=b"abc", headers=self.headers)
        self.assertEqual(response.status_code, 202)
        result = await self.client.post(f"/recordings/{self.recording}/reconcile?session_id={self.session}", headers=self.headers)
        self.assertEqual(result.status_code, 202)
        self.assertEqual(result.json()["outcome"], "publication_unresolved")

    async def test_timeout_cancels_attempt_and_requires_same_recording_reconciliation(self):
        cancelled = []
        async def slow(**kwargs):
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                cancelled.append(True)
                raise
        self.ingress.finalize.side_effect = slow
        with patch("app.files.recording_router.UPLOAD_SECONDS", 0.001):
            response = await self.client.put(self.path, content=b"abc", headers=self.headers)
        self.assertEqual(response.status_code, 408)
        self.assertEqual(cancelled, [True])


if __name__ == "__main__":
    unittest.main()
