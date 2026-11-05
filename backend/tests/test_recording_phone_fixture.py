"""Offline fixture contracts only: no SQL, sockets, native processes or providers."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.responses import Response
import httpx

from app.files.storage import JarvisFileStorageClient, SignedRecordingUpload
from app.storage.supabase_config import SupabaseStorageConfig
from evals.second_brain.recording_phone_fixture import (
    CAPTURE_MAX_BYTES, CLEANUP_SECONDS, NATIVE_SECONDS, PARENT_SECONDS, SETUP_SECONDS,
    RecordingStorage, RecordingTrace, bounded_json, build_manifest, evaluate, save,
)


def complete_evidence():
    session, recording, file, request, generation = (str(uuid4()) for _ in range(5))
    checksum, version, size = "a" * 64, "exact:opaque-version", 128
    descriptor = {"filename": "synthetic-recording.mp4", "description": "Synthetic fixture",
        "size_bytes": size, "content_sha256": checksum,
        "client_report": {"generation_id": generation, "timeline_end_ms": 300_000,
            "video_frames": 7200, "source_audio_frames": 14_400_000, "padded_audio_frames": 0}}
    result = dict(outcome="saved", recording_id=recording, file_id=file, source_id=str(uuid4()),
                  version_id=version, download_request_id=request, download_status="pending")
    manifest = {"session_id": session, "start_command": {"recording_id": recording}}
    receipt = {"request_id": request, "artifact_kind": "file", "action": "download",
        "content_ready": True, "transfer_policy": "mp4_capture_v1", "state": "saved_local",
        "local_receipt": {"file_id": file, "version_id": version, "byte_count": size,
                          "content_sha256": checksum, "destination": "app_storage"}}
    state = {"finalizations": [{"status": 200, "response_complete": True,
                                "result": result, "descriptor": descriptor}],
        "inspections": [{"video_codec": "h264", "audio_codec": "aac", "duration_ms": 300_000}],
        "content_reads": [{"path": "/files/" + file + "/content", "status": 200, "response_complete": True,
            "response_bytes": size, "observed_sha256": checksum, "content_sha256": checksum,
            "version_id": version, "response_version_id": version, "session_id": session}],
        "requests": [{"path": "/workspace/requests/" + request + "/ack", "status": 200,
            "authenticated": True, "session_id": session, "ack_state": ack} for ack in ("claimed", "saved_local")],
        "cloud_reads": [{"size_bytes": size, "content_sha256": checksum} for _ in range(4)]}
    return manifest, state, receipt


class FixtureContractTests(unittest.TestCase):
    def test_manifest_is_private_exact_and_parent_budget_covers_phases(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            command = {"type": "recording.command", "max_bytes": CAPTURE_MAX_BYTES, "max_duration_ms": 300_000}
            value = build_manifest(root=root, run_id=str(uuid4()), owner="fixture", namespace="fixture-namespace",
                session=uuid4(), token="private-fixture-only", command=command)
            save(root / "fixture.json", value)
            self.assertEqual((root / "fixture.json").stat().st_mode & 0o777, 0o600)
            self.assertEqual(bounded_json(root / "fixture.json"), value)
            self.assertEqual(value["capture"], dict(duration_ms=300_000, video_frames=7200, synthetic_input=True))
            self.assertEqual(value["budgets"]["max_tus_patches"], 43)
            self.assertEqual(value["budgets"]["max_backend_read_bytes"], 4 * 256 * 1024**2)
            self.assertLess(SETUP_SECONDS + NATIVE_SECONDS + CLEANUP_SECONDS, PARENT_SECONDS)

    def test_control_file_is_bounded(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "control.json"
            path.write_bytes(b" " * 16_385)
            with self.assertRaises(ValueError):
                bounded_json(path)

    def test_complete_evidence_requires_all_independent_boundaries(self):
        manifest, state, receipt = complete_evidence()
        self.assertTrue(all(evaluate(manifest, state, receipt).values()))
        mutations = (
            lambda s, r: s["finalizations"][0].update(status=202),
            lambda s, r: s["finalizations"][0]["descriptor"]["client_report"].update(timeline_end_ms=299_000),
            lambda s, r: s["finalizations"][0]["descriptor"]["client_report"].update(video_frames=7199),
            lambda s, r: s["inspections"][0].update(duration_ms=300_101),
            lambda s, r: s["inspections"][0].update(audio_codec=None),
            lambda s, r: r.update(state="pending"),
            lambda s, r: r["local_receipt"].update(content_sha256="b" * 64),
            lambda s, r: s["content_reads"][0].update(response_complete=False),
            lambda s, r: s["content_reads"][0].update(observed_sha256="b" * 64),
            lambda s, r: s["content_reads"][0].update(version_id="other-version"),
            lambda s, r: s["requests"][0].update(authenticated=False),
            lambda s, r: s["cloud_reads"].pop(),
        )
        for index, mutate in enumerate(mutations):
            with self.subTest(index=index):
                changed, changed_receipt = deepcopy(state), deepcopy(receipt)
                mutate(changed, changed_receipt)
                self.assertFalse(all(evaluate(manifest, changed, changed_receipt).values()))


class FixtureStorageTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.state = {}
        self.storage = RecordingStorage(SupabaseStorageConfig("https://fixture.invalid", "unused-fixture-key",
            "fixture-frames", "fixture-files"), self.state, lambda: None)
        self.addAsyncCleanup(self.storage.close)
        self.owner = "direct-image-eval-second-brain-" + uuid4().hex + "-a"
        self.path = self.storage.build_file_path(owner_id=self.owner, session_id=uuid4(), file_id=uuid4(), filename="x.mp4")
        self.storage.bind(self.owner, self.path)

    async def test_exact_scope_no_backend_upload_and_single_signing(self):
        target = SignedRecordingUpload(self.storage.bucket, self.path, "https://fixture.invalid/signed",
            datetime.now(timezone.utc) + timedelta(hours=1), "never-record-this-signed-token")
        with patch.object(JarvisFileStorageClient, "sign_recording_upload", AsyncMock(return_value=target)) as sign:
            with self.assertRaises(ValueError):
                await self.storage.sign_recording_upload(path=self.path + "-other")
            self.assertIs(await self.storage.sign_recording_upload(path=self.path), target)
            with self.assertRaises(ValueError):
                await self.storage.sign_recording_upload(path=self.path)
            self.assertEqual(sign.await_count, 1)
        self.assertNotIn(target.token, json.dumps(self.state))
        for method in (self.storage.upload_file, self.storage.upload_bytes):
            with self.assertRaises(AssertionError):
                await method(path=self.path)

    async def test_read_budget_is_four_scoped_snapshots_and_cleanup_checks_absence_only(self):
        local = SimpleNamespace(size_bytes=10, content_sha256="a" * 64)
        with patch.object(JarvisFileStorageClient, "read_object_file", AsyncMock(return_value=local)) as read:
            for _ in range(4):
                self.assertIs(await self.storage.read_object_file(bucket=self.storage.bucket,
                    path=self.path, max_bytes=CAPTURE_MAX_BYTES), local)
            with self.assertRaises(ValueError):
                await self.storage.read_object_file(bucket=self.storage.bucket, path=self.path, max_bytes=10)
            self.assertEqual(read.await_count, 4)
        with patch.object(JarvisFileStorageClient, "delete_object", AsyncMock()) as delete, \
                patch.object(JarvisFileStorageClient, "read_object_file", AsyncMock(return_value=None)) as absent:
            await self.storage.cleanup_object()
            delete.assert_awaited_once_with(bucket=self.storage.bucket, path=self.path)
            absent.assert_awaited_once_with(bucket=self.storage.bucket, path=self.path, max_bytes=1)
        self.assertEqual(self.state["cloud_cleanup"], "object_absent")
        self.assertFalse(self.state["signed_capability_revoked"])


class FixtureTraceTests(unittest.IsolatedAsyncioTestCase):
    async def test_trace_hashes_actual_stream_and_never_captures_signed_target(self):
        app, state = FastAPI(), {}
        app.add_middleware(RecordingTrace, state=state, persist=lambda: None)
        content = b"synthetic stream bytes"
        checksum = hashlib.sha256(content).hexdigest()

        @app.post("/recordings/fixture/upload-target")
        async def signed():
            return {"request_headers": {"x-signature": "do-not-log-fixture-token"}}

        @app.get("/files/fixture/content")
        async def get_content():
            return Response(content, headers={"X-Content-SHA256": checksum, "X-Jarvis-Version-Id": "opaque:v1"})

        async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://fixture") as client:
            self.assertEqual((await client.post("/recordings/fixture/upload-target")).status_code, 200)
            session = str(uuid4())
            response = await client.get("/files/fixture/content", params={"session_id": session, "version_id": "opaque:v1"})
            self.assertEqual(response.content, content)
            self.assertEqual((await client.put("/recordings/fixture/content", content=b"forbidden")).status_code, 403)
        self.assertNotIn("do-not-log", json.dumps(state))
        read = state["content_reads"][0]
        self.assertEqual((read["observed_sha256"], read["response_bytes"], read["session_id"]),
                         (checksum, len(content), session))
        self.assertTrue(read["response_complete"])

    async def test_finalization_metadata_is_bounded_and_repeat_does_not_run(self):
        app, state = FastAPI(), {}
        app.add_middleware(RecordingTrace, state=state, persist=lambda: None)
        _, evidence, _ = complete_evidence()
        final = evidence["finalizations"][0]
        calls = []

        @app.post("/recordings/fixture/finalize-upload")
        async def finalize(request: Request):
            await request.json()
            calls.append(True)
            result = dict(final["result"])
            result["scope"] = {"recording_id": result.pop("recording_id")}
            return result

        async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://fixture") as client:
            payload = {"upload_id": str(uuid4()), "descriptor": final["descriptor"]}
            self.assertEqual((await client.post("/recordings/fixture/finalize-upload", json=payload)).status_code, 200)
            self.assertEqual((await client.post("/recordings/fixture/finalize-upload", json=payload)).status_code, 429)
        self.assertEqual(len(calls), 1)
        self.assertEqual(state["finalizations"][0]["descriptor"]["client_report"]["timeline_end_ms"], 300_000)
        self.assertEqual(state["finalizations"][0]["result"]["outcome"], "saved")
        self.assertEqual(state["finalizations"][0]["result"]["recording_id"], final["result"]["recording_id"])
