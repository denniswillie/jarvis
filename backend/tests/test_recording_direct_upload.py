"""Offline boundaries only: storage, inspector and persistence are substituted."""
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch
from uuid import UUID, uuid4

from app.files.local_file import _snapshot_chunks
from app.files.publication import FilePublicationService, file_id_for_operation
from app.files.recording_ingress import _spec
from app.files.media_inspection import MediaInspectionError
from app.files.storage import JarvisFileStorageClient, SignedRecordingUpload
from app.knowledge.models import SourceError
import test_recording_ingress as legacy_ingress
import test_recording_router as legacy_routes


class DirectIngressTests(legacy_ingress.RecordingIngressTests):
    """Exercise new transport alongside the inherited legacy ingress checks."""
    def setUp(self):
        super().setUp()
        self.storage = SimpleNamespace(bucket="files", upload_file=AsyncMock(), upload_bytes=AsyncMock(),
            read_object_file=AsyncMock(side_effect=self.remote_read))
        self.storage.build_file_path = lambda **kw: JarvisFileStorageClient.build_file_path(self.storage, **kw)
        self.storage.sign_recording_upload = AsyncMock(side_effect=lambda path: SignedRecordingUpload(
            "files", path, "https://fixture.storage.supabase.co/storage/v1/upload/resumable/sign",
            datetime.now(timezone.utc) + timedelta(hours=1), "fixture-signed-token"))
        self.ingress.storage = self.storage
        self.publications.adopt_recording_upload = AsyncMock(side_effect=self.adopt)
        self.upload_id = file_id_for_operation(self.owner, "recording:" + str(self.recording))
        self.remote_data = self.data
        self.returned = []

    async def remote_read(self, **kwargs):
        self.assertEqual(kwargs, {"bucket": "files", "path": self.ingress._upload_target(self.scope)[2],
                                  "max_bytes": self.upload.size_bytes})
        async def chunks():
            yield self.remote_data
        local = await _snapshot_chunks(chunks(), max_bytes=kwargs["max_bytes"])
        self.returned.append(local)
        return local

    async def adopt(self, **kwargs):
        self.assertEqual(kwargs["recording_id"], self.recording)
        self.assertEqual((kwargs["bucket"], kwargs["path"]), self.ingress._upload_target(self.scope)[1:])
        return await self.publish(operation_key="recording:" + str(self.recording), **kwargs)

    async def direct(self, **changed):
        return await self.ingress.finalize_upload(**({"owner_id": self.owner, "session_id": self.session,
            "recording_id": self.recording, "upload_id": self.upload_id, "upload": self.upload} | changed))

    async def test_signed_target_uses_stable_file_identity_and_capture_recheck(self):
        first = await self.ingress.create_upload_target(owner_id=self.owner, session_id=self.session,
                                                        recording_id=self.recording)
        second = await self.ingress.create_upload_target(owner_id=self.owner, session_id=self.session,
                                                         recording_id=self.recording)
        self.assertEqual(first["upload_id"], str(self.upload_id))
        self.assertEqual(first["object_path"], second["object_path"])
        self.assertEqual(first["chunk_bytes"], 6291456)
        self.assertEqual(first["max_bytes"], self.scope.max_bytes)
        self.assertEqual(first["request_headers"], {"x-signature": "fixture-signed-token"})
        self.assertEqual(len(self.admissions.require_admitted_recording.await_args_list), 4)
        self.assertTrue(all(call.kwargs == {"for_capture": True}
            for call in self.admissions.require_admitted_recording.await_args_list))
        self.storage.upload_file.assert_not_awaited(); self.storage.read_object_file.assert_not_awaited()

    async def test_rejected_admission_prevents_signing_and_post_sign_revocation_hides_target(self):
        self.admissions.require_admitted_recording.side_effect = SourceError("recording_revoked", "Revoked", 403)
        with self.assertRaises(SourceError):
            await self.ingress.create_upload_target(owner_id=self.owner, session_id=self.session, recording_id=self.recording)
        self.storage.sign_recording_upload.assert_not_awaited()
        self.admissions.require_admitted_recording.side_effect = [self.scope, SourceError("recording_revoked", "Revoked", 403)]
        with self.assertRaises(SourceError):
            await self.ingress.create_upload_target(owner_id=self.owner, session_id=self.session, recording_id=self.recording)
        self.storage.sign_recording_upload.assert_awaited_once()

    async def test_direct_finalization_inspects_cloud_and_adopts_without_any_upload(self):
        result = await self.direct()
        self.assertEqual(result["outcome"], "saved")
        self.storage.read_object_file.assert_awaited_once()
        self.publications.adopt_recording_upload.assert_awaited_once()
        self.publications.publish_file.assert_not_awaited()
        self.storage.upload_file.assert_not_awaited(); self.storage.upload_bytes.assert_not_awaited()
        self.assert_private_files_closed()
        self.assertTrue(all(not file.path.exists() for file in self.returned))
        self.assertTrue(all(not call.kwargs for call in self.admissions.require_admitted_recording.await_args_list))

    async def test_duration_failure_is_typed_and_never_publishes(self):
        self.ingress.inspect_file = AsyncMock(side_effect=MediaInspectionError("media_duration_exceeded"))
        with self.assertRaises(SourceError) as raised:
            await self.direct()
        self.assertEqual(raised.exception.code, "media_duration_exceeded")
        self.assertEqual(raised.exception.status_code, 422)
        self.admissions.bind_finalization.assert_not_awaited()
        self.publications.adopt_recording_upload.assert_not_awaited()
        self.assertTrue(all(not file.path.exists() for file in self.returned))

    async def test_other_upload_identity_never_reads_cloud(self):
        with self.assertRaises(SourceError):
            await self.direct(upload_id=uuid4())
        self.storage.read_object_file.assert_not_awaited(); self.publications.adopt_recording_upload.assert_not_awaited()

    async def test_cloud_mismatch_or_missing_object_never_binds_or_adopts(self):
        for data in (b"x" * len(self.data), self.data[:-1]):
            self.remote_data = data
            with self.subTest(length=len(data)), self.assertRaises(SourceError):
                await self.direct()
        self.storage.read_object_file.side_effect = None
        self.storage.read_object_file.return_value = None
        self.assertEqual((await self.direct())["outcome"], "not_saved")
        self.admissions.bind_finalization.assert_not_awaited()
        self.publications.adopt_recording_upload.assert_not_awaited()
        self.assertTrue(all(not file.path.exists() for file in self.returned))

    async def test_withdrawal_during_cloud_inspection_prevents_binding_and_closes_snapshot(self):
        self.admissions.require_admitted_recording.side_effect = [self.scope, SourceError("recording_revoked", "Revoked", 403)]
        with self.assertRaises(SourceError):
            await self.direct()
        self.admissions.bind_finalization.assert_not_awaited()
        self.publications.adopt_recording_upload.assert_not_awaited(); self.assert_private_files_closed()

    async def test_direct_audio_and_current_policy_limits_still_apply(self):
        self.inspection = self.inspection.model_copy(update={"audio_codec": None})
        with self.assertRaises(SourceError):
            await self.direct()
        self.publications.adopt_recording_upload.assert_not_awaited(); self.assert_private_files_closed()
        self.ingress.policy = type(self.policy)("mp4_capture_v1", len(self.data) - 1)
        self.storage.read_object_file.reset_mock()
        with self.assertRaises(SourceError):
            await self.direct()
        self.storage.read_object_file.assert_not_awaited()

    async def test_direct_replay_and_conflict_preserve_existing_publication_identity(self):
        self.freeze(); self.prior = {"id": uuid4(), "phase": "outcome_unknown"}
        self.assertEqual((await self.direct())["outcome"], "publication_unresolved")
        self.publications.reconcile.assert_awaited_once_with(owner_id=self.owner, operation_id=self.prior["id"])
        with self.assertRaises(SourceError):
            await self.direct(upload=self.upload.model_copy(update={"description": "Different"}))
        self.storage.read_object_file.assert_not_awaited(); self.publications.adopt_recording_upload.assert_not_awaited()

    async def test_external_settle_checks_exact_target_preserves_withdrawal_and_never_uploads(self):
        self.freeze()
        local = await self.remote_read(bucket="files", path=self.ingress._upload_target(self.scope)[2],
                                       max_bytes=self.upload.size_bytes)
        self.addCleanup(local.close)
        service = object.__new__(FilePublicationService)
        service.owner_id, service.access_check = self.owner, None
        service.storage, service.sources, service.transfer_policy = self.storage, self.sources, self.policy
        service.adopt_file = AsyncMock()
        service.provenance = SimpleNamespace(tasks=self.admissions.tasks)
        prepared = SimpleNamespace(operation_id=uuid4())
        service.prepare_manifest = AsyncMock(return_value=prepared)
        row = {"id": prepared.operation_id, "file_id": self.upload_id, "storage_bucket": "files",
               "storage_path": self.ingress._upload_target(self.scope)[2], "phase": "prepared", "upload_token": None}
        service._row = AsyncMock(return_value=row); service._active = AsyncMock()
        service.reconcile = AsyncMock(return_value=SimpleNamespace(phase="available"))
        self.connection.execute = AsyncMock()
        args = dict(owner_id=self.owner, recording_id=self.recording, spec=_spec(self.scope, self.frozen),
            bucket="files", path=row["storage_path"], local_file=local, inspection=self.inspection,
            source_payload_ids=(self.scope.admission_payload_id,))
        with self.assertRaises(SourceError):
            await service.adopt_recording_upload(**(args | {"path": "owners/foreign/content"}))
        service.prepare_manifest.assert_not_awaited()
        result = await service.adopt_recording_upload(**args)
        self.assertEqual(result.phase, "available")
        call = self.connection.execute.await_args
        self.assertIn("upload_settled=true,phase='uploaded'", call.args[0])
        self.assertEqual(call.args[1:], (self.owner, prepared.operation_id, self.upload_id))
        service.reconcile.assert_awaited_once_with(owner_id=self.owner, operation_id=prepared.operation_id)
        self.storage.upload_file.assert_not_awaited(); self.storage.upload_bytes.assert_not_awaited()
        service._active.side_effect = SourceError("file_publication_revoked", "Revoked")
        self.connection.execute.reset_mock(); service.reconcile.reset_mock()
        with self.assertRaises(SourceError):
            await service.adopt_recording_upload(**args)
        self.connection.execute.assert_not_awaited(); service.reconcile.assert_not_awaited()


class DirectRouteTests(legacy_routes.RecordingRouteTests):
    """Real ASGI/auth/JSON handling; backend persistence and presentation mocked."""
    async def asyncSetUp(self):
        await super().asyncSetUp()
        self.upload_id, self.file_id = uuid4(), uuid4()
        self.saved = {"outcome": "saved", "file_id": str(self.file_id), "version_id": "source:exact-version"}
        self.ingress.create_upload_target = AsyncMock(return_value={"upload_id": str(self.upload_id)})
        self.ingress.finalize_upload = AsyncMock(return_value=self.saved)
        self.presentations = SimpleNamespace(present=AsyncMock(side_effect=self.present))
        self.app.state.runtime_services.workspace_presentations = self.presentations
        self.body = {"upload_id": str(self.upload_id), "descriptor": self.manifest}
        self.direct_path = f"/recordings/{self.recording}/finalize-upload?session_id={self.session}"
        self.auth = {"Authorization": self.headers["Authorization"]}

    async def present(self, ctx, **kwargs):
        self.assertEqual(ctx.owner_id, "recording-route-owner")
        self.assertEqual(ctx.session_id, self.session)
        self.assertEqual(kwargs, {"action": "download", "artifact_kind": "file", "artifact_id": self.file_id,
                                  "version_id": self.saved["version_id"]})
        return {"outcome": "accepted", "presentation": {"request_id": str(ctx.operation_id), "state": "pending"}}

    async def test_direct_routes_require_bearer_original_session_and_private_response(self):
        target = f"/recordings/{self.recording}/upload-target?session_id={self.session}"
        for path in (target, self.direct_path):
            self.assertEqual((await self.client.post(path, json=self.body)).status_code, 401)
            self.assertEqual((await self.client.post(path, json=self.body,
                headers={"Authorization": "Bearer wrong"})).status_code, 403)
            self.assertEqual((await self.client.post(path.split("?")[0], json=self.body, headers=self.auth)).status_code, 422)
        self.ingress.create_upload_target.assert_not_awaited(); self.ingress.finalize_upload.assert_not_awaited()
        response = await self.client.post(target, headers=self.auth)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["cache-control"], "private, no-store")
        self.ingress.create_upload_target.assert_awaited_once_with(owner_id="recording-route-owner",
            session_id=self.session, recording_id=self.recording)

    async def test_final_save_and_reconcile_never_request_a_phone_download(self):
        for _ in range(2):
            response = await self.client.post(self.direct_path, json=self.body, headers=self.auth)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json(), self.saved)
        self.ingress.reconcile.return_value = self.saved
        response = await self.client.post(f"/recordings/{self.recording}/reconcile?session_id={self.session}", headers=self.auth)
        self.assertEqual(response.json(), self.saved)
        self.presentations.present.assert_not_awaited()

    async def test_oversize_unknown_path_fields_or_invalid_json_never_reaches_finalize(self):
        for body in (b" " * 8449, b"{", b'{"upload_id":"bad"}',
                     __import__("json").dumps(self.body | {"object_path": "foreign"}).encode()):
            response = await self.client.post(self.direct_path, content=body,
                headers=self.auth | {"Content-Type": "application/json"})
            self.assertEqual(response.status_code, 422)
        self.ingress.finalize_upload.assert_not_awaited(); self.presentations.present.assert_not_awaited()

    async def test_unavailable_presentations_do_not_affect_cloud_save(self):
        self.presentations.present.side_effect = RuntimeError("must-not-be-called")
        response = await self.client.post(self.direct_path, json=self.body, headers=self.auth)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), self.saved)
        self.presentations.present.assert_not_awaited()
        self.app.state.runtime_services.workspace_presentations = None
        response = await self.client.post(self.direct_path, json=self.body, headers=self.auth)
        self.assertEqual(response.json(), self.saved)

    async def test_pending_cloud_adoption_does_not_request_download(self):
        self.ingress.finalize_upload.return_value = {"outcome": "not_saved", "phase": "upload_not_found"}
        response = await self.client.post(self.direct_path, json=self.body, headers=self.auth)
        self.assertEqual(response.status_code, 202)
        self.presentations.present.assert_not_awaited()


    async def test_direct_finalize_timeout_does_not_request_download(self):
        import asyncio
        async def slow(**kwargs):
            await asyncio.sleep(10)
        self.ingress.finalize_upload.side_effect = slow
        with patch("app.files.recording_router.FINALIZE_SECONDS", 0.001):
            response = await self.client.post(self.direct_path, json=self.body, headers=self.auth)
        self.assertEqual(response.status_code, 408)
        self.presentations.present.assert_not_awaited()

