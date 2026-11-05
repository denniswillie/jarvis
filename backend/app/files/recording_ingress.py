"""Retain admitted recordings through the existing exact Files publication owner.

Client timing/capture reports are retained as user input, never sensor/hearing
proof. This adapter owns no camera, clock, command journal or transfer lifecycle.
"""
from __future__ import annotations

import json
from typing import Literal
from uuid import UUID

from pydantic import AwareDatetime, Field, field_validator, model_validator, model_serializer

from app.evidence.models import FrozenRecord
from app.files.local_file import VerifiedLocalFile, _snapshot_chunks
from app.files.publication import FilePublicationSpec, file_id_for_operation
from app.files.transfer import (
    CAPTURE_MAX_BYTES, CAPTURE_MAX_DURATION_MS, CAPTURE_ENCODING_OVERHEAD_MS, FileManifest, FileTransferPolicy,
    MP4Inspection, MP4_CAPTURE_POLICY,
)
from app.knowledge.files import SourceFileRepository
from app.knowledge.models import SourceError, SourcePublication
from app.knowledge.product_files import FILE_NAMESPACE, RECORDING_ORIGINAL_FORMAT
from app.recordings.models import FINALIZATION_FORMAT
from app.files.media_inspection import MediaInspectionError


class RecordingGap(FrozenRecord):
    track: Literal["video", "audio"]
    start_us: int = Field(ge=0, le=CAPTURE_MAX_DURATION_MS * 1000, strict=True)
    end_us: int = Field(ge=0, le=CAPTURE_MAX_DURATION_MS * 1000, strict=True)
    reason: str = Field(min_length=1, max_length=64)
    inferred_video_duration: bool = Field(default=False, strict=True)

    @model_validator(mode="after")
    def ordered(self):
        if self.end_us < self.start_us or "\x00" in self.reason:
            raise ValueError("A reported gap must have ordered bounded times")
        return self


class RecordingClientReport(FrozenRecord):
    format: Literal["jarvis.recording_client_report.v1"] = "jarvis.recording_client_report.v1"
    generation_id: UUID
    timeline_end_ms: int = Field(ge=1, le=CAPTURE_MAX_DURATION_MS, strict=True)
    video_frames: int = Field(ge=1, le=100_000, strict=True)
    source_audio_frames: int = Field(ge=0, le=CAPTURE_MAX_DURATION_MS * 48, strict=True)
    padded_audio_frames: int = Field(ge=0, le=CAPTURE_MAX_DURATION_MS * 48, strict=True)
    missing_intervals: tuple[RecordingGap, ...] = Field(default=(), max_length=128)
    omitted_interval_details: int = Field(default=0, ge=0, le=1_000_000, strict=True)
    capture_started_at: AwareDatetime | None = None
    capture_ended_at: AwareDatetime | None = None
    recording_mode: Literal["ordinary", "voiceover"] | None = None
    include_media_audio: bool | None = Field(default=None, strict=True)
    media_coverage: dict[str, int] | None = None

    @field_validator("media_coverage", mode="before")
    @classmethod
    def media_counts(cls, value):
        if value is None:
            return None
        keys = {"consumed_output_frames", "resolved_source_frames", "missing_source_frames",
                "unknown_output_frames", "receipt_dropped_output_frames", "zero_gain_source_frames",
                "admitted_source_frames", "rejected_source_frames", "recorded_source_frames", "output_capture_dropped_frames"}
        if (type(value) is not dict or set(value) != keys
                or any(type(count) is not int or not 0 <= count <= 9_007_199_254_740_991 for count in value.values())):
            raise ValueError("Media coverage needs exact bounded frame counters")
        return value

    @model_serializer(mode="wrap")
    def preserve_legacy_report(self, handler):
        result = handler(self)
        # New optional metadata must not change hashes of old pending/frozen
        # v1 reports merely because the server has been upgraded.
        for key in ("recording_mode", "include_media_audio", "media_coverage"):
            if result.get(key) is None:
                result.pop(key, None)
        return result

    @model_validator(mode="after")
    def bounded_report(self):
        if ((self.recording_mode is None) != (self.include_media_audio is None)
                or (self.recording_mode == "voiceover" and self.include_media_audio)
                or (self.media_coverage is not None and self.include_media_audio is not True)):
            raise ValueError("Recording mode and media selection must agree")
        if any(gap.end_us > self.timeline_end_ms * 1000 for gap in self.missing_intervals):
            raise ValueError("Reported gaps must fit the declared recording timeline")
        if self.source_audio_frames + self.padded_audio_frames > self.timeline_end_ms * 48:
            raise ValueError("Reported audio frames exceed the declared 48kHz timeline")
        return self


class RecordingUpload(FrozenRecord):
    filename: str = Field(min_length=1, max_length=240)
    description: str = Field(min_length=1, max_length=2000)
    size_bytes: int = Field(ge=1, le=CAPTURE_MAX_BYTES, strict=True)
    content_sha256: str = Field(pattern="^[0-9a-f]{64}$")
    client_report: RecordingClientReport

    @field_validator("filename", "description")
    @classmethod
    def stored_text(cls, value):
        if not value.strip() or "\x00" in value:
            raise ValueError("Recording labels must be nonblank stored text")
        value.encode("utf-8")
        return value

    @field_validator("filename")
    @classmethod
    def movie_name(cls, value):
        if "/" in value or "\\" in value or not value.lower().endswith(".mp4"):
            raise ValueError("A plain MP4 display filename is required")
        return value


def _key(scope):
    return "recording:" + str(scope.recording_id)


def _metadata(scope, frozen):
    return {"recording_original": {"recording_id": str(scope.recording_id),
            "manifest_digest": frozen.manifest_digest, "finalization_manifest": frozen.manifest},
            "media_inspection": frozen.manifest["media_inspection"]}


def _spec(scope, frozen):
    manifest = frozen.manifest
    return FilePublicationSpec(session_id=scope.session_id, filename=manifest["filename"],
        description=manifest["description"], content_type="video/mp4", source="upload",
        metadata=_metadata(scope, frozen))


class _RecordingFiles(SourceFileRepository):
    """A capture producer adapter, not general permission for source='upload'."""
    def __init__(self, ingress, scope):
        self.ingress, self.recording_scope = ingress, scope
        super().__init__(ingress.sources, ingress.storage, owner_id=scope.owner_id, area_id=scope.area_id,
            access_check=ingress.admissions._authorize,
            media_policy=FileTransferPolicy(MP4_CAPTURE_POLICY, scope.max_bytes),
            media_inspector=ingress.inspect_file)

    async def _adopt_media(self, connection, file, spec, local_file, inspection, metadata_payload_id):
        owner, scope = self.ingress.admissions, self.recording_scope
        await owner.tasks.lock_owner(connection, scope.owner_id)
        current = await owner.require_admitted_recording(connection, scope.owner_id, scope.session_id, scope.recording_id)
        frozen = await owner.get_finalization(connection, scope)
        if (current != scope or frozen is None or not isinstance(local_file, VerifiedLocalFile)
                or not isinstance(inspection, MP4Inspection) or spec != _spec(scope, frozen)):
            raise SourceError("recording_adoption_mismatch", "Stored media differs from its admitted recording.", 403)
        manifest = frozen.manifest
        inspection.validate_file(FileManifest.from_file(local_file), self.media_policy)
        if (inspection.model_dump(mode="json") != manifest["media_inspection"]
                or local_file.content_sha256 != manifest["content_sha256"]
                or local_file.size_bytes != manifest["size_bytes"]):
            raise SourceError("recording_adoption_mismatch", "Stored bytes differ from the frozen recording.", 403)
        original = {"format": RECORDING_ORIGINAL_FORMAT, "record": file.model_dump(mode="json"),
            "content_sha256": local_file.content_sha256, "recording_id": str(scope.recording_id),
            "finalization_manifest": manifest, "media_inspection": inspection.model_dump(mode="json"),
            "media_origin": "client_reported",
            "meaning": "Exact retained client media. Capture origin, timing and heard speech are not independently proved."}
        if "timing_correction" in manifest:
            original["meaning"] = ("Client-reported capture with server-side picture timing correction; "
                "audio is unchanged. The manifest retains the original upload digest and applied offset.")
        payload = await connection.fetchval("""insert into jarvis_context_payloads(owner_id,availability,content)
            values($1,'available',$2::jsonb) returning id""", scope.owner_id, json.dumps(original, allow_nan=False))
        await connection.execute("""insert into jarvis_context_payload_dependencies
            (owner_id,dependent_payload_id,source_payload_id) values($1,$2,$3)""",
            scope.owner_id, payload, metadata_payload_id)
        source = await self.sources.publish_in_transaction(connection, scope.owner_id, SourcePublication(
            area_id=scope.area_id, namespace=FILE_NAMESPACE, source_key=str(file.id), payload_id=payload,
            source_type="recording_file", trust="user_input", source_time=file.created_at, session_id=scope.session_id,
            expected_revision=0, time_basis="Verified adoption time; client capture times are not independently verified."))
        publication_id = await connection.fetchval("""select id from jarvis_file_publications
            where owner_id=$1 and operation_key=$2 and file_id=$3""", scope.owner_id, _key(scope), file.id)
        await owner.complete_in_transaction(connection, scope, frozen.manifest_digest,
            publication_id=publication_id, file_id=file.id, source_id=source.source_id, version_id=source.document_id)


class RecordingIngress:
    def __init__(self, sources, storage, admissions, *, policy, inspect_file, timing_options=None):
        from app.recordings.saved_timing import configured_timing_options
        self.timing_options = timing_options if timing_options is not None else configured_timing_options()
        if (admissions.sources is not sources or not isinstance(policy, FileTransferPolicy)
                or policy.policy_id != MP4_CAPTURE_POLICY or not callable(inspect_file)):
            raise ValueError("Recording ingress requires one trusted admission, capture policy and inspector")
        self.sources, self.storage, self.admissions = sources, storage, admissions
        self.policy, self.inspect_file = policy, inspect_file

    async def _existing(self, connection, scope):
        frozen = await self.admissions.get_finalization(connection, scope)
        publication = await connection.fetchrow("""select id,phase from jarvis_file_publications
            where owner_id=$1 and area_id=$2 and operation_key=$3""", scope.owner_id, scope.area_id, _key(scope))
        if publication is not None and frozen is None:
            raise SourceError("recording_finalization_missing", "A recording publication lacks its frozen binding.")
        return frozen, publication

    @staticmethod
    def _matches_upload(frozen, upload):
        expected = upload.model_dump(mode="json")
        original = frozen.manifest.get("uploaded_original", frozen.manifest)
        if any(original.get(key) != value for key, value in expected.items()):
            raise SourceError("recording_finalization_conflict", "This recording already binds different bytes or metadata.")

    async def _result(self, scope, publication):
        if publication.phase != "available":
            return {"recording_id": str(scope.recording_id), "outcome": "publication_unresolved",
                "publication_id": str(publication.operation_id), "phase": publication.phase}
        async with self.sources.transaction(scope.owner_id) as connection:
            await self.admissions.tasks.lock_owner(connection, scope.owner_id)
            receipt = await self.admissions.get_completion(connection, scope)
            if receipt is None:
                raise SourceError("recording_receipt_missing", "No committed recording save receipt is available.")
            frozen = await self.admissions.get_finalization(connection, scope)
        from app.recordings.saved_timing import finish_timing_save
        return await finish_timing_save(self, scope, frozen, receipt.model_dump(mode="json"))

    async def _inspect(self, local, policy, upload):
        try:
            inspection = await self.inspect_file(local, policy)
        except MediaInspectionError as error:
            message = ("This recording is too long to save. Try a shorter recording."
                if error.code == "media_duration_exceeded"
                else "This recording could not be verified as a playable video.")
            raise SourceError(error.code, message, 422) from None
        if not isinstance(inspection, MP4Inspection):
            raise SourceError("recording_inspection_unavailable", "The recording has no trusted media inspection.")
        inspection.validate_file(FileManifest.from_file(local), policy)
        if inspection.duration_ms > upload.client_report.timeline_end_ms + CAPTURE_ENCODING_OVERHEAD_MS:
            raise SourceError("media_duration_exceeded",
                "The video is longer than the captured recording. Try a shorter recording.", 422)
        return inspection

    async def finalize(self, *, owner_id, session_id, recording_id, upload, chunks):
        """Consume bounded raw chunks only when an exact upload is still needed."""
        if type(upload) is not RecordingUpload:
            raise ValueError("A validated recording upload descriptor is required")
        self.admissions._authorize(owner_id)
        async with self.sources.transaction(owner_id) as connection:
            await self.admissions.tasks.lock_owner(connection, owner_id)
            scope = await self.admissions.require_admitted_recording(connection, owner_id, session_id, recording_id)
            frozen, prior = await self._existing(connection, scope)
            if frozen is not None:
                self._matches_upload(frozen, upload)
        files = _RecordingFiles(self, scope)
        if prior is not None and prior["phase"] != "prepared":
            # Once an upload might have happened, inspect that same object. Do
            # not consume/re-send client bytes or clear the existing upload token.
            return await self._result(scope, await files.publications.reconcile(owner_id=owner_id, operation_id=prior["id"]))
        if (upload.size_bytes > min(scope.max_bytes, self.policy.max_bytes)
                or upload.client_report.timeline_end_ms > scope.max_duration_ms):
            raise SourceError("recording_limit_exceeded", "The recording exceeds its admitted capture bounds.", 413)
        local = await _snapshot_chunks(chunks, max_bytes=min(scope.max_bytes, self.policy.max_bytes),
            expected_size=upload.size_bytes, expected_sha256=upload.content_sha256)
        prepared = None
        try:
            inspection = await self._inspect(local, files.media_policy, upload)
            if scope.include_jarvis_audio and inspection.audio_codec != "aac":
                raise SourceError("recording_audio_missing", "The requested audio track is missing; no speech was synthesized.")
            from app.recordings.saved_timing import prepare_recording
            prepared = await prepare_recording(self, scope, upload, local, inspection, frozen)
            manifest = prepared.manifest
            async with self.sources.transaction(owner_id) as connection:
                await self.admissions.tasks.lock_owner(connection, owner_id)
                current = await self.admissions.require_admitted_recording(connection, owner_id, session_id, recording_id)
                if current != scope:
                    raise SourceError("recording_scope_changed", "The original recording admission changed.", 403)
                frozen = await self.admissions.bind_finalization(connection, scope, manifest)
                # Validate the full bounded storage specification before commit.
                spec = _spec(scope, frozen)
            result = await files.publications.publish_file(owner_id=owner_id, operation_key=_key(scope), spec=spec,
                local_file=prepared.local, source_payload_ids=(scope.admission_payload_id,))
            return await self._result(scope, result)
        finally:
            if prepared is not None:
                prepared.close()
            local.close()

    def _upload_target(self, scope):
        file_id = file_id_for_operation(scope.owner_id, _key(scope))
        path = self.storage.build_file_path(owner_id=scope.owner_id, session_id=scope.session_id,
            file_id=file_id, filename="recording.mp4")
        return file_id, self.storage.bucket, path

    async def create_upload_target(self, *, owner_id, session_id, recording_id):
        """Authorize a direct, non-overwriting upload while capture is admitted."""
        self.admissions._authorize(owner_id)
        async with self.sources.transaction(owner_id) as connection:
            await self.admissions.tasks.lock_owner(connection, owner_id)
            scope = await self.admissions.require_admitted_recording(
                connection, owner_id, session_id, recording_id, for_capture=True)
        upload_id, bucket, path = self._upload_target(scope)
        target = await self.storage.sign_recording_upload(path=path)
        if (target.bucket, target.path) != (bucket, path):
            raise SourceError("recording_upload_target_mismatch", "Storage returned another upload target.", 502)
        # Signing has external latency. Recheck authority before releasing a
        # credential; no database transaction is held across storage I/O.
        async with self.sources.transaction(owner_id) as connection:
            await self.admissions.tasks.lock_owner(connection, owner_id)
            current = await self.admissions.require_admitted_recording(
                connection, owner_id, session_id, recording_id, for_capture=True)
            if current != scope:
                raise SourceError("recording_scope_changed", "The original recording admission changed.", 403)
        return {"recording_id": str(recording_id), "session_id": str(session_id),
            "upload_id": str(upload_id), "bucket_name": bucket, "object_path": path,
            "resumable_url": target.resumable_url, "request_headers": {"x-signature": target.token},
            "credential_expires_at": target.expires_at.isoformat(),
            "max_bytes": min(scope.max_bytes, self.policy.max_bytes), "chunk_bytes": 6 * 1024**2}

    async def finalize_upload(self, *, owner_id, session_id, recording_id, upload_id, upload):
        """Verify the final cloud object, then use the normal recording owner.

        Only the admitted server target is read. Client bytes/paths/inspection
        claims cannot replace the bounded cloud snapshot or trusted inspector.
        """
        if type(upload) is not RecordingUpload or not isinstance(upload_id, UUID):
            raise ValueError("An exact upload identity and recording descriptor are required")
        self.admissions._authorize(owner_id)
        async with self.sources.transaction(owner_id) as connection:
            await self.admissions.tasks.lock_owner(connection, owner_id)
            scope = await self.admissions.require_admitted_recording(connection, owner_id, session_id, recording_id)
            expected_id, bucket, path = self._upload_target(scope)
            if upload_id != expected_id:
                raise SourceError("recording_upload_target_mismatch", "This upload belongs to another recording.", 403)
            frozen, prior = await self._existing(connection, scope)
            if frozen is not None:
                self._matches_upload(frozen, upload)
        files = _RecordingFiles(self, scope)
        if prior is not None and prior["phase"] != "prepared":
            return await self._result(scope, await files.publications.reconcile(owner_id=owner_id, operation_id=prior["id"]))
        if (upload.size_bytes > min(scope.max_bytes, self.policy.max_bytes)
                or upload.client_report.timeline_end_ms > scope.max_duration_ms):
            raise SourceError("recording_limit_exceeded", "The recording exceeds its admitted capture bounds.", 413)
        local = await self.storage.read_object_file(bucket=bucket, path=path, max_bytes=upload.size_bytes)
        if local is None:
            return {"recording_id": str(recording_id), "outcome": "not_saved", "phase": "upload_not_found"}
        prepared = None
        try:
            if (not isinstance(local, VerifiedLocalFile) or local.size_bytes != upload.size_bytes
                    or local.content_sha256 != upload.content_sha256):
                raise SourceError("recording_content_mismatch", "The cloud object differs from the final recording manifest.", 422)
            inspection = await self._inspect(local, files.media_policy, upload)
            if scope.include_jarvis_audio and inspection.audio_codec != "aac":
                raise SourceError("recording_audio_missing", "The requested audio track is missing; no speech was synthesized.")
            async for _ in local.iter_chunks():
                pass
            from app.recordings.saved_timing import prepare_recording
            prepared = await prepare_recording(self, scope, upload, local, inspection, frozen, direct=True)
            manifest = prepared.manifest
            async with self.sources.transaction(owner_id) as connection:
                await self.admissions.tasks.lock_owner(connection, owner_id)
                current = await self.admissions.require_admitted_recording(connection, owner_id, session_id, recording_id)
                if current != scope:
                    raise SourceError("recording_scope_changed", "The original recording admission changed.", 403)
                frozen = await self.admissions.bind_finalization(connection, scope, manifest)
                spec = _spec(scope, frozen)
            if prepared.owns_local:
                result = await files.publications.publish_file(owner_id=owner_id,
                    operation_key=_key(scope), spec=spec, local_file=prepared.local,
                    source_payload_ids=(scope.admission_payload_id,))
            else:
                result = await files.publications.adopt_recording_upload(owner_id=owner_id,
                    recording_id=recording_id, spec=spec, bucket=bucket, path=path,
                    local_file=local, inspection=inspection, source_payload_ids=(scope.admission_payload_id,))
            return await self._result(scope, result)
        finally:
            if prepared is not None:
                prepared.close()
            local.close()

    async def reconcile(self, *, owner_id, session_id, recording_id):
        self.admissions._authorize(owner_id)
        async with self.sources.transaction(owner_id) as connection:
            await self.admissions.tasks.lock_owner(connection, owner_id)
            scope = await self.admissions.require_admitted_recording(connection, owner_id, session_id, recording_id)
            frozen, prior = await self._existing(connection, scope)
        if prior is None:
            return {"recording_id": str(scope.recording_id), "outcome": "not_saved",
                "finalization_bound": frozen is not None}
        files = _RecordingFiles(self, scope)
        return await self._result(scope, await files.publications.reconcile(owner_id=owner_id, operation_id=prior["id"]))
