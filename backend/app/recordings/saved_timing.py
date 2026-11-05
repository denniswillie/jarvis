"""Optional saved-file processing; never part of microphone/capture startup."""
import asyncio
from dataclasses import dataclass, replace
import logging
import os

from app.files.transfer import FileManifest, MP4Inspection
from app.knowledge.models import SourceError
from app.recordings.models import FINALIZATION_FORMAT, freeze_manifest
from app.recordings.timing import RecordingTimingOptions, correct_recording_timing


log = logging.getLogger(__name__)


def configured_timing_options():
    # Opt-in calibration for the deployed capture route, not a universal offset.
    try:
        return RecordingTimingOptions(
            video_delay_ms=int(os.getenv("JARVIS_RECORDING_VIDEO_DELAY_MS", "0")),
            timeout_seconds=20,
            ffmpeg=os.getenv("JARVIS_RECORDING_SYNC_FFMPEG", "ffmpeg"),
            ffprobe=os.getenv("JARVIS_RECORDING_SYNC_FFPROBE", "ffprobe"))
    except (TypeError, ValueError):
        log.warning("Invalid recording timing configuration; saving without correction")
        return RecordingTimingOptions()


@dataclass
class PreparedRecording:
    local: object
    inspection: MP4Inspection
    manifest: dict
    owns_local: bool = False

    def close(self):
        if self.owns_local:
            self.local.close()


async def prepare_recording(ingress, scope, upload, local, inspection, frozen, *, direct=False):
    raw = {"format": FINALIZATION_FORMAT, **upload.model_dump(mode="json"),
           "duration_ms": inspection.duration_ms, "media_inspection": inspection.model_dump(mode="json")}
    previous = frozen.manifest if frozen is not None else None
    if previous is not None and "timing_correction" not in previous:
        return PreparedRecording(local, inspection, previous)
    options = ingress.timing_options
    if previous is not None:
        options = replace(options, video_delay_ms=previous["timing_correction"]["video_delay_ms"])
    if options.video_delay_ms == 0:
        return PreparedRecording(local, inspection, raw)
    options = replace(options, max_bytes=min(scope.max_bytes, ingress.policy.max_bytes),
                      max_duration_ms=scope.max_duration_ms)
    result = await correct_recording_timing(local, options)
    failure = {"status": result.status, "reason": result.reason}
    if result.corrected is not None:
        try:
            measured = await ingress.inspect_file(result.corrected, ingress.policy)
            if not isinstance(measured, MP4Inspection):
                raise ValueError("No trusted corrected-file inspection")
            measured.validate_file(FileManifest.from_file(result.corrected), ingress.policy)
            if measured.audio_codec != inspection.audio_codec:
                raise ValueError("Corrected recording lost its audio track")
            manifest = {**raw, "size_bytes": result.corrected.size_bytes,
                "content_sha256": result.corrected.content_sha256,
                "duration_ms": measured.duration_ms, "media_inspection": measured.model_dump(mode="json"),
                "uploaded_original": raw, "timing_correction": result.metadata,
                "temporary_direct_upload": direct}
            candidate = freeze_manifest(scope, manifest)
            if frozen is not None and candidate != frozen:
                raise ValueError("Retry produced different bytes or metadata")
            return PreparedRecording(result.corrected, measured, manifest, owns_local=True)
        except BaseException as error:
            result.corrected.close()
            if not isinstance(error, Exception):
                raise
            failure = {"status": "failed", "reason": "corrected_inspection_failed"}
    if previous is not None:
        raise SourceError("recording_timing_retry_unavailable",
            "The pending corrected recording could not be reproduced; its original is retained.", 503)
    # Never let an optional correction failure discard an otherwise valid capture.
    log.info("Recording timing fallback: %s", failure["reason"])
    return PreparedRecording(local, inspection, {**raw, "timing_processing": failure})


async def finish_timing_save(ingress, scope, frozen, receipt):
    if frozen is None:
        return receipt
    manifest = frozen.manifest
    correction = manifest.get("timing_correction")
    if correction is None:
        if "timing_processing" in manifest:
            return {**receipt, "timing_processing": manifest["timing_processing"]}
        return receipt
    result = {**receipt, "timing_processing": {"status": "corrected",
              "video_delay_ms": correction["video_delay_ms"]}}
    if not manifest.get("temporary_direct_upload"):
        return result
    # Called only after normal publication has verified stored bytes and the
    # committed completion receipt. Never delete the corrected /synced object.
    _, bucket, raw_path = ingress._upload_target(scope)
    try:
        ingress.admissions._authorize(scope.owner_id)
        async with asyncio.timeout(5):
            await ingress.storage.delete_object(bucket=bucket, path=raw_path)
    except Exception:
        # Reconcile/finalize retries repeat this idempotent cleanup. A cleanup
        # outage must not disguise a successfully saved video as a failed save.
        log.warning("Saved corrected recording; temporary upload cleanup pending")
        result["temporary_upload_cleanup_pending"] = True
    return result
