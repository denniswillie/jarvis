"""Immutable authority snapshots; admission does not prove capture or hearing."""
from dataclasses import dataclass
import hashlib
import json
import re
from typing import Literal
from uuid import UUID

from pydantic import Field

from app.evidence.models import FrozenRecord, OwnedReference
from app.files.transfer import CAPTURE_MAX_BYTES, CAPTURE_MAX_DURATION_MS, CAPTURE_ENCODING_OVERHEAD_MS, MP4_CAPTURE_POLICY


FINALIZATION_FORMAT = "jarvis.recording_finalization.v1"
MAX_MANIFEST_BYTES = 16_384


class RecordingScope(FrozenRecord):
    recording_id: UUID
    owner_id: str
    session_id: UUID
    area_id: UUID
    include_jarvis_audio: bool = Field(strict=True)
    transfer_policy: Literal["mp4_capture_v1"] = MP4_CAPTURE_POLICY
    max_bytes: int = Field(ge=1, le=CAPTURE_MAX_BYTES, strict=True)
    max_duration_ms: int = Field(ge=1, le=CAPTURE_MAX_DURATION_MS, strict=True)
    admission_event_id: UUID
    admission_payload_id: UUID


class RecordingReceipt(FrozenRecord):
    scope: RecordingScope
    operation_id: UUID
    outcome: Literal["admitted", "capture_closed", "saved"]
    receipt_event_id: UUID
    receipt_payload_id: UUID
    replayed: bool
    capture_allowed: bool
    publication_id: UUID | None = None
    file_id: UUID | None = None
    source_id: UUID | None = None
    version_id: str | None = None


@dataclass(frozen=True)
class FrozenFinalization:
    manifest_json: str
    manifest_digest: str

    @property
    def manifest(self):
        return json.loads(self.manifest_json)


def freeze_manifest(scope: RecordingScope, manifest: dict) -> FrozenFinalization:
    """Ingress validates nested media/client schemas; domain freezes exact data."""
    if type(scope) is not RecordingScope or type(manifest) is not dict:
        raise ValueError("An admitted scope and exact manifest object are required")
    OwnedReference.validate_owner(scope.owner_id)
    # Reject Python-only values, non-string keys, nonfinite numbers and excessive
    # nesting before canonicalization; JSON coercion must not merge identities.
    def json_value(value, depth=0):
        if depth > 12:
            raise ValueError("Recording manifest exceeds its nesting bound")
        if type(value) is dict:
            if any(type(key) is not str for key in value):
                raise ValueError("Recording manifest keys must be strings")
            for item in value.values():
                json_value(item, depth + 1)
        elif type(value) is list:
            for item in value:
                json_value(item, depth + 1)
        elif value is not None and type(value) not in {str, int, float, bool}:
            raise ValueError("Recording manifest must contain only JSON values")
    json_value(manifest)
    report = manifest.get("client_report", {})
    captured_ms = report.get("timeline_end_ms") if isinstance(report, dict) else None
    if type(captured_ms) is not int or not 1 <= captured_ms <= scope.max_duration_ms:
        raise ValueError("Recording capture timeline differs from its admitted bounds")
    padding_ms = 0
    if "timing_correction" in manifest or "uploaded_original" in manifest:
        original, correction = manifest.get("uploaded_original"), manifest.get("timing_correction")
        if (type(original) is not dict or type(correction) is not dict
                or "uploaded_original" in original or "timing_correction" in original):
            raise ValueError("Invalid recording timing provenance")
        freeze_manifest(scope, original)
        padding_ms = correction.get("video_delay_ms")
        if (type(padding_ms) is not int or not 1 <= padding_ms <= 2000
                or correction.get("format") != "jarvis.recording_timing.v1"
                or correction.get("audio_reencoded") is not False
                or correction.get("original_sha256") != original["content_sha256"]
                or correction.get("corrected_sha256") != manifest.get("content_sha256")
                or any(manifest.get(key) != original.get(key) for key in ("filename", "description", "client_report"))
                or type(manifest.get("temporary_direct_upload")) is not bool
                or type(manifest.get("duration_ms")) is not int
                or manifest["duration_ms"] > original["duration_ms"] + padding_ms + CAPTURE_ENCODING_OVERHEAD_MS):
            raise ValueError("Recording correction differs from its original capture")
    if (manifest.get("format") != FINALIZATION_FORMAT
            or type(manifest.get("size_bytes")) is not int
            or not 1 <= manifest["size_bytes"] <= scope.max_bytes
            or type(manifest.get("duration_ms")) is not int
            or not 1 <= manifest["duration_ms"] <= min(
                captured_ms + padding_ms + CAPTURE_ENCODING_OVERHEAD_MS,
                scope.max_duration_ms + CAPTURE_ENCODING_OVERHEAD_MS)
            or not isinstance(manifest.get("content_sha256"), str)
            or re.fullmatch("[0-9a-f]{64}", manifest["content_sha256"]) is None):
        raise ValueError("Recording manifest differs from its admitted capture bounds")
    text = json.dumps(manifest, sort_keys=True, ensure_ascii=True, separators=(",", ":"), allow_nan=False)
    encoded = text.encode("utf-8")
    if len(encoded) > MAX_MANIFEST_BYTES:
        raise ValueError("Recording manifest exceeds its byte bound")
    return FrozenFinalization(text, hashlib.sha256(encoded).hexdigest())
