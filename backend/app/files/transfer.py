"""Trusted transfer ceilings; these are not request/model selectable grants."""
from dataclasses import dataclass
from typing import Literal

from pydantic import Field

from app.evidence.models import FrozenRecord

DOCUMENT_POLICY = "document_v1"
MP4_CAPTURE_POLICY = "mp4_capture_v1"
MP4_OUTPUT_POLICY = "mp4_output_v1"
MP4_OUTPUT_MAX_BYTES = 500_000_000
MP4_OUTPUT_MAX_DURATION_MS = 1_800_000
CAPTURE_MAX_BYTES = 1024**3
CAPTURE_MAX_DURATION_MS = 1_800_000
CAPTURE_ENCODING_OVERHEAD_MS = 100
CAPTURE_MAX_ENCODED_DURATION_MS = CAPTURE_MAX_DURATION_MS + CAPTURE_ENCODING_OVERHEAD_MS
POLICY_LIMITS = {
    DOCUMENT_POLICY: (8 * 1024 * 1024, None),
    MP4_CAPTURE_POLICY: (CAPTURE_MAX_BYTES, CAPTURE_MAX_DURATION_MS),
    MP4_OUTPUT_POLICY: (MP4_OUTPUT_MAX_BYTES, MP4_OUTPUT_MAX_DURATION_MS),
}


@dataclass(frozen=True)
class FileTransferPolicy:
    policy_id: str = DOCUMENT_POLICY
    deployment_limit_bytes: int | None = None

    def __post_init__(self):
        if self.policy_id not in POLICY_LIMITS:
            raise ValueError("Unknown trusted file policy")
        limit = self.deployment_limit_bytes
        if limit is not None and (type(limit) is not int or limit < 1):
            raise ValueError("Deployment limit must be a verified positive byte count")
        if self.is_media and limit is None:
            raise ValueError("Media requires an explicit verified deployment/bucket limit")

    @property
    def is_media(self):
        return self.policy_id != DOCUMENT_POLICY

    @property
    def max_bytes(self):
        ceiling = POLICY_LIMITS[self.policy_id][0]
        return min(ceiling, self.deployment_limit_bytes or ceiling)

    @property
    def max_duration_ms(self):
        return POLICY_LIMITS[self.policy_id][1]

    @property
    def max_encoded_duration_ms(self):
        # Capture admission stays exact. Only trusted capture inspection/playback
        # allows the small AAC/container tail; generated outputs do not inherit it.
        if self.policy_id == MP4_CAPTURE_POLICY:
            return CAPTURE_MAX_ENCODED_DURATION_MS
        return self.max_duration_ms

    def validate(self, manifest, spec):
        if not isinstance(manifest, FileManifest) or manifest.size_bytes > self.max_bytes:
            raise ValueError("File exceeds its trusted effective transfer limit")
        if self.is_media and (manifest.size_bytes == 0 or spec.content_type != "video/mp4"
                              or not spec.filename.lower().endswith(".mp4")):
            raise ValueError("The media transfer policy requires a nonempty MP4")

    @classmethod
    def from_row(cls, row):
        # Pre-migration records are documents; no wider policy is inferred from MIME.
        policy = cls(row.get("transfer_policy", DOCUMENT_POLICY), row.get("transfer_limit_bytes"))
        if row["size_bytes"] > policy.max_bytes or row["size_bytes"] < 0:
            raise ValueError("Publication exceeds its persisted transfer policy")
        return policy


class FileManifest(FrozenRecord):
    size_bytes: int = Field(ge=0, le=CAPTURE_MAX_BYTES, strict=True)
    content_sha256: str = Field(pattern="^[0-9a-f]{64}$")

    @classmethod
    def from_file(cls, local_file):
        return cls(size_bytes=local_file.size_bytes, content_sha256=local_file.content_sha256)


class MP4Inspection(FrozenRecord):
    """Measured by a trusted inspector outside transactions, never a client claim."""
    format: Literal["jarvis.mp4_inspection.v1"] = "jarvis.mp4_inspection.v1"
    content_sha256: str = Field(pattern="^[0-9a-f]{64}$")
    size_bytes: int = Field(ge=1, le=CAPTURE_MAX_BYTES, strict=True)
    duration_ms: int = Field(ge=1, le=CAPTURE_MAX_ENCODED_DURATION_MS, strict=True)
    width: int = Field(ge=1, le=7680, strict=True)
    height: int = Field(ge=1, le=7680, strict=True)
    video_codec: Literal["h264"]
    audio_codec: Literal["aac"] | None

    def validate_file(self, manifest, policy):
        if (not policy.is_media or self.size_bytes != manifest.size_bytes
                or self.content_sha256 != manifest.content_sha256
                or self.size_bytes > policy.max_bytes
                or self.duration_ms > policy.max_encoded_duration_ms):
            raise ValueError("Media inspection differs from exact bytes or trusted duration policy")
