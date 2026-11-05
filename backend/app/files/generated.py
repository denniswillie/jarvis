"""Bounded publication metadata for exact originals; document inspection belongs in the sandbox."""
from __future__ import annotations

import hashlib
from pathlib import PurePosixPath
from typing import Literal
from uuid import UUID

from pydantic import Field, field_validator, model_serializer, model_validator

from app.evidence.models import FrozenRecord
from app.files.transfer import FileManifest, FileTransferPolicy, MP4Inspection, MP4_OUTPUT_MAX_BYTES

ArtifactFormat = Literal["text", "csv", "json", "pdf", "xlsx", "mp4"]
MEDIA_TYPES = {
    "text": "text/plain; charset=utf-8",
    "csv": "text/csv; charset=utf-8",
    "json": "application/json",
    "pdf": "application/pdf",
    "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "mp4": "video/mp4",
}
EXTENSIONS = {
    "text": {".txt", ".md", ".py", ".js", ".ts", ".sh", ".sql"},
    "csv": {".csv"}, "json": {".json"}, "pdf": {".pdf"}, "xlsx": {".xlsx"}, "mp4": {".mp4"},
}


class ArtifactPolicy(FrozenRecord):
    max_files: int = Field(default=8, ge=1, le=32, strict=True)
    max_file_bytes: int = Field(default=8 * 1024 * 1024, ge=1, le=MP4_OUTPUT_MAX_BYTES, strict=True)
    max_total_bytes: int = Field(default=32 * 1024 * 1024, ge=1, le=MP4_OUTPUT_MAX_BYTES, strict=True)
    formats: tuple[ArtifactFormat, ...] = ("text", "csv", "json", "pdf", "xlsx")
    transfer_policy: Literal["document_v1", "mp4_output_v1"] = "document_v1"
    mp4_max_file_bytes: int | None = Field(default=None, ge=1, le=MP4_OUTPUT_MAX_BYTES, strict=True)

    @model_validator(mode="after")
    def scoped_policy(self):
        if not self.supports_media and self.max_total_bytes > 128 * 1024 * 1024:
            raise ValueError("Document allowances retain their 128 MiB aggregate ceiling")
        if self.transfer_policy == "document_v1":
            if self.max_file_bytes > 8 * 1024 * 1024:
                raise ValueError("Document allowances retain their 8 MiB ceiling")
            if ("mp4" in self.formats) != (self.mp4_max_file_bytes is not None):
                raise ValueError("Mixed output requires explicit MP4 format and byte authority")
        elif self.formats != ("mp4",) or self.mp4_max_file_bytes is not None:
            raise ValueError("Legacy MP4-only allowances use their single existing ceiling")
        return self

    @property
    def supports_media(self):
        return "mp4" in self.formats

    def limit_for(self, format):
        if format not in self.formats:
            raise ValueError("The output format is not authorized")
        return (self.mp4_max_file_bytes if format == "mp4" and self.mp4_max_file_bytes is not None
                else self.max_file_bytes)

    def for_format(self, format, *, deployment_limit_bytes=None):
        """Select one concrete transfer policy from an already-granted format.

        The caller supplies a verified deployment bound when one is configured;
        this method does not discover storage capacity or authorize new formats.
        """
        limit = self.limit_for(format)
        if deployment_limit_bytes is not None:
            if type(deployment_limit_bytes) is not int or deployment_limit_bytes < 1:
                raise ValueError("An explicit positive deployment limit is required")
            limit = min(limit, deployment_limit_bytes)
        return FileTransferPolicy("mp4_output_v1" if format == "mp4" else "document_v1", limit)

    def accepts(self, granted):
        """Semantic subset check, including legacy narrower grant encodings."""
        return (isinstance(granted, ArtifactPolicy) and set(granted.formats) <= set(self.formats)
                and granted.max_files <= self.max_files and granted.max_total_bytes <= self.max_total_bytes
                and all(granted.limit_for(format) <= self.limit_for(format) for format in granted.formats))

    @model_serializer(mode="wrap")
    def compatible_document_scope(self, handler):
        result = handler(self)
        if self.transfer_policy == "document_v1":
            result.pop("transfer_policy", None)
        if self.mp4_max_file_bytes is None:
            result.pop("mp4_max_file_bytes", None)
        return result

    @field_validator("formats")
    @classmethod
    def distinct_formats(cls, value):
        if not value or len(value) != len(set(value)):
            raise ValueError("Select distinct nonempty artifact formats")
        return value


class ArtifactRequest(FrozenRecord):
    operation_key: str = Field(min_length=1, max_length=100)
    filename: str = Field(min_length=1, max_length=120)
    format: ArtifactFormat
    description: str = Field(min_length=1, max_length=2000)

    @field_validator("operation_key", "filename", "description")
    @classmethod
    def ordinary_text(cls, value):
        if not value.strip() or any(ord(character) < 32 for character in value):
            raise ValueError("Artifact metadata must be nonblank and contain no control characters")
        return value

    @field_validator("filename")
    @classmethod
    def leaf_name(cls, value):
        if value in {".", ".."} or "/" in value or "\\" in value or value != value.strip():
            raise ValueError("An artifact filename must be a single filename, not a path")
        return value


class ArtifactInspection(FrozenRecord):
    format: ArtifactFormat
    content_sha256: str = Field(pattern="^[0-9a-f]{64}$")
    size_bytes: int = Field(ge=1, le=MP4_OUTPUT_MAX_BYTES, strict=True)
    media_inspection: MP4Inspection | None = None
    text_excerpt: str | None = Field(default=None, max_length=12000)
    text_complete: bool = False
    # Retain historical parsed counts without inventing counts for new originals.
    item_count: int | None = Field(default=None, ge=0, strict=True)
    meaning: str = ("Publication metadata only; format is declared, not parsed. "
                    "Not verification of document validity, factual claims, formula results or malware safety.")

    @model_validator(mode="after")
    def measured_media(self):
        if self.format != "mp4":
            if self.size_bytes > 8 * 1024 * 1024 or self.media_inspection is not None:
                raise ValueError("Document inspection retains its existing ceiling")
        else:
            if self.media_inspection is None or self.text_excerpt is not None or self.text_complete:
                raise ValueError("MP4 requires measured media metadata, not text")
            self.media_inspection.validate_file(
                FileManifest(size_bytes=self.size_bytes, content_sha256=self.content_sha256),
                FileTransferPolicy("mp4_output_v1", MP4_OUTPUT_MAX_BYTES))
        return self

    @model_serializer(mode="wrap")
    def compatible_document_inspection(self, handler):
        result = handler(self)
        if self.media_inspection is None:
            result.pop("media_inspection", None)
        return result

    @property
    def content_type(self):
        return MEDIA_TYPES[self.format]

    def summary(self):
        return self.model_dump(mode="json", exclude={"text_excerpt"})


class ArtifactSnapshot(FrozenRecord):
    request: ArtifactRequest
    inspection: ArtifactInspection
    parent_grant_id: UUID
    task_id: UUID
    evidence_payload_id: UUID
    transfer_limit_bytes: int | None = Field(default=None, ge=1, le=MP4_OUTPUT_MAX_BYTES, strict=True)

    @model_validator(mode="after")
    def frozen_transfer_limit(self):
        if self.transfer_limit_bytes is not None:
            ceiling = MP4_OUTPUT_MAX_BYTES if self.inspection.format == "mp4" else 8 * 1024 * 1024
            if (self.request.format != self.inspection.format
                    or not self.inspection.size_bytes <= self.transfer_limit_bytes <= ceiling):
                raise ValueError("The frozen transfer limit must fit the exact authorized format and bytes")
        return self

    @model_serializer(mode="wrap")
    def compatible_snapshot(self, handler):
        result = handler(self)
        if self.transfer_limit_bytes is None:
            result.pop("transfer_limit_bytes", None)
        return result

    @property
    def publication_policy(self):
        if self.transfer_limit_bytes is None:
            return None
        return FileTransferPolicy("mp4_output_v1" if self.inspection.format == "mp4" else "document_v1",
                                  self.transfer_limit_bytes)

    def matches(self, content):
        if self.inspection.format == "mp4":
            from app.files.local_file import VerifiedLocalFile
            return (isinstance(content, VerifiedLocalFile) and content.size_bytes == self.inspection.size_bytes
                    and content.content_sha256 == self.inspection.content_sha256)
        return (type(content) is bytes and len(content) == self.inspection.size_bytes
                and hashlib.sha256(content).hexdigest() == self.inspection.content_sha256)


def inspect_artifact(request: ArtifactRequest, content: bytes, policy: ArtifactPolicy) -> ArtifactInspection:
    """Check delivery bounds, not document internals. Never decompress or parse an original."""
    if request.format == "mp4" or type(content) is not bytes or not 1 <= len(content) <= policy.limit_for(request.format):
        raise ValueError("Generated content exceeds its nonempty file-byte allowance")
    if request.format not in policy.formats or PurePosixPath(request.filename).suffix.lower() not in EXTENSIONS[request.format]:
        raise ValueError("The filename and selected format must match an allowed delivery type")
    text = None
    if request.format in {"text", "csv", "json"}:
        # A bounded source excerpt is optional indexing, not format validation.
        # Malformed text remains downloadable without claiming extracted content.
        try:
            candidate = content.decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            pass
        else:
            if "\x00" not in candidate:
                text = candidate
    return ArtifactInspection(format=request.format, content_sha256=hashlib.sha256(content).hexdigest(),
        size_bytes=len(content), text_excerpt=text[:12000] if text is not None else None,
        text_complete=text is not None and len(text) <= 12000)


def inspect_file_artifact(request, local_file, policy, media_inspection):
    """Bind an independently measured MP4 to the existing immutable output snapshot."""
    from app.files.local_file import VerifiedLocalFile
    if (not isinstance(local_file, VerifiedLocalFile) or request.format != "mp4"
            or not policy.supports_media
            or PurePosixPath(request.filename).suffix.lower() != ".mp4"
            or not 0 < local_file.size_bytes <= policy.limit_for("mp4")
            or not isinstance(media_inspection, MP4Inspection)):
        raise ValueError("The exact file is outside the trusted MP4 allowance")
    media_inspection.validate_file(FileManifest.from_file(local_file),
                                   policy.for_format("mp4"))
    return ArtifactInspection(format="mp4", size_bytes=local_file.size_bytes,
        content_sha256=local_file.content_sha256, media_inspection=media_inspection,
        meaning="Measured container/track metadata for exact generated bytes; not proof of scene meaning, "
                "complete decoding, playback, delivery or sensor origin.")
