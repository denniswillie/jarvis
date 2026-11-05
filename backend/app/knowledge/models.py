"""Trusted source publication records, not model-supplied storage authority."""
from __future__ import annotations

from uuid import UUID
from typing import Literal

from pydantic import AwareDatetime, Field, JsonValue, field_validator

from app.errors import JarvisHTTPError
from app.evidence.models import FrozenRecord, SourceTrust


class SourceError(JarvisHTTPError):
    def __init__(self, code: str, message: str, status: int = 409):
        super().__init__(status, code, message)


class CanonicalOwnerBusy(SourceError):
    """A nonblocking lock attempt failed; no source authority was established."""

    def __init__(self):
        super().__init__("canonical_owner_busy", "Canonical ownership is temporarily busy.")


class MemorySourceReference(FrozenRecord):
    """Exact source interpretation selected by a trusted reader, not model JSON."""
    source_id: UUID
    revision: int = Field(ge=1)
    generation: int = Field(ge=1)
    content_digest: str = Field(pattern="^[0-9a-f]{64}$")
    source_usage: Literal["current", "historical"]

    @classmethod
    def from_metadata(cls, source):
        if type(source["is_current"]) is not bool:
            raise ValueError("Source currentness must be explicit")
        return cls(source_id=source["source_id"], revision=source["revision"],
                   generation=source["generation"], content_digest=source["content_digest"],
                   source_usage="current" if source["is_current"] else "historical")


class KnowledgeAreaDefinition(FrozenRecord):
    """Descriptive area metadata, never a permission or source-access policy."""
    name: str = Field(min_length=1, max_length=200, strict=True)
    domain: str = Field(min_length=1, max_length=200, strict=True)
    description: str = Field(default="", max_length=4000, strict=True)

    @field_validator("name", "domain")
    @classmethod
    def nonblank_label(cls, value):
        if not value.strip() or "\x00" in value:
            raise ValueError("Area labels must be nonblank and contain no NUL")
        return value.strip()

    @field_validator("description")
    @classmethod
    def descriptive_text(cls, value):
        if "\x00" in value:
            raise ValueError("Area descriptions must contain no NUL")
        return value


class SourcePublication(FrozenRecord):
    area_id: UUID
    namespace: str = Field(min_length=1, max_length=100)
    source_key: str = Field(min_length=1, max_length=512)
    payload_id: UUID
    source_type: str = Field(min_length=1, max_length=80)
    trust: SourceTrust
    source_time: AwareDatetime | None
    time_basis: str = Field(min_length=1, max_length=200)
    expected_revision: int = Field(ge=0, strict=True)
    session_id: UUID | None = None

    @field_validator("namespace", "source_key", "source_type", "time_basis")
    @classmethod
    def exact_identity(cls, value):
        if not value.strip() or "\x00" in value:
            raise ValueError("Source metadata must be nonblank and contain no NUL")
        return value


class SourceVersion(FrozenRecord):
    owner_id: str
    area_id: UUID
    source_id: UUID
    revision: int
    generation: int
    payload_id: UUID
    document_id: str
    bank_name: str
    content_digest: str
    source_type: str
    trust: SourceTrust
    source_time: AwareDatetime | None
    time_basis: str
    recorded_at: AwareDatetime
    content: dict[str, JsonValue]
    is_current: bool
    ingestion_status: str
    ingestion_reason: str | None = None
    semantic_index_exclusion: str | None = None
    session_id: UUID | None = None
    context_event_id: UUID | None = None
    context_event_revision: int | None = None
    context_session_id: UUID | None = None
    context_recorded_at: AwareDatetime | None = None

    def original_evidence(self) -> dict:
        """The conversational answer uses original provenance, not vendor labels."""
        return self.model_dump(mode="json", exclude={"bank_name"})
