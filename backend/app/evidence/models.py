"""Immutable evidence snapshots, not a substitute for checking current access.

Source timestamps may be ahead of the server clock. Preserve them, but never
use future, missing, or provisional evidence as current positive support.
"""
from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Self
from uuid import UUID

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, field_validator, model_validator


OwnerID = Annotated[str, Field(min_length=1, max_length=200)]
ReferenceID = Annotated[str, Field(min_length=1, max_length=512)]


class ClaimBasis(StrEnum):
    OBSERVED = "observed"
    USER_REPORTED = "user_reported"
    SOURCE_REPORTED = "source_reported"
    INFERRED = "inferred"
    UNKNOWN = "unknown"


class SourceAvailability(StrEnum):
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    EXPIRED = "expired"
    DELETED = "deleted"
    NOT_FINALIZED = "not_finalized"


class SourceTrust(StrEnum):
    USER_INPUT = "user_input"
    SENSOR = "sensor"
    EXTERNAL_CONTENT = "external_content"
    ASSISTANT_OUTPUT = "assistant_output"
    TOOL_RECEIPT = "tool_receipt"
    DERIVED = "derived"


class FrozenRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class OwnedReference(FrozenRecord):
    owner_id: OwnerID
    id: UUID

    @field_validator("owner_id")
    @classmethod
    def validate_owner(cls, value: str) -> str:
        if value != value.strip() or not value or "\x00" in value:
            raise ValueError("owner identity must be nonblank, exact, and contain no NUL")
        return value


class EvidenceRef(FrozenRecord):
    owner_id: OwnerID
    source_type: Annotated[str, Field(min_length=1, max_length=80)]
    source_id: ReferenceID
    source_version: Annotated[str, Field(min_length=1, max_length=256)]
    event_time: AwareDatetime
    recorded_at: AwareDatetime
    captured_at: AwareDatetime | None = None
    knowledge_area_id: UUID | None = None
    availability: SourceAvailability
    trust: SourceTrust

    @field_validator("owner_id")
    @classmethod
    def validate_owner(cls, value: str) -> str:
        return OwnedReference.validate_owner(value)

    @field_validator("source_type", "source_id", "source_version")
    @classmethod
    def validate_identity(cls, value: str) -> str:
        if not value.strip() or "\x00" in value:
            raise ValueError("source identity must be nonblank and contain no NUL")
        return value

    def require_scope(self, *, owner_id: str, allowed_area_ids: frozenset[UUID] | None = None) -> None:
        if self.owner_id != owner_id:
            raise ValueError("evidence belongs to another owner")
        if (allowed_area_ids is not None and self.knowledge_area_id is not None
                and self.knowledge_area_id not in allowed_area_ids):
            raise ValueError("evidence lies outside the authorized knowledge areas")

    def is_available_at(self, at: datetime) -> bool:
        if at.tzinfo is None or at.utcoffset() is None:
            raise ValueError("evidence comparison requires a timezone-aware timestamp")
        return (
            self.availability == SourceAvailability.AVAILABLE
            and self.recorded_at <= at and self.event_time <= at
            and (self.captured_at is None or self.captured_at <= at)
        )


class Claim(FrozenRecord):
    owner_id: OwnerID
    id: UUID
    content: Annotated[str, Field(min_length=1, max_length=32_000)]
    basis: ClaimBasis
    evidence_refs: tuple[EvidenceRef, ...] = Field(default=(), max_length=128)
    valid_from: AwareDatetime | None = None
    valid_to: AwareDatetime | None = None
    uncertainty: Annotated[str, Field(max_length=4_000)] = ""
    confidence: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)] | None = None
    superseded_by: OwnedReference | None = None

    @field_validator("owner_id")
    @classmethod
    def validate_owner(cls, value: str) -> str:
        return OwnedReference.validate_owner(value)

    @field_validator("content")
    @classmethod
    def validate_content(cls, value: str) -> str:
        if not value.strip() or "\x00" in value:
            raise ValueError("claim content must be nonblank and contain no NUL")
        return value

    @model_validator(mode="after")
    def validate_references(self) -> Self:
        for reference in self.evidence_refs:
            reference.require_scope(owner_id=self.owner_id)
        if self.superseded_by is not None:
            if self.superseded_by.owner_id != self.owner_id or self.superseded_by.id == self.id:
                raise ValueError("replacement must be a different claim of the same owner")
        if self.valid_to is not None and self.valid_from is None:
            raise ValueError("a validity end requires a known validity start")
        if self.valid_to is not None and self.valid_to <= self.valid_from:
            raise ValueError("claim validity must be a nonempty interval")
        if self.basis != ClaimBasis.UNKNOWN and not self.evidence_refs:
            raise ValueError("a non-unknown claim requires attributed evidence")
        if self.basis == ClaimBasis.UNKNOWN and not self.uncertainty.strip():
            raise ValueError("unknown claims must describe the missing knowledge")
        required_trust = {
            ClaimBasis.OBSERVED: {SourceTrust.SENSOR, SourceTrust.TOOL_RECEIPT},
            ClaimBasis.USER_REPORTED: {SourceTrust.USER_INPUT},
            ClaimBasis.SOURCE_REPORTED: {SourceTrust.EXTERNAL_CONTENT, SourceTrust.TOOL_RECEIPT},
        }.get(self.basis)
        if required_trust is not None and not any(ref.trust in required_trust for ref in self.evidence_refs):
            raise ValueError("claim basis lacks a source of the stated origin")
        return self

    def available_evidence(self, at: datetime) -> tuple[EvidenceRef, ...]:
        """Availability only; this does not establish semantic truth or freshness."""
        return tuple(ref for ref in self.evidence_refs if ref.is_available_at(at))
