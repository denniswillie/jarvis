from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal
from uuid import UUID


RetrievalRecordType = Literal[
    "experience_event",
    "episode",
    "semantic_fact",
    "reflection",
    "procedure",
    "goal",
]
EvidenceReinspectionStatus = Literal["verified", "structured_only"]

RETRIEVAL_RECORD_TYPES = frozenset(
    {
        "experience_event",
        "episode",
        "semantic_fact",
        "reflection",
        "procedure",
        "goal",
    }
)


@dataclass(frozen=True, slots=True)
class RetrievalDocument:
    id: UUID
    owner_id: str
    session_id: UUID | None
    record_type: RetrievalRecordType
    occurred_at: datetime
    valid_from: datetime | None
    valid_to: datetime | None
    title: str
    summary: str
    content: str
    importance: float
    confidence: float
    record_status: str
    is_explicit: bool
    evidence_count: int
    keyframe_count: int
    evidence_types: tuple[str, ...]
    retrieval_count: int = 0

    def temporal_status_at(self, now: datetime) -> Literal["current", "historical", "future"]:
        if self.record_status != "active" or (self.valid_to is not None and self.valid_to <= now):
            return "historical"
        if self.valid_from is not None and self.valid_from > now:
            return "future"
        return "current"

    def __post_init__(self) -> None:
        if not self.owner_id.strip():
            raise ValueError("retrieval document owner_id cannot be blank")
        if self.record_type not in RETRIEVAL_RECORD_TYPES:
            raise ValueError(f"unsupported retrieval record type: {self.record_type}")
        if not 0 <= self.importance <= 1:
            raise ValueError("retrieval document importance must be between 0 and 1")
        if not 0 <= self.confidence <= 1:
            raise ValueError("retrieval document confidence must be between 0 and 1")
        if self.evidence_count < 0 or self.keyframe_count < 0:
            raise ValueError("retrieval evidence counts cannot be negative")
        if self.retrieval_count < 0:
            raise ValueError("retrieval_count cannot be negative")


@dataclass(frozen=True, slots=True)
class RetrievalSourceHit:
    document: RetrievalDocument
    source: str
    raw_score: float
    exact_match: bool = False

    def __post_init__(self) -> None:
        if not self.source.strip():
            raise ValueError("retrieval source cannot be blank")


@dataclass(frozen=True, slots=True)
class FusedRetrievalCandidate:
    document: RetrievalDocument
    score: float
    sources: tuple[str, ...]
    exact_match: bool


@dataclass(frozen=True, slots=True)
class MemoryRetrievalResult:
    id: UUID
    record_type: RetrievalRecordType
    session_id: UUID | None
    occurred_at: datetime
    valid_from: datetime | None
    valid_to: datetime | None
    title: str
    summary: str
    confidence: float
    importance: float
    status: str
    is_historical: bool
    evidence_available: bool
    keyframe_count: int
    match_explanation: tuple[str, ...]
    evidence_metadata: Mapping[str, Any]
    temporal_status: Literal["current", "historical", "future"] = "current"


@dataclass(frozen=True, slots=True)
class EvidenceReinspectionResult:
    status: EvidenceReinspectionStatus
    verified_details: str
    uncertainty: str | None
    confidence: float
    pixel_evidence_available: bool
    supporting_frame_ids: tuple[UUID, ...]
    provider: str | None
    model: str | None
    fallback_reason: str | None

    def __post_init__(self) -> None:
        if not self.verified_details.strip():
            raise ValueError("reinspection details cannot be blank")
        if not 0 <= self.confidence <= 1:
            raise ValueError("reinspection confidence must be between 0 and 1")
        if len(self.supporting_frame_ids) > 6:
            raise ValueError("reinspection may reference at most six frames")
        if len(self.supporting_frame_ids) != len(set(self.supporting_frame_ids)):
            raise ValueError("reinspection supporting frame IDs must be unique")
        if self.status == "verified" and not self.supporting_frame_ids:
            raise ValueError("verified reinspection requires supporting frames")
        if self.status == "verified" and not self.pixel_evidence_available:
            raise ValueError("verified reinspection requires available pixel evidence")
        if self.status == "verified" and (not self.provider or not self.model):
            raise ValueError("verified reinspection requires provider and model")
        if self.status == "structured_only" and self.supporting_frame_ids:
            raise ValueError("structured-only reinspection cannot claim supporting frames")
        if self.status == "structured_only" and not self.fallback_reason:
            raise ValueError("structured-only reinspection requires a fallback reason")


@dataclass(frozen=True, slots=True)
class MemoryRetrievalResponse:
    retrieval_run_id: UUID
    results: tuple[MemoryRetrievalResult, ...]
    candidate_count: int
    provider_spaces: tuple[str, ...]
    evidence_reinspection: EvidenceReinspectionResult | None = None
