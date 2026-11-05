from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from uuid import UUID

from app.memory.retrieval.models import RETRIEVAL_RECORD_TYPES, RetrievalRecordType


@dataclass(frozen=True, slots=True)
class MemoryRetrievalQuery:
    query: str | None = None
    start_time: datetime | None = None
    end_time: datetime | None = None
    seconds_before_now: int | None = None
    memory_types: tuple[RetrievalRecordType, ...] = ()
    session_id: UUID | None = None
    include_superseded: bool = False
    include_evidence_metadata: bool = True


@dataclass(frozen=True, slots=True)
class NormalizedMemoryRetrievalQuery:
    query: str
    start_time: datetime
    end_time: datetime
    memory_types: tuple[RetrievalRecordType, ...]
    session_id: UUID | None
    include_superseded: bool
    include_evidence_metadata: bool

    def audit_constraints(self) -> dict[str, object]:
        return {
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "memory_types": list(self.memory_types),
            "session_id": str(self.session_id) if self.session_id else None,
            "include_superseded": self.include_superseded,
            "include_evidence_metadata": self.include_evidence_metadata,
        }


def normalize_memory_query(
    value: MemoryRetrievalQuery,
    *,
    now: datetime,
    default_seconds: int,
    max_days: int,
) -> NormalizedMemoryRetrievalQuery:
    current_time = _utc_datetime(now, "now")
    if default_seconds < 1:
        raise ValueError("default retrieval seconds must be positive")
    if max_days < 1:
        raise ValueError("maximum retrieval days must be positive")
    if value.seconds_before_now is not None and (
        value.start_time is not None or value.end_time is not None
    ):
        raise ValueError(
            "seconds_before_now cannot be combined with start_time or end_time"
        )
    if (value.start_time is None) != (value.end_time is None):
        raise ValueError("start_time and end_time must be provided together")

    max_window = timedelta(days=max_days)
    if value.start_time is not None and value.end_time is not None:
        start_time = _utc_datetime(value.start_time, "start_time")
        end_time = _utc_datetime(value.end_time, "end_time")
    else:
        seconds = (
            default_seconds
            if value.seconds_before_now is None
            else value.seconds_before_now
        )
        if isinstance(seconds, bool) or seconds < 1:
            raise ValueError("seconds_before_now must be positive")
        if seconds > int(max_window.total_seconds()):
            raise ValueError(f"retrieval window cannot exceed {max_days} days")
        end_time = current_time
        start_time = current_time - timedelta(seconds=seconds)

    if end_time <= start_time:
        raise ValueError("end_time must be after start_time")
    if end_time - start_time > max_window:
        raise ValueError(f"retrieval window cannot exceed {max_days} days")

    query = (value.query or "").strip()
    if len(query.encode("utf-8")) > 8192:
        raise ValueError("retrieval query exceeds 8192 bytes")
    memory_types = tuple(dict.fromkeys(value.memory_types))
    unsupported = set(memory_types) - RETRIEVAL_RECORD_TYPES
    if unsupported:
        raise ValueError(
            "unsupported retrieval memory types: " + ", ".join(sorted(unsupported))
        )

    return NormalizedMemoryRetrievalQuery(
        query=query,
        start_time=start_time,
        end_time=end_time,
        memory_types=memory_types,
        session_id=value.session_id,
        include_superseded=value.include_superseded,
        include_evidence_metadata=value.include_evidence_metadata,
    )


def build_embedding_query(value: str) -> str:
    query = value.strip()
    if not query:
        raise ValueError("embedding query cannot be blank")
    return f"task: retrieve relevant personal memory | query: {query}"


def _utc_datetime(value: datetime, label: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{label} must be timezone-aware")
    return value.astimezone(UTC)
