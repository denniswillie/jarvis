from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal, Mapping, Sequence
from uuid import UUID


MemoryJobStatus = Literal["queued", "claimed", "completed", "failed", "dead"]
MEMORY_JOB_STATUSES: tuple[MemoryJobStatus, ...] = (
    "queued",
    "claimed",
    "completed",
    "failed",
    "dead",
)
ACTIVE_MEMORY_JOB_STATUSES = frozenset({"queued", "claimed", "failed"})
TERMINAL_MEMORY_JOB_STATUSES = frozenset({"completed", "dead"})

MAX_JOB_JSON_BYTES = 262_144
MAX_JOB_JSON_DEPTH = 12
MAX_JOB_ERROR_LENGTH = 4_000


class InvalidMemoryJobPayloadError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class MemoryJob:
    id: UUID
    owner_id: str
    session_id: UUID | None
    job_type: str
    dedupe_key: str
    payload: Mapping[str, Any]
    status: MemoryJobStatus
    priority: int
    attempt_count: int
    max_attempts: int
    available_at: datetime
    claimed_at: datetime | None
    claimed_by: str | None
    claim_token: UUID | None
    result: Mapping[str, Any] | None
    last_error: str | None
    completed_at: datetime | None
    created_at: datetime
    updated_at: datetime

    def __post_init__(self) -> None:
        _require_text(self.owner_id, "owner_id")
        _require_text(self.job_type, "job_type")
        _require_text(self.dedupe_key, "dedupe_key")
        if self.status not in ACTIVE_MEMORY_JOB_STATUSES | TERMINAL_MEMORY_JOB_STATUSES:
            raise ValueError(f"unsupported memory job status: {self.status}")
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        if self.attempt_count < 0 or self.attempt_count > self.max_attempts:
            raise ValueError("attempt_count is outside the allowed range")
        if self.status == "queued":
            if self.attempt_count >= self.max_attempts:
                raise ValueError("queued job must have a remaining attempt")
            self._require_unclaimed()
            self._require_no_outcome()
        elif self.status == "claimed":
            if self.attempt_count < 1:
                raise ValueError("claimed job must have an attempt")
            if self.claimed_at is None or self.claim_token is None:
                raise ValueError("claimed job requires claim metadata")
            _require_text(self.claimed_by, "claimed_by")
            self._require_no_outcome()
        elif self.status == "completed":
            if self.attempt_count < 1 or self.completed_at is None:
                raise ValueError("completed job requires an attempt and completion time")
            self._require_unclaimed()
            if self.last_error is not None:
                raise ValueError("completed job cannot have an error")
        elif self.status == "failed":
            if not 1 <= self.attempt_count < self.max_attempts:
                raise ValueError("failed job must remain retryable")
            self._require_unclaimed()
            self._require_failure()
        else:
            if self.attempt_count != self.max_attempts:
                raise ValueError("dead job must have exhausted its attempts")
            self._require_unclaimed()
            self._require_failure()

    @property
    def is_active(self) -> bool:
        return self.status in ACTIVE_MEMORY_JOB_STATUSES

    @property
    def is_terminal(self) -> bool:
        return self.status in TERMINAL_MEMORY_JOB_STATUSES

    def _require_unclaimed(self) -> None:
        if self.claimed_at is not None or self.claimed_by is not None or self.claim_token is not None:
            raise ValueError(f"{self.status} job cannot retain an active claim")

    def _require_no_outcome(self) -> None:
        if self.result is not None or self.last_error is not None or self.completed_at is not None:
            raise ValueError(f"{self.status} job cannot have a terminal outcome")

    def _require_failure(self) -> None:
        _require_text(self.last_error, "last_error")
        if self.result is not None or self.completed_at is not None:
            raise ValueError(f"{self.status} job cannot have a completion result")


@dataclass(frozen=True, slots=True)
class MemoryJobEnqueueResult:
    job: MemoryJob
    created: bool


@dataclass(frozen=True, slots=True)
class MemoryJobQueueDiagnostics:
    counts_by_status: Mapping[MemoryJobStatus, int]
    oldest_queued_age_seconds: float | None
    active_job_types: tuple[str, ...]
    last_completed_at: datetime | None


def normalize_job_json_object(
    value: Mapping[str, Any],
    *,
    label: str,
    max_bytes: int = MAX_JOB_JSON_BYTES,
    max_depth: int = MAX_JOB_JSON_DEPTH,
) -> dict[str, Any]:
    if type(max_depth) is not int or not 0 <= max_depth <= 128:
        raise ValueError("Invalid JSON nesting bound")
    normalized = _normalize_json(value, label=label, depth=0, max_depth=max_depth)
    if not isinstance(normalized, dict):
        raise InvalidMemoryJobPayloadError(f"{label} must be a JSON object")
    encoded = json.dumps(
        normalized,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    if len(encoded) > max_bytes:
        raise InvalidMemoryJobPayloadError(f"{label} exceeds {max_bytes} bytes")
    return normalized


def normalize_job_error(error: str) -> str:
    normalized = error.strip()
    if not normalized:
        raise ValueError("job error cannot be blank")
    return normalized[:MAX_JOB_ERROR_LENGTH]


def _normalize_json(value: Any, *, label: str, depth: int, max_depth: int = MAX_JOB_JSON_DEPTH) -> Any:
    if depth > max_depth:
        raise InvalidMemoryJobPayloadError(f"{label} is nested too deeply")
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise InvalidMemoryJobPayloadError(f"{label} contains a non-finite number")
        return value
    if isinstance(value, bytes | bytearray | memoryview):
        raise InvalidMemoryJobPayloadError(f"{label} cannot contain raw bytes")
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise InvalidMemoryJobPayloadError(f"{label} keys must be strings")
            normalized[key] = _normalize_json(item, label=label, depth=depth + 1, max_depth=max_depth)
        return normalized
    if isinstance(value, Sequence) and not isinstance(value, str):
        return [_normalize_json(item, label=label, depth=depth + 1, max_depth=max_depth) for item in value]
    raise InvalidMemoryJobPayloadError(
        f"{label} contains unsupported value type {type(value).__name__}"
    )


def _require_text(value: str | None, label: str) -> str:
    if value is None or not value.strip():
        raise ValueError(f"{label} cannot be blank")
    return value.strip()
