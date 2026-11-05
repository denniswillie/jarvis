from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from math import isfinite
from typing import Any, Literal, Mapping
from urllib.parse import parse_qsl, urlsplit
from uuid import UUID


ToolExecutionStatus = Literal["started", "completed", "failed"]
ExperienceEventStatus = Literal["open", "closed", "consolidating", "consolidated", "failed"]
ExperienceEventKind = Literal["mixed", "visual", "interaction", "tool_activity"]
ExperienceModality = Literal["audio", "visual", "text", "tool"]
ExperienceEvidenceType = Literal["observation", "message", "tool_execution"]
KeyframeSelectedBy = Literal["deterministic", "gpt", "user"]
RetentionReasonType = Literal[
    "event_candidate",
    "ordinary_event_keyframe",
    "important_event_keyframe",
    "user_capture",
    "explicit_memory",
    "visual_interrupt",
]

PERMANENT_RETENTION_REASON_TYPES = frozenset(
    {"important_event_keyframe", "user_capture", "explicit_memory"}
)
MAX_TOOL_PAYLOAD_BYTES = 65536
MAX_EVIDENCE_METADATA_BYTES = 8192

_FORBIDDEN_PAYLOAD_KEYS = frozenset(
    {
        "access_token",
        "api_key",
        "audio_base64",
        "authorization",
        "bearer_token",
        "client_secret",
        "file_bytes",
        "id_token",
        "image_base64",
        "password",
        "raw_bytes",
        "refresh_token",
        "secret",
        "service_role_key",
        "signed_download_url",
        "signed_upload_url",
        "signed_url",
    }
)
_SIGNED_QUERY_KEYS = frozenset(
    {"token", "signature", "sig", "x_amz_signature", "x_goog_signature"}
)


class UnsafeToolPayloadError(ValueError):
    pass


@dataclass(frozen=True)
class ToolExecution:
    id: UUID
    owner_id: str
    session_id: UUID
    provider_call_id: str
    tool_name: str
    mutates_state: bool
    arguments: Mapping[str, Any]
    result: Mapping[str, Any] | None
    status: ToolExecutionStatus
    error: Mapping[str, Any] | None
    started_at: datetime
    completed_at: datetime | None
    created_at: datetime

    def __post_init__(self) -> None:
        _require_text(self.owner_id, "owner id")
        _require_text(self.provider_call_id, "provider call id")
        _require_text(self.tool_name, "tool name")
        if self.status == "started":
            if self.result is not None or self.error is not None or self.completed_at is not None:
                raise ValueError("started tool execution cannot contain terminal fields")
        elif self.status == "completed":
            if self.result is None or self.error is not None or self.completed_at is None:
                raise ValueError("completed tool execution requires result and completion time")
        elif self.status == "failed":
            if self.error is None or self.completed_at is None:
                raise ValueError("failed tool execution requires error and completion time")
        else:
            raise ValueError(f"unsupported tool execution status: {self.status}")
        if self.completed_at is not None and self.completed_at < self.started_at:
            raise ValueError("tool completion cannot be before start")


@dataclass(frozen=True)
class ExperienceEvent:
    id: UUID
    owner_id: str
    session_id: UUID
    status: ExperienceEventStatus
    event_kind: ExperienceEventKind
    modalities: tuple[ExperienceModality, ...]
    started_at: datetime
    ended_at: datetime | None
    title: str
    summary: str
    scene: str | None
    activity: str | None
    state: Mapping[str, Any]
    boundary_reason: str | None
    importance: float
    confidence: float
    keyframes_finalized_at: datetime | None
    consolidation_ready_at: datetime | None
    evidence_reduced_at: datetime | None
    retention_expires_at: datetime | None
    consolidation_result: Mapping[str, Any] | None
    consolidation_provider: str | None
    consolidation_model: str | None
    consolidated_at: datetime | None
    created_at: datetime
    updated_at: datetime

    def __post_init__(self) -> None:
        _require_text(self.owner_id, "owner id")
        if self.event_kind not in {"mixed", "visual", "interaction", "tool_activity"}:
            raise ValueError(f"unsupported event kind: {self.event_kind}")
        if len(set(self.modalities)) != len(self.modalities):
            raise ValueError("event modalities must be unique")
        if not set(self.modalities).issubset({"audio", "visual", "text", "tool"}):
            raise ValueError("event contains an unsupported modality")
        _require_score(self.importance, "importance")
        _require_score(self.confidence, "confidence")
        if self.status == "open":
            if (
                self.ended_at is not None
                or self.keyframes_finalized_at is not None
                or self.consolidation_ready_at is not None
            ):
                raise ValueError("open event cannot contain closure markers")
        elif self.status in {"closed", "consolidating", "consolidated", "failed"}:
            if (
                self.ended_at is None
                or self.keyframes_finalized_at is None
                or self.consolidation_ready_at is None
            ):
                raise ValueError("closed event state requires atomic readiness markers")
        else:
            raise ValueError(f"unsupported experience event status: {self.status}")
        if self.ended_at is not None and self.ended_at < self.started_at:
            raise ValueError("event end cannot be before start")
        if self.status == "consolidated" and (
            self.consolidation_result is None
            or not self.consolidation_provider
            or not self.consolidation_model
            or self.consolidated_at is None
        ):
            raise ValueError("consolidated event requires model provenance and result")


@dataclass(frozen=True)
class ExperienceEventEvidence:
    id: UUID
    event_id: UUID
    evidence_type: ExperienceEvidenceType
    evidence_ref_id: UUID
    occurred_at: datetime
    source_sequence: int | None
    ingested_at: datetime
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.evidence_type not in {"observation", "message", "tool_execution"}:
            raise ValueError(f"unsupported evidence type: {self.evidence_type}")
        if self.source_sequence is not None and self.source_sequence < 0:
            raise ValueError("source sequence cannot be negative")


@dataclass(frozen=True)
class ExperienceEventKeyframe:
    event_id: UUID
    visual_frame_id: UUID
    rank: int
    reason: str
    quality_score: float
    selected_by: KeyframeSelectedBy
    created_at: datetime

    def __post_init__(self) -> None:
        if self.rank < 1:
            raise ValueError("keyframe rank must be positive")
        _require_text(self.reason, "keyframe reason")
        _require_score(self.quality_score, "quality score")
        if self.selected_by not in {"deterministic", "gpt", "user"}:
            raise ValueError(f"unsupported keyframe selector: {self.selected_by}")


@dataclass(frozen=True)
class VisualFrameRetentionReason:
    id: UUID
    owner_id: str
    visual_frame_id: UUID
    reason_type: RetentionReasonType
    reason_ref_id: UUID
    expires_at: datetime | None
    created_at: datetime

    def __post_init__(self) -> None:
        _require_text(self.owner_id, "owner id")
        if self.reason_type not in {
            "event_candidate",
            "ordinary_event_keyframe",
            "important_event_keyframe",
            "user_capture",
            "explicit_memory",
            "visual_interrupt",
        }:
            raise ValueError(f"unsupported retention reason: {self.reason_type}")
        if self.expires_at is None and self.reason_type not in PERMANENT_RETENTION_REASON_TYPES:
            raise ValueError(f"{self.reason_type} requires a finite expiry")


def normalize_durable_tool_payload(
    payload: Mapping[str, Any],
    *,
    label: str,
    max_bytes: int = MAX_TOOL_PAYLOAD_BYTES,
) -> dict[str, Any]:
    normalized = _normalize_json_value(payload, path=label, depth=0)
    if not isinstance(normalized, dict):
        raise UnsafeToolPayloadError(f"{label} must be a JSON object")
    encoded = json.dumps(normalized, allow_nan=False, separators=(",", ":"))
    if len(encoded.encode("utf-8")) > max_bytes:
        raise UnsafeToolPayloadError(f"{label} exceeds {max_bytes} bytes")
    return normalized


def normalize_evidence_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    return normalize_durable_tool_payload(
        metadata,
        label="evidence metadata",
        max_bytes=MAX_EVIDENCE_METADATA_BYTES,
    )


def _normalize_json_value(value: Any, *, path: str, depth: int) -> Any:
    if depth > 12:
        raise UnsafeToolPayloadError(f"{path} exceeds maximum nesting depth")
    if value is None or isinstance(value, (str, bool, int)):
        if isinstance(value, str):
            _reject_signed_url(value, path)
            if value.lower().startswith("bearer "):
                raise UnsafeToolPayloadError(f"{path} contains a bearer credential")
        return value
    if isinstance(value, float):
        if not isfinite(value):
            raise UnsafeToolPayloadError(f"{path} contains a non-finite number")
        return value
    if isinstance(value, bytes | bytearray | memoryview):
        raise UnsafeToolPayloadError(f"{path} contains raw bytes")
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for raw_key, item in value.items():
            if not isinstance(raw_key, str):
                raise UnsafeToolPayloadError(f"{path} contains a non-string key")
            key = _normalized_key(raw_key)
            if key in _FORBIDDEN_PAYLOAD_KEYS:
                raise UnsafeToolPayloadError(f"{path}.{raw_key} is not durable-safe")
            normalized[raw_key] = _normalize_json_value(
                item,
                path=f"{path}.{raw_key}",
                depth=depth + 1,
            )
        return normalized
    if isinstance(value, (list, tuple)):
        return [
            _normalize_json_value(item, path=f"{path}[{index}]", depth=depth + 1)
            for index, item in enumerate(value)
        ]
    raise UnsafeToolPayloadError(f"{path} contains non-JSON value {type(value).__name__}")


def _reject_signed_url(value: str, path: str) -> None:
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.query:
        return
    query_keys = {_normalized_key(key) for key, _ in parse_qsl(parsed.query)}
    if query_keys & _SIGNED_QUERY_KEYS:
        raise UnsafeToolPayloadError(f"{path} contains a signed or credentialed URL")


def _normalized_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


def _require_text(value: str, label: str) -> None:
    if not value.strip():
        raise ValueError(f"{label} cannot be blank")


def _require_score(value: float, label: str) -> None:
    if not isfinite(value) or value < 0 or value > 1:
        raise ValueError(f"{label} must be between zero and one")
