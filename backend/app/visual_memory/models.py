from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal
from uuid import UUID


VisualFrameSource = Literal["glasses", "iphone_camera"]
RawImageStatus = Literal["available", "expiring", "expired", "deleted", "failed"]
AnalyzerStatus = Literal["pending", "skipped", "completed", "failed"]


@dataclass(frozen=True)
class ValidatedVisualFrame:
    frame_sequence: int
    captured_at: datetime
    source: VisualFrameSource
    content_type: str
    width: int
    height: int
    image_bytes: bytes


@dataclass(frozen=True)
class VisualFrameStorageObject:
    bucket: str
    path: str
    size_bytes: int


@dataclass(frozen=True)
class VisualFrameRecord:
    id: UUID
    owner_id: str
    session_id: UUID
    audio_session_id: str | None
    visual_session_id: UUID
    frame_sequence: int
    source: VisualFrameSource
    captured_at: datetime
    server_received_at: datetime
    content_type: str
    width: int
    height: int
    storage_bucket: str
    storage_path: str
    raw_image_status: RawImageStatus
    raw_image_expires_at: datetime
    analyzer_status: AnalyzerStatus
    frame_description: str | None
    memory_tags: list[str]
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class BufferedVisualFrame:
    frame_id: UUID
    owner_id: str
    session_id: UUID
    audio_session_id: str | None
    visual_session_id: UUID
    frame_sequence: int
    source: VisualFrameSource
    captured_at: datetime
    server_received_at: datetime
    content_type: str
    width: int
    height: int
    storage_bucket: str
    storage_path: str
    raw_image_expires_at: datetime


@dataclass(frozen=True)
class RecentVisualInterrupt:
    id: UUID
    visual_frame_id: UUID | None
    message: str
    reason: str | None
    confidence: float
    dedupe_bucket: str
    created_at: datetime


@dataclass(frozen=True)
class ActiveVisualReminder:
    id: UUID
    title: str
    details: str
    remind_at: datetime | None
    condition: str | None
    updated_at: datetime


@dataclass(frozen=True)
class VisualInterruptRecord:
    id: UUID
    owner_id: str
    session_id: UUID
    visual_frame_id: UUID | None
    message: str
    reason: str | None
    confidence: float
    dedupe_bucket: str
    spoken_response_id: str | None
    created_at: datetime


@dataclass(frozen=True)
class UserPreference:
    key: str
    value: str
    source: str
    updated_at: datetime
