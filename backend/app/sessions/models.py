from __future__ import annotations

from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field


SessionStatus = Literal["active", "closed"]
MessageRole = Literal["user", "assistant", "tool", "system"]
MessageModality = Literal["text", "audio", "visual", "tool"]


class JarvisSession(BaseModel):
    id: UUID
    title: str
    status: SessionStatus
    created_at: datetime
    updated_at: datetime
    closed_at: datetime | None
    last_activity_at: datetime
    behavior_instructions: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class JarvisMessage(BaseModel):
    id: UUID
    session_id: UUID
    role: MessageRole
    modality: MessageModality
    content: str
    provider_item_id: str | None
    provider_response_id: str | None
    created_at: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)
    source_payload_id: UUID | None = None


class StartSessionRequest(BaseModel):
    title: str | None = None
    resume_active: bool = True


class UpdateBehaviorInstructionsRequest(BaseModel):
    behavior_instructions: str


class RealtimeConfig(BaseModel):
    audio_websocket_url: str
    visual_websocket_url: str


class SessionStartResponse(BaseModel):
    session: JarvisSession
    realtime: RealtimeConfig


class ActiveSessionResponse(BaseModel):
    session: JarvisSession | None
    realtime: RealtimeConfig | None = None


class SessionDetailResponse(BaseModel):
    session: JarvisSession
    realtime: RealtimeConfig
    recent_messages: list[JarvisMessage] | None = None
