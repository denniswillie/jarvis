from __future__ import annotations

from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field


ReminderStatus = Literal["active", "completed", "failed", "deleted"]
MAX_REMINDER_DELAY_SECONDS = 31_536_000  # One year; longer schedules can use absolute time.
ReminderEventType = Literal[
    "generated",
    "dispatched_to_session",
    "delivered_to_session",
    "missed",
    "failed",
    "interrupted",
]


class JarvisReminder(BaseModel):
    id: UUID
    session_id: UUID | None
    title: str
    details: str
    remind_at: datetime | None
    condition: str | None
    status: ReminderStatus
    created_at: datetime
    updated_at: datetime
    completed_at: datetime | None
    triggered_at: datetime | None
    trigger_claimed_at: datetime | None
    trigger_claim_token: UUID | None
    trigger_attempt_count: int
    last_trigger_error: str | None
    metadata: dict[str, Any] = Field(default_factory=dict)


class JarvisReminderEvent(BaseModel):
    id: UUID
    reminder_id: UUID
    session_id: UUID | None
    event_type: ReminderEventType
    message: str
    provider_response_id: str | None
    created_at: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)


class ActiveReminderResponse(BaseModel):
    id: UUID
    title: str
    details: str
    remind_at: datetime | None
    condition: str | None
    status: ReminderStatus
    updated_at: datetime


class ActiveRemindersResponse(BaseModel):
    reminders: list[ActiveReminderResponse]


class ReminderEventResponse(BaseModel):
    id: UUID
    reminder_id: UUID
    session_id: UUID | None
    event_type: ReminderEventType
    message: str
    provider_response_id: str | None
    created_at: datetime


class ReminderEventsResponse(BaseModel):
    events: list[ReminderEventResponse]
