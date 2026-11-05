"""Durable delivery identity; no inferred human hearing or model-selected owner."""
from dataclasses import dataclass
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from app.errors import JarvisHTTPError


class AttentionError(JarvisHTTPError):
    def __init__(self, code, message, status=409):
        super().__init__(status, code, message)


class InboxMutation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    request_id: UUID
    expected_generation: int = Field(ge=1, strict=True)


@dataclass(frozen=True, slots=True)
class DeliveryClaim:
    owner_id: str
    intent_id: UUID
    attempt_id: UUID
    generation: int
    session_id: UUID
    worker_id: str
    source_payload_id: UUID
    source_namespace: str = "task_result"
    subject_type: str | None = None
    subject_id: UUID | None = None


ACTIVE_PHASES = ("claimed", "submitted", "forwarding", "awaiting_receipt")
