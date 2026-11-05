from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
import hashlib
import json
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.errors import JarvisHTTPError
from app.memory.jobs.models import normalize_job_json_object
from app.runtime.feature_config import TaskLimits


class TaskStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    WAITING = "waiting_for_input"
    PAUSED = "paused"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCEL_REQUESTED = "cancel_requested"
    CANCELLED = "cancelled"
    UNKNOWN = "outcome_unknown"


TRANSITIONS = {
    TaskStatus.QUEUED: {TaskStatus.RUNNING, TaskStatus.PAUSED, TaskStatus.CANCELLED},
    TaskStatus.RUNNING: {TaskStatus.QUEUED, TaskStatus.WAITING, TaskStatus.PAUSED, TaskStatus.SUCCEEDED,
                        TaskStatus.FAILED, TaskStatus.CANCEL_REQUESTED, TaskStatus.CANCELLED, TaskStatus.UNKNOWN},
    TaskStatus.WAITING: {TaskStatus.QUEUED, TaskStatus.PAUSED, TaskStatus.CANCELLED},
    TaskStatus.PAUSED: {TaskStatus.QUEUED, TaskStatus.CANCELLED, TaskStatus.CANCEL_REQUESTED, TaskStatus.UNKNOWN},
    TaskStatus.SUCCEEDED: {TaskStatus.QUEUED},
    TaskStatus.FAILED: {TaskStatus.QUEUED, TaskStatus.CANCELLED},
    TaskStatus.CANCEL_REQUESTED: {TaskStatus.CANCELLED, TaskStatus.UNKNOWN},
    TaskStatus.CANCELLED: {TaskStatus.QUEUED},
    TaskStatus.UNKNOWN: {TaskStatus.QUEUED, TaskStatus.RUNNING, TaskStatus.WAITING, TaskStatus.CANCEL_REQUESTED, TaskStatus.CANCELLED,
                         TaskStatus.SUCCEEDED, TaskStatus.FAILED, TaskStatus.PAUSED},
}


class TaskError(JarvisHTTPError):
    def __init__(self, code: str, message: str, status: int = 409):
        super().__init__(status, code, message)


class ClaimLost(TaskError):
    def __init__(self):
        super().__init__("task_claim_lost", "This task attempt no longer owns the current work.")


class BudgetExceeded(TaskError):
    def __init__(self, code: str = "task_budget_reached"):
        super().__init__(code, "The configured work budget has been reached.", 429)


def require_transition(old: TaskStatus, new: TaskStatus) -> None:
    if old != new and new not in TRANSITIONS[old]:
        raise TaskError("task_state_conflict", f"Cannot transition a task from {old} to {new}.")


def json_object(value: Any) -> dict:
    if isinstance(value, str):
        value = json.loads(value)
    return normalize_job_json_object(value, label="background task payload")


def json_text(value: dict) -> str:
    return json.dumps(json_object(value), sort_keys=True, ensure_ascii=True, separators=(",", ":"))


def digest(value: dict) -> str:
    return hashlib.sha256(json_text(value).encode()).hexdigest()


class TaskCreate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    request_id: UUID
    title: str = Field(min_length=1, max_length=200)
    requested_outcome: str = Field(min_length=1, max_length=32_000)
    session_id: UUID | None = None
    knowledge_area_id: UUID | None = None
    backend_kind: Literal["frontier", "sandbox"] = "frontier"
    execution_environment: Literal["none", "offline_sandbox", "networked_sandbox",
                                   "desktop_sandbox", "networked_desktop_sandbox"] = "none"
    publish_artifacts: bool = Field(default=False, strict=True)
    edit_memory: bool = Field(default=False, strict=True)
    grant_ids: tuple[UUID, ...] = Field(default=(), max_length=32)

    @model_validator(mode="after")
    def artifact_scope(self):
        if self.edit_memory and self.session_id is None:
            raise ValueError("Memory editing requires an owned conversation session")
        if self.publish_artifacts and (self.execution_environment == "none" or self.session_id is None):
            raise ValueError("Generated-file publication requires an admitted sandbox and an owned session")
        return self

    @field_validator("title", "requested_outcome")
    @classmethod
    def nonblank(cls, value):
        if not value.strip() or "\x00" in value:
            raise ValueError("task text must be nonblank and contain no NUL")
        return value


class TaskMutation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    request_id: UUID
    expected_revision: int = Field(ge=1)
    message: str = Field(default="", max_length=32_000)

    @field_validator("message")
    @classmethod
    def validate_message(cls, value):
        if "\x00" in value:
            raise ValueError("task messages must contain no NUL")
        return value


@dataclass(frozen=True, slots=True)
class TaskVoiceSource:
    """An exact finalized utterance from trusted tool invocation context."""
    event_id: UUID
    revision: int
    session_id: UUID

    def __post_init__(self):
        if (not isinstance(self.event_id, UUID) or not isinstance(self.session_id, UUID)
                or type(self.revision) is not int or self.revision < 1):
            raise ValueError("invalid voice request source")

    def parameters(self):
        return {"event_id": str(self.event_id), "revision": self.revision,
                "session_id": str(self.session_id)}


class TaskView(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)
    id: UUID
    owner_id: str
    session_id: UUID | None
    knowledge_area_id: UUID | None
    backend_kind: str
    model: str
    status: TaskStatus
    revision: int
    context_watermark: int
    title: str | None = None
    mode: Literal["consult", "background"] | None = None
    requested_outcome: str | None = None
    description_payload_id: UUID
    request_event_id: UUID
    request_revision: int
    current_attempt_id: UUID | None
    fence: int
    attempt_count: int
    active_seconds: float
    budget: TaskLimits
    result_payload_id: UUID | None
    reason: str | None
    pending_status: str | None
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class TaskClaim:
    task: TaskView
    attempt_id: UUID
    task_revision: int
    fence: int
    lease_until: datetime
    provider_state: str
    provider_handle: str | None
    checkpoint_payload_id: UUID | None
    recovering: bool = False


@dataclass(frozen=True, slots=True)
class WorkerOutcome:
    status: TaskStatus
    payload: dict
    reason: str | None = None

    def __post_init__(self):
        if self.status not in {TaskStatus.SUCCEEDED, TaskStatus.FAILED, TaskStatus.WAITING, TaskStatus.PAUSED}:
            raise ValueError("worker must report a result, failure, clarification, or pause")
        json_object(self.payload)


@dataclass(frozen=True, slots=True)
class Reconciliation:
    state: Literal["running", "not_started", "terminal", "unknown"]
    outcome: WorkerOutcome | None = None

    def __post_init__(self):
        if self.state not in {"running", "not_started", "terminal", "unknown"}:
            raise ValueError("unknown provider reconciliation state")
        if (self.state == "terminal") != (self.outcome is not None):
            raise ValueError("only a proved terminal provider outcome carries a result")
