"""Canonical context contracts. Only trusted ingress adapters construct inputs."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.tasks.models import json_object


ContextKind = Literal["user_request", "user_transcript", "assistant_output", "played_output_boundary",
                      "tool_outcome", "observation", "task_update", "correction", "context_gap",
                      "command_request"]
ContextTrust = Literal["user_input", "sensor", "external_content", "assistant_output", "tool_receipt", "derived"]
ContextFinality = Literal["provisional", "final", "revised", "unavailable"]


class ContextIngress(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    session_id: UUID | None
    source_namespace: str = Field(min_length=1, max_length=100)
    source_event_id: str = Field(min_length=1, max_length=512)
    receipt_id: str = Field(min_length=1, max_length=512)
    kind: ContextKind
    trust: ContextTrust
    finality: ContextFinality = "final"
    source_time: datetime | None = None
    payload: dict[str, Any]
    expected_revision: int | None = Field(default=None, ge=0)
    source_payload_ids: tuple[UUID, ...] = Field(default=(), max_length=128)

    @field_validator("source_payload_ids")
    @classmethod
    def exact_dependencies(cls, value):
        return tuple(sorted(set(value)))

    @field_validator("source_namespace", "source_event_id", "receipt_id")
    @classmethod
    def identity(cls, value):
        if not value.strip() or "\x00" in value:
            raise ValueError("context identity must be nonblank and contain no NUL")
        return value

    @field_validator("source_time")
    @classmethod
    def aware_time(cls, value):
        if value is not None and (value.tzinfo is None or value.utcoffset() is None):
            raise ValueError("context source time must be timezone-aware")
        return value

    @field_validator("payload")
    @classmethod
    def bounded_payload(cls, value):
        return json_object(value)

    @model_validator(mode="after")
    def origin_matches_kind(self):
        # This labels a server-constructed command, not user-input authority.
        # Reserve both the kind and namespace so neither can impersonate an
        # ordinary transcript, request, observation, or receipt.
        if self.kind == "command_request" or self.source_namespace == "voice_command":
            if (self.source_namespace, self.kind, self.trust) != (
                "voice_command", "command_request", "derived"
            ):
                raise ValueError("voice_command requires command_request kind and derived trust")
        required = {"user_request": "user_input", "user_transcript": "user_input",
                    "assistant_output": "assistant_output", "tool_outcome": "tool_receipt"}
        if self.kind in required and self.trust != required[self.kind]:
            raise ValueError("context kind does not match its trusted source")
        if self.kind == "user_transcript" and self.finality in {"final", "revised"}:
            text = self.payload.get("text")
            if not isinstance(text, str) or not text.strip() or "\x00" in text:
                raise ValueError("a finalized user transcript requires actual nonblank text")
        return self


class ContextRecord(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)
    id: UUID
    owner_id: str
    session_id: UUID | None
    sequence: int
    origin_sequence: int
    source_namespace: str
    source_event_id: str
    revision: int
    supersedes_event_id: UUID | None
    kind: ContextKind
    trust: ContextTrust
    finality: ContextFinality
    source_time: datetime | None
    recorded_at: datetime
    payload_id: UUID
    availability: str
    content_digest: str | None
    content: dict[str, Any] | None
    is_current: bool

    @property
    def authorizes_request(self) -> bool:
        return (self.is_current and self.kind in {"user_request", "user_transcript"}
                and self.trust == "user_input" and self.finality in {"final", "revised"}
                and self.availability == "available")

    @property
    def streaming_assistant_snapshot(self) -> bool:
        return self.kind == "assistant_output" and self.finality == "provisional"

    @property
    def dependency_usage(self) -> str:
        # A stream checkpoint is historical partial output, not a promise that
        # no further output will arrive. Deletion still revokes its payload.
        # Expert text is immutable per delivery; later revisions only advance
        # delivery/playback telemetry. The exact script snapshot remains valid
        # historical context, while its source dependency still enforces removal.
        expert_snapshot = self.kind == "assistant_output" and (self.content or {}).get("authored_by") == "frontier_expert"
        return "historical" if self.streaming_assistant_snapshot or expert_snapshot or not self.is_current else "current"

    def as_context(self) -> dict:
        result = self.model_dump(mode="json", exclude={"owner_id", "payload_id"})
        if self.kind == "assistant_output":
            # Generated, forwarded and played are separate facts. A transcript
            # does not become heard text in the absence of a linked client ACK.
            result["heard_state"] = "unknown"
            if self.streaming_assistant_snapshot:
                result["streaming_snapshot"] = "Partial output at capture time; not a completed answer or confirmed playback."
        return result


class ContextWindow(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    max_messages: int = Field(default=128, ge=1, le=512)
    max_characters: int = Field(default=64_000, ge=1024, le=160_000)
    max_pins: int = Field(default=32, ge=1, le=128)


class TaskContextManifest(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)
    id: UUID
    owner_id: str
    task_id: UUID
    task_revision: int
    context_watermark: int
    payload_id: UUID
    status: Literal["active", "superseded", "revoked"]
    created_at: datetime
    content: dict[str, Any] | None
