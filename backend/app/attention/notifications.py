"""Source-backed notification text; no second reasoning or delivery loop."""
from dataclasses import dataclass
from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_serializer

from app.attention.models import AttentionError
from app.tasks.models import digest, json_object


class NotificationContent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    format: Literal["jarvis.attention.notification.v1"] = "jarvis.attention.notification.v1"
    source_namespace: Literal["reminder_event"]
    source_id: UUID
    reminder_id: UUID
    authored_by: Literal["stored_reminder"] = "stored_reminder"
    title: str
    details: str
    message: str = Field(min_length=1)


class PerceptionFocusCondition(BaseModel):
    """Backend-bound applicability, not a model claim or settlement authority."""
    model_config = ConfigDict(extra="forbid", frozen=True)
    subject_id: UUID
    generation: int = Field(gt=0, le=9223372036854775807, strict=True)


class VisualNotificationContent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    format: Literal["jarvis.attention.visual_notification.v1"] = "jarvis.attention.visual_notification.v1"
    source_namespace: Literal["visual_alert", "prospective_event"]
    source_id: UUID
    session_id: UUID
    observation_payload_id: UUID
    authored_by: Literal["visual_analyzer"] = "visual_analyzer"
    message: str = Field(min_length=1)
    supporting_frame_ids: tuple[UUID, ...] = Field(min_length=1, max_length=64)
    source_captured_at: datetime
    source_received_at: datetime | None
    expires_at: datetime
    subject_type: Literal["visual_focus", "conditional_reminder"] | None = None
    subject_id: UUID | None = None
    subject_generation: int | None = None
    claim_token: UUID | None = None
    perception_focus: PerceptionFocusCondition | None = None

    @model_serializer(mode="wrap")
    def serialize(self, handler):
        content = handler(self)
        # Keep legacy payloads and their speech-plan digests exactly unchanged.
        if self.perception_focus is None:
            content.pop("perception_focus", None)
        return content


class QuietSummaryMember(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    intent_id: UUID
    generation: int = Field(gt=0)
    source_namespace: Literal["task_result", "reminder_event", "visual_alert", "prospective_event"]
    source_id: UUID
    source_revision: int = Field(gt=0)
    source_payload_id: UUID
    task_revision: int | None


class QuietSummaryContent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    format: Literal["jarvis.attention.quiet_summary.v1"] = "jarvis.attention.quiet_summary.v1"
    source_namespace: Literal["quiet_resume_summary"] = "quiet_resume_summary"
    source_id: UUID
    authored_by: Literal["attention_ledger"] = "attention_ledger"
    cutoff: datetime
    members: tuple[QuietSummaryMember, ...] = Field(max_length=3)
    message: str = Field(max_length=400)


def speech_attribution(namespace):
    if namespace == "quiet_resume_summary":
        return "notification_tts", "attention_ledger", "Source-checked quiet summary TTS output"
    if namespace == "task_result":
        return "expert_tts", "frontier_expert", "Expert-authored TTS output"
    if namespace == "reminder_event":
        return "notification_tts", "stored_reminder", "Saved-reminder TTS output"
    if namespace in {"visual_alert", "prospective_event"}:
        return "notification_tts", "visual_analyzer", "Observation-backed visual-notification TTS output"
    raise AttentionError("attention_source_unsupported", "This source has no speech adapter.")


def selected_notification_repository(manager):
    if getattr(manager, "attention_notifications_enabled", False) is not True:
        return None
    repository = getattr(manager, "attention_repository", None)
    if repository is None:
        raise AttentionError("attention_unavailable", "The selected notification inbox is unavailable.")
    return repository


def reminder_text(reminder):
    """Read the saved instruction, not a generated claim about its completion."""
    parts = [reminder["title"].strip(), reminder["details"].strip()]
    return "Reminder: " + ". ".join(part for part in parts if part)


def speech_chunks(text, *, byte_limit=1600, max_chunks=64):
    """Lossless UTF-8 transport splitting, never a semantic rewrite/truncation."""
    if not text.strip() or "\x00" in text or len(text.encode("utf-8")) > byte_limit * max_chunks:
        return ()
    chunks = []
    while text:
        prefix = text.encode("utf-8")[:byte_limit].decode("utf-8", errors="ignore")
        if len(prefix) < len(text):
            boundary = max((i + 1 for i, value in enumerate(prefix) if value.isspace()), default=0)
            if boundary:
                prefix = prefix[:boundary]
        if not prefix or len(chunks) == max_chunks:
            return ()
        chunks.append(prefix)
        text = text[len(prefix):]
    return tuple(chunks)


@dataclass(frozen=True)
class NotificationOutput:
    content: NotificationContent | VisualNotificationContent | QuietSummaryContent

    @property
    def speech_texts(self):
        return speech_chunks(self.content.message)

    @property
    def speech_ready(self):
        return bool(self.speech_texts)

    @property
    def result_digest(self):
        return digest(self.content.model_dump(mode="json"))


async def reminder_source(connection, row, *, lock=False):
    payload = await connection.fetchrow(
        """select content,availability from jarvis_context_payloads where owner_id=$1 and id=$2"""
        + (" for share" if lock else ""), row["owner_id"], row["source_payload_id"])
    if payload is None or payload["availability"] != "available" or payload["content"] is None:
        return None, "source_payload_unavailable"
    try:
        content = NotificationContent.model_validate(json_object(payload["content"]))
    except (ValidationError, ValueError, TypeError):
        return None, "notification_source_invalid"
    # No row locks here: mutations invalidate the payload in their transaction.
    # Delivery holds that payload fence, avoiding payload -> reminder row inversion.
    valid = await connection.fetchval(
        """select exists(select 1 from jarvis_reminder_attention_sources b
           join jarvis_reminder_events e on e.owner_id=b.owner_id and e.id=b.event_id
           join jarvis_reminders r on r.owner_id=b.owner_id and r.id=b.reminder_id
           where b.owner_id=$1 and b.event_id=$2 and b.payload_id=$3 and b.reminder_id=$4
             and r.status<>'deleted' and r.title=$5 and r.details=$6 and e.message=$7)""",
        row["owner_id"], row["source_id"], row["source_payload_id"], content.reminder_id,
        content.title, content.details, content.message)
    if (not valid or row["source_namespace"] != content.source_namespace
            or row["source_id"] != content.source_id or row["source_revision"] != 1
            or content.message != reminder_text({"title":content.title, "details":content.details})):
        return None, "notification_source_changed"
    return NotificationOutput(content), None


async def enqueue_reminder(tasks, connection, *, owner_id, reminder, event):
    """Called inside finalization's transaction; failure rolls back both records."""
    content = NotificationContent(source_namespace="reminder_event", source_id=event["id"],
        reminder_id=reminder["id"], title=reminder["title"], details=reminder["details"],
        message=event["message"])
    if content.message != reminder_text(reminder):
        raise AttentionError("notification_source_mismatch", "Use the exact saved reminder instruction.")
    payload = await tasks.payload(connection, owner_id, content.model_dump(mode="json"))
    await connection.execute(
        """insert into jarvis_reminder_attention_sources(owner_id,reminder_id,event_id,payload_id)
           values($1,$2,$3,$4)""", owner_id, reminder["id"], event["id"], payload)
    return await connection.fetchval(
        """insert into jarvis_attention_intents(owner_id,source_namespace,source_id,source_revision,
           source_payload_id,preferred_session_id,privacy)
           values($1,'reminder_event',$2,1,$3,$4,'sensitive') returning id""",
        owner_id, event["id"], payload, event["session_id"])
