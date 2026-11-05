from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Mapping
from uuid import UUID

from app.experience.models import (
    ExperienceEvent,
    ExperienceEventKind,
    ExperienceEvidenceType,
    ExperienceModality,
    ToolExecution,
    UnsafeToolPayloadError,
    normalize_durable_tool_payload,
)
from app.perception.models import VisualObservation
from app.sessions.models import JarvisMessage


_MODALITY_ORDER: tuple[ExperienceModality, ...] = (
    "audio",
    "visual",
    "text",
    "tool",
)
_MAX_EVENT_TAGS = 32
_MAX_RECENT_SUMMARIES = 8
_MAX_SUMMARY_LENGTH = 1_000


@dataclass(frozen=True)
class EventEvidenceDescriptor:
    evidence_type: ExperienceEvidenceType
    evidence_ref_id: UUID
    occurred_at: datetime
    modality: ExperienceModality
    summary: str
    scene: str | None = None
    activity: str | None = None
    tags: tuple[str, ...] = ()
    confidence: float = 1.0
    importance: float = 0.0
    starts_new_event: bool = False
    boundary_reason: str | None = None
    source_sequence: int | None = None
    metadata: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class EventAggregate:
    event_kind: ExperienceEventKind
    modalities: tuple[ExperienceModality, ...]
    title: str
    summary: str
    scene: str | None
    activity: str | None
    state: Mapping[str, Any]
    importance: float
    confidence: float


def evidence_from_observation(observation: VisualObservation) -> EventEvidenceDescriptor:
    return EventEvidenceDescriptor(
        evidence_type="observation",
        evidence_ref_id=observation.id,
        occurred_at=observation.ended_at,
        modality="visual",
        summary=observation.summary,
        scene=observation.scene,
        activity=observation.activity,
        tags=observation.memory_tags,
        confidence=observation.confidence,
        importance=(
            observation.confidence
            if observation.interrupt_candidate is not None
            else 0.7 if observation.starts_new_event else 0.35
        ),
        starts_new_event=observation.starts_new_event,
        boundary_reason=observation.boundary_reason,
        metadata={
            "source": observation.source,
            "primary_visual_frame_id": (
                str(observation.primary_visual_frame_id)
                if observation.primary_visual_frame_id is not None
                else None
            ),
        },
    )


def evidence_from_message(message: JarvisMessage) -> EventEvidenceDescriptor:
    modality: ExperienceModality = "audio" if message.modality == "audio" else "text"
    metadata = {"role": message.role, "source_modality": message.modality}
    summary = message.content
    if message.metadata.get("audio_provider") == "gemini":
        metadata.update({key: message.metadata.get(key) for key in (
            "audio_provider", "completion_state", "delivery_status", "playback_status",
        )})
        if message.role == "assistant":
            summary = "Assistant-generated response; playback not confirmed: " + summary
    return EventEvidenceDescriptor(
        evidence_type="message",
        evidence_ref_id=message.id,
        occurred_at=message.created_at,
        modality=modality,
        summary=_bounded_text(summary),
        confidence=1.0,
        importance=0.3,
        metadata=metadata,
    )


def evidence_from_tool_execution(execution: ToolExecution) -> EventEvidenceDescriptor:
    occurred_at = execution.completed_at or execution.started_at
    return EventEvidenceDescriptor(
        evidence_type="tool_execution",
        evidence_ref_id=execution.id,
        occurred_at=occurred_at,
        modality="tool",
        summary=f"{execution.tool_name}: {execution.status}",
        tags=(execution.tool_name,),
        confidence=1.0,
        importance=0.5,
        metadata={
            "tool_name": execution.tool_name,
            "status": execution.status,
            "mutates_state": execution.mutates_state,
        },
    )


def event_last_evidence_at(event: ExperienceEvent) -> datetime:
    value = event.state.get("last_evidence_at")
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError:
            pass
        else:
            if parsed.tzinfo is not None and parsed.utcoffset() is not None:
                return parsed
    return event.started_at


def reduce_event_aggregate(
    event: ExperienceEvent,
    evidence: EventEvidenceDescriptor,
) -> EventAggregate:
    state = dict(event.state)
    counts = _integer_mapping(state.get("evidence_counts"))
    previous_count = sum(counts.values())
    counts[evidence.evidence_type] = counts.get(evidence.evidence_type, 0) + 1

    modalities = tuple(
        modality
        for modality in _MODALITY_ORDER
        if modality in set(event.modalities) | {evidence.modality}
    )
    tags = _bounded_unique_strings([*state.get("tags", []), *evidence.tags])
    summaries = [
        *(
            item
            for item in state.get("recent_summaries", [])
            if isinstance(item, str) and item.strip()
        ),
        _bounded_text(evidence.summary),
    ][-_MAX_RECENT_SUMMARIES:]
    confidence = (
        (event.confidence * previous_count + evidence.confidence) / (previous_count + 1)
        if previous_count
        else evidence.confidence
    )
    latest = previous_count == 0 or evidence.occurred_at >= event_last_evidence_at(event)
    scene = (evidence.scene or event.scene) if latest else event.scene
    activity = (evidence.activity or event.activity) if latest else event.activity
    state.update({"evidence_counts": counts, "recent_summaries": summaries, "tags": tags})
    if latest:
        state.update({
            "last_evidence_at": evidence.occurred_at.isoformat(),
            "last_evidence_type": evidence.evidence_type,
            "last_evidence_ref_id": str(evidence.evidence_ref_id),
        })
    else:
        previous_late = state.get("late_evidence_count", 0)
        state["late_evidence_count"] = (previous_late if type(previous_late) is int else 0) + 1
    return EventAggregate(
        event_kind=event_kind_for_modalities(modalities),
        modalities=modalities,
        title=_event_title(activity=activity, scene=scene, evidence=evidence) if latest else event.title,
        summary=_bounded_text(evidence.summary) if latest else event.summary,
        scene=scene,
        activity=activity,
        state=state,
        importance=max(event.importance, evidence.importance),
        confidence=max(0.0, min(1.0, confidence)),
    )


def event_kind_for_modalities(
    modalities: tuple[ExperienceModality, ...],
) -> ExperienceEventKind:
    values = set(modalities)
    if values == {"visual"}:
        return "visual"
    if values == {"tool"}:
        return "tool_activity"
    if values and values.issubset({"audio", "text"}):
        return "interaction"
    return "mixed"


def durable_tool_payload(payload: Mapping[str, Any], *, status: str) -> dict[str, Any]:
    try:
        return normalize_durable_tool_payload(payload, label="tool outcome")
    except UnsafeToolPayloadError:
        return {"status": status, "redacted": True}


def _event_title(
    *,
    activity: str | None,
    scene: str | None,
    evidence: EventEvidenceDescriptor,
) -> str:
    if activity and activity.strip():
        return _bounded_text(activity, maximum=120)
    if scene and scene.strip():
        return _bounded_text(scene, maximum=120)
    if evidence.evidence_type == "tool_execution":
        return _bounded_text(evidence.summary, maximum=120)
    return "Conversation"


def _integer_mapping(value: object) -> dict[str, int]:
    if not isinstance(value, Mapping):
        return {}
    return {
        str(key): int(item)
        for key, item in value.items()
        if isinstance(item, int) and item >= 0
    }


def _bounded_unique_strings(values: list[object]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, str):
            continue
        normalized = value.strip()
        key = normalized.casefold()
        if not normalized or key in seen:
            continue
        seen.add(key)
        result.append(normalized)
    return result[-_MAX_EVENT_TAGS:]


def _bounded_text(value: str, *, maximum: int = _MAX_SUMMARY_LENGTH) -> str:
    normalized = " ".join(value.split())
    return normalized[:maximum]
