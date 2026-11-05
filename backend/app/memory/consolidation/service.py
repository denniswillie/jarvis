from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from typing import Any, Protocol
from uuid import UUID

from app.experience.models import ExperienceEvent, ExperienceEventEvidence
from app.experience.repository import ExperienceEventMissingError, ExperienceRepository
from app.memory.consolidation.contracts import (
    ConsolidationEvidenceInput,
    ConsolidationKeyframeInput,
    ConsolidationRequest,
    ConsolidationResult,
    ConsolidationValidationError,
    parse_consolidation_result,
)
from app.memory.consolidation.evidence import event_state_for_evidence, is_citable_source, source_content_values
from app.memory.consolidation.openai_client import OpenAIStructuredOutputClient
from app.memory.consolidation.prompts import build_consolidation_repair_feedback
from app.perception.models import (
    OwnerWorkingMemoryState,
    VisualObservation,
    serialize_entities,
    serialize_keyframe_candidates,
    serialize_state_changes,
    serialize_visible_text,
)
from app.perception.repository import PerceptionRepository
from app.reminders.models import JarvisReminder, ReminderStatus


CONTEXT_MESSAGE_MARGIN_SECONDS = 5
MAX_ACTIVE_REMINDERS = 20
MAX_SEMANTIC_CONTEXT_FACTS = 40


class ConsolidationImageLoader(Protocol):
    async def download_object(self, *, bucket: str, path: str) -> bytes: ...


class OwnerWorkingMemoryReader(Protocol):
    async def get_owner_working_memory(
        self,
        *,
        owner_id: str,
    ) -> OwnerWorkingMemoryState | None: ...


class ActiveReminderReader(Protocol):
    async def list_reminders(
        self,
        *,
        owner_id: str,
        status: ReminderStatus | None,
        limit: int,
    ) -> list[JarvisReminder]: ...


class SemanticFactContextReader(Protocol):
    async def list_semantic_fact_context(
        self, *, owner_id: str, at: datetime, limit: int, query_text: str = "",
    ) -> list[Mapping[str, Any]]: ...


def _semantic_context_query(evidence: Sequence[ConsolidationEvidenceInput]) -> str:
    """Rank prior identities using recent citable content, not diagnostic metadata."""
    values = (
        value for source in reversed(evidence)
        if is_citable_source(source)
        and (source.evidence_type != "message" or source.payload.get("role") == "user")
        for _, value in source_content_values(source) if isinstance(value, str)
    )
    return "\n".join(values).encode("utf-8")[:8192].decode("utf-8", errors="ignore")


@dataclass(frozen=True, slots=True)
class PreparedConsolidation:
    event_id: UUID
    owner_id: str
    session_id: UUID
    result: ConsolidationResult
    provider: str
    model: str


class ExperienceConsolidationService:
    def __init__(
        self,
        *,
        experience_repository: ExperienceRepository,
        perception_repository: PerceptionRepository,
        client: OpenAIStructuredOutputClient,
        image_loader: ConsolidationImageLoader | None,
        owner_working_memory_reader: OwnerWorkingMemoryReader | None,
        active_reminder_reader: ActiveReminderReader | None,
        max_keyframes: int,
        max_transcripts: int,
        semantic_context_reader: SemanticFactContextReader | None = None,
    ) -> None:
        if max_keyframes < 1:
            raise ValueError("max_keyframes must be positive")
        if max_transcripts < 1:
            raise ValueError("max_transcripts must be positive")
        self._experience_repository = experience_repository
        self._perception_repository = perception_repository
        self._client = client
        self._image_loader = image_loader
        self._owner_working_memory_reader = owner_working_memory_reader
        self._active_reminder_reader = active_reminder_reader
        self._max_keyframes = max_keyframes
        self._max_transcripts = max_transcripts
        self._semantic_context_reader = semantic_context_reader

    @property
    def provider(self) -> str:
        return self._client.provider

    @property
    def model(self) -> str:
        return self._client.model

    async def close(self) -> None:
        await self._client.close()

    async def consolidate(
        self,
        *,
        owner_id: str,
        session_id: UUID,
        event_id: UUID,
    ) -> PreparedConsolidation:
        request = await self.build_request(
            owner_id=owner_id,
            session_id=session_id,
            event_id=event_id,
        )
        feedback = None
        for model_attempt in (1, 2):
            if feedback is None:
                raw = await self._client.generate(request, owner_id=owner_id)
            else:
                raw = await self._client.generate(
                    request, owner_id=owner_id, validation_feedback=feedback,
                )
            try:
                result = parse_consolidation_result(raw, request=request)
                break
            except ConsolidationValidationError as error:
                error.model_attempt = model_attempt
                if model_attempt == 2:
                    raise
                feedback = build_consolidation_repair_feedback(error, request)
        return PreparedConsolidation(
            event_id=event_id,
            owner_id=owner_id,
            session_id=session_id,
            result=result,
            provider=self.provider,
            model=self.model,
        )

    async def build_request(
        self,
        *,
        owner_id: str,
        session_id: UUID,
        event_id: UUID,
    ) -> ConsolidationRequest:
        event = await self._experience_repository.get_event(
            owner_id=owner_id,
            session_id=session_id,
            event_id=event_id,
        )
        if event is None or (
            event.id != event_id or event.owner_id != owner_id or event.session_id != session_id
        ):
            raise ExperienceEventMissingError()
        if event.status not in {"closed", "consolidating", "consolidated"}:
            raise ValueError("event is not ready for consolidation")
        if event.ended_at is None:
            raise ValueError("finalized event is missing ended_at")

        evidence = await self._experience_repository.list_event_evidence(
            owner_id=owner_id,
            session_id=session_id,
            event_id=event_id,
        )
        observation_ids = tuple(
            item.evidence_ref_id
            for item in evidence
            if item.evidence_type == "observation"
        )
        observations = await self._perception_repository.list_observations_by_ids(
            owner_id=owner_id,
            session_id=session_id,
            observation_ids=observation_ids,
        )
        observation_by_id = {item.id: item for item in observations}
        source_payloads = await self._experience_repository.list_nonvisual_source_payloads(
            owner_id=owner_id,
            session_id=session_id,
            event_id=event_id,
        )
        attached = _build_attached_evidence(
            evidence,
            event=event,
            observation_by_id=observation_by_id,
            source_payloads=source_payloads,
            max_transcripts=self._max_transcripts,
        )
        attached_message_ids = {
            item.evidence_ref_id
            for item in attached
            if item.evidence_type == "message"
        }
        attached_transcript_count = sum(
            item.evidence_type in {"message", "tool_execution"} for item in attached
        )
        remaining_context_count = self._max_transcripts - attached_transcript_count
        if remaining_context_count > 0:
            context_payloads = await self._experience_repository.list_context_messages(
                owner_id=owner_id,
                session_id=session_id,
                started_at=event.started_at
                - timedelta(seconds=CONTEXT_MESSAGE_MARGIN_SECONDS),
                ended_at=event.ended_at
                + timedelta(seconds=CONTEXT_MESSAGE_MARGIN_SECONDS),
                limit=remaining_context_count + len(attached_message_ids),
            )
            context_messages = tuple(
                ConsolidationEvidenceInput(
                    evidence_id=None,
                    evidence_type="message",
                    evidence_ref_id=item["id"],
                    occurred_at=item["created_at"],
                    source_sequence=None,
                    payload=item["payload"],
                    citation_allowed=False,
                )
                for item in context_payloads
                if item["id"] not in attached_message_ids
            )[:remaining_context_count]
        else:
            context_messages = ()
        keyframe_rows = await self._experience_repository.list_event_keyframe_sources(
            owner_id=owner_id,
            session_id=session_id,
            event_id=event_id,
            limit=self._max_keyframes,
        )
        keyframes = tuple(
            [await self._keyframe_input(row) for row in keyframe_rows]
        )
        owner_working_memory = (
            await self._owner_working_memory_reader.get_owner_working_memory(
                owner_id=owner_id
            )
            if self._owner_working_memory_reader is not None
            else None
        )
        active_reminders = (
            await self._active_reminder_reader.list_reminders(
                owner_id=owner_id,
                status="active",
                limit=MAX_ACTIVE_REMINDERS,
            )
            if self._active_reminder_reader is not None
            else []
        )
        # Keep persisted search metadata intact; only the model's evidence view changes.
        evidence_event = replace(event, state=event_state_for_evidence(event.state))
        prior_facts = (
            await self._semantic_context_reader.list_semantic_fact_context(
                owner_id=owner_id, at=event.ended_at, limit=MAX_SEMANTIC_CONTEXT_FACTS + 1,
                query_text=_semantic_context_query(attached),
            )
            if self._semantic_context_reader is not None else []
        )
        return ConsolidationRequest(
            event=evidence_event,
            evidence=attached,
            context_messages=context_messages,
            keyframes=keyframes,
            event_end_working_state=dict(evidence_event.state),
            owner_working_memory=(
                dict(owner_working_memory.state)
                if owner_working_memory is not None
                else {}
            ),
            active_reminders=tuple(
                _active_reminder_payload(reminder) for reminder in active_reminders
            ),
            semantic_fact_context=tuple(prior_facts[:MAX_SEMANTIC_CONTEXT_FACTS]),
            semantic_fact_context_truncated=len(prior_facts) > MAX_SEMANTIC_CONTEXT_FACTS,
        )

    async def _keyframe_input(
        self,
        row: Mapping[str, Any],
    ) -> ConsolidationKeyframeInput:
        image_bytes = None
        if self._image_loader is not None and row["raw_image_status"] == "available":
            image_bytes = await self._image_loader.download_object(
                bucket=row["storage_bucket"],
                path=row["storage_path"],
            )
            if not _is_jpeg(image_bytes):
                raise ValueError("consolidation keyframe is not a valid JPEG")
        return ConsolidationKeyframeInput(
            visual_frame_id=row["visual_frame_id"],
            rank=row["rank"],
            reason=row["reason"],
            quality_score=float(row["quality_score"]),
            captured_at=row["captured_at"],
            frame_description=row["frame_description"],
            storage_bucket=row["storage_bucket"],
            storage_path=row["storage_path"],
            content_type=row["content_type"],
            image_bytes=image_bytes,
        )


def _build_attached_evidence(
    evidence: Sequence[ExperienceEventEvidence],
    *,
    event: ExperienceEvent,
    observation_by_id: Mapping[UUID, VisualObservation],
    source_payloads: Mapping[tuple[str, UUID], Mapping[str, Any]],
    max_transcripts: int,
) -> tuple[ConsolidationEvidenceInput, ...]:
    items: list[ConsolidationEvidenceInput] = []
    transcript_count = 0
    for item in evidence:
        if item.event_id != event.id:
            raise ValueError("attached evidence belongs to another event")
        if item.evidence_type == "observation":
            observation = observation_by_id.get(item.evidence_ref_id)
            if observation is None:
                raise ValueError("attached observation evidence is missing")
            if observation.owner_id != event.owner_id or observation.session_id != event.session_id:
                raise ValueError("attached observation belongs to another owner or session")
            payload = _observation_payload(observation)
        else:
            if transcript_count >= max_transcripts:
                continue
            payload = source_payloads.get((item.evidence_type, item.evidence_ref_id))
            if payload is None:
                raise ValueError("attached transcript/tool evidence is missing")
            transcript_count += 1
        items.append(
            ConsolidationEvidenceInput(
                evidence_id=item.id,
                evidence_type=item.evidence_type,
                evidence_ref_id=item.evidence_ref_id,
                occurred_at=item.occurred_at,
                source_sequence=item.source_sequence,
                payload=payload,
            )
        )
    return tuple(items)


def _observation_payload(observation: VisualObservation) -> dict[str, Any]:
    return {
        "summary": observation.summary,
        "scene": observation.scene,
        "activity": observation.activity,
        "entities": serialize_entities(observation.entities),
        "visible_text": serialize_visible_text(observation.visible_text),
        "state_changes": serialize_state_changes(observation.state_changes),
        "keyframe_candidates": serialize_keyframe_candidates(
            observation.keyframe_candidates
        ),
        "confidence": observation.confidence,
        "compacted": observation.is_compacted,
    }


def _active_reminder_payload(reminder: JarvisReminder) -> dict[str, Any]:
    return {
        "id": str(reminder.id),
        "title": reminder.title,
        "details": reminder.details,
        "remind_at": reminder.remind_at.isoformat() if reminder.remind_at else None,
        "condition": reminder.condition,
        "updated_at": reminder.updated_at.isoformat(),
    }


def _is_jpeg(value: bytes) -> bool:
    return len(value) >= 4 and value[:3] == b"\xff\xd8\xff" and value[-2:] == b"\xff\xd9"
