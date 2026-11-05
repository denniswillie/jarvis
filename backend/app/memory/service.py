from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from uuid import UUID

from app.memory.jobs.repository import MemoryJobRepository
from app.memory.entities.repository import EntityRepository
from app.memory.models import (
    ExplicitMemoryEvidenceInput,
    ExplicitMemoryKind,
    ExplicitMemoryWriteResult,
    MemoryEvidenceDetail,
    MemoryExplanation,
    MemoryItem,
    MemoryType,
)
from app.memory.repository import MemoryMissingError, MemoryRepository
from app.memory.retrieval.models import MemoryRetrievalResponse
from app.memory.retrieval.query import MemoryRetrievalQuery
from app.memory.retrieval.service import HierarchicalMemoryRetrievalService
from app.perception.frame_window import RecentVisualFrameWindow
from app.perception.models import OwnerWorkingMemoryState, SessionWorkingMemoryState
from app.perception.services import WorkingMemoryService
from app.sessions.repository import SessionRepository


_EXPLICIT_MEMORY_TYPES: dict[ExplicitMemoryKind, MemoryType] = {
    "fact": "semantic_fact",
    "experience": "reflection",
    "procedure": "procedure",
    "goal": "goal",
}


class MemoryApplicationService:
    """Owner-scoped memory use cases shared by agent tools and REST handlers."""

    def __init__(
        self,
        *,
        repository: MemoryRepository,
        retrieval_service: HierarchicalMemoryRetrievalService,
        session_repository: SessionRepository,
        frame_window: RecentVisualFrameWindow,
        working_memory_service: WorkingMemoryService | None,
        embedding_job_repository: MemoryJobRepository | None,
        current_frame_max_age_seconds: int,
        entity_repository: EntityRepository | None = None,
    ) -> None:
        if current_frame_max_age_seconds < 1:
            raise ValueError("current_frame_max_age_seconds must be positive")
        self._repository = repository
        self._retrieval_service = retrieval_service
        self._session_repository = session_repository
        self._frame_window = frame_window
        self._working_memory_service = working_memory_service
        self._embedding_job_repository = embedding_job_repository
        self._current_frame_max_age_seconds = current_frame_max_age_seconds
        self._entity_repository = entity_repository

    @property
    def retrieval_service(self) -> HierarchicalMemoryRetrievalService:
        return self._retrieval_service

    async def search(
        self,
        *,
        owner_id: str,
        request: MemoryRetrievalQuery,
        reinspect_evidence: bool,
    ) -> MemoryRetrievalResponse:
        return await self._retrieval_service.search(
            owner_id=owner_id,
            request=request,
            reinspect_evidence=reinspect_evidence,
        )

    async def remember(
        self,
        *,
        owner_id: str,
        session_id: UUID | None,
        kind: ExplicitMemoryKind,
        content: str,
        occurred_at: datetime | None,
        entity_references: tuple[str, ...],
        attach_triggering_message: bool,
        attach_current_evidence: bool,
        supersedes_memory_id: UUID | None,
        operation_id: str | None = None,
    ) -> ExplicitMemoryWriteResult:
        normalized_content = content.strip()
        if not normalized_content:
            raise ValueError("content cannot be blank")
        if len(normalized_content) > 8_000:
            raise ValueError("content must be at most 8000 characters")
        memory_type = _EXPLICIT_MEMORY_TYPES.get(kind)
        if memory_type is None:
            raise ValueError(f"unsupported explicit memory kind: {kind}")
        effective_time = occurred_at or datetime.now(UTC)
        if effective_time.tzinfo is None or effective_time.utcoffset() is None:
            raise ValueError("occurred_at must be timezone-aware")
        effective_time = effective_time.astimezone(UTC)

        evidence: list[ExplicitMemoryEvidenceInput] = []
        warnings: list[str] = []
        latest_user_message_id: UUID | None = None
        if attach_triggering_message:
            if session_id is None:
                warnings.append("triggering_message_unavailable_without_session")
            else:
                messages = await self._session_repository.get_recent_messages(
                    owner_id,
                    session_id,
                    limit=20,
                )
                latest_user_message = next(
                    (message for message in reversed(messages) if message.role == "user"),
                    None,
                )
                if latest_user_message is None:
                    warnings.append("triggering_user_message_unavailable")
                else:
                    latest_user_message_id = latest_user_message.id
                    evidence.append(
                        ExplicitMemoryEvidenceInput(
                            evidence_type="message",
                            evidence_ref_id=str(latest_user_message.id),
                            evidence_time=latest_user_message.created_at,
                            excerpt=latest_user_message.content[:500],
                            metadata={"role": latest_user_message.role},
                        )
                    )

        latest_frame_id: UUID | None = None
        if attach_current_evidence:
            if session_id is None:
                warnings.append("current_visual_evidence_unavailable_without_session")
            else:
                frame = self._frame_window.latest(
                    owner_id,
                    session_id,
                    max_age_seconds=self._current_frame_max_age_seconds,
                )
                if frame is None:
                    warnings.append("current_visual_evidence_unavailable")
                else:
                    latest_frame_id = frame.frame_id
                    evidence.append(
                        ExplicitMemoryEvidenceInput(
                            evidence_type="frame",
                            evidence_ref_id=str(frame.frame_id),
                            evidence_time=frame.captured_at,
                            excerpt="Current glasses frame attached by explicit request.",
                            metadata={"source": frame.source},
                        )
                    )

        normalized_entities = tuple(
            dict.fromkeys(
                value.strip()
                for value in entity_references
                if value.strip()
            )
        )
        if len(normalized_entities) > 20:
            raise ValueError("entity_references cannot contain more than 20 values")
        if any(len(value) > 300 for value in normalized_entities):
            raise ValueError("entity references must be at most 300 characters")
        resolved_entity_ids = await self._resolve_entity_references(
            owner_id=owner_id,
            references=normalized_entities,
            warnings=warnings,
        )
        idempotency_key = _explicit_memory_idempotency_key(
            owner_id=owner_id,
            session_id=session_id,
            kind=kind,
            content=normalized_content,
            occurred_at=occurred_at,
            entity_references=normalized_entities,
            attach_triggering_message=attach_triggering_message,
            attach_current_evidence=attach_current_evidence,
            message_id=latest_user_message_id,
            supersedes_memory_id=supersedes_memory_id,
            operation_id=operation_id,
        )
        return await self._repository.create_explicit_memory(
            owner_id=owner_id,
            session_id=session_id,
            memory_type=memory_type,
            title=_memory_title(kind, normalized_content),
            content=normalized_content,
            occurred_at=effective_time,
            idempotency_key=idempotency_key,
            evidence=evidence,
            entity_ids=resolved_entity_ids,
            metadata={
                "explicit_kind": kind,
                "entity_references": list(normalized_entities),
            },
            supersedes_memory_id=supersedes_memory_id,
            job_repository=self._embedding_job_repository,
            warnings=warnings,
        )

    async def _resolve_entity_references(
        self,
        *,
        owner_id: str,
        references: tuple[str, ...],
        warnings: list[str],
    ) -> tuple[UUID, ...]:
        if not references or self._entity_repository is None:
            if references and self._entity_repository is None:
                warnings.append("entity_resolution_not_configured")
            return ()

        resolved: list[UUID] = []
        for reference in references:
            matches = await self._entity_repository.find_by_name(
                owner_id=owner_id,
                name=reference,
                limit=2,
            )
            unique_ids = tuple(dict.fromkeys(item.id for item in matches))
            if len(unique_ids) == 1:
                if unique_ids[0] not in resolved:
                    resolved.append(unique_ids[0])
            elif not unique_ids:
                warnings.append(f"entity_reference_unresolved:{reference}")
            else:
                warnings.append(f"entity_reference_ambiguous:{reference}")
        return tuple(resolved)

    async def list_recent(
        self,
        *,
        owner_id: str,
        limit: int,
        memory_types: tuple[MemoryType, ...],
        session_id: UUID | None,
        start_time: datetime | None,
        end_time: datetime | None,
        cursor_time: datetime | None,
        cursor_id: UUID | None,
    ) -> list[MemoryItem]:
        return await self._repository.list_recent_memory_items(
            owner_id=owner_id,
            limit=limit,
            memory_types=memory_types,
            session_id=session_id,
            start_time=start_time,
            end_time=end_time,
            cursor_time=cursor_time,
            cursor_id=cursor_id,
        )

    async def evidence(
        self,
        *,
        owner_id: str,
        memory_id: UUID,
        limit: int = 100,
    ) -> tuple[MemoryEvidenceDetail, ...]:
        memory = await self._repository.get_memory_item(
            owner_id=owner_id,
            memory_item_id=memory_id,
        )
        if memory is None:
            raise MemoryMissingError()
        return tuple(
            await self._repository.list_memory_evidence_details(
                owner_id=owner_id,
                memory_item_id=memory_id,
                limit=limit,
            )
        )

    async def explain(
        self,
        *,
        owner_id: str,
        memory_id: UUID,
    ) -> MemoryExplanation:
        memory = await self._repository.get_memory_item(
            owner_id=owner_id,
            memory_item_id=memory_id,
        )
        if memory is None:
            raise MemoryMissingError()
        evidence = tuple(
            await self._repository.list_memory_evidence_details(
                owner_id=owner_id,
                memory_item_id=memory_id,
            )
        )
        links = tuple(
            await self._repository.list_memory_links(
                owner_id=owner_id,
                memory_item_id=memory_id,
            )
        )
        related_ids: list[UUID] = []
        for link in links:
            candidate = (
                link.target_memory_id
                if link.source_memory_id == memory_id
                else link.source_memory_id
            )
            if candidate not in related_ids:
                related_ids.append(candidate)
        if (
            memory.superseded_by_memory_id is not None
            and memory.superseded_by_memory_id not in related_ids
        ):
            related_ids.append(memory.superseded_by_memory_id)
        superseded = await self._repository.list_memories_superseded_by(
            owner_id=owner_id,
            replacement_memory_id=memory_id,
        )
        related: list[MemoryItem] = list(superseded)
        existing_ids = {item.id for item in related}
        for related_id in related_ids[:20]:
            if related_id in existing_ids:
                continue
            item = await self._repository.get_memory_item(
                owner_id=owner_id,
                memory_item_id=related_id,
            )
            if item is not None:
                related.append(item)
                existing_ids.add(item.id)
        return MemoryExplanation(
            memory=memory,
            evidence=evidence,
            links=links,
            related_memories=tuple(related[:20]),
        )

    async def working_state(
        self,
        *,
        owner_id: str,
        session_id: UUID | None,
    ) -> tuple[SessionWorkingMemoryState | None, OwnerWorkingMemoryState | None]:
        if self._working_memory_service is None:
            return None, None
        session_state = (
            await self._working_memory_service.get_session_working_memory(
                owner_id=owner_id,
                session_id=session_id,
            )
            if session_id is not None
            else None
        )
        owner_state = await self._working_memory_service.get_owner_working_memory(
            owner_id=owner_id
        )
        return session_state, owner_state


def _memory_title(kind: ExplicitMemoryKind, content: str) -> str:
    prefix = {
        "fact": "Remembered fact",
        "experience": "Remembered experience",
        "procedure": "Remembered procedure",
        "goal": "Remembered goal",
    }[kind]
    first_line = content.splitlines()[0].strip()
    return f"{prefix}: {first_line[:120]}" if first_line else prefix


def _explicit_memory_idempotency_key(
    *,
    owner_id: str,
    session_id: UUID | None,
    kind: ExplicitMemoryKind,
    content: str,
    occurred_at: datetime | None,
    entity_references: tuple[str, ...],
    attach_triggering_message: bool,
    attach_current_evidence: bool,
    message_id: UUID | None,
    supersedes_memory_id: UUID | None,
    operation_id: str | None,
) -> str:
    normalized_operation_id = operation_id.strip() if operation_id else None
    if normalized_operation_id:
        payload = {
            "owner_id": owner_id,
            "session_id": str(session_id) if session_id else None,
            "operation_id": normalized_operation_id,
        }
    else:
        payload = {
            "owner_id": owner_id,
            "session_id": str(session_id) if session_id else None,
            "kind": kind,
            "content": content,
            "occurred_at": occurred_at.isoformat() if occurred_at else None,
            "entity_references": list(entity_references),
            "attach_triggering_message": attach_triggering_message,
            "attach_current_evidence": attach_current_evidence,
            "message_id": str(message_id) if message_id else None,
            "supersedes_memory_id": (
                str(supersedes_memory_id) if supersedes_memory_id else None
            ),
        }
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
