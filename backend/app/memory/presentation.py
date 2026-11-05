from __future__ import annotations

from typing import Any

from app.memory.models import (
    ExplicitMemoryWriteResult,
    MemoryEvidence,
    MemoryEvidenceDetail,
    MemoryExplanation,
    MemoryItem,
    MemoryLink,
)
from app.memory.retrieval.models import MemoryRetrievalResponse
from app.perception.models import OwnerWorkingMemoryState, SessionWorkingMemoryState


def memory_item_payload(item: MemoryItem) -> dict[str, Any]:
    return {
        "id": str(item.id),
        "session_id": str(item.session_id) if item.session_id else None,
        "memory_type": item.memory_type,
        "status": item.status,
        "title": item.title,
        "content": item.content,
        "summary": item.summary,
        "source_experience_event_id": (
            str(item.source_experience_event_id)
            if item.source_experience_event_id
            else None
        ),
        "occurred_at": item.occurred_at.isoformat() if item.occurred_at else None,
        "valid_from": item.valid_from.isoformat() if item.valid_from else None,
        "valid_to": item.valid_to.isoformat() if item.valid_to else None,
        "importance": item.importance,
        "confidence": item.confidence,
        "source": item.source,
        "is_explicit": item.is_explicit,
        "superseded_by_memory_id": (
            str(item.superseded_by_memory_id)
            if item.superseded_by_memory_id
            else None
        ),
        "created_at": item.created_at.isoformat(),
        "updated_at": item.updated_at.isoformat(),
    }


def evidence_payload(item: MemoryEvidenceDetail) -> dict[str, Any]:
    evidence = item.evidence
    return {
        "id": str(evidence.id),
        "memory_item_id": str(evidence.memory_item_id),
        "evidence_type": evidence.evidence_type,
        "evidence_ref_id": evidence.evidence_ref_id,
        "evidence_time": evidence.evidence_time.isoformat(),
        "excerpt": evidence.excerpt,
        "confidence": evidence.confidence,
        "source_available": item.source_available,
        "retained_image_available": item.retained_image_available,
        "created_at": evidence.created_at.isoformat(),
    }


def link_payload(item: MemoryLink) -> dict[str, Any]:
    return {
        "id": str(item.id),
        "source_memory_id": str(item.source_memory_id),
        "target_memory_id": str(item.target_memory_id),
        "relationship": item.relationship,
        "confidence": item.confidence,
        "valid_from": item.valid_from.isoformat(),
        "valid_to": item.valid_to.isoformat() if item.valid_to else None,
    }


def explanation_payload(value: MemoryExplanation) -> dict[str, Any]:
    uncertainty: list[str] = []
    if not value.evidence:
        uncertainty.append("No supporting evidence is currently available.")
    unavailable_sources = sum(
        1 for item in value.evidence if not item.source_available
    )
    if unavailable_sources:
        uncertainty.append(
            f"{unavailable_sources} supporting evidence source(s) are no longer available."
        )
    unavailable_images = sum(
        1
        for item in value.evidence
        if item.retained_image_available is False
    )
    if unavailable_images:
        uncertainty.append(
            f"{unavailable_images} supporting frame image(s) have expired or are unavailable."
        )
    return {
        "memory": memory_item_payload(value.memory),
        "evidence": [evidence_payload(item) for item in value.evidence],
        "links": [link_payload(item) for item in value.links],
        "related_memories": [
            {
                "id": str(item.id),
                "memory_type": item.memory_type,
                "status": item.status,
                "title": item.title,
                "summary": item.summary or item.content[:500],
            }
            for item in value.related_memories
        ],
        "uncertainty": " ".join(uncertainty) if uncertainty else None,
    }


def explicit_write_payload(value: ExplicitMemoryWriteResult) -> dict[str, Any]:
    return {
        "memory": memory_item_payload(value.memory),
        "evidence_count": len(value.evidence),
        "indexing_status": value.indexing_status,
        "replayed": value.replayed,
        "warnings": list(value.warnings),
    }


def retrieval_payload(value: MemoryRetrievalResponse) -> dict[str, Any]:
    reinspection = value.evidence_reinspection
    return {
        "retrieval_run_id": str(value.retrieval_run_id),
        "candidate_count": value.candidate_count,
        "provider_spaces": list(value.provider_spaces),
        "results": [
            {
                "id": str(item.id),
                "memory_type": item.record_type,
                "session_id": str(item.session_id) if item.session_id else None,
                "occurred_at": item.occurred_at.isoformat(),
                "valid_from": item.valid_from.isoformat() if item.valid_from else None,
                "valid_to": item.valid_to.isoformat() if item.valid_to else None,
                "title": item.title,
                "summary": item.summary,
                "confidence": item.confidence,
                "importance": item.importance,
                "status": item.status,
                "is_historical": item.is_historical,
                "temporal_status": item.temporal_status,
                "evidence_available": item.evidence_available,
                "keyframe_count": item.keyframe_count,
                "match_explanation": list(item.match_explanation),
                "evidence_metadata": dict(item.evidence_metadata),
            }
            for item in value.results
        ],
        "evidence_reinspection": (
            {
                "status": reinspection.status,
                "verified_details": reinspection.verified_details,
                "uncertainty": reinspection.uncertainty,
                "confidence": reinspection.confidence,
                "pixel_evidence_available": reinspection.pixel_evidence_available,
                "supporting_frame_ids": [
                    str(item) for item in reinspection.supporting_frame_ids
                ],
                "fallback_reason": reinspection.fallback_reason,
            }
            if reinspection is not None
            else None
        ),
    }


def working_state_payload(
    session_state: SessionWorkingMemoryState | None,
    owner_state: OwnerWorkingMemoryState | None,
) -> dict[str, Any]:
    return {
        "session": (
            {
                "session_id": str(session_state.session_id),
                "revision": session_state.revision,
                "state": dict(session_state.state),
                "expires_at": session_state.expires_at.isoformat(),
                "updated_at": session_state.updated_at.isoformat(),
            }
            if session_state is not None
            else None
        ),
        "owner": (
            {
                "revision": owner_state.revision,
                "state": dict(owner_state.state),
                "expires_at": (
                    owner_state.expires_at.isoformat()
                    if owner_state.expires_at
                    else None
                ),
                "updated_at": owner_state.updated_at.isoformat(),
            }
            if owner_state is not None
            else None
        ),
    }
