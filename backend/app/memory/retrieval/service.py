from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from time import monotonic

from app.memory.repository import MemoryRepository
from app.images import ImageUnavailableError
from app.memory.retrieval.image_evidence import HistoricalImageService
from app.memory.retrieval.candidates import RetrievalCandidateCollector
from app.memory.retrieval.evidence_inspector import (
    HistoricalEvidenceInspector,
    build_structured_evidence_fallback,
)
from app.memory.retrieval.fusion import reciprocal_rank_fuse
from app.memory.retrieval.models import (
    FusedRetrievalCandidate,
    MemoryRetrievalResponse,
    MemoryRetrievalResult,
)
from app.memory.retrieval.query import MemoryRetrievalQuery, normalize_memory_query
from app.perception.repository import PerceptionRepository


class HierarchicalMemoryRetrievalService:
    def __init__(
        self,
        *,
        memory_repository: MemoryRepository,
        perception_repository: PerceptionRepository,
        candidate_collector: RetrievalCandidateCollector,
        candidate_limit: int = 40,
        result_limit: int = 12,
        default_seconds: int = 600,
        max_days: int = 3650,
        evidence_inspector: HistoricalEvidenceInspector | None = None,
        image_service: HistoricalImageService | None = None,
    ) -> None:
        if candidate_limit < 1:
            raise ValueError("candidate_limit must be positive")
        if result_limit < 1 or result_limit > candidate_limit:
            raise ValueError("result_limit must be positive and not exceed candidate_limit")
        self._memory_repository = memory_repository
        self._perception_repository = perception_repository
        self._candidate_collector = candidate_collector
        self._candidate_limit = candidate_limit
        self._result_limit = result_limit
        self._default_seconds = default_seconds
        self._max_days = max_days
        self._evidence_inspector = evidence_inspector
        self._image_service = image_service

    async def search(
        self,
        *,
        owner_id: str,
        request: MemoryRetrievalQuery,
        now: datetime | None = None,
        reinspect_evidence: bool = False,
    ) -> MemoryRetrievalResponse:
        normalized_owner_id = owner_id.strip()
        if not normalized_owner_id:
            raise ValueError("owner_id cannot be blank")
        started = monotonic()
        current_time = now or datetime.now(UTC)
        query = normalize_memory_query(
            request,
            now=current_time,
            default_seconds=self._default_seconds,
            max_days=self._max_days,
        )
        collection = await self._candidate_collector.collect(
            owner_id=normalized_owner_id,
            query=query,
            limit=self._candidate_limit,
        )
        fused = reciprocal_rank_fuse(
            {
                source: [
                    hit for hit in hits
                    if hit.document.owner_id == normalized_owner_id
                    and (
                        hit.document.record_type == "experience_event"
                        or (
                            hit.document.record_status in {"active", "superseded", "retracted"}
                            and (
                                query.include_superseded
                                or hit.document.temporal_status_at(current_time) == "current"
                                or (
                                    hit.document.record_type == "episode"
                                    and hit.document.record_status == "active"
                                    and (hit.document.valid_from is None or hit.document.valid_from <= current_time)
                                )
                            )
                        )
                    )
                ]
                for source, hits in collection.rankings.items()
            },
            candidate_limit=self._candidate_limit,
            now=current_time,
        )
        selected = fused[: self._result_limit]
        evidence_reinspection = None
        if reinspect_evidence and selected:
            evidence_reinspection = (
                await self._evidence_inspector.inspect(
                    owner_id=normalized_owner_id,
                    query=query.query,
                    candidates=selected,
                )
                if self._evidence_inspector is not None
                else build_structured_evidence_fallback(
                    candidates=selected,
                    reason="pixel_reinspection_not_configured",
                )
            )
        results = tuple(
            _to_result(
                candidate,
                include_evidence_metadata=query.include_evidence_metadata,
                now=current_time,
            )
            for candidate in selected
        )
        if query.include_evidence_metadata and self._image_service is not None and results:
            preview_status = "metadata_only"
            try:
                previews = await self._image_service.image_previews(
                    owner_id=normalized_owner_id,
                    records=tuple((result.record_type, result.id) for result in results),
                )
            except ImageUnavailableError:
                # Optional pixel metadata must never take down useful text search.
                previews = {}
                preview_status = "unavailable"
            results = tuple(
                replace(result, evidence_metadata={
                    **result.evidence_metadata,
                    "image_refs": previews.get((result.record_type, result.id), []),
                    "image_refs_status": preview_status,
                    "image_refs_bounded": True,
                    "image_refs_unique_frame_limit": self._image_service.preview_limit,
                })
                for result in results
            )
        provider_spaces = list(collection.provider_spaces)
        if (
            evidence_reinspection is not None
            and evidence_reinspection.status == "verified"
            and evidence_reinspection.provider
            and evidence_reinspection.model
        ):
            provider_spaces.append(
                f"{evidence_reinspection.provider}:"
                f"{evidence_reinspection.model}:historical_evidence"
            )
        audited_provider_spaces = tuple(dict.fromkeys(provider_spaces))
        latency_ms = max(0, int((monotonic() - started) * 1000))
        audit_constraints = query.audit_constraints()
        audit_constraints["lifecycle_as_of"] = current_time.isoformat()
        audit_constraints["reinspect_evidence_requested"] = reinspect_evidence
        retrieval_run_id = await self._memory_repository.record_retrieval_run(
            owner_id=normalized_owner_id,
            session_id=query.session_id,
            query=query.query,
            constraints=audit_constraints,
            candidate_ids=[item.document.id for item in fused],
            result_ids=[item.id for item in results],
            reinspection_performed=(
                evidence_reinspection is not None
                and evidence_reinspection.status == "verified"
            ),
            provider_spaces=audited_provider_spaces,
            latency_ms=latency_ms,
        )
        return MemoryRetrievalResponse(
            retrieval_run_id=retrieval_run_id,
            results=results,
            candidate_count=len(fused),
            provider_spaces=audited_provider_spaces,
            evidence_reinspection=evidence_reinspection,
        )


def _to_result(
    candidate: FusedRetrievalCandidate,
    *,
    include_evidence_metadata: bool,
    now: datetime,
) -> MemoryRetrievalResult:
    document = candidate.document
    evidence_available = document.evidence_count > 0 or document.keyframe_count > 0
    metadata = (
        {
            "evidence_count": document.evidence_count,
            "evidence_types": list(document.evidence_types),
            "keyframe_count": document.keyframe_count,
        }
        if include_evidence_metadata
        else {}
    )
    return MemoryRetrievalResult(
        id=document.id,
        record_type=document.record_type,
        session_id=document.session_id,
        occurred_at=document.occurred_at,
        valid_from=document.valid_from,
        valid_to=document.valid_to,
        title=document.title,
        summary=document.summary or document.content[:500],
        confidence=document.confidence,
        importance=document.importance,
        status=document.record_status,
        is_historical=document.temporal_status_at(now) == "historical",
        temporal_status=document.temporal_status_at(now),
        evidence_available=evidence_available,
        keyframe_count=document.keyframe_count,
        match_explanation=tuple(_source_explanation(source) for source in candidate.sources),
        evidence_metadata=metadata,
    )


def _source_explanation(source: str) -> str:
    labels = {
        "time": "within the requested time window",
        "lexical": "matched memory text",
        "entity_exact": "matched a known entity name or alias",
        "entity_embedding": "matched a semantically related known entity",
        "visible_text_exact": "matched text visible in event evidence",
        "temporal_adjacency": "occurred next to another relevant memory",
    }
    if source.startswith("embedding:"):
        return "matched semantic or multimodal meaning"
    return labels.get(source, source.replace("_", " "))
