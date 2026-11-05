from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from uuid import UUID

import asyncpg

from app.images import ImageReference
from app.memory.retrieval.models import (
    RETRIEVAL_RECORD_TYPES,
    FusedRetrievalCandidate,
    RetrievalRecordType,
)


MAX_HISTORICAL_EVIDENCE_FRAMES = 6


@dataclass(frozen=True, slots=True)
class RetainedEvidenceFrame:
    candidate_id: UUID
    visual_frame_id: UUID
    captured_at: datetime
    storage_bucket: str
    storage_path: str
    content_type: str
    existing_description: str | None


@dataclass(frozen=True, slots=True)
class HistoricalImageEvidence:
    reference: ImageReference
    storage_bucket: str = field(repr=False)
    storage_path: str = field(repr=False)


class HistoricalEvidenceRepository:
    """One owner-scoped evidence lookup for direct pixels and legacy inspection."""

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def get_image_evidence(
        self, *, owner_id: str, record_id: UUID,
        record_type: RetrievalRecordType, frame_id: UUID | None = None,
    ) -> HistoricalImageEvidence | None:
        evidence = await self.list_image_evidence(
            owner_id=owner_id, records=((record_type, record_id),),
            limit=1, frame_id=frame_id,
        )
        return evidence[0] if evidence else None

    async def list_image_evidence(
        self, *, owner_id: str,
        records: Sequence[tuple[RetrievalRecordType, UUID]],
        limit: int, frame_id: UUID | None = None,
    ) -> tuple[HistoricalImageEvidence, ...]:
        rows = await self._evidence_rows(
            owner_id=owner_id, records=records, limit=limit,
            frame_id=frame_id, strict_record_types=True,
        )
        return tuple(
            HistoricalImageEvidence(
                reference=ImageReference(
                    frame_id=row["visual_frame_id"], owner_id=owner_id.strip(),
                    session_id=row["session_id"], source="historical",
                    captured_at=row["captured_at"], content_type=row["content_type"],
                    width=row["width"], height=row["height"],
                    raw_image_expires_at=row["raw_image_expires_at"],
                    record_id=row["candidate_id"], record_type=row["candidate_type"],
                ),
                storage_bucket=row["storage_bucket"], storage_path=row["storage_path"],
            )
            for row in rows
        )

    async def list_retained_evidence_frames(
        self, *, owner_id: str, candidates: Sequence[FusedRetrievalCandidate], limit: int,
    ) -> tuple[RetainedEvidenceFrame, ...]:
        # The legacy API accepts fused candidates, including projected record types.
        # The new record-identity API always checks the underlying memory type.
        rows = await self._evidence_rows(
            owner_id=owner_id,
            records=tuple((item.document.record_type, item.document.id) for item in candidates),
            limit=limit, frame_id=None, strict_record_types=False,
        )
        seen: set[UUID] = set()
        frames = []
        for row in rows:
            if row["visual_frame_id"] in seen:
                continue
            seen.add(row["visual_frame_id"])
            frames.append(RetainedEvidenceFrame(
                candidate_id=row["candidate_id"], visual_frame_id=row["visual_frame_id"],
                captured_at=row["captured_at"], storage_bucket=row["storage_bucket"],
                storage_path=row["storage_path"], content_type=row["content_type"],
                existing_description=row["frame_description"],
            ))
        return tuple(frames)

    async def _evidence_rows(
        self, *, owner_id: str, records: Sequence[tuple[RetrievalRecordType, UUID]],
        limit: int, frame_id: UUID | None, strict_record_types: bool,
    ) -> list[asyncpg.Record]:
        owner_id = owner_id.strip()
        if not owner_id:
            raise ValueError("owner_id cannot be blank")
        if not 1 <= limit <= MAX_HISTORICAL_EVIDENCE_FRAMES:
            raise ValueError("historical evidence frame limit must be between 1 and 6")
        records = tuple(dict.fromkeys(records))
        if any(record_type not in RETRIEVAL_RECORD_TYPES for record_type, _ in records):
            raise ValueError("unsupported historical record type")
        if not records:
            return []
        async with self._pool.acquire() as conn:
            return await conn.fetch(
                """
                with requested as (
                    select candidate_id, candidate_type, candidate_rank
                    from unnest($2::uuid[], $3::text[]) with ordinality
                        as request(candidate_id, candidate_type, candidate_rank)
                ),
                available_frames as (
                    select * from visual_frames
                    where owner_id = $1
                      and raw_image_status = 'available'
                      and raw_image_expires_at > now()
                      and ($5::uuid is null or id = $5)
                ),
                direct_event_frames as (
                    select request.candidate_id, request.candidate_type,
                           request.candidate_rank, 1::integer as source_priority,
                           keyframe.rank::bigint as evidence_rank,
                           frame.id as visual_frame_id, frame.session_id,
                           frame.captured_at, frame.storage_bucket, frame.storage_path,
                           frame.content_type, frame.frame_description,
                           frame.width, frame.height, frame.raw_image_expires_at
                    from requested request
                    join experience_events event
                      on request.candidate_type = 'experience_event'
                     and event.id = request.candidate_id and event.owner_id = $1
                    join experience_event_keyframes keyframe on keyframe.event_id = event.id
                    join available_frames frame
                      on frame.id = keyframe.visual_frame_id
                     and frame.session_id = event.session_id
                ),
                direct_memory_frames as (
                    select request.candidate_id, request.candidate_type,
                           request.candidate_rank, 0::integer as source_priority,
                           0::bigint as evidence_rank,
                           frame.id as visual_frame_id, frame.session_id,
                           frame.captured_at, frame.storage_bucket, frame.storage_path,
                           frame.content_type, frame.frame_description,
                           frame.width, frame.height, frame.raw_image_expires_at
                    from requested request
                    join memory_items memory
                      on request.candidate_type <> 'experience_event'
                     and memory.id = request.candidate_id and memory.owner_id = $1
                     and memory.status <> 'forgotten'
                     and (not $6::boolean or memory.memory_type = request.candidate_type)
                    join memory_evidence evidence
                      on evidence.memory_item_id = memory.id
                     and evidence.owner_id = memory.owner_id
                     and evidence.evidence_type = 'frame'
                    join available_frames frame on frame.id = evidence.evidence_ref_id::uuid
                ),
                memory_event_frames as (
                    select request.candidate_id, request.candidate_type,
                           request.candidate_rank, 2::integer as source_priority,
                           keyframe.rank::bigint as evidence_rank,
                           frame.id as visual_frame_id, frame.session_id,
                           frame.captured_at, frame.storage_bucket, frame.storage_path,
                           frame.content_type, frame.frame_description,
                           frame.width, frame.height, frame.raw_image_expires_at
                    from requested request
                    join memory_items memory
                      on request.candidate_type <> 'experience_event'
                     and memory.id = request.candidate_id and memory.owner_id = $1
                     and memory.status <> 'forgotten'
                     and (not $6::boolean or memory.memory_type = request.candidate_type)
                    join memory_evidence evidence
                      on evidence.memory_item_id = memory.id
                     and evidence.owner_id = memory.owner_id
                     and evidence.evidence_type = 'experience_event'
                    join experience_events event
                      on event.id = evidence.evidence_ref_id::uuid
                     and event.owner_id = memory.owner_id
                    join experience_event_keyframes keyframe on keyframe.event_id = event.id
                    join available_frames frame
                      on frame.id = keyframe.visual_frame_id
                     and frame.session_id = event.session_id
                ),
                combined as (
                    select * from direct_event_frames
                    union all select * from direct_memory_frames
                    union all select * from memory_event_frames
                ),
                ranked as (
                    select combined.*,
                           row_number() over (
                               partition by visual_frame_id
                               order by candidate_rank, source_priority, evidence_rank,
                                        captured_at, visual_frame_id
                           ) as frame_rank,
                           row_number() over (
                               partition by candidate_type, candidate_id, visual_frame_id
                               order by source_priority, evidence_rank, captured_at
                           ) as relation_rank
                    from combined
                ),
                selected_frames as (
                    select visual_frame_id from ranked
                    where frame_rank = 1
                    order by candidate_rank, source_priority, evidence_rank,
                             captured_at, visual_frame_id
                    limit $4
                )
                select ranked.* from ranked
                join selected_frames using (visual_frame_id)
                where relation_rank = 1
                order by candidate_rank, source_priority, evidence_rank,
                         captured_at, visual_frame_id
                """,
                owner_id, [record_id for _, record_id in records],
                [record_type for record_type, _ in records], limit, frame_id,
                strict_record_types,
            )
