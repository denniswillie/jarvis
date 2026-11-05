from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from uuid import UUID


HISTORICAL_EVIDENCE_INSPECTION_INSTRUCTIONS = """
You are reinspecting retained historical image evidence for a personal memory query.

Answer only from the supplied images and structured candidate summaries. Use the
images to recover concrete visual details that the summaries may have omitted. Do
not treat a candidate summary as visual proof. If the pixels do not establish a
detail, state that uncertainty instead of guessing.

Return exactly one JSON object with this shape:
{
  "evidence_sufficient": true,
  "verified_details": "concise answer to the query grounded in retained pixels",
  "uncertainty": "remaining uncertainty, or null",
  "confidence": 0.0,
  "supporting_frame_ids": ["UUID from the supplied frame list"]
}

Set evidence_sufficient to true only when the retained pixels directly establish
the verified details. When the images are ambiguous, irrelevant, or do not answer
the query, set evidence_sufficient to false, leave supporting_frame_ids empty, and
describe the limitation in uncertainty. Never cite a frame merely because it was
provided. When evidence_sufficient is true, supporting_frame_ids must contain only
frames that visibly support the answer. Never return storage paths, URLs, image
bytes, credentials, or commentary outside the JSON object.
""".strip()


@dataclass(frozen=True, slots=True)
class HistoricalEvidencePromptCandidate:
    id: UUID
    record_type: str
    occurred_at: datetime
    title: str
    summary: str


@dataclass(frozen=True, slots=True)
class HistoricalEvidencePromptFrame:
    id: UUID
    candidate_id: UUID
    captured_at: datetime
    existing_description: str | None


def build_historical_evidence_prompt(
    *,
    query: str,
    candidates: tuple[HistoricalEvidencePromptCandidate, ...],
    frames: tuple[HistoricalEvidencePromptFrame, ...],
) -> str:
    normalized_query = query.strip() or "What happened in the retrieved memory?"
    payload = {
        "query": normalized_query,
        "candidate_memories": [
            {
                "id": str(candidate.id),
                "record_type": candidate.record_type,
                "occurred_at": candidate.occurred_at.isoformat(),
                "title": candidate.title,
                "summary": candidate.summary,
            }
            for candidate in candidates
        ],
        "retained_frames": [
            {
                "frame_id": str(frame.id),
                "candidate_id": str(frame.candidate_id),
                "captured_at": frame.captured_at.isoformat(),
                "existing_description": frame.existing_description,
            }
            for frame in frames
        ],
    }
    return (
        f"{HISTORICAL_EVIDENCE_INSPECTION_INSTRUCTIONS}\n\n"
        "Evidence bundle:\n"
        f"{json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}"
    )
