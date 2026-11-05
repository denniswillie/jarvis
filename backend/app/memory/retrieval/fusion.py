from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from math import log1p

from app.memory.retrieval.models import FusedRetrievalCandidate, RetrievalSourceHit


RRF_RANK_CONSTANT = 60
EXACT_MATCH_BOOST = 0.015
EXPLICIT_MEMORY_BOOST = 0.012
IMPORTANCE_BOOST_MAX = 0.010
CONFIDENCE_BOOST_MAX = 0.005
RETRIEVAL_FREQUENCY_BOOST_MAX = 0.006
AGE_DECAY_HALF_LIFE_DAYS = 30.0
AGE_DECAY_FLOOR = 0.35


@dataclass(slots=True)
class _Accumulator:
    score: float
    sources: list[str]
    exact_match: bool
    hit: RetrievalSourceHit


def reciprocal_rank_fuse(
    rankings: Mapping[str, Sequence[RetrievalSourceHit]],
    *,
    candidate_limit: int,
    now: datetime | None = None,
) -> list[FusedRetrievalCandidate]:
    if candidate_limit < 1:
        raise ValueError("candidate_limit must be positive")
    current_time = now or datetime.now(UTC)
    accumulators: dict[tuple[str, object], _Accumulator] = {}

    for source, hits in rankings.items():
        seen: set[tuple[str, object]] = set()
        for rank, hit in enumerate(hits, start=1):
            key = (hit.document.record_type, hit.document.id)
            if key in seen:
                continue
            seen.add(key)
            contribution = 1.0 / (RRF_RANK_CONSTANT + rank)
            accumulator = accumulators.get(key)
            if accumulator is None:
                accumulator = _Accumulator(
                    score=0.0,
                    sources=[],
                    exact_match=False,
                    hit=hit,
                )
                accumulators[key] = accumulator
            accumulator.score += contribution
            accumulator.sources.append(source)
            accumulator.exact_match = accumulator.exact_match or hit.exact_match

    fused: list[FusedRetrievalCandidate] = []
    for accumulator in accumulators.values():
        document = accumulator.hit.document
        score = accumulator.score * _age_decay(document.occurred_at, current_time)
        if accumulator.exact_match:
            score += EXACT_MATCH_BOOST
        if document.is_explicit:
            score += EXPLICIT_MEMORY_BOOST
        score += document.importance * IMPORTANCE_BOOST_MAX
        score += document.confidence * CONFIDENCE_BOOST_MAX
        score += _retrieval_frequency_boost(document.retrieval_count)
        fused.append(
            FusedRetrievalCandidate(
                document=document,
                score=score,
                sources=tuple(dict.fromkeys(accumulator.sources)),
                exact_match=accumulator.exact_match,
            )
        )

    fused.sort(
        key=lambda item: (
            -item.score,
            -_timestamp(item.document.occurred_at, current_time),
            str(item.document.id),
        )
    )
    return fused[:candidate_limit]


def group_rankings(
    hits: Sequence[RetrievalSourceHit],
) -> dict[str, list[RetrievalSourceHit]]:
    rankings: dict[str, list[RetrievalSourceHit]] = defaultdict(list)
    for hit in hits:
        rankings[hit.source].append(hit)
    return dict(rankings)


def _timestamp(value: datetime, fallback: datetime) -> float:
    try:
        return value.timestamp()
    except (OverflowError, OSError):
        return fallback.timestamp()


def _age_decay(occurred_at: datetime, now: datetime) -> float:
    age_seconds = max(0.0, (now - occurred_at).total_seconds())
    age_days = age_seconds / 86_400.0
    decayed = 0.5 ** (age_days / AGE_DECAY_HALF_LIFE_DAYS)
    return AGE_DECAY_FLOOR + ((1.0 - AGE_DECAY_FLOOR) * decayed)


def _retrieval_frequency_boost(retrieval_count: int) -> float:
    bounded = min(10, max(0, retrieval_count))
    return (log1p(bounded) / log1p(10)) * RETRIEVAL_FREQUENCY_BOOST_MAX
