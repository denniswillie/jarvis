from dataclasses import replace
from datetime import UTC, datetime
from uuid import UUID

from app.memory.retrieval.fusion import reciprocal_rank_fuse
from app.memory.retrieval.models import RetrievalDocument, RetrievalSourceHit


def document(value: int, *, explicit: bool = False, importance: float = 0.5):
    return RetrievalDocument(
        id=UUID(int=value),
        owner_id="owner",
        session_id=UUID(int=100),
        record_type="episode",
        occurred_at=datetime(2026, 8, 27, 12, value, tzinfo=UTC),
        valid_from=None,
        valid_to=None,
        title=f"Memory {value}",
        summary="summary",
        content="content",
        importance=importance,
        confidence=0.8,
        record_status="active",
        is_explicit=explicit,
        evidence_count=1,
        keyframe_count=0,
        evidence_types=("experience_event",),
    )


def hit(item: RetrievalDocument, source: str, raw: float, *, exact: bool = False):
    return RetrievalSourceHit(
        document=item,
        source=source,
        raw_score=raw,
        exact_match=exact,
    )


def test_fusion_rewards_agreement_between_incompatible_sources() -> None:
    first = document(1)
    second = document(2)

    results = reciprocal_rank_fuse(
        {
            "lexical": [hit(first, "lexical", 0.01), hit(second, "lexical", 999)],
            "embedding:event_text": [hit(first, "embedding:event_text", -0.2)],
        },
        candidate_limit=10,
    )

    assert [item.document.id for item in results] == [first.id, second.id]
    assert results[0].sources == ("lexical", "embedding:event_text")


def test_fusion_applies_bounded_exact_and_explicit_boosts() -> None:
    ordinary = document(1)
    explicit = document(2, explicit=True, importance=1.0)

    results = reciprocal_rank_fuse(
        {
            "lexical": [
                hit(ordinary, "lexical", 1.0),
                hit(explicit, "lexical", 0.8, exact=True),
            ]
        },
        candidate_limit=10,
    )

    assert results[0].document.id == explicit.id
    assert results[0].exact_match is True


def test_fusion_deduplicates_subject_within_one_source() -> None:
    item = document(1)

    results = reciprocal_rank_fuse(
        {"lexical": [hit(item, "lexical", 1), hit(item, "lexical", 0.5)]},
        candidate_limit=10,
    )

    assert len(results) == 1


def test_fusion_decays_old_records_without_erasing_them() -> None:
    recent = document(1)
    old = replace(
        recent,
        id=UUID(int=2),
        occurred_at=datetime(2025, 8, 27, 12, 0, tzinfo=UTC),
    )

    results = reciprocal_rank_fuse(
        {"lexical": [hit(old, "lexical", 1), hit(recent, "lexical", 1)]},
        candidate_limit=10,
        now=datetime(2026, 8, 28, 12, 0, tzinfo=UTC),
    )

    assert [item.document.id for item in results] == [recent.id, old.id]
    assert results[1].score > 0


def test_fusion_applies_bounded_recent_retrieval_boost() -> None:
    ordinary = document(1)
    frequently_retrieved = replace(
        ordinary,
        id=UUID(int=2),
        retrieval_count=100,
    )

    results = reciprocal_rank_fuse(
        {"lexical": [hit(ordinary, "lexical", 1), hit(frequently_retrieved, "lexical", 1)]},
        candidate_limit=10,
        now=datetime(2026, 8, 28, 12, 0, tzinfo=UTC),
    )

    assert results[0].document.id == frequently_retrieved.id
