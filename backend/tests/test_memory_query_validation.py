from datetime import UTC, datetime, timedelta

import pytest

from app.memory.retrieval.query import MemoryRetrievalQuery, normalize_memory_query


NOW = datetime(2026, 8, 27, 12, 0, tzinfo=UTC)


def normalize(query: MemoryRetrievalQuery):
    return normalize_memory_query(
        query,
        now=NOW,
        default_seconds=600,
        max_days=3650,
    )


def test_query_defaults_to_recent_memory_window() -> None:
    result = normalize(MemoryRetrievalQuery())

    assert result.start_time == NOW - timedelta(seconds=600)
    assert result.end_time == NOW


def test_relative_and_absolute_time_forms_are_mutually_exclusive() -> None:
    with pytest.raises(ValueError, match="cannot be combined"):
        normalize(
            MemoryRetrievalQuery(
                seconds_before_now=60,
                start_time=NOW - timedelta(minutes=1),
                end_time=NOW,
            )
        )


def test_absolute_time_requires_both_bounds() -> None:
    with pytest.raises(ValueError, match="provided together"):
        normalize(MemoryRetrievalQuery(start_time=NOW - timedelta(minutes=1)))


def test_time_bounds_must_be_timezone_aware() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        normalize(
            MemoryRetrievalQuery(
                start_time=datetime(2026, 8, 27, 11, 0),
                end_time=NOW,
            )
        )


def test_query_rejects_excessive_time_range() -> None:
    with pytest.raises(ValueError, match="cannot exceed"):
        normalize(MemoryRetrievalQuery(seconds_before_now=3651 * 24 * 60 * 60))


@pytest.mark.parametrize("seconds", [0, False])
def test_query_rejects_zero_and_boolean_relative_windows(seconds: object) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        normalize(
            MemoryRetrievalQuery(seconds_before_now=seconds)  # type: ignore[arg-type]
        )


def test_query_rejects_unknown_memory_type() -> None:
    with pytest.raises(ValueError, match="unsupported retrieval memory types"):
        normalize(MemoryRetrievalQuery(memory_types=("unknown",)))  # type: ignore[arg-type]


def test_memory_query_does_not_keyword_classify_current_view_language() -> None:
    result = normalize(MemoryRetrievalQuery(query="What am I looking at right now?"))

    assert result.query == "What am I looking at right now?"
    assert result.start_time == NOW - timedelta(seconds=600)
