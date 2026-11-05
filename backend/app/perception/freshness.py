from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from math import isfinite
from typing import Literal


CAPTURE_CLOCK_TOLERANCE_SECONDS = 1.0


@dataclass(frozen=True)
class FrameFreshness:
    status: Literal["fresh", "stale", "uncertain"]
    age_seconds: float
    reason: Literal["age_limit_exceeded", "clock_uncertain", "monotonic_regressed"] | None = None


def require_aware_timestamp(value: datetime) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Visual evidence timestamps must include a timezone.")


def assess_frame_freshness(
    *,
    captured_at: datetime,
    server_received_at: datetime,
    sampled_at: datetime,
    sampled_monotonic: float,
    now: datetime,
    monotonic_now: float,
    max_age_seconds: float,
) -> FrameFreshness:
    """Conservative source age; neither late upload nor clock rollback refreshes it.

    Keep the first local sample for an immutable frame. Monotonic elapsed time
    prevents subsequent wall-clock changes from making that sample younger.
    Capture time remains evidence, not an inferred client-clock correction.
    """
    for timestamp in (captured_at, server_received_at, sampled_at, now):
        require_aware_timestamp(timestamp)
    if not all(isfinite(value) for value in (sampled_monotonic, monotonic_now, max_age_seconds)):
        raise ValueError("Visual freshness durations must be finite.")
    if max_age_seconds <= 0:
        raise ValueError("Visual freshness age limit must be positive.")

    initial_age = max(
        0.0,
        (sampled_at - captured_at).total_seconds(),
        (sampled_at - server_received_at).total_seconds(),
    )
    elapsed = monotonic_now - sampled_monotonic
    age = max(
        initial_age + max(0.0, elapsed),
        (now - captured_at).total_seconds(),
        (now - server_received_at).total_seconds(),
    )
    if elapsed < 0:
        return FrameFreshness("uncertain", age, "monotonic_regressed")
    if (
        (captured_at - server_received_at).total_seconds() > CAPTURE_CLOCK_TOLERANCE_SECONDS
        or (server_received_at - sampled_at).total_seconds() > CAPTURE_CLOCK_TOLERANCE_SECONDS
        or (sampled_at - now).total_seconds() > CAPTURE_CLOCK_TOLERANCE_SECONDS
    ):
        return FrameFreshness("uncertain", age, "clock_uncertain")
    if age > max_age_seconds:
        return FrameFreshness("stale", age, "age_limit_exceeded")
    return FrameFreshness("fresh", age)
