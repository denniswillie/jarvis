from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from math import isfinite


class ProactiveInterruptGate:
    """Bounds consecutive proactive responses until the user speaks again."""

    def __init__(self, *, max_chain_length: int) -> None:
        self._max_chain_length = max(1, max_chain_length)
        self._chain_length = 0

    def can_begin(self) -> bool:
        return self._chain_length < self._max_chain_length

    def try_begin(self) -> bool:
        if not self.can_begin():
            return False
        self._chain_length += 1
        return True

    def reset_for_user_turn(self) -> None:
        self._chain_length = 0


GENERIC_VISUAL_QUIET_SECONDS = 5.0


@dataclass(frozen=True)
class GenericSpeechActivity:
    """A snapshot of session facts, not an independently updated turn registry."""

    closed: bool
    transport_ready: bool
    user_audio_pending: bool
    normal_turn_pending: bool
    speech_pending_or_active: bool
    last_user_turn_settled_at: datetime | None
    last_user_turn_settled_monotonic: float | None


def generic_visual_speech_rejection(
    activity: GenericSpeechActivity,
    *,
    source_matches: bool,
    source_received_at: datetime | None,
    now: datetime,
    monotonic_now: float,
) -> str | None:
    """Generic comments are disposable; durable reminders use their own path."""
    if activity.closed or not activity.transport_ready:
        return "session_unavailable"
    if not source_matches:
        return "source_session_mismatch"
    if activity.user_audio_pending or activity.normal_turn_pending:
        return "user_turn_busy"
    if activity.speech_pending_or_active:
        return "speech_busy"
    if (
        source_received_at is None
        or source_received_at.utcoffset() is None
        or now.utcoffset() is None
        or source_received_at > now
        or not isfinite(monotonic_now)
    ):
        return "source_time_unavailable"
    settled_at = activity.last_user_turn_settled_at
    settled_monotonic = activity.last_user_turn_settled_monotonic
    if settled_at is not None:
        if settled_at.utcoffset() is None or settled_monotonic is None:
            return "turn_time_unavailable"
        elapsed = monotonic_now - settled_monotonic
        if not isfinite(elapsed) or elapsed < GENERIC_VISUAL_QUIET_SECONDS:
            return "user_turn_quiet_interval"
        if source_received_at <= settled_at:
            return "superseded_observation"
    return None
