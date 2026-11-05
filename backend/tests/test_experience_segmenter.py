from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

from app.experience.segmenter import EventSegmentationPolicy, ExperienceEventSegmenter


NOW = datetime(2026, 8, 21, 12, 0, tzinfo=UTC)


def _event(*, started_at: datetime, last_evidence_at: datetime, evidence_count: int = 1):
    return SimpleNamespace(
        started_at=started_at,
        ended_at=None,
        state={
            "last_evidence_at": last_evidence_at.isoformat(),
            "evidence_counts": {"observation": evidence_count},
        },
    )


def _evidence(
    occurred_at: datetime,
    *,
    starts_new_event: bool = False,
    boundary_reason: str | None = None,
):
    return SimpleNamespace(
        occurred_at=occurred_at,
        starts_new_event=starts_new_event,
        boundary_reason=boundary_reason,
    )


def _segmenter() -> ExperienceEventSegmenter:
    return ExperienceEventSegmenter(
        EventSegmentationPolicy(idle_close_seconds=12, max_event_seconds=120)
    )


def test_first_evidence_starts_event() -> None:
    decision = _segmenter().decide(current=None, evidence=_evidence(NOW))

    assert decision.action == "start"
    assert decision.reason == "first_evidence"


def test_idle_gap_splits_event_at_last_evidence() -> None:
    last_evidence_at = NOW - timedelta(seconds=20)
    decision = _segmenter().decide(
        current=_event(
            started_at=NOW - timedelta(minutes=1),
            last_evidence_at=last_evidence_at,
        ),
        evidence=_evidence(NOW),
    )

    assert decision.action == "split"
    assert decision.reason == "idle_gap"
    assert decision.previous_event_ended_at == last_evidence_at


def test_analyzer_boundary_splits_nonempty_event() -> None:
    decision = _segmenter().decide(
        current=_event(started_at=NOW - timedelta(seconds=5), last_evidence_at=NOW),
        evidence=_evidence(
            NOW + timedelta(seconds=1),
            starts_new_event=True,
            boundary_reason="scene_change",
        ),
    )

    assert decision.action == "split"
    assert decision.reason == "scene_change"


def test_bounded_late_evidence_joins_open_event_without_changing_its_timestamp() -> None:
    decision = _segmenter().decide(
        current=_event(started_at=NOW, last_evidence_at=NOW),
        evidence=_evidence(NOW - timedelta(seconds=1)),
    )

    assert decision.action == "join"
    assert decision.reason == "late_open_evidence"


def test_max_duration_takes_precedence_when_event_is_due() -> None:
    reason = _segmenter().due_close_reason(
        _event(
            started_at=NOW - timedelta(seconds=121),
            last_evidence_at=NOW,
        ),
        now=NOW,
    )

    assert reason == "max_duration"


def test_late_evidence_cannot_cross_closed_boundary_or_reopen_old_windows() -> None:
    current = _event(started_at=NOW, last_evidence_at=NOW)
    closed = SimpleNamespace(ended_at=NOW - timedelta(seconds=1))
    decision = _segmenter().decide(current=current, evidence=_evidence(closed.ended_at),
                                  latest_closed=closed)
    assert (decision.action, decision.reason) == ("unattached", "finalized_boundary")
    for evidence, event in (
        (_evidence(NOW - timedelta(seconds=13)), current),
        (_evidence(NOW - timedelta(seconds=1), starts_new_event=True), current),
        (_evidence(NOW - timedelta(seconds=1)),
         _event(started_at=NOW, last_evidence_at=NOW + timedelta(seconds=119))),
    ):
        assert _segmenter().decide(current=event, evidence=evidence).action == "unattached"
    accepted = _segmenter().decide(
        current=current, evidence=_evidence(NOW - timedelta(milliseconds=500)), latest_closed=closed)
    assert accepted.action == "join"
