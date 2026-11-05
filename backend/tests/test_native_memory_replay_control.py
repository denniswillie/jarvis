"""Offline doubles only: no native, SQL, model, service or authorization calls."""
from copy import deepcopy
from datetime import UTC, datetime, timedelta
import json
from pathlib import Path
from uuid import uuid4

import pytest

from evals.responsive_voice.native_memory_replay_control import (
    ContractError, DomainCalendar, EvidenceRef, MemoryReplayController, Scope, digest,
)


FIXTURE = Path(__file__).parents[1] / "evals/responsive_voice/fixtures/native_memory_replay_c02c.json"
PLAN = json.loads(FIXTURE.read_text())
EPOCH = datetime(2026, 1, 1, tzinfo=UTC)  # Synthetic domain label, not a real capture date.
WALL = datetime(2026, 10, 31, tzinfo=UTC)
HASH = "a" * 64
REF = EvidenceRef("offline-double.json", HASH, "/synthetic/receipt")
RUN = "1" * 32
SCOPE = Scope(RUN, f"direct-image-eval-second-brain-{RUN}-a",
              f"jarvis_sb_eval_{RUN}_0", str(uuid4()), "jarvis_" + "2" * 32)


class Clock:
    def __init__(self):
        self.ms = 0
        self.wall = WALL

    def monotonic(self):
        return self.ms / 1000


def controller(plan=None):
    clock = Clock()
    result = MemoryReplayController(plan or PLAN, SCOPE, epoch=EPOCH,
                                    monotonic=clock.monotonic, wall=lambda: clock.wall)
    return result, clock


def reserve(c, **changes):
    values = dict(authorization_id=str(uuid4()), scope=SCOPE, plan_sha256=c.plan_sha256,
                  input_manifest_sha256=HASH, provider_bindings_sha256=HASH,
                  expires_at=WALL + timedelta(hours=1), consumed_ledger_ref=REF)
    values.update(changes)
    c.reserve_admission(**values)
    return values


def ready(c, **changes):
    phase = c.active["phase"]
    values = dict(scope=SCOPE, session_id="session-" + phase["id"],
                  audio_session_id="audio-" + phase["id"], app_instance_id="app-" + phase["id"],
                  host_instance_id="host-" + phase["id"], origin_run_ms=phase["start_ms"],
                  input_manifest_sha256=HASH, evidence=REF)
    values.update(changes)
    return c.ready(**values)


def publish(c, clock, **changes):
    frame = c.active["phase"]["frames"][10]  # source 20 s, publication at T55.
    clock.ms = frame["at_ms"]
    values = dict(source_id="source-observed", revision=1, generation=1,
                  frame_id=frame["id"], frame_sha256=frame["sha256"],
                  source_pts_ms=frame["source_pts_ms"], source_day=0,
                  published_at_ms=clock.ms, evidence=REF)
    values.update(changes)
    return c.publish(**values)


def close(c, clock, **changes):
    phase = c.active["phase"]
    clock.ms = phase["end_ms"]
    values = dict(scope=SCOPE, ended_at_ms=clock.ms,
                  retired={k: True for k in c.RETIREMENT}, canonical=REF,
                  provider_finalization="confirmed")
    values.update(changes)
    c.close_phase(**values)


def observed():
    c, clock = controller()
    reserve(c)
    ready(c)
    source = publish(c, clock)
    close(c, clock)
    return c, clock, source


def retrieve(c, clock, source, **changes):
    clock.ms = c.active["phase"]["query_ms"]
    values = dict(source_key=source, scope=SCOPE, session_id=c.active["ready"]["ids"][0],
                  query_at_ms=clock.ms, evidence=REF, selected_context=REF)
    values.update(changes)
    return c.retrieve(**values)


def test_frozen_c02c_contract_is_not_two_runnable_manifests():
    assert PLAN["evidence_mode"] == "offline_controller_contract_not_native_acceptance"
    observe, recall = PLAN["phases"]
    assert (observe["end_ms"], recall["start_ms"], recall["query_ms"], recall["end_ms"]) == (
        67000, 67000, 75000, 95000)
    assert observe["speech_ids"] == ["u03"]
    assert recall["speech_ids"] == ["u04"]
    assert [f["source_pts_ms"] for f in observe["frames"]] == list(range(0, 32000, 2000))
    assert [f["at_ms"] for f in observe["frames"]] == list(range(35000, 67000, 2000))
    assert recall["frames"] == []  # No dummy frame, repeated image or answer/context.
    assert all("answer" not in f and "context" not in f for f in observe["frames"])
    assert PLAN["provenance"]["reference_is_not_a_multisession_manifest"] is True


def test_offline_two_phase_contract_retains_scope_but_never_claims_e2e():
    c, clock, source = observed()
    assert c.state == "between_sessions" and c.scope == SCOPE
    reserve(c)
    clock.ms = 70000
    assert ready(c) == 5000  # Not eight seconds after the second readiness.
    assert retrieve(c, clock, source) == 20  # Source20 -> query40, not publication age.
    close(c, clock)
    c.cleanup(scope=SCOPE, checks={k: True for k in c.CLEANUP}, evidence=REF,
              started_at_ms=clock.ms)
    report = c.report()
    assert report["state"] == "closed"
    assert report["sessions_retired"] == report["admissions_reserved"] == 2
    assert report["native_e2e_accepted"] is False
    assert report["semantic_acceptance"] == "not_evaluated"
    assert report["scheduler_time_travel_qualified"] is False
    assert report["remaining_native_gates"]


@pytest.mark.parametrize("kind,value", [
    ("scene", 180001), ("day", -1), ("day", 91), ("day", True),
    ("spacing", 35001), ("future", 32000), ("source_order", 0),
    ("lease", 754), ("query", 95000), ("overlap", 66000),
    ("oracle", "expected location"), ("duplicate_phase", "observe"),
])
def test_invalid_plan_bounds(kind, value):
    p = deepcopy(PLAN)
    a, b = p["phases"]
    if kind == "scene":
        a["end_ms"] = value
    elif kind == "day":
        b["day"] = value
    elif kind == "spacing":
        a["frames"][0]["at_ms"] = value
    elif kind == "future":
        a["frames"][-1]["source_pts_ms"] = value
    elif kind == "source_order":
        a["frames"][1]["source_pts_ms"] = value
    elif kind == "lease":
        p["max_wall_seconds"] = value
    elif kind == "query":
        b["query_ms"] = value
    elif kind == "overlap":
        b["start_ms"] = value
    elif kind == "oracle":
        a["frames"][0]["expected_answer"] = value
    else:
        b["id"] = value
    with pytest.raises(ContractError):
        controller(p)


@pytest.mark.parametrize("duration", [1, 180000])
def test_scene_boundary_inclusive(duration):
    p = deepcopy(PLAN)
    a, b = p["phases"]
    a.update(end_ms=duration, frames=[])
    b.update(start_ms=duration, end_ms=duration + 1, query_ms=duration)
    controller(p)


@pytest.mark.parametrize("changes", [
    {"public_demo": True}, {"plan_sha256": "b" * 64},
    {"expires_at": WALL - timedelta(seconds=1)},
    {"expires_at": WALL + timedelta(seconds=487)},
    {"input_manifest_sha256": "invented"}, {"provider_bindings_sha256": "invented"},
    {"consumed_ledger_ref": None},
])
def test_admission_rejects_unbound_or_expired_receipt(changes):
    c, _ = controller()
    with pytest.raises(ContractError):
        reserve(c, **changes)
    assert c.report()["admissions_reserved"] == 0


def test_rejected_or_ambiguous_attempt_is_burned_no_automatic_retry():
    c, clock = controller()
    values = reserve(c)
    with pytest.raises(ContractError, match="admission_not_allowed"):
        reserve(c)
    c.abort("native_admission_rejected_or_ambiguous")
    c.close_phase(scope=SCOPE, ended_at_ms=0,
                  retired={k: True for k in c.RETIREMENT}, canonical=REF,
                  provider_finalization="unknown")
    assert c.state == "cleanup_ready"
    with pytest.raises(ContractError):
        reserve(c, authorization_id=values["authorization_id"])
    assert c.report()["admissions_reserved"] == 1


def test_scope_changed_or_released_authorization_reused():
    c, clock, _ = observed()
    prior_id = next(iter(c.authorizations))
    with pytest.raises(ContractError, match="authorization_reused"):
        reserve(c, authorization_id=prior_id)
    other = Scope(RUN, SCOPE.owner_id, SCOPE.database_schema, SCOPE.area_id, "jarvis_" + "3" * 32)
    with pytest.raises(ContractError, match="authorization_scope_changed"):
        reserve(c, scope=other)


@pytest.mark.parametrize("field", ["session_id", "audio_session_id", "app_instance_id", "host_instance_id"])
def test_old_session_conversation_or_process_cannot_be_reused(field):
    c, _, _ = observed()
    reserve(c)
    old = dict(zip(("session_id", "audio_session_id", "app_instance_id", "host_instance_id"),
                   c.receipts[0]["ready"]["ids"]))
    with pytest.raises(ContractError, match="identity_reused"):
        ready(c, **{field: old[field]})


@pytest.mark.parametrize("at_ms,allowed", [(74999, True), (75000, True), (75001, False)])
def test_frozen_eight_second_readiness_deadline_is_not_extended(at_ms, allowed):
    c, clock, _ = observed()
    reserve(c)
    clock.ms = at_ms
    if allowed:
        assert ready(c) == 75000 - at_ms
    else:
        with pytest.raises(ContractError, match="fixed_source_deadline_missed"):
            ready(c)


def test_second_native_origin_must_not_reset_to_readiness():
    c, clock, _ = observed()
    reserve(c)
    clock.ms = 70000
    with pytest.raises(ContractError, match="origin_rebased"):
        ready(c, origin_run_ms=70000)


@pytest.mark.parametrize("changes", [
    {"source_pts_ms": 30000}, {"frame_sha256": "b" * 64},
    {"frame_id": "not-selected"}, {"source_day": 1},
    {"published_at_ms": 55001}, {"published_at_ms": 34999},
    {"revision": 0}, {"generation": True}, {"evidence": None},
])
def test_publication_cannot_invent_frame_version_or_future_evidence(changes):
    c, clock = controller()
    reserve(c)
    ready(c)
    with pytest.raises(ContractError):
        publish(c, clock, **changes)


@pytest.mark.parametrize("changes", [
    {"source_key": ("not-published", 1, 1)},
    {"source_key": ("source-observed", 2, 1)},
    {"session_id": "session-observe"}, {"query_at_ms": 75001},
    {"selected_context": None}, {"evidence": None},
])
def test_retrieval_requires_prior_published_version_actual_query_and_selection(changes):
    c, clock, source = observed()
    reserve(c)
    ready(c)
    with pytest.raises(ContractError):
        retrieve(c, clock, source, **changes)


@pytest.mark.parametrize("field", MemoryReplayController.RETIREMENT)
def test_no_next_session_or_fixture_cleanup_before_all_handles_join(field):
    c, clock = controller()
    reserve(c)
    ready(c)
    publish(c, clock)
    values = {k: True for k in c.RETIREMENT}
    values[field] = False
    with pytest.raises(ContractError, match="unjoined_session"):
        close(c, clock, retired=values)
    with pytest.raises(ContractError, match="admission_not_allowed"):
        reserve(c)
    with pytest.raises(ContractError, match="cleanup_before_join"):
        c.cleanup(scope=SCOPE, checks={k: True for k in c.CLEANUP},
                  evidence=REF, started_at_ms=clock.ms)


def test_missing_export_or_memory_evidence_preserves_active_scope():
    c, clock = controller()
    reserve(c)
    ready(c)
    with pytest.raises(ContractError, match="no_persisted"):
        close(c, clock)
    c.abort("no_persisted_publication")
    with pytest.raises(ContractError, match="evidence_reference"):
        close(c, clock, canonical=None)
    assert c.active is not None


def test_no_retrieval_cannot_complete_recall_phase():
    c, clock, _ = observed()
    reserve(c)
    ready(c)
    with pytest.raises(ContractError, match="no_actual_retrieval"):
        close(c, clock)


def test_unknown_remote_provider_finalization_is_not_local_cleanup_success():
    c, clock = controller()
    reserve(c)
    ready(c)
    publish(c, clock)
    close(c, clock, provider_finalization="unknown")
    with pytest.raises(ContractError, match="previous_provider"):
        reserve(c)
    reserve(c, prior_unknown_risk_accepted=True)
    assert c.report()["provider_finalization"] == "unknown"


def test_day_month_checkpoint_math_preserves_real_clocks():
    c = DomainCalendar(EPOCH)
    for day in (0, 1, 7, 30, 90):
        old = c.day
        assert c.advance(day) == (day - old) * 86400
        assert c.event_time(0, 20000) == EPOCH + timedelta(seconds=20)
        assert c.event_time(day, 40000) == EPOCH + timedelta(days=day, seconds=40)
    with pytest.raises(ContractError):
        c.advance(30)
    with pytest.raises(ContractError):
        c.advance(91)
    with pytest.raises(ContractError):
        c.event_time(91, 0)


@pytest.mark.parametrize("day", [1, 7, 30, 90])
def test_advance_only_after_retirement_requires_explicit_native_clock_receipt(day):
    p = deepcopy(PLAN)
    p["case_id"] = "calendar-contract-not-c02c-v1"
    p["phases"][1]["day"] = day
    c, clock = controller(p)
    reserve(c)
    ready(c)
    source = publish(c, clock)
    assert c.calendar.day == 0
    with pytest.raises(ContractError):
        reserve(c)
    close(c, clock)
    reserve(c)
    assert c.calendar.day == day
    with pytest.raises(ContractError, match="native_domain_clock_not_wired"):
        ready(c)
    ready(c, clock_components=tuple(c.CLOCK_COMPONENTS))
    assert retrieve(c, clock, source) == day * 86400 + 20
    assert c.report()["native_e2e_accepted"] is False


@pytest.mark.parametrize("component", ["sql", "authorization", "lease", "scheduler", "transport"])
def test_security_and_scheduler_clocks_never_claimed_as_domain_consumers(component):
    c, _, _ = observed()
    reserve(c)
    with pytest.raises(ContractError, match="unsupported_domain_clock_scope"):
        ready(c, clock_components=tuple(c.CLOCK_COMPONENTS | {component}))


def test_monotonic_and_real_authorization_clocks_remain_independent():
    c, clock = controller()
    independent_domain = DomainCalendar(EPOCH)
    independent_domain.advance(90)
    clock.wall = WALL + timedelta(days=1)
    with pytest.raises(ContractError, match="real_wall_expired"):
        reserve(c)
    clock.wall = WALL
    reserve(c)
    clock.ms = -1
    with pytest.raises(ContractError, match="monotonic_reversed"):
        ready(c)


def test_retirement_and_whole_controller_deadlines():
    c, clock = controller()
    reserve(c)
    c.abort("cancelled")
    clock.ms = 65001
    with pytest.raises(ContractError, match="retirement_join_deadline"):
        c.close_phase(scope=SCOPE, ended_at_ms=0,
                      retired={k: True for k in c.RETIREMENT}, canonical=REF,
                      provider_finalization="unknown")
    clock.ms = 900001
    with pytest.raises(ContractError, match="controller_wall_deadline"):
        ready(c)


@pytest.mark.parametrize("field", MemoryReplayController.CLEANUP)
def test_cleanup_audits_all_retained_resources(field):
    c, clock = controller()
    c.abort("not_launched")
    checks = {k: True for k in c.CLEANUP}
    checks[field] = False
    with pytest.raises(ContractError, match="cleanup_audit_incomplete"):
        c.cleanup(scope=SCOPE, checks=checks, evidence=REF, started_at_ms=clock.ms)


@pytest.mark.parametrize("path", ["/absolute", "../escape", "a/../b", "a//b", r"a\b", ""])
def test_evidence_receipt_requires_bounded_relative_artifact_reference(path):
    with pytest.raises(ContractError):
        EvidenceRef(path, HASH, "/selector")


def test_plan_hash_is_canonical_contract_not_original_manifest_hash():
    c, _ = controller()
    assert c.plan_sha256 == digest(PLAN)
    assert c.plan_sha256 != PLAN["provenance"]["reference_c02a_manifest_sha256"]
    assert c.plan_sha256 != HASH


def test_closed_controller_cannot_reopen_cleanup_or_admissions():
    c, clock = controller()
    c.abort("not_launched")
    c.cleanup(scope=SCOPE, checks={k: True for k in c.CLEANUP},
              evidence=REF, started_at_ms=clock.ms)
    with pytest.raises(ContractError, match="already_closed"):
        c.abort("try_again")
    with pytest.raises(ContractError, match="admission_not_allowed"):
        reserve(c)


def test_source_and_retrieval_receipts_are_bounded():
    c, clock = controller()
    reserve(c)
    ready(c)
    for i in range(256):
        publish(c, clock, source_id=f"source-{i}")
    with pytest.raises(ContractError, match="source_receipt_budget"):
        publish(c, clock, source_id="one-too-many")
    close(c, clock)
    reserve(c)
    ready(c)
    for _ in range(16):
        retrieve(c, clock, ("source-0", 1, 1))
    with pytest.raises(ContractError, match="retrieval_receipt_budget"):
        retrieve(c, clock, ("source-0", 1, 1))

