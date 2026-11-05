"""Offline doubles only; no canonical admission, SQL, native or model evidence.

Real private-file IO qualifies checkpoint mechanics. Supplied source records and
evidence references are controlled inputs, not a claim that frontend time is wired.
"""
from dataclasses import replace
from datetime import UTC, datetime, timedelta
import hashlib
import json
from uuid import uuid4

import pytest

from app.knowledge.models import SourceVersion
from evals.responsive_voice import native_memory_domain_clock as module
from evals.responsive_voice.native_memory_domain_clock import Checkpoint, EvalDomainClock
from evals.responsive_voice.native_memory_replay_control import ContractError, EvidenceRef, MemoryReplayController
import test_native_memory_replay_control as replay_tests


SCOPE = replay_tests.SCOPE
EPOCH = datetime(2026, 8, 1, 12, tzinfo=UTC)
MANIFEST = b'{"format":"offline-controlled-manifest","phase":"calendar"}'
REF = EvidenceRef("offline-double.json", "a" * 64, "/controlled/join")


@pytest.fixture
def clock(tmp_path):
    return EvalDomainClock.create(root=tmp_path.resolve(), scope=SCOPE, manifest_bytes=MANIFEST, epoch=EPOCH)


def advance(clock, day, **changes):
    args = dict(day=day, scope=SCOPE, manifest_sha256=clock.manifest_sha256,
                previous=clock.checkpoint, phase_id=f"phase-{clock.checkpoint.sequence}",
                session_id=uuid4(), retired={key: True for key in MemoryReplayController.RETIREMENT}, evidence=REF)
    args.update(changes)
    return clock.advance_after_join(**args)


def source(**changes):
    value = SourceVersion(owner_id=SCOPE.owner_id, area_id=SCOPE.area_id, source_id=uuid4(),
        revision=1, generation=1, payload_id=uuid4(), document_id="controlled-document",
        bank_name=SCOPE.memory_bank_id, content_digest="b" * 64, source_type="observation",
        trust="derived", source_time=EPOCH - timedelta(hours=2), time_basis="controlled original time",
        recorded_at=replay_tests.WALL, content={"finding": "controlled original"},
        is_current=True, ingestion_status="searchable")
    return value.model_copy(update=changes)


def restore(clock, **changes):
    args = dict(root=clock.root, scope=SCOPE, manifest_bytes=MANIFEST, epoch=EPOCH,
                checkpoint=clock.checkpoint)
    args.update(changes)
    return EvalDomainClock.restore(**args)


def test_real_checkpoint_roundtrip_days_months_and_immutable_predecessor(clock):
    original = source()
    untouched = original.model_dump()
    stamp = clock.bind_source(original, source_day=0, source_pts_ms=1000, evidence=REF)
    first = clock.checkpoint
    first_bytes = (clock.root / first.name).read_bytes()
    for day in (1, 31, 61):
        checkpoint = advance(clock, day)
        clock = restore(clock)
        assert clock.day == day and clock.checkpoint == checkpoint
        result = clock.logical_age(stamp, current_source=original, phase_pts_ms=1000,
                                   max_age_seconds=30 * 86400)
        assert result.age_seconds == day * 86400
        assert result.eligible is (day < 30)
        assert result.domain_as_of == EPOCH + timedelta(days=day, seconds=1)
    assert original.model_dump() == untouched
    assert (clock.root / first.name).read_bytes() == first_bytes
    assert (clock.root / clock.checkpoint.name).stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize("flag", MemoryReplayController.RETIREMENT)
def test_each_unjoined_boundary_rejected_without_advance(clock, flag):
    retired = {key: True for key in MemoryReplayController.RETIREMENT}
    retired[flag] = False
    before = clock.checkpoint
    with pytest.raises(ContractError, match="phase_not_joined"):
        advance(clock, 1, retired=retired)
    assert clock.day == 0 and clock.checkpoint == before
    assert not (clock.root / "domain-clock-001.json").exists()


@pytest.mark.parametrize("variant", ["missing", "extra", "truthy_not_true"])
def test_join_flags_are_exact_not_truthy(clock, variant):
    retired = {key: True for key in MemoryReplayController.RETIREMENT}
    if variant == "missing":
        retired.pop("host_joined")
    elif variant == "extra":
        retired["provider_ack"] = True
    else:
        retired["host_joined"] = 1
    with pytest.raises(ContractError, match="phase_not_joined"):
        advance(clock, 1, retired=retired)


@pytest.mark.parametrize("day", [-1, 0, 91, True, 1.5])
def test_invalid_logical_advance_is_not_persisted(clock, day):
    with pytest.raises(ContractError):
        advance(clock, day)
    assert clock.day == 0


def test_repeated_boundary_and_stale_checkpoint_cannot_advance_twice(clock):
    first = clock.checkpoint
    session = uuid4()
    advance(clock, 1, phase_id="closed-a", session_id=session)
    with pytest.raises(ContractError, match="stale_checkpoint"):
        advance(clock, 2, previous=first)
    for changes in ({"phase_id": "closed-a"}, {"session_id": session}):
        with pytest.raises(ContractError, match="retirement_reused"):
            advance(clock, 2, **changes)
    with pytest.raises(ContractError, match="later_checkpoint_exists"):
        restore(clock, checkpoint=first)
    assert clock.day == 1


@pytest.mark.parametrize("change", ["owner", "area", "bank", "manifest", "epoch"])
def test_replaced_binding_cannot_restore_or_advance(clock, change):
    changed_scope = SCOPE
    changes = {}
    if change == "owner":
        changed_scope = replace(SCOPE, owner_id=SCOPE.owner_id[:-1] + "b")
    elif change == "area":
        changed_scope = replace(SCOPE, area_id=str(uuid4()))
    elif change == "bank":
        changed_scope = replace(SCOPE, memory_bank_id="jarvis_" + "3" * 32)
    elif change == "manifest":
        changes["manifest_bytes"] = MANIFEST + b" "
    else:
        changes["epoch"] = EPOCH + timedelta(days=1)
    with pytest.raises(ContractError, match="checkpoint_binding_changed"):
        restore(clock, scope=changed_scope, **changes)
    if change != "epoch":
        with pytest.raises(ContractError, match="boundary_scope_or_manifest_changed"):
            advance(clock, 1, scope=changed_scope,
                    manifest_sha256="f" * 64 if change == "manifest" else clock.manifest_sha256)


@pytest.mark.parametrize("change", [
    "unavailable", "historical", "revision", "generation", "payload_id", "document_id",
    "content_digest", "source_time", "owner_id", "area_id", "bank_name", "source_id",
])
def test_correction_deletion_permission_loss_and_replacement_reject(clock, change):
    original = source()
    stamp = clock.bind_source(original, source_day=0, source_pts_ms=0, evidence=REF)
    values = {"historical": ("is_current", False), "revision": ("revision", 2),
        "generation": ("generation", 2), "payload_id": ("payload_id", uuid4()),
        "document_id": ("document_id", "replacement"), "content_digest": ("content_digest", "c" * 64),
        "source_time": ("source_time", EPOCH), "owner_id": ("owner_id", "foreign-owner"),
        "area_id": ("area_id", uuid4()), "bank_name": ("bank_name", "jarvis_" + "4" * 32),
        "source_id": ("source_id", uuid4())}
    current = None if change == "unavailable" else original.model_copy(update=dict([values[change]]))
    result = clock.logical_age(stamp, current_source=current, phase_pts_ms=0, max_age_seconds=10)
    assert not result.eligible and result.age_seconds is None
    assert result.reason in {"source_unavailable", "source_not_current", "source_version_changed", "source_scope_changed"}


def test_new_source_after_jump_is_fresh_without_reviving_old_source(clock):
    old = source()
    stamp = clock.bind_source(old, source_day=0, source_pts_ms=0, evidence=REF)
    advance(clock, 31)
    current = source()
    fresh = clock.bind_source(current, source_day=31, source_pts_ms=0, evidence=REF)
    assert not clock.logical_age(stamp, current_source=old, phase_pts_ms=0, max_age_seconds=10).eligible
    assert clock.logical_age(fresh, current_source=current, phase_pts_ms=9999, max_age_seconds=10).eligible
    assert not clock.logical_age(fresh, current_source=current, phase_pts_ms=10000, max_age_seconds=10).eligible
    progressed = current.model_copy(update={"ingestion_status": "accepted"})
    assert clock.logical_age(fresh, current_source=progressed, phase_pts_ms=0, max_age_seconds=10).eligible
    # Logical eligibility intentionally makes no searchable/index-readiness claim.


def test_source_stamp_cannot_cross_scope_manifest_or_epoch(clock, tmp_path):
    original = source()
    stamp = clock.bind_source(original, source_day=0, source_pts_ms=0, evidence=REF)
    for index, changes in enumerate(({"scope": replace(SCOPE, area_id=str(uuid4()))},
                                    {"manifest_bytes": MANIFEST + b" "},
                                    {"epoch": EPOCH + timedelta(days=1)})):
        root = tmp_path / f"other-{index}"
        root.mkdir(mode=0o700)
        args = dict(root=root.resolve(), scope=SCOPE, manifest_bytes=MANIFEST, epoch=EPOCH)
        args.update(changes)
        other = EvalDomainClock.create(**args)
        result = other.logical_age(stamp, current_source=original, phase_pts_ms=0, max_age_seconds=10)
        assert not result.eligible and result.reason == "source_binding_changed"


def test_future_source_mapping_and_missing_evidence_rejected(clock):
    with pytest.raises(ContractError, match="future_source_day"):
        clock.bind_source(source(), source_day=1, source_pts_ms=0, evidence=REF)
    with pytest.raises(ContractError, match="source_evidence_required"):
        clock.bind_source(source(), source_day=0, source_pts_ms=0, evidence=None)
    original = source()
    stamp = clock.bind_source(original, source_day=0, source_pts_ms=1000, evidence=REF)
    result = clock.logical_age(stamp, current_source=original, phase_pts_ms=999, max_age_seconds=10)
    assert not result.eligible and result.reason == "source_domain_time_in_future"


@pytest.mark.parametrize("limit", [0, -1, True, float("nan"), float("inf")])
def test_logical_age_budget_is_finite(clock, limit):
    original = source()
    stamp = clock.bind_source(original, source_day=0, source_pts_ms=0, evidence=REF)
    with pytest.raises(ContractError, match="invalid_logical_age_bound"):
        clock.logical_age(stamp, current_source=original, phase_pts_ms=0, max_age_seconds=limit)


def test_domain_advance_does_not_extend_existing_real_authorization_or_deadline(clock):
    control, operational = replay_tests.controller()
    advance(clock, 90)
    assert operational.wall == replay_tests.WALL and operational.ms == 0
    operational.wall += timedelta(days=1)
    with pytest.raises(ContractError, match="authorization_real_wall_expired_or_short"):
        replay_tests.reserve(control)
    operational.wall = replay_tests.WALL
    replay_tests.reserve(control)
    operational.ms = -1
    with pytest.raises(ContractError, match="monotonic_reversed"):
        replay_tests.ready(control)
    operational.ms = control.plan["max_wall_seconds"] * 1000 + 1
    with pytest.raises(ContractError, match="controller_wall_deadline"):
        replay_tests.ready(control)
    assert clock.day == 90


def test_checkpoint_budget_matches_existing_six_phase_contract(clock):
    for day in range(1, 6):
        advance(clock, day)
    with pytest.raises(ContractError, match="checkpoint_sequence"):
        advance(clock, 6)
    assert clock.day == 5


def test_second_writer_cannot_overwrite_newer_checkpoint(clock):
    other = restore(clock)
    advance(clock, 1)
    with pytest.raises(FileExistsError):
        advance(other, 2)
    with pytest.raises(ContractError, match="checkpoint_write_uncertain"):
        other.logical_now(phase_pts_ms=0)
    assert restore(clock).day == 1


@pytest.mark.parametrize("written", [False, True])
def test_failed_or_unacknowledged_write_fences_without_automatic_retry(clock, monkeypatch, written):
    original = clock.checkpoint
    write = module.write_private

    def fail(*args, **kwargs):
        if written:
            write(*args, **kwargs)
        raise OSError("controlled checkpoint failure")

    monkeypatch.setattr(module, "write_private", fail)
    with pytest.raises(OSError, match="controlled checkpoint failure"):
        advance(clock, 1)
    assert clock.day == 0 and clock.checkpoint == original
    with pytest.raises(ContractError, match="checkpoint_write_uncertain"):
        advance(clock, 1)
    if written:
        with pytest.raises(ContractError, match="later_checkpoint_exists"):
            restore(clock)
    else:
        assert restore(clock).day == 0


def test_changed_checkpoint_bytes_and_symlink_fail_closed(clock):
    path = clock.root / clock.checkpoint.name
    raw = path.read_bytes()
    path.write_bytes(raw + b" ")
    with pytest.raises(ContractError, match="checkpoint_digest_changed"):
        restore(clock)
    target = clock.root / "controlled-target.json"
    target.write_bytes(raw)
    target.chmod(0o600)
    path.unlink()
    path.symlink_to(target)
    with pytest.raises(OSError):
        restore(clock)


def test_duplicate_checkpoint_keys_are_rejected_even_with_matching_hash(clock):
    path = clock.root / clock.checkpoint.name
    raw = path.read_bytes()
    raw = b'{"day":0,' + raw[1:]
    path.write_bytes(raw)
    reference = Checkpoint(0, hashlib.sha256(raw).hexdigest())
    with pytest.raises(ContractError, match="duplicate_checkpoint_key"):
        restore(clock, checkpoint=reference)


def test_checkpoint_contains_only_bound_time_and_retirement_metadata(clock):
    advance(clock, 1)
    document = json.loads((clock.root / clock.checkpoint.name).read_bytes())
    assert set(document) == {"format", "binding", "sequence", "day", "previous_sha256", "boundaries"}
    assert document["binding"] == {"scope": module.asdict(SCOPE),
        "manifest_sha256": hashlib.sha256(MANIFEST).hexdigest(), "epoch": EPOCH.isoformat()}
    assert "finding" not in json.dumps(document)


@pytest.mark.parametrize("manifest", [b"", "not-bytes", b"x" * (module.MAX_MANIFEST_BYTES + 1)])
def test_manifest_input_is_bounded_private_bytes(tmp_path, manifest):
    with pytest.raises(ContractError, match="manifest_bytes_required"):
        EvalDomainClock.create(root=tmp_path.resolve(), scope=SCOPE, manifest_bytes=manifest, epoch=EPOCH)
