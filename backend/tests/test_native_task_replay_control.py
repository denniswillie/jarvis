"""Offline app-observation/release validation, never native task acceptance."""
from datetime import UTC, datetime, timedelta
import hashlib
import json
import os
from types import SimpleNamespace
from uuid import uuid4

import pytest

from evals.responsive_voice.native_task_barrier import BarrierReceipt
from evals.responsive_voice.native_task_release_control import (
    ClaimIdentity, HoldReceipt, create_control, publish_release, receipt_name, write_private,
)
from evals.responsive_voice.native_task_replay_control import (
    NativeTaskReplayRelease, NativeTaskScope, ReplayControlError,
)


@pytest.fixture
def case(tmp_path):
    root = tmp_path.resolve()
    worker, controls = root / "worker", root / "controls"
    worker.mkdir(mode=0o700)
    controls.mkdir(mode=0o700)
    now = datetime.now(UTC)
    run, barrier, session, task, attempt = (uuid4() for _ in range(5))
    owner = f"direct-image-eval-second-brain-{run.hex}-a"
    schema = f"jarvis_sb_eval_{run.hex}_0"
    host = f"native-task-{run.hex}-0123456789ab"
    worker_id = "frontier-" + uuid4().hex
    scope = NativeTaskScope(run_id=run, barrier_id=barrier, owner_id=owner,
        database_schema=schema, worker_id=worker_id, runtime_host_id=host,
        model="gpt-5.6-sol", host_expires_at=now + timedelta(seconds=120))
    scope_bytes = scope.model_dump_json().encode()
    manifest = {"version": 1, "run_id": run.hex, "owner_id": owner, "voice_route": "realtime",
        "task_barrier": {"version": 1, "scope_file": "task-scope.json", "request_input_ordinal": 0},
        "utterances": [
            {"trigger": {"kind": "at_time", "at_seconds": 7}},
            {"trigger": {"kind": "after_task_claim", "after_input_ordinal": 0, "timeout_seconds": 30}},
            {"trigger": {"kind": "response_output_start", "require_task_claim": True}},
            {"trigger": {"kind": "after_input_complete", "after_input_ordinal": 2, "delay_seconds": 1}},
        ]}
    manifest_bytes = json.dumps(manifest).encode()
    config = create_control(worker, run_id=run, owner_id=owner, database_schema=schema,
        worker_host_id=host, model=scope.model, expires_at=now + timedelta(seconds=100), max_hold_seconds=30)
    identity = ClaimIdentity(session_id=session, task_id=task, task_revision=1,
        attempt_id=attempt, fence=1, worker_id=worker_id, worker_pid=os.getpid())
    held = HoldReceipt(scope=config.scope, identity=identity, hold_id=uuid4(), phase="entered",
        entered_at=now - timedelta(seconds=1), updated_at=now, expires_at=now + timedelta(seconds=29))
    write_private(config.control_root, receipt_name(identity), held.model_dump_json().encode())
    wire = {"format": "jarvis.native_task_barrier.v1",
        "scope": {**scope.model_dump(mode="json", exclude={"format"}), "session_id": str(session)},
        "admission": {"owner_id": owner, "session_id": str(session), "request_event_id": str(uuid4()),
            "request_revision": 1, "request_id": str(uuid4()), "task_id": str(task), "accepted_revision": 1,
            "parameters_digest": "a" * 64, "provider_call_id": "actual-call-fixture", "operation": "create"},
        "sequence": 5, "observed_at": (now - timedelta(seconds=.2)).isoformat(),
        "expires_at": (now + timedelta(seconds=1.8)).isoformat(), "status": "eligible",
        "reason": "active_claim_observed", "claim": {"task_id": str(task), "task_revision": 1,
            "attempt_id": str(attempt), "attempt_number": 1, "fence": 1,
            "worker_id": worker_id, "runtime_host_id": host,
            "lease_until": (now + timedelta(seconds=20)).isoformat(),
            "worker_expires_at": (now + timedelta(seconds=10)).isoformat(),
            "task_status": "running", "attempt_status": "running", "provider_state": "live"}}
    current = BarrierReceipt.model_validate_json(json.dumps(wire))
    value = SimpleNamespace(now=now, root=controls, config=config, held=held, current=current,
        manifest=manifest, manifest_bytes=manifest_bytes, scope_bytes=scope_bytes, scope=scope, calls=[])
    bridge = SimpleNamespace(scope=current.scope, admission=current.admission, claim=current.claim,
        exchange=SimpleNamespace(read=lambda: value.current.model_dump_json().encode()))
    readiness = {"status": "ready", "registry_ready": True, "run_id": run.hex,
        "owner_id": owner, "database_schema": schema, "model": scope.model,
        "worker_id": worker_id, "host_id": host, "host_expires_at": scope.host_expires_at.isoformat(),
        "pid": os.getpid()}
    binding = {"format": "jarvis.native_task_session_binding.v1", "run_id": str(run),
        "barrier_id": str(barrier), "owner_id": owner, "session_id": str(session),
        "manifest_file_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "scope_file_sha256": hashlib.sha256(scope_bytes).hexdigest(),
        "recorded_at": (now - timedelta(seconds=.5)).isoformat()}
    completion = {**binding, "format": "jarvis.native_task_stimulus_completion.v1",
        "recorded_at": (now - timedelta(seconds=.1)).isoformat(), "inputs": [
            {"input_ordinal": i, "status": "sdk_input_complete", "sample_count": 48000,
             "accepted_samples": 48000, "first_sdk_callback_monotonic": 100.0 + i * 2,
             "last_sdk_callback_end_monotonic": 101.0 + i * 2,
             "task_claim_sequence_at_enqueue": 3 if i in (1, 2) else None} for i in range(4)]}
    value.binding, value.completion, value.bridge, value.readiness = binding, completion, bridge, readiness

    def write(name, data):
        write_private(controls, name, json.dumps(data).encode())

    def publisher(*args, **kwargs):
        value.calls.append(kwargs["command_id"])
        return publish_release(*args, **kwargs)

    value.write = write
    value.arguments = dict(observer_host=bridge, worker_readiness=readiness, control_root=controls,
        manifest_bytes=manifest_bytes, scope_bytes=scope_bytes, input_sample_counts=(48000,) * 4,
        hold_config=config, publisher=publisher)
    value.controller = NativeTaskReplayRelease(**value.arguments)
    write("session-binding.json", binding)
    write("stimulus-completion.json", completion)
    return value


def simulate_worker_ack(case, command_id):
    held = case.held.model_copy(update={"phase": "released", "command_id": command_id,
        "updated_at": case.now})
    write_private(case.config.control_root, receipt_name(held.identity), held.model_dump_json().encode())


def test_complete_inputs_request_release_once_and_ack_can_follow_task_settlement(case):
    controller = case.controller
    assert controller.step(now=case.now)["status"] == "release_requested"
    controller.step(now=case.now)
    assert len(case.calls) == 1
    case.current = case.current.model_copy(update={"status": "blocked", "reason": "canonical_unavailable", "claim": None})
    simulate_worker_ack(case, case.calls[0])
    result = controller.step(now=case.now)
    assert result["status"] == "released"
    assert result["semantic_acceptance"] == "unmeasured"
    controller.step(now=case.now)
    assert len(case.calls) == 1


@pytest.mark.parametrize("name", ["session-binding.json", "stimulus-completion.json"])
def test_missing_app_observation_waits_without_release(case, name):
    (case.root / name).unlink()
    assert case.controller.step(now=case.now)["status"] == "waiting"
    assert not case.calls


@pytest.mark.parametrize("field,value", [
    ("owner_id", "foreign-owner"), ("run_id", str(uuid4())), ("barrier_id", str(uuid4())),
    ("session_id", str(uuid4())), ("manifest_file_sha256", "b" * 64), ("scope_file_sha256", "c" * 64),
])
def test_foreign_or_changed_app_binding_is_rejected(case, field, value):
    case.binding[field] = value
    case.write("session-binding.json", case.binding)
    assert case.controller.step(now=case.now)["status"] == "failed"
    assert not case.calls


@pytest.mark.parametrize("change", ["missing", "duplicate", "count", "partial", "overlap", "no_claim", "future_claim"])
def test_incomplete_or_unbound_stimuli_never_release(case, change):
    rows = case.completion["inputs"]
    if change == "missing":
        rows.pop()
    elif change == "duplicate":
        rows[2]["input_ordinal"] = 1
    elif change == "count":
        rows[2].update(sample_count=47000, accepted_samples=47000)
    elif change == "partial":
        rows[2]["accepted_samples"] = 47999
    elif change == "overlap":
        rows[2]["first_sdk_callback_monotonic"] = rows[1]["first_sdk_callback_monotonic"]
    elif change == "no_claim":
        rows[2]["task_claim_sequence_at_enqueue"] = None
    else:
        rows[2]["task_claim_sequence_at_enqueue"] = 6
    case.write("stimulus-completion.json", case.completion)
    assert case.controller.step(now=case.now)["status"] == "failed"
    assert not case.calls


@pytest.mark.parametrize("kind", ["blocked", "expired", "replaced"])
def test_noncurrent_canonical_claim_does_not_release(case, kind):
    if kind == "blocked":
        case.current = case.current.model_copy(update={"status": "blocked", "claim": None, "reason": "canonical_unavailable"})
    elif kind == "expired":
        case.current = case.current.model_copy(update={"observed_at": case.now - timedelta(seconds=3),
            "expires_at": case.now - timedelta(seconds=1)})
    else:
        case.current = case.current.model_copy(update={"claim": case.current.claim.model_copy(update={"attempt_id": uuid4()})})
    assert case.controller.step(now=case.now)["status"] == "waiting"
    assert not case.calls


def test_observer_has_not_committed_a_claim_yet(case):
    case.bridge.claim = None
    assert case.controller.step(now=case.now)["reason"] == "canonical_claim_pending"
    assert not case.calls


def test_result_not_yet_returned_waits(case):
    (case.config.control_root / receipt_name(case.held.identity)).unlink()
    assert case.controller.step(now=case.now)["status"] == "waiting"
    assert not case.calls


def test_foreign_held_attempt_worker_identity_rejected(case):
    changed = case.held.model_copy(update={"identity": case.held.identity.model_copy(update={"worker_pid": os.getpid() + 1})})
    write_private(case.config.control_root, receipt_name(changed.identity), changed.model_dump_json().encode())
    assert case.controller.step(now=case.now)["reason"] == "held_claim_mismatch"
    assert not case.calls


def test_unknown_release_write_is_not_retried(case):
    original = case.arguments["publisher"]

    def uncertain(*args, **kwargs):
        original(*args, **kwargs)
        raise OSError("simulated_post_write_failure")

    controller = NativeTaskReplayRelease(**{**case.arguments, "publisher": uncertain})
    assert controller.step(now=case.now)["reason"] == "release_write_outcome_unknown"
    controller.step(now=case.now)
    assert len(case.calls) == 1
    simulate_worker_ack(case, case.calls[0])
    assert controller.step(now=case.now)["status"] == "released"


def test_unowned_ack_is_not_accepted(case):
    case.controller.step(now=case.now)
    simulate_worker_ack(case, uuid4())
    assert case.controller.step(now=case.now)["status"] == "failed"


def test_expiry_prevents_new_release(case):
    assert case.controller.step(now=case.scope.host_expires_at)["reason"] == "replay_scope_expired"
    assert not case.calls


def test_future_completion_timestamp_rejected(case):
    case.completion["recorded_at"] = (case.now + timedelta(seconds=1)).isoformat()
    case.write("stimulus-completion.json", case.completion)
    assert case.controller.step(now=case.now)["status"] == "failed"


def test_missing_task_gate_is_not_enabled_implicitly(case):
    manifest = {**case.manifest, "utterances": [{"trigger": {"kind": "at_time", "at_seconds": i}} for i in range(4)]}
    with pytest.raises(ReplayControlError, match="invalid_task_manifest"):
        NativeTaskReplayRelease(**{**case.arguments, "manifest_bytes": json.dumps(manifest).encode()})


def test_requested_worker_cannot_substitute_for_actual_readiness(case):
    readiness = {**case.readiness, "registry_ready": False}
    with pytest.raises(ReplayControlError, match="worker_scope_mismatch"):
        NativeTaskReplayRelease(**{**case.arguments, "worker_readiness": readiness})
