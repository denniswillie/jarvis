"""Offline canonical-adapter and private-file tests, not SQL/native qualification."""

import asyncio
import json
import os
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from pydantic import ValidationError

from app.tasks.models import TaskVoiceSource
from evals.responsive_voice.native_task_barrier import (
    MAX_RECEIPT_BYTES, AdmissionWitness, AtomicReceiptFile, BarrierError, BarrierReceipt,
    CanonicalTaskObserver, FixtureScope, ReceiptConsumer, SNAPSHOT_SQL, decode_receipt,
    witness_from_observation,
)


NOW = datetime(2026, 10, 31, 12, tzinfo=UTC)


@pytest.fixture
def case():
    run = uuid4()
    scope = FixtureScope(run_id=run, barrier_id=uuid4(), owner_id="isolated-owner-a",
        session_id=uuid4(), database_schema=f"jarvis_sb_eval_{run.hex}_0",
        worker_id="frontier-fixture", runtime_host_id="native-task-fixture", model="gpt-5.6-sol",
        host_expires_at=NOW + timedelta(seconds=60))
    witness = AdmissionWitness(owner_id=scope.owner_id, session_id=scope.session_id,
        request_event_id=uuid4(), request_revision=2, request_id=uuid4(), task_id=uuid4(),
        accepted_revision=1, parameters_digest="a" * 64, provider_call_id="real-call-observation")
    attempt = uuid4()
    row = dict(task_id=witness.task_id, owner_id=scope.owner_id, session_id=scope.session_id,
        request_event_id=witness.request_event_id, request_revision=2, revision=1, status="running",
        pending_status=None, result_payload_id=None, current_attempt_id=attempt, fence=4,
        backend_kind="frontier", model=scope.model, candidate_count=1, candidate_matches=True,
        committed_request_id=witness.request_id, operation="create", accepted_revision=1,
        parameters_digest=witness.parameters_digest, source_kind="user_transcript", source_trust="user_input",
        source_finality="final", source_revision=2, source_session_id=scope.session_id,
        source_availability="available", source_current=True, attempt_id=attempt, task_revision=1,
        attempt_number=1, attempt_fence=4, worker_id=scope.worker_id, runtime_host_id=scope.runtime_host_id,
        attempt_status="running", provider_state="live", lease_until=NOW + timedelta(seconds=30),
        ended_at=None, registered_host_id=scope.runtime_host_id, registered_owner_id=scope.owner_id,
        registered_model=scope.model, draining=False, worker_expires_at=NOW + timedelta(seconds=20))
    return scope, witness, row


class Pool:
    def __init__(self, scope, row):
        self.schema, self.row = scope.database_schema, row
        self.borrowed = False
        self.transactions, self.queries = [], []
        self.released = 0

    @asynccontextmanager
    async def acquire(self):
        try:
            yield self
        finally:
            self.released += 1

    def is_in_transaction(self):
        return self.borrowed

    @asynccontextmanager
    async def transaction(self, **kwargs):
        self.transactions.append(kwargs)
        yield

    async def fetchval(self, sql):
        assert sql == "select current_schema()"
        return self.schema

    async def fetchrow(self, sql, *args):
        assert sql == SNAPSHOT_SQL
        self.queries.append(args)
        return self.row


def observer(case, **kwargs):
    scope, witness, row = case
    pool = Pool(scope, row)
    return CanonicalTaskObserver(pool, scope, witness, ttl_seconds=2.0, clock=lambda: NOW, **kwargs), pool


def observe(subject):
    return asyncio.run(subject.observe())


def wire(receipt):
    return receipt.model_dump_json().encode()


def test_real_field_bindings_committed_snapshot_and_immutable_roundtrip(case):
    subject, pool = observer(case)
    result = observe(subject)
    assert result.status == "eligible"
    assert result.claim.task_revision == 1 and result.admission.request_revision == 2
    assert result.claim.fence == 4
    assert decode_receipt(wire(result)) == result
    assert pool.transactions == [dict(isolation="repeatable_read", readonly=True)]
    assert pool.released == 1
    assert pool.queries == [(case[0].owner_id, case[0].session_id, case[1].request_event_id,
                             case[1].request_revision, case[1].task_id, case[1].request_id)]
    with pytest.raises(ValidationError):
        result.sequence = 9


def test_handler_observation_does_not_claim_commit_or_tool_return_delivery(case):
    scope, witness, row = case
    source = TaskVoiceSource(witness.request_event_id, witness.request_revision, scope.session_id)
    record = dict(owner_id=scope.owner_id, operation="create", request_event_id=str(witness.request_event_id),
        request_revision=witness.request_revision, request_id=str(witness.request_id), task_id=str(witness.task_id),
        accepted_revision=witness.accepted_revision, parameters_digest=witness.parameters_digest,
        voice_source=source.parameters(), call_id=witness.provider_call_id, committed=True)
    actual = witness_from_observation(scope, record,
        [dict(tool="start_background_task", call_id=witness.provider_call_id)])
    assert actual == witness
    subject, pool = observer((scope, actual, None))
    assert observe(subject).reason == "admission_not_committed"
    pool.row = row
    assert observe(subject).status == "eligible"
    for change in ({"owner_id": "other"}, {"operation": "update"}, {"voice_source": {}}, {"call_id": "other"}):
        with pytest.raises(BarrierError, match="invalid_handler_witness"):
            witness_from_observation(scope, {**record, **change},
                [dict(tool="start_background_task", call_id=witness.provider_call_id)])


@pytest.mark.parametrize("field,value,reason", [
    ("committed_request_id", None, "admission_not_committed"),
    ("owner_id", "other", "admission_mismatch"),
    ("session_id", uuid4(), "admission_mismatch"),
    ("parameters_digest", "b" * 64, "admission_mismatch"),
    ("operation", "update", "admission_mismatch"),
    ("request_event_id", uuid4(), "source_ineligible"),
    ("request_revision", 3, "source_ineligible"),
    ("source_revision", 3, "source_ineligible"),
    ("source_session_id", uuid4(), "source_ineligible"),
    ("source_finality", "provisional", "source_ineligible"),
    ("source_trust", "assistant_output", "source_ineligible"),
    ("source_availability", "removed", "source_ineligible"),
    ("source_current", False, "source_replaced"),
    ("candidate_count", 2, "ambiguous_tasks"),
    ("candidate_matches", False, "ambiguous_tasks"),
    ("revision", 2, "task_revision_replaced"),
    ("status", "queued", "task_not_active"),
    ("status", "succeeded", "task_not_active"),
    ("status", "cancel_requested", "task_not_active"),
    ("status", "cancelled", "task_not_active"),
    ("status", "paused", "task_not_active"),
    ("status", "outcome_unknown", "task_not_active"),
    ("pending_status", "cancelled", "task_not_active"),
    ("result_payload_id", uuid4(), "task_not_active"),
    ("attempt_id", None, "claim_ineligible"),
    ("task_revision", 2, "claim_ineligible"),
    ("attempt_fence", 5, "claim_ineligible"),
    ("attempt_status", "recovering", "claim_ineligible"),
    ("ended_at", NOW, "claim_ineligible"),
    ("worker_id", "other", "worker_ineligible"),
    ("runtime_host_id", "other", "worker_ineligible"),
    ("registered_owner_id", "other", "worker_ineligible"),
    ("registered_model", "other", "worker_ineligible"),
    ("draining", True, "worker_ineligible"),
    ("lease_until", NOW, "lease_expired"),
    ("worker_expires_at", NOW, "lease_expired"),
])
def test_ineligible_canonical_states_fail_prerequisite(case, field, value, reason):
    case[2][field] = value
    subject, _ = observer(case)
    result = observe(subject)
    assert result.status == "blocked" and result.claim is None and result.reason == reason


def test_repeated_observation_deduplicates_identity_but_replaced_attempt_is_blocked(case):
    subject, pool = observer(case)
    first, second = observe(subject), observe(subject)
    assert first.claim.identity() == second.claim.identity()
    assert (first.sequence, second.sequence) == (1, 2)
    pool.row.update(current_attempt_id=uuid4(), attempt_number=2, fence=5, attempt_fence=5)
    pool.row["attempt_id"] = pool.row["current_attempt_id"]
    assert observe(subject).reason == "claim_replaced"


def test_receipt_expiry_is_clamped_to_attempt_worker_and_host(case):
    subject, pool = observer(case)
    pool.row["lease_until"] = NOW + timedelta(seconds=1)
    assert observe(subject).expires_at == pool.row["lease_until"]
    pool.row["worker_expires_at"] = NOW + timedelta(milliseconds=500)
    assert observe(subject).expires_at == pool.row["worker_expires_at"]
    subject.scope = case[0].model_copy(update={"host_expires_at": NOW + timedelta(milliseconds=250)})
    assert observe(subject).expires_at == subject.scope.host_expires_at
    subject.clock = lambda: NOW + timedelta(seconds=2)
    with pytest.raises(BarrierError, match="host_or_lease_expired"):
        observe(subject)


@pytest.mark.parametrize("mode,reason", [("schema", "schema_mismatch"), ("borrowed", "borrowed_transaction")])
def test_public_schema_and_borrowed_transactions_never_observe(case, mode, reason):
    subject, pool = observer(case)
    if mode == "schema":
        pool.schema = "public"
    else:
        pool.borrowed = True
    assert observe(subject).reason == reason
    assert not pool.queries and pool.released == 1


def test_canonical_failure_redacted_and_cancellation_propagates(case):
    subject, pool = observer(case)
    async def failing(*_):
        raise RuntimeError("secret DSN and user content")
    pool.fetchrow = failing
    result = observe(subject)
    assert result.reason == "canonical_unavailable" and b"secret" not in wire(result)
    async def cancelled(*_):
        raise asyncio.CancelledError()
    pool.fetchrow = cancelled
    with pytest.raises(asyncio.CancelledError):
        observe(subject)
    assert subject.sequence == 1 and not subject._observing and pool.released == 2


def test_consumer_replay_wrong_scope_stale_future_and_reconnect_watermark(case):
    subject, _ = observer(case)
    first, second = observe(subject), observe(subject)
    consumer = ReceiptConsumer(case[0])
    assert consumer.consume(wire(first), now=NOW) == first
    with pytest.raises(BarrierError, match="receipt_replayed"):
        consumer.consume(wire(first), now=NOW)
    resumed = ReceiptConsumer(case[0], after_sequence=consumer.sequence,
                              admission=consumer.admission, claim=consumer.claim)
    assert resumed.consume(wire(second), now=NOW) == second
    for when in (NOW - timedelta(microseconds=1), second.expires_at):
        with pytest.raises(BarrierError, match="receipt_expired_or_future"):
            ReceiptConsumer(case[0]).consume(wire(second), now=when)
    wrong = case[0].model_copy(update={"session_id": uuid4()})
    with pytest.raises(BarrierError, match="receipt_scope_mismatch"):
        ReceiptConsumer(wrong).consume(wire(second), now=NOW)


def test_blocked_receipt_withdraws_eligibility_and_preserves_watermark(case):
    subject, pool = observer(case)
    consumer = ReceiptConsumer(case[0])
    first = observe(subject)
    consumer.consume(wire(first), now=NOW)
    pool.row["status"] = "cancel_requested"
    blocked = observe(subject)
    with pytest.raises(BarrierError, match="task_not_active"):
        consumer.consume(wire(blocked), now=NOW)
    assert consumer.sequence == blocked.sequence
    with pytest.raises(BarrierError, match="receipt_replayed"):
        consumer.consume(wire(first), now=NOW)


def test_consumer_rejects_replaced_claim_and_admission(case):
    subject, _ = observer(case)
    first = observe(subject)
    consumer = ReceiptConsumer(case[0])
    consumer.consume(wire(first), now=NOW)
    changed = first.model_copy(update={"sequence": 2, "claim": first.claim.model_copy(update={"attempt_id": uuid4()})})
    with pytest.raises(BarrierError, match="claim_replaced"):
        consumer.consume(wire(changed), now=NOW)
    changed = first.model_copy(update={"sequence": 3, "admission": first.admission.model_copy(update={"request_id": uuid4()})})
    with pytest.raises(BarrierError, match="admission_replaced"):
        consumer.consume(wire(changed), now=NOW)


def test_strict_json_capacity_unknown_fields_duplicate_keys_and_expiry(case):
    subject, _ = observer(case)
    receipt = observe(subject)
    data = json.loads(wire(receipt))
    for change in ({"unknown": 1}, {"sequence": True}, {"expires_at": "2099-01-01T00:00:00Z"},
                   {"observed_at": "2026-10-31T12:00:00"}):
        with pytest.raises(BarrierError, match="invalid_receipt"):
            decode_receipt(json.dumps({**data, **change}).encode())
    with pytest.raises(BarrierError, match="invalid_receipt"):
        decode_receipt(wire(receipt)[:-1] + b',"sequence":2}')
    with pytest.raises(BarrierError, match="receipt_size"):
        decode_receipt(b" " * (MAX_RECEIPT_BYTES + 1))


def test_atomic_exchange_sequence_privacy_and_no_temporary_leaks(case, tmp_path):
    tmp_path.chmod(0o700)
    subject, pool = observer(case)
    first = observe(subject)
    with AtomicReceiptFile(tmp_path) as exchange:
        exchange.publish(first, now=NOW)
        assert decode_receipt(exchange.read()) == first
        with pytest.raises(BarrierError, match="receipt_replayed"):
            exchange.publish(first, now=NOW)
        pool.row["status"] = "cancelled"
        second = observe(subject)
        exchange.publish(second, now=NOW)
        assert decode_receipt(exchange.read()).status == "blocked"
        assert {p.name for p in tmp_path.iterdir()} == {"task-barrier.json", "task-barrier.json.lock"}
        assert (tmp_path / "task-barrier.json").stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize("name", ["../escape.json", "/tmp/escape.json", "child/file.json", "..", "a.json/../b.json"])
def test_control_basename_breakout_rejected(tmp_path, name):
    with pytest.raises(BarrierError, match="invalid_control_name"):
        AtomicReceiptFile(tmp_path, name=name)


def test_root_and_ancestor_symlinks_permissions_and_replacement_rejected(tmp_path):
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    link = tmp_path / "link"
    link.symlink_to(private, target_is_directory=True)
    with pytest.raises(BarrierError, match="unsafe_control_root"):
        AtomicReceiptFile(link)
    child = private / "child"
    child.mkdir(mode=0o700)
    with pytest.raises(BarrierError, match="unsafe_control_root"):
        AtomicReceiptFile(link / "child")
    private.chmod(0o755)
    with pytest.raises(BarrierError, match="control_root_not_private"):
        AtomicReceiptFile(private)
    private.chmod(0o700)
    with AtomicReceiptFile(private) as exchange:
        private.rename(tmp_path / "retired")
        private.mkdir(mode=0o700)
        with pytest.raises(BarrierError, match="control_root_replaced"):
            exchange.read()


@pytest.mark.parametrize("kind", ["symlink", "hardlink", "fifo", "oversize", "public"])
def test_unsafe_control_targets_rejected_without_following(case, tmp_path, kind):
    tmp_path.chmod(0o700)
    subject, _ = observer(case)
    receipt = observe(subject)
    target, outside = tmp_path / "task-barrier.json", tmp_path / "untouched"
    outside.write_bytes(b"sentinel")
    outside.chmod(0o600)
    if kind == "symlink":
        target.symlink_to(outside)
    elif kind == "hardlink":
        os.link(outside, target)
    elif kind == "fifo":
        os.mkfifo(target, 0o600)
    else:
        target.write_bytes(b"x" * (MAX_RECEIPT_BYTES + 1) if kind == "oversize" else wire(receipt))
        target.chmod(0o644 if kind == "public" else 0o600)
    with AtomicReceiptFile(tmp_path) as exchange:
        with pytest.raises(BarrierError):
            exchange.read()
        with pytest.raises(BarrierError):
            exchange.publish(receipt, now=NOW)
    assert outside.read_bytes() == b"sentinel"


def test_atomic_failure_preserves_previous_receipt_and_removes_owned_temporary(case, tmp_path, monkeypatch):
    tmp_path.chmod(0o700)
    subject, _ = observer(case)
    first, second = observe(subject), observe(subject)
    with AtomicReceiptFile(tmp_path) as exchange:
        exchange.publish(first, now=NOW)
        def fail(*args, **kwargs):
            raise OSError("controlled rename failure")
        monkeypatch.setattr(os, "replace", fail)
        with pytest.raises(BarrierError, match="control_publication_failed"):
            exchange.publish(second, now=NOW)
        assert decode_receipt(exchange.read()) == first
        assert len(list(tmp_path.iterdir())) == 2
