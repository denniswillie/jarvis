"""Test-only, read-only canonical task prerequisite for native replay.

No admission, claim, renewal, cancellation, executor hold or audio is performed
here. The caller supplies an observation from the real registered create handler
and its real tool invocation. A *different*, non-borrowed pool lease establishes
committed state. Provider call IDs are observation evidence, not invented SQL
columns. No result/receipt implies ongoing model computation or audible output.

Wire contract: jarvis.native_task_barrier.v1, strict immutable JSON, <=16 KiB.
The five-second maximum is a stale-evidence safety cap, not a latency target.
Consumers must retain their sequence/identity watermark across reconnects. This
is a point-in-time prerequisite, not an atomic task reservation at audio enqueue.
"""

from __future__ import annotations

import fcntl
import json
import math
import os
import re
import stat
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from app.tasks.models import TaskVoiceSource


MAX_RECEIPT_BYTES = 16 * 1024
MAX_TTL_SECONDS = 5.0
FORMAT = "jarvis.native_task_barrier.v1"


class BarrierError(ValueError):
    """Static code only: never propagate DB details, source text or secrets."""

    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


def _aware(value: datetime) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timezone_required")
    return value.astimezone(UTC)


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)


class FixtureScope(_Frozen):
    run_id: UUID
    barrier_id: UUID
    owner_id: str = Field(min_length=1, max_length=200)
    session_id: UUID
    database_schema: str
    worker_id: str = Field(min_length=1, max_length=100)
    runtime_host_id: str = Field(min_length=1, max_length=200)
    model: str = Field(min_length=1, max_length=100)
    host_expires_at: datetime

    _time = field_validator("host_expires_at")(_aware)

    @model_validator(mode="after")
    def isolated(self):
        if not re.fullmatch(r"jarvis_sb_eval_" + self.run_id.hex + r"_[0-9]+", self.database_schema):
            raise ValueError("isolated_manifest_schema_required")
        return self


class AdmissionWitness(_Frozen):
    owner_id: str = Field(min_length=1, max_length=200)
    session_id: UUID
    request_event_id: UUID
    request_revision: int = Field(gt=0)
    request_id: UUID
    task_id: UUID
    accepted_revision: int = Field(gt=0)
    parameters_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    provider_call_id: str = Field(min_length=1, max_length=512)
    operation: Literal["create"] = "create"


def witness_from_observation(
    scope: FixtureScope, observation: Mapping[str, Any], tool_calls: Sequence[Mapping[str, Any]],
) -> AdmissionWitness:
    """Adapt the ObservedAdmissionRepository pattern; its committed flag is ignored.

    These arguments must originate in the real handler/executor, never from a
    replay manifest, transcript-text detector, synthetic tool call or client JSON.
    No claim is made that the tool return reached the foreground provider.
    """
    try:
        source = TaskVoiceSource(UUID(str(observation["request_event_id"])),
                                 observation["request_revision"], scope.session_id)
        matching = [row for row in tool_calls if row.get("call_id") == observation["call_id"]]
        if (observation["owner_id"] != scope.owner_id or observation["operation"] != "create"
                or observation["voice_source"] != source.parameters()
                or not matching or any(row.get("tool") != "start_background_task" for row in matching)):
            raise ValueError("witness")
        return AdmissionWitness(
            owner_id=scope.owner_id, session_id=scope.session_id,
            request_event_id=source.event_id,
            request_revision=observation["request_revision"],
            request_id=UUID(str(observation["request_id"])), task_id=UUID(str(observation["task_id"])),
            accepted_revision=observation["accepted_revision"],
            parameters_digest=observation["parameters_digest"], provider_call_id=observation["call_id"],
        )
    except (KeyError, TypeError, ValueError, AttributeError):
        raise BarrierError("invalid_handler_witness") from None


class ActiveClaim(_Frozen):
    task_id: UUID
    task_revision: int = Field(gt=0)
    attempt_id: UUID
    attempt_number: int = Field(gt=0)
    fence: int = Field(gt=0)
    worker_id: str = Field(min_length=1, max_length=100)
    runtime_host_id: str = Field(min_length=1, max_length=200)
    lease_until: datetime
    worker_expires_at: datetime
    task_status: Literal["running"] = "running"
    attempt_status: Literal["running"] = "running"
    provider_state: Literal["not_started", "submitting", "live", "terminal"]

    _times = field_validator("lease_until", "worker_expires_at")(_aware)

    def identity(self) -> tuple:
        return (self.task_id, self.task_revision, self.attempt_id, self.attempt_number,
                self.fence, self.worker_id, self.runtime_host_id)


Reason = Literal[
    "active_claim_observed", "admission_not_committed", "admission_mismatch", "ambiguous_tasks",
    "source_ineligible", "source_replaced", "task_revision_replaced", "task_not_active",
    "claim_ineligible", "claim_replaced", "lease_expired", "worker_ineligible",
    "schema_mismatch", "borrowed_transaction", "canonical_unavailable",
]


class BarrierReceipt(_Frozen):
    format: Literal["jarvis.native_task_barrier.v1"] = FORMAT
    scope: FixtureScope
    admission: AdmissionWitness
    sequence: int = Field(gt=0)
    observed_at: datetime
    expires_at: datetime
    status: Literal["eligible", "blocked"]
    reason: Reason
    claim: ActiveClaim | None = None

    _times = field_validator("observed_at", "expires_at")(_aware)

    @model_validator(mode="after")
    def bounded(self):
        if (self.admission.owner_id != self.scope.owner_id
                or self.admission.session_id != self.scope.session_id):
            raise ValueError("admission_scope_mismatch")
        ttl = (self.expires_at - self.observed_at).total_seconds()
        if not 0 < ttl <= MAX_TTL_SECONDS or self.expires_at > self.scope.host_expires_at:
            raise ValueError("invalid_expiry")
        if self.status == "eligible":
            c = self.claim
            if (self.reason != "active_claim_observed" or c is None
                    or c.task_id != self.admission.task_id
                    or c.task_revision != self.admission.accepted_revision
                    or c.worker_id != self.scope.worker_id
                    or c.runtime_host_id != self.scope.runtime_host_id
                    or self.expires_at > min(c.lease_until, c.worker_expires_at)):
                raise ValueError("invalid_claim_binding")
        elif self.claim is not None or self.reason == "active_claim_observed":
            raise ValueError("invalid_blocked_receipt")
        return self


# Actual columns from migrations 0001/0002/0005. No payload revision exists.
# The candidate limit detects ambiguity, rather than hiding a second task behind
# multiple idempotency receipts for the first one. No FOR UPDATE or source-lock
# function is appropriate for this read-only observer.
SNAPSHOT_SQL = """
with candidates as (
    select id from jarvis_background_tasks
    where owner_id=$1 and session_id=$2 and request_event_id=$3 and request_revision=$4
    limit 2
)
select t.id as task_id,t.owner_id,t.session_id,t.request_event_id,t.request_revision,
    t.revision,t.status,t.pending_status,t.result_payload_id,t.current_attempt_id,t.fence,
    t.backend_kind,t.model,
    (select count(*) from candidates) as candidate_count,
    (select bool_and(id=$5) from candidates) as candidate_matches,
    r.request_id as committed_request_id,r.operation,r.accepted_revision,r.parameters_digest,
    e.kind as source_kind,e.trust as source_trust,e.finality as source_finality,
    e.revision as source_revision,e.session_id as source_session_id,
    p.availability as source_availability,
    not exists(select 1 from jarvis_context_events newer where newer.owner_id=e.owner_id
        and newer.source_namespace=e.source_namespace and newer.source_event_id=e.source_event_id
        and newer.revision>e.revision) as source_current,
    a.id as attempt_id,a.task_revision,a.attempt_number,a.fence as attempt_fence,
    a.worker_id,a.runtime_host_id,a.status as attempt_status,a.provider_state,a.lease_until,a.ended_at,
    w.host_id as registered_host_id,w.owner_id as registered_owner_id,
    w.model as registered_model,w.draining,w.expires_at as worker_expires_at
from jarvis_background_tasks t
left join jarvis_task_requests r on r.owner_id=t.owner_id and r.task_id=t.id and r.request_id=$6
left join jarvis_context_events e on e.owner_id=t.owner_id and e.id=t.request_event_id
    and e.revision=t.request_revision
left join jarvis_context_payloads p on p.owner_id=e.owner_id and p.id=e.payload_id
left join jarvis_task_attempts a on a.owner_id=t.owner_id and a.task_id=t.id and a.id=t.current_attempt_id
left join jarvis_task_workers w on w.worker_id=a.worker_id and w.backend_kind=t.backend_kind
where t.owner_id=$1 and t.session_id=$2 and t.id=$5
"""


class CanonicalTaskObserver:
    def __init__(self, pool, scope: FixtureScope, admission: AdmissionWitness, *,
                 ttl_seconds: float, initial_sequence: int = 0, expected_claim: ActiveClaim | None = None,
                 clock=lambda: datetime.now(UTC)):
        if (admission.owner_id != scope.owner_id or admission.session_id != scope.session_id):
            raise BarrierError("admission_scope_mismatch")
        if (isinstance(ttl_seconds, bool) or not math.isfinite(ttl_seconds)
                or not 0 < ttl_seconds <= MAX_TTL_SECONDS
                or type(initial_sequence) is not int or initial_sequence < 0):
            raise BarrierError("invalid_observer_bounds")
        self.pool, self.scope, self.admission = pool, scope, admission
        self.ttl_seconds, self.sequence, self.clock = ttl_seconds, initial_sequence, clock
        self._claim = expected_claim
        self._observing = False

    async def observe(self) -> BarrierReceipt:
        if self._observing:
            raise BarrierError("concurrent_observation")
        self._observing = True
        try:
            claim = None
            try:
                async with self.pool.acquire() as connection:
                    if connection.is_in_transaction():
                        raise BarrierError("borrowed_transaction")
                    async with connection.transaction(isolation="repeatable_read", readonly=True):
                        if await connection.fetchval("select current_schema()") != self.scope.database_schema:
                            raise BarrierError("schema_mismatch")
                        a = self.admission
                        row = await connection.fetchrow(SNAPSHOT_SQL, self.scope.owner_id,
                            self.scope.session_id, a.request_event_id, a.request_revision, a.task_id, a.request_id)
                now = _aware(self.clock())
                claim = self._evaluate(row, now)
                reason: Reason = "active_claim_observed"
            except BarrierError as error:
                reason = error.code
            except Exception:
                reason = "canonical_unavailable"
            now = _aware(self.clock())
            expiry = min(now + timedelta(seconds=self.ttl_seconds), self.scope.host_expires_at)
            if claim is not None:
                expiry = min(expiry, claim.lease_until, claim.worker_expires_at)
            if expiry <= now:
                raise BarrierError("host_or_lease_expired")
            self.sequence += 1
            return BarrierReceipt(scope=self.scope, admission=self.admission, sequence=self.sequence,
                observed_at=now, expires_at=expiry, status="eligible" if claim else "blocked",
                reason=reason, claim=claim)
        finally:
            self._observing = False

    def _evaluate(self, r, now: datetime) -> ActiveClaim:
        a, s = self.admission, self.scope
        if r is None or r["committed_request_id"] is None:
            raise BarrierError("admission_not_committed")
        if any((r["task_id"] != a.task_id, r["owner_id"] != s.owner_id, r["session_id"] != s.session_id,
                r["committed_request_id"] != a.request_id, r["operation"] != "create",
                r["accepted_revision"] != a.accepted_revision, r["parameters_digest"] != a.parameters_digest)):
            raise BarrierError("admission_mismatch")
        if any((r["request_event_id"] != a.request_event_id, r["request_revision"] != a.request_revision,
                r["source_revision"] != a.request_revision, r["source_session_id"] != s.session_id,
                r["source_kind"] not in {"user_request", "user_transcript"}, r["source_trust"] != "user_input",
                r["source_finality"] not in {"final", "revised"}, r["source_availability"] != "available")):
            raise BarrierError("source_ineligible")
        if r["source_current"] is not True:
            raise BarrierError("source_replaced")
        if r["candidate_count"] != 1 or r["candidate_matches"] is not True:
            raise BarrierError("ambiguous_tasks")
        if r["revision"] != a.accepted_revision:
            raise BarrierError("task_revision_replaced")
        if r["status"] != "running" or r["pending_status"] is not None or r["result_payload_id"] is not None:
            raise BarrierError("task_not_active")
        if (r["attempt_id"] is None or r["attempt_id"] != r["current_attempt_id"]
                or r["task_revision"] != r["revision"] or r["attempt_fence"] != r["fence"]
                or r["attempt_status"] != "running" or r["ended_at"] is not None):
            raise BarrierError("claim_ineligible")
        if (r["worker_id"] != s.worker_id or r["runtime_host_id"] != s.runtime_host_id
                or r["registered_host_id"] != s.runtime_host_id or r["registered_owner_id"] != s.owner_id
                or r["backend_kind"] != "frontier" or r["model"] != s.model
                or r["registered_model"] != s.model or r["draining"] is not False):
            raise BarrierError("worker_ineligible")
        if r["lease_until"] <= now or r["worker_expires_at"] <= now:
            raise BarrierError("lease_expired")
        claim = ActiveClaim(task_id=r["task_id"], task_revision=r["revision"], attempt_id=r["attempt_id"],
            attempt_number=r["attempt_number"], fence=r["attempt_fence"], worker_id=r["worker_id"],
            runtime_host_id=r["runtime_host_id"], lease_until=r["lease_until"],
            worker_expires_at=r["worker_expires_at"], provider_state=r["provider_state"])
        if self._claim is not None and self._claim.identity() != claim.identity():
            raise BarrierError("claim_replaced")
        self._claim = claim
        return claim


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate_json_key")
        result[key] = value
    return result


def decode_receipt(data: bytes) -> BarrierReceipt:
    if not isinstance(data, bytes) or not 0 < len(data) <= MAX_RECEIPT_BYTES:
        raise BarrierError("receipt_size")
    try:
        json.loads(data.decode("utf-8"), object_pairs_hook=_unique_pairs,
                   parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite")))
        return BarrierReceipt.model_validate_json(data)
    except (ValueError, UnicodeError, ValidationError):
        raise BarrierError("invalid_receipt") from None


class ReceiptConsumer:
    """Retain this object (or all its watermarks) across route reconnects."""

    def __init__(self, scope: FixtureScope, *, after_sequence: int = 0,
                 admission: AdmissionWitness | None = None, claim: ActiveClaim | None = None):
        if type(after_sequence) is not int or after_sequence < 0:
            raise BarrierError("invalid_sequence")
        self.scope, self.sequence, self.admission, self.claim = scope, after_sequence, admission, claim

    def consume(self, data: bytes, *, now: datetime) -> BarrierReceipt:
        receipt = decode_receipt(data)
        now = _aware(now)
        if receipt.scope != self.scope:
            raise BarrierError("receipt_scope_mismatch")
        if receipt.sequence <= self.sequence:
            raise BarrierError("receipt_replayed")
        if not receipt.observed_at <= now < receipt.expires_at:
            raise BarrierError("receipt_expired_or_future")
        if self.admission is not None and receipt.admission != self.admission:
            raise BarrierError("admission_replaced")
        self.sequence = receipt.sequence
        self.admission = receipt.admission
        if receipt.status != "eligible":
            raise BarrierError(receipt.reason)
        if self.claim is not None and receipt.claim.identity() != self.claim.identity():
            raise BarrierError("claim_replaced")
        self.claim = receipt.claim
        return receipt


def _open_root(path: Path) -> int:
    raw = os.fspath(path)
    if not os.path.isabs(raw) or raw != os.path.normpath(raw) or raw == "/":
        raise BarrierError("invalid_control_root")
    fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in raw.split("/")[1:]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = child
        info = os.fstat(fd)
        if info.st_uid != os.geteuid() or info.st_mode & 0o077:
            raise BarrierError("control_root_not_private")
        return fd
    except BaseException:
        os.close(fd)
        raise


def _regular_private(fd: int):
    info = os.fstat(fd)
    if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid()
            or info.st_mode & 0o077 or info.st_nlink != 1):
        raise BarrierError("control_file_not_private_regular")
    return info


class AtomicReceiptFile:
    """One latest-state file, not a queue. Root must already exist privately.

    The trusted runner supplies the absolute staged control root and basename.
    All ancestors are opened without following symlinks. A private per-file lock
    serializes only local atomic publication, never SQL or network operations.
    """

    def __init__(self, root: Path, *, name: str = "task-barrier.json"):
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,99}\.json", name):
            raise BarrierError("invalid_control_name")
        self.root, self.name = root, name
        try:
            self.fd = _open_root(root)
        except OSError:
            raise BarrierError("unsafe_control_root") from None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None

    def _check_root(self):
        if self.fd is None:
            raise BarrierError("control_file_closed")
        current = _open_root(self.root)
        try:
            old, new = os.fstat(self.fd), os.fstat(current)
            if (old.st_dev, old.st_ino) != (new.st_dev, new.st_ino):
                raise BarrierError("control_root_replaced")
        finally:
            os.close(current)

    def read(self) -> bytes:
        try:
            self._check_root()
            fd = os.open(self.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=self.fd)
            try:
                before = _regular_private(fd)
                if not 0 < before.st_size <= MAX_RECEIPT_BYTES:
                    raise BarrierError("receipt_size")
                data = os.read(fd, MAX_RECEIPT_BYTES + 1)
                after = os.fstat(fd)
                if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                    raise BarrierError("receipt_changed_during_read")
                decode_receipt(data)
                return data
            finally:
                os.close(fd)
        except FileNotFoundError:
            raise BarrierError("receipt_missing") from None
        except OSError:
            raise BarrierError("unsafe_control_file") from None

    def publish(self, receipt: BarrierReceipt, *, now: datetime):
        data = receipt.model_dump_json().encode("utf-8")
        receipt = decode_receipt(data)
        if not receipt.observed_at <= _aware(now) < receipt.expires_at:
            raise BarrierError("receipt_expired_or_future")
        lock = None
        temporary = None
        try:
            self._check_root()
            lock = os.open(self.name + ".lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK,
                           0o600, dir_fd=self.fd)
            _regular_private(lock)
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                previous = decode_receipt(self.read())
            except BarrierError as error:
                if error.code != "receipt_missing":
                    raise
            else:
                if previous.scope != receipt.scope or previous.admission != receipt.admission:
                    raise BarrierError("control_identity_replaced")
                if receipt.sequence <= previous.sequence:
                    raise BarrierError("receipt_replayed")
            temporary = "." + self.name + "." + uuid4().hex
            fd = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600, dir_fd=self.fd)
            try:
                remaining = memoryview(data)
                while remaining:
                    count = os.write(fd, remaining)
                    if count <= 0:
                        raise OSError("short_write")
                    remaining = remaining[count:]
                os.fsync(fd)
            finally:
                os.close(fd)
            self._check_root()
            os.replace(temporary, self.name, src_dir_fd=self.fd, dst_dir_fd=self.fd)
            temporary = None
            os.fsync(self.fd)
        except OSError:
            raise BarrierError("control_publication_failed") from None
        finally:
            if temporary is not None:
                os.unlink(temporary, dir_fd=self.fd)
            if lock is not None:
                os.close(lock)
