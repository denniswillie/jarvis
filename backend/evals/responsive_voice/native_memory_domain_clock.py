"""Opt-in eval domain time, not a runner, source authority, or runtime clock.

The existing retained-run owner supplies approved manifest bytes, actual scope,
joined phase evidence, and freshly authorized canonical source reads. Evidence
references/hashes validate structure and identity, not authenticity. No business
records, original timestamps, authorization, leases, or provider clocks change.

Checkpoints are private, immutable and fsynced before their reference is returned.
The caller must retain that reference in its existing run evidence. A failed
write is uncertain: the instance fences, and no retry/recovery is automatic.
Source stamps are caller-retained evidence values, not a second source registry.
Nothing in this module propagates time into application state or model context.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
from uuid import UUID

from app.knowledge.models import SourceVersion
from evals.responsive_voice.native_memory_replay_control import (
    ContractError, DomainCalendar, EvidenceRef, MemoryReplayController, Scope,
    digest, identifier, integer, require, sha,
)
from evals.responsive_voice.native_task_release_control import read_private, write_private


FORMAT = "jarvis.native_memory_replay.domain_clock.v1"
MAX_PHASES = 6  # The existing MemoryReplayController phase budget.
MAX_MANIFEST_BYTES = 256 * 1024


@dataclass(frozen=True)
class Checkpoint:
    sequence: int
    sha256: str

    def __post_init__(self):
        integer(self.sequence, 0, MAX_PHASES - 1, "checkpoint_sequence")
        sha(self.sha256)

    @property
    def name(self):
        return f"domain-clock-{self.sequence:03d}.json"


@dataclass(frozen=True)
class SourceStamp:
    binding_sha256: str
    source_id: UUID
    identity_sha256: str
    source_day: int
    source_pts_ms: int
    evidence: EvidenceRef

    def __post_init__(self):
        sha(self.binding_sha256)
        sha(self.identity_sha256)
        require(isinstance(self.source_id, UUID), "source_id_required")
        integer(self.source_day, 0, 90, "invalid_source_day")
        integer(self.source_pts_ms, 0, 3_600_000, "invalid_source_pts")
        require(isinstance(self.evidence, EvidenceRef), "source_evidence_required")


@dataclass(frozen=True)
class LogicalAge:
    """An additional predicate only, never permission or current-world truth."""

    eligible: bool
    reason: str
    age_seconds: float | None
    domain_as_of: datetime


def _pairs(items):
    result = {}
    for key, value in items:
        require(key not in result, "duplicate_checkpoint_key")
        result[key] = value
    return result


def _encode(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _hash(data):
    return hashlib.sha256(data).hexdigest()


def _source_identity(source):
    # Operational ingestion progress is deliberately not a source revision.
    return digest({
        "owner_id": source.owner_id, "area_id": str(source.area_id),
        "bank_name": source.bank_name, "source_id": str(source.source_id),
        "revision": source.revision, "generation": source.generation,
        "payload_id": str(source.payload_id), "document_id": source.document_id,
        "content_digest": source.content_digest,
        "source_time": source.source_time.isoformat() if source.source_time else None,
        "recorded_at": source.recorded_at.isoformat(),
    })


class EvalDomainClock:
    """Single-owner, synchronous checkpoint helper; no tasks or timer callbacks.

    Use create/restore. The root must already be a private operator artifact
    directory. Logical positions are explicit phase-relative milliseconds, never
    sampled wall/monotonic time. Advance requires the existing retirement flags.
    """

    def __init__(self, *, root, scope, manifest_bytes, epoch):
        require(isinstance(scope, Scope), "fixture_scope_required")
        require(isinstance(manifest_bytes, bytes)
                and 0 < len(manifest_bytes) <= MAX_MANIFEST_BYTES, "manifest_bytes_required")
        self.root, self.scope = Path(root), scope
        require(self.root.is_absolute(), "absolute_checkpoint_root_required")
        self._calendar = DomainCalendar(epoch)
        self._binding = {"scope": asdict(scope), "manifest_sha256": _hash(manifest_bytes),
                         "epoch": self._calendar.epoch.isoformat()}
        self._document = None
        self._checkpoint = None
        self._uncertain = False

    @property
    def day(self):
        return self._calendar.day

    @property
    def checkpoint(self):
        return self._checkpoint

    @property
    def manifest_sha256(self):
        return self._binding["manifest_sha256"]

    @property
    def binding_sha256(self):
        return digest(self._binding)

    @classmethod
    def create(cls, *, root, scope, manifest_bytes, epoch):
        clock = cls(root=root, scope=scope, manifest_bytes=manifest_bytes, epoch=epoch)
        clock._persist({"format": FORMAT, "binding": clock._binding, "sequence": 0,
                        "day": 0, "previous_sha256": None, "boundaries": []})
        return clock

    @classmethod
    def restore(cls, *, root, scope, manifest_bytes, epoch, checkpoint):
        require(isinstance(checkpoint, Checkpoint), "checkpoint_reference_required")
        clock = cls(root=root, scope=scope, manifest_bytes=manifest_bytes, epoch=epoch)
        raw = read_private(clock.root, checkpoint.name)
        require(_hash(raw) == checkpoint.sha256, "checkpoint_digest_changed")
        document = json.loads(raw, object_pairs_hook=_pairs)
        clock._validate(document)
        require(document["sequence"] == checkpoint.sequence, "checkpoint_sequence_changed")
        # A newer or partially written checkpoint prevents silent rollback.
        for sequence in range(checkpoint.sequence + 1, MAX_PHASES):
            try:
                read_private(clock.root, Checkpoint(sequence, "0" * 64).name)
            except FileNotFoundError:
                continue
            raise ContractError("later_checkpoint_exists")
        clock._document, clock._checkpoint = document, checkpoint
        clock._calendar.advance(document["day"])
        return clock

    def _validate(self, document):
        require(isinstance(document, dict) and set(document) == {
            "format", "binding", "sequence", "day", "previous_sha256", "boundaries"
        }, "invalid_checkpoint")
        require(document["format"] == FORMAT and document["binding"] == self._binding,
                "checkpoint_binding_changed")
        sequence = integer(document["sequence"], 0, MAX_PHASES - 1, "checkpoint_sequence")
        integer(document["day"], 0, 90, "invalid_domain_day")
        boundaries = document["boundaries"]
        require(isinstance(boundaries, list) and len(boundaries) == sequence, "checkpoint_history_changed")
        if sequence:
            sha(document["previous_sha256"])
        else:
            require(document["previous_sha256"] is None, "invalid_initial_checkpoint")
        day, phases, sessions = 0, set(), set()
        for boundary in boundaries:
            require(isinstance(boundary, dict) and set(boundary) == {
                "phase_id", "session_id", "previous_day", "next_day", "retired", "evidence"
            }, "invalid_joined_boundary")
            phase = identifier(boundary["phase_id"])
            session = boundary["session_id"]
            require(isinstance(session, str) and str(UUID(session)) == session, "session_id_required")
            require(phase not in phases and session not in sessions, "retirement_reused")
            phases.add(phase)
            sessions.add(session)
            require(type(boundary["previous_day"]) is int and boundary["previous_day"] == day,
                    "boundary_day_changed")
            day = integer(boundary["next_day"], day + 1, 90, "domain_day_must_advance")
            retired = boundary["retired"]
            require(isinstance(retired, dict) and set(retired) == set(MemoryReplayController.RETIREMENT)
                    and all(value is True for value in retired.values()), "phase_not_joined")
            evidence = boundary["evidence"]
            require(isinstance(evidence, dict) and set(evidence) == {"path", "sha256", "selector"},
                    "retirement_evidence_required")
            EvidenceRef(**evidence)
        require(day == document["day"], "checkpoint_day_changed")

    def _persist(self, document):
        self._validate(document)
        raw = _encode(document)
        reference = Checkpoint(document["sequence"], _hash(raw))
        try:
            # The immutable name is not advertised until write + directory fsync.
            write_private(self.root, reference.name, raw, exclusive=True)
        except BaseException:
            self._uncertain = True
            raise
        self._document, self._checkpoint = document, reference
        self._calendar.advance(document["day"])
        return reference

    def _ready(self):
        require(not self._uncertain, "checkpoint_write_uncertain")
        require(self._checkpoint is not None, "checkpoint_not_initialized")

    def advance_after_join(self, *, day, scope, manifest_sha256, previous,
                           phase_id, session_id, retired, evidence):
        self._ready()
        require(scope == self.scope and manifest_sha256 == self.manifest_sha256,
                "boundary_scope_or_manifest_changed")
        require(previous == self.checkpoint, "stale_checkpoint")
        require(isinstance(session_id, UUID), "session_id_required")
        require(isinstance(evidence, EvidenceRef), "retirement_evidence_required")
        require(_hash(read_private(self.root, previous.name)) == previous.sha256,
                "checkpoint_digest_changed")
        require(isinstance(retired, dict), "phase_not_joined")
        boundary = {"phase_id": phase_id, "session_id": str(session_id),
                    "previous_day": self.day, "next_day": day,
                    "retired": dict(retired), "evidence": asdict(evidence)}
        return self._persist({"format": FORMAT, "binding": self._binding,
            "sequence": previous.sequence + 1, "day": day, "previous_sha256": previous.sha256,
            "boundaries": [*self._document["boundaries"], boundary]})

    def logical_now(self, *, phase_pts_ms):
        self._ready()
        return self._calendar.event_time(self.day, phase_pts_ms)

    def _source_scope(self, source):
        return (source.owner_id == self.scope.owner_id and str(source.area_id) == self.scope.area_id
                and source.bank_name == self.scope.memory_bank_id)

    def bind_source(self, source, *, source_day, source_pts_ms, evidence):
        """Bind a real source read to supplied exact admission/publication evidence.

        Caller must join source identity to its admitted phase, not infer a phase
        from temporal proximity. This helper neither reads nor invents that join.
        """
        self._ready()
        require(isinstance(source, SourceVersion) and self._source_scope(source), "source_scope_changed")
        require(source.is_current, "source_not_current")
        self._calendar.event_time(source_day, source_pts_ms)
        return SourceStamp(self.binding_sha256, source.source_id, _source_identity(source),
                           source_day, source_pts_ms, evidence)

    def logical_age(self, stamp, *, current_source, phase_pts_ms, max_age_seconds):
        """Pure additional eligibility; a cached row is NOT fresh authorization.

        The caller must first use existing canonical permission/deletion guards.
        Supply None when that read is unavailable or access is withdrawn. A
        successful result cannot replace real freshness or action/source guards.
        """
        now = self.logical_now(phase_pts_ms=phase_pts_ms)
        require(isinstance(stamp, SourceStamp), "source_stamp_required")
        require(type(max_age_seconds) in (int, float) and math.isfinite(max_age_seconds)
                and max_age_seconds > 0, "invalid_logical_age_bound")
        reason = None
        if stamp.binding_sha256 != self.binding_sha256:
            reason = "source_binding_changed"
        elif current_source is None:
            reason = "source_unavailable"
        elif not isinstance(current_source, SourceVersion) or not self._source_scope(current_source):
            reason = "source_scope_changed"
        elif not current_source.is_current:
            reason = "source_not_current"
        elif (current_source.source_id != stamp.source_id
              or _source_identity(current_source) != stamp.identity_sha256):
            reason = "source_version_changed"
        elif stamp.source_day > self.day:
            reason = "source_domain_time_in_future"
        if reason is not None:
            return LogicalAge(False, reason, None, now)
        observed = self._calendar.event_time(stamp.source_day, stamp.source_pts_ms)
        age = (now - observed).total_seconds()
        if age < 0:
            return LogicalAge(False, "source_domain_time_in_future", age, now)
        eligible = age < max_age_seconds
        return LogicalAge(eligible, "within_logical_age" if eligible else "logical_age_expired", age, now)
