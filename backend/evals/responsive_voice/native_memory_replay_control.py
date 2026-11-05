"""Offline contract controller for retained-scope native memory replay.

No launcher, SQL, provider client, fixture allocator or authorization consumer.
A future adapter must supply receipts from the EXISTING runner/host and real
native/canonical evidence. Receipt validation is not evidence authenticity or
E2E acceptance. In particular, two run_once calls destroy/reallocate the bank.

Calendar time is explicit evaluator/domain time, never a replacement for wall,
security, database or monotonic clocks. Current native clock wiring is absent.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
import hashlib
import json
import math
import re
import time
from uuid import UUID


class ContractError(ValueError):
    pass


def require(condition, code):
    if not condition:
        raise ContractError(code)


def integer(value, minimum, maximum, code):
    require(type(value) is int and minimum <= value <= maximum, code)
    return value


def digest(value):
    """Hash canonical contract JSON, NOT original file bytes."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def sha(value):
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value), "invalid_sha256")
    return value


def utc(value):
    require(isinstance(value, datetime) and value.tzinfo is not None
            and value.utcoffset() == timedelta(0), "utc_time_required")
    return value


def identifier(value):
    require(isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_.:-]{1,160}", value),
            "invalid_evidence_identity")
    return value


@dataclass(frozen=True)
class EvidenceRef:
    # Relative artifact path + immutable digest + selector into retained evidence.
    path: str
    sha256: str
    selector: str

    def __post_init__(self):
        require(isinstance(self.path, str) and 0 < len(self.path) <= 512
                and not self.path.startswith("/") and "\\" not in self.path
                and all(part not in {"", ".", ".."} for part in self.path.split("/")),
                "evidence_path_not_relative")
        sha(self.sha256)
        require(isinstance(self.selector, str) and 0 < len(self.selector) <= 256,
                "evidence_selector_required")


@dataclass(frozen=True)
class Scope:
    run_id: str
    owner_id: str
    database_schema: str
    area_id: str
    memory_bank_id: str

    def __post_init__(self):
        require(UUID(self.run_id).hex == self.run_id, "invalid_fixture_run")
        require(self.owner_id in {
            f"direct-image-eval-second-brain-{self.run_id}-a",
            f"direct-image-eval-second-brain-{self.run_id}-b"}, "foreign_owner")
        require(self.database_schema == f"jarvis_sb_eval_{self.run_id}_0", "foreign_schema")
        UUID(self.area_id)
        require(re.fullmatch(r"jarvis_[0-9a-f]{32}", self.memory_bank_id), "invalid_bank")


class DomainCalendar:
    """Checkpoint math only; changing this object does not advance native services."""
    def __init__(self, epoch):
        self.epoch, self.day = utc(epoch), 0

    def advance(self, day):
        integer(day, self.day, 90, "domain_time_cannot_reverse_or_exceed_90_days")
        delta, self.day = day - self.day, day
        return delta * 86400

    def event_time(self, day, source_pts_ms):
        integer(day, 0, self.day, "future_source_day")
        integer(source_pts_ms, 0, 3_600_000, "invalid_source_pts")
        return self.epoch + timedelta(days=day, milliseconds=source_pts_ms)


class MemoryReplayController:
    """Fail-closed, synchronous state machine; caller owns every external handle.

    Each receipt is an assertion by the future adapter with an evidence pointer.
    Offline doubles exercise ordering/bounds, not real app/ingestion/retention.
    No release template is consumable here and no retry is automatic.
    """
    CLOCK_COMPONENTS = frozenset({"live_state", "working_memory", "source_date_filters"})
    RETIREMENT = ("app_terminated", "audio_callbacks_joined", "old_conversation_retired",
                  "frame_ring_retired", "host_joined", "workers_joined")
    CLEANUP = ("sql_clean", "bank_clean", "storage_clean", "tokens_removed")

    def __init__(self, plan, scope, *, epoch, monotonic=time.monotonic,
                 wall=lambda: datetime.now(UTC)):
        # Detach caller-owned containers; all runtime inputs remain outside this object.
        self.plan = json.loads(json.dumps(plan, allow_nan=False))
        self.scope, self.calendar = scope, DomainCalendar(epoch)
        self._validate_plan()
        self.plan_sha256 = digest(self.plan)
        self.monotonic, self.wall = monotonic, wall
        self.origin, self.last_tick = monotonic(), 0.0
        require(type(self.origin) in (int, float) and math.isfinite(self.origin),
                "invalid_monotonic_origin")
        self.index, self.state = 0, "between_sessions"
        self.active = None
        self.authorizations, self.identities, self.sources = set(), set(), {}
        self.receipts, self.failure = [], None
        self.provider_finalization = "not_started"

    def _validate_plan(self):
        p = self.plan
        require(p.get("format") == "jarvis.native_memory_replay.contract.v1", "invalid_plan_format")
        integer(p.get("media_origin_ms"), 0, 180_000, "invalid_media_origin")
        integer(p.get("source_end_ms"), 1, 180_000, "invalid_source_end")
        integer(p.get("max_wall_seconds"), 1, 3600, "invalid_wall_budget")
        phases = p.get("phases")
        require(isinstance(phases, list) and 2 <= len(phases) <= 6, "phase_budget")
        previous_end, previous_day, ids, frames = 0, 0, set(), set()
        for phase in phases:
            identity = identifier(phase["id"])
            require(identity not in ids, "duplicate_phase")
            ids.add(identity)
            integer(phase["day"], previous_day, 90, "nonmonotonic_day")
            integer(phase["start_ms"], previous_end, 3_600_000, "overlapping_phases")
            integer(phase["end_ms"], phase["start_ms"] + 1,
                    phase["start_ms"] + 180_000, "scene_duration_bound")
            if phase["query_ms"] is not None:
                integer(phase["query_ms"], phase["start_ms"], phase["end_ms"] - 1,
                        "query_outside_phase")
            require(isinstance(phase["frames"], list) and len(phase["frames"]) <= 240,
                    "frame_budget")
            last_pts = -1
            for frame in phase["frames"]:
                require(set(frame) == {"id", "source_pts_ms", "at_ms", "sha256"},
                        "runtime_or_oracle_fields_not_allowed")
                identity = identifier(frame["id"])
                require(identity not in frames, "duplicate_frame")
                frames.add(identity)
                integer(frame["source_pts_ms"], last_pts + 1, p["source_end_ms"] - 1,
                        "nonchronological_or_future_frame")
                require(frame["at_ms"] == p["media_origin_ms"] + frame["source_pts_ms"]
                        and phase["start_ms"] <= frame["at_ms"] < phase["end_ms"],
                        "source_clock_spacing_changed")
                sha(frame["sha256"])
                last_pts = frame["source_pts_ms"]
            previous_end, previous_day = phase["end_ms"], phase["day"]
        # Existing runner reserves 240 s per session; final audited cleanup 180 s.
        require(previous_end / 1000 + len(phases) * 240 + 180 <= p["max_wall_seconds"],
                "lease_does_not_cover_phases_and_cleanup")

    def _tick(self):
        value = (self.monotonic() - self.origin) * 1000
        require(math.isfinite(value) and value >= self.last_tick, "monotonic_reversed")
        self.last_tick = value
        require(value <= self.plan["max_wall_seconds"] * 1000, "controller_wall_deadline")
        return value

    def _ref(self, value):
        require(isinstance(value, EvidenceRef), "retained_evidence_reference_required")
        return value

    def reserve_admission(self, *, authorization_id, scope, plan_sha256,
                          input_manifest_sha256, provider_bindings_sha256,
                          expires_at, consumed_ledger_ref, public_demo=False,
                          prior_unknown_risk_accepted=False):
        """Accept an EXISTING consumed-ledger receipt, never consume a grant here.

        The real adapter must retain the existing simulator lock/active marker,
        release validation and O_EXCL spent ledger. In-memory sets are not a
        replacement. An ambiguous/rejected native admission is never retried.
        """
        now = self._tick()
        require(self.state == "between_sessions" and self.index < len(self.plan["phases"]),
                "admission_not_allowed")
        phase = self.plan["phases"][self.index]
        require(scope == self.scope and plan_sha256 == self.plan_sha256, "authorization_scope_changed")
        identity = str(UUID(authorization_id))
        require(identity not in self.authorizations, "authorization_reused")
        require(public_demo is False, "public_demo_forbidden")
        # Reserve cleanup plus the remaining phase duration against REAL wall time.
        remaining = max(0, phase["end_ms"] - now) / 1000 + 240 + 180
        require((utc(expires_at) - utc(self.wall())).total_seconds() > remaining,
                "authorization_real_wall_expired_or_short")
        require(now >= phase["start_ms"], "phase_started_before_boundary")
        require(now < phase["end_ms"] and (phase["query_ms"] is None or now <= phase["query_ms"]),
                "fixed_source_deadline_missed")
        require(self.provider_finalization != "unknown" or prior_unknown_risk_accepted is True,
                "previous_provider_finalization_unknown")
        self._ref(consumed_ledger_ref)
        sha(input_manifest_sha256)
        sha(provider_bindings_sha256)
        self.authorizations.add(identity)  # Burn before native result, including ambiguous rejection.
        self.calendar.advance(phase["day"])
        self.active = {"phase": phase, "authorization_id": identity, "input_sha256": input_manifest_sha256,
                       "provider_bindings_sha256": provider_bindings_sha256, "ledger": consumed_ledger_ref,
                       "reserved_at_ms": now, "ready": None, "retrievals": []}
        self.state = "admission_reserved"

    def ready(self, *, scope, session_id, audio_session_id, app_instance_id, host_instance_id,
              origin_run_ms, input_manifest_sha256, evidence, clock_components=()):
        now = self._tick()
        require(self.state == "admission_reserved", "readiness_without_reserved_admission")
        phase = self.active["phase"]
        require(scope == self.scope and input_manifest_sha256 == self.active["input_sha256"],
                "native_scope_or_input_changed")
        require(origin_run_ms == phase["start_ms"], "native_origin_rebased_source_clock")
        require(now < phase["end_ms"] and (phase["query_ms"] is None or now <= phase["query_ms"]),
                "fixed_source_deadline_missed")
        identities = tuple(identifier(x) for x in
                           (session_id, audio_session_id, app_instance_id, host_instance_id))
        require(len(set(identities)) == 4 and not (set(identities) & self.identities),
                "session_or_process_identity_reused")
        require(not clock_components or set(clock_components) == self.CLOCK_COMPONENTS,
                "unsupported_domain_clock_scope")
        if phase["day"]:
            require(set(clock_components) == self.CLOCK_COMPONENTS, "native_domain_clock_not_wired")
        self.identities.update(identities)
        self.active["ready"] = {"ids": identities, "at_ms": now, "evidence": self._ref(evidence)}
        self.state = "running"
        # Delay is from the fixed shared origin, never a new fixed 8 s after readiness.
        return None if phase["query_ms"] is None else phase["query_ms"] - now

    def publish(self, *, source_id, revision, generation, frame_id, frame_sha256,
                source_pts_ms, source_day, published_at_ms, evidence):
        now = self._tick()
        require(self.state == "running", "publication_outside_running_session")
        require(len(self.sources) < 256, "source_receipt_budget")
        phase = self.active["phase"]
        frame = next((f for f in phase["frames"] if f["id"] == frame_id), None)
        require(frame is not None and frame_sha256 == frame["sha256"]
                and source_pts_ms == frame["source_pts_ms"] and source_day == phase["day"],
                "source_not_in_admitted_manifest")
        require(frame["at_ms"] <= published_at_ms <= min(now, phase["end_ms"]),
                "publication_future_or_after_retirement")
        identifier(source_id)
        integer(revision, 1, 1_000_000, "invalid_revision")
        integer(generation, 1, 1_000_000, "invalid_generation")
        key = (source_id, revision, generation)
        require(key not in self.sources, "duplicate_publication")
        self.sources[key] = {"phase_index": self.index, "frame_id": frame_id,
                             "source_time": self.calendar.event_time(source_day, source_pts_ms),
                             "published_at_ms": published_at_ms, "evidence": self._ref(evidence)}
        return key

    def retrieve(self, *, source_key, scope, session_id, query_at_ms, evidence, selected_context):
        now = self._tick()
        require(self.state == "running" and self.index > 0, "retrieval_without_fresh_session")
        require(len(self.active["retrievals"]) < 16, "retrieval_receipt_budget")
        phase = self.active["phase"]
        require(scope == self.scope and session_id == self.active["ready"]["ids"][0],
                "retrieval_wrong_session_or_scope")
        require(query_at_ms == phase["query_ms"] and query_at_ms <= now < phase["end_ms"],
                "query_retimed_or_future")
        source = self.sources.get(tuple(source_key))
        require(source is not None and source["phase_index"] < self.index,
                "retrieval_not_from_prior_persisted_publication")
        source_now_ms = query_at_ms - self.plan["media_origin_ms"]
        at = self.calendar.event_time(phase["day"], source_now_ms)
        age = (at - source["source_time"]).total_seconds()
        require(age >= 0, "future_source_retrieved")
        self.active["retrievals"].append({"source_key": tuple(source_key), "age_seconds": age,
            "retrieval": self._ref(evidence), "selected_context": self._ref(selected_context)})
        return age

    def close_phase(self, *, scope, ended_at_ms, retired, canonical, provider_finalization):
        now = self._tick()
        require(self.state in {"running", "admission_reserved", "aborted"} and self.active is not None,
                "no_active_phase")
        require(scope == self.scope, "retirement_wrong_scope")
        require(set(retired) == set(self.RETIREMENT) and all(v is True for v in retired.values()),
                "unjoined_session_preserve_fixture")
        phase = self.active["phase"]
        require(0 <= ended_at_ms <= now and now - ended_at_ms <= 65_000,
                "retirement_join_deadline")
        if self.state != "aborted":
            require(ended_at_ms >= phase["end_ms"], "early_session_end")
        require(provider_finalization in {"confirmed", "unknown"}, "invalid_provider_finalization")
        self._ref(canonical)
        if self.state != "aborted":
            require(self.active["ready"] is not None, "no_native_readiness")
            if self.index == 0:
                require(bool(self.sources), "no_persisted_publication")
            else:
                require(bool(self.active["retrievals"]), "no_actual_retrieval_evidence")
        self.provider_finalization = provider_finalization
        self.receipts.append({**self.active, "ended_at_ms": ended_at_ms, "joined_at_ms": now,
                              "canonical": canonical, "retired": dict(retired)})
        self.active = None
        self.index += 1
        self.state = "cleanup_ready" if self.failure or self.index == len(self.plan["phases"]) else "between_sessions"

    def abort(self, code):
        require(self.state != "closed", "controller_already_closed")
        self.failure = identifier(code)
        self.state = "aborted" if self.active else "cleanup_ready"

    def cleanup(self, *, scope, checks, evidence, started_at_ms):
        now = self._tick()
        require(self.state == "cleanup_ready" and self.active is None, "cleanup_before_join_and_export")
        require(scope == self.scope and set(checks) == set(self.CLEANUP)
                and all(v is True for v in checks.values()), "cleanup_audit_incomplete")
        require(0 <= started_at_ms <= now and now - started_at_ms <= 180_000,
                "cleanup_deadline")
        self._ref(evidence)
        self.state = "closed"

    def report(self):
        return {"format": "jarvis.native_memory_replay.controller.v1", "state": self.state,
                "plan_sha256": self.plan_sha256, "sessions_retired": len(self.receipts),
                "admissions_reserved": len(self.authorizations), "failure": self.failure,
                "domain_day": self.calendar.day, "source_versions": len(self.sources),
                "provider_finalization": self.provider_finalization,
                "native_e2e_accepted": False, "semantic_acceptance": "not_evaluated",
                "evidence_authenticity": "adapter_assertions_require_actual_artifact_join",
                "scheduler_time_travel_qualified": False,
                "database_clock_changed": False,
                "remaining_native_gates": list(self.plan.get("required_native_seams", ()))}

