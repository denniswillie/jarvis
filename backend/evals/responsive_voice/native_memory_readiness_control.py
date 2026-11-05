"""Bounded sibling IO for the existing typed replay memory-readiness adapter.

Only explicitly configured replay hosts start this observer loop. No memory/tool
actions, worker claims, provider calls or clock rebasing occur in this module.
"""
from __future__ import annotations

import asyncio
from dataclasses import asdict, fields
from datetime import UTC, datetime, timedelta
import hashlib
import json
import math
import time
from pathlib import Path
from uuid import UUID

from evals.responsive_voice.native_memory_readiness import ReadinessRequest
from evals.responsive_voice.native_memory_replay_control import Scope
from evals.responsive_voice.native_task_replay_control import _object
from evals.responsive_voice.native_task_release_control import write_private

REQUEST_NAME = "memory-readiness-request.json"
RECEIPT_NAME = "memory-readiness-receipt.json"
SCOPE_NAME = "memory-readiness-scope.json"
REQUEST_FORMAT = "jarvis.native_memory_readiness.request.v1"
RECEIPT_FORMAT = "jarvis.native_memory_readiness.receipt.v1"
HOST_FORMAT = "jarvis.native_memory_readiness.host.v1"
CONFIG_FORMAT = "jarvis.native_memory_readiness.configuration.v1"
STATUS_FORMAT = "jarvis.native_memory_readiness.status.v1"


def check(condition, code):
    if not condition:
        raise ValueError(code)


def number(value, low, high):
    check(type(value) in (int, float) and math.isfinite(value) and low <= value <= high,
          "invalid_memory_control_number")
    return value


def memory_gate_rows(plan):
    """Projection of approved manifest triggers, not a new identity schema."""
    rows = []
    for target, item in enumerate(plan.get("utterances", [])):
        trigger = item.get("trigger", {}) if isinstance(item, dict) else {}
        if trigger.get("kind") != "after_manual_response":
            continue
        required = {"kind", "after_input_ordinal", "at_seconds", "timeout_seconds", "canonical_requirement"}
        check(required <= set(trigger) <= required | {"playback_requirement"}, "invalid_memory_trigger_fields")
        prior = trigger["after_input_ordinal"]
        check(type(prior) is int and 0 <= prior < target <= 19, "invalid_memory_trigger_prior")
        at = number(trigger["at_seconds"], 0, 180)
        timeout = number(trigger["timeout_seconds"], .001, 120)
        check(at + timeout <= 180, "memory_trigger_deadline_exceeds_scene")
        check(trigger["canonical_requirement"] in {"finalized_input", "retain_searchable"}
              and trigger.get("playback_requirement", "none") in {"none", "authenticated_buffer_stop"},
              "invalid_memory_trigger_requirement")
        rows.append({"target_input_ordinal": target, "trigger": dict(trigger)})
    return rows


def declared_memory_readiness(plan):
    rows = memory_gate_rows(plan)
    if rows:
        check(plan.get("capture_mode") == "manual" and plan.get("voice_route") == "realtime"
              and plan.get("memory_readiness") == {"version": 1, "scope_file": SCOPE_NAME}
              and type(plan["memory_readiness"]["version"]) is int,
              "memory_readiness_declaration_required")
    else:
        check("memory_readiness" not in plan, "unused_memory_readiness_declaration")
    return bool(rows)


def validate_host_config(config):
    from evals.responsive_voice.frontend_host import _replay_path
    for path in (config.configuration_path, config.status_path):
        check(isinstance(path, Path), "memory_control_paths_required")
        _replay_path(path)
    check(config.configuration_path != config.status_path
          and config.configuration_path.parent == config.status_path.parent,
          "memory_control_sibling_metadata_required")
    check(isinstance(config.origin_invocation_id, UUID), "memory_control_origin_required")
    number(config.phase_start_seconds, 0, 900)
    if config.source_anchor_sha256 is not None:
        check(isinstance(config.source_anchor_sha256, str) and len(config.source_anchor_sha256) == 64
              and all(c in "0123456789abcdef" for c in config.source_anchor_sha256), "invalid_anchor_hash")
    check(isinstance(config.approved_gates, tuple) and 1 <= len(config.approved_gates) <= 19,
          "approved_memory_gates_required")
    prior_target = 0
    for row in config.approved_gates:
        check(isinstance(row, dict) and set(row) == {"target_input_ordinal", "trigger"}, "invalid_approved_gate")
        target = row["target_input_ordinal"]
        check(type(target) is int and prior_target < target <= 19, "invalid_approved_target")
        check(memory_gate_rows({"utterances": [{}] * target + [{"trigger": row["trigger"]}]}) == [row],
              "invalid_approved_trigger")
        prior_target = target


def config_from_metadata(value):
    from evals.responsive_voice.native_memory_readiness_observer import MemoryObserverConfig
    expected = {"format", "invocation_id", "manifest_sha256", "configuration_path", "status_path",
                "approved_gates", "origin_invocation_id", "phase_start_seconds", "source_anchor_sha256"}
    check(isinstance(value, dict) and set(value) == expected and value["format"] == HOST_FORMAT,
          "invalid_memory_host_metadata")
    return MemoryObserverConfig(invocation_id=UUID(value["invocation_id"]),
        manifest_sha256=value["manifest_sha256"], configuration_path=Path(value["configuration_path"]),
        status_path=Path(value["status_path"]), approved_gates=tuple(value["approved_gates"]),
        origin_invocation_id=UUID(value["origin_invocation_id"]), phase_start_seconds=value["phase_start_seconds"],
        source_anchor_sha256=value["source_anchor_sha256"])


def parse_request(data):
    check(isinstance(data, bytes) and 0 < len(data) <= 16384, "memory_request_byte_bound")
    envelope = _object(data)
    check(set(envelope) == {"format", "request"} and envelope["format"] == REQUEST_FORMAT,
          "memory_request_format")
    value = envelope["request"]
    check(isinstance(value, dict) and set(value) == {f.name for f in fields(ReadinessRequest)},
          "memory_request_fields")
    scope = value["scope"]
    check(isinstance(scope, dict) and set(scope) == {f.name for f in fields(Scope)}, "memory_request_scope_fields")
    value = dict(value, scope=Scope(**scope), session_id=UUID(value["session_id"]))
    for key in ("issued_at", "expires_at"):
        check(isinstance(value[key], str), "memory_request_clock_type")
        value[key] = datetime.fromisoformat(value[key])
    return ReadinessRequest(**value)


def _source_clock_wire_label(actual, expected):
    """Accept exact UTC or its nearest-millisecond native ISO8601 encoding.

    This is only a wire representation check, never a deadline allowance. The
    original same-boot uptime window still fences entry and result publication.
    """
    return actual == expected or (
        isinstance(actual, datetime) and actual.utcoffset() is not None
        and actual.microsecond % 1000 == 0
        and abs(actual - expected) <= timedelta(microseconds=500))


def approved_deadline(config, scope, anchor, request, sample):
    """Use the SAME v1 ProcessInfo.systemUptime contract as retained replay.

    UTC fields are source-anchor-derived labels, at exact or native millisecond
    wire precision. The original same-boot uptime guard, not rounded client
    labels or host/provider UTC equality, decides the window.
    """
    expected = {"format", "run_id", "origin_invocation_id", "monotonic_before_seconds",
                "monotonic_after_seconds", "utc_unix_seconds", "boot_session_id"}
    check(isinstance(anchor, dict) and set(anchor) == expected
          and anchor["format"] == "jarvis.replay.source_clock.v1"
          and UUID(anchor["run_id"]) == UUID(scope.run_id)
          and UUID(anchor["origin_invocation_id"]) == config.origin_invocation_id,
          "memory_source_anchor_scope")
    before = number(anchor["monotonic_before_seconds"], 0, 1e12)
    after = number(anchor["monotonic_after_seconds"], before, before + 1)
    utc = number(anchor["utc_unix_seconds"], 1, 4102444799)
    check(isinstance(sample, dict) and sample.get("clock") == "ProcessInfo.systemUptime"
          and UUID(sample["boot_session_id"]) == UUID(anchor["boot_session_id"]), "memory_source_clock_unavailable")
    uptime = number(sample["uptime_seconds"], after, 1e12)
    gates = [g for g in config.approved_gates if g["target_input_ordinal"] == request.target_input_ordinal]
    check(len(gates) == 1, "memory_target_not_approved")
    trigger = gates[0]["trigger"]
    check(request.prior_input_ordinal == trigger["after_input_ordinal"]
          and request.canonical_requirement == trigger["canonical_requirement"], "memory_trigger_binding")
    phase = config.phase_start_seconds
    start_utc = (utc + phase) + trigger["at_seconds"]
    issued = datetime.fromtimestamp(start_utc, UTC)
    expires = datetime.fromtimestamp(start_utc + trigger["timeout_seconds"], UTC)
    check(_source_clock_wire_label(request.issued_at, issued)
          and _source_clock_wire_label(request.expires_at, expires), "memory_original_deadline_mismatch")
    arm = ((before + phase) + (after + phase)) / 2 + trigger["at_seconds"]
    deadline = arm + trigger["timeout_seconds"]
    return {"clock": "ProcessInfo.systemUptime", "boot_session_id": anchor["boot_session_id"],
        "phase_start_seconds": phase, "arm_uptime_seconds": arm, "deadline_uptime_seconds": deadline,
        "sampled_uptime_seconds": uptime, "remaining_seconds": deadline-uptime,
        "status": "expired" if uptime >= deadline else "waiting" if uptime < arm else "open"}


class MemoryReadinessFileBridge:
    def __init__(self, observer, *, sample_clock=None):
        from evals.responsive_voice.frontend_host import _private_directory_identity
        self.observer, self.config = observer, observer.config
        self.root = self.config.configuration_path.parent
        self.root_identity = _private_directory_identity(self.root)
        self.sample_clock = sample_clock
        self.task = None
        self.failure = None
        self.configured = False
        self.configuration_bytes = self.runtime_bytes = self.scope_bytes = self.anchor_bytes = None
        self.control_root = self.control_identity = self.plan_path = self.anchor_path = None
        self.nonce_hashes, self.control_bindings = {}, {}
        self.last_hash = None
        self.published = self.superseded = 0
        self.lease_deadline = time.monotonic() + max(0,
            (observer.host_scope.expires_at-observer.clock()).total_seconds())

    def _identity(self):
        from evals.responsive_voice.frontend_host import _private_directory_identity
        check(_private_directory_identity(self.root) == self.root_identity, "memory_operator_root_replaced")
        if self.control_root is not None:
            check(_private_directory_identity(self.control_root) == self.control_identity, "memory_control_root_replaced")

    def status(self):
        return {"configured": self.configured, "failure_code": self.failure,
                "requests": len(self.nonce_hashes), "published": self.published, "superseded": self.superseded,
                "loop_joined": self.task is None or self.task.done()}

    def publish_status(self, detail=None):
        self._identity()
        value = {"format": STATUS_FORMAT, "run_id": self.observer.scope.run_id,
            "invocation_id": str(self.config.invocation_id), "manifest_sha256": self.config.manifest_sha256,
            "status": "failed" if self.failure else "ready" if self.configured else "awaiting_configuration",
            "semantic_acceptance": "not_evaluated", **self.status()}
        if detail is not None:
            value.update(detail)
        write_private(self.root, self.config.status_path.name, json.dumps(value).encode())

    def start(self):
        for path in (self.config.configuration_path, self.config.status_path):
            check(not path.exists() and not path.is_symlink(), "memory_operator_path_not_fresh")
        self.task = asyncio.create_task(self._run(), name="native-memory-readiness-" + self.observer.scope.run_id)

    def configure(self, data):
        from evals.responsive_voice.frontend_host import read_task_replay_file, _private_directory_identity
        value = _object(data)
        expected = {"format", "run_id", "invocation_id", "manifest_sha256", "scope_file",
                    "runtime_manifest_file", "runtime_manifest_sha256", "control_root", "source_anchor_path"}
        check(set(value) == expected and value["format"] == CONFIG_FORMAT
            and value["run_id"] == self.observer.scope.run_id
            and value["invocation_id"] == str(self.config.invocation_id)
            and value["manifest_sha256"] == self.config.manifest_sha256, "memory_configuration_scope")
        plan_path, controls = Path(value["runtime_manifest_file"]), Path(value["control_root"])
        scope_path, anchor_path = Path(value["scope_file"]), Path(value["source_anchor_path"])
        check(plan_path.name == "replay.json" and plan_path.parent.name == "inputs"
            and controls == plan_path.parent.parent / "controls"
            and scope_path == plan_path.parent / SCOPE_NAME
            and anchor_path == self.root / "source-clock.json", "memory_staging_layout")
        raw = read_task_replay_file(plan_path, limit=512*1024)
        check(hashlib.sha256(raw).hexdigest() == value["runtime_manifest_sha256"], "memory_runtime_manifest_hash")
        scope_bytes = read_task_replay_file(scope_path)
        check(_object(scope_bytes) == asdict(self.observer.scope), "memory_actual_fixture_scope_mismatch")
        plan = _object(raw)
        check(plan.get("capture_mode") == "manual" and plan.get("voice_route") == "realtime"
            and plan.get("memory_readiness") == {"version": 1, "scope_file": SCOPE_NAME,
                "manifest_sha256": self.config.manifest_sha256}
            and plan.get("replay_invocation_id") == str(self.config.invocation_id)
            and plan.get("run_id") == self.observer.scope.run_id
            and plan.get("owner_id") == self.observer.scope.owner_id
            and str(plan.get("area_id")) == self.observer.scope.area_id
            and plan.get("memory_bank_id") == self.observer.scope.memory_bank_id
            and tuple(memory_gate_rows(plan)) == self.config.approved_gates,
            "memory_runtime_scope_or_triggers")
        clock = plan.get("replay_source_clock")
        check(isinstance(clock, dict) and type(clock.get("version")) is int and clock["version"] == 1
            and clock.get("phase_start_seconds") == self.config.phase_start_seconds
            and clock.get("anchor_sha256") == self.config.source_anchor_sha256,
            "memory_runtime_source_clock_binding")
        identity = _private_directory_identity(controls)
        for name in (REQUEST_NAME, RECEIPT_NAME):
            check(not (controls/name).exists() and not (controls/name).is_symlink(), "memory_control_path_not_fresh")
        check(not anchor_path.exists() and not anchor_path.is_symlink(), "memory_anchor_not_fresh")
        self.control_root, self.control_identity = controls, identity
        self.plan_path, self.anchor_path = plan_path, anchor_path
        self.configuration_bytes, self.runtime_bytes, self.scope_bytes = data, raw, scope_bytes
        self.configured = True

    async def step(self):
        from evals.responsive_voice.frontend_host import read_task_replay_file
        self._identity()
        data = read_task_replay_file(self.config.configuration_path)
        if not self.configured:
            self.configure(data)
            self.publish_status()
            return
        check(data == self.configuration_bytes, "memory_configuration_replaced")
        try:
            request_bytes = read_task_replay_file(self.control_root/REQUEST_NAME)
        except FileNotFoundError:
            return
        sha = hashlib.sha256(request_bytes).hexdigest()
        if sha == self.last_hash:
            return
        request = parse_request(request_bytes)
        check(request.scope == self.observer.scope and request.invocation_id == str(self.config.invocation_id)
            and request.manifest_sha256 == self.config.manifest_sha256, "memory_request_scope")
        check(request.nonce not in self.nonce_hashes, "memory_nonce_reused_or_rebound")
        check(len(self.nonce_hashes) < self.config.max_consumes, "memory_consume_budget")
        self.nonce_hashes[request.nonce], self.last_hash = sha, sha
        check(read_task_replay_file(self.plan_path, limit=512*1024) == self.runtime_bytes
            and read_task_replay_file(self.plan_path.parent/SCOPE_NAME) == self.scope_bytes, "memory_staged_inputs_replaced")
        try:
            raw_anchor = read_task_replay_file(self.anchor_path)
        except FileNotFoundError:
            result = self.observer._failure(request, "validated_source_anchor_pending", status="waiting")
            clock = None
        else:
            checksum = hashlib.sha256(raw_anchor).hexdigest()
            check(self.anchor_bytes is None or self.anchor_bytes == raw_anchor, "memory_source_anchor_changed")
            check(self.config.source_anchor_sha256 is None or checksum == self.config.source_anchor_sha256,
                  "memory_source_anchor_hash_mismatch")
            if self.sample_clock is None:
                from evals.responsive_voice.native_replay_run import sample_replay_source_clock
                sample = sample_replay_source_clock()
            else:
                sample = self.sample_clock()
            clock = approved_deadline(self.config, self.observer.scope, _object(raw_anchor), request, sample)
            self.anchor_bytes = raw_anchor
            clock["anchor_sha256"] = checksum
            binding = asdict(request)
            for key in ("nonce", "response_id", "item_id"):
                binding.pop(key)
            previous = self.control_bindings.setdefault(request.target_input_ordinal, binding)
            check(previous == binding, "memory_original_control_binding_changed")
            if clock["status"] != "open":
                result = self.observer._failure(request, "approved_source_clock_window", status=clock["status"])
            else:
                try:
                    async with asyncio.timeout(min(clock["remaining_seconds"], self.config.read_timeout_seconds)):
                        result = await self.observer.observe(request)
                except TimeoutError:
                    result = self.observer._failure(request, "approved_memory_read_deadline", status="expired")
                # A result is not ready if the exact original window closed while reading.
                fresh_sample = self.sample_clock() if self.sample_clock else sample_replay_source_clock()
                clock = approved_deadline(self.config, self.observer.scope, _object(raw_anchor), request, fresh_sample)
                clock["anchor_sha256"] = checksum
                if clock["status"] != "open":
                    result = self.observer._failure(request, "approved_source_clock_window", status=clock["status"])
        self._identity()
        if read_task_replay_file(self.control_root/REQUEST_NAME) != request_bytes:
            self.superseded += 1
            return
        receipt = {"format": RECEIPT_FORMAT, "request_sha256": sha,
            "recorded_at": self.observer.clock().isoformat(), "readiness": result}
        if clock is not None:
            receipt["readiness"] = result | {"source_clock": clock}
        encoded = json.dumps(receipt, separators=(",", ":"), allow_nan=False).encode()
        check(len(encoded) <= 32768, "memory_receipt_byte_bound")
        write_private(self.control_root, RECEIPT_NAME, encoded)
        self.published += 1

    async def _run(self):
        try:
            self.publish_status()
            while (self.observer.active and self.observer.clock() < self.observer.host_scope.expires_at
                   and time.monotonic() < self.lease_deadline):
                try:
                    await self.step()
                except FileNotFoundError:
                    if self.configured:
                        raise ValueError("memory_configuration_or_request_removed") from None
                await asyncio.sleep(.05)
        except asyncio.CancelledError:
            pass
        except Exception:
            self.failure = "memory_file_control_failed"
        finally:
            self.publish_status()
