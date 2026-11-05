"""Test-only app observations -> explicitly authorized worker-result release.

The host owns scheduling and shutdown. This component creates no task, thread,
lease or business record. A released hold is not an end-to-end passing verdict.
"""
from __future__ import annotations

from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
from typing import Literal
from uuid import UUID, uuid4

from pydantic import Field, field_validator, model_validator

from evals.responsive_voice.native_task_barrier import BarrierError, ReceiptConsumer
from evals.responsive_voice.native_task_release_control import (
    Frozen, ReleaseControlError, decode, publish_release, read_hold, read_private, utc,
)

ALL_SDK_INPUTS = "all_sdk_inputs"
NO_AUTO_RELEASE = "no_auto_release"
ResultReleasePolicy = Literal["all_sdk_inputs", "no_auto_release"]


class ReplayControlError(ValueError):
    def __init__(self, code):
        self.code = code
        super().__init__(code)


class NativeTaskScope(Frozen):
    format: Literal["jarvis.native_task_scope.v1"] = "jarvis.native_task_scope.v1"
    run_id: UUID
    barrier_id: UUID
    owner_id: str = Field(min_length=1, max_length=200)
    database_schema: str = Field(min_length=1, max_length=200)
    worker_id: str = Field(min_length=1, max_length=100)
    runtime_host_id: str = Field(min_length=1, max_length=200)
    model: str = Field(min_length=1, max_length=100)
    host_expires_at: datetime

    _time = field_validator("host_expires_at")(utc)


class SessionBinding(Frozen):
    format: Literal["jarvis.native_task_session_binding.v1"] = "jarvis.native_task_session_binding.v1"
    run_id: UUID
    barrier_id: UUID
    owner_id: str = Field(min_length=1, max_length=200)
    session_id: UUID
    manifest_file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    scope_file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    recorded_at: datetime

    _time = field_validator("recorded_at")(utc)


class CompletedInput(Frozen):
    input_ordinal: int = Field(ge=0, lt=20)
    status: Literal["sdk_input_complete"] = "sdk_input_complete"
    sample_count: int = Field(gt=0, le=60 * 48000)
    accepted_samples: int = Field(gt=0, le=60 * 48000)
    first_sdk_callback_monotonic: float = Field(ge=0, allow_inf_nan=False)
    last_sdk_callback_end_monotonic: float = Field(ge=0, allow_inf_nan=False)
    task_claim_sequence_at_enqueue: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def complete(self):
        if (self.accepted_samples != self.sample_count
                or self.last_sdk_callback_end_monotonic < self.first_sdk_callback_monotonic):
            raise ValueError("incomplete_sdk_input")
        return self


class StimulusCompletion(SessionBinding):
    format: Literal["jarvis.native_task_stimulus_completion.v1"] = "jarvis.native_task_stimulus_completion.v1"
    inputs: tuple[CompletedInput, ...] = Field(min_length=1, max_length=20)


def _object(data):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate_key")
            result[key] = value
        return result
    try:
        if not isinstance(data, bytes) or not 0 < len(data) <= 512 * 1024:
            raise ValueError("size")
        value = json.loads(data, object_pairs_hook=pairs,
            parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite")))
        if not isinstance(value, dict):
            raise ValueError("object_required")
        return value
    except (ValueError, UnicodeError, RecursionError):
        raise ReplayControlError("invalid_task_manifest") from None


class NativeTaskReplayRelease:
    """Call step from the isolated host's owned, bounded control loop.

    Missing files/claims/results are waits. Malformed app observations fail the
    trial closed. Once a release write is attempted it is never retried, even if
    the write outcome is unknown; the same hold receipt supplies acknowledgment.
    """

    def __init__(self, *, observer_host, worker_readiness, control_root,
                 manifest_bytes, scope_bytes, input_sample_counts, hold_config,
                 publisher=publish_release, result_release_policy=ALL_SDK_INPUTS,
                 scenario=None):
        if (type(result_release_policy) is not str
                or result_release_policy not in {ALL_SDK_INPUTS, NO_AUTO_RELEASE}
                or result_release_policy == NO_AUTO_RELEASE and scenario != "explicit_cancel"):
            raise ReplayControlError("invalid_result_release_policy")
        self.result_release_policy = result_release_policy
        self.bridge, self.root, self.config = observer_host, Path(control_root), hold_config
        self.scope = decode(NativeTaskScope, scope_bytes)
        self.manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
        self.scope_hash = hashlib.sha256(scope_bytes).hexdigest()
        self.samples = tuple(input_sample_counts)
        if (not 1 <= len(self.samples) <= 20
                or any(type(n) is not int or not 0 < n <= 60 * 48000 for n in self.samples)):
            raise ReplayControlError("invalid_expected_samples")
        manifest = _object(manifest_bytes)
        try:
            declaration = manifest["task_barrier"]
            utterances = manifest["utterances"]
            if (manifest["version"] != 1 or manifest["voice_route"] != "realtime"
                    or UUID(manifest["run_id"]) != self.scope.run_id
                    or manifest["owner_id"] != self.scope.owner_id
                    or declaration != {"version": 1, "scope_file": "task-scope.json", "request_input_ordinal": 0}
                    or not isinstance(utterances, list) or len(utterances) != len(self.samples)):
                raise ValueError("binding")
            self.required_claims = tuple(i for i, item in enumerate(utterances)
                if item["trigger"]["kind"] == "after_task_claim"
                or item["trigger"].get("require_task_claim") is True)
            if not any(item["trigger"]["kind"] == "after_task_claim" for item in utterances):
                raise ValueError("missing_claim_gate")
        except (KeyError, TypeError, ValueError, AttributeError):
            raise ReplayControlError("invalid_task_manifest") from None
        r, s, h = worker_readiness, self.scope, hold_config.scope
        try:
            if (r["status"] != "ready" or r["registry_ready"] is not True
                    or UUID(str(r["run_id"])) != s.run_id or r["owner_id"] != s.owner_id
                    or r["database_schema"] != s.database_schema or r["model"] != s.model
                    or r["worker_id"] != s.worker_id or r["host_id"] != s.runtime_host_id
                    or utc(datetime.fromisoformat(r["host_expires_at"])) != s.host_expires_at
                    or type(r["pid"]) is not int or r["pid"] <= 1
                    or (h.run_id, h.owner_id, h.database_schema, h.worker_host_id, h.model)
                    != (s.run_id, s.owner_id, s.database_schema, s.runtime_host_id, s.model)
                    or h.expires_at > s.host_expires_at):
                raise ValueError("readiness")
        except (KeyError, TypeError, ValueError, AttributeError):
            raise ReplayControlError("worker_scope_mismatch") from None
        self.worker_pid = r["pid"]
        self.publisher = publisher
        self.state, self.reason = "waiting", "session_binding_missing"
        self.command_id = self.held = None

    def status(self):
        result = {"status": self.state, "reason": self.reason,
            "release_command_id": str(self.command_id) if self.command_id else None,
            "task_id": str(self.held.identity.task_id) if self.held else None,
            "attempt_id": str(self.held.identity.attempt_id) if self.held else None,
            "semantic_acceptance": "unmeasured"}
        if self.result_release_policy == NO_AUTO_RELEASE:
            result["result_release_policy"] = self.result_release_policy
            result["cancellation_qualification"] = "unmeasured"
        return result

    def _binding(self, value, now):
        s = self.scope
        if ((value.run_id, value.barrier_id, value.owner_id) != (s.run_id, s.barrier_id, s.owner_id)
                or value.manifest_file_sha256 != self.manifest_hash
                or value.scope_file_sha256 != self.scope_hash
                or value.recorded_at > now or value.recorded_at >= s.host_expires_at):
            raise ReplayControlError("app_binding_mismatch")

    def _candidate(self, now):
        binding = decode(SessionBinding, read_private(self.root, "session-binding.json"))
        self._binding(binding, now)
        completion = decode(StimulusCompletion, read_private(self.root, "stimulus-completion.json"))
        self._binding(completion, now)
        if completion.session_id != binding.session_id or completion.recorded_at < binding.recorded_at:
            raise ReplayControlError("app_session_mismatch")
        bridge = self.bridge
        if bridge.scope is None or bridge.admission is None or bridge.claim is None:
            self.reason = "canonical_claim_pending"
            return None
        for key in type(self.scope).model_fields:
            if key != "format" and getattr(self.scope, key) != getattr(bridge.scope, key):
                raise ReplayControlError("canonical_scope_mismatch")
        if bridge.scope.session_id != binding.session_id:
            raise ReplayControlError("canonical_session_mismatch")
        receipt = ReceiptConsumer(bridge.scope, admission=bridge.admission, claim=bridge.claim).consume(
            bridge.exchange.read(), now=now)
        if len(completion.inputs) != len(self.samples):
            raise ReplayControlError("input_coverage_mismatch")
        last = -1.0
        for ordinal, (actual, expected) in enumerate(zip(completion.inputs, self.samples, strict=True)):
            if (actual.input_ordinal != ordinal or actual.sample_count != expected
                    or actual.first_sdk_callback_monotonic < last):
                raise ReplayControlError("input_coverage_mismatch")
            sequence = actual.task_claim_sequence_at_enqueue
            if ((ordinal in self.required_claims and sequence is None)
                    or (sequence is not None and sequence > receipt.sequence)):
                raise ReplayControlError("input_claim_evidence_mismatch")
            last = actual.last_sdk_callback_end_monotonic
        held = read_hold(self.config, attempt_id=receipt.claim.attempt_id, fence=receipt.claim.fence)
        identity, claim = held.identity, receipt.claim
        if ((identity.session_id, identity.task_id, identity.task_revision, identity.attempt_id,
             identity.fence, identity.worker_id, identity.worker_pid)
                != (binding.session_id, claim.task_id, claim.task_revision, claim.attempt_id,
                    claim.fence, claim.worker_id, self.worker_pid)):
            raise ReplayControlError("held_claim_mismatch")
        if held.phase != "entered" or now >= held.expires_at:
            raise ReplayControlError("hold_not_pending")
        return held

    def _acknowledge(self, now):
        held = read_hold(self.config, attempt_id=self.held.identity.attempt_id, fence=self.held.identity.fence)
        if held.hold_id != self.held.hold_id or held.identity != self.held.identity:
            raise ReplayControlError("held_claim_mismatch")
        if held.phase == "released" and held.command_id == self.command_id:
            self.state, self.reason = "released", "worker_acknowledged_release"
        elif held.phase != "entered" or now >= held.expires_at:
            raise ReplayControlError("release_not_acknowledged")

    def _observe_withheld(self, now):
        """Observe worker cleanup only; a cancelled waiter is not task cancellation proof."""
        held = read_hold(self.config, attempt_id=self.held.identity.attempt_id, fence=self.held.identity.fence)
        if held.hold_id != self.held.hold_id or held.identity != self.held.identity:
            raise ReplayControlError("held_claim_mismatch")
        if held.phase == "released" or held.command_id is not None:
            raise ReplayControlError("unexpected_result_release")
        if held.phase == "cancelled":
            self.state, self.reason = "hold_ended", "hold_cancelled_outcome_unmeasured"
        elif held.phase == "timeout" or now >= held.expires_at:
            raise ReplayControlError("hold_bound_expired_without_cancel_proof")
        else:
            self.state, self.reason = "withheld", "automatic_release_disabled"

    def step(self, *, now=None):
        now = utc(now or datetime.now(UTC))
        if self.state in {"released", "failed", "hold_ended"}:
            return self.status()
        try:
            if self.command_id is not None:
                # Settlement may already make the canonical claim terminal.
                # Acknowledge our exact command instead of demanding RUNNING again.
                self._acknowledge(now)
            elif now >= min(self.scope.host_expires_at, self.config.scope.expires_at):
                raise ReplayControlError("replay_scope_expired")
            elif self.result_release_policy == NO_AUTO_RELEASE and self.held is not None:
                self._observe_withheld(now)
            else:
                held = self._candidate(now)
                if held is not None:
                    if self.result_release_policy == NO_AUTO_RELEASE:
                        self.held = held
                        self.state, self.reason = "withheld", "automatic_release_disabled"
                        return self.status()
                    self.held, self.command_id = held, uuid4()
                    self.state, self.reason = "release_requested", "awaiting_worker_ack"
                    try:
                        self.publisher(self.config, held, command_id=self.command_id, now=now)
                    except Exception:
                        # An atomic write may have succeeded before its fsync/return
                        # failed. Never issue another command on that uncertainty.
                        self.reason = "release_write_outcome_unknown"
        except FileNotFoundError:
            self.reason = "required_observation_missing"
        except BarrierError:
            self.reason = "canonical_claim_not_eligible"
        except (ReplayControlError, ReleaseControlError, ValueError, OSError) as error:
            self.state = "failed"
            self.reason = error.code if isinstance(error, ReplayControlError) else "invalid_control_observation"
        return self.status()
