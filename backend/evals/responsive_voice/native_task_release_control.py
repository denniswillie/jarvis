"""Private, fixture-only returned-result release. Never task or speech authority.

The controller must separately establish its stimulus evidence. An eligible
native_task_barrier observation is NOT a release command. No SQL, model calls,
lease changes, task settlement or output text belong in this module.
"""
from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from evals.responsive_voice.native_task_barrier import _open_root, _regular_private
from evals.responsive_voice.native_task_result_hold import ResultHoldTimeout

MAX_BYTES = 8192
MAX_RELEASE_TTL_SECONDS = 5.0
CONFIG_NAME = "hold-config.json"


class ReleaseControlError(ValueError):
    def __init__(self, code):
        self.code = code
        super().__init__(code)


def utc(value):
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timezone_required")
    return value.astimezone(UTC)


class Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)


class HoldScope(Frozen):
    run_id: UUID
    control_id: UUID
    owner_id: str
    database_schema: str
    worker_host_id: str
    model: str = Field(min_length=1, max_length=100)
    expires_at: datetime

    _time = field_validator("expires_at")(utc)

    @model_validator(mode="after")
    def isolated(self):
        if (self.owner_id not in {f"direct-image-eval-second-brain-{self.run_id.hex}-{suffix}" for suffix in ("a", "b")}
                or not re.fullmatch(r"jarvis_sb_eval_" + self.run_id.hex + r"_[0-9]+", self.database_schema)
                or not re.fullmatch(r"native-task-" + self.run_id.hex + r"-[0-9a-f]{12}", self.worker_host_id)):
            raise ValueError("invalid_hold_fixture_scope")
        return self


class HoldConfig(Frozen):
    format: Literal["jarvis.native_task_result_hold.config.v1"] = "jarvis.native_task_result_hold.config.v1"
    scope: HoldScope
    worker_root: Path
    max_hold_seconds: float = Field(gt=0, le=60, allow_inf_nan=False)
    poll_seconds: float = Field(default=.05, ge=.005, le=.5, allow_inf_nan=False)

    @field_validator("worker_root")
    @classmethod
    def absolute(cls, value):
        if not value.is_absolute() or value != value.resolve():
            raise ValueError("canonical_worker_root_required")
        return value

    @property
    def control_root(self):
        return self.worker_root / "result-control"

    @property
    def config_path(self):
        return self.control_root / CONFIG_NAME

    def readiness(self):
        return {"format": "jarvis.native_task_result_hold.control.v1",
            "control_id": str(self.scope.control_id), "control_root": str(self.control_root),
            "config_path": str(self.config_path), "max_hold_seconds": self.max_hold_seconds,
            "expires_at": self.scope.expires_at.isoformat(),
            "hold_receipt_template": "hold-{attempt_id_hex}-{fence}.json",
            "release_command_template": "release-{hold_id_hex}.json",
            "meaning": "Explicit returned-result hold; not ongoing model computation or task completion."}


class ClaimIdentity(Frozen):
    session_id: UUID
    task_id: UUID
    task_revision: int = Field(gt=0)
    attempt_id: UUID
    fence: int = Field(gt=0)
    worker_id: str = Field(pattern=r"^frontier-[0-9a-f]{32}$")
    worker_pid: int = Field(gt=1)


class HoldReceipt(Frozen):
    format: Literal["jarvis.native_task_result_hold.receipt.v1"] = "jarvis.native_task_result_hold.receipt.v1"
    scope: HoldScope
    identity: ClaimIdentity
    hold_id: UUID
    phase: Literal["entered", "released", "timeout", "cancelled"]
    entered_at: datetime
    expires_at: datetime
    updated_at: datetime
    command_id: UUID | None = None
    rejected_commands: int = Field(default=0, ge=0, le=10000)
    last_rejection: Literal["invalid_release", "scope_mismatch", "claim_mismatch", "expired_release",
                            "early_release", "release_replayed"] | None = None

    _times = field_validator("entered_at", "expires_at", "updated_at")(utc)

    @model_validator(mode="after")
    def bounded(self):
        if (not 0 < (self.expires_at - self.entered_at).total_seconds() <= 60
                or self.expires_at > self.scope.expires_at or self.updated_at < self.entered_at
                or (self.phase == "released") != (self.command_id is not None)):
            raise ValueError("invalid_hold_receipt")
        return self


class ReleaseCommand(Frozen):
    format: Literal["jarvis.native_task_result_release.v1"] = "jarvis.native_task_result_release.v1"
    action: Literal["release_returned_result"] = "release_returned_result"
    scope: HoldScope
    identity: ClaimIdentity
    hold_id: UUID
    command_id: UUID
    sequence: Literal[1] = 1
    issued_at: datetime
    expires_at: datetime

    _times = field_validator("issued_at", "expires_at")(utc)

    @model_validator(mode="after")
    def bounded(self):
        if (type(self.sequence) is not int or not 0 < (self.expires_at - self.issued_at).total_seconds() <= MAX_RELEASE_TTL_SECONDS
                or self.expires_at > self.scope.expires_at):
            raise ValueError("invalid_release_expiry")
        return self


def _pairs(items):
    value = {}
    for key, item in items:
        if key in value:
            raise ValueError("duplicate_key")
        value[key] = item
    return value


def decode(model, data):
    try:
        if not isinstance(data, bytes) or not 0 < len(data) <= MAX_BYTES:
            raise ValueError("size")
        json.loads(data.decode(), object_pairs_hook=_pairs,
                   parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite")))
        return model.model_validate_json(data)
    except (ValueError, UnicodeError, RecursionError):
        raise ReleaseControlError("invalid_release") from None


def _name(name):
    if not re.fullmatch(r"[a-z0-9][a-z0-9_.-]{0,180}", name) or ".." in name:
        raise ReleaseControlError("invalid_control_name")


def read_private(root, name):
    _name(name)
    directory = _open_root(root)
    try:
        fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
        try:
            before = _regular_private(fd)
            if not 0 < before.st_size <= MAX_BYTES:
                raise ReleaseControlError("invalid_release")
            data = os.read(fd, MAX_BYTES + 1)
            after = os.fstat(fd)
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise ReleaseControlError("invalid_release")
            return data
        finally:
            os.close(fd)
    finally:
        os.close(directory)


def write_private(root, name, data, *, exclusive=False):
    _name(name)
    if not isinstance(data, bytes) or not 0 < len(data) <= MAX_BYTES:
        raise ReleaseControlError("invalid_release")
    directory = _open_root(root)
    temporary = name if exclusive else "pending-" + uuid4().hex
    created = False
    try:
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=directory)
        created = True
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        if not exclusive:
            os.replace(temporary, name, src_dir_fd=directory, dst_dir_fd=directory)
        os.fsync(directory)
    finally:
        if created and not exclusive:
            try:
                os.unlink(temporary, dir_fd=directory)
            except FileNotFoundError:
                pass
        os.close(directory)


def create_control(worker_root, *, run_id, owner_id, database_schema, worker_host_id,
                   model, expires_at, max_hold_seconds, poll_seconds=.05):
    if type(max_hold_seconds) not in (int, float) or not math.isfinite(max_hold_seconds):
        raise ReleaseControlError("invalid_hold_budget")
    config = HoldConfig(scope=HoldScope(run_id=UUID(str(run_id)), control_id=uuid4(), owner_id=owner_id,
        database_schema=database_schema, worker_host_id=worker_host_id, model=model, expires_at=expires_at),
        worker_root=Path(worker_root).resolve(strict=True), max_hold_seconds=float(max_hold_seconds),
        poll_seconds=float(poll_seconds))
    if not datetime.now(UTC) < config.scope.expires_at <= datetime.now(UTC) + timedelta(hours=1):
        raise ReleaseControlError("expired_hold_scope")
    config.control_root.mkdir(mode=0o700)
    write_private(config.control_root, CONFIG_NAME, config.model_dump_json().encode(), exclusive=True)
    return config


def load_config(path):
    path = Path(path)
    if not path.is_absolute() or path.name != CONFIG_NAME:
        raise ReleaseControlError("invalid_hold_config_path")
    config = decode(HoldConfig, read_private(path.parent, path.name))
    if config.config_path != path or not datetime.now(UTC) < config.scope.expires_at <= datetime.now(UTC) + timedelta(hours=1):
        raise ReleaseControlError("invalid_hold_config_scope")
    return config


def receipt_name(identity):
    return f"hold-{identity.attempt_id.hex}-{identity.fence}.json"


def read_hold(config, *, attempt_id, fence):
    name = f"hold-{UUID(str(attempt_id)).hex}-{int(fence)}.json"
    receipt = decode(HoldReceipt, read_private(config.control_root, name))
    if receipt.scope != config.scope or receipt_name(receipt.identity) != name:
        raise ReleaseControlError("scope_mismatch")
    return receipt


def _validate_release(command, receipt, now):
    if command.scope != receipt.scope:
        raise ReleaseControlError("scope_mismatch")
    if command.identity != receipt.identity or command.hold_id != receipt.hold_id:
        raise ReleaseControlError("claim_mismatch")
    if command.issued_at < receipt.entered_at or command.issued_at > now:
        raise ReleaseControlError("early_release")
    if not now < command.expires_at <= receipt.expires_at:
        raise ReleaseControlError("expired_release")


def publish_release(config, receipt, *, command_id, ttl_seconds=2.0, now=None):
    """Controller command only; caller, not this helper, proves stimulus evidence."""
    now = utc(now or datetime.now(UTC))
    if (type(ttl_seconds) not in (int, float) or not math.isfinite(ttl_seconds)
            or not 0 < ttl_seconds <= MAX_RELEASE_TTL_SECONDS):
        raise ReleaseControlError("invalid_release")
    current = read_hold(config, attempt_id=receipt.identity.attempt_id, fence=receipt.identity.fence)
    if current != receipt or receipt.phase != "entered":
        raise ReleaseControlError("claim_mismatch")
    command = ReleaseCommand(scope=config.scope, identity=receipt.identity, hold_id=receipt.hold_id,
        command_id=command_id, issued_at=now,
        expires_at=min(now + timedelta(seconds=ttl_seconds), receipt.expires_at))
    _validate_release(command, receipt, now)
    write_private(config.control_root, f"release-{receipt.hold_id.hex}.json", command.model_dump_json().encode())
    return command


class BoundReleaseWaiter:
    def __init__(self, config):
        self.config = config

    def _identity(self, context):
        config, claim, task = self.config, context.claim, context.task
        marker = json.loads(read_private(config.worker_root / "native", "worker.json"), object_pairs_hook=_pairs)
        if (set(marker) != {"format", "worker_id", "owner_digest", "host_id", "pid"}
                or marker["format"] != "jarvis.task_worker.v1"
                or marker["host_id"] != config.scope.worker_host_id or marker["pid"] != os.getpid()
                or marker["owner_digest"] != hashlib.sha256(config.scope.owner_id.encode()).hexdigest()
                or task.owner_id != config.scope.owner_id or task.model != config.scope.model
                or task.backend_kind != "frontier" or task.session_id is None or claim.recovering
                or task.current_attempt_id != claim.attempt_id or task.fence != claim.fence
                or task.revision != claim.task_revision):
            raise ReleaseControlError("claim_mismatch")
        return ClaimIdentity(session_id=task.session_id, task_id=task.id, task_revision=claim.task_revision,
            attempt_id=claim.attempt_id, fence=claim.fence, worker_id=marker["worker_id"], worker_pid=marker["pid"])

    async def __call__(self, context):
        config = self.config
        identity = self._identity(context)
        entered = datetime.now(UTC)
        # Leave a small metadata-write reserve inside ResultHoldProvider's cap.
        budget = config.max_hold_seconds - min(.1, config.max_hold_seconds / 10)
        expires = min(entered + timedelta(seconds=budget), config.scope.expires_at)
        if expires <= entered:
            raise ReleaseControlError("expired_hold_scope")
        receipt = HoldReceipt(scope=config.scope, identity=identity, hold_id=uuid4(), phase="entered",
                              entered_at=entered, expires_at=expires, updated_at=entered)
        name = receipt_name(identity)
        write_private(config.control_root, name, receipt.model_dump_json().encode(), exclusive=True)
        rejected, last_reason, seen = 0, None, None

        def terminal(phase, command_id=None):
            final = receipt.model_copy(update={"phase": phase, "command_id": command_id,
                "updated_at": datetime.now(UTC), "rejected_commands": rejected, "last_rejection": last_reason})
            write_private(config.control_root, name, final.model_dump_json().encode())

        try:
            async with asyncio.timeout((expires - entered).total_seconds()):
                while True:
                    try:
                        data = read_private(config.control_root, f"release-{receipt.hold_id.hex}.json")
                    except FileNotFoundError:
                        data = None
                    except (ValueError, OSError):
                        data = b"invalid"
                    if data is not None and data != seen:
                        seen = data
                        try:
                            command = decode(ReleaseCommand, data)
                            _validate_release(command, receipt, datetime.now(UTC))
                            consumed = json.dumps({"control_id": str(config.scope.control_id),
                                "hold_id": str(receipt.hold_id), "command_id": str(command.command_id)}).encode()
                            try:
                                write_private(config.control_root, f"consumed-{command.command_id.hex}.json", consumed, exclusive=True)
                            except FileExistsError:
                                raise ReleaseControlError("release_replayed") from None
                            terminal("released", command.command_id)
                            return
                        except ReleaseControlError as error:
                            rejected = min(10000, rejected + 1)
                            last_reason = error.code if error.code in {"scope_mismatch", "claim_mismatch",
                                "expired_release", "early_release", "release_replayed"} else "invalid_release"
                    await asyncio.sleep(config.poll_seconds)
        except TimeoutError:
            try:
                terminal("timeout")
            except Exception:
                pass
            raise ResultHoldTimeout("native_result_hold_timeout") from None
        except asyncio.CancelledError:
            try:
                terminal("cancelled")
            except Exception:
                pass
            raise
