"""Fixture-only supervision of the existing frontier CLI, never another worker.

Calling start performs real DB/provider/native work. Importing this module does
not. Main must authorize the integrated trial separately. Host/runner callers
must await close before export or fixture deletion, and require status=closed.
"""
from __future__ import annotations

import asyncio
from copy import deepcopy
from datetime import UTC, datetime, timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import stat
import sys
import tempfile
import time
from urllib.parse import parse_qs, urlsplit
from uuid import uuid4

import asyncpg
from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.runtime.feature_config import FrontierModelConfiguration, TaskLimits
from app.tasks.catalog import TaskWorkerRegistry
from evals.responsive_voice.frontend_host import ReplayHostScope, scoped_database_url
from evals.second_brain.fixtures import validate_manifest


BACKEND = Path(__file__).resolve().parents[2]
HERMES_VERSION = "0.19.0"  # Enforced by the existing worker's native probe.

# Exact static names/codes emitted by app.tasks.worker.main/run_command and its
# local worker/configuration paths. Imported or future errors stay redacted.
_STARTUP_FAILURE_MAX_LINE_BYTES = 512
_STARTUP_FAILURE_TYPES = frozenset({"TaskError", "ValueError", "ValidationError"})
_STARTUP_FAILURE_CODES = frozenset({
    "task_worker_startup_failed", "task_worker_not_configured", "task_schema_not_ready",
    "task_worker_root_in_use", "task_worker_stopped", "sandbox_cleanup_worker_stopped",
    "artifact_configuration_invalid", "sandbox_configuration_invalid",
    "task_model_not_accessible", "task_memory_not_configured", "artifact_worker_unavailable",
})
_STARTUP_DIAGNOSTIC_SOURCES = {
    ("worker", "worker.py"): frozenset({"main", "run_command", "worker_settings", "probe_native",
        "worker_health", "__init__", "from_flags", "native", "absolute_path", "stable_host", "run"}),
    ("knowledge_runtime", "runtime.py"): frozenset({"create", "__init__", "start", "close", "set_file_publications"}),
    ("knowledge_client", "hindsight.py"): frozenset({"__init__"}),
    ("knowledge_outbox", "outbox.py"): frozenset({"__init__"}),
    ("knowledge_ingestion", "ingestion.py"): frozenset({"__init__"}),
    ("sandbox_runtime", "runtime.py"): frozenset({"load_sandbox_runtime"}),
    ("hermes_initialization", "hermes.py"): frozenset({"__init__", "ensure_private_runtime_root"}),
    ("model_gateway", "model_gateway.py"): frozenset({"__init__"}),
    ("storage_configuration", "supabase_config.py"): frozenset({"get_supabase_storage_config"}),
}


def _valid_startup_diagnostic(value):
    if (type(value) is not dict or set(value) != {"version", "stage", "frame", "basis", "truncated"}
            or type(value["version"]) is not int or value["version"] != 1
            or type(value["stage"]) is not str or type(value["basis"]) is not str
            or type(value["truncated"]) is not bool):
        return False
    frame = value["frame"]
    if frame is None:
        return value["stage"] == "unavailable" and value["basis"] == "unavailable"
    if (type(frame) is not dict or set(frame) != {"source", "function", "line"}
            or type(frame["source"]) is not str or type(frame["function"]) is not str
            or type(frame["line"]) is not int or not 1 <= frame["line"] <= 1_000_000):
        return False
    functions = _STARTUP_DIAGNOSTIC_SOURCES.get((value["stage"], frame["source"]), ())
    return (frame["function"] in functions
            and value["basis"] in {"exception_origin", "filtered_caller"}
            and not (value["basis"] == "exception_origin" and value["truncated"]))


def _unique_json_fields(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate field")
        result[key] = value
    return result


class NativeTaskWorkerError(RuntimeError):
    def __init__(self, code, *, receipt=None, resource_manifest=None):
        super().__init__(code)
        self.code, self.receipt, self.resource_manifest = code, receipt, resource_manifest


class NativeTaskWorkerOptions(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    model: str
    api_key_file: Path
    hermes_python: Path
    artifact_dir: Path
    result_hold_seconds: float | None = None
    reasoning_effort: str = "low"
    worker_python: Path = Field(default_factory=lambda: Path(sys.executable))
    readiness_timeout_seconds: float = Field(default=45, gt=0, le=120, allow_inf_nan=False)
    poll_seconds: float = Field(default=.1, gt=0, le=1, allow_inf_nan=False)
    stop_timeout_seconds: float = Field(default=5, gt=0, le=15, allow_inf_nan=False)
    kill_timeout_seconds: float = Field(default=2, gt=0, le=10, allow_inf_nan=False)
    withdrawal_timeout_seconds: float = Field(default=5, gt=0, le=20, allow_inf_nan=False)
    io_timeout_seconds: float = Field(default=5, gt=0, le=10, allow_inf_nan=False)
    pipe_cap_bytes: int = Field(default=65536, ge=1, le=1048576, strict=True)
    task_limits: TaskLimits = Field(default_factory=lambda: TaskLimits(
        owner_concurrency=1, global_concurrency=1, active_seconds=120, tool_steps=20,
        request_output_tokens=4096, task_output_tokens=8192, task_input_tokens=64000,
        owner_daily_tokens=128000, max_attempts=2))

    @field_validator("result_hold_seconds", mode="before")
    @classmethod
    def result_hold_bound(cls, value):
        if value is not None and (type(value) not in (int, float)
                or not math.isfinite(value) or not 0 < value <= 60):
            raise ValueError("invalid_result_hold_budget")
        return value

    @field_validator("api_key_file", "hermes_python", "artifact_dir", "worker_python")
    @classmethod
    def absolute(cls, value):
        if not value.is_absolute():
            raise ValueError("Explicit absolute fixture paths required")
        return value

    @property
    def shutdown_budget(self):
        return (self.stop_timeout_seconds + self.kill_timeout_seconds +
                self.withdrawal_timeout_seconds + 3 * self.io_timeout_seconds + 1)


def expected_host_environment(options: NativeTaskWorkerOptions) -> dict[str, str]:
    """Caller applies BEFORE host startup; this helper never edits the host."""
    FrontierModelConfiguration(model=options.model, reasoning_effort=options.reasoning_effort)
    return {**{"SECOND_BRAIN_" + key.upper(): str(value)
               for key, value in options.task_limits.model_dump().items()},
        "SECOND_BRAIN_ENABLED": "true", "SECOND_BRAIN_TASKS_ENABLED": "true",
        "SECOND_BRAIN_FRONTIER_ENABLED": "true", "SECOND_BRAIN_FRONTIER_MODEL": options.model,
        "SECOND_BRAIN_FRONTIER_REASONING_EFFORT": options.reasoning_effort,
        "SECOND_BRAIN_ATTENTION_VOICE_ENABLED": "true", "REMINDER_SCHEDULER_ENABLED": "true",
        "SECOND_BRAIN_SANDBOX_ENABLED": "false", "SECOND_BRAIN_ARTIFACTS_ENABLED": "false",
        "SECOND_BRAIN_PUSH_ENABLED": "false", "SECOND_BRAIN_MAIL_ENABLED": "false"}


def _require(value, code):
    if not value:
        raise NativeTaskWorkerError(code)


def _read_private(path, *, maximum=131072):
    path = Path(path)
    _require(path.is_absolute() and not path.is_symlink(), "fixture_private_path_required")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(descriptor, "rb") as handle:
        info = os.fstat(handle.fileno())
        _require(stat.S_ISREG(info.st_mode) and info.st_uid == os.getuid()
                 and info.st_mode & 0o077 == 0, "fixture_private_permissions_required")
        data = handle.read(maximum + 1)
    _require(len(data) <= maximum, "fixture_file_too_large")
    return data


def _secret(path):
    value = _read_private(path, maximum=8193).decode("utf-8").strip()
    _require(value and len(value) <= 8192 and not any(c in value for c in "\n\r\x00"),
             "fixture_secret_invalid")
    return value


def _write_private(path, value):
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(value)


async def _open_pool(dsn):
    return await asyncpg.create_pool(dsn=dsn, min_size=1, max_size=1, timeout=5, command_timeout=5)


async def _launch(*args, **kwargs):
    return await asyncio.create_subprocess_exec(*args, **kwargs)


class _PipeDrain:
    def __init__(self, stream, cap):
        self.stream, self.cap = stream, cap
        self.tail = bytearray()
        self.bytes_seen, self.eof, self.error = 0, False, None
        self._startup_line = bytearray()
        self._startup_line_oversized = False
        self._startup_failure = None

    def _capture_startup_line(self):
        if self._startup_failure is not None or self._startup_line_oversized:
            return
        try:
            value = json.loads(self._startup_line.decode("utf-8"), object_pairs_hook=_unique_json_fields)
        except (ValueError, UnicodeError, RecursionError):
            return
        if (type(value) is not dict or set(value) not in (
                {"ready", "error_type", "code"}, {"ready", "error_type", "code", "diagnostic"})
                or value["ready"] is not False
                or type(value["error_type"]) is not str or type(value["code"]) is not str):
            return
        if "diagnostic" in value and not _valid_startup_diagnostic(value["diagnostic"]):
            return
        self._startup_failure = {
            "ready": False,
            "error_type": (value["error_type"] if value["error_type"] in _STARTUP_FAILURE_TYPES
                           else "unrecognized_error_type"),
            "code": value["code"] if value["code"] in _STARTUP_FAILURE_CODES else "unrecognized_code",
        }
        if "diagnostic" in value:
            self._startup_failure["diagnostic"] = value["diagnostic"]

    def _consume_startup_lines(self, data):
        if self._startup_failure is not None:
            return
        parts = data.split(b"\n")
        for index, part in enumerate(parts):
            if not self._startup_line_oversized:
                if len(self._startup_line) + len(part) > _STARTUP_FAILURE_MAX_LINE_BYTES:
                    self._startup_line.clear()
                    self._startup_line_oversized = True
                else:
                    self._startup_line.extend(part)
            if index < len(parts) - 1:
                self._capture_startup_line()
                self._startup_line.clear()
                self._startup_line_oversized = False
                if self._startup_failure is not None:
                    return

    async def run(self):
        try:
            while data := await self.stream.read(8192):
                self.bytes_seen += len(data)
                self._consume_startup_lines(data)
                self.tail.extend(data)
                del self.tail[:-self.cap]
            # A complete final JSON object needs no newline; an incomplete one
            # is not evidence. Cancelled/failed drains do not parse their tail.
            self._capture_startup_line()
            self.eof = True
        except asyncio.CancelledError:
            self.error = "drain_cancelled"
            raise
        except Exception:
            self.error = "drain_failed"
        finally:
            self._startup_line.clear()

    def receipt(self):
        # Raw child output may contain credentials or task text. Never export it.
        # The CLI can report failure after startup too: this parsed report alone
        # establishes neither startup chronology, readiness nor clean shutdown.
        return {"bytes_seen": self.bytes_seen, "retained_bytes": len(self.tail),
                "truncated": self.bytes_seen > self.cap, "eof": self.eof, "error": self.error,
                "startup_failure": deepcopy(self._startup_failure) if self._startup_failure is not None else None}


class NativeTaskWorker:
    def __init__(self, options, resources, dsn):
        self.options, self.resources = options, dict(resources)
        self.readiness_receipt = self.close_receipt = None
        self.resource_manifest = self.root = self.pool = self.process = None
        self.worker_id = self.host_id = None
        self.result_hold_control = None
        self._join = self._watcher = self._close_task = None
        self._drains, self._drain_tasks, self._secrets = [], [], []
        self._dsn = dsn
        self._reason = None

    def _persist(self, status):
        if self.resource_manifest is None:
            return
        value = {"format": "jarvis.native_task_worker.v1", "status": status,
            "run_id": self.scope.run_id, "owner_id": self.scope.owner_id,
            "database_schema": self.scope.database_schema, "area_id": str(self.scope.area_id),
            "memory_bank_id": self.scope.memory_bank_id, "model": self.options.model,
            "reasoning_effort": self.options.reasoning_effort, "host_id": self.host_id,
            "worker_id": self.worker_id, "pid": self.process.pid if self.process else None,
            "root": str(self.root), "host_expires_at": self.scope.expires_at.isoformat(),
            "readiness": self.readiness_receipt, "close": self.close_receipt}
        if self.result_hold_control is not None:
            value["result_hold_control"] = self.result_hold_control
        temporary = self.resource_manifest.with_name("receipt-" + uuid4().hex + ".tmp")
        _write_private(temporary, json.dumps(value, indent=2) + "\n")
        temporary.replace(self.resource_manifest)

    def _validate(self):
        options, resources = self.options, self.resources
        expected_host_environment(options)
        _require(options.artifact_dir.is_dir() and not options.artifact_dir.is_symlink(),
                 "fixture_artifact_directory_invalid")
        for name in ("fixture_manifest", "scope_file"):
            path = Path(resources[name])
            _require(path.parent.resolve() == options.artifact_dir.resolve(), "fixture_manifest_directory_mismatch")
        fixture = json.loads(_read_private(resources["fixture_manifest"]))
        validate_manifest(fixture)
        self.scope = ReplayHostScope.model_validate_json(_read_private(resources["scope_file"]))
        scope = self.scope
        _require(resources.get("format") == "jarvis.frontend_host.resources.v1"
                 and resources.get("status") in {"prepared", "serving"}
                 and resources.get("capability_profile") == "native_realtime_media", "fixture_host_not_prepared")
        _require(fixture["status"] == "prepared" and fixture["run_id"] == scope.run_id
                 and scope.owner_id in fixture["owners"] and scope.database_schema in fixture["schemas"],
                 "fixture_database_manifest_mismatch")
        for key, expected in {"run_id": scope.run_id, "owner_id": scope.owner_id,
                "database_schema": scope.database_schema, "area_id": str(scope.area_id),
                "bank_name": scope.memory_bank_id}.items():
            _require(resources.get(key) == expected, "fixture_scope_binding_mismatch")
        _require(resources.get("memory_bank_id", scope.memory_bank_id) == scope.memory_bank_id,
                 "fixture_bank_binding_mismatch")
        if resources.get("expires_at"):
            _require(datetime.fromisoformat(resources["expires_at"]) == scope.expires_at, "fixture_expiry_mismatch")
        for key, expected in (("worker_model", options.model), ("worker_reasoning_effort", options.reasoning_effort)):
            _require(resources.get(key, expected) == expected, "fixture_model_binding_mismatch")
        remaining = (scope.expires_at - datetime.now(UTC)).total_seconds()
        _require(options.shutdown_budget + 1 < remaining <= 3600, "fixture_worker_lease_insufficient")
        self._deadline = time.monotonic() + remaining - options.shutdown_budget
        self.stop_at = scope.expires_at - timedelta(seconds=options.shutdown_budget)
        for interpreter in (options.worker_python, options.hermes_python):
            _require(interpreter.is_file() and os.access(interpreter, os.X_OK), "fixture_interpreter_unavailable")
        parts = urlsplit(self._dsn)
        _require(parts.scheme in {"postgres", "postgresql"} and parts.hostname and parts.path not in {"", "/"},
                 "fixture_database_url_invalid")
        query = parse_qs(parts.query, keep_blank_values=True)
        expected = f'"{scope.database_schema}",pg_catalog,extensions'
        _require("options" not in query and ("search_path" not in query or query["search_path"] == [expected]),
                 "fixture_database_path_mismatch")
        self._dsn = scoped_database_url(self._dsn, scope.database_schema)
        memory = urlsplit(resources["hindsight_url"])
        _require(memory.scheme == "http" and memory.hostname == "127.0.0.1"
                 and memory.port is not None and not memory.username and not memory.password
                 and not memory.query and not memory.fragment and memory.path in {"", "/"},
                 "fixture_memory_origin_invalid")

    async def _scope_check(self, connection):
        schema = await connection.fetchval("select current_schema()")
        paths = await connection.fetchval("select current_schemas(false)")
        _require(schema == self.scope.database_schema and paths and paths[0] == schema
                 and not set(paths) - {schema, "pg_catalog", "extensions"}, "fixture_actual_schema_mismatch")
        row = await connection.fetchrow("""select b.bank_name,a.status
            from jarvis_memory_banks b join jarvis_knowledge_areas a
              on a.owner_id=b.owner_id and a.id=b.area_id
            where b.owner_id=$1 and b.area_id=$2""", self.scope.owner_id, self.scope.area_id)
        _require(row is not None and row["bank_name"] == self.scope.memory_bank_id and row["status"] == "active",
                 "fixture_actual_bank_mismatch")

    async def _registry_state(self):
        async with asyncio.timeout(self.options.io_timeout_seconds), self.pool.acquire() as connection:
            async with connection.transaction(readonly=True):
                await self._scope_check(connection)
                rows = await connection.fetch("""select worker_id,model,draining,
                    expires_at>clock_timestamp() as unexpired from jarvis_task_workers
                    where owner_id=$1 and host_id=$2 and backend_kind='frontier'""",
                    self.scope.owner_id, self.host_id)
                _require(len(rows) <= 1 and all(row["model"] == self.options.model
                    and (self.worker_id is None or row["worker_id"] == self.worker_id) for row in rows),
                    "fixture_worker_registry_binding_mismatch")
                ready = await TaskWorkerRegistry(self.pool).ready(self.scope.owner_id, "frontier",
                    self.options.model, worker_id=self.worker_id, connection=connection) if self.worker_id else False
        if not rows:
            return "absent", False
        row = rows[0]
        if row["draining"]:
            return "withdrawn", ready
        if not row["unexpired"]:
            return "expired", ready
        return "active", ready

    def _marker(self):
        path = self.root / "native" / "worker.json"
        if not path.exists():
            return False
        marker = json.loads(_read_private(path, maximum=4096))
        _require(marker.get("format") == "jarvis.task_worker.v1"
                 and marker.get("host_id") == self.host_id and marker.get("pid") == self.process.pid
                 and marker.get("owner_digest") == hashlib.sha256(self.scope.owner_id.encode()).hexdigest()
                 and isinstance(marker.get("worker_id"), str) and 0 < len(marker["worker_id"]) <= 200,
                 "fixture_worker_marker_mismatch")
        _require(self.worker_id in {None, marker["worker_id"]}, "fixture_worker_identity_changed")
        self.worker_id = marker["worker_id"]
        return True

    async def _start(self):
        self._validate()
        api_key = _secret(self.options.api_key_file)
        memory_key = _secret(self.resources["hindsight_api_key_file"])
        self.root = Path(tempfile.mkdtemp(prefix=f"native-task-worker-{self.scope.run_id}-",
                                         dir=self.options.artifact_dir))
        self.root.chmod(0o700)
        self.resource_manifest = self.root / "worker-supervision.json"
        self.host_id = "native-task-" + self.scope.run_id + "-" + uuid4().hex[:12]
        self._persist("preparing")
        for name in ("cwd", "home"):
            (self.root / name).mkdir(mode=0o700)
        for name, value in (("db.secret", self._dsn), ("openai.secret", api_key), ("hindsight.secret", memory_key)):
            path = self.root / name
            _write_private(path, value)
            self._secrets.append(path)
        del api_key, memory_key
        self.pool = await asyncio.wait_for(_open_pool(self._dsn), self.options.io_timeout_seconds)
        async with asyncio.timeout(self.options.io_timeout_seconds), self.pool.acquire() as connection:
            async with connection.transaction(readonly=True):
                await self._scope_check(connection)
        env = {**expected_host_environment(self.options), "PATH": "/usr/bin:/bin",
            "HOME": str(self.root / "home"), "TMPDIR": str(self.root / "cwd"), "LANG": "C.UTF-8",
            "PYTHONPATH": str(BACKEND), "PYTHONUNBUFFERED": "1", "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1", "JARVIS_OWNER_ID": self.scope.owner_id,
            "JARVIS_FRONTIER_DB_URL_FILE": str(self.root / "db.secret"),
            "JARVIS_FRONTIER_OPENAI_API_KEY_FILE": str(self.root / "openai.secret"),
            "SECOND_BRAIN_HERMES_PYTHON": str(self.options.hermes_python),
            "SECOND_BRAIN_HERMES_WORK_ROOT": str(self.root / "native"),
            "SECOND_BRAIN_WORKER_HOST_ID": self.host_id, "SECOND_BRAIN_WORKER_HEARTBEAT_SECONDS": "1",
            "SECOND_BRAIN_KNOWLEDGE_ENABLED": "true", "SECOND_BRAIN_MEMORY_AUTHORITY": "hindsight",
            "SECOND_BRAIN_KNOWLEDGE_AREA_ID": str(self.scope.area_id),
            "SECOND_BRAIN_HINDSIGHT_URL": self.resources["hindsight_url"],
            "SECOND_BRAIN_HINDSIGHT_PRIVATE_HTTP_ORIGIN": self.resources["hindsight_url"],
            "SECOND_BRAIN_HINDSIGHT_API_KEY_FILE": str(self.root / "hindsight.secret"),
            "SECOND_BRAIN_ATTENTION_VOICE_ENABLED": "false", "REMINDER_SCHEDULER_ENABLED": "false",
            "SECOND_BRAIN_RESPONSIBILITIES_ENABLED": "false", "MEMORY_JOB_RUNNER_ENABLED": "false",
            "MEMORY_MAINTENANCE_ENABLED": "false", "SUPABASE_URL": "", "SUPABASE_SERVICE_ROLE_KEY": ""}
        command = ("app.tasks.worker", "run")
        if self.options.result_hold_seconds is not None:
            from evals.responsive_voice.native_task_release_control import create_control
            control = create_control(self.root, run_id=self.scope.run_id, owner_id=self.scope.owner_id,
                database_schema=self.scope.database_schema, worker_host_id=self.host_id,
                model=self.options.model, expires_at=self.stop_at,
                max_hold_seconds=self.options.result_hold_seconds)
            self.result_hold_control = control.readiness()
            env["JARVIS_NATIVE_RESULT_HOLD_CONFIG"] = str(control.config_path)
            command = ("evals.responsive_voice.native_task_worker_entry",)
        _require(time.monotonic() < self._deadline, "fixture_worker_lease_expired_before_spawn")
        self.process = await _launch(str(self.options.worker_python), "-m", *command,
            cwd=str(self.root / "cwd"), env=env, start_new_session=True,
            stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        _require(type(self.process.pid) is int and self.process.pid > 1, "fixture_child_identity_invalid")
        self._join = asyncio.create_task(self.process.wait())
        self._drains = [_PipeDrain(stream, self.options.pipe_cap_bytes)
                        for stream in (self.process.stdout, self.process.stderr)]
        self._drain_tasks = [asyncio.create_task(drain.run()) for drain in self._drains]
        self._watcher = asyncio.create_task(self._watch(), name="native-fixture-worker-lease")
        self._persist("starting")
        deadline = min(self._deadline, time.monotonic() + self.options.readiness_timeout_seconds)
        while time.monotonic() < deadline:
            _require(self.process.returncode is None and self._close_task is None, "fixture_worker_exited_before_ready")
            if self._marker():
                state, ready = await self._registry_state()
                if ready and state == "active":
                    _require(self.process.returncode is None and self._close_task is None
                             and time.monotonic() < self._deadline, "fixture_worker_no_longer_current")
                    self.readiness_receipt = {"status": "ready", "registry_ready": True,
                        "run_id": self.scope.run_id, "owner_id": self.scope.owner_id,
                        "database_schema": self.scope.database_schema, "area_id": str(self.scope.area_id),
                        "memory_bank_id": self.scope.memory_bank_id, "model": self.options.model,
                        "reasoning_effort": self.options.reasoning_effort, "worker_id": self.worker_id,
                        "host_id": self.host_id, "pid": self.process.pid, "hermes_version_pin": HERMES_VERSION,
                        "host_expires_at": self.scope.expires_at.isoformat(), "stop_at": self.stop_at.isoformat(),
                        "resource_manifest": str(self.resource_manifest),
                        "expected_host_environment": expected_host_environment(self.options),
                        "meaning": "Observed worker lease, not task success, delivery or hearing."}
                    if self.result_hold_control is not None:
                        self.readiness_receipt["result_hold_control"] = self.result_hold_control
                    self._persist("ready")
                    return self
            await asyncio.sleep(self.options.poll_seconds)
        raise NativeTaskWorkerError("fixture_worker_readiness_timeout")

    async def _watch(self):
        await asyncio.wait((self._join,), timeout=max(0, self._deadline - time.monotonic()))
        await self.close(reason="worker_exit" if self._join.done() else "host_lease")

    def _signal(self, sig):
        # Never accept a PID from a manifest or caller. Only this new-session
        # child handle grants signalling authority, while it remains unjoined.
        if self.process.returncode is not None:
            return
        _require(os.getpgid(self.process.pid) == self.process.pid, "fixture_process_group_not_owned")
        os.killpg(self.process.pid, sig)

    async def close(self, *, reason="caller_close"):
        if self._close_task is None:
            self._reason = reason
            self._close_task = asyncio.create_task(self._close())
        return await asyncio.shield(self._close_task)

    async def _close(self):
        errors, forced, joined, registry_state = [], False, self.process is None, "unverified"
        if self._watcher is not None:
            self._watcher.cancel()
            await asyncio.gather(self._watcher, return_exceptions=True)
        if self.process is not None:
            try:
                self._signal(signal.SIGTERM)
                await asyncio.wait_for(asyncio.shield(self._join), self.options.stop_timeout_seconds)
                joined = True
            except ProcessLookupError:
                try:
                    await asyncio.wait_for(asyncio.shield(self._join), self.options.kill_timeout_seconds)
                    joined = True
                except Exception:
                    errors.append("worker_join_unconfirmed")
            except TimeoutError:
                forced = True
                try:
                    self._signal(signal.SIGKILL)
                    await asyncio.wait_for(asyncio.shield(self._join), self.options.kill_timeout_seconds)
                    joined = True
                except Exception:
                    errors.append("worker_kill_or_join_unconfirmed")
            except Exception:
                errors.append("worker_stop_unconfirmed")
        if self.pool is not None:
            try:
                deadline = time.monotonic() + self.options.withdrawal_timeout_seconds
                while True:
                    registry_state, ready = await self._registry_state()
                    if registry_state != "active" or time.monotonic() >= deadline:
                        break
                    await asyncio.sleep(self.options.poll_seconds)
                if ready or registry_state == "active":
                    errors.append("worker_readiness_still_active")
            except Exception:
                errors.append("worker_readiness_unverified")
        elif self.process is None:
            registry_state = "not_spawned"
        if self._drain_tasks:
            _, pending = await asyncio.wait(self._drain_tasks, timeout=self.options.io_timeout_seconds)
            for task in pending:
                task.cancel()
            await asyncio.gather(*self._drain_tasks, return_exceptions=True)
        pipes = [drain.receipt() for drain in self._drains]
        pipes_complete = all(item["eof"] and item["error"] is None for item in pipes)
        for drain in self._drains:
            drain.tail.clear()
        secrets_removed = False
        if joined and pipes_complete:
            try:
                for path in self._secrets:
                    path.unlink(missing_ok=True)
                secrets_removed = True
            except Exception:
                errors.append("worker_secret_cleanup_failed")
        if self.pool is not None:
            try:
                await asyncio.wait_for(self.pool.close(), self.options.io_timeout_seconds)
            except Exception:
                errors.append("worker_verification_pool_close_failed")
                self.pool.terminate()
        inactive = registry_state in {"withdrawn", "expired", "absent", "not_spawned"}
        returncode = self.process.returncode if self.process else None
        safe = joined and inactive and pipes_complete and secrets_removed and not forced and not errors
        if self.process is not None:
            safe = safe and returncode == 0 and self._reason != "worker_exit"
        self.close_receipt = {"status": "closed" if safe else "failed", "reason": self._reason,
            "worker_joined": joined, "readiness_inactive": inactive, "registry_state": registry_state,
            "pipes_complete": pipes_complete, "pipes": pipes, "secrets_removed": secrets_removed,
            "forced_kill": forced, "returncode": returncode, "errors": errors,
            "worker_id": self.worker_id, "host_id": self.host_id,
            "pid": self.process.pid if self.process else None,
            "resource_manifest": str(self.resource_manifest) if self.resource_manifest else None,
            "native_cleanup": "unverified_after_forced_stop" if forced else "existing_worker_shutdown_contract",
            "task_completion_or_hearing": "not_established"}
        try:
            self._persist(self.close_receipt["status"])
        except Exception:
            self.close_receipt["status"] = "failed"
            errors.append("worker_receipt_persistence_failed")
        self._dsn = None
        return self.close_receipt


async def start(options: NativeTaskWorkerOptions, resources: dict, dsn: str) -> NativeTaskWorker:
    """Real opt-in: failures await bounded owned cleanup before raising.

    Caller-owned secret files, host resources, exports, bank and schema are never
    removed. A failed close receipt must block host fixture deletion. Cancellation
    is re-raised only after cleanup; its receipt is also attached to the exception.
    """
    worker = NativeTaskWorker(options, resources, dsn)
    try:
        return await worker._start()
    except BaseException as error:
        cleanup = asyncio.create_task(worker.close(reason="startup_failed"))
        while not cleanup.done():
            try:
                await asyncio.shield(cleanup)
            except asyncio.CancelledError:
                continue
        receipt = cleanup.result()
        if isinstance(error, asyncio.CancelledError):
            error.receipt, error.resource_manifest = receipt, worker.resource_manifest
            raise
        code = error.code if isinstance(error, NativeTaskWorkerError) else "fixture_worker_start_failed"
        raise NativeTaskWorkerError(code, receipt=receipt, resource_manifest=worker.resource_manifest) from None
