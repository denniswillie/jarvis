"""Test-only ASGI boundary around the normal Jarvis application.

The caller owns creation and audited cleanup of the manifested disposable schema,
area, native memory bank and storage objects. This module neither allocates those
resources nor asserts that a configured Hindsight service actually works. It
attests the inspected application binding, not memory quality or native readiness.
No production entry point imports or enables this module.
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import re
import time
from typing import Literal
from urllib.parse import parse_qs, parse_qsl, urlencode, urlsplit, urlunsplit
from uuid import UUID

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, TypeAdapter, model_serializer, model_validator
from starlette.responses import JSONResponse

from app.auth.static_bearer import authenticate_static_bearer
from app.errors import JarvisHTTPError
from app.migrations.runner import quote_schema
from app.runtime.feature_config import MemoryAuthority
from evals.responsive_voice.native_task_barrier import _open_root, _regular_private
from evals.responsive_voice.native_task_barrier_host import NativeTaskBarrierHost
from evals.responsive_voice.native_task_release_control import Frozen, load_config, write_private
from evals.responsive_voice.native_task_replay_control import (
    ALL_SDK_INPUTS, NO_AUTO_RELEASE, ResultReleasePolicy,
    NativeTaskReplayRelease, NativeTaskScope, _object,
)


class ReplayHostError(RuntimeError):
    pass


def _replay_path(path):
    if (not path.is_absolute() or '..' in path.parts
            or not re.fullmatch(r'[a-z0-9][a-z0-9_.-]{0,180}', path.name)
            or '..' in path.name):
        raise ReplayHostError('replay_task_path_invalid')
    return path


def _private_directory_identity(path):
    fd = _open_root(path)
    try:
        value = os.fstat(fd)
        return value.st_dev, value.st_ino
    finally:
        os.close(fd)


def read_task_replay_file(path, *, limit=16384):
    """Bounded, no-follow reads of the operator's immutable private inputs."""
    path = _replay_path(Path(path))
    directory = _open_root(path.parent)
    try:
        fd = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
        try:
            before = _regular_private(fd)
            if not 0 < before.st_size <= limit:
                raise ReplayHostError('replay_task_file_size')
            data = bytearray()
            while len(data) <= limit:
                part = os.read(fd, min(65536, limit + 1 - len(data)))
                if not part:
                    break
                data.extend(part)
            after = os.fstat(fd)
            if (len(data) != before.st_size
                    or (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns)):
                raise ReplayHostError('replay_task_file_changed')
            return bytes(data)
        finally:
            os.close(fd)
    finally:
        os.close(directory)


def _task_document(model, data, *, limit=16384):
    if not isinstance(data, bytes) or not 0 < len(data) <= limit:
        raise ReplayHostError('replay_task_document_size')
    _object(data)  # Reject duplicate keys and nonfinite JSON before typed parsing.
    return model.model_validate_json(data)


class NativeTaskReplayMetadata(Frozen):
    format: Literal['jarvis.native_task_replay.host.v1']
    barrier_id: UUID
    model: str = Field(min_length=1, max_length=100)
    configuration_path: Path
    status_path: Path
    stop_path: Path
    result_release_policy: ResultReleasePolicy = ALL_SDK_INPUTS
    scenario: Literal['explicit_cancel'] | None = None

    @model_serializer(mode='wrap')
    def legacy_wire(self, handler):
        value = handler(self)
        if self.result_release_policy == ALL_SDK_INPUTS and self.scenario is None:
            value.pop('result_release_policy', None)
            value.pop('scenario', None)
        return value

    @model_validator(mode='after')
    def private_paths(self):
        if self.result_release_policy == NO_AUTO_RELEASE and self.scenario != 'explicit_cancel':
            raise ValueError('held_pending_cancel_scope_required')
        paths = (self.configuration_path, self.status_path, self.stop_path)
        for path in paths:
            _replay_path(path)
        if len(set(paths)) != 3 or len({path.parent for path in paths}) != 1:
            raise ValueError('distinct_sibling_operator_paths_required')
        return self


class _TaskReplayConfiguration(Frozen):
    format: Literal['jarvis.native_task_replay.configuration.v1']
    run_id: UUID
    barrier_id: UUID
    worker_readiness: dict
    control_root: Path
    manifest_file: Path
    scope_file: Path
    input_sample_counts: tuple[int, ...] = Field(min_length=1, max_length=20)
    result_release_policy: ResultReleasePolicy = ALL_SDK_INPUTS
    scenario: Literal['explicit_cancel'] | None = None

    @model_serializer(mode='wrap')
    def legacy_wire(self, handler):
        value = handler(self)
        if self.result_release_policy == ALL_SDK_INPUTS and self.scenario is None:
            value.pop('result_release_policy', None)
            value.pop('scenario', None)
        return value


class _TaskReplayStop(Frozen):
    format: Literal['jarvis.native_task_replay.stop.v1']
    run_id: UUID
    barrier_id: UUID
    requested_at: AwareDatetime


class FrontendTaskReplayControl:
    """Single-run operator handshake; no business writes, worker or pool ownership.

    Server readiness precedes configuration readiness. Only the existing release
    controller may authorize a result release; observing an eligible task cannot.
    A closed receipt describes joins, not scenario success or provider acceptance.
    """

    def __init__(self, scope, metadata, *, clock=lambda: datetime.now(UTC)):
        self.scope, self.clock = scope, clock
        self.metadata = (metadata if isinstance(metadata, NativeTaskReplayMetadata) else
                         _task_document(NativeTaskReplayMetadata, json.dumps(metadata).encode()))
        self.root = self.metadata.configuration_path.parent
        self.root_identity = _private_directory_identity(self.root)
        self.bridge = self.release = self.task = None
        self.configured = False
        self.failure_code = self.close_reason = None
        self.observer_joined = False
        self._stop = asyncio.Event()
        self._configuration_bytes = self._last_payload = None
        self._status = None
        self._sequence = 0
        self._started_at = self.clock()
        self._deadline = time.monotonic() + max(0, (scope.expires_at - self._started_at).total_seconds())

    @property
    def accepting_inputs(self):
        return (self.configured and self.failure_code is None and not self._stop.is_set()
                and self.task is not None and not self.task.done())

    def status(self):
        return dict(self._status) if self._status is not None else None

    def _root_current(self):
        if _private_directory_identity(self.root) != self.root_identity:
            raise ReplayHostError('replay_task_operator_root_replaced')

    def _publish(self, state):
        self._root_current()
        value = {'format': 'jarvis.native_task_replay.status.v1',
            'run_id': self.scope.run_id, 'barrier_id': str(self.metadata.barrier_id),
            'status': state, 'configured': self.configured,
            'observer_joined': self.observer_joined,
            'release_loop_joined': self.task is None or self.task.done(),
            'failure_code': self.failure_code, 'close_reason': self.close_reason,
            'release': self.release.status() if self.release is not None else None,
            'semantic_acceptance': 'unmeasured'}
        if value == self._last_payload:
            return
        self._sequence += 1
        receipt = {**value, 'sequence': self._sequence, 'recorded_at': self.clock().isoformat()}
        write_private(self.root, self.metadata.status_path.name, json.dumps(receipt).encode())
        self._last_payload, self._status = value, receipt

    async def start(self, runtime):
        if self.task is not None:
            raise ReplayHostError('replay_task_already_started')
        self._root_current()
        for path in (self.metadata.configuration_path, self.metadata.status_path, self.metadata.stop_path):
            try:
                path.lstat()
            except FileNotFoundError:
                continue
            raise ReplayHostError('replay_task_operator_path_not_fresh')
        self.bridge = NativeTaskBarrierHost.attach(runtime, run_id=UUID(self.scope.run_id),
            barrier_id=self.metadata.barrier_id, owner_id=self.scope.owner_id,
            database_schema=self.scope.database_schema, model=self.metadata.model,
            host_expires_at=self.scope.expires_at)
        self.task = asyncio.create_task(self._run(), name='native-task-replay-control-' + self.scope.run_id)
        self.task.add_done_callback(self._finished)

    async def _configure(self, data):
        value = _task_document(_TaskReplayConfiguration, data)
        if value.run_id != UUID(self.scope.run_id) or value.barrier_id != self.metadata.barrier_id:
            raise ReplayHostError('replay_task_configuration_scope')
        if ((value.result_release_policy, value.scenario) !=
                (self.metadata.result_release_policy, self.metadata.scenario)):
            raise ReplayHostError('replay_task_release_policy_mismatch')
        manifest, scope_file, controls = value.manifest_file, value.scope_file, value.control_root
        _replay_path(manifest)
        _replay_path(scope_file)
        _replay_path(controls)
        if (manifest.parent.name != 'inputs' or scope_file != manifest.parent / 'task-scope.json'
                or controls != manifest.parent.parent / 'controls'):
            raise ReplayHostError('replay_task_staging_layout')
        _private_directory_identity(controls)
        manifest_bytes = read_task_replay_file(manifest, limit=512 * 1024)
        scope_bytes = read_task_replay_file(scope_file, limit=8192)
        task_scope = _task_document(NativeTaskScope, scope_bytes, limit=8192)
        if ((task_scope.run_id, task_scope.barrier_id, task_scope.owner_id, task_scope.database_schema,
             task_scope.model, task_scope.host_expires_at) !=
                (UUID(self.scope.run_id), self.metadata.barrier_id, self.scope.owner_id,
                 self.scope.database_schema, self.metadata.model, self.scope.expires_at)):
            raise ReplayHostError('replay_task_staged_scope')
        hold_receipt = value.worker_readiness['result_hold_control']
        hold = load_config(hold_receipt['config_path'])
        if hold.readiness() != hold_receipt:
            raise ReplayHostError('replay_task_hold_binding')
        release = NativeTaskReplayRelease(observer_host=self.bridge,
            worker_readiness=value.worker_readiness, control_root=controls,
            manifest_bytes=manifest_bytes, scope_bytes=scope_bytes,
            input_sample_counts=value.input_sample_counts, hold_config=hold,
            result_release_policy=value.result_release_policy, scenario=value.scenario)
        await self.bridge.configure(worker_readiness=value.worker_readiness, control_root=controls,
                                    history_root=self.root / 'task-barrier-history')
        self.release, self._configuration_bytes, self.configured = release, data, True
        self._publish('ready')

    def _requested_stop(self):
        try:
            data = read_task_replay_file(self.metadata.stop_path, limit=8192)
        except FileNotFoundError:
            return False
        value = _task_document(_TaskReplayStop, data, limit=8192)
        if (value.run_id != UUID(self.scope.run_id) or value.barrier_id != self.metadata.barrier_id
                or not self._started_at <= value.requested_at <= self.clock()):
            raise ReplayHostError('replay_task_stop_scope')
        return True

    async def _run(self):
        try:
            while not self._stop.is_set():
                self._root_current()
                if self._requested_stop():
                    self.close_reason = 'operator_stop'
                    break
                if self.clock() >= self.scope.expires_at or time.monotonic() >= self._deadline:
                    self.failure_code, self.close_reason = 'host_scope_expired', 'host_scope_expired'
                    break
                try:
                    data = read_task_replay_file(self.metadata.configuration_path)
                except FileNotFoundError:
                    if self.configured:
                        raise ReplayHostError('replay_task_configuration_removed') from None
                else:
                    if not self.configured:
                        await self._configure(data)
                    elif data != self._configuration_bytes:
                        raise ReplayHostError('replay_task_configuration_replaced')
                    if self._stop.is_set() or self._requested_stop():
                        self.close_reason = 'operator_stop'
                        break
                    history = getattr(self.bridge, 'history', None)
                    if history is not None and history.error:
                        self.failure_code = 'receipt_history_failed'
                        self._publish('failed')
                        break
                    result = self.release.step(now=self.clock())
                    if result['status'] == 'failed':
                        self.failure_code = 'release_control_failed'
                        self._publish('failed')
                        break
                    self._publish('ready')
                try:
                    await asyncio.wait_for(self._stop.wait(), timeout=.05)
                except TimeoutError:
                    pass
        except asyncio.CancelledError:
            self.failure_code, self.close_reason = 'control_loop_cancelled', 'cancelled'
        except Exception:
            # Neither operator payloads, paths nor exception text are diagnostics.
            self.failure_code = 'task_replay_control_invalid'
            try:
                self._publish('failed')
            except Exception:
                pass
        finally:
            self._stop.set()
            self.close_reason = self.close_reason or 'control_failed' if self.failure_code else self.close_reason or 'host_shutdown'
            try:
                receipt = await self.bridge.close()
                self.observer_joined = receipt.get('joined') is True
                if (receipt.get('receipt_history') or {}).get('error'):
                    self.failure_code = self.failure_code or 'receipt_history_failed'
                if not self.observer_joined:
                    self.failure_code = self.failure_code or 'observer_join_pending'
            except Exception:
                self.failure_code = self.failure_code or 'observer_close_failed'

    def _finished(self, task):
        # This callback runs only AFTER the release loop has actually joined.
        try:
            task.result()
        except BaseException:
            self.failure_code = self.failure_code or 'control_loop_failed'
        try:
            self._publish('closed' if self.observer_joined else 'failed')
        except Exception:
            self.failure_code = self.failure_code or 'status_write_failed'

    async def close(self):
        self._stop.set()
        self.close_reason = self.close_reason or 'host_shutdown'
        if self.task is not None:
            await asyncio.wait((self.task,), timeout=3)
            if not self.task.done():
                self.failure_code = self.failure_code or 'release_loop_join_pending'
                self._publish('failed')
            else:
                self._finished(self.task)
        return self.status()


class ReplayHostScope(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    run_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    owner_id: str
    database_schema: str
    area_id: UUID
    memory_bank_id: str = Field(pattern=r"^jarvis_[0-9a-f]{32}$")
    expires_at: AwareDatetime

    @model_validator(mode="after")
    def fixture_identity(self):
        prefix = f"direct-image-eval-second-brain-{self.run_id}-"
        if self.owner_id not in {prefix + "a", prefix + "b"}:
            raise ValueError("replay_owner_not_in_fixture")
        prefix = f"jarvis_sb_eval_{self.run_id}_"
        suffix = self.database_schema.removeprefix(prefix)
        if (not self.database_schema.startswith(prefix) or not suffix.isdecimal()
                or str(int(suffix)) != suffix):
            raise ValueError("replay_schema_not_in_fixture")
        quote_schema(self.database_schema)
        return self

    def public(self):
        return {
            "run_id": self.run_id, "owner_id": self.owner_id,
            "area_id": str(self.area_id), "memory_bank_id": self.memory_bank_id,
            "provenance": "frontend_replay", "isolation_enforced": True,
            "fresh_session_required": True, "external_notifications_enabled": False,
            "expires_at": self.expires_at.isoformat(),
        }


def scoped_database_url(dsn: str, schema: str) -> str:
    """asyncpg forwards this DSN parameter as a PostgreSQL startup setting.

    public is deliberately NOT a lookup fallback for application tables. Never
    print the returned URL: it retains the caller's existing DB credentials.
    """
    parts = urlsplit(dsn)
    if parts.scheme not in {"postgres", "postgresql"}:
        raise ValueError("replay_requires_postgresql_dsn")
    query = [(key, value) for key, value in parse_qsl(parts.query, keep_blank_values=True)
             if key != "search_path"]
    query.append(("search_path", f"{quote_schema(schema)},pg_catalog,extensions"))
    return urlunsplit(parts._replace(query=urlencode(query)))


def validate_configuration(scope, settings, flags):
    if settings.JARVIS_OWNER_ID != scope.owner_id:
        raise ReplayHostError("replay_owner_configuration_mismatch")
    token = settings.JARVIS_STATIC_BEARER_TOKEN
    if not isinstance(token, str) or len(token) < 24:
        raise ReplayHostError("replay_requires_private_bearer")
    query = parse_qs(urlsplit(settings.SUPABASE_DB_URL or "").query, keep_blank_values=True)
    expected = f"{quote_schema(scope.database_schema)},pg_catalog,extensions"
    if query.get("search_path") != [expected]:
        raise ReplayHostError("replay_database_path_not_isolated")
    boolean = TypeAdapter(bool)
    for name in ("push_enabled", "mail_enabled", "sandbox_enabled", "artifacts_enabled"):
        if boolean.validate_python(getattr(flags, name)):
            raise ReplayHostError("replay_external_capability_enabled")
    for name in ("enabled", "knowledge_enabled", "tasks_enabled", "attention_enabled"):
        if not boolean.validate_python(getattr(flags, name)):
            raise ReplayHostError("replay_required_capability_disabled")
    if (flags.selected_memory_authority() != MemoryAuthority.HINDSIGHT
            or str(flags.knowledge_area_id) != str(scope.area_id)):
        raise ReplayHostError("replay_memory_configuration_mismatch")


class FrontendReplayHost:
    """Authenticated, expiring admission gate; downstream media paths unchanged."""

    def __init__(self, app, scope, *, settings, flags, clock=lambda: datetime.now(UTC),
                 native_task_replay=None, native_memory_readiness=None):
        validate_configuration(scope, settings, flags)
        remaining = (scope.expires_at - clock()).total_seconds()
        if not 0 < remaining <= 3600:
            raise ReplayHostError("replay_scope_lifetime_invalid")
        self.app, self.scope, self.settings, self.flags, self.clock = app, scope, settings, flags, clock
        self.deadline = time.monotonic() + remaining
        self.ready, self.runtime, self.pool, self.started_at = False, None, None, None
        self.task_replay = (FrontendTaskReplayControl(scope, native_task_replay, clock=clock)
                            if native_task_replay is not None else None)
        self.memory_readiness = None
        if native_memory_readiness is not None:
            from evals.responsive_voice.native_memory_readiness_observer import MemoryReadinessObserver
            self.memory_readiness = MemoryReadinessObserver(scope, native_memory_readiness, clock=clock)
        original_lifespan = app.router.lifespan_context

        @asynccontextmanager
        async def lifespan(application):
            validate_configuration(scope, settings, flags)
            async with original_lifespan(application):
                await self.activate(application.state.runtime_services)
                try:
                    yield
                finally:
                    self.ready = False
                    try:
                        if self.memory_readiness is not None:
                            await self.memory_readiness.shutdown()
                    finally:
                        if self.task_replay is not None:
                            receipt = await self.task_replay.close()
                            if not receipt or not (receipt['observer_joined'] and receipt['release_loop_joined']):
                                raise ReplayHostError('replay_task_shutdown_unjoined')

        app.router.lifespan_context = lifespan

    def remaining(self):
        return min(self.deadline - time.monotonic(),
                   (self.scope.expires_at - self.clock()).total_seconds())

    async def activate(self, runtime):
        self.ready = False
        validate_configuration(self.scope, self.settings, self.flags)
        if self.remaining() <= 0:
            raise ReplayHostError("replay_scope_expired")
        knowledge = getattr(runtime, "knowledge_runtime", None)
        pool = getattr(runtime, "db_pool", None)
        if (pool is None or knowledge is None or runtime.settings is not self.settings
                or runtime.memory_authority != MemoryAuthority.HINDSIGHT
                or knowledge.owner_id != self.scope.owner_id
                or knowledge.configuration.area_id != self.scope.area_id
                or knowledge.sources.pool is not pool
                or knowledge.context_repository.tasks.pool is not pool
                or knowledge._closed or knowledge.task is None or knowledge.task.done()):
            raise ReplayHostError("replay_runtime_binding_missing")
        attention = getattr(runtime, "attention_service", None)
        if attention is None or attention.push is not None:
            raise ReplayHostError("replay_external_notification_binding")
        async with asyncio.timeout(5), pool.acquire() as connection:
            schema = await connection.fetchval("select current_schema()")
            paths = await connection.fetchval("select current_schemas(false)")
            if (schema != self.scope.database_schema or not paths or paths[0] != schema
                    or set(paths) - {schema, "pg_catalog", "extensions"}):
                raise ReplayHostError("replay_runtime_schema_mismatch")
            bank = await connection.fetchrow("""select b.bank_name,a.status
                from jarvis_memory_banks b join jarvis_knowledge_areas a
                  on a.owner_id=b.owner_id and a.id=b.area_id
                where b.owner_id=$1 and b.area_id=$2""", self.scope.owner_id, self.scope.area_id)
            if bank is None or bank["bank_name"] != self.scope.memory_bank_id or bank["status"] != "active":
                raise ReplayHostError("replay_canonical_bank_mismatch")
            self.started_at = await connection.fetchval("select clock_timestamp()")
        self.runtime, self.pool = runtime, pool
        try:
            if self.task_replay is not None:
                await self.task_replay.start(runtime)
            if self.memory_readiness is not None:
                self.memory_readiness.start(runtime, started_at=self.started_at)
        except BaseException:
            try:
                if self.memory_readiness is not None:
                    await self.memory_readiness.shutdown()
            finally:
                if self.task_replay is not None:
                    await self.task_replay.close()
            raise
        self.ready = True

    def check_live(self):
        if not self.ready or self.remaining() <= 0:
            raise ReplayHostError("replay_scope_unavailable")
        validate_configuration(self.scope, self.settings, self.flags)
        runtime, knowledge = self.runtime, self.runtime.knowledge_runtime
        if (runtime._closed or runtime.db_pool is not self.pool or knowledge._closed
                or knowledge.owner_id != self.scope.owner_id
                or knowledge.configuration.area_id != self.scope.area_id
                or knowledge.sources.pool is not self.pool
                or runtime.attention_service.push is not None):
            raise ReplayHostError("replay_runtime_binding_changed")

    @staticmethod
    def single_header(asgi, name):
        values = [value.decode("latin1") for key, value in asgi.get("headers", []) if key.lower() == name]
        if len(values) != 1:
            raise ReplayHostError("replay_header_missing_or_ambiguous")
        return values[0]

    async def reject(self, asgi, receive, send, code, status=403):
        if asgi["type"] == "websocket":
            await send({"type": "websocket.close", "code": 1008, "reason": code})
        else:
            await JSONResponse({"error": {"code": code}}, status_code=status)(asgi, receive, send)

    async def fresh_websocket(self, asgi):
        if asgi["path"] not in {"/realtime/audio/ws", "/realtime/visual/ws"}:
            raise ReplayHostError("replay_websocket_path_unsupported")
        try:
            values = parse_qs(asgi.get("query_string", b"").decode("ascii"))["session_id"]
            if len(values) != 1:
                raise ValueError()
            session_id = UUID(values[0])
        except (KeyError, ValueError, UnicodeError):
            raise ReplayHostError("replay_fresh_session_required") from None
        async with asyncio.timeout(min(3, self.remaining())), self.pool.acquire() as connection:
            fresh = await connection.fetchval("""select exists(select 1 from jarvis_sessions
                where owner_id=$1 and id=$2 and status='active' and created_at >= $3)""",
                self.scope.owner_id, session_id, self.started_at)
        if not fresh:
            raise ReplayHostError("replay_fresh_session_required")

    async def fresh_body(self, receive):
        body = bytearray()
        async with asyncio.timeout(min(5, self.remaining())):
            while True:
                message = await receive()
                if message["type"] != "http.request":
                    raise ReplayHostError("replay_session_body_incomplete")
                body.extend(message.get("body", b""))
                if len(body) > 16384:
                    raise ReplayHostError("replay_session_body_too_large")
                if not message.get("more_body", False):
                    break
        try:
            payload = json.loads(body)
        except (ValueError, UnicodeError):
            raise ReplayHostError("replay_session_body_invalid") from None
        if not isinstance(payload, dict) or payload.get("resume_active") is not False:
            raise ReplayHostError("replay_fresh_session_required")
        consumed = False

        async def replay_receive():
            nonlocal consumed
            if not consumed:
                consumed = True
                return {"type": "http.request", "body": bytes(body), "more_body": False}
            return await receive()

        return replay_receive

    async def __call__(self, asgi, receive, send):
        if asgi["type"] == "lifespan":
            return await self.app(asgi, receive, send)
        if asgi["type"] not in {"http", "websocket"}:
            raise ReplayHostError("replay_scope_type_unsupported")
        try:
            self.check_live()
            run_id = self.single_header(asgi, b"x-jarvis-evaluation-run")
            if run_id != self.scope.run_id:
                raise ReplayHostError("replay_run_mismatch")
            principal = authenticate_static_bearer(
                authorization=self.single_header(asgi, b"authorization"), settings=self.settings)
            if principal.owner_id != self.scope.owner_id:
                raise ReplayHostError("replay_principal_mismatch")
            if (self.task_replay is not None and not self.task_replay.accepting_inputs
                    and asgi['path'] != f'/eval/replay/{self.scope.run_id}/scope'):
                raise ReplayHostError('replay_task_control_not_ready')
            if asgi["type"] == "websocket":
                await self.fresh_websocket(asgi)
            elif asgi["path"] == "/sessions/start" and asgi["method"] == "POST":
                receive = await self.fresh_body(receive)
        except JarvisHTTPError as error:
            return await self.reject(asgi, receive, send, error.code, error.status_code)
        except (ReplayHostError, TimeoutError) as error:
            code = str(error) if isinstance(error, ReplayHostError) else "replay_admission_timeout"
            return await self.reject(asgi, receive, send, code)
        scoped = {**asgi, "state": {**asgi.get("state", {}), "jarvis_evaluation": self.scope.public()}}
        endpoint = f"/eval/replay/{self.scope.run_id}/scope"
        if asgi["type"] == "http" and asgi["path"] == endpoint:
            if asgi["method"] != "GET":
                return await self.reject(asgi, receive, send, "replay_scope_method", 405)
            return await JSONResponse(self.scope.public())(scoped, receive, send)
        response_started = False
        websocket_closed = False

        async def guarded_send(message):
            nonlocal response_started, websocket_closed
            self.check_live()
            response_started |= message["type"] == "http.response.start"
            websocket_closed |= message["type"] == "websocket.close"
            await send(message)

        try:
            async with asyncio.timeout(self.remaining()):
                await self.app(scoped, receive, guarded_send)
        except (ReplayHostError, TimeoutError):
            if response_started:
                # A partially sent HTTP response cannot truthfully be replaced by
                # a fresh success/error response; let the server abort transport.
                raise ReplayHostError("replay_scope_expired_during_response") from None
            if not websocket_closed:
                await self.reject(asgi, receive, send, "replay_scope_unavailable", 503)


def create_replay_application(scope: ReplayHostScope, *, native_task_replay=None, native_memory_readiness=None):
    """Use in a dedicated configured process, never mutate a running live app.

    Set the unique owner/token, scoped DSN and SecondBrain area BEFORE importing
    the normal server. The normal lifecycle constructs every downstream service.
    A resource launcher/cleanup audit is still required; this is not that runner.
    """
    from app.settings import get_settings
    from app.runtime.feature_config import get_second_brain_settings
    settings, flags = get_settings(), get_second_brain_settings()
    validate_configuration(scope, settings, flags)
    from app.main import create_app
    return FrontendReplayHost(create_app(), scope, settings=settings, flags=flags,
                              native_task_replay=native_task_replay,
                              native_memory_readiness=native_memory_readiness)
