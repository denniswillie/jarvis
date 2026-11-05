"""Opt-in, single-run replay-host observation. Default applications are untouched.

Attach before session tool construction, configure only after real worker
readiness and private simulator staging exist, then close before pool teardown.
Only the original registered handler/repository write business state. This
adapter observes them and independently reads committed state. Eligibility is
never worker-result release authorization; no worker control root is accessed.
"""

from __future__ import annotations

import asyncio
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
import json
import math
import os
from pathlib import Path
import re
from types import MethodType
from uuid import UUID, uuid4

from app.tasks.models import TaskVoiceSource, digest
from app.tasks.voice_tools import BackgroundTaskToolModule
from app.tools.invocation import current_tool_invocation
from app.tools.module import FunctionToolModule
from evals.responsive_voice.native_task_barrier import (
    AtomicReceiptFile, BarrierError, BarrierReceipt, CanonicalTaskObserver,
    FixtureScope, MAX_TTL_SECONDS, MAX_RECEIPT_BYTES, _open_root, decode_receipt,
    witness_from_observation,
)


MAX_HISTORY_RECEIPTS = 4096


class BarrierReceiptHistory:
    """Bounded evidence, never an alternate receipt authority or replay queue.

    Prepare private bytes before publishing; retain an immutable sequence file
    only after reading those exact bytes back from the atomic live exchange.
    A crash/failure between the two filesystem commits is NOT called complete.
    Readers require closed+complete index, contiguous numeric files, and each
    file's original v1 identity. Pending files are diagnostics, not receipts.
    """

    def __init__(self, root, *, binding, max_receipts=MAX_HISTORY_RECEIPTS):
        if type(max_receipts) is not int or not 1 <= max_receipts <= MAX_HISTORY_RECEIPTS:
            raise BarrierError("invalid_history_bound")
        self.root = Path(root)
        if self.root.name != "task-barrier-history":
            raise BarrierError("invalid_history_root")
        self.binding, self.maximum = dict(binding), max_receipts
        self.fd = None
        self.scope = self.admission = None
        self.last_published = self.last_archived = 0
        self.error = self.pending_file = self.unconfirmed_sequence = None
        self.closed = False
        parent = _open_root(self.root.parent)
        try:
            os.mkdir(self.root.name, mode=0o700, dir_fd=parent)
            os.fsync(parent)
            self.fd = _open_root(self.root)
            self._index()
        except BaseException:
            if self.fd is not None:
                os.close(self.fd)
                self.fd = None
            raise
        finally:
            os.close(parent)

    def status(self):
        return dict(format="jarvis.native_task_barrier_history.v1", root=str(self.root),
            run_id=str(self.binding["run_id"]), barrier_id=str(self.binding["barrier_id"]),
            filename_pattern="{sequence:020d}.json", max_receipts=self.maximum,
            max_receipt_bytes=MAX_RECEIPT_BYTES, last_published_sequence=self.last_published,
            last_archived_sequence=self.last_archived, receipt_count=self.last_archived,
            complete=self.error is None and self.last_published == self.last_archived,
            closed=self.closed, error=self.error, pending_file=self.pending_file,
            unconfirmed_sequence=self.unconfirmed_sequence)

    def _check_root(self):
        if self.fd is None:
            raise BarrierError("history_closed")
        current = _open_root(self.root)
        try:
            a, b = os.fstat(self.fd), os.fstat(current)
            if (a.st_dev, a.st_ino) != (b.st_dev, b.st_ino):
                raise BarrierError("history_root_replaced")
        finally:
            os.close(current)

    def _stage(self, data):
        self._check_root()
        if not 0 < len(data) <= MAX_RECEIPT_BYTES:
            raise BarrierError("history_receipt_size")
        name = ".pending-" + uuid4().hex
        fd = os.open(name, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600, dir_fd=self.fd)
        try:
            remaining = memoryview(data)
            while remaining:
                count = os.write(fd, remaining)
                if count <= 0:
                    raise OSError("short_write")
                remaining = remaining[count:]
            os.fsync(fd)
        except BaseException:
            os.unlink(name, dir_fd=self.fd)
            raise
        finally:
            os.close(fd)
        return name

    def _commit(self, temporary, name):
        self._check_root()
        # link is atomic and refuses an existing destination, unlike replace.
        # The temporary hardlink is removed before reporting archival success.
        os.link(temporary, name, src_dir_fd=self.fd, dst_dir_fd=self.fd, follow_symlinks=False)
        os.unlink(temporary, dir_fd=self.fd)
        os.fsync(self.fd)

    def _index(self):
        temporary = self._stage(json.dumps(self.status(), sort_keys=True).encode())
        try:
            self._check_root()
            os.replace(temporary, "index.json", src_dir_fd=self.fd, dst_dir_fd=self.fd)
            os.fsync(self.fd)
        finally:
            try:
                os.unlink(temporary, dir_fd=self.fd)
            except FileNotFoundError:
                pass

    def _failure(self, code):
        self.error = self.error or code
        try:
            self._index()
        except Exception:
            pass  # The host also fails explicitly; a stale/unclosed index is not completeness.

    def publish(self, exchange, receipt, *, now):
        phase = "history_preflight_failed"
        try:
            if self.closed or self.error:
                raise BarrierError("history_unavailable")
            self._check_root()
            data = receipt.model_dump_json().encode("utf-8")
            actual = decode_receipt(data)
            if (any(getattr(actual.scope, key) != value for key, value in self.binding.items())
                    or (self.scope is not None and (actual.scope != self.scope or actual.admission != self.admission))):
                raise BarrierError("history_scope_mismatch")
            if actual.sequence != self.last_published + 1 or actual.sequence > self.maximum:
                raise BarrierError("history_sequence_or_capacity")
            name = f"{actual.sequence:020d}.json"
            try:
                os.stat(name, dir_fd=self.fd, follow_symlinks=False)
            except FileNotFoundError:
                pass
            else:
                raise BarrierError("history_sequence_preexists")
            phase = "history_stage_failed"
            self.pending_file = self._stage(data)
            self.unconfirmed_sequence = actual.sequence
            publication_error = None
            try:
                exchange.publish(receipt, now=now)
            except Exception as error:
                publication_error = error
            phase = "live_publication_unconfirmed"
            observed = exchange.read()
            if observed != data:
                raise BarrierError("live_receipt_bytes_mismatch")
            # Observed visibility is weaker than a successful/durable publish return.
            self.last_published = actual.sequence
            self.scope, self.admission = actual.scope, actual.admission
            self.unconfirmed_sequence = None
            phase = "history_archive_failed"
            self._commit(self.pending_file, name)
            self.pending_file = None
            self.last_archived = actual.sequence
            phase = "history_index_failed"
            self._index()
            if publication_error is not None:
                phase = "live_publication_failed_after_visibility"
                raise publication_error
        except Exception:
            self._failure(phase)
            raise BarrierError("receipt_history_failed") from None

    def close(self):
        if self.fd is not None:
            self.closed = True
            try:
                self._index()
            except Exception:
                self._failure("history_close_failed")
            finally:
                os.close(self.fd)
                self.fd = None
        return self.status()


def _utc(value):
    if isinstance(value, str):
        value = datetime.fromisoformat(value)
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("aware_time_required")
    return value.astimezone(UTC)


@dataclass
class _Call:
    call_id: str
    event_id: UUID
    revision: int
    session_id: UUID
    notes: list = field(default_factory=list)
    done: bool = False


class NativeTaskBarrierHost:
    @classmethod
    def attach(cls, runtime, *, run_id: UUID, barrier_id: UUID, owner_id: str,
               database_schema: str, model: str, host_expires_at: datetime,
               poll_interval_seconds=0.25, query_timeout_seconds=1.0,
               ttl_seconds=2.0, close_timeout_seconds=2.0, max_calls=16,
               clock=lambda: datetime.now(UTC), observer_factory=CanonicalTaskObserver):
        """Install instance-local hooks; neither open a file nor start polling."""
        self = cls()
        self.runtime, self.run_id, self.barrier_id = runtime, run_id, barrier_id
        self.owner_id, self.database_schema, self.model = owner_id, database_schema, model
        self.host_expires_at, self.clock = _utc(host_expires_at), clock
        if (not isinstance(run_id, UUID) or not isinstance(barrier_id, UUID)
                or not isinstance(owner_id, str) or not 1 <= len(owner_id) <= 200
                or not isinstance(model, str) or not 1 <= len(model) <= 100
                or not re.fullmatch(r"jarvis_sb_eval_" + run_id.hex + r"_[0-9]+", database_schema)
                or not 0 < (self.host_expires_at - _utc(clock())).total_seconds() <= 3600):
            raise BarrierError("invalid_host_binding")
        for value, maximum in ((poll_interval_seconds, 5), (query_timeout_seconds, 10),
                               (ttl_seconds, MAX_TTL_SECONDS), (close_timeout_seconds, 10)):
            if isinstance(value, bool) or not math.isfinite(value) or not 0 < value <= maximum:
                raise BarrierError("invalid_host_bounds")
        if type(max_calls) is not int or not 1 <= max_calls <= 64:
            raise BarrierError("invalid_host_bounds")
        self.interval, self.query_timeout = poll_interval_seconds, query_timeout_seconds
        self.ttl, self.close_timeout, self.max_calls = ttl_seconds, close_timeout_seconds, max_calls
        self.observer_factory = observer_factory
        service = getattr(runtime, "background_task_service", None)
        self.repository = getattr(service, "repository", None)
        self.pool = getattr(runtime, "db_pool", None)
        modules = tuple(getattr(runtime, "tool_modules", ()))
        matches = [m for m in modules if isinstance(m, BackgroundTaskToolModule) and m.service is service]
        if (self.pool is None or self.repository is None or self.repository.pool is not self.pool
                or len(matches) != 1 or not callable(getattr(self.repository, "_save_request", None))
                or getattr(self.repository, "_native_task_barrier_owner", None) is not None):
            raise BarrierError("runtime_hook_binding_unavailable")
        self.original_module = matches[0]
        self.original_save = self.repository._save_request
        self.had_local_save = "_save_request" in vars(self.repository)
        self.original_local_save = vars(self.repository).get("_save_request")
        self.calls, self.sequence = [], 0
        self.scope = self.admission = self.claim = self.exchange = self.poll_task = None
        self.history = None
        self.worker = None
        self.reason = "unconfigured"
        self.terminal_reason = None
        self._closing = self._closed = False
        self._hooks_restored = None
        self._active = ContextVar("native_task_barrier_call", default=None)
        self._wake = asyncio.Event()
        self._stop = asyncio.Event()

        async def observed_save(repository, connection, task, request_id, operation, parameters):
            result = await self.original_save(connection, task, request_id, operation, parameters)
            # Reporting must never turn an accepted mutation into a handler error.
            if not self._closing:
                try:
                    self._note(task, request_id, operation, parameters)
                except Exception:
                    self._fail("observation_failed")
            return result

        self.installed_save = MethodType(observed_save, self.repository)
        self.module = FunctionToolModule(self.original_module.name, self._build_tools)
        self.repository._save_request = self.installed_save
        self.repository._native_task_barrier_owner = self
        runtime.tool_modules = tuple(self.module if m is self.original_module else m for m in modules)
        return self

    def _fail(self, reason):
        self.terminal_reason = reason
        self.reason = reason
        self._wake.set()

    def _build_tools(self, context):
        definitions = list(self.original_module.build_tools(context))
        if self._closing or context.owner_id != self.owner_id or context.session_id is None:
            return definitions
        result = []
        for definition in definitions:
            if definition.name != "start_background_task":
                result.append(definition)
                continue
            original = definition.handler

            async def observed_start(arguments, original=original):
                call = None
                if not self._closing:
                    try:
                        inv = current_tool_invocation()
                        if (inv.task is None and isinstance(inv.request_event_id, UUID)
                                and type(inv.request_revision) is int and inv.request_revision > 0
                                and isinstance(inv.provider_call_id, str) and 0 < len(inv.provider_call_id) <= 512):
                            if len(self.calls) >= self.max_calls:
                                self._fail("observation_capacity")
                            else:
                                call = _Call(inv.provider_call_id, inv.request_event_id,
                                             inv.request_revision, context.session_id)
                                self.calls.append(call)
                    except Exception:
                        self._fail("observation_failed")
                token = self._active.set(call)
                try:
                    return await original(arguments)
                finally:
                    self._active.reset(token)
                    if call is not None:
                        call.done = True
                    self._wake.set()

            result.append(replace(definition, handler=observed_start))
        return result

    def _note(self, task, request_id, operation, parameters):
        call = self._active.get()
        if call is None or operation != "create":
            return
        inv = current_tool_invocation()
        source = TaskVoiceSource(call.event_id, call.revision, call.session_id)
        if (inv.task is not None or inv.provider_call_id != call.call_id
                or inv.request_event_id != call.event_id or inv.request_revision != call.revision
                or task.owner_id != self.owner_id or task.session_id != call.session_id
                or parameters.get("voice_source") != source.parameters()):
            self._fail("observed_source_mismatch")
            return
        if sum(len(c.notes) for c in self.calls) >= self.max_calls:
            self._fail("observation_capacity")
            return
        call.notes.append(dict(owner_id=task.owner_id, request_id=str(request_id), task_id=str(task.id),
            operation=operation, parameters_digest=digest(parameters), accepted_revision=task.revision,
            call_id=call.call_id, request_event_id=str(call.event_id), request_revision=call.revision,
            voice_source=source.parameters(), committed=False))

    async def configure(self, *, worker_readiness, control_root: Path, name="task-barrier.json",
                        history_root: Path | None = None):
        """Caller supplies actual readiness and the FINAL simulator control root.

        Worker release/control-root fields, if present, are intentionally unused.
        Session identity is learned only inside the observed registered handler.
        """
        if self._closing or self.exchange is not None:
            raise BarrierError("host_already_configured_or_closed")
        try:
            r = worker_readiness
            if (r["status"] != "ready" or r["registry_ready"] is not True
                    or UUID(str(r["run_id"])) != self.run_id or r["owner_id"] != self.owner_id
                    or r["database_schema"] != self.database_schema or r["model"] != self.model
                    or _utc(r["host_expires_at"]) != self.host_expires_at
                    or not isinstance(r["worker_id"], str) or not 1 <= len(r["worker_id"]) <= 100
                    or not isinstance(r["host_id"], str) or not 1 <= len(r["host_id"]) <= 200
                    or _utc(self.clock()) >= self.host_expires_at):
                raise ValueError("readiness")
        except (KeyError, TypeError, ValueError):
            raise BarrierError("worker_readiness_mismatch") from None
        exchange = AtomicReceiptFile(control_root, name=name)
        try:
            try:
                exchange.read()
            except BarrierError as error:
                if error.code != "receipt_missing":
                    raise
            else:
                raise BarrierError("control_receipt_preexists")
            if history_root is not None:
                self.history = BarrierReceiptHistory(history_root, binding={
                    "run_id": self.run_id, "barrier_id": self.barrier_id, "owner_id": self.owner_id,
                    "database_schema": self.database_schema, "model": self.model,
                    "host_expires_at": self.host_expires_at})
        except BaseException:
            exchange.close()
            raise
        self.worker = {"worker_id": r["worker_id"], "runtime_host_id": r["host_id"]}
        self.exchange = exchange
        self.reason = "missing_finalized_create"
        self.poll_task = asyncio.create_task(self._run(), name="native-task-barrier-" + self.run_id.hex)
        self._wake.set()
        return self.status()

    def _bind_admission(self):
        observed = [(call, note) for call in self.calls if call.done for note in call.notes]
        keys = {(note["request_id"], note["task_id"], note["accepted_revision"], note["parameters_digest"],
                 note["request_event_id"], note["request_revision"], call.session_id) for call, note in observed}
        if len(keys) > 1:
            self._fail("ambiguous_admissions")
            return
        if self.admission is not None or not observed:
            return
        call, note = observed[0]
        scope = FixtureScope(run_id=self.run_id, barrier_id=self.barrier_id, owner_id=self.owner_id,
            session_id=call.session_id, database_schema=self.database_schema, model=self.model,
            host_expires_at=self.host_expires_at, **self.worker)
        self.admission = witness_from_observation(scope, note,
            [{"tool": "start_background_task", "call_id": call.call_id}])
        self.scope = scope

    def _withdraw(self, reason="canonical_unavailable"):
        if self.admission is None or self.exchange is None:
            return
        now = _utc(self.clock())
        expiry = min(now + timedelta(seconds=self.ttl), self.host_expires_at)
        if expiry <= now:
            return  # Every previously published receipt already expires by this boundary.
        receipt = BarrierReceipt(scope=self.scope, admission=self.admission, sequence=self.sequence + 1,
            observed_at=now, expires_at=expiry, status="blocked", reason=reason)
        self._publish_receipt(receipt, now=now)
        self.sequence = receipt.sequence

    def _publish_receipt(self, receipt, *, now):
        if self.history is None:
            self.exchange.publish(receipt, now=now)
            return
        try:
            self.history.publish(self.exchange, receipt, now=now)
        except BarrierError:
            self._fail("receipt_history_failed")
            self._stop.set()
            raise
        finally:
            self.sequence = max(self.sequence, self.history.last_published)

    async def _poll(self):
        try:
            if self.terminal_reason is None:
                self._bind_admission()
            if self.terminal_reason:
                self.reason = self.terminal_reason
                self._withdraw("ambiguous_tasks" if self.terminal_reason == "ambiguous_admissions"
                               else "canonical_unavailable")
                return
            if self.admission is None:
                self.reason = "handler_in_flight" if any(not c.done for c in self.calls) else "missing_finalized_create"
                return
            remaining = (self.host_expires_at - _utc(self.clock())).total_seconds()
            if remaining <= 0:
                self.reason = "host_expired"
                self._stop.set()
                return
            observer = self.observer_factory(self.pool, self.scope, self.admission,
                ttl_seconds=self.ttl, initial_sequence=self.sequence, expected_claim=self.claim, clock=self.clock)
            async with asyncio.timeout(min(self.query_timeout, remaining)):
                receipt = await observer.observe()
            if self._closing:
                return
            self._publish_receipt(receipt, now=_utc(self.clock()))
            self.sequence = receipt.sequence
            self.reason = receipt.reason
            if receipt.claim is not None:
                self.claim = receipt.claim
        except TimeoutError:
            self.reason = "canonical_poll_timeout"
            self._withdraw()
        except BarrierError as error:
            self.reason = error.code
            if self.history is not None and self.history.error:
                return  # Do not produce further live bytes without their archive.
            if self.admission is not None:
                self._withdraw()

    async def _run(self):
        try:
            while not self._stop.is_set():
                if _utc(self.clock()) >= self.host_expires_at:
                    self.reason = "host_expired"
                    return
                self._wake.clear()
                await self._poll()
                # An already-set wake event is not a scheduling/cancellation
                # boundary. Always yield, even on terminal prerequisite paths.
                await asyncio.sleep(0)
                try:
                    async with asyncio.timeout(self.interval):
                        await self._wake.wait()
                except TimeoutError:
                    pass
        except asyncio.CancelledError:
            raise
        except Exception:
            self.reason = "receipt_history_failed" if self.history is not None and self.history.error else "observer_failed"
            try:
                self._withdraw()
            except Exception:
                pass

    def status(self):
        return dict(configured=self.exchange is not None, reason=self.reason, sequence=self.sequence,
            observed_calls=len(self.calls), inflight_handlers=sum(not c.done for c in self.calls),
            session_id=str(self.scope.session_id) if self.scope else None,
            task_id=str(self.admission.task_id) if self.admission else None,
            polling=self.poll_task is not None and not self.poll_task.done(), closed=self._closed,
            receipt_history=self.history.status() if self.history is not None else None)

    async def close(self):
        """Cancel/join only our polling task, never business handlers or workers.

        A non-cooperative task returns close_pending with its handle retained;
        call close again on this same object, not a replacement supervisor.
        """
        self._closing = True
        self._stop.set()
        self._wake.set()
        if self.poll_task is not None and not self.poll_task.done():
            self.poll_task.cancel()
            await asyncio.wait({self.poll_task}, timeout=self.close_timeout)
        joined = self.poll_task is None or self.poll_task.done()
        if joined and self.poll_task is not None:
            await asyncio.gather(self.poll_task, return_exceptions=True)
        withdrawal = "not_applicable"
        if self.exchange is not None and not self._closed:
            try:
                self._withdraw()
                withdrawal = "blocked_or_expired" if self.admission is not None else "no_admission"
            except Exception:
                withdrawal = "failed_existing_receipt_expires"
        if self._hooks_restored is None:
            owned = vars(self.repository).get("_save_request") is self.installed_save
            if owned:
                if self.had_local_save:
                    self.repository._save_request = self.original_local_save
                else:
                    del self.repository._save_request
            modules = tuple(self.runtime.tool_modules)
            module_owned = any(m is self.module for m in modules)
            if module_owned:
                self.runtime.tool_modules = tuple(self.original_module if m is self.module else m for m in modules)
            if getattr(self.repository, "_native_task_barrier_owner", None) is self:
                del self.repository._native_task_barrier_owner
            self._hooks_restored = owned and module_owned
        if joined and self.exchange is not None and not self._closed:
            self.exchange.close()
            if self.history is not None:
                self.history.close()
        self._closed = joined
        return dict(status="closed" if joined else "close_pending", joined=joined,
            hooks_restored=self._hooks_restored,
            receipt_withdrawal=withdrawal, sequence=self.sequence,
            inflight_handlers=sum(not c.done for c in self.calls),
            pending_task_name=self.poll_task.get_name() if not joined else None,
            worker_release_authorized=False,
            receipt_history=self.history.status() if self.history is not None else None)
