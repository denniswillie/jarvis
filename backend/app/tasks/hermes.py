"""Maintained Hermes execution under Jarvis task/action ownership.

This module runs in the ordinary worker environment. Only hermes_child.py runs
in the separately pinned native interpreter; no Hermes imports enter voice.
"""
from __future__ import annotations

import asyncio
from contextlib import suppress
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import stat
from typing import Awaitable, Callable
from uuid import UUID, uuid4

import uvicorn

from app.runtime.feature_config import FrontierConfiguration
from app.tasks.context import TaskContextBuilder
from app.tasks.context_repository import ContextRepository
from app.tasks.context_events import ContextWindow
from app.tasks.context_tools import TaskContextToolModule
from app.tasks.hermes_context import native_conversation_context
from app.tasks.hermes_result import HermesTaskResult, WORKER_SYSTEM_PROMPT
from app.tasks.model_gateway import OpenAIResponsesHTTP, TaskModelGateway
from app.tasks.models import ClaimLost, Reconciliation, TaskError, TaskStatus, WorkerOutcome, json_object
from app.tasks.tool_bridge import TaskToolBridge


@dataclass(frozen=True)
class HermesRuntimeConfiguration:
    python: Path
    work_root: Path
    host_id: str = ""
    max_iterations: int = 40
    tool_timeout_seconds: int = 120
    stop_seconds: float = 1
    context_tokens: int | None = None
    compression_tail_messages: int = 20

    def __post_init__(self):
        if not self.python.is_absolute() or not self.work_root.is_absolute():
            raise ValueError("Native interpreter and private state root must be absolute")
        if type(self.max_iterations) is not int or not 1 <= self.max_iterations <= 200:
            raise ValueError("Invalid native iteration bound")
        if not 1 <= self.tool_timeout_seconds <= 300 or not 0 < self.stop_seconds <= 10:
            raise ValueError("Invalid native timeout")
        if self.context_tokens is not None and (type(self.context_tokens) is not int or not 65536 <= self.context_tokens <= 2000000):
            raise ValueError("Invalid native context window")
        if type(self.compression_tail_messages) is not int or not 3 <= self.compression_tail_messages <= 100:
            raise ValueError("Invalid native compression tail")


@dataclass(frozen=True)
class PreparedHermesTask:
    manifest: object
    tools: TaskToolBridge


class HermesContextFactory:
    """Assemble S03 context and the same production tool modules for each owner."""

    def __init__(self, *, modules=(), read_tools=(), capability_tools=None,
                 images_factory=None, shared_context_factory=None, memory_evidence=None, receipt_revalidator=None,
                 context_window: ContextWindow | None = None, bridge_factory=None):
        self.modules, self.read_tools = tuple(modules), tuple(read_tools)
        self.context_window = context_window
        self.bridge_factory = bridge_factory
        self.memory_evidence = memory_evidence
        self.receipt_revalidator = receipt_revalidator
        self.capability_tools = dict(capability_tools or {})
        self.images_factory, self.shared_context_factory = images_factory, shared_context_factory

    async def __call__(self, context):
        repository = ContextRepository(context.tasks)
        builder = TaskContextBuilder(repository,window=self.context_window)
        images = await self.images_factory(context, repository) if self.images_factory else None
        module = TaskContextToolModule(builder, context.claim, images=images)
        names = ("read_task_context_archive", "read_task_context_source")
        if images is not None:
            names += ("read_task_context_image",)
        bridge = await TaskToolBridge.from_modules(
            context, (*self.modules, module), read_tools=(*self.read_tools, *names),
            capability_tools=self.capability_tools, memory_evidence=self.memory_evidence,
            receipt_revalidator=self.receipt_revalidator,
        )
        if self.bridge_factory is not None:
            bridge = await self.bridge_factory(bridge)
        shared = await self.shared_context_factory(context) if self.shared_context_factory else {}
        manifest = await builder.build(context.claim, shared_context=shared | {
            "authorized_operations": bridge.authorized_operations(),
            "worker": {"runtime":"hermes", "memory":"Jarvis tools only",
                       "visual_evidence":"Metadata is not pixels; retrieve original images when needed."},
        })
        return PreparedHermesTask(manifest, bridge)


def ensure_private_runtime_root(root):
    if root.is_symlink():
        raise ValueError("Native state root cannot be a symlink")
    root.mkdir(parents=True,exist_ok=True,mode=0o700)
    info = root.stat()
    if info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise ValueError("Native state root must be private to the worker user")
    return root


def _check_private_directory(directory):
    info = directory.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise PermissionError("Native state directory is not private to this worker")


def _write_run_marker(directory, marker):
    _check_private_directory(directory)
    temporary = directory / "run.next"
    # Do not follow a leftover native symlink when replacing a receipt.
    temporary.unlink(missing_ok=True)
    with open(temporary, "x", opener=lambda path, flags: os.open(path, flags, 0o600)) as stream:
        stream.write(json.dumps(marker))
    temporary.replace(directory / "run.json")


def _cleanup_private_home(directory, marker):
    """Only after the native process has ended; preserve its non-content receipt."""
    try:
        _check_private_directory(directory)
        home = directory / "home"
        if home.is_symlink():
            home.unlink()
        else:
            try:
                shutil.rmtree(home)
            except FileNotFoundError:
                # A missing descendant is not proof that the whole home is gone.
                if home.exists() or home.is_symlink():
                    raise
        marker["private_home_cleanup"] = {"state":"complete"}
        _write_run_marker(directory, marker)
    except OSError as error:
        marker["private_home_cleanup"] = {"state":"failed", "error_type":type(error).__name__}
        with suppress(OSError):
            _write_run_marker(directory, marker)
        raise TaskError("hermes_private_cleanup_pending",
                        "Native private-file cleanup is unconfirmed; reconcile the existing run before retrying.") from None


class _NativeRun:
    def __init__(self, configuration, context, gateway, *, on_progress=None, work_context=None):
        self.configuration, self.context, self.gateway = configuration, context, gateway
        self.run_id = gateway.run_id
        root = ensure_private_runtime_root(configuration.work_root)
        self.directory = root / UUID(self.run_id.removeprefix("hermes-run:")).hex
        self.directory.mkdir(mode=0o700)
        self.home = self.directory / "home"
        self.home.mkdir(mode=0o700)
        self.marker = {
            "format":"jarvis.hermes.run_receipt.v1", "run_id":self.run_id,
            "host_id":configuration.host_id or socket.gethostname(),
            "task_id":str(context.task.id), "attempt_id":str(context.claim.attempt_id),
            "owner_digest":hashlib.sha256(context.task.owner_id.encode()).hexdigest(),
            "phase":"starting", "pid":None, "exit_code":None,
            "private_home_cleanup":{"state":"pending"},
        }
        self.write_marker()
        self.process = self.server = self.server_task = self.sock = None
        self._stop_lock = asyncio.Lock()
        self.outcome = None
        self.finished = asyncio.Event()
        self.on_progress = on_progress
        self.work_context = work_context
        self._ending = False

    def write_marker(self):
        _write_run_marker(self.directory, self.marker)

    async def launch(self, prepared, frontier):
        self.sock = socket.socket()
        self.sock.bind(("127.0.0.1",0))
        self.sock.listen(16)
        self.sock.setblocking(False)
        self.server = uvicorn.Server(uvicorn.Config(
            self.gateway.application(tool_bridge=prepared.tools), log_level="error",
            access_log=False, lifespan="off", timeout_graceful_shutdown=1,
        ))
        self.server_task = asyncio.create_task(self.server.serve(sockets=[self.sock]))
        for _ in range(100):
            if self.server.started:
                break
            if self.server_task.done():
                await self.server_task
                raise TaskError("hermes_gateway_unavailable", "The native task transport did not start.")
            await asyncio.sleep(.02)
        if not self.server.started:
            raise TaskError("hermes_gateway_unavailable", "The native task transport did not start.")
        url = "http://127.0.0.1:" + str(self.sock.getsockname()[1]) + "/v1"
        aux = {"provider":"custom", "model":frontier.model, "base_url":url,
               "api_key":self.gateway.auxiliary_token, "api_mode":"codex_responses"}
        config_path = self.home / "config.yaml"
        config_path.write_text(json.dumps({
            "model": {"model":frontier.model,"provider":"openai","base_url":url,"api_mode":"codex_responses",
                      **({"context_length":self.configuration.context_tokens} if self.configuration.context_tokens else {})},
            "compression": {"enabled":True,"abort_on_summary_failure":True,
                            "protect_last_n":self.configuration.compression_tail_messages},
            "auxiliary": {"compression":aux},
            "sessions": {"write_json_snapshots":False},
            "agent": {"api_max_retries":1,"tool_use_enforcement":False},
            "memory": {"memory_enabled":False,"user_profile_enabled":False},
        }))
        config_path.chmod(0o600)
        self.process = await asyncio.create_subprocess_exec(
            str(self.configuration.python), str(Path(__file__).with_name("hermes_child.py")),
            cwd=self.home, env={
                "PATH":"/usr/bin:/bin:/usr/sbin:/sbin", "HOME":str(self.home), "HERMES_HOME":str(self.home),
                "TMPDIR":str(self.home), "PYTHONNOUSERSITE":"1", "PYTHONUNBUFFERED":"1",
                # Hermes' Codex transport otherwise kills small requests after
                # 12 seconds without SSE events, including valid public-API
                # reasoning pauses. Jarvis owns upstream I/O and task deadlines;
                # disabling this nested idle watchdog does not enable retries
                # or remove cancellation/unknown-usage fencing.
                "HERMES_CODEX_EVENT_STALE_TIMEOUT_SECONDS":"0",
            }, start_new_session=True, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE, limit=262144,
        )
        # Record identity before allowing the child to receive any task/token.
        self.marker.update(phase="running",pid=self.process.pid,gateway_url=url)
        self.write_marker()
        rendered = native_conversation_context(prepared.manifest.content)
        if self.work_context is not None:
            rendered["manifest"]["work_conversation"] = self.work_context
        config = {
            "run_id":self.run_id, "task_id":str(self.context.task.id),
            "attempt_id":str(self.context.claim.attempt_id), "base_url":url, "token":self.gateway.token,
            "model":frontier.model, "reasoning_effort":frontier.reasoning_effort,
            "tools":prepared.tools.catalog(), **rendered,
            "system_prompt":WORKER_SYSTEM_PROMPT,
            "max_iterations":self.configuration.max_iterations,
            "tool_timeout_seconds":self.configuration.tool_timeout_seconds,
            "max_output_tokens":self.context.task.budget.request_output_tokens,
            "progress_enabled": self.on_progress is not None,
        }
        encoded = (json.dumps(config)+"\n").encode()
        if len(encoded)>32000000:
            raise TaskError("hermes_context_too_large", "The native handoff exceeds its explicit transport bound.")
        self.process.stdin.write(encoded)
        await self.process.stdin.drain()

    async def steer(self, text):
        """Ordinary parent-owned message, not a claim that a correction was applied."""
        if type(text) is not str or not text.strip() or len(text.encode("utf-8")) > 64000:
            raise TaskError("hermes_input_invalid", "The follow-up input is invalid.")
        if self._ending or self.process is None or self.process.returncode is not None:
            raise TaskError("hermes_input_run_ended", "The existing worker is no longer running.")
        self.process.stdin.write((json.dumps({"operation": "steer", "text": text},
                                             ensure_ascii=False) + "\n").encode("utf-8"))
        await self.process.stdin.drain()

    async def result(self):
        async def discard_diagnostics():
            # Drain without retaining native transcript/tool arguments or keys.
            while await self.process.stderr.read(16384):
                pass
        draining = asyncio.create_task(discard_diagnostics())
        output = None
        try:
            for _ in range(self.configuration.max_iterations * 2 + 4):
                line = await self.process.stdout.readline()
                if not line:
                    break
                event = json.loads(line)
                if event.get("run_id") != self.run_id:
                    raise TaskError("hermes_protocol_mismatch", "The native response belongs to another run.")
                if event.get("type") == "result":
                    output = event
                    self._ending = True
                elif event.get("type") == "progress":
                    text = event.get("text")
                    if not isinstance(text, str) or not text.strip() or len(text.encode("utf-8")) > 16384:
                        raise TaskError("hermes_protocol_invalid", "The native progress event is invalid.")
                    if self.on_progress is not None:
                        await self.on_progress(text)
                elif event.get("type") == "error":
                    self.marker["native_error"] = event
                    self.write_marker()
                    raise TaskError("hermes_native_exception",
                                    "Native startup/execution failed: "+str(event.get("error_type","unknown")))
                elif event.get("type") != "ready":
                    raise TaskError("hermes_protocol_invalid", "The native process returned an invalid event.")
            else:
                raise TaskError("hermes_protocol_capacity", "The native event bound was exceeded.")
            await self.process.wait()
            if self.process.returncode != 0 or output is None:
                raise TaskError("hermes_process_failed", "The isolated native worker did not return a complete result.")
            self.marker["native_result"] = {key:output.get(key) for key in
                ("completed","failed","interrupted","partial","api_calls","turn_exit_reason","context_compressions")}
            self.marker["model_diagnostics"] = list(self.gateway.diagnostics)
            self.write_marker()
            return output
        finally:
            draining.cancel()
            await asyncio.gather(draining, return_exceptions=True)

    async def stop(self):
        async with self._stop_lock:
            self._ending = True
            self.gateway.close()
            if self.process and self.process.returncode is None:
                with suppress(BrokenPipeError, ConnectionResetError):
                    self.process.stdin.write(b'{"operation":"cancel"}\n')
                    await self.process.stdin.drain()
                try:
                    await asyncio.wait_for(self.process.wait(), self.configuration.stop_seconds)
                except TimeoutError:
                    with suppress(ProcessLookupError):
                        os.killpg(self.process.pid, signal.SIGTERM)
                    try:
                        await asyncio.wait_for(self.process.wait(), 1)
                    except TimeoutError:
                        with suppress(ProcessLookupError):
                            os.killpg(self.process.pid, signal.SIGKILL)
                        await self.process.wait()
            if self.server_task:
                self.server.should_exit = True
                try:
                    await asyncio.wait_for(self.server_task, 1.5)
                except TimeoutError:
                    self.server_task.cancel()
                    await asyncio.gather(self.server_task, return_exceptions=True)
            if self.sock:
                self.sock.close()
            self.marker.update(phase="exited",exit_code=self.process.returncode if self.process else None)
            _cleanup_private_home(self.directory, self.marker)
            self.finished.set()


class HermesTaskProvider:
    def __init__(self, frontier: FrontierConfiguration, configuration: HermesRuntimeConfiguration, *,
                 prepare: Callable[..., Awaitable[PreparedHermesTask]] | None = None, upstream_factory=None):
        self.frontier, self.configuration = frontier, configuration
        self.model = frontier.model
        self.prepare = prepare or HermesContextFactory()
        self.upstream_factory = upstream_factory or (
            lambda: OpenAIResponsesHTTP(frontier.api_key.get_secret_value())
        )
        self._runs = {}

    async def finish_outcome(self, context, outcome):
        cursor = outcome.payload.get("work_input_through")
        factory = getattr(self.prepare, "conversation_for", None)
        if type(cursor) is not int or factory is None:
            return False
        mailbox = await factory(context)
        if mailbox is None:
            return False
        return await mailbox.finish(context.claim, outcome, through_sequence=cursor)

    async def execute(self, context):
        if context.task.model != self.model:
            raise TaskError("task_model_configuration_changed", "This task's admitted model no longer matches the worker.")
        if self.configuration.stop_seconds+3>context.task.budget.shutdown_seconds:
            raise TaskError("hermes_shutdown_budget_invalid","Native shutdown must fit the task worker shutdown bound.")
        prepared = await self.prepare(context)
        if (prepared.manifest.owner_id != context.task.owner_id or prepared.manifest.task_id != context.task.id
                or prepared.manifest.task_revision != context.claim.task_revision
                or prepared.tools.context.claim != context.claim):
            raise TaskError("hermes_context_scope_mismatch", "The worker context belongs to another task attempt.")
        factory = getattr(self.prepare, "conversation_for", None)
        mailbox = await factory(context) if factory is not None else None
        work = None
        work_context = None
        if mailbox is not None:
            from app.tasks.hermes_work import HermesWorkSession
            work = HermesWorkSession(context, mailbox)
            work_context = await work.prepare()
        upstream = self.upstream_factory()
        run_id = "hermes-run:" + uuid4().hex
        gateway = TaskModelGateway(context,upstream,tools=prepared.tools.catalog(),
                                   run_id=run_id,managed_runtime=True,
                                   fence_poll_seconds=context.task.budget.poll_seconds,
                                   context_guard=work.guard if work is not None else None,
                                   response_text={"format":{"type":"json_schema","name":"jarvis_task_result",
                                                           "schema":HermesTaskResult.model_json_schema(),"strict":True}})
        run = pump = result_waiter = None
        try:
            run = _NativeRun(self.configuration,context,gateway,
                             on_progress=work.offer_progress if work is not None else None,
                             work_context=work_context)
            self._runs[context.claim.attempt_id] = run
            await context.tasks.start_runtime(context.claim,run_id,{
                "format":"jarvis.hermes.run.v1","run_id":run_id,"phase":"starting",
                "host_id":run.marker["host_id"],"manifest_id":str(prepared.manifest.id),
            },host_id=run.marker["host_id"])
            await run.launch(prepared,self.frontier)
            if work is None:
                result = await run.result()
            else:
                pump = asyncio.create_task(work.pump(run))
                result_waiter = asyncio.create_task(run.result())
                done, _ = await asyncio.wait((pump, result_waiter), return_when=asyncio.FIRST_COMPLETED)
                if pump in done:
                    await pump
                result = await result_waiter
                pump.cancel()
                with suppress(asyncio.CancelledError):
                    await pump
            if (result.get("completed") is not True
                    or any(result.get(flag) is not False for flag in ("failed","partial","interrupted"))):
                outcome = self.incomplete("hermes_native_incomplete")
            else:
                await gateway.revalidate(tuple(gateway.images))
                parsed = HermesTaskResult.model_validate_json(result["final_response"])
                outcome = await parsed.verified(context,prepared.tools)
            outcome = WorkerOutcome(outcome.status,outcome.payload | {"tool_receipts":gateway.tool_receipts},outcome.reason)
            if work is not None:
                outcome = WorkerOutcome(outcome.status,
                    outcome.payload | {"work_input_through": work.cursor}, outcome.reason)
            run.outcome = outcome
            await context.checkpoint({
                "format":"jarvis.hermes.run.v1","run_id":run_id,"phase":"terminal",
                "outcome":{"status":outcome.status.value,"payload":outcome.payload,"reason":outcome.reason},
            })
            return outcome
        finally:
            for pending in (pump, result_waiter):
                if pending is not None and not pending.done():
                    pending.cancel()
            for pending in (pump, result_waiter):
                if pending is not None:
                    with suppress(asyncio.CancelledError, Exception):
                        await pending
            try:
                if run:
                    stopping = asyncio.create_task(run.stop())
                    try:
                        await asyncio.shield(stopping)
                    except asyncio.CancelledError:
                        # Observe shutdown even if cancellation arrives during it.
                        await asyncio.shield(stopping)
                        raise
            finally:
                if run:
                    self._runs.pop(context.claim.attempt_id,None)
                close = getattr(upstream,"aclose",None)
                if close:
                    await close()

    @staticmethod
    def incomplete(reason):
        return WorkerOutcome(TaskStatus.PAUSED,{
            "format":"jarvis.hermes.runtime_outcome.v1", "authorship":{"kind":"runtime"},
            "answer":"I could not finish this task. Its recorded operation outcomes must be checked before retrying effects.",
            "uncertainties":[reason],
        },reason=reason)

    async def reconcile(self, context, *, cancel):
        handle = context.claim.provider_handle
        if not handle or not handle.startswith("hermes-run:"):
            return Reconciliation("unknown")
        try:
            identifier = UUID(handle.removeprefix("hermes-run:"))
        except ValueError:
            return Reconciliation("unknown")
        local = self._runs.get(context.claim.attempt_id)
        if local is not None:
            try:
                await local.stop()
            except (TaskError, OSError):
                return Reconciliation("unknown")
        directory = self.configuration.work_root / identifier.hex
        try:
            _check_private_directory(self.configuration.work_root)
            _check_private_directory(directory)
            marker = json.loads((directory/"run.json").read_text())
        except (OSError, ValueError):
            return Reconciliation("unknown")
        expected = {
            "format":"jarvis.hermes.run_receipt.v1","run_id":handle,
            "host_id":self.configuration.host_id or socket.gethostname(),
            "task_id":str(context.task.id),"attempt_id":str(context.claim.attempt_id),
            "owner_digest":hashlib.sha256(context.task.owner_id.encode()).hexdigest(),
        }
        if not isinstance(marker, dict) or any(marker.get(key)!=value for key,value in expected.items()):
            return Reconciliation("unknown")
        pid = marker.get("pid")
        if marker.get("phase") != "exited":
            if pid is not None:
                if type(pid) is not int or pid < 2:
                    return Reconciliation("unknown")
                try:
                    os.kill(pid,0)
                except ProcessLookupError:
                    pass
                except OSError:
                    return Reconciliation("unknown")
                else:
                    # A reused PID is not permission to signal an unrelated process.
                    return Reconciliation("unknown")
            elif marker.get("phase") != "starting":
                return Reconciliation("unknown")
        # Old exited receipts also need this check: earlier workers ignored
        # rmtree errors. Process exit is not a private-copy cleanup receipt.
        marker["phase"] = "exited"
        try:
            _cleanup_private_home(directory, marker)
        except TaskError:
            return Reconciliation("unknown")
        async with context.tasks.transaction(context.task.owner_id) as connection:
            _, attempt = await context.tasks.lock_claim(connection,context.claim,recovery=True)
            row = await connection.fetchrow(
                "select availability,content from jarvis_context_payloads where owner_id=$1 and id=$2",
                context.task.owner_id,attempt["checkpoint_payload_id"],
            )
        checkpoint = json_object(row["content"]) if row and row["availability"]=="available" else {}
        if (checkpoint.get("format")=="jarvis.hermes.run.v1" and checkpoint.get("run_id")==handle
                and checkpoint.get("phase")=="terminal"):
            value = checkpoint["outcome"]
            return Reconciliation("terminal",WorkerOutcome(TaskStatus(value["status"]),value["payload"],value.get("reason")))
        # The owned native process ended or never received its configuration.
        # Unknown model subrequest usage remains charged. This is NOT a claim
        # that OpenAI stopped remotely, nor permission to repeat external effects.
        return Reconciliation("terminal",self.incomplete("hermes_runtime_ended_without_result"))
