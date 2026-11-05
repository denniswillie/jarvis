"""Optional real-English mode for the existing managed-media qualification.

No CLI, credential discovery, new executor or renderer. Core injects the exact
approved model/runtime and allocates execution through managed_media_qualification.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import time
from uuid import uuid4

from app.files.workspace_access import WorkspaceFileAccess
from app.live_state.admission import CommandAdmission
from app.runtime.feature_config import FrontierConfiguration, TaskLimits
from app.sandbox.tools import SandboxToolBridge
from app.tasks.artifacts import result_artifacts
from app.tasks.hermes import HermesRuntimeConfiguration, HermesTaskProvider
from app.tasks.hermes_result import inspect_expert_output
from app.tasks.media_editing import MEDIA_TOOL, MediaInstruction, SavedRecordingReference
from app.tasks.model_gateway import OpenAIResponsesHTTP
from app.tasks.models import TaskStatus, json_object
from app.tasks.runner import TaskRunner
from app.workspace.media_agent import WorkspaceMediaAgent
from app.workspace.models import WorkspaceContext


MAIN_SECONDS, PARENT_SECONDS = 300, 390
MODEL, REASONING_EFFORT, HERMES_VERSION = "gpt-5.6-sol", "medium", "0.19.0"
INSTRUCTION = (
    "Edit this recording into a 10-second vertical video. Use seconds 2 to 7, "
    "followed by seconds 12 to 17. Keep the original sound and fit the whole image "
    "inside the portrait frame without cropping. Export at 720 by 1280, 24 frames "
    "per second. Show the caption 'Jarvis managed edit' from second 1 until second "
    "4 of the finished video. Save the edited video."
)
NOT_PROVEN = ["voice input, continued conversation, TTS or heard speech",
              "standalone worker CLI or scheduling lifecycle", "physical glasses capture",
              "human creative judgement or subjective audio quality",
              "authenticated phone delivery, playback or offline access",
              "correction MR02", "full concert", "Instagram publication"]


@dataclass(frozen=True)
class EnglishQualification:
    frontier: FrontierConfiguration
    runtime: HermesRuntimeConfiguration

    def validate(self, directory):
        if type(self.frontier) is not FrontierConfiguration or type(self.runtime) is not HermesRuntimeConfiguration:
            raise ValueError("Explicit approved model and native configuration are required")
        if (self.frontier.model, self.frontier.reasoning_effort) != (MODEL, REASONING_EFFORT):
            raise ValueError("This case is allocated only for the approved development model and reasoning")
        if (self.runtime.max_iterations > 12 or self.runtime.tool_timeout_seconds > 120
                or self.runtime.stop_seconds > 1 or not self.runtime.host_id):
            raise ValueError("Use the reviewed English-case native bounds and host")
        if (self.runtime.work_root != directory / "hermes" or self.runtime.work_root.exists()
                or self.runtime.work_root.is_symlink()
                or not self.runtime.python.is_file() or not os.access(self.runtime.python, os.X_OK)):
            raise ValueError("Use a fresh run-owned native root and explicit pinned interpreter")

    def public_metadata(self):
        return {"model": self.frontier.model, "reasoning_effort": self.frontier.reasoning_effort,
                "native_python": str(self.runtime.python), "work_root": str(self.runtime.work_root),
                "hermes_version_pin": HERMES_VERSION, "hermes_source_commit": None,
                "native_iterations": self.runtime.max_iterations, "tool_steps": 24,
                "input_tokens": 100000, "output_tokens": 8000, "response_tokens": 4000,
                "instruction": INSTRUCTION}


def task_limits():
    return TaskLimits(lease_seconds=300, active_seconds=MAIN_SECONDS, max_attempts=1,
        owner_concurrency=1, global_concurrency=1, tool_steps=24,
        task_input_tokens=100000, task_output_tokens=8000, request_output_tokens=4000)


class PrivateTrace:
    """Chronological bounded JSONL; payloads only, never transport headers/keys.

    Overflow is explicitly recorded and fails the case, never a partial pass.
    Each entry is flushed before the associated external effect is attempted.
    """
    MAX_BYTES, MAX_RECORD = 32 * 1024 * 1024, 8 * 1024 * 1024

    def __init__(self, path):
        self.path, self.count, self.size, self.truncated = path, 0, 0, False
        self.started = time.monotonic_ns()
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        self.stream = os.fdopen(fd, "wb")

    def append(self, kind, *, identity, payload):
        self.count += 1
        value = {"sequence": self.count, "at": datetime.now(UTC).isoformat(),
                 "elapsed_ns": time.monotonic_ns() - self.started, "kind": kind,
                 "identity": identity, "truncated": False, "payload": payload}
        raw = (json.dumps(value, sort_keys=True, ensure_ascii=False) + "\n").encode()
        overflow = len(raw) > self.MAX_RECORD or self.size + len(raw) > self.MAX_BYTES - 2048
        if overflow:
            self.truncated = True
            value.update(truncated=True, payload={"error": "trace_bound_exceeded",
                "omitted_bytes": len(raw), "omitted_sha256": hashlib.sha256(raw).hexdigest()})
            raw = (json.dumps(value, sort_keys=True) + "\n").encode()
        if self.size + len(raw) <= self.MAX_BYTES:
            self.stream.write(raw)
            self.stream.flush()
            self.size += len(raw)
        if overflow:
            raise ValueError("Detailed trace exceeded the reviewed bound")

    def error(self, kind, identity, error):
        # Exception strings may include URLs/credentials or arbitrary server text.
        self.append(kind, identity=identity, payload={"error_type": type(error).__name__})

    def close(self):
        self.stream.close()


class TracedUpstream:
    def __init__(self, upstream, trace, probe):
        self.upstream, self.trace, self.probe = upstream, trace, probe

    async def count(self, payload):
        identity = "count-" + uuid4().hex
        self.trace.append("model_count_request", identity=identity, payload=payload)
        self.probe.record["model_called"] = True
        self.probe.persist()
        try:
            count = await self.upstream.count(payload)
            self.trace.append("model_count_result", identity=identity, payload={"input_tokens": count})
            return count
        except BaseException as error:
            self.trace.error("model_count_error", identity, error)
            raise

    async def events(self, payload):
        identity = "response-" + uuid4().hex
        self.trace.append("model_request", identity=identity, payload=payload)
        self.probe.record["model_called"] = True
        self.probe.persist()
        try:
            async for event in self.upstream.events(payload):
                self.trace.append("model_event", identity=identity, payload=event.data)
                yield event  # Preserve actual data and wire bytes without rewriting.
            self.trace.append("model_stream_end", identity=identity, payload={})
        except BaseException as error:
            self.trace.error("model_error", identity, error)
            raise

    async def aclose(self):
        await self.upstream.aclose()


class TracedContext:
    def __init__(self, factory, trace, probe):
        self.factory, self.trace, self.probe, self.prepared = factory, trace, probe, None

    @property
    def capabilities(self):
        return self.factory.capabilities

    async def finish_outcome(self, context, outcome):
        return await self.factory.finish_outcome(context, outcome)

    async def __call__(self, context):
        self.prepared = await self.factory(context)
        bridge = self.prepared.tools
        if not isinstance(bridge, SandboxToolBridge) or bridge.media is None:
            raise ValueError("The ordinary managed media bridge is required")
        original = bridge.invoke

        async def invoke(name, arguments, *, provider_call_id):
            identity = str(provider_call_id)
            self.trace.append("tool_request", identity=identity,
                payload={"name": name, "arguments": arguments})
            try:
                result = await original(name, arguments, provider_call_id=provider_call_id)
                self.trace.append("tool_result", identity=identity, payload={"name": name, "result": result})
                operation = result.get("media_operation", {}) if isinstance(result, dict) else {}
                if name == "media_publish" and operation.get("state") == "published":
                    self.probe.record.setdefault("english_publications", []).append(operation)
                    self.probe.persist()
                return result
            except BaseException as error:
                self.trace.error("tool_error", identity, error)
                raise

        bridge.invoke = invoke
        return self.prepared


class ObservedHermesProvider(HermesTaskProvider):
    """Keep the single executor's completion observable after runner cancellation."""
    active = None

    async def execute(self, context):
        self.active = asyncio.current_task()
        try:
            return await super().execute(context)
        except BaseException as error:
            from evals.second_brain.managed_media_qualification import failure_location
            self.prepare.trace.append("executor_error", identity=str(context.claim.attempt_id),
                                      payload=failure_location(error))
            raise

    async def observe_stopped(self):
        if self.active is not None and not self.active.done():
            self.active.cancel()
            _, pending = await asyncio.wait((self.active,), timeout=5)
            if pending:
                raise ValueError("Native executor cleanup remains unresolved")


async def admit(service, files, recordings, owner, session, recording_id, saved):
    args = MediaInstruction(instruction=INSTRUCTION, recording_id=recording_id,
        saved=SavedRecordingReference(**{key: saved[key] for key in SavedRecordingReference.model_fields}))
    command = CommandAdmission(owner_id=owner, session_id=session,
        provider_session_key="synthetic-english-media-text-ingress",
        provider_call_id="english-media-" + uuid4().hex, tool_name=MEDIA_TOOL,
        capability="tasks.create", normalized_arguments=args.model_dump(mode="json", exclude_none=True))
    receipt = await WorkspaceMediaAgent(service, WorkspaceFileAccess(files), recordings).submit(
        WorkspaceContext(owner, session, command.operation_id), command.normalized_arguments,
        command_admission=command)
    if receipt.get("outcome") != "accepted" or receipt.get("effect_committed") is not False:
        raise ValueError("No ordinary English task admission receipt")
    async with service.repository.transaction(owner) as connection:
        row = await connection.fetchrow("select task_id from jarvis_task_requests where owner_id=$1 and request_id=$2",
                                        owner, command.operation_id)
        if row is None:
            raise ValueError("No exact durable command/task link")
        task = await service.repository.load(connection, owner, row["task_id"], lock="share")
    return task, receipt


def verify_recipe(recipe, reference, template):
    # Compare meaning only AFTER actual model execution/publication; identifiers
    # and the recipe itself are chosen by Hermes, never injected by this harness.
    if (recipe.previous_recipe is not None or recipe.template != template or len(recipe.sources) != 1
            or recipe.sources[0].file != reference or len(recipe.segments) != 2
            or [(s.in_ms, s.out_ms) for s in recipe.segments] != [(2000, 7000), (12000, 17000)]
            or any(s.source_id != recipe.sources[0].source_id for s in recipe.segments)
            or len(recipe.captions) != 1
            or (recipe.captions[0].start_frame, recipe.captions[0].end_frame, recipe.captions[0].text)
                != (24, 96, "Jarvis managed edit")
            or (recipe.output_profile, recipe.fit, recipe.audio) != ("portrait_720p24_v1", "contain", "preserve")):
        raise ValueError("The saved model-chosen recipe does not implement the English request")


async def execute_claim(configuration, factory, tasks, claim, probe):
    trace = PrivateTrace(probe.root / "english-trace.jsonl")
    prepared = TracedContext(factory, trace, probe)
    provider = ObservedHermesProvider(configuration.frontier, configuration.runtime, prepare=prepared,
        upstream_factory=lambda: TracedUpstream(
            OpenAIResponsesHTTP(configuration.frontier.api_key.get_secret_value()), trace, probe))
    try:
        trace.append("task_claim", identity=str(claim.attempt_id), payload={
            "task_id": str(claim.task.id), "request": INSTRUCTION,
            "model": configuration.frontier.model, "reasoning_effort": configuration.frontier.reasoning_effort})
        # Real production execution/heartbeat/settlement on this single admitted
        # claim. This deliberately does not qualify the standalone worker CLI.
        await TaskRunner(tasks, {"frontier": provider})._execute(claim)
        task = await tasks.get(claim.task.owner_id, claim.task.id)
        probe.record["english_task_terminal"] = {"status": task.status.value,
            "result_payload_id": str(task.result_payload_id) if task.result_payload_id else None}
        if task.status != TaskStatus.SUCCEEDED or task.result_payload_id is None:
            raise ValueError("The real English task did not succeed with a durable result")
        async with tasks.transaction(task.owner_id) as connection:
            raw = await connection.fetchval("select content from jarvis_context_payloads where owner_id=$1 and id=$2 and availability='available'",
                                           task.owner_id, task.result_payload_id)
            payload = json_object(raw) if raw is not None else {}
            inspected = inspect_expert_output(payload, task=task)
            artifacts = await result_artifacts(connection, tasks, task, inspected)
            unsettled = await connection.fetchval("""select count(*) from jarvis_task_reservations
                where owner_id=$1 and task_id=$2 and status in ('reserved','submitted','unknown')""",
                task.owner_id, task.id)
        if unsettled != 0:
            raise ValueError("Successful English task has unsettled model/tool usage")
        outputs = payload.get("verified_media", {}).get("outputs", [])
        publications = probe.record.get("english_publications", [])
        if (not inspected.expert_available or len(outputs) != 1 or not publications
                or any(item.get("output") != outputs[0] for item in publications)
                or not any(item["file_id"] == outputs[0]["file"]["file_id"] for item in artifacts)):
            raise ValueError("Expert result, accepted output and ordinary inbox artifact do not agree")
        trace.append("task_result", identity=str(task.result_payload_id), payload=payload)
        probe.record.update(english_result=payload, inbox_artifacts=artifacts)
        return prepared.prepared.tools, publications[-1]
    finally:
        try:
            await provider.observe_stopped()
        finally:
            trace.close()
            probe.record["english_trace"] = {"path": str(trace.path), "bytes": trace.size,
                "records": trace.count, "truncated": trace.truncated,
                "sha256": hashlib.sha256(trace.path.read_bytes()).hexdigest()}
            probe.persist()


def verify_native_cleanup(configuration, probe):
    # Native stop owns termination/deletion. Inspect only our fresh, exact root;
    # never signal processes or reconcile unrelated production worker state.
    root = configuration.runtime.work_root
    receipts = []
    for path in sorted(root.glob("*/run.json")) if root.exists() else []:
        if path.is_symlink() or path.stat().st_size > 16384:
            raise ValueError("Unbounded native receipt")
        value = json.loads(path.read_text())
        receipts.append(value)
        if (value.get("task_id") != probe.record.get("task_id")
                or value.get("attempt_id") != probe.record.get("attempt_id")
                or value.get("phase") != "exited"
                or value.get("private_home_cleanup", {}).get("state") != "complete"
                or (path.parent / "home").exists()):
            raise ValueError("Native exit/private-home cleanup remains unconfirmed")
        pid = value.get("pid")
        if pid is not None:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                pass
            else:
                raise ValueError("Recorded native PID is still present; no cleanup claim")
    probe.record["native_cleanup_receipts"] = receipts
    if probe.record.get("model_called") and len(receipts) != 1:
        raise ValueError("No unique native execution receipt for the actual model call")
    probe.persist()
