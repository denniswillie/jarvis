"""Opt-in native S05 speech through production sessions, task tools and Hermes.

Only the phone and original utterances are synthetic. This harness does not
replace turn admission, canonical capture, tool execution or the agent loop.
Forwarded PCM proves adapter output, never physical playback. Semantic speech
acceptance is reported separately from mechanically checked task ownership.
This version checks foreground task control, not expert-answer retelling or
direct-TTS delivery, which requires its own S06 native acceptance population.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
from contextlib import AsyncExitStack, ExitStack, aclosing
from datetime import UTC, datetime
import json
import logging
from pathlib import Path
import subprocess
import time
from uuid import uuid4
from unittest.mock import patch

from app.assistant.context import AssistantContextBuilder
from app.assistant.context_providers import (
    CurrentTimeContextProvider, RecentMessagesContextProvider, SessionContextProvider,
)
from app.experience.repository import ExperienceRepository, ToolExecutionRepository
from app.experience.segmenter import EventSegmentationPolicy, ExperienceEventSegmenter
from app.experience.service import ExperienceEventService
from app.realtime.gemini.client import GeminiLiveConnection
from app.realtime.gemini.session import GeminiAudioSession
from app.realtime.models import RealtimeSessionContext
from app.realtime.session import RealtimeAudioSession
from app.runtime.feature_config import DEFAULT_FRONTIER_MODEL, FrontierConfiguration, TaskLimits
from app.sessions.repository import SessionRepository
from app.settings import JarvisSettings
from app.tasks.catalog import ConfiguredTaskProvider, TaskWorkerRegistry
from app.tasks.context_capture import SessionContextCapture
from app.tasks.context_events import ContextIngress
from app.tasks.context_repository import ContextRepository
from app.tasks.hermes import HermesTaskProvider, _NativeRun
from app.tasks.model_gateway import TaskModelGateway
from app.tasks.models import TaskStatus, digest
from app.tasks.repository import TaskRepository
from app.tasks.service import TaskService
from app.tasks.session_context import SessionTaskContextProvider, SessionTaskContextService
from app.tasks.voice_tools import BackgroundTaskToolModule
from app.tasks.worker import TaskWorkerRole, WorkerContextFactory, WorkerRoleConfiguration
from evals.gemini_live.session_probe import ProbePhone
from evals.gemini_live.transport_probe import read_pcm, write_wav
from evals.second_brain.source_fixture import source_fixture
from evals.second_brain.voice_phase_profile import CURRENT_CALL, Profile, distribution


UTTERANCES = {
    "ordinary": "What is seventeen plus six? Just tell me the answer.",
    "consult": (
        "I need a careful second opinion before changing my binary search. The midpoint "
        "is low plus high floor divided by two. Ask your expert to check operator precedence "
        "and give me the corrected expression. I am using Python and an ascending list."
    ),
    "background": (
        "Work in the background on a rigorous review of Python binary search for ascending lists. "
        "Include loop invariants, duplicate handling, boundary updates, empty inputs, and a test plan. "
        "Do not run tests or access files. Tell me once you have started and let me keep talking."
    ),
    "explicit_deep": (
        "Please use deeper reasoning for this design question. How should an assistant avoid duplicate "
        "side effects if its speech is interrupted while a task is still running? Compare cancellation "
        "before submission, after submission, and after completion, including uncertain outcomes. "
        "Delegate the analysis and let me keep talking. Do not make any changes."
    ),
    "multitask": (
        "Start two separate background jobs and let me keep talking. First, have your expert design "
        "a rigorous Python binary search tutorial for ascending lists, with loop invariants, duplicate "
        "handling, and a detailed test plan. Second, have another expert design a distributed task "
        "cancellation protocol with a full failure matrix and recovery invariants. "
        "These are independent jobs. Do not run code or write files."
    ),
    "status": "What background work is active, and what has actually finished?",
    "revise": (
        "Update the binary search job: it must return the first matching index when duplicates exist. "
        "Keep the ascending list requirement and the earlier constraints. Have the expert handle this "
        "correction instead of answering the technical question yourself."
    ),
    "cancel": "Cancel only the distributed task cancellation design job. Keep the binary search job.",
}
CASES = ("ordinary", "consult", "background", "explicit_deep", "unavailable", "disabled",
         "multitask", "interrupt", "reconnect", "missing_transcript", "context_overhead")
TERMINAL = {TaskStatus.SUCCEEDED, TaskStatus.FAILED, TaskStatus.CANCELLED,
            TaskStatus.PAUSED, TaskStatus.WAITING, TaskStatus.UNKNOWN}



def native_metadata(value):
    """Retain native control flags, not response bodies, task prose or secrets."""
    selected = {key: value.get(key) for key in ("completed", "failed", "partial", "interrupted")
                if type(value.get(key)) is bool or value.get(key) is None}
    for key in ("api_calls", "context_compressions"):
        if type(value.get(key)) is int and 0 <= value[key] <= 1000000:
            selected[key] = value[key]
    reason = value.get("turn_exit_reason")
    if isinstance(reason, str) and 0 < len(reason) <= 64 and all(
            character in "abcdefghijklmnopqrstuvwxyz0123456789_-" for character in reason):
        selected["turn_exit_reason"] = reason
    return selected


def diagnostic_code(value):
    """Only bounded machine classifications; never exception messages/bodies."""
    if isinstance(value, str) and 0 < len(value) <= 96 and all(
            char in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-" for char in value):
        return value
    return None


def terminal_metadata(event):
    kind = event.get("type")
    if kind not in {"response.completed", "response.incomplete", "response.failed"}:
        return None
    response = event.get("response")
    response = response if isinstance(response, dict) else {}
    selected = {"event_type": kind, "status": diagnostic_code(response.get("status"))}
    for source, field, target in (("incomplete_details", "reason", "incomplete_reason"),
                                  ("error", "code", "error_code")):
        value = response.get(source)
        if isinstance(value, dict):
            selected[target] = diagnostic_code(value.get(field))
    usage = response.get("usage")
    selected["provider_usage"] = {
        field: usage[field] for field in ("input_tokens", "output_tokens", "total_tokens")
        if isinstance(usage, dict) and type(usage.get(field)) is int
        and 0 <= usage[field] <= 1000000000
    }
    return selected


class ObservedResponsesUpstream:
    """Observe receipts without changing transport, payloads or exceptions."""
    def __init__(self, upstream, rows, clock, identity):
        self.upstream, self.rows, self.clock, self.identity = upstream, rows, clock, identity

    def start(self, phase):
        row = {**self.identity(), "phase": phase,
               "started_seconds": time.monotonic() - self.clock}
        self.rows.append(row)
        return row

    async def count(self, payload):
        row = self.start("input_count")
        try:
            value = await self.upstream.count(payload)
            if type(value) is int:
                row["input_tokens"] = value
            return value
        except BaseException as error:
            row.update(error_type=type(error).__name__, error_code=diagnostic_code(getattr(error, "code", None)))
            raise
        finally:
            row["ended_seconds"] = time.monotonic() - self.clock

    async def events(self, payload):
        row = self.start("provider_stream")
        row["event_count"] = 0
        try:
            async with aclosing(self.upstream.events(payload)) as events:
                async for event in events:
                    row["event_count"] += 1
                    row["last_event_type"] = diagnostic_code(event.data.get("type"))
                    terminal = terminal_metadata(event.data)
                    if terminal is not None:
                        row["terminal_receipt"] = terminal
                    yield event
            row["eof"] = True
        except GeneratorExit:
            row["closed_by_consumer"] = True
            raise
        except BaseException as error:
            row.update(error_type=type(error).__name__, error_code=diagnostic_code(getattr(error, "code", None)))
            raise
        finally:
            row["ended_seconds"] = time.monotonic() - self.clock


class ObservedTaskModelGateway(TaskModelGateway):
    def __init__(self, *args, diagnostics, clock, **kwargs):
        super().__init__(*args, **kwargs)
        self.diagnostics, self.diagnostic_clock = diagnostics, clock
        self.upstream = ObservedResponsesUpstream(self.upstream, diagnostics, clock, self.identity)

    def identity(self):
        return {"task_id": str(self.context.task.id),
                "attempt_id": str(self.context.claim.attempt_id), "request_index": self._requests}

    async def stream(self, body, *, auxiliary=False):
        row = {**self.identity(), "phase": "gateway_stream", "auxiliary": auxiliary,
               "started_seconds": time.monotonic() - self.diagnostic_clock}
        self.diagnostics.append(row)
        try:
            async with aclosing(super().stream(body, auxiliary=auxiliary)) as stream:
                async for wire in stream:
                    yield wire
            row["completed"] = True
        except BaseException as error:
            row.update(error_type=type(error).__name__, error_code=diagnostic_code(getattr(error, "code", None)))
            raise
        finally:
            row["request_index"] = self._requests
            row["uncertain"] = self._uncertain
            row["ended_seconds"] = time.monotonic() - self.diagnostic_clock


def completed_audio_ids(events):
    audible = {e.get("response_id") for e in events if e.get("type") == "audio.output.delta"}
    audible.discard(None)
    return {e.get("response_id") for e in events
            if e.get("type") == "response.done" and e.get("response_id") in audible
            and not e.get("interrupted") and e.get("status", "completed") == "completed"}


def speech_texts(events, response_ids=None):
    return [e.get("text", "") for e in events
            if e.get("type") == "transcript.done"
            and (response_ids is None or e.get("response_id") in response_ids)]


def safe_app_event(event):
    # Session/call IDs are synthetic. No configuration, credentials or native
    # resume handles are retained; PCM is saved separately with explicit origin.
    return {key: value for key, value in event.items() if key not in {"audio_base64", "audio"}}


class ObservedPhone(ProbePhone):
    def __init__(self, active_native):
        super().__init__()
        self.started = time.monotonic()
        self.observations = []
        self.active_native = active_native
        self.interrupt_when = None
        self.interruption = None

    async def send_text(self, payload):
        await self.send_json(json.loads(payload))

    async def send_json(self, event):
        self.observations.append({
            "seconds": time.monotonic() - self.started,
            "event": safe_app_event(event),
            "native_running_tasks": self.active_native(),
        })
        await super().send_json(event)
        if (event.get("type") == "audio.output.delta" and self.interrupt_when is not None
                and self.interrupt_when()):
            self.interrupt_when = None
            self.interruption = {
                "seconds": time.monotonic() - self.started,
                "response_id": event.get("response_id"),
                "response_already_completed": any(
                    earlier.get("type") == "response.done"
                    and earlier.get("response_id") == event.get("response_id") for earlier in self.events),
                "native_running_tasks": self.active_native(),
                "playback": "unknown",
            }
            # A real client cancel event, not direct manipulation of task state.
            await self.send("response.cancel")


class ObservedGeminiConnection(GeminiLiveConnection):
    """Observe the real serialized send, not just a successful tool handler."""

    def __init__(self, *, rows, clock, **kwargs):
        super().__init__(**kwargs)
        self.rows, self.clock = rows, clock

    async def _write(self, event):
        responses = event.get("toolResponse", {}).get("functionResponses", [])
        observations = []
        for response in responses:
            payload = response.get("response", {})
            task = payload.get("task") or {}
            row = {
                "started_seconds": time.monotonic() - self.clock,
                "call_id": response.get("id"), "tool": response.get("name"),
                "result_digest": digest(payload), "task_ref": task.get("task_ref"),
                "task_revision": task.get("revision"), "task_state": task.get("state"),
                "expert_output_available": payload.get("expert_output_available"),
                "answer_delivery": payload.get("answer_delivery"),
                "socket_write_completed": False,
                "meaning": "Socket write completion is not provider consumption or user playback.",
            }
            self.rows.append(row)
            observations.append(row)
        started = time.monotonic()
        try:
            await super()._write(event)
        except BaseException as error:
            for row in observations:
                row.update(error_type=type(error).__name__, error_code=getattr(error, "code", None))
            raise
        else:
            for row in observations:
                row["socket_write_completed"] = True
        finally:
            for row in observations:
                row["duration_ms"] = (time.monotonic() - started) * 1000


class MissingTranscriptCapture(SessionContextCapture):
    """Only the synthetic canonical-transcript receipt is faulted, not inference."""

    def input_final(self, identity, text):
        self.input_unavailable(identity, reason="synthetic_missing_provider_transcript")


class ObservedExecutor:
    """Timing wrapper around the real executor, never a handler replacement."""
    def __init__(self, executor, rows, clock, *, profile=None):
        self.executor, self.rows, self.clock = executor, rows, clock
        self.profile = profile

    async def execute(self, name, arguments, **kwargs):
        started = time.monotonic()
        row = {"tool": name, "arguments": arguments, "started_seconds": started - self.clock,
               "call_id": kwargs.get("provider_call_id")}
        self.rows.append(row)
        token = None
        if self.profile is not None:
            row["profile_call"] = "native:" + uuid4().hex
            token = CURRENT_CALL.set(row["profile_call"])
        try:
            result = await self.executor.execute(name, arguments, **kwargs)
            row["result"] = result
            return result
        except Exception as error:
            row["error_type"] = type(error).__name__
            raise
        finally:
            row["duration_ms"] = (time.monotonic() - started) * 1000
            if token is not None:
                CURRENT_CALL.reset(token)


async def wait_until(predicate, *, running=(), timeout=60):
    async with asyncio.timeout(timeout):
        while not predicate():
            for task in running:
                if task.done():
                    if not task.cancelled() and task.exception():
                        raise task.exception()
                    raise RuntimeError("A production session/worker stopped before the boundary")
            await asyncio.sleep(.025)


async def speak(phone, speech_dir, name):
    pcm = read_pcm(speech_dir / (name + ".wav"))
    segment = name + "-" + uuid4().hex
    for offset in range(0, len(pcm), 4800):
        chunk = pcm[offset:offset + 4800]
        await phone.send("audio.input.append", segment_id=segment,
                         audio_base64=base64.b64encode(chunk).decode())
        await asyncio.sleep(len(chunk) / 48000)
    ended = time.monotonic()
    await phone.send("audio.input.commit", segment_id=segment)
    await phone.send("response.create")
    return ended


async def answer(phone, speech_dir, name, *, running):
    start = len(phone.events)
    observed_start = len(phone.observations)
    ended = await speak(phone, speech_dir, name)
    await wait_until(lambda: bool(completed_audio_ids(phone.events[start:])), running=running, timeout=75)
    selected = phone.events[start:]
    ids = completed_audio_ids(selected)
    observations = phone.observations[observed_start:]
    first = next((e for e in observations if e["event"].get("type") == "audio.output.delta"
                  and e["event"].get("response_id") in ids), None)
    return {
        "utterance": name, "text": speech_texts(selected, ids),
        "input_end_seconds": ended - phone.started,
        "input_end_to_forwarded_audio_ms": (
            (first["seconds"] - (ended - phone.started)) * 1000 if first else None),
        "forwarded_audio_while_native_running": any(
            e["native_running_tasks"] for e in observations
            if e["event"].get("type") == "audio.output.delta" and e["event"].get("response_id") in ids),
        "response_ids": sorted(ids),
    }


def foreground_context_providers(task_context=None):
    return (
        CurrentTimeContextProvider(), SessionContextProvider(), RecentMessagesContextProvider(12),
        *((SessionTaskContextProvider(task_context),) if task_context is not None else ()),
    )


async def paired_context_overhead(sources, request, active_native, *, samples=20):
    """Pair the real context pipeline; never manufacture a native workload."""
    if type(samples) is not int or not 1 <= samples <= 30:
        raise ValueError("paired context samples must be 1..30")
    builders = {
        False: AssistantContextBuilder(providers=foreground_context_providers()),
        True: AssistantContextBuilder(providers=foreground_context_providers(SessionTaskContextService(sources))),
    }
    rows = []
    for ordinal in range(samples):
        before = sorted(active_native())
        if len(before) != 2:
            break
        row = {"ordinal": ordinal, "native_tasks_before": before, "measurements": []}
        payloads = {}
        order = (False, True) if ordinal % 2 == 0 else (True, False)
        for enabled in order:
            measurement = {"task_context_enabled": enabled, "native_tasks_before": sorted(active_native())}
            started = time.monotonic()
            try:
                context = await builders[enabled].build(**request)
                payload = context.as_prompt_payload()
                wire = json.dumps(payload, ensure_ascii=True, separators=(",", ":"), allow_nan=False)
                measurement["payload_bytes"] = len(wire.encode())
                measurement["context_available"] = (
                    not enabled or (payload.get("second_brain") or {}).get("status") == "available")
                payloads[enabled] = {key: value for key, value in payload.items() if key != "second_brain"}
            except Exception as error:
                measurement.update(error_type=type(error).__name__, context_available=False)
            finally:
                measurement["duration_ms"] = (time.monotonic() - started) * 1000
                measurement["native_tasks_after"] = sorted(active_native())
                row["measurements"].append(measurement)
        row["baseline_payload_equal"] = (
            len(payloads) == 2 and payloads[False] == payloads[True])
        row["qualified_fixed_load"] = row["baseline_payload_equal"] and all(
            item["context_available"] and item["native_tasks_before"] == before
            and item["native_tasks_after"] == before for item in row["measurements"])
        durations = {item["task_context_enabled"]: item["duration_ms"] for item in row["measurements"]}
        row["added_application_context_ms"] = durations[True] - durations[False]
        rows.append(row)
    qualified = [row["added_application_context_ms"] for row in rows if row["qualified_fixed_load"]]
    timing = distribution(qualified) if qualified else None
    return {
        "format": "jarvis.s05.paired_context_overhead.v1", "requested_pairs": samples,
        "qualified_pairs": len(qualified), "all_requested_pairs_qualified": len(qualified) == samples,
        "required_native_tasks": 2, "rows": rows, "added_application_context": timing,
        "target_p95_ms": 500,
        "target_met": len(qualified) == samples and timing is not None and timing["p95_ms"] <= 500,
        "meaning": (
            "Alternating enabled/disabled production context build and JSON serialization with frozen "
            "foreground inputs and the same two live native task IDs at each boundary. Negative paired "
            "differences are retained. Short or failed workloads are not substituted or counted. "
            "This measures application context overhead, not provider inference, input ingestion or playback."),
    }


async def run_case(args, settings, provider, case, iteration):
    label = f"{provider}-{case}-{iteration}"
    output = args.output / label
    output.mkdir(mode=0o700)
    limits = TaskLimits()
    result = {
        "format": "jarvis.s05.native_voice_task.v1", "case": label, "provider": provider,
        "scenario": case, "mechanical_pass": False, "semantic_review": "pending",
        "harness_revision": "s05-control-20", "task_limits": limits.model_dump(mode="json"),
        "acceptance_scope": "foreground_task_control_not_expert_speech",
        "direct_tts_acceptance": "not_implemented",
        "device_playback": "not_tested", "synthetic_speech": True,
        "native_worker": case not in {"unavailable", "disabled"},
        "foreground_model": settings.OPENAI_REALTIME_MODEL if provider == "openai" else settings.GEMINI_LIVE_MODEL,
        "frontier_model": args.frontier_model, "answers": [], "tools": [], "worker_events": [], "native_outcomes": [],
        "context_sections": [], "gateway_events": [], "native_tool_writes": [],
        "budget_accounting_meaning": (
            "charged_task_budget is whole-task accounting, including full unsettled reservations. "
            "Only settled actual amounts or provider terminal receipts establish actual usage."),
    }
    session = session_job = worker_job = capture = phone = worker = native = experience = native_observer = None
    stop = asyncio.Event()
    journals = []
    profile = Profile() if args.profile_tools else None
    profile_patches = ExitStack()
    try:
        if profile is not None:
            profile.install(profile_patches)
        async with source_fixture(settings.SUPABASE_DB_URL, output, test_name=label,
            pool_init=profile.initialize_connection if profile is not None else None) as fixture:
            # Exercise production task defaults rather than an artificial token
            # cap or aggressive poll cadence. Fixture waits remain bounded, and
            # cleanup drains the real worker even when a scenario times out.
            tasks = TaskRepository(fixture.pool, limits)
            providers = {"frontier": ConfiguredTaskProvider(args.frontier_model)}
            catalog = TaskWorkerRegistry(fixture.pool)
            service = TaskService(tasks, providers, registry=catalog)
            sources = ContextRepository(tasks)
            repository = SessionRepository(fixture.pool)
            experience = ExperienceEventService(repository=ExperienceRepository(fixture.pool),
                segmenter=ExperienceEventSegmenter(EventSegmentationPolicy(300, 900)))
            async with fixture.pool.acquire() as connection:
                session_id = await connection.fetchval(
                    "insert into jarvis_sessions(owner_id) values($1) returning id", fixture.owner)
            result["owner_id"], result["session_id"] = fixture.owner, str(session_id)
            if case == "context_overhead":
                result["fixed_context_history_sources"] = 32
                for index in range(32):
                    await sources.append(fixture.owner, ContextIngress(
                        session_id=session_id, source_namespace="paired-context-history",
                        source_event_id=str(index), receipt_id=uuid4().hex,
                        kind="user_transcript", trust="user_input",
                        payload={"text": (
                            f"Synthetic context entry {index}: these exercises are theoretical. "
                            "Do not run code, access files, or perform external actions.")},
                    ))

            def active_native():
                if native is None:
                    return []
                return sorted(str(run.context.task.id) for run in tuple(native._runs.values())
                              if run.process is not None and run.process.returncode is None)

            phone = ObservedPhone(active_native)
            try:
                if result["native_worker"]:
                    configuration = WorkerRoleConfiguration(
                        python=args.hermes_python, work_root=output / "native-state",
                        host_id="s05-native-" + uuid4().hex, heartbeat_seconds=1,
                    )
                    class ObservedRun(_NativeRun):
                        async def result(self):
                            value = await super().result()
                            result["native_outcomes"].append({
                                "task_id": str(self.context.task.id),
                                "attempt_id": str(self.context.claim.attempt_id), **native_metadata(value),
                            })
                            return value

                    class ObservedGateway(ObservedTaskModelGateway):
                        def __init__(self, *gateway_args, **gateway_kwargs):
                            super().__init__(*gateway_args, diagnostics=result["gateway_events"],
                                             clock=phone.started, **gateway_kwargs)

                    native_observer = patch.multiple("app.tasks.hermes",
                        _NativeRun=ObservedRun, TaskModelGateway=ObservedGateway)
                    native_observer.start()

                    class ObservedHermes(HermesTaskProvider):
                        async def execute(self, context):
                            event = {"task_id": str(context.task.id),
                                     "attempt_id": str(context.claim.attempt_id),
                                     "started_seconds": time.monotonic() - phone.started}
                            result["worker_events"].append(event)
                            try:
                                outcome = await super().execute(context)
                                event.update(status=outcome.status.value, reason=outcome.reason)
                                return outcome
                            except BaseException as error:
                                event.update(error_type=type(error).__name__, code=getattr(error, "code", None))
                                raise
                            finally:
                                event["ended_seconds"] = time.monotonic() - phone.started
                                try:
                                    event["charged_task_budget"] = await context.budgets.usage(
                                        context.task.owner_id, context.task.id)
                                    async with fixture.pool.acquire() as connection:
                                        event["attempt_reservations"] = [dict(row) for row in await connection.fetch(
                                            """select status,input_tokens,output_tokens,tool_steps,
                                                actual_input_tokens,actual_output_tokens,actual_tool_steps
                                                from jarvis_task_reservations
                                                where owner_id=$1 and task_id=$2 and attempt_id=$3""",
                                            context.task.owner_id, context.task.id, context.claim.attempt_id)]
                                except Exception as error:
                                    event["accounting_error_type"] = type(error).__name__

                    native = ObservedHermes(FrontierConfiguration(
                        model=args.frontier_model, reasoning_effort=args.reasoning_effort,
                        api_key=settings.OPENAI_API_KEY,
                    ), configuration.native(), prepare=WorkerContextFactory(fixture.pool))
                    worker = TaskWorkerRole(tasks, {"frontier": native},
                        owner_id=fixture.owner, configuration=configuration)
                    worker_job = asyncio.create_task(worker.run(stop))
                    async with asyncio.timeout(20):
                        while not await catalog.ready(fixture.owner, "frontier", args.frontier_model):
                            if worker_job.done():
                                await worker_job
                                raise RuntimeError("Native worker ended before readiness")
                            await asyncio.sleep(.1)
                class ObservedTaskContext(SessionTaskContextService):
                    async def describe(self, request):
                        row = {"started_seconds": time.monotonic() - phone.started}
                        result["context_sections"].append(row)
                        began = time.monotonic()
                        try:
                            return await super().describe(request)
                        except BaseException as error:
                            row["error_type"] = type(error).__name__
                            raise
                        finally:
                            row["duration_ms"] = (time.monotonic() - began) * 1000

                async def connect():
                    nonlocal capture, session, session_job
                    capture_type = MissingTranscriptCapture if case == "missing_transcript" else SessionContextCapture
                    capture = capture_type(sources, owner_id=fixture.owner, session_id=session_id,
                        provider=provider, audio_session_id=str(uuid4()))
                    modules = () if case == "disabled" else (BackgroundTaskToolModule(service),)
                    context_providers = foreground_context_providers(
                        ObservedTaskContext(sources) if case != "disabled" else None)
                    session_type = RealtimeAudioSession if provider == "openai" else GeminiAudioSession
                    session = session_type(websocket=phone, context=RealtimeSessionContext(
                        audio_session_id=capture.audio_session_id, owner_id=fixture.owner,
                        jarvis_session_id=session_id, client_kind="phone_audio",
                        session=await repository.get_session(fixture.owner, session_id),
                    ), settings=settings, repository=repository, tool_modules=modules,
                        context_providers=context_providers, context_capture=capture,
                        tool_execution_repository=ToolExecutionRepository(fixture.pool),
                        experience_event_service=experience,
                        **({"connection_factory": lambda **kwargs: ObservedGeminiConnection(
                            rows=result["native_tool_writes"], clock=phone.started, **kwargs)}
                           if provider == "gemini" else {}))
                    if provider == "openai":
                        session.tool_executor = ObservedExecutor(
                            session.tool_executor, result["tools"], phone.started, profile=profile)
                    else:
                        session.operations.executor = ObservedExecutor(
                            session.operations.executor, result["tools"], phone.started, profile=profile)
                    if provider == "gemini":
                        journals.append(session.journal)
                    boundary = len(phone.events)
                    session_job = asyncio.create_task(session.run())
                    jobs = tuple(job for job in (session_job, worker_job) if job is not None)
                    await wait_until(lambda: any(e["type"] == "session.created" for e in phone.events[boundary:]),
                                     running=jobs, timeout=25)
                    return jobs

                jobs = await connect()
                utterance = ("multitask" if case == "context_overhead" else "background"
                    if case in {"unavailable", "disabled", "interrupt", "reconnect", "missing_transcript"} else case)
                if case == "interrupt":
                    phone.interrupt_when = lambda: any(
                        row["tool"] == "start_background_task"
                        and row.get("result", {}).get("accepted") is True for row in result["tools"])
                    await speak(phone, args.speech_dir, utterance)
                    await wait_until(lambda: phone.interruption is not None, running=jobs, timeout=75)
                    result["interruption"] = phone.interruption
                else:
                    result["answers"].append(await answer(phone, args.speech_dir, utterance, running=jobs))
                if case == "reconnect":
                    before = await tasks.list(fixture.owner, limit=20)
                    result["reconnect"] = {
                        "task_ids_before": [str(task.id) for task in before],
                        "statuses_before": [task.status.value for task in before],
                        "native_running_at_disconnect": active_native(),
                        "old_audio_session_id": capture.audio_session_id,
                    }
                    await session.close("s05_synthetic_phone_disconnect")
                    await asyncio.wait_for(session_job, 20)
                    await capture.close()
                    result["reconnect"]["old_capture_issue"] = capture.issue
                    previous = phone
                    phone = ObservedPhone(active_native)
                    # New transport and queues; retain a single observation clock
                    # and original output evidence across the two connections.
                    phone.started = previous.started
                    phone.events, phone.audio = previous.events, previous.audio
                    phone.observations = previous.observations
                    jobs = await connect()
                    result["reconnect"]["new_audio_session_id"] = capture.audio_session_id
                    result["reconnect"]["task_ids_after"] = [
                        str(task.id) for task in await tasks.list(fixture.owner, limit=20)]
                if case in {"consult", "background", "explicit_deep", "multitask", "interrupt", "reconnect", "context_overhead"}:
                    result["answers"].append(await answer(phone, args.speech_dir, "ordinary", running=jobs))
                    if case == "context_overhead":
                        await capture.flush()
                        request = {
                            "owner_id": fixture.owner,
                            "session": await repository.get_session(fixture.owner, session_id),
                            "recent_messages": await repository.get_recent_messages(
                                fixture.owner, session_id, limit=12),
                            "current_utc_time": datetime.now(UTC),
                        }
                        result["paired_context"] = await paired_context_overhead(
                            sources, request, active_native, samples=args.paired_context_samples)
                    elif case == "multitask":
                        for name in ("status", "revise", "cancel", "status"):
                            result["answers"].append(await answer(phone, args.speech_dir, name, running=jobs))
                    elif case in {"interrupt", "reconnect"}:
                        for name in ("status", "revise", "status"):
                            result["answers"].append(await answer(phone, args.speech_dir, name, running=jobs))
                    else:
                        # Observation, not admission: wait for actual worker status
                        # before asking for current status. No guessed completion timer.
                        async with asyncio.timeout(150):
                            while True:
                                current = await tasks.list(fixture.owner, limit=20)
                                if not current or all(t.status.value in TERMINAL for t in current):
                                    break
                                await asyncio.sleep(.5)
                        result["answers"].append(await answer(phone, args.speech_dir, "status", running=jobs))
                await capture.flush()
                current = await tasks.list(fixture.owner, limit=20)
                result["tasks"] = []
                for task in current:
                    _, details = await tasks.voice_detail(fixture.owner, task.id)
                    result["tasks"].append({
                        "id": str(task.id), "task_ref": tasks.reference(task.id),
                        "revision": task.revision, "title": task.title,
                        "requested_outcome": task.requested_outcome, "mode": task.mode,
                        "status": task.status.value, "details": details,
                    })
                expected = 0 if case in {"ordinary", "unavailable", "disabled", "missing_transcript"} else 2 if case in {"multitask", "context_overhead"} else 1
                async with fixture.pool.acquire() as connection:
                    result["ledger"] = {
                        table: [dict(row) for row in await connection.fetch(
                            "select * from " + table + " where owner_id=$1", fixture.owner)]
                        for table in ("jarvis_task_events", "jarvis_task_attempts", "jarvis_task_workers")
                    }
                result["expected_task_count"] = expected
                result["capture_issue"] = capture.issue
                result["mechanical_pass"] = (
                    len(current) == expected and capture.issue is None
                    and not any(e["type"] == "error" for e in phone.events)
                    and all(item["response_ids"] for item in result["answers"])
                )
                if case in {"consult", "background", "explicit_deep"}:
                    result["mechanical_pass"] &= bool(current) and all(t.status == TaskStatus.SUCCEEDED for t in current)
                if case == "missing_transcript":
                    async with fixture.pool.acquire() as connection:
                        result["missing_input_evidence"] = [dict(row) for row in await connection.fetch(
                            """select e.id,e.finality,p.content->>'transcript_status' as transcript_status
                               from jarvis_context_events e join jarvis_context_payloads p
                               on p.owner_id=e.owner_id and p.id=e.payload_id
                               where e.owner_id=$1 and e.kind='user_transcript'""", fixture.owner)]
                    result["mechanical_pass"] &= bool(result["missing_input_evidence"]) and all(
                        row["finality"] == "unavailable" for row in result["missing_input_evidence"])
                    result["canonical_transcript_fault_injected"] = True
                if case == "multitask":
                    accepted = {row["tool"] for row in result["tools"]
                                if row.get("result", {}).get("accepted") is True}
                    result["mechanical_pass"] &= {
                        "update_background_task", "cancel_background_task"} <= accepted
                if case in {"interrupt", "reconnect"}:
                    accepted_updates = [row for row in result["tools"]
                        if row["tool"] == "update_background_task"
                        and row.get("result", {}).get("accepted") is True]
                    result["mechanical_pass"] &= (
                        len(current) == 1 and current[0].revision == 2
                        and current[0].status != TaskStatus.CANCELLED and len(accepted_updates) == 1)
                if case == "interrupt":
                    result["mechanical_pass"] &= not result["interruption"]["response_already_completed"]
                if case == "reconnect":
                    reconnect = result["reconnect"]
                    result["mechanical_pass"] &= (
                        len(reconnect["task_ids_before"]) == 1
                        and reconnect["task_ids_before"] == reconnect["task_ids_after"]
                        and reconnect["old_audio_session_id"] != reconnect["new_audio_session_id"]
                        and reconnect["old_capture_issue"] is None
                        and bool(reconnect["native_running_at_disconnect"]))
                result["speech_overlap_observed"] = any(
                    item["forwarded_audio_while_native_running"] for item in result["answers"][1:])
            finally:
                stop.set()
                # Every owner is drained even if an earlier cleanup callback
                # fails. The disposable schema closes only after this stack.
                async with AsyncExitStack() as cleanup:
                    if native_observer is not None:
                        cleanup.callback(native_observer.stop)
                    if worker_job is not None:
                        cleanup.push_async_callback(asyncio.wait_for, worker_job, 20)
                    if experience is not None:
                        cleanup.push_async_callback(experience.close)
                    if capture is not None:
                        cleanup.push_async_callback(capture.close)
                    if session_job is not None:
                        cleanup.push_async_callback(asyncio.wait_for, session_job, 20)
                    if session is not None:
                        cleanup.push_async_callback(session.close, "s05_native_fixture_finished")
                result["native_processes_remaining"] = active_native()
                if result["native_processes_remaining"]:
                    raise RuntimeError("Native worker did not drain")
    except Exception as error:
        result["error_type"] = type(error).__name__
        result["error_code"] = getattr(error, "code", None)
        result["mechanical_pass"] = False
    finally:
        profile_patches.close()
        result["tool_profile"] = {
            "enabled": profile is not None,
            "spans": profile.spans if profile is not None else [],
            "queries": profile.queries if profile is not None else [],
            "meaning": (
                "Foreground call-context phase/query tracing only; SQL digests, not SQL or arguments. "
                "Native background work outside that call context is not attributed to admission. "
                "Nested/concurrent spans must not be summed. Instrumented timing is diagnostic, "
                "not the uninstrumented latency acceptance population."),
        }
        result["conversation_journals"] = [journal.diagnostics() for journal in journals]
        if phone is not None:
            result["application_errors"] = [
                row for row in phone.observations if row["event"]["type"] == "error"]
            result["observations"] = phone.observations
            result["all_spoken_texts"] = speech_texts(phone.events)
            result["all_user_transcripts"] = [
                safe_app_event(e) for e in phone.events if e["type"] == "user.transcript.done"]
            result["audio_files"] = []
            for index, (item, pcm) in enumerate(phone.audio.items(), 1):
                name = f"forwarded-{index}.wav"
                write_wav(output / name, bytes(pcm))
                result["audio_files"].append({"item_id": item, "file": name, "source": "adapter_forwarded"})
        (output / "result.json").write_text(json.dumps(result, indent=2, default=str) + "\n")
    print(json.dumps({key: result.get(key) for key in (
        "case", "mechanical_pass", "error_type", "error_code", "speech_overlap_observed",
        "expected_task_count", "capture_issue", "answers",
    )}), flush=True)
    return result


def prepare_speech(directory):
    directory.mkdir(parents=True, exist_ok=False, mode=0o700)
    (directory / "utterances.json").write_text(json.dumps(UTTERANCES, indent=2) + "\n")
    for name, text in UTTERANCES.items():
        original, aiff, wav = (directory / (name + suffix) for suffix in (".txt", ".aiff", ".wav"))
        original.write_text(text)
        subprocess.run(["/usr/bin/say", "-r", "185", "-f", str(original), "-o", str(aiff)], check=True)
        subprocess.run(["/usr/bin/afconvert", "-f", "WAVE", "-d", "LEI16@24000", "-c", "1",
                        str(aiff), str(wav)], check=True)
        read_pcm(wav)
        aiff.unlink()


async def run(args):
    args.output.mkdir(parents=True, exist_ok=False, mode=0o700)
    settings = JarvisSettings()
    results = []
    try:
        for iteration in range(1, args.repeats + 1):
            for case in args.cases:
                for provider in args.providers:
                    results.append(await run_case(args, settings, provider, case, iteration))
    finally:
        timings = {}
        for provider in args.providers:
            rows = [row for result in results if result["provider"] == provider for row in result["tools"]]
            timings[provider] = {name: distribution([
                row["duration_ms"] for row in rows if row["tool"] == name])
                for name in sorted({row["tool"] for row in rows})}
        summary = {
            "format": "jarvis.s05.native_voice_matrix.v1",
            "configuration": {"providers": args.providers, "cases": args.cases, "repeats": args.repeats,
                              "frontier_model": args.frontier_model, "reasoning_effort": args.reasoning_effort,
                              "task_limits": TaskLimits().model_dump(mode="json"),
                              "fixture_result_wait_seconds": 150, "paired_context_samples": args.paired_context_samples,
                              "profile_tools": args.profile_tools},
            "results": [{key: value for key, value in r.items() if key not in {"observations", "tools"}}
                        for r in results],
            "tool_timings": timings,
            "task_context_timings": {
                provider: distribution([row["duration_ms"] for r in results if r["provider"] == provider
                                        for row in r["context_sections"]])
                for provider in args.providers
                if any(r["context_sections"] for r in results if r["provider"] == provider)},
            "task_context_timing_meaning": (
                "Elapsed added task-context section, including failures. Not end-to-end voice latency "
                "or a paired disabled-feature baseline."),
            "semantic_acceptance": "pending_independent_review",
            "physical_playback": "not_tested", "independent_cleanup": "required",
        }
        (args.output / "matrix.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    expected = args.repeats * len(args.cases) * len(args.providers)
    return 0 if len(results) == expected and all(r["mechanical_pass"] for r in results) else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    speech = commands.add_parser("prepare-speech")
    speech.add_argument("--output", type=Path, required=True)
    native = commands.add_parser("run")
    native.add_argument("--output", type=Path, required=True)
    native.add_argument("--speech-dir", type=Path, required=True)
    native.add_argument("--hermes-python", type=Path, required=True)
    native.add_argument("--frontier-model", default=DEFAULT_FRONTIER_MODEL)
    native.add_argument("--reasoning-effort", default="medium")
    native.add_argument("--providers", nargs="+", choices=("openai",), default=["openai"])
    native.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    native.add_argument("--repeats", type=int, choices=range(1, 4), default=3)
    native.add_argument("--paired-context-samples", type=int, choices=range(1, 31), default=20)
    native.add_argument("--profile-tools", action="store_true")
    args = parser.parse_args()
    logging.disable(logging.CRITICAL)
    if args.command == "prepare-speech":
        prepare_speech(args.output)
        return 0
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
