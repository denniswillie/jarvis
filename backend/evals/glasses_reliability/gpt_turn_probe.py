"""Paid GPT turn acceptance using real sessions, tools, and isolated DB owners.

Only synthetic camera/speech, the phone sink, and an explicit writer-lock fault
are fixtures. No physical playback or historical-stall root cause is claimed.
"""
from __future__ import annotations

import argparse
import asyncio
from contextlib import AsyncExitStack
from datetime import UTC, datetime, timedelta
import inspect
import json
import logging
import os
from pathlib import Path
import re
import time
from unittest.mock import patch
from uuid import uuid4
import base64

from app.assistant.context_providers import CurrentTimeContextProvider, RecentMessagesContextProvider, SessionContextProvider
from app.experience.repository import ExperienceRepository, ToolExecutionRepository
from app.experience.segmenter import EventSegmentationPolicy, ExperienceEventSegmenter
from app.experience.service import ExperienceEventService
from app.memory.prospective.context import ProspectiveContextProvider
from app.memory.prospective.repository import ProspectiveRepository
from app.memory.prospective.service import ProspectiveMemoryService
from app.memory.prospective.tools import ProspectiveToolModule
from app.perception.current_image import CurrentImageService
from app.perception.frame_window import RecentVisualFrameWindow, SensoryVisualFrame
from app.perception.tools import CurrentImageToolModule
from app.realtime.models import RealtimeSessionContext
from app.realtime.openai_client import OpenAIRealtimeConnection
from app.realtime.session import RealtimeAudioSession
from app.realtime.session_manager import RealtimeSessionManager
from app.sessions.repository import SessionRepository
from app.settings import JarvisSettings
from evals.gemini_live.image_probe import marker_jpeg
from evals.gemini_live.interruption import DatabaseFixtures
from evals.gemini_live.transport_probe import read_pcm, write_wav
from evals.glasses_reliability.search_tool_probe import FocusOnlyContext, NativePhone
from evals.perception_memory.fixture_cleanup import cleanup_fixture_owners


PROMPTS = {
    "ordinary": "What is two plus two? Reply with only the answer.",
    "image": "Read the two printed words in my camera view. Reply with only those two words.",
    "followup": "Repeat just the second word, with nothing else.",
    "count": "Count slowly from one to one hundred, saying every number.",
    "interrupt": "Stop counting. What is two plus two? Reply with only the answer.",
    "tool": "Help me find my silver keys with a black fob. Tell me once you find them."
}
CASES = ("ordinary", "early_create", "image_followup", "interruption", "tool", "queued_write_timeout")


class Phone(NativePhone):
    def __init__(self):
        super().__init__()
        self.timeline = []

    async def send_json(self, event):
        self.timeline.append({
            "at": time.monotonic(), "type": event["type"],
            **{key: event[key] for key in ("response_id", "item_id", "code") if key in event},
        })
        await super().send_json(event)


class ObservedConnection(OpenAIRealtimeConnection):
    """Observe identities and clocks without copying media, instructions, or keys."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.writes, self.response_generations, self.first_audio = [], {}, {}
        self.commits = 0

    async def send_event(self, event):
        kind = None

        async def observed_payload():
            nonlocal kind
            payload = event() if callable(event) else event
            if inspect.isawaitable(payload):
                payload = await payload
            if payload is None:
                return None
            kind = payload.get("type")
            metadata = payload.get("response", {}).get("metadata", {})
            item = payload.get("item", {})
            self.writes.append({
                "type": kind, "at": time.monotonic(),
                "request_id": metadata.get("jarvis_request_id"),
                "turn_generation": metadata.get("jarvis_turn_generation"),
                "item_type": item.get("type"), "call_id": item.get("call_id"),
            })
            return payload

        sent = await super().send_event(observed_payload)
        if sent and kind == "input_audio_buffer.commit":
            self.commits += 1
        return sent

    async def receive_event(self):
        event = await super().receive_event()
        if event.get("type") == "response.created":
            response = event.get("response", {})
            self.response_generations[response.get("id")] = response.get("metadata", {}).get("jarvis_turn_generation")
        if event.get("type") == "response.output_audio.delta":
            self.first_audio.setdefault(event.get("response_id"), time.monotonic())
        return event


class TraceCapture(logging.Handler):
    def __init__(self, audio_session_id):
        super().__init__(logging.INFO)
        self.audio_session_id = audio_session_id
        self.records = []
        self.overflowed = False

    def emit(self, record):
        if record.msg != "realtime_trace %s":
            return
        value = json.loads(record.args[0])
        if value.get("audio_session_id") != self.audio_session_id:
            return
        if len(self.records) >= 4096:
            self.overflowed = True
            return
        self.records.append(value)

    def audit(self, wire):
        admitted = {e["request_id"]: e for e in self.records if e["event"] == "response_admitted"}
        started = {e.get("request_id") for e in self.records if e["event"] == "provider_response_started"}
        not_submitted = {e.get("request_id") for e in self.records if e["event"] == "response_not_submitted"}
        rejected = {e.get("request_id") for e in self.records if e["event"] == "response_rejected"}
        native_requests = {e["request_id"] for e in wire.writes if e["type"] == "response.create"}
        commits = {str(e["turn_generation"]): e["monotonic_s"] for e in self.records
                   if e["event"] == "input_commit_received"}
        forwarded = {}
        for event in self.records:
            if event["event"] == "first_audio_forwarded":
                forwarded.setdefault(event["response_id"], event["monotonic_s"])
        metrics = []
        for generation, committed_at in commits.items():
            response_ids = [key for key, value in wire.response_generations.items() if value == generation]
            native = [wire.first_audio[key] for key in response_ids if key in wire.first_audio]
            app = [forwarded[key] for key in response_ids if key in forwarded]
            metrics.append({
                "turn_generation": generation,
                "commit_to_first_provider_audio_ms": round((min(native) - committed_at) * 1000, 1) if native else None,
                "commit_to_first_forwarded_audio_ms": round((min(app) - committed_at) * 1000, 1) if app else None,
            })
        return {
            "admitted_requests": len(admitted),
            "unexplained_requests": sorted(set(admitted) - started - not_submitted - rejected),
            "rejected_requests": len(rejected & set(admitted)),
            "untraced_native_requests": sorted(native_requests - set(admitted)),
            "trace_overflowed": self.overflowed,
            "process_count": len({e["process_id"] for e in self.records}),
            "timing": metrics,
        }


async def stop_task(task):
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)


async def run_case(name, *, fixtures, speech_dir, output):
    owner, session_id = await fixtures.create("gpt-turn-" + name)
    audio_session_id = str(uuid4())
    trace, phone = TraceCapture(audio_session_id), Phone()
    result = {"case": name, "passed": False, "device_playback": "not_tested",
              "historical_cause": "unproven", "checks": {}}
    logger = logging.getLogger("app.realtime.telemetry")
    logger.addHandler(trace)
    session = wire = None
    utterances = 0
    try:
        async with AsyncExitStack() as stack, asyncio.timeout(150):
            repository = SessionRepository(fixtures.pool)
            prospective = ProspectiveRepository(fixtures.pool)
            manager = RealtimeSessionManager()
            settings = JarvisSettings(_env_file=None, VISUAL_REALTIME_ENABLED=name == "image_followup")
            service = ProspectiveMemoryService(repository=prospective, realtime_session_manager=manager, settings=settings)
            experience = ExperienceEventService(
                repository=ExperienceRepository(fixtures.pool),
                segmenter=ExperienceEventSegmenter(EventSegmentationPolicy(300, 900)),
            )
            stack.push_async_callback(experience.close)
            modules = (ProspectiveToolModule(service),) if name == "tool" else ()
            window = None
            if name == "image_followup":
                window = RecentVisualFrameWindow(max_frames_per_session=4, ttl_seconds=10)
                current = CurrentImageService(
                    frame_window=window, max_age_seconds=settings.VISUAL_REALTIME_MAX_CONTEXT_AGE_SECONDS,
                    max_image_bytes=settings.VISUAL_FRAME_MAX_BYTES, max_width=1024, max_height=1024,
                )
                modules = (CurrentImageToolModule(current),)
            with patch("app.realtime.session.OpenAIRealtimeConnection", ObservedConnection):
                session = RealtimeAudioSession(
                    websocket=phone, context=RealtimeSessionContext(
                        audio_session_id=audio_session_id, owner_id=owner, jarvis_session_id=session_id,
                        client_kind="phone_audio", session=await repository.get_session(owner, session_id),
                    ), settings=settings, repository=repository, tool_modules=modules,
                    context_providers=(CurrentTimeContextProvider(), SessionContextProvider(),
                                       RecentMessagesContextProvider(12),
                                       ProspectiveContextProvider(FocusOnlyContext(prospective))),
                    sensory_frame_window=window,
                    tool_execution_repository=ToolExecutionRepository(fixtures.pool),
                    experience_event_service=experience,
                )
            wire = session.openai
            await manager.register(audio_session_id=audio_session_id, owner_id=owner,
                                   jarvis_session_id=session_id, session=session)
            stack.push_async_callback(manager.unregister, audio_session_id=audio_session_id,
                                      owner_id=owner, jarvis_session_id=session_id)
            running = asyncio.create_task(session.run())

            async def shutdown():
                try:
                    await session.close("native_turn_probe_finished")
                finally:
                    if not running.done():
                        running.cancel()
                    await asyncio.gather(running, return_exceptions=True)

            stack.push_async_callback(shutdown)

            async def until(predicate, timeout=35):
                async with asyncio.timeout(timeout):
                    while not predicate():
                        if running.done() or phone.closed:
                            raise RuntimeError("Session ended before the expected boundary")
                        await asyncio.sleep(.005)

            if window is not None:
                lease = window.begin_stream(owner, session_id, uuid4())
                stack.callback(window.end_stream, lease)
                jpeg = marker_jpeg("copper orchard")
                (output / "camera-marker.jpg").write_bytes(jpeg)

                async def camera():
                    sequence = 0
                    while True:
                        sequence += 1
                        now = datetime.now(UTC)
                        window.put(SensoryVisualFrame(
                            frame_id=uuid4(), owner_id=owner, session_id=session_id,
                            audio_session_id=audio_session_id, visual_session_id=lease.visual_session_id,
                            frame_sequence=sequence, source="glasses", captured_at=now, server_received_at=now,
                            content_type="image/jpeg", width=640, height=360, storage_bucket="synthetic-fixture",
                            storage_path="marker.jpg", raw_image_expires_at=now + timedelta(minutes=1),
                            image_bytes=jpeg,
                        ), stream_lease=lease)
                        session.notify_visual_frame_available()
                        await asyncio.sleep(.5)

                stack.push_async_callback(stop_task, asyncio.create_task(camera()))
            await until(lambda: any(e["type"] == "session.created" for e in phone.events))

            async def speech(label, *, create=True, early=False):
                nonlocal utterances
                start = len(phone.events)
                utterances += 1
                segment = name + "-" + str(utterances)
                pcm = read_pcm(speech_dir / (label + ".wav"))
                for offset in range(0, len(pcm), 4800):
                    chunk = pcm[offset:offset + 4800]
                    await phone.send("audio.input.append", segment_id=segment,
                                     audio_base64=base64.b64encode(chunk).decode())
                    if early and offset == 0:
                        await phone.send("response.create")
                    await asyncio.sleep(len(chunk) / 48000)
                await phone.send("audio.input.commit", segment_id=segment)
                if create and not early:
                    await phone.send("response.create")
                return start

            async def answer(start):
                def completed():
                    generation = str(session._turn_generation)
                    audio = {e.get("response_id") for e in phone.events[start:]
                             if e["type"] == "audio.output.delta"
                             and wire.response_generations.get(e.get("response_id")) == generation}
                    done = {e.get("response_id") for e in phone.events[start:] if e["type"] == "response.done"}
                    return bool(audio & done) and not session._normal_turn_pending

                await until(completed)
                generation = str(session._turn_generation)
                return " ".join(
                    e.get("text", "") for e in phone.events[start:] if e["type"] == "transcript.done"
                    and wire.response_generations.get(e.get("response_id")) == generation
                )

            checks, answers = result["checks"], []
            if name == "image_followup":
                answers.append(await answer(await speech("image")))
                answers.append(await answer(await speech("followup")))
                checks["printed_words"] = re.findall(r"[a-z]+", answers[0].lower()) == ["copper", "orchard"]
                checks["followup_word"] = re.findall(r"[a-z]+", answers[1].lower()) == ["orchard"]
                checks["actual_image_ack"] = any(e["event"] == "image_provider_acknowledged" for e in trace.records)
            elif name == "interruption":
                start = await speech("count")
                await until(lambda: any(e["type"] == "audio.output.delta" for e in phone.events[start:]))
                old_response = next(e["response_id"] for e in phone.events[start:] if e["type"] == "audio.output.delta")
                checks["interrupted_while_active"] = session._active_responses[old_response].completed_response is None
                await phone.send("response.cancel")
                answers.append(await answer(await speech("interrupt")))
                cancelled_at = next(e["monotonic_s"] for e in trace.records if e["event"] == "response_cancel_requested")
                checks["no_old_audio_after_cancel"] = not any(
                    e["type"] == "audio.output.delta" and e.get("response_id") == old_response
                    and e["at"] >= cancelled_at for e in phone.timeline
                )
            elif name == "queued_write_timeout":
                before_commit = wire.commits
                await speech("ordinary", create=False)
                await until(lambda: wire.commits > before_commit)
                await wire._send_lock.acquire()
                try:
                    fault_started = time.monotonic()
                    await phone.send("response.create")
                    await until(lambda: any(e["type"] == "error" and e.get("code") == "response_preparation_failed"
                                            for e in phone.events))
                    await until(lambda: session._response_preparation_task is None)
                    result["forced_wait_seconds"] = round(time.monotonic() - fault_started, 3)
                    checks["bounded_write_failure"] = result["forced_wait_seconds"] < settings.OPENAI_REALTIME_WRITE_TIMEOUT_SECONDS + 3
                    checks["never_submitted"] = not any(e["type"] == "response.create" for e in wire.writes)
                    checks["reservation_released"] = session._pending_response is None and not session._normal_turn_pending
                finally:
                    wire._send_lock.release()
                answers.append(await answer(await speech("ordinary")))
            else:
                answers.append(await answer(await speech("tool" if name == "tool" else "ordinary",
                                                        early=name == "early_create")))
            if name in {"ordinary", "early_create", "interruption", "queued_write_timeout"}:
                checks["correct_spoken_answer"] = bool(set(re.findall(r"[a-z0-9]+", answers[-1].lower())) & {"four", "4"})
            if name == "early_create":
                checks["deferred_then_released"] = all(
                    any(e["event"] == kind for e in trace.records)
                    for kind in ("response_deferred", "deferred_response_released")
                )
                checks["one_native_response"] = sum(e["type"] == "response.create" for e in wire.writes) == 1
            if name == "tool":
                focus = await prospective.get_active_visual_focus(owner_id=owner)
                async with fixtures.pool.acquire() as connection:
                    executions = [dict(row) for row in await connection.fetch(
                        "select tool_name, status from jarvis_tool_executions where owner_id=$1 and session_id=$2",
                        owner, session_id,
                    )]
                result["executions"] = executions
                result["focus"] = focus.as_prompt_payload() if focus is not None else None
                checks["real_tool_succeeded"] = any(e["tool_name"] == "set_visual_focus" and e["status"] == "completed"
                                                     for e in executions)
                checks["correct_focus"] = focus is not None and focus.mode == "find_once" and "key" in focus.target.casefold()
                checks["tool_output_followed_by_native_continuation"] = (
                    any(e.get("item_type") == "function_call_output" for e in wire.writes)
                    and sum(e["type"] == "response.create" for e in wire.writes) >= 2
                )
            await until(lambda: sum(e["type"] == "user.transcript.done" for e in phone.events) >= utterances)
            errors = [e.get("code") for e in phone.events if e["type"] == "error"]
            checks["expected_errors_only"] = (
                errors == ["response_preparation_failed"] if name == "queued_write_timeout" else not errors
            )
            result.update(answers=answers, configured_model=settings.OPENAI_REALTIME_MODEL,
                          audit=trace.audit(wire), wire=wire.writes)
            checks["all_admissions_explained"] = not result["audit"]["unexplained_requests"]
            checks["native_writes_have_admission_traces"] = (
                result["audit"]["admitted_requests"] > 0 and not result["audit"]["untraced_native_requests"]
            )
            checks["complete_single_process_trace"] = not trace.overflowed and result["audit"]["process_count"] == 1
            forwarded_times = [row["commit_to_first_forwarded_audio_ms"] for row in result["audit"]["timing"]]
            checks["forwarded_audio_measured"] = (
                len(forwarded_times) == utterances
                and sum(value is None for value in forwarded_times) == (1 if name == "queued_write_timeout" else 0)
                and all(value >= 0 for value in forwarded_times if value is not None)
            )
            result["passed"] = all(checks.values())
    except Exception as error:
        result["error_type"] = type(error).__name__
        if wire is not None:
            result["audit"] = trace.audit(wire)
    finally:
        logger.removeHandler(trace)
        result["trace"] = trace.records
        result["app_timeline"] = phone.timeline
        for index, pcm in enumerate(phone.audio.values()):
            write_wav(output / (name + "-" + str(index) + ".wav"), bytes(pcm))
        result["cleanup"] = await cleanup_fixture_owners(fixtures.dsn, {owner})
        (output / (name + ".json")).write_text(json.dumps(result, indent=2, default=str) + "\n")
    print(json.dumps({key: result.get(key) for key in ("case", "passed", "checks", "answers", "audit", "error_type")}
                     | {"remaining_rows": result["cleanup"]["remaining_rows"]}), flush=True)
    return result


async def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    fixtures = DatabaseFixtures(os.environ["JARVIS_TEST_DB_URL"], args.output)
    results = []
    logging.getLogger("app.realtime.telemetry").setLevel(logging.INFO)
    logging.getLogger("app.realtime.telemetry").propagate = False
    try:
        await fixtures.connect()
        for name in args.cases.split(","):
            results.append(await run_case(name, fixtures=fixtures, speech_dir=args.speech_dir, output=args.output))
    finally:
        await fixtures.cleanup()
        (args.output / "matrix.json").write_text(json.dumps({
            "results": [{key: result.get(key) for key in ("case", "passed", "checks", "audit", "configured_model")}
                        for result in results],
            "cleanup": fixtures.cleanup_result,
        }, indent=2, default=str) + "\n")
    return 0 if results and all(result["passed"] for result in results) else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--speech-dir", type=Path, required=True)
    parser.add_argument("--cases", default=",".join(CASES))
    args = parser.parse_args()
    if any(name not in CASES for name in args.cases.split(",")):
        parser.error("Unknown case")
    logging.basicConfig(level=logging.CRITICAL)
    raise SystemExit(asyncio.run(run(args)))
