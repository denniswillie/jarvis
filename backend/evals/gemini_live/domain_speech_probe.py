"""Paid streamed-speech interruption checks through the production Gemini adapter.

Only the phone sink and precise observation barriers are evaluation code. Tools,
admission, receipts, context, messages, and experience evidence use real services.
Each case has a tracked disposable database owner and mandatory independent cleanup.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
from collections import defaultdict
from contextlib import suppress
import json
import logging
from pathlib import Path
import sys
import time
from uuid import uuid4

from app.assistant.context_providers import CurrentTimeContextProvider, RecentMessagesContextProvider, SessionContextProvider
from app.experience.repository import ExperienceRepository, ToolExecutionRepository
from app.experience.segmenter import EventSegmentationPolicy, ExperienceEventSegmenter
from app.experience.service import ExperienceEventService
from app.realtime.gemini.client import GeminiLiveConnection
from app.realtime.gemini.events import decode_pcm16
from app.realtime.gemini.session import GeminiAudioSession
from app.realtime.models import RealtimeSessionContext
from app.sessions.repository import SessionRepository
from app.settings import JarvisSettings
from app.tools.module import FunctionToolModule
from evals.gemini_live.interruption import Barrier, DatabaseFixtures, Trace, instrument_registry, tool_result_succeeded
from evals.gemini_live.session_probe import ProbePhone
from evals.gemini_live.transport_probe import read_pcm, redact_native, write_wav
from evals.perception_memory.fixture_cleanup import cleanup_fixture_owners


STAGES = ("before_admission", "during_execution", "after_commit", "after_receipt")


async def case(*, stage, iteration, fixtures, keys, speech_dir, output):
    label = f"{stage}-{iteration}"
    owner, session_id = await fixtures.create("production-audio-" + label)
    trace, phone = Trace(), ProbePhone()
    barrier = Barrier(stage, trace)
    registry = instrument_registry(fixtures.registry(owner, session_id), barrier, trace)
    repository = SessionRepository(fixtures.pool)
    experience = ExperienceEventService(
        repository=ExperienceRepository(fixtures.pool),
        segmenter=ExperienceEventSegmenter(EventSegmentationPolicy(300, 900)),
    )
    native_trace, connections = [], []
    native_audio = defaultdict(bytearray)
    receipt_sent = False

    class TracedConnection(GeminiLiveConnection):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            connections.append(self)
            self.generation, self.native_turn = len(connections), 1

        async def send_event(self, event):
            nonlocal receipt_sent
            await super().send_event(event)
            realtime = event.get("realtimeInput", {})
            if "text" in realtime:
                trace.add("application_context_sent", generation=self.generation,
                          context=json.loads(realtime["text"]))
            if "activityEnd" in realtime:
                trace.add("activity_end_sent", generation=self.generation)
            if any(response.get("name") == "remember" and tool_result_succeeded(response.get("response"))
                   for response in event.get("toolResponse", {}).get("functionResponses", [])):
                receipt_sent = True
                trace.add("native_receipt_sent", generation=self.generation)

        async def receive_event(self):
            event = await super().receive_event()
            native_trace.append({"at_seconds": time.monotonic() - trace.started,
                                 "generation": self.generation, "native_turn": self.native_turn,
                                 "event": redact_native(event)})
            content = event.get("serverContent", {})
            for part in content.get("modelTurn", {}).get("parts", []):
                inline = part.get("inlineData", {})
                if inline.get("mimeType", "").startswith("audio/"):
                    native_audio[(self.generation, self.native_turn)].extend(
                        decode_pcm16(inline.get("data"), max_bytes=8 * 1024 * 1024)
                    )
                    if stage == "after_receipt" and receipt_sent and not barrier.claimed:
                        barrier.claimed = True
                        trace.add("barrier_reached", stage=stage)
                        barrier.reached.set()
            if content.get("turnComplete"):
                self.native_turn += 1
            return event

    settings = JarvisSettings(_env_file=None, GEMINI_API_KEY=keys["gemini"],
                             GEMINI_LIVE_MODEL=keys.get("gemini_model", "gemini-3.1-flash-live-preview"))
    session = GeminiAudioSession(
        websocket=phone, context=RealtimeSessionContext(
            audio_session_id=str(uuid4()), owner_id=owner, jarvis_session_id=session_id,
            client_kind="phone_audio", session=await repository.get_session(owner, session_id),
        ), settings=settings, repository=repository, connection_factory=TracedConnection,
        tool_modules=(FunctionToolModule("real_memory_fixture", lambda _: registry.list()),),
        context_providers=(CurrentTimeContextProvider(), SessionContextProvider(), RecentMessagesContextProvider(12)),
        tool_execution_repository=ToolExecutionRepository(fixtures.pool), experience_event_service=experience,
    )
    original_execute = session.operations.execute

    async def observed_admission(call, **kwargs):
        if call.name == "remember":
            trace.add("native_call_known", native_id=call.native_id, status_ref=call.status_ref)
            await barrier.pause("before_admission")
        return await original_execute(call, **kwargs)

    session.operations.execute = observed_admission
    real_executor = session.operations.executor

    class ObservedExecutor:
        async def execute(self, name, arguments, **kwargs):
            trace.add("execution_admitted", tool=name, call_id=kwargs.get("provider_call_id"))
            result = await real_executor.execute(name, arguments, **kwargs)
            trace.add("execution_returned", tool=name, result=result)
            if name == "remember" and tool_result_succeeded(result):
                await barrier.pause("after_commit")
            return result

    session.operations.executor = ObservedExecutor()
    running = asyncio.create_task(session.run())
    result = {"case": label, "passed": False, "device_playback": "not_tested"}

    async def until(predicate, *, timeout=35):
        async with asyncio.timeout(timeout):
            while not predicate():
                if phone.closed or running.done():
                    raise RuntimeError("The production audio run ended before the expected boundary")
                await asyncio.sleep(.01)

    async def speak(name, segment, *, release_boundary=False):
        pcm = read_pcm(speech_dir / f"{name}.wav")
        for offset in range(0, len(pcm), 4800):
            chunk = pcm[offset:offset + 4800]
            await phone.send("audio.input.append", segment_id=segment, audio_base64=base64.b64encode(chunk).decode())
            if offset == 0 and release_boundary:
                await until(lambda: segment in session._turns)
                # The application has received the new activity before releasing
                # the old action barrier. No arbitrary sleep defines admission.
                barrier.release.set()
            await asyncio.sleep(len(chunk) / 48000)
        await phone.send("audio.input.commit", segment_id=segment)
        await phone.send("response.create")

    async def finished(segment):
        await until(lambda: segment in session._turns and session._turns[segment].output.completed)

    try:
        async with asyncio.timeout(150):
            await phone.wait_for("session.created")
            await speak("save", "save")
            await until(barrier.reached.is_set)
            await phone.send("response.cancel")
            await speak("interrupt", "interrupt", release_boundary=True)
            await finished("interrupt")
            await speak("status", "status")
            await finished("status")
            await until(lambda: not session._domain_tasks)
            original = await fixtures.snapshot(owner, session_id)
            await speak("followup", "followup")
            await finished("followup")
            await until(lambda: not session._domain_tasks)
            final = await fixtures.snapshot(owner, session_id)
            expected = 0 if stage == "before_admission" else 1
            result.update({"original_memory_count": len(original["memories"]),
                           "final_memory_count": len(final["memories"]),
                           "expected_original_memory_count": expected,
                           "original_snapshot": original, "final_snapshot": final})
            assert len(original["memories"]) == expected
            assert len(final["memories"]) == expected + 1
            lookups = [call for call in session.operations.calls.values() if call.name == "get_operation_status"]
            assert all(call.result and call.result.get("status") == "success" for call in lookups)
            result["passed"] = True
    except Exception as error:
        result["error_type"] = type(error).__name__
    finally:
        barrier.release.set()
        await session.close("domain_probe_finished")
        with suppress(asyncio.CancelledError):
            await running
        await experience.close()
        diagnostics = session.diagnostics()
        if diagnostics["domain_tasks"] or diagnostics["conversation_persistence"]["degraded"]:
            result["passed"] = False
        result.update({"boundary_observed": barrier.reached.is_set(),
                       "diagnostics": diagnostics, "application_trace": trace.events,
                       "native_trace": native_trace,
                       "app_events": [event for event in phone.events if event["type"] != "audio.output.delta"],
                       "operations": [{"native_id": call.native_id, "status_ref": call.status_ref,
                                       "turn_number": call.turn_number, "namespace": call.namespace,
                                       "tool": call.name, "arguments": call.arguments, "state": call.state,
                                       "admitted": call.admitted, "delivery_cancelled": call.delivery_cancelled,
                                       "delivered": call.delivered, "result": call.result}
                                      for call in session.operations.calls.values()]})
        for (generation, native_turn), pcm in native_audio.items():
            write_wav(output / f"{label}-native-g{generation}-t{native_turn}.wav", bytes(pcm))
        for index, pcm in enumerate(phone.audio.values(), 1):
            write_wav(output / f"{label}-app-{index}.wav", bytes(pcm))
        # Drain all production work before independently cleaning this owner.
        # Prompt cleanup also avoids leaving an earlier case for live sweepers.
        result["cleanup"] = await cleanup_fixture_owners(keys["database"], {owner})
        (output / f"{label}.json").write_text(json.dumps(result, indent=2, default=str) + "\n")
    print(json.dumps({"case": label, "passed": result["passed"], "error_type": result.get("error_type"),
                      "original_memories": result.get("original_memory_count"),
                      "final_memories": result.get("final_memory_count"),
                      "persistence": diagnostics["conversation_persistence"],
                      "cleanup_remaining": result["cleanup"]["remaining_rows"],
                      "answers": [event for event in phone.events if event["type"] in {"transcript.done", "error"}]}), flush=True)
    return result


async def run(args, keys):
    stages = args.stages.split(",")
    if any(stage not in STAGES for stage in stages) or not 1 <= args.repeats <= 4:
        raise ValueError("Invalid stage matrix")
    args.output.mkdir(parents=True, exist_ok=False)
    fixtures = DatabaseFixtures(keys["database"], args.output)
    results = []
    try:
        await fixtures.connect()
        for iteration in range(1, args.repeats + 1):
            for stage in stages:
                result = await case(stage=stage, iteration=iteration, fixtures=fixtures, keys=keys,
                                    speech_dir=args.speech_dir, output=args.output)
                results.append({"case": result["case"], "passed": result["passed"]})
    finally:
        await fixtures.cleanup()
        (args.output / "matrix.json").write_text(json.dumps({"results": results, "cleanup": fixtures.cleanup_result}, indent=2) + "\n")
    return 0 if results and all(result["passed"] for result in results) else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--speech-dir", type=Path, required=True)
    parser.add_argument("--stages", default=",".join(STAGES))
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    logging.disable(logging.CRITICAL)
    raise SystemExit(asyncio.run(run(args, json.load(sys.stdin))))
