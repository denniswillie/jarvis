"""Matched paid GPT control for the production Gemini interruption matrix.

Only observation barriers, native transport tracing, and a fixture phone are
added. The existing GPT coordinator and real memory handlers are not replaced.
Durable counts and spoken-status correctness are reported separately.
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
from unittest.mock import patch
from uuid import uuid4

from app.assistant.context_providers import CurrentTimeContextProvider, RecentMessagesContextProvider, SessionContextProvider
from app.experience.repository import ExperienceRepository, ToolExecutionRepository
from app.experience.segmenter import EventSegmentationPolicy, ExperienceEventSegmenter
from app.experience.service import ExperienceEventService
from app.realtime.models import RealtimeSessionContext
from app.realtime.openai_client import OpenAIRealtimeConnection
from app.realtime.session import RealtimeAudioSession
from app.sessions.repository import SessionRepository
from app.settings import JarvisSettings
from app.tools.module import FunctionToolModule
from evals.gemini_live.domain_speech_probe import STAGES
from evals.gemini_live.interruption import Barrier, DatabaseFixtures, Trace, instrument_registry, tool_result_succeeded
from evals.gemini_live.paired_probe import PairedPhone, observed_openai_write, safe_event
from evals.gemini_live.transport_probe import read_pcm, write_wav
from evals.perception_memory.fixture_cleanup import cleanup_fixture_owners


def completed_new_audio_response(events, previous_ids):
    audible = {event.get("response_id") for event in events if event["type"] == "audio.output.delta"}
    audible.difference_update(previous_ids)
    audible.discard(None)
    return any(event["type"] == "response.done" and event.get("response_id") in audible for event in events)


def successful_memory_receipt(event, successful_calls):
    item = event.get("item", {})
    return (event.get("type") == "conversation.item.create" and isinstance(item, dict)
            and item.get("type") == "function_call_output" and item.get("call_id") in successful_calls)


async def case(*, stage, iteration, fixtures, keys, speech_dir, output):
    label = f"{stage}-{iteration}"
    owner, session_id = await fixtures.create("production-gpt-audio-" + label)
    trace, phone = Trace(), PairedPhone()
    barrier = Barrier(stage, trace)
    registry = instrument_registry(fixtures.registry(owner, session_id), barrier, trace)
    repository = SessionRepository(fixtures.pool)
    experience = ExperienceEventService(repository=ExperienceRepository(fixtures.pool),
        segmenter=ExperienceEventSegmenter(EventSegmentationPolicy(300, 900)))
    native_audio = defaultdict(bytearray)
    receipt_sent = False
    successful_calls = set()
    session = running = None
    result = {"case": label, "provider": "openai", "durable_counts_passed": False,
              "spoken_status": "requires_independent_audio_review", "device_playback": "not_tested"}

    def sent(direction, event):
        nonlocal receipt_sent
        trace.add("native_sent", payload=safe_event(event))
        if successful_memory_receipt(event, successful_calls):
            receipt_sent = True

    class TracedConnection(OpenAIRealtimeConnection):
        async def send_event(self, event):
            return await observed_openai_write(super().send_event, event, sent)

        async def receive_event(self):
            event = await super().receive_event()
            trace.add("native_received", payload=safe_event(event))
            if event.get("type") in {"response.output_audio.delta", "response.audio.delta"}:
                native_audio[event["item_id"]].extend(base64.b64decode(event["delta"], validate=True))
                if stage == "after_receipt" and receipt_sent and not barrier.claimed:
                    barrier.claimed = True
                    trace.add("barrier_reached", stage=stage)
                    barrier.reached.set()
            return event

    async def until(predicate):
        async with asyncio.timeout(40):
            while not predicate():
                if phone.closed or running.done():
                    raise RuntimeError("GPT run ended before the expected boundary")
                await asyncio.sleep(.01)

    async def speak(name, *, release_boundary=False):
        pcm = read_pcm(speech_dir / f"{name}.wav")
        generation = session._turn_generation
        for offset in range(0, len(pcm), 4800):
            chunk = pcm[offset:offset + 4800]
            await phone.send("audio.input.append", segment_id=name, audio_base64=base64.b64encode(chunk).decode())
            if offset == 0 and release_boundary:
                await until(lambda: session._turn_generation > generation and session._user_audio_pending)
                barrier.release.set()
            await asyncio.sleep(len(chunk) / 48000)
        await phone.send("audio.input.commit", segment_id=name)
        await phone.send("response.create")

    async def answer(name, *, release_boundary=False):
        start = len(phone.events)
        previous_ids = {event.get("response_id") for event in phone.events} | set(session._active_responses)
        await speak(name, release_boundary=release_boundary)
        await until(lambda: completed_new_audio_response(phone.events[start:], previous_ids))
        result.setdefault("answers", {})[name] = [event["text"] for event in phone.events[start:]
            if event["type"] == "transcript.done" and event.get("response_id") not in previous_ids]

    try:
        async with asyncio.timeout(180):
            settings = JarvisSettings(_env_file=None, **{
                **keys.get("settings", {}), "OPENAI_API_KEY": keys["openai"],
                "OPENAI_REALTIME_MODEL": keys["openai_model"],
            })
            with patch("app.realtime.session.OpenAIRealtimeConnection", TracedConnection):
                session = RealtimeAudioSession(websocket=phone, context=RealtimeSessionContext(
                    audio_session_id=str(uuid4()), owner_id=owner, jarvis_session_id=session_id,
                    client_kind="phone_audio", session=await repository.get_session(owner, session_id)),
                    settings=settings, repository=repository,
                    tool_modules=(FunctionToolModule("real_memory_fixture", lambda _: registry.list()),),
                    context_providers=(CurrentTimeContextProvider(), SessionContextProvider(), RecentMessagesContextProvider(12)),
                    tool_execution_repository=ToolExecutionRepository(fixtures.pool), experience_event_service=experience)
            original_admission = session._handle_function_call_item

            async def admission(item, *, response_id):
                if item.get("name") == "remember":
                    trace.add("native_call_known", call_id=item.get("call_id"))
                    await barrier.pause("before_admission")
                return await original_admission(item, response_id=response_id)

            session._handle_function_call_item = admission
            executor = session.tool_executor

            class ObservedExecutor:
                async def execute(self, name, arguments, **kwargs):
                    trace.add("execution_admitted", tool=name, call_id=kwargs.get("provider_call_id"))
                    value = await executor.execute(name, arguments, **kwargs)
                    trace.add("execution_returned", tool=name, result=value)
                    if name == "remember" and tool_result_succeeded(value):
                        successful_calls.add(kwargs.get("provider_call_id"))
                        await barrier.pause("after_commit")
                    return value

            session.tool_executor = ObservedExecutor()
            running = asyncio.create_task(session.run())
            await phone.wait_for("session.created")
            await speak("save")
            await until(barrier.reached.is_set)
            await phone.send("response.cancel")
            await answer("interrupt", release_boundary=True)
            await answer("status")
            original = await fixtures.snapshot(owner, session_id)
            await answer("followup")
            await phone.wait_for("user.transcript.done", count=4)
            final = await fixtures.snapshot(owner, session_id)
            expected = 0 if stage == "before_admission" else 1
            result.update(original_snapshot=original, final_snapshot=final,
                original_memory_count=len(original["memories"]), final_memory_count=len(final["memories"]),
                expected_original_memory_count=expected)
            result["durable_counts_passed"] = (len(original["memories"]) == expected
                and len(final["memories"]) == expected + 1
                and not any(event["type"] == "error" for event in phone.events))
    except Exception as error:
        result["error_type"] = type(error).__name__
    finally:
        barrier.release.set()
        if session is not None:
            await session.close("gpt_domain_probe_finished")
        if running is not None:
            with suppress(asyncio.CancelledError):
                await running
        await experience.close()
        result["trace"] = trace.events
        result["boundary_observed"] = barrier.reached.is_set()
        result["app_events"] = [safe_event(event) for event in phone.events if event["type"] != "audio.output.delta"]
        result["audio_files"] = []
        for index, (item, pcm) in enumerate(native_audio.items(), 1):
            name = f"{label}-native-{index}.wav"
            write_wav(output / name, bytes(pcm))
            result["audio_files"].append({"item_id": item, "file": name, "source": "native"})
        for index, (item, pcm) in enumerate(phone.audio.items(), 1):
            name = f"{label}-app-{index}.wav"
            write_wav(output / name, bytes(pcm))
            result["audio_files"].append({"item_id": item, "file": name, "source": "forwarded"})
        result["cleanup"] = await cleanup_fixture_owners(keys["database"], {owner})
        (output / f"{label}.json").write_text(json.dumps(result, indent=2, default=str) + "\n")
    print(json.dumps({key: result.get(key) for key in ("case", "durable_counts_passed", "error_type",
        "original_memory_count", "final_memory_count", "answers")} | {"cleanup_remaining": result["cleanup"]["remaining_rows"]}), flush=True)
    return result


async def run(args, keys):
    stages = args.stages.split(",")
    if any(stage not in STAGES for stage in stages) or not 1 <= args.repeats <= 4:
        raise ValueError("Invalid GPT control matrix")
    args.output.mkdir(parents=True, exist_ok=False)
    fixtures = DatabaseFixtures(keys["database"], args.output)
    results = []
    try:
        await fixtures.connect()
        for iteration in range(1, args.repeats + 1):
            for stage in stages:
                results.append(await case(stage=stage, iteration=iteration, fixtures=fixtures, keys=keys,
                    speech_dir=args.speech_dir, output=args.output))
    finally:
        await fixtures.cleanup()
        (args.output / "matrix.json").write_text(json.dumps({
            "results": [{key: r.get(key) for key in ("case", "durable_counts_passed", "error_type",
                "original_memory_count", "final_memory_count", "answers")} for r in results],
            "cleanup": fixtures.cleanup_result, "spoken_status_acceptance": "pending_independent_review",
        }, indent=2) + "\n")
    return 0 if len(results) == len(stages) * args.repeats and all(r["durable_counts_passed"] for r in results) else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--speech-dir", type=Path, required=True)
    parser.add_argument("--stages", default=",".join(STAGES))
    parser.add_argument("--repeats", type=int, default=2)
    logging.disable(logging.CRITICAL)
    raise SystemExit(asyncio.run(run(parser.parse_args(), json.load(sys.stdin))))
