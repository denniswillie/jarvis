"""Paid native resumption checks with real Jarvis tools and disposable DB owners.

GoAway timing and the dropped socket are controlled triggers. The resumption
handle, native reconnect, streamed speech, domain execution, and persistence are
real. No ambiguous audio is replayed and no application history supplies the
conversation-only recall marker after resumption.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
from collections import defaultdict
from contextlib import suppress
from hashlib import sha256
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
from evals.gemini_live.interruption import DatabaseFixtures
from evals.gemini_live.session_probe import ProbePhone
from evals.gemini_live.transport_probe import read_pcm, redact_native, write_wav
from evals.perception_memory.fixture_cleanup import cleanup_fixture_owners


async def case(*, kind, fixtures, keys, speech_dir, output):
    owner, session_id = await fixtures.create("production-recovery-" + kind)
    repository = SessionRepository(fixtures.pool)
    experience = ExperienceEventService(
        repository=ExperienceRepository(fixtures.pool),
        segmenter=ExperienceEventSegmenter(EventSegmentationPolicy(300, 900)),
    )
    registry = fixtures.registry(owner, session_id)
    phone, connections, trace = ProbePhone(), [], []
    native_audio = defaultdict(bytearray)
    started = time.monotonic()

    class TracedConnection(GeminiLiveConnection):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            connections.append(self)
            self.generation, self.native_turn = len(connections), 1

        async def connect(self, **kwargs):
            trace.append({"direction": "setup", "generation": self.generation,
                          "has_resume_handle": bool(kwargs.get("resume_handle")),
                          "history_turns": len(kwargs.get("history", [])),
                          "setup_digest": sha256(json.dumps({key: kwargs.get(key) for key in ("instructions", "tools")},
                                                            sort_keys=True).encode()).hexdigest(),
                          "model": self.config.model, "voice": self.config.voice})
            await super().connect(**kwargs)

        async def send_event(self, event):
            await super().send_event(event)
            realtime = event.get("realtimeInput", {})
            if "audio" not in realtime:
                trace.append({"direction": "sent", "generation": self.generation,
                              "at_seconds": time.monotonic() - started, "event": redact_native(event)})

        async def receive_event(self):
            event = await super().receive_event()
            trace.append({"direction": "received", "generation": self.generation,
                          "at_seconds": time.monotonic() - started, "event": redact_native(event)})
            content = event.get("serverContent", {})
            for part in content.get("modelTurn", {}).get("parts", []):
                inline = part.get("inlineData", {})
                if inline.get("mimeType", "").startswith("audio/"):
                    native_audio[(self.generation, self.native_turn)].extend(
                        decode_pcm16(inline.get("data"), max_bytes=8 * 1024 * 1024)
                    )
            if content.get("turnComplete"):
                self.native_turn += 1
            return event

    session = GeminiAudioSession(
        websocket=phone, context=RealtimeSessionContext(
            audio_session_id=str(uuid4()), owner_id=owner, jarvis_session_id=session_id,
            client_kind="phone_audio", session=await repository.get_session(owner, session_id),
        ), settings=JarvisSettings(_env_file=None, GEMINI_API_KEY=keys["gemini"]),
        repository=repository, connection_factory=TracedConnection,
        tool_modules=(FunctionToolModule("real_fixture_memory", lambda _: registry.list()),),
        context_providers=(CurrentTimeContextProvider(), SessionContextProvider(), RecentMessagesContextProvider(12)),
        tool_execution_repository=ToolExecutionRepository(fixtures.pool), experience_event_service=experience,
    )
    running = asyncio.create_task(session.run())
    stage = "setup"
    result = {"case": kind, "passed": False, "device_playback": "not_tested",
              "recovery_trigger": "controlled_goaway_message" if kind == "goaway" else "controlled_native_socket_close"}

    async def until(predicate, timeout=35):
        async with asyncio.timeout(timeout):
            while not predicate():
                if running.done() or phone.closed:
                    raise RuntimeError("Audio session ended before expected recovery state")
                await asyncio.sleep(.01)

    async def speak(name):
        pcm = read_pcm(speech_dir / f"{name}.wav")
        for offset in range(0, len(pcm), 4800):
            chunk = pcm[offset:offset + 4800]
            await phone.send("audio.input.append", segment_id=name, audio_base64=base64.b64encode(chunk).decode())
            await asyncio.sleep(len(chunk) / 48000)
        await phone.send("audio.input.commit", segment_id=name)
        await phone.send("response.create")
        await until(lambda: name in session._turns and session._turns[name].output.completed
                    and session._turns[name].input_finalized and not session._domain_tasks)

    try:
        async with asyncio.timeout(180):
            await phone.wait_for("session.created")
            stage = "save"
            await speak("save")
            stage = "ordinary_conversation_marker"
            await speak("secret")
            original = await fixtures.snapshot(owner, session_id)
            assert len(original["memories"]) == 1
            stage = "safe_native_checkpoint"
            await until(lambda: session.recovery.handle is not None and session._checkpoint_safe())
            stage = "native_resumption"
            if kind == "goaway":
                await session._queue.put(("provider", (session._connection_generation, {"goAway": {"timeLeft": "15s"}})))
            else:
                await session.connection.close()
            await until(lambda: session.recovery.resumptions == 1)
            assert session._native_namespace == 1
            setups = [event for event in trace if event["direction"] == "setup"]
            assert all(item["has_resume_handle"] and item["history_turns"] == 0 for item in setups[1:])
            assert len({item["setup_digest"] for item in setups}) == 1
            assert len({item["model"] for item in setups}) == 1
            stage = "native_history_recall"
            await speak("recall")
            recall = session._turns["recall"].output.transcript
            assert "jade lantern" in recall.lower()
            for event in trace:
                if event["direction"] == "sent" and event["generation"] > 1:
                    assert "jade lantern" not in json.dumps(event["event"]).lower()
            recalled = await fixtures.snapshot(owner, session_id)
            assert len(recalled["memories"]) == 1
            stage = "new_intentional_save"
            await speak("followup")
            final = await fixtures.snapshot(owner, session_id)
            assert len(final["memories"]) == 2
            for turn in session._turns.values():
                assert turn.output.completed and not turn.output.interrupted
                assert turn.output.provider_audio_bytes > 0
                assert turn.output.provider_audio_bytes == turn.output.forwarded_audio_bytes
                assert turn.output.diagnostics()["provider_audio_sha256"] == turn.output.diagnostics()["forwarded_audio_sha256"]
            result.update(passed=True, recall=recall, original_memory_count=1, after_recall_memory_count=1,
                          final_memory_count=2, original_snapshot=original, final_snapshot=final)
    except Exception as error:
        result.update(failed_stage=stage, error_type=type(error).__name__)
    finally:
        await session.close("recovery_probe_finished")
        with suppress(asyncio.CancelledError):
            await running
        await experience.close()
        result["diagnostics"] = session.diagnostics()
        if session.journal.diagnostics()["degraded"] or session.diagnostics()["owned_tasks"] or session.diagnostics()["domain_tasks"]:
            result["passed"] = False
        result["trace"] = trace
        result["app_events"] = [event for event in phone.events if event["type"] != "audio.output.delta"]
        result["operations"] = [{"tool": call.name, "native_namespace": call.namespace, "state": call.state,
                                 "execution_id": call.execution_id, "native_id": call.native_id}
                                for call in session.operations.calls.values()]
        for (generation, native_turn), pcm in native_audio.items():
            write_wav(output / f"{kind}-native-g{generation}-t{native_turn}.wav", bytes(pcm))
        for index, pcm in enumerate(phone.audio.values(), 1):
            write_wav(output / f"{kind}-app-{index}.wav", bytes(pcm))
        result["cleanup"] = await cleanup_fixture_owners(keys["database"], {owner})
        (output / f"{kind}.json").write_text(json.dumps(result, indent=2, default=str) + "\n")
    print(json.dumps({key: result.get(key) for key in (
        "case", "passed", "failed_stage", "error_type", "recall", "original_memory_count", "final_memory_count",
    )} | {"recovery": result["diagnostics"]["recovery"], "persistence": result["diagnostics"]["conversation_persistence"],
          "cleanup_remaining": result["cleanup"]["remaining_rows"]}), flush=True)
    return result


async def run(args, keys):
    args.output.mkdir(parents=True, exist_ok=False)
    fixtures = DatabaseFixtures(keys["database"], args.output)
    results = []
    try:
        await fixtures.connect()
        for kind in ("goaway", "disconnect"):
            result = await case(kind=kind, fixtures=fixtures, keys=keys, speech_dir=args.speech_dir, output=args.output)
            results.append({"case": kind, "passed": result["passed"]})
    finally:
        await fixtures.cleanup()
        (args.output / "matrix.json").write_text(json.dumps({"results": results, "cleanup": fixtures.cleanup_result}, indent=2) + "\n")
    return 0 if results and all(item["passed"] for item in results) else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--speech-dir", type=Path, required=True)
    args = parser.parse_args()
    logging.disable(logging.CRITICAL)
    raise SystemExit(asyncio.run(run(args, json.load(sys.stdin))))
