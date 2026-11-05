"""Paid controlled speech through the production Gemini audio-session coordinator.

Context retrieval, message storage, experience attachment, and the phone's output
sink are explicit in-memory fixtures. Input handling,
native transport, preparation scheduling, interruption, and cleanup are production
code. No domain mutations, database access, or device playback are performed.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
from collections import defaultdict
from contextlib import suppress
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from uuid import uuid4

from app.realtime.gemini.client import GeminiLiveConnection
from app.realtime.gemini.events import decode_pcm16
from app.realtime.gemini.session import GeminiAudioSession, PreparedInput
from app.realtime.models import RealtimeSessionContext
from app.settings import JarvisSettings
from evals.gemini_live.transport_probe import read_pcm, redact_native, write_wav


class ProbePhone:
    def __init__(self):
        self.input = asyncio.Queue(maxsize=32)
        self.events = []
        self.audio = defaultdict(bytearray)
        self.changed = asyncio.Condition()
        self.closed = False

    async def iter_text(self):
        while not self.closed:
            yield json.dumps(await self.input.get())

    async def send(self, kind, **fields):
        await self.input.put({"type": kind, "client_event_id": str(uuid4()), **fields})

    async def send_json(self, event):
        self.events.append(event)
        if event["type"] == "audio.output.delta":
            self.audio[event["item_id"]].extend(decode_pcm16(event["audio_base64"], max_bytes=8 * 1024 * 1024))
        async with self.changed:
            self.changed.notify_all()

    async def close(self, **kwargs):
        self.closed = True
        async with self.changed:
            self.changed.notify_all()

    async def wait_for(self, kind, count=1):
        async with self.changed:
            await self.changed.wait_for(lambda: self.closed or sum(e["type"] == kind for e in self.events) >= count)
        if self.closed:
            raise RuntimeError("Session closed before the expected event")


class ContextFixtureSession(GeminiAudioSession):
    case = "ordinary"

    async def _prepare_turn(self, turn):
        if self.case == "supersede_preparation" and turn.segment_id == "a":
            self.preparation_entered.set()
            await asyncio.Event().wait()
        return PreparedInput(json.dumps({
            "kind": "jarvis_application_context", "segment_id": turn.segment_id,
            "cutoff_at": turn.cutoff_at, "current_test_marker": "silver meadow",
            "current_visual_evidence": "unavailable", "operations": [],
        }))


class ProbeMessageRepository:
    """Transport-only storage fixture; never evidence of database durability."""

    def __init__(self):
        self.messages = {}
        self.attached = set()

    async def create_or_reuse_gemini_message(self, **fields):
        key = tuple(fields[name] for name in (
            "owner_id", "session_id", "audio_session_id", "logical_item_id", "role",
        ))
        existing = self.messages.get(key)
        if existing is not None:
            if existing.content != fields["content"]:
                raise ValueError("Conflicting fixture transcript")
            return existing
        message = SimpleNamespace(id=uuid4(), **fields)
        self.messages[key] = message
        return message

    async def gemini_message_has_experience_evidence(self, *, owner_id, session_id, message_id):
        return (owner_id, session_id, message_id) in self.attached


class ProbeExperienceService:
    def __init__(self, repository):
        self.repository = repository

    async def attach_message(self, *, owner_id, message):
        self.repository.attached.add((owner_id, message.session_id, message.id))
        return SimpleNamespace(attached=True)


async def probe(*, case: str, key: str, speech_dir: Path, output: Path) -> dict:
    phone = ProbePhone()
    started = time.monotonic()
    trace = []
    raw_audio = defaultdict(bytearray)
    connections = []

    class TracedConnection(GeminiLiveConnection):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            connections.append(self)
            self.generation = len(connections)
            self.native_turn = 1

        async def receive_event(self):
            event = await super().receive_event()
            trace.append({"at_seconds": time.monotonic() - started,
                          "generation": self.generation, "native_turn": self.native_turn,
                          "event": redact_native(event)})
            content = event.get("serverContent", {})
            for part in content.get("modelTurn", {}).get("parts", []):
                inline = part.get("inlineData", {})
                if inline.get("mimeType", "").startswith("audio/"):
                    raw_audio[(self.generation, self.native_turn)].extend(
                        decode_pcm16(inline.get("data"), max_bytes=8 * 1024 * 1024)
                    )
            if content.get("turnComplete"):
                self.native_turn += 1
            return event

    settings = JarvisSettings(_env_file=None, GEMINI_API_KEY=key)
    repository = ProbeMessageRepository()
    session = ContextFixtureSession(
        websocket=phone, context=RealtimeSessionContext(
            audio_session_id=str(uuid4()), owner_id="isolated-session-probe",
            jarvis_session_id=uuid4(), client_kind="phone_audio",
            session=SimpleNamespace(behavior_instructions=""),
        ), settings=settings, repository=repository, connection_factory=TracedConnection,
        experience_event_service=ProbeExperienceService(repository),
    )
    session.case = case
    session.preparation_entered = asyncio.Event()
    run = asyncio.create_task(session.run())
    error = None

    async def speak(name, segment, *, commit=True):
        pcm = read_pcm(speech_dir / f"{name}.wav")
        for offset in range(0, len(pcm), 4800):
            chunk = pcm[offset:offset + 4800]
            await phone.send("audio.input.append", segment_id=segment,
                             audio_base64=base64.b64encode(chunk).decode())
            await asyncio.sleep(len(chunk) / 48000)
        if commit:
            await phone.send("audio.input.commit", segment_id=segment)
            await phone.send("response.create")

    try:
        async with asyncio.timeout(100):
            await phone.wait_for("session.created")
            if case in {"ordinary", "explanation", "long_input"}:
                await speak("marker" if case == "ordinary" else case, "a")
                await phone.wait_for("response.done")
            else:
                await speak("explanation", "a")
                if case == "barge_in":
                    await phone.wait_for("audio.output.delta")
                    await asyncio.sleep(.35)
                elif case == "supersede_preparation":
                    await session.preparation_entered.wait()
                elif case == "early_barge":
                    # Allow the first commit to release, but interrupt before its
                    # usual first-audio latency. The trace records the actual case.
                    await asyncio.sleep(.08)
                await phone.send("response.cancel")
                await speak("marker", "b")
                await phone.wait_for("response.done", count=2)
            await asyncio.sleep(.5)
    except Exception as exc:
        error = type(exc).__name__
    finally:
        await session.close("probe_complete")
        with suppress(asyncio.CancelledError):
            await run

    native_files = []
    for (generation, native_turn), pcm in raw_audio.items():
        name = f"{case}-native-g{generation}-t{native_turn}.wav"
        write_wav(output / name, bytes(pcm))
        native_files.append(name)
    for index, (item, pcm) in enumerate(phone.audio.items(), 1):
        write_wav(output / f"{case}-app-item-{index}.wav", bytes(pcm))
    result = {
        "case": case, "error_type": error, "connections": len(connections),
        "passed": error is None and not session.journal.diagnostics()["degraded"]
                  and not any(event["type"] == "error" for event in phone.events),
        "diagnostics": session.diagnostics(), "native_audio_files": native_files,
        "app_events": redact_native([
            {k: v for k, v in event.items() if k != "audio_base64"}
            for event in phone.events if event["type"] != "audio.output.delta"
        ]), "native_trace": trace,
        "device_playback": "not_tested", "database_access": False,
        "persistence_fixture": "in_memory_only_not_database_evidence",
    }
    (output / f"{case}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"case": case, "error_type": error, "connections": len(connections),
                      "close_reason": session.diagnostics()["close_reason"],
                      "responses": [e for e in phone.events if e["type"] in {"transcript.done", "response.done", "error"}]}), flush=True)
    return result


async def main(args, keys):
    args.output.mkdir(parents=True, exist_ok=False)
    results = []
    for case in args.cases.split(","):
        if case not in {"ordinary", "explanation", "long_input", "barge_in", "early_barge", "supersede_preparation"}:
            raise ValueError("Unknown session probe case")
        results.append(await probe(case=case, key=keys["gemini"], speech_dir=args.speech_dir, output=args.output))
    return 0 if results and all(result["passed"] for result in results) else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--speech-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", default="ordinary,explanation,barge_in,early_barge,supersede_preparation")
    args = parser.parse_args()
    raise SystemExit(asyncio.run(main(args, json.load(sys.stdin))))
