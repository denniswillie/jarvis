"""Opt-in native TTS -> common writer -> iOS playback -> durable receipt probe.

The expert answer is a synthetic fixture, NOT a native frontier reasoning run.
Only disposable manifested storage and loopback clients are admitted. The
foreground transport is inert; it cannot retell or synthesize the answer.
"""
from __future__ import annotations

import argparse
import asyncio
from contextlib import suppress
import json
from pathlib import Path
import time
from types import SimpleNamespace
from uuid import uuid4

from openai import AsyncOpenAI
from websockets.asyncio.server import serve

from app.attention.repository import AttentionRepository
from app.attention.tts import OpenAISpeechProvider, SpeechConfiguration
from app.realtime.events import encode_event
from app.realtime.models import RealtimeSessionContext
from app.realtime.session import RealtimeAudioSession
from app.runtime.feature_config import TaskLimits
from app.sessions.repository import SessionRepository
from app.settings import JarvisSettings
from app.tasks.context_capture import SessionContextCapture
from app.tasks.context_repository import ContextRepository
from app.tasks.hermes_result import HermesTaskResult
from app.tasks.models import TaskCreate
from app.tasks.repository import TaskRepository
from app.tasks.runner import TaskRunner
from evals.second_brain.source_fixture import source_fixture


class InertForeground:
    def __init__(self):
        self.events = []

    async def send_event(self, event):
        self.events.append(event["type"])
        return True

    async def close(self):
        pass


class ProbeContextRepository(ContextRepository):
    """Record timings/types only, never private payload or exception text."""
    def __init__(self, tasks):
        super().__init__(tasks)
        self.writes = []

    async def append(self, owner_id, event):
        started = time.perf_counter()
        outcome = "committed"
        try:
            return await super().append(owner_id, event)
        except BaseException as error:
            outcome = type(error).__name__
            raise
        finally:
            if len(self.writes) < 256:
                self.writes.append({"kind":event.kind, "outcome":outcome,
                                    "elapsed_ms":round((time.perf_counter()-started)*1000, 2)})


async def run(output: Path, *, timeout=120):
    output.mkdir(mode=0o700, parents=True, exist_ok=True)
    settings = JarvisSettings()
    result = {"ok":False, "native_tts":True, "native_frontier_reasoning":False,
              "physical_glasses":False, "human_hearing":"not_established"}
    try:
        async with source_fixture(settings.SUPABASE_DB_URL, output, test_name="ios-native-tts-delivery") as fixture:
            tasks = TaskRepository(fixture.pool, TaskLimits(lease_seconds=300))
            messages = SessionRepository(fixture.pool)
            session = await messages.create_session(fixture.owner, "Synthetic playback integration")
            task = await tasks.create(fixture.owner, TaskCreate(request_id=uuid4(), session_id=session.id,
                title="Synthetic playback check", requested_outcome="Only speak the synthetic test result."), model="fixture-expert")
            worker_claim = await tasks.claim_next("fixture-expert", ("frontier",))
            context = TaskRunner(tasks, {}).context(worker_claim)
            outcome = await HermesTaskResult(status="succeeded",
                answer="This is a synthetic playback integration check, not a claim about your room.",
                speech=[{"text":"This is a synthetic playback integration check."},
                        {"text":"It is not a claim about your room or your activities."}],
                evidence=[], completed_operation_refs=[], uncertainties=[]).verified(context, SimpleNamespace())
            await tasks.finish(worker_claim, outcome)
            inbox = AttentionRepository(tasks)
            done = asyncio.Event()
            admitted = False
            async with AsyncOpenAI(api_key=settings.OPENAI_API_KEY, max_retries=0) as client:
                speech = OpenAISpeechProvider(client, SpeechConfiguration(timeout_seconds=25, max_audio_seconds=30))

                async def handle(socket):
                    nonlocal admitted, result
                    if admitted or socket.request.headers.get("Authorization") != "Bearer fixture-token":
                        await socket.close(code=1008)
                        return
                    admitted = True
                    started = time.perf_counter()
                    delivery = None
                    incoming = None
                    audio_id = "fixture-audio-" + uuid4().hex
                    context_repository = ProbeContextRepository(tasks)
                    capture = SessionContextCapture(context_repository, owner_id=fixture.owner,
                        session_id=session.id, provider="openai", audio_session_id=audio_id)
                    adapter = SimpleNamespace(send_text=socket.send, close=socket.close)
                    voice = RealtimeAudioSession(websocket=adapter,
                        context=RealtimeSessionContext(audio_session_id=audio_id, owner_id=fixture.owner,
                            jarvis_session_id=session.id, client_kind="phone_audio", session=session),
                        settings=settings, repository=messages, tool_modules=(), context_providers=(), context_capture=capture)
                    foreground = InertForeground()
                    voice.openai = foreground
                    voice._realtime_context_ready.set()
                    try:
                        await voice._send_app_event("audio.provider.selected", provider="openai", protocol_version=1,
                                                    audio_session_id=audio_id, model="inert-foreground-fixture")
                        await voice._send_app_event("session.created", session_id=str(session.id), audio_session_id=audio_id,
                                                    playback_receipts={"protocol_version":1,"sample_rate":24000})
                        delivery = await voice.try_attention(inbox, speech, worker_id="ios-fixture", receipt_seconds=40)
                        if delivery is None:
                            raise RuntimeError("Fixture delivery was not admitted")
                        async with asyncio.timeout(timeout):
                            while True:
                                incoming = asyncio.create_task(socket.recv())
                                finished, _ = await asyncio.wait({incoming, delivery.task}, return_when=asyncio.FIRST_COMPLETED)
                                if incoming not in finished:
                                    raise RuntimeError("Delivery stopped before a completed playback receipt")
                                raw = incoming.result()
                                incoming = None
                                event = json.loads(raw)
                                if not event.get("type", "").startswith("playback."):
                                    raise RuntimeError("Fixture accepts playback receipts only")
                                await voice._handle_client_event(event)
                                if delivery._played.is_set():
                                    await delivery.task
                                    if not await capture.flush():
                                        raise RuntimeError("Canonical capture did not flush")
                                    detail = await inbox.detail(fixture.owner, delivery.claim.intent_id)
                                    async with fixture.pool.acquire() as connection:
                                        chunks = await connection.fetch("""select chunk_index,forwarded_samples,played_samples,
                                            first_bytes_ms,generated_at is not null as generated,
                                            playback_completed_at is not null as completed
                                            from jarvis_attention_speech_chunks where owner_id=$1 and attempt_id=$2 order by chunk_index""",
                                            fixture.owner, delivery.claim.attempt_id)
                                        boundaries = await connection.fetchval("""select count(*) from jarvis_context_events
                                            where owner_id=$1 and session_id=$2 and kind='played_output_boundary'""", fixture.owner, session.id)
                                    result |= {"state":detail["state"], "chunks":[dict(c) for c in chunks],
                                        "canonical_playback_boundaries":boundaries,
                                        "foreground_events":foreground.events,
                                        "ok":detail["state"] == "played" and len(chunks) == 2 and boundaries >= 2
                                            and all(c["completed"] and c["forwarded_samples"] == c["played_samples"] > 0 for c in chunks)
                                            and "response.create" not in foreground.events}
                                    await voice._app_sender.send_text(encode_event("fixture.result", **result))
                                    break
                    except Exception as error:
                        result["error_type"] = type(error).__name__
                        if delivery is not None:
                            result["delivery_reason"] = delivery.reason
                        result["capture"] = capture.diagnostics()
                        with suppress(Exception):
                            await voice._app_sender.send_text(encode_event("fixture.result", **result))
                    finally:
                        if incoming is not None and not incoming.done():
                            incoming.cancel()
                            await asyncio.gather(incoming, return_exceptions=True)
                        await voice.close("fixture_complete")
                        result["elapsed_ms"] = round((time.perf_counter()-started)*1000, 2)
                        result["context_writes"] = context_repository.writes
                        done.set()

                async with serve(handle, "127.0.0.1", 0, max_size=65536) as server:
                    print(json.dumps({"ready":True,"url":"ws://127.0.0.1:"+str(server.sockets[0].getsockname()[1])}), flush=True)
                    await asyncio.wait_for(done.wait(), timeout+60)
        result["fixture_cleanup"] = "completed"
    except Exception as error:
        result["error_type"] = type(error).__name__
    (output / "result.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result), flush=True)
    return result["ok"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-native-tts", action="store_true", required=True)
    arguments = parser.parse_args()
    raise SystemExit(0 if asyncio.run(run(arguments.output)) else 1)
