"""Opt-in simulator reset/reconnect probe using only manifested synthetic data.

Real Jarvis sessions, ordered writers, capture, source deletion and native iOS
client/player are exercised. The OpenAI network transport is synthetic: this
does not measure model reasoning, native TTS or human hearing.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
from contextlib import suppress
import json
from pathlib import Path
import re
from types import SimpleNamespace
from uuid import uuid4

from websockets.asyncio.server import serve

from app.assistant.context_providers import (
    CurrentTimeContextProvider, RecentMessagesContextProvider, SessionContextProvider,
)
from app.realtime.models import RealtimeSessionContext
from app.realtime.session import RealtimeAudioSession
from app.runtime.feature_config import TaskLimits
from app.sessions.repository import SessionRepository
from app.settings import JarvisSettings
from app.tasks.context_capture import SessionContextCapture
from app.tasks.context_repository import ContextRepository
from app.tasks.repository import TaskRepository
from evals.second_brain.source_fixture import source_fixture

REMOVABLE = "Synthetic secret: the fixture marker is CORAL-SEVEN."
INDEPENDENT = "Independent interrupted fixture answer."


class SyntheticProviderWire:
    def __init__(self):
        self.incoming = asyncio.Queue()
        self.instructions = []
        self.events = []
        self.closed = False

    async def send(self, raw):
        event = json.loads(raw)
        self.events.append(event["type"])
        if event["type"] == "session.update":
            self.instructions.append(event["session"]["instructions"])
            self.incoming.put_nowait({"type":"session.updated"})
        elif event["type"] == "input_audio_buffer.commit":
            self.incoming.put_nowait({"type":"input_audio_buffer.committed", "item_id":"unfinished-input"})
        elif event["type"] == "response.create":
            self.incoming.put_nowait({"type":"response.created",
                "response":{"id":"synthetic-response","metadata":event["response"].get("metadata", {})}})
            self.incoming.put_nowait({"type":"response.output_audio.delta", "response_id":"synthetic-response",
                "item_id":"synthetic-output", "delta":base64.b64encode(bytes(48_000)).decode()})
            self.incoming.put_nowait({"type":"response.output_audio_transcript.delta",
                "response_id":"synthetic-response", "item_id":"synthetic-output",
                "delta":"Synthetic source-dependent partial answer."})

    async def recv(self):
        return json.dumps(await self.incoming.get())

    async def close(self):
        self.closed = True


async def _delete_fixture_source(repository, fixture, payload_id):
    # Match canonical capture's owner mutation lane before lifecycle triggers
    # acquire payload/source and owner-journal row locks.
    async with repository.tasks.transaction(fixture.owner) as connection:
        await connection.execute("""update jarvis_context_payloads
                                    set availability='deleted',content=null where owner_id=$1 and id=$2""",
                                    fixture.owner, payload_id)


def _safe_failure(error):
    sqlstate = getattr(error, "sqlstate", None)
    return {"error_type":type(error).__name__, "sqlstate":sqlstate
        if isinstance(sqlstate, str) and re.fullmatch(r"[A-Z0-9]{5}", sqlstate) else None}


async def run(output: Path, *, timeout=300):
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    settings = JarvisSettings()
    result = {"ok":False, "native_client":True, "native_player":True, "native_model":False,
        "native_tts":False, "physical_glasses":False, "human_hearing":"not_established"}
    stage = "fixture_setup"
    try:
        async with source_fixture(settings.SUPABASE_DB_URL, output / "manifests",
                test_name="ios-source-context-reset") as fixture:
            messages = SessionRepository(fixture.pool)
            repository = ContextRepository(TaskRepository(fixture.pool, TaskLimits()))
            session = await messages.create_session(fixture.owner, "Synthetic context recovery")
            original = await fixture.publish(REMOVABLE)
            await messages.create_message(owner_id=fixture.owner, session_id=session.id,
                role="assistant", modality="audio", content=REMOVABLE, source_payload_id=original.payload_id)
            earlier = SessionContextCapture(repository, owner_id=fixture.owner,
                session_id=session.id, provider="openai", audio_session_id="fixture-earlier")
            earlier.observe_output({"type":"transcript.delta", "item_id":"independent-output",
                "response_id":"independent-response", "text":INDEPENDENT}, forwarded=True)
            await earlier.close()
            complete, serial = asyncio.Event(), asyncio.Lock()
            connection_count = 0
            checks = {}

            async def handle(socket):
                nonlocal connection_count, result, stage
                if (socket.request.headers.get("Authorization") != "Bearer fixture-token"
                        or socket.request.path != "/context-reset?session_id=" + str(session.id)):
                    await socket.close(code=1008)
                    return
                async with serial:
                    connection_count += 1
                    number = connection_count
                    if number > 2 or complete.is_set():
                        await socket.close(code=1008)
                        return
                    wire = SyntheticProviderWire()
                    ready, audio_sent = asyncio.Event(), asyncio.Event()
                    audio_id = "fixture-context-" + uuid4().hex
                    capture = SessionContextCapture(repository, owner_id=fixture.owner,
                        session_id=session.id, provider="openai", audio_session_id=audio_id)

                    async def send_text(raw):
                        await socket.send(raw)
                        event = json.loads(raw)
                        if event["type"] == "session.created":
                            ready.set()
                        elif event["type"] == "audio.output.delta":
                            audio_sent.set()

                    async def incoming():
                        async for raw in socket:
                            yield raw

                    voice = RealtimeAudioSession(
                        websocket=SimpleNamespace(send_text=send_text, close=socket.close, iter_text=incoming),
                        context=RealtimeSessionContext(audio_session_id=audio_id, owner_id=fixture.owner,
                            jarvis_session_id=session.id, client_kind="phone_audio", session=session),
                        settings=settings, repository=messages, tool_modules=(), context_providers=(
                            CurrentTimeContextProvider(), SessionContextProvider(),
                            RecentMessagesContextProvider(max_items=24)),
                        context_capture=capture)

                    async def connect_synthetic():
                        voice.openai._websocket = wire
                    voice.openai.connect = connect_synthetic
                    running = asyncio.create_task(voice.run())
                    try:
                        stage = f"connection_{number}_ready"
                        await asyncio.wait_for(ready.wait(), 20)
                        if number == 1:
                            checks["initial_context_contains_source"] = any(REMOVABLE in x for x in wire.instructions)
                            stage = "first_audio"
                            await asyncio.wait_for(audio_sent.wait(), 20)
                            stage = "source_delete_transaction"
                            await _delete_fixture_source(repository, fixture, original.payload_id)
                            stage = "reset_await"
                            await asyncio.wait_for(running, 12)
                            stage = "reset_ack_checks"
                            checks["reset_acknowledged"] = voice._context_reset_ack.is_set()
                            checks["old_provider_closed"] = wire.closed
                            stage = "reset_session_preservation"
                            checks["session_preserved_after_reset"] = (
                                await messages.get_session(fixture.owner, session.id)).status == "active"
                        else:
                            stage = "recovered_context"
                            rebuilt = voice._assistant_context
                            records = rebuilt.voice_continuity["records"]
                            checks["removed_text_absent"] = all(REMOVABLE not in x for x in wire.instructions)
                            checks["removed_source_not_rebound"] = original.payload_id not in rebuilt.source_payload_ids
                            checks["independent_interruption_preserved"] = any(
                                r["kind"] == "assistant_output"
                                and INDEPENDENT in json.dumps(r["content"])
                                and (r["content"] or {}).get("completion_state") == "interrupted"
                                and (r["content"] or {}).get("heard_transcript") is None for r in records)
                            checks["missing_input_is_explicit"] = any(
                                (r["content"] or {}).get("transcript_status") == "session_closed_without_transcript"
                                and (r["content"] or {}).get("text") is None for r in records)
                            checks["clear_ack_retained"] = any(
                                (r["content"] or {}).get("reason") == "source_context_rebuild"
                                and (r["content"] or {}).get("client_reported_audio_cleared") is True for r in records)
                            checks["conversation_preserved"] = (
                                await messages.get_session(fixture.owner, session.id)).status == "active"
                            result |= {"ok":all(checks.values()), "checks":checks, "connections":number}
                            await voice._send_app_event("fixture.result", **result)
                            await voice.close("fixture_complete")
                            await asyncio.wait_for(running, 5)
                            complete.set()
                    except Exception as error:
                        result |= {**_safe_failure(error), "stage":stage, "checks":checks}
                        with suppress(Exception):
                            await voice._send_app_event("fixture.result", **result)
                        complete.set()
                    finally:
                        await voice.close("fixture_complete")
                        if not running.done():
                            running.cancel()
                        await asyncio.gather(running, return_exceptions=True)

            async with serve(handle, "127.0.0.1", 0, max_size=131072) as server:
                address = ("ws://127.0.0.1:" + str(server.sockets[0].getsockname()[1])
                           + "/context-reset?session_id=" + str(session.id))
                stage = "await_simulator"
                print(json.dumps({"ready":True,"url":address}), flush=True)
                await asyncio.wait_for(complete.wait(), timeout)
        result["fixture_cleanup"] = "completed"
    except Exception as error:
        result |= {"ok":False, **_safe_failure(error), "stage":stage}
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    return result["ok"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-isolated-database", action="store_true", required=True)
    args = parser.parse_args()
    raise SystemExit(0 if asyncio.run(run(args.output)) else 1)
