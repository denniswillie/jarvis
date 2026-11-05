"""Paid ASGI-router acceptance with real providers and disposable DB owners.

No application lifespan is started: shared background queues and external
mutation transports are deliberately excluded. Only the phone/ASGI transport is
simulated. Provider constructors, routing, auth, sessions, tools, and storage are
production implementations. Credentials arrive on stdin, never in reports.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
from contextlib import ExitStack, suppress
import json
import logging
from pathlib import Path
import sys
import time
from urllib.parse import urlencode
from uuid import uuid4
import wave
from unittest.mock import AsyncMock, patch

from fastapi import FastAPI

from app.assistant.context_providers import CurrentTimeContextProvider, RecentMessagesContextProvider, SessionContextProvider
from app.experience.repository import ExperienceRepository
from app.experience.segmenter import EventSegmentationPolicy, ExperienceEventSegmenter
from app.experience.service import ExperienceEventService
from app.perception.frame_window import RecentVisualFrameWindow
from app.realtime.gemini.client import GeminiLiveConnection, GeminiTransportError
from app.realtime.router import _create_gemini_session, router
from app.realtime.session import RealtimeAudioSession
from app.realtime.session_manager import RealtimeSessionManager
from app.runtime import JarvisRuntimeServices
from app.settings import JarvisSettings
from app.tools.module import FunctionToolModule
from evals.gemini_live.interruption import DatabaseFixtures
from evals.gemini_live.session_probe import ProbePhone
from evals.gemini_live.transport_probe import read_pcm


class RoutedPhone(ProbePhone):
    """ASGI phone peer; Starlette still owns the actual server WebSocket."""

    def __init__(self):
        super().__init__()
        self.connected = False
        self.accepted = False
        self.close_code = None

    async def asgi_receive(self):
        if not self.connected:
            self.connected = True
            return {"type": "websocket.connect"}
        event = await self.input.get()
        if event is None:
            return {"type": "websocket.disconnect", "code": 1000}
        return {"type": "websocket.receive", "text": json.dumps(event)}

    async def asgi_send(self, event):
        if event["type"] == "websocket.accept":
            self.accepted = True
        elif event["type"] == "websocket.send":
            await self.send_json(json.loads(event.get("text") or event["bytes"]))
        elif event["type"] == "websocket.close":
            self.close_code = event.get("code", 1000)
            await self.close()

    async def disconnect(self):
        await self.input.put(None)


def completed_spoken_answer(events):
    first_audio = next((i for i, event in enumerate(events)
                        if event["type"] == "audio.output.delta"), None)
    return first_audio is not None and any(
        event["type"] == "response.done" for event in events[first_audio + 1:]
    )


def completed_after_resume(events):
    return any(
        event.get("type") == "audio.provider.reconnected"
        and event.get("provider") == "gemini" and event.get("resumed") is True
        and completed_spoken_answer(events[index + 1:])
        for index, event in enumerate(events)
    )


def successful_execution(snapshot, name, *, memory_id=None):
    for execution in snapshot["executions"]:
        result = execution["result"]
        if isinstance(result, str):
            result = json.loads(result)
        arguments = execution["arguments"]
        if isinstance(arguments, str):
            arguments = json.loads(arguments)
        if (execution["tool_name"] == name and isinstance(result, dict)
                and result.get("status") == "success"
                and (memory_id is None or arguments.get("memory_id") == str(memory_id))):
            return True
    return False


def public_event(event):
    return {key: value for key, value in event.items() if key != "audio_base64"}


async def routed_case(*, app, settings, owner, session_id, provider, label,
                      speech, output, expected_error=None, missing_bearer=False,
                      fail_gemini_connect=False, soak_seconds=0):
    phone, constructed = RoutedPhone(), []
    recovery_events = []
    native_receive = GeminiLiveConnection.receive_event

    async def observe_recovery(connection):
        event = await native_receive(connection)
        if "goAway" in event:
            recovery_events.append({"kind": "goAway", "observed_at": time.monotonic(),
                                    "time_left": event["goAway"].get("timeLeft")})
        # Resumption handles and unrelated provider media never enter this log.
        return event
    selected_settings = settings.model_copy(update={"JARVIS_OWNER_ID": owner})
    params = [("session_id", str(session_id)), ("client_kind", "phone_audio")]
    if provider is not None:
        params.append(("audio_provider", provider))
    headers = [] if missing_bearer else [
        (b"authorization", ("Bearer " + settings.JARVIS_STATIC_BEARER_TOKEN).encode())
    ]
    scope = {"type": "websocket", "asgi": {"version": "3.0", "spec_version": "2.4"},
             "http_version": "1.1", "scheme": "ws", "path": "/realtime/audio/ws",
             "raw_path": b"/realtime/audio/ws", "root_path": "",
             "query_string": urlencode(params).encode(), "headers": headers,
             "server": ("isolated-fixture", 80), "client": ("isolated-phone", 1), "state": {}}

    def observed(factory):
        def construct(**kwargs):
            session = factory(**kwargs)
            constructed.append(session)
            return session
        return construct

    result = {"case": label, "provider": provider or "legacy_openai", "passed": False,
              "expected_error": expected_error, "simulated_phone": True,
              "controlled_connect_failure": fail_gemini_connect}
    running = None
    completed_turns = 0
    with ExitStack() as stack:
        stack.enter_context(patch("app.realtime.router.get_settings", return_value=selected_settings))
        stack.enter_context(patch("app.realtime.router._create_gemini_session", observed(_create_gemini_session)))
        stack.enter_context(patch("app.realtime.router.RealtimeAudioSession", observed(RealtimeAudioSession)))
        if soak_seconds:
            stack.enter_context(patch.object(GeminiLiveConnection, "receive_event", observe_recovery))
        if fail_gemini_connect:
            stack.enter_context(patch.object(
                GeminiLiveConnection, "connect", AsyncMock(side_effect=GeminiTransportError("gemini_connect_failed"))))
        try:
            running = asyncio.create_task(app(scope, phone.asgi_receive, phone.asgi_send))
            async with asyncio.timeout(125 + soak_seconds):
                if expected_error is not None:
                    await running
                else:
                    await phone.wait_for("session.created")
                    pcm = read_pcm(speech)
                    run_started = time.monotonic()
                    while True:
                        event_start = len(phone.events)
                        segment = f"{label}-{completed_turns}"
                        started = time.monotonic()
                        for offset in range(0, len(pcm), 4800):
                            chunk = pcm[offset:offset + 4800]
                            await phone.send("audio.input.append", segment_id=segment,
                                             audio_base64=base64.b64encode(chunk).decode())
                            await asyncio.sleep(max(0, started + (offset + len(chunk)) / 48000 - time.monotonic()))
                        await phone.send("audio.input.commit", segment_id=segment)
                        await phone.send("response.create")
                        async with phone.changed:
                            await phone.changed.wait_for(lambda: phone.closed or completed_spoken_answer(phone.events[event_start:]))
                        if not completed_spoken_answer(phone.events[event_start:]):
                            raise RuntimeError("No completed spoken answer")
                        completed_turns += 1
                        remaining = soak_seconds - (time.monotonic() - run_started)
                        if remaining <= 0:
                            break
                        await asyncio.sleep(min(45, remaining))
                    if soak_seconds:
                        diagnostics = constructed[0].diagnostics()
                        result["natural_turnover"] = {
                            "elapsed_seconds": time.monotonic() - run_started,
                            "completed_turns": completed_turns,
                            "provider": diagnostics["provider"], "model": diagnostics["model"],
                            "connection_generation": diagnostics["connection_generation"],
                            "recovery": diagnostics["recovery"],
                            "goaway_events": recovery_events,
                            "completed_speech_after_resume": completed_after_resume(phone.events),
                            "controlled_disconnects": 0,
                        }
                    await phone.disconnect()
                    await running
            errors = [event.get("code") for event in phone.events if event["type"] == "error"]
            kinds = [event["type"] for event in phone.events]
            acknowledgements = [event for event in phone.events if event["type"] == "audio.provider.selected"]
            created = [event for event in phone.events if event["type"] == "session.created"]
            result.update(error_codes=errors, constructed_sessions=len(constructed),
                          acknowledgements=acknowledgements, created_count=len(created))
            if expected_error is not None:
                result["passed"] = errors == [expected_error] and not created and not phone.audio
                if not fail_gemini_connect:
                    result["passed"] = result["passed"] and not constructed and not acknowledgements
            else:
                if provider is None:
                    routing_matches = not acknowledgements
                else:
                    routing_matches = (
                        len(acknowledgements) == 1 and len(created) == 1
                        and acknowledgements[0]["provider"] == provider
                        and acknowledgements[0]["audio_session_id"] == created[0]["audio_session_id"]
                        and kinds.index("audio.provider.selected") < kinds.index("session.created"))
                result["passed"] = not errors and routing_matches and completed_spoken_answer(phone.events)
                if soak_seconds:
                    result["passed"] = result["passed"] and bool(recovery_events) and (
                        result["natural_turnover"]["recovery"]["resumptions"] >= 1
                        and completed_after_resume(phone.events))
        except Exception as error:
            result["error_type"] = type(error).__name__
        finally:
            for session in constructed:
                with suppress(Exception):
                    await asyncio.wait_for(session.close("routing_probe_finished"), 20)
            if running is not None and not running.done():
                running.cancel()
            if running is not None:
                with suppress(asyncio.CancelledError, Exception):
                    await running
    diagnostics = await app.state.runtime_services.realtime_session_manager.get_session_diagnostics(session_id)
    result["manager_unregistered"] = not diagnostics["realtime_connected"]
    result["passed"] = result["passed"] and result["manager_unregistered"]
    result["events"] = [public_event(event) for event in phone.events]
    result["audio_files"] = []
    for index, (item_id, pcm) in enumerate(phone.audio.items()):
        name = f"{label}-{index}.wav"
        with wave.open(str(output / name), "wb") as audio:
            audio.setnchannels(1)
            audio.setsampwidth(2)
            audio.setframerate(24000)
            audio.writeframes(pcm)
        result["audio_files"].append({"file": name, "item_id": item_id, "pcm_bytes": len(pcm)})
    (output / f"{label}.json").write_text(json.dumps(result, indent=2, default=str) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key not in {"events", "audio_files"}}), flush=True)
    return result


async def evaluate(args, keys):
    args.output.mkdir(parents=True, exist_ok=False)
    fixtures = DatabaseFixtures(keys["database"], args.output)
    results, runtime = [], None
    report = {"scope": "real ASGI router and providers, isolated domain capabilities and DB owners",
              "physical_glasses_tested": False, "production_deployed": False, "passed": False}
    try:
        await fixtures.connect()
        owner, session_id = await fixtures.create("routed-provider-rollback")
        foreign_owner, closed_session = await fixtures.create("routed-security-control")
        async with fixtures.pool.acquire() as connection:
            await connection.execute("update jarvis_sessions set status='closed' where owner_id=$1 and id=$2",
                                     foreign_owner, closed_session)
        settings = JarvisSettings(
            _env_file=None, **keys.get("settings", {}), OPENAI_API_KEY=keys["openai"],
            GEMINI_API_KEY=keys["gemini"], GEMINI_LIVE_ENABLED=True,
            JARVIS_STATIC_BEARER_TOKEN="fixture-" + uuid4().hex, JARVIS_OWNER_ID=owner)
        experience = ExperienceEventService(
            repository=ExperienceRepository(fixtures.pool),
            segmenter=ExperienceEventSegmenter(EventSegmentationPolicy(300, 900)))
        runtime = JarvisRuntimeServices(
            settings=settings, realtime_session_manager=RealtimeSessionManager(),
            sensory_frame_window=RecentVisualFrameWindow(max_frames_per_session=4, ttl_seconds=5),
            # No lifespan/background services are started by this isolated probe.
            latest_visual_frame_buffer=None, domain_event_bus=None, model_limiter=None,
            db_pool=fixtures.pool, experience_event_service=experience,
            tool_modules=(FunctionToolModule("fixture_memory", lambda context: fixtures.registry(
                context.owner_id, context.session_id, names=("remember", "explain_memory")).list()),),
            assistant_context_providers=(CurrentTimeContextProvider(), SessionContextProvider(), RecentMessagesContextProvider(12)))
        app = FastAPI()
        app.include_router(router)
        app.state.runtime_services = runtime

        async def run(label, provider="gemini", **overrides):
            fields = dict(app=app, settings=settings, owner=owner, session_id=session_id,
                          provider=provider, label=label, speech=args.speech_dir / "simple-24k.wav", output=args.output)
            fields.update(overrides)
            result = await routed_case(**fields)
            results.append(result)
            return result

        await run("missing-bearer", missing_bearer=True, expected_error="jarvis_auth_required")
        await run("wrong-owner", owner=foreign_owner, expected_error="session_not_found")
        await run("closed-session", owner=foreign_owner, session_id=closed_session, expected_error="session_closed")
        settings.GEMINI_LIVE_ENABLED = False
        await run("disabled-gemini", expected_error="audio_provider_unavailable")
        settings.GEMINI_LIVE_ENABLED = True
        await run("failed-gemini-startup", fail_gemini_connect=True, expected_error="gemini_connect_failed")
        await run("legacy-gpt-after-failure", provider=None)
        if args.soak_seconds:
            await run("gemini-natural-turnover", soak_seconds=args.soak_seconds)
        await run("gemini-save", speech=args.speech_dir / "save.wav")
        snapshot = await fixtures.snapshot(owner, session_id)
        (args.output / "after-gemini-save.json").write_text(json.dumps(snapshot, indent=2, default=str) + "\n")
        saved = len(snapshot["memories"]) == 1 and successful_execution(snapshot, "remember")
        report["gemini_save_verified"] = saved
        if saved:
            memory_id = snapshot["memories"][0]["id"]
            # Only the record identifier is supplied, not its content or a fake
            # retrieval result. The following GPT run must use the real handler.
            async with fixtures.pool.acquire() as connection:
                await connection.execute(
                    "update jarvis_sessions set behavior_instructions=$3 where owner_id=$1 and id=$2",
                    owner, session_id, "The stored test memory identifier is " + str(memory_id) +
                    ". When asked to check it, use explain_memory with that memory_id. Do not save it again.")
            settings.GEMINI_LIVE_ENABLED = False
            await run("disabled-after-gemini", expected_error="audio_provider_unavailable")
            await run("gpt-rollback-memory-read", provider="openai", speech=args.speech_dir / "retrieve.wav")
            after = await fixtures.snapshot(owner, session_id)
            (args.output / "after-gpt-rollback.json").write_text(json.dumps(after, indent=2, default=str) + "\n")
            report["gpt_read_same_memory"] = successful_execution(after, "explain_memory", memory_id=memory_id)
            report["memory_unchanged_after_rollback"] = after["memories"] == snapshot["memories"]
        else:
            report["rollback_skipped_reason"] = "No verified successful Gemini save; not inventing a memory fixture."
        report["passed"] = all(result["passed"] for result in results) and all(report.get(key) for key in (
            "gemini_save_verified", "gpt_read_same_memory", "memory_unchanged_after_rollback"))
    except Exception as error:
        report["error_type"] = type(error).__name__
    finally:
        try:
            if runtime is not None:
                await runtime.realtime_session_manager.close_all()
                await runtime.experience_event_service.close()
        except Exception as error:
            report["resource_close_error_type"] = type(error).__name__
            report["passed"] = False
        finally:
            # A resource-close failure must not bypass database cleanup.
            try:
                await fixtures.cleanup()
            except Exception as error:
                report["cleanup_error_type"] = type(error).__name__
                report["passed"] = False
        report["cleanup"] = fixtures.cleanup_result
        report["cleanup_confirmed"] = bool(fixtures.cleanup_result is not None and
                                          fixtures.cleanup_result["remaining_rows"] == 0)
        report["passed"] = report["passed"] and report["cleanup_confirmed"]
        report["results"] = [{key: value for key, value in result.items() if key != "events"} for result in results]
        (args.output / "report.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--speech-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--soak-seconds", type=int, default=0,
                        help="Optional natural-turnover observation window, 600 to 900 seconds")
    args = parser.parse_args()
    if args.soak_seconds and not 600 <= args.soak_seconds <= 900:
        parser.error("--soak-seconds must be zero or between 600 and 900")
    logging.disable(logging.CRITICAL)
    report = asyncio.run(evaluate(args, json.load(sys.stdin)))
    print(json.dumps({key: value for key, value in report.items() if key != "results"}), flush=True)
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
