"""Paid matched probes through both production audio-session coordinators.

The phone sink and camera source are controlled fixtures. Provider transports,
session coordination, image tools, messages, and experience attachment are real.
Transport success is not an OCR grade or evidence of physical device playback.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
from collections import defaultdict
from contextlib import suppress
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from io import BytesIO
import inspect
import json
import logging
from pathlib import Path
import statistics
import sys
import time
from unittest.mock import patch
from uuid import uuid4

from PIL import Image

from app.assistant.context_providers import CurrentTimeContextProvider, RecentMessagesContextProvider, SessionContextProvider
from app.experience.repository import ExperienceRepository, ToolExecutionRepository
from app.experience.segmenter import EventSegmentationPolicy, ExperienceEventSegmenter
from app.experience.service import ExperienceEventService
from app.perception.current_image import CurrentImageService
from app.perception.frame_window import RecentVisualFrameWindow, SensoryVisualFrame
from app.perception.tools import CurrentImageToolModule
from app.realtime.gemini.client import GeminiLiveConnection
from app.realtime.gemini.events import decode_pcm16
from app.realtime.gemini.session import GeminiAudioSession
from app.realtime.models import RealtimeSessionContext
from app.realtime.openai_client import OpenAIRealtimeConnection
from app.realtime.output_policy import RealtimeOutputItemState
from app.realtime.session import RealtimeAudioSession
from app.sessions.repository import SessionRepository
from app.settings import JarvisSettings
from evals.gemini_live.interruption import DatabaseFixtures
from evals.gemini_live.session_probe import ProbePhone
from evals.gemini_live.transport_probe import read_pcm, redact_native, write_wav
from evals.perception_memory.fixture_cleanup import cleanup_fixture_owners


class PairedPhone(ProbePhone):
    async def send_text(self, payload):
        await self.send_json(json.loads(payload))


class ExpectedOpenAIAudio:
    """Apply the existing phase policy independently of the observed phone sink."""
    def __init__(self):
        self.items = {}

    def observe_item(self, item):
        if isinstance(item, dict) and isinstance(item.get("id"), str):
            self.items.setdefault(item["id"], RealtimeOutputItemState(item_id=item["id"])).observe_phase(item.get("phase"))

    def admits_chunk(self, item_id):
        state = self.items.setdefault(item_id, RealtimeOutputItemState(item_id=item_id))
        allowed = state.record_generated()
        if allowed:
            state.record_forwarded(audio_chunk=True)
        return allowed


async def observed_openai_write(write, event, observe):
    """Preserve lazy resolution inside the real writer and its boolean result."""
    payload = None

    async def resolve():
        nonlocal payload
        payload = event() if callable(event) else event
        if inspect.isawaitable(payload):
            payload = await payload
        return payload

    sent = await write(resolve)
    if sent and payload is not None and payload.get("type") != "input_audio_buffer.append":
        observe("sent", payload)
    return sent


def safe_event(event: dict) -> dict:
    """Keep evidence useful without copying PCM, image bytes, or resume handles."""
    if event.get("type") == "session.update":
        return {"type": "session.update", "sha256": sha256(
            json.dumps(event, sort_keys=True).encode()).hexdigest()}
    result = redact_native(event)
    for key in ("audio", "audio_base64"):
        if key in result:
            result[key] = "<redacted>"
    if event.get("type") in {"response.output_audio.delta", "response.audio.delta"}:
        result["delta"] = "<redacted>"
    # OpenAI input images are data URLs, unlike Gemini's inlineData.data.
    def images(value):
        if isinstance(value, dict):
            return {key: "<redacted>" if key == "image_url" else images(item)
                    for key, item in value.items()}
        if isinstance(value, list):
            return [images(item) for item in value]
        return value
    return images(result)


def latency_summary(results: list[dict]) -> dict:
    summary = {}
    for provider in ("openai", "gemini"):
        selected = [r for r in results if r["provider"] == provider]
        measured = [r["latency"]["input_end_to_app_audio_seconds"] for r in selected
                    if r.get("latency", {}).get("input_end_to_app_audio_seconds") is not None]
        ordered = sorted(measured)
        summary[provider] = {
            "attempts": len(selected), "transport_failures": sum(not r["transport_passed"] for r in selected),
            "measured_samples": len(measured), "missing_samples": len(selected) - len(measured),
            "median_seconds": statistics.median(measured) if measured else None,
            "max_seconds": max(measured) if measured else None,
            "p95_nearest_rank_seconds": ordered[max(0, (95 * len(ordered) + 99) // 100 - 1)] if ordered else None,
        }
    return summary


def pcm_evidence(native_audio, forwarded: bytes, completed: set[str], interrupted: set[str], *, eligible_audio=None) -> dict:
    native = b"".join(bytes(value) for value in native_audio.values())
    eligible = native if eligible_audio is None else b"".join(bytes(value) for value in eligible_audio.values())
    return {"native_bytes": len(native), "forwarded_bytes": len(forwarded),
            "native_sha256": sha256(native).hexdigest(), "forwarded_sha256": sha256(forwarded).hexdigest(),
            "eligible_native_bytes": len(eligible), "eligible_native_sha256": sha256(eligible).hexdigest(),
            "suppressed_native_bytes": len(native) - len(eligible),
            "exact_match": bool(eligible) and eligible == forwarded,
            "native_completed": bool(native_audio) and set(native_audio) <= completed and not interrupted,
            "native_audio_responses": len(native_audio)}


async def case(*, provider, label, fixtures, keys, speech, image_path, automatic, output):
    result = {"case": label, "provider": provider, "transport_passed": False,
              "answer_quality": "requires_source_and_independent_audio_review",
              "device_playback": "not_tested", "turn_mode": "application_commit_create",
              "image_delivery": "current_view" if automatic else "requested_tool"}
    owner, session_id = await fixtures.create("paired-" + label)
    phone = PairedPhone()
    repository = SessionRepository(fixtures.pool)
    experience = ExperienceEventService(repository=ExperienceRepository(fixtures.pool),
        segmenter=ExperienceEventSegmenter(EventSegmentationPolicy(300, 900)))
    session = running = camera = lease = None
    window = RecentVisualFrameWindow(max_frames_per_session=4, ttl_seconds=10)
    native_audio = defaultdict(bytearray)
    eligible_audio = defaultdict(bytearray)
    expected_openai = ExpectedOpenAIAudio()
    native_completed, native_interrupted = set(), set()
    trace, times = [], {}
    started = time.monotonic()
    native_turn = 1

    def observe(direction, event):
        nonlocal native_turn
        now = time.monotonic()
        trace.append({"direction": direction, "seconds": now - started, "event": safe_event(event)})
        if direction == "sent":
            if "activityEnd" in event.get("realtimeInput", {}) or event.get("type") == "response.create":
                times.setdefault("provider_release", now)
            return
        if provider == "openai":
            if event.get("type") in {"response.output_item.added", "response.output_item.created",
                                     "response.output_item.done", "response.output_item.completed"}:
                expected_openai.observe_item(event.get("item"))
            if event.get("type") in {"response.output_audio.delta", "response.audio.delta"}:
                pcm = decode_pcm16(event.get("delta"), max_bytes=8 * 1024 * 1024)
                native_audio[event["response_id"]].extend(pcm)
                times.setdefault("provider_audio", now)
                if expected_openai.admits_chunk(event.get("item_id")):
                    eligible_audio[event["response_id"]].extend(pcm)
                    times.setdefault("eligible_provider_audio", now)
            if event.get("type") == "response.done":
                response = event.get("response", {})
                if response.get("status") == "completed":
                    native_completed.add(response.get("id"))
                else:
                    native_interrupted.add(response.get("id"))
        else:
            content = event.get("serverContent", {})
            for part in content.get("modelTurn", {}).get("parts", []):
                inline = part.get("inlineData", {})
                if inline.get("mimeType", "").startswith("audio/"):
                    pcm = decode_pcm16(inline.get("data"), max_bytes=8 * 1024 * 1024)
                    native_audio[str(native_turn)].extend(pcm)
                    eligible_audio[str(native_turn)].extend(pcm)
                    times.setdefault("provider_audio", now)
                    times.setdefault("eligible_provider_audio", now)
            if content.get("turnComplete"):
                native_completed.add(str(native_turn))
                native_turn += 1
            if content.get("interrupted"):
                native_interrupted.add("interrupted")

    class TracedGemini(GeminiLiveConnection):
        async def send_event(self, event):
            await super().send_event(event)
            if "audio" not in event.get("realtimeInput", {}):
                observe("sent", event)

        async def receive_event(self):
            event = await super().receive_event()
            observe("received", event)
            return event

    class TracedOpenAI(OpenAIRealtimeConnection):
        async def send_event(self, event):
            return await observed_openai_write(super().send_event, event, observe)

        async def send_visual_event(self, payload_factory, *, timeout_seconds):
            payload = None

            def resolve():
                nonlocal payload
                payload = payload_factory()
                return payload

            sent = await super().send_visual_event(resolve, timeout_seconds=timeout_seconds)
            if sent and payload is not None:
                observe("sent", payload)
            return sent

        async def receive_event(self):
            event = await super().receive_event()
            observe("received", event)
            return event

    original_send = phone.send_json

    async def observed_phone(event):
        if event["type"] == "audio.output.delta":
            times.setdefault("app_audio", time.monotonic())
        await original_send(event)

    phone.send_json = observed_phone

    async def until(predicate):
        while not predicate():
            if phone.closed or running.done():
                raise RuntimeError("Production session ended before the expected answer")
            await asyncio.sleep(.01)

    try:
        async with asyncio.timeout(100):
            settings = JarvisSettings(_env_file=None, **{
                **keys.get("settings", {}), "GEMINI_API_KEY": keys["gemini"],
                "OPENAI_API_KEY": keys["openai"], "OPENAI_REALTIME_MODEL": keys["openai_model"],
                "VISUAL_REALTIME_ENABLED": automatic,
            })
            result["model"] = settings.OPENAI_REALTIME_MODEL if provider == "openai" else settings.GEMINI_LIVE_MODEL
            modules = ()
            if image_path is not None:
                jpeg = image_path.read_bytes()
                with Image.open(BytesIO(jpeg)) as source:
                    width, height = source.size
                    if source.format != "JPEG":
                        raise ValueError("The paired image fixture must be an original JPEG")
                result["image"] = {"file": image_path.name, "sha256": sha256(jpeg).hexdigest(),
                                   "width": width, "height": height}
                current = CurrentImageService(frame_window=window,
                    max_age_seconds=settings.VISUAL_REALTIME_MAX_CONTEXT_AGE_SECONDS,
                    max_image_bytes=settings.VISUAL_FRAME_MAX_BYTES, max_width=1024, max_height=1024)
                modules = (CurrentImageToolModule(current),)
                lease = window.begin_stream(owner, session_id, uuid4())

                async def frames():
                    sequence = 0
                    while True:
                        sequence += 1
                        now = datetime.now(UTC)
                        window.put(SensoryVisualFrame(
                            frame_id=uuid4(), owner_id=owner, session_id=session_id,
                            audio_session_id=session.context.audio_session_id,
                            visual_session_id=lease.visual_session_id, frame_sequence=sequence,
                            source="glasses", captured_at=now, server_received_at=now,
                            content_type="image/jpeg", width=width, height=height,
                            storage_bucket="paired-fixture", storage_path=image_path.name,
                            raw_image_expires_at=now + timedelta(minutes=1), image_bytes=jpeg,
                        ), stream_lease=lease)
                        await asyncio.sleep(.5)

            context = RealtimeSessionContext(audio_session_id=str(uuid4()), owner_id=owner,
                jarvis_session_id=session_id, client_kind="phone_audio",
                session=await repository.get_session(owner, session_id))
            kwargs = dict(websocket=phone, context=context, settings=settings, repository=repository,
                tool_modules=modules, context_providers=(CurrentTimeContextProvider(), SessionContextProvider(),
                    RecentMessagesContextProvider(12)), sensory_frame_window=window if image_path else None,
                tool_execution_repository=ToolExecutionRepository(fixtures.pool), experience_event_service=experience)
            if provider == "gemini":
                session = GeminiAudioSession(**kwargs, connection_factory=TracedGemini)
            else:
                # Replace only the transport constructor synchronously, with a
                # tracing subclass that delegates every operation unchanged.
                with patch("app.realtime.session.OpenAIRealtimeConnection", TracedOpenAI):
                    session = RealtimeAudioSession(**kwargs)
            running = asyncio.create_task(session.run())
            if image_path is not None:
                camera = asyncio.create_task(frames())
            await phone.wait_for("session.created")
            pcm = read_pcm(speech)
            result["input"] = {"sha256": sha256(pcm).hexdigest(), "bytes": len(pcm), "sample_rate": 24000}
            for offset in range(0, len(pcm), 4800):
                chunk = pcm[offset:offset + 4800]
                await phone.send("audio.input.append", segment_id="paired", audio_base64=base64.b64encode(chunk).decode())
                await asyncio.sleep(len(chunk) / 48000)
            times["input_end"] = time.monotonic()
            await phone.send("audio.input.commit", segment_id="paired")
            await phone.send("response.create")

            def answered():
                audible = {event.get("response_id") for event in phone.events if event["type"] == "audio.output.delta"}
                return any(event["type"] == "response.done" and event.get("response_id") in audible
                           for event in phone.events)

            await until(answered)
            await phone.wait_for("user.transcript.done")
            await asyncio.sleep(.5)
            forwarded = b"".join(bytes(value) for value in phone.audio.values())
            result["pcm"] = pcm_evidence(native_audio, forwarded, native_completed, native_interrupted,
                                         eligible_audio=eligible_audio)
            result["pcm"]["eligibility_policy"] = "existing_GPT_message_phase_policy" if provider == "openai" else "all_native_turn_audio"
            result["transport_passed"] = result["pcm"]["exact_match"] and result["pcm"]["native_completed"] and not any(
                event["type"] == "error" for event in phone.events)
    except Exception as error:
        result["error_type"] = type(error).__name__
    finally:
        if camera is not None:
            camera.cancel()
            await asyncio.gather(camera, return_exceptions=True)
        if lease is not None:
            window.end_stream(lease)
        if session is not None:
            await session.close("paired_probe_finished")
        if running is not None:
            with suppress(asyncio.CancelledError):
                await running
        await experience.close()
        if provider == "gemini" and session is not None:
            result["diagnostics"] = session.diagnostics()
            if (session.journal.diagnostics()["degraded"] or session.diagnostics()["domain_tasks"]
                    or session.diagnostics()["owned_tasks"]):
                result["transport_passed"] = False
        result["answers"] = [event["text"] for event in phone.events if event["type"] == "transcript.done"]
        result["app_events"] = [safe_event(event) for event in phone.events if event["type"] != "audio.output.delta"]
        result["trace"] = trace
        result["latency"] = {
            "clock": "single_runner_monotonic_not_device_clock",
            "input_end_to_app_audio_seconds": times.get("app_audio", 0) - times["input_end"]
                if "app_audio" in times and "input_end" in times else None,
            "input_end_to_provider_audio_seconds": times.get("provider_audio", 0) - times["input_end"]
                if "provider_audio" in times and "input_end" in times else None,
            "input_end_to_provider_release_seconds": times.get("provider_release", 0) - times["input_end"]
                if "provider_release" in times and "input_end" in times else None,
            "input_end_to_eligible_provider_audio_seconds": times["eligible_provider_audio"] - times["input_end"]
                if "eligible_provider_audio" in times and "input_end" in times else None,
            "eligible_provider_audio_to_app_audio_seconds": times["app_audio"] - times["eligible_provider_audio"]
                if "app_audio" in times and "eligible_provider_audio" in times else None,
        }
        for index, pcm in enumerate(native_audio.values(), 1):
            write_wav(output / f"{label}-native-{index}.wav", bytes(pcm))
        for index, pcm in enumerate(eligible_audio.values(), 1):
            write_wav(output / f"{label}-eligible-{index}.wav", bytes(pcm))
        for index, pcm in enumerate(phone.audio.values(), 1):
            write_wav(output / f"{label}-app-{index}.wav", bytes(pcm))
        result["snapshot"] = await fixtures.snapshot(owner, session_id)
        if result["snapshot"]["memories"]:
            result["transport_passed"] = False
        result["cleanup"] = await cleanup_fixture_owners(keys["database"], {owner})
        (output / f"{label}.json").write_text(json.dumps(result, indent=2, default=str) + "\n")
    print(json.dumps({key: result.get(key) for key in ("case", "provider", "transport_passed", "error_type",
        "answers", "pcm", "latency")} | {"cleanup_remaining": result["cleanup"]["remaining_rows"]}), flush=True)
    return result


async def run(args, keys):
    groups = set(args.groups.split(","))
    if not groups or groups - {"all", "audio", "vision"} or not 1 <= args.repeats <= 5:
        raise ValueError("Unknown paired group or repeat count")
    args.output.mkdir(parents=True, exist_ok=False)
    fixtures = DatabaseFixtures(keys["database"], args.output)
    results = []
    scenarios = []
    if groups & {"all", "audio"}:
        scenarios.append(("ordinary", args.speech_dir / "simple-24k.wav", None, False))
    if groups & {"all", "vision"}:
        scenarios.extend((f"image-{index}-{mode}", args.speech_dir / "code-24k.wav", image, automatic)
            for index, image in enumerate(args.images, 1) for mode, automatic in (("current", True), ("tool", False)))
    try:
        await fixtures.connect()
        for iteration in range(1, args.repeats + 1):
            # Counterbalance ordering; GPT is exercised after Gemini as well as before.
            providers = ("openai", "gemini") if iteration % 2 else ("gemini", "openai")
            for scenario, speech, image, automatic in scenarios:
                for provider in providers:
                    label = f"{scenario}-{iteration}-{provider}"
                    results.append(await case(provider=provider, label=label, fixtures=fixtures, keys=keys,
                        speech=speech, image_path=image, automatic=automatic, output=args.output))
    finally:
        await fixtures.cleanup()
        compact = [{key: result.get(key) for key in ("case", "provider", "model", "transport_passed",
            "error_type", "input", "image", "answers", "pcm", "latency", "answer_quality")} for result in results]
        (args.output / "matrix.json").write_text(json.dumps({
            "scope": "production adapters with isolated owners; not deployed routing or physical playback",
            "results": compact, "cleanup": fixtures.cleanup_result,
            "latency": latency_summary(results), "quality_acceptance": "pending_independent_review",
        }, indent=2) + "\n")
    return 0 if len(results) == len(scenarios) * 2 * args.repeats and all(r["transport_passed"] for r in results) else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keys-stdin", action="store_true", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--speech-dir", type=Path, required=True)
    parser.add_argument("--images", type=Path, nargs="+", required=True)
    parser.add_argument("--groups", default="all")
    parser.add_argument("--repeats", type=int, default=2, choices=range(1, 6))
    logging.disable(logging.CRITICAL)
    raise SystemExit(asyncio.run(run(parser.parse_args(), json.load(sys.stdin))))
