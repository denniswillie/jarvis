"""Paid native image acceptance through Jarvis's production Gemini audio adapter.

Images and phone input/output are controlled fixtures. Native transport, context,
current-image resolution, tools, conversation persistence, and experience evidence
are real. This proves image delivery, not blurry-photo OCR or glasses playback.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
from contextlib import suppress
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from io import BytesIO
import json
import logging
from pathlib import Path
import sys
import time
from uuid import uuid4

from PIL import Image, ImageDraw, ImageFont

from app.assistant.context_providers import CurrentTimeContextProvider, SessionContextProvider
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
from app.sessions.repository import SessionRepository
from app.settings import JarvisSettings
from evals.gemini_live.interruption import DatabaseFixtures
from evals.gemini_live.session_probe import ProbePhone
from evals.gemini_live.transport_probe import read_pcm, redact_native, write_wav
from evals.perception_memory.fixture_cleanup import cleanup_fixture_owners


def marker_jpeg(marker: str) -> bytes:
    image = Image.new("RGB", (640, 360), "white")
    ImageDraw.Draw(image).text((48, 145), marker, fill="black", font=ImageFont.load_default(size=44))
    output = BytesIO()
    image.save(output, format="JPEG", quality=95)
    return output.getvalue()


async def case(*, automatic, fixtures, keys, speech, output):
    name = "automatic" if automatic else "tool_requested"
    marker = "copper orchard" if automatic else "silver maple"
    jpeg = marker_jpeg(marker)
    (output / f"{name}.jpg").write_bytes(jpeg)
    owner, session_id = await fixtures.create("production-image-" + name)
    repository = SessionRepository(fixtures.pool)
    experience = ExperienceEventService(
        repository=ExperienceRepository(fixtures.pool),
        segmenter=ExperienceEventSegmenter(EventSegmentationPolicy(300, 900)),
    )
    phone, trace, native_audio = ProbePhone(), [], bytearray()
    started = time.monotonic()

    class TracedConnection(GeminiLiveConnection):
        async def send_event(self, event):
            await super().send_event(event)
            realtime = event.get("realtimeInput", {})
            if "audio" not in realtime:
                trace.append({"direction": "sent", "at_seconds": time.monotonic() - started,
                              "event": redact_native(event)})

        async def receive_event(self):
            event = await super().receive_event()
            trace.append({"direction": "received", "at_seconds": time.monotonic() - started,
                          "event": redact_native(event)})
            for part in event.get("serverContent", {}).get("modelTurn", {}).get("parts", []):
                inline = part.get("inlineData", {})
                if inline.get("mimeType", "").startswith("audio/"):
                    native_audio.extend(decode_pcm16(inline.get("data"), max_bytes=8 * 1024 * 1024))
            return event

    settings = JarvisSettings(_env_file=None, GEMINI_API_KEY=keys["gemini"], VISUAL_REALTIME_ENABLED=automatic)
    window = RecentVisualFrameWindow(max_frames_per_session=4, ttl_seconds=10)
    lease = window.begin_stream(owner, session_id, uuid4())
    current = CurrentImageService(frame_window=window,
                                  max_age_seconds=settings.VISUAL_REALTIME_MAX_CONTEXT_AGE_SECONDS,
                                  max_image_bytes=settings.VISUAL_FRAME_MAX_BYTES,
                                  max_width=1024, max_height=1024)
    session = GeminiAudioSession(
        websocket=phone, context=RealtimeSessionContext(
            audio_session_id=str(uuid4()), owner_id=owner, jarvis_session_id=session_id,
            client_kind="phone_audio", session=await repository.get_session(owner, session_id),
        ), settings=settings, repository=repository, connection_factory=TracedConnection,
        tool_modules=(CurrentImageToolModule(current),), sensory_frame_window=window,
        context_providers=(CurrentTimeContextProvider(), SessionContextProvider()),
        tool_execution_repository=ToolExecutionRepository(fixtures.pool), experience_event_service=experience,
    )

    async def camera_fixture():
        sequence = 0
        while True:
            sequence += 1
            now = datetime.now(UTC)
            window.put(SensoryVisualFrame(
                frame_id=uuid4(), owner_id=owner, session_id=session_id,
                audio_session_id=session.context.audio_session_id, visual_session_id=lease.visual_session_id,
                frame_sequence=sequence, source="glasses", captured_at=now, server_received_at=now,
                content_type="image/jpeg", width=640, height=360, storage_bucket="synthetic-fixture",
                storage_path="current.jpg", raw_image_expires_at=now + timedelta(minutes=1), image_bytes=jpeg,
            ), stream_lease=lease)
            # Populate only the normal sensory window. No automatic video loop
            # is added to the native connection; production decides what to send.
            await asyncio.sleep(.5)

    running = asyncio.create_task(session.run())
    camera = asyncio.create_task(camera_fixture())
    result = {"case": name, "passed": False, "expected_marker": marker,
              "production_transport_unmodified": True,
              "image_source": "synthetic_fixture_not_hardware", "device_playback": "not_tested"}
    try:
        async with asyncio.timeout(90):
            await phone.wait_for("session.created")
            pcm = read_pcm(speech)
            for offset in range(0, len(pcm), 4800):
                chunk = pcm[offset:offset + 4800]
                await phone.send("audio.input.append", segment_id="image", audio_base64=base64.b64encode(chunk).decode())
                await asyncio.sleep(len(chunk) / 48000)
            await phone.send("audio.input.commit", segment_id="image")
            await phone.send("response.create")
            await phone.wait_for("response.done")
            # Input transcription is independent of output; wait for its final
            # rather than closing early and pretending persistence was tested.
            await phone.wait_for("user.transcript.done")
            answers = [event["text"] for event in phone.events if event["type"] == "transcript.done"]
            result["answers"] = answers
            forwarded = b"".join(bytes(pcm) for pcm in phone.audio.values())
            result["pcm"] = {"native_bytes": len(native_audio), "forwarded_bytes": len(forwarded),
                             "native_sha256": sha256(native_audio).hexdigest(),
                             "forwarded_sha256": sha256(forwarded).hexdigest(),
                             "exact_match": bytes(native_audio) == forwarded}
            sent = [item["event"] for item in trace if item["direction"] == "sent"]
            videos = [event for event in sent if "video" in event.get("realtimeInput", {})]
            appended = [part for event in sent for turn in event.get("clientContent", {}).get("turns", [])
                        for part in turn.get("parts", []) if "inlineData" in part]
            image_responses = [response for event in sent
                               for response in event.get("toolResponse", {}).get("functionResponses", [])
                               if response.get("name") == "get_current_camera_view" and response.get("parts")]
            result["automatic_images"] = len(videos)
            result["appended_images"] = len(appended)
            result["image_tool_responses"] = len(image_responses)
            assert len(answers) == 1 and marker in answers[0].lower()
            assert native_audio and bytes(native_audio) == forwarded
            assert len(videos) + len(appended) == 1 if automatic else not videos and not appended and len(image_responses) >= 1
            assert not any(event["type"] == "error" for event in phone.events)
            result["passed"] = True
    except Exception as error:
        result["error_type"] = type(error).__name__
    finally:
        camera.cancel()
        await asyncio.gather(camera, return_exceptions=True)
        window.end_stream(lease)
        await session.close("image_probe_finished")
        with suppress(asyncio.CancelledError):
            await running
        await experience.close()
        result["diagnostics"] = session.diagnostics()
        if session.journal.diagnostics()["degraded"] or session.diagnostics()["domain_tasks"]:
            result["passed"] = False
        result["trace"] = trace
        result["app_events"] = [event for event in phone.events if event["type"] != "audio.output.delta"]
        write_wav(output / f"{name}-native.wav", bytes(native_audio))
        write_wav(output / f"{name}-app.wav", b"".join(bytes(pcm) for pcm in phone.audio.values()))
        result["cleanup"] = await cleanup_fixture_owners(keys["database"], {owner})
        (output / f"{name}.json").write_text(json.dumps(result, indent=2, default=str) + "\n")
    print(json.dumps({key: result.get(key) for key in (
        "case", "passed", "error_type", "answers", "pcm", "automatic_images", "appended_images", "image_tool_responses",
    )} | {"cleanup_remaining": result["cleanup"]["remaining_rows"]}), flush=True)
    return result


async def run(args, keys):
    args.output.mkdir(parents=True, exist_ok=False)
    fixtures = DatabaseFixtures(keys["database"], args.output)
    results = []
    try:
        await fixtures.connect()
        for automatic in (True, False):
            result = await case(automatic=automatic, fixtures=fixtures, keys=keys,
                                speech=args.speech, output=args.output)
            results.append({"case": result["case"], "passed": result["passed"]})
    finally:
        await fixtures.cleanup()
        (args.output / "matrix.json").write_text(json.dumps({"results": results, "cleanup": fixtures.cleanup_result}, indent=2) + "\n")
    return 0 if results and all(result["passed"] for result in results) else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--speech", type=Path, required=True)
    args = parser.parse_args()
    logging.disable(logging.CRITICAL)
    raise SystemExit(asyncio.run(run(args, json.load(sys.stdin))))
