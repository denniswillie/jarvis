"""Opt-in N03 trace through real foreground providers and disposable DB schemas.

This uses synthetic speech, synthetic pixels and a read-only image tool. It is
not glasses acceptance, latency benchmarking, or permission for external work.
The production backend and its feature/provider settings are never modified.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
from contextlib import AsyncExitStack
from datetime import UTC, datetime, timedelta
import hashlib
import io
import json
import logging
import os
from pathlib import Path
import subprocess
from uuid import uuid4
import wave

import asyncpg
from PIL import Image, ImageDraw, ImageFont

from app.assistant.context_providers import CurrentTimeContextProvider, RecentMessagesContextProvider, SessionContextProvider
from app.images import ImageUnavailableError
from app.migrations.runner import apply_migrations, quote_schema
from app.perception.current_image import CurrentImageService
from app.perception.frame_window import RecentVisualFrameWindow, SensoryVisualFrame
from app.perception.tools import CurrentImageToolModule
from app.realtime.gemini.session import GeminiAudioSession
from app.realtime.models import RealtimeSessionContext
from app.realtime.session import RealtimeAudioSession
from app.runtime.feature_config import TaskLimits
from app.sessions.repository import SessionRepository
from app.settings import JarvisSettings, is_configured_secret
from app.tasks.context import TaskContextBuilder
from app.tasks.context_capture import SessionContextCapture
from app.tasks.context_images import TaskContextImages
from app.tasks.context_repository import ContextRepository
from app.tasks.context_tools import TaskContextToolModule
from app.tasks.models import TaskCreate
from app.tasks.repository import TaskRepository
from app.tasks.session_context import SessionTaskContextProvider, SessionTaskContextService
from app.tools.module import ToolBuildContext
from app.tools.registry import create_tool_registry
from evals.second_brain.fixtures import DatabaseFixture


QUESTION = (
    "What is the title of my active background task, and what single word is shown "
    "in the camera image? This is only a read-only test. Do not start any work."
)
TASK_TITLE = "Prepare the synthetic lantern report"
IMAGE_WORD = "MANGO"


class ProbePhone:
    """Application-side transport, with no microphone, speaker or playback ACK."""
    def __init__(self):
        self.inputs = asyncio.Queue()
        self.events = []

    async def iter_text(self):
        while True:
            event = await self.inputs.get()
            if event is None:
                return
            yield json.dumps(event)

    async def send_text(self, text):
        await self.send_json(json.loads(text))

    async def send_json(self, event):
        # The output audio is counted and hashed, never played or retained.
        payload = dict(event)
        audio = payload.pop("audio_base64", None)
        if audio is not None:
            raw = base64.b64decode(audio, validate=True)
            payload["audio_bytes"] = len(raw)
            payload["audio_sha256"] = hashlib.sha256(raw).hexdigest()
        if payload.get("type") == "error":
            payload.pop("message", None)
        self.events.append(payload)

    async def send(self, kind, **fields):
        await self.inputs.put({"type": kind, "client_event_id": str(uuid4()), **fields})

    async def close(self, *args, **kwargs):
        await self.inputs.put(None)


def completed_answer(events):
    completed = {
        event.get("response_id") for event in events
        if event.get("type") == "response.done" and event.get("status", "completed") == "completed"
    }
    return "\n".join(
        event["text"] for event in events
        if event.get("type") == "transcript.done" and isinstance(event.get("text"), str)
        and event.get("response_id") in completed
    ).strip()


async def wait_for(predicate, running, *, seconds=60):
    async with asyncio.timeout(seconds):
        while not predicate():
            if running.done():
                error = running.exception()
                if error is not None:
                    raise error
                raise RuntimeError("foreground_session_ended_before_expected_evidence")
            await asyncio.sleep(0.02)


async def close_probe_resources(result, *, fixture, pool=None, running=None, voice=None):
    async def cleanup_fixture():
        result["cleanup"] = await fixture.cleanup()

    async def stop_running():
        if not running.done():
            running.cancel()
        await asyncio.gather(running, return_exceptions=True)

    # Every registered release runs even if an earlier one raises.
    async with AsyncExitStack() as stack:
        stack.push_async_callback(cleanup_fixture)
        if pool is not None:
            stack.push_async_callback(pool.close)
        if running is not None:
            stack.push_async_callback(stop_running)
        if voice is not None:
            stack.push_async_callback(voice.close, "native_context_trace_complete")


def write_json(path, value):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=True, default=str)
        handle.write("\n")


def make_media(directory):
    aiff, wav_path = directory / "request.aiff", directory / "request.wav"
    subprocess.run(["/usr/bin/say", "-o", str(aiff), QUESTION], check=True, timeout=30,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(["/usr/bin/afconvert", str(aiff), str(wav_path),
                    "-f", "WAVE", "-d", "LEI16@24000", "-c", "1"], check=True, timeout=30,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    with wave.open(str(wav_path), "rb") as handle:
        if (handle.getframerate(), handle.getsampwidth(), handle.getnchannels()) != (24000, 2, 1):
            raise ValueError("native_context_probe_requires_pcm16_24khz_mono")
        pcm = handle.readframes(handle.getnframes())
    picture = Image.new("RGB", (640, 400), "white")
    drawing = ImageDraw.Draw(picture)
    drawing.text((100, 140), IMAGE_WORD, fill="black", font=ImageFont.load_default(size=96))
    output = io.BytesIO()
    picture.save(output, format="JPEG", quality=95)
    pixels = output.getvalue()
    (directory / "frame.jpg").write_bytes(pixels)
    return pcm, pixels


async def probe(provider, root, pcm, pixels):
    directory = root / provider
    directory.mkdir(mode=0o700)
    base = JarvisSettings()
    if not is_configured_secret(base.SUPABASE_DB_URL):
        raise ValueError("database_not_configured")
    key = base.OPENAI_API_KEY if provider == "openai" else base.GEMINI_API_KEY
    if not is_configured_secret(key):
        raise ValueError("native_provider_key_not_configured")
    fixture = DatabaseFixture(base.SUPABASE_DB_URL, directory, test_name=f"N03-{provider}")
    pool, voice, running = None, None, None
    phone, contexts = ProbePhone(), []
    result = {"provider": provider, "status": "failed", "physical_glasses_test": False}
    try:
        connection = await asyncpg.connect(base.SUPABASE_DB_URL, timeout=10, command_timeout=90)
        try:
            schema = await fixture.create_schema(connection)
            await apply_migrations(connection, schema=schema, fresh=True)
        finally:
            await connection.close()

        async def setup(connection):
            await connection.execute(f"set search_path to {quote_schema(schema)},pg_catalog,public,extensions")

        pool = await asyncpg.create_pool(base.SUPABASE_DB_URL, min_size=1, max_size=4,
                                        timeout=10, command_timeout=30, setup=setup)
        owner = fixture.owners[0]
        repository = SessionRepository(pool)
        session = await repository.create_session(owner, "Synthetic native context trace")
        tasks = TaskRepository(pool, TaskLimits(lease_seconds=180))
        task = await tasks.create(
            owner, TaskCreate(request_id=uuid4(), title=TASK_TITLE,
                              requested_outcome="Synthetic read-only context trace; no external actions.",
                              session_id=session.id), model="fixture-no-worker",
        )
        claim = await tasks.claim_next("native-context-fixture", ("frontier",))
        if claim is None or claim.task.id != task.id:
            raise RuntimeError("fixture_task_claim_missing")
        canonical = ContextRepository(tasks)
        audio_session_id = str(uuid4())
        capture = SessionContextCapture(canonical, owner_id=owner, session_id=session.id,
                                        provider=provider, audio_session_id=audio_session_id)
        # A generous fixture freshness window removes setup time from this
        # context-integrity check. It is not a proposed production policy.
        settings = base.model_copy(update={
            "VISUAL_REALTIME_ENABLED": True, "VISUAL_REALTIME_MAX_CONTEXT_AGE_SECONDS": 60.0,
            "GEMINI_LIVE_ENABLED": True,
        })
        window = RecentVisualFrameWindow(max_frames_per_session=4, ttl_seconds=90)
        lease = window.begin_stream(owner, session.id, uuid4())
        now = datetime.now(UTC)
        frame = SensoryVisualFrame(
            frame_id=uuid4(), owner_id=owner, session_id=session.id, audio_session_id=audio_session_id,
            visual_session_id=lease.visual_session_id, frame_sequence=1, source="glasses",
            captured_at=now, server_received_at=now, content_type="image/jpeg",
            width=640, height=400, storage_bucket="fixture-not-uploaded",
            storage_path="fixture-not-uploaded/frame.jpg", raw_image_expires_at=now + timedelta(minutes=5),
            image_bytes=pixels,
        )
        window.put(frame, stream_lease=lease)
        images = CurrentImageService(
            frame_window=window, max_age_seconds=60, max_image_bytes=settings.VISUAL_FRAME_MAX_BYTES,
            max_width=settings.VISUAL_FRAME_MAX_WIDTH, max_height=settings.VISUAL_FRAME_MAX_HEIGHT,
        )
        context_provider = SessionTaskContextProvider(SessionTaskContextService(canonical, timeout_seconds=10))
        providers = (CurrentTimeContextProvider(), SessionContextProvider(),
                     RecentMessagesContextProvider(max_items=20), context_provider)
        voice = (RealtimeAudioSession if provider == "openai" else GeminiAudioSession)(
            websocket=phone,
            context=RealtimeSessionContext(audio_session_id=audio_session_id, owner_id=owner,
                jarvis_session_id=session.id, client_kind="phone_audio", session=session),
            settings=settings, repository=repository,
            tool_modules=(CurrentImageToolModule(images),), context_providers=providers,
            sensory_frame_window=window, context_capture=capture,
        )
        builder = voice.assistant_context_builder if provider == "openai" else voice._context_service.builder
        original_build = builder.build

        async def recorded_build(**kwargs):
            built = await original_build(**kwargs)
            contexts.append(built.as_prompt_payload())
            return built

        builder.build = recorded_build
        running = asyncio.create_task(voice.run())
        await wait_for(lambda: any(e.get("type") == "session.created" for e in phone.events), running, seconds=30)
        for start in range(0, len(pcm), 4800):
            await phone.send("audio.input.append", segment_id="synthetic-question",
                             audio_base64=base64.b64encode(pcm[start:start+4800]).decode("ascii"))
            await asyncio.sleep(0)
        await phone.send("audio.input.commit", segment_id="synthetic-question")
        await phone.send("response.create")
        await wait_for(lambda: bool(completed_answer(phone.events)), running)
        await wait_for(lambda: any(e.get("type") == "user.transcript.done" for e in phone.events), running, seconds=15)
        if not await capture.flush():
            raise RuntimeError("canonical_capture_incomplete")
        context_builder = TaskContextBuilder(canonical)
        manifest = await context_builder.build(claim, shared_context=contexts[-1] if contexts else {})
        registry = create_tool_registry(
            modules=[TaskContextToolModule(context_builder, claim, TaskContextImages(canonical, current=images))],
            context=ToolBuildContext(settings=settings, owner_id=owner, session_id=session.id),
        )
        archive = await registry.get("read_task_context_archive").handler({"limit": 100})
        image_sources = [
            event for event in archive.get("events", [])
            if event.get("kind") == "observation" and isinstance(event.get("content"), dict)
            and any(ref.get("frame_id") == str(frame.frame_id) for ref in event["content"].get("images", []))
        ]
        same_pixels = False
        if image_sources:
            image_tool = registry.get("read_task_context_image")
            selection = await image_tool.handler({"event_id": image_sources[-1]["id"]})
            resolved = await image_tool.image_resolver(selection)
            await resolved.revalidate()
            same_pixels = resolved.image_bytes == pixels
        records = await canonical.archive(owner, session.id, through=manifest.context_watermark)
        answer = completed_answer(phone.events)
        checks = {
            "native_audio_response": any(e.get("audio_bytes", 0) > 0 for e in phone.events),
            "task_title_recalled": "lantern" in answer.casefold(),
            "image_word_read": "mango" in answer.casefold(),
            "finalized_user_transcript": any(r.kind == "user_transcript" and r.authorizes_request for r in records),
            "canonical_image_source": bool(image_sources),
            "worker_resolved_identical_original_pixels": same_pixels,
            "capture_complete": capture.issue is None,
            "playback_not_fabricated": all(
                event.get("heard_state") == "unknown" for event in manifest.content["messages"]
                if event.get("kind") == "assistant_output"
            ),
        }
        result.update(
            status="passed" if all(checks.values()) else "failed", checks=checks,
            task_id=str(task.id), session_id=str(session.id), manifest_id=str(manifest.id),
            image_sha256=hashlib.sha256(pixels).hexdigest(),
            model=settings.OPENAI_REALTIME_MODEL if provider == "openai" else voice.config.model,
            answer=answer, fixture_freshness_seconds=60,
        )
        write_json(directory / "context-manifest.json", manifest.model_dump(mode="json"))
        write_json(directory / "archive.json", archive)
    except Exception as error:
        result["failure"] = {"type": type(error).__name__, "code": getattr(error, "code", None)}
    finally:
        try:
            await close_probe_resources(result, fixture=fixture, pool=pool, running=running, voice=voice)
        except Exception as error:
            result["status"] = "failed"
            result["cleanup_error"] = {"type": type(error).__name__, "code": getattr(error, "code", None)}
        finally:
            write_json(directory / "foreground-contexts.json", contexts)
            write_json(directory / "phone-events.json", phone.events)
            write_json(directory / "result.json", result)
    return result


async def run(args):
    os.umask(0o077)
    root = args.output_dir.resolve()
    root.mkdir(mode=0o700, parents=True, exist_ok=False)
    pcm, pixels = make_media(root)
    write_json(root / "fixture.json", {
        "question": QUESTION, "task_title": TASK_TITLE, "image_word": IMAGE_WORD,
        "audio_sha256": hashlib.sha256(pcm).hexdigest(), "image_sha256": hashlib.sha256(pixels).hexdigest(),
        "real_personal_data": False, "real_email_or_external_actions": False,
    })
    results = []
    for provider in ("openai", "gemini") if args.provider == "both" else (args.provider,):
        result = await probe(provider, root, pcm, pixels)
        results.append(result)
        print(json.dumps(result), flush=True)
    write_json(root / "results.json", results)
    return 0 if all(item["status"] == "passed" for item in results) else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=("openai", "gemini", "both"), default="both")
    parser.add_argument("--output-dir", required=True, type=Path, help="A new private evidence directory")
    args = parser.parse_args()
    logging.disable(logging.CRITICAL)
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
