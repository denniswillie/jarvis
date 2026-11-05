"""Matched native code/continuity trials using production sessions and real DB storage.

Only media, a non-playing phone, and a declared optional-preparation fault are
fixtures. Ground truth stays in private reports, never application instructions.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
from contextlib import AsyncExitStack
from datetime import UTC, datetime, timedelta
from hashlib import sha256
import inspect
import json
import logging
import os
from pathlib import Path
import time
from unittest.mock import patch
from uuid import uuid4

from app.assistant.context_providers import CurrentTimeContextProvider, RecentMessagesContextProvider, SessionContextProvider
from app.experience.repository import ExperienceRepository, ToolExecutionRepository
from app.experience.segmenter import EventSegmentationPolicy, ExperienceEventSegmenter
from app.experience.service import ExperienceEventService
from app.perception.current_image import CurrentImageService
from app.perception.frame_window import RecentVisualFrameWindow, SensoryVisualFrame
from app.perception.tools import CurrentImageToolModule
from app.realtime.gemini.client import GeminiLiveConnection
from app.realtime.gemini.session import GeminiAudioSession
from app.realtime.models import RealtimeSessionContext
from app.realtime.session import RealtimeAudioSession
from app.sessions.repository import SessionRepository
from app.settings import JarvisSettings
from evals.gemini_live.interruption import DatabaseFixtures
from evals.gemini_live.transport_probe import read_pcm, write_wav
from evals.glasses_reliability.gemini_continuity_probe import compact, incomplete_records
from evals.glasses_reliability.gpt_turn_probe import ObservedConnection as GPTConnection, Phone, TraceCapture, stop_task
from evals.glasses_reliability.quality_cases import BY_NAME, definition_manifest
from evals.perception_memory.fixture_cleanup import cleanup_fixture_owners


def image_digests(payload):
    """Observe actual image bytes at the transport boundary without storing them."""
    result = []

    def visit(value):
        if isinstance(value, dict):
            mime = value.get("mimeType", "")
            if isinstance(mime, str) and mime.startswith("image/") and isinstance(value.get("data"), str):
                result.append(sha256(base64.b64decode(value["data"], validate=True)).hexdigest())
            url = value.get("image_url")
            if isinstance(url, str) and url.startswith("data:image/") and ";base64," in url:
                result.append(sha256(base64.b64decode(url.split(";base64,", 1)[1], validate=True)).hexdigest())
            for nested in value.values():
                visit(nested)
        elif isinstance(value, list):
            for nested in value:
                visit(nested)

    visit(payload)
    return result


def native_event_summary(event):
    """Keep protocol boundaries, never provider text, tool arguments, or media."""
    content = event.get("serverContent", {})
    return {
        "boundaries": [key for key in ("setupComplete", "toolCall", "toolCallCancellation", "goAway", "error") if key in event],
        "content_flags": {key: content[key] for key in ("turnComplete", "generationComplete", "interrupted")
                          if type(content.get(key)) is bool},
        "has_input_transcript": "inputTranscription" in content,
        "has_output_transcript": "outputTranscription" in content,
        "model_part_count": len(content.get("modelTurn", {}).get("parts", [])),
    }


def answer_observation_deadline(started, *, first_audio_at=None, pcm_bytes=0):
    """Bounded Gemini protocol allowance, not measured device playback.

    Google documents turnComplete after assumed playout, unlike generationComplete.
    A 55-second answer must not be declared hung after an arbitrary 50-second wait.
    Retain a 120-second cap; long or malformed answers remain quality evidence.
    """
    if pcm_bytes < 0:
        raise ValueError("Negative PCM length")
    expected_end = started + 50
    if first_audio_at is not None:
        expected_end = max(expected_end, first_audio_at + pcm_bytes / 48000 + 10)
    return min(started + 120, expected_end)


async def observe_visual_write(send, payload_factory, *, timeout_seconds, written_images):
    """GPT ambient images use their own priority-aware writer, not send_event."""
    digests = []

    def observed():
        payload = payload_factory()
        if payload is not None:
            digests.extend(image_digests(payload))
        return payload

    sent = await send(observed, timeout_seconds=timeout_seconds)
    if sent:
        written_images.extend({"sha256": digest, "at": time.monotonic()} for digest in digests)
    return sent


def select_matrix(cases, repeats, providers=("openai", "gemini")):
    if not cases or repeats < 1 or len(set(cases)) != len(cases) or any(name not in BY_NAME for name in cases):
        raise ValueError("Invalid or duplicate matrix cases")
    if not providers or len(set(providers)) != len(providers) or any(p not in {"openai", "gemini"} for p in providers):
        raise ValueError("Invalid or duplicate providers")
    return [(BY_NAME[name], repeat + 1, provider)
            for repeat in range(repeats) for name in cases
            for provider in (providers if repeat % 2 == 0 else tuple(reversed(providers)))]


def summarize(results, *, expected_runs):
    complete = expected_runs > 0 and len(results) == expected_runs
    return {
        "expected_runs": expected_runs, "recorded_runs": len(results), "matrix_complete": complete,
        "infrastructure_passed": complete and all(row["infrastructure_passed"] for row in results),
        "quality_acceptance": "pending_independent_answer_and_audio_review",
        "results": [{key: row.get(key) for key in (
            "case", "repeat", "provider", "infrastructure_passed", "checks", "error_type", "artifact_sha256",
        )} for row in results],
    }


async def run_case(case, repeat, provider, *, fixtures, settings, assets, output):
    label = f"{case.name}-{repeat}-{provider}"
    row = {"case": case.name, "repeat": repeat, "provider": provider,
           "stratum": case.stratum, "infrastructure_passed": False, "checks": {},
           "quality_acceptance": "not_graded", "device_playback": "not_tested"}
    owner = None
    phone = Phone()
    audio_session_id = str(uuid4())
    trace = TraceCapture(audio_session_id)
    logger = logging.getLogger("app.realtime.telemetry")
    logger.addHandler(trace)
    connections, written_images, usage, input_hashes, turns, frames = [], [], [], [], [], []
    native_events = []
    session = None

    native_terminals = {}

    class ObservedGPT(GPTConnection):
        async def send_visual_event(self, payload_factory, *, timeout_seconds):
            return await observe_visual_write(
                super().send_visual_event, payload_factory,
                timeout_seconds=timeout_seconds, written_images=written_images,
            )

        async def send_event(self, event):
            digests = []

            async def observed():
                payload = event() if callable(event) else event
                if inspect.isawaitable(payload):
                    payload = await payload
                if payload is not None:
                    digests.extend(image_digests(payload))
                return payload

            sent = await super().send_event(observed)
            if sent:
                written_images.extend({"sha256": digest, "at": time.monotonic()} for digest in digests)
            return sent

        async def receive_event(self):
            event = await super().receive_event()
            if event.get("type") == "response.done":
                response = event.get("response", {})
                native_terminals[response.get("id")] = response.get("status")
            if event.get("type") == "response.done" and event.get("response", {}).get("usage"):
                usage.append(event["response"]["usage"])
            return event

    class ObservedGemini(GeminiLiveConnection):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.history = []
            self.input_bytes = 0
            connections.append(self)

        async def connect(self, **kwargs):
            self.history = kwargs.get("history", [])
            return await super().connect(**kwargs)

        async def send_event(self, event):
            digests = image_digests(event)
            result = await super().send_event(event)
            written_images.extend({"sha256": digest, "at": time.monotonic()} for digest in digests)
            audio = event.get("realtimeInput", {}).get("audio")
            if audio:
                self.input_bytes += len(base64.b64decode(audio["data"]))
            return result

        async def receive_event(self):
            event = await super().receive_event()
            if len(native_events) < 4096:
                native_events.append({"at": time.monotonic(), "generation": len(connections), **native_event_summary(event)})
            else:
                row["native_events_overflowed"] = True
            if event.get("usageMetadata"):
                usage.append(event["usageMetadata"])
            return event

    try:
        async with AsyncExitStack() as stack, asyncio.timeout(180):
            owner, session_id = await fixtures.create("quality-" + label)
            repository = SessionRepository(fixtures.pool)
            experience = ExperienceEventService(
                repository=ExperienceRepository(fixtures.pool),
                segmenter=ExperienceEventSegmenter(EventSegmentationPolicy(300, 900)),
            )
            stack.push_async_callback(experience.close)
            window = RecentVisualFrameWindow(max_frames_per_session=4, ttl_seconds=10)
            lease = window.begin_stream(owner, session_id, uuid4())
            stack.callback(window.end_stream, lease)
            current = CurrentImageService(
                frame_window=window, max_age_seconds=settings.VISUAL_REALTIME_MAX_CONTEXT_AGE_SECONDS,
                max_image_bytes=settings.VISUAL_FRAME_MAX_BYTES, max_width=1024, max_height=1024,
            )
            kwargs = {"connection_factory": ObservedGemini} if provider == "gemini" else {}
            cls = GeminiAudioSession if provider == "gemini" else RealtimeAudioSession
            with patch("app.realtime.session.OpenAIRealtimeConnection", ObservedGPT):
                session = cls(
                    websocket=phone, context=RealtimeSessionContext(
                        audio_session_id=audio_session_id, owner_id=owner, jarvis_session_id=session_id,
                        client_kind="phone_audio", session=await repository.get_session(owner, session_id),
                    ), settings=settings, repository=repository,
                    context_providers=(CurrentTimeContextProvider(), SessionContextProvider(), RecentMessagesContextProvider(12)),
                    tool_modules=(CurrentImageToolModule(current),), sensory_frame_window=window,
                    tool_execution_repository=ToolExecutionRepository(fixtures.pool),
                    experience_event_service=experience, **kwargs,
                )
            barrier_entered = asyncio.Event()
            if case.interruption == "superseded_preparation" and provider == "gemini":
                prepare = session._prepare_turn

                async def preparation_barrier(turn):
                    if turn.segment_id == "bridge":
                        barrier_entered.set()
                        await asyncio.Future()
                    return await prepare(turn)

                session._prepare_turn = preparation_barrier
            running = asyncio.create_task(session.run())

            async def shutdown():
                try:
                    async with asyncio.timeout(25):
                        await session.close("quality_probe_complete")
                finally:
                    if not running.done():
                        running.cancel()
                    await asyncio.gather(running, return_exceptions=True)

            stack.push_async_callback(shutdown)

            async def until(predicate, timeout=50, deadline=None):
                async with asyncio.timeout(120 if deadline is not None else timeout):
                    while not predicate():
                        if running.done() or phone.closed:
                            raise RuntimeError("Native adapter ended before the required boundary")
                        if deadline is not None and time.monotonic() >= deadline():
                            raise TimeoutError("Answer observation budget exhausted")
                        await asyncio.sleep(.01)

            row["waiting_boundary"] = "session_ready"
            await until(lambda: any(e["type"] == "session.created" for e in phone.events))
            asset = assets[case.name]
            image = asset["jpeg"]

            def add_frame(stale=False):
                now = datetime.now(UTC)
                captured = now - timedelta(seconds=60) if stale else now
                frame_id = uuid4()
                sequence = len(frames) + 1
                accepted = window.put(SensoryVisualFrame(
                    frame_id=frame_id, owner_id=owner, session_id=session_id,
                    audio_session_id=audio_session_id, visual_session_id=lease.visual_session_id,
                    frame_sequence=sequence, source="glasses", captured_at=captured, server_received_at=now,
                    content_type="image/jpeg", width=asset["size"][0], height=asset["size"][1],
                    storage_bucket="synthetic-fixture-not-uploaded", storage_path=case.name + ".jpg",
                    raw_image_expires_at=now + timedelta(minutes=1), image_bytes=image,
                ), stream_lease=lease)
                if not accepted:
                    raise RuntimeError("Fixture stream rejected its own frame")
                frames.append({"frame_id": str(frame_id), "sequence": sequence,
                               "captured_at": captured.isoformat(), "sha256": sha256(image).hexdigest()})
                session.notify_visual_frame_available()

            async def camera():
                while True:
                    add_frame()
                    await asyncio.sleep(.5)

            if case.image_condition == "stale":
                add_frame(stale=True)
            elif case.expects_image:
                # Populate before the first input; subsequent frames follow the declared cadence.
                add_frame()
                stack.push_async_callback(stop_task, asyncio.create_task(camera()))

            async def speak(segment, pcm):
                start = len(phone.events)
                for offset in range(0, len(pcm), 4800):
                    chunk = pcm[offset:offset + 4800]
                    await phone.send("audio.input.append", segment_id=segment,
                                     audio_base64=base64.b64encode(chunk).decode())
                    await asyncio.sleep(len(chunk) / 48000)
                committed = time.monotonic()
                await phone.send("audio.input.commit", segment_id=segment)
                await phone.send("response.create")
                input_hashes.append({"segment": segment, "sha256": sha256(pcm).hexdigest(), "bytes": len(pcm)})
                turns.append({"segment": segment, "commit_requested_at": committed, "start_event": start})
                return start

            def answer_events(start):
                return phone.events[start:]

            async def finish(segment, start):
                row["waiting_boundary"] = "complete_answer_" + segment
                wait_started = time.monotonic()

                def gemini_deadline():
                    turn = session._turns.get(segment)
                    first_audio = next((e["at"] for e in phone.timeline
                                        if turn and e["type"] == "audio.output.delta"
                                        and e.get("response_id") == turn.output.response_id), None)
                    pcm_bytes = turn.output.forwarded_audio_bytes if turn else 0
                    allowed_until = answer_observation_deadline(
                        wait_started, first_audio_at=first_audio, pcm_bytes=pcm_bytes,
                    )
                    turns[-1]["observation_allowance"] = {
                        "seconds": round(allowed_until - wait_started, 3),
                        "forwarded_pcm_seconds": round(pcm_bytes / 48000, 3),
                        "basis": "documented_assumed_playout_not_device_playback",
                    }
                    return allowed_until
                def done():
                    if provider == "gemini":
                        turn = session._turns.get(segment)
                        return turn and turn.input_finalized and turn.output.completed and not session._domain_tasks
                    generation = str(session._turn_generation)
                    audio = {e.get("response_id") for e in answer_events(start) if e["type"] == "audio.output.delta"
                             and session.openai.response_generations.get(e.get("response_id")) == generation}
                    completed = {e.get("response_id") for e in answer_events(start) if e["type"] == "response.done"}
                    return bool(audio & completed) and not session._normal_turn_pending

                await until(done, deadline=gemini_deadline if provider == "gemini" else None)
                if provider == "gemini":
                    output_turn = session._turns[segment].output
                    complete_audio = (output_turn.completed and not output_turn.interrupted
                                      and output_turn.forwarded_audio_bytes > 0
                                      and output_turn.forwarded_audio_bytes == output_turn.provider_audio_bytes)
                    response_ids = [output_turn.response_id]
                else:
                    generation = str(session._turn_generation)
                    response_ids = sorted({e["response_id"] for e in answer_events(start)
                                           if e["type"] == "audio.output.delta"
                                           and session.openai.response_generations.get(e.get("response_id")) == generation})
                    complete_audio = bool(response_ids) and all(native_terminals.get(r) == "completed" for r in response_ids)
                row["checks"]["complete_answer_audio_" + segment] = bool(complete_audio)
                turns[-1]["response_ids"] = response_ids

            first_start = await speak("step", asset["speech"][0])
            if case.interruption != "none":
                row["waiting_boundary"] = "first_answer_interruptible"
                def interruptible():
                    if provider == "gemini":
                        turn = session._turns.get("step")
                        return turn and turn.input_finalized and turn.output.forwarded_audio_bytes and all(
                            word in compact(turn.output.transcript) for word in ("elif", "mid"))
                    text = " ".join(e.get("delta", e.get("text", "")) for e in answer_events(first_start)
                                    if e["type"] in {"transcript.delta", "transcript.done"})
                    return any(e["type"] == "audio.output.delta" for e in answer_events(first_start)) and all(
                        word in compact(text) for word in ("elif", "mid"))

                await until(interruptible)
                if provider == "gemini":
                    active = not session._turns["step"].output.completed
                else:
                    ids = {e.get("response_id") for e in answer_events(first_start) if e["type"] == "audio.output.delta"}
                    active = any(session._active_responses[r].completed_response is None for r in ids if r in session._active_responses)
                row["checks"]["interrupted_during_active_output"] = bool(active)
                await phone.send("response.cancel")
                if case.interruption == "superseded_preparation":
                    if provider == "gemini":
                        await speak("bridge", asset["bridge"])
                        row["waiting_boundary"] = "bridge_preparation_entered"
                        await until(barrier_entered.is_set, timeout=5)
                    else:
                        # GPT does not use Gemini's preparation-driven reconnect.
                        # Match the spoken bridge but do not invent a GPT reset.
                        await finish("bridge", await speak("bridge", asset["bridge"]))
                    row["controlled_reset_scope"] = "gemini_preparation_only_gpt_keeps_connection"
            else:
                await finish("step", first_start)
            if len(case.prompts) == 2:
                await finish("followup", await speak("followup", asset["speech"][1]))
            row["waiting_boundary"] = "final_user_transcripts"
            await until(lambda: sum(e["type"] == "user.transcript.done" for e in phone.events)
                        >= len(asset["speech"]))
            row["checks"].update({
                "no_provider_errors": not any(e["type"] == "error" for e in phone.events),
                "forwarded_audio_present": bool(phone.audio) and any(phone.audio.values()),
                "image_boundary": (bool(written_images) if case.expects_image else not written_images),
                "only_supplied_image_written": all(e["sha256"] == sha256(image).hexdigest() for e in written_images),
                "trace_not_overflowed": not trace.overflowed,
                "native_events_not_overflowed": not row.get("native_events_overflowed", False),
            })
            if provider == "gemini":
                expected_connections = 2 if case.interruption == "superseded_preparation" else 1
                row["checks"]["expected_connections"] = len(connections) == expected_connections
                row["checks"]["input_not_replayed"] = sum(c.input_bytes for c in connections) == sum(h["bytes"] for h in input_hashes)
                if expected_connections == 2:
                    history = connections[-1].history
                    row["restored_history"] = history
                    row["checks"]["interrupted_context_marked_uncertain"] = any(
                        r.get("assistant_output", {}).get("heard_text_status") == "unknown"
                        and "elif" in compact(r.get("assistant_output", {}).get("context_transcript", ""))
                        for r in incomplete_records(history)
                    )
            else:
                row["gpt_request_audit"] = trace.audit(session.openai)
                row["checks"]["requests_accounted_for"] = not (
                    row["gpt_request_audit"]["unexplained_requests"] or row["gpt_request_audit"]["untraced_native_requests"])
            # Independent committed readback, not a provider's own claim of storage.
            snapshot = await fixtures.snapshot(owner, session_id)
            row["tool_executions"] = snapshot["executions"]
            row["checks"]["no_domain_mutations"] = not snapshot["memories"] and not any(
                e["mutates_state"] for e in snapshot["executions"])
            row["waiting_boundary"] = "finished"
    except Exception as error:
        row["error_type"] = type(error).__name__
    finally:
        logger.removeHandler(trace)
        row.update(frames=frames, written_images=written_images, input_pcm=input_hashes, turns=turns,
                   traces=trace.records, usage=usage, phone_timeline=phone.timeline,
                   native_terminal_statuses=native_terminals, native_events=native_events)
        row["provider_errors"] = [{"code": e.get("code")} for e in phone.events if e["type"] == "error"]
        row["transcripts"] = [e for e in phone.events if e["type"] in {"transcript.done", "user.transcript.done"}]
        row["terminal_events"] = [e for e in phone.events if e["type"] in {"response.done", "audio.output.clear", "audio.provider.reconnected"}]
        row["audio_files"] = []
        for index, (item, pcm) in enumerate(phone.audio.items(), 1):
            name = f"{label}-{index}.wav"
            write_wav(output / name, bytes(pcm))
            row["audio_files"].append({"file": name, "item_id": item, "pcm_sha256": sha256(pcm).hexdigest(), "bytes": len(pcm)})
        if session is not None and provider == "gemini":
            row["diagnostics"] = session.diagnostics()
            row["checks"]["clean_shutdown"] = not (
                row["diagnostics"]["owned_tasks"] or row["diagnostics"]["domain_tasks"]
                or row["diagnostics"]["conversation_persistence"]["degraded"])
        if owner is not None:
            row["cleanup"] = await cleanup_fixture_owners(fixtures.dsn, {owner})
            row["checks"]["cleanup_zero"] = row["cleanup"]["remaining_rows"] == 0
        row["infrastructure_passed"] = bool(row["checks"]) and all(row["checks"].values()) and "error_type" not in row
        encoded = json.dumps(row, indent=2, default=str) + "\n"
        (output / (label + ".json")).write_text(encoded)
        row["artifact_sha256"] = sha256(encoded.encode()).hexdigest()
    print(json.dumps({k: row.get(k) for k in ("case", "repeat", "provider", "infrastructure_passed", "checks", "error_type")}), flush=True)
    return row


def load_assets(media, names):
    manifest = json.loads((media / "manifest.json").read_text())
    expected = definition_manifest()
    if any(manifest.get(key) != expected[key] for key in ("schema", "cases", "schedule", "scoring", "limits")):
        raise ValueError("Prepared case definitions do not match this evaluator")
    assets, hashes = {}, {}
    for name in names:
        jpeg = (media / (name + ".jpg")).read_bytes()
        if sha256(jpeg).hexdigest() != manifest["media"][name]["image_sha256"]:
            raise ValueError("Image changed after predeclaration")
        speech = [read_pcm(media / f"{name}-{i}.wav") for i in range(1, len(BY_NAME[name].prompts) + 1)]
        if not all(speech):
            raise ValueError("Empty input speech is not a valid native case")
        bridge = read_pcm(media / "bridge.wav")
        if not bridge:
            raise ValueError("Empty bridge speech is not a valid native case")
        assets[name] = {"jpeg": jpeg, "size": manifest["media"][name]["size"], "speech": speech, "bridge": bridge}
        hashes[name] = {"image_sha256": sha256(jpeg).hexdigest(),
                        "input_pcm_sha256": [sha256(pcm).hexdigest() for pcm in speech],
                        "bridge_pcm_sha256": sha256(assets[name]["bridge"]).hexdigest()}
    return assets, hashes


async def run(args):
    names = args.cases.split(",")
    providers = ("openai", "gemini") if args.provider == "both" else (args.provider,)
    matrix = select_matrix(names, args.repeats, providers)
    assets, hashes = load_assets(args.media, names)
    settings = JarvisSettings(_env_file=None)
    args.output.mkdir(parents=True, exist_ok=False)
    predeclared = {**definition_manifest(), "selected_cases": names, "repeats": args.repeats,
                   "providers": providers, "input_hashes": hashes, "source_revision": args.source_revision,
                   "models": {"openai": settings.OPENAI_REALTIME_MODEL, "gemini": settings.GEMINI_LIVE_MODEL},
                   "created_at": datetime.now(UTC).isoformat()}
    (args.output / "run-manifest.json").write_text(json.dumps(predeclared, indent=2) + "\n")
    fixtures = DatabaseFixtures(os.environ["JARVIS_TEST_DB_URL"], args.output)
    rows = []
    try:
        await fixtures.connect()
        for case, repeat, provider in matrix:
            rows.append(await run_case(case, repeat, provider, fixtures=fixtures, settings=settings,
                                       assets=assets, output=args.output))
            (args.output / "matrix.json").write_text(json.dumps(summarize(rows, expected_runs=len(matrix)), indent=2) + "\n")
    finally:
        await fixtures.cleanup()
        (args.output / "matrix.json").write_text(json.dumps({**summarize(rows, expected_runs=len(matrix)),
                                                           "cleanup": fixtures.cleanup_result}, indent=2) + "\n")
    return 0 if summarize(rows, expected_runs=len(matrix))["infrastructure_passed"] else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--media", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--cases", default=",".join(BY_NAME))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--provider", choices=("both", "openai", "gemini"), default="both")
    args = parser.parse_args()
    if not 1 <= args.repeats <= 3:
        parser.error("Use 1-3 repeats; fewer than three is explicitly a pilot, not acceptance")
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("app.realtime.telemetry").setLevel(logging.INFO)
    logging.getLogger("app.realtime.telemetry").propagate = False
    raise SystemExit(asyncio.run(run(args)))
