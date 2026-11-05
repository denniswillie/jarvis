"""One real English-note journey; author/offline checks do not start this run.

prepare generates local synthetic speech. execute burns one explicit attempt,
uses the normal isolated app/worker, and never polls tasks or injects results.
Only AUDIO-model wire input is traced. Worker wire is NOT CAPTURED.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
from contextlib import redirect_stdout
from datetime import UTC, datetime, timedelta
from io import BytesIO
import json
import os
from pathlib import Path
import re
import time
from uuid import UUID
import wave

from evals.responsive_voice import lean_system_replay as base


FORMAT = "jarvis.lean_note_replay.v1"
SCRIPT = (
    ("note", "Please add this to my note: today I did eight slow arm raises using the blue resistance band."),
    ("unrelated", "While you work on that, explain the difference between affect and effect in one sentence."),
    ("correct", "Please correct that entry: I did nine arm raises, not eight. Keep the blue resistance band detail."),
)
LIMITS = {**base.LIMITS, "utterances": 3, "late_wait_seconds": 60,
          "correction_min_remaining_seconds": 45, "worker_start_seconds": 75,
          "worker_stop_seconds": 35, "host_lease_seconds": 600}
WORKER_LIMITS = {"active_seconds": 120, "tool_steps": 20, "request_output_tokens": 4096,
                 "task_output_tokens": 8192, "task_input_tokens": 64000,
                 "owner_daily_tokens": 128000, "max_attempts": 2,
                 "owner_concurrency": 1, "global_concurrency": 1}
TRACE_SCOPE = {
    "audio_model": "Existing private Realtime wire JSONL and line-by-line rendering; client budget input recorded separately.",
    "audio_input": "Hashed source WAVs and actual browser input sample times; not provider-received media-byte proof.",
    "worker_request_response_wire": "NOT CAPTURED",
    "worker_effects": "Actual canonical task/admission/action/result/usage/events, not worker prose as a save receipt.",
}

# Passive metadata observation on the SAME peer/data channel used by the shared
# HTML. No send, response.create, transcript insertion, result hold or transport.
OBSERVER = r"""(() => {
  const original = RTCPeerConnection.prototype.createDataChannel;
  const notifications = {}, events = [];
  let failed = null, bytes = 0;
  RTCPeerConnection.prototype.createDataChannel = function(...args) {
    const channel = original.apply(this, args);
    channel.addEventListener('message', event => {
      try {
        const p = JSON.parse(event.data), response = p.response || {};
        const id = p.response_id || response.id;
        if (p.type === 'response.created' && response.metadata?.jarvis_offer_id) {
          notifications[id] = {response_id:id, metadata:response.metadata,
            created_frame:window.leanState().audio_frame, utc:new Date().toISOString()};
        }
        if (p.type === 'response.created' || p.type === 'response.done' ||
            p.type === 'response.output_audio_transcript.done' ||
            p.type === 'response.audio_transcript.done') {
          const row = {type:p.type, response_id:id, metadata:response.metadata,
            status:response.status, transcript:p.transcript,
            utc:new Date().toISOString(), audio_frame:window.leanState().audio_frame};
          bytes += new TextEncoder().encode(JSON.stringify(row)).length;
          if (bytes > 2097152 || events.length >= 512) {failed='observer_bound'; return;}
          events.push(row);
        }
      } catch {failed='observer_event_invalid';}
    });
    return channel;
  };
  window.noteState = () => ({...window.leanState(), notifications, observer_error:failed});
  window.noteEvidence = () => ({notifications, events, observer_error:failed});
  return true;
})()"""


def validate_models(model, voice, worker_model):
    base.require(all(isinstance(x, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:/-]{0,127}", x)
                     for x in (model, voice, worker_model)), "explicit_models_and_voice_required")


def prepare(output, *, model, voice, worker_model, speech_builder=None):
    validate_models(model, voice, worker_model)
    root = Path(output).absolute()
    base.require(not root.is_symlink(), "symlink_run_directory")
    root.mkdir(mode=0o700, parents=True, exist_ok=False)
    mask = os.umask(0o077)
    try:
        if speech_builder is None:
            from evals.responsive_voice.speech_assets import prepare_speech_bundle
            speech_builder = prepare_speech_bundle
        speech_builder(root / "speech", list(SCRIPT))
        manifest = {"format": FORMAT, "created": base.stamp(), "model": model, "voice": voice,
            "worker_model": worker_model, "worker_reasoning_effort": "low", "worker_limits": WORKER_LIMITS,
            "script": [list(row) for row in SCRIPT], "limits": LIMITS, "trace_scope": TRACE_SCOPE,
            "speech_manifest_sha256": base.digest(base.private_read(root / "speech/speech.json", 65536)),
            "oracle": "Initial note records eight arm raises and the blue band; optional correction changes only eight to nine. Original version stays historical.",
            "oracle_boundary": "External review only; never injected into model context.",
            "substitutions": ["macOS synthetic speech", "desktop WebRTC peer", "muted decoded PCM sink"],
            "normal_worker_timing": True, "automatic_runner_retry": False,
            "acceptance": "not_evaluated", "phone_displayed_or_heard": "not_tested"}
        base.save(root / "manifest.json", manifest, exclusive=True)
        validate_bundle(root)
        return manifest
    finally:
        os.umask(mask)


def validate_bundle(root):
    root = Path(root)
    base.require(root.is_dir() and not root.is_symlink() and not root.stat().st_mode & 0o077,
                 "private_run_directory_required")
    manifest = json.loads(base.private_read(root / "manifest.json", 65536))
    base.require(manifest.get("format") == FORMAT and manifest.get("limits") == LIMITS
        and manifest.get("script") == [list(row) for row in SCRIPT]
        and manifest.get("worker_limits") == WORKER_LIMITS and manifest.get("worker_reasoning_effort") == "low"
        and manifest.get("trace_scope") == TRACE_SCOPE and manifest.get("normal_worker_timing") is True
        and manifest.get("automatic_runner_retry") is False, "frozen_note_contract_changed")
    validate_models(manifest["model"], manifest["voice"], manifest["worker_model"])
    raw = base.private_read(root / "speech/speech.json", 65536)
    base.require(base.digest(raw) == manifest["speech_manifest_sha256"], "speech_manifest_hash_mismatch")
    speech = json.loads(raw)
    clips = speech.get("clips", [])
    base.require(speech.get("format") == "jarvis.speech_assets.v1"
        and [clip.get("id") for clip in clips] == [row[0] for row in SCRIPT], "speech_order_mismatch")
    total = 0
    for clip, (identity, text) in zip(clips, SCRIPT):
        base.require(clip.get("text") == text and clip.get("relative_path") == identity + ".wav",
                     "speech_script_or_path_mismatch")
        raw = base.private_read(root / "speech" / clip["relative_path"], 44 + 16 * base.RATE * 2)
        base.require(base.digest(raw) == clip.get("sha256"), "speech_hash_mismatch")
        with wave.open(BytesIO(raw), "rb") as source:
            count = source.getnframes()
            base.require((source.getnchannels(), source.getsampwidth(), source.getframerate(), source.getcomptype())
                == (1, 2, base.RATE, "NONE"), "speech_pcm_format_invalid")
            base.require(0 < count <= LIMITS["clip_seconds"] * base.RATE and count == clip.get("sample_count")
                and len(source.readframes(count)) == count * 2, "speech_duration_invalid")
        first, last = clip.get("first_active_sample"), clip.get("last_active_sample_exclusive")
        base.require(type(first) is int and type(last) is int and 0 <= first < last <= count,
                     "speech_activity_bounds_invalid")
        total += count
    base.require(total <= LIMITS["source_speech_seconds"] * base.RATE, "source_speech_budget")
    return manifest, clips


def notification_finished(state, *, after_frame, excluded=()):
    base.require(not state.get("fatal") and not state.get("observer_error"), "media_observation_failed")
    last_sample = max((turn.get("last_output_sample") or -1 for turn in state.get("turns", [])), default=-1)
    for identity, observed in state.get("notifications", {}).items():
        metadata = observed.get("metadata", {})
        try:
            UUID(metadata["jarvis_offer_id"])
            UUID(metadata["jarvis_attempt_id"])
            base.require(isinstance(metadata["jarvis_request_id"], str) and bool(metadata["jarvis_request_id"]),
                         "notification_request_identity")
        except (KeyError, ValueError, TypeError):
            continue
        response = state.get("responses", {}).get(identity, {})
        created = response.get("created_frame")
        if (identity not in excluded and type(created) is int and created > after_frame
                and response.get("status") == "completed" and response.get("output_observed") is True
                and response.get("has_function_calls") is False and response.get("function_call_ids") == []
                and response.get("buffer_started") is True and response.get("buffer_stopped") is True
                and last_sample >= created and state.get("audio_frame", 0) - last_sample >= base.RATE // 4):
            return observed
    return None


def worker_cleanup_confirmed(receipt):
    return (isinstance(receipt, dict) and receipt.get("status") == "closed"
        and receipt.get("worker_joined") is True and receipt.get("readiness_inactive") is True
        and receipt.get("pipes_complete") is True and receipt.get("secrets_removed") is True
        and receipt.get("forced_kill") is False and not receipt.get("errors")
        and receipt.get("returncode") in (None, 0))


def correction_fits(deadline, *, now=None):
    return deadline - (time.monotonic() if now is None else now) >= LIMITS["correction_min_remaining_seconds"]


def evidence_summary(canonical, report):
    """Join actual identities, not natural-language claims or synthetic receipts."""
    tables = canonical.get("tables", {})
    late = report.get("late_results", [])
    summary = {"publication_linked": False, "worker_overlap": False,
               "semantic_acceptance": "not_evaluated", "worker_wire": "NOT CAPTURED"}
    if not late or len(report.get("turns", [])) < 2:
        return summary
    attempt_id = late[0]["metadata"]["jarvis_attempt_id"]
    attempt = next((r for r in tables.get("jarvis_attention_attempts", []) if r["id"] == attempt_id), None)
    intent = next((r for r in tables.get("jarvis_attention_intents", [])
                   if attempt and r["id"] == attempt["intent_id"] and r["source_namespace"] == "task_result"), None)
    task = next((r for r in tables.get("jarvis_background_tasks", [])
                 if intent and r["id"] == intent["task_id"] and r.get("session_id") == report.get("session_id")), None)
    if task is None:
        return summary
    actions = {r["id"] for r in tables.get("jarvis_task_actions", []) if r["task_id"] == task["id"]
               and (r["status"] == "accepted" or r["status"] == "reconciled" and r.get("reconciled_outcome") == "accepted")}
    operations = [r for r in tables.get("jarvis_note_operations", []) if r["operation_id"] in actions]
    snapshot = report.get("note_reads", [{}])[0].get("artifact", {}) if report.get("note_reads") else {}
    matches = [r for r in operations if r["artifact_id"] == snapshot.get("artifact_id")
               and r["result_version_id"] == snapshot.get("version_id") and r.get("outcome") == "completed"]
    summary.update(task_id=task["id"], task_state=task["status"], intent_id=intent["id"],
        attention_attempt_id=attempt_id, provider_response_id=late[0]["response_id"],
        publication_linked=len(matches) == 1, publications=operations,
        distinction="Worker status, committed publication and observed native output are independent facts.")
    instant = datetime.fromisoformat(report["turns"][1]["input_utc"])
    for worker in tables.get("jarvis_task_attempts", []):
        if worker.get("task_id") != task["id"] or not worker.get("started_at") or not worker.get("ended_at"):
            continue
        if datetime.fromisoformat(worker["started_at"]) <= instant < datetime.fromisoformat(worker["ended_at"]):
            summary["worker_overlap"] = True
    return summary


def collection_complete(report):
    return (base.collection_complete(report) and worker_cleanup_confirmed(report.get("worker_cleanup"))
        and report.get("evidence", {}).get("publication_linked") is True
        and report.get("evidence", {}).get("worker_overlap") is True)


async def browser_start(root, clips, report):
    from evals.responsive_voice.synthetic_webrtc import Browser
    directory = root / "browser"
    directory.mkdir(mode=0o700)
    browser = Browser(directory, root / "speech" / clips[0]["relative_path"])
    try:
        await browser.start()
        html = Path(base.__file__).with_suffix(".html").read_text()
        await browser.cdp.evaluate("document.open();document.write(" + json.dumps(html) + ");document.close();true")
        await browser.cdp.evaluate(OBSERVER)
        # Bound from AudioContext creation, not from a later conversational turn.
        report["capture_started"] = base.stamp()
        await browser.cdp.evaluate("window.leanPrepare(" + json.dumps(LIMITS) + ")")
        for clip in clips:
            raw = base.private_read(root / "speech" / clip["relative_path"], 44 + 16 * base.RATE * 2)
            await browser.cdp.evaluate("window.leanLoad(" + json.dumps(clip["id"]) + "," +
                                      json.dumps(base64.b64encode(raw).decode()) + ")")
        return browser
    except BaseException:
        async with asyncio.timeout(20):
            report["browser_cleanup"] = await browser.close()
        raise


async def note_snapshot(http, root, report, label, *, original=None):
    async def get(path, params=None):
        response = await http.get(path, params=params)
        base.require(response.status_code == 200 and len(response.content) <= 1024 * 1024, "note_http_read_failed")
        return response.json()
    current = await get("/workspace/note")
    note = current.get("note")
    base.require(isinstance(note, dict) and isinstance(note.get("artifact"), dict), "account_note_not_present")
    artifact = note["artifact"]
    identity = str(UUID(artifact["artifact_id"]))
    history = await get("/workspace/artifacts/" + identity + "/versions",
                        {"session_id": report["session_id"], "limit": 8})
    result = {"observed": base.stamp(), "current": current, "history": history}
    if original is not None:
        base.require(original["artifact_id"] == identity, "account_note_identity_changed")
        result["original"] = await get("/workspace/artifacts/" + identity,
            {"session_id": report["session_id"], "version_id": original["version_id"]})
    base.save(root / (label + "-http.json"), result, exclusive=True)
    report["note_reads"].append({"label": label, "observed": result["observed"], "artifact": artifact})
    return artifact


async def drive_conversation(root, browser, http, report, *, deadline, control_state):
    async def media():
        base.require(control_state.get("error") is None, "app_control_failed")
        return await asyncio.wait_for(browser.cdp.evaluate("window.noteState()"), 3)

    async def play(identity):
        ordinal = len(report["turns"])
        started = await browser.cdp.evaluate("(() => {const turn=window.leanStart(" + json.dumps(identity) +
            ");return {utc:new Date().toISOString(),audio_frame:window.leanState().audio_frame,turn};})()")
        async with asyncio.timeout_at(min(deadline, time.monotonic() + LIMITS["turn_wait_seconds"])):
            while True:
                state = await media()
                if base.completed_turn(state, ordinal):
                    turn = state["turns"][ordinal]
                    input_time = datetime.fromisoformat(started["utc"]) + timedelta(
                        seconds=(turn["first_input_sample"] - started["audio_frame"]) / base.RATE)
                    report["turns"].append({**turn, "input_utc": input_time.isoformat(), "observed": base.stamp()})
                    return turn
                await asyncio.sleep(.1)

    async def late(after_frame):
        async with asyncio.timeout_at(min(deadline, time.monotonic() + LIMITS["late_wait_seconds"])):
            while True:
                result = notification_finished(await media(), after_frame=after_frame,
                    excluded={row["response_id"] for row in report["late_results"]})
                if result is not None:
                    report["late_results"].append({**result, "observed": base.stamp(),
                        "audio_boundary": "Decoded signal in the response control interval; not per-packet attribution or hearing."})
                    return result
                await asyncio.sleep(.1)

    await play("note")
    unrelated = await play("unrelated")
    await late(unrelated["last_input_sample"])
    original = await note_snapshot(http, root, report, "initial")
    report["sequence_completed"] = True
    report["correction"] = "skipped_insufficient_remaining_budget"
    base.save(root / "result.json", report)
    if correction_fits(deadline):
        report["correction"] = "attempted"
        try:
            corrected = await play("correct")
            await late(corrected["last_input_sample"])
            await note_snapshot(http, root, report, "corrected", original=original)
            report["correction"] = "captured_semantic_review_required"
        except TimeoutError:
            report["correction"] = "incomplete_at_fixed_deadline_no_retry"


async def run_session(root, info, browser, report):
    import httpx
    import websockets
    origin = info["backend_base_url"]
    base.audio_url(origin, str(UUID(int=0)))
    token = base.private_read(Path(info["bearer_token_file"]), 4096).decode().strip()
    headers = {"Authorization": "Bearer " + token, "X-Jarvis-Evaluation-Run": info["run_id"]}
    websocket = reader = session_id = None
    state = {"answer": None, "created": None, "error": None, "bytes": 0}
    stream = os.fdopen(os.open(root / "app-control.jsonl", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb")
    deadline = report["capture_started"]["monotonic_s"] + LIMITS["system_seconds"] - 2

    async def receive():
        try:
            async for raw in websocket:
                event = json.loads(raw)
                base.require(isinstance(event, dict), "app_event_shape")
                safe = {k: v for k, v in event.items() if k not in {"sdp", "authorization", "token"}}
                data = base.encode_event({**base.stamp(), "event": safe})
                state["bytes"] += len(data)
                base.require(state["bytes"] <= 4 * 1024 * 1024, "app_control_capture_bound")
                stream.write(data)
                stream.flush()
                if event.get("type") == "audio.webrtc.answer":
                    base.require(state["answer"] is None and event.get("conversation_mode") == "lean_v1", "lean_answer_required")
                    state["answer"] = event["sdp"]
                elif event.get("type") == "session.created":
                    base.require(event.get("conversation_mode") == "lean_v1" and event.get("session_id") == session_id,
                                 "lean_session_required")
                    state["created"] = safe
                elif event.get("type") == "error":
                    raise base.ReplayError("app_error")
        except Exception as error:
            state["error"] = type(error).__name__
        finally:
            state["error"] = state["error"] or "app_control_closed"

    async def wait_control(key, seconds):
        async with asyncio.timeout(seconds):
            while state[key] is None:
                base.require(state["error"] is None, "app_control_failed")
                await asyncio.sleep(.05)
            return state[key]

    async with httpx.AsyncClient(base_url=origin, headers=headers, timeout=10, trust_env=False) as http:
        try:
            async with asyncio.timeout_at(deadline):
                report["system_started"] = base.stamp()
                response = await http.post("/sessions/start", json={"resume_active": False, "title": "English note replay"})
                base.require(response.status_code == 200, "session_start_failed")
                session_id = str(UUID(response.json()["session"]["id"]))
                report["session_id"] = session_id
                offer = await browser.cdp.evaluate("window.leanOffer()")
                websocket = await websockets.connect(base.audio_url(origin, session_id), additional_headers=headers,
                    proxy=None, open_timeout=10, close_timeout=2, max_size=2 * 1024 * 1024, max_queue=64)
                reader = asyncio.create_task(receive())
                await websocket.send(json.dumps({"type": "audio.webrtc.offer", "sdp": offer["sdp"],
                    "capture_mode": "auto", "conversation_mode": "lean_v1"}))
                answer = await wait_control("answer", 20)
                await browser.cdp.evaluate("window.leanAnswer(" + json.dumps(answer) + ")")
                report["app_session"] = await wait_control("created", 10)
                budget = {"type": "session.update", "event_id": "s01_output_budget",
                    "session": {"type": "realtime", "max_output_tokens": LIMITS["max_output_tokens_per_response"]}}
                base.save(root / "client-audio-model-input.json", {"attempted": base.stamp(), "payload": budget}, exclusive=True)
                await browser.cdp.evaluate("window.leanConfigure()")
                async with asyncio.timeout(8):
                    while True:
                        media = await browser.cdp.evaluate("window.noteState()")
                        base.require(not media.get("fatal"), "media_configuration_failed")
                        if media.get("configured"):
                            report["token_configuration"] = media["configuration"]
                            break
                        await asyncio.sleep(.05)
                await drive_conversation(root, browser, http, report, deadline=deadline, control_state=state)
        finally:
            report["system_stopped"] = base.stamp()
            if websocket is not None:
                try:
                    async with asyncio.timeout(5):
                        await websocket.send('{"type":"session.close"}')
                        await websocket.close()
                except Exception:
                    report["control_close_error"] = "unconfirmed"
            if reader is not None:
                reader.cancel()
                await asyncio.gather(reader, return_exceptions=True)
            stream.close()
            if session_id is not None:
                try:
                    async with asyncio.timeout(5):
                        response = await http.post("/sessions/" + session_id + "/close")
                        base.require(response.status_code == 200, "session_close_failed")
                except Exception:
                    report["session_close_error"] = "unconfirmed"


async def execute(root, *, execute_once, hindsight_url, api_key_file, openai_key_file, hermes_python):
    base.require(execute_once is True, "execute_once_required")
    root = Path(root).absolute()
    manifest, clips = validate_bundle(root)
    base.require(all(Path(p).is_absolute() for p in (api_key_file, openai_key_file, hermes_python)), "absolute_prerequisite_paths_required")
    base.claim(root)
    report = {"format": FORMAT, "status": "failed", "acceptance": "not_evaluated", "started": base.stamp(),
        "limits": LIMITS, "trace_scope": TRACE_SCOPE, "turns": [], "late_results": [], "note_reads": [],
        "phone_displayed_or_heard": "not_tested", "normal_worker_timing": True}
    host_task = browser = worker = resource = None
    worker_attempted = False
    stop = asyncio.Event()
    mask = os.umask(0o077)
    host_root = root / "host"
    try:
        root.joinpath("wire").mkdir(mode=0o700)
        host_root.mkdir(mode=0o700)
        dsn = os.environ.get("JARVIS_SECOND_BRAIN_TEST_DB_URL")
        base.require(bool(dsn), "explicit_test_database_required")
        key = base.private_read(Path(openai_key_file), 4096).decode().strip()
        base.require(bool(key), "openai_key_file_empty")
        env = {"LEAN_VOICE_ENABLED": "true", "OPENAI_API_KEY": key,
            "OPENAI_REALTIME_MODEL": manifest["model"], "OPENAI_REALTIME_VOICE": manifest["voice"],
            "JARVIS_REALTIME_MAX_SESSION_SECONDS": str(LIMITS["system_seconds"]),
            "JARVIS_EVAL_REALTIME_WIRE_DIR": str(root / "wire")}
        with base.environment(env):
            from evals.responsive_voice.frontend_host_qualification import (
                NATIVE_MEDIA, REASONING_REMINDER, qualify_frontend_host, serve_prepared_host, cleanup_prepared_host)
            from evals.responsive_voice.native_task_worker import NativeTaskWorkerOptions, start
            from evals.responsive_voice.native_canonical_export import export
            from evals.responsive_voice.model_wire_trace import readable, render
            try:
                async with asyncio.timeout(LIMITS["prepare_host_seconds"]):
                    prepared = await qualify_frontend_host(dsn=dsn, directory=host_root,
                        hindsight_url=hindsight_url, api_key_file=api_key_file, prepare_only=True,
                        lifetime_seconds=LIMITS["host_lease_seconds"], profile=NATIVE_MEDIA,
                        workload=REASONING_REMINDER, frontier_model=manifest["worker_model"],
                        frontier_reasoning_effort="low", voice_route="realtime")
                resource = Path(prepared["resource_manifest"])
                report["resource_manifest"] = str(resource)
                base.require(prepared["status"] == "prepared", "host_preparation_failed")
                resources = json.loads(base.private_read(resource, 1024 * 1024))
                options = NativeTaskWorkerOptions(model=manifest["worker_model"], api_key_file=Path(openai_key_file),
                    hermes_python=Path(hermes_python), artifact_dir=host_root, result_hold_seconds=None)
                actual_limits = options.task_limits.model_dump()
                base.require(all(actual_limits[k] == v for k, v in WORKER_LIMITS.items()), "worker_defaults_changed")
                worker_attempted = True
                try:
                    async with asyncio.timeout(LIMITS["worker_start_seconds"]):
                        worker = await start(options, resources, dsn)
                except Exception as error:
                    report["worker_cleanup"] = getattr(error, "receipt", None)
                    raise
                report["worker_readiness"] = worker.readiness_receipt
                report["worker_limits"] = actual_limits
                ready = asyncio.get_running_loop().create_future()

                async def on_ready(info):
                    if not ready.done():
                        ready.set_result(info)

                host_task = asyncio.create_task(serve_prepared_host(manifest=resource, dsn=dsn,
                    stop_event=stop, ready_callback=on_ready, auto_cleanup=False))
                done, _ = await asyncio.wait({ready, host_task}, timeout=LIMITS["host_ready_seconds"],
                                             return_when=asyncio.FIRST_COMPLETED)
                base.require(ready in done and host_task not in done, "host_not_ready")
                info = ready.result()
                report.update(run_id=info["run_id"], provider_configuration=info.get("provider_configuration"))
                async with asyncio.timeout(LIMITS["browser_setup_seconds"]):
                    browser = await browser_start(root, clips, report)
                await run_session(root, info, browser, report)
            except Exception as error:
                report["failure"] = str(error) if isinstance(error, base.ReplayError) else type(error).__name__
            finally:
                if browser is not None:
                    try:
                        async with asyncio.timeout(LIMITS["capture_export_seconds"]):
                            media = await browser.cdp.evaluate("window.leanStop()")
                            media["notification_observer"] = await browser.cdp.evaluate("window.noteEvidence()")
                            base.save(root / "media-timeline.json", media)

                            async def chunk(start_sample, count):
                                return await browser.cdp.evaluate(f"window.leanChunk({start_sample},{count})")

                            report["output_audio"] = await base.record_output(root, media, chunk)
                            report["decoded_signal"] = media.get("decoded_signal_samples", 0)
                    except Exception as error:
                        report["capture_error"] = type(error).__name__
                    try:
                        async with asyncio.timeout(20):
                            report["browser_cleanup"] = await browser.close()
                            base.require(base.browser_cleanup_confirmed(report["browser_cleanup"]), "browser_cleanup_unconfirmed")
                    except Exception as error:
                        report["browser_cleanup_error"] = type(error).__name__
                if worker is not None:
                    try:
                        async with asyncio.timeout(LIMITS["worker_stop_seconds"]):
                            report["worker_cleanup"] = await worker.close(reason="note_replay_finished")
                    except Exception as error:
                        report["worker_cleanup_error"] = type(error).__name__
                stop.set()
                if host_task is not None:
                    try:
                        report["host_shutdown"] = await asyncio.wait_for(host_task, LIMITS["host_stop_seconds"])
                    except Exception as error:
                        report["host_stop_error"] = type(error).__name__
                if resource is None:
                    candidates = list(host_root.glob("frontend-host-resources-*.json"))
                    if len(candidates) == 1:
                        resource = candidates[0]
                        report["resource_manifest"] = str(resource)
                safe = ((not worker_attempted or worker_cleanup_confirmed(report.get("worker_cleanup")))
                    and (host_task is None or report.get("host_shutdown", {}).get("status") == "passed"))
                if resource is not None:
                    resources = json.loads(base.private_read(resource, 1024 * 1024))
                    if not safe:
                        report["cleanup"] = {"status": "blocked_unconfirmed_owned_process_shutdown"}
                    elif resources.get("database_schema") and resources.get("status") != "cleaned":
                        try:
                            async with asyncio.timeout(LIMITS["canonical_export_seconds"]):
                                canonical = await export(dsn, resources)
                                base.save(root / "canonical.json", canonical)
                                report["canonical_status"] = canonical["status"]
                                report["evidence"] = evidence_summary(canonical, report)
                        except Exception as error:
                            report["canonical_error"] = type(error).__name__
                        try:
                            async with asyncio.timeout(LIMITS["cleanup_seconds"]):
                                report["cleanup"] = await cleanup_prepared_host(manifest=resource, dsn=dsn)
                        except Exception as error:
                            report["cleanup_error"] = type(error).__name__
                    else:
                        report["cleanup"] = {"status": resources.get("status"), "evidence": resources.get("cleanup")}
                try:
                    report["wiretrace"] = render(root / "wire", root / "audio-model-wire-timeline.md")
                    path = root / "client-audio-model-input.json"
                    if path.exists():
                        lines = readable(json.loads(base.private_read(path, 65536)))
                        fd = os.open(root / "client-audio-model-input.txt", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                        with os.fdopen(fd, "w") as stream:
                            stream.write("\n".join(lines) + "\n")
                except Exception as error:
                    report["wiretrace_error"] = type(error).__name__
    except Exception as error:
        report.setdefault("failure", str(error) if isinstance(error, base.ReplayError) else type(error).__name__)
    finally:
        if collection_complete(report):
            report["status"] = "collected"
        report["finished"] = base.stamp()
        base.save(root / "result.json", report)
        os.umask(mask)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    prep = sub.add_parser("prepare", help="Local macOS speech only; no provider, browser or SQL")
    prep.add_argument("--output", type=Path, required=True)
    prep.add_argument("--model", required=True)
    prep.add_argument("--voice", required=True)
    prep.add_argument("--worker-model", required=True)
    run = sub.add_parser("execute", help="One separately authorized real-system execution")
    run.add_argument("--run-dir", type=Path, required=True)
    run.add_argument("--execute-once", action="store_true")
    run.add_argument("--hindsight-url", required=True)
    run.add_argument("--hindsight-api-key-file", type=Path, required=True)
    run.add_argument("--openai-api-key-file", type=Path, required=True)
    run.add_argument("--hermes-python", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.mode == "prepare":
        prepare(args.output, model=args.model, voice=args.voice, worker_model=args.worker_model)
        print(json.dumps({"status": "prepared", "manifest": str(args.output.absolute() / "manifest.json")}))
        return 0
    base.require(args.execute_once, "execute_once_required")
    validate_bundle(args.run_dir)
    fd = os.open(args.run_dir / "runner.log", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as log, redirect_stdout(log):
        result = asyncio.run(execute(args.run_dir, execute_once=True, hindsight_url=args.hindsight_url,
            api_key_file=args.hindsight_api_key_file, openai_key_file=args.openai_api_key_file,
            hermes_python=args.hermes_python))
    print(json.dumps({"status": result["status"], "acceptance": result["acceptance"],
                      "report": str(args.run_dir / "result.json")}))
    return 0 if result["status"] == "collected" else 1


if __name__ == "__main__":
    raise SystemExit(main())
