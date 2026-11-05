"""S01 desktop speech -> normal Jarvis -> decoded audio/canonical evidence.

prepare only synthesizes local speech. execute requires --execute-once and burns
one attempt before any external work. No provider shortcut, automatic retry,
transcript/tool injection, presentation ACK, or semantic pass is implemented.
The existing native host still requires SQL, storage, Gemini and local Hindsight.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
from contextlib import contextmanager, redirect_stdout
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import time
from urllib.parse import urlencode, urlsplit
from uuid import UUID
import wave


ROOT = Path(__file__).resolve().parents[3]
CONTRACT = ROOT / "jarvis-side-chat/implementation/SYSTEM_INPUT_SEAM_AUDIT.md"
FORMAT = "jarvis.lean_system_replay.s01.v1"
RATE = 48000
LIMITS = {
    "system_seconds": 120, "utterances": 4, "clip_seconds": 15,
    "source_speech_seconds": 45, "turn_wait_seconds": 25,
    "response_stop_count": 16, "max_output_tokens_per_response": 1024,
    "capture_samples": 120 * RATE, "provider_calls": 1,
    "prepare_host_seconds": 300, "host_ready_seconds": 90,
    "browser_setup_seconds": 45, "capture_export_seconds": 30,
    "host_stop_seconds": 60, "canonical_export_seconds": 55,
    "cleanup_seconds": 210, "host_lease_seconds": 420,
}
IDS = ("create", "correct", "read_current", "read_original")


class ReplayError(ValueError):
    pass


def require(condition, code):
    if not condition:
        raise ReplayError(code)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def stamp():
    return {"utc": datetime.now(UTC).isoformat(), "monotonic_s": time.monotonic()}


def encode(value):
    return (json.dumps(value, ensure_ascii=True, allow_nan=False, indent=2) + "\n").encode()


def encode_event(value):
    return (json.dumps(value, ensure_ascii=True, allow_nan=False, separators=(",", ":")) + "\n").encode()


def save(path, value, *, exclusive=False):
    flags = os.O_WRONLY | os.O_CREAT | (os.O_EXCL if exclusive else os.O_TRUNC)
    fd = os.open(path, flags | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(encode(value))


def private_read(path, limit):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, "rb") as stream:
        info = os.fstat(stream.fileno())
        require(stat.S_ISREG(info.st_mode) and not info.st_mode & 0o077, "private_file_required")
        data = stream.read(limit + 1)
    require(len(data) <= limit, "file_byte_bound")
    return data


@contextmanager
def environment(values):
    previous = {key: os.environ.get(key) for key in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def read_contract(path=CONTRACT):
    data = Path(path).read_bytes()
    require(len(data) <= 65536, "contract_byte_bound")
    text = data.decode("utf-8")
    marker = "## Coordinator first-case proposal"
    require(text.count(marker) == 1, "s01_appendix_required")
    section = text.split(marker, 1)[1]
    rows = re.findall(r'^\|([1-4])\|\u201c([^\n]+?)\u201d\|([^\n]+)\|$', section, re.M)
    require([row[0] for row in rows] == ["1", "2", "3", "4"], "four_ordered_s01_utterances_required")
    return {"source": str(Path(path).resolve()), "sha256": digest(data),
            "utterances": [{"id": identity, "text": row[1], "oracle": row[2]}
                           for identity, row in zip(IDS, rows)]}


def prepare(output, *, model, voice, contract=CONTRACT, speech_builder=None):
    require(all(isinstance(x, str) and re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", x)
                for x in (model, voice)), "explicit_model_and_voice_required")
    frozen = read_contract(contract)
    root = Path(output).absolute()
    require(not root.is_symlink(), "symlink_run_directory")
    root.mkdir(mode=0o700, parents=True, exist_ok=False)
    old_mask = os.umask(0o077)
    try:
        if speech_builder is None:
            from evals.responsive_voice.speech_assets import prepare_speech_bundle
            speech_builder = prepare_speech_bundle
        speech_builder(root / "speech", [(row["id"], row["text"]) for row in frozen["utterances"]])
        manifest = {"format": FORMAT, "case_id": "S01", "created": stamp(),
            "contract": frozen, "speech_manifest_sha256": digest(private_read(root / "speech/speech.json", 65536)),
            "model": model, "voice": voice, "limits": LIMITS,
            "substitutions": ["macOS synthetic speech, not a microphone",
                "desktop WebRTC peer, not Phone", "muted decoded PCM sink, not a speaker",
                "one budget-only data-channel session.update before input"],
            "response_budget_boundary": "stop on the 16th observed response.created; not a pre-dispatch server quota",
            "oracle_boundary": "external only; never sent as context or tool JSON",
            "acceptance": "not_evaluated", "video": False}
        save(root / "manifest.json", manifest, exclusive=True)
        validate_bundle(root)
        return manifest
    finally:
        os.umask(old_mask)


def validate_bundle(root):
    root = Path(root)
    require(not root.is_symlink() and root.is_dir() and not root.stat().st_mode & 0o077,
            "private_run_directory_required")
    manifest = json.loads(private_read(root / "manifest.json", 65536))
    require(manifest.get("format") == FORMAT and manifest.get("case_id") == "S01"
            and manifest.get("limits") == LIMITS, "frozen_run_contract_changed")
    frozen = read_contract(manifest["contract"]["source"])
    require(frozen == manifest["contract"], "coordinator_contract_changed")
    raw = private_read(root / "speech/speech.json", 65536)
    require(digest(raw) == manifest["speech_manifest_sha256"], "speech_manifest_hash_mismatch")
    speech = json.loads(raw)
    require(speech.get("format") == "jarvis.speech_assets.v1", "speech_format_invalid")
    clips = speech.get("clips", [])
    require([clip.get("id") for clip in clips] == list(IDS), "speech_order_mismatch")
    total = 0
    for clip, row in zip(clips, frozen["utterances"]):
        require(clip.get("text") == row["text"] and clip.get("relative_path") == row["id"] + ".wav",
                "speech_script_or_path_mismatch")
        data = private_read(root / "speech" / clip["relative_path"], 44 + 16 * RATE * 2)
        require(digest(data) == clip.get("sha256"), "speech_hash_mismatch")
        from io import BytesIO
        with wave.open(BytesIO(data), "rb") as source:
            require((source.getnchannels(), source.getsampwidth(), source.getframerate(), source.getcomptype())
                    == (1, 2, RATE, "NONE"), "speech_pcm_format_invalid")
            count = source.getnframes()
            require(0 < count <= LIMITS["clip_seconds"] * RATE and count == clip.get("sample_count"),
                    "speech_duration_invalid")
            require(len(source.readframes(count)) == count * 2, "speech_pcm_truncated")
        require(type(clip.get("first_active_sample")) is int and type(clip.get("last_active_sample_exclusive")) is int
                and 0 <= clip["first_active_sample"] < clip["last_active_sample_exclusive"] <= count,
                "speech_activity_bounds_invalid")
        total += count
    require(total <= LIMITS["source_speech_seconds"] * RATE, "source_speech_budget")
    return manifest, clips


def claim(root):
    # A failed/ambiguous execution is still consumed. This is not domain authority.
    save(Path(root) / "execution-attempt.json", {"started": stamp(), "automatic_retry": False}, exclusive=True)


def audio_url(origin, session_id):
    parts = urlsplit(origin)
    require(parts.scheme == "http" and parts.hostname == "127.0.0.1" and parts.port
            and not parts.username and not parts.password and parts.path in {"", "/"}
            and not parts.query and not parts.fragment, "isolated_loopback_origin_required")
    identity = str(UUID(session_id))
    return "ws://" + parts.netloc + "/realtime/audio/ws?" + urlencode({
        "session_id": identity, "client_kind": "phone_audio", "audio_provider": "openai",
        "audio_transport": "webrtc", "conversation_mode": "lean_v1"})


def shared_buffer_drained(state, turn, ids):
    """A known ordered drain, not invented per-response playback receipts."""
    playback = state.get("playback", {})
    generation = turn.get("playback_generation")
    if (type(generation) is not int or generation < 1 or playback.get("generation") != generation
            or playback.get("interrupted") is not False or playback.get("uncertain") is not False
            or playback.get("error") is not None):
        return False
    responses = state.get("responses", {})
    orders = []
    for identity in ids:
        response = responses.get(identity, {})
        order = response.get("creation_order")
        if (response.get("playback_generation") != generation or type(order) is not int
                or order < 1 or orders and order <= orders[-1]):
            return False
        orders.append(order)
    if not orders or playback.get("latest_response_order") != orders[-1]:
        return False
    drain = playback.get("drain")
    if not isinstance(drain, dict):
        return False
    source = responses.get(drain.get("response_id"), {})
    sequence = drain.get("event_sequence")
    if (drain.get("generation") != generation or source.get("playback_generation") != generation
            or type(drain.get("response_order")) is not int
            or drain["response_order"] != source.get("creation_order")
            or drain["response_order"] != orders[-1] or type(sequence) is not int or sequence < 1):
        return False
    start = playback.get("last_start")
    return start is None or (isinstance(start, dict) and start.get("generation") == generation
        and type(start.get("event_sequence")) is int and start["event_sequence"] < sequence
        and type(start.get("response_order")) is int and start["response_order"] <= drain["response_order"])


def completed_turn(state, ordinal):
    require(not state.get("fatal"), "media_" + str(state.get("fatal")))
    turns = state.get("turns", [])
    if len(turns) <= ordinal:
        return False
    turn = turns[ordinal]
    ids = turn.get("response_ids", [])
    if (not turn.get("input_ended") or turn.get("last_input_sample") is None
            or not turn.get("input_item_ids") or not ids or turn.get("first_output_sample") is None
            or turn.get("last_output_sample") is None):
        return False
    responses = state.get("responses", {})
    if any(responses.get(identity, {}).get("status") != "completed"
           or responses.get(identity, {}).get("output_observed") is not True for identity in ids):
        return False
    final = responses[ids[-1]]
    # A completed ACK can still carry a tool whose domain work is pending.
    # Only an actual subsequent non-tool audio response can finish that chain.
    # Shared drain can cover a continuation with no separate started event.
    # This is control-interval/decoded evidence, not per-packet or hearing proof.
    return (final.get("has_audio_output") is True and final.get("has_function_calls") is False
            and final.get("function_call_ids") == []
            and type(final.get("created_frame")) is int
            and turn["last_output_sample"] >= final["created_frame"]
            and shared_buffer_drained(state, turn, ids)
            and state["audio_frame"] - turn["last_output_sample"] >= RATE // 4)


def browser_cleanup_confirmed(result):
    if not isinstance(result, dict) or any(key.endswith("_error") for key in result):
        return False
    media = result.get("media")
    return (isinstance(media, dict) and media.get("peer") == "closed" and media.get("context") == "closed"
            and not any(key.endswith("_error") for key in media)
            and result.get("loopback_server_stopped") is True and result.get("process_group_gone") is True
            # Browser.close normally terminates its owned Chrome with SIGTERM;
            # an observed negative return code is not itself a cleanup failure.
            and type(result.get("chrome_returncode")) is int)


def collection_complete(report):
    return (report.get("sequence_completed") and report.get("decoded_signal", 0) > 0
            and report.get("canonical_status") == "captured" and report.get("cleanup", {}).get("status") == "passed"
            and report.get("host_shutdown", {}).get("status") == "passed"
            and report.get("wiretrace", {}).get("incomplete") is False
            and browser_cleanup_confirmed(report.get("browser_cleanup"))
            and not any(key.endswith("_error") for key in report) and "failure" not in report)


async def record_output(root, metadata, fetch_chunk):
    count = metadata.get("captured_samples")
    require(type(count) is int and 0 <= count <= LIMITS["capture_samples"], "output_capture_bound")
    root = Path(root)
    raw_path, wav_path = root / "output.pcm", root / "output.wav"
    checksum = hashlib.sha256()
    with os.fdopen(os.open(raw_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb") as raw:
        with wave.open(str(wav_path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(RATE)
            for start in range(0, count, 48000):
                size = min(48000, count - start)
                encoded = await asyncio.wait_for(fetch_chunk(start, size), 3)
                block = base64.b64decode(encoded, validate=True)
                require(len(block) == size * 2, "output_chunk_size_mismatch")
                raw.write(block)
                wav.writeframesraw(block)
                checksum.update(block)
    wav_path.chmod(0o600)
    return {"pcm_path": str(raw_path), "wav_path": str(wav_path), "sample_rate_hz": RATE,
            "channels": 1, "sample_width_bytes": 2, "samples": count,
            "pcm_sha256": checksum.hexdigest(), "wav_sha256": digest(private_read(wav_path, count * 2 + 44)),
            "boundary": "decoded desktop WebRTC samples, not physical playback or hearing"}


async def browser_start(root, clips):
    from evals.responsive_voice.synthetic_webrtc import Browser
    directory = root / "browser"
    directory.mkdir(mode=0o700)
    browser = Browser(directory, root / "speech" / clips[0]["relative_path"])
    try:
        await browser.start()
        html = Path(__file__).with_suffix(".html").read_text()
        await browser.cdp.evaluate("document.open();document.write(" + json.dumps(html) + ");document.close();true")
        await browser.cdp.evaluate("window.leanPrepare(" + json.dumps(LIMITS) + ")")
        for clip in clips:
            data = private_read(root / "speech" / clip["relative_path"], 44 + 16 * RATE * 2)
            await browser.cdp.evaluate("window.leanLoad(" + json.dumps(clip["id"]) + "," +
                                       json.dumps(base64.b64encode(data).decode()) + ")")
        return browser
    except BaseException:
        await browser.close()
        raise


async def run_session(root, info, browser, report):
    import httpx
    import websockets
    token = private_read(Path(info["bearer_token_file"]), 4096).decode().strip()
    headers = {"Authorization": "Bearer " + token, "X-Jarvis-Evaluation-Run": info["run_id"]}
    origin = info["backend_base_url"]
    audio_url(origin, str(UUID(int=0)))  # Validate destination before sending credentials.
    websocket = reader = None
    session_id = None
    control = root / "app-control.jsonl"
    stream = os.fdopen(os.open(control, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb")
    state = {"answer": None, "created": None, "error": None, "bytes": 0}

    async def receive():
        try:
            async for raw in websocket:
                event = json.loads(raw)
                require(isinstance(event, dict), "app_event_shape")
                # SDP and auth never enter this evidence file. Full model/tool
                # content is captured by the existing private provider wiretrace.
                safe = {key: value for key, value in event.items() if key not in {"sdp", "authorization", "token"}}
                data = encode_event({**stamp(), "event": safe})
                state["bytes"] += len(data)
                require(state["bytes"] <= 4 * 1024 * 1024, "app_control_capture_bound")
                stream.write(data)
                stream.flush()
                if event.get("type") == "audio.webrtc.answer":
                    require(state["answer"] is None and event.get("conversation_mode") == "lean_v1", "lean_answer_required")
                    state["answer"] = event["sdp"]
                elif event.get("type") == "session.created":
                    require(event.get("conversation_mode") == "lean_v1" and event.get("session_id") == session_id,
                            "lean_session_required")
                    state["created"] = safe
                elif event.get("type") == "error":
                    raise ReplayError("app_error")
        except Exception as error:
            state["error"] = type(error).__name__
        finally:
            if state["error"] is None:
                state["error"] = "app_control_closed"

    async def wait_control(key, seconds):
        async with asyncio.timeout(seconds):
            while state[key] is None:
                require(state["error"] is None, "app_control_failed")
                await asyncio.sleep(.05)
            return state[key]

    async with httpx.AsyncClient(base_url=origin, headers=headers, timeout=10, trust_env=False) as http:
        try:
            async with asyncio.timeout(LIMITS["system_seconds"]):
                report["system_started"] = stamp()
                response = await http.post("/sessions/start", json={"resume_active": False, "title": "S01 system replay"})
                require(response.status_code == 200, "session_start_failed")
                session_id = str(UUID(response.json()["session"]["id"]))
                report["session_id"] = session_id
                offer = await browser.cdp.evaluate("window.leanOffer()")
                websocket = await websockets.connect(audio_url(origin, session_id), additional_headers=headers,
                    proxy=None, open_timeout=10, close_timeout=2, max_size=2 * 1024 * 1024, max_queue=64)
                reader = asyncio.create_task(receive())
                await websocket.send(json.dumps({"type": "audio.webrtc.offer", "sdp": offer["sdp"],
                    "capture_mode": "auto", "conversation_mode": "lean_v1"}))
                answer = await wait_control("answer", 20)
                await browser.cdp.evaluate("window.leanAnswer(" + json.dumps(answer) + ")")
                await wait_control("created", 10)
                report["app_session"] = state["created"]
                await browser.cdp.evaluate("window.leanConfigure()")
                async with asyncio.timeout(8):
                    while True:
                        media = await browser.cdp.evaluate("window.leanState()")
                        require(not media.get("fatal"), "media_configuration_failed")
                        if media.get("configured"):
                            break
                        await asyncio.sleep(.05)
                report["token_configuration"] = media["configuration"]
                for ordinal, identity in enumerate(IDS):
                    require(state["error"] is None, "app_control_failed")
                    await browser.cdp.evaluate("window.leanStart(" + json.dumps(identity) + ")")
                    async with asyncio.timeout(LIMITS["turn_wait_seconds"]):
                        while True:
                            require(state["error"] is None, "app_control_failed")
                            media = await browser.cdp.evaluate("window.leanState()")
                            if completed_turn(media, ordinal):
                                report["turns"].append({"observed": stamp(), **media["turns"][ordinal]})
                                save(root / "result.json", report)
                                break
                            await asyncio.sleep(.1)
                report["sequence_completed"] = True
        finally:
            report["system_stopped"] = stamp()
            if websocket is not None:
                try:
                    async with asyncio.timeout(5):
                        await websocket.send('{"type":"session.close"}')
                        await websocket.close()
                except Exception:
                    report["control_close_unconfirmed"] = True
            if reader is not None:
                reader.cancel()
                await asyncio.gather(reader, return_exceptions=True)
            stream.close()
            if session_id is not None:
                try:
                    async with asyncio.timeout(5):
                        response = await http.post("/sessions/" + session_id + "/close")
                        report["app_session_close_http_status"] = response.status_code
                except Exception:
                    report["app_session_close_unconfirmed"] = True


async def execute(root, *, execute_once, hindsight_url, api_key_file):
    require(execute_once is True, "execute_once_required")
    root = Path(root).absolute()
    manifest, clips = validate_bundle(root)
    claim(root)
    report = {"format": FORMAT, "case_id": "S01", "status": "failed", "acceptance": "not_evaluated",
              "started": stamp(), "turns": [], "limits": LIMITS,
              "semantic_review": "Compare recorded audio and canonical original/current versions against manifest oracle.",
              "phone_displayed": "not_tested", "provider_closure": "consult owned host diagnostics; not inferred"}
    host_task = browser = resource = None
    stop = asyncio.Event()
    old_mask = os.umask(0o077)
    root.joinpath("wire").mkdir(mode=0o700)
    env = {"LEAN_VOICE_ENABLED": "true", "OPENAI_REALTIME_MODEL": manifest["model"],
           "OPENAI_REALTIME_VOICE": manifest["voice"], "JARVIS_REALTIME_MAX_SESSION_SECONDS": "120",
           "JARVIS_EVAL_REALTIME_WIRE_DIR": str(root / "wire")}
    dsn = None
    try:
        with environment(env):
            from app.settings import JarvisSettings
            from evals.responsive_voice.frontend_host_qualification import (
                NATIVE_MEDIA, NO_WORKLOAD, qualify_frontend_host, serve_prepared_host, cleanup_prepared_host)
            from evals.responsive_voice.native_canonical_export import export
            from evals.responsive_voice.model_wire_trace import render
            dsn = os.environ.get("JARVIS_SECOND_BRAIN_TEST_DB_URL") or JarvisSettings().SUPABASE_DB_URL
            require(bool(dsn), "isolated_database_configuration_required")
            host_root = root / "host"
            host_root.mkdir(mode=0o700)
            try:
                async with asyncio.timeout(LIMITS["prepare_host_seconds"]):
                    prepared = await qualify_frontend_host(dsn=dsn, directory=host_root,
                        hindsight_url=hindsight_url, api_key_file=api_key_file, prepare_only=True,
                        lifetime_seconds=LIMITS["host_lease_seconds"], profile=NATIVE_MEDIA,
                        workload=NO_WORKLOAD, voice_route="realtime")
                resource = Path(prepared["resource_manifest"])
                report["resource_manifest"] = str(resource)
                require(prepared["status"] == "prepared", "host_preparation_failed")
                ready = asyncio.get_running_loop().create_future()

                async def on_ready(info):
                    if not ready.done():
                        ready.set_result(info)

                host_task = asyncio.create_task(serve_prepared_host(manifest=resource, dsn=dsn,
                    stop_event=stop, ready_callback=on_ready, auto_cleanup=False))
                done, _ = await asyncio.wait({ready, host_task}, timeout=LIMITS["host_ready_seconds"],
                                             return_when=asyncio.FIRST_COMPLETED)
                require(ready in done and host_task not in done, "host_not_ready")
                info = ready.result()
                report["run_id"] = info["run_id"]
                report["provider_configuration"] = info.get("provider_configuration")
                async with asyncio.timeout(LIMITS["browser_setup_seconds"]):
                    browser = await browser_start(root, clips)
                await run_session(root, info, browser, report)
            except Exception as error:
                report["failure"] = str(error) if isinstance(error, ReplayError) else type(error).__name__
            finally:
                if browser is not None:
                    try:
                        async with asyncio.timeout(LIMITS["capture_export_seconds"]):
                            media = await browser.cdp.evaluate("window.leanStop()")
                            save(root / "media-timeline.json", media)

                            async def chunk(start, count):
                                return await browser.cdp.evaluate(f"window.leanChunk({start},{count})")

                            report["output_audio"] = await record_output(root, media, chunk)
                            report["decoded_signal"] = media.get("decoded_signal_samples", 0)
                    except Exception as error:
                        report["capture_error"] = type(error).__name__
                    try:
                        async with asyncio.timeout(20):
                            report["browser_cleanup"] = await browser.close()
                            if not browser_cleanup_confirmed(report["browser_cleanup"]):
                                report["browser_cleanup_error"] = "closure_unconfirmed"
                    except Exception as error:
                        report["browser_cleanup_error"] = type(error).__name__
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
                if resource is not None:
                    resources = json.loads(private_read(resource, 1024 * 1024))
                    if resources.get("database_schema") and resources.get("status") != "cleaned":
                        try:
                            async with asyncio.timeout(LIMITS["canonical_export_seconds"]):
                                canonical = await export(dsn, resources)
                                save(root / "canonical.json", canonical)
                                report["canonical_status"] = canonical["status"]
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
                    report["wiretrace"] = render(root / "wire", root / "wire-timeline.md")
                except Exception as error:
                    report["wiretrace_error"] = type(error).__name__
    except Exception as error:
        report.setdefault("failure", str(error) if isinstance(error, ReplayError) else type(error).__name__)
    finally:
        report["finished"] = stamp()
        if collection_complete(report):
            report["status"] = "collected"
        save(root / "result.json", report)
        os.umask(old_mask)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    prep = sub.add_parser("prepare", help="Local speech generation only; no services/provider/browser")
    prep.add_argument("--output", type=Path, required=True)
    prep.add_argument("--model", required=True)
    prep.add_argument("--voice", required=True)
    run = sub.add_parser("execute", help="One explicitly authorized real-service/provider session")
    run.add_argument("--run-dir", type=Path, required=True)
    run.add_argument("--execute-once", action="store_true")
    run.add_argument("--hindsight-url", required=True)
    run.add_argument("--hindsight-api-key-file", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.mode == "prepare":
        prepare(args.output, model=args.model, voice=args.voice)
        print(json.dumps({"status": "prepared", "manifest": str(args.output.absolute() / "manifest.json")}))
        return 0
    require(args.execute_once, "execute_once_required")
    # Host launch metadata and incidental stdout are private, never the console.
    validate_bundle(args.run_dir)
    fd = os.open(args.run_dir / "runner.log", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as log, redirect_stdout(log):
        result = asyncio.run(execute(args.run_dir, execute_once=True,
            hindsight_url=args.hindsight_url, api_key_file=args.hindsight_api_key_file))
    print(json.dumps({"status": result["status"], "acceptance": result["acceptance"],
                      "report": str(args.run_dir / "result.json")}))
    return 0 if result["status"] == "collected" else 1


if __name__ == "__main__":
    raise SystemExit(main())
