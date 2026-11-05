#!/usr/bin/env python3
"""Bounded real retained-Hermes/E2B/Supabase media acceptance, not glasses proof.

Uses an already-running Jarvis server and native profile. Does not start services,
change the profile, delete data or install packages. Two native master turns:
create/publish a synthetic H.264+AAC MP4, then stage that JarvisFile, trim/publish.
Checks normal Files API metadata/gallery/preview, downloaded and E2B byte hashes,
and full local audio/video decoding. All generated files and receipts are kept.

Example (only after the serving owner releases this profile for testing):
  .jarvis-dev/core-test-env/bin/python scripts/retained_media_acceptance.py \
    --profile .jarvis-dev/integration-demo/profile \
    --output .jarvis-dev/retained-media-acceptance-RUN \
    --env-file backend/.env \
    --e2b-api-key-file .jarvis-dev/sandbox-view-runtime/provider-api-key \
    --execute-paid

Requires httpx, Pillow, the pinned E2B SDK, and local ffmpeg/ffprobe. Native launch
must already declare max_turns <= --max-native-turns. Wall-clock cancellation and
that native limit bound work; this script does not promise a monetary spend cap.
A missing prerequisite fails explicitly, never substitutes mocks or another path.
To continue an interrupted acceptance without recreating its source, supply all
of --resume-file-id, --resume-session-id and --resume-run-id from its receipts.
"""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
import logging
import os
from pathlib import Path
import shutil
import sqlite3
import sys
import time
from urllib.parse import urlsplit
from uuid import UUID, uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
FORMAT = "jarvis.retained_media_acceptance.v1"
TOOLS = frozenset({"list_files", "get_file_metadata", "create_text_file", "sandbox_stage_file", "sandbox_publish_file"})


class Failure(RuntimeError):
    pass


def require(condition, code):
    if not condition:
        raise Failure(code)


def private_write(path, value):
    data = (json.dumps(value, indent=2, ensure_ascii=True, default=str) + "\n").encode()
    with os.fdopen(os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600), "wb") as file:
        file.write(data)


def bounded_json(path, limit=1024 * 1024):
    require(path.is_file() and not path.is_symlink() and path.stat().st_size <= limit, "invalid_json_file")
    return json.loads(path.read_text())


class Evidence:
    def __init__(self, root):
        self.root = root
        root.mkdir(mode=0o700, parents=True, exist_ok=False)
        self.started = time.monotonic()
        self.report = {"format": FORMAT, "status": "running", "synthetic_fixture": True,
            "glasses_capture_tested": False, "voice_audio_tested": False, "artifacts": [], "turns": [],
            "deletions": [], "sandbox_preserved": True}

    def event(self, kind, **values):
        entry = {"at": datetime.now(timezone.utc).isoformat(), "elapsed_seconds": round(time.monotonic() - self.started, 3),
                 "event": kind, **values}
        with os.fdopen(os.open(self.root / "timeline.jsonl", os.O_CREAT | os.O_APPEND | os.O_WRONLY | os.O_NOFOLLOW, 0o600), "a") as file:
            file.write(json.dumps(entry, default=str) + "\n")
        print(kind, flush=True)


async def json_request(http, method, path, **kwargs):
    async with http.stream(method, path, **kwargs) as response:
        body = bytearray()
        async for block in response.aiter_bytes():
            body.extend(block)
            require(len(body) <= 2 * 1024**2, "api_json_too_large")
        if response.status_code >= 300:
            code = "unknown"
            try:
                payload = json.loads(body)
                detail = payload.get("detail", payload.get("error", payload))
                if isinstance(detail, dict):
                    candidate = detail.get("code")
                    if isinstance(candidate, str) and candidate.replace("_", "").isalnum():
                        code = candidate
            except (ValueError, AttributeError):
                pass
            raise Failure(f"api_status_{response.status_code}_{code}")
        return json.loads(body)


async def download(http, path, destination, *, params, maximum):
    checksum, size = hashlib.sha256(), 0
    async with http.stream("GET", path, params=params) as response:
        require(response.status_code == 200, f"download_status_{response.status_code}")
        with os.fdopen(os.open(destination, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600), "wb") as file:
            async for block in response.aiter_bytes():
                size += len(block)
                require(size <= maximum, "download_byte_limit")
                checksum.update(block)
                file.write(block)
        return {"sha256": checksum.hexdigest(), "size_bytes": size,
            "header_sha256": response.headers.get("x-content-sha256") or response.headers.get("x-jarvis-content-sha256"),
            "content_type": response.headers.get("content-type", "").split(";")[0]}


async def process(command, *, timeout=30, output_limit=128 * 1024):
    child = await asyncio.create_subprocess_exec(*command, stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)

    async def read(stream):
        result = bytearray()
        while block := await stream.read(4096):
            result.extend(block)
            require(len(result) <= output_limit, "decoder_output_limit")
        return bytes(result)

    readers = [asyncio.create_task(read(child.stdout)), asyncio.create_task(read(child.stderr))]
    try:
        async with asyncio.timeout(timeout):
            stdout, stderr = await asyncio.gather(*readers)
            code = await child.wait()
        return code, stdout, stderr
    finally:
        if child.returncode is None:
            child.kill()
        await child.wait()
        for task in readers:
            task.cancel()
        await asyncio.gather(*readers, return_exceptions=True)


async def decode(path, *, expected_duration, ffmpeg, ffprobe, evidence):
    code, stdout, stderr = await process([ffprobe, "-v", "error", "-protocol_whitelist", "file",
        "-count_frames", "-show_entries", "format=duration,size:stream=codec_type,codec_name,width,height,nb_read_frames",
        "-of", "json", str(path)])
    require(code == 0, "ffprobe_failed")
    probe = json.loads(stdout)
    video = [stream for stream in probe["streams"] if stream.get("codec_type") == "video"]
    audio = [stream for stream in probe["streams"] if stream.get("codec_type") == "audio"]
    require(len(video) == 1 and video[0].get("codec_name") == "h264", "expected_h264_video")
    require(len(audio) == 1 and audio[0].get("codec_name") == "aac", "expected_aac_audio")
    require(int(video[0].get("nb_read_frames", 0)) > 0, "no_decoded_video_frames")
    require(abs(float(probe["format"]["duration"]) - expected_duration) <= 0.35, "unexpected_clip_duration")
    code, _, stderr = await process([ffmpeg, "-nostdin", "-v", "error", "-xerror", "-threads", "1",
        "-protocol_whitelist", "file", "-i", str(path), "-map", "0:v:0", "-map", "0:a:0",
        "-threads", "1", "-f", "null", "-"])
    private_write(evidence.root / (path.stem + "-decode.json"),
                  {"probe": probe, "full_decode_exit_code": code, "stderr": stderr.decode(errors="replace")})
    require(code == 0, "full_audio_video_decode_failed")
    return {"duration_seconds": float(probe["format"]["duration"]),
            "video_frames": int(video[0]["nb_read_frames"]), "video_codec": "h264", "audio_codec": "aac", "full_decode": True}


def native_tool_evidence(result):
    """Keep structured tool names/arguments only, never private reasoning/history."""
    found = []

    def visit(value, depth=0):
        if depth > 10:
            return
        if isinstance(value, list):
            for item in value[:2000]:
                visit(item, depth + 1)
        elif isinstance(value, dict):
            function = value.get("function")
            if isinstance(function, dict) and function.get("name") in TOOLS:
                found.append({"name": function["name"], "arguments": function.get("arguments")})
            if isinstance(function, dict) and function.get("name") == "tool_call":
                wrapped = function.get("arguments")
                if isinstance(wrapped, str):
                    try:
                        wrapped = json.loads(wrapped)
                    except ValueError:
                        wrapped = None
                calls = wrapped.get("calls") if isinstance(wrapped, dict) else None
                if isinstance(calls, list):
                    for call in calls[:32]:
                        if isinstance(call, dict) and call.get("name") in TOOLS:
                            found.append({"name": call["name"], "arguments": call.get("arguments")})
            if value.get("type") == "tool_use" and value.get("name") in TOOLS:
                found.append({"name": value["name"], "arguments": value.get("input")})
            for key in ("messages", "tool_calls", "content", "output"):
                if key in value:
                    visit(value[key], depth + 1)
    visit(result)
    return found


async def native_turn(service, master, session_id, *, input_id, prompt, seconds, evidence):
    await service.bind_input_scope(master.logical_id, input_id=input_id, jarvis_session_id=session_id,
        allowed_tools=TOOLS, request_event_id=None, request_revision=None)
    started = time.monotonic()
    receipt = None
    terminal = False
    evidence.event("native_submit", input_id=input_id)
    private_write(evidence.root / f"{input_id}-request.json", {"input_id": input_id, "text": prompt})
    try:
        async with asyncio.timeout(seconds):
            receipt = await service.send(master.logical_id, input_id=input_id, text=prompt)
            evidence.report["turns"].append(asdict(receipt))
            last_status = None
            while True:
                value = await service.receipt(master.logical_id, receipt.admission_id)
                status = value.get("status")
                if status != last_status:
                    evidence.event("native_status", input_id=input_id, status=status)
                    last_status = status
                if status == "terminal":
                    terminal = True
                    result = value.get("result") or {}
                    selected = {"receipt": asdict(receipt), "outcome": value.get("outcome"),
                        "duration_seconds": round(time.monotonic() - started, 3),
                        "final_response": result.get("final_response") if isinstance(result, dict) else None,
                        "tool_calls": native_tool_evidence(result)}
                    private_write(evidence.root / f"{input_id}-result.json", selected)
                    require(value.get("outcome") == "completed", "native_turn_not_completed")
                    return selected
                await asyncio.sleep(1)
    finally:
        if not terminal:
            evidence.report["native_work_uncertain"] = True
            if receipt is not None:
                try:
                    async with asyncio.timeout(10):
                        cancelled = await service.client.request("prompt.cancel", session_id=master.native_id,
                                                                admission_id=receipt.admission_id)
                    evidence.report["cancel_requested"] = {"admission_id": receipt.admission_id,
                                                          "status": cancelled.get("status")}
                except Exception as error:
                    evidence.report["cancel_error_type"] = type(error).__name__
            evidence.event("native_work_needs_reconciliation", input_id=input_id,
                           admission_id=receipt.admission_id if receipt else None)


async def lookup_file(http, session_id, filename):
    result = await json_request(http, "GET", "/files", params={"session_id": str(session_id),
        "query": filename, "limit": 8, "media_only": "true"})
    matches = [item for item in result["files"] if item.get("filename") == filename and item.get("session_id") == str(session_id)]
    require(len(matches) == 1, "expected_one_published_fixture_file")
    file_id = str(UUID(matches[0]["id"]))
    detail = await json_request(http, "GET", "/files/" + file_id, params={"session_id": str(session_id)})
    return detail["file"]


async def sandbox_hash(provider, profile, owner, logical_id, path, maximum):
    from app.hermes_sessions.sandbox_provider import RetainedSandboxScope
    from app.sandbox.models import SandboxPolicy
    from app.files.transfer import FileTransferPolicy
    database = profile / "jarvis" / "sandboxes.sqlite"
    with sqlite3.connect(database.as_uri() + "?mode=ro", uri=True, timeout=2) as connection:
        connection.row_factory = sqlite3.Row
        row = connection.execute("SELECT * FROM sandboxes WHERE logical_session_id=?", (logical_id,)).fetchone()
    require(row is not None and row["sandbox_id"] and row["pending_action"] is None, "sandbox_binding_unavailable")
    policy = SandboxPolicy.model_validate_json(row["policy_json"])
    require(policy.template == "desktop", "bound_sandbox_is_not_desktop")
    scope = RetainedSandboxScope(owner_id=owner, profile_id=str(profile), logical_session_id=logical_id,
                                 allocation_id=UUID(row["allocation_id"]))
    handle = await provider.info(row["sandbox_id"], scope, policy)
    require(handle.state == "running", "sandbox_not_running")
    local = await provider.read_workspace_file_snapshot(handle, path,
        policy=FileTransferPolicy("mp4_output_v1", maximum), max_bytes=maximum)
    try:
        local._check()
        return {"sandbox_id": handle.sandbox_id, "allocation_id": str(scope.allocation_id),
                "sha256": local.content_sha256, "size_bytes": local.size_bytes, "template": policy.template}
    finally:
        local.close()


async def verify_file(http, provider, args, evidence, *, session_id, owner, master, file, path, duration, label):
    from PIL import Image
    require(file.get("content_type") == "video/mp4", "published_file_is_not_mp4")
    require(file.get("version_id"), "media_version_missing")
    source = file.get("memory_source")
    expected = file.get("content_sha256")
    require(isinstance(source, dict) and source.get("document_id") == file["version_id"],
            "source_manifest_version_mismatch")
    require(isinstance(expected, str) and len(expected) == 64
            and all(character in "0123456789abcdef" for character in expected), "source_manifest_hash_missing")
    params = {"session_id": str(session_id), "version_id": file["version_id"]}
    artifact = {"label": label, "file_id": file["id"], "version_id": file["version_id"],
                "filename": file["filename"], "publication_preview_present": file.get("thumbnail") is not None}
    evidence.report["artifacts"].append(artifact)
    private_write(evidence.root / (label + "-metadata.json"), file)
    downloaded = await download(http, "/files/" + file["id"] + "/content", evidence.root / (label + ".mp4"),
                                params=params, maximum=args.max_file_bytes)
    remote = await sandbox_hash(provider, args.profile, owner, master.logical_id, path, args.max_file_bytes)
    artifact.update(download=downloaded, sandbox_original=remote)
    require(downloaded["sha256"] == downloaded["header_sha256"] == expected == remote["sha256"], "roundtrip_hash_mismatch")
    require(downloaded["size_bytes"] == file["size_bytes"] == remote["size_bytes"], "roundtrip_size_mismatch")
    artifact["decode"] = await decode(evidence.root / (label + ".mp4"), expected_duration=duration,
        ffmpeg=args.ffmpeg, ffprobe=args.ffprobe, evidence=evidence)
    preview = file.get("thumbnail")
    require(isinstance(preview, dict), "publication_preview_missing")
    require(preview.get("path") == "/files/" + file["id"] + "/thumbnail", "invalid_thumbnail_path")
    require(preview.get("version_id") == file["version_id"], "preview_version_mismatch")
    thumbnail = await download(http, preview["path"], evidence.root / (label + "-thumbnail.jpg"),
                               params=params, maximum=65536)
    require(thumbnail["sha256"] == thumbnail["header_sha256"] == preview["content_sha256"], "preview_hash_mismatch")
    require(thumbnail["size_bytes"] == preview["size_bytes"], "preview_size_mismatch")
    with Image.open(evidence.root / (label + "-thumbnail.jpg")) as image:
        image.load()
        require(image.format == "JPEG" and image.size == (preview["width"], preview["height"]), "preview_decode_mismatch")
    artifact["preview"] = thumbnail
    evidence.event("artifact_verified", label=label, file_id=file["id"], sha256=expected)
    return artifact


async def run(args):
    import httpx
    from app.hermes_sessions.client import HermesGatewayClient
    from app.hermes_sessions.domain_service import compose_domain_work
    from app.hermes_sessions.sandbox_provider import RetainedE2BProvider
    from app.sandbox.models import E2BConfiguration
    from app.settings import JarvisSettings

    evidence = Evidence(args.output)
    session_id = None
    created_session = False
    http = None
    try:
        config = bounded_json(args.profile / "jarvis-adapter.json")
        launch = config.get("launch", {})
        require(config.get("domain_tools") is True, "profile_domain_tools_disabled")
        require(type(launch.get("max_turns")) is int and 0 < launch["max_turns"] <= args.max_native_turns,
                "profile_needs_bounded_max_turns")
        require("jarvis_domain" in launch.get("toolsets", []), "profile_domain_toolset_missing")
        require(args.ffmpeg and args.ffprobe, "local_ffmpeg_or_ffprobe_missing")
        settings = JarvisSettings(_env_file=args.env_file)
        token = args.bearer_file.read_text().strip() if args.bearer_file else settings.JARVIS_STATIC_BEARER_TOKEN
        require(token and config["owner_id"] == settings.JARVIS_OWNER_ID, "credential_owner_or_bearer_missing")
        key = args.e2b_api_key_file.read_text().strip()
        require(key, "e2b_key_missing")
        provider = RetainedE2BProvider(E2BConfiguration(api_key=key))
        http = httpx.AsyncClient(base_url=args.base_url, headers={"Authorization": "Bearer " + token},
                                timeout=30, follow_redirects=False, trust_env=False)
        run_id = args.resume_run_id or uuid4().hex[:12]
        prefix = "retained-media-" + run_id
        folder = "acceptance-" + run_id
        source_path, edited_path = folder + "/source.mp4", folder + "/edited.mp4"
        source_name, edited_name = prefix + "-source.mp4", prefix + "-edited.mp4"
        evidence.report.update(run_id=run_id, profile=str(args.profile), limits={
            "native_prompts": 1 if args.resume_file_id else 2, "native_max_turns_per_prompt": launch["max_turns"],
            "turn_seconds": args.turn_seconds, "total_seconds": args.total_seconds,
            "maximum_file_bytes": args.max_file_bytes, "monetary_cap_enforced": False})
        async with asyncio.timeout(args.total_seconds):
            async with HermesGatewayClient(args.profile, timeout=15) as native:
                bridge = compose_domain_work(args.profile, native)
                master = await bridge.service.master()
                require(args.master is None or master.logical_id == args.master, "unexpected_retained_master")
                evidence.report["master"] = master.logical_id
                snapshot = await native.snapshot(master.native_id)
                require(snapshot.get("running") is False and snapshot.get("pending") == [], "native_master_not_idle")
                if args.resume_file_id:
                    session_id = args.resume_session_id
                    evidence.report.update(resumed_from_file_id=str(args.resume_file_id), existing_session_preserved=True)
                    evidence.event("test_session_reused", session_id=str(session_id), file_id=str(args.resume_file_id))
                else:
                    started = await json_request(http, "POST", "/sessions/start", json={"title": prefix, "resume_active": False})
                    session_id = UUID(started["session"]["id"])
                    created_session = True
                    evidence.event("test_session_created", session_id=str(session_id))
                evidence.report["jarvis_session_id"] = str(session_id)
                instructions = (
                    "Run this bounded synthetic media transport acceptance in YOUR current master sandbox, not a child. "
                    "Do not touch existing files, start other jobs, enable keepalive, install packages or use personal media. "
                    "Use installed ffmpeg; if unavailable report the missing prerequisite. All paths below are relative to "
                    "the normal sandbox workspace. Use the real tools and do not claim success without their saved-file receipt. ")
                prompt = instructions + (
                    f"Create directory {folder}, then create {source_path} using: "
                    f"ffmpeg -nostdin -v error -f lavfi -i testsrc2=size=320x240:rate=15 "
                    f"-f lavfi -i sine=frequency=440:sample_rate=48000 -t 6 -c:v libx264 -preset ultrafast "
                    f"-pix_fmt yuv420p -c:a aac -b:a 64k -movflags +faststart {source_path}. "
                    f"Publish via sandbox_publish_file with path={source_path}, filename={source_name}, format=mp4, "
                    "description='Synthetic acceptance fixture, not glasses capture'. Return its exact file ID.")
                if args.resume_file_id:
                    detail = await json_request(http, "GET", "/files/" + str(args.resume_file_id),
                                                params={"session_id": str(session_id)})
                    original = detail["file"]
                    require(original.get("id") == str(args.resume_file_id)
                            and original.get("session_id") == str(session_id)
                            and original.get("filename") == source_name, "resume_source_identity_mismatch")
                else:
                    await native_turn(bridge.service, master, session_id, input_id=prefix + "-create", prompt=prompt,
                                      seconds=args.turn_seconds, evidence=evidence)
                    original = await lookup_file(http, session_id, source_name)
                first = await verify_file(http, provider, args, evidence, session_id=session_id, owner=config["owner_id"],
                    master=master, file=original, path=source_path, duration=6, label="source")
                prompt = instructions + (
                    f"Continue the SAME work. Call sandbox_stage_file for saved JarvisFile {original['id']}. "
                    "Use that tool's returned read-only input path, NOT the original workspace file. "
                    f"Trim seconds 1 through 3 into {edited_path}: ffmpeg -nostdin -v error -ss 1 "
                    f"-i RETURNED_INPUT_PATH -t 2 -map 0:v:0 -map 0:a:0 -c:v libx264 -preset ultrafast "
                    f"-pix_fmt yuv420p -c:a aac -b:a 64k -movflags +faststart {edited_path}. "
                    f"Publish via sandbox_publish_file, path={edited_path}, filename={edited_name}, format=mp4. "
                    "Return its exact file ID and the staging input path. Do not delete the source or either output.")
                second_turn = await native_turn(bridge.service, master, session_id, input_id=prefix + "-edit", prompt=prompt,
                                               seconds=args.turn_seconds, evidence=evidence)
                edited = await lookup_file(http, session_id, edited_name)
                second = await verify_file(http, provider, args, evidence, session_id=session_id, owner=config["owner_id"],
                    master=master, file=edited, path=edited_path, duration=2, label="edited")
                require(first["download"]["sha256"] != second["download"]["sha256"], "edited_bytes_unchanged")
                require(first["sandbox_original"]["sandbox_id"] == second["sandbox_original"]["sandbox_id"], "sandbox_identity_changed")
                stage_calls = [call for call in second_turn["tool_calls"] if call["name"] == "sandbox_stage_file"
                               and original["id"] in json.dumps(call["arguments"])]
                evidence.report["native_stage_call_observed"] = bool(stage_calls)
                require(stage_calls, "native_stage_call_evidence_unavailable")
                gallery = await json_request(http, "GET", "/files", params={"session_id": str(session_id),
                    "query": prefix, "media_only": "true", "limit": 8})
                private_write(evidence.root / "gallery.json", gallery)
                require({original["id"], edited["id"]}.issubset({item["id"] for item in gallery["files"]}), "gallery_missing_output")
                evidence.report["status"] = "passed"
                evidence.event("acceptance_passed")
    except BaseException as error:
        evidence.report.update(status="failed", error_type=type(error).__name__,
                               error_code=str(error) if isinstance(error, Failure) else "inspect_private_stage_receipts")
        evidence.event("acceptance_failed", error_type=type(error).__name__, error_code=evidence.report["error_code"])
    finally:
        if http is not None and session_id is not None and created_session:
            try:
                async with asyncio.timeout(10):
                    await json_request(http, "POST", f"/sessions/{session_id}/close")
                evidence.report["test_session_closed"] = True
            except Exception as error:
                evidence.report["session_close_error_type"] = type(error).__name__
        if http is not None:
            await http.aclose()
        evidence.report["elapsed_seconds"] = round(time.monotonic() - evidence.started, 3)
        private_write(evidence.root / "report.json", evidence.report)
    print(json.dumps({"status": evidence.report["status"], "report": str(evidence.root / "report.json")}))
    return 0 if evidence.report["status"] == "passed" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--profile", type=Path, default=ROOT / ".jarvis-dev/integration-demo/profile")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:9000")
    parser.add_argument("--env-file", type=Path, default=ROOT / "backend/.env")
    parser.add_argument("--bearer-file", type=Path)
    parser.add_argument("--e2b-api-key-file", type=Path, default=ROOT / ".jarvis-dev/sandbox-view-runtime/provider-api-key")
    parser.add_argument("--master")
    parser.add_argument("--resume-file-id", type=UUID)
    parser.add_argument("--resume-session-id", type=UUID)
    parser.add_argument("--resume-run-id")
    parser.add_argument("--turn-seconds", type=int, default=180)
    parser.add_argument("--total-seconds", type=int, default=480)
    parser.add_argument("--max-native-turns", type=int, default=24)
    parser.add_argument("--max-file-bytes", type=int, default=8 * 1024**2)
    parser.add_argument("--ffmpeg", default=shutil.which("ffmpeg"))
    parser.add_argument("--ffprobe", default=shutil.which("ffprobe"))
    parser.add_argument("--execute-paid", action="store_true")
    args = parser.parse_args()
    if not args.execute_paid:
        parser.error("No execution performed. Pass --execute-paid only after the serving owner releases the profile.")
    endpoint = urlsplit(args.base_url)
    require(endpoint.scheme == "https" or (endpoint.scheme == "http" and endpoint.hostname in {"localhost", "127.0.0.1", "::1"}),
            "unsafe_api_origin")
    require(not endpoint.username and not endpoint.password and not endpoint.query and not endpoint.fragment
            and endpoint.path in {"", "/"}, "invalid_api_origin")
    require(30 <= args.turn_seconds <= 300 and args.turn_seconds <= args.total_seconds <= 900, "invalid_time_budget")
    require(1 <= args.max_native_turns <= 32 and 1024 <= args.max_file_bytes <= 32 * 1024**2, "invalid_work_budget")
    resume_values = (args.resume_file_id, args.resume_session_id, args.resume_run_id)
    require(not any(resume_values) or all(resume_values), "incomplete_resume_identity")
    require(args.resume_run_id is None or (len(args.resume_run_id) == 12
            and all(character in "0123456789abcdef" for character in args.resume_run_id)), "invalid_resume_run_id")
    args.profile = args.profile.resolve(strict=True)
    args.output = args.output.absolute()
    os.umask(0o077)
    for name in ("httpx", "httpcore", "websockets", "e2b"):
        logging.getLogger(name).setLevel(logging.WARNING)
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
