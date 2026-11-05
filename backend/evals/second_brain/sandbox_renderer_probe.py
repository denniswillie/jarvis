"""One synthetic MR01 renderer attempt; live template binding is Core-owned.

SOURCE ONLY until a reviewed binding is supplied by trusted Python orchestration.
This module never maps logical templates or constructs a real provider. The CLI
cannot allocate. An approved caller must use a hard300s process-group wrapper:
main<=210s, cleanup<=60s, one offline VM/TTL300, one render, no retries/downloads.
Actual decode/frame/audio/playback inspection is a separate acceptance gate.
"""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import time
from typing import Awaitable, Callable

from app.files.local_file import snapshot_file
from app.files.transfer import FileTransferPolicy, MP4_CAPTURE_POLICY, MP4_OUTPUT_POLICY
from app.sandbox.file_transfer import INPUT_ROOT
from app.sandbox.models import SandboxError, SandboxPolicy
from evals.second_brain.sandbox_media_input_probe import safe_error, verify_absent

REPO = Path(__file__).resolve().parents[3]
SOURCE_PATH = REPO / ".jarvis-dev/media-qualification-20261101T232553Z-f3d2b5/retry-bt709/synthetic-30s.mp4"
SOURCE_SHA256 = "f4f885c568d5bee8310e1dea75ec5b36860c954b03dc37d2654e9dee4a001ce5"
SOURCE_SIZE = 15431453
PROPS_PATH = REPO / "media/clip-template/fixtures/mr01-props-10s.json"
PROPS_SHA256 = "1057e5837f6803a1d0605eba6a42513b198859bb957d90dff19bc9db98fcd634"
BYTE_CEILING = 20_000_000
MAIN_SECONDS, CLEANUP_SECONDS = 210, 60
JOB = "/home/jarvis/workspace/renderer-mr01"
OUTPUT = "renderer-mr01/render/output.mp4"
COMMAND_STAGES = {"verify_system_trust", "verify_installed_software", "unprivileged_software_protection", "prepare_fixed_MR01", "one_render"}
COMMAND_OUTPUT_BYTES = 16384
SYSTEM_TRUST_SOURCE = Path(__file__).with_name("sandbox_system_trust.py")
SYSTEM_TRUST_FORMAT = "jarvis.renderer.system-trust.v1"
RENDER_ENTRYPOINT = "/opt/jarvis-render/render-clip"
PROTECTED_SOFTWARE_PATHS = ("software-inventory.json", "render-clip", "clip/src/Clip.mjs",
    "clip/node_modules/@fontsource/noto-sans/600.css", "node/bin/node",
    "chrome/chrome-linux64/chrome", "clip/node_modules/@remotion/compositor-linux-x64-gnu/ffmpeg")


@dataclass(frozen=True)
class RendererBinding:
    """Trusted test-operator injection, never request JSON or authority itself.

    Core must review/create this adapter. create(probe, policy), if used, MUST
    retain Probe.create's allocation-first records through an already-bound
    provider. confirm(probe, handle) must obtain native identity/network evidence
    and normalize exactly sandbox_id/template_id/state/internet fields. SDK2.49.1
    exposes no native build_id. Build provenance is the pinned actual receipt
    plus Core's reviewed selection; installed software is measured separately.
    """
    template_id: str
    build_id: str
    software_sha256: str
    build_receipt: Path
    build_receipt_sha256: str
    confirm: Callable[..., Awaitable[dict]]
    create: Callable[..., Awaitable[object]] | None = None


def fixed_props():
    raw = PROPS_PATH.read_bytes()
    if len(raw) > 65536 or hashlib.sha256(raw).hexdigest() != PROPS_SHA256:
        raise SandboxError("renderer_fixture_changed")
    value = json.loads(raw)
    # Only the known synthetic retained-byte mapping replaces placeholder data.
    # The qualified cuts/caption/fit/audio/fps/duration stay byte-semantically the same.
    value["sources"][0].update(assetPath="inputs/" + SOURCE_SHA256 + ".mp4", contentSha256=SOURCE_SHA256)
    return value


INVENTORY_PROTECTION = r'''
import hashlib,json,os,stat
def receipt():
    root="/opt/jarvis-render/"
    path=root+"software-inventory.json"
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
    try:
        info=os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_uid!=0 or info.st_mode&0o222 or info.st_size>16000000:
            raise ValueError("inventory_protection")
        data=bytearray()
        while True:
            block=os.read(fd,65536)
            if not block: break
            data.extend(block)
            if len(data)>16000000: raise ValueError("inventory_limit")
        saved=json.loads(data)
    finally:
        os.close(fd)
    measured=hashlib.sha256(json.dumps(saved["software"],sort_keys=True,separators=(",",":")).encode()).hexdigest()
    if saved["software_sha256"]!=measured: raise ValueError("inventory_digest")
    checks=[]
    for relative in __PROTECTED_PATHS__:
        target=root+relative
        info=os.lstat(target)
        denied=False
        try:
            writable=os.open(target,os.O_WRONLY|os.O_APPEND|os.O_NOFOLLOW)
        except PermissionError:
            denied=True
        else:
            os.close(writable) # Never modify even on a failed protection check.
        checks.append({"path":relative,"root_owned":info.st_uid==0,"readonly":not bool(info.st_mode&0o222),
                       "regular":stat.S_ISREG(info.st_mode),"write_denied":denied})
    return {"software_sha256":measured,"uid":os.getuid(),"protection":checks}
print(json.dumps(receipt()))
'''.replace("__PROTECTED_PATHS__", repr(PROTECTED_SOFTWARE_PATHS))

STAGE_FIXTURE = r'''
import hashlib,json,os,stat
def stage(request):
    job="/home/jarvis/workspace/renderer-mr01"
    os.mkdir(job,0o700)
    os.mkdir(job+"/staging",0o700)
    os.mkdir(job+"/staging/inputs",0o700)
    source=os.open(request["path"],os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
    target=None
    try:
        info=os.fstat(source)
        if not stat.S_ISREG(info.st_mode) or info.st_uid!=0 or info.st_mode&0o222 or info.st_size!=request["size"]:
            raise ValueError("source_protection")
        target=os.open(job+"/staging/inputs/"+request["sha256"]+".mp4",os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o400)
        digest=hashlib.sha256(); size=0
        while True:
            block=os.read(source,65536)
            if not block: break
            size+=len(block)
            if size>request["size"] or size>20000000: raise ValueError("source_size")
            digest.update(block)
            view=memoryview(block)
            while view:
                written=os.write(target,view)
                if written<=0: raise ValueError("copy_failed")
                view=view[written:]
        os.fsync(target)
        after=os.fstat(source)
        if any(getattr(info,key)!=getattr(after,key) for key in ("st_dev","st_ino","st_mode","st_size","st_mtime_ns","st_ctime_ns")):
            raise ValueError("source_changed")
        if size!=request["size"] or digest.hexdigest()!=request["sha256"]: raise ValueError("source_digest")
    finally:
        if target is not None: os.close(target)
        os.close(source)
    encoded=json.dumps(request["props"],sort_keys=True,separators=(",",":")).encode()
    if len(encoded)>65536: raise ValueError("props_size")
    fd=os.open(job+"/props.json",os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o400)
    with os.fdopen(fd,"wb") as output:
        output.write(encoded); output.flush(); os.fsync(output.fileno())
    return {"sha256":digest.hexdigest(),"size_bytes":size,"props_sha256":hashlib.sha256(encoded).hexdigest()}
'''


def validate_binding(binding):
    if (not isinstance(binding, RendererBinding) or not callable(binding.confirm)
            or any(not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,200}", value) for value in (binding.template_id, binding.build_id))
            or any(not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value) for value in (binding.software_sha256, binding.build_receipt_sha256))
            or not isinstance(binding.build_receipt, Path) or not binding.build_receipt.is_absolute()):
        raise SandboxError("reviewed_renderer_binding_required")
    path = binding.build_receipt
    if path.resolve() != path or not path.is_file() or path.stat().st_size > 65536:
        raise SandboxError("renderer_build_receipt_invalid")
    with path.open("rb") as source:
        raw = source.read(65537)
    if len(raw) > 65536 or hashlib.sha256(raw).hexdigest() != binding.build_receipt_sha256:
        raise SandboxError("renderer_build_receipt_changed")
    receipt = json.loads(raw)
    if receipt.get("template_id") != binding.template_id or receipt.get("build_id") != binding.build_id or receipt.get("status") != "ready":
        raise SandboxError("renderer_build_receipt_mismatch")


async def retain_output(probe, local):
    """Own a declared private artifact before releasing the transport snapshot."""
    directory = probe.root / "inspection"
    directory.mkdir(mode=0o700)
    target, partial = directory / "mr01-output.mp4", directory / "mr01-output.mp4.part"
    record = {"path": str(target), "partial_path": str(partial), "status": "planned",
              "semantic_status": "uninspected", "content_sha256": local.content_sha256,
              "expected_size_bytes": local.size_bytes}
    probe.record["inspection_artifact"] = record
    probe.persist()
    digest, size = hashlib.sha256(), 0
    try:
        descriptor = os.open(partial, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(descriptor, "wb") as output:
            async for block in local.iter_chunks():
                size += len(block)
                if size > BYTE_CEILING or size > local.size_bytes:
                    raise SandboxError("renderer_output_limit")
                output.write(block)
                digest.update(block)
            output.flush()
            os.fsync(output.fileno())
            if size < 1 or size != local.size_bytes or digest.hexdigest() != local.content_sha256:
                raise SandboxError("renderer_output_digest")
            os.fchmod(output.fileno(), 0o400)
        os.link(partial, target, follow_symlinks=False)  # exclusive, never overwrite
        partial.unlink()
        record.update(status="retained_uninspected", size_bytes=size, content_sha256=digest.hexdigest())
    except BaseException as error:
        record.update(status="partial_unready", error=safe_error(error), partial_size_bytes=size,
                      partial_sha256=digest.hexdigest())
        raise
    finally:
        probe.persist()


def retain_command_result(probe, stage, result):
    """Keep fixed in-VM command diagnostics private before validating results.

    Raw output never enters console/manifest summaries. No SDK exception text,
    transport URLs, command strings or environment values are collected. Both
    channels share the command's16KiB byte ceiling, with room reserved for stderr
    when oversized output is returned. Truncation and original lengths are explicit.
    """
    if stage not in COMMAND_STAGES:
        raise SandboxError("renderer_command_stage_invalid")
    stdout = getattr(result, "stdout", "")
    stderr = getattr(result, "stderr", "")
    if not isinstance(stdout, str) or not isinstance(stderr, str):
        raise SandboxError("renderer_command_output_invalid")
    stdout, stderr = stdout.encode("utf-8", errors="replace"), stderr.encode("utf-8", errors="replace")
    out_limit = min(len(stdout), COMMAND_OUTPUT_BYTES // 2)
    err_limit = min(len(stderr), COMMAND_OUTPUT_BYTES - out_limit)
    out_limit = min(len(stdout), COMMAND_OUTPUT_BYTES - err_limit)
    # Ignore only an incomplete final UTF-8 code point at the bounded cut.
    out_text = stdout[:out_limit].decode("utf-8", errors="ignore")
    err_text = stderr[:err_limit].decode("utf-8", errors="ignore")
    lengths = {"stdout": len(stdout), "stderr": len(stderr)}
    captured = {"stdout": len(out_text.encode()), "stderr": len(err_text.encode())}
    truncated = {name: captured[name] != lengths[name] for name in lengths}
    directory = probe.root / "command-results"
    directory.mkdir(mode=0o700, exist_ok=True)
    path = directory / (stage + ".json")
    exit_code = getattr(result, "exit_code", None)
    receipt = {"format": "jarvis.renderer.command-result.v1", "stage": stage,
               "exit_code": exit_code if type(exit_code) is int else None,
               "stdout": out_text, "stderr": err_text, "original_utf8_bytes": lengths,
               "captured_utf8_bytes": captured, "truncated": truncated}
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "w") as target:
        json.dump(receipt, target, ensure_ascii=True, indent=2)
        target.flush()
        os.fsync(target.fileno())
    probe.record.setdefault("command_receipts", {})[stage] = {
        "path": str(path), "exit_code": receipt["exit_code"],
        "captured_utf8_bytes": captured, "truncated": truncated}
    probe.persist()


async def exercise(probe, args, *, binding=None, handle=None):
    """args: live=True, synthetic=True, input_path (absolute pinned source bytes).

    Caller supplies the already-bound provider through Probe. No binding is
    constructed from CLI/request fields. An optional supplied handle must already
    be registered in this same Probe; normally Core's create adapter runs after
    private preflight and delegates to Probe.create. This is a test seam, not a
    product authority, template registry, or provider implementation.
    """
    local = downloaded = None
    failure = cleanup_failure = None
    started = time.monotonic()
    probe.record.update(purpose="one_synthetic_MR01_renderer_attempt", bounds={"vm_count": 1,
        "ttl_seconds": 300, "main_seconds": MAIN_SECONDS, "cleanup_seconds": CLEANUP_SECONDS,
        "outer_seconds_required": 300, "bytes_per_input_output": BYTE_CEILING, "renders": 1})
    probe.persist()

    async def stage(name, operation):
        probe.record["phase"] = name
        probe.persist()
        before = time.monotonic()
        result = await operation
        if name in COMMAND_STAGES:
            retain_command_result(probe, name, result)
        probe.record.setdefault("stage_seconds", {})[name] = round(time.monotonic() - before, 3)
        probe.persist()
        return result

    def data(result):
        if result.exit_code != 0 or len(result.stdout) > 16384:
            raise SandboxError("renderer_probe_command_failed")
        return json.loads(result.stdout)

    try:
        async with asyncio.timeout(MAIN_SECONDS):
            validate_binding(binding)
            if not args.live or not args.synthetic or not args.input_path.is_absolute():
                raise SandboxError("explicit_synthetic_renderer_scope_required")
            props = fixed_props()
            probe.record["selected_build_provenance"] = {"template_id": binding.template_id,
                "build_id": binding.build_id, "build_receipt": str(binding.build_receipt),
                "build_receipt_sha256": binding.build_receipt_sha256,
                "selection": "Core_reviewed_binding_not_native_build_id_field",
                "expected_software_sha256": binding.software_sha256}
            local = await stage("private_preflight", snapshot_file(args.input_path, max_bytes=BYTE_CEILING,
                expected_size=SOURCE_SIZE, expected_sha256=SOURCE_SHA256))
            with SYSTEM_TRUST_SOURCE.open("rb") as source:
                trust_bytes = source.read(16001)
            if not 0 < len(trust_bytes) <= 16000:
                raise SandboxError("renderer_system_trust_source_limit")
            trust_code = trust_bytes.decode("utf-8")
            trust_command = "/usr/bin/timeout --signal=KILL 20s /usr/bin/python3 -I -S -B -c " + shlex.quote(trust_code)
            if len(trust_command) > 16000:
                raise SandboxError("renderer_system_trust_source_limit")
            probe.record["system_trust_source_sha256"] = hashlib.sha256(trust_bytes).hexdigest()
            policy = SandboxPolicy(template="base", lifetime_seconds=300, command_seconds=180, output_bytes=16384)
            if handle is None:
                if not callable(binding.create) or probe.scopes or probe.record["resources"]:
                    raise SandboxError("reviewed_renderer_creation_adapter_required")
                handle = await stage("create_prepare", binding.create(probe, policy))
            resources = probe.record["resources"]
            probe.check("one_owned_bound_allocation", len(resources) == 1 and len(probe.scopes) == 1
                and resources[0].get("sandbox_id") == handle.sandbox_id
                and resources[0]["scope"]["allocation_id"] == str(handle.scope.allocation_id)
                and handle.policy == policy and handle.template_id == binding.template_id)
            expected_binding = {"sandbox_id": handle.sandbox_id, "template_id": binding.template_id,
                                "state": "running", "allow_internet_access": False}
            confirmation = await stage("native_binding", binding.confirm(probe, handle))
            probe.check("native_template_and_offline_binding", confirmation == expected_binding
                        and confirmation.get("allow_internet_access") is False)
            probe.record["native_binding"] = expected_binding
            trust = data(await stage("verify_system_trust", probe.run(handle, trust_command)))
            probe.check("actual_system_image_protection_before_task", trust.get("format") == SYSTEM_TRUST_FORMAT
                and trust.get("passed") is True, measured=trust)
            verified = data(await stage("verify_installed_software", probe.run(handle,
                "/usr/bin/python3 /opt/jarvis-render/inventory.py --verify")))
            probe.check("actual_software_fingerprint", verified.get("software_sha256") == binding.software_sha256)
            protected = data(await stage("unprivileged_software_protection", probe.python(handle, INVENTORY_PROTECTION)))
            checks = protected.get("protection", [])
            probe.check("software_root_owned_immutable_to_renderer", protected.get("software_sha256") == binding.software_sha256
                and type(protected.get("uid")) is int and protected["uid"] != 0
                and len(checks) == len(PROTECTED_SOFTWARE_PATHS)
                and {value.get("path") for value in checks} == set(PROTECTED_SOFTWARE_PATHS)
                and all(all(value.get(field) is True for field in ("root_owned", "readonly", "regular", "write_denied")) for value in checks),
                measured=protected)
            expected = {"path": INPUT_ROOT + "/" + SOURCE_SHA256, "sha256": SOURCE_SHA256, "size_bytes": SOURCE_SIZE}
            receipt = await stage("stage_input", probe.provider.stage_input_file(handle, SOURCE_SHA256, local,
                policy=FileTransferPolicy(MP4_CAPTURE_POLICY, BYTE_CEILING)))
            probe.check("exact_staged_input", receipt == expected and type(receipt.get("size_bytes")) is int)
            request = {"path": expected["path"], "size": SOURCE_SIZE, "sha256": SOURCE_SHA256, "props": props}
            staged = data(await stage("prepare_fixed_MR01", probe.python(handle,
                STAGE_FIXTURE + "\nprint(json.dumps(stage(" + repr(request) + ")))\n")))
            props_sha = hashlib.sha256(json.dumps(props, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            probe.check("fixed_MR01_exact_source_and_props", staged == {"sha256": SOURCE_SHA256,
                        "size_bytes": SOURCE_SIZE, "props_sha256": props_sha})
            probe.record["recipe"] = {"fixture_sha256": PROPS_SHA256, "resolved_props_sha256": props_sha,
                                      "duration_frames": 240, "fps": 24}
            result = await stage("one_render", probe.run(handle,
                RENDER_ENTRYPOINT + " " + JOB + "/props.json " + JOB + "/staging "
                + JOB + "/render " + JOB + "/render/output.mp4"))
            probe.check("renderer_returned_zero_only", result.exit_code == 0)
            downloaded = await stage("freeze_download", probe.provider.read_workspace_file_snapshot(handle, OUTPUT,
                policy=FileTransferPolicy(MP4_OUTPUT_POLICY, BYTE_CEILING), max_bytes=BYTE_CEILING))
            await stage("retain_for_inspection", retain_output(probe, downloaded))
    except BaseException as error:
        failure = error
        probe.record["failure"] = {"code": safe_error(error)}
        probe.persist()
    finally:
        for name, value in (("input", local), ("output", downloaded)):
            if value is not None:
                path = value.path
                try:
                    value.close()
                except Exception as error:
                    failure = failure or error
                probe.record.setdefault("private_transport_files", {})[name] = {"closed": value._closed, "removed": not path.exists()}
        probe.persist()

        async def cleanup():
            nonlocal failure
            async with asyncio.timeout(CLEANUP_SECONDS):
                if local is not None:
                    original = None
                    try:
                        async with asyncio.timeout(5):
                            original = await snapshot_file(args.input_path, max_bytes=BYTE_CEILING,
                                expected_size=SOURCE_SIZE, expected_sha256=SOURCE_SHA256)
                            probe.check("original_hash_unchanged", True)
                    except Exception as error:
                        failure = failure or error
                        probe.record["original_postflight_error"] = safe_error(error)
                    finally:
                        if original is not None:
                            original.close()
                error = None
                try:
                    await probe.cleanup()
                except Exception as caught:
                    error = caught
                await verify_absent(probe)
                if error:
                    raise error
        task = asyncio.create_task(cleanup())
        try:
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError as error:
                failure = failure or error
                await task
        except BaseException as error:
            cleanup_failure = error
            probe.record.update(status="cleanup_unresolved", cleanup_error=safe_error(error))
        probe.record["qualification"] = {"execution_and_transfer_passed": failure is None and cleanup_failure is None,
            "semantic_passed": False, "overall_passed": False,
            "remaining_gate": "decode_and_selected_frames_audio_alignment_and_playback_inspection",
            "elapsed_seconds": round(time.monotonic() - started, 3)}
        probe.persist()
    if failure:
        raise failure
    if cleanup_failure:
        raise cleanup_failure


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args(argv)
    print(json.dumps({"live_creation_enabled": False, "error": "reviewed_Core_binding_not_supplied",
                      "source_only": True, "input_sha256": SOURCE_SHA256, "input_size": SOURCE_SIZE}))
    return 2 if args.live else 0


if __name__ == "__main__":
    raise SystemExit(main())
