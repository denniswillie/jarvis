"""One explicit synthetic media transport probe, using the existing E2B Probe.

Launch ONLY under the coordinator's manifested process-group wrapper with a
hard 300-second deadline. This module bounds main work to 210 seconds and all
postflight/cleanup to a separate 60 seconds. One base VM has a 300-second TTL;
one upload and one same-VM copy/download are allowed, each at most 20,000,000 B.
No retry, media inspection/rendering, ordinary task authority, SQL or model run.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import time

from dotenv import dotenv_values

from app.files.local_file import snapshot_file
from app.files.transfer import FileTransferPolicy, MP4_CAPTURE_POLICY, MP4_OUTPUT_POLICY
from app.sandbox.e2b_provider import E2BProvider
from app.sandbox.file_transfer import INPUT_ROOT
from app.sandbox.models import E2BConfiguration, SandboxError, SandboxPolicy
from evals.second_brain.e2b_probe import Probe

BYTE_CEILING = 20_000_000
MAIN_SECONDS = 210
CLEANUP_SECONDS = 60
OUTPUT_PATH = "media-input-roundtrip.mp4"

# Only this fixed program and a data-only retained path/manifest enter the VM.
# No local source path, caption, task text, provider URL or credential is passed.
READ_COPY = r'''
import hashlib,json,os,shutil,stat,subprocess
def inspect_and_copy(request):
    fd=os.open(request["path"],os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
    output=None
    try:
        before=os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_size!=request["size"]:
            raise ValueError("input_type_or_size")
        output=os.open("media-input-roundtrip.mp4",os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
        digest=hashlib.sha256()
        size=0
        while True:
            block=os.read(fd,65536)
            if not block:
                break
            size+=len(block)
            if size>request["size"] or size>20000000:
                raise ValueError("input_size")
            digest.update(block)
            view=memoryview(block)
            while view:
                written=os.write(output,view)
                if written<=0:
                    raise ValueError("output_write")
                view=view[written:]
        os.fsync(output)
        after=os.fstat(fd)
        if any(getattr(before,key)!=getattr(after,key) for key in
               ("st_dev","st_ino","st_mode","st_size","st_mtime_ns","st_ctime_ns")):
            raise ValueError("input_changed")
        denied=False
        try:
            writable=os.open(request["path"],os.O_WRONLY|os.O_APPEND|os.O_NOFOLLOW)
        except PermissionError:
            denied=True
        else:
            os.close(writable)  # Never modify even if protection is wrong.
        result={"path":request["path"],"sha256":digest.hexdigest(),"size_bytes":size,
                "uid":os.getuid(),"input_uid":before.st_uid,"input_mode":stat.S_IMODE(before.st_mode),
                "write_permission_denied":denied,"output_path":"media-input-roundtrip.mp4"}
    finally:
        if output is not None:
            os.close(output)
        os.close(fd)
    if request["inventory"]:
        inventory={}
        for name,flag in (("node","--version"),("npm","--version"),("ffmpeg","-version"),
                          ("ffprobe","-version"),("chromium","--version"),
                          ("chromium-browser","--version"),("google-chrome","--version")):
            path=shutil.which(name)
            item={"path":path}
            if path:
                try:
                    value=subprocess.run([path,flag],capture_output=True,text=True,timeout=2)
                    item.update(exit_code=value.returncode,version=(value.stdout or value.stderr).splitlines()[0][:200])
                except Exception:
                    item["error"]="version_unavailable"
            inventory[name]=item
        result["inventory"]=inventory
    return result
'''


def safe_error(error):
    value = getattr(error, "code", type(error).__name__)
    if isinstance(value, str) and len(value) <= 80 and value.isascii() and value.replace("_", "").isalnum():
        return value
    return "probe_failed"


def validate(args, *, new_directory):
    if not args.live or not args.synthetic:
        raise ValueError("Explicit live synthetic scope required")
    credentials = [value for value in (args.credential_file, args.api_key_file) if value is not None]
    if len(credentials) != 1:
        raise ValueError("One explicit credential source required")
    for value in (*credentials, args.manifest_dir, args.input_path):
        if not value.is_absolute() or ".." in value.parts:
            raise ValueError("Explicit absolute paths required")
    if new_directory and args.manifest_dir.exists():
        raise ValueError("Use a new manifest directory")
    if (args.media_policy not in {MP4_CAPTURE_POLICY, MP4_OUTPUT_POLICY}
            or not isinstance(args.input_sha256, str) or len(args.input_sha256) != 64
            or any(c not in "0123456789abcdef" for c in args.input_sha256)
            or type(args.deployment_limit_bytes) is not int
            or not 0 < args.deployment_limit_bytes <= BYTE_CEILING
            or type(args.input_size) is not int or not 0 < args.input_size <= args.deployment_limit_bytes):
        raise ValueError("Exact input hash/size and bounded deployment limit required")
    return FileTransferPolicy(args.media_policy, args.deployment_limit_bytes)


def read_key(args):
    """Read only the selected existing source; never fall back to environment."""
    if args.api_key_file is not None:
        with args.api_key_file.open("r") as source:
            value = source.read(8193)
        if len(value) > 8192:
            raise ValueError("Credential file exceeds bound")
        return value.strip()
    return dotenv_values(args.credential_file, interpolate=False).get("E2B_API_KEY")


async def verify_absent(probe):
    """A list result alone cannot prove that a known sandbox was removed."""
    observed = probe.record.setdefault("individual_absence", [])
    for resource in probe.record["resources"]:
        ids = set(resource.get("cleanup_observed_ids", []))
        if resource.get("sandbox_id"):
            ids.add(resource["sandbox_id"])
        for identity in sorted(ids):
            try:
                await probe.provider._request(probe.provider.sdk.AsyncSandbox.get_info, identity,
                                               **probe.provider._options())
            except Exception as error:
                code = safe_error(error)
                absent = code == "sandbox_not_found"
            else:
                code, absent = "sandbox_still_observed", False
            observed.append({"sandbox_id": identity, "absent": absent, "code": code})
            probe.persist()
            if not absent:
                raise SandboxError("probe_cleanup_unresolved")


async def exercise(probe, args):
    """Doubles may exercise control flow; only the CLI admits a real provider."""
    local = downloaded = None
    failure = cleanup_failure = None
    started = time.monotonic()
    probe.record.update(purpose="synthetic_media_transport_only", bounds={
        "vm_count": 1, "ttl_seconds": 300, "main_seconds": MAIN_SECONDS,
        "cleanup_seconds": CLEANUP_SECONDS, "required_outer_seconds": 300,
        "input_output_byte_ceiling": BYTE_CEILING, "uploads": 1, "downloads": 1})
    probe.persist()

    async def stage(name, operation):
        probe.record["phase"] = name
        probe.persist()
        before = time.monotonic()
        result = await operation
        probe.record.setdefault("stage_seconds", {})[name] = round(time.monotonic() - before, 3)
        probe.persist()
        return result

    try:
        async with asyncio.timeout(MAIN_SECONDS):
            policy = validate(args, new_directory=False)
            local = await stage("private_preflight", snapshot_file(args.input_path, max_bytes=policy.max_bytes,
                expected_size=args.input_size, expected_sha256=args.input_sha256))
            identity = hashlib.sha256(("jarvis.synthetic.media-input.v1:" + args.input_sha256
                                       + ":" + str(args.input_size)).encode()).hexdigest()
            expected = {"path": INPUT_ROOT + "/" + identity, "sha256": args.input_sha256,
                        "size_bytes": args.input_size}
            probe.record["input"] = {"identity": identity, "sha256": args.input_sha256,
                                     "size_bytes": args.input_size, "policy": policy.policy_id,
                                     "effective_limit_bytes": policy.max_bytes}
            probe.check("private_file_preflight", local.size_bytes == args.input_size
                        and local.content_sha256 == args.input_sha256)
            handle = await stage("create_prepare", probe.create(SandboxPolicy(
                template="base", lifetime_seconds=300, command_seconds=25, output_bytes=16384)))
            receipt = await stage("upload_install", probe.provider.stage_input_file(
                handle, identity, local, policy=policy))
            probe.check("exact_input_receipt", receipt == expected)
            request = {"path": expected["path"], "size": args.input_size, "inventory": args.inventory}
            code = READ_COPY + "\nprint(json.dumps(inspect_and_copy(" + repr(request) + ")))\n"
            result = await stage("unprivileged_read_copy", probe.python(handle, code))
            probe.check("read_copy_exit", result.exit_code == 0 and len(result.stdout) <= 16384)
            measured = json.loads(result.stdout)
            probe.check("unprivileged_exact_read_and_protection", all(measured.get(k) == v for k, v in expected.items())
                        and type(measured.get("size_bytes")) is int
                        and type(measured.get("uid")) is int and measured["uid"] != 0
                        and type(measured.get("input_uid")) is int and measured["input_uid"] == 0
                        and type(measured.get("input_mode")) is int and measured["input_mode"] & 0o222 == 0
                        and measured.get("write_permission_denied") is True
                        and measured.get("output_path") == OUTPUT_PATH,
                        measured=measured)
            output_policy = FileTransferPolicy(MP4_OUTPUT_POLICY, args.deployment_limit_bytes)
            downloaded = await stage("freeze_download", probe.provider.read_workspace_file_snapshot(
                handle, OUTPUT_PATH, policy=output_policy, max_bytes=output_policy.max_bytes))
            digest, size = hashlib.sha256(), 0
            async for block in downloaded.iter_chunks():
                digest.update(block)
                size += len(block)
            probe.check("downloaded_bytes_equal_original", size == args.input_size
                        and digest.hexdigest() == args.input_sha256
                        and downloaded.size_bytes == args.input_size
                        and downloaded.content_sha256 == args.input_sha256)
    except BaseException as error:
        failure = error
        probe.record["failure"] = {"code": safe_error(error)}
        probe.persist()
    finally:
        # File ownership closes on every exit, including task cancellation.
        for name, value in (("input", local), ("output", downloaded)):
            if value is not None:
                path = value.path
                try:
                    value.close()
                except Exception as error:
                    failure = failure or error
                    probe.record.setdefault("private_cleanup_errors", {})[name] = safe_error(error)
                probe.record.setdefault("private_files", {})[name] = {
                    "closed": value._closed, "removed": not path.exists()}
        probe.persist()

        async def cleanup():
            nonlocal failure
            async with asyncio.timeout(CLEANUP_SECONDS):
                if local is not None:
                    original = None
                    try:
                        async with asyncio.timeout(5):
                            original = await snapshot_file(args.input_path, max_bytes=args.deployment_limit_bytes,
                                expected_size=args.input_size, expected_sha256=args.input_sha256)
                            probe.check("original_hash_unchanged", original.content_sha256 == args.input_sha256)
                    except Exception as error:
                        probe.record["original_postflight_error"] = safe_error(error)
                        failure = failure or error
                    finally:
                        if original is not None:
                            original.close()
                # Existing Probe owns allocation-first manifests, discovery,
                # ambiguous creation and kill reconciliation. Never replace it.
                cleanup_error = None
                try:
                    await probe.cleanup()
                except Exception as error:
                    cleanup_error = error
                await verify_absent(probe)
                if cleanup_error:
                    raise cleanup_error

        task = asyncio.create_task(cleanup())
        try:
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError as error:
                failure = failure or error
                await task
        except BaseException as error:
            cleanup_failure = error
            probe.record["cleanup_error"] = safe_error(error)
            probe.record["status"] = "cleanup_unresolved"
        probe.record["qualification"] = {"passed": failure is None and cleanup_failure is None,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "scope": "synthetic_transport_only_not_capture_render_or_ordinary_task"}
        probe.persist()
    if failure is not None:
        raise failure
    if cleanup_failure is not None:
        raise cleanup_failure


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", required=True)
    parser.add_argument("--synthetic", action="store_true", required=True)
    credential = parser.add_mutually_exclusive_group(required=True)
    credential.add_argument("--credential-file", type=Path, help="Existing dotenv-format E2B_API_KEY file")
    credential.add_argument("--api-key-file", type=Path, help="Existing raw E2B API key file")
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument("--input-sha256", required=True)
    parser.add_argument("--input-size", type=int, required=True)
    parser.add_argument("--media-policy", choices=(MP4_CAPTURE_POLICY, MP4_OUTPUT_POLICY), required=True)
    parser.add_argument("--deployment-limit-bytes", type=int, required=True)
    parser.add_argument("--inventory", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    probe = None
    try:
        args = parse_args(argv)
        validate(args, new_directory=True)
        args.manifest_dir.mkdir(mode=0o700)  # exclusive new directory, no parents
        key = read_key(args)
        provider = E2BProvider(E2BConfiguration(api_key=key))
        probe = Probe(provider, args.manifest_dir)
        print(json.dumps({"manifest": str(probe.path), "outer_deadline_required_seconds": 300}), flush=True)
        asyncio.run(exercise(probe, args))
    except BaseException as error:
        if isinstance(error, SystemExit):
            return error.code
        print(json.dumps({"passed": False, "error": safe_error(error),
                          "manifest": str(probe.path) if probe else None}), flush=True)
        return 1
    print(json.dumps({"passed": True, "manifest": str(probe.path),
                      "qualification": probe.record["qualification"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
