"""Default is offline preflight; --live-build requires a separate approved slot.

One build submission. A 900s observation deadline is NOT remote cancellation.
Use an external 960s process-group wrapper and preserve the manifest on loss.
No Sandbox.create(), config activation, retry or qualification occurs here.
"""
import argparse
import asyncio
from importlib.metadata import version
import json
import os
from pathlib import Path
import re

from template import ROOT, make_template, preflight


def save(path, record):
    temporary = path.with_suffix(".tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "w") as target:
        json.dump(record, target, sort_keys=True, indent=2)
        target.flush()
        os.fsync(target.fileno())
    temporary.replace(path)


def identity_observer(path, record):
    # Pinned SDK2.49.1 emits this one message immediately after request_build
    # returns, before hashing/uploading COPY data or triggering the build.
    # It is a message field, not a public structured-ID event. No other log
    # message is retained, interpreted or printed.
    pattern = re.compile(r"Template created with ID: ([A-Za-z0-9_-]{1,200}), Build ID: ([A-Za-z0-9_-]{1,200})")
    def observe(entry):
        message = getattr(entry, "message", None)
        if getattr(entry, "level", None) != "info" or not isinstance(message, str):
            return
        match = pattern.fullmatch(message)
        if match is None:
            return
        pair = {"template_id": match[1], "build_id": match[2]}
        if "template_id" in record and any(record[key] != value for key, value in pair.items()):
            record["identity_conflict"] = pair
            save(path, record)
            raise ValueError("build_identity_conflict")
        record.update(pair, status="identity_observed_before_upload", identity_observed_early=True)
        save(path, record)
    return observe


def read_key(args):
    sources = [value for value in (args.api_key_file, args.credential_file) if value is not None]
    if len(sources) != 1 or not sources[0].is_absolute():
        raise ValueError("one_explicit_credential_source_required")
    if args.api_key_file is not None:
        with args.api_key_file.open() as stream:
            key = stream.read(8193).strip()
    else:
        from dotenv import dotenv_values
        key = dotenv_values(args.credential_file, interpolate=False).get("E2B_API_KEY")
    if not key or len(key) > 8192:
        raise ValueError("invalid_explicit_key_file")
    return key


async def submit(args, candidate):
    if version("e2b") != "2.49.1":
        raise ValueError("sdk_version_mismatch")
    from e2b import AsyncTemplate, Template
    if (not args.manifest.is_absolute() or args.manifest.exists()
            or not re.fullmatch(r"jarvis-clip-candidate-[a-z0-9-]{6,64}", args.name)):
        raise ValueError("new_manifest_and_candidate_name_required")
    # Nothing from the credential source is put in COPY context or image env.
    key = read_key(args)
    descriptor = os.open(args.manifest, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.close(descriptor)
    record = {"format": "jarvis.clip.template-build.v1", "status": "planned", "name": args.name,
              "recipe_sha256": candidate["recipe_sha256"], "bounds": candidate["pins"]["proposed_build"],
              "runtime_qualified": False, "allocation_created": False}
    save(args.manifest, record)
    try:
        async with asyncio.timeout(900):
            opts = {"api_key": key, "request_timeout": 60, "retries": 0, "debug": False}
            if await AsyncTemplate.exists(args.name, **opts):
                raise ValueError("candidate_name_already_exists")
            template = make_template(Template)
            if preflight(require_complete=True)["recipe_sha256"] != candidate["recipe_sha256"]:
                raise ValueError("recipe_changed_before_submission")
            record["status"] = "submitting"
            save(args.manifest, record)
            info = await AsyncTemplate.build_in_background(template, name=args.name,
                cpu_count=2, memory_mb=4096, min_free_disk_mb=4096, skip_cache=True,
                on_build_logs=identity_observer(args.manifest, record), **opts)
            returned = {"template_id": info.template_id, "build_id": info.build_id}
            if not record.get("identity_observed_early") or any(record[key] != value for key, value in returned.items()):
                record["returned_identity_conflict"] = returned
                save(args.manifest, record)
                raise ValueError("returned_build_identity_mismatch")
            record.update(status="building", returned_identity_confirmed=True)
            save(args.manifest, record)
            while True:
                status = await AsyncTemplate.get_build_status(info, logs_offset=0, **opts)
                state = str(status.status.value)
                record["status"] = state
                save(args.manifest, record)
                if state == "ready":
                    if preflight(require_complete=True)["recipe_sha256"] != candidate["recipe_sha256"]:
                        raise ValueError("recipe_changed_during_build")
                    return record
                if state == "error":
                    raise ValueError("build_failed_no_retry")
                await asyncio.sleep(5)
    except BaseException as error:
        record["error"] = type(error).__name__
        record["observation_lost"] = record["status"] in {"submitting", "identity_observed_before_upload", "building", "waiting"}
        record["cleanup_required"] = record["status"] != "planned"
        save(args.manifest, record)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live-build", action="store_true")
    parser.add_argument("--name")
    credential = parser.add_mutually_exclusive_group()
    credential.add_argument("--api-key-file", type=Path)
    credential.add_argument("--credential-file", type=Path)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    try:
        candidate = preflight(require_complete=args.live_build)
        if not args.live_build:
            print(json.dumps(candidate, indent=2))
            return 0
        if not args.name or (args.api_key_file is None and args.credential_file is None) or args.manifest is None:
            raise ValueError("explicit_build_inputs_required")
        print(json.dumps(asyncio.run(submit(args, candidate)), indent=2))
        return 0
    except Exception as error:
        print(json.dumps({"passed": False, "error": type(error).__name__, "build_or_runtime_qualified": False}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
