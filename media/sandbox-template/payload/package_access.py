"""Build-only payload sealing and bounded unprivileged access gate."""
import argparse
import json
import os
from pathlib import Path
import re
import selectors
import signal
import stat
import subprocess
import time

ROOT = Path("/opt/jarvis-render")
WORKER_UID = WORKER_GID = 65534
GATE_SECONDS, OUTPUT_BYTES = 30, 16384
FORMAT = "jarvis.clip.unprivileged-build-gate.v1"


def normalized_mode(mode):
    if stat.S_ISDIR(mode):
        return 0o555
    if stat.S_ISREG(mode):
        return 0o555 if mode & 0o111 else 0o444
    if stat.S_ISLNK(mode):
        return None  # Symlink mode is not access authority; owner/target are.
    raise ValueError("payload_unsupported_type")


def seal(root=ROOT):
    if os.geteuid() != 0 or root.is_symlink() or not root.is_dir():
        raise ValueError("payload_seal_requires_root_owned_build_directory")
    paths = [root]
    for directory, folders, files in os.walk(root, followlinks=False,
            onerror=lambda error: (_ for _ in ()).throw(error)):
        paths.extend(Path(directory) / name for name in folders + files)
        if len(paths) > 30000:
            raise ValueError("payload_seal_node_limit")
    plans = []
    for path in paths:
        info = path.lstat()
        mode = normalized_mode(info.st_mode)
        if mode is None:
            target = path.resolve(strict=True)
            if not target.is_relative_to(root.resolve(strict=True)):
                raise ValueError("payload_symlink_escape:" + str(path))
        plans.append((path, mode))
    # Preserve all bytes and existing execute designations. Directories are
    # sealed last; no chmod ever follows a symlink outside the payload.
    for path, mode in sorted(plans, key=lambda item: len(item[0].parts), reverse=True):
        os.chown(path, 0, 0, follow_symlinks=False)
        if mode is not None:
            os.chmod(path, mode, follow_symlinks=False)
    return {"format": "jarvis.clip.payload-permissions.v1", "nodes": len(plans),
            "directories": sum(path.is_dir() and not path.is_symlink() for path, _ in plans),
            "regular_files": sum(mode is not None and path.is_file() for path, mode in plans)}


CHILD = r'''
import json,os,runpy,sys
identity={"uid":os.getuid(),"euid":os.geteuid(),"gid":os.getgid(),"egid":os.getegid(),"groups":os.getgroups()}
record={"format":"jarvis.clip.unprivileged-build-gate.v1",**identity,"passed":False,"render_qualified":False}
try:
    if any(identity[key]!=65534 for key in ("uid","euid","gid","egid")) or identity["groups"]:
        raise ValueError("worker_identity_not_unprivileged")
    if not (sys.flags.isolated and sys.flags.no_site and sys.flags.dont_write_bytecode):
        raise ValueError("worker_python_not_isolated")
    namespace=runpy.run_path("/opt/jarvis-render/inventory.py",run_name="jarvis_build_worker_inventory")
    checksum=namespace["verify"]()
    saved=namespace["load_receipt"]()
    root=namespace["ROOT"]
    paths=[root,root/"software-inventory.json"]+[root/entry["path"] for entry in saved["software"]["files"]]
    for path in paths:
        flags=os.R_OK|(os.X_OK if path.is_dir() or path.stat().st_mode&0o111 else 0)
        if not os.access(path,flags,effective_ids=True):
            raise ValueError("worker_payload_inaccessible:"+str(path))
    record.update(passed=True,software_sha256=checksum,access_paths_checked=len(paths))
except Exception as error:
    record["failure"]={"code":type(error).__name__,"detail":str(error)[:2048]}
print(json.dumps(record,sort_keys=True))
raise SystemExit(0 if record["passed"] else 1)
'''


def require_worker_receipt(receipt, expected_sha):
    if (not isinstance(receipt, dict) or receipt.get("format") != FORMAT
            or receipt.get("passed") is not True or receipt.get("render_qualified") is not False
            or any(type(receipt.get(key)) is not int or receipt[key] != 65534
                   for key in ("uid", "euid", "gid", "egid"))
            or receipt.get("groups") != [] or receipt.get("software_sha256") != expected_sha
            or type(receipt.get("access_paths_checked")) is not int or receipt["access_paths_checked"] < 3):
        raise ValueError("unprivileged_build_gate_receipt_invalid")


def worker_check(root=ROOT):
    if os.geteuid() != 0 or root != ROOT:
        raise ValueError("unprivileged_build_gate_requires_root_supervisor")
    with (root / "software-inventory.json").open("rb") as source:
        raw = source.read(16000001)
    if len(raw) > 16000000:
        raise ValueError("inventory_receipt_size_limit")
    expected = json.loads(raw)["software_sha256"]
    if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError("inventory_receipt_digest_invalid")
    process = subprocess.Popen(["/usr/bin/python3", "-I", "-S", "-B", "-c", CHILD],
        user=WORKER_UID, group=WORKER_GID, extra_groups=(), cwd="/", start_new_session=True,
        close_fds=True, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        env={"PATH": "/usr/sbin:/usr/bin:/sbin:/bin", "HOME": "/nonexistent", "LC_ALL": "C",
             "PYTHONDONTWRITEBYTECODE": "1"})
    selector = selectors.DefaultSelector()
    buffers = {"stdout": bytearray(), "stderr": bytearray()}
    deadline = time.monotonic() + GATE_SECONDS
    try:
        for name in buffers:
            stream = getattr(process, name)
            os.set_blocking(stream.fileno(), False)
            selector.register(stream, selectors.EVENT_READ, name)
        while selector.get_map():
            if time.monotonic() >= deadline:
                raise ValueError("unprivileged_build_gate_timeout")
            for key, _ in selector.select(min(0.05, max(0, deadline-time.monotonic()))):
                block = os.read(key.fileobj.fileno(), 65536)
                if not block:
                    selector.unregister(key.fileobj)
                    continue
                if sum(map(len, buffers.values())) + len(block) > OUTPUT_BYTES:
                    raise ValueError("unprivileged_build_gate_output_limit")
                buffers[key.data].extend(block)
        process.wait(timeout=max(0.01, deadline-time.monotonic()))
        if process.returncode != 0:
            detail = bytes(buffers["stdout"] + buffers["stderr"]).decode("utf-8", errors="replace")[:4096]
            raise ValueError("unprivileged_build_gate_failed:" + detail)
        receipt = json.loads(buffers["stdout"])
        require_worker_receipt(receipt, expected)
        return receipt
    finally:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=1)
        finally:
            selector.close()
            process.stdout.close()
            process.stderr.close()


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--seal", action="store_true")
    group.add_argument("--worker-check", action="store_true")
    args = parser.parse_args()
    try:
        result = seal() if args.seal else worker_check()
    except Exception as error:
        print(json.dumps({"format": FORMAT if args.worker_check else "jarvis.clip.payload-permissions.v1",
                          "passed": False, "failure": {"code": type(error).__name__, "detail": str(error)[:4096]},
                          "render_qualified": False}))
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
