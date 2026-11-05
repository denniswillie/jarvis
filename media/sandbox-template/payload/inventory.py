"""Protected software identity; executable version evidence is build-time only."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess

ROOT = Path("/opt/jarvis-render")
INVENTORY = "software-inventory.json"
CORE_COMMANDS = ("/usr/sbin/useradd", "/usr/sbin/groupadd", "/bin/bash", "/usr/bin/install",
                 "/usr/bin/timeout", "/usr/bin/python3")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def digest(path):
    checksum = hashlib.sha256()
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as source:
        before = os.fstat(source.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("software_hash_not_regular:" + str(path))
        while block := source.read(65536):
            checksum.update(block)
        after = os.fstat(source.fileno())
    if any(getattr(before, key) != getattr(after, key)
           for key in ("st_dev", "st_ino", "st_size", "st_mode", "st_uid", "st_mtime_ns", "st_ctime_ns")):
        raise ValueError("software_hash_changed_during_read:" + str(path))
    return checksum.hexdigest()


def protected(path, *, kind=None, readonly=False, mode=None):
    info = path.lstat()
    if info.st_uid != 0:
        raise ValueError(f"software_uid_mismatch:{path}:expected=0:observed={info.st_uid}")
    actual_mode = stat.S_IMODE(info.st_mode)
    if kind == "file" and not stat.S_ISREG(info.st_mode):
        raise ValueError("software_type_not_regular:" + str(path))
    if kind == "directory" and not stat.S_ISDIR(info.st_mode):
        raise ValueError("software_type_not_directory:" + str(path))
    if mode is not None and actual_mode != mode:
        raise ValueError(f"software_mode_mismatch:{path}:expected={mode:04o}:observed={actual_mode:04o}")
    if not stat.S_ISLNK(info.st_mode) and actual_mode & (0o222 if readonly else 0o022):
        raise ValueError(f"software_mode_unprotected:{path}:observed={actual_mode:04o}")
    if readonly and not stat.S_ISLNK(info.st_mode):
        permitted = {0o555} if stat.S_ISDIR(info.st_mode) else {0o444, 0o555}
        if actual_mode not in permitted:
            raise ValueError(f"software_mode_not_worker_readable:{path}:observed={actual_mode:04o}")
    return {"path": str(path), "uid": info.st_uid, "mode": actual_mode}


def protected_root():
    # No symlink or writable ancestor can redirect the trusted launcher/receipt.
    chain = [protected(path, kind="directory") for path in reversed(ROOT.parents)]
    chain.append(protected(ROOT, kind="directory", readonly=True))
    protected(ROOT / "render-clip", kind="file", readonly=True, mode=0o555)
    return chain


def entries(root=ROOT, *, require_root=True):
    result = []
    for path in sorted(root.rglob("*")):
        relative = str(path.relative_to(root))
        if relative == INVENTORY:
            continue  # Definition excludes only its own resulting receipt.
        info = path.lstat()
        if require_root:
            protected(path, readonly=True)
        value = {"path": relative, "mode": stat.S_IMODE(info.st_mode), "uid": info.st_uid}
        if path.is_symlink():
            if not path.resolve(strict=True).is_relative_to(root.resolve(strict=True)):
                raise ValueError("software_symlink_escape")
            value.update(kind="symlink", target=os.readlink(path))
        elif path.is_file():
            value.update(kind="file", size_bytes=info.st_size, sha256=digest(path))
        elif path.is_dir():
            value.update(kind="directory")
        else:
            raise ValueError("unsupported_software_entry")
        result.append(value)
    return result


def verify_source(root=ROOT):
    lock = json.loads((root / "source-lock.json").read_text())
    for entry in lock["files"]:
        source = root / entry["path"]
        if not source.is_file() or source.is_symlink() or digest(source) != entry["sha256"]:
            raise ValueError("qualified_source_mismatch")
        if entry["path"].startswith("font-proof/"):
            installed = root / "clip/node_modules/@fontsource/noto-sans" / entry["path"].removeprefix("font-proof/")
            if digest(installed) != entry["sha256"]:
                raise ValueError("qualified_font_mismatch")


def command(args):
    result = subprocess.run(args, check=True, capture_output=True, text=True, timeout=5,
                            env={"PATH": str(ROOT / "node/bin") + ":/usr/bin:/bin", "HOME": "/tmp"})
    return result.stdout.strip()


def validate_versions(versions):
    names = {"node", "npm", "chrome", "ffmpeg", "ffprobe", "remotion"}
    if (not isinstance(versions, dict) or set(versions) != names
            or any(not isinstance(value, str) or not 0 < len(value) <= 4096 for value in versions.values())):
        raise ValueError("inventory_build_versions_invalid")
    pins = json.loads((ROOT / "pins.json").read_text())
    if (versions["node"] != "v" + pins["node"]["version"] or versions["remotion"] != pins["remotion"]
            or pins["chrome"]["version"] not in versions["chrome"]):
        raise ValueError("inventory_build_version_pin_mismatch")


def build_versions():
    """Build evidence only. Runtime identity verification never executes these."""
    binaries = ROOT / "clip/node_modules/@remotion/compositor-linux-x64-gnu"
    versions = {"node": command([str(ROOT / "node/bin/node"), "--version"]),
                "npm": command([str(ROOT / "node/bin/npm"), "--version"]),
                "chrome": command([str(ROOT / "chrome/chrome-linux64/chrome"), "--version"]),
                "ffmpeg": command([str(binaries / "ffmpeg"), "-version"]).splitlines()[0],
                "ffprobe": command([str(binaries / "ffprobe"), "-version"]).splitlines()[0],
                "remotion": json.loads((binaries / "package.json").read_text())["version"]}
    validate_versions(versions)
    return versions


def core_identity(path):
    # Debian's /bin and python3 aliases are supported, but their exact resolved
    # file, metadata and contents remain part of the build/runtime comparison.
    alias = protected(path)
    resolved = path.resolve(strict=True)
    parents = [protected(parent, kind="directory") for parent in reversed(resolved.parents)]
    target = protected(resolved, kind="file")
    if not os.access(resolved, os.X_OK):
        raise ValueError(f"core_command_mode_not_executable:{path}:observed={target['mode']:04o}")
    return {"alias": alias, "resolved": target, "parents": parents, "sha256": digest(resolved)}


def software(*, versions):
    """Recompute immutable identity; the sole runtime subprocess is dpkg-query."""
    protection = protected_root()
    validate_versions(versions)
    return {"format": "jarvis.clip.software.v2", "platform": "linux/amd64",
            "protection": protection, "files": entries(ROOT), "versions": versions,
            "system_packages": command(["/usr/bin/dpkg-query", "-W", "-f=${Package}\t${Version}\t${Architecture}\n"]).splitlines(),
            "core_commands": {path: core_identity(Path(path)) for path in CORE_COMMANDS},
            "entrypoint_sha256": digest(ROOT / "render-clip")}


def load_receipt():
    protected_root()
    path = ROOT / INVENTORY
    protected(path, kind="file", readonly=True)
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as source:
        before = os.fstat(source.fileno())
        if before.st_size > 16000000:
            raise ValueError("inventory_receipt_size_limit")
        data = source.read(16000001)
        after = os.fstat(source.fileno())
    if len(data) > 16000000 or any(getattr(before, key) != getattr(after, key)
            for key in ("st_dev", "st_ino", "st_size", "st_mode", "st_uid", "st_mtime_ns", "st_ctime_ns")):
        raise ValueError("inventory_receipt_changed_during_read")
    saved = json.loads(data)
    if (not isinstance(saved, dict) or not isinstance(saved.get("software"), dict)
            or not isinstance(saved.get("software_sha256"), str)
            or not re.fullmatch(r"[0-9a-f]{64}", saved["software_sha256"])):
        raise ValueError("inventory_receipt_invalid")
    if hashlib.sha256(canonical(saved["software"])).hexdigest() != saved["software_sha256"]:
        raise ValueError("inventory_receipt_sha256_mismatch")
    if saved["software"].get("format") != "jarvis.clip.software.v2":
        raise ValueError("inventory_receipt_format_mismatch")
    return saved


def assert_identity(expected, observed):
    if set(expected) != set(observed):
        raise ValueError("inventory_identity_fields_mismatch")
    # Exact paths and operand names make a permission/hash failure actionable.
    for section in ("protection", "files"):
        old = {item["path"]: item for item in expected[section]}
        new = {item["path"]: item for item in observed[section]}
        if set(old) != set(new):
            raise ValueError("inventory_" + section + "_paths_mismatch")
        for path in sorted(old):
            for key in sorted(old[path].keys() | new[path].keys()):
                if old[path].get(key) != new[path].get(key):
                    raise ValueError(f"inventory_{section}_{key}_mismatch:{path}:expected={old[path].get(key)}:observed={new[path].get(key)}")
    if set(expected["core_commands"]) != set(observed["core_commands"]):
        raise ValueError("inventory_core_command_paths_mismatch")
    for path, value in expected["core_commands"].items():
        actual = observed["core_commands"][path]
        for key in ("alias", "resolved", "parents", "sha256"):
            if value.get(key) != actual.get(key):
                raise ValueError(f"inventory_core_{key}_mismatch:{path}:expected={value.get(key)}:observed={actual.get(key)}")
    for key in sorted(expected.keys() - {"protection", "files", "core_commands"}):
        if expected[key] != observed[key]:
            raise ValueError("inventory_" + key + "_mismatch")
    if expected != observed:
        raise ValueError("inventory_identity_mismatch")


def verify():
    saved = load_receipt()  # Validate protection/self-digest before reusing versions.
    observed = software(versions=saved["software"]["versions"])
    assert_identity(saved["software"], observed)
    checksum = hashlib.sha256(canonical(observed)).hexdigest()
    if checksum != saved["software_sha256"]:
        raise ValueError("inventory_software_sha256_mismatch")
    return checksum


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--verify-source", action="store_true")
    group.add_argument("--write", action="store_true")
    group.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify_source:
        verify_source()
        return
    target = ROOT / INVENTORY
    if args.write:
        if os.getuid() != 0 or target.exists():
            raise ValueError("inventory_write_requires_new_root_owned_build")
        observed = software(versions=build_versions())
        checksum = hashlib.sha256(canonical(observed)).hexdigest()
        descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o444)
        with os.fdopen(descriptor, "w") as stream:
            json.dump({"software_sha256": checksum, "software": observed,
                       "render_qualified": False}, stream, sort_keys=True, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
            os.fchmod(stream.fileno(), 0o444)  # Contract survives a restrictive build umask.
    else:
        checksum = verify()
    print(json.dumps({"software_sha256": checksum, "render_qualified": False}))


if __name__ == "__main__":
    main()
