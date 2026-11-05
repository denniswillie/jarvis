"""Fixed first-use permission evidence, not OS/boot attestation. No media inputs."""
import hashlib
import json
import os
import selectors
import signal
import stat
import subprocess
import sys
import time

TOOLS = ("/usr/bin/env", "/bin/sh", "/bin/bash", "/usr/bin/install", "/usr/bin/cat",
         "/usr/sbin/useradd", "/usr/sbin/groupadd", "/usr/bin/chmod", "/usr/bin/timeout",
         "/usr/bin/python3", "/usr/bin/ldd")
MAX_NODES, MAX_ELFS, MAX_CAPTURE = 25000, 160, 262144
SECONDS = 15
WORK_SECONDS = 14  # reserve bounded child reaping and compact receipt emission


class Failure(Exception):
    def __init__(self, code, path="", **details):
        self.code, self.path, self.details = code, str(path), details
        super().__init__(code)


class Guard:
    def __init__(self):
        self.deadline = time.monotonic() + WORK_SECONDS
        self.nodes = {}
        self.absent = []
        self.stage = "launch"

    def tick(self):
        if time.monotonic() >= self.deadline:
            raise Failure("internal_deadline", self.stage)

    def node(self, path):
        self.tick()
        if path in self.nodes:
            return self.nodes[path]
        if len(self.nodes) >= MAX_NODES:
            raise Failure("node_limit", path, limit=MAX_NODES)
        info = os.lstat(path)
        mode = stat.S_IMODE(info.st_mode)
        if info.st_uid != 0:
            raise Failure("uid_not_root", path, observed_uid=info.st_uid)
        if not stat.S_ISLNK(info.st_mode) and mode & 0o022:
            raise Failure("group_other_writable", path, observed_mode=f"{mode:04o}")
        kind = ("link" if stat.S_ISLNK(info.st_mode) else "directory" if stat.S_ISDIR(info.st_mode)
                else "file" if stat.S_ISREG(info.st_mode) else "unsupported")
        if kind == "unsupported":
            raise Failure("unsupported_file_type", path)
        row = {"path": path, "kind": kind, "uid": info.st_uid, "mode": mode, "size": info.st_size}
        if kind == "link":
            row["target"] = os.readlink(path)
            if len(row["target"]) > 4096:
                raise Failure("symlink_target_limit", path)
        self.nodes[path] = row
        return row

    def path(self, path, *, kind=None, executable=False, absent_ok=False):
        if not isinstance(path, str) or not path.startswith("/") or "\x00" in path or len(path) > 4096:
            raise Failure("absolute_path_required", str(path)[:160])
        pending, parts, hops = path.split("/")[1:], [], 0
        self.node("/")
        while pending:
            self.tick()
            part = pending.pop(0)
            if part in ("", "."):
                continue
            if part == "..":
                if parts:
                    parts.pop()
                continue
            candidate = "/" + "/".join(parts + [part])
            try:
                row = self.node(candidate)
            except FileNotFoundError:
                if absent_ok and not pending:
                    self.absent.append(candidate)
                    return None
                raise Failure("path_missing", candidate) from None
            if row["kind"] == "link":
                hops += 1
                if hops > 40:
                    raise Failure("symlink_cycle_or_limit", candidate)
                target = row["target"]
                if target.startswith("/"):
                    parts = []
                pending = target.split("/") + pending
                continue
            parts.append(part)
            if pending and row["kind"] != "directory":
                raise Failure("ancestor_not_directory", candidate)
        resolved = "/" + "/".join(parts)
        row = self.node(resolved)
        if kind is not None and row["kind"] != kind:
            raise Failure("wrong_file_type", resolved, expected=kind, observed=row["kind"])
        if executable and not row["mode"] & 0o111:
            raise Failure("not_executable", resolved, observed_mode=f"{row['mode']:04o}")
        return resolved


def import_paths(guard, paths, flags, cwd, version, prefix, executable):
    required = ("isolated", "no_site", "ignore_environment", "safe_path", "dont_write_bytecode")
    for flag in required:
        if not getattr(flags, flag, False):
            raise Failure("python_flag_missing", flag)
    if prefix != "/usr" or version[0] != 3:
        raise Failure("unexpected_python_prefix", prefix)
    base = f"/usr/lib/python{version[0]}.{version[1]}"
    zip_path = f"/usr/lib/python{version[0]}{version[1]}.zip"
    allowed = {base, base + "/lib-dynload", zip_path}
    if not paths or base not in paths:
        raise Failure("stdlib_import_path_missing", base)
    for path in paths:
        if (not isinstance(path, str) or path not in allowed or path == cwd
                or "site-packages" in path or "dist-packages" in path):
            raise Failure("unsafe_python_import_path", str(path)[:160])
        guard.path(path, kind="file" if path == zip_path else "directory", absent_ok=path == zip_path)
    actual = guard.path(executable, kind="file", executable=True)
    expected = guard.path("/usr/bin/python3", kind="file", executable=True)
    if actual != expected:
        raise Failure("python_executable_mismatch", actual)
    return base


def stdlib(guard, root):
    """Walk just the actual stdlib, including extensions, with a global node cap."""
    directories, visited, extensions = [guard.path(root, kind="directory")], set(), []
    while directories:
        guard.tick()
        current = directories.pop()
        if current in visited:
            continue
        visited.add(current)
        with os.scandir(current) as entries:
            for entry in entries:
                guard.tick()
                resolved = guard.path(entry.path)
                row = guard.node(resolved)
                if row["kind"] == "directory":
                    if resolved != root and not resolved.startswith(root + "/"):
                        raise Failure("stdlib_directory_escape", entry.path, resolved=resolved)
                    directories.append(resolved)
                elif resolved.endswith(".so"):
                    extensions.append(resolved)
                    if len(extensions) > MAX_ELFS:
                        raise Failure("extension_limit", root, limit=MAX_ELFS)
    return sorted(set(extensions)), len(visited)


def mapped_files(guard, text):
    files = set()
    for line in text.splitlines():
        guard.tick()
        fields = line.split(None, 5)
        if len(fields) < 5:
            raise Failure("maps_line_invalid", "/proc/self/maps")
        if len(fields) == 5 or fields[5].startswith("["):
            continue  # anonymous/kernel mappings are outside filesystem evidence
        path = fields[5]
        if not path.startswith("/") or path.endswith(" (deleted)"):
            raise Failure("maps_path_unsupported", path[:160])
        for escaped, value in ((r"\040", " "), (r"\011", "\t"), (r"\012", "\n"), (r"\134", "\\")):
            path = path.replace(escaped, value)
        files.add(guard.path(path, kind="file"))
    if not files:
        raise Failure("maps_file_evidence_missing", "/proc/self/maps")
    return files


def ldd_paths(guard, text, inputs):
    libraries = set()
    for line in text.splitlines():
        guard.tick()
        value = line.strip()
        if not value:
            continue
        if value.endswith(":") and value[:-1] in inputs:
            continue
        if value.startswith("linux-vdso.so.") or value in {"statically linked", "not a dynamic executable"}:
            if value == "not a dynamic executable":
                raise Failure("ldd_not_dynamic", value)
            continue
        if "=> not found" in value:
            raise Failure("loader_library_missing", value[:160])
        path = value.split("=>", 1)[1].strip() if "=>" in value else value
        path = path.rsplit(" (", 1)[0]
        if not path.startswith("/"):
            raise Failure("loader_output_unrecognized", value[:160])
        libraries.add(guard.path(path, kind="file"))
    if not libraries:
        raise Failure("loader_file_evidence_missing", "/usr/bin/ldd")
    return libraries


def run_ldd(guard, paths):
    """One protected ldd inspection, bounded output/time and child-group cleanup."""
    if not paths or len(paths) > MAX_ELFS or sum(map(len, paths)) > 16000:
        raise Failure("loader_input_limit", "ldd", count=len(paths))
    ldd = guard.path("/usr/bin/ldd", kind="file", executable=True)
    guard.path("/bin/bash", kind="file", executable=True)
    process = subprocess.Popen([ldd, *paths], stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, cwd="/", env={"PATH": "/usr/sbin:/usr/bin:/sbin:/bin", "HOME": "/", "LC_ALL": "C"},
        start_new_session=True, close_fds=True)
    buffers = {"stdout": bytearray(), "stderr": bytearray()}
    selector = selectors.DefaultSelector()
    deadline = min(guard.deadline - 0.5, time.monotonic() + 6)
    try:
        for name in buffers:
            stream = getattr(process, name)
            os.set_blocking(stream.fileno(), False)
            selector.register(stream, selectors.EVENT_READ, name)
        while selector.get_map():
            guard.tick()
            if time.monotonic() >= deadline:
                raise Failure("loader_inspection_timeout", "/usr/bin/ldd")
            for key, _ in selector.select(min(0.05, max(0, deadline - time.monotonic()))):
                block = os.read(key.fileobj.fileno(), 65536)
                if not block:
                    selector.unregister(key.fileobj)
                    continue
                if sum(map(len, buffers.values())) + len(block) > MAX_CAPTURE:
                    raise Failure("loader_output_limit", "/usr/bin/ldd", limit=MAX_CAPTURE)
                buffers[key.data].extend(block)
        process.wait(timeout=max(0.01, deadline - time.monotonic()))
        if process.returncode != 0:
            raise Failure("loader_inspection_nonzero", "/usr/bin/ldd", exit_code=process.returncode,
                          stderr=bytes(buffers["stderr"][:512]).decode("utf-8", errors="replace"))
        return bytes(buffers["stdout"]).decode("utf-8", errors="strict")
    finally:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=0.4)
        finally:
            selector.close()
            process.stdout.close()
            process.stderr.close()


def inspect(guard):
    guard.stage = "python_import_paths"
    base = import_paths(guard, sys.path, sys.flags, os.getcwd(), sys.version_info, sys.base_prefix, sys.executable)
    guard.stage = "system_tools"
    tools = {path: guard.path(path, kind="file", executable=True) for path in TOOLS}
    for path in ("/etc/ld.so.cache", "/etc/ld.so.preload"):
        guard.path(path, kind="file", absent_ok=True)
    guard.stage = "stdlib"
    extensions, directories = stdlib(guard, base)
    guard.stage = "mapped_loaders"
    with open("/proc/self/maps", "r") as source:
        maps = source.read(1048577)
    if len(maps) > 1048576:
        raise Failure("maps_size_limit", "/proc/self/maps")
    mapped = mapped_files(guard, maps)
    guard.stage = "other_tool_and_extension_loaders"
    elfs = sorted(set(tools.values()) - {tools["/usr/bin/ldd"]} | set(extensions))
    for path in elfs:
        guard.tick()
        with open(path, "rb") as source:
            if source.read(4) != b"\x7fELF":
                raise Failure("known_tool_not_elf", path)
    libraries = ldd_paths(guard, run_ldd(guard, elfs), set(elfs))
    guard.tick()
    rows = [guard.nodes[path] for path in sorted(guard.nodes)]
    return {"tools": tools, "stdlib_root": base, "stdlib_directory_count": directories,
            "extension_count": len(extensions), "mapped_file_count": len(mapped),
            "ldd_elf_count": len(elfs), "ldd_library_count": len(libraries),
            "metadata_node_count": len(rows), "absent_protected_parent_paths": guard.absent,
            "metadata_sha256": hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest()}


def main():
    guard = Guard()
    started = time.monotonic()
    def timeout(*_):
        raise Failure("internal_deadline", guard.stage)
    signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, WORK_SECONDS)
    record = {"format": "jarvis.renderer.system-trust.v1", "passed": False,
              "uid": os.getuid(), "python_flags": {name: getattr(sys.flags, name, None) for name in
                  ("isolated", "no_site", "ignore_environment", "safe_path", "dont_write_bytecode")},
              "cwd": os.getcwd(), "deadline_seconds": SECONDS,
              "scope": "system_tool_loader_stdlib_permission_evidence_after_root_prepare",
              "not_proven": ["kernel_or_envd_attestation", "OS_reference_content_hashes", "future_root_mutation", "renderer_readiness"]}
    try:
        record.update(inspect(guard), passed=True)
    except Failure as error:
        record["failure"] = {"code": error.code, "path": error.path, **error.details}
    except Exception as error:
        record["failure"] = {"code": type(error).__name__, "stage": guard.stage,
                             "path": str(getattr(error, "filename", "") or "")[:512]}
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
    record["elapsed_seconds"] = round(time.monotonic() - started, 3)
    encoded = json.dumps(record, sort_keys=True)
    if len(encoded.encode()) > 8192:
        record["passed"] = False
        encoded = json.dumps({"format": record["format"], "passed": False, "failure": {"code": "receipt_size_limit"}})
    print(encoded)
    return 0 if record["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
