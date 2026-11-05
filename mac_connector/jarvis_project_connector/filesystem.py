"""Pinned roots, descriptor-relative reads and immutable bounded snapshots."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
import fnmatch
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import subprocess
from uuid import uuid4

from .protocol import ConnectorError, Limits, RootRegistration


EXCLUDED = {".git", ".hg", ".svn", ".ssh", ".aws", ".azure", ".config", ".venv", "venv",
            "node_modules", "__pycache__", "build", "dist", "coverage", "target"}
SECRET_PATTERNS = (".env*", "*.pem", "*.key", "*.p12", "*.pfx", "*.keystore",
                   "*credentials*", "*secrets*", "id_rsa*", "id_ed25519*")
DIRECTORY_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def allowed_name(name: str) -> bool:
    lowered = name.casefold()
    return not (lowered.startswith(".") or lowered in EXCLUDED
                or any(fnmatch.fnmatchcase(lowered, pattern) for pattern in SECRET_PATTERNS))


def relative_parts(value: str) -> tuple[str, ...]:
    if not value or "\\" in value or any(ord(char) < 32 for char in value):
        raise ConnectorError("invalid_path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or value != "/".join(path.parts):
        raise ConnectorError("invalid_path")
    if not all(allowed_name(part) for part in path.parts):
        raise ConnectorError("excluded_path")
    return path.parts


def register_root(path: Path, label: str) -> RootRegistration:
    resolved = path.expanduser().resolve(strict=True)
    with _directory(resolved) as fd:
        info = os.fstat(fd)
    return RootRegistration(root_id=uuid4(), label=label, path=str(resolved),
                            device=info.st_dev, inode=info.st_ino)


@contextmanager
def _directory(path):
    fd = os.open(path, DIRECTORY_FLAGS)
    try:
        yield fd
    finally:
        os.close(fd)


def _stamp(value):
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns


@dataclass(frozen=True)
class Original:
    path: str
    text: str
    digest: str
    stamp: tuple

    def metadata(self):
        return {"path": self.path, "content_digest": self.digest, "characters": len(self.text),
                "bytes": len(self.text.encode("utf-8"))}


@dataclass(frozen=True)
class Snapshot:
    files: dict[str, Original]
    directories: dict[str, tuple]
    omitted: dict[str, int]
    git: dict
    snapshot_id: str
    captured_at: str


class ProjectReader:
    def __init__(self, registration: RootRegistration, *, limits: Limits | None = None):
        self.registration = registration
        self.limits = limits or Limits()

    @contextmanager
    def root(self):
        try:
            fd = os.open(self.registration.path, DIRECTORY_FLAGS)
        except FileNotFoundError as error:
            raise ConnectorError("root_unavailable") from error
        except OSError as error:
            raise ConnectorError("root_unavailable") from error
        try:
            info = os.fstat(fd)
            if (info.st_dev, info.st_ino) != (self.registration.device, self.registration.inode):
                raise ConnectorError("root_changed")
            yield fd
            try:
                current = os.stat(self.registration.path, follow_symlinks=False)
            except OSError as error:
                raise ConnectorError("root_unavailable") from error
            if not stat.S_ISDIR(current.st_mode) or (current.st_dev, current.st_ino) != (
                    self.registration.device, self.registration.inode):
                raise ConnectorError("root_changed")
        finally:
            os.close(fd)

    @contextmanager
    def parent(self, root_fd, parts):
        fd = os.dup(root_fd)
        try:
            for part in parts[:-1]:
                self.exact_name(fd, part)
                child = os.open(part, DIRECTORY_FLAGS, dir_fd=fd)
                os.close(fd)
                fd = child
            yield fd
        finally:
            os.close(fd)

    def exact_name(self, directory_fd, name):
        with os.scandir(directory_fd) as entries:
            for index, entry in enumerate(entries):
                if index >= self.limits.entries:
                    raise ConnectorError("directory_scan_limit")
                if entry.name == name:
                    return
        raise ConnectorError("source_unavailable")

    def _read(self, root_fd, path):
        parts = relative_parts(path)
        with self.parent(root_fd, parts) as parent:
            self.exact_name(parent, parts[-1])
            fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC, dir_fd=parent)
            try:
                before = os.fstat(fd)
                if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
                    raise ConnectorError("excluded_file_type")
                if before.st_size > self.limits.file_bytes:
                    raise ConnectorError("file_too_large")
                chunks, remaining = [], self.limits.file_bytes + 1
                while remaining:
                    block = os.read(fd, min(remaining, 65536))
                    if not block:
                        break
                    chunks.append(block)
                    remaining -= len(block)
                raw = b"".join(chunks)
                if len(raw) > self.limits.file_bytes:
                    raise ConnectorError("file_too_large")
                after = os.fstat(fd)
                linked = os.stat(parts[-1], dir_fd=parent, follow_symlinks=False)
                if _stamp(before) != _stamp(after) or _stamp(after) != _stamp(linked):
                    raise ConnectorError("source_changed")
            finally:
                os.close(fd)
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ConnectorError("binary_or_non_utf8") from error
        if "\x00" in text:
            raise ConnectorError("binary_or_non_utf8")
        return Original(path, text, digest(text), _stamp(after))

    def read(self, path: str, expected_digest: str | None = None) -> Original:
        try:
            with self.root() as fd:
                original = self._read(fd, path)
                if expected_digest is not None and original.digest != expected_digest:
                    raise ConnectorError("source_changed")
                return original
        except OSError as error:
            raise ConnectorError("source_unavailable") from error

    def git_state(self):
        executable = shutil.which("git")
        if executable is None:
            return {"available": False, "reason": "git_unavailable"}
        environment = {"PATH": "/usr/bin:/bin", "GIT_CONFIG_NOSYSTEM": "1",
                       "GIT_CONFIG_GLOBAL": os.devnull, "GIT_TERMINAL_PROMPT": "0",
                       "GIT_OPTIONAL_LOCKS": "0", "LC_ALL": "C"}
        base = [executable, "--no-pager", "--no-optional-locks", "-c", "core.fsmonitor=false",
                "-c", "core.hooksPath=" + os.devnull, "-C", self.registration.path]
        def run(arguments):
            result = subprocess.run(base + arguments, env=environment, stdout=subprocess.PIPE,
                                    stderr=subprocess.DEVNULL, timeout=5, check=False)
            if len(result.stdout) > 1048576:
                raise ConnectorError("git_output_limit")
            return result.returncode, result.stdout
        try:
            code, top = run(["rev-parse", "--show-toplevel"])
            if code or Path(os.fsdecode(top).strip()).resolve() != Path(self.registration.path):
                return {"available": False, "reason": "not_a_repository_root"}
            _, commit = run(["rev-parse", "--verify", "HEAD"])
            _, branch = run(["symbolic-ref", "--quiet", "--short", "HEAD"])
            code, status_bytes = run(["status", "--porcelain=v1", "-z", "--untracked-files=all"])
            if code:
                raise ConnectorError("git_status_unavailable")
            changes, fields, index = [], status_bytes.split(b"\0"), 0
            while index < len(fields) and fields[index]:
                row = os.fsdecode(fields[index]); index += 1
                state, path = row[:2], row[3:]
                old_path = None
                if "R" in state or "C" in state:
                    old_path = os.fsdecode(fields[index]); index += 1
                try:
                    relative_parts(path)
                    if old_path is not None:
                        relative_parts(old_path)
                except ConnectorError:
                    continue
                changes.append({"path": path, "status": state, "previous_path": old_path})
            return {"available": True, "commit": commit.decode("ascii").strip() or None,
                    "branch": branch.decode("utf-8").strip() or None,
                    "dirty": bool(status_bytes), "changes": changes,
                    "coverage": "Git paths filtered by the same local source policy."}
        except (OSError, subprocess.TimeoutExpired, UnicodeError, IndexError) as error:
            raise ConnectorError("git_status_unavailable") from error

    def snapshot(self):
        files, directories, omitted = {}, {}, {}
        total, entries, stopped = 0, 0, False
        def omit(reason):
            omitted[reason] = omitted.get(reason, 0) + 1
        with self.root() as root_fd:
            git = self.git_state()
            def walk(fd, prefix, depth):
                nonlocal total, entries, stopped
                directories[prefix] = _stamp(os.fstat(fd))
                if depth >= self.limits.depth:
                    omit("depth_limit")
                    return
                names = []
                with os.scandir(fd) as iterator:
                    for entry in iterator:
                        entries += 1
                        if entries > self.limits.entries:
                            stopped = True; omit("entry_limit"); break
                        names.append(entry.name)
                for name in sorted(names):
                    if stopped:
                        break
                    if not allowed_name(name):
                        omit("excluded_policy"); continue
                    path = prefix + "/" + name if prefix else name
                    info = os.stat(name, dir_fd=fd, follow_symlinks=False)
                    if stat.S_ISDIR(info.st_mode):
                        child = os.open(name, DIRECTORY_FLAGS, dir_fd=fd)
                        try:
                            walk(child, path, depth + 1)
                        finally:
                            os.close(child)
                    elif stat.S_ISREG(info.st_mode):
                        if len(files) >= self.limits.files:
                            stopped = True; omit("file_limit"); break
                        try:
                            original = self._read(root_fd, path)
                        except ConnectorError as error:
                            if error.code in {"file_too_large", "binary_or_non_utf8", "excluded_file_type"}:
                                omit(error.code); continue
                            raise
                        total += len(original.text.encode("utf-8"))
                        if total > self.limits.snapshot_bytes:
                            stopped = True; omit("snapshot_byte_limit"); break
                        files[path] = original
                    else:
                        omit("excluded_file_type")
            try:
                walk(root_fd, "", 0)
            except OSError as error:
                raise ConnectorError("source_changed") from error
        identity = {"root_id": str(self.registration.root_id), "git": git, "omitted": omitted,
                    "files": [(path, original.digest) for path, original in sorted(files.items())]}
        result = Snapshot(files, directories, omitted, git,
                          digest(json.dumps(identity, sort_keys=True, separators=(",", ":"))),
                          datetime.now(UTC).isoformat())
        self.verify(result)
        return result

    def verify(self, snapshot: Snapshot):
        with self.root() as root_fd:
            try:
                for path, expected in snapshot.directories.items():
                    if not path:
                        actual = os.fstat(root_fd)
                    else:
                        parts = relative_parts(path)
                        with self.parent(root_fd, parts) as parent:
                            actual = os.stat(parts[-1], dir_fd=parent, follow_symlinks=False)
                    if _stamp(actual) != expected:
                        raise ConnectorError("source_changed")
                for path, source in snapshot.files.items():
                    actual = self._read(root_fd, path)
                    if actual.digest != source.digest or actual.stamp != source.stamp:
                        raise ConnectorError("source_changed")
            except OSError as error:
                raise ConnectorError("source_changed") from error
        if self.git_state() != snapshot.git:
            raise ConnectorError("source_changed")
