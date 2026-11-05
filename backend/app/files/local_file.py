"""Private, verified disk snapshots for bounded retained-file transfers.

These objects carry bytes, never publication authority. The caller owns close();
neither a caller path nor its mutable directory entry is the upload source.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import hashlib
import os
from pathlib import Path
import stat
import tempfile


CHUNK_BYTES = 64 * 1024


class LocalFileError(ValueError):
    """A stable, content-free failure; paths and input values are not included."""
    def __init__(self, code):
        self.code = code
        super().__init__(code)


def _bounds(max_bytes, expected_size=None, expected_sha256=None):
    if type(max_bytes) is not int or max_bytes < 0:
        raise ValueError("A nonnegative file byte bound is required")
    if expected_size is not None and (type(expected_size) is not int or not 0 <= expected_size <= max_bytes):
        raise ValueError("The expected file size must fit its byte bound")
    if expected_sha256 is not None and (type(expected_sha256) is not str or len(expected_sha256) != 64
            or any(character not in "0123456789abcdef" for character in expected_sha256)):
        raise ValueError("An exact lowercase SHA256 is required")


def _open_regular(path):
    """Walk without following symlinks, pinning each parent before the next."""
    path = Path(path)
    if not path.is_absolute():
        path = Path.cwd() / path
    if ".." in path.parts or len(path.parts) < 2:
        raise LocalFileError("local_file_invalid")
    directory = None
    descriptor = None
    try:
        directory = os.open(path.anchor, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        for part in path.parts[1:-1]:
            parent = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                             dir_fd=directory)
            os.close(directory)
            directory = parent
        descriptor = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC,
                             dir_fd=directory)
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise LocalFileError("local_file_invalid")
        result, descriptor = descriptor, None
        return result
    except OSError:
        raise LocalFileError("local_file_invalid") from None
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if directory is not None:
            os.close(directory)


def _fingerprint(value):
    return (value.st_dev, value.st_ino, value.st_mode, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


@dataclass(frozen=True, slots=True)
class VerifiedLocalFile:
    path: Path
    size_bytes: int
    content_sha256: str
    _fd: int = field(repr=False)
    _directory: tempfile.TemporaryDirectory = field(repr=False)
    _chunks: tuple[bytes, ...] = field(repr=False)
    _identity: tuple = field(repr=False)
    _closed: bool = field(default=False, init=False, repr=False)

    def _check(self):
        if self._closed:
            raise LocalFileError("local_file_closed")
        try:
            current = os.fstat(self._fd)
            entry = self.path.lstat()
            if (_fingerprint(current) != self._identity or not stat.S_ISREG(entry.st_mode)
                    or (entry.st_dev, entry.st_ino) != (current.st_dev, current.st_ino)):
                raise LocalFileError("local_file_changed")
        except OSError:
            raise LocalFileError("local_file_changed") from None

    async def iter_chunks(self):
        """Yield only checked immutable chunks, with no shared read cursor."""
        self._check()
        offset = 0
        for expected in self._chunks:
            self._check()
            try:
                block = os.pread(self._fd, min(CHUNK_BYTES, self.size_bytes - offset), offset)
            except OSError:
                raise LocalFileError("local_file_changed") from None
            if hashlib.sha256(block).digest() != expected:
                raise LocalFileError("local_file_changed")
            offset += len(block)
            yield block
            await asyncio.sleep(0)
        self._check()
        if offset != self.size_bytes:
            raise LocalFileError("local_file_length_mismatch")

    def close(self):
        if not self._closed:
            object.__setattr__(self, "_closed", True)
            try:
                os.close(self._fd)
            finally:
                self._directory.cleanup()

    def __enter__(self):
        self._check()
        return self

    def __exit__(self, *_):
        self.close()

    def __del__(self):
        # Explicit ownership is primary; this also prevents leaked descriptors
        # when a caller discards an otherwise unused snapshot.
        try:
            self.close()
        except (OSError, AttributeError):
            pass


async def _snapshot_chunks(chunks, *, max_bytes, expected_size=None, expected_sha256=None):
    """Internal streaming sink; incomplete files never become verified handles."""
    _bounds(max_bytes, expected_size, expected_sha256)
    directory = tempfile.TemporaryDirectory(prefix="jarvis-file-")
    path = Path(directory.name) / "content"
    descriptor = None
    reader = None
    owned = False
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
        checksum, chunk_hashes, pending, size = hashlib.sha256(), [], bytearray(), 0
        async for block in chunks:
            if not isinstance(block, bytes):
                raise LocalFileError("local_file_invalid")
            size += len(block)
            if size > max_bytes:
                raise LocalFileError("local_file_size_exceeded")
            if expected_size is not None and size > expected_size:
                raise LocalFileError("local_file_length_mismatch")
            checksum.update(block)
            view = memoryview(block)
            while view:
                part = view[:CHUNK_BYTES - len(pending)]
                pending.extend(part)
                view = view[len(part):]
                if len(pending) == CHUNK_BYTES:
                    _write_all(descriptor, pending)
                    chunk_hashes.append(hashlib.sha256(pending).digest())
                    pending.clear()
            await asyncio.sleep(0)
        if pending:
            _write_all(descriptor, pending)
            chunk_hashes.append(hashlib.sha256(pending).digest())
        if expected_size is not None and size != expected_size:
            raise LocalFileError("local_file_length_mismatch")
        if expected_sha256 is not None and checksum.hexdigest() != expected_sha256:
            raise LocalFileError("local_file_digest_mismatch")
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o400)
        writer_identity = os.fstat(descriptor)
        reader = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
        identity = os.fstat(reader)
        if (identity.st_dev, identity.st_ino, identity.st_size) != (
                writer_identity.st_dev, writer_identity.st_ino, size):
            raise LocalFileError("local_file_changed")
        result = VerifiedLocalFile(path, size, checksum.hexdigest(), reader, directory,
                                   tuple(chunk_hashes), _fingerprint(identity))
        owned = True
        return result
    except OSError:
        raise LocalFileError("local_file_io_failed") from None
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if not owned:
            if reader is not None:
                os.close(reader)
            directory.cleanup()


def _write_all(descriptor, block):
    view = memoryview(block)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise LocalFileError("local_file_io_failed")
        view = view[written:]


async def snapshot_file(path, *, max_bytes, expected_size=None, expected_sha256=None):
    """Copy an exact regular-file original to privately owned, verified bytes."""
    _bounds(max_bytes, expected_size, expected_sha256)
    descriptor = _open_regular(path)
    try:
        before = os.fstat(descriptor)
        if before.st_size > max_bytes:
            raise LocalFileError("local_file_size_exceeded")
        if expected_size is not None and before.st_size != expected_size:
            raise LocalFileError("local_file_length_mismatch")

        async def chunks():
            while block := os.read(descriptor, CHUNK_BYTES):
                yield block
            if _fingerprint(os.fstat(descriptor)) != _fingerprint(before):
                raise LocalFileError("local_file_changed")

        return await _snapshot_chunks(chunks(), max_bytes=max_bytes, expected_size=before.st_size,
                                      expected_sha256=expected_sha256)
    except OSError:
        raise LocalFileError("local_file_io_failed") from None
    finally:
        os.close(descriptor)
