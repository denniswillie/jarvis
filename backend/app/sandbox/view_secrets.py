"""Service-private handoff of credentials already obtained by an authorized worker.

No provider request or lifecycle operation belongs here. Both processes must
select the same explicit private directory; there is no temporary/public fallback.
"""
from contextlib import contextmanager
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import stat
from uuid import uuid4

from app.sandbox.models import SandboxError


class SandboxViewSecrets:
    def __init__(self, root):
        self.root = Path(root)
        if not self.root.is_absolute() or ".." in self.root.parts:
            raise ValueError("Select an absolute private credential directory")
        with self._directory():
            pass

    @classmethod
    def configured(cls):
        root = os.environ.get("SANDBOX_VIEW_SECRET_ROOT")
        return cls(root) if root else None

    @contextmanager
    def _directory(self):
        descriptor = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
        try:
            for index, part in enumerate(self.root.parts[1:]):
                final = index == len(self.root.parts) - 2
                if final:
                    try:
                        os.mkdir(part, 0o700, dir_fd=descriptor)
                    except FileExistsError:
                        pass
                next_descriptor = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor)
                os.close(descriptor)
                descriptor = next_descriptor
            info = os.fstat(descriptor)
            if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o700:
                raise SandboxError("sandbox_view_secret_directory_invalid")
            yield descriptor
        finally:
            os.close(descriptor)

    @staticmethod
    def _name(sandbox_id, scope):
        return hashlib.sha256((scope.owner_id + "\0" + sandbox_id).encode()).hexdigest() + ".json"

    def capture(self, handle, token):
        if handle.policy.template != "desktop":
            return
        if not isinstance(token, str) or not 1 <= len(token) <= 4096 or any(ord(c) < 33 for c in token):
            raise SandboxError("sandbox_view_credential_unavailable")
        record = {"sandbox_id": handle.sandbox_id, "scope": handle.scope.model_dump(mode="json"),
                  "native_scope": handle.provider_scope.model_dump(mode="json"),
                  "policy": handle.policy.fingerprint, "expires_at": handle.expires_at.isoformat(), "token": token}
        name, temporary = self._name(handle.sandbox_id, handle.scope), uuid4().hex + ".tmp"
        with self._directory() as directory:
            try:
                descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                                     0o600, dir_fd=directory)
                with os.fdopen(descriptor, "w") as output:
                    json.dump(record, output, separators=(",", ":"))
                    output.flush()
                    os.fsync(output.fileno())
                os.replace(temporary, name, src_dir_fd=directory, dst_dir_fd=directory)
                os.fsync(directory)
            finally:
                try:
                    os.unlink(temporary, dir_fd=directory)
                except FileNotFoundError:
                    pass

    def read(self, handle):
        try:
            with self._directory() as directory:
                descriptor = os.open(self._name(handle.sandbox_id, handle.scope),
                                     os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory)
                with os.fdopen(descriptor, "rb") as source:
                    info = os.fstat(source.fileno())
                    if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                            or stat.S_IMODE(info.st_mode) != 0o600 or info.st_nlink != 1 or info.st_size > 8192):
                        raise ValueError()
                    value = json.load(source)
            if (value["sandbox_id"] != handle.sandbox_id
                    or value["scope"] != handle.scope.model_dump(mode="json")
                    or value["native_scope"] != handle.provider_scope.model_dump(mode="json")
                    or value["policy"] != handle.policy.fingerprint):
                raise ValueError()
            if datetime.fromisoformat(value["expires_at"]) <= datetime.now(UTC):
                self.remove(handle.sandbox_id, handle.scope)
                raise ValueError()
            token = value["token"]
            if not isinstance(token, str) or not 1 <= len(token) <= 4096 or any(ord(c) < 33 for c in token):
                raise ValueError()
            return token
        except (OSError, ValueError, KeyError, TypeError):
            raise SandboxError("sandbox_view_credential_unavailable") from None

    def remove(self, sandbox_id, scope):
        with self._directory() as directory:
            try:
                os.unlink(self._name(sandbox_id, scope), dir_fd=directory)
            except FileNotFoundError:
                pass
