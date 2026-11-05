"""Outbound HTTPS only. Scoped credentials never enter project subprocesses."""
from __future__ import annotations

from datetime import UTC, datetime
from hashlib import sha256
import json
import os
from contextlib import contextmanager
from pathlib import Path
import stat
import time
from urllib.parse import urlsplit
from uuid import UUID

import httpx
from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError

from .filesystem import ProjectReader
from .navigation import ProjectInspector
from .protocol import ConnectorError, ReadRequest


MAX_BODY_BYTES = 2_000_000


class Connection(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    protocol_version: int = Field(default=1, ge=1, le=1)
    url: str
    connector_id: UUID
    token: SecretStr


def origin(url):
    parsed = urlsplit(url)
    if (parsed.username or parsed.password or parsed.query or parsed.fragment or parsed.path not in ("", "/")
            or not parsed.hostname or parsed.scheme not in ("https", "http")):
        raise ConnectorError("invalid_backend_origin")
    if parsed.scheme == "http" and parsed.hostname not in {"127.0.0.1", "::1", "localhost"}:
        raise ConnectorError("https_required")
    return url.rstrip("/")


def offer(root):
    identity = json.dumps({"path": root.path, "device": root.device, "inode": root.inode},
                          sort_keys=True, separators=(",", ":"))
    return {"local_root_id": str(root.root_id), "label": root.label,
            "fingerprint": sha256(identity.encode()).hexdigest()}


def save_connection(path, connection):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        value = connection.model_dump(mode="json")
        value["token"] = connection.token.get_secret_value()
        json.dump(value, handle)
        handle.write("\n")


@contextmanager
def reserved_connection(path):
    """Fail before contacting the server if the private credential path is occupied."""
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    identity = os.fstat(fd)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            def save(connection):
                value = connection.model_dump(mode="json")
                value["token"] = connection.token.get_secret_value()
                json.dump(value, handle)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            yield save
    except BaseException:
        try:
            current = os.stat(path, follow_symlinks=False)
            if (current.st_dev, current.st_ino) == (identity.st_dev, identity.st_ino):
                os.unlink(path)
        except FileNotFoundError:
            pass
        raise


def load_connection(path):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, encoding="utf-8") as handle:
        info = os.fstat(handle.fileno())
        if (not stat.S_ISREG(info.st_mode) or info.st_mode & 0o077 or info.st_uid != os.getuid()
                or info.st_size > 8192 or info.st_nlink != 1):
            raise ConnectorError("credential_permissions")
        connection = Connection.model_validate_json(handle.read())
    origin(connection.url)
    return connection


class OutboundConnector:
    def __init__(self, url, token, *, node=None, typescript=None):
        self.url = origin(url)
        self.node, self.typescript = node, typescript
        self.client = httpx.Client(headers={"Authorization": "Bearer " + token},
            timeout=httpx.Timeout(15, connect=5), follow_redirects=False, trust_env=False)

    def close(self):
        self.client.close()

    def post(self, path, payload, *, method="POST"):
        raw = json.dumps(payload, ensure_ascii=True, allow_nan=False).encode("ascii")
        if len(raw) > MAX_BODY_BYTES:
            raise ConnectorError("response_too_large")
        try:
            with self.client.stream(method, self.url + path, content=raw,
                                    headers={"Content-Type": "application/json"}) as response:
                if response.status_code != 200:
                    code = "connector_authorization_failed" if response.status_code in (401, 403) else "backend_request_rejected"
                    raise ConnectorError(code)
                pieces, size = [], 0
                for part in response.iter_bytes():
                    size += len(part)
                    if size > MAX_BODY_BYTES:
                        raise ConnectorError("backend_response_too_large")
                    pieces.append(part)
                return json.loads(b"".join(pieces))
        except (httpx.HTTPError, ValueError) as error:
            raise ConnectorError("backend_unavailable") from error

    def register(self, roots, *, area_id, label):
        return self.post("/projects/connectors", {"area_id": str(area_id), "label": label,
                                                 "roots": [offer(root) for root in roots]})

    def once(self, registry_loader):
        roots = registry_loader()
        response = self.post("/projects/connector/poll", {
            "protocol_version": 1, "roots": [offer(root) for root in roots]})
        if response.get("protocol_version") != 1:
            raise ConnectorError("unsupported_protocol")
        work = response.get("request")
        if work is None:
            return False
        if set(work) != {"request_id", "lease_id", "lease_until", "local_root_id", "fingerprint", "read"}:
            raise ConnectorError("invalid_backend_request")
        try:
            UUID(work["request_id"])
            UUID(work["lease_id"])
            if datetime.fromisoformat(work["lease_until"]) <= datetime.now(UTC):
                raise ConnectorError("lease_expired")
            root = next((root for root in roots if str(root.root_id) == work["local_root_id"]), None)
            if root is None or offer(root)["fingerprint"] != work["fingerprint"]:
                raise ConnectorError("root_not_enrolled")
            request = ReadRequest.model_validate(work["read"])
            result = ProjectInspector(ProjectReader(root), node=self.node, typescript=self.typescript).inspect(
                request, include_originals=True)
            originals = result.pop("_originals", [])
            if root not in registry_loader():
                raise ConnectorError("root_not_enrolled")
            if len(originals) > 30:
                raise ConnectorError("response_too_large")
            # Reserve room for canonical provenance; never present a page as the full original.
            if (len(json.dumps(result, ensure_ascii=True)) > 150000 or
                    any(len(json.dumps(original, ensure_ascii=True)) > 220000 for original in originals)):
                raise ConnectorError("source_storage_limit")
            if len(json.dumps({"result": result, "originals": originals}, ensure_ascii=True).encode()) > MAX_BODY_BYTES - 2000:
                raise ConnectorError("response_too_large")
        except (ConnectorError, ValidationError, OSError, ValueError, KeyError) as error:
            code = error.code if isinstance(error, ConnectorError) else "invalid_or_unavailable_source"
            result, originals = {"status": "unavailable", "error": code}, []
        captured_at = result.get("captured_at") or result.get("snapshot", {}).get("captured_at") or datetime.now(UTC).isoformat()
        completed = {"protocol_version": 1,
            "request_id": work["request_id"], "lease_id": work["lease_id"],
            "captured_at": captured_at, "result": result, "originals": originals}
        try:
            self.post("/projects/connector/complete", completed)
        except ConnectorError as error:
            if error.code != "backend_request_rejected":
                raise
            # A rejected source must not leave the caller waiting as if inspection succeeded.
            # The same lease still fences this acknowledgment; it cannot overwrite an accepted result.
            self.post("/projects/connector/complete", completed | {
                "result": {"status": "unavailable", "error": "source_result_rejected"}, "originals": []})
        return True

    def run(self, registry_loader, *, stop=None, poll_seconds=1):
        while stop is None or not stop.is_set():
            try:
                handled = self.once(registry_loader)
            except ConnectorError as error:
                # Status codes only. Never print credentials, source text or host paths.
                print(json.dumps({"status": "unavailable", "error": error.code}), flush=True)
                if error.code == "connector_authorization_failed":
                    raise
                handled = False
            if not handled:
                if stop is not None:
                    stop.wait(poll_seconds)
                else:
                    time.sleep(poll_seconds)
