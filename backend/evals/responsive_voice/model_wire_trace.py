"""Opt-in, private, complete Realtime payload capture for synthetic evaluations.

Not production telemetry. No credentials/HTTP headers/SDP are recorded. WebRTC
media and phone-side control are outside the sideband boundary and must be
provided as separate evidence. This module never initiates a model connection.
"""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
from email import policy
from email.parser import BytesParser
import hashlib
import json
import os
from pathlib import Path
import time
from urllib.parse import urlsplit
from uuid import uuid4


ENV = "JARVIS_EVAL_REALTIME_WIRE_DIR"
MAX_BYTES = 64 * 1024 * 1024
SECRET_KEYS = frozenset({"client_secret", "api_key", "authorization", "access_token"})
_installed = False


def sanitized_payload(value):
    """Only credential fields are elided; prompts and schemas remain complete."""
    redactions = []
    def walk(item, path):
        if isinstance(item, dict):
            result = {}
            for key, child in item.items():
                here = path + [str(key)]
                if key.lower() in SECRET_KEYS:
                    result[key] = "[credential omitted]"
                    redactions.append(".".join(here))
                else:
                    result[key] = walk(child, here)
            return result
        if isinstance(item, list):
            return [walk(child, path + [str(i)]) for i, child in enumerate(item)]
        return item
    return walk(value, []), redactions


class WireTrace:
    def __init__(self, root, *, limit=MAX_BYTES):
        self.root = Path(root)
        if not self.root.is_absolute() or self.root.is_symlink():
            raise ValueError("private_absolute_trace_directory_required")
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        if self.root.stat().st_mode & 0o077:
            raise ValueError("trace_directory_must_be_private")
        self.path = self.root / ("wire-" + str(os.getpid()) + "-" + uuid4().hex + ".jsonl")
        fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        self.stream = os.fdopen(fd, "w", encoding="utf-8")
        self.limit, self.bytes, self.sequence = limit, 0, 0
        self.incomplete = None
        self.record("capture_started", boundary="host_provider_transport_not_phone_media",
                    credentials="headers_and_sdp_never_captured")

    def record(self, stage, *, raw=None, **fields):
        if self.incomplete is not None:
            return None
        self.sequence += 1
        row = dict(sequence=self.sequence, stage=stage,
                   utc=datetime.now(timezone.utc).isoformat(), monotonic_s=time.monotonic(), **fields)
        if raw is not None:
            try:
                payload = json.loads(raw)
            except (ValueError, UnicodeDecodeError, TypeError):
                # Do not copy an unexpected non-JSON provider frame into a secret log.
                self.incomplete = "non_json_frame"
                row.update(stage="capture_incomplete", reason=self.incomplete)
            else:
                payload, redactions = sanitized_payload(payload)
                original = raw.decode("utf-8") if isinstance(raw, bytes) else raw
                retained = json.dumps(payload, ensure_ascii=True) if redactions else original
                row.update(raw=retained, payload=payload, credential_redactions=redactions,
                           raw_sha256=hashlib.sha256(retained.encode()).hexdigest(),
                           raw_exact_except_credentials=not bool(redactions))
        encoded = json.dumps(row, ensure_ascii=True, separators=(",", ":")) + "\n"
        if self.bytes + len(encoded.encode()) > self.limit:
            self.incomplete = "capture_byte_limit"
            encoded = json.dumps(dict(sequence=self.sequence, stage="capture_incomplete",
                                      reason=self.incomplete)) + "\n"
        try:
            self.stream.write(encoded)
            self.stream.flush()
            self.bytes += len(encoded.encode())
        except OSError:
            self.incomplete = "capture_io_error"
        return self.sequence

    def close(self):
        if not self.stream.closed:
            self.record("capture_closed", complete=self.incomplete is None)
            self.stream.close()


class TracedSocket:
    def __init__(self, socket, trace, connection_id):
        self.socket, self.trace, self.connection_id = socket, trace, connection_id

    def __getattr__(self, key):
        return getattr(self.socket, key)

    async def send(self, raw, *args, **kwargs):
        sequence = self.trace.record("send_attempt", raw=raw, connection_id=self.connection_id)
        try:
            result = await self.socket.send(raw, *args, **kwargs)
        except BaseException as error:
            self.trace.record("send_failed", attempt_sequence=sequence,
                              connection_id=self.connection_id, error_type=type(error).__name__)
            raise
        self.trace.record("send_returned", attempt_sequence=sequence, connection_id=self.connection_id)
        return result

    async def recv(self, *args, **kwargs):
        raw = await self.socket.recv(*args, **kwargs)
        self.trace.record("receive", raw=raw, connection_id=self.connection_id)
        return raw

    async def close(self, *args, **kwargs):
        try:
            return await self.socket.close(*args, **kwargs)
        finally:
            self.trace.record("socket_close_returned_or_failed", connection_id=self.connection_id,
                              remote_closure_confirmed=False)

    def __aiter__(self):
        return self

    async def __anext__(self):
        from websockets.exceptions import ConnectionClosedOK
        try:
            return await self.recv()
        except ConnectionClosedOK:
            raise StopAsyncIteration


class TracedConnect:
    """Keep both await and async-context-manager connector interfaces."""
    def __init__(self, connector, trace):
        self.connector, self.trace = connector, trace
        self.connection_id = uuid4().hex

    async def _connect(self):
        socket = await self.connector
        self.trace.record("socket_connected", connection_id=self.connection_id)
        return TracedSocket(socket, self.trace, self.connection_id)

    def __await__(self):
        return self._connect().__await__()

    async def __aenter__(self):
        self.socket = await self._connect()
        return self.socket

    async def __aexit__(self, *args):
        await self.socket.close()


async def creation_session(request):
    """Extract only the session part, never SDP, HTTP auth or ICE credentials."""
    content_type = request.headers.get("content-type", "")
    if "multipart/form-data" not in content_type:
        raise ValueError("unexpected_creation_content_type")
    # HTTPX retains the same buffered body for the actual send after aread().
    # This changes local buffering timing, not the multipart body bytes.
    content = await request.aread()
    message = BytesParser(policy=policy.default).parsebytes(
        b"Content-Type: " + content_type.encode() + b"\r\nMIME-Version: 1.0\r\n\r\n" + content)
    for part in message.iter_parts():
        if part.get_param("name", header="content-disposition") == "session":
            return part.get_payload(decode=True).decode("utf-8")
    raise ValueError("creation_session_part_missing")


def install_from_environment():
    """Called by the eval child only; absent opt-in is a strict no-op."""
    global _installed
    root = os.environ.get(ENV)
    if _installed or not root or os.environ.get("APP_ENV") != "frontend-replay-qualification":
        return
    import atexit
    import httpx
    import websockets
    trace = WireTrace(root)
    atexit.register(trace.close)
    original_connect, original_send = websockets.connect, httpx.AsyncClient.send

    def connect(uri, *args, **kwargs):
        connector = original_connect(uri, *args, **kwargs)
        parsed = urlsplit(str(uri))
        if parsed.hostname == "api.openai.com" and parsed.path == "/v1/realtime":
            return TracedConnect(connector, trace)
        return connector

    async def send(client, request, *args, **kwargs):
        matched = (request.url.host == "api.openai.com" and
                   request.url.path == "/v1/realtime/calls" and request.method == "POST")
        if not matched:
            return await original_send(client, request, *args, **kwargs)
        try:
            raw = await creation_session(request)
        except Exception as error:
            trace.record("creation_configuration_unavailable", error_type=type(error).__name__)
        else:
            trace.record("creation_configuration_send_attempt", raw=raw)
        try:
            response = await original_send(client, request, *args, **kwargs)
        except BaseException as error:
            trace.record("creation_request_failed", error_type=type(error).__name__)
            raise
        trace.record("creation_request_returned", http_status=response.status_code)
        return response

    websockets.connect, httpx.AsyncClient.send = connect, send
    _installed = True


def readable(value, path="$", *, assets=None, stem="payload"):
    """Every field/string line, without silently shortening prompts or schemas."""
    lines = []
    if isinstance(value, dict):
        if not value:
            lines.append(path + " = {}")
        for key, child in value.items():
            lines.extend(readable(child, path + "." + key, assets=assets, stem=stem))
    elif isinstance(value, list):
        if not value:
            lines.append(path + " = []")
        for i, child in enumerate(value):
            lines.extend(readable(child, path + f"[{i}]", assets=assets, stem=stem))
    elif isinstance(value, str):
        encoded = None
        suffix = ".bin"
        if value.startswith("data:image/") and ";base64," in value:
            encoded = value.split(";base64,", 1)[1]
            suffix = ".jpg" if value.startswith("data:image/jpeg;") else ".bin"
        elif path.endswith(".audio") or path.endswith(".delta") and len(value) > 1024:
            try:
                base64.b64decode(value, validate=True)
                encoded = value
                suffix = ".pcm"
            except ValueError:
                pass
        if encoded is not None and assets is not None:
            blob = base64.b64decode(encoded, validate=True)
            digest = hashlib.sha256(blob).hexdigest()
            target = assets / (digest + suffix)
            if not target.exists():
                target.write_bytes(blob)
                target.chmod(0o600)
            lines.append(f"{path} = BINARY {len(blob)} bytes; sha256={digest}; file={target}")
        else:
            for number, line in enumerate(value.split("\n"), 1):
                lines.append(f"{path}:L{number:04d} | {line}")
    else:
        lines.append(path + " = " + json.dumps(value))
    return lines


def render(root, output):
    root, output = Path(root), Path(output)
    rows = []
    for source in sorted(root.glob("wire-*.jsonl")):
        for line_no, line in enumerate(source.read_text().splitlines(), 1):
            row = json.loads(line)
            row["source"], row["source_line"] = str(source), line_no
            rows.append(row)
    if not rows:
        raise ValueError("no_captured_wire_messages")
    rows.sort(key=lambda row: (row.get("monotonic_s", float("inf")), row["source_line"]))
    assets = output.parent / (output.stem + "-assets")
    assets.mkdir(mode=0o700, exist_ok=False)
    lines = ["# Complete captured Realtime transport timeline", "",
        "New captured baseline, not a reconstructed historical run.", "",
        "Boundary: host HTTP session configuration and host/provider WebSocket. Direct phone WebRTC "
        "audio and phone-side control are NOT captured here. Original WAVs and phone traces must "
        "be supplied separately; they do not prove provider-received media bytes.", "",
        "A send attempt is not delivery. Send returned is not model adoption. Provider events "
        "are recorded separately. No credentials, authorization headers or SDP are retained.", "",
        "All text and tool schema fields are expanded below, including repeated prompts. Binary "
        "payloads are externalized with hashes. Raw JSONL retains complete captured message strings.", ""]
    for i, row in enumerate(rows, 1):
        payload = row.get("payload")
        event = payload.get("type", "configuration") if isinstance(payload, dict) else ""
        lines += [f"## {i:04d} | {row.get('utc', 'timestamp unavailable')} | {row['stage']} | {event}", "",
                  f"Raw: [{Path(row['source']).name}:{row['source_line']}]({row['source']})", "", "``````text"]
        lines.extend(readable({k:v for k,v in row.items() if k not in {"raw", "payload", "source", "source_line"}}))
        if payload is not None:
            lines.extend(readable(payload, assets=assets, stem=f"{i:04d}"))
        lines += ["``````", ""]
    fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as stream:
        stream.write("\n".join(lines))
    return dict(rows=len(rows), output=str(output), incomplete=any(
        row["stage"] in {"capture_incomplete", "creation_configuration_unavailable"} for row in rows))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    print(json.dumps(render(arguments.trace_dir, arguments.output)))
