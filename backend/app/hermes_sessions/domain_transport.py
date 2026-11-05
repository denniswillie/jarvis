"""Private local tool RPC, not a second agent/API provider.

Imports stay standard-library-only: the native Hermes plugin does not need the
backend's database dependencies. The async server receives its existing adapter.
Filesystem access to the private profile is the local process trust boundary;
neither this descriptor nor the socket belongs in a remote sandbox.
"""
from __future__ import annotations

import asyncio
import base64
import fcntl
import json
import math
import os
from pathlib import Path
import socket
import stat
import tempfile


MAX_REQUEST = 256 * 1024
MAX_RESPONSE = 1024 * 1024
FILE_TOOL_TIMEOUT_SECONDS = 900.0
_FIELDS = {"version", "native_session_id", "input_id", "admission_id", "call_id", "name", "arguments"}


def _file_transfer_request(request):
    return (isinstance(request, dict) and request.get("channel") is None
            and request.get("name") in ("sandbox_stage_file", "sandbox_publish_file"))


def _failed(code, message):
    return {"status": "failed", "error": {"code": code, "message": message}}


async def _native_image_result(outcome):
    """Transient pixels use Hermes's generic multimodal handler convention.

    Tool receipts were already captured without bytes by the domain adapter.
    Keep the existing RPC limit rather than growing every local response.
    """
    image = outcome.image
    if image is None:
        return outcome.payload
    content = getattr(image, "image_bytes", None)
    content_type = getattr(getattr(image, "reference", None), "content_type", None)
    if (type(content) is not bytes or not 0 < len(content) <= 512 * 1024
            or content_type not in {"image/png", "image/jpeg"}):
        return _failed("domain_image_transport_unavailable", "No bounded PNG/JPEG pixels were delivered.")
    if image.revalidate is not None:
        await image.revalidate()
    summary = json.dumps(outcome.payload, ensure_ascii=True, allow_nan=False)
    result = {"_multimodal": True, "content": [
        {"type": "text", "text": summary},
        {"type": "image_url", "image_url": {
            "url": f"data:{content_type};base64," + base64.b64encode(content).decode("ascii")}}],
        "text_summary": summary}
    if len(json.dumps(result).encode()) + 1024 > MAX_RESPONSE:
        return _failed("domain_image_transport_unavailable", "Image response exceeds the local transport limit.")
    return result


def _private_directory(home):
    home = Path(home).resolve(strict=True)
    state = home / "jarvis"
    if state.is_symlink():
        raise ValueError("Profile state cannot be a symlink")
    state.mkdir(mode=0o700, exist_ok=True)
    if state.stat().st_uid != os.getuid():
        raise ValueError("Profile state must belong to this process owner")
    state.chmod(0o700)
    return home, state


def _publish(path, payload):
    encoded = json.dumps(payload, ensure_ascii=True, allow_nan=False).encode()
    if len(encoded) > MAX_RESPONSE:
        raise ValueError("Profile descriptor is too large")
    fd, pending = tempfile.mkstemp(prefix=".domain-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(pending, path)
    finally:
        if os.path.exists(pending):
            os.unlink(pending)


def write_domain_catalog(profile_home, tools):
    """Publish derived read schemas; this manifest grants no execution authority."""
    _, state = _private_directory(profile_home)
    if not isinstance(tools, list) or len(tools) > 64:
        raise ValueError("Invalid domain catalog")
    names = set()
    for tool in tools:
        if (not isinstance(tool, dict) or set(tool) != {"name", "description", "parameters"}
                or not isinstance(tool["name"], str) or not 1 <= len(tool["name"]) <= 256
                or tool["name"] in names or not isinstance(tool["description"], str)
                or not isinstance(tool["parameters"], dict)):
            raise ValueError("Invalid domain tool definition")
        names.add(tool["name"])
    _publish(state / "domain-tools.json", {"version": 1, "tools": tools})


def _validate_request(payload):
    if (not isinstance(payload, dict) or set(payload) != _FIELDS
            or type(payload["version"]) is not int or payload["version"] != 1
            or not isinstance(payload["arguments"], dict)):
        raise ValueError("Invalid domain request")
    for field in ("native_session_id", "input_id", "admission_id", "call_id", "name"):
        value = payload[field]
        if not isinstance(value, str) or not value.strip() or len(value) > 256 or "\0" in value:
            raise ValueError("Invalid domain request identity")


def _validate_rpc_request(request):
    if isinstance(request, dict) and request.get("channel") == "terminal":
        from .terminal_transport import validate_terminal_request
        validate_terminal_request(request)
    elif isinstance(request, dict) and request.get("channel") == "sandbox_control":
        from .terminal_transport import validate_sandbox_control_request
        validate_sandbox_control_request(request)
    else:
        _validate_request(request)


def invoke_domain_tool(profile_home, *, native_session_id, input_id, admission_id,
                       call_id, name, arguments, timeout=None):
    """Called inside a native tool worker; one attempt, never a blind retry."""
    request = {"version": 1, "native_session_id": native_session_id, "input_id": input_id,
               "admission_id": admission_id, "call_id": call_id, "name": name, "arguments": arguments}
    if timeout is None:
        timeout = FILE_TOOL_TIMEOUT_SECONDS + 5 if _file_transfer_request(request) else 65.0
    return _invoke_private(profile_home, request, timeout=timeout)


def _invoke_private(profile_home, request, *, timeout):
    """Shared private framing only; domain and terminal authority stay separate."""
    try:
        ceiling = FILE_TOOL_TIMEOUT_SECONDS + 5 if _file_transfer_request(request) else 180
        if not math.isfinite(timeout) or not 0 < timeout <= ceiling:
            raise ValueError("Invalid deadline")
        home = Path(profile_home).resolve(strict=True)
        path = home / "jarvis" / "domain-api.json"
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(fd, "rb") as stream:
            info = os.fstat(stream.fileno())
            if info.st_uid != os.getuid() or info.st_mode & 0o077:
                raise ValueError("Untrusted descriptor permissions")
            encoded = stream.read(8193)
        if len(encoded) > 8192:
            raise ValueError("Invalid descriptor size")
        descriptor = json.loads(encoded)
        if (type(descriptor.get("version")) is not int or descriptor["version"] != 1
                or descriptor.get("profile_home") != str(home)):
            raise ValueError("Different profile descriptor")
        address = Path(descriptor["socket_path"])
        info = address.lstat()
        parent = address.parent.stat()
        if (not address.is_absolute() or address.resolve() != address
                or not stat.S_ISSOCK(info.st_mode) or info.st_uid != os.getuid()
                or info.st_mode & 0o077 or parent.st_uid != os.getuid() or parent.st_mode & 0o077):
            raise ValueError("Untrusted socket")
        _validate_rpc_request(request)
        wire = json.dumps(request, ensure_ascii=True, allow_nan=False).encode() + b"\n"
        if len(wire) > MAX_REQUEST:
            raise ValueError("Request exceeds local transport limit")
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.settimeout(timeout)
            connection.connect(str(address))
            connection.sendall(wire)
            response = bytearray()
            while not response.endswith(b"\n"):
                chunk = connection.recv(min(65536, MAX_RESPONSE + 1 - len(response)))
                if not chunk:
                    raise ConnectionError("No complete tool reply")
                response.extend(chunk)
                if len(response) > MAX_RESPONSE:
                    raise ValueError("Response exceeds local transport limit")
        reply = json.loads(response)
        if (not isinstance(reply, dict) or reply.get("version") != 1
                or not isinstance(reply.get("payload"), dict)):
            raise ValueError("Invalid tool reply")
        return reply["payload"]
    except (OSError, ConnectionError, TimeoutError):
        return _failed("domain_transport_unavailable",
                       "The domain service did not return a complete reply. No automatic retry was made.")
    except (KeyError, TypeError, ValueError):
        return _failed("domain_protocol_error", "The local tool connection or request is invalid.")


class DomainToolServer:
    def __init__(self, profile_home, adapter, *, terminal=None, sandbox_controls=None,
                 max_in_flight=8, timeout=60.0):
        if (type(max_in_flight) is not int or not 1 <= max_in_flight <= 32
                or not math.isfinite(timeout) or not 0 < timeout <= 120):
            raise ValueError("Invalid domain transport limits")
        self.home, self.state = _private_directory(profile_home)
        if terminal is not None and terminal.profile_home != self.home:
            raise ValueError("Terminal adapter belongs to another profile")
        self.terminal = terminal
        self.sandbox_controls = None
        if sandbox_controls is not None:
            self.install_sandbox_controls(sandbox_controls)
        self.adapter, self.max_in_flight, self.timeout = adapter, max_in_flight, timeout
        self.server = None
        self.socket_path = None
        self._lock = None
        self._published = False
        self._active = set()
        self._closing = False

    def install_sandbox_controls(self, controls):
        """Trusted staged composition, after master setup and before model inputs.

        The server borrows the controller; it does not start its timer or own its
        journal/resource lease. A second controller cannot silently replace it.
        """
        if controls.profile_home != self.home:
            raise ValueError("Sandbox controls belong to another profile")
        if self.sandbox_controls is not None and self.sandbox_controls is not controls:
            raise RuntimeError("Sandbox controls already installed")
        self.sandbox_controls = controls

    async def start(self):
        if self.server is not None or self._lock is not None:
            raise RuntimeError("Domain server already started")
        self._closing = False
        self._lock = os.open(self.state / "domain-api.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException:
            os.close(self._lock)
            self._lock = None
            raise
        try:
            # macOS Unix addresses are short. A private tmp directory avoids
            # weakening isolation or depending on a short checkout/home path.
            directory = Path(tempfile.mkdtemp(prefix="jarvis-domain-", dir="/tmp")).resolve()
            self.socket_path = directory / "api.sock"
            self.server = await asyncio.start_unix_server(self._handle, path=str(self.socket_path),
                                                          limit=MAX_REQUEST + 1, backlog=16)
            self.socket_path.chmod(0o600)
            _publish(self.state / "domain-api.json", {"version": 1,
                "profile_home": str(self.home), "socket_path": str(self.socket_path)})
            self._published = True
            return self
        except BaseException:
            await self.close()
            raise

    async def _reply(self, writer, payload):
        try:
            reply = json.dumps({"version": 1, "payload": payload}, ensure_ascii=True,
                               allow_nan=False).encode() + b"\n"
            if len(reply) > MAX_RESPONSE:
                raise ValueError("Result too large")
        except (TypeError, ValueError):
            reply = json.dumps({"version": 1, "payload": _failed("domain_result_unserializable",
                "The tool result cannot be delivered within this transport; request a smaller result.")}).encode() + b"\n"
        writer.write(reply)
        await writer.drain()

    async def _handle(self, reader, writer):
        task = asyncio.current_task()
        if self._closing:
            writer.close()
            return
        if len(self._active) >= self.max_in_flight:
            try:
                await self._reply(writer, _failed("domain_service_busy", "The bounded domain service is busy."))
            finally:
                writer.close()
            return
        self._active.add(task)
        try:
            async with asyncio.timeout(self.timeout) as deadline:
                raw = await reader.readline()
                if not raw.endswith(b"\n") or len(raw) > MAX_REQUEST:
                    raise ValueError("Invalid request frame")
                request = json.loads(raw)
                _validate_rpc_request(request)
                if _file_transfer_request(request):
                    # File I/O has its own finite budget; ordinary requests and
                    # slow/incomplete request frames retain the short deadline.
                    deadline.reschedule(asyncio.get_running_loop().time() + FILE_TOOL_TIMEOUT_SECONDS)
                if request.get("channel") == "terminal":
                    if self.terminal is None:
                        payload = _failed("terminal_disabled", "Retained terminal is not enabled.")
                    else:
                        payload = await self.terminal.invoke(operation=request["operation"],
                            context=request["context"], attachment=request["attachment"],
                            arguments=request["arguments"])
                elif request.get("channel") == "sandbox_control":
                    if self.sandbox_controls is None:
                        payload = _failed("sandbox_controls_disabled", "Sandbox controls are not enabled.")
                    else:
                        payload = await self.sandbox_controls.invoke(
                            context=request["context"], arguments=request["arguments"])
                elif self.adapter is None:
                    payload = _failed("domain_disabled", "Domain tools are not enabled.")
                else:
                    outcome = await self.adapter.invoke(native_session_id=request["native_session_id"],
                        input_id=request["input_id"], call_id=request["call_id"], name=request["name"],
                        arguments=request["arguments"])
                    payload = await _native_image_result(outcome)
                await self._reply(writer, payload)
        except asyncio.CancelledError:
            raise
        except (ValueError, KeyError, TypeError):
            await self._reply(writer, _failed("domain_protocol_error", "Invalid local domain request."))
        except TimeoutError:
            await self._reply(writer, _failed("domain_request_timeout",
                "The operation did not return before its deadline. Effects may have occurred; do not resubmit blindly."))
        except Exception:
            await self._reply(writer, _failed("domain_request_unavailable",
                                             "The requested input, source or domain service is unavailable."))
        finally:
            self._active.discard(task)
            writer.close()
            try:
                await writer.wait_closed()
            except (OSError, ConnectionError):
                pass

    async def close(self):
        self._closing = True
        server, self.server = self.server, None
        if server is not None:
            server.close()
            # On recent Python versions wait_closed also waits for clients.
            # Stop reads first, including callbacks already scheduled at close.
            await asyncio.sleep(0)
        active = tuple(self._active)
        for task in active:
            task.cancel()
        if active:
            await asyncio.gather(*active, return_exceptions=True)
        if server is not None:
            await server.wait_closed()
        if self.terminal is not None:
            self.terminal.close()
        if self._published:
            (self.state / "domain-api.json").unlink(missing_ok=True)
            self._published = False
        if self.socket_path is not None:
            self.socket_path.unlink(missing_ok=True)
            self.socket_path.parent.rmdir()
            self.socket_path = None
        if self._lock is not None:
            os.close(self._lock)
            self._lock = None
