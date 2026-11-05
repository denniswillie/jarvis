"""Authenticated local transport to the pinned Hermes session authority.

This is an internal service client, never a model-facing arbitrary RPC tool.
Its profile directory must be selected by trusted owner-scoped composition.
No model SDK, database, task runner, voice provider or sandbox dependency belongs
here. Notifications are coalesced invalidations; native snapshots/event replay
remain authoritative, and no notification automatically triggers speech.
"""

from __future__ import annotations

import asyncio
from contextlib import suppress
import json
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit
from uuid import uuid4

from websockets.asyncio.client import connect


class HermesGatewayError(RuntimeError):
    def __init__(self, reason: str, *, method: str | None = None):
        self.reason, self.method = reason, method
        super().__init__(f"Hermes gateway: {reason}" + (f" ({method})" if method else ""))


class HermesOutcomeUnknown(HermesGatewayError):
    """A create/submit may have committed. Reconcile using its SAME identity."""

    def __init__(self, method: str, request_id: str | None):
        self.request_id = request_id
        super().__init__("outcome_unknown", method=method)


class HermesGatewayClient:
    """One profile-bound connection with nonblocking notification handling.

Use one instance per trusted profile. The caller owns reconnection and native
event cursors; this class never silently resubmits a mutation or starts services.
The operation allowlist deliberately excludes archival/deletion and arbitrary
administration. A single receive task dispatches replies, so a slow response does
not hold up other requests or accumulate an unbounded notification queue.
"""

    METHODS = frozenset({"session.create", "session.resume", "session.events.since", "prompt.submit",
                         "prompt.receipt", "prompt.cancel", "prompt.steer_queued", "session.redirect", "session.steer"})
    MUTATIONS = frozenset({"session.create", "prompt.submit", "prompt.cancel", "prompt.steer_queued", "session.redirect", "session.steer"})

    def __init__(self, profile_home: Path, *, timeout: float = 10, max_message_bytes: int = 4 * 1024 * 1024):
        if not 0 < timeout <= 120 or not 65536 <= max_message_bytes <= 16 * 1024 * 1024:
            raise ValueError("Invalid Hermes transport bounds")
        self.profile_home = Path(profile_home).resolve()
        self.timeout, self.max_message_bytes = timeout, max_message_bytes
        self._socket = None
        self._reader = None
        self._pending: dict[str, asyncio.Future] = {}
        self._changed = asyncio.Condition()
        self._change_sequence = 0
        self._failure: HermesGatewayError | None = None
        self.connection_generation = 0

    @property
    def change_sequence(self) -> int:
        return self._change_sequence

    async def __aenter__(self):
        await self.open()
        return self

    async def __aexit__(self, *_args):
        await self.close()

    def _control_path(self) -> str:
        pointer = self.profile_home / "gateway.sock.path"
        path = pointer.read_text(encoding="utf-8").strip() if pointer.exists() else str(self.profile_home / "gateway.sock")
        if not path or "\x00" in path or not Path(path).is_absolute():
            raise HermesGatewayError("invalid_control_path")
        return path

    async def _control(self, verb: str, params: dict | None = None) -> dict:
        writer = None
        try:
            async with asyncio.timeout(self.timeout):
                path = await asyncio.to_thread(self._control_path)
                reader, writer = await asyncio.open_unix_connection(path, limit=65536)
                request = {"protocol": 1, "id": 1, "verb": verb, "params": params or {}}
                writer.write(json.dumps(request).encode() + b"\n")
                await writer.drain()
                reply = json.loads(await reader.readline())
                if not isinstance(reply, dict) or reply.get("ok") is not True or not isinstance(reply.get("result"), dict):
                    raise HermesGatewayError("control_refused")
                return reply["result"]
        except (OSError, ValueError, TimeoutError) as exc:
            raise HermesGatewayError("control_unavailable") from exc
        finally:
            if writer is not None:
                writer.close()
                with suppress(OSError):
                    await writer.wait_closed()

    @staticmethod
    def _websocket_url(origin: Any) -> str:
        if not isinstance(origin, str):
            raise HermesGatewayError("invalid_gateway_origin")
        try:
            parsed = urlsplit(origin)
            valid = (parsed.scheme == "http" and parsed.hostname in {"127.0.0.1", "::1"}
                     and parsed.port is not None and not parsed.username and not parsed.password
                     and parsed.path in {"", "/"} and not parsed.query and not parsed.fragment)
        except ValueError as exc:
            raise HermesGatewayError("invalid_gateway_origin") from exc
        if not valid:
            raise HermesGatewayError("invalid_gateway_origin")
        return "ws://" + parsed.netloc + "/api/ws"

    async def open(self):
        if self._socket is not None:
            raise HermesGatewayError("already_connected")
        description = await self._control("identify")
        profiles = description.get("served_profiles", [])
        if (description.get("state") != "ready" or "session-authority-v1" not in description.get("capabilities", [])
                or not any(isinstance(profile, dict) and profile.get("profile_id") == str(self.profile_home)
                           for profile in profiles)):
            raise HermesGatewayError("profile_not_ready")
        instance = description.get("instance_id")
        if not isinstance(instance, str) or not instance:
            raise HermesGatewayError("invalid_gateway_identity")
        url = self._websocket_url(description.get("api_origin"))
        grant = await self._control("session-ticket", {
            "profile_id": str(self.profile_home), "instance_id": instance, "purpose": "interactive",
        })
        ticket = grant.get("ticket")
        if not isinstance(ticket, str) or not ticket:
            raise HermesGatewayError("invalid_gateway_ticket")
        try:
            self._socket = await connect(
                url, subprotocols=["hermes-gateway-v1", "hermes-gateway-ticket." + ticket],
                open_timeout=self.timeout, close_timeout=2, max_size=self.max_message_bytes,
                max_queue=16, proxy=None,
            )
        except Exception as exc:
            raise HermesGatewayError("connect_failed") from exc
        if self._socket.subprotocol != "hermes-gateway-v1":
            await self._socket.close()
            self._socket = None
            raise HermesGatewayError("protocol_mismatch")
        self._failure = None
        self.connection_generation += 1
        self._reader = asyncio.create_task(self._receive(), name="hermes-gateway-replies")

    async def _receive(self):
        try:
            async for raw in self._socket:
                message = json.loads(raw)
                if not isinstance(message, dict):
                    raise HermesGatewayError("invalid_gateway_message")
                identity = message.get("id")
                if isinstance(identity, str) and identity in self._pending:
                    future = self._pending[identity]
                    if not future.done():
                        future.set_result(message)
                elif "id" not in message:
                    async with self._changed:
                        self._change_sequence += 1
                        self._changed.notify_all()
        except asyncio.CancelledError:
            raise
        except Exception:
            self._failure = HermesGatewayError("connection_lost")
        finally:
            self._failure = self._failure or HermesGatewayError("connection_closed")
            for future in self._pending.values():
                if not future.done():
                    future.set_exception(self._failure)
            async with self._changed:
                self._changed.notify_all()

    async def close(self):
        if self._socket is not None:
            await self._socket.close()
        if self._reader is not None:
            self._reader.cancel()
            with suppress(asyncio.CancelledError):
                await self._reader
        self._socket = self._reader = None

    async def wait_for_change(self, after: int, *, timeout: float | None = None) -> int:
        """Wake to read native state; no event payload is retained or interpreted."""
        if type(after) is not int or after < 0:
            raise ValueError("A nonnegative change sequence is required")
        async with asyncio.timeout(timeout):
            async with self._changed:
                await self._changed.wait_for(lambda: self._change_sequence > after or self._failure is not None)
                if self._failure is not None:
                    raise self._failure
                return self._change_sequence

    async def request(self, method: str, *, _operation_id: str | None = None, **params) -> dict:
        """Internal native call. Mutating requests require a stable caller identity.

        Cancellation after sending does not cancel native work. Reconcile
        admissions with the same request_id/input_id. Live steer/redirect controls
        have no replay identity and must not be resent after an unknown outcome.
        """
        if method not in self.METHODS:
            raise ValueError("This Hermes operation is not exposed by Jarvis")
        identity_field = {"prompt.submit": "input_id", "prompt.cancel": "admission_id",
                          "prompt.steer_queued": "admission_id"}.get(method, "request_id")
        # Native live controls are generation-bound, not replay-idempotent. Their local
        # operation ID identifies uncertainty only and must never go on the wire.
        durable_id = _operation_id if method in {"session.redirect", "session.steer"} else params.get(identity_field)
        if method in self.MUTATIONS and (not isinstance(durable_id, str) or not durable_id or len(durable_id) > 256):
            raise ValueError("Native mutations require a stable bounded request identity")
        if self._socket is None or self._failure is not None:
            raise HermesGatewayError("not_connected", method=method)
        identity = uuid4().hex
        payload = json.dumps({"jsonrpc": "2.0", "id": identity, "method": method, "params": params}, ensure_ascii=False)
        if len(payload.encode("utf-8")) > self.max_message_bytes:
            raise ValueError("Hermes request exceeds the transport bound")
        future = asyncio.get_running_loop().create_future()
        self._pending[identity] = future
        try:
            try:
                async with asyncio.timeout(self.timeout):
                    await self._socket.send(payload)
                    reply = await asyncio.shield(future)
            except (Exception, TimeoutError) as exc:
                if method in self.MUTATIONS:
                    raise HermesOutcomeUnknown(method, durable_id) from exc
                raise HermesGatewayError("request_failed", method=method) from exc
            if "error" in reply:
                error = reply["error"]
                reason = error.get("message") if isinstance(error, dict) else None
                raise HermesGatewayError(reason if isinstance(reason, str) else "native_request_refused", method=method)
            result = reply.get("result")
            if not isinstance(result, dict):
                raise HermesGatewayError("invalid_native_result", method=method)
            return result
        finally:
            self._pending.pop(identity, None)
            if not future.done():
                future.cancel()

    async def submit(self, session_id: str, input_id: str, text: str) -> dict:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Original nonempty input text is required")
        return await self.request("prompt.submit", session_id=session_id, input_id=input_id, text=text)

    async def snapshot(self, session_id: str) -> dict:
        return await self.request("session.resume", session_id=session_id)

    async def events_since(self, session_id: str, replay_epoch: str | int, last_sequence: int) -> dict:
        return await self.request("session.events.since", session_id=session_id,
                                  replay_epoch=replay_epoch, last_sequence=last_sequence)
