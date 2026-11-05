"""Authenticated app-side Live media signaling, not a provider event reader.

The host supplies its authenticated connection identity and actual writer. The
writer must call the supplied predicate immediately before writing while holding
its own serialization lock. Receiving a message never waits for provider, model,
database, or another client message. This broker proves only correlated client
readiness/gate receipts; it never proves output attribution, playback or hearing.

Closing this object invalidates signaling only. The route owner must separately
fence the phone's media through the existing interruption/close contract.
"""
from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from hashlib import sha256
import json
import math
from uuid import UUID

from app.voice_routes.contracts import RouteIdentity
from app.voice_routes.live_adapter import LiveMediaCommand, LiveMediaReceipt


class LiveMediaSignalingError(RuntimeError):
    """Stable diagnostic codes; never includes SDP or untrusted payload bodies."""


@dataclass
class _Pending:
    command: LiveMediaCommand
    current: Callable[[], bool]
    future: asyncio.Future
    submission_started: bool = False
    writer: asyncio.Task | None = None


class LiveMediaSignaling:
    """One single-use prepare/activate handshake for an authenticated generation.

    Bind ``request`` to LiveRouteBindings.media_control. The existing app reader
    may pass only authenticated ``live.media.receipt`` messages to ``receive``;
    these are never passed to the provider sideband parser. The host must close
    the broker on connection loss, local fencing or route replacement.
    """

    def __init__(self, *, route: RouteIdentity, audio_session_id: str,
                 connection_id: UUID,
                 send: Callable[[dict, Callable[[], bool]], Awaitable[bool]],
                 timeout_seconds: float = 15):
        if (not isinstance(route, RouteIdentity) or not isinstance(connection_id, UUID)
                or not isinstance(audio_session_id, str) or not 0 < len(audio_session_id) <= 256
                or not callable(send) or type(timeout_seconds) not in (int, float)
                or not math.isfinite(timeout_seconds) or not 0 < timeout_seconds <= 60):
            raise ValueError("Invalid Live media signaling configuration.")
        self.route, self.audio_session_id, self.connection_id = route, audio_session_id, connection_id
        self._send, self._timeout = send, timeout_seconds
        self._pending: dict[tuple[UUID, str], _Pending] = {}
        self._fingerprints: dict[tuple[UUID, str], bytes] = {}
        self._completed: dict[tuple[UUID, str], LiveMediaReceipt] = {}
        self._receipt_ids: dict[UUID, LiveMediaReceipt] = {}
        self._prepared: tuple[UUID, str] | None = None
        self._closed = False

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    def _validate(self, command):
        if (not isinstance(command, LiveMediaCommand) or command.route != self.route
                or command.audio_session_id != self.audio_session_id
                or not isinstance(command.command_id, UUID)
                or not isinstance(command.provider_session_id, str)
                or not 0 < len(command.provider_session_id) <= 256
                or command.phase not in {"prepare", "activate"}):
            raise LiveMediaSignalingError("media_command_scope")
        if command.phase == "prepare":
            if (not isinstance(command.answer_sdp, str) or not command.answer_sdp.startswith("v=0")
                    or len(command.answer_sdp.encode()) > 65536):
                raise LiveMediaSignalingError("media_answer_invalid")
        elif command.answer_sdp is not None:
            raise LiveMediaSignalingError("media_activation_shape")

    async def request(self, command: LiveMediaCommand, current: Callable[[], bool]) -> LiveMediaReceipt:
        self._validate(command)
        if self._closed or current() is not True:
            raise LiveMediaSignalingError("media_signaling_closed_or_stale")
        key = (command.command_id, command.phase)
        fingerprint = sha256(json.dumps(command.payload(), sort_keys=True).encode()).digest()
        if key in self._fingerprints:
            if self._fingerprints[key] != fingerprint:
                raise LiveMediaSignalingError("media_command_identity_conflict")
            if key in self._completed:
                return self._completed[key]
            raise LiveMediaSignalingError("media_command_already_submitted")
        if len(self._fingerprints) >= 2:
            raise LiveMediaSignalingError("media_handshake_already_used")
        if command.phase == "prepare":
            if self._fingerprints:
                raise LiveMediaSignalingError("media_preparation_already_submitted")
        elif self._prepared != (command.command_id, command.provider_session_id):
            raise LiveMediaSignalingError("media_activation_without_preparation")
        self._fingerprints[key] = fingerprint
        pending = _Pending(command, current, asyncio.get_running_loop().create_future())
        self._pending[key] = pending

        def allow_write():
            if (self._closed or self._pending.get(key) is not pending or current() is not True):
                return False
            pending.submission_started = True
            return True

        try:
            async with asyncio.timeout(self._timeout):
                pending.writer = asyncio.create_task(self._send(command.payload(), allow_write),
                    name="live-client-media-write")
                written = await pending.writer
                if (written is not True or not pending.submission_started
                        or self._closed or current() is not True):
                    raise LiveMediaSignalingError("media_write_unconfirmed")
                # A receipt may arrive while the writer is returning. It is not
                # accepted as a successful handshake until the write completes.
                receipt = await pending.future
                if self._closed or current() is not True:
                    raise LiveMediaSignalingError("media_receipt_stale")
                self._completed[key] = receipt
                if command.phase == "prepare":
                    self._prepared = (command.command_id, command.provider_session_id)
                return receipt
        except asyncio.CancelledError:
            already_closed = self._closed
            self.close()
            if already_closed:
                raise LiveMediaSignalingError("media_signaling_closed") from None
            raise
        except TimeoutError:
            self.close()
            raise LiveMediaSignalingError("media_receipt_timeout") from None
        except Exception as error:
            self.close()
            if isinstance(error, LiveMediaSignalingError):
                raise
            raise LiveMediaSignalingError("media_signaling_failed") from None
        finally:
            if not pending.future.done():
                pending.future.cancel()
            if pending.writer is not None and not pending.writer.done():
                pending.writer.cancel()
                await asyncio.gather(pending.writer, return_exceptions=True)
            self._pending.pop(key, None)

    def receive(self, payload: dict, *, connection_id: UUID) -> bool:
        """Call only after app authentication. Never await on the reader lane."""
        if self._closed or connection_id != self.connection_id:
            return False
        try:
            receipt = LiveMediaReceipt.from_payload(payload)
        except (TypeError, ValueError, KeyError, AttributeError):
            return False
        if receipt.route != self.route or receipt.audio_session_id != self.audio_session_id:
            return False
        key = (receipt.command_id, receipt.phase)
        previous = self._completed.get(key)
        if previous is not None:
            return previous == receipt
        pending = self._pending.get(key)
        if (pending is None or not pending.submission_started
                or pending.command.provider_session_id != receipt.provider_session_id
                or receipt.provider_started is not True
                or receipt.output_gated is not (receipt.phase == "prepare")):
            return False
        if pending.current() is not True:
            self.close()
            return False
        prior_id = self._receipt_ids.get(receipt.receipt_id)
        if prior_id is not None and prior_id != receipt:
            return False
        if pending.future.done():
            return not pending.future.cancelled() and pending.future.result() == receipt
        self._receipt_ids[receipt.receipt_id] = receipt
        pending.future.set_result(receipt)
        return True

    def close(self) -> None:
        """Invalidate pending signaling immediately, without claiming media stopped."""
        self._closed = True
        for pending in tuple(self._pending.values()):
            if not pending.future.done():
                pending.future.cancel()
            if pending.writer is not None and not pending.writer.done():
                pending.writer.cancel()

    async def aclose(self) -> None:
        self.close()
        writers = tuple(p.writer for p in self._pending.values() if p.writer is not None)
        if writers:
            await asyncio.gather(*writers, return_exceptions=True)
