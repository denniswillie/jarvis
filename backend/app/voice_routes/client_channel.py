"""Authenticated phone-control binding, not another media or speech owner.

The existing socket reader calls the synchronous accept methods. It must not
await a route command while that command needs a receipt from the same reader.
All writes use the existing serialized control writer supplied by composition.
"""
from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
import math
from uuid import UUID, uuid4

from app.voice_routes.contracts import (
    ClientPlaybackObservation, CloseRouteCommand, CommandReceipt, DeliveryEvidence,
    DeliveryEvidenceState, InterruptSpeechCommand, NotificationOffer, OutputIdentity,
    RouteCapabilities, RouteIdentity, Support,
)


MEDIA_TYPES = frozenset({
    "audio.webrtc.response", "audio.webrtc.output_item", "audio.webrtc.started",
    "audio.webrtc.stopped", "audio.webrtc.cleared", "audio.output.delta",
    "audio.output.done", "audio.output.clear", "audio.input.ready", "audio.capture_mode.ready",
})


class ClientChannelError(RuntimeError):
    pass


@dataclass
class _PendingCommand:
    command: InterruptSpeechCommand | CloseRouteCommand
    future: asyncio.Future


@dataclass(frozen=True)
class _OutputBinding:
    offer: NotificationOffer
    output: OutputIdentity


class VoiceRouteClientChannel:
    def __init__(self, identity: RouteIdentity, audio_session_id: str,
                 capabilities: RouteCapabilities, send: Callable[[dict], Awaitable[bool | None]], *,
                 clock: Callable[[], datetime] = lambda: datetime.now(UTC),
                 ready_timeout: float = 15.0, control_timeout: float = 2.0):
        if not audio_session_id or len(audio_session_id) > 256:
            raise ValueError("An audio-session identity is required")
        if not all(math.isfinite(value) and value > 0 for value in (ready_timeout, control_timeout)):
            raise ValueError("Client deadlines must be finite and positive")
        self.identity, self.audio_session_id, self.capabilities = identity, audio_session_id, capabilities
        self._send, self.clock = send, clock
        self.ready_timeout, self.control_timeout = ready_timeout, control_timeout
        self._selected = self._selection_written = self._client_ready = self._closed = False
        self._closing = False
        self._ready = asyncio.Event()
        self._commands: dict[UUID, _PendingCommand] = {}
        self._bindings: dict[tuple[UUID, UUID], _OutputBinding] = {}
        self._retired_outputs: set[UUID] = set()
        self._playback_receipts: set[UUID] = set()
        self._rendered: dict[tuple[UUID, UUID], int] = {}
        self._writes: set[asyncio.Task] = set()

    @property
    def ready(self):
        return self._selection_written and self._client_ready and not self._closed and not self._closing

    @staticmethod
    def _keys(message, required, optional=()):
        if not isinstance(message, dict) or not set(required) <= message.keys() or (
                message.keys() - set(required) - set(optional)):
            raise ClientChannelError("unexpected_wire_fields")

    def _outer(self, message, kind, *, payload=False, extra=()):
        required = {"type", "protocol_version", "audio_session_id", "payload" if payload else "route", *extra}
        self._keys(message, required, {"client_event_id"})
        if (message["type"] != kind or type(message["protocol_version"]) is not int
                or message["protocol_version"] != 1 or message["audio_session_id"] != self.audio_session_id):
            raise ClientChannelError("wire_scope_mismatch")
        metadata = message.get("client_event_id")
        if metadata is not None and (not isinstance(metadata, str) or not 0 < len(metadata) <= 256):
            raise ClientChannelError("invalid_transport_identity")
        data = message["payload"] if payload else message
        if not isinstance(data, dict) or RouteIdentity.model_validate(data.get("route")) != self.identity:
            raise ClientChannelError("route_identity_mismatch")
        return data

    @staticmethod
    def _list(capabilities, name, *, integers=False, allow_empty=False):
        values = capabilities[name]
        expected = int if integers else str
        if (not isinstance(values, list) or len(values) > 8 or not values and not allow_empty
                or any(type(value) is not expected for value in values)
                or any(not integers and (not value or len(value) > 128) for value in values)
                or len(set(values)) != len(values)):
            raise ClientChannelError("invalid_client_capabilities")
        return values

    def _compatible(self, client):
        keys = {"supported_versions", "supported_transports", "supported_media_topologies",
                "input_codecs", "output_codecs", "local_output_fencing", "client_playback_boundaries"}
        self._keys(client, keys)
        versions = self._list(client, "supported_versions", integers=True)
        transports = self._list(client, "supported_transports")
        topologies = self._list(client, "supported_media_topologies")
        inputs, outputs = self._list(client, "input_codecs"), self._list(client, "output_codecs")
        playback = self._list(client, "client_playback_boundaries", allow_empty=True)
        if (1 not in versions or self.capabilities.transport.value not in transports
                or self.capabilities.media_topology not in topologies
                or self.capabilities.input_codec not in inputs or self.capabilities.output_codec not in outputs
                or client["local_output_fencing"] is not True
                or any(value != "client_render_lower_bound" for value in playback)
                or self.capabilities.client_playback_evidence == Support.SUPPORTED
                and "client_render_lower_bound" not in playback):
            raise ClientChannelError("incompatible_client_route")

    def _envelope(self, kind, *, payload=None):
        result = {"type": kind, "protocol_version": 1, "audio_session_id": self.audio_session_id}
        if payload is not None:
            result["payload"] = payload
        else:
            result["route"] = self.identity.model_dump(mode="json")
        return result

    async def _write(self, message):
        task = asyncio.create_task(self._send(message))
        self._writes.add(task)
        def finished(done):
            self._writes.discard(done)
            if not done.cancelled():
                done.exception()
        task.add_done_callback(finished)
        try:
            done, _ = await asyncio.wait({task}, timeout=self.control_timeout)
            if not done:
                task.cancel()
                raise ClientChannelError("client_control_write_timeout")
            if task.result() is False:
                raise ClientChannelError("client_control_write_rejected")
        except asyncio.CancelledError:
            task.cancel()
            raise

    async def select(self, client_capabilities):
        if self._selected or self._closed:
            raise ClientChannelError("route_selection_reused")
        self._compatible(client_capabilities)
        self._selected = True
        message = self._envelope("voice_route.selected")
        message["capabilities"] = self.capabilities.model_dump(mode="json")
        try:
            await self._write(message)
        except BaseException:
            self.close()
            raise
        self._selection_written = True
        if self._client_ready:
            self._ready.set()

    def accept_ready(self, message) -> bool:
        if not self._selected or self._closed:
            return False
        try:
            self._outer(message, "voice_route.ready")
        except (ClientChannelError, ValueError, TypeError):
            return False
        self._client_ready = True
        if self._selection_written:
            self._ready.set()
        return True

    async def wait_ready(self) -> bool:
        if self._closed or not self._selected:
            return False
        try:
            async with asyncio.timeout(self.ready_timeout):
                await self._ready.wait()
        except TimeoutError:
            self.close()
            return False
        return self.ready

    def media_envelope(self, payload):
        if not self.ready or not isinstance(payload, dict) or payload.get("type") not in MEDIA_TYPES:
            raise ClientChannelError("media_not_admitted")
        owned = {"protocol_version", "audio_session_id", "route"}
        for key in owned & payload.keys():
            expected = self._envelope(payload["type"])[key]
            if payload[key] != expected:
                raise ClientChannelError("media_identity_conflict")
        return {**payload, **self._envelope(payload["type"])}

    def _unknown(self, command, reason):
        return CommandReceipt(route=self.identity, command_id=command.command_id,
            operation="close" if isinstance(command, CloseRouteCommand) else "interrupt_speech",
            observed_at=self.clock(), status="unknown", output_fenced=None,
            provider_stopped=None, reason=reason)

    def retire_outputs(self):
        identities = {binding.output.output_id for binding in self._bindings.values()}
        if len(self._retired_outputs | identities) > 512:
            self.close()
            raise ClientChannelError("output_retirement_capacity")
        self._retired_outputs.update(identities)
        self._bindings.clear()
        self._rendered.clear()

    async def request_fence(self, command):
        if command.route != self.identity:
            raise ClientChannelError("command_scope_mismatch")
        saved = self._commands.get(command.command_id)
        if saved is not None:
            if saved.command != command:
                raise ClientChannelError("command_identity_reused")
            return await asyncio.shield(saved.future)
        if not self.ready or len(self._commands) >= 128:
            return self._unknown(command, "client_fence_not_available")
        future = asyncio.get_running_loop().create_future()
        self._commands[command.command_id] = _PendingCommand(command, future)
        self.retire_outputs()
        closing = isinstance(command, CloseRouteCommand)
        if closing:
            self._closing = True
        try:
            await self._write(self._envelope("voice_route.close" if closing else "voice_route.interrupt",
                payload=command.model_dump(mode="json")))
            async with asyncio.timeout(self.control_timeout):
                return await asyncio.shield(future)
        except asyncio.CancelledError:
            if not future.done():
                future.set_result(self._unknown(command, "client_fence_cancelled"))
            raise
        except Exception:
            if not future.done():
                future.set_result(self._unknown(command, "client_fence_unconfirmed"))
            return future.result()

    def accept_receipt(self, message) -> bool:
        if self._closed:
            return False
        try:
            payload = self._outer(message, "voice_route.event", payload=True)
            receipt = CommandReceipt.model_validate(payload)
        except (ClientChannelError, ValueError, TypeError):
            return False
        saved = self._commands.get(receipt.command_id)
        if saved is None or saved.future.done() or receipt.provider_stopped is not None:
            return False
        expected = "close" if isinstance(saved.command, CloseRouteCommand) else "interrupt_speech"
        if receipt.operation != expected:
            return False
        saved.future.set_result(receipt)
        return True

    def parse_interrupt(self, message) -> InterruptSpeechCommand:
        if not self.ready:
            raise ClientChannelError("client_interrupt_before_ready")
        command = InterruptSpeechCommand.model_validate(
            self._outer(message, "voice_route.interrupt", payload=True))
        self.retire_outputs()
        return command

    async def send_route_event(self, event):
        if event.route != self.identity or self._closed:
            raise ClientChannelError("route_event_scope_mismatch")
        await self._write(self._envelope("voice_route.event", payload=event.model_dump(mode="json")))

    async def bind_output(self, offer: NotificationOffer, output: OutputIdentity):
        if (not self.ready or offer.route != self.identity or offer.expires_at <= self.clock()
                or output.output_id in self._retired_outputs):
            raise ClientChannelError("output_binding_not_admitted")
        key = (offer.offer_id, output.output_id)
        binding = _OutputBinding(offer, output)
        if key in self._bindings:
            if self._bindings[key] != binding:
                raise ClientChannelError("output_binding_conflict")
            return
        if len(self._bindings) >= 128 or any(
                item.output.output_id == output.output_id and item.output != output
                for item in self._bindings.values()):
            raise ClientChannelError("output_identity_conflict_or_capacity")
        self._bindings[key] = binding
        message = self._envelope("voice_route.output_bound") | {
            "offer_id": str(offer.offer_id), "attempt_id": str(offer.attempt.attempt_id),
            "attempt_generation": offer.attempt.attempt_generation, "output": output.model_dump(mode="json")}
        try:
            await self._write(message)
        except BaseException:
            self._bindings.pop(key, None)
            self._retired_outputs.add(output.output_id)
            raise

    def parse_playback(self, message) -> DeliveryEvidence | None:
        if not self.ready or self.capabilities.client_playback_evidence != Support.SUPPORTED:
            return None
        try:
            data = self._outer(message, "voice_route.playback", extra={
                "offer_id", "attempt_id", "attempt_generation", "output_id", "playback"})
            key = UUID(data["offer_id"]), UUID(data["output_id"])
            binding = self._bindings.get(key)
            if (binding is None or type(data["attempt_generation"]) is not int
                    or UUID(data["attempt_id"]) != binding.offer.attempt.attempt_id
                    or data["attempt_generation"] != binding.offer.attempt.attempt_generation):
                return None
            playback = ClientPlaybackObservation.model_validate(data["playback"])
            if (playback.receipt_id in self._playback_receipts or len(self._playback_receipts) >= 512
                    or playback.rendered_ms is None or playback.rendered_ms <= self._rendered.get(key, 0)):
                return None
            evidence = DeliveryEvidence(evidence_id=uuid4(), route=self.identity,
                transport=self.capabilities.transport, observed_at=self.clock(),
                state=DeliveryEvidenceState.CLIENT_PLAYBACK_OBSERVED, attribution="explicit",
                offer_id=binding.offer.offer_id, attempt=binding.offer.attempt,
                output=binding.output, playback=playback)
        except (ClientChannelError, ValueError, TypeError, KeyError):
            return None
        self._playback_receipts.add(playback.receipt_id)
        self._rendered[key] = playback.rendered_ms
        return evidence

    def close(self):
        self._closed = True
        self._ready.set()
        self._bindings.clear()
        for pending in self._commands.values():
            if not pending.future.done():
                pending.future.set_result(self._unknown(pending.command, "client_channel_closed"))
        for task in tuple(self._writes):
            task.cancel()

    async def aclose(self) -> bool:
        self.close()
        pending = {task for task in self._writes if not task.done()}
        if pending:
            _, pending = await asyncio.wait(pending, timeout=self.control_timeout)
        return not pending
