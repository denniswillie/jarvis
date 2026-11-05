"""Live route composition over the existing authenticated application socket.

The HTTP/router owner authenticates this connection and supplies the offered SDP
and its current connection token. This object binds selection, received client
readiness, bootstrap proof, media receipts and permanent output fencing to that
same connection. It opens no transport and owns no microphone or task scheduler.

Only the qualified native Live driver's selected-route ready boundary establishes
the initial gate: that driver must hold output disabled before sending ready.
This is client-reported readiness, not proof of physical silence or user hearing.
"""
from __future__ import annotations

from collections.abc import Callable
from hashlib import sha256
from uuid import UUID

from app.realtime.socket_sender import WebSocketEventSender
from app.voice_routes.client_channel import ClientChannelError, VoiceRouteClientChannel
from app.voice_routes.contracts import (
    CloseRouteCommand, InterruptSpeechCommand, RouteCapabilities, RouteIdentity, RouteStart, Support,
    VoiceTransport,
)
from app.voice_routes.live_adapter import LiveBootstrapGateProof
from app.voice_routes.live_media import LiveMediaSignaling


class LiveClientChannelError(RuntimeError):
    pass


class LiveClientChannel:
    """Single-generation app channel; route operations run outside its reader.

    Application composition must select this channel and continue its existing
    reader concurrently. Bind ``bootstrap_gate``, ``media_control`` and
    ``fence_output`` into LiveRouteBindings. Forward authenticated app messages to
    ``receive`` with their actual connection token. The interruption callback is
    a synchronous bounded enqueue into the route owner, never provider I/O.

    ``close`` invalidates signaling only. A healthy shutdown first calls the
    route's close/fence operation and waits for its actual client receipt.
    """

    def __init__(self, *, route: RouteIdentity, audio_session_id: str,
                 connection_id: UUID, offer_sdp: str, capabilities: RouteCapabilities,
                 sender: WebSocketEventSender, current_connection: Callable[[], UUID | None],
                 on_interrupt: Callable[[InterruptSpeechCommand], None],
                 ready_timeout: float = 15, control_timeout: float = 2):
        if (not isinstance(connection_id, UUID) or not isinstance(offer_sdp, str)
                or not offer_sdp.startswith("v=0") or len(offer_sdp.encode()) > 65536
                or not isinstance(sender, WebSocketEventSender)
                or not callable(current_connection) or not callable(on_interrupt)):
            raise ValueError("A real authenticated Live client writer and offer are required.")
        if (capabilities.transport != VoiceTransport.NATIVE_VOICE
                or capabilities.media_topology != "phone_to_provider"
                or capabilities.input_codec != "webrtc_negotiated"
                or capabilities.output_codec != "webrtc_negotiated"
                or capabilities.explicit_input_commit != Support.UNSUPPORTED
                or capabilities.explicit_response_create != Support.UNSUPPORTED
                or capabilities.notification_output_attribution != Support.UNSUPPORTED
                or capabilities.client_playback_evidence != Support.UNSUPPORTED):
            raise ValueError("This channel requires the native Live capability contract.")
        self.route, self.audio_session_id, self.connection_id = route, audio_session_id, connection_id
        self._offer_digest = sha256(offer_sdp.encode()).hexdigest()
        self._sender, self._connection, self._on_interrupt = sender, current_connection, on_interrupt
        self._closed = self._retired = self._gated = False
        self._selection_write_started = self._selection_complete = self._ready_received = False
        self._activation_submitted = False
        self._interrupts: dict[UUID, InterruptSpeechCommand] = {}
        self.channel = VoiceRouteClientChannel(route, audio_session_id, capabilities, self._send_control,
            ready_timeout=ready_timeout, control_timeout=control_timeout)
        self.media = LiveMediaSignaling(route=route, audio_session_id=audio_session_id,
            connection_id=connection_id, send=self._send_media, timeout_seconds=ready_timeout)

    def _transport_current(self):
        return not self._closed and not self._sender.closed and self._connection() == self.connection_id

    def _route_current(self):
        return self._transport_current() and not self._retired

    def _proof_current(self):
        return self._route_current() and self._selection_complete and self._ready_received and self.channel.ready

    def _proof_gated(self):
        return self._proof_current() and self._gated and not self._activation_submitted

    async def _send_control(self, payload):
        stopping = payload.get("type") in {"voice_route.interrupt", "voice_route.close"}

        def current():
            if not (self._transport_current() if stopping else self._route_current()):
                return False
            if payload.get("type") == "voice_route.selected":
                self._selection_write_started = True
            return True

        return await self._sender.send_event(payload, should_send=current)

    async def _send_media(self, payload, eligible):
        def current():
            if not self._proof_current() or eligible() is not True:
                return False
            if payload["type"] == "live.media.prepare":
                return self._proof_gated()
            if payload["type"] != "live.media.activate":
                return False
            # Once opening is submitted its outcome may be unknown. Never claim
            # the old closed gate merely because its receipt has not arrived.
            self._activation_submitted = True
            self._gated = False
            return True

        return await self._sender.send_event(payload, should_send=current)

    async def select(self, client_capabilities: dict) -> None:
        if not self._route_current():
            raise LiveClientChannelError("live_client_connection_stale")
        try:
            await self.channel.select(client_capabilities)
            self._selection_complete = True
            if not await self.channel.wait_ready() or not self._proof_gated():
                raise LiveClientChannelError("live_client_ready_unconfirmed")
        except BaseException:
            self.close()
            raise

    async def bootstrap_gate(self, request: RouteStart) -> LiveBootstrapGateProof:
        if request.route != self.route or not self._proof_gated():
            raise LiveClientChannelError("live_bootstrap_client_proof_unavailable")
        return LiveBootstrapGateProof(route=self.route, audio_session_id=self.audio_session_id,
            connection_id=self.connection_id, offer_sha256=self._offer_digest,
            selection_written=self._selection_complete, client_ready_received=self._ready_received,
            output_gated=self._gated, is_current=self._proof_current, is_gated=self._proof_gated)

    async def media_control(self, command, current):
        if not self._proof_current():
            raise LiveClientChannelError("live_media_client_unavailable")
        return await self.media.request(command, current)

    def _retire(self):
        self._retired = True
        self._gated = False
        self.media.close()

    async def fence_output(self, command: InterruptSpeechCommand | CloseRouteCommand):
        if command.route != self.route:
            raise LiveClientChannelError("live_fence_scope_mismatch")
        self._retire()
        # Retirement prevents activation but keeps the existing reader/writer
        # alive for this actual stop command and its correlated receipt.
        return await self.channel.request_fence(command)

    def receive(self, payload: dict, *, connection_id: UUID) -> bool:
        if (connection_id != self.connection_id or not self._transport_current()
                or not isinstance(payload, dict)):
            return False
        kind = payload.get("type")
        if kind == "voice_route.event":
            return self.channel.accept_receipt(payload)
        if not self._route_current():
            return False
        if kind == "voice_route.ready":
            if not self._selection_write_started or not self.channel.accept_ready(payload):
                return False
            self._ready_received = True
            if not self._activation_submitted:
                self._gated = True
            return True
        if kind == "live.media.receipt":
            return self.media.receive(payload, connection_id=connection_id)
        if kind == "voice_route.interrupt":
            try:
                command = self.channel.parse_interrupt(payload)
            except (ClientChannelError, TypeError, ValueError):
                return False
            previous = self._interrupts.get(command.command_id)
            if previous is not None:
                return previous == command
            if len(self._interrupts) >= 128:
                self._retire()
                return False
            self._interrupts[command.command_id] = command
            self._retire()
            self._on_interrupt(command)
            return True
        return False

    def close(self):
        self._closed = True
        self._retire()
        self.channel.close()

    async def aclose(self) -> bool:
        self.close()
        await self.media.aclose()
        return await self.channel.aclose()
