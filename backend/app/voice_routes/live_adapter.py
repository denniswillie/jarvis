"""Inactive GPT Live VoiceRoute with direct phone WebRTC and one sideband owner.

Composition is deliberately explicit: canonical guards, authenticated media
receipts and a client output fence are required, not replaced with defaults.
No readiness, completed utterance, notification output identity or hearing is
inferred from a successful socket write, timeline proximity or transcript gap.

The extra live.media.* signaling below is a media-driver seam, NOT a change to
the frozen VoiceRoute event/control vocabulary. The application must authenticate
and correlate it. It must not let phone JSON enter the provider reader.
"""
from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import AbstractAsyncContextManager
from dataclasses import asdict, dataclass, field
from hashlib import sha256
import json
import logging
import math
from typing import Any, Literal
from uuid import UUID, uuid4, uuid5

from app.voice_routes.bridge import Clock, utc_now
from app.voice_routes.contracts import (
    CloseRouteCommand, CommandReceipt, ContextReceipt, DeliveryEvidence,
    DeliveryEvidenceState as Evidence, InterruptSpeechCommand, NotificationOffer,
    OperationCapability, RouteCapabilities, RouteConnectionEvent, RouteError,
    RouteIdentity, RouteOperation, RouteStart, SourceRetirementCommand,
    SourceRetirementReceipt, Support, TranscriptRevision, VoiceContextUpdate,
    VoiceRouteEvent, VoiceTransport, WorkProposal,
)
from app.voice_routes.live_protocol import (
    LiveAppendKind, LiveCommandObservation, LiveCommandStatus as Status,
    LiveControlProtocol, LiveDelegationRequest, LiveHistoryMessage, LiveProtocolError,
    LiveProviderFailure, LiveSessionAnswer, LiveSessionConfiguration,
    LiveSessionFinalization, LiveTranscriptFragment, TokenCounter,
    webrtc_session_request,
)
from app.voice_routes.live_transport import LiveBackendTransport, LiveSendEvidence
from app.voice_routes.live_context_content import encode_live_context_content


_LIFECYCLE_LOGGER = logging.getLogger("app.voice_routes.live_adapter.lifecycle")
_RETIREMENT_CODES = frozenset({
    "live_start_cancelled", "live_start_unconfirmed", "context_replacement_requires_fenced_reconnect",
    "notification_source_replaced", "delegation_result_submission_unconfirmed",
    "delegation_result_revoked_after_submission", "append_capacity_exceeded", "append_cancelled",
    "append_write_unconfirmed", "uncorrelated_live_provider_error", "live_reader_failed",
    "remote_context_expired_requires_replacement", "live_deadline_monitor_failed",
    "live_append_acknowledgment_unknown", "live_event_backpressure", "source_retirement",
    "route_closed", "provider_finalized",
})


FenceCommand = InterruptSpeechCommand | CloseRouteCommand
GuardedInput = RouteStart | VoiceContextUpdate | NotificationOffer


@dataclass(frozen=True)
class LiveMediaCommand:
    route: RouteIdentity
    audio_session_id: str
    command_id: UUID
    provider_session_id: str
    phase: Literal["prepare", "activate"]
    answer_sdp: str | None = None

    def payload(self) -> dict:
        """Send through authenticated app signaling, never the provider socket.

        prepare: apply SDP, keep output muted/queued audio discarded, await the
        actual Live session.started on the data channel and selected-route ready.
        activate: open ONLY this still-current prepared generation. Client-local
        close/interrupt invalidates pending activation, even when messages race.
        Both phases need the exact matching live.media.receipt below.
        """
        result = dict(type="live.media." + self.phase, protocol_version=1,
                      route=self.route.model_dump(mode="json"),
                      audio_session_id=self.audio_session_id,
                      command_id=str(self.command_id), provider_session_id=self.provider_session_id)
        if self.phase == "prepare":
            result["answer_sdp"] = self.answer_sdp
        return result


@dataclass(frozen=True)
class LiveMediaReceipt:
    route: RouteIdentity
    audio_session_id: str
    command_id: UUID
    provider_session_id: str
    phase: Literal["prepare", "activate"]
    receipt_id: UUID
    provider_started: bool
    output_gated: bool

    @classmethod
    def from_payload(cls, raw: dict) -> LiveMediaReceipt:
        """Parse AFTER application authentication; this is not authorization."""
        keys = {"type", "protocol_version", "route", "audio_session_id", "command_id",
                "provider_session_id", "phase", "receipt_id", "provider_started", "output_gated"}
        if (not isinstance(raw, dict) or set(raw) != keys or raw["type"] != "live.media.receipt"
                or type(raw["protocol_version"]) is not int or raw["protocol_version"] != 1
                or raw["phase"] not in {"prepare", "activate"}
                or type(raw["provider_started"]) is not bool or type(raw["output_gated"]) is not bool
                or any(not isinstance(raw[k], str) or not 0 < len(raw[k]) <= 256
                       for k in ("audio_session_id", "provider_session_id"))):
            raise ValueError("Invalid Live media receipt.")
        return cls(RouteIdentity.model_validate(raw["route"]), raw["audio_session_id"],
                   UUID(raw["command_id"]), raw["provider_session_id"], raw["phase"],
                   UUID(raw["receipt_id"]), raw["provider_started"], raw["output_gated"])


@dataclass(frozen=True)
class LiveBootstrapGateProof:
    """Host attestation of actual selected-client readiness, NOT a wire receipt.

    Construct only from the authenticated client's received readiness evidence
    for this exact selected route, audio session, connection and offered peer.
    A successful selection send or an SDP offer alone is insufficient. The
    qualified Live client readiness boundary must establish an output gate that
    is already closed, without destroying the peer used to create the offer.

    is_current must synchronously recheck connection/route/peer ownership and
    become false on disconnect, replacement or permanent fencing. is_gated
    reflects the actual gated-readiness state and remains true until authorized
    activation. Neither predicate may perform I/O or infer readiness from time.
    A local UUID or these booleans alone are not evidence: composition owns the
    authenticated receipt history backing this proof. No third media wire phase
    is introduced; prepare/activate still have their own genuine receipts.
    """
    route: RouteIdentity
    audio_session_id: str
    connection_id: UUID
    offer_sha256: str
    selection_written: bool
    client_ready_received: bool
    output_gated: bool
    is_current: Callable[[], bool]
    is_gated: Callable[[], bool]


@dataclass(frozen=True)
class LiveRouteBindings:
    """Dependencies for ONE authenticated route generation.

    source_guard must validate committed roots, exact dependencies and scope at
    the actual write. For RouteStart it covers authorization AND ALL supplied
    history. media_control sends LiveMediaCommand.payload(), awaits a genuine
    matching authenticated receipt, and honors current at its write boundary.
    It must not return readiness merely because it forwarded the command.

    bootstrap_gate returns LiveBootstrapGateProof from an actual authenticated
    selected-client/gated readiness boundary BEFORE the provider POST. Bind
    client_connection_id to the same connection as LiveMediaSignaling. If that
    evidence is unavailable, start fails closed; it never substitutes a send
    acknowledgment or invokes permanent fencing to manufacture a startup gate.

    fence_output uses existing voice_route.interrupt/close + CommandReceipt.
    The phone must keep the gate latched after interruption; this component has
    no qualified same-session fresh-audio boundary for reopening it. Replacement
    closes/fences this generation before another speaker can be admitted.
    """
    identity: RouteIdentity
    audio_session_id: str
    transport: LiveBackendTransport
    configuration: LiveSessionConfiguration
    offer_sdp: str
    count_tokens: TokenCounter
    authorize_start: Callable[[RouteStart], Awaitable[bool]]
    source_guard: Callable[[GuardedInput], AbstractAsyncContextManager[Any]]
    media_control: Callable[[LiveMediaCommand, Callable[[], bool]], Awaitable[LiveMediaReceipt]]
    fence_output: Callable[[FenceCommand], Awaitable[CommandReceipt]]
    client_connection_id: UUID | None = None
    bootstrap_gate: Callable[[RouteStart], Awaitable[LiveBootstrapGateProof]] | None = None


@dataclass
class _Append:
    payload: VoiceContextUpdate | NotificationOffer
    command_id: str
    delegation_id: str | None = None
    last: ContextReceipt | DeliveryEvidence | None = None
    status: Status = Status.PREPARED
    submitted: bool = False
    done: asyncio.Event = field(default_factory=asyncio.Event)


@dataclass(frozen=True)
class LiveTranscriptPart:
    """Approximate provider timing, separate from canonical turn/output identity."""
    transcript_id: UUID
    revision: int
    fragment: LiveTranscriptFragment


@dataclass(frozen=True)
class LiveDelegationWindow:
    """Received before a delegation event; neither a complete turn nor authority."""
    request: LiveDelegationRequest
    parts: tuple[LiveTranscriptPart, ...]
    received_part_count: int

    @property
    def omitted_prefix_parts(self):
        return self.received_part_count - len(self.parts)


class LiveRouteError(RuntimeError):
    pass


class LiveVoiceRoute:
    def __init__(self, bindings: LiveRouteBindings, *, io_timeout: float = 3,
                 start_timeout: float = 20, ack_timeout: float = 5,
                 event_limit: int = 256, append_limit: int = 256,
                 transcript_limit: int = 1024, delegation_limit: int = 32,
                 clock: Clock = utc_now) -> None:
        if any(type(v) not in (int, float) or not math.isfinite(v) or not 0 < v <= 120
               for v in (io_timeout, start_timeout, ack_timeout)):
            raise ValueError("Invalid route deadline.")
        if any(type(v) is not int or not 1 <= v <= 4096
               for v in (event_limit, append_limit, transcript_limit, delegation_limit)):
            raise ValueError("Invalid route capacity.")
        if not bindings.audio_session_id or len(bindings.audio_session_id) > 256:
            raise ValueError("An audio-session binding is required.")
        self._b, self._clock = bindings, clock
        self._io_timeout, self._start_timeout, self._ack_timeout = io_timeout, start_timeout, ack_timeout
        self._event_limit, self._append_limit = event_limit, append_limit
        self._transcript_limit, self._delegation_limit = transcript_limit, delegation_limit
        self._protocol: LiveControlProtocol | None = None
        self._events: deque[VoiceRouteEvent] = deque()
        self._wake, self._delegation_wake, self._finalized = asyncio.Event(), asyncio.Event(), asyncio.Event()
        self._overflow: RouteError | None = None
        self._io: set[asyncio.Task] = set()
        self._reader: asyncio.Task | None = None
        self._timer: asyncio.Task | None = None
        self._shutdown: asyncio.Task | None = None
        self._finalization: LiveSessionFinalization | None = None
        self._append_failure_reports = 0
        self._appends: dict[str, _Append] = {}
        self._notices: dict[UUID, tuple[tuple, dict]] = {}
        self._context: VoiceContextUpdate | None = None
        self._sources: dict[tuple, object] = {}
        self._context_sources: set = set()
        self._history_sources: set = set()
        self._transcripts: dict[tuple[UUID, int], TranscriptRevision] = {}
        self._parts: list[LiveTranscriptPart] = []
        self._current_transcript: dict[str, TranscriptRevision] = {}
        self._chunk_number = 0
        self._delegations: dict[str, LiveDelegationRequest] = {}
        self._delegation_queue: deque[LiveDelegationRequest] = deque()
        self._delegation_windows: dict[str, LiveDelegationWindow] = {}
        self._proposals: dict[str, WorkProposal] = {}
        self._delegation_results = {}
        self._delegation_result_revoked = set()
        self._delegation_result_submitted = set()
        self._media_receipts: set[UUID] = set()
        self._bootstrap_gate: LiveBootstrapGateProof | None = None
        self._epoch = 0
        self._start_attempted = self._starting = self._started = False
        self._retired = self._closed = self._output_held = self._context_busy = False
        self._event_reader = self._delegation_reader = False
        self._capabilities = RouteCapabilities(
            transport=VoiceTransport.NATIVE_VOICE, media_topology="phone_to_provider",
            control_owner="backend", credential_mode="server_only",
            input_codec="webrtc_negotiated", output_codec="webrtc_negotiated", vad_owner="provider",
            explicit_input_commit=Support.UNSUPPORTED, explicit_response_create=Support.UNSUPPORTED,
            full_duplex=Support.SUPPORTED, image_input=Support.UNSUPPORTED,
            provider_context_receipts=Support.SUPPORTED, notification_output_attribution=Support.UNSUPPORTED,
            client_playback_evidence=Support.UNSUPPORTED, provider_interruption=Support.UNSUPPORTED,
            operations=tuple(OperationCapability(operation=op, support=Support.SUPPORTED)
                             for op in RouteOperation),
        )

    @property
    def identity(self) -> RouteIdentity:
        return self._b.identity

    @property
    def capabilities(self) -> RouteCapabilities:
        return self._capabilities

    @property
    def requires_fenced_replacement(self) -> bool:
        return self._output_held or self._retired

    @property
    def finalization(self) -> LiveSessionFinalization | None:
        return self._finalization

    @property
    def transcript_parts(self) -> tuple[LiveTranscriptPart, ...]:
        return tuple(self._parts)

    def delegation_window(self, request: LiveDelegationRequest) -> LiveDelegationWindow:
        window = self._delegation_windows.get(request.delegation_id)
        if window is None or window.request != request:
            raise LiveRouteError("delegation_window_not_owned")
        return window

    async def start(self, request: RouteStart) -> CommandReceipt:
        self._identity(request.route)
        if self._start_attempted or self._retired:
            return self._command(request, "start", "failed", reason="start_reused_or_retired")
        self._start_attempted = self._starting = True
        epoch = self._epoch
        self._emit(RouteConnectionEvent(route=self.identity, observed_at=self._clock(), state="starting"))
        try:
            if request.initial_context.mode != "snapshot" or self._expired(request.initial_context):
                raise LiveRouteError("invalid_initial_context")
            # Validate ALL provider size limits before any paid session creation.
            if request.publish_initial_context:
                self._content(request.initial_context)
            body = webrtc_session_request(self._b.configuration, self._b.offer_sdp,
                history=tuple(LiveHistoryMessage(t.role, t.text) for t in request.relevant_history),
                count_tokens=self._b.count_tokens)
            async with asyncio.timeout(self._start_timeout):
                if await self._call(self._b.authorize_start(request)) is not True:
                    raise LiveRouteError("start_unauthorized")
                await self._acquire_bootstrap_gate(request, epoch)
                self._history_sources = {d for t in request.relevant_history for d in t.dependencies}
                self._remember_sources(self._history_sources)
                answer = await self._call(self._b.transport.create(body,
                    guard=lambda: self._b.source_guard(request), current=lambda: self._bootstrap_current(epoch)),
                    timeout=self._start_timeout)
                await self._call(self._b.transport.attach(answer), timeout=self._start_timeout)
                if not self._bootstrap_current(epoch):
                    raise LiveRouteError("start_fenced")
                self._protocol = LiveControlProtocol(route_id=self.identity.route_id,
                    route_generation=self.identity.route_generation, provider_session_id=answer.session_id,
                    count_tokens=self._b.count_tokens, max_commands=max(32, self._append_limit + 16))
                self._protocol.mark_attached(authorized_session_id=answer.session_id)
                self._reader = asyncio.create_task(self._read(), name="live-sideband-reader")
                self._timer = asyncio.create_task(self._watch(), name="live-route-deadlines")
                await self._media(request, answer, "prepare", epoch)
                if request.publish_initial_context:
                    receipt = await self.publish_context(request.initial_context)
                    state = self._appends.get("ctx_" + request.initial_context.update_id.hex)
                    if state is None or receipt.status not in {"written", "provider_acknowledged"}:
                        raise LiveRouteError("initial_context_unwritten")
                    await self._call(state.done.wait(), timeout=self._ack_timeout)
                    if state.status != Status.ACCEPTED or not self._bootstrap_current(epoch):
                        raise LiveRouteError("initial_context_unacknowledged")
                # Recheck every initial source before admitting audible output.
                async with self._b.source_guard(request):
                    if self._expired(request.initial_context) or not self._bootstrap_current(epoch):
                        raise LiveRouteError("initial_context_or_gate_expired")
                    await self._media(request, answer, "activate", epoch)
                if not self._start_current(epoch):
                    raise LiveRouteError("start_fenced")
                self._started = True
                self._emit(RouteConnectionEvent(route=self.identity, observed_at=self._clock(), state="ready"))
                if not self._start_current(epoch):
                    raise LiveRouteError("start_fenced")
                return self._command(request, "start", "completed")
        except asyncio.CancelledError:
            self._fail("live_start_cancelled")
            raise
        except Exception:
            self._fail("live_start_unconfirmed")
            return self._command(request, "start", "failed", reason="live_start_unconfirmed")
        finally:
            self._starting = False

    async def _acquire_bootstrap_gate(self, request, epoch):
        if (self._b.bootstrap_gate is None or not isinstance(self._b.client_connection_id, UUID)
                or not self._current(epoch) or self._output_held):
            raise LiveRouteError("bootstrap_gate_not_configured_or_current")
        proof = await self._call(self._b.bootstrap_gate(request))
        if (not isinstance(proof, LiveBootstrapGateProof) or proof.route != self.identity
                or proof.audio_session_id != self._b.audio_session_id
                or proof.connection_id != self._b.client_connection_id
                or proof.offer_sha256 != sha256(self._b.offer_sdp.encode("utf-8")).hexdigest()
                or proof.selection_written is not True or proof.client_ready_received is not True
                or proof.output_gated is not True or not callable(proof.is_current)
                or not callable(proof.is_gated)):
            raise LiveRouteError("bootstrap_gate_unconfirmed")
        self._bootstrap_gate = proof
        if not self._bootstrap_current(epoch):
            raise LiveRouteError("bootstrap_gate_stale")

    def _start_current(self, epoch):
        proof = self._bootstrap_gate
        return (self._current(epoch) and not self._output_held and proof is not None
                and proof.is_current() is True)

    def _bootstrap_current(self, epoch):
        return self._start_current(epoch) and self._bootstrap_gate.is_gated() is True

    async def _media(self, request, answer, phase, epoch):
        command = LiveMediaCommand(self.identity, self._b.audio_session_id, request.command_id,
                                   answer.session_id, phase, answer.answer_sdp if phase == "prepare" else None)
        # Activation intentionally opens the gate, so its post-write/receipt
        # predicate checks ownership, not the old gate-closed condition. Local
        # interruption invalidates it synchronously via _output_held; no late
        # receipt may reopen a permanently fenced peer.
        current = (lambda: self._bootstrap_current(epoch)) if phase == "prepare" else (lambda: self._start_current(epoch))
        receipt = await self._call(self._b.media_control(command, current),
                                   timeout=self._start_timeout)
        if (not isinstance(receipt, LiveMediaReceipt) or receipt.route != self.identity
                or receipt.audio_session_id != command.audio_session_id or receipt.command_id != command.command_id
                or receipt.provider_session_id != command.provider_session_id or receipt.phase != phase
                or not isinstance(receipt.receipt_id, UUID) or receipt.receipt_id in self._media_receipts
                or receipt.provider_started is not True or receipt.output_gated is not (phase == "prepare")
                or not current()):
            raise LiveRouteError("media_readiness_unconfirmed")
        self._media_receipts.add(receipt.receipt_id)

    async def publish_context(self, update: VoiceContextUpdate) -> ContextReceipt:
        self._identity(update.route)
        key = "ctx_" + update.update_id.hex
        saved = self._appends.get(key)
        if saved:
            if saved.payload != update:
                raise LiveRouteError("context_identity_reused")
            return saved.last or self._context_receipt(update, "queued")
        if not self._open() or self._expired(update) or self._context_busy:
            return self._context_receipt(update, "rejected", "context_not_admissible")
        if self._context and (update.cursor.epoch != self._context.cursor.epoch
                             or update.cursor.revision <= self._context.cursor.revision):
            return self._context_receipt(update, "rejected", "context_cursor_mismatch")
        sources = set(update.dependencies)
        roots_removed = (self._context and not self._context.dependencies and self._context.guard_binding
            and (update.guard_binding is None or not set(self._context.guard_binding.payload_root_ids)
                 <= set(update.guard_binding.payload_root_ids)))
        historical = ({source for observation in self._context.observation_history
                       for source in observation.dependencies} if self._context else set())
        # Omission from a current view does not revoke an explicitly timestamped
        # observation. The cumulative _sources conflict map and session source
        # watcher retain ALL remote-source obligations; no roots are forgotten.
        removed_nonhistorical = self._context_sources - sources - historical
        if (self._source_conflict(sources) or update.mode == "snapshot"
                and (removed_nonhistorical or roots_removed)):
            self._fail("context_replacement_requires_fenced_reconnect")
            return self._context_receipt(update, "rejected", "context_replacement_requires_fenced_reconnect")
        self._context_busy = True
        try:
            state = self._prepare(update, key, LiveAppendKind.THINKING)
            await self._write_append(state, LiveAppendKind.THINKING)
            if state.submitted and self._open():
                self._context = update
                self._context_sources = sources if update.mode == "snapshot" else self._context_sources | sources
            return state.last
        except LiveProtocolError as error:
            return self._context_receipt(update, "rejected", error.code)
        finally:
            self._context_busy = False

    async def offer_notification(self, offer: NotificationOffer, *,
                                 delegation_id: str | None = None) -> DeliveryEvidence:
        """Commentary submission is speech-eligible; its ACK is NOT output evidence."""
        self._identity(offer.route)
        key = "offer_" + offer.offer_id.hex
        saved = self._appends.get(key)
        if saved:
            if saved.payload != offer or saved.delegation_id != delegation_id:
                raise LiveRouteError("offer_identity_reused")
            return saved.last
        if (not self._started or not self._open() or self._output_held or self._expired(offer)
                or any(isinstance(s.payload, NotificationOffer)
                       and s.payload.attempt.attempt_id == offer.attempt.attempt_id for s in self._appends.values())):
            return self._delivery(offer, Evidence.FAILED, "notification_not_admissible")
        if self._source_conflict({offer.attempt.source}):
            self._fail("notification_source_replaced")
            return self._delivery(offer, Evidence.FAILED, "notification_source_replaced")
        try:
            state = self._prepare(offer, key, LiveAppendKind.COMMENTARY, delegation_id)
        except LiveProtocolError as error:
            return self._delivery(offer, Evidence.FAILED, error.code)
        await self._write_append(state, LiveAppendKind.COMMENTARY)
        return state.last

    def _content(self, payload) -> str:
        if isinstance(payload, VoiceContextUpdate):
            # Cursor identity remains on the local update, guard and receipt.
            # The shared helper budgets exactly the factual provider content.
            result = encode_live_context_content(payload.text, mode=payload.mode)
        else:
            result = json.dumps({"kind": "authorized_speakable_result", "text": payload.text},
                                ensure_ascii=True, separators=(",", ":"))
        count = self._b.count_tokens(result)
        if type(count) is not int or not 0 <= count <= 500:
            raise LiveProtocolError("append_token_budget_exceeded")
        return result

    async def submit_delegation_result(self, result, *, guard):
        """Reply via the existing protocol/reader, without notification receipts.

        Host guard validates exact task/result/source ownership at EVERY write.
        A lost ACK is not replayed. Source withdrawal after injection requires
        the host's existing output fence, just like ordinary remote context.
        """
        from app.voice_routes.live_results import LiveDelegationResult, LiveDelegationSubmission
        if not isinstance(result, LiveDelegationResult):
            raise TypeError("A typed delegation result is required")
        self._identity(result.route)
        key = result.delegation_id
        if key in self._delegation_result_revoked:
            return LiveDelegationSubmission(key, "rejected", reason="delegation_result_revoked")
        # A delegation can receive acceptance, progress and completion. Each
        # immutable source payload has its own non-reissued submission receipt.
        result_key = (key, result.result_payload_id)
        saved = self._delegation_results.get(result_key)
        if saved is not None:
            if saved[0] != result:
                raise LiveRouteError("delegation_result_identity_reused")
            return saved[1]
        if (key not in self._delegations or not self._started or not self._open()
                or self._output_held or self._expired(result) or len(self._delegation_results) >= self._delegation_limit):
            return LiveDelegationSubmission(key, "rejected", reason="delegation_result_not_current")
        try:
            chunks = result.chunks(self._b.count_tokens)
        except ValueError:
            return LiveDelegationSubmission(key, "rejected", reason="delegation_result_budget")
        self._delegation_results[result_key] = (result, LiveDelegationSubmission(key, "pending"))
        epoch, submitted = self._epoch, False
        def current():
            return (self._current(epoch) and not self._output_held and not self._expired(result)
                    and key not in self._delegation_result_revoked)
        try:
            for index, content in enumerate(chunks):
                kind = LiveAppendKind.COMMENTARY if result.speak and index == len(chunks) - 1 else LiveAppendKind.THINKING
                command = self._protocol.prepare_append(kind, content,
                    command_id=f"dr_{sha256(key.encode()).hexdigest()}_{result.result_payload_id.hex}_{index}",
                    delegation_id=key, timeout=self._ack_timeout)
                def take():
                    nonlocal submitted
                    submitted = True
                    self._delegation_result_submitted.add(key)
                    return self._protocol.take_for_write(command.command_id)
                sent = await self._call(self._b.transport.send(take, guard=lambda: guard(result), current=current))
                if not sent or not current():
                    raise LiveRouteError("delegation_result_write_fenced")
                self._protocol.mark_written(command.command_id)
                async with asyncio.timeout(self._ack_timeout + .1):
                    while current():
                        observed = self._protocol.observation(command.command_id)
                        if observed.status == Status.ACCEPTED:
                            break
                        if observed.status in {Status.REJECTED, Status.UNKNOWN}:
                            raise LiveRouteError("delegation_result_unacknowledged")
                        await asyncio.sleep(.01)
                    else:
                        raise LiveRouteError("delegation_result_fenced")
            receipt = LiveDelegationSubmission(key, "provider_received", provider_received=True)
        except BaseException as error:
            from app.attention.quiet_delivery import QuietDeliverySuppressed
            quiet = isinstance(error, QuietDeliverySuppressed)
            receipt = LiveDelegationSubmission(key, "unknown" if submitted else "rejected",
                                               reason=error.code if quiet else "delegation_result_submission_unconfirmed")
            self._delegation_results[result_key] = (result, receipt)
            # Earlier context-only chunks remain source-valid. Suppression of
            # final commentary is not source withdrawal and must not close a
            # route/cancel its other work. This receipt is still non-replayable.
            if submitted and not quiet:
                self._fail("delegation_result_submission_unconfirmed")
            if isinstance(error, (asyncio.CancelledError, KeyboardInterrupt, SystemExit)):
                raise
            return receipt
        self._delegation_results[result_key] = (result, receipt)
        return receipt

    def revoke_delegation_result(self, delegation_id):
        """Synchronous local fence; the host separately revokes durable admission.

        Never treat natural barge-in or a transcript fragment as this command.
        Once even quiet result context may have been sent, replace the route:
        partial remote history cannot be reliably retracted in place.
        """
        if delegation_id not in self._delegations:
            raise LiveRouteError("delegation_result_not_owned")
        self._delegation_result_revoked.add(delegation_id)
        if delegation_id in self._delegation_result_submitted:
            self._fail("delegation_result_revoked_after_submission")

    async def submit_notice(self, *, notice_id, text, expires_at, guard, kind="authorized_reminder"):
        """Authorized operational notice, not a fabricated client delegation.

        Reuses the existing sideband protocol/reader. The owning service gates
        current reminder/source generation or retained completion at the actual write.
        A provider ACK is never labeled as speech heard by the user.
        """
        if not isinstance(notice_id, UUID) or not isinstance(text, str) or not text.strip():
            raise ValueError("An exact notice ID and nonempty text are required")
        if kind not in {"authorized_reminder", "hermes_completion", "app_action_result"}:
            raise ValueError("Unknown operational notice kind")
        content = json.dumps({"kind": kind, "text": text},
                             ensure_ascii=True, separators=(",", ":"))
        identity = (kind, text, expires_at)
        if notice_id in self._notices:
            previous, receipt = self._notices[notice_id]
            if previous != identity:
                raise LiveRouteError("notice_identity_reused")
            return dict(receipt)
        if (not self._started or not self._open() or self._output_held
                or expires_at.tzinfo is None or expires_at <= self._clock()
                or len(self._notices) >= self._append_limit or self._b.count_tokens(content) > 500):
            return {"status": "rejected", "provider_received": False}
        receipt = {"status": "unknown", "provider_received": False}
        self._notices[notice_id] = (identity, receipt)
        epoch, submitted = self._epoch, False
        def current():
            return self._current(epoch) and not self._output_held and self._clock() < expires_at
        try:
            command = self._protocol.prepare_append(LiveAppendKind.COMMENTARY, content,
                command_id="notice_" + notice_id.hex, timeout=self._ack_timeout)
            def take():
                nonlocal submitted
                submitted = True
                return self._protocol.take_for_write(command.command_id)
            sent = await self._call(self._b.transport.send(take, guard=guard, current=current))
            if not sent or not current():
                raise LiveRouteError("notice_write_fenced")
            self._protocol.mark_written(command.command_id)
            async with asyncio.timeout(self._ack_timeout + .1):
                while current():
                    observation = self._protocol.observation(command.command_id)
                    if observation.status == Status.ACCEPTED:
                        receipt.update(status="provider_received", provider_received=True)
                        return dict(receipt)
                    if observation.status in {Status.REJECTED, Status.UNKNOWN}:
                        break
                    await asyncio.sleep(.01)
                raise LiveRouteError("notice_unacknowledged")
        except BaseException as error:
            from app.attention.quiet_delivery import QuietDeliverySuppressed
            receipt["status"] = "unknown" if submitted else "rejected"
            if submitted and not isinstance(error, QuietDeliverySuppressed):
                self._fail("append_write_unconfirmed")
            if isinstance(error, (asyncio.CancelledError, KeyboardInterrupt, SystemExit)):
                raise
            return dict(receipt)
    def _prepare(self, payload, key, kind, delegation_id=None):
        if len(self._appends) >= self._append_limit:
            self._fail("append_capacity_exceeded")
            raise LiveProtocolError("append_capacity_exceeded")
        command = self._protocol.prepare_append(kind, self._content(payload), command_id=key,
            delegation_id=delegation_id, timeout=self._ack_timeout)
        state = _Append(payload, command.command_id, delegation_id)
        self._appends[key] = state
        state.last = (self._context_receipt(payload, "queued") if isinstance(payload, VoiceContextUpdate)
                      else self._delivery(payload, Evidence.OFFERED))
        return state

    async def _write_append(self, state, kind):
        epoch = self._epoch
        evidence = None
        def observed(value):
            nonlocal evidence
            evidence = value
        def current():
            return (self._current(epoch) and not self._expired(state.payload)
                    and (kind != LiveAppendKind.COMMENTARY or not self._output_held))
        def take():
            data = self._protocol.take_for_write(state.command_id)
            state.submitted = True
            deps = (state.payload.dependencies if isinstance(state.payload, VoiceContextUpdate)
                    else (state.payload.attempt.source,))
            self._remember_sources(deps)
            if isinstance(state.payload, NotificationOffer):
                state.last = self._delivery(state.payload, Evidence.SUBMITTED,
                                           "commentary_write_started_output_unattributed")
            return data
        try:
            sent = await self._call(self._b.transport.send(take,
                guard=lambda: self._b.source_guard(state.payload), current=current, observe=observed))
            if not sent or not current():
                raise LiveRouteError("append_write_fenced")
            self._command_observed(self._protocol.mark_written(state.command_id))
        except asyncio.CancelledError:
            self._append_failure_diagnostic(state, evidence, "cancelled")
            self._unknown(state, "append_cancelled_without_replay")
            self._fail("append_cancelled")
            raise
        except Exception:
            self._append_failure_diagnostic(state, evidence, "error")
            self._unknown(state, "append_write_unconfirmed")
            self._fail("append_write_unconfirmed")

    def _append_failure_diagnostic(self, state, evidence, outcome):
        try:
            self._append_failure_reports += 1
            if self._append_failure_reports > 9:
                return
            record = {"schema_version": 1, "route_id": str(self.identity.route_id),
                "route_generation": self.identity.route_generation}
            if self._append_failure_reports == 9:
                record["truncated"] = True
            else:
                context = isinstance(state.payload, VoiceContextUpdate)
                record.update(stage="append_failure", outcome=outcome,
                    payload_kind="context" if context else "notification",
                    payload_id=str(state.payload.update_id if context else state.payload.offer_id),
                    cursor_revision=state.payload.cursor.revision if context else None,
                    payload_taken=state.submitted, append_status=state.status.value,
                    freshness_expired=self._expired(state.payload),
                    expires_at=state.payload.expires_at.isoformat() if state.payload.expires_at is not None else None,
                    transport=asdict(evidence) if isinstance(evidence, LiveSendEvidence) else None)
            _LIFECYCLE_LOGGER.warning("live_append_failure_diagnostic=%s", json.dumps(record, sort_keys=True))
        except Exception:
            pass

    async def interrupt_speech(self, command: InterruptSpeechCommand) -> CommandReceipt:
        self._identity(command.route)
        self._output_held = True
        # No wait on the provider writer or source/persistence transactions.
        fenced = await self._fence(command)
        for state in self._appends.values():
            if isinstance(state.payload, NotificationOffer):
                self._unknown(state, "interrupted_output_not_attributable")
        # Steering is best effort and NEVER opens the playback gate. Keep only
        # one outstanding steering command; repeated interrupts still fence.
        if self._protocol and self._protocol.ready and not getattr(self, "_steering", None):
            self._steering = asyncio.create_task(self._steer_stop())
            self._steering.add_done_callback(lambda t: None if t.cancelled() else t.exception())
        return self._command(command, "interrupt_speech", "completed" if fenced else "unknown",
            fenced=fenced, reason="client_gate_latched_fenced_replacement_required")

    async def _steer_stop(self):
        try:
            record = self._protocol.prepare_append(LiveAppendKind.INSTRUCTIONS,
                "Stop speaking. Wait for further application guidance.", timeout=self._ack_timeout)
            sent = await self._call(self._b.transport.send(
                lambda: self._protocol.take_for_write(record.command_id), current=self._open))
            if sent:
                self._protocol.mark_written(record.command_id)
        except (Exception, asyncio.CancelledError):
            pass

    async def retire_context(self, command: SourceRetirementCommand) -> SourceRetirementReceipt:
        self._identity(command.route)
        task = self._begin_shutdown("source_retirement")
        fenced, _, _ = await asyncio.shield(task)
        receipt = SourceRetirementReceipt(route=self.identity, command_id=command.command_id,
            cursor=command.cursor, observed_at=self._clock(),
            status="reconnect_required" if fenced else "unknown",
            reason="live_history_has_no_verified_in_place_retirement")
        self._emit(receipt)
        return receipt

    async def close(self, command: CloseRouteCommand) -> CommandReceipt:
        self._identity(command.route)
        fenced, stopped, settled = await asyncio.shield(self._begin_shutdown("route_closed", command))
        return self._command(command, "close", "completed" if fenced and settled else "unknown",
            fenced=fenced, stopped=stopped,
            reason=None if fenced and settled else "close_or_output_fence_unconfirmed")

    async def _fence(self, command):
        expected = "close" if isinstance(command, CloseRouteCommand) else "interrupt_speech"
        try:
            receipt = await self._call(self._b.fence_output(command))
            return (isinstance(receipt, CommandReceipt) and receipt.route == self.identity
                and receipt.command_id == command.command_id and receipt.operation == expected
                and receipt.status == "completed" and receipt.output_fenced is True)
        except Exception:
            return False

    def _begin_shutdown(self, reason, command=None):
        if self._shutdown is not None:
            return self._shutdown
        self._retired = self._output_held = True
        self._epoch += 1
        self._emit(RouteConnectionEvent(route=self.identity, observed_at=self._clock(), state="closing", reason=reason))
        command = command or CloseRouteCommand(route=self.identity, command_id=uuid4(),
            requested_at=self._clock(), reason=reason)
        self._retirement_diagnostic(reason, command)
        self._shutdown = asyncio.create_task(self._stop(command), name="live-route-close")
        return self._shutdown

    def _retirement_diagnostic(self, reason, command):
        # At the actual emitter, before scheduling the client fence or losing
        # the initiating cause behind downstream disconnect/finalization.
        try:
            record = {"schema_version": 1, "stage": "adapter_retirement",
                "route_id": str(self.identity.route_id), "route_generation": self.identity.route_generation,
                "command_id": str(command.command_id), "requested_at": command.requested_at.isoformat(),
                "reason": reason if type(reason) is str and reason in _RETIREMENT_CODES else "unrecognized_code",
                "startup_completed": self._started,
                "current_context_id": str(self._context.update_id) if self._context is not None else None,
                "current_context_expired": self._expired(self._context) if self._context is not None else None}
            _LIFECYCLE_LOGGER.warning("live_retirement_diagnostic=%s", json.dumps(record, sort_keys=True))
        except Exception:
            pass

    async def _stop(self, command):
        for task in tuple(self._io):
            task.cancel()
        if self._timer:
            self._timer.cancel()
        steering = getattr(self, "_steering", None)
        if steering:
            steering.cancel()
        fenced = await self._fence(command)
        try:
            if self._protocol and self._protocol.ready and not self._finalized.is_set():
                record = self._protocol.prepare_close(timeout=self._io_timeout)
                sent = await self._call(self._b.transport.send(
                    lambda: self._protocol.take_for_write(record.command_id)))
                if sent:
                    self._protocol.mark_written(record.command_id)
                    await self._call(self._finalized.wait())
        except Exception:
            pass
        # Keep the reader until finalization or the explicit close deadline.
        if self._reader:
            self._reader.cancel()
        stopped = True if self._finalization is not None else None
        try:
            transport_closed = await self._call(self._b.transport.aclose()) is True
        except Exception:
            transport_closed = False
        if self._protocol:
            self._protocol.retire()
        for state in self._appends.values():
            if state.status not in {Status.ACCEPTED, Status.REJECTED, Status.UNKNOWN} or isinstance(state.payload, NotificationOffer):
                self._unknown(state, "route_closed_delivery_unconfirmed")
        pending = {t for t in (self._reader, self._timer, steering, *self._io) if t and not t.done()}
        for task in pending:
            task.cancel()
        if pending:
            _, pending = await asyncio.wait(pending, timeout=self._io_timeout)
        self._closed = True
        self._emit(RouteConnectionEvent(route=self.identity, observed_at=self._clock(),
            state="closed" if transport_closed and not pending else "failed",
            reason="provider_finalized" if stopped else "provider_finalization_unconfirmed"))
        self._wake.set()
        self._delegation_wake.set()
        return fenced, stopped, transport_closed and not pending

    async def _read(self):
        try:
            while True:
                raw = await self._b.transport.receive()
                observed = self._protocol.accept(raw, provider_session_id=self._protocol.provider_session_id)
                if isinstance(observed, LiveSessionFinalization):
                    self._finalization = observed
                    self._finalized.set()
                    if self._shutdown is None:
                        self._begin_shutdown("provider_finalized")
                    return
                if self._retired:
                    continue  # Late evidence cannot reopen this generation.
                if isinstance(observed, LiveCommandObservation):
                    self._command_observed(observed)
                elif isinstance(observed, LiveTranscriptFragment):
                    self._transcript(observed)
                elif isinstance(observed, LiveDelegationRequest):
                    if len(self._delegations) >= self._delegation_limit:
                        raise LiveRouteError("delegation_capacity_exceeded")
                    self._delegations[observed.delegation_id] = observed
                    # Freeze in the sole provider reader, before later captions
                    # can race the asynchronous delegation consumer. Drop only
                    # whole old fragments; retained original deltas are exact.
                    parts, encoded_bytes = [], 0
                    for part in reversed(self._parts):
                        size = len(part.fragment.delta.encode("utf-8"))
                        if encoded_bytes + size > 32768:
                            break
                        parts.append(part)
                        encoded_bytes += size
                    self._delegation_windows[observed.delegation_id] = LiveDelegationWindow(
                        observed, tuple(reversed(parts)), len(self._parts))
                    self._delegation_queue.append(observed)
                    self._delegation_wake.set()
                elif isinstance(observed, LiveProviderFailure):
                    if observed.command_id in self._appends:
                        self._command_observed(self._protocol.observation(observed.command_id))
                    else:
                        self._fail("uncorrelated_live_provider_error")
        except asyncio.CancelledError:
            raise
        except Exception:
            self._fail("live_reader_failed")

    async def _watch(self):
        try:
            while not self._retired:
                self._protocol.expire()
                for state in tuple(self._appends.values()):
                    self._command_observed(self._protocol.observation(state.command_id))
                    # expires_at is still hard expiry. Opted-in observational
                    # fresh_until never extends it or authorizes stale-as-current
                    # claims; reducer expiry publishes an explicit unknown view.
                    if state.submitted and self._expired(state.payload):
                        self._fail("remote_context_expired_requires_replacement")
                        return
                await asyncio.sleep(min(0.1, self._ack_timeout / 4))
        except asyncio.CancelledError:
            raise
        except Exception:
            self._fail("live_deadline_monitor_failed")

    def _command_observed(self, observed):
        state = self._appends.get(observed.command_id)
        if state is None or state.status == observed.status or state.status in {Status.ACCEPTED, Status.REJECTED, Status.UNKNOWN}:
            return
        state.status = observed.status
        if isinstance(state.payload, VoiceContextUpdate):
            status = {Status.WRITTEN: "written", Status.ACCEPTED: "provider_acknowledged",
                      Status.REJECTED: "rejected", Status.UNKNOWN: "unknown"}.get(observed.status)
            if status:
                state.last = self._context_receipt(state.payload, status, observed.reason,
                    observed.command_id if status == "provider_acknowledged" else None)
        elif observed.status in {Status.ACCEPTED, Status.REJECTED, Status.UNKNOWN}:
            status = {Status.ACCEPTED: Evidence.PROVIDER_RECEIVED, Status.REJECTED: Evidence.FAILED,
                      Status.UNKNOWN: Evidence.UNKNOWN}[observed.status]
            state.last = self._delivery(state.payload, status,
                "commentary_injected_output_unattributed" if status == Evidence.PROVIDER_RECEIVED else observed.reason)
        if observed.status in {Status.ACCEPTED, Status.REJECTED, Status.UNKNOWN}:
            state.done.set()
        if observed.status == Status.UNKNOWN:
            self._fail("live_append_acknowledgment_unknown")

    def _transcript(self, fragment):
        text = fragment.delta
        if not text:
            return
        while text:
            previous = self._current_transcript.get(fragment.speaker)
            # Canonical caption capture accepts at most 128 original fragments
            # per caption. Short deltas reach that bound before 4096 characters.
            # Rotate a provisional chunk, not a fabricated finalized utterance.
            if previous is None or len(previous.text) >= 4096 or previous.revision >= 128:
                self._chunk_number += 1
                identity = uuid5(self.identity.route_id, f"live-caption-{self._chunk_number}")
                before, revision = "", 1
            else:
                identity, before, revision = previous.transcript_id, previous.text, previous.revision + 1
            piece, text = text[:4096 - len(before)], text[4096 - len(before):]
            if len(self._transcripts) >= self._transcript_limit:
                raise LiveRouteError("transcript_capacity_exceeded")
            value = TranscriptRevision(route=self.identity, transcript_id=identity, revision=revision,
                observed_at=self._clock(), role=fragment.speaker, text=before + piece, final=False)
            self._transcripts[(identity, revision)] = value
            self._current_transcript[fragment.speaker] = value
            self._parts.append(LiveTranscriptPart(identity, revision, LiveTranscriptFragment(
                fragment.event_id, fragment.speaker, piece, fragment.start_ms, fragment.end_ms)))
            self._emit(value)

    async def delegation_requests(self) -> AsyncIterator[LiveDelegationRequest]:
        """One canonical context/admission consumer; metadata alone is NOT a task.

        Resolve the offset against canonical conversation and retained fragments,
        then call submit_work_proposal. Do not infer a final turn from silence.
        """
        if self._delegation_reader:
            raise LiveRouteError("delegation_reader_already_owned")
        self._delegation_reader = True
        try:
            while True:
                if self._delegation_queue:
                    yield self._delegation_queue.popleft()
                elif self._closed:
                    return
                else:
                    self._delegation_wake.clear()
                    await self._delegation_wake.wait()
        finally:
            self._delegation_reader = False

    def submit_work_proposal(self, proposal: WorkProposal) -> bool:
        """Accept an explicitly correlated proposal, never execute or authorize it.

        The application chooses when sufficient user intent exists. Display
        transcript chunks are not canonical turns. Corrections retain the same
        proposal ID; the shared admission coordinator decides their effects.
        """
        self._identity(proposal.route)
        transcript = self._transcripts.get((proposal.transcript_id, proposal.transcript_revision))
        if (not self._open() or not self._started or proposal.provider_delegation_ref not in self._delegations
                or transcript is None or transcript.role != "user" or transcript.text != proposal.text
                or self._context is None or proposal.cursor != self._context.cursor):
            raise LiveRouteError("work_proposal_not_correlated")
        previous = self._proposals.get(proposal.provider_delegation_ref)
        if previous == proposal:
            return False
        if previous and (previous.proposal_id != proposal.proposal_id
                         or previous.transcript_id != proposal.transcript_id
                         or previous.transcript_revision >= proposal.transcript_revision):
            raise LiveRouteError("delegation_proposal_conflict")
        self._proposals[proposal.provider_delegation_ref] = proposal
        self._emit(proposal)
        return True

    async def events(self) -> AsyncIterator[VoiceRouteEvent]:
        if self._event_reader:
            raise LiveRouteError("event_reader_already_owned")
        self._event_reader = True
        try:
            while True:
                if self._overflow:
                    event, self._overflow = self._overflow, None
                    yield event
                elif self._events:
                    yield self._events.popleft()
                elif self._closed:
                    return
                else:
                    self._wake.clear()
                    await self._wake.wait()
        finally:
            self._event_reader = False

    def _source_conflict(self, sources):
        return any((d.source_namespace, d.source_id) in self._sources
                   and self._sources[(d.source_namespace, d.source_id)] != d for d in sources)

    def _remember_sources(self, sources):
        if self._source_conflict(sources):
            raise LiveRouteError("source_version_conflict")
        self._sources.update({(d.source_namespace, d.source_id): d for d in sources})
        if len(self._sources) > 256:
            raise LiveRouteError("source_capacity_exceeded")

    def _unknown(self, state, reason):
        if state.status == Status.UNKNOWN:
            return
        state.status = Status.UNKNOWN
        state.last = (self._context_receipt(state.payload, "unknown", reason)
            if isinstance(state.payload, VoiceContextUpdate) else self._delivery(state.payload, Evidence.UNKNOWN, reason))
        state.done.set()

    def _context_receipt(self, update, status, reason=None, provider_ref=None):
        receipt = ContextReceipt(route=self.identity, update_id=update.update_id, cursor=update.cursor,
            observed_at=self._clock(), status=status, reason=reason, provider_receipt_ref=provider_ref)
        self._emit(receipt)
        return receipt

    def _delivery(self, offer, state, reason=None):
        evidence = DeliveryEvidence(evidence_id=uuid4(), route=self.identity, transport=VoiceTransport.NATIVE_VOICE,
            observed_at=self._clock(), state=state, attribution="explicit", offer_id=offer.offer_id,
            attempt=offer.attempt, reason=reason)
        self._emit(evidence)
        return evidence

    def _command(self, request, operation, status, *, fenced=None, stopped=None, reason=None):
        receipt = CommandReceipt(route=self.identity, command_id=request.command_id, operation=operation,
            observed_at=self._clock(), status=status, output_fenced=fenced, provider_stopped=stopped, reason=reason)
        self._emit(receipt)
        return receipt

    def _identity(self, identity):
        if identity != self.identity:
            raise LiveRouteError("route_identity_mismatch")

    def _open(self):
        return (self._starting or self._started) and not self._retired and not self._closed

    def _current(self, epoch):
        return self._open() and self._epoch == epoch

    def _expired(self, payload):
        return payload.expires_at is not None and payload.expires_at <= self._clock()

    def _fail(self, code):
        if self._retired:
            return
        self._emit(RouteError(route=self.identity, observed_at=self._clock(), code=code,
            message="Live route requires fenced replacement; no automatic replay.", recoverable=False))
        self._begin_shutdown(code)

    def _emit(self, event):
        if len(self._events) >= self._event_limit:
            if not self._retired:
                self._overflow = RouteError(route=self.identity, observed_at=self._clock(),
                    code="live_event_backpressure", message="Route evidence requires reconciliation.", recoverable=False)
                self._begin_shutdown("live_event_backpressure")
        else:
            self._events.append(event)
        self._wake.set()

    async def _call(self, operation, *, timeout=None):
        if len(self._io) >= 64:
            operation.close()
            raise LiveRouteError("io_capacity_exceeded")
        task = asyncio.create_task(operation)
        self._io.add(task)
        def finished(done):
            self._io.discard(done)
            if not done.cancelled():
                done.exception()
        task.add_done_callback(finished)
        try:
            done, _ = await asyncio.wait({task}, timeout=timeout or self._io_timeout)
        except asyncio.CancelledError:
            task.cancel()
            raise
        if not done:
            task.cancel()
            raise LiveRouteError("operation_deadline")
        if task.cancelled():
            # Shutdown cancels owned I/O independently of the public caller.
            # Report that as an operation failure, not cancellation of start()
            # or publish_context(). Explicit caller cancellation is propagated
            # by the asyncio.wait branch above and remains distinguishable.
            raise LiveRouteError("owned_operation_cancelled")
        return task.result()
