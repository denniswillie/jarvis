"""Explicit, inactive-by-default wiring on ONE existing Realtime session.

Install before provider readiness and before exposing the session to dispatchers.
The same provider connection, client writer, response owner and receive handlers
remain authoritative. Only typed adapter events have an additional consumer.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import time
from collections.abc import Awaitable, Callable
from contextlib import AbstractAsyncContextManager, AsyncExitStack, asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any
from uuid import UUID, uuid4

from app.attention.models import AttentionError
from app.guard_diagnostics import GuardCancellationDiagnostics
from app.voice_routes.bridge import Clock, ContextOfferResult, ContextValidity, OfferDisposition, utc_now
from app.voice_routes.client_channel import VoiceRouteClientChannel
from app.voice_routes.contracts import (
    CloseRouteCommand, CommandReceipt, ContextReceipt, DeliveryEvidence, DeliveryEvidenceState,
    InterruptSpeechCommand, NotificationOffer, RouteError, RouteStart, SourceRetirementCommand,
    VoiceContextUpdate, VoiceRouteEvent,
)
from app.voice_routes.coordinator import CoordinationPhase, VoiceRouteCoordinator
from app.voice_routes.realtime_adapter import GuardFactory, GuardedInput, RealtimeRouteBindings, RealtimeVoiceRoute
from app.realtime.response_control import ResponsePrewriteRejected
from app.realtime.source_context import LiveContextError
from app.realtime.telemetry import latency_phase


_MEDIA = frozenset({
    "audio.webrtc.response", "audio.webrtc.output_item", "audio.webrtc.started",
    "audio.webrtc.stopped", "audio.webrtc.cleared", "audio.output.delta",
    "audio.output.done", "audio.output.clear", "audio.input.ready", "audio.capture_mode.ready",
})
_OUTPUT_END = frozenset({"audio.webrtc.stopped", "audio.webrtc.cleared"})
_INPUT_READY = frozenset({"audio.input.ready", "audio.capture_mode.ready"})
_READINESS_LOGGER = logging.getLogger(__name__)
_READINESS_LOG_LIMIT = 64
_CONTEXT_RECEIPT_REASONS = frozenset({
    "context_not_current_snapshot", "context_create_rejected", "context_cursor_mismatch",
    "context_write_fenced_or_expired", "context_write_failed",
    "context_response_active",
    "context_acknowledged_stale",
    "context_owner_busy",
})
_COORDINATOR_REASONS = frozenset({
    "context_validity_rejected", "context_expired_before_write", "context_publication_deadline",
    "context_publication_cancelled", "context_publication_failed", "context_receipt_unconfirmed",
    "context_receipt_mismatch_or_unsupported", "route_start_cancelled", "route_start_failed",
    "route_start_invalidated", "route_close_cancelled", "route_close_failed", "source_retirement",
    "context_refresh_deadline", "context_expired_awaiting_snapshot",
    "context_replacement_inflight",
    "context_empty_awaiting_snapshot",
    "context_owner_busy_awaiting_snapshot",
})
_SOURCE_GUARD_CODES = frozenset({
    "live_context_changed", "live_context_revoked", "live_context_storage_unavailable",
    "live_context_transaction_required", "live_context_source_invalid", "live_context_source_limit",
    "voice_context_expired", "voice_context_not_admitted", "live_state_runtime_unavailable",
    "live_state_context_revoked", "live_state_event_changed", "live_state_source_changed",
    "live_state_scope_changed", "live_state_scope_unavailable", "live_task_snapshot_changed",
    "live_task_context_expired", "live_episode_source_changed", "live_episode_context_expired",
    "canonical_owner_busy",
})
_ATTENTION_GUARD_CODES = frozenset({
    "attention_offer_not_current", "attention_route_not_ready", "attention_not_submitted",
    "attention_claim_lost", "attention_session_unavailable", "attention_source_unavailable",
    "attention_source_changed", "attention_source_generation_unavailable",
    "attention_plan_unavailable", "attention_plan_conflict", "attention_offer_conflict",
    "attention_submission_state", "attention_write_ticket_used", "attention_legacy_attempt",
    "owner_quiet_active", "owner_quiet_unavailable", "owner_quiet_release_unavailable",
    "owner_quiet_coalesced",
})


def context_receipt_diagnostic(receipt: ContextReceipt) -> dict:
    """Only typed receipt identity/outcome; never source text or provider references."""
    return {
        "route_id": str(receipt.route.route_id),
        "route_generation": receipt.route.route_generation,
        "update_id": str(receipt.update_id), "cursor_epoch": str(receipt.cursor.epoch),
        "cursor_revision": receipt.cursor.revision, "status": receipt.status,
        "reason": (receipt.reason if receipt.reason in _CONTEXT_RECEIPT_REASONS
                   else None if receipt.reason is None else "unrecognized_code"),
        "observed_at": receipt.observed_at.isoformat(),
    }


@dataclass
class _Dispatch:
    offer: NotificationOffer
    request_id: str
    guard: GuardFactory
    current: Callable[[], bool]
    used: bool = False
    written: bool = False
    response_request_id: str | None = None
    inflight_current: Callable[[], bool] | None = None


class VoiceRouteRuntimeError(RuntimeError):
    pass


class RealtimeVoiceRouteRuntime:
    """Composition supplies already-bound snapshots and canonical guard callbacks.

payload_roots is a synchronous extraction of committed removable payload roots,
not a lookup. atomic_guard validates authorization, source lifecycle and exact
binding under the actual provider/client writer. It must obey the existing
source-lock order. validity runs only in the context bridge's background task.

on_event, if supplied, is a synchronous bounded enqueue into the external ledger
owner, never inline persistence. Notification attempts are prepared/committed by
that owner before offer_notification(). Existing try_attention fails closed until
that separate sink integration exists; no legacy TTS fallback is performed.
"""

    def __init__(
        self, session, request: RouteStart, client_capabilities: dict, *,
        atomic_guard: Callable[[GuardedInput], AbstractAsyncContextManager[Any]],
        payload_roots: Callable[[GuardedInput], tuple[UUID, ...]],
        validity: ContextValidity,
        authorize_start: Callable[[RouteStart], Awaitable[bool]],
        authorize_context_replacement: Callable[[RouteStart], Awaitable[bool]] | None = None,
        normal_turn_scope_guard: GuardFactory | None = None,
        on_event: Callable[[VoiceRouteEvent], None] | None = None,
        input_codec: str = "opus", output_codec: str = "opus",
        clock: Clock = utc_now, ready_timeout: float = 15.0, control_timeout: float = 2.0,
    ) -> None:
        context = session.context
        if (request.route.scope.owner_id != context.owner_id
                or request.route.scope.session_id != context.jarvis_session_id):
            raise ValueError("Voice route scope does not match the authenticated session.")
        if getattr(session.openai, "transport", "websocket") != "webrtc":
            raise ValueError("Opt-in runtime currently supports native WebRTC only; PCM negotiation differs.")
        self.session, self.request = session, request
        self._client_capabilities = client_capabilities
        self._atomic_guard, self._payload_roots = atomic_guard, payload_roots
        self._authorize_start, self._on_event = authorize_start, on_event
        self._authorize_context_replacement = authorize_context_replacement or authorize_start
        self._normal_turn_scope_guard = normal_turn_scope_guard
        self._normal_turn_context = None
        self._clock, self._control_timeout = clock, control_timeout
        self._dispatch_context: ContextVar[_Dispatch | None] = ContextVar("voice_route_dispatch", default=None)
        self._notification_output_context: ContextVar[tuple | None] = ContextVar(
            "voice_route_notification_output", default=None)
        self._dispatches: dict[str, _Dispatch] = {}
        self._response_offers: dict[str, _Dispatch] = {}
        self._offers: dict[UUID, NotificationOffer] = {}
        self._interrupts: dict[UUID, InterruptSpeechCommand] = {}
        self._tasks: set[asyncio.Task] = set()
        self._installed = self._closing = self._closed = False
        self._unavailability_reported = False
        self._context_replacement_active = False
        self._context_replacement_settled = asyncio.Event()
        self._context_replacement_settled.set()
        self._bootstrap_task = self._pump_task = self._shutdown_task = self._finish_task = None
        self.failure: str | None = None
        self.shutdown_confirmed = False
        self._owner_provider_send = session.openai.send_event
        self._input_provider = session.openai
        self._input_audio_session_id = context.audio_session_id
        self._owner_client_handler = session._handle_client_event
        self._owner_provider_handler = session._handle_openai_event
        self._owner_send_text = session._app_sender.send_text
        bindings = RealtimeRouteBindings(
            identity=request.route, audio_session_id=context.audio_session_id, provider=session.openai,
            wait_ready=self._wait_ready, authorize_start=self._authorize,
            source_guard=self._source_guard, admit_response=self._admit_response,
            cancel_response=self._cancel_response, close_control=self._close_control,
            fence_output=self._fence_output,
            context_replacement_guard=self._context_replacement_guard,
            context_reconciliation_current=lambda: self.context_publication_ready,
            on_source_handoff=self._source_handoff,
        )
        self.adapter = RealtimeVoiceRoute(bindings, media_transport="webrtc",
            input_codec=input_codec, output_codec=output_codec, client_playback_supported=False,
            io_timeout=2 * control_timeout + 1, start_timeout=ready_timeout, clock=clock)
        self.channel = VoiceRouteClientChannel(request.route, context.audio_session_id,
            self.adapter.capabilities, session._app_sender.send_event, clock=clock,
            ready_timeout=ready_timeout, control_timeout=control_timeout)
        self.coordinator = VoiceRouteCoordinator(request.route.scope, validity=validity,
            control_timeout=ready_timeout + 2 * control_timeout + 2, clock=clock,
            preflight_expiry_refresh=True)

    @property
    def ready(self) -> bool:
        ready = self.context_publication_ready and not self.context_refresh_pending
        self._readiness_diagnostic("ready_observed", ready=ready)
        return ready

    @property
    def context_publication_ready(self) -> bool:
        return (self._installed and not self._closing and not self._closed and self.channel.ready
                and self.coordinator.status.phase == CoordinationPhase.ACTIVE
                and not self.session._closed.is_set() and not self.session._context_revoked)

    @property
    def context_refresh_pending(self) -> bool:
        bridge = getattr(self.coordinator, "_bridge", None)
        return (getattr(self, "_context_replacement_active", False) is True
                or getattr(bridge, "awaiting_fresh_context", False) is True
                or getattr(bridge, "publication_in_flight", False) is True)

    def _answer_owns_context(self) -> bool:
        pending = getattr(self.session, "_pending_response", None)
        pending_busy = pending is not None and (pending.submitted or
            getattr(self.session, "_context_waiting_request_id", None) != pending.request_id)
        return bool(pending_busy or getattr(self.session, "_active_responses", {})
                    or getattr(self.session, "_direct_output_pending", set())
                    or getattr(self.session, "_native_playback", {}))

    @property
    def context_preparation_ready(self) -> bool:
        """Allow one private binding while an answer owns provider context.

        This does not authorize an offer, provider mutation or output. The
        publication-idle predicate and actual source guards remain unchanged.
        """
        return (self.context_publication_ready
                and not getattr(self, "_context_replacement_active", False))

    @property
    def context_publication_idle(self) -> bool:
        return (self.context_publication_ready and not self._answer_owns_context()
                and not getattr(self, "_context_replacement_active", False))

    async def wait_for_context_ready(self, timeout: float) -> bool:
        bridge = getattr(self.coordinator, "_bridge", None)
        if not self.context_publication_ready:
            return False
        try:
            async with asyncio.timeout(timeout):
                if getattr(self, "_context_replacement_active", False):
                    await self._context_replacement_settled.wait()
                if (getattr(bridge, "awaiting_fresh_context", False) is True
                        and not await bridge.wait_for_fresh_context(timeout)):
                    return False
        except TimeoutError:
            return False
        return self.ready

    def _normal_turn_owned(self, reservation, *, pending=True):
        session, route = self.session, self.request.route
        return (reservation is not None and reservation.kind == "normal_user_turn"
                and self.context_publication_ready
                and session._voice_route_runtime is self and session.openai is self._input_provider
                and self.coordinator.status.identity == route and self.channel.identity == route
                and session.context.owner_id == route.scope.owner_id
                and session.context.jarvis_session_id == route.scope.session_id
                and session.context.audio_session_id == self._input_audio_session_id == self.channel.audio_session_id
                and reservation.turn_generation == session._turn_generation
                and (not pending or session._pending_response is reservation))

    def normal_turn_context_unavailable(self, reservation):
        grant = getattr(self, "_normal_turn_context", None)
        return (grant is not None and grant[0] is reservation
                and not self._context_replacement_active and self._normal_turn_owned(reservation)
                and self.adapter.response_context_empty_proof is grant[1])

    def ready_for_response(self, reservation):
        # Notifications and all legacy callers keep global readiness. An empty
        # optional briefing alone grants neither a new request nor tool authority.
        return self.ready or self.normal_turn_context_unavailable(reservation)

    async def prepare_normal_turn_context(self, reservation, timeout):
        if self.ready_for_response(reservation):
            return True
        guard = getattr(self, "_normal_turn_scope_guard", None)
        if guard is None or not self._normal_turn_owned(reservation):
            return False
        self._normal_turn_context = None
        diagnostics = getattr(self, "_normal_context_guard_diagnostics", None)
        if diagnostics is None:
            diagnostics = self._normal_context_guard_diagnostics = GuardCancellationDiagnostics()
        trace = self.session._trace
        fields = dict(request_id=reservation.request_id, turn_generation=reservation.turn_generation)
        async with asyncio.timeout(timeout):
            # Join the whole optional publication, including preflight outside
            # the adapter replacement guard. No old job may fence a later turn.
            bridge = self.coordinator._bridge
            if bridge is None or not await bridge.quiesce_for_normal_turn():
                return False
            if self._context_replacement_active:
                with latency_phase(trace, "response_preparation_latency", phase="normal_context_settled_wait", fields=fields):
                    await self._context_replacement_settled.wait()
            if not self._normal_turn_owned(reservation):
                return False
            # A publication joined above may have installed fresh context.
            if self.ready_for_response(reservation):
                return True
            with latency_phase(trace, "response_preparation_latency", phase="normal_context_scope", fields=fields):
                with diagnostics.observe(trace, reservation.request_id):
                    async with guard():
                        if not self._normal_turn_owned(reservation):
                            return False
            self._context_replacement_active = True
            self._context_replacement_settled.clear()
            try:
                with latency_phase(trace, "response_preparation_latency", phase="normal_context_delete", fields=fields):
                    await self.adapter.clear_optional_context_for_response(
                        current=lambda: self._normal_turn_owned(reservation))
            except BaseException:
                if self.adapter._response_clear_unresolved:
                    self._fail("response_context_clear_unknown")
                raise
            finally:
                self._context_replacement_active = False
                self._context_replacement_settled.set()
            proof = self.adapter.response_context_empty_proof
            if proof is None or not self._normal_turn_owned(reservation):
                return False
            bridge.note_normal_turn_clear()
            self._normal_turn_context = (reservation, proof)
            self._readiness_diagnostic("normal_turn_optional_context_unavailable",
                request_id=reservation.request_id)
            return self.ready_for_response(reservation)

    def _normal_output_context_current(self, state):
        grant = getattr(self, "_normal_turn_context", None)
        return (grant is not None and state is not None and not state.interrupted and not state.dismissed
                and grant[0].submitted is True and self._normal_turn_owned(grant[0], pending=False)
                and not self._context_replacement_active
                and self.adapter.response_context_empty_proof is grant[1]
                and state.metadata.get("jarvis_request_id") == grant[0].request_id
                and state.metadata.get("voice_route_offer_id") is None)

    def _readiness_diagnostic(self, stage, *, ready=None, request_id=None, receipt=None,
                              context_update=None, refresh_attempt=None, source_error=None,
                              guard_yielded=None):
        """Bounded observations, not a replacement state machine or recovery policy."""
        if stage not in {"ready_observed", "response_prewrite_rejected", "context_receipt",
                         "source_guard_prewrite_refresh", "source_guard_rejected",
                         "normal_turn_optional_context_unavailable"}:
            return
        try:
            status = self.coordinator.status
            phase = getattr(status.phase, "value", status.phase)
            phase = phase if phase in {item.value for item in CoordinationPhase} else "unrecognized_code"
            reason = (status.reason if status.reason in _COORDINATOR_REASONS
                      else None if status.reason is None else "unrecognized_code")
            if reason is None and self.context_refresh_pending:
                reason = ("context_replacement_inflight" if getattr(self, "_context_replacement_active", False)
                          else "context_expired_awaiting_snapshot")
            flags = dict(installed=self._installed is True, closing=self._closing is True,
                         closed=self._closed is True, channel_ready=self.channel.ready is True,
                         session_closed=self.session._closed.is_set(),
                         context_revoked=self.session._context_revoked is True,
                         context_refresh_pending=self.context_refresh_pending)
            state = (phase, reason, *flags.values())
            if stage == "ready_observed":
                if getattr(self, "_last_readiness_observation", None) == state:
                    return
                self._last_readiness_observation = state
            count = getattr(self, "_readiness_log_count", 0)
            if count > _READINESS_LOG_LIMIT:
                return
            self._readiness_log_count = count + 1
            def identity(value):
                try:
                    if isinstance(value, UUID):
                        return str(value)
                    if type(value) is str and len(value) <= 45:
                        return str(UUID(value))
                except ValueError:
                    pass
                return None
            route = getattr(getattr(self, "request", None), "route", None)
            generation = getattr(route, "route_generation", None)
            record = dict(stage=stage, route_id=identity(getattr(route, "route_id", None)),
                          route_generation=generation if type(generation) is int and generation > 0 else None,
                          audio_session_id=identity(self.session.context.audio_session_id),
                          monotonic_s=time.monotonic())
            if count == _READINESS_LOG_LIMIT:
                record.update(stage="truncated", record_limit=_READINESS_LOG_LIMIT,
                              absence_after_limit_is_inconclusive=True)
            else:
                record.update(flags, phase=phase, reason=reason)
                if type(ready) is bool:
                    record["ready"] = ready
                if request_id is not None:
                    record["request_id"] = identity(request_id)
                if isinstance(receipt, ContextReceipt):
                    record["context_receipt"] = context_receipt_diagnostic(receipt)
                if isinstance(context_update, VoiceContextUpdate):
                    record.update(update_id=str(context_update.update_id),
                                  cursor_revision=context_update.cursor.revision)
                if type(refresh_attempt) is int and 1 <= refresh_attempt <= 2:
                    record["refresh_attempt"] = refresh_attempt
                if type(guard_yielded) is bool:
                    record["guard_yielded"] = guard_yielded
                if source_error is not None:
                    error_type = type(source_error).__name__
                    code = getattr(source_error, "code", None)
                    record["error_type"] = (error_type if error_type in {
                        "LiveContextError", "SourceError", "TimeoutError"} else "unrecognized_error")
                    record["error_code"] = (code if type(code) is str and code in _SOURCE_GUARD_CODES
                                            else "guard_timeout" if isinstance(source_error, TimeoutError)
                                            else "unrecognized_code")
                    if isinstance(source_error, AttentionError):
                        record["error_type"] = "AttentionError"
                        record["error_code"] = (code if type(code) is str and code in _ATTENTION_GUARD_CODES
                                                else "unrecognized_code")
            _READINESS_LOGGER.warning("voice_route_readiness_diagnostic %s",
                                      json.dumps(record, sort_keys=True, separators=(",", ":")))
        except Exception:
            # Logging failure cannot authorize, reject, retire or recover a route.
            pass

    def _output_ready(self, ending: bool, response_id=None) -> bool:
        if not ending:
            if self.ready:
                return True
            state = getattr(self.session, "_active_responses", {}).get(response_id)
            # Only already admitted, pinned output may outlive current-view
            # freshness. Canonical authorization is checked at its actual write.
            return (self.context_publication_ready and state is not None
                    and not state.interrupted and not state.dismissed
                    and (getattr(state, "source_snapshot", None) is not None
                         or self._normal_output_context_current(state)))
        # An end-only control carries no source content and cannot authorize
        # new audio. Keep it available when publication has degraded, while
        # retaining the exact live session/channel/route ownership boundary.
        status = self.coordinator.status
        return (self._installed and not self._closing and not self._closed
                and self.channel.ready and not self.session._closed.is_set()
                and status.identity == self.request.route
                and status.phase in {CoordinationPhase.ACTIVE, CoordinationPhase.DEGRADED,
                                     CoordinationPhase.RETIRING})

    def install(self) -> None:
        asyncio.get_running_loop()
        if (self._installed or getattr(self.session, "_voice_route_runtime", None) is not None
                or self.session._closed.is_set() or self.session._realtime_context_ready.is_set()
                or getattr(self.session, "_attention_delivery", None) is not None):
            raise VoiceRouteRuntimeError("runtime_installation_not_admitted")
        self._installed = True
        self.session._voice_route_runtime = self
        # Instance-local wrappers preserve the most-derived direct-session
        # handler and the same writer locks. They do not create transport owners.
        self.session._handle_client_event = self._client_event
        self.session._handle_openai_event = self._provider_event
        self.session.openai.send_event = self._provider_send
        self.session._app_sender.send_text = self._app_send_text
        self.session.try_attention = self._attention_unavailable
        self._pump_task = self._spawn(self._pump_events())
        self._bootstrap_task = self._spawn(self._bootstrap())

    async def _bootstrap(self) -> None:
        try:
            await self.session._realtime_context_ready.wait()
            if self._closing or self.session._closed.is_set():
                return
            await self.channel.select(self._client_capabilities)
            if not await self.channel.wait_ready():
                raise VoiceRouteRuntimeError("client_not_ready")
            await self.coordinator.start(self.adapter, self.request)
        except asyncio.CancelledError:
            raise
        except Exception:
            self._fail("voice_route_start_failed")

    async def _wait_ready(self) -> bool:
        await self.session._realtime_context_ready.wait()
        return not self._closing and not self.session._closed.is_set() and await self.channel.wait_ready()

    async def _authorize(self, request: RouteStart) -> bool:
        return (request == self.request and not self._closing
                and await self._authorize_start(request) is True)

    @asynccontextmanager
    async def _source_guard(self, payload: GuardedInput):
        roots = self._payload_roots(payload)
        if not isinstance(roots, tuple) or any(not isinstance(root, UUID) for root in roots):
            raise VoiceRouteRuntimeError("invalid_committed_source_roots")
        # Validate candidates without enrolling them into provider history.
        # Only the actual adapter handoff calls _source_handoff below.
        yielded = False

        @asynccontextmanager
        async def current_sources():
            nonlocal yielded
            refreshes = 0
            while True:
                try:
                    async with self.session._source_context.context_guard(extra_sources=roots):
                        # No provider factory has run before this point. Once
                        # yielded, even a same-code error may follow a write.
                        yielded = True
                        yield
                    return
                except LiveContextError as error:
                    if yielded or error.code != "live_context_changed" or refreshes >= 2:
                        raise
                    refreshes += 1
                    # Refresh the union of roots and lock all of them again,
                    # under the SAME borrowed canonical owner transaction. This
                    # neither repeats deletion/creation nor extends deadlines.
                    self._readiness_diagnostic("source_guard_prewrite_refresh",
                        context_update=payload, refresh_attempt=refreshes, guard_yielded=False)

        try:
            async with self._atomic_guard(payload) as connection:
                if connection is None:
                    async with current_sources():
                        yield
                else:
                    async with self.session._source_context.use_transaction(connection):
                        async with current_sources():
                            yield
        except Exception as error:
            self._readiness_diagnostic("source_guard_rejected", context_update=payload,
                                       source_error=error, guard_yielded=yielded)
            raise

    def _source_handoff(self, payload: GuardedInput):
        roots = self._payload_roots(payload)
        if not isinstance(roots, tuple) or any(not isinstance(root, UUID) for root in roots):
            raise VoiceRouteRuntimeError("invalid_committed_source_roots")
        # Conservative possible exposure, not successful delivery. Keep these
        # roots after unknown writes or later item deletion acknowledgments.
        self.session._source_context.observe(roots)

    @asynccontextmanager
    async def _context_replacement_guard(self, update: VoiceContextUpdate):
        def eligible():
            return (isinstance(update, VoiceContextUpdate) and update.route == self.request.route
                    and not self._answer_owns_context() and not self._closing and not self._closed
                    and not self.session._closed.is_set() and not self.session._context_revoked)

        if not eligible():
            yield False
            return
        settled = getattr(self, "_context_replacement_settled", None)
        if settled is None:
            settled = self._context_replacement_settled = asyncio.Event()
        self._context_replacement_active = True
        settled.clear()
        try:
            # Removing this route's owned item needs canonical owner/session/
            # selected-scope authority, not a second validation of NEW content.
            # The existing callback releases its transaction before returning.
            # Preflight and the actual create's full source guard are unchanged;
            # previously exposed roots remain enrolled even after deletion.
            authorize = (self._authorize_start if update == getattr(self.request, "initial_context", None)
                         else getattr(self, "_authorize_context_replacement", self._authorize_start))
            authorized = await authorize(self.request)
            yield authorized is True and eligible()
        finally:
            self._context_replacement_active = False
            settled.set()

    def offer_context(self, update: VoiceContextUpdate) -> ContextOfferResult:
        if not self.context_publication_ready:
            return ContextOfferResult(OfferDisposition.REJECTED, "runtime_not_ready")
        if self._answer_owns_context():
            return ContextOfferResult(OfferDisposition.REJECTED, "runtime_response_active")
        return self.coordinator.offer_context(update)

    @property
    def context_publication_status(self):
        """Read the selected coordinator's evidence, never a second retry owner."""
        bridge = self.coordinator._bridge
        if not self.context_publication_ready or bridge is None:
            return None
        status = bridge.status
        return status if status.identity == self.request.route else None

    async def retire_context(self, command: SourceRetirementCommand):
        try:
            return await self.coordinator.retire_context(command)
        except Exception:
            self._fail("voice_route_retirement_failed")
            raise

    async def offer_notification(self, offer: NotificationOffer) -> DeliveryEvidence:
        if not self.ready or offer.route != self.request.route:
            raise VoiceRouteRuntimeError("notification_route_not_ready")
        previous = self._offers.get(offer.offer_id)
        if previous is not None and previous != offer:
            raise VoiceRouteRuntimeError("notification_identity_reused")
        if previous is None and len(self._offers) >= 64:
            raise VoiceRouteRuntimeError("notification_binding_capacity")
        self._offers[offer.offer_id] = offer
        return await self.adapter.offer_notification(offer)

    async def _attention_unavailable(self, *args, **kwargs):
        # Keep maintenance/reconciliation running, but never fall through to
        # ExpertVoiceDelivery while native mode lacks its ledger/sink binding.
        return None

    async def _admit_response(self, offer, request_id, guard, current) -> bool:
        session = self.session
        async with session._response_admission_lock:
            if (not self.ready or not current() or session._user_audio_pending
                    or session._unacknowledged_response_requests
                    or (getattr(session, "_attention_delivery", None) is not None
                        and not getattr(session._attention_delivery, "accepts_offer", lambda _: False)(offer))
                    or getattr(session, "_direct_output_pending", None)
                    or any(not item.interrupted and item.completed_response is None
                           for item in session._active_responses.values())):
                return False
            generation = session._turn_generation
        if request_id in self._dispatches or len(self._dispatches) >= 64:
            return False
        dispatch = _Dispatch(offer, request_id, guard,
            lambda: self.ready and current() and session._turn_generation == generation,
            inflight_current=lambda: current() and session._turn_generation == generation)
        self._dispatches[request_id] = dispatch
        token = self._dispatch_context.set(dispatch)
        try:
            # The existing owner reserves, prepares source snapshots, submits
            # and correlates its own response. No parallel reservation is made.
            accepted = await session.request_route_notification(offer, turn_generation=generation)
            return accepted is True and dispatch.written
        finally:
            self._dispatch_context.reset(token)

    async def _provider_send(self, event, *, write_guard=None):
        dispatch = self._dispatch_context.get()
        holder = {}
        touched = False

        def owned_pending(payload):
            request_id = payload.get("event_id")
            pending = self.session._unacknowledged_response_requests.get(request_id)
            if pending is None:
                pending = getattr(self.session, "_pending_response", None)
            metadata = (payload.get("response") or {}).get("metadata") or {}
            return (pending if pending is not None and pending.request_id == request_id
                    and metadata.get("jarvis_request_id") == request_id else None)

        def reject_response(payload, reason):
            request_id = payload.get("event_id")
            self._readiness_diagnostic("response_prewrite_rejected", request_id=request_id)
            pending = owned_pending(payload)
            metadata = (payload.get("response") or {}).get("metadata") or {}
            if (pending is not None and pending.submitted is True
                    and pending.request_id == request_id
                    and metadata.get("jarvis_request_id") == request_id):
                # No socket send has begun: the enclosing real writer is still
                # awaiting this factory. Distinguish this from send/guard-exit
                # exceptions, whose remote outcome must remain unknown.
                raise ResponsePrewriteRejected(request_id, reason)
            return None

        @asynccontextmanager
        async def combined_guard():
            async with AsyncExitStack() as stack:
                if dispatch is not None and not dispatch.used:
                    # Own canonical owner/source locks before the original
                    # response guard; it borrows this same transaction.
                    await stack.enter_async_context(dispatch.guard())
                elif self.normal_turn_context_unavailable(getattr(self.session, "_pending_response", None)):
                    connection = await stack.enter_async_context(self._normal_turn_scope_guard())
                    if connection is not None:
                        await stack.enter_async_context(self.session._source_context.use_transaction(connection))
                if write_guard is not None:
                    await stack.enter_async_context(write_guard())
                holder["stack"] = stack
                yield

        async def prepared():
            nonlocal touched
            payload = event() if callable(event) else event
            if inspect.isawaitable(payload):
                payload = await payload
            if not isinstance(payload, dict) or payload.get("type") != "response.create":
                return payload
            pending = owned_pending(payload)
            if not (self.ready if dispatch is not None else self.ready_for_response(pending)):
                return reject_response(payload, "route_not_ready")
            if dispatch is None or dispatch.used:
                return payload
            if not dispatch.current():
                return reject_response(payload, "notification_not_current")
            if not dispatch.current():
                return reject_response(payload, "notification_not_current")
            response = dict(payload.get("response") or {})
            metadata = dict(response.get("metadata") or {})
            metadata.update(voice_route_id=str(self.request.route.route_id),
                voice_route_generation=str(self.request.route.route_generation),
                voice_route_offer_id=str(dispatch.offer.offer_id), voice_route_request_id=dispatch.request_id)
            actual_request_id = payload.get("event_id")
            pending = self.session._unacknowledged_response_requests.get(actual_request_id)
            if (not isinstance(actual_request_id, str) or pending is None
                    or metadata.get("jarvis_request_id") != actual_request_id
                    or dispatch.response_request_id not in (None, actual_request_id)):
                raise VoiceRouteRuntimeError("notification_request_binding_missing")
            dispatch.response_request_id = actual_request_id
            # Preserve the exact provider-write binding for cancellation before
            # response.created. This is routing metadata, not new source authority.
            pending.metadata.update({key: metadata[key] for key in (
                "voice_route_id", "voice_route_generation", "voice_route_offer_id", "voice_route_request_id")})
            response["metadata"] = metadata
            touched = dispatch.used = True
            return {**payload, "response": response}

        sent = await self._owner_provider_send(prepared, write_guard=combined_guard)
        if touched:
            dispatch.written = sent is True
        # The original writer and its canonical guards have exited. A rejected
        # factory can return False without an exception/error event for the app.
        await self._report_unavailable_status()
        return sent

    async def _report_unavailable_status(self) -> None:
        """One status projection, not another readiness/recovery/lifecycle owner."""
        status = self.coordinator.status
        if (self._unavailability_reported or not self._installed or self._closing or self._closed
                or self.session._closed.is_set() or not self.channel.ready
                or getattr(self.session, "_voice_route_runtime", None) is not self
                or status.identity != self.request.route
                or status.phase != CoordinationPhase.DEGRADED or not status.reconnect_required
                or self._context_replacement_active):
            return
        # ACTIVE awaiting-fresh-context is recoverable and cannot enter here.
        # Keep the original route state untouched, including after a failed
        # bounded channel write. No retry, close, cancellation or recovery.
        self._unavailability_reported = True
        try:
            await self.channel.send_route_event(RouteError(
                route=self.request.route, observed_at=self._clock(),
                code="voice_route_unavailable", recoverable=False,
                message="Voice is unavailable for this connection. Stop and start a new voice session.",
            ))
        except asyncio.CancelledError:
            raise
        except Exception:
            pass  # Status transport failure cannot change the owner's outcome.

    def _inflight_notification_current(self, dispatch):
        """Written request and unchanged transport/epochs, not source authority.

        Only ordinary ambient freshness may differ. Destructive replacement,
        unresolved publication, source/speech epoch changes and new turns remain
        fences. Admission and the response writer still require self.ready.
        """
        session, route, channel = self.session, self.request.route, self.channel
        status = self.coordinator.status
        delivery = getattr(session, "_attention_delivery", None)
        return (dispatch is not None and dispatch.written is True
                and isinstance(dispatch.response_request_id, str) and bool(dispatch.response_request_id)
                and callable(dispatch.inflight_current) and dispatch.inflight_current() is True
                and self._dispatches.get(dispatch.request_id) is dispatch
                and dispatch.offer.route == route and channel.identity == route
                and status.identity == route and status.phase == CoordinationPhase.ACTIVE
                and self.context_publication_ready and not self._context_replacement_active
                and getattr(self.adapter, "context_reconciliation_pending", True) is False
                and session._voice_route_runtime is self and session.openai is self._input_provider
                and session.context.owner_id == route.scope.owner_id
                and session.context.jarvis_session_id == route.scope.session_id
                and session.context.audio_session_id == self._input_audio_session_id
                and channel.audio_session_id == self._input_audio_session_id
                and delivery is not None and delivery.runtime is self
                and delivery.owns_offer(dispatch.offer))

    def _bound_notification_current(self, dispatch, response_id):
        state = getattr(self.session, "_active_responses", {}).get(response_id)
        if (not self._inflight_notification_current(dispatch)
                or self._response_offers.get(response_id) is not dispatch
                or state is None or state.interrupted or state.dismissed
                or getattr(state, "source_snapshot", None) is None):
            return False
        metadata = getattr(state, "metadata", {})
        return (metadata.get("jarvis_request_id") == dispatch.response_request_id
                and metadata.get("voice_route_request_id") == dispatch.request_id
                and metadata.get("voice_route_offer_id") == str(dispatch.offer.offer_id)
                and metadata.get("voice_route_id") == str(self.request.route.route_id)
                and metadata.get("voice_route_generation") == str(self.request.route.route_generation))

    def notification_output_current(self, offer):
        """One app-writer task's accepted response, never a used-ticket shortcut."""
        context = getattr(self, "_notification_output_context", None)
        owned = context.get() if context is not None else None
        return (owned is not None and owned[2] is asyncio.current_task()
                and owned[0].offer == offer
                and self._bound_notification_current(owned[0], owned[1]))

    def _response_dispatch(self, payload):
        response = payload.get("response")
        if not isinstance(response, dict) or not isinstance(response.get("id"), str):
            return None
        metadata = response.get("metadata")
        if not isinstance(metadata, dict):
            return None
        dispatch = self._dispatches.get(metadata.get("voice_route_request_id"))
        if (not self._inflight_notification_current(dispatch)
                or metadata.get("jarvis_request_id") != dispatch.response_request_id
                or metadata.get("voice_route_id") != str(self.request.route.route_id)
                or metadata.get("voice_route_generation") != str(self.request.route.route_generation)
                or metadata.get("voice_route_offer_id") != str(dispatch.offer.offer_id)):
            return None
        return dispatch

    async def _provider_event(self, payload):
        # Item lifecycle ACKs have no admitted response. Observe before the
        # owner handler can await app output, while preserving all error routing.
        self.adapter.observe_item_event(self.request.route, payload)
        response = payload.get("response") if isinstance(payload.get("response"), dict) else {}
        response_id = payload.get("response_id") or response.get("id")
        before = self.session._active_responses.get(response_id)
        dispatch = self._response_dispatch(payload) if payload.get("type") == "response.created" else None
        if dispatch is not None:
            self._response_offers[response_id] = dispatch
        await self._owner_provider_handler(payload)
        state = self.session._active_responses.get(response_id) or before
        admitted = state is not None and not state.interrupted and not state.dismissed
        if dispatch is not None:
            if admitted and self._bound_notification_current(dispatch, response_id):
                self.adapter.bind_response(self.request.route, dispatch.offer.offer_id, dispatch.request_id, response_id)
            else:
                self._response_offers.pop(response_id, None)
        self.adapter.observe_provider_event(self.request.route, payload, admitted=admitted)

    async def _send_input_ready(self, payload, *, should_send=None, write_guard=None):
        """A cleared-input receipt is not an output permit or a context receipt."""
        route, channel = self.request.route, self.channel
        generation = self.session._turn_generation

        def current():
            session, status = self.session, self.coordinator.status
            owned = (self._installed and not self._closing and not self._closed
                and session._voice_route_runtime is self and self.channel is channel
                and channel.ready and channel.identity == route and self.request.route == route
                and status.identity == route and status.phase == CoordinationPhase.ACTIVE
                and session.openai is self._input_provider
                and session.context.owner_id == route.scope.owner_id
                and session.context.jarvis_session_id == route.scope.session_id
                and session.context.audio_session_id == self._input_audio_session_id == channel.audio_session_id
                and session._realtime_context_ready.is_set()
                and not session._closed.is_set() and not session._context_revoked
                and session._turn_generation == generation)
            if payload["type"] == "audio.input.ready":
                segment = payload.get("segment_id")
                owned = (owned and set(payload) == {"type", "segment_id"}
                    and isinstance(segment, str) and 0 < len(segment) <= 256
                    and session.openai.capture_mode == "manual" and session._user_audio_pending
                    and getattr(session, "_direct_manual_segment_id", None) == segment
                    and getattr(session, "_direct_manual_clear_generation", None) == generation)
            else:
                owned = (owned and set(payload) == {"type", "capture_mode"}
                    and payload.get("capture_mode") in {"auto", "manual"}
                    and session.openai.capture_mode == payload["capture_mode"])
            if not owned:
                self._native_control_diagnostic(payload, "runtime_write_guard_rejected",
                    reason="input_control_not_current", send_result=False)
                return False
            if should_send is not None and not should_send():
                self._native_control_diagnostic(payload, "runtime_write_guard_rejected",
                    reason="owner_send_guard", send_result=False)
                return False
            return True

        if not current():
            return False
        envelope = channel.media_envelope(payload)
        try:
            sent = await self._owner_send_text(json.dumps(envelope, ensure_ascii=True, separators=(",", ":")),
                should_send=current, write_guard=write_guard)
        except BaseException as error:
            self._native_control_diagnostic(payload, "runtime_send_raised",
                reason="write_cancelled" if isinstance(error, asyncio.CancelledError) else "write_exception")
            raise
        self._native_control_diagnostic(payload, "runtime_send_result", send_result=sent)
        return sent

    async def _stop_unready_notification(self, dispatch, response_id):
        """Retire only this owned announcement, after app/source locks unwind.

        This is not a new output authorization or a retry. Uncertain provider
        exposure remains the existing delivery's unknown settlement; the
        existing fence path closes the session if its fence is unconfirmed.
        """
        session, route, channel = self.session, self.request.route, self.channel
        status = self.coordinator.status
        delivery = getattr(session, "_attention_delivery", None)
        state = getattr(session, "_active_responses", {}).get(response_id)
        if (dispatch is None or not dispatch.written
                or self._response_offers.get(response_id) is not dispatch
                or dispatch.offer.route != route or channel.identity != route
                or status.identity != route or status.phase != CoordinationPhase.ACTIVE
                or not self._installed or self._closing or self._closed or not channel.ready
                or session._closed.is_set() or session._context_revoked
                or session._voice_route_runtime is not self or session.openai is not self._input_provider
                or session.context.owner_id != route.scope.owner_id
                or session.context.jarvis_session_id != route.scope.session_id
                or session.context.audio_session_id != self._input_audio_session_id
                or channel.audio_session_id != self._input_audio_session_id
                or state is None or state.interrupted or state.dismissed
                or getattr(state, "source_snapshot", None) is None
                or delivery is None or delivery.runtime is not self
                or not delivery.owns_offer(dispatch.offer)):
            return False
        await delivery.stop("native_context_not_ready")
        return True

    async def _app_send_text(self, text, *, should_send=None, write_guard=None):
        try:
            payload = json.loads(text)
        except (TypeError, ValueError):
            return await self._owner_send_text(text, should_send=should_send, write_guard=write_guard)
        if not isinstance(payload, dict) or payload.get("type") not in _MEDIA:
            return await self._owner_send_text(text, should_send=should_send, write_guard=write_guard)
        self._native_control_diagnostic(payload, "runtime_attempt")
        if payload["type"] in _INPUT_READY:
            return await self._send_input_ready(payload, should_send=should_send, write_guard=write_guard)
        # Native mode cannot receive a second PCM/TTS speaker, even if an old
        # producer retained a reference to the existing serialized writer.
        if (payload["type"] in {"audio.output.delta", "audio.output.done"}
                or payload.get("audio_origin") in {"expert_tts", "notification_tts"}):
            return False
        ending = (payload["type"] in _OUTPUT_END
            and isinstance(payload.get("response_id"), str) and bool(payload["response_id"])
            and set(payload) <= {"type", "response_id", "protocol_version", "audio_session_id", "route"})
        if not self._output_ready(ending, payload.get("response_id")):
            self._native_control_diagnostic(payload, "runtime_rejected", reason="runtime_not_ready", send_result=False)
            return False
        dispatch = self._response_offers.get(payload.get("response_id"))
        state = getattr(self.session, "_active_responses", {}).get(payload.get("response_id"))
        if (not ending and dispatch is None
                and getattr(state, "metadata", {}).get("voice_route_offer_id") is not None):
            # A foreign/unconfirmed notification must not fall through to the
            # ordinary-answer path merely because correlation was rejected.
            self._native_control_diagnostic(payload, "runtime_rejected",
                reason="notification_not_current", send_result=False)
            return False
        envelope = self.channel.media_envelope(payload)
        readiness_denied = False
        guard_yielded = False

        def current():
            nonlocal readiness_denied
            if not self._output_ready(ending, payload.get("response_id")):
                self._native_control_diagnostic(payload, "runtime_write_guard_rejected",
                    reason="runtime_not_ready", send_result=False)
                return False
            if should_send is not None and not should_send():
                self._native_control_diagnostic(payload, "runtime_write_guard_rejected",
                    reason="owner_send_guard", send_result=False)
                return False
            if dispatch is not None and not ending and not self._bound_notification_current(
                    dispatch, payload.get("response_id")):
                readiness_denied = not self.ready
                self._native_control_diagnostic(payload, "runtime_write_guard_rejected",
                    reason="notification_not_current", send_result=False)
                return False
            return True

        @asynccontextmanager
        async def output_guard():
            nonlocal guard_yielded
            context = getattr(self, "_notification_output_context", None)
            token = (context.set((dispatch, payload.get("response_id"), asyncio.current_task()))
                     if context is not None and dispatch is not None and not ending else None)
            try:
                async with AsyncExitStack() as stack:
                    # Owner/notification authority precedes copied-source locks.
                    # _source_guard lends its transaction to the pin guard in this
                    # exact task; never take a second owner lock after pinning roots.
                    if dispatch is not None and not ending:
                        await stack.enter_async_context(self._source_guard(dispatch.offer))
                    state = getattr(self.session, "_active_responses", {}).get(payload.get("response_id"))
                    if dispatch is None and not ending and self._normal_output_context_current(state):
                        connection = await stack.enter_async_context(self._normal_turn_scope_guard())
                        if connection is not None:
                            await stack.enter_async_context(self.session._source_context.use_transaction(connection))
                    snapshot = getattr(state, "source_snapshot", None)
                    if not ending and snapshot is not None:
                        await stack.enter_async_context(self.session._source_context.output_guard(snapshot))
                    marker = await stack.enter_async_context(write_guard()) if write_guard is not None else None
                    guard_yielded = True
                    yield marker  # Preserve the existing writer's actual-forwarding marker.
            finally:
                if token is not None:
                    context.reset(token)

        try:
            sent = await self._owner_send_text(json.dumps(envelope, ensure_ascii=True, separators=(",", ":")),
                                               should_send=current, write_guard=output_guard)
        except BaseException as error:
            if (not ending and not guard_yielded and isinstance(error, AttentionError)
                    and error.code == "attention_route_not_ready"
                    and await self._stop_unready_notification(dispatch, payload.get("response_id"))):
                self._native_control_diagnostic(payload, "runtime_send_result", send_result=False)
                return False
            state = getattr(self.session, "_active_responses", {}).get(payload.get("response_id"))
            if (not ending and getattr(state, "source_snapshot", None) is not None
                    and isinstance(error, (LiveContextError, TimeoutError))):
                self.session._begin_context_rebuild()
            self._native_control_diagnostic(payload, "runtime_send_raised",
                reason="write_cancelled" if isinstance(error, asyncio.CancelledError) else "write_exception")
            raise
        if not sent and readiness_denied:
            await self._stop_unready_notification(dispatch, payload.get("response_id"))
        self._native_control_diagnostic(payload, "runtime_send_result", send_result=sent)
        return sent

    def _native_control_diagnostic(self, payload, stage, *, reason=None, send_result=None):
        diagnostic = getattr(self.session, "_native_control_diagnostic", None)
        if callable(diagnostic):
            diagnostic(stage, payload.get("type"), payload.get("response_id"),
                reason=reason, send_result=send_result, runtime=self,
                **({"segment_id": payload.get("segment_id"), "turn_generation": self.session._turn_generation}
                   if payload.get("type") in _INPUT_READY else {}))

    async def _client_event(self, event):
        kind = event.get("type")
        if kind == "voice_route.ready":
            self.channel.accept_ready(event)
            return
        if kind == "voice_route.event":
            self.channel.accept_receipt(event)
            return
        if kind == "voice_route.playback":
            evidence = self.channel.parse_playback(event)
            if evidence is not None:
                self.adapter.accept_client_playback(evidence, audio_session_id=self.session.context.audio_session_id)
            return
        if kind == "voice_route.interrupt":
            try:
                command = self.channel.parse_interrupt(event)
            except (ValueError, RuntimeError, TypeError):
                return
            self._queue_interrupt(command)
            return
        if kind == "response.cancel":
            if self._interrupt_owner_current():
                self._queue_interrupt(InterruptSpeechCommand(route=self.request.route, command_id=uuid4(),
                    requested_at=self._clock(), reason="client_response_cancel"))
            return
        if kind == "session.close":
            self._spawn(self.before_session_close("client_closed"))
            return
        if isinstance(kind, str) and kind.startswith("voice_route."):
            return  # Unknown new-mode commands never fall through to a legacy control.
        await self._owner_client_handler(event)

    def _interrupt_owner_current(self):
        # Stopping owned audio never depends on optional briefing freshness.
        return (self._installed and not self._closed and not self._closing
                and self.session._voice_route_runtime is self
                and self.session.openai is self._input_provider and self.channel.ready
                and self.channel.identity == self.request.route
                and self.channel.audio_session_id == self._input_audio_session_id
                and self.session.context.audio_session_id == self._input_audio_session_id
                and self.coordinator.status.identity == self.request.route
                and self.coordinator.status.phase in {CoordinationPhase.ACTIVE,
                    CoordinationPhase.DEGRADED, CoordinationPhase.RETIRING})

    def _queue_interrupt(self, command):
        previous = self._interrupts.get(command.command_id)
        if previous is not None:
            if previous != command:
                self._fail("interrupt_identity_reused")
            return
        if (command.route != self.request.route or not self._interrupt_owner_current()
                or len(self._interrupts) >= 128):
            self._fail("interrupt_not_admitted")
            return
        self._interrupts[command.command_id] = command
        self._spawn(self._interrupt(command))

    async def _interrupt(self, command):
        try:
            await self.coordinator.interrupt_speech(command)
        except Exception:
            self._fail("voice_route_interruption_unconfirmed")

    async def _cancel_response(self):
        # Preserve the existing generation advance, turn settlement and lock.
        await self._owner_client_handler({"type": "response.cancel"})

    async def _close_control(self):
        self.session._mark_closing("voice_route_closed")
        await self.session.openai.close()

    async def _fence_output(self, command):
        return await self.channel.request_fence(command)

    async def fence_notification_output(self, command, *, delivery):
        if (command.route != self.request.route or self.session._attention_delivery is not delivery
                or not delivery.revoked):
            raise VoiceRouteRuntimeError("notification_fence_not_owned")
        offer = getattr(delivery, "offer", None)
        for dispatch in tuple(self._dispatches.values()):
            if offer is not None and dispatch.offer == offer and dispatch.response_request_id is not None:
                await self.session.cancel_route_notification(
                    request_id=dispatch.response_request_id, offer_id=str(offer.offer_id))
        # The shared session already owns cancellation/turn generation. Calling
        # coordinator.interrupt_speech here would spawn another generic cancel
        # that can arrive after a newer user turn has been admitted.
        return await self.adapter.fence_output_only(command)

    async def _pump_events(self):
        try:
            async for event in self.adapter.events():
                if isinstance(event, ContextReceipt):
                    self._readiness_diagnostic("context_receipt", receipt=event)
                if self._on_event is not None:
                    result = self._on_event(event)
                    if inspect.isawaitable(result):
                        if inspect.iscoroutine(result):
                            result.close()
                        raise VoiceRouteRuntimeError("event_sink_must_enqueue_synchronously")
                if (isinstance(event, DeliveryEvidence)
                        and event.state == DeliveryEvidenceState.OUTPUT_STARTED and event.output is not None):
                    offer = self._offers.get(event.offer_id)
                    if offer is None or offer.attempt != event.attempt:
                        raise VoiceRouteRuntimeError("output_has_no_prepared_offer")
                    await self.channel.bind_output(offer, event.output)
                await self.channel.send_route_event(event)
        except asyncio.CancelledError:
            raise
        except Exception:
            self._fail("voice_route_event_intake_failed")

    def _fail(self, reason):
        self.failure = reason
        if not self._closing:
            self._spawn(self.before_session_close(reason))

    async def before_session_close(self, reason: str) -> bool:
        if self._shutdown_task is None:
            self._closing = True
            self._shutdown_task = asyncio.create_task(self._shutdown(reason))
        return await asyncio.shield(self._shutdown_task)

    async def _shutdown(self, reason):
        if self._bootstrap_task is not None and not self._bootstrap_task.done():
            self._bootstrap_task.cancel()
            _, pending = await asyncio.wait({self._bootstrap_task}, timeout=self._control_timeout)
            if pending:
                self.failure = "bootstrap_not_joined"
                return False
        try:
            receipt = await self.coordinator.close()
            if receipt is None and self.channel.ready:
                command = CloseRouteCommand(route=self.request.route, command_id=uuid4(),
                    requested_at=self._clock(), reason="session_shutdown")
                receipt = await self.channel.request_fence(command)
                await self._close_control()
            self.shutdown_confirmed = (isinstance(receipt, CommandReceipt)
                and receipt.status == "completed" and receipt.output_fenced is True)
        except Exception:
            self.failure = "voice_route_shutdown_unconfirmed"
        if self._pump_task is not None and not self._pump_task.done():
            await asyncio.wait({self._pump_task}, timeout=self._control_timeout)
        return self.shutdown_confirmed

    async def after_session_close(self) -> bool:
        if self._finish_task is None:
            self._finish_task = asyncio.create_task(self._finish())
        return await asyncio.shield(self._finish_task)

    async def _finish(self):
        self._closing = self._closed = True
        for task in tuple(self._tasks):
            if not task.done():
                task.cancel()
        channel_joined = await self.channel.aclose()
        pending = {task for task in self._tasks if not task.done()}
        if pending:
            _, pending = await asyncio.wait(pending, timeout=self._control_timeout)
        if pending or not channel_joined:
            self.failure = "voice_route_owned_tasks_unjoined"
            return False
        return True

    def _spawn(self, operation):
        task = asyncio.create_task(operation)
        self._tasks.add(task)
        def finished(done):
            self._tasks.discard(done)
            if not done.cancelled():
                done.exception()
        task.add_done_callback(finished)
        return task
