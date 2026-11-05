"""Opt-in route ownership and fencing; no provider, ledger or task implementation."""

import asyncio
import math
from dataclasses import dataclass
from enum import Enum
from uuid import UUID, uuid4

from app.live_state.models import SessionScope
from app.voice_routes.bridge import (
    Clock, ContextOfferResult, ContextValidity, OfferDisposition, VoiceContextBridge, utc_now,
)
from app.voice_routes.contracts import (
    CloseRouteCommand, CommandReceipt, InterruptSpeechCommand, RouteIdentity,
    RouteOperation, RouteStart, SourceRetirementCommand, SourceRetirementReceipt,
    Support, UnsupportedOperation, VoiceContextUpdate,
)
from app.voice_routes.protocol import VoiceRoute


class CoordinationPhase(str, Enum):
    EMPTY = "empty"
    STARTING = "starting"
    ACTIVE = "active"
    RETIRING = "retiring"
    CLOSING = "closing"
    DEGRADED = "degraded"
    CLOSED = "closed"


@dataclass(frozen=True)
class CoordinatorStatus:
    phase: CoordinationPhase
    identity: RouteIdentity | None
    reason: str | None
    reconnect_required: bool


class RouteCoordinationError(RuntimeError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class VoiceRouteCoordinator:
    """Single-event-loop owner with serialized lifecycle and independent interrupt.

Caller authorization/source guards remain mandatory, including route.start's
initial context and history. Adapter methods must safely support concurrent
control/publication. This owner requires completed start/stop receipts; it does
not consume asynchronous ready events or infer success from elapsed time.

Timeouts cancel and retain owned tasks until they finish. A successful close or
replacement joins them; uncooperative tasks leave degraded ownership, never a
second speaker. No automatic replacement, retry, task cancellation or ledger
settlement is performed.
"""

    def __init__(
        self, scope: SessionScope, *, validity: ContextValidity,
        control_timeout: float = 5.0, publish_timeout: float = 2.0,
        clock: Clock = utc_now,
        preflight_expiry_refresh: bool = False,
    ) -> None:
        if any(not math.isfinite(value) or value <= 0 for value in (control_timeout, publish_timeout)):
            raise ValueError("Timeouts must be finite and positive.")
        self._scope = scope
        self._validity = validity
        self._control_timeout = control_timeout
        self._publish_timeout = publish_timeout
        self._preflight_expiry_refresh = preflight_expiry_refresh
        self._clock = clock
        self._lifecycle = asyncio.Lock()
        self._route: VoiceRoute | None = None
        self._identity: RouteIdentity | None = None
        self._bridge: VoiceContextBridge | None = None
        self._phase = CoordinationPhase.EMPTY
        self._reason: str | None = None
        self._reconnect_required = False
        self._highest_generation = 0
        self._seen_route_ids: set[UUID] = set()
        self._control_epoch = 0
        self._calls: set[asyncio.Task] = set()

    @property
    def status(self) -> CoordinatorStatus:
        return CoordinatorStatus(self._phase, self._identity, self._reason, self._reconnect_required)

    def offer_context(self, update: VoiceContextUpdate) -> ContextOfferResult:
        if self._phase != CoordinationPhase.ACTIVE or self._bridge is None:
            return ContextOfferResult(OfferDisposition.REJECTED, "route_not_active")
        return self._bridge.offer(update)

    async def start(self, route: VoiceRoute, request: RouteStart) -> RouteIdentity:
        async with self._lifecycle:
            if self._phase == CoordinationPhase.CLOSED or self._route is not None:
                raise RouteCoordinationError("coordinator_not_empty")
            self._check_candidate(route, request)
            return await self._start_owned(route, request)

    async def replace(self, route: VoiceRoute, request: RouteStart) -> RouteIdentity:
        async with self._lifecycle:
            if self._route is None or self._phase == CoordinationPhase.CLOSED:
                raise RouteCoordinationError("no_route_to_replace")
            self._check_candidate(route, request)
            await self._stop_owned("route_replacement")
            return await self._start_owned(route, request)

    async def close(self) -> CommandReceipt | None:
        async with self._lifecycle:
            receipt = await self._stop_owned("coordinator_closed") if self._route is not None else None
            self._phase = CoordinationPhase.CLOSED
            self._reason = None
            self._reconnect_required = False
            return receipt

    async def interrupt_speech(self, command: InterruptSpeechCommand) -> CommandReceipt:
        """No lifecycle lock and no await of the context sender."""
        route = self._matching_owner(command.route)
        if self._phase not in {CoordinationPhase.ACTIVE, CoordinationPhase.DEGRADED, CoordinationPhase.RETIRING}:
            raise RouteCoordinationError("route_not_interruptible")
        try:
            receipt = await self._bounded(route.interrupt_speech(command), "interrupt")
            self._confirm(receipt, command.route, command.command_id, "interrupt_speech", fenced=True)
            return receipt
        except asyncio.CancelledError:
            self._degrade("interruption_cancelled", owner=route)
            raise
        except RouteCoordinationError as error:
            self._degrade(error.code, owner=route)
            raise

    async def retire_context(
        self, command: SourceRetirementCommand,
    ) -> SourceRetirementReceipt | UnsupportedOperation:
        route = self._matching_owner(command.route)
        # Fence synchronously before waiting for any serialized lifecycle work.
        self._control_epoch += 1
        if self._bridge is not None:
            self._bridge.fence("source_retirement")
        self._phase = CoordinationPhase.RETIRING
        self._reconnect_required = True
        async with self._lifecycle:
            if self._route is not route or self._identity != command.route:
                raise RouteCoordinationError("retirement_owner_changed")
            try:
                interrupt = InterruptSpeechCommand(route=command.route, command_id=uuid4(),
                    requested_at=self._clock(), reason="source_retirement")
                stopped = await self._bounded(route.interrupt_speech(interrupt), "retirement_interrupt")
                self._confirm(stopped, command.route, interrupt.command_id, "interrupt_speech", fenced=True)
                receipt = await self._bounded(route.retire_context(command), "source_retirement")
                if isinstance(receipt, SourceRetirementReceipt):
                    valid = (receipt.route == command.route and receipt.command_id == command.command_id
                             and receipt.cursor == command.cursor)
                elif isinstance(receipt, UnsupportedOperation):
                    valid = (receipt.route == command.route and receipt.request_id == command.command_id
                             and receipt.operation == RouteOperation.RETIRE_CONTEXT)
                else:
                    valid = False
                if not valid:
                    raise RouteCoordinationError("retirement_receipt_mismatch")
                # The current receipt vocabulary has no effective-forgetting or
                # replacement-output-fence guarantee. Never reopen this bridge.
                self._degrade("source_retirement_requires_reconnect", owner=route)
                return receipt
            except asyncio.CancelledError:
                self._degrade("source_retirement_cancelled", owner=route)
                raise
            except RouteCoordinationError as error:
                self._degrade(error.code, owner=route)
                raise

    def _check_candidate(self, route: VoiceRoute, request: RouteStart) -> None:
        identity = request.route
        if route.identity != identity or identity.scope != self._scope:
            raise RouteCoordinationError("candidate_scope_or_identity_mismatch")
        if identity.route_generation <= self._highest_generation or identity.route_id in self._seen_route_ids:
            raise RouteCoordinationError("candidate_generation_or_identity_reused")
        if request.initial_context.mode != "snapshot":
            raise RouteCoordinationError("initial_delta_not_supported")
        expiry = request.initial_context.expires_at
        if expiry is not None and expiry <= self._clock():
            raise RouteCoordinationError("initial_context_expired")
        capabilities = {item.operation: item.support for item in route.capabilities.operations}
        required = (RouteOperation.START, RouteOperation.PUBLISH_CONTEXT,
                    RouteOperation.INTERRUPT_SPEECH, RouteOperation.CLOSE)
        if any(capabilities.get(operation) != Support.SUPPORTED for operation in required):
            raise RouteCoordinationError("required_route_operation_unsupported")

    async def _start_owned(self, route: VoiceRoute, request: RouteStart) -> RouteIdentity:
        # Replacement may spend time fencing/joining the old owner. Revalidate
        # at the candidate boundary before assigning or starting a new speaker.
        self._check_candidate(route, request)
        self._route, self._identity = route, request.route
        self._highest_generation = request.route.route_generation
        self._seen_route_ids.add(request.route.route_id)
        self._phase = CoordinationPhase.STARTING
        self._reason = None
        self._reconnect_required = False
        self._control_epoch += 1
        epoch = self._control_epoch
        try:
            receipt = await self._bounded(route.start(request), "route_start")
            self._confirm(receipt, request.route, request.command_id, "start", fenced=False)
            if epoch != self._control_epoch or route.identity != request.route:
                raise RouteCoordinationError("route_start_invalidated")
            bridge = VoiceContextBridge(route, request.initial_context, validity=self._validity,
                publish_timeout=self._publish_timeout, clock=self._clock,
                preflight_expiry_refresh=self._preflight_expiry_refresh,
                on_failure=lambda reason: self._publication_failed(request.route, reason))
            self._bridge = bridge
            await bridge.start()
            self._phase = CoordinationPhase.ACTIVE
            return request.route
        except asyncio.CancelledError:
            self._degrade("route_start_cancelled", owner=route)
            raise
        except Exception as error:
            code = error.code if isinstance(error, RouteCoordinationError) else "route_start_failed"
            self._degrade(code, owner=route)
            raise RouteCoordinationError(code) from None

    async def _stop_owned(self, reason: str) -> CommandReceipt:
        route, identity, bridge = self._route, self._identity, self._bridge
        if route is None or identity is None:
            raise RouteCoordinationError("no_owned_route")
        self._control_epoch += 1
        self._phase = CoordinationPhase.CLOSING
        if bridge is not None:
            bridge.fence(reason)
        try:
            command = CloseRouteCommand(route=identity, command_id=uuid4(),
                                        requested_at=self._clock(), reason=reason)
            # Close the speaker/provider before joining writes its shutdown may
            # need to unblock. The context sender never owns this control lane.
            receipt = await self._bounded(route.close(command), "route_close")
            self._confirm(receipt, identity, command.command_id, "close", fenced=True)
            if bridge is not None:
                await self._bounded(bridge.close(), "context_join")
            pending = {task for task in self._calls if not task.done()}
            for task in pending:
                task.cancel()
            if pending:
                _, remaining = await asyncio.wait(pending, timeout=self._control_timeout)
                if remaining:
                    raise RouteCoordinationError("old_route_tasks_unjoined")
            self._route = None
            self._identity = None
            self._bridge = None
            self._phase = CoordinationPhase.EMPTY
            self._reason = None
            self._reconnect_required = False
            return receipt
        except asyncio.CancelledError:
            self._degrade("route_close_cancelled", owner=route)
            raise
        except Exception as error:
            code = error.code if isinstance(error, RouteCoordinationError) else "route_close_failed"
            self._degrade(code, owner=route)
            raise RouteCoordinationError(code) from None

    def _matching_owner(self, identity: RouteIdentity) -> VoiceRoute:
        if self._route is None or self._identity != identity:
            raise RouteCoordinationError("route_owner_mismatch")
        return self._route

    def _publication_failed(self, identity: RouteIdentity, reason: str) -> None:
        if self._identity == identity and self._phase == CoordinationPhase.ACTIVE:
            self._degrade(reason)

    def _degrade(self, reason: str, *, owner: VoiceRoute | None = None) -> None:
        if owner is not None and self._route is not owner:
            return
        self._phase = CoordinationPhase.DEGRADED
        self._reason = reason
        self._reconnect_required = True
        if self._bridge is not None:
            self._bridge.fence(reason)

    @staticmethod
    def _confirm(receipt, identity, command_id, operation, *, fenced: bool) -> None:
        if (not isinstance(receipt, CommandReceipt) or receipt.route != identity
                or receipt.command_id != command_id or receipt.operation != operation
                or receipt.status != "completed" or (fenced and receipt.output_fenced is not True)):
            raise RouteCoordinationError(operation + "_receipt_unconfirmed")

    def _call_finished(self, task: asyncio.Task) -> None:
        self._calls.discard(task)
        if not task.cancelled():
            task.exception()

    async def _bounded(self, operation, label: str):
        task = asyncio.create_task(operation)
        self._calls.add(task)
        task.add_done_callback(self._call_finished)
        try:
            done, _ = await asyncio.wait({task}, timeout=self._control_timeout)
        except asyncio.CancelledError:
            task.cancel()
            raise
        if not done:
            task.cancel()
            raise RouteCoordinationError(label + "_deadline")
        try:
            return task.result()
        except asyncio.CancelledError:
            raise RouteCoordinationError(label + "_cancelled") from None
        except Exception:
            raise RouteCoordinationError(label + "_failed") from None
