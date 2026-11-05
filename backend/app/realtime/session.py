from __future__ import annotations

import asyncio
import base64
import binascii
import json
import logging
import re
import time
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, replace
from collections.abc import Awaitable, Callable, Sequence
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4, uuid5

from app.memory.prospective.delivery import ProspectiveDeliveryAttempt

from fastapi import WebSocket, WebSocketDisconnect
from websockets.exceptions import ConnectionClosed, ConnectionClosedOK

from app.assistant.context import AssistantContext, AssistantContextBuilder
from app.assistant.memory_context import (
    session_working_memory_revision,
    working_memory_context_available,
    working_memory_next_expires_at,
    working_memory_app_event_payload,
)
from app.assistant.proactive_policy import (
    GenericSpeechActivity,
    ProactiveInterruptGate,
    generic_visual_speech_rejection,
)
from app.assistant.context_providers import AssistantContextProvider
from app.assistant.tool_limits import ToolCallBudget
from app.attention.admission_diagnostics import record_admission
from app.experience.repository import ToolExecutionRepository
from app.experience.evidence import durable_tool_payload
from app.experience.service import ExperienceEventService
from app.perception.frame_window import RecentVisualFrameWindow
from app.realtime.visual_delivery import RealtimeVisualCoordinator
from app.realtime.events import (
    ClientEventError,
    encode_event,
    optional_int,
    optional_string,
    parse_client_event,
    require_string,
)
from app.realtime.control_tools import create_visual_interrupt_control_registry
from app.realtime.context_refresh import RealtimeContextRefreshCoordinator
from app.realtime.instructions import (
    build_realtime_instructions,
    build_tool_followup_instructions,
    build_visual_interrupt_instructions,
    utc_now_iso,
)
from app.realtime.models import (
    PendingRealtimeResponse, RealtimeResponseContinuation, RealtimeResponseState,
    RealtimeSessionContext, ResponseKind,
    RealtimeToolRequest, RealtimeToolReceipt, RealtimeToolFollowup,
)
from app.realtime.openai_client import OpenAIRealtimeConnection, build_session_update_event
from app.realtime.output_policy import RealtimeOutputItemState
from app.realtime.socket_sender import WebSocketEventSender
from app.realtime.telemetry import RealtimeTrace, latency_phase
from app.realtime.tool_images import (
    ToolImageRequest, image_unavailable, resolve_and_deliver_tool_image,
)
from app.realtime.visual_context import (
    VisualConnectionKey, VisualContextState,
    VisualDeliveryPolicy, VisualResponseSnapshot,
    RequestedImageDeliveryError, VisualContextNotReadyError,
)
from app.reminders.repository import ReminderRepository
from app.runtime.events import DomainEvent, MemoryItemChangedPayload
from app.sessions.repository import SessionRepository
from app.settings import JarvisSettings
from app.tools.executor import JarvisToolExecutor
from app.tools.module import ToolBuildContext, ToolModule
from app.tools.registry import create_tool_registry
from app.tools.provenance import SourceBoundToolResult
from app.images import ImageResolver, image_source_payload_ids
from app.visual_memory.repository import VisualFrameRepository
from app.tasks.context_capture import SessionContextCapture
from app.realtime.source_context import LiveResponseSources, LiveContextError
from app.attention.quiet_delivery import (
    QuietDeliverySuppressed, quiet_response_guard, unsolicited_response_origin,
)
from app.realtime.continuity import recover_voice_context
from app.tasks.context_repository import ContextRepository
from app.realtime.request_sources import (
    REQUEST_SOURCE_ARGUMENT, RequestSourceScope, followup_request_sources, source_error,
)
from app.realtime.request_source_diagnostics import (
    close_request_source_diagnostics, record_request_source,
)
from app.tasks.models import TaskError


LOGGER = logging.getLogger(__name__)
OPENAI_SESSION_READY_TIMEOUT_SECONDS = 10.0
OPENAI_RESPONSE_START_TIMEOUT_SECONDS = 10.0
MAX_RECENT_CANCELLATIONS = 64
NO_ACTIVE_RESPONSE_CANCEL_CODE = "response_cancel_not_active"


@dataclass(frozen=True)
class _ResponseCancellation:
    response_id: str | None
    response_request_id: str | None


class RealtimeAudioSession:
    def __init__(
        self,
        *,
        websocket: WebSocket,
        context: RealtimeSessionContext,
        settings: JarvisSettings,
        repository: SessionRepository,
        tool_modules: Sequence[ToolModule],
        context_providers: Sequence[AssistantContextProvider],
        visual_repository: VisualFrameRepository | None = None,
        sensory_frame_window: RecentVisualFrameWindow | None = None,
        reminder_repository: ReminderRepository | None = None,
        tool_execution_repository: ToolExecutionRepository | None = None,
        experience_event_service: ExperienceEventService | None = None,
        context_capture: SessionContextCapture | None = None,
        provider_connection: OpenAIRealtimeConnection | None = None,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.websocket = websocket
        self.context = context
        self.settings = settings
        self.repository = repository
        self.visual_repository = visual_repository
        self.reminder_repository = reminder_repository
        self.experience_event_service = experience_event_service
        self._context_capture = context_capture
        self._source_context = LiveResponseSources(
            getattr(repository, "_pool", None), owner_id=context.owner_id,
            session_id=context.jarvis_session_id, audio_session_id=context.audio_session_id)
        self._context_revoked = False
        self._context_reset_id: str | None = None
        self._context_reset_ack = asyncio.Event()
        self._context_retirement_task: asyncio.Task | None = None
        self.assistant_context_builder = AssistantContextBuilder(
            providers=context_providers,
        )
        self.openai = provider_connection or OpenAIRealtimeConnection(settings=settings, owner_id=context.owner_id)
        self.tool_registry = create_tool_registry(
            modules=tool_modules,
            context=ToolBuildContext(
                settings=settings,
                owner_id=context.owner_id,
                session_id=context.jarvis_session_id,
            ),
        )
        self.tool_executor = JarvisToolExecutor(
            self.tool_registry,
            execution_repository=tool_execution_repository,
            experience_event_service=experience_event_service,
            owner_id=context.owner_id,
            session_id=context.jarvis_session_id,
            context_capture=context_capture,
            require_source_tracking=True,
            defer_experience_attachment=True,
            on_request_evidence=self._bind_tool_followup_request,
        )

        # Response controls are local, not domain tools or durable memory evidence.
        self._visual_interrupt_control_registry = create_visual_interrupt_control_registry()
        self._visual_interrupt_control_executor = JarvisToolExecutor(
            self._visual_interrupt_control_registry,
        )

        self._closed = asyncio.Event()
        self._close_task: asyncio.Task[None] | None = None
        self._provider_connect_task: asyncio.Task[None] | None = None
        self._realtime_context_ready = asyncio.Event()
        self._trace = RealtimeTrace(provider="openai", audio_session_id=context.audio_session_id,
                                    jarvis_session_id=context.jarvis_session_id)
        self._source_context.bind_trace(self._trace)
        if isinstance(self.openai, OpenAIRealtimeConnection):
            self.openai.bind_trace(self._trace)
        self._app_sender = WebSocketEventSender(
            websocket, on_closed=lambda: self._mark_closing("client_disconnect"),
            trace=self._trace,
        )
        # This lock owns response-control admission, not provider media transport.
        self._response_admission_lock = asyncio.Lock()
        self._attention_delivery = None
        self._attention_receipts = {}
        self._native_playback = {}
        self._native_playback_history = {}
        self._now = now
        self._monotonic = monotonic
        self._turn_generation = 0
        self._user_audio_pending = False
        self._normal_turn_pending = False
        self._normal_response_requested = False
        self._last_user_turn_settled_at: datetime | None = None
        self._last_user_turn_settled_monotonic: float | None = None
        self._pending_response: PendingRealtimeResponse | None = None
        self._latest_response_request_id: str | None = None
        self._response_start_task: asyncio.Task[None] | None = None
        self._response_preparation_task: asyncio.Task[bool] | None = None
        self._client_receive_task: asyncio.Task | None = None
        self._provider_receive_task: asyncio.Task | None = None
        self._assistant_context: AssistantContext | None = None
        self._audio_commit_request: tuple[str, int] | None = None
        self._tool_followup: RealtimeResponseContinuation | None = None
        # Keep task-only instructions locally, including across late provider ACKs.
        self._unacknowledged_response_requests: dict[str, PendingRealtimeResponse] = {}
        self._aborted_continuation_requests: set[str] = set()
        self._continuation_settlement_tasks: set[asyncio.Task[None]] = set()
        self._provider_completion_tasks: dict[str, asyncio.Task[None]] = {}
        self._provider_completion_capacity = asyncio.Semaphore(2)
        from app.realtime.response_control import ResponseControlLane
        self._response_controls = ResponseControlLane(
            send=self._write_response_control, fail=self._mark_closing,
            deadline=self.settings.OPENAI_REALTIME_WRITE_TIMEOUT_SECONDS,
        )
        self._visual_context = VisualContextState(
            connection=VisualConnectionKey(
                context.owner_id, context.jarvis_session_id, uuid4()
            ),
            policy=VisualDeliveryPolicy.from_settings(settings),
        )
        self._visual_delivery = (
            RealtimeVisualCoordinator(
                state=self._visual_context, frame_window=sensory_frame_window,
                provider=self.openai, max_image_bytes=settings.VISUAL_FRAME_MAX_BYTES,
                now=now, monotonic=monotonic,
                image_resolution_timeout_seconds=settings.VISUAL_IMAGE_RETRIEVAL_TIMEOUT_SECONDS,
                on_image_delivery=self._capture_image_delivery,
            )
            if sensory_frame_window is not None else None
        )
        self._current_response_id: str | None = None
        self._current_output_item_id: str | None = None
        self._active_responses: dict[str, RealtimeResponseState] = {}
        self._recent_cancellations: dict[str, _ResponseCancellation] = {}
        # Keep only identity tombstones after completion, not output text/media.
        # They last for this bounded-duration socket and reject delayed events.
        self._retired_response_ids: set[str] = set()
        self._output_policy_notices: set[str] = set()
        self._response_id_by_output_item_id: dict[str, str] = {}
        self._handled_tool_call_ids: set[str] = set()
        self._image_tool_tasks: dict[str, asyncio.Task[None]] = {}
        self._ordinary_tool_tasks: dict[str, asyncio.Task[None]] = {}
        self._ordinary_tool_chains: dict[str, str] = {}
        self._ordinary_tool_locks: dict[str, asyncio.Lock] = {}
        self._ordinary_tool_capacity = asyncio.Semaphore(4)
        self._image_tool_requests: dict[str, ToolImageRequest] = {}
        self._pending_reminder_events: dict[str, str] = {}
        self._prospective_attempts: dict[str, ProspectiveDeliveryAttempt] = {}
        self._pending_prospective_deliveries: dict[
            str,
            tuple[str, str, Callable[[str, str | None], Awaitable[None]]],
        ] = {}
        self._tool_budgets: dict[str, ToolCallBudget] = {}
        self._tool_chain_by_response_id: dict[str, str] = {}
        self._responses_with_tool_calls: set[str] = set()
        self._proactive_interrupt_gate = ProactiveInterruptGate(
            max_chain_length=settings.JARVIS_REALTIME_MAX_INTERRUPT_CHAIN
        )
        self._close_reason = "unknown"
        self._working_memory_expiry_task: asyncio.Task[None] | None = None
        self._context_refresh = RealtimeContextRefreshCoordinator(
            self._refresh_realtime_context,
        )
        self._voice_route_runtime = None

    def install_voice_route_runtime(self, runtime) -> None:
        """Explicit pre-run composition only; legacy sessions install nothing."""
        if runtime.session is not self:
            raise ValueError("Voice runtime belongs to another session.")
        if self._voice_route_runtime is not None or self._realtime_context_ready.is_set():
            raise RuntimeError("Voice runtime must be installed once before provider readiness.")
        runtime.install()

    async def run(self) -> None:
        owned_tasks: set[asyncio.Task] = set()
        try:
            if self._closed.is_set():
                return
            closed_task = asyncio.create_task(self._closed.wait())
            setup_task = asyncio.create_task(self._initialize_openai_session())
            owned_tasks.update((closed_task, setup_task))
            try:
                await asyncio.wait(
                    {setup_task, closed_task}, return_when=asyncio.FIRST_COMPLETED,
                )
                if self._closed.is_set():
                    return
                setup_task.result()
            except Exception as error:
                if self._context_revoked:
                    self._mark_closing("live_context_rebuild_required")
                    return
                from app.realtime.webrtc import log_webrtc_setup_failure
                if not log_webrtc_setup_failure(LOGGER, self.openai, error):
                    LOGGER.error("Jarvis OpenAI realtime setup failed error_type=%s", type(error).__name__)
                self._mark_closing("openai_realtime_connect_failed")
                await self._send_error(
                    "openai_realtime_connect_failed",
                    "OpenAI Realtime connection failed.",
                    recoverable=True,
                )
                return

            await self._send_app_event(
                "session.created",
                session_id=str(self.context.jarvis_session_id),
                audio_session_id=self.context.audio_session_id,
                client_kind=self.context.client_kind,
                server_time=utc_now_iso(),
                playback_receipts={"protocol_version":1, "sample_rate":24000},
                audio_transport=getattr(self.openai, "transport", "websocket"),
            )
            if self._closed.is_set():
                return

            client_task = asyncio.create_task(self._client_to_openai_loop())
            openai_task = asyncio.create_task(self._openai_to_client_loop())
            timeout_task = asyncio.create_task(self._max_duration_timer())
            source_task = asyncio.create_task(self._watch_source_context())
            owned_tasks.update((client_task, openai_task, timeout_task, source_task))
            owned_tasks.add(asyncio.create_task(self._recover_tool_acknowledgment()))

            done, _ = await asyncio.wait(
                {client_task, openai_task, timeout_task, source_task, closed_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in done:
                task.result()
        except Exception as error:
            LOGGER.error("Jarvis realtime audio session failed error_type=%s", type(error).__name__)
            self._mark_closing("realtime_session_failed")
            await self._send_error(
                "realtime_session_failed",
                "Realtime audio session failed.",
                recoverable=True,
            )
        finally:
            if self._voice_route_runtime is not None:
                # Keep the existing reader alive for correlated client fencing
                # before cancelling the session's transport tasks.
                await self._voice_route_runtime.before_session_close(self._close_reason)
            self._mark_closing(self._close_reason)
            for task in owned_tasks:
                if not task.done():
                    task.cancel()
            try:
                # Close the provider before joining writes that may need its
                # shutdown to unblock. Relay cancellation still releases locks.
                await self.close(self._close_reason)
            finally:
                await asyncio.gather(*owned_tasks, return_exceptions=True)

    async def _initialize_openai_session(self) -> None:
        if self._closed.is_set():
            return
        self._provider_connect_task = asyncio.create_task(self.openai.connect())
        self._trace.event("session_provider_connect_started")
        # Cleanup owns the handshake even if run() cancels this setup waiter.
        await asyncio.shield(self._provider_connect_task)
        if self._closed.is_set():
            return
        self._trace.event("session_provider_control_connected")
        await self._send_transport_setup()
        if self._closed.is_set():
            return
        initial_revision = await self._send_session_update()
        if self._closed.is_set():
            return
        self._trace.event("session_configuration_sent")
        self._context_refresh.initialize(initial_revision)
        await self._wait_for_openai_session_ready()
        if self._closed.is_set():
            return
        self._trace.event("session_configuration_acknowledged")
        self._realtime_context_ready.set()
        if self._visual_delivery is not None:
            self._visual_delivery.start()

    def _mark_closing(self, reason: str) -> None:
        # A send-side departure only signals the run loop. Never await cleanup
        # under the sender's lock, because cleanup can send final app events.
        if getattr(self, "_close_reason", "unknown") == "unknown":
            self._close_reason = reason
        self._closed.set()

    async def close(self, reason: str = "closed") -> None:
        if self._voice_route_runtime is not None:
            await self._voice_route_runtime.before_session_close(reason)
        # Admission stops immediately, but every caller must await actual cleanup.
        self._mark_closing(reason)
        close_task = getattr(self, "_close_task", None)
        if close_task is None:
            close_task = asyncio.create_task(self._close_resources(self._close_reason))
            self._close_task = close_task
        if close_task is asyncio.current_task():
            return
        # run() may cancel the client loop when provider shutdown ends its receive
        # loop. Cancelling that waiter must not cancel the shared cleanup task.
        await asyncio.shield(close_task)
        if self._voice_route_runtime is not None:
            await self._voice_route_runtime.after_session_close()

    async def _close_resources(self, reason: str) -> None:
        self._response_controls.retire()
        attention = getattr(self, "_attention_delivery", None)
        if attention is not None:
            await attention.stop(reason)
            if attention.task is not None and attention.task is not asyncio.current_task():
                await asyncio.gather(attention.task, return_exceptions=True)
        context_refresh = getattr(self, "_context_refresh", None)
        if context_refresh is not None:
            context_refresh.stop()
        producer_tasks = [
            task for task in (
                getattr(self, "_response_preparation_task", None),
                getattr(self, "_response_start_task", None),
                getattr(self, "_working_memory_expiry_task", None),
                getattr(self, "_context_retirement_task", None),
                *getattr(self, "_image_tool_tasks", {}).values(),
                *getattr(self, "_ordinary_tool_tasks", {}).values(),
                *getattr(self, "_provider_completion_tasks", {}).values(),
            )
            if task is not None and task is not asyncio.current_task()
        ]
        self._response_start_task = None
        self._working_memory_expiry_task = None
        self._cancel_image_tool_work()
        for task in producer_tasks:
            task.cancel()
        connecting = getattr(self, "_provider_connect_task", None)
        if connecting is not None:
            if not connecting.done():
                connecting.cancel()
            # Only connection acquisition is joined here, never provider writes.
            # A handshake completing during shutdown must precede transport close.
            await asyncio.gather(connecting, return_exceptions=True)
            self._provider_connect_task = None
        delivery = getattr(self, "_visual_delivery", None)
        if delivery is not None:
            await delivery.close()
        # Closing the transport must not wait behind a stalled response-control write.
        # The provider close path does not acquire the response admission lock.
        with suppress(Exception):
            await self.openai.close()
        admission_lock = getattr(self, "_response_admission_lock", None)
        if admission_lock is not None:
            async with admission_lock:
                self._turn_generation += 1
                self._pending_response = None
                self._latest_response_request_id = None
                self._tool_followup = None
                self._normal_turn_pending = self._user_audio_pending = False
                self._cancel_response_start_timer()
                self._visual_context.close()
                self._recent_cancellations.clear()
                self._active_responses.clear()
                self._retired_response_ids.clear()
                self._response_id_by_output_item_id.clear()
                self._output_policy_notices.clear()
                self._unacknowledged_response_requests.clear()
                self._aborted_continuation_requests.clear()
        await asyncio.gather(*producer_tasks, return_exceptions=True)
        await self._response_controls.close()
        if context_refresh is not None:
            with suppress(Exception):
                await context_refresh.close()
        # No new settlement task is admitted once closing starts. Join writes
        # already in flight before retrying any still-pending delivery at close.
        await asyncio.gather(
            *tuple(getattr(self, "_continuation_settlement_tasks", ())),
            return_exceptions=True,
        )
        await self._reconcile_pending_prospective_deliveries_on_close(reason=reason)
        await self._reconcile_pending_reminders_on_close(reason=reason)
        if capture := getattr(self, "_context_capture", None):
            await capture.close()
        with suppress(Exception):
            await self._send_app_event("session.closed", reason=reason,
                **({"session_id":str(self.context.jarvis_session_id), "audio_session_id":self.context.audio_session_id,
                    "context_reset":self._context_reset_notice()} if self._context_reset_id is not None else {}))
        with suppress(Exception):
            await self._app_sender.close()
        close_request_source_diagnostics(self)

    async def speak_visual_interrupt(
        self,
        *,
        owner_id: str,
        jarvis_session_id: UUID,
        source_received_at: datetime | None,
        message: str,
        interrupt_id: UUID,
        visual_frame_id: UUID | None = None,
        source_captured_at: datetime | None = None,
        reason: str | None = None,
        confidence: float | None = None,
        source_payload_id: UUID | None = None,
    ) -> bool:
        async with self._response_admission_lock:
            rejection = generic_visual_speech_rejection(
                self._generic_speech_activity(),
                source_matches=(
                    owner_id == self.context.owner_id
                    and jarvis_session_id == self.context.jarvis_session_id
                ),
                source_received_at=source_received_at,
                now=self._now(),
                monotonic_now=self._monotonic(),
            )
            if rejection is None:
                try:
                    async with asyncio.timeout(self.settings.OPENAI_REALTIME_WRITE_TIMEOUT_SECONDS):
                        async with quiet_response_guard(self._source_context, origin="visual_interrupt"):
                            pass
                except QuietDeliverySuppressed as error:
                    rejection = error.code
                except Exception:
                    rejection = "owner_quiet_unavailable"
            if rejection is None and not self._proactive_interrupt_gate.try_begin():
                rejection = "proactive_chain_limit"
            if rejection is not None:
                LOGGER.info(
                    "generic visual speech rejected interrupt_id=%s reason=%s",
                    interrupt_id, rejection,
                )
                return False
            self._observe_notification_source(source_payload_id)
            reservation = self._reserve_response_locked(
                "visual_interrupt", {"visual_interrupt_id": str(interrupt_id), "jarvis_initiation": "unsolicited"}
            )
        try:
            if not await self._send_reserved_response(
                reservation,
                instructions=build_visual_interrupt_instructions(
                    message=message,
                    visual_frame_id=visual_frame_id,
                    source_captured_at=source_captured_at,
                    source_received_at=source_received_at,
                    reason=reason,
                    confidence=confidence,
                ),
            ):
                return False
        except asyncio.CancelledError:
            raise
        except Exception:
            LOGGER.info("generic visual response send failed interrupt_id=%s", interrupt_id)
            return False
        if self._closed.is_set() or reservation.request_id in self._aborted_continuation_requests:
            return False
        await self._send_speech_notification(
            "visual.interrupt.started", interrupt_id=str(interrupt_id), message=message,
        )
        return True

    async def speak_reminder(
        self,
        *,
        message: str,
        reminder_id: UUID,
        reminder_event_id: UUID,
    ) -> None:
        if self._closed.is_set():
            raise RuntimeError("Realtime session is closed.")
        if self.reminder_repository is None:
            raise RuntimeError("Reminder repository is not configured for this realtime session.")
        reservation = await self._reserve_durable_speech(
            "reminder",
            {"reminder_id": str(reminder_id), "reminder_event_id": str(reminder_event_id)},
        )
        if reservation is None:
            raise RuntimeError("Realtime reminder speech could not be admitted.")
        event_key = str(reminder_event_id)
        # Keep reconciliation possible even if the dispatch write is cancelled.
        self._pending_reminder_events[event_key] = str(reminder_id)
        try:
            await self.reminder_repository.mark_event_delivery(
                owner_id=self.context.owner_id,
                event_id=reminder_event_id,
                event_type="dispatched_to_session",
                session_id=self.context.jarvis_session_id,
                metadata={"delivery_route": "realtime"},
            )
            if not await self._send_reserved_response(
                reservation,
                instructions=(
                    "Speak this Jarvis reminder now. Preserve its meaning, keep it concise, "
                    f"and do not add unrelated commentary. Reminder: {message}"
                ),
            ):
                raise RuntimeError("Reminder speech was superseded before submission.")
        except (Exception, asyncio.CancelledError) as error:
            cancelled = isinstance(error, asyncio.CancelledError)
            # A started write can time out after the provider accepted it.
            # Retain ownership so the caller cannot falsely mark it failed.
            if reservation.submitted:
                if cancelled:
                    raise
                return
            async with self._response_admission_lock:
                self._release_response_reservation(reservation)
            persisted = await self._persist_reminder_event_delivery(
                reminder_event_id=event_key,
                event_type="failed",
                metadata={
                    "delivery_route": "realtime",
                    "reason": "response_create_cancelled" if cancelled else "response_create_failed",
                },
            )
            if persisted:
                self._pending_reminder_events.pop(event_key, None)
            await self._send_speech_notification(
                "reminder.failed",
                reminder_id=str(reminder_id),
                reminder_event_id=event_key,
                message="Reminder audio could not be started.",
            )
            raise
        if not self._closed.is_set() and reservation.request_id not in self._aborted_continuation_requests:
            await self._send_speech_notification(
                "reminder.started",
                reminder_id=str(reminder_id),
                reminder_event_id=event_key,
                message=message,
            )

    async def speak_prospective_interrupt(
        self,
        *,
        message: str,
        event_id: UUID,
        subject_type: str,
        subject_id: UUID,
        on_outcome: Callable[[str, str | None], Awaitable[None]],
        delivery_attempt: ProspectiveDeliveryAttempt | None = None,
        source_payload_id: UUID | None = None,
    ) -> bool:
        if delivery_attempt is not None and delivery_attempt.revoked:
            return False
        self._observe_notification_source(source_payload_id)
        event_key = str(event_id)
        reservation = await self._reserve_durable_speech(
            "prospective",
            {
                "prospective_event_id": event_key,
                "prospective_subject_type": subject_type,
                "prospective_subject_id": str(subject_id),
            },
        )
        if reservation is None:
            if delivery_attempt is not None:
                delivery_attempt.reason = "session_unavailable_or_interrupt_limit"
            return False
        if delivery_attempt is not None:
            delivery_attempt.provider_owned = True
            delivery_attempt.on_revoke = lambda: self._clear_prospective_playback(event_key)
            self._prospective_attempts[event_key] = delivery_attempt
        self._pending_prospective_deliveries[event_key] = (
            subject_type,
            str(subject_id),
            on_outcome,
        )
        try:
            if not await self._send_reserved_response(
                reservation,
                instructions=(
                    "Say only this evidence-backed proactive Jarvis message now, then stop. "
                    "Do not resume earlier work or perform actions described in the message. "
                    f"Message: {message}"
                ),
            ):
                await self._complete_prospective_delivery(
                    event_id=event_key, outcome="failed", response_id=None,
                    reason="response_not_submitted",
                )
                return False
        except (Exception, asyncio.CancelledError) as error:
            if reservation.submitted:
                # A failed/abandoned socket write is not proof that the model
                # received nothing. Keep correlation and the durable fence for
                # a late provider outcome or transport-close reconciliation.
                raise
            async with self._response_admission_lock:
                self._release_response_reservation(reservation)
            await self._complete_prospective_delivery(
                event_id=event_key, outcome="failed", response_id=None,
                reason="response_preparation_failed",
            )
            raise
        if reservation.request_id in self._aborted_continuation_requests:
            return False
        if not self._closed.is_set():
            await self._send_speech_notification(
                "prospective.started",
                prospective_event_id=event_key,
                subject_type=subject_type,
                subject_id=str(subject_id),
                message=message,
            )
        return True

    def _observe_notification_source(self, source_payload_id: UUID | None) -> None:
        if source_payload_id is None:
            return
        try:
            self._source_context.observe((source_payload_id,))
        except LiveContextError:
            self._begin_context_rebuild()
            raise

    def _generic_speech_activity(self) -> GenericSpeechActivity:
        return GenericSpeechActivity(
            closed=self._closed.is_set(),
            transport_ready=self._realtime_context_ready.is_set(),
            user_audio_pending=self._user_audio_pending,
            normal_turn_pending=self._normal_turn_pending,
            speech_pending_or_active=(
                self._pending_response is not None or bool(self._active_responses)
                or self._attention_delivery is not None
            ),
            last_user_turn_settled_at=self._last_user_turn_settled_at,
            last_user_turn_settled_monotonic=self._last_user_turn_settled_monotonic,
        )

    async def _clear_prospective_playback(self, event_id: str) -> None:
        matching = [
            (state.response_id, item.item_id)
            for state in self._active_responses.values()
            if state.metadata.get("prospective_event_id") == event_id
            for item in state.output_items.values()
            if item.item_id is not None
        ]
        for response_id, item_id in matching:
            await self._send_app_event(
                "audio.output.clear", provider="openai",
                response_id=response_id, item_id=item_id, reason="prospective_revoked",
            )

    def _reserve_response_locked(
        self, kind: ResponseKind, metadata: dict[str, str]
    ) -> PendingRealtimeResponse:
        if self._pending_response is not None:
            raise RuntimeError("Another response is already reserved.")
        reservation = PendingRealtimeResponse(
            request_id=uuid4().hex, kind=kind, turn_generation=self._turn_generation,
            metadata=dict(metadata),
        )
        self._pending_response = reservation
        self._latest_response_request_id = reservation.request_id
        self._trace.event("response_admitted", request_id=reservation.request_id,
                          turn_generation=reservation.turn_generation, kind=kind,
                          event_id=metadata.get("jarvis_event_id"))
        return reservation

    def _visual_prompt_metadata(
        self, preparation: VisualResponseSnapshot | None = None,
    ) -> dict[str, object] | None:
        state = getattr(self, "_visual_context", None)
        if state is None:
            return None
        metadata: dict[str, object] = {
            "sampled_at": self._now().isoformat(),
            "camera_stream_active": bool(getattr(getattr(self, "_visual_delivery", None), "camera_stream_active", False)),
            "max_context_age_seconds": state.policy.max_context_age_seconds,
            "availability": (
                preparation.availability if preparation is not None
                else state.availability(now=self._now(), monotonic_now=self._monotonic())
            ).to_metadata(),
        }
        if preparation is not None:
            metadata["response_cutoff"] = preparation.to_metadata()
        else:
            metadata["latest_acknowledged"] = (
                state.latest_acknowledged.to_metadata() if state.latest_acknowledged else None
            )
        return metadata

    async def _send_reserved_response_locked(
        self, reservation: PendingRealtimeResponse, *, instructions: str | None = None,
        preparation: VisualResponseSnapshot | None = None,
    ) -> bool:
        from app.realtime.response_control import ResponseControlPending, ResponsePrewriteRejected
        # Legacy callers hold admission through this operation. Responsive
        # callers own the same reservation, but leave admission unlocked during
        # source preparation and the guarded provider write. The synchronous
        # factory below is the final eligibility check at the actual writer.
        route_runtime, provider = self._voice_route_runtime, self.openai
        capture = getattr(self, "_context_capture", None)
        if (reservation.kind == "normal_user_turn" and capture is not None
                and hasattr(capture, "request_sources")):
            reservation.request_sources = capture.request_sources(
                reservation.turn_generation, RequestSourceScope.from_session(self), reservation.tool_followup)
        request_sources = reservation.request_sources
        extra_roots = request_sources.roots if request_sources is not None else ()
        rejection_reason = rejection_boundary = None
        gate_trace_count = 0
        diagnostic_write_outcome = "no_return"
        latency_fields = {"request_id": reservation.request_id,
                          "turn_generation": reservation.turn_generation}

        def write_trace(name: str, *, reason: str, boundary: str, status: str) -> None:
            nonlocal gate_trace_count
            if name == "response_write_gate":
                if gate_trace_count > 8:
                    return
                if gate_trace_count == 8:
                    reason, boundary, status = "diagnostic_limit_reached", "response_write", "truncated"
                gate_trace_count += 1
            try:
                self._trace.event(name, request_id=reservation.request_id,
                                  turn_generation=reservation.turn_generation,
                                  reason=reason, boundary=boundary, status=status)
            except Exception:
                # Diagnostics cannot reject a response or retire its sources.
                pass

        def reject(reason: str, boundary: str) -> bool:
            nonlocal rejection_reason, rejection_boundary
            rejection_reason, rejection_boundary = reason, boundary
            write_trace("response_write_gate", reason=reason, boundary=boundary, status="rejected")
            return False

        def reservation_eligible(boundary: str) -> bool:
            attempt = self._prospective_attempt_for(reservation.metadata)
            if attempt is not None and attempt.revoked:
                return reject("prospective_revoked", boundary)
            if self._closed.is_set():
                return reject("session_closed", boundary)
            if self._context_revoked:
                return reject("context_revoked", boundary)
            if self._pending_response is not reservation:
                return reject("reservation_replaced", boundary)
            if reservation.turn_generation != self._turn_generation:
                return reject("turn_generation_mismatch", boundary)
            if self._voice_route_runtime is not route_runtime:
                return reject("route_runtime_replaced", boundary)
            if self.openai is not provider:
                return reject("provider_replaced", boundary)
            return True

        def payload_at_write() -> dict[str, Any] | None:
            if self._voice_route_runtime is not None and self._response_controls.busy:
                write_trace("response_write_gate", reason="response_control_pending",
                            boundary="payload_at_write", status="deferred")
                raise ResponseControlPending()
            if not reservation_eligible("payload_at_write"):
                return None
            if request_sources is not None and not request_sources.current(self, capture):
                reject("request_sources_changed", "payload_at_write")
                return None
            response_ready = getattr(route_runtime, "ready_for_response", None)
            if route_runtime is not None and not (
                    response_ready(reservation) if callable(response_ready) else route_runtime.ready):
                # The native wrapper also checks this after evaluating the
                # factory. Check before mutating visual/submission ownership:
                # a locally rejected payload is not an unknown provider write.
                reject("route_not_ready", "payload_at_write")
                try:
                    route_status = route_runtime.coordinator.status
                    phase = getattr(route_status.phase, "value", route_status.phase)
                    phase = phase if phase in {
                        "empty", "starting", "active", "retiring", "closing", "closed", "degraded",
                    } else "unknown"
                    reason = route_status.reason
                    reason = reason if reason in {
                        "context_validity_rejected", "context_expired_before_write",
                        "context_publication_deadline", "context_publication_cancelled",
                        "context_publication_failed", "context_receipt_mismatch_or_unsupported",
                        "context_receipt_unconfirmed", "route_start_cancelled", "route_start_failed",
                        "route_start_invalidated", "route_close_cancelled", "route_close_failed",
                        "source_retirement",
                    } else "none" if reason is None else "unknown"
                    write_trace("response_route_state", reason=reason,
                                boundary="payload_at_write", status=phase)
                except Exception:
                    # Diagnostics cannot replace the already-established rejection.
                    pass
                return None
            if snapshot is None and self._source_context.has_sources:
                # Sources may arrive while a formerly source-free request waits
                # for the writer. It needs lineage before any response write.
                raise LiveContextError("live_context_changed")
            delivery = reservation.tool_followup
            if delivery is not None:
                request = delivery.request
                if delivery.binding_conflict:
                    reject("tool_followup_binding_conflict", "payload_at_write")
                    return None
                if request is not None:
                    if (request.owner_id != self.context.owner_id
                            or request.session_id != self.context.jarvis_session_id
                            or request.turn_generation != reservation.turn_generation):
                        reject("tool_followup_scope_mismatch", "payload_at_write")
                        return None
                    roots = {request.payload_id, *(r.payload_id for r in delivery.additional_requests),
                             *((delivery.conversation_request.payload_id,) if delivery.conversation_request else ()),
                             *(item.payload_id for item in delivery.receipts
                                                   if item.payload_id is not None)}
                    if snapshot is None or not roots.issubset(snapshot.sources):
                        raise LiveContextError("live_context_source_invalid")
            if preparation is not None and not self.mark_visual_response_submitted(
                preparation, request_id=reservation.request_id,
            ):
                reject("visual_preparation_ineligible", "payload_at_write")
                if reservation.requested_image_call_ids:
                    raise RequestedImageDeliveryError(
                        "image_inputs_changed", "The requested images are no longer eligible.",
                    )
                return None
            cutoff = self._visual_context.active_snapshot if preparation is not None else None
            live_view = self._visual_prompt_metadata(cutoff)
            if reservation.unavailable_requested_images:
                live_view = {
                    **(live_view or {}),
                    "unavailable_requested_images": list(reservation.unavailable_requested_images),
                }
            complete_instructions = build_realtime_instructions(
                context=self._assistant_context,
                live_view=live_view,
                tasks_enabled=self.tool_registry.get("start_background_task") is not None,
            )
            task_instructions = instructions if instructions is not None else reservation.task_instructions
            if task_instructions is not None:
                complete_instructions += "\n\nCurrent response task:\n" + task_instructions
            context_unavailable = getattr(route_runtime, "normal_turn_context_unavailable", None)
            if callable(context_unavailable) and context_unavailable(reservation) is True:
                complete_instructions += (
                    "\n\nCurrent Live State briefing is temporarily unavailable. Its previous briefing "
                    "was removed, not refreshed. Do not present earlier visual observations or task "
                    "status as current. Use the user's current request and independently supplied "
                    "tool results; acknowledge missing information or ask when needed. This is not "
                    "evidence that memories, tasks, or responsibilities are absent.")
            followup_anchor = build_tool_followup_instructions(delivery) if delivery is not None else ""
            complete_instructions += followup_anchor
            complete_instructions += followup_request_sources(delivery)
            if request_sources is not None:
                complete_instructions += request_sources.instructions()
            response: dict[str, Any] = {
                "instructions": complete_instructions,
                "metadata": {
                    **reservation.metadata,
                    "jarvis_kind": reservation.kind,
                    "jarvis_request_id": reservation.request_id,
                    "jarvis_turn_generation": str(reservation.turn_generation),
                    "audio_session_id": self.context.audio_session_id,
                    "jarvis_session_id": str(self.context.jarvis_session_id),
                },
            }
            if reservation.metadata.get("jarvis_delivery_kind", reservation.kind) == "visual_interrupt":
                response["tools"] = [
                    *self.tool_registry.openai_realtime_tools(),
                    *self._visual_interrupt_control_registry.openai_realtime_tools(),
                ]
                response["tool_choice"] = "auto"
            if request_sources is not None:
                response["tools"] = request_sources.tools(self.tool_registry.openai_realtime_tools())
            if cutoff is not None and cutoff.requested_images:
                response["metadata"]["jarvis_image_response"] = "true"
            if reservation.kind == "prospective":
                response.update(tools=[], tool_choice="none")
            if reservation.metadata.get("jarvis_failure_notice"):
                # One bounded spoken error, not a retry of the task or its tools.
                # Exclude the failed image and conversation inputs from inference.
                response.update(tool_choice="none", tools=[], input=[], conversation="none")
            self._trace.tool_followup_write(request_id=reservation.request_id,
                anchor=followup_anchor, instructions=complete_instructions)
            reservation.submitted = True
            if cutoff is not None:
                LOGGER.info(
                    "visual response cutoff request_id=%s generation_id=%s synchronized=%s "
                    "availability=%s frame_ids=%s prepared_at=%s submitted_at=%s",
                    reservation.request_id, cutoff.generation_id, cutoff.synchronized,
                    cutoff.availability.status.value,
                    [str(item.frame.frame_id) for item in cutoff.acknowledged_items],
                    cutoff.prepared_at.isoformat(), cutoff.response_submitted_at.isoformat(),
                )
            write_trace("response_write_gate", reason="payload_prepared",
                        boundary="payload_at_write", status="accepted")
            record_request_source(self, "payload_factory", reservation=reservation,
                                  offer=request_sources, response=response)
            return {"type": "response.create", "event_id": reservation.request_id, "response": response}

        try:
            # Mark submission only at the common writer, never while queued.
            # This distinguishes a safe prewrite failure from an unknown write.
            snapshot = None
            refresh_stale, needs_prepare, source_retries = False, True, 0
            preparation_deadline = (asyncio.get_running_loop().time()
                                    + self.settings.OPENAI_REALTIME_WRITE_TIMEOUT_SECONDS)
            while True:
                if not reservation_eligible("preparation_preflight"):
                    sent = False
                    break
                if self._voice_route_runtime is not None:
                    with latency_phase(self._trace, "response_preparation_latency", phase="control_wait", fields=latency_fields):
                        await self._response_controls.wait_idle()
                try:
                    if needs_prepare and (self._source_context.has_sources or extra_roots):
                        # Reuse the original preparation deadline, not a fresh
                        # timeout on each retry. The writer retains its own bound.
                        with latency_phase(self._trace, "response_preparation_latency", phase="source_prepare", fields=latency_fields):
                            async with asyncio.timeout_at(preparation_deadline):
                                snapshot = await self._source_context.prepare(
                                    reservation.request_id,
                                    **({"refresh_stale": True} if refresh_stale else {}),
                                    **({"extra_sources": extra_roots} if extra_roots else {}),
                                )
                        needs_prepare = False
                    if not reservation_eligible("prepared_preflight"):
                        sent = False
                        break
                    source_guard = (lambda: self._source_context.guard(snapshot)) if snapshot is not None else None
                    origin = unsolicited_response_origin(reservation.kind, reservation.metadata)
                    write_guard = (lambda: quiet_response_guard(
                        self._source_context, origin=origin, source_guard=source_guard)) if origin else source_guard
                    with latency_phase(self._trace, "response_preparation_latency", phase="guarded_write", fields=latency_fields):
                        sent = await self.openai.send_event(payload_at_write, **(
                            {"write_guard": write_guard} if write_guard is not None else {}))
                    break
                except ResponseControlPending:
                    # Only the synchronous prewrite factory can raise this.
                    # Nothing was submitted; drain controls and recheck both
                    # reservation and source guards at the actual writer.
                    continue
                except ResponsePrewriteRejected as error:
                    if error.request_id != reservation.request_id:
                        # Never clear another request's possibly-written state.
                        raise
                    # The runtime raises this only before the underlying socket
                    # writer receives the payload, including an awaited factory
                    # that lost readiness after marking its local reservation.
                    # This is not a retry or a claim about any previous response.
                    reservation.submitted = False
                    reject(error.reason, "route_prewrite")
                    sent = False
                    break
                except LiveContextError as error:
                    if error.code != "live_context_changed" or reservation.submitted is not False:
                        # Once the payload factory has marked submission, even
                        # an exception after send/guard exit is an unknown write.
                        raise
                    if not reservation_eligible("source_retry_preflight"):
                        sent = False
                        break
                    if source_retries >= 2:
                        self._trace.event("response_source_preparation", request_id=reservation.request_id,
                                          turn_generation=reservation.turn_generation, status="retry_exhausted")
                        # Surface failure through existing retirement and owned
                        # preparation settlement, never silently lose/replay a turn.
                        raise
                    source_retries += 1
                    refresh_stale, needs_prepare = True, True
                    self._trace.event("response_source_preparation", request_id=reservation.request_id,
                                      turn_generation=reservation.turn_generation, status="retry_prewrite")
            diagnostic_write_outcome = "local_send_returned" if sent else "not_sent"
            write_trace("response_write_finished",
                        reason=("transport_write_completed" if sent else rejection_reason
                                or "transport_not_written_without_factory_reason"),
                        boundary="transport_return" if sent else rejection_boundary or "transport_return",
                        status="written" if sent else "not_submitted")
            return sent
        except QuietDeliverySuppressed:
            # Quiet is not source invalidation, route retirement or task cancellation.
            return False
        except Exception:
            if self._source_context.has_sources:
                # A stale provider session is not repaired by changing its prompt.
                # Retire it rather than silently continuing with revoked history.
                self._begin_context_rebuild()
            raise
        finally:
            record_request_source(self, "writer_finished", reservation=reservation,
                                  offer=request_sources, outcome=diagnostic_write_outcome)
            if reservation.submitted and extra_roots:
                # History enrollment follows possible exposure, never mere
                # preparation. Unknown writes retain withdrawal fencing too.
                self._source_context.observe(extra_roots)
            if (
                reservation.submitted and self._pending_response is reservation
                and not self._closed.is_set()
            ):
                self._cancel_response_start_timer()
                self._response_start_task = asyncio.create_task(
                    self._watch_response_start(reservation.request_id)
                )

    async def _send_reserved_response(
        self, reservation: PendingRealtimeResponse, *, instructions: str | None = None,
    ) -> bool:
        if instructions is not None:
            reservation.task_instructions = instructions
        self._unacknowledged_response_requests[reservation.request_id] = reservation
        previous = self._response_preparation_task
        task = asyncio.create_task(
            self._prepare_and_send_response(reservation, instructions, previous)
        )
        self._response_preparation_task = task
        task.add_done_callback(lambda finished: self._response_preparation_finished(finished, reservation))
        # Receiving continues while the owned task waits for image ACKs.
        # Direct callers (including durable delivery) still await the actual send.
        if asyncio.current_task() in {self._client_receive_task, self._provider_receive_task}:
            return True
        try:
            return await task
        except asyncio.CancelledError:
            caller = asyncio.current_task()
            if caller is None or caller.cancelling():
                # Actual producer shutdown still owns cancellation and cleanup.
                raise
            # Barge-in/session close cancelled our owned child, not its producer.
            # Before submission this is simply a rejected admission. After a
            # write begins its outcome remains unknown: retain durable delivery
            # correlation and image pins until provider settlement or close.
            return reservation.submitted

    def _response_preparation_finished(
        self, task: asyncio.Task[bool], reservation: PendingRealtimeResponse,
    ) -> None:
        if self._response_preparation_task is task:
            self._response_preparation_task = None
        error = None if task.cancelled() else task.exception()
        if error is not None:
            LOGGER.warning("realtime response preparation failed")
            # Error messages may contain user/provider content; retain only the
            # class and the existing request identity, not str(error) or payloads.
            self._trace.event("response_preparation_failed", request_id=reservation.request_id,
                              turn_generation=reservation.turn_generation,
                              error_type=type(error).__name__)
        capture = getattr(self, "_context_capture", None)
        if capture is not None and hasattr(capture, "retain_unwritten"):
            if reservation.submitted:
                capture.consume_unwritten(reservation.request_sources)
            elif (reservation.preserve_unwritten_request and reservation.user_admission_scope is not None
                    and not self._closed.is_set() and not self._context_revoked
                    and reservation.user_admission_scope.same(RequestSourceScope.from_session(self))):
                # A done callback is joined preparation evidence, including a
                # coroutine cancelled before its first instruction ran.
                capture.retain_unwritten(reservation.turn_generation, reservation.user_admission_scope)
        if (
            reservation.submitted
            or (not task.cancelled() and error is None and task.result())
        ):
            # A write that started can still be acknowledged after cancellation
            # or transport failure. Only provider rejection/settlement is final.
            return
        self._trace.event(
            "response_not_submitted", request_id=reservation.request_id,
            turn_generation=reservation.turn_generation, kind=reservation.kind,
            status="cancelled" if task.cancelled() else "failed" if error is not None else "superseded",
        )
        if self._unacknowledged_response_requests.get(reservation.request_id) is not reservation:
            return
        self._unacknowledged_response_requests.pop(reservation.request_id)
        self._release_tool_images(reservation.requested_image_call_ids)
        if not reservation.is_delivery_continuation:
            return
        self._aborted_continuation_requests.add(reservation.request_id)
        if self._closed.is_set():
            return
        # A task cancelled before its coroutine starts has no finally block.
        # The done callback owns this case too, without DB writes under admission.
        reason = (
            "continuation_cancelled" if task.cancelled()
            else "continuation_preparation_failed" if error is not None
            else "continuation_superseded"
        )
        settlement = asyncio.create_task(self._settle_unstarted_continuation(
            reservation.metadata, request_id=reservation.request_id, reason=reason,
        ))
        self._continuation_settlement_tasks.add(settlement)
        settlement.add_done_callback(self._continuation_settlement_finished)

    def _continuation_settlement_finished(self, task: asyncio.Task[None]) -> None:
        self._continuation_settlement_tasks.discard(task)
        if not task.cancelled() and task.exception() is not None:
            LOGGER.warning("continuation delivery settlement failed")

    async def _prepare_and_send_response(
        self, reservation: PendingRealtimeResponse, instructions: str | None,
        previous: asyncio.Task[bool] | None,
    ) -> bool:
        preparation = None
        outcome = "superseded"
        context_refresh_waits = 0
        latency_fields = {"request_id": reservation.request_id,
                          "turn_generation": reservation.turn_generation}
        self._trace.event("preparation_started", request_id=reservation.request_id,
                          turn_generation=reservation.turn_generation)
        try:
            # Barge-in cancels a never-submitted waiter; let its finally release
            # the old fence before the next generation uses the same coordinator.
            if previous is not None and not previous.done():
                with latency_phase(self._trace, "response_preparation_latency", phase="prior_preparation_join", fields=latency_fields):
                    await asyncio.gather(previous, return_exceptions=True)
            async with self._response_admission_lock:
                if self._closed.is_set() or self._pending_response is not reservation:
                    return False
            while True:
                try:
                    route = self._voice_route_runtime
                    if getattr(route, "context_refresh_pending", False) is True:
                        if context_refresh_waits >= 1:
                            raise TimeoutError("context_refresh_wait_exhausted")
                        context_refresh_waits += 1
                        self._trace.event("response_context_wait", request_id=reservation.request_id,
                                          turn_generation=reservation.turn_generation, status="waiting")
                        normal_prepare = getattr(route, "prepare_normal_turn_context", None)
                        normal_prepare = (normal_prepare if reservation.kind == "normal_user_turn"
                                          and callable(normal_prepare) else None)
                        if normal_prepare is None:
                            self._context_waiting_request_id = reservation.request_id
                        try:
                            with latency_phase(self._trace, "response_preparation_latency", phase="context_ready_wait", fields=latency_fields):
                                available = (await normal_prepare(reservation,
                                    self.settings.OPENAI_REALTIME_WRITE_TIMEOUT_SECONDS)
                                    if normal_prepare is not None else await route.wait_for_context_ready(
                                        self.settings.OPENAI_REALTIME_WRITE_TIMEOUT_SECONDS))
                        finally:
                            if getattr(self, "_context_waiting_request_id", None) == reservation.request_id:
                                self._context_waiting_request_id = None
                        self._trace.event("response_context_wait", request_id=reservation.request_id,
                                          turn_generation=reservation.turn_generation,
                                          status="ready" if available else "unavailable")
                        if not available:
                            raise TimeoutError("context_refresh_unavailable")
                        async with self._response_admission_lock:
                            if (self._closed.is_set() or self._voice_route_runtime is not route
                                    or self._pending_response is not reservation
                                    or reservation.turn_generation != self._turn_generation):
                                return False
                    requested = reservation.requested_image_call_ids
                    if requested and any(
                        self._visual_context.requested_pin(call_id) is None
                        for call_id in requested
                    ):
                        raise RequestedImageDeliveryError(
                            "image_inputs_changed", "A requested image is no longer available.",
                        )
                    with latency_phase(self._trace, "response_preparation_latency", phase="visual_prepare", fields=latency_fields):
                        preparation = await self.prepare_visual_response(
                            turn_id=str(reservation.turn_generation),
                            generation_id=UUID(hex=reservation.request_id),
                            **({"requested_call_ids": requested} if requested else {}),
                        )
                    self._trace.event("image_context_prepared", request_id=reservation.request_id,
                                      turn_generation=reservation.turn_generation,
                                      status=preparation.availability.status.value if preparation is not None else "disabled")
                    for item in preparation.acknowledged_items if preparation is not None else ():
                        self._trace.event("image_provider_acknowledged", request_id=reservation.request_id,
                                          frame_id=item.frame.frame_id,
                                          captured_at=getattr(item.frame, "captured_at", None),
                                          server_received_at=getattr(item.frame, "server_received_at", None))
                    attempt = self._prospective_attempt_for(reservation.metadata)
                    if attempt is not None and not await attempt.prepare():
                        async with self._response_admission_lock:
                            self._release_response_reservation(reservation)
                        return False
                    async with self._response_admission_lock:
                        if (
                            self._closed.is_set() or self._pending_response is not reservation
                            or reservation.turn_generation != self._turn_generation
                        ):
                            return False
                        if self._voice_route_runtime is None:
                            # Keep default/legacy admission behavior unchanged.
                            sent = await self._send_reserved_response_locked(
                                reservation, instructions=instructions, preparation=preparation,
                            )
                            if not sent:
                                self._release_response_reservation(reservation)
                            outcome = "written" if sent else "not_submitted"
                            return sent
                    # The existing owned preparation task remains the only
                    # submitter. Do not hold control admission while waiting for
                    # sources, a provider writer, or its canonical transaction.
                    # Barge-in can cancel this task and replace its reservation;
                    # the at-write factory must still reject that old generation
                    # even if an external await suppresses cancellation.
                    sent = await self._send_reserved_response_locked(
                        reservation, instructions=instructions, preparation=preparation,
                    )
                    if (not sent and not reservation.submitted and attempt is None
                            and reservation.kind == "normal_user_turn"
                            and self._voice_route_runtime is route
                            and getattr(route, "context_refresh_pending", False) is True
                            and self._pending_response is reservation
                            and reservation.turn_generation == self._turn_generation
                            and not self._closed.is_set()):
                        # Only a proven-unwritten normal turn may wait once for
                        # fresh context. No tool/prospective work or uncertain
                        # provider write is replayed. All admission checks repeat.
                        if context_refresh_waits >= 1:
                            raise TimeoutError("context_refresh_wait_exhausted")
                        if preparation is not None:
                            self.cancel_visual_response_preparation(preparation)
                            preparation = None
                        continue
                    async with self._response_admission_lock:
                        if not sent:
                            # Identity-aware release cannot clear a newer turn
                            # or a reservation already consumed by provider ACK.
                            self._release_response_reservation(reservation)
                    outcome = "written" if sent else "not_submitted"
                    return sent
                except (RequestedImageDeliveryError, VisualContextNotReadyError):
                    if not reservation.requested_image_call_ids or reservation.submitted:
                        raise
                    # One never-submitted transport fallback, not a model/tool retry.
                    # Previously attached pixels must not certify a current view.
                    if preparation is not None:
                        self.cancel_visual_response_preparation(preparation)
                        preparation = None
                    reservation.unavailable_requested_images = tuple(
                        {
                            "provider_call_id": call_id, "image_status": "unavailable",
                            "reason": "image_inputs_changed",
                            "image": request.reference.to_metadata() if request and request.reference else None,
                        }
                        for call_id in reservation.requested_image_call_ids
                        for request in (self._image_tool_requests.get(call_id),)
                    )
                    self._release_tool_images(reservation.requested_image_call_ids)
                    reservation.requested_image_call_ids = ()
        except (Exception, asyncio.CancelledError) as error:
            outcome = "cancelled" if isinstance(error, asyncio.CancelledError) else "failed"
            async with self._response_admission_lock:
                if not reservation.submitted:
                    self._release_response_reservation(reservation)
            if isinstance(error, asyncio.CancelledError):
                raise
            if not self._closed.is_set() and not (self._context_revoked and not reservation.submitted):
                uncertain = reservation.submitted
                uncertain_code = (
                    "prospective_delivery_outcome_unknown" if reservation.kind == "prospective"
                    else "response_write_outcome_unknown"
                )
                await self._send_error(
                    uncertain_code if uncertain else "response_preparation_failed",
                    ("The response write did not finish. Its outcome is unknown; it will not be automatically replayed."
                     if uncertain else "The response could not be prepared. Please try speaking again."),
                    recoverable=True,
                )
            raise
        finally:
            self._trace.event("preparation_finished", request_id=reservation.request_id,
                              turn_generation=reservation.turn_generation, status=outcome)
            if preparation is not None:
                self.cancel_visual_response_preparation(preparation)
            self._release_tool_images(reservation.requested_image_call_ids)

    def _cancel_response_start_timer(self) -> None:
        task = self._response_start_task
        self._response_start_task = None
        if task is not None and task is not asyncio.current_task():
            task.cancel()

    async def _watch_response_start(self, request_id: str) -> None:
        await asyncio.sleep(OPENAI_RESPONSE_START_TIMEOUT_SECONDS)
        async with self._response_admission_lock:
            pending = self._pending_response
            if self._closed.is_set() or pending is None or pending.request_id != request_id:
                return
        # Unknown outcome remains reserved. Do not automatically retry speech.
        await self._send_error(
            "response_start_timeout",
            "The response has not started yet. Its outcome is still unknown.",
            recoverable=True,
        )

    def _release_response_reservation(self, reservation: PendingRealtimeResponse) -> None:
        if self._pending_response is not reservation:
            return
        self._pending_response = None
        self._cancel_response_start_timer()
        if self._latest_response_request_id == reservation.request_id:
            self._latest_response_request_id = None
        if reservation.kind == "normal_user_turn":
            self._settle_normal_turn(reservation.turn_generation)
        # Never reset the proactive chain on failure.

    async def _cancel_response_locked(self, *, keep_image_calls: tuple[str, ...] = ()) -> None:
        if self._attention_delivery is not None:
            delivery = self._attention_delivery
            request_stop = getattr(delivery, "request_stop", None)
            if self._voice_route_runtime is not None and request_stop is not None:
                fence = request_stop("user_interrupted")
                self._response_controls.hold_fence(str(id(delivery)), fence)
                # The delivery releases itself after its actual fence and
                # settlement. Never give a queued old clear a newer speaker.
            else:
                await delivery.stop("user_interrupted")
                self._attention_delivery = None
        for item_id in tuple(self._native_playback):
            self._retire_native_playback(item_id)
        self._cancel_image_tool_work(keep_call_ids=keep_image_calls)
        preparing = self._response_preparation_task
        if preparing is not None and not preparing.done() and preparing is not asyncio.current_task():
            preparing.cancel()
        pending = self._pending_response
        current = self._active_responses.get(self._current_response_id)
        should_cancel = (
            (pending is not None and pending.submitted)
            or (current is not None and not current.interrupted and current.completed_response is None)
        )
        if current is not None:
            current.interrupted = True
            if capture := getattr(self, "_context_capture", None):
                capture.finish_response(current.response_id, interrupted=True)
        self._pending_response = None
        self._latest_response_request_id = None
        self._current_response_id = self._current_output_item_id = None
        self._cancel_response_start_timer()
        if should_cancel:
            await self._send_response_cancel_locked(
                response_id=current.response_id if current is not None else None,
                response_request_id=(
                    pending.request_id if pending is not None and pending.submitted
                    else current.metadata.get("jarvis_request_id") if current is not None else None
                ),
            )

    async def _send_response_cancel_locked(
        self, *, response_id: str | None, response_request_id: str | None,
    ) -> None:
        if self._voice_route_runtime is not None:
            self._response_controls.cancel(response_id, response_request_id)
            return
        await self._write_response_cancel(response_id=response_id, response_request_id=response_request_id)

    async def _write_response_control(self, kind, response_id, response_request_id) -> None:
        if self._closed.is_set():
            return
        if kind == "clear":
            await self.openai.send_event({"type": "output_audio_buffer.clear"})
        else:
            await self._write_response_cancel(response_id=response_id, response_request_id=response_request_id)

    async def cancel_route_notification(self, *, request_id: str, offer_id: str) -> None:
        """Revoke only an exactly bound notification, never the current user turn."""
        async with self._response_admission_lock:
            if self._closed.is_set():
                return
            pending = self._unacknowledged_response_requests.get(request_id)
            if (pending is not None and pending.submitted
                    and pending.metadata.get("voice_route_offer_id") == offer_id):
                self._aborted_continuation_requests.add(request_id)
                if self._pending_response is pending:
                    self._release_response_reservation(pending)
                await self._send_response_cancel_locked(response_id=None, response_request_id=request_id)
            for state in tuple(self._active_responses.values()):
                if (state.metadata.get("jarvis_request_id") != request_id
                        or state.metadata.get("voice_route_offer_id") != offer_id
                        or state.interrupted or state.completed_response is not None):
                    continue
                state.interrupted = True
                if (capture := getattr(self, "_context_capture", None)) is not None:
                    capture.finish_response(state.response_id, interrupted=True)
                if self._current_response_id == state.response_id:
                    self._current_response_id = self._current_output_item_id = None
                await self._send_response_cancel_locked(
                    response_id=state.response_id, response_request_id=request_id)

    async def _write_response_cancel(self, *, response_id, response_request_id) -> None:
        if self._closed.is_set():
            return
        request_id = uuid4().hex
        self._recent_cancellations[request_id] = _ResponseCancellation(
            response_id=response_id, response_request_id=response_request_id,
        )
        # Successful cancellation has response.done, not a separate cancel ACK.
        # Keep bounded tombstones so a delayed no-active-response error remains
        # attributable even after a newer user turn has started.
        while len(self._recent_cancellations) > MAX_RECENT_CANCELLATIONS:
            self._recent_cancellations.pop(next(iter(self._recent_cancellations)))
        payload = {"type": "response.cancel", "event_id": request_id}
        if response_id is not None:
            payload["response_id"] = response_id
        await self.openai.send_event(payload)

    async def _consume_cancel_race(self, error: Any) -> bool:
        request_id = _string_value(error, "event_id")
        if request_id is None:
            return False
        async with self._response_admission_lock:
            cancellation = self._recent_cancellations.pop(request_id, None)
            if cancellation is None or _string_value(error, "code") != NO_ACTIVE_RESPONSE_CANCEL_CODE:
                return False
            if self._visual_delivery is not None and cancellation.response_request_id is not None:
                self._visual_delivery.settle_response(
                    request_id=cancellation.response_request_id,
                    response_id=cancellation.response_id,
                )
        # Do not settle/reset the current user turn, its audio commit, reminders,
        # or a newer response. Only this cancellation's old visual fence settles.
        LOGGER.debug("realtime cancellation already settled request_id=%s", request_id)
        return True

    async def _reserve_durable_speech(
        self, kind: ResponseKind, metadata: dict[str, str]
    ) -> PendingRealtimeResponse | None:
        async with self._response_admission_lock:
            if self._attention_delivery is not None:
                return None
            if self._closed.is_set() or not self._realtime_context_ready.is_set():
                return None
            if not self._proactive_interrupt_gate.try_begin():
                return None
            # Preserve explicit reminders' interrupt behavior, not the generic quiet rule.
            await self._cancel_response_locked()
            if self._normal_turn_pending and not self._user_audio_pending:
                self._turn_generation += 1
                self._settle_normal_turn(self._turn_generation)
            return self._reserve_response_locked(kind, metadata)

    async def _begin_user_turn_locked(self) -> None:
        pending = self._pending_response
        if pending is not None and pending.user_admission_scope is not None:
            pending.preserve_unwritten_request = True
        self._turn_generation += 1
        self._normal_turn_pending = True
        self._normal_response_requested = False
        self._tool_followup = None
        self._proactive_interrupt_gate.reset_for_user_turn()
        await self._cancel_response_locked()

    def _settle_normal_turn(self, generation: int) -> None:
        if generation != self._turn_generation:
            return
        self._normal_turn_pending = self._normal_response_requested = False
        self._user_audio_pending = False
        self._tool_followup = None
        self._last_user_turn_settled_at = self._now()
        self._last_user_turn_settled_monotonic = self._monotonic()

    async def request_route_notification(self, offer, *, turn_generation: int) -> bool:
        """Reserve one authorized callback without manufacturing a user turn."""
        async with self._response_admission_lock:
            delivery = self._attention_delivery
            if (self._closed.is_set() or not self._realtime_context_ready.is_set()
                    or turn_generation != self._turn_generation or self._user_audio_pending
                    or self._normal_turn_pending or self._pending_response is not None
                    or self._active_responses or self._native_playback
                    or delivery is None or not getattr(delivery, "accepts_offer", lambda _: False)(offer)):
                return False
            reservation = self._reserve_response_locked("prospective", {
                "voice_route_offer_id": str(offer.offer_id),
                "attention_attempt_id": str(offer.attempt.attempt_id),
            })
            reservation.task_instructions = (
                "Deliver the authorized notification offer identified by voice_route_offer_id. "
                "Preserve its numbers, negation, instructions and uncertainty. Do not redo its reasoning, "
                "invent success, execute actions, or repeat earlier notifications."
            )
        return await self._send_reserved_response(reservation)

    async def _request_normal_response(
        self,
        *,
        turn_generation: int,
        metadata: dict[str, str] | None = None,
        continuation: bool = False,
        task_instructions: str | None = None,
        requested_image_call_ids: tuple[str, ...] = (),
        tool_followup: RealtimeToolFollowup | None = None,
    ) -> bool:
        # Reserve first; synchronization happens outside admission and revalidates at send.
        reservation = None
        async with self._response_admission_lock:
            if (
                not self._closed.is_set()
                and turn_generation == self._turn_generation
                and self._normal_turn_pending
                and not self._user_audio_pending
                and (not self._normal_response_requested or continuation)
            ):
                await self._cancel_response_locked(keep_image_calls=requested_image_call_ids)
                reservation = self._reserve_response_locked("normal_user_turn", metadata or {})
                reservation.task_instructions = task_instructions
                reservation.requested_image_call_ids = requested_image_call_ids
                reservation.tool_followup = tool_followup
                if not continuation:
                    reservation.user_admission_scope = RequestSourceScope.from_session(self)
                self._normal_response_requested = True
        if reservation is None:
            self._trace.event("response_rejected", turn_generation=turn_generation,
                              reason="admission_state", closed=self._closed.is_set(),
                              user_audio_pending=self._user_audio_pending,
                              normal_turn_pending=self._normal_turn_pending,
                              response_requested=self._normal_response_requested)
            self._release_tool_images(requested_image_call_ids)
            if continuation and not self._closed.is_set():
                await self._settle_unstarted_continuation(
                    metadata or {}, reason="continuation_superseded",
                )
            return False
        return await self._send_reserved_response(reservation)

    async def _finish_response_admission(
        self, state: RealtimeResponseState | None, *, response_status: str | None,
        response_used_tools: bool,
    ) -> RealtimeResponseContinuation | None:
        if state is None:
            return None
        async with self._response_admission_lock:
            if self._current_response_id == state.response_id:
                self._current_response_id = self._current_output_item_id = None
            if (
                state.metadata.get("jarvis_request_id") != self._latest_response_request_id
                or state.metadata.get("jarvis_turn_generation") != str(self._turn_generation)
            ):
                return None
            self._latest_response_request_id = None
            if (
                response_status == "failed" and state.delivery_kind == "normal_user_turn"
                and not state.interrupted and not state.dismissed
                and state.metadata.get("jarvis_image_response") == "true"
                and not state.metadata.get("jarvis_failure_notice")
                and self._normal_turn_pending
            ):
                self._tool_followup = None
                return RealtimeResponseContinuation(
                    turn_generation=self._turn_generation,
                    metadata={"jarvis_failure_notice": "image_response_failed"},
                    source_response_id=state.response_id,
                    task_instructions=(
                        "The image-backed response failed. Give Dennis one brief final_answer "
                        "saying you could not read that image and he can try again. "
                        "Do not describe or infer its contents, repeat the original task, "
                        "call tools, or claim that an action succeeded."
                    ),
                )
            if (
                not state.interrupted and not state.dismissed and response_status == "completed"
                and response_used_tools and self._tool_followup is not None
                and self._tool_followup.source_response_id == state.response_id
            ):
                followup = self._tool_followup
                self._tool_followup = None
                return followup
            if state.kind == "normal_user_turn" or response_used_tools:
                self._settle_normal_turn(self._turn_generation)
        return None

    def _is_passive_visual_event(self, payload: dict[str, Any]) -> bool:
        item = payload.get("item")
        item_id = _string_value(payload, "item_id")
        if item_id is None and isinstance(item, dict):
            item_id = _string_value(item, "id")
        return item_id is not None and (
            (self._visual_delivery is not None and self._visual_delivery.is_passive_item(item_id))
            or any(candidate.item_id == item_id for candidate in self._visual_context.items)
        )

    def notify_visual_frame_available(self) -> None:
        if not self._closed.is_set() and self._visual_delivery is not None:
            self._visual_delivery.notify_frame_available()

    def visual_delivery_diagnostics(self) -> dict[str, object]:
        if self._visual_delivery is not None:
            return self._visual_delivery.diagnostics()
        return self._visual_context.to_metadata(now=self._now(), monotonic_now=self._monotonic())

    async def prepare_visual_response(
        self, *, turn_id: str, generation_id: UUID, requested_call_ids: tuple[str, ...] = (),
    ) -> VisualResponseSnapshot | None:
        """Prepare outside admission; the ordered response writer revalidates."""
        if self._visual_delivery is None:
            return None
        return await self._visual_delivery.prepare_response(
            turn_id=turn_id, generation_id=generation_id, requested_call_ids=requested_call_ids,
        )

    def mark_visual_response_submitted(
        self, preparation: VisualResponseSnapshot, *, request_id: str,
    ) -> bool:
        return self._visual_delivery is not None and self._visual_delivery.mark_response_submitted(
            preparation, request_id=request_id,
        )

    def cancel_visual_response_preparation(self, preparation: VisualResponseSnapshot) -> bool:
        return self._visual_delivery is not None and self._visual_delivery.cancel_preparation(preparation)

    async def notify_working_memory_updated(self, revision: int) -> None:
        if self._closed.is_set():
            return
        await self._context_refresh.notify(revision)

    async def notify_owner_working_memory_updated(self, revision: int) -> None:
        del revision
        if self._closed.is_set():
            return
        await self._context_refresh.invalidate()

    async def notify_memory_item_changed(self, event: DomainEvent) -> None:
        if self._closed.is_set() or not isinstance(event.payload, MemoryItemChangedPayload):
            return
        payload = event.payload
        try:
            await self._send_app_event(
                "memory.updated",
                change_id=str(event.event_id),
                changed_at=event.occurred_at.isoformat(),
                affected_memory_ids=[str(payload.memory_item_id)],
                affected_memory_types=(
                    [payload.memory_type] if payload.memory_type is not None else []
                ),
                reason=payload.reason or payload.change_type,
                change_type=payload.change_type,
            )
        except Exception:
            LOGGER.warning(
                "failed to send memory update hint memory_id=%s",
                payload.memory_item_id,
                exc_info=True,
            )

    def _context_reset_notice(self):
        return {"protocol_version":1, "id":self._context_reset_id, "preserve_session":True}

    def _begin_context_rebuild(self):
        if self._closed.is_set() or self._context_revoked:
            return
        self._context_revoked = True
        self._context_reset_id = str(uuid4())
        if self._close_reason == "unknown":
            self._close_reason = "live_context_rebuild_required"
        try:
            # This runs once per retirement, after output has been fenced. Keep
            # the initiating guard distinguishable without logging its text.
            import sys
            error = sys.exception()
            error_type = type(error).__name__ if error is not None else "none"
            if error_type not in {"none", "LiveContextError", "SourceError", "TimeoutError"}:
                error_type = "unrecognized_error"
            code = getattr(error, "code", None)
            if not isinstance(code, str) or code not in {
                "live_context_revoked", "live_context_changed", "live_context_transaction_required",
                "live_context_source_invalid", "live_context_source_limit",
                "live_context_storage_unavailable", "live_context_response_limit",
            }:
                code = "unrecognized_code" if error is not None else "none"
            caller = sys._getframe(1)
            caller_line = caller.f_lineno if caller.f_code.co_filename == __file__ else None
            LOGGER.warning("source_context_retirement_diagnostic cause_type=%s cause_code=%s caller_line=%s",
                           error_type, code, caller_line)
        except Exception:
            # Diagnostic failure cannot prevent or change source retirement.
            pass
        self._context_retirement_task = asyncio.create_task(self._retire_source_context())

    async def _watch_source_context(self):
        try:
            await self._source_context.wait_for_invalidation(
                timeout=self.settings.OPENAI_REALTIME_WRITE_TIMEOUT_SECONDS)
        except asyncio.CancelledError:
            raise
        except Exception:
            self._begin_context_rebuild()
            await self._closed.wait()

    async def _retire_source_context(self):
        input_pending = self._user_audio_pending
        try:
            # Keep the client receive loop alive briefly for the exact clear ACK.
            # Revoked media/turns are already fenced before this notice queues.
            async with asyncio.timeout(2):
                await self._send_app_event("session.context_invalidated",
                    session_id=str(self.context.jarvis_session_id),
                    audio_session_id=self.context.audio_session_id,
                    context_reset=self._context_reset_notice())
                await self._context_reset_ack.wait()
        except (TimeoutError, Exception):
            pass
        finally:
            try:
                if capture := self._context_capture:
                    capture.context_reset(self._context_reset_id,
                        client_cleared=self._context_reset_ack.is_set(), input_was_pending=input_pending)
            except Exception:
                LOGGER.warning("Context reset evidence capture failed")
            finally:
                self._mark_closing("live_context_rebuild_required")

    async def _send_transport_setup(self) -> None:
        """Transport-specific handshake; this does not admit microphone input."""

    async def _send_session_update(self) -> int | None:
        if self._closed.is_set():
            return None
        assistant_context = await self._build_assistant_context()
        if self._closed.is_set():
            return None
        await self._send_complete_session_update(assistant_context)
        self._schedule_working_memory_expiry(assistant_context.working_memory)
        return session_working_memory_revision(assistant_context.working_memory)

    async def _build_assistant_context(self):
        self._trace.event("session_context_build_started")
        recent_messages = await self.repository.get_recent_messages(
            self.context.owner_id,
            self.context.jarvis_session_id,
            limit=self.settings.JARVIS_REALTIME_RECENT_MESSAGES_LIMIT,
        )
        self._trace.event("session_context_messages_loaded")
        assistant_context = await self.assistant_context_builder.build(
            owner_id=self.context.owner_id,
            session=self.context.session,
            recent_messages=recent_messages,
        )
        self._trace.event("session_context_sections_loaded")
        capture = self._context_capture
        if capture is not None and isinstance(capture.repository, ContextRepository):
            async with asyncio.timeout(self.settings.OPENAI_REALTIME_WRITE_TIMEOUT_SECONDS):
                history = await capture.repository.voice_history(self.context.owner_id, self.context.jarvis_session_id)
            assistant_context = recover_voice_context(assistant_context, history)
        self._trace.event("session_context_build_completed")
        return assistant_context


    async def _recover_tool_acknowledgment(self):
        """Finish an unspoken receipt after reconnect without replaying its tool."""
        capture = self._context_capture
        continuity = getattr(self._assistant_context, "voice_continuity", None) or {}
        expected = continuity.get("pending_tool_acknowledgment")
        if not expected or capture is None or not isinstance(capture.repository, ContextRepository):
            return
        try:
            # A newly admitted user turn always takes precedence over recovery.
            if self._closed.is_set() or self._turn_generation != 0 or self._context_revoked:
                return
            async with asyncio.timeout(self.settings.OPENAI_REALTIME_WRITE_TIMEOUT_SECONDS):
                instructions = await capture.repository.claim_tool_acknowledgment(
                    self.context.owner_id, self.context.jarvis_session_id, expected,
                    audio_session_id=self.context.audio_session_id)
            if instructions is None:
                return
            async with self._response_admission_lock:
                if self._closed.is_set() or self._turn_generation != 0 or self._context_revoked:
                    return
                await self._begin_user_turn_locked()
                generation = self._turn_generation
            # No input is admitted or copied into the new capture. Consequently
            # this acknowledgment cannot grant authority to repeat a mutation.
            await self._request_normal_response(turn_generation=generation,
                task_instructions=instructions, metadata={"jarvis_context_recovery": "tool_acknowledgment"})
        except asyncio.CancelledError:
            raise
        except Exception:
            LOGGER.warning("Tool acknowledgment recovery unavailable; accepted operations are not replayed.")
    async def _send_complete_session_update(self, assistant_context) -> None:
        try:
            self._source_context.observe(getattr(assistant_context, "source_payload_ids", ()))
        except LiveContextError:
            self._begin_context_rebuild()
            raise
        self._assistant_context = assistant_context
        tools = self.tool_registry.openai_realtime_tools()
        instructions = build_realtime_instructions(
            context=assistant_context, live_view=self._visual_prompt_metadata(),
            tasks_enabled=any(tool.get("name") == "start_background_task" for tool in tools),
        )
        try:
            await self.openai.send_event(
                build_session_update_event(
                    settings=self.settings, instructions=instructions, tools=tools,
                ),
                **({"write_guard": self._source_context.context_guard}
                   if self._source_context.has_sources else {}),
            )
        except Exception:
            if self._source_context.has_sources:
                self._begin_context_rebuild()
            raise

    async def _refresh_realtime_context(
        self,
        requested_revision: int | None,
    ) -> int | None:
        if self._closed.is_set():
            return None
        await self._realtime_context_ready.wait()
        if self._closed.is_set():
            return None
        assistant_context = await self._build_assistant_context()
        if self._closed.is_set():
            return None
        if not working_memory_context_available(assistant_context.working_memory):
            raise RuntimeError("Working-memory context is temporarily unavailable.")
        applied_revision = session_working_memory_revision(
            assistant_context.working_memory
        )
        if (
            requested_revision is not None
            and (applied_revision is None or applied_revision < requested_revision)
        ):
            return None
        try:
            await self._send_app_event(
                "working_memory.updated",
                session_id=str(self.context.jarvis_session_id),
                **working_memory_app_event_payload(assistant_context.working_memory),
            )
        except Exception:
            LOGGER.warning(
                "failed to send working-memory update hint session_id=%s revision=%s",
                self.context.jarvis_session_id,
                applied_revision,
                exc_info=True,
            )
        if self._closed.is_set():
            return None
        await self._send_complete_session_update(assistant_context)
        if self._closed.is_set():
            return None
        self._schedule_working_memory_expiry(assistant_context.working_memory)
        return applied_revision

    def _schedule_working_memory_expiry(self, payload: dict[str, Any] | None) -> None:
        existing = self._working_memory_expiry_task
        if existing is not None and existing is not asyncio.current_task():
            existing.cancel()
        self._working_memory_expiry_task = None
        expires_at = working_memory_next_expires_at(payload)
        if expires_at is None or self._closed.is_set():
            return
        self._working_memory_expiry_task = asyncio.create_task(
            self._invalidate_working_memory_at_expiry(expires_at)
        )

    async def _invalidate_working_memory_at_expiry(self, expires_at: datetime) -> None:
        try:
            delay = max(0.0, (expires_at - datetime.now(UTC)).total_seconds())
            await asyncio.sleep(delay)
            if not self._closed.is_set():
                await self._context_refresh.invalidate()
        finally:
            if self._working_memory_expiry_task is asyncio.current_task():
                self._working_memory_expiry_task = None

    async def _wait_for_openai_session_ready(self) -> None:
        while not self._closed.is_set():
            payload = await asyncio.wait_for(
                self.openai.receive_event(),
                timeout=OPENAI_SESSION_READY_TIMEOUT_SECONDS,
            )
            event_type = payload.get("type")
            if event_type == "session.updated":
                return
            if event_type == "session.created":
                continue
            if event_type == "error":
                raise RuntimeError(_openai_error_message(payload))

    async def _client_to_openai_loop(self) -> None:
        self._client_receive_task = asyncio.current_task()
        try:
            async for raw_message in self.websocket.iter_text():
                dequeued_at = self._trace.control_clock()
                if self._closed.is_set():
                    return
                try:
                    self._enforce_client_message_size(raw_message)
                    event = parse_client_event(raw_message)
                    with self._trace.control_handler(
                        event, origin="client_control", dequeued_at=dequeued_at,
                    ):
                        await self._handle_client_event(event)
                except ClientEventError as error:
                    await self._send_error("invalid_client_event", str(error), recoverable=True)
                if self._closed.is_set():
                    return
        except WebSocketDisconnect:
            pass
        # Starlette's iter_text() normally consumes WebSocketDisconnect itself.
        self._app_sender.mark_closed()

    async def _handle_client_event(self, event: dict[str, Any]) -> None:
        event_type = event["type"]
        if event_type == "session.context_cleared":
            if (not self._context_revoked or type(event.get("protocol_version")) is not int
                    or event.get("protocol_version") != 1 or event.get("reset_id") != self._context_reset_id
                    or event.get("audio_session_id") != self.context.audio_session_id):
                raise ClientEventError("Context clear acknowledgement does not match this connection.")
            self._context_reset_ack.set()
            return
        if self._context_revoked:
            return
        self._trace.client_event(event, turn_generation=self._turn_generation)

        if event_type == "audio.input.append":
            audio_base64 = self._validated_audio_base64(require_string(event, "audio_base64"))
            async with self._response_admission_lock:
                if self._closed.is_set():
                    return
                if not self._user_audio_pending:
                    await self._begin_user_turn_locked()
                    self._user_audio_pending = True
            await self.openai.send_event(
                {
                    "type": "input_audio_buffer.append",
                    "audio": audio_base64,
                }
            )
            return

        if event_type == "audio.input.commit":
            async with self._response_admission_lock:
                if self._closed.is_set():
                    return
                if not self._user_audio_pending:
                    await self._begin_user_turn_locked()
                # While audio is pending, this flag can represent an early
                # response.create, not a reservation. Release that intent only
                # after commit; new-turn/cancel settlement already clears it.
                response_requested = self._normal_response_requested
                self._normal_response_requested = False
                self._user_audio_pending = False
                generation = self._turn_generation
                request_id = uuid4().hex
                self._audio_commit_request = (request_id, self._turn_generation)
                if capture := getattr(self, "_context_capture", None):
                    capture.admit_openai_commit(request_id, generation)
                self._trace.event("input_commit_admitted", request_id=request_id,
                                  turn_generation=self._turn_generation, segment_id=event.get("segment_id"),
                                  client_event_id=event.get("client_event_id"))
                await self.openai.send_event(
                    {"type": "input_audio_buffer.commit", "event_id": request_id}
                )
            if response_requested:
                self._trace.event("deferred_response_released", turn_generation=generation,
                                  reason="audio_committed")
                await self._request_normal_response(turn_generation=generation)
            return

        if event_type == "response.create":
            async with self._response_admission_lock:
                if self._closed.is_set():
                    return
                if not self._normal_turn_pending:
                    await self._begin_user_turn_locked()
                generation = self._turn_generation
                if self._user_audio_pending:
                    self._normal_response_requested = True
                    self._trace.event("response_deferred", turn_generation=generation,
                                      reason="audio_commit_pending")
                    return
            await self._request_normal_response(turn_generation=generation)
            return

        if event_type == "response.cancel":
            async with self._response_admission_lock:
                self._turn_generation += 1
                self._settle_normal_turn(self._turn_generation)
                await self._cancel_response_locked()
            return

        if event_type == "playback.truncated":
            item_id = require_string(event, "item_id")
            if delivery := self._attention_receipts.get(item_id):
                await delivery.receipt(event, completed=False)
                return
            if item_id.startswith("tts_"):
                raise ClientEventError("Unknown expert playback item.")
            content_index = optional_int(event, "content_index", 0)
            audio_end_ms = optional_int(event, "audio_end_ms", 0)
            if capture := getattr(self, "_context_capture", None):
                capture.playback(item_id, end_ms=audio_end_ms, content_index=content_index)
            await self.openai.send_event(
                {
                    "type": "conversation.item.truncate",
                    "item_id": item_id,
                    "content_index": content_index,
                    "audio_end_ms": audio_end_ms,
                }
            )
            response_id = self._response_id_by_output_item_id.get(item_id)
            if response_id is not None:
                state = self._active_responses.get(response_id)
                if state is not None:
                    state.interrupted = True
                    self._cancel_image_tool_work(response_id=state.response_id)
            return

        if event_type in {"playback.started", "playback.progress", "playback.interrupted", "playback.completed"}:
            item_id = require_string(event, "item_id")
            if (event.get("audio_session_id", self.context.audio_session_id) != self.context.audio_session_id
                    or type(event.get("sample_rate", 24000)) is not int or event.get("sample_rate", 24000) != 24000
                    or type(event.get("protocol_version", 1)) is not int or event.get("protocol_version", 1) != 1
                    or event.get("boundary", "av_audio_data_played_back_lower_bound") != "av_audio_data_played_back_lower_bound"):
                raise ClientEventError("Playback receipt belongs to another session or protocol.")
            if delivery := self._attention_receipts.get(item_id):
                await delivery.receipt(event, completed=event_type == "playback.completed")
                return
            native = self._native_playback.get(item_id) or self._native_playback_history.get(item_id)
            samples = event.get("played_samples")
            if (native is None or type(samples) is not int or not 0 <= samples <= native["samples"]
                    or event.get("response_id") != native["response_id"]
                    or type(event.get("content_index", 0)) is not int or event.get("content_index", 0) != 0
                    or event_type == "playback.completed" and (not native["done"] or samples != native["samples"])):
                raise ClientEventError("Playback receipt does not match this session's output.")
            if samples < native.get("played_samples", 0):
                return  # A reordered historical progress receipt cannot move time backwards.
            native["played_samples"] = samples
            if capture := self._context_capture:
                capture.playback_receipt(item_id, response_id=native["response_id"],
                                         samples=samples, state=event_type.removeprefix("playback."))
            if event_type == "playback.completed":
                self._retire_native_playback(item_id)
            elif event_type == "playback.interrupted":
                async with self._response_admission_lock:
                    if self._current_response_id == native["response_id"]:
                        await self._cancel_response_locked()
            return

        if event_type == "session.close":
            await self.close("client_closed")
            return

        await self._send_error(
            "unsupported_event_type",
            f"Unsupported client event type: {event_type}.",
            recoverable=True,
        )

    async def _openai_to_client_loop(self) -> None:
        self._provider_receive_task = asyncio.current_task()
        try:
            while not self._closed.is_set():
                receive_started_at = self._trace.control_clock()
                payload = await self.openai.receive_event()
                dequeued_at = self._trace.control_clock()
                with self._trace.control_handler(
                    payload, origin="provider_control", dequeued_at=dequeued_at,
                    receive_started_at=receive_started_at,
                ):
                    await self._handle_openai_event(payload)
        except ConnectionClosed as error:
            if self._closed.is_set():
                return
            if isinstance(error, ConnectionClosedOK):
                self._mark_closing("openai_closed")
            else:
                raise

    async def _handle_openai_event(self, payload: dict[str, Any]) -> None:
        if self._closed.is_set():
            return
        event_type = payload.get("type")
        if not isinstance(event_type, str):
            return
        if self._context_revoked and event_type not in {
            "input_audio_buffer.committed", "conversation.item.input_audio_transcription.failed",
            "conversation.item.input_audio_transcription.completed", "input_audio_transcription.completed",
        }:
            return
        if self._visual_delivery is not None and self._visual_delivery.handle_provider_event(payload):
            return

        if event_type in {
            "conversation.item.input_audio_transcription.completed",
            "input_audio_transcription.completed",
            "input_audio_buffer.speech_started",
        } and self._is_passive_visual_event(payload):
            return

        if event_type == "response.created":
            response = payload.get("response")
            if isinstance(response, dict):
                metadata = response.get("metadata")
                metadata = metadata if isinstance(metadata, dict) else {}
                self._trace.event("provider_response_started", response_id=response.get("id"),
                                  request_id=metadata.get("jarvis_request_id"))
                await self._register_response(response)
            return

        if event_type == "input_audio_buffer.committed":
            if (capture := getattr(self, "_context_capture", None)) and (item_id := _string_value(payload, "item_id")):
                capture.openai_committed(item_id)
            return

        if event_type == "conversation.item.input_audio_transcription.failed":
            if (capture := getattr(self, "_context_capture", None)) and (item_id := _string_value(payload, "item_id")):
                capture.openai_transcript(item_id, None)
            return

        if event_type in {"response.output_item.added", "response.output_item.created"}:
            response_id = _string_value(payload, "response_id") or self._current_response_id
            item = payload.get("item")
            self._observe_output_item(
                response_id=response_id, item=item, output_index=payload.get("output_index"),
            )
            return

        if event_type in {"response.output_item.done", "response.output_item.completed"}:
            item = payload.get("item")
            self._observe_output_item(
                response_id=_string_value(payload, "response_id") or self._current_response_id,
                item=item, output_index=payload.get("output_index"),
            )
            if isinstance(item, dict) and item.get("type") == "function_call":
                await self._handle_function_call_item(
                    item,
                    response_id=_string_value(payload, "response_id"),
                )
            return

        if event_type == "response.function_call_arguments.done":
            await self._handle_function_call_item(
                payload,
                response_id=_string_value(payload, "response_id"),
            )
            return

        if event_type in {
            "response.output_audio.delta", "response.output_audio.done",
            "response.output_audio_transcript.delta", "response.output_audio_transcript.done",
        }:
            await self._forward_output_media(payload, event_type=event_type)
            return

        if event_type in {
            "conversation.item.input_audio_transcription.completed",
            "input_audio_transcription.completed",
        }:
            transcript = _string_value(payload, "transcript") or ""
            provider_item_id = _string_value(payload, "item_id")
            if capture := getattr(self, "_context_capture", None):
                # Canonical capture atomically projects its source-bound message.
                # A capture failure must not leave an independent durable copy.
                capture.openai_transcript(provider_item_id, transcript)
            elif transcript.strip():
                # Audio segment admission, not delayed transcription, owns turn state.
                message = await self.repository.create_message(
                    owner_id=self.context.owner_id,
                    session_id=self.context.jarvis_session_id,
                    role="user",
                    modality="audio",
                    content=transcript.strip(),
                    provider_item_id=provider_item_id,
                    metadata={
                        "audio_session_id": self.context.audio_session_id,
                        "provider_event_type": event_type,
                    },
                )
                await self._attach_message_to_experience(message)
            await self._send_app_event("user.transcript.done", text=transcript)
            return

        if event_type == "response.done":
            response = payload.get("response")
            if isinstance(response, dict):
                self._trace.event("provider_response_completed", response_id=response.get("id"),
                                  status=response.get("status"))
            response_id = (
                _string_value(response, "id")
                if isinstance(response, dict)
                else self._current_response_id
            )
            if response_id is None:
                await self._send_app_event("response.done", response_id=None)
                return
            state = self._active_responses.get(response_id)
            if response_id in self._retired_response_ids or (
                state is not None and state.completed_response is not None
            ):
                return
            if isinstance(response, dict) and isinstance(response.get("output"), list):
                for index, item in enumerate(response["output"]):
                    self._observe_output_item(response_id=response_id, item=item, output_index=index)
                    if isinstance(item, dict) and item.get("type") == "function_call":
                        await self._handle_function_call_item(item, response_id=response_id)
            completion = {"id": response_id, "status": _string_value(response, "status")}
            if state is not None:
                # Keep only terminal identity/status, not a provider output/media envelope.
                state.completed_response = completion
                self._retain_cancelled_empty_request(state, response)
                if self._visual_delivery is not None:
                    request_id = state.metadata.get("jarvis_request_id")
                    if isinstance(request_id, str):
                        self._visual_delivery.settle_response(request_id=request_id, response_id=response_id)
                if completion["status"] != "completed" or state.interrupted or state.dismissed:
                    self._cancel_image_tool_work(response_id=response_id)
                if state.pending_tool_call_ids:
                    return
            if self._voice_route_runtime is not None:
                self._schedule_provider_completion(response_id, completion)
            else:
                await self._finish_provider_response(response_id, completion)
            return

        if event_type == "error":
            error = payload.get("error")
            if await self._consume_cancel_race(error):
                return
            LOGGER.warning("%s", _openai_error_message(payload))
            request_id = _string_value(error, "event_id")
            if (capture := getattr(self, "_context_capture", None)) and request_id is not None:
                capture.openai_commit_rejected(request_id)
            if request_id is not None and self._visual_delivery is not None:
                self._visual_delivery.settle_response(request_id=request_id)
            async with self._response_admission_lock:
                rejected = self._unacknowledged_response_requests.pop(request_id, None)
                pending = self._pending_response
                if pending is not None and request_id == pending.request_id:
                    self._release_response_reservation(pending)
                if rejected is not None:
                    # Fence initial rejections as well as aborted continuations.
                    # Neither can be reopened by a contradictory late ACK.
                    self._aborted_continuation_requests.add(rejected.request_id)
                    self._response_controls.reject(rejected.request_id)
                commit = self._audio_commit_request
                if commit is not None and request_id == commit[0]:
                    self._settle_normal_turn(commit[1])
                    self._audio_commit_request = None
            if rejected is not None:
                self._trace.event(
                    "response_rejected", request_id=rejected.request_id,
                    turn_generation=rejected.turn_generation, kind=rejected.kind,
                    reason="provider_rejected",
                )
                await self._settle_rejected_continuation(
                    {
                        **rejected.metadata,
                        "jarvis_delivery_kind": (
                            rejected.metadata.get("jarvis_delivery_kind", "")
                            if rejected.is_delivery_continuation else rejected.kind
                        ),
                    },
                    request_id=rejected.request_id,
                    reason="continuation_rejected" if rejected.is_delivery_continuation else "response_rejected",
                )
            await self._send_error(
                "openai_realtime_error",
                _openai_error_message(payload),
                recoverable=True,
            )

    async def _settle_rejected_continuation(self, *args, **kwargs) -> None:
        if self._voice_route_runtime is None:
            await self._settle_unstarted_continuation(*args, **kwargs)
            return
        if self._closed.is_set():
            return  # Close reconciles remaining canonical pending deliveries.
        if len(self._continuation_settlement_tasks) >= 32:
            self._mark_closing("response_rejection_settlement_queue_exhausted")
            return
        task = asyncio.create_task(
            self._run_rejected_continuation_settlement(*args, **kwargs),
            name="realtime-rejected-response-settlement",
        )
        self._continuation_settlement_tasks.add(task)
        task.add_done_callback(self._rejected_continuation_settlement_finished)

    async def _run_rejected_continuation_settlement(self, *args, **kwargs) -> None:
        # Use the existing bounded external-write budget. A timeout leaves
        # canonical pending delivery state for the normal close reconciliation;
        # it never retries a provider response or invents successful delivery.
        async with asyncio.timeout(self.settings.OPENAI_REALTIME_WRITE_TIMEOUT_SECONDS):
            await self._settle_unstarted_continuation(*args, **kwargs)

    def _rejected_continuation_settlement_finished(self, task) -> None:
        self._continuation_settlement_tasks.discard(task)
        if not task.cancelled() and task.exception() is not None:
            self._mark_closing("response_rejection_settlement_failed")

    def _retain_cancelled_empty_request(self, state, response) -> None:
        """Reoffer an unfulfilled request, never replay a possibly executed operation."""
        scope, generation = state.user_admission_scope, state.user_admission_generation
        capture = getattr(self, "_context_capture", None)
        if (capture is None or not hasattr(capture, "retain_cancelled_empty")
                or self._closed.is_set() or self._context_revoked
                or scope is None or type(generation) is not int
                or generation >= self._turn_generation
                or not scope.same(RequestSourceScope.from_session(self))
                or state.delivery_kind != "normal_user_turn" or not state.interrupted or state.dismissed
                or response.get("status") != "cancelled" or response.get("output") != []
                or state.output_items or state.pending_tool_call_ids
                or state.response_id in self._responses_with_tool_calls
                or state.tool_followup.receipts or state.requested_image_call_ids):
            return
        # completed_response has already sealed dispatch. Sticky admission history
        # is still present here, including tools that returned or became uncertain.
        # A duplicate terminal event cannot reach this boundary a second time.
        capture.retain_cancelled_empty(generation, scope, state.request_sources)

    def _schedule_provider_completion(self, response_id, response):
        if (self._closed.is_set() or response_id in self._retired_response_ids
                or response_id in self._provider_completion_tasks):
            return
        if len(self._provider_completion_tasks) >= 32:
            self._mark_closing("provider_completion_queue_exhausted")
            return
        task = asyncio.create_task(self._run_provider_completion(response_id, response),
                                   name="realtime-response-completion")
        self._provider_completion_tasks[response_id] = task
        task.add_done_callback(lambda finished: self._provider_completion_finished(response_id, finished))

    async def _run_provider_completion(self, response_id, response):
        async with self._provider_completion_capacity:
            if not self._closed.is_set():
                await self._finish_provider_response(response_id, response)

    def _provider_completion_finished(self, response_id, task):
        if self._provider_completion_tasks.get(response_id) is task:
            self._provider_completion_tasks.pop(response_id, None)
        if not task.cancelled() and task.exception() is not None:
            LOGGER.warning("Provider response settlement failed response_id=%s", response_id)
            if not self._closed.is_set():
                self._mark_closing("provider_completion_failed")

    async def _finish_provider_response(self, response_id: str, response: dict[str, Any]) -> None:
        """The existing terminal path, reached only after all accepted outputs settle."""
        if self._closed.is_set() or response_id in self._retired_response_ids:
            return
        state = self._active_responses.get(response_id)
        if state is not None:
            if state.completion_started or state.pending_tool_call_ids:
                return
            state.completion_started = True
        response_used_tools = response_id in self._responses_with_tool_calls
        if capture := getattr(self, "_context_capture", None):
            capture.finish_response(response_id, interrupted=state is not None and state.interrupted,
                                    completed=response.get("status") == "completed")
        if not (state is not None and state.interrupted):
            await self._persist_assistant_transcript(response_id=response_id)
        state = self._active_responses.pop(response_id, None)
        self._retired_response_ids.add(response_id)
        followup = await self._finish_response_admission(
            state, response_status=_string_value(response, "status"),
            response_used_tools=response_used_tools,
        )
        completed = _string_value(response, "status") == "completed"
        if (
            _string_value(response, "status") == "failed"
            and state is not None and not state.interrupted and not state.dismissed
        ):
            await self._send_error(
                "response_failed", "The response could not be completed. Please try again.",
                recoverable=True,
            )
        forwarded_answer = (
            state is not None and not state.interrupted and completed and state.has_forwarded_answer
        )
        if (
            state is not None and not state.interrupted and completed
            and followup is None and not response_used_tools and not forwarded_answer
        ):
            await self._send_error(
                "no_final_answer", "The response ended without a final answer. Please try again.",
                recoverable=True,
            )
        if state is not None:
            LOGGER.debug(
                "realtime output completed response_id=%s items=%s", response_id,
                [item.as_metadata() for item in state.ordered_output_items],
            )
        settle_delivery = (
            state is not None and followup is None
            and state.metadata.get("jarvis_request_id") not in self._aborted_continuation_requests
        )
        if state is not None and state.delivery_kind == "visual_interrupt" and settle_delivery:
            interrupt_id = state.metadata.get("visual_interrupt_id")
            if isinstance(interrupt_id, str) and interrupt_id:
                if forwarded_answer:
                    await self._persist_visual_interrupt_response_id(
                        interrupt_id=interrupt_id, response_id=response_id,
                    )
                await self._send_speech_notification(
                    "visual.interrupt.done", interrupt_id=interrupt_id, response_id=response_id,
                    forwarded_answer=forwarded_answer,
                    dismissed=state.dismissed and completed and not state.interrupted,
                )
        if state is not None and state.delivery_kind == "reminder" and settle_delivery:
            await self._finish_reminder_response(state=state, response=response, response_id=response_id)
        if state is not None and state.delivery_kind == "prospective" and settle_delivery:
            await self._finish_prospective_response(state=state, response=response, response_id=response_id)
        self._finish_tool_response(
            response_id=response_id,
            continuing=followup is not None and not followup.metadata.get("jarvis_failure_notice"),
        )
        await self._send_app_event("response.done", response_id=response_id)
        if followup is not None:
            await self._request_normal_response(
                turn_generation=followup.turn_generation, metadata=followup.metadata,
                continuation=True, task_instructions=followup.task_instructions,
                requested_image_call_ids=followup.requested_image_call_ids,
                tool_followup=followup.tool_followup,
            )

    def _output_notice(self, reason: str) -> None:
        # Fixed reason codes, once per socket: no raw phase values or speech.
        if reason not in self._output_policy_notices:
            self._output_policy_notices.add(reason)
            LOGGER.warning("realtime output policy notice reason=%s", reason)

    def _output_item(
        self, state: RealtimeResponseState, *, item_id: str | None,
        output_index: Any = None,
    ) -> RealtimeOutputItemState:
        key = item_id or ""
        item = state.output_items.get(key)
        if item is None:
            item = RealtimeOutputItemState(item_id=item_id)
            state.output_items[key] = item
        if type(output_index) is int and output_index >= 0 and item.output_index is None:
            item.output_index = output_index
        return item

    def _observe_output_item(
        self, *, response_id: str | None, item: Any, output_index: Any = None,
    ) -> None:
        state = self._active_responses.get(response_id) if response_id else None
        if state is None or state.interrupted or state.completed_response is not None or not isinstance(item, dict):
            return
        item_id = _string_value(item, "id")
        if item_id is None:
            return
        self._response_id_by_output_item_id[item_id] = state.response_id
        if item.get("type") != "message":
            return
        state.output_item_id = item_id
        if response_id == self._current_response_id:
            self._current_output_item_id = item_id
        output = self._output_item(state, item_id=item_id, output_index=output_index)
        notice = output.observe_phase(item.get("phase"))
        if notice is not None:
            self._output_notice(notice)

    async def _forward_output_media(self, payload: dict[str, Any], *, event_type: str) -> None:
        item_id = _string_value(payload, "item_id")
        response_id = (
            _string_value(payload, "response_id")
            or self._response_id_by_output_item_id.get(item_id)
            or self._current_response_id
        )
        if response_id in self._retired_response_ids:
            return
        state = self._active_responses.get(response_id) if response_id else None
        if self._source_context.has_sources and state is None:
            # Unknown/early output has no accepted response-source binding.
            # Do not forward it or create an untraceable canonical copy.
            return
        attempt = self._prospective_attempt_for(state.metadata) if state is not None else None
        if attempt is not None and attempt.revoked:
            state.interrupted = True
        if state is not None and (state.interrupted or state.dismissed or state.completed_response is not None):
            return
        item_id = item_id or (state.output_item_id if state else None)
        output = (
            self._output_item(state, item_id=item_id, output_index=payload.get("output_index"))
            if state is not None else RealtimeOutputItemState(item_id=item_id)
        )
        if state is not None and item_id is not None:
            self._response_id_by_output_item_id[item_id] = state.response_id
        audio = event_type.startswith("response.output_audio.")
        done = event_type.endswith(".done")
        if (output.audio_done if audio else output.transcript_done):
            return
        if done:
            if audio:
                output.audio_done = True
            else:
                output.transcript_done = True
        if output.phase is None:
            self._output_notice("missing_phase_legacy_passthrough")
        if not output.record_generated():
            return

        if audio:
            delta = _string_value(payload, "delta") or ""
            sent = await self._send_app_event(
                "audio.output.done" if done else "audio.output.delta",
                item_id=item_id, response_id=response_id,
                **({} if done else {"audio_base64": delta}),
            )
            text = None
        else:
            text = (
                (_string_value(payload, "transcript") or output.transcript) if done
                else (_string_value(payload, "delta") or "")
            )
            sent = await self._send_app_event(
                "transcript.done" if done else "transcript.delta",
                role="assistant", text=text, response_id=response_id,
            )
        # The sender retires a departed peer without raising. Do not claim that
        # its discarded writes were forwarded (let alone played or heard).
        forwarded = sent is not False and not self._closed.is_set() and not self._app_sender.closed
        if capture := getattr(self, "_context_capture", None):
            capture.observe_output({"type": ("audio.output." if audio else "transcript.") + ("done" if done else "delta"),
                "item_id": item_id, "response_id": response_id,
                **({"audio_base64": delta} if audio else {"text": text})}, forwarded=forwarded)
        if not forwarded:
            return
        output.record_forwarded(audio_chunk=audio and not done and bool(delta))
        if text is not None:
            output.transcript = text if done else output.transcript + text

    async def _register_response(self, response: dict[str, Any]) -> None:
        response_id = _string_value(response, "id")
        if (
            response_id is None or response_id in self._active_responses
            or response_id in self._retired_response_ids
        ):
            return
        metadata = response.get("metadata")
        kind = "normal_user_turn"
        if isinstance(metadata, dict) and metadata.get("jarvis_kind") == "visual_interrupt":
            kind = "visual_interrupt"
        elif isinstance(metadata, dict) and metadata.get("jarvis_kind") == "reminder":
            kind = "reminder"
        elif isinstance(metadata, dict) and metadata.get("jarvis_kind") == "prospective":
            kind = "prospective"
        response_metadata = dict(metadata) if isinstance(metadata, dict) else {}
        tool_chain_id = response_metadata.get("jarvis_tool_chain_id")
        if not isinstance(tool_chain_id, str) or not tool_chain_id:
            tool_chain_id = response_id
        response_metadata["jarvis_tool_chain_id"] = tool_chain_id
        self._tool_chain_by_response_id[response_id] = tool_chain_id
        self._tool_budgets.setdefault(
            tool_chain_id,
            ToolCallBudget(
                max_total_calls=self.settings.JARVIS_MAX_TOOL_CALLS_PER_RESPONSE,
                max_repeated_calls=self.settings.JARVIS_MAX_REPEATED_TOOL_CALLS_PER_RESPONSE,
            ),
        )
        state = RealtimeResponseState(
            response_id=response_id,
            kind=kind,  # type: ignore[arg-type]
            metadata=response_metadata,
        )
        async with self._response_admission_lock:
            request_id = response_metadata.get("jarvis_request_id")
            local_request = self._unacknowledged_response_requests.pop(request_id, None)
            if local_request is not None:
                state.task_instructions = local_request.task_instructions
                state.request_sources = local_request.request_sources
                if (local_request.submitted and local_request.kind == "normal_user_turn"
                        and not local_request.is_delivery_continuation
                        and local_request.tool_followup is None):
                    state.user_admission_scope = local_request.user_admission_scope
                    state.user_admission_generation = local_request.turn_generation
                if local_request.tool_followup is not None:
                    state.tool_followup = local_request.tool_followup
            if capture := getattr(self, "_context_capture", None):
                generation = (local_request.turn_generation if local_request is not None
                    and local_request.metadata.get("jarvis_delivery_kind", local_request.kind) == "normal_user_turn" else None)
                source_payload_id = self._source_context.source_for(request_id)
                capture.bind_openai_response(response_id, generation, **(
                    {"source_payload_id": source_payload_id} if source_payload_id is not None else {}))
            state.interrupted = (
                self._closed.is_set()
                or request_id is None
                or request_id != self._latest_response_request_id
                or request_id in self._aborted_continuation_requests
                or (
                    kind == "normal_user_turn"
                    and response_metadata.get("jarvis_turn_generation") != str(self._turn_generation)
                )
            )
            if local_request is not None and local_request.submitted and not state.interrupted:
                state.source_snapshot = self._source_context.snapshot_for(request_id)
            self._active_responses[response_id] = state
            pending = self._pending_response
            if pending is not None and request_id == pending.request_id:
                self._pending_response = None
                self._cancel_response_start_timer()
            if not state.interrupted:
                self._current_response_id = response_id
                self._current_output_item_id = None
            else:
                # Late acknowledgement cannot take over a newer turn.
                with suppress(Exception):
                    await self._send_response_cancel_locked(
                        response_id=response_id,
                        response_request_id=request_id if isinstance(request_id, str) else None,
                    )
        if self._visual_delivery is not None and isinstance(request_id, str):
            self._visual_delivery.response_created(request_id=request_id, response_id=response_id)
        if request_id in self._aborted_continuation_requests:
            # A contradictory late ACK still gets cancelled/settled, but cannot
            # reopen a delivery already failed before this response existed.
            return
        if state.delivery_kind == "visual_interrupt":
            interrupt_id = state.metadata.get("visual_interrupt_id")
            if isinstance(interrupt_id, str) and interrupt_id:
                await self._send_speech_notification(
                    "visual.interrupt.response_created",
                    interrupt_id=interrupt_id,
                    response_id=response_id,
                )
        elif state.delivery_kind == "reminder":
            reminder_id = state.metadata.get("reminder_id")
            reminder_event_id = state.metadata.get("reminder_event_id")
            if isinstance(reminder_event_id, str) and reminder_event_id:
                await self._persist_reminder_response_id(
                    reminder_event_id=reminder_event_id,
                    response_id=response_id,
                )
            await self._send_speech_notification(
                "reminder.response_created",
                reminder_id=reminder_id,
                reminder_event_id=reminder_event_id,
                response_id=response_id,
            )
        elif state.delivery_kind == "prospective":
            await self._send_speech_notification(
                "prospective.response_created",
                prospective_event_id=state.metadata.get("prospective_event_id"),
                subject_type=state.metadata.get("prospective_subject_type"),
                subject_id=state.metadata.get("prospective_subject_id"),
                response_id=response_id,
            )

    async def _handle_function_call_item(
        self,
        item: dict[str, Any],
        *,
        response_id: str | None,
    ) -> None:
        call_id = _string_value(item, "call_id")
        name = _string_value(item, "name")
        arguments_text = _string_value(item, "arguments") or "{}"
        if call_id is None or name is None:
            return
        response_id = response_id or self._current_response_id
        source_state = self._active_responses.get(response_id) if response_id else None
        if source_state is None or source_state.interrupted or source_state.completed_response is not None:
            return
        generation = self._turn_generation
        if call_id in self._handled_tool_call_ids:
            return
        self._handled_tool_call_ids.add(call_id)
        if (capture := getattr(self, "_context_capture", None)) and source_state.request_sources is None:
            capture.bind_openai_call(call_id, source_state.response_id)
        response_id = response_id or self._current_response_id
        tool_chain_id = self._tool_chain_by_response_id.get(response_id) if response_id else None
        if tool_chain_id is None:
            tool_chain_id = response_id or call_id
        budget = self._tool_budgets.setdefault(
            tool_chain_id,
            ToolCallBudget(
                max_total_calls=self.settings.JARVIS_MAX_TOOL_CALLS_PER_RESPONSE,
                max_repeated_calls=self.settings.JARVIS_MAX_REPEATED_TOOL_CALLS_PER_RESPONSE,
            ),
        )
        if response_id is not None:
            self._responses_with_tool_calls.add(response_id)
        source_state.pending_tool_call_ids.add(call_id)
        definition = self.tool_registry.get(name)
        if definition is not None and definition.image_resolver is not None:
            request = ToolImageRequest(
                connection_epoch=self._visual_context.connection.connection_epoch,
                provider_call_id=call_id, source_response_id=source_state.response_id,
                turn_generation=generation, tool_chain_id=tool_chain_id,
            )
            self._image_tool_requests[call_id] = request
            task = asyncio.create_task(self._execute_function_call(
                name=name, arguments_text=arguments_text, call_id=call_id, generation=generation,
                source_state=source_state, tool_chain_id=tool_chain_id, budget=budget,
                image_request=request, image_resolver=definition.image_resolver,
            ))
            self._image_tool_tasks[call_id] = task
            task.add_done_callback(lambda finished: self._image_tool_finished(request, source_state, finished))
            return
        if self._voice_route_runtime is not None:
            # Bound owned work; never await a database/tool/provider operation
            # in the provider event reader. Preserve order within each chain.
            if len(self._ordinary_tool_tasks) >= 32:
                self._tool_call_settled(source_state, call_id)
                self._mark_closing("ordinary_tool_queue_exhausted")
                return
            chain_lock = self._ordinary_tool_locks.setdefault(tool_chain_id, asyncio.Lock())
            task = asyncio.create_task(self._run_ordinary_tool(
                chain_lock=chain_lock, name=name, arguments_text=arguments_text,
                call_id=call_id, generation=generation, source_state=source_state,
                tool_chain_id=tool_chain_id, budget=budget,
            ), name="realtime-ordinary-tool")
            self._ordinary_tool_tasks[call_id] = task
            self._ordinary_tool_chains[call_id] = tool_chain_id
            task.add_done_callback(lambda finished: self._ordinary_tool_finished(call_id, source_state, finished))
            return
        try:
            await self._execute_function_call(
                name=name, arguments_text=arguments_text, call_id=call_id, generation=generation,
                source_state=source_state, tool_chain_id=tool_chain_id, budget=budget,
            )
        finally:
            self._tool_call_settled(source_state, call_id)

    async def _run_ordinary_tool(self, *, chain_lock, **work):
        async with chain_lock, self._ordinary_tool_capacity:
            source = work["source_state"]
            if (self._closed.is_set() or work["generation"] != self._turn_generation
                    or source.interrupted or source.dismissed
                    or (source.completed_response is not None
                        and source.completed_response.get("status") != "completed")):
                return
            # Once admitted, canonical tool execution owns effects and receipts.
            # Barge-in suppresses stale voice output; it does not replay/cancel
            # an already-running mutation or a durable background task.
            await self._execute_function_call(**work)

    def _ordinary_tool_finished(self, call_id, source, task):
        self._ordinary_tool_tasks.pop(call_id, None)
        chain = self._ordinary_tool_chains.pop(call_id, None)
        if chain not in self._ordinary_tool_chains.values():
            self._ordinary_tool_locks.pop(chain, None)
        if not task.cancelled() and task.exception() is not None:
            source.interrupted = True
            if not self._closed.is_set():
                LOGGER.warning("Ordinary tool worker failed call_id=%s", call_id)
                self._mark_closing("ordinary_tool_worker_failed")
        self._tool_call_settled(source, call_id)

    async def _select_tool_result(
        self, *, name: str, arguments_text: str, call_id: str,
        source_state: RealtimeResponseState, budget: ToolCallBudget,
    ) -> dict[str, Any]:

        is_control = self._visual_interrupt_control_registry.get(name) is not None
        offer = source_state.request_sources
        selected = None
        selection_present = False

        def selection_trace(stage, *, code=None, outcome=None, result=None):
            record_request_source(self, stage, state=source_state, offer=offer,
                                  selected=selected, call_id=call_id, name=name,
                                  selection_present=selection_present, code=code, outcome=outcome,
                                  result=result)

        try:
            parsed_arguments = json.loads(arguments_text)
            if is_control and not isinstance(parsed_arguments, dict):
                raise ValueError("Response control arguments must be an object.")
            arguments = parsed_arguments if isinstance(parsed_arguments, dict) else {}
            if not is_control:
                selection_present = REQUEST_SOURCE_ARGUMENT in arguments
                selection = arguments.pop(REQUEST_SOURCE_ARGUMENT, None)
                if offer is not None:
                    capture = self._context_capture
                    if (capture is None or not offer.current(self, capture)
                            or source_state.interrupted or source_state.dismissed):
                        raise source_error()
                    selected = offer.select(selection)
                    capture.bind_selected_call(call_id, selected)
                elif selection is not None:
                    raise source_error("tool_request_source_invalid")
        except TaskError as error:
            selection_trace("selection", code=error.code, outcome="rejected_before_executor")
            return {"status": "failed", "error": {"code": error.code,
                    "message": "The selected user request is missing, changed, or unavailable; no tool was invoked."}}
        except ValueError:
            selection_trace("selection", code="invalid_tool_arguments", outcome="rejected_before_executor")
            arguments = {}
            result = {
                "status": "failed",
                "error": {
                    "code": "invalid_tool_arguments",
                    "message": "Tool arguments must be a JSON object.",
                },
            }
        else:
            selection_trace("selection", outcome=("selected_hot_cache" if selected is not None else
                                                   "control_path" if is_control else "single_request_path"))
            if source_state.dismissed:
                result = {
                    "status": "failed",
                    "error": {"code": "response_dismissed", "message": "This response has ended."},
                }
            elif is_control and source_state.delivery_kind != "visual_interrupt":
                result = {
                    "status": "failed",
                    "error": {
                        "code": "tool_unavailable_for_response",
                        "message": "Dismissal is only available for unsolicited visual candidates.",
                    },
                }
            else:
                executor = self._visual_interrupt_control_executor if is_control else self.tool_executor
                try:
                    result = await executor.execute(
                        name,
                        arguments,
                        budget=budget,
                        provider_call_id=call_id,
                    )
                except asyncio.CancelledError:
                    selection_trace("executor_return", outcome="cancelled")
                    raise
                except Exception:
                    selection_trace("executor_return", outcome="exception")
                    raise
                selection_trace("executor_return", result=result)
                if is_control and result.get("status") == "success":
                    source_state.dismissed = True
                    self._cancel_image_tool_work(response_id=source_state.response_id)
        return result

    def _bind_tool_followup_request(self, call_id, record) -> None:
        """Reuse loaded canonical evidence for a guarded, transient quotation."""
        if record is None or self._closed.is_set() or self._context_revoked:
            return
        matches = [state for state in self._active_responses.values()
                   if call_id in state.pending_tool_call_ids and not state.interrupted
                   and not state.dismissed and state.delivery_kind == "normal_user_turn"
                   and state.metadata.get("jarvis_turn_generation") == str(self._turn_generation)]
        if len(matches) != 1:
            return
        request = RealtimeToolRequest.from_record(record, owner_id=self.context.owner_id,
            session_id=self.context.jarvis_session_id, turn_generation=self._turn_generation,
            source_namespace=self._context_capture.namespace)
        if request is None:
            return
        state = matches[0]
        prior = state.tool_followup.request
        if prior is not None and prior != request:
            if state.request_sources is None:
                state.tool_followup = replace(state.tool_followup, binding_conflict=True)
                raise LiveContextError("live_context_changed")
            additional = state.tool_followup.additional_requests
            if request not in additional:
                if len(additional) >= 4:
                    raise LiveContextError("live_context_source_limit")
                state.tool_followup = replace(state.tool_followup, additional_requests=(*additional, request))
        # This original is already present as the provider's admitted input.
        # Enroll it in the existing final source guard, not a new DB preparation.
        self._source_context.observe((request.payload_id,))
        state.tool_followup = replace(state.tool_followup, request=prior or request)
        if state.request_sources is not None:
            current_id = self._context_capture._generations.get(self._turn_generation)
            current_record = self._context_capture.cached_request(current_id)
            current = RealtimeToolRequest.from_record(current_record, owner_id=self.context.owner_id,
                session_id=self.context.jarvis_session_id, turn_generation=self._turn_generation,
                source_namespace=self._context_capture.namespace)
            if current is not None:
                self._source_context.observe((current.payload_id,))
            calls = state.tool_followup.call_request_ids
            if not any(key == call_id for key, _ in calls) and len(calls) < 16:
                calls = (*calls, (call_id, request.event_id))
            state.tool_followup = replace(state.tool_followup, conversation_request=current,
                                          call_request_ids=calls)

    async def _execute_function_call(
        self, *, name: str, arguments_text: str, call_id: str, generation: int,
        source_state: RealtimeResponseState, tool_chain_id: str, budget: ToolCallBudget,
        image_request: ToolImageRequest | None = None, image_resolver: ImageResolver | None = None,
    ) -> None:
        def eligible() -> bool:
            return (
                not self._closed.is_set() and generation == self._turn_generation
                and not source_state.interrupted and not source_state.dismissed
                and (source_state.completed_response is None or source_state.completed_response["status"] == "completed")
                and (image_request is None or (
                    not image_request.cancelled
                    and image_request.connection_epoch == self._visual_context.connection.connection_epoch
                ))
            )

        if image_request is not None:
            image_request.started = True
        elif self._voice_route_runtime is not None and not eligible():
            return
        try:
            if image_request is not None and not eligible():
                result = image_unavailable("request_cancelled", "The image request has ended.")
            else:
                result = await self._select_tool_result(
                    name=name, arguments_text=arguments_text, call_id=call_id,
                    source_state=source_state, budget=budget,
                )
            if image_request is None or eligible():
                await self._notify_tool_result(name, result)
            if image_request is not None and image_resolver is not None:
                result = await resolve_and_deliver_tool_image(
                    result, resolver=image_resolver, coordinator=self._visual_delivery,
                    request=image_request, is_request_active=eligible, monotonic=self._monotonic,
                )
        except asyncio.CancelledError:
            if image_request is None:
                raise
            image_request.cancelled = True
            result = image_unavailable("request_cancelled", "The image request has ended.")

        if image_request is not None:
            result = await self._write_image_tool_output(image_request, result, eligible)
            if result is None:
                return
            if result.get("image_status") == "attached" and eligible():
                source_state.requested_image_call_ids.append(call_id)
        else:
            current = eligible if self._voice_route_runtime is not None else None
            if not await self._write_tool_output(call_id, result, is_current=current):
                return

        async with self._response_admission_lock:
            if not self._closed.is_set() and generation == self._turn_generation and not source_state.interrupted:
                if source_state.dismissed:
                    self._tool_followup = None
                else:
                    self._normal_turn_pending = True
                    if source_state.delivery_kind == "normal_user_turn":
                        try:
                            receipt = RealtimeToolReceipt.from_result(call_id, name, result)
                            source_state.tool_followup = source_state.tool_followup.append(receipt)
                        except ValueError:
                            source_state.tool_followup = replace(source_state.tool_followup,
                                omitted_receipts=source_state.tool_followup.omitted_receipts + 1)
                        source_state.tool_followup = replace(
                            source_state.tool_followup, source_offer=source_state.request_sources)
                    self._tool_followup = RealtimeResponseContinuation(
                        turn_generation=generation, source_response_id=source_state.response_id,
                        metadata={
                            "tool_name": name, "jarvis_tool_chain_id": tool_chain_id,
                            **source_state.delivery_continuation_metadata(),
                        },
                        task_instructions=source_state.task_instructions,
                        requested_image_call_ids=tuple(source_state.requested_image_call_ids),
                        tool_followup=(source_state.tool_followup
                            if source_state.delivery_kind == "normal_user_turn" else None),
                    )

    async def _notify_tool_result(self, name: str, result: dict[str, Any]) -> None:

        if name in {"create_reminder", "complete_reminder", "delete_reminder"} and result.get("status") == "success":
            await self._send_app_event("reminders.changed", tool_name=name)
        if name == "set_behavior_instructions" and result.get("status") == "success":
            await self._send_session_update()

    async def _write_tool_output(self, call_id, result, *, is_current=None):
        if is_current is not None and not is_current():
            return False
        if not isinstance(result, SourceBoundToolResult):
            if is_current is not None:
                return await self.openai.send_event(
                    lambda: self._tool_output_event(call_id, result) if is_current() else None)
            return await self.openai.send_event(self._tool_output_event(call_id, result))
        if is_current is not None:
            from contextlib import asynccontextmanager

            @asynccontextmanager
            async def source_guard():
                if not is_current():
                    yield
                    return
                async with self._source_context.context_guard(extra_sources=(result.payload_id,)):
                    yield

            def guarded_payload():
                if self._closed.is_set() or self._context_revoked or not is_current():
                    return None
                self._source_context.observe((result.payload_id,))
                return self._tool_output_event(call_id, result)

            try:
                return await self.openai.send_event(guarded_payload, write_guard=source_guard)
            except Exception:
                if is_current():
                    self._begin_context_rebuild()
                return False
        try:
            self._source_context.observe((result.payload_id,))
            def at_write():
                if (self._closed.is_set() or self._context_revoked
                        or (is_current is not None and not is_current())):
                    return None
                return self._tool_output_event(call_id, result)
            return await self.openai.send_event(at_write, write_guard=self._source_context.context_guard)
        except Exception:
            self._begin_context_rebuild()
            return False

    @staticmethod
    def _tool_output_event(call_id: str, result: dict[str, Any]) -> dict[str, Any]:
        return {
            "type": "conversation.item.create",
            "item": {
                "type": "function_call_output", "call_id": call_id,
                "output": json.dumps(result, ensure_ascii=True, separators=(",", ":")),
            },
        }

    def _capture_image_delivery(self, reference, state, delivery_id):
        if reference is not None and reference.source_payload_ids and state in {
            "socket_written", "provider_acknowledged", "write_outcome_unknown",
        }:
            try:
                self._source_context.observe(reference.source_payload_ids)
            except LiveContextError:
                self._begin_context_rebuild()
        if self._context_capture is not None:
            self._context_capture.image_delivery(reference, state, delivery_id)

    async def _write_image_tool_output(
        self, request: ToolImageRequest, result: dict[str, Any], eligible: Callable[[], bool],
    ) -> dict[str, Any] | None:
        started = False
        effective = result
        sources = getattr(self, "_source_context", None)
        roots = image_source_payload_ids(result, owner_id=self.context.owner_id)
        if roots:
            if sources is None:
                raise LiveContextError("live_context_storage_unavailable")
            sources.observe(roots)
        guard = {"write_guard": sources.context_guard} if sources is not None and sources.has_sources else {}

        def payload_at_write() -> dict[str, Any] | None:
            nonlocal started, effective
            if self._closed.is_set():
                return None
            if not eligible():
                effective = image_unavailable("request_cancelled", "The image request has ended.")
            effective = durable_tool_payload(effective, status=str(effective.get("status", "failed")))
            event = self._tool_output_event(request.provider_call_id, effective)
            started = True
            return event

        try:
            sent = await self.openai.send_event(payload_at_write, **guard)
        except LiveContextError:
            self._begin_context_rebuild()
            return None
        except asyncio.CancelledError:
            if self._closed.is_set() or started:
                # A started write has an unknown outcome; never duplicate it.
                raise
            request.cancelled = True
            sent = await self.openai.send_event(payload_at_write, **guard)
        return effective if sent else None

    def _image_tool_finished(
        self, request: ToolImageRequest, source: RealtimeResponseState, task: asyncio.Task[None],
    ) -> None:
        self._image_tool_tasks.pop(request.provider_call_id, None)
        failed = task.cancelled() or task.exception() is not None
        if failed:
            source.interrupted = True
            if not task.cancelled() and not self._closed.is_set():
                LOGGER.warning("image tool output failed call_id=%s", request.provider_call_id)
                self._mark_closing("image_tool_output_failed")
        if failed or request.cancelled or request.provider_call_id not in source.requested_image_call_ids:
            self._release_tool_images((request.provider_call_id,))
        self._tool_call_settled(source, request.provider_call_id)

    def _tool_call_settled(self, source: RealtimeResponseState, call_id: str) -> None:
        source.pending_tool_call_ids.discard(call_id)
        if (
            not self._closed.is_set() and not source.pending_tool_call_ids
            and source.completed_response is not None and not source.completion_started
        ):
            if self._voice_route_runtime is not None:
                self._schedule_provider_completion(source.response_id, source.completed_response)
                return
            task = asyncio.create_task(self._finish_provider_response(source.response_id, source.completed_response))
            self._continuation_settlement_tasks.add(task)
            task.add_done_callback(self._continuation_settlement_finished)

    def _release_tool_images(self, call_ids: tuple[str, ...]) -> None:
        for call_id in call_ids:
            request = self._image_tool_requests.pop(call_id, None)
            if request is not None:
                request.cancelled = True
                if self._visual_delivery is not None:
                    self._visual_delivery.cancel_requested_image(
                        provider_call_id=call_id, turn_generation=request.turn_generation,
                    )

    def _cancel_image_tool_work(
        self, *, response_id: str | None = None, keep_call_ids: tuple[str, ...] = (),
    ) -> None:
        for call_id, request in tuple(getattr(self, "_image_tool_requests", {}).items()):
            if call_id in keep_call_ids or (response_id is not None and request.source_response_id != response_id):
                continue
            request.cancelled = True
            if self._visual_delivery is not None:
                self._visual_delivery.cancel_requested_image(
                    provider_call_id=call_id, turn_generation=request.turn_generation,
                )
            task = self._image_tool_tasks.get(call_id)
            if task is None:
                self._image_tool_requests.pop(call_id, None)
            elif request.started and not task.done() and not task.cancelling() and task is not asyncio.current_task():
                task.cancel()

    def _mark_current_response_interrupted(self) -> None:
        response_id = self._current_response_id
        if response_id is None:
            return
        state = self._active_responses.get(response_id)
        if state is not None:
            state.interrupted = True
            self._cancel_image_tool_work(response_id=state.response_id)

    def _finish_tool_response(self, *, response_id: str, continuing: bool) -> None:
        tool_chain_id = self._tool_chain_by_response_id.pop(response_id, None)
        self._responses_with_tool_calls.discard(response_id)
        if tool_chain_id is not None and not continuing:
            self._tool_budgets.pop(tool_chain_id, None)

    async def _persist_visual_interrupt_response_id(
        self,
        *,
        interrupt_id: str,
        response_id: str,
    ) -> None:
        try:
            if self.visual_repository is None:
                return
            await self.visual_repository.mark_visual_interrupt_spoken(
                owner_id=self.context.owner_id,
                interrupt_id=UUID(interrupt_id),
                spoken_response_id=response_id,
            )
        except Exception:
            LOGGER.info("failed to persist visual interrupt response id", exc_info=True)

    async def _persist_reminder_response_id(
        self,
        *,
        reminder_event_id: str,
        response_id: str,
    ) -> None:
        try:
            if self.reminder_repository is None:
                return
            await self.reminder_repository.set_event_provider_response_id(
                owner_id=self.context.owner_id,
                event_id=UUID(reminder_event_id),
                provider_response_id=response_id,
            )
        except Exception:
            LOGGER.info("failed to persist reminder response id", exc_info=True)

    async def _finish_reminder_response(
        self,
        *,
        state: RealtimeResponseState,
        response: dict[str, Any] | None,
        response_id: str,
    ) -> None:
        reminder_event_id = _string_value(state.metadata, "reminder_event_id")
        if reminder_event_id is None:
            return
        response_status = _string_value(response, "status")
        if state.interrupted:
            event_type = "interrupted"
            message = "Reminder speech was interrupted."
        elif response_status == "completed" and state.has_forwarded_answer:
            event_type = "delivered_to_session"
            message = None
        else:
            event_type = "failed"
            message = "Reminder audio did not complete."
        await self._complete_reminder_delivery(
            reminder_event_id=reminder_event_id,
            event_type=event_type,
            response_id=response_id,
            message=message,
            metadata={
                "delivery_route": "realtime",
                "provider_response_status": response_status,
                "forwarded_answer": state.has_forwarded_answer and not state.interrupted,
            },
        )

    async def _complete_reminder_delivery(
        self,
        *,
        reminder_event_id: str,
        event_type: str,
        metadata: dict[str, Any],
        response_id: str | None = None,
        message: str | None = None,
    ) -> None:
        reminder_id = self._pending_reminder_events.get(reminder_event_id)
        if reminder_id is None:
            return
        persisted = await self._persist_reminder_event_delivery(
            reminder_event_id=reminder_event_id,
            event_type=event_type,
            metadata=metadata,
        )
        if persisted:
            self._pending_reminder_events.pop(reminder_event_id, None)
        LOGGER.info(
            "reminder realtime response finished reminder_event_id=%s response_id=%s event_type=%s",
            reminder_event_id,
            response_id,
            event_type,
        )
        payload: dict[str, Any] = {
            "reminder_id": reminder_id,
            "reminder_event_id": reminder_event_id,
            "response_id": response_id,
        }
        if message is not None:
            payload["message"] = message
        app_event_type = "reminder.done" if event_type == "delivered_to_session" else f"reminder.{event_type}"
        await self._send_speech_notification(app_event_type, **payload)

    async def _settle_unstarted_continuation(
        self, metadata: dict[str, str], *, reason: str, request_id: str | None = None,
    ) -> None:
        """Settle only a definite no-response outcome, never a submitted unknown."""
        if request_id is not None:
            async with self._response_admission_lock:
                pending = self._pending_response
                if pending is not None and pending.request_id == request_id and not pending.submitted:
                    self._release_response_reservation(pending)
        kind = metadata.get("jarvis_delivery_kind")
        interrupted = reason in {"continuation_cancelled", "continuation_superseded"}
        if kind == "reminder":
            event_id = metadata.get("reminder_event_id")
            if event_id is not None:
                await self._complete_reminder_delivery(
                    reminder_event_id=event_id,
                    event_type="interrupted" if interrupted else "failed",
                    message=(
                        "Reminder audio was interrupted before it could start."
                        if interrupted else "Reminder audio could not be started."
                    ),
                    metadata={
                        "delivery_route": "realtime",
                        "continuation_request_id": request_id,
                        "reason": reason,
                        "forwarded_answer": False,
                    },
                )
        elif kind == "prospective":
            event_id = metadata.get("prospective_event_id")
            if event_id is not None:
                await self._complete_prospective_delivery(
                    event_id=event_id,
                    outcome="interrupted" if interrupted else "failed",
                    response_id=None,
                    reason=reason,
                )
        elif kind == "visual_interrupt":
            await self._send_speech_notification(
                "visual.interrupt.done",
                interrupt_id=metadata.get("visual_interrupt_id"),
                response_id=None,
                forwarded_answer=False,
                reason=reason,
            )

    async def _persist_reminder_event_delivery(
        self,
        *,
        reminder_event_id: str,
        event_type: str,
        metadata: dict[str, Any],
    ) -> bool:
        try:
            if self.reminder_repository is None:
                return False
            await self.reminder_repository.mark_event_delivery(
                owner_id=self.context.owner_id,
                event_id=UUID(reminder_event_id),
                event_type=event_type,
                session_id=self.context.jarvis_session_id,
                metadata=metadata,
            )
            return True
        except Exception:
            LOGGER.warning(
                "failed to persist reminder delivery reminder_event_id=%s event_type=%s",
                reminder_event_id,
                event_type,
                exc_info=True,
            )
            return False

    async def _reconcile_pending_reminders_on_close(self, *, reason: str) -> None:
        for reminder_event_id, reminder_id in list(self._pending_reminder_events.items()):
            persisted = await self._persist_reminder_event_delivery(
                reminder_event_id=reminder_event_id,
                event_type="interrupted",
                metadata={
                    "delivery_route": "realtime",
                    "reason": "session_closed",
                    "session_close_reason": reason,
                },
            )
            if persisted:
                self._pending_reminder_events.pop(reminder_event_id, None)
            with suppress(Exception):
                await self._send_app_event(
                    "reminder.interrupted",
                    reminder_id=reminder_id,
                    reminder_event_id=reminder_event_id,
                    message="Reminder speech was interrupted because the session closed.",
                )

    def _prospective_attempt_for(self, metadata) -> ProspectiveDeliveryAttempt | None:
        return getattr(self, "_prospective_attempts", {}).get(metadata.get("prospective_event_id"))

    async def _finish_prospective_response(
        self,
        *,
        state: RealtimeResponseState,
        response: dict[str, Any] | None,
        response_id: str,
    ) -> None:
        event_id = _string_value(state.metadata, "prospective_event_id")
        if event_id is None:
            return
        attempt = self._prospective_attempt_for(state.metadata)
        if attempt is not None and attempt.revoked:
            state.interrupted = True
        response_status = _string_value(response, "status")
        outcome = (
            "interrupted"
            if state.interrupted
            else "delivered"
            if response_status == "completed" and state.has_forwarded_answer
            else "failed"
        )
        await self._complete_prospective_delivery(
            event_id=event_id, outcome=outcome, response_id=response_id,
        )

    async def _complete_prospective_delivery(
        self, *, event_id: str, outcome: str, response_id: str | None,
        reason: str | None = None,
    ) -> None:
        pending = self._pending_prospective_deliveries.get(event_id)
        if pending is None:
            return
        subject_type, subject_id, on_outcome = pending
        attempt = getattr(self, "_prospective_attempts", {}).get(event_id)
        if attempt is not None and attempt.observed_outcome is not None:
            outcome, response_id = attempt.observed_outcome
        if attempt is not None and attempt.reason is None:
            attempt.reason = reason
        try:
            await on_outcome(outcome, response_id)
        except Exception:
            # Keep failed persistence available to the existing close reconciler.
            LOGGER.warning(
                "failed to persist prospective delivery event_id=%s outcome=%s",
                event_id,
                outcome,
                exc_info=True,
            )
            return
        else:
            self._pending_prospective_deliveries.pop(event_id, None)
            getattr(self, "_prospective_attempts", {}).pop(event_id, None)
        await self._send_speech_notification(
            f"prospective.{outcome}",
            prospective_event_id=event_id,
            subject_type=subject_type,
            subject_id=subject_id,
            response_id=response_id,
            **({"reason": reason} if reason is not None else {}),
        )

    async def _reconcile_pending_prospective_deliveries_on_close(
        self,
        *,
        reason: str,
    ) -> None:
        pending_deliveries = getattr(self, "_pending_prospective_deliveries", None)
        if not pending_deliveries:
            return

        for event_id, (subject_type, subject_id, on_outcome) in list(
            pending_deliveries.items()
        ):
            attempt = getattr(self, "_prospective_attempts", {}).get(event_id)
            outcome, response_id = (
                attempt.observed_outcome if attempt is not None and attempt.observed_outcome is not None
                else ("interrupted", None)
            )
            if attempt is not None and attempt.reason is None:
                attempt.reason = reason
            try:
                await on_outcome(outcome, response_id)
            except Exception:
                LOGGER.warning(
                    "failed to reconcile prospective delivery event_id=%s",
                    event_id,
                    exc_info=True,
                )
                continue
            self._pending_prospective_deliveries.pop(event_id, None)
            getattr(self, "_prospective_attempts", {}).pop(event_id, None)
            with suppress(Exception):
                await self._send_app_event(
                    f"prospective.{outcome}",
                    prospective_event_id=event_id,
                    subject_type=subject_type,
                    subject_id=subject_id,
                    response_id=response_id,
                    reason=attempt.reason if attempt is not None else reason,
                )

    async def _persist_assistant_transcript(self, *, response_id: str) -> None:
        state = self._active_responses.get(response_id)
        if state is None or state.interrupted:
            return
        for item in state.ordered_output_items:
            if (
                item.persisted or item.delivery_disposition != "forwarded"
                or not item.transcript.strip()
            ):
                continue
            metadata = {
                "audio_session_id": self.context.audio_session_id,
                "response_kind": state.delivery_kind,
                "realtime_output": item.as_metadata(),
            }
            if capture := getattr(self, "_context_capture", None):
                # This local flag prevents duplicate submissions, not a claim of
                # durable storage. Canonical capture owns the atomic message write.
                item.persisted = capture.assistant_message(
                    response_id=response_id, item_id=item.item_id,
                    text=item.transcript, metadata=metadata,
                )
                continue
            message = await self.repository.create_message(
                owner_id=self.context.owner_id,
                session_id=self.context.jarvis_session_id,
                role="assistant",
                modality="audio",
                content=item.transcript.strip(),
                **({"source_payload_id": self._source_context.source_for(state.metadata.get("jarvis_request_id"))}
                   if self._source_context.source_for(state.metadata.get("jarvis_request_id")) is not None else {}),
                provider_item_id=item.item_id,
                provider_response_id=response_id,
                metadata=metadata,
            )
            item.persisted = True
            await self._attach_message_to_experience(message)

    async def _attach_message_to_experience(self, message: Any) -> None:
        if self.experience_event_service is None or getattr(message, "source_payload_id", None) is not None:
            return
        try:
            await self.experience_event_service.attach_message(
                owner_id=self.context.owner_id,
                message=message,
            )
        except Exception:
            LOGGER.exception(
                "failed to attach durable message to experience event message=%s",
                getattr(message, "id", None),
            )

    async def _max_duration_timer(self) -> None:
        await asyncio.sleep(self.settings.JARVIS_REALTIME_MAX_SESSION_SECONDS)
        self._mark_closing("max_session_duration")

    async def _send_speech_notification(self, event_type: str, **payload: Any) -> None:
        """App status hints must not determine authoritative speech delivery outcomes."""
        try:
            await self._send_app_event(event_type, **payload)
        except Exception:
            LOGGER.warning("realtime speech notification failed event_type=%s", event_type)

    async def _send_app_event(self, event_type: str, **payload: Any) -> bool:
        attempt = None
        origin = None
        snapshot = None
        media = event_type in {"audio.output.delta", "audio.output.done", "transcript.delta", "transcript.done"}
        if media:
            state = self._active_responses.get(payload.get("response_id"))
            if state is not None:
                attempt = self._prospective_attempt_for(state.metadata)
                origin = unsolicited_response_origin(state.delivery_kind, state.metadata)
                snapshot = getattr(state, "source_snapshot", None)
        if not media:
            origin = {"visual.interrupt.started": "visual_interrupt", "prospective.started": "prospective",
                      "reminder.started": "reminder"}.get(event_type)
        source_bound = (snapshot is not None or self._source_context.has_sources) and (
            media or event_type in {"visual.interrupt.started", "prospective.started", "reminder.started"}
        )
        @asynccontextmanager
        async def current_output_sources():
            if snapshot is not None:
                # Source-authorized as-of delivery never refreshes its evidence
                # into a different answer. New response/tool guards stay strict.
                async with self._source_context.output_guard(snapshot):
                    yield
                return
            # context_guard can wait for canonical locks while observe() adds
            # unrelated sources. Before yield, live_context_changed proves this
            # guard has not admitted a client write. Revalidate the full current
            # root set, under the same writer/deadline (and borrowed owner
            # transaction, if any), rather than retiring a healthy response.
            # After yield, even the same error code is an uncertain write and
            # must not retry the output or its forwarding marker.
            entered = False
            refreshes = 0
            while True:
                try:
                    async with self._source_context.context_guard():
                        entered = True
                        yield
                    return
                except LiveContextError as error:
                    if entered or error.code != "live_context_changed" or refreshes >= 2:
                        raise
                    refreshes += 1
                    with suppress(Exception):
                        self._trace.event("app_output_source_refresh",
                            event_type=event_type, response_id=payload.get("response_id"),
                            reason="source_addition_before_write", status=f"refresh_{refreshes}")

        source_guard = current_output_sources if source_bound else None
        @asynccontextmanager
        async def quiet_app_write_guard():
            # The quiet guard lends a connection to its legitimate callers.
            # The app writer instead interprets a yielded value as an optional
            # post-send forwarding callback. Keep authority through the send
            # without exposing the non-callable connection as that callback.
            async with quiet_response_guard(
                self._source_context, origin=origin, source_guard=source_guard
            ):
                yield

        write_guard = quiet_app_write_guard if origin else source_guard
        try:
            async with asyncio.timeout(
                self.settings.OPENAI_REALTIME_WRITE_TIMEOUT_SECONDS if source_bound or origin else None
            ):
                sent = await self._app_sender.send_text(
                    encode_event(event_type, **payload),
                    **({"should_send": lambda: not self._closed.is_set() and not self._context_revoked
                        and (attempt is None or not attempt.revoked)} if media or source_bound else {}),
                    **({"write_guard": write_guard} if write_guard is not None else {}),
                )
        except QuietDeliverySuppressed:
            # Already-forwarded/buffered audio is not recalled by this write fence.
            return False
        except (LiveContextError, TimeoutError):
            if not source_bound:
                raise
            self._begin_context_rebuild()
            return False
        if sent and event_type == "audio.output.delta" and payload.get("item_id"):
            item = self._native_playback.setdefault(payload["item_id"], {
                "response_id":payload.get("response_id"), "samples":0, "done":False})
            item["samples"] += len(base64.b64decode(payload.get("audio_base64", ""))) // 2
        elif sent and event_type == "audio.output.done" and payload.get("item_id") in self._native_playback:
            self._native_playback[payload["item_id"]]["done"] = True
        return sent

    def _retire_native_playback(self, item_id):
        if item := self._native_playback.pop(item_id, None):
            self._native_playback_history[item_id] = item
        while len(self._native_playback_history) > 256:
            self._native_playback_history.pop(next(iter(self._native_playback_history)))

    async def try_attention(self, repository, speech, *, worker_id, receipt_seconds):
        """One admission lane for native and expert speech; never interrupt a user."""
        from app.attention.voice import ExpertVoiceDelivery
        async with self._response_admission_lock:
            if (self._closed.is_set() or not self._realtime_context_ready.is_set()
                    or self._attention_delivery is not None or self._user_audio_pending
                    or self._normal_turn_pending or self._pending_response is not None
                    or self._active_responses or self._native_playback):
                record_admission(
                    "route_not_ready" if self._closed.is_set() or not self._realtime_context_ready.is_set()
                    else "foreground_busy",
                    owner=self.context.owner_id,
                    session_id=self.context.jarvis_session_id,
                )
                return None
            def admit(row, output):
                if row["source_namespace"] == "visual_alert":
                    rejection = generic_visual_speech_rejection(self._generic_speech_activity(),
                        source_matches=row["preferred_session_id"] == self.context.jarvis_session_id,
                        source_received_at=output.content.source_received_at, now=self._now(),
                        monotonic_now=self._monotonic())
                    if rejection is not None:
                        return rejection
                if row["source_namespace"] != "task_result" and not self._proactive_interrupt_gate.can_begin():
                    return "proactive_chain_limit"
                return None
            claim = await repository.claim_next(self.context.owner_id, self.context.jarvis_session_id,
                                                worker_id, admission=admit)
            if claim is None:
                return None
            if claim.source_namespace != "task_result":
                self._proactive_interrupt_gate.try_begin()
            delivery_factory = getattr(self, "_attention_delivery_factory", ExpertVoiceDelivery)
            delivery = delivery_factory(repository, speech, claim, self, receipt_seconds=receipt_seconds)
            self._attention_delivery = delivery
            delivery.task = asyncio.create_task(delivery.run())
            return delivery

    def remember_attention_item(self, item_id, delivery):
        # Keep tombstones after cancellation: the client's subsequent truncate
        # must never be forwarded to OpenAI for an application-owned TTS item.
        self._attention_receipts[item_id] = delivery
        while len(self._attention_receipts) > 256:
            self._attention_receipts.pop(next(iter(self._attention_receipts)))

    async def release_attention(self, delivery):
        async with self._response_admission_lock:
            if self._attention_delivery is delivery:
                self._attention_delivery = None

    async def record_expert_speech(self, delivery, index, text):
        return await self.record_attention_speech(delivery, index, text)

    async def record_attention_speech(self, delivery, index, text):
        # Exact attributed output, never a request for the foreground to retell it.
        expert = getattr(delivery.claim, "source_namespace", "task_result") == "task_result"
        from app.attention.notifications import speech_attribution
        origin, author, label = speech_attribution(getattr(delivery.claim, "source_namespace", "task_result"))
        async with asyncio.timeout(5):
            async with delivery.repository.guard(delivery.claim) as (connection, _, _, _):
                if not delivery.current():
                    return
                item_id = delivery.item_id(index)
                # Reuse the guarded transaction. Acquiring another pool slot
                # while an ingress writer waits on our owner lock can deadlock
                # a bounded pool, even though both individual queries are valid.
                self._source_context.observe((delivery.claim.source_payload_id,))
                await self.repository.create_message(connection=connection, owner_id=self.context.owner_id,
                    source_payload_id=delivery.claim.source_payload_id,
                    session_id=self.context.jarvis_session_id, role="assistant", modality="audio", content=text,
                    provider_item_id=item_id, provider_response_id=delivery.response_id,
                    metadata={"audio_session_id":self.context.audio_session_id, "response_kind":origin,
                              "intent_id":str(delivery.claim.intent_id), "chunk_index":index, "playback_status":"unknown",
                              "authored_by":author})
                if delivery.current():
                    await self.openai.send_event({"type":"conversation.item.create", "item":{
                        "id":uuid5(delivery.claim.attempt_id, f"{'expert' if expert else 'notification'}:{index}").hex,
                        "type":"message", "role":"assistant", "content":[{"type":"output_text",
                        "text":f"[{label}; playback is not confirmed. Do not repeat automatically.]\n" + text}]}})
        # Source-backed canonical capture is the expert's memory intake. Do not
        # create an independent legacy experience copy that removal cannot trace.

    async def _send_error(self, code: str, message: str, *, recoverable: bool) -> None:
        with suppress(Exception):
            await self._send_app_event(
                "error",
                code=code,
                message=message,
                recoverable=recoverable,
            )

    def _enforce_client_message_size(self, raw_message: str | bytes) -> None:
        size_bytes = len(raw_message.encode("utf-8")) if isinstance(raw_message, str) else len(raw_message)
        if size_bytes > self.settings.JARVIS_REALTIME_MAX_CLIENT_MESSAGE_BYTES:
            raise ClientEventError("Client event is too large.")

    def _validated_audio_base64(self, value: str) -> str:
        try:
            decoded = base64.b64decode(value, validate=True)
        except binascii.Error as error:
            raise ClientEventError("audio_base64 must be valid base64.") from error
        if not decoded:
            raise ClientEventError("audio_base64 must not be empty.")
        if len(decoded) > self.settings.JARVIS_REALTIME_MAX_AUDIO_CHUNK_BYTES:
            raise ClientEventError("audio_base64 chunk is too large.")
        return value


def _string_value(payload: dict[str, Any] | None, key: str) -> str | None:
    if not isinstance(payload, dict):
        return None
    value = payload.get(key)
    return value if isinstance(value, str) and value else None


def _openai_error_message(payload: dict[str, Any]) -> str:
    # Provider error text can echo request content, including image data or
    # authorization values. Expose only a bounded machine-readable error code.
    code = _string_value(payload.get("error"), "code")
    if code is not None and re.fullmatch(r"[a-z][a-z0-9_]{0,79}", code):
        return f"Realtime provider error ({code})."
    return "Realtime provider error."
