from __future__ import annotations

from app.memory.prospective.delivery import ProspectiveDeliveryAttempt

from app.errors import JarvisHTTPError

import asyncio
from collections import OrderedDict, deque
from collections.abc import Awaitable, Callable
from contextlib import suppress
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import UTC, datetime
import hashlib
import inspect
import json
import logging
import time
from typing import Any
from uuid import UUID

from fastapi import WebSocket

from app.assistant.tool_limits import ToolCallBudget
from app.assistant.proactive_policy import (
    GenericSpeechActivity, ProactiveInterruptGate, generic_visual_speech_rejection,
)
from app.images import ImageUnavailableError
from app.realtime.gemini.client import GeminiLiveConnection, GeminiTransportError
from app.realtime.gemini.config import GeminiLiveConfig
from app.realtime.gemini.context import GeminiContext, PreparedImage, bounded_context
from app.realtime.gemini.context_updates import GeminiContextUpdates
from app.realtime.gemini.events import GeminiOutputTurn, decode_pcm16
from app.realtime.gemini.instructions import build_gemini_instructions
from app.realtime.gemini.operations import GeminiOperations, NativeCall, OPERATION_INSTRUCTIONS, native_declarations
from app.realtime.gemini.persistence import FinalTranscript, GeminiConversationJournal
from app.realtime.gemini.proactive import (
    GeminiProactiveDeliveries, ProactiveDelivery, PROACTIVE_INSTRUCTIONS,
)
from app.realtime.gemini.recovery import GeminiRecovery, MAX_RESUME_ATTEMPTS
from app.realtime.models import RealtimeSessionContext
from app.realtime.socket_sender import WebSocketEventSender
from app.realtime.telemetry import RealtimeTrace
from app.settings import JarvisSettings
from app.tools.module import ToolBuildContext
from app.tools.registry import create_tool_registry


LOGGER = logging.getLogger(__name__)
MAX_QUEUED_EVENTS = 32
MAX_PENDING_RESPONSES = 8
MAX_RETAINED_TURNS = 256
MAX_CONTEXT_CHARACTERS = 24000
DOMAIN_DRAIN_TIMEOUT_SECONDS = 10.0
TOOL_TIMEOUT_SECONDS = 30.0


@dataclass(frozen=True, slots=True)
class PreparedInput:
    context_text: str
    evidence_status: str = "available"
    image: PreparedImage | None = None
    image_state: str = "unavailable"


class OptionalPreparationUnavailable(Exception):
    """Only optional context is missing; required operation state must stay intact."""


@dataclass(slots=True)
class GeminiInputTurn:
    segment_id: str
    number: int
    ingress_epoch: int
    connection_generation: int
    output: GeminiOutputTurn
    input_bytes: int = 0
    sealed: bool = False
    cancelled: bool = False
    released: bool = False
    provider_started: bool = False
    input_finalized: bool = False
    input_status: str = "pending"
    input_transcript: str = ""
    preparation_task: asyncio.Task | None = None
    response_timeout_task: asyncio.Task | None = None
    committed_at: float | None = None
    commit_received_at: float | None = None
    released_at: float | None = None
    preparation_deadline: float | None = None
    cutoff_at: str | None = None
    evidence_status: str | None = None
    first_provider_audio_at: float | None = None
    first_forwarded_audio_at: float | None = None
    playback_end_ms: int | None = None
    timing: dict[str, str] = field(default_factory=dict)
    tool_budget: ToolCallBudget | None = None
    proactive: ProactiveDelivery | None = None
    native_namespace: int = 1

    @property
    def input_item_id(self) -> str:
        return f"{self.output.audio_session_id}:g{self.connection_generation}:u{self.number}"


class GeminiAudioSession:
    """Jarvis phone protocol over a separately owned Gemini native connection.

    The dispatcher is the sole provider writer. Preparation only computes data;
    it cannot write a delayed end marker. Ingress epochs invalidate a preparation
    even while the dispatcher is awaiting a socket write. Domain preparation,
    tools, and persistence are separate integration hooks, not GPT emulation.
    """

    def __init__(
        self, *, websocket: WebSocket, context: RealtimeSessionContext,
        settings: JarvisSettings, repository: Any,
        tool_modules=(), context_providers=(), visual_repository=None,
        sensory_frame_window=None, reminder_repository=None,
        tool_execution_repository=None, experience_event_service=None,
        context_capture=None,
        connection_factory: Callable[..., GeminiLiveConnection] = GeminiLiveConnection,
        config: GeminiLiveConfig | None = None,
        connection_admission: Callable[[], None] | None = None,
    ) -> None:
        self.websocket = websocket
        self.context = context
        self.settings = settings
        self.repository = repository
        self.tool_modules = tuple(tool_modules)
        self.context_providers = tuple(context_providers)
        self.visual_repository = visual_repository
        self.sensory_frame_window = sensory_frame_window
        self.reminder_repository = reminder_repository
        self.tool_execution_repository = tool_execution_repository
        self.experience_event_service = experience_event_service
        self._context_capture = context_capture
        self.config = config if config is not None else GeminiLiveConfig.from_settings(settings)
        self._connection_admission = connection_admission
        self._connection_factory = connection_factory
        self.connection = connection_factory(config=self.config)
        self._connection_generation = 1
        self._native_namespace = 1
        self._native_setup: dict[str, Any] | None = None
        self._queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue(MAX_QUEUED_EVENTS)
        self._tasks: set[asyncio.Task] = set()
        self._provider_task: asyncio.Task | None = None
        self._closed = asyncio.Event()
        self._close_task: asyncio.Task | None = None
        self._close_reason = "unknown"
        self._running = False
        self._run_task: asyncio.Task | None = None
        self._ready = False
        self._trace = RealtimeTrace(provider="gemini", audio_session_id=context.audio_session_id,
                                    jarvis_session_id=context.jarvis_session_id)
        self._sender = WebSocketEventSender(websocket, on_closed=self._peer_departed, trace=self._trace)
        self._turns: OrderedDict[str, GeminiInputTurn] = OrderedDict()
        self._pending_responses: deque[GeminiInputTurn] = deque()
        self._pending_transcripts: deque[GeminiInputTurn] = deque()
        self._active_input: GeminiInputTurn | None = None
        self._marker_turn: GeminiInputTurn | None = None
        self._turn_number = 0
        self._ingress_epoch = 0
        self._ingress_segment: str | None = None
        self._ingress_sealed = False
        self._seen_events: OrderedDict[str, str] = OrderedDict()
        self.recovery = GeminiRecovery()
        self._recovery_timer: asyncio.Task | None = None
        self.registry = create_tool_registry(
            modules=self.tool_modules,
            context=ToolBuildContext(settings, context.owner_id, context.jarvis_session_id),
        )
        self.operations = GeminiOperations(
            registry=self.registry, repository=tool_execution_repository,
            owner_id=context.owner_id, session_id=context.jarvis_session_id,
            audio_session_id=context.audio_session_id,
            experience_event_service=experience_event_service,
            context_capture=context_capture,
        )
        self._domain_tasks: set[asyncio.Task] = set()
        self.proactive = GeminiProactiveDeliveries(
            context=context, reminder_repository=reminder_repository,
            emit=self._emit, on_failure=self._persistence_failure,
        )
        self._preparing_proactive: ProactiveDelivery | None = None
        self._proactive_gate = ProactiveInterruptGate(
            max_chain_length=settings.JARVIS_REALTIME_MAX_INTERRUPT_CHAIN,
        )
        self._last_user_turn_settled_at: datetime | None = None
        self._last_user_turn_settled_monotonic: float | None = None
        self._visual_frame_notifications = 0
        self._context_service = GeminiContext(
            context=context, settings=settings, repository=repository,
            providers=self.context_providers, registry=self.registry,
        )
        self.journal = GeminiConversationJournal(
            repository=repository, experience_service=experience_event_service,
            owner_id=context.owner_id, session_id=context.jarvis_session_id,
            audio_session_id=context.audio_session_id, on_failure=self._persistence_failure,
        )
        self._initial_context_payload: dict | None = None
        self.context_updates = GeminiContextUpdates(
            context=context, build=self._context_service.shared_payload, emit=self._emit,
            closed=self._closed.is_set, timeout_seconds=self.config.preparation_timeout_seconds,
        )

    async def run(self) -> None:
        if self._running or self._closed.is_set():
            raise RuntimeError("A Gemini audio session cannot be reused")
        self._running = True
        self._run_task = asyncio.current_task()
        self._spawn(self._client_loop())
        self._spawn(self._initial_setup())
        stop = asyncio.create_task(self._closed.wait())
        try:
            async with asyncio.timeout(self.settings.JARVIS_REALTIME_MAX_SESSION_SECONDS):
                while not self._closed.is_set():
                    receive = asyncio.create_task(self._queue.get())
                    try:
                        done, _ = await asyncio.wait({receive, stop}, return_when=asyncio.FIRST_COMPLETED)
                        if stop in done:
                            break
                        kind, payload = receive.result()
                    finally:
                        receive.cancel()
                        with suppress(asyncio.CancelledError):
                            await receive
                    await self._dispatch(kind, payload)
        except asyncio.CancelledError:
            if not self._closed.is_set():
                self._mark_closed("cancelled")
                raise
        except Exception as error:
            if not self._closed.is_set():
                code = error.code if isinstance(error, GeminiTransportError) else (
                    "session_timeout" if isinstance(error, TimeoutError) else "gemini_session_failed"
                )
                await self._error(code, recoverable=False)
                self._mark_closed(code)
        finally:
            stop.cancel()
            with suppress(asyncio.CancelledError):
                await stop
            await self.close(self._close_reason)

    def _spawn(self, operation: Awaitable) -> asyncio.Task:
        operation_started = False

        async def owned():
            nonlocal operation_started
            operation_started = True
            try:
                await operation
            except asyncio.CancelledError:
                raise
            except Exception as error:
                if not self._closed.is_set():
                    code = error.code if isinstance(error, GeminiTransportError) else "gemini_background_failed"
                    await self._queue.put(("failure", code))

        task = asyncio.create_task(owned())
        self._tasks.add(task)

        def finished(completed):
            self._tasks.discard(completed)
            # A wrapper cancelled before its first step never awaits the owned
            # coroutine. Explicitly close it rather than leaking an unawaited
            # timeout/provider loop (notably immediate checkpoint recovery).
            if not operation_started and inspect.iscoroutine(operation):
                operation.close()

        task.add_done_callback(finished)
        return task

    async def _initial_setup(self) -> None:
        await self._connect_current()
        await self._queue.put(("ready", None))

    def _check_connection_admission(self) -> None:
        if self._connection_admission is not None:
            try:
                self._connection_admission()
            except JarvisHTTPError as error:
                raise GeminiTransportError(error.code) from None

    async def _connect_current(self) -> None:
        self._check_connection_admission()
        setup = await self._connection_setup()
        self._check_connection_admission()
        task = asyncio.create_task(self.connection.connect(**setup))
        self._tasks.add(task)
        try:
            await task
            self._native_setup = deepcopy(setup)
        finally:
            self._tasks.discard(task)

    async def _connection_setup(self) -> dict[str, Any]:
        shared = None
        try:
            async with asyncio.timeout(self.config.preparation_timeout_seconds):
                shared = await self._context_service.shared_payload(include_history=True)
        except Exception:
            pass  # Optional context can be rebuilt at the next user-turn cutoff.
        self._initial_context_payload = shared
        base_instructions = getattr(self.context.session, "behavior_instructions", "")
        if shared is not None:
            base_instructions = shared.get("behavior_instructions", {}).get("effective", base_instructions)
        history = self._conversation_history()
        initial_context = bounded_context({
            "kind": "jarvis_application_context", "memory_context": shared or "unavailable",
            "operation_context": self.operations.snapshot(),
        }, max_characters=MAX_CONTEXT_CHARACTERS)
        history.insert(0, {"role": "user", "parts": [{"text": initial_context}]})
        return {
            "instructions": build_gemini_instructions(base_instructions=base_instructions,
                tasks_enabled=self.registry.get("start_background_task") is not None)
                            + "\n\n" + OPERATION_INSTRUCTIONS + "\n\n" + PROACTIVE_INSTRUCTIONS,
            "history": history,
            "tools": native_declarations(self.registry),
        }

    def _conversation_history(self) -> list[dict[str, Any]]:
        history = []
        remaining = MAX_CONTEXT_CHARACTERS
        # Logical input order, not asynchronous transcript arrival order. Build
        # newest-first for the budget, then restore chronological native turns.
        groups = []
        for turn in reversed(self._turns.values()):
            group = []
            if turn.input_finalized:
                prefix = (
                    "[Earlier input from an interrupted exchange; context only, not a new command] "
                    if turn.cancelled else ""
                )
                group.append({"role": "user", "parts": [{"text": prefix + turn.input_transcript}]})
            output = turn.output
            complete = (
                output.completed and not output.interrupted and not turn.cancelled
                and bool(output.transcript.strip())
                and output.forwarded_audio_bytes > 0
                and output.forwarded_audio_bytes == output.provider_audio_bytes
                and turn.playback_end_ms is None
            )
            input_committed = turn.sealed and turn.input_bytes > 0
            if (
                (input_committed and not turn.input_finalized)
                or (not complete and (turn.sealed or output.forwarded_audio_bytes))
            ):
                # This is evidence about an incomplete exchange, not a completed
                # model turn or a fabricated transcript of missing user speech.
                record = {
                    "kind": "jarvis_incomplete_exchange",
                    "turn_number": turn.number,
                    "segment_id": turn.segment_id,
                    "original_connection_generation": turn.connection_generation,
                    "input_status": turn.input_status if turn.input_bytes else "not_applicable",
                    "input_committed": input_committed,
                }
                if not complete:
                    assistant = {
                        "completion_state": (
                            "truncated" if turn.playback_end_ms is not None
                            else "interrupted" if output.interrupted or turn.cancelled else "incomplete"
                        ),
                        "forwarded_audio_bytes": output.forwarded_audio_bytes,
                        "playback_end_ms": turn.playback_end_ms,
                        "heard_text_status": "unknown",
                    }
                    if output.forwarded_audio_bytes and output.context_transcript:
                        assistant["context_transcript"] = output.context_transcript
                    record["assistant_output"] = assistant
                group.append({"role": "user", "parts": [{"text": json.dumps(record, separators=(",", ":"))}]})
            if complete:
                group.append({"role": "model", "parts": [{"text": output.transcript}]})
            if not group:
                continue
            if len(groups) >= 16:
                break
            size = sum(len(item["parts"][0]["text"]) for item in group)
            if size > remaining:
                if groups:
                    break
                # One long recent exchange must not erase all reset context.
                # Excerpt source strings, not serialized JSON or operation receipts.
                text = self._history_excerpt(turn, complete=complete, max_characters=remaining)
                group = [{"role": "user", "parts": [{"text": text}]}]
                size = len(text)
            remaining -= size
            groups.append(group)
        for group in reversed(groups):
            history.extend(group)
        return history

    @staticmethod
    def _history_excerpt(turn: GeminiInputTurn, *, complete: bool, max_characters: int) -> str:
        output = turn.output
        record = {
            "kind": "jarvis_history_excerpt",
            "purpose": "historical context only, not a new command",
            "truncated": True,
            "turn_number": turn.number,
            "original_connection_generation": turn.connection_generation,
            "input_status": turn.input_status if turn.input_bytes else "not_applicable",
            "input_committed": turn.sealed and turn.input_bytes > 0,
            "assistant_completion_state": (
                "completed" if complete else "truncated" if turn.playback_end_ms is not None
                else "interrupted" if output.interrupted or turn.cancelled else "incomplete"
            ),
            "heard_text_status": "unknown",
            "playback_end_ms": turn.playback_end_ms,
        }
        sources = {}
        if turn.input_finalized:
            sources["input_transcript_excerpt"] = turn.input_transcript
        if output.forwarded_audio_bytes:
            sources["assistant_transcript_excerpt"] = (
                output.transcript if complete else output.context_transcript
            )
        keep = max((len(text) for text in sources.values()), default=0)
        while True:
            excerpts = {
                key: text if len(text) <= keep else (
                    text[:keep // 2] + " [... omitted due to history budget ...] "
                    + (text[-(keep // 2):] if keep >= 2 else "")
                )
                for key, text in sources.items()
            }
            encoded = json.dumps({**record, **excerpts}, separators=(",", ":"))
            if len(encoded) <= max_characters:
                return encoded
            if keep == 0:
                raise GeminiTransportError("gemini_history_metadata_too_large")
            keep //= 2

    async def _client_loop(self) -> None:
        async for raw in self.websocket.iter_text():
            if self._closed.is_set():
                return
            if len(raw.encode("utf-8")) > self.settings.JARVIS_REALTIME_MAX_CLIENT_MESSAGE_BYTES:
                raise GeminiTransportError("client_message_too_large")
            try:
                event = json.loads(raw)
            except (ValueError, TypeError):
                await self._queue.put(("protocol_error", "invalid_client_json"))
                continue
            if not isinstance(event, dict):
                await self._queue.put(("protocol_error", "invalid_client_event"))
                continue
            event_id = event.get("client_event_id")
            if event_id is not None:
                if not isinstance(event_id, str) or not 0 < len(event_id) <= 128:
                    await self._queue.put(("protocol_error", "invalid_client_event_id"))
                    continue
                fingerprint = hashlib.sha256(json.dumps(event, sort_keys=True).encode()).hexdigest()
                if event_id in self._seen_events:
                    if self._seen_events[event_id] != fingerprint:
                        await self._queue.put(("protocol_error", "conflicting_client_event"))
                    continue
                self._seen_events[event_id] = fingerprint
                if len(self._seen_events) > MAX_RETAINED_TURNS:
                    self._seen_events.popitem(last=False)
            if event.get("type") == "session.close":
                self._mark_closed("client_closed")
                self._ensure_close_task()
                return
            # Invalidate obsolete preparation before a congested dispatcher can
            # release it. This reader does not perform provider writes.
            segment = event.get("segment_id")
            if (event.get("type") == "audio.input.append"
                    and isinstance(segment, str) and 0 < len(segment) <= 128
                    and segment != self._ingress_segment and segment not in self._turns
                    and (self._ingress_segment is None or self._ingress_sealed)):
                try:
                    decode_pcm16(event.get("audio_base64"), max_bytes=self.config.max_input_chunk_bytes)
                except ValueError:
                    pass
                else:
                    self._ingress_epoch += 1
                    self._ingress_segment = segment
                    self._ingress_sealed = False
            if event.get("type") == "audio.input.commit" and segment == self._ingress_segment:
                self._ingress_sealed = True
            if event.get("type") == "response.cancel" and self._ingress_sealed:
                self._ingress_epoch += 1
                self._ingress_segment = None
                self._ingress_sealed = False
            await self._queue.put(("client", (event, self._ingress_epoch, time.monotonic())))
        self._mark_closed("client_disconnect")
        self._ensure_close_task()

    async def _provider_loop(self, connection, generation: int) -> None:
        try:
            while not self._closed.is_set():
                event = await connection.receive_event()
                await self._queue.put(("provider", (generation, event)))
        except asyncio.CancelledError:
            raise
        except Exception as error:
            if not self._closed.is_set():
                code = error.code if isinstance(error, GeminiTransportError) else "gemini_receive_failed"
                await self._queue.put(("provider_failure", (generation, code)))

    async def _dispatch(self, kind: str, payload: Any) -> None:
        if kind == "failure":
            raise GeminiTransportError(payload)
        if kind == "protocol_error":
            await self._error(payload, recoverable=True)
        elif kind == "ready":
            self._ready = True
            self._provider_task = self._spawn(self._provider_loop(self.connection, self._connection_generation))
            await self._emit({
                "type": "session.created", "session_id": str(self.context.jarvis_session_id),
                "audio_session_id": self.context.audio_session_id,
                "client_kind": self.context.client_kind, "server_time": datetime.now(UTC).isoformat(),
            })
            self.context_updates.initialize(self._initial_context_payload)
        elif kind == "client":
            if self._should_defer_client(payload):
                self.recovery.defer(payload)
            else:
                await self._handle_client(*payload)
        elif kind == "provider":
            generation, event = payload
            if generation == self._connection_generation:
                await self._handle_provider(event)
        elif kind == "provider_failure":
            generation, code = payload
            if generation == self._connection_generation:
                if code not in {"gemini_receive_failed", "gemini_connection_closed"}:
                    raise GeminiTransportError(code)
                if self._checkpoint_safe() and self.recovery.handle is not None:
                    await self._resume_connection(reason="provider_disconnect")
                else:
                    raise GeminiTransportError("gemini_recovery_not_safe")
        elif kind == "prepared":
            turn, prepared = payload
            await self._release_turn(turn, prepared)
        elif kind == "preparation_failed":
            turn, code = payload
            if self._current_preparation(turn):
                raise GeminiTransportError(code)
        elif kind == "response_timeout":
            if not payload.output.completed and payload.connection_generation == self._connection_generation:
                raise GeminiTransportError("gemini_response_timeout")
        elif kind == "tool_ready":
            await self._deliver_tool_result(*payload)
        elif kind == "persistence_error":
            await self._error(payload, recoverable=True)
        elif kind == "proactive":
            await self._begin_proactive(payload)
        elif kind == "proactive_prepared":
            await self._release_proactive(*payload)
        elif kind == "recovery_expired":
            if self.recovery.deadline == payload:
                raise GeminiTransportError("gemini_recovery_deadline")
        await self._maybe_resume()

    def _checkpoint_safe(self) -> bool:
        if self._preparing_proactive is not None:
            return False
        return (self._ready and not self._closed.is_set() and not self._pending_responses
                and not self._pending_transcripts and not self._domain_tasks
                and (self._active_input is None or (self._active_input.sealed and self._active_input.released)))

    def _should_defer_client(self, payload: tuple) -> bool:
        if not self.recovery.pending:
            return False
        event = payload[0]
        kind = event.get("type")
        if kind in {"response.cancel", "playback.truncated"}:
            return False
        if self.recovery.has_deferred:
            return True
        if kind != "audio.input.append":
            return False
        active = self._active_input
        if active is not None and event.get("segment_id") == active.segment_id:
            return False  # Finish the activity already sent to this connection.
        unresolved = self._pending_responses[-1] if self._pending_responses else None
        if ((active is not None and active.sealed and not active.released)
                or (unresolved is not None and not unresolved.provider_started)
                or (unresolved is not None and any(
                    call.namespace == self._native_namespace and call.turn_number == unresolved.number
                    and not call.delivered for call in self.operations.calls.values()
                ))):
            # Preserve the already-proven early/tool barge-in fresh-session path.
            # It retires the unsafe native state, not admitted application work.
            return False
        return True

    async def _maybe_resume(self) -> None:
        if self.recovery.pending and self._checkpoint_safe() and self.recovery.handle is not None:
            await self._resume_connection(reason="goaway")

    async def _recovery_deadline(self, deadline: float) -> None:
        await asyncio.sleep(max(0, deadline - time.monotonic()))
        if not self._closed.is_set():
            await self._queue.put(("recovery_expired", deadline))

    async def _resume_connection(self, *, reason: str) -> None:
        self._check_connection_admission()
        if not self._checkpoint_safe() or self.recovery.handle is None or self._native_setup is None:
            raise GeminiTransportError("gemini_recovery_not_safe")
        handle = self.recovery.handle
        deadline = self.recovery.deadline or time.monotonic() + self.config.setup_timeout_seconds
        self._ready = False
        if self._recovery_timer is not None:
            self._recovery_timer.cancel()
            self._recovery_timer = None
        if self._provider_task is not None:
            self._provider_task.cancel()
            await asyncio.gather(self._provider_task, return_exceptions=True)
        await self.connection.close()
        await self._emit({"type": "audio.provider.reconnecting", "provider": "gemini", "reason": reason,
                          "audio_session_id": self.context.audio_session_id})
        try:
            async with asyncio.timeout_at(deadline):
                for attempt in range(MAX_RESUME_ATTEMPTS):
                    self._check_connection_admission()
                    self.recovery.attempts += 1
                    self._connection_generation += 1
                    self.connection = self._connection_factory(config=self.config)
                    setup = deepcopy({key: self._native_setup[key] for key in ("instructions", "tools")})
                    task = asyncio.create_task(self.connection.connect(**setup, resume_handle=handle))
                    self._tasks.add(task)
                    try:
                        await task
                        break
                    except GeminiTransportError:
                        await self.connection.close()
                        if attempt + 1 == MAX_RESUME_ATTEMPTS:
                            raise
                    finally:
                        self._tasks.discard(task)
        except asyncio.CancelledError:
            raise
        except Exception:
            raise GeminiTransportError("gemini_resume_failed") from None
        self.recovery.resumptions += 1
        deferred = self.recovery.release()
        self._ready = True
        self._provider_task = self._spawn(self._provider_loop(self.connection, self._connection_generation))
        await self._emit({"type": "audio.provider.reconnected", "provider": "gemini", "resumed": True,
                          "audio_session_id": self.context.audio_session_id, "reason": reason,
                          "connection_generation": self._connection_generation})
        # These are newly received, never-forwarded phone events, not a replay
        # of ambiguous audio from the retired connection. One dispatcher writes.
        for event in deferred:
            if self._closed.is_set():
                break
            await self._handle_client(*event)

    async def _handle_client(self, event: dict, ingress_epoch: int, received_at: float | None = None) -> None:
        kind = event.get("type")
        self._trace.client_event(event, connection_generation=self._connection_generation)
        if not self._ready:
            await self._error("audio_session_not_ready", recoverable=True)
            return
        try:
            if kind == "audio.input.append":
                await self._append(event, ingress_epoch)
            elif kind == "audio.input.commit":
                await self._commit(event, received_at)
            elif kind == "response.create":
                if self._marker_turn is None:
                    raise ValueError("response_without_committed_input")
                self._marker_turn = None
            elif kind == "response.cancel":
                await self._cancel_output()
            elif kind == "playback.truncated":
                self._record_playback(event)
            else:
                raise ValueError("unsupported_client_event")
        except ValueError as error:
            # Only application-owned protocol codes are exposed, not arbitrary
            # decoder text or client-supplied content.
            allowed = {
                "response_without_committed_input", "unsupported_client_event",
                "invalid_segment_id", "conflicting_segment", "append_after_commit",
                "empty_audio_commit", "invalid_playback_progress",
            }
            code = str(error) if str(error) in allowed else "invalid_audio_input"
            await self._error(code, recoverable=True)

    @staticmethod
    def _segment_id(event: dict) -> str:
        segment = event.get("segment_id")
        if not isinstance(segment, str) or not 0 < len(segment) <= 128:
            raise ValueError("invalid_segment_id")
        return segment

    async def _append(self, event: dict, ingress_epoch: int) -> None:
        segment = self._segment_id(event)
        pcm = decode_pcm16(event.get("audio_base64"), max_bytes=self.config.max_input_chunk_bytes)
        turn = self._active_input
        if turn is None or turn.segment_id != segment:
            if segment in self._turns:
                raise ValueError("append_after_commit")
            if turn is not None and not turn.sealed:
                raise ValueError("conflicting_segment")
            if len(self._turns) >= MAX_RETAINED_TURNS:
                oldest_id, oldest = next(iter(self._turns.items()))
                if not oldest.output.completed or oldest.input_status == "pending":
                    raise GeminiTransportError("gemini_turn_retention_limit")
                del self._turns[oldest_id]
            unresolved = self._pending_responses[-1] if self._pending_responses else None
            pending_native_tool = unresolved is not None and any(
                call.namespace == self._native_namespace and call.turn_number == unresolved.number
                and not call.delivered for call in self.operations.calls.values()
            )
            if ((turn is not None and turn.sealed and not turn.released)
                    or (unresolved is not None and not unresolved.provider_started)
                    or pending_native_tool):
                # Native start/start and immediate end/start can merge or lose
                # speech before generation begins. Never replay either segment.
                # A cancelled native tool response can also cause this model to
                # regenerate the old action despite an authoritative receipt.
                # Retire that native conversation, not the admitted domain task.
                await self._reset_before_new_input(
                    reason="interrupted_tool_response" if pending_native_tool else "superseded_before_response",
                )
            await self._cancel_output()
            self._proactive_gate.reset_for_user_turn()
            self._turn_number += 1
            turn = GeminiInputTurn(
                segment_id=segment, number=self._turn_number, ingress_epoch=ingress_epoch,
                connection_generation=self._connection_generation,
                native_namespace=self._native_namespace,
                output=GeminiOutputTurn(self.context.audio_session_id, self._connection_generation, self._turn_number),
            )
            self._turns[segment] = turn
            self._active_input = turn
            self._pending_transcripts.append(turn)
            if self._context_capture is not None:
                self._context_capture.input_admitted(turn.input_item_id)
            self.recovery.handle = None
            await self.connection.send_event({"realtimeInput": {"activityStart": {}}})
        elif turn.sealed:
            raise ValueError("append_after_commit")
        await self.connection.send_event({"realtimeInput": {"audio": {
            "data": event["audio_base64"], "mimeType": f"audio/pcm;rate={self.config.input_sample_rate}",
        }}})
        turn.input_bytes += len(pcm)

    async def _commit(self, event: dict, received_at: float | None = None) -> None:
        segment = self._segment_id(event)
        turn = self._turns.get(segment)
        if turn is not None and turn.sealed:
            return
        if turn is None or turn.input_bytes == 0:
            raise ValueError("empty_audio_commit")
        if turn is not self._active_input:
            raise ValueError("conflicting_segment")
        turn.sealed = True
        turn.committed_at = time.monotonic()
        turn.commit_received_at = received_at if received_at is not None else turn.committed_at
        turn.preparation_deadline = turn.committed_at + self.config.preparation_timeout_seconds
        turn.cutoff_at = datetime.now(UTC).isoformat()
        self._trace_turn("input_commit_admitted", turn, cutoff_at=turn.cutoff_at,
                         client_event_id=event.get("client_event_id"))
        turn.timing = {
            name: event[name] for name in ("audio_started_at", "audio_ended_at")
            if isinstance(event.get(name), str) and len(event[name]) <= 64
        }
        self._marker_turn = turn
        turn.preparation_task = self._spawn(self._prepare_and_enqueue(turn))

    def _current_preparation(self, turn: GeminiInputTurn) -> bool:
        return (not self._closed.is_set() and turn is self._active_input
                and not turn.cancelled and not turn.released
                and turn.ingress_epoch == self._ingress_epoch
                and turn.connection_generation == self._connection_generation)

    async def _prepare_and_enqueue(self, turn: GeminiInputTurn) -> None:
        self._trace_turn("preparation_started", turn)
        try:
            # Reserve part of the single total budget for the required context
            # write and end marker. A timeout never starts a new budget.
            reserve = min(.05, self.config.preparation_timeout_seconds / 5)
            try:
                async with asyncio.timeout_at(turn.preparation_deadline - reserve):
                    prepared = await self._prepare_turn(turn)
            except (TimeoutError, OptionalPreparationUnavailable):
                prepared = self._fallback_preparation(turn)
                self._trace_turn("preparation_fallback", turn, reason="optional_context_unavailable")
            self._trace_turn("preparation_finished", turn, status=prepared.evidence_status)
            if self._current_preparation(turn):
                await self._queue.put(("prepared", (turn, prepared)))
            else:
                self._trace_turn("response_rejected", turn, reason="preparation_superseded")
        except asyncio.CancelledError:
            self._trace_turn("preparation_finished", turn, status="cancelled")
            raise
        except Exception as error:
            code = error.code if isinstance(error, GeminiTransportError) else "gemini_preparation_failed"
            self._trace_turn("preparation_finished", turn, status="failed", reason=code)
            if self._current_preparation(turn):
                await self._queue.put(("preparation_failed", (turn, code)))

    async def _prepare_turn(self, turn: GeminiInputTurn) -> PreparedInput:
        # Snapshot the current-frame reference before any slower context I/O.
        # Never substitute a frame that arrived after this turn's fixed cutoff.
        image, state = await self._context_service.select_current(
            namespace=turn.connection_generation, not_after=datetime.fromisoformat(turn.cutoff_at),
        )
        self._trace_image("image_selected", turn, image, state)
        shared = None
        try:
            shared = await self._context_service.shared_payload()
        except Exception:
            pass
        payload = self._context_payload(turn, shared=shared)
        return PreparedInput(
            bounded_context(payload, max_characters=MAX_CONTEXT_CHARACTERS),
            evidence_status="available" if image is not None else "unavailable",
            image=image, image_state=state,
        )

    def _context_payload(self, turn: GeminiInputTurn, *, shared=None) -> dict[str, Any]:
        return {
            "kind": "jarvis_application_context", "segment_id": turn.segment_id,
            "cutoff_at": turn.cutoff_at, "current_visual_evidence": "unavailable",
            "memory_context": shared or "unavailable",
            # This mandatory local receipt ledger is also retained on optional
            # preparation timeout. Running/unknown must never become 'no actions'.
            "operation_context": self.operations.snapshot(),
        }

    def _fallback_preparation(self, turn: GeminiInputTurn) -> PreparedInput:
        return PreparedInput(bounded_context(
            self._context_payload(turn), max_characters=MAX_CONTEXT_CHARACTERS,
        ), evidence_status="unavailable")

    async def _release_turn(self, turn: GeminiInputTurn, prepared: PreparedInput) -> None:
        if not self._current_preparation(turn):
            self._trace_turn("response_rejected", turn, reason="preparation_superseded")
            return
        if not isinstance(prepared, PreparedInput) or len(prepared.context_text) > MAX_CONTEXT_CHARACTERS:
            raise GeminiTransportError("gemini_preparation_not_representable")
        async with asyncio.timeout_at(turn.preparation_deadline):
            effective_evidence = await self._write_prepared_context(turn, prepared)
            if not self._current_preparation(turn):
                return
            if len(self._pending_responses) >= MAX_PENDING_RESPONSES:
                raise GeminiTransportError("gemini_pending_response_limit")
            self._pending_responses.append(turn)
            self._trace_turn("response_admitted", turn)
            # The ledger is installed before the write; a failed/ambiguous write
            # is terminal and is never retried on another connection.
            await self.connection.send_event({"realtimeInput": {"activityEnd": {}}})
            self._trace_turn("response_write_finished", turn, status="written")
            turn.released = True
            turn.released_at = time.monotonic()
            turn.response_timeout_task = self._spawn(self._response_deadline(turn))
            turn.evidence_status = effective_evidence or prepared.evidence_status
            LOGGER.info("Gemini input released audio_session_id=%s turn=%s preparation_ms=%.1f evidence=%s",
                        self.context.audio_session_id, turn.number,
                        (time.monotonic() - turn.committed_at) * 1000, turn.evidence_status)

    async def _response_deadline(self, turn: GeminiInputTurn) -> None:
        await asyncio.sleep(self.config.response_timeout_seconds)
        if not turn.output.completed and not self._closed.is_set():
            await self._queue.put(("response_timeout", turn))

    async def _write_prepared_context(self, turn: GeminiInputTurn, prepared: PreparedInput) -> str | None:
        image, state = prepared.image, prepared.image_state
        if image is not None:
            try:
                await self._context_service.revalidate(image)
            except ImageUnavailableError as error:
                image, state = None, error.code
        self._trace_image("image_prewrite", turn, image, state)
        if not self._current_preparation(turn):
            return
        if image is not None and state != "reuse":
            # Realtime audio/video/text are concurrent provider streams: writing
            # a video frame before activityEnd does not guarantee it is included
            # in that answer. Append these occasional stills to conversation
            # content instead, without requesting generation. The existing end
            # marker remains the only answer trigger. Native probes cover this
            # mixed-input sequence; socket-write success is not model vision ACK.
            await self.connection.send_event({"clientContent": {
                "turns": [{"role": "user", "parts": [self._context_service.inline_part(image)]}],
                "turnComplete": False,
            }})
            self._context_service.mark_delivered(image, namespace=turn.connection_generation)
            if self._context_capture is not None:
                self._context_capture.image_delivery(
                    image.resolved.reference, "socket_written",
                    f"g{turn.connection_generation}:frame:{image.resolved.reference.frame_id}",
                )
            self._trace_image("image_socket_written", turn, image, "written_not_acknowledged")
        if not self._current_preparation(turn):
            return
        text = prepared.context_text
        try:
            payload = json.loads(text)
        except ValueError:
            payload = None
        if isinstance(payload, dict) and payload.get("kind") == "jarvis_application_context":
            payload["operation_context"] = self.operations.snapshot()
            payload["current_visual_evidence"] = "available" if image is not None else "unavailable"
            payload["live_view"] = {
                "status": ("reused" if state == "reuse" else "included") if image is not None else "unavailable",
                "reason": state,
                "image": image.resolved.reference.to_metadata() if image is not None else None,
                "meaning": "Use adequate current pixels already supplied. Fetch only when needed evidence is missing.",
            }
            text = bounded_context(payload, max_characters=MAX_CONTEXT_CHARACTERS)
        await self.connection.send_event({"realtimeInput": {"text": text}})
        if isinstance(payload, dict) and payload.get("kind") == "jarvis_application_context":
            return "available" if image is not None else "unavailable"
        return prepared.evidence_status

    async def _cancel_output(self) -> None:
        if self._active_input is not None and self._active_input.sealed and not self._active_input.released:
            self._active_input.cancelled = True
            if self._active_input.preparation_task is not None:
                self._active_input.preparation_task.cancel()
        for turn in self._pending_responses:
            turn.cancelled = True
            if turn.proactive is not None:
                self.proactive.finish(turn.proactive, "interrupted",
                                      response_id=turn.output.response_id, reason="speech_interrupted")
            self.operations.cancel(namespace=turn.native_namespace, turn_number=turn.number)
            for event in turn.output.interrupt(source="client"):
                await self._emit(event)

    async def _reset_before_new_input(self, *, reason: str = "superseded_before_response") -> None:
        self._trace.event("provider_reset_started", reason=reason,
                          connection_generation=self._connection_generation)
        await self._cancel_output()
        old_generation = self._connection_generation
        if self._provider_task is not None:
            self._provider_task.cancel()
            await asyncio.gather(self._provider_task, return_exceptions=True)
        await self.connection.close()
        for turn in self._turns.values():
            if (
                turn.connection_generation == old_generation
                and not turn.input_finalized and turn.input_bytes > 0
            ):
                turn.input_status = "unavailable_after_reset"
                if turn.sealed:
                    self._trace.event(
                        "input_transcript_unavailable", segment_id=turn.segment_id,
                        turn_number=turn.number, connection_generation=old_generation,
                        reason=reason,
                    )
            if turn.connection_generation == old_generation and not turn.output.completed:
                turn.cancelled = True
                if turn.response_timeout_task is not None:
                    turn.response_timeout_task.cancel()
                for event in turn.output.cancel_locally(reason=reason):
                    await self._emit(event)
                if turn.output.transcript:
                    await self._assistant_transcript_finalized(turn)
        self._pending_responses.clear()
        self._pending_transcripts.clear()
        self.recovery.clear()
        if self._recovery_timer is not None:
            self._recovery_timer.cancel()
            self._recovery_timer = None
        self._connection_generation += 1
        self._native_namespace += 1
        self.connection = self._connection_factory(config=self.config)
        await self._connect_current()
        if self._closed.is_set():
            return
        self._provider_task = self._spawn(self._provider_loop(self.connection, self._connection_generation))
        await self._emit({"type": "audio.provider.reconnected", "provider": "gemini",
                          "audio_session_id": self.context.audio_session_id,
                          "connection_generation": self._connection_generation,
                          "reason": reason})

    async def _handle_provider(self, event: dict) -> None:
        tool_owner = self._pending_responses[0] if self._pending_responses else None
        cancellation = event.get("toolCallCancellation")
        if isinstance(cancellation, dict):
            ids = cancellation.get("ids", [])
            if not isinstance(ids, list) or any(not isinstance(item, str) for item in ids):
                raise GeminiTransportError("gemini_invalid_tool_cancellation")
            self.operations.cancel(namespace=self._native_namespace, native_ids=ids)
        update = event.get("sessionResumptionUpdate")
        if isinstance(update, dict):
            self.recovery.observe(update, checkpoint_safe=self._checkpoint_safe())
        content = event.get("serverContent", {})
        transcription = content.get("inputTranscription", {})
        if isinstance(transcription, dict) and isinstance(transcription.get("text"), str):
            await self._final_input(transcription["text"])
        if content and self._pending_responses:
            turn = self._pending_responses[0]
            if (turn.ingress_epoch < self._ingress_epoch
                    or (turn.proactive is not None and turn.proactive.delivery_attempt is not None
                        and turn.proactive.delivery_attempt.revoked)):
                turn.cancelled = True
                for emitted in turn.output.interrupt(source="client"):
                    await self._emit(emitted)
            if content.get("modelTurn") or content.get("outputTranscription"):
                if not turn.provider_started:
                    self._trace_turn("provider_response_started", turn)
                turn.provider_started = True
            # Duplicate/late terminal-only events must not finish a new output
            # that has not reached its own generation/interruption boundary.
            terminal_allowed = (turn.output.audio_done or content.get("generationComplete")
                                or content.get("interrupted") or turn.output.interrupted)
            if content.get("turnComplete") and not terminal_allowed:
                content = {k: v for k, v in content.items() if k != "turnComplete"}
            if content.get("modelTurn") and turn.first_provider_audio_at is None:
                if any(part.get("inlineData", {}).get("mimeType", "").startswith("audio/")
                       for part in content["modelTurn"].get("parts", [])):
                    turn.first_provider_audio_at = time.monotonic()
            for emitted in turn.output.consume({"serverContent": content}):
                if emitted["type"] == "audio.output.delta" and (
                    turn.ingress_epoch < self._ingress_epoch
                    or (turn.proactive is not None and turn.proactive.delivery_attempt is not None
                        and turn.proactive.delivery_attempt.revoked)
                ):
                    continue
                sent = await self._emit(emitted)
                if sent is not False and not self._sender.closed and not self._closed.is_set():
                    turn.output.record_forwarded(emitted)
                    if emitted["type"] == "audio.output.delta" and turn.first_forwarded_audio_at is None:
                        turn.first_forwarded_audio_at = time.monotonic()
            if self._context_capture is not None:
                self._context_capture.gemini_output(turn.output)
            if turn.output.completed:
                self._trace_turn("provider_response_completed", turn, interrupted=turn.output.interrupted)
                self._pending_responses.popleft()
                self._settle_output(turn)
                if turn.response_timeout_task is not None:
                    turn.response_timeout_task.cancel()
                if turn.output.transcript:
                    await self._assistant_transcript_finalized(turn)
        elif content.get("modelTurn") or content.get("outputTranscription"):
            raise GeminiTransportError("gemini_unsolicited_output")
        if event.get("toolCall"):
            if self._pending_responses:
                if not self._pending_responses[0].provider_started:
                    self._trace_turn("provider_response_started", self._pending_responses[0])
                self._pending_responses[0].provider_started = True
            await self._handle_tool_call(event["toolCall"], turn=tool_owner)
        if event.get("goAway"):
            deadline = self.recovery.request(event["goAway"], now=time.monotonic(),
                                             max_wait=self.config.response_timeout_seconds)
            if self._recovery_timer is not None:
                self._recovery_timer.cancel()
            self._recovery_timer = self._spawn(self._recovery_deadline(deadline))

    async def _final_input(self, text: str) -> None:
        if not text:
            return
        if not self._pending_transcripts:
            raise GeminiTransportError("gemini_unattributed_input_transcript")
        turn = self._pending_transcripts.popleft()
        turn.input_transcript = text
        turn.input_finalized = True
        turn.input_status = "finalized"
        await self._input_transcript_finalized(turn)
        await self._emit({
            "type": "user.transcript.done", "text": text, "role": "user",
            "item_id": turn.input_item_id, "segment_id": turn.segment_id,
            "audio_session_id": self.context.audio_session_id,
        })

    async def _input_transcript_finalized(self, turn: GeminiInputTurn) -> None:
        if self._context_capture is not None:
            self._context_capture.input_final(turn.input_item_id, turn.input_transcript)
        self.journal.submit(FinalTranscript(
            logical_item_id=f"turn:{turn.number}:user", role="user", content=turn.input_transcript,
            # Message creation follows transcript finalization, like the shared
            # repository's existing flow. Original speech timing stays explicit
            # in metadata; it must not backdate a late message before the current
            # experience event and cause the segmenter to reject its evidence.
            occurred_at=datetime.now(UTC),
            provider_item_id=turn.input_item_id,
            metadata={"segment_id": turn.segment_id, "completion_state": "finalized",
                      "original_connection_generation": turn.connection_generation,
                      "input_cutoff_at": turn.cutoff_at,
                      "timing": dict(turn.timing), "experience_eligible": True},
        ))

    async def _assistant_transcript_finalized(self, turn: GeminiInputTurn) -> None:
        output = turn.output
        if self._context_capture is not None:
            self._context_capture.gemini_output(output)
        eligible = (output.completed and not output.interrupted and output.forwarded_audio_bytes > 0
                    and output.forwarded_audio_bytes == output.provider_audio_bytes)
        content = output.transcript
        if not eligible:
            content = "[Diagnostic: interrupted or unforwarded generated output; not a heard conversation.] " + content
        self.journal.submit(FinalTranscript(
            logical_item_id=f"turn:{turn.number}:assistant", role="assistant", content=content,
            occurred_at=datetime.now(UTC), provider_item_id=output.item_id,
            provider_response_id=output.response_id, eligible_for_experience=eligible,
            metadata={"segment_id": turn.segment_id, "experience_eligible": eligible,
                      "response_kind": ("visual_interrupt" if turn.proactive and turn.proactive.kind == "visual"
                                        else turn.proactive.kind if turn.proactive else "normal_user_turn"),
                      "proactive_event_id": str(turn.proactive.event_id) if turn.proactive else None,
                      "completion_state": "interrupted" if output.interrupted else "completed",
                      "delivery_status": "forwarded" if eligible else "incomplete",
                      "playback_status": "truncated" if turn.playback_end_ms is not None else "unconfirmed",
                      "playback_end_ms": turn.playback_end_ms,
                      "provider_audio_bytes": output.provider_audio_bytes,
                      "forwarded_audio_bytes": output.forwarded_audio_bytes},
        ))

    async def _handle_tool_call(self, payload: dict, *, turn: GeminiInputTurn | None) -> None:
        calls = payload.get("functionCalls")
        if turn is None or not isinstance(calls, list) or not calls or len(calls) > 8:
            raise GeminiTransportError("gemini_unattributed_tool_call")
        received = []
        for call in calls:
            if not isinstance(call, dict):
                raise GeminiTransportError("gemini_invalid_tool_call")
            operation, created = self.operations.receive(
                call, namespace=turn.native_namespace, turn_number=turn.number,
            )
            if created:
                received.append(operation)
        if len(self._domain_tasks) + len(received) > 8:
            raise GeminiTransportError("gemini_tool_backlog_limit")
        if turn.tool_budget is None:
            turn.tool_budget = ToolCallBudget(
                self.settings.JARVIS_MAX_TOOL_CALLS_PER_RESPONSE,
                self.settings.JARVIS_MAX_REPEATED_TOOL_CALLS_PER_RESPONSE,
            )
        for call in received:
            if turn.proactive is not None:
                # A prepared speech message grants no permission to run actions.
                call.settled, call.reason = True, "proactive_speech_only"
                await self._deliver_tool_result(call, turn, {
                    "status": "failed", "error": {"code": "proactive_speech_only"},
                    "message": "Speak the supplied message without tools. No operation was started.",
                }, None)
                continue
            task = asyncio.create_task(self._execute_tool(call, turn))
            call.task = task
            self._domain_tasks.add(task)
            task.add_done_callback(self._domain_tasks.discard)

    async def _execute_tool(self, call: NativeCall, turn: GeminiInputTurn) -> None:
        image = None
        if self._context_capture is not None:
            self._context_capture.bind_call(call.execution_id, turn.input_item_id if turn.proactive is None else None)
        try:
            async with asyncio.timeout(TOOL_TIMEOUT_SECONDS):
                result = await self.operations.execute(
                    call, budget=turn.tool_budget,
                    still_current=lambda: (not self._closed.is_set() and not turn.cancelled
                                           and turn.ingress_epoch == self._ingress_epoch
                                           and turn.connection_generation == self._connection_generation),
                )
                if not call.delivery_cancelled:
                    try:
                        async with asyncio.timeout(self.settings.VISUAL_IMAGE_RETRIEVAL_TIMEOUT_SECONDS):
                            image = await self._context_service.resolve_tool_image(call.name, result)
                    except (ImageUnavailableError, TimeoutError) as error:
                        code = error.code if isinstance(error, ImageUnavailableError) else "image_resolution_timeout"
                        result = {"status": "failed", "image_status": "unavailable", "error": {"code": code}}
        except asyncio.CancelledError:
            raise
        except Exception:
            result = {"status": "failed", "error": {"code": "operation_outcome_unavailable"},
                      "message": "The operation outcome is uncertain. Use status lookup; do not repeat it to check."}
        if not self._closed.is_set():
            await self._queue.put(("tool_ready", (call, turn, result, image)))

    async def _deliver_tool_result(self, call: NativeCall, turn: GeminiInputTurn, result: dict, image: PreparedImage | None) -> None:
        def current():
            return (not call.delivery_cancelled and not call.delivered and not turn.cancelled
                    and call.namespace == self._native_namespace and turn.connection_generation == self._connection_generation
                    and turn.ingress_epoch == self._ingress_epoch)

        if not current():
            return
        if call.mutates_state:
            if call.state == "unknown":
                result = {"status": "failed", "error": {"code": "operation_outcome_unknown"}}
            result = {**result, "application_operation_receipt": self.operations.receipt(call)}
        response = {"id": call.native_id, "name": call.name, "response": result}
        if image is not None:
            try:
                async with asyncio.timeout(self.settings.VISUAL_IMAGE_RETRIEVAL_TIMEOUT_SECONDS):
                    await self._context_service.revalidate(image)
            except (ImageUnavailableError, TimeoutError) as error:
                code = error.code if isinstance(error, ImageUnavailableError) else "image_resolution_timeout"
                response["response"] = {"status": "failed", "image_status": "unavailable", "error": {"code": code}}
                image = None
            else:
                response["parts"] = [self._context_service.inline_part(image)]
                response["response"] = {**result, "image_status": "included_in_response"}
        if not current():
            return
        await self.connection.send_event({"toolResponse": {"functionResponses": [response]}})
        call.delivered = True
        if image is not None:
            self._context_service.mark_delivered(image, namespace=turn.connection_generation)
            if self._context_capture is not None:
                self._context_capture.image_delivery(
                    image.resolved.reference, "socket_written",
                    f"g{turn.connection_generation}:tool:{call.execution_id}",
                )

    async def _persistence_failure(self, code: str) -> None:
        if not self._closed.is_set():
            try:
                self._queue.put_nowait(("persistence_error", code))
            except asyncio.QueueFull:
                self._mark_closed(code)

    def notify_visual_frame_available(self) -> None:
        # The shared frame window already owns the frame. Read at the next user
        # turn or explicit image tool; never start inference on the event bus.
        if not self._closed.is_set():
            self._visual_frame_notifications += 1

    async def speak_visual_interrupt(
        self, *, owner_id: str, jarvis_session_id: UUID, source_received_at: datetime | None,
        message: str, interrupt_id: UUID, visual_frame_id: UUID | None = None,
        source_captured_at: datetime | None = None, reason: str | None = None,
        confidence: float | None = None,
    ) -> bool:
        return await self._request_proactive(
            kind="visual", event_id=interrupt_id, message=message,
            fields={"owner_id": owner_id, "jarvis_session_id": jarvis_session_id,
                    "source_received_at": source_received_at, "visual_frame_id": visual_frame_id,
                    "source_captured_at": source_captured_at, "reason": reason, "confidence": confidence},
        )

    async def speak_reminder(self, *, message: str, reminder_id: UUID, reminder_event_id: UUID) -> None:
        accepted = await self._request_proactive(kind="reminder", event_id=reminder_event_id,
                                                message=message, fields={"reminder_id": reminder_id})
        if not accepted:
            # The existing manager ignores this callback's return value.
            raise GeminiTransportError("gemini_reminder_not_accepted")

    async def speak_prospective_interrupt(
        self, *, message: str, event_id: UUID, subject_type: str, subject_id: UUID,
        on_outcome: Callable[[str, str | None], Awaitable[None]],
        delivery_attempt: ProspectiveDeliveryAttempt | None = None,
    ) -> bool:
        if delivery_attempt is not None:
            delivery_attempt.on_revoke = lambda: self._clear_prospective_playback(event_id)
        return await self._request_proactive(
            kind="prospective", event_id=event_id, message=message,
            fields={"subject_type": subject_type, "subject_id": subject_id}, on_outcome=on_outcome,
            delivery_attempt=delivery_attempt,
        )

    async def _clear_prospective_playback(self, event_id: UUID) -> None:
        matching = [
            (turn.output.response_id, turn.output.item_id)
            for turn in self._pending_responses
            if turn.proactive is not None and turn.proactive.event_id == event_id
            and turn.output.item_id is not None
        ]
        for response_id, item_id in matching:
            await self._emit({
                "type": "audio.output.clear", "provider": "gemini",
                "response_id": response_id, "item_id": item_id,
                "reason": "prospective_revoked",
            })

    async def _request_proactive(self, **kwargs) -> bool:
        delivery, created = self.proactive.request(**kwargs)
        if delivery is None:
            return False
        if created:
            try:
                self._queue.put_nowait(("proactive", delivery))
            except asyncio.QueueFull:
                self.proactive.finish(delivery, "failed", reason="session_busy")
        try:
            return await asyncio.shield(delivery.accepted)
        except asyncio.CancelledError:
            delivery.cancelled = True
            raise

    def _proactive_rejection(self, delivery: ProactiveDelivery) -> str | None:
        if delivery.delivery_attempt is not None and delivery.delivery_attempt.revoked:
            return delivery.delivery_attempt.reason or "subject_no_longer_current"
        active = self._active_input
        user_busy = ((self._ingress_segment is not None
                      and (active is None or self._ingress_segment != active.segment_id))
                     or (active is not None and (not active.sealed or not active.released)))
        early_answer = any(not turn.provider_started and not turn.cancelled
                           for turn in self._pending_responses if turn.proactive is None)
        activity = GenericSpeechActivity(
            closed=self._closed.is_set(), transport_ready=self._ready,
            user_audio_pending=user_busy, normal_turn_pending=early_answer,
            speech_pending_or_active=bool(self._pending_responses),
            last_user_turn_settled_at=self._last_user_turn_settled_at,
            last_user_turn_settled_monotonic=self._last_user_turn_settled_monotonic,
        )
        if delivery.cancelled or delivery.outcome is not None or activity.closed or not activity.transport_ready:
            return "session_unavailable"
        if self.recovery.deadline is not None:
            return "provider_recovering"
        if user_busy or early_answer:
            return "user_turn_busy"
        if self._preparing_proactive not in (None, delivery):
            return "proactive_busy"
        if any(t.proactive is not None and not t.cancelled for t in self._pending_responses):
            return "proactive_busy"
        if len(self._turns) >= MAX_RETAINED_TURNS or len(self._pending_responses) >= MAX_PENDING_RESPONSES:
            return "session_capacity"
        if delivery.kind == "visual":
            return generic_visual_speech_rejection(
                activity, source_matches=(delivery.fields["owner_id"] == self.context.owner_id
                                          and delivery.fields["jarvis_session_id"] == self.context.jarvis_session_id),
                source_received_at=delivery.fields["source_received_at"],
                now=datetime.now(UTC), monotonic_now=time.monotonic(),
            )
        return None

    async def _begin_proactive(self, delivery: ProactiveDelivery) -> None:
        rejection = self._proactive_rejection(delivery)
        if rejection is not None or not self._proactive_gate.try_begin():
            self._trace.event("response_rejected", event_id=delivery.event_id, kind=delivery.kind,
                              reason=rejection or "interrupt_chain_limit",
                              connection_generation=self._connection_generation)
            self.proactive.finish(delivery, "failed", reason=rejection or "interrupt_chain_limit")
            return
        delivery.ingress_epoch = self._ingress_epoch
        self._preparing_proactive = delivery
        task = asyncio.create_task(self._prepare_proactive(delivery))
        self._domain_tasks.add(task)
        task.add_done_callback(self._domain_tasks.discard)

    async def _prepare_proactive(self, delivery: ProactiveDelivery) -> None:
        prepared = False
        try:
            await self.proactive.prepare(delivery)
            prepared = True
        except asyncio.CancelledError:
            raise
        except Exception:
            pass
        if not self._closed.is_set():
            await self._queue.put(("proactive_prepared", (delivery, prepared)))

    async def _release_proactive(self, delivery: ProactiveDelivery, prepared: bool) -> None:
        if self._preparing_proactive is not delivery:
            return
        try:
            rejection = self._proactive_rejection(delivery)
            if not prepared or rejection or delivery.ingress_epoch != self._ingress_epoch:
                self._trace.event("response_rejected", event_id=delivery.event_id, kind=delivery.kind,
                                  reason=rejection or "preparation_superseded",
                                  connection_generation=self._connection_generation)
                self.proactive.finish(delivery, "failed", reason=rejection or "preparation_superseded")
                return
            unresolved = self._pending_responses[-1] if self._pending_responses else None
            if unresolved is not None and any(
                not call.delivered and call.namespace == self._native_namespace
                and call.turn_number == unresolved.number for call in self.operations.calls.values()
            ):
                await self._reset_before_new_input(reason="proactive_during_tool_response")
            await self._cancel_output()
            if (self._closed.is_set() or delivery.cancelled or delivery.ingress_epoch != self._ingress_epoch
                    or (delivery.delivery_attempt is not None and delivery.delivery_attempt.revoked)):
                self.proactive.finish(delivery, "failed", reason="preparation_superseded")
                return
            self._turn_number += 1
            segment = f"proactive:{delivery.kind}:{delivery.event_id}"
            if segment in self._turns:
                self.proactive.finish(delivery, "failed", reason="segment_identity_conflict")
                return
            turn = GeminiInputTurn(
                segment_id=segment, number=self._turn_number, ingress_epoch=self._ingress_epoch,
                connection_generation=self._connection_generation, native_namespace=self._native_namespace,
                output=GeminiOutputTurn(self.context.audio_session_id, self._connection_generation, self._turn_number),
                sealed=True, released=True, input_status="not_applicable", proactive=delivery,
            )
            self._turns[segment] = turn
            self._pending_responses.append(turn)
            self.recovery.handle = None
            delivery.response_id = turn.output.response_id
            delivery.write_started = True
            self._trace_turn("response_admitted", turn, event_id=delivery.event_id, kind=delivery.kind)
            await self.connection.send_event({"clientContent": {
                "turns": [{"role": "user", "parts": [{"text": json.dumps({
                    "kind": "jarvis_proactive_delivery", "delivery_kind": delivery.kind,
                    "event_id": str(delivery.event_id), "message": delivery.message,
                    "operation_context": self.operations.snapshot(),
                }, ensure_ascii=True)}]}], "turnComplete": True,
            }})
            delivery.submitted = True
            self._trace_turn("response_write_finished", turn, status="written", event_id=delivery.event_id)
            if not delivery.accepted.done():
                delivery.accepted.set_result(True)
            turn.released_at = time.monotonic()
            turn.response_timeout_task = self._spawn(self._response_deadline(turn))
            await self._emit({"type": f"{delivery.event_prefix}.started", **delivery.notifications,
                              "message": delivery.message, "response_id": delivery.response_id})
        finally:
            self._preparing_proactive = None

    def _settle_output(self, turn: GeminiInputTurn) -> None:
        if turn.proactive is None:
            self._last_user_turn_settled_at = datetime.now(UTC)
            self._last_user_turn_settled_monotonic = time.monotonic()
            return
        output = turn.output
        completed_audio = (output.forwarded_audio_bytes > 0
                           and output.forwarded_audio_bytes == output.provider_audio_bytes)
        revoked = turn.proactive.delivery_attempt is not None and turn.proactive.delivery_attempt.revoked
        outcome = ("interrupted" if output.interrupted or turn.cancelled or revoked else
                   "delivered" if output.completed and completed_audio else "failed")
        self.proactive.finish(turn.proactive, outcome, response_id=output.response_id)

    async def notify_working_memory_updated(self, revision: int) -> None:
        await self.context_updates.notify_working_memory_updated(revision)

    async def notify_owner_working_memory_updated(self, revision: int) -> None:
        await self.context_updates.notify_owner_working_memory_updated(revision)

    async def notify_memory_item_changed(self, event) -> None:
        await self.context_updates.notify_memory_item_changed(event)

    def _record_playback(self, event: dict) -> None:
        end_ms = event.get("audio_end_ms")
        if type(end_ms) is not int or end_ms < 0 or event.get("content_index", 0) != 0:
            raise ValueError("invalid_playback_progress")
        turn = next((item for item in self._turns.values() if item.output.item_id == event.get("item_id")), None)
        if turn is None or end_ms > turn.output.forwarded_audio_bytes * 1000 // 48000:
            raise ValueError("invalid_playback_progress")
        turn.playback_end_ms = end_ms
        if self._context_capture is not None:
            self._context_capture.playback(turn.output.item_id, end_ms=end_ms)

    def _trace_turn(self, name: str, turn: GeminiInputTurn, **fields) -> None:
        self._trace.event(name, segment_id=turn.segment_id, turn_number=turn.number,
                          connection_generation=turn.connection_generation,
                          response_id=turn.output.response_id, **fields)

    def _trace_image(self, name: str, turn: GeminiInputTurn, image: PreparedImage | None, state: str) -> None:
        reference = image.resolved.reference if image is not None else None
        self._trace_turn(name, turn, namespace=turn.connection_generation, status=state,
                         cutoff_at=turn.cutoff_at, frame_id=getattr(reference, "frame_id", None),
                         captured_at=getattr(reference, "captured_at", None),
                         server_received_at=getattr(reference, "server_received_at", None))

    async def _emit(self, event: dict) -> bool:
        if self._closed.is_set():
            return False
        attempt = None
        if event.get("type") in {"audio.output.delta", "audio.output.done", "transcript.delta", "transcript.done"}:
            turn = next((turn for turn in self._pending_responses
                         if turn.output.response_id == event.get("response_id")), None)
            if turn is not None and turn.proactive is not None:
                attempt = turn.proactive.delivery_attempt
        async with asyncio.timeout(self.config.write_timeout_seconds):
            forwarded = await self._sender.send_event(
                event, **({"should_send": lambda: not attempt.revoked} if attempt is not None else {}),
            )
        if self._context_capture is not None:
            self._context_capture.observe_output(event, forwarded=forwarded)
        return forwarded

    async def _error(self, code: str, *, recoverable: bool) -> None:
        await self._emit({"type": "error", "code": code,
                          "message": ("Gemini could not safely restore the conversation. Start a new audio session; "
                                      "audio and actions were not automatically replayed."
                                      if code.startswith(("gemini_recovery", "gemini_resume"))
                                      else "Gemini audio could not process this request."),
                          "recoverable": recoverable, "details": {"provider": "gemini"}})

    def _peer_departed(self) -> None:
        self._mark_closed("client_disconnect")

    def _mark_closed(self, reason: str) -> None:
        if not self._closed.is_set():
            self.operations.accepting = False
            self.proactive.stop()
            self.context_updates.stop()
            self.recovery.clear()
            self._close_reason = reason
            self._closed.set()

    def _ensure_close_task(self) -> None:
        if self._close_task is None:
            self._close_task = asyncio.create_task(self._close_resources())
            # External Stop must also release a dispatcher blocked in a send or
            # setup wait. run() distinguishes this owned stop from cancellation
            # by its caller, and still joins the same shielded cleanup task.
            if self._run_task is not None and self._run_task is not asyncio.current_task():
                self._run_task.cancel()

    async def prepare_for_resource_close(self) -> None:
        await self.close("service_shutdown")

    async def close(self, reason: str) -> None:
        self._mark_closed(reason)
        self._ensure_close_task()
        await asyncio.shield(self._close_task)

    async def _close_resources(self) -> None:
        self._ready = False
        await self.context_updates.close()
        tasks = tuple(self._tasks)
        for task in tasks:
            task.cancel()
        with suppress(Exception):
            async with asyncio.timeout(self.config.close_timeout_seconds):
                await asyncio.gather(*tasks, return_exceptions=True)
        await self.connection.close()
        # Admitted domain tasks are not in the media task set cancelled above.
        # Let them record an outcome before draining transcript evidence work.
        domain_tasks = tuple(self._domain_tasks)
        if domain_tasks:
            _, pending = await asyncio.wait(domain_tasks, timeout=DOMAIN_DRAIN_TIMEOUT_SECONDS)
            for task in pending:
                task.cancel()
            await asyncio.gather(*domain_tasks, return_exceptions=True)
        for turn in self._turns.values():
            if not turn.output.completed:
                turn.output.cancel_locally(reason=self._close_reason)
                if turn.output.transcript:
                    await self._assistant_transcript_finalized(turn)
            if self._context_capture is not None:
                self._context_capture.gemini_output(turn.output)
        await self.proactive.drain(timeout_seconds=DOMAIN_DRAIN_TIMEOUT_SECONDS)
        await self.journal.drain(timeout_seconds=DOMAIN_DRAIN_TIMEOUT_SECONDS)
        if self._context_capture is not None:
            await self._context_capture.close()
        with suppress(Exception):
            async with asyncio.timeout(self.config.close_timeout_seconds):
                await self._sender.send_event({
                    "type": "session.closed", "audio_session_id": self.context.audio_session_id,
                    "session_id": str(self.context.jarvis_session_id), "reason": self._close_reason,
                })
                await self._sender.close()

    def diagnostics(self) -> dict[str, Any]:
        return {"provider": "gemini", "model": self.config.model,
                "audio_session_id": self.context.audio_session_id,
                "connection_generation": self._connection_generation,
                "native_namespace": self._native_namespace, "recovery": self.recovery.diagnostics(),
                "closed": self._closed.is_set(), "close_reason": self._close_reason,
                "queued_events": self._queue.qsize(), "owned_tasks": len(self._tasks),
                "domain_tasks": len(self._domain_tasks), "conversation_persistence": self.journal.diagnostics(),
                "context_updates": self.context_updates.diagnostics(),
                **({"canonical_context": self._context_capture.diagnostics()} if self._context_capture is not None else {}),
                "proactive": self.proactive.diagnostics(),
                "visual_frame_notifications": self._visual_frame_notifications,
                "operation_states": {call.status_ref: call.state for call in self.operations.calls.values() if call.mutates_state},
                "turns": [{"segment_id": turn.segment_id,
                           "connection_generation": turn.connection_generation,
                           "released": turn.released, "cancelled": turn.cancelled,
                           "input_status": turn.input_status,
                           "input_bytes": turn.input_bytes, "cutoff_at": turn.cutoff_at,
                           "evidence_status": turn.evidence_status,
                           "playback_end_ms": turn.playback_end_ms,
                           "commit_received_at": turn.commit_received_at,
                           "preparation_started_at": turn.committed_at,
                           "released_at": turn.released_at,
                           "preparation_duration_ms": (
                               (turn.released_at - turn.committed_at) * 1000
                               if turn.released_at is not None and turn.committed_at is not None else None
                           ),
                           "first_provider_audio_at": turn.first_provider_audio_at,
                           "first_forwarded_audio_at": turn.first_forwarded_audio_at,
                           **turn.output.diagnostics()} for turn in self._turns.values()]}
