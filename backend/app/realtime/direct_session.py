"""Reuse Jarvis turn/tool/attention ownership with a direct WebRTC media path."""
from __future__ import annotations

import asyncio
import json as _native_control_json
import logging as _native_control_logging
import re as _native_control_re
from contextlib import asynccontextmanager
from dataclasses import replace
from uuid import uuid4

from app.assistant.context import AssistantContextBuilder
from app.assistant.memory_context import session_working_memory_revision
from app.attention.admission_diagnostics import record_admission
from app.realtime.events import ClientEventError
from app.realtime.instructions import build_realtime_instructions
from app.realtime.openai_client import build_session_update_event
from app.realtime.session import RealtimeAudioSession
from app.realtime.webrtc import OpenAIWebRTCConnection


STARTUP_CONTEXT_LOAD_TIMEOUT_SECONDS = 15.0


_NATIVE_CONTROL_LOGGER = _native_control_logging.getLogger(__name__)
_NATIVE_CONTROL_LOG_LIMIT = 128
_NATIVE_CONTROL_EVENTS = frozenset({
    "response.created", "response.done", "response.output_item.added",
    "response.output_item.created", "response.output_item.done", "response.output_item.completed",
    "output_audio_buffer.started", "output_audio_buffer.stopped", "output_audio_buffer.cleared",
    "audio.webrtc.response", "audio.webrtc.output_item", "audio.webrtc.started",
    "audio.webrtc.stopped", "audio.webrtc.cleared",
    "audio.input.ready", "audio.capture_mode.ready",
})
_NATIVE_CONTROL_STAGES = frozenset({
    "provider_handler_received", "direct_precheck_rejected", "direct_admitted",
    "direct_send_result", "direct_send_raised", "runtime_attempt", "runtime_rejected",
    "runtime_write_guard_rejected", "runtime_send_result", "runtime_send_raised",
})
_NATIVE_CONTROL_REASONS = frozenset({
    "session_closed", "context_revoked", "response_id_missing", "response_not_pending",
    "response_unknown", "response_interrupted", "response_completed", "invalid_output_item",
    "runtime_not_ready", "owner_send_guard", "notification_not_current",
    "write_cancelled", "write_exception",
    "input_control_not_current", "input_clear_unconfirmed",
})
_NATIVE_COORDINATOR_PHASES = frozenset({
    "empty", "starting", "active", "retiring", "closing", "closed", "degraded",
})
_NATIVE_COORDINATOR_REASONS = frozenset({
    "context_validity_rejected", "context_expired_before_write",
    "context_publication_deadline", "context_publication_cancelled", "context_publication_failed",
    "context_receipt_mismatch_or_unsupported", "context_receipt_unconfirmed",
    "route_start_cancelled", "route_start_failed", "route_start_invalidated",
    "route_close_cancelled", "route_close_failed", "source_retirement",
})


class DirectRealtimeAudioSession(RealtimeAudioSession):
    def __init__(self, *, offer, capture_mode="auto", **kwargs):
        connection = OpenAIWebRTCConnection(settings=kwargs["settings"],
            owner_id=kwargs["context"].owner_id, offer=offer, capture_mode=capture_mode)
        super().__init__(provider_connection=connection, **kwargs)
        self._direct_inputs = {}
        self._direct_committed = set()
        self._direct_output_pending = set()
        self._native_control_log_count = 0
        self._native_control_log_truncated = False
        self._direct_audio_responses = set()
        self._direct_tasks = set()
        self._direct_manual_segment_id = None
        self._direct_manual_clear_generation = None
        self._direct_initial_context = None
        self._direct_startup_context_task = None
        self._direct_startup_context_status = "not_started"
        self._direct_warming_initial = False
        self._direct_warming_context = None
        self._direct_context_load_serial = 0
        self._direct_startup_context_builder = AssistantContextBuilder(
            providers=tuple(provider for provider in kwargs["context_providers"] if provider.critical))

    async def _initialize_openai_session(self):
        if self._closed.is_set():
            return
        if self._voice_route_runtime is not None:
            self._direct_warming_initial = True
            self._direct_startup_context_status = "warming"
            task = asyncio.create_task(
                self._enrich_startup_context(self._direct_context_load_serial),
                name="realtime-startup-context")
            self._direct_startup_context_task = task
            try:
                # Connection, SDP and the actual first session.updated ACK are
                # still required. Only optional context/history leaves startup.
                await super()._initialize_openai_session()
            finally:
                if self._closed.is_set() or not self._realtime_context_ready.is_set():
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
            return
        # Context and provider connection are independent reads/setup. Keep the
        # existing guarded session update and readiness admission unchanged.
        context_task = asyncio.create_task(super()._build_assistant_context())
        self._direct_initial_context = context_task
        try:
            await super()._initialize_openai_session()
        finally:
            self._direct_initial_context = None
            if not context_task.done():
                context_task.cancel()
            await asyncio.gather(context_task, return_exceptions=True)

    async def _build_assistant_context(self):
        if self._voice_route_runtime is not None:
            if self._direct_warming_initial:
                self._direct_warming_initial = False
                context = await self._direct_startup_context_builder.build(
                    owner_id=self.context.owner_id, session=self.context.session, recent_messages=[])
                context = self._startup_context_availability(context, "warming")
                self._direct_warming_context = context
                return context
            # A newer full refresh takes precedence even while it is loading;
            # the startup snapshot must not overwrite a correction's refresh.
            self._direct_context_load_serial += 1
        task, self._direct_initial_context = self._direct_initial_context, None
        if task is not None:
            return await task
        return await super()._build_assistant_context()

    @staticmethod
    def _startup_context_availability(context, availability):
        return replace(context, voice_continuity={
            "availability": availability,
            "records": [],
            "meaning": [
                "Prior conversation, memory, preferences and reminders are not yet available here. "
                "Empty fields do not mean that no history, preferences or responsibilities exist.",
                "Use current conversation and independently supplied verified live context. "
                "Do not claim to remember missing details or resume an old task; ask when needed.",
            ],
        })

    async def _enrich_startup_context(self, serial):
        started_at, generation = self._now(), self._turn_generation
        try:
            # One owned load, with the original full provider set and canonical
            # voice-history reconstruction. No substitute summary or extra LLM.
            async with asyncio.timeout(STARTUP_CONTEXT_LOAD_TIMEOUT_SECONDS):
                context = await super()._build_assistant_context()
            context = replace(context, voice_continuity={
                **(context.voice_continuity or {}),
                "startup_snapshot": {
                    "read_started_at": started_at.isoformat(),
                    "read_started_turn_generation": generation,
                    "meaning": "Historical context, not a new request. Newer live conversation takes precedence.",
                },
            })
            await self._realtime_context_ready.wait()
            if await self._publish_startup_context(context, serial):
                self._direct_startup_context_status = "submitted"
                self._schedule_working_memory_expiry(context.working_memory)
                self._context_refresh.initialize(session_working_memory_revision(context.working_memory))
                # run() may have checked recovery before history was available.
                # Its canonical claim and turn-generation checks remain intact.
                await self._recover_tool_acknowledgment()
            else:
                self._direct_startup_context_status = "superseded"
        except asyncio.CancelledError:
            self._direct_startup_context_status = "cancelled"
            raise
        except Exception as error:
            self._direct_startup_context_status = "unavailable"
            self._trace.event("session_context_enrichment_unavailable", error_type=type(error).__name__)
            # A failed optional load is not an empty or successfully recalled
            # history. Replace the warming notice only if it still belongs to us.
            await self._realtime_context_ready.wait()
            warming = self._direct_warming_context
            if warming is not None:
                try:
                    await self._publish_startup_context(
                        self._startup_context_availability(warming, "unavailable"), serial)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    self._trace.event("session_context_enrichment_notice_unavailable")

    async def _publish_startup_context(self, context, serial):
        def current():
            return (not self._closed.is_set() and not self._context_revoked
                    and serial == self._direct_context_load_serial
                    and self._direct_warming_context is not None
                    and self._assistant_context is self._direct_warming_context)

        if not current():
            return False
        roots = context.source_payload_ids
        tools = self.tool_registry.openai_realtime_tools()
        instructions = build_realtime_instructions(
            context=context, live_view=self._visual_prompt_metadata(),
            tasks_enabled=any(tool.get("name") == "start_background_task" for tool in tools))

        @asynccontextmanager
        async def guard():
            if not current():
                yield
                return
            # Do not enroll a queued/stale history in remote-context ownership.
            # Existing originals stay locked through the actual provider write.
            async with self._source_context.context_guard(extra_sources=roots):
                yield

        def at_write():
            if not current():
                return None
            self._source_context.observe(roots)
            self._assistant_context = context
            return build_session_update_event(settings=self.settings, instructions=instructions, tools=tools)

        try:
            async with asyncio.timeout(self.settings.OPENAI_REALTIME_WRITE_TIMEOUT_SECONDS):
                sent = await self.openai.send_event(at_write, write_guard=guard)
            self._trace.event("session_context_enrichment_write", status="written" if sent else "superseded")
            return sent is True
        except Exception:
            if self._source_context.has_sources:
                self._begin_context_rebuild()
            raise

    async def _send_transport_setup(self):
        if not self._closed.is_set():
            await self._send_app_event("audio.webrtc.answer", sdp=self.openai.answer)

    async def _send_input_ready(self, kind, owner, **payload):
        provider, generation, runtime, route = owner
        current = (not self._closed.is_set() and not self._context_revoked
            and self.openai is provider and self._turn_generation == generation
            and self._voice_route_runtime is runtime
            and (runtime is None or runtime.request.route == route))
        if kind == "audio.input.ready":
            current = (current and self.openai.capture_mode == "manual" and self._user_audio_pending
                and self._direct_manual_segment_id == payload.get("segment_id")
                and self._direct_manual_clear_generation == generation)
        else:
            current = current and self.openai.capture_mode == payload.get("capture_mode")
        if not current:
            self._native_control_diagnostic("direct_precheck_rejected", kind,
                reason="input_control_not_current", send_result=False,
                segment_id=payload.get("segment_id"), turn_generation=generation)
            return False
        return await self._send_app_event(kind, **payload)

    def _input_ready_owner(self):
        runtime = self._voice_route_runtime
        return (self.openai, self._turn_generation, runtime,
                runtime.request.route if runtime is not None else None)

    async def _handle_client_event(self, event):
        kind = event.get("type")
        if self._context_revoked:
            return await super()._handle_client_event(event)
        if kind == "audio.input.append":
            raise ClientEventError("WebRTC microphone audio must not be sent over the control socket.")
        if kind == "audio.capture_mode":
            mode = event.get("capture_mode", event.get("mode"))
            if mode not in {"auto", "manual"}:
                raise ClientEventError("Unsupported capture mode.")
            async with self._response_admission_lock:
                await self._begin_user_turn_locked()
                self._settle_normal_turn(self._turn_generation)
                self.openai.capture_mode = mode
                self._direct_manual_segment_id = None
                self._direct_manual_clear_generation = None
                self._user_audio_pending = False
                owner = self._input_ready_owner()
                cleared = await self.openai.send_event({"type": "input_audio_buffer.clear"})
            if cleared is not True:
                self._native_control_diagnostic("direct_precheck_rejected", "audio.capture_mode.ready",
                    reason="input_clear_unconfirmed", send_result=False)
                return
            await self._send_session_update()
            await self._send_input_ready("audio.capture_mode.ready", owner, capture_mode=mode)
            return
        if kind == "audio.input.begin":
            if self.openai.capture_mode != "manual":
                raise ClientEventError("Manual speech begin is unavailable during semantic turn detection.")
            segment_id = event.get("segment_id")
            if not isinstance(segment_id, str) or not segment_id or len(segment_id) > 256:
                raise ClientEventError("Manual speech begin requires a segment identifier.")
            async with self._response_admission_lock:
                duplicate = self._direct_manual_segment_id is not None
                if duplicate and self._direct_manual_segment_id != segment_id:
                    raise ClientEventError("Manual speech begin conflicts with the active segment.")
                if not duplicate:
                    await self._begin_user_turn_locked()
                    self._user_audio_pending = True
                    self._direct_manual_segment_id = segment_id
                    self._direct_manual_clear_generation = None
                owner = self._input_ready_owner()
            if not duplicate:
                cleared = await self.openai.send_event({"type": "input_audio_buffer.clear"})
                if cleared is not True:
                    self._native_control_diagnostic("direct_precheck_rejected", "audio.input.ready",
                        reason="input_clear_unconfirmed", send_result=False,
                        segment_id=segment_id, turn_generation=owner[1])
                    return
                # A late clear may not ready a replacement segment/provider/turn.
                if (self.openai is owner[0] and self._turn_generation == owner[1]
                        and self._direct_manual_segment_id == segment_id):
                    self._direct_manual_clear_generation = owner[1]
            await self._send_input_ready("audio.input.ready", owner, segment_id=segment_id)
            return
        if kind == "audio.input.abort":
            if (self.openai.capture_mode == "manual"
                    and event.get("segment_id") == self._direct_manual_segment_id):
                async with self._response_admission_lock:
                    self._direct_manual_segment_id = None
                    self._direct_manual_clear_generation = None
                    self._user_audio_pending = False
                    self._settle_normal_turn(self._turn_generation)
                    await self.openai.send_event({"type": "input_audio_buffer.clear"})
            return
        if kind == "audio.input.commit" and self.openai.capture_mode != "manual":
            raise ClientEventError("Semantic turn detection owns audio commits in this session.")
        if kind == "audio.input.commit":
            if event.get("segment_id") != self._direct_manual_segment_id:
                raise ClientEventError("Manual audio commit does not match the active segment.")
            self._direct_manual_segment_id = None
            self._direct_manual_clear_generation = None
        return await super()._handle_client_event(event)

    async def _handle_openai_event(self, payload):
        kind, item_id = payload.get("type"), payload.get("item_id")
        response = payload.get("response")
        diagnostic_response_id = payload.get("response_id") or (
            response.get("id") if isinstance(response, dict) else None)
        self._native_control_diagnostic("provider_handler_received", kind, diagnostic_response_id)
        if self._closed.is_set() or self._context_revoked:
            self._native_control_diagnostic("direct_precheck_rejected", kind, diagnostic_response_id,
                reason="session_closed" if self._closed.is_set() else "context_revoked")
            return await super()._handle_openai_event(payload)
        if kind == "input_audio_buffer.speech_started" and self.openai.capture_mode == "auto":
            if not isinstance(item_id, str) or item_id in self._direct_inputs:
                return
            async with self._response_admission_lock:
                await self._begin_user_turn_locked()
                self._user_audio_pending = True
                request_id = uuid4().hex
                self._direct_inputs[item_id] = self._turn_generation
                if capture := self._context_capture:
                    capture.admit_openai_commit(request_id, self._turn_generation)
            while len(self._direct_inputs) > 256:
                self._direct_inputs.pop(next(iter(self._direct_inputs)))
            self._trace.event("provider_speech_started", item_id=item_id,
                              turn_generation=self._turn_generation)
            await self._send_app_event("audio.input.started", item_id=item_id)
            return
        if kind == "input_audio_buffer.speech_stopped":
            self._trace.event("provider_speech_stopped", item_id=item_id)
            await self._send_app_event("audio.input.stopped", item_id=item_id)
            return
        if kind == "input_audio_buffer.committed" and self.openai.capture_mode == "auto":
            await super()._handle_openai_event(payload)
            generation = self._direct_inputs.get(item_id)
            if generation != self._turn_generation or item_id in self._direct_committed:
                return
            self._direct_committed.add(item_id)
            if len(self._direct_committed) > 256:
                self._direct_committed.intersection_update(self._direct_inputs)
            async with self._response_admission_lock:
                if generation != self._turn_generation:
                    return
                self._user_audio_pending = False
            self._trace.event("input_commit_admitted", item_id=item_id, turn_generation=generation)
            task = asyncio.create_task(self._request_normal_response(turn_generation=generation))
            self._direct_tasks.add(task)
            task.add_done_callback(self._direct_request_finished)
            return
        if kind in {"output_audio_buffer.started", "output_audio_buffer.stopped", "output_audio_buffer.cleared"}:
            response_id = payload.get("response_id")
            if isinstance(response_id, str):
                if kind == "output_audio_buffer.started":
                    if response_id not in self._direct_output_pending:
                        self._native_control_diagnostic("direct_precheck_rejected", kind, response_id,
                            reason="response_not_pending")
                        return
                    self._direct_audio_responses.add(response_id)
                else:
                    self._direct_output_pending.discard(response_id)
                    self._direct_audio_responses.discard(response_id)
                # Provider output buffering is NOT a client-render or hearing receipt.
                self._trace.event(kind.replace(".", "_"), response_id=response_id,
                                  boundary="provider_webrtc_buffer")
                await self._send_native_control("audio.webrtc." + kind.rsplit(".", 1)[-1],
                                                response_id=response_id)
            else:
                self._native_control_diagnostic("direct_precheck_rejected", kind, response_id,
                    reason="response_id_missing")
            return
        if kind == "response.created":
            await super()._handle_openai_event(payload)
            response_id = (payload.get("response") or {}).get("id")
            state = self._active_responses.get(response_id)
            if state is not None and not state.interrupted:
                self._direct_output_pending.add(response_id)
                await self._send_native_control("audio.webrtc.response", response_id=response_id,
                                                turn_generation=self._turn_generation)
            else:
                self._native_control_diagnostic("direct_precheck_rejected", kind, response_id,
                    reason="response_unknown" if state is None else "response_interrupted")
            return
        if kind in {"response.output_item.added", "response.output_item.created",
                    "response.output_item.done", "response.output_item.completed"}:
            await super()._handle_openai_event(payload)
            response_id = payload.get("response_id") or self._current_response_id
            state = self._active_responses.get(response_id)
            item = payload.get("item")
            if (response_id in self._direct_output_pending and state is not None
                    and not state.interrupted and state.completed_response is None
                    and isinstance(item, dict) and isinstance(item.get("id"), str)):
                # The phone must not depend on a second event stream to learn
                # the phase/type that gates an already-admitted response.
                metadata = {key: item[key] for key in ("id", "type", "phase") if key in item}
                await self._send_native_control("audio.webrtc.output_item",
                                                response_id=response_id, item=metadata)
            else:
                reason = ("response_not_pending" if response_id not in self._direct_output_pending else
                    "response_unknown" if state is None else
                    "response_interrupted" if state.interrupted else
                    "response_completed" if state.completed_response is not None else "invalid_output_item")
                self._native_control_diagnostic("direct_precheck_rejected", kind, response_id, reason=reason)
            return
        if kind == "response.done":
            response = payload.get("response") or {}
            response_id = response.get("id")
            has_audio = response_id in self._direct_audio_responses or any(
                part.get("type") in {"audio", "output_audio"}
                for item in response.get("output", []) if isinstance(item, dict)
                for part in item.get("content", []) if isinstance(part, dict)
            )
            if response.get("status") != "completed" or not has_audio:
                self._direct_output_pending.discard(response_id)
            return await super()._handle_openai_event(payload)
        return await super()._handle_openai_event(payload)

    def _native_control_diagnostic(self, stage, event_type, response_id=None, *,
                                   reason=None, send_result=None, runtime=None,
                                   segment_id=None, turn_generation=None):
        """Bounded metadata only; handler receipt is not socket arrival or playback.

        Absence is inconclusive after the truncation marker. Logging is best effort
        and must never change provider admission, writes, or cancellation behavior.
        """
        try:
            if stage not in _NATIVE_CONTROL_STAGES or event_type not in _NATIVE_CONTROL_EVENTS:
                return
            if self._native_control_log_count >= _NATIVE_CONTROL_LOG_LIMIT:
                if not self._native_control_log_truncated:
                    self._native_control_log_truncated = True
                    _NATIVE_CONTROL_LOGGER.warning("native_control_diagnostic %s", _native_control_json.dumps({
                        "stage": "truncated", "record_limit": _NATIVE_CONTROL_LOG_LIMIT,
                        "absence_after_limit_is_inconclusive": True}, sort_keys=True))
                return
            self._native_control_log_count += 1
            state = self._active_responses.get(response_id) if isinstance(response_id, str) else None
            runtime = runtime if runtime is not None else self._voice_route_runtime
            status = getattr(getattr(runtime, "coordinator", None), "status", None)
            phase = getattr(getattr(status, "phase", None), "value", None)
            coordinator_reason = getattr(status, "reason", None)
            safe_response = (response_id if isinstance(response_id, str)
                and _native_control_re.fullmatch(r"resp_[A-Za-z0-9]{1,64}", response_id) else None)
            record = {
                "stage": stage, "event_type": event_type, "response_id": safe_response,
                "response_id_redacted": response_id is not None and safe_response is None,
                "session_closed": self._closed.is_set(), "context_revoked": bool(self._context_revoked),
                "response_pending": isinstance(response_id, str) and response_id in self._direct_output_pending,
                "response_active": state is not None,
                "response_interrupted": bool(getattr(state, "interrupted", False)),
                "response_dismissed": bool(getattr(state, "dismissed", False)),
                "response_completed": state is not None and getattr(state, "completed_response", None) is not None,
                "runtime_present": runtime is not None,
                "runtime_ready": bool(runtime.ready) if runtime is not None else None,
                "coordinator_phase": phase if phase in _NATIVE_COORDINATOR_PHASES else "unavailable",
                "coordinator_reason": (coordinator_reason if coordinator_reason in _NATIVE_COORDINATOR_REASONS
                    else "unrecognized_code" if coordinator_reason is not None else None),
                "reason": reason if reason in _NATIVE_CONTROL_REASONS else "unrecognized_code" if reason is not None else None,
                "send_result": send_result if type(send_result) is bool else None,
            }
            if event_type in {"audio.input.ready", "audio.capture_mode.ready"}:
                # Segment strings can be arbitrary user data. Only UUID-shaped
                # identities are retained; no payload, transcript or error text.
                safe_segment = (segment_id if isinstance(segment_id, str)
                    and _native_control_re.fullmatch(
                        r"[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", segment_id) else None)
                identity = getattr(status, "identity", None)
                record.update(segment_id=safe_segment,
                    segment_id_redacted=segment_id is not None and safe_segment is None,
                    turn_generation=turn_generation if type(turn_generation) is int and turn_generation >= 0 else None,
                    route_id=str(identity.route_id) if identity is not None else None,
                    route_generation=identity.route_generation if identity is not None else None)
            _NATIVE_CONTROL_LOGGER.warning("native_control_diagnostic %s",
                _native_control_json.dumps(record, sort_keys=True, separators=(",", ":")))
        except Exception:
            # A diagnostic sink failure must not become a media-control failure.
            return

    async def _send_native_control(self, event_type, **payload):
        response_id = payload.get("response_id")
        self._native_control_diagnostic("direct_admitted", event_type, response_id)
        try:
            sent = await self._send_app_event(event_type, **payload)
        except BaseException as error:
            self._native_control_diagnostic("direct_send_raised", event_type, response_id,
                reason="write_cancelled" if isinstance(error, asyncio.CancelledError) else "write_exception")
            raise
        self._native_control_diagnostic("direct_send_result", event_type, response_id, send_result=sent)
        return sent

    def _direct_request_finished(self, task):
        self._direct_tasks.discard(task)
        if not task.cancelled() and task.exception() is not None:
            self._mark_closing("webrtc_response_control_failed")

    async def _forward_output_media(self, payload, *, event_type):
        if event_type in {"response.output_audio.delta", "response.output_audio.done"}:
            # No PCM is relayed or counted as backend-forwarded audio. The existing
            # transcript path retains generated text with playback explicitly unknown.
            return
        return await super()._forward_output_media(payload, event_type=event_type)

    async def _cancel_response_locked(self, *, keep_image_calls=()):
        had_audio = bool(self._direct_audio_responses)
        await super()._cancel_response_locked(keep_image_calls=keep_image_calls)
        self._direct_output_pending.clear()
        self._direct_audio_responses.clear()
        if had_audio:
            if self._voice_route_runtime is not None:
                self._response_controls.clear_output()
            else:
                await self.openai.send_event({"type": "output_audio_buffer.clear"})

    async def try_attention(self, *args, **kwargs):
        if self._direct_output_pending:
            record_admission("foreground_busy", owner=self.context.owner_id,
                             session_id=self.context.jarvis_session_id)
            return None
        return await super().try_attention(*args, **kwargs)

    async def _close_resources(self, reason):
        startup = self._direct_startup_context_task
        tasks = (*tuple(self._direct_tasks), *((startup,) if startup is not None else ()))
        for task in tasks:
            if task is not asyncio.current_task():
                task.cancel()
        try:
            await super()._close_resources(reason)
        finally:
            await asyncio.gather(*(task for task in tasks if task is not asyncio.current_task()),
                                 return_exceptions=True)
