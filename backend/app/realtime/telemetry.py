"""Content-free lifecycle evidence. A successful write is not audible playback."""
from __future__ import annotations

from asyncio import CancelledError
from collections import OrderedDict
from contextlib import contextmanager
from datetime import UTC, datetime
import hashlib
import json
import hashlib
import logging
import math
import re
import time
from uuid import UUID, uuid4


LOGGER = logging.getLogger(__name__)
PROCESS_ID = uuid4().hex
MAX_RECENT_AUDIO_ITEMS = 256
MAX_LATENCY_RECORDS = 2048
TOOL_CONFIGURATION_NAMES = frozenset({
    "track_visual_memory", "track_episode_check", "list_episode_checks",
    "change_episode_check", "get_current_camera_view", "search_memory", "remember",
})
_TOKEN = re.compile(r"[A-Za-z0-9_.:-]{1,160}\Z")
_TOKENS = frozenset({
    "provider", "audio_session_id", "jarvis_session_id", "segment_id", "request_id",
    "response_id", "item_id", "client_event_id", "frame_id", "event_id", "focus_id",
    "reason", "status", "boundary", "event_type", "kind", "error_type",
    "origin", "dispatch_id", "phase", "source_snapshot_id", "route_id", "update_id",
    "configuration_dispatch_id",
})
_NUMBERS = frozenset({
    "connection_generation", "turn_generation", "turn_number", "namespace",
    "duration_ms", "audio_end_ms", "receive_wait_ms", "decode_dispatch_ms",
    "dequeued_monotonic_s",
    "tool_count", "other_tool_count",
    "guard_completed_pool_acquire_ms", "guard_completed_transaction_enter_ms",
    "guard_completed_memory_owner_lock_ms", "guard_completed_task_owner_lock_ms",
    "guard_completed_scope_revalidation_ms", "guard_completed_canonical_source_lock_ms",
    "guard_completed_dependency_revalidation_ms", "guard_completed_response_source_lock_ms",
    "guard_completed_unclassified_guard_ms",
})
_TIMES = frozenset({"captured_at", "server_received_at", "cutoff_at"})
_FLAGS = frozenset({"closed", "user_audio_pending", "normal_turn_pending", "response_requested", "interrupted",
                    "write_possible", "send_returned", "guard_present", "anchor_present"})
_LATENCY_EVENTS = frozenset({"provider_write_latency", "response_source_latency",
                             "response_preparation_latency", "provider_item_latency", "tool_followup_write"})
_ANCHOR_STATUSES = frozenset({"absent", "available", "unavailable", "omitted_byte_budget"})
_OUTCOME_FIELDS = frozenset({"memory_status", "source_ingestion_status", "error_code", "replayed",
    "quiet_effective_state", "quiet_revision", "quiet_suppresses_unsolicited", "quiet_accepted_revision",
    "quiet_delivery_propagation", "quiet_playback_stop_confirmed"})
_WIRE_EVENTS = {
    "session.created": "session_ready_forwarded",
    "session.closed": "session_closed_forwarded",
    "audio.provider.reconnecting": "provider_reconnecting_forwarded",
    "audio.provider.reconnected": "provider_reconnected_forwarded",
    "response.done": "response_completion_forwarded",
    "audio.output.clear": "playback_clear_forwarded",
    "audio.output.done": "audio_generation_done_forwarded",
}
_CLIENT_EVENTS = {
    "audio.input.commit": "input_commit_received",
    "response.create": "response_requested",
    "response.cancel": "response_cancel_requested",
    "playback.truncated": "playback_truncation_received",
    "session.close": "session_close_requested",
}

_CONTROL_EVENTS = frozenset({
    "audio.input.begin", "audio.input.commit", "audio.input.abort", "audio.capture_mode",
    "response.create", "response.cancel", "session.close", "session.context_cleared",
    "playback.truncated", "playback.completed", "input_audio_buffer.speech_started",
    "input_audio_buffer.speech_stopped", "input_audio_buffer.committed",
    "response.created", "response.done", "response.function_call_arguments.done",
    "response.output_item.done", "response.output_item.completed",
    "output_audio_buffer.started", "output_audio_buffer.stopped", "output_audio_buffer.cleared",
})
_SLOW_CONTROL_HANDLER_MS = 250.0


def safe_fields(fields: dict) -> dict:
    """Allow identifiers/codes and finite clocks, never arbitrary event payloads."""
    result = {}
    for key, value in fields.items():
        if key == "tool_names" and isinstance(value, (list, tuple)) and len(value) <= 64:
            result[key] = sorted({name for name in value if isinstance(name, str)
                                  and name in TOOL_CONFIGURATION_NAMES})
        elif key == "tool_configuration_sha256":
            if isinstance(value, str) and re.fullmatch(r"[a-f0-9]{64}", value):
                result[key] = value
        elif key in {"tool_scope", "configuration_state", "configuration_match", "tool_choice"}:
            allowed = {
                "tool_scope": {"session", "response"},
                "configuration_state": {"explicit", "inherited", "unavailable"},
                "configuration_match": {"configuration_only", "different", "unavailable"},
                "tool_choice": {"auto", "none", "required", "forced", "omitted", "unavailable"},
            }
            if isinstance(value, str) and value in allowed[key]:
                result[key] = value
        elif key in _TOKENS:
            if isinstance(value, UUID):
                value = str(value)
            if isinstance(value, str) and _TOKEN.fullmatch(value):
                result[key] = value
        elif key in _NUMBERS and type(value) in (int, float) and -1e15 <= value <= 1e15 and math.isfinite(value):
            result[key] = value
        elif key in _FLAGS and type(value) is bool:
            result[key] = value
        elif key == "anchor_status" and isinstance(value, str) and value in _ANCHOR_STATUSES:
            result[key] = value
        elif key == "instructions_sha256" and isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value):
            result[key] = value
        elif (key == "outcome_fields" and isinstance(value, (list, tuple)) and len(value) <= len(_OUTCOME_FIELDS)
                and all(isinstance(item, str) and item in _OUTCOME_FIELDS for item in value)):
            result[key] = sorted(set(value))
        elif key in _TIMES:
            try:
                parsed = value if isinstance(value, datetime) else datetime.fromisoformat(value)
                if parsed.tzinfo is not None:
                    result[key] = parsed.astimezone(UTC).isoformat()
            except (ValueError, TypeError):
                pass
    return result


def tool_configuration_fields(configuration) -> dict:
    """Fingerprint tools/choice only, never retain instructions or tool bodies.

    A provider echo can establish equal configuration, not which identical update
    caused it, model adoption, invocation, or successful execution.
    """
    unavailable = {"configuration_state": "unavailable"}
    try:
        if not isinstance(configuration, dict):
            return unavailable
        choice = configuration.get("tool_choice", "omitted")
        choice_label = (choice if isinstance(choice, str) and choice in
                        {"auto", "none", "required", "omitted"} else
                        "forced" if isinstance(choice, dict) else "unavailable")
        if "tools" not in configuration:
            return {"configuration_state": "inherited", "tool_choice": choice_label}
        tools = configuration["tools"]
        if not isinstance(tools, list) or len(tools) > 64 or any(not isinstance(t, dict) for t in tools):
            return unavailable
        names = [t.get("name") for t in tools]
        retained = sorted({n for n in names if isinstance(n, str) and n in TOOL_CONFIGURATION_NAMES})
        encoded = json.dumps({"tools": tools, "tool_choice": choice}, sort_keys=True,
                             ensure_ascii=True, separators=(",", ":"), allow_nan=False).encode("ascii")
        if len(encoded) > 262144:
            return unavailable
        return {"configuration_state": "explicit", "tool_choice": choice_label,
                "tool_count": len(tools), "tool_names": retained,
                "other_tool_count": sum(not isinstance(n, str) or n not in TOOL_CONFIGURATION_NAMES for n in names),
                "tool_configuration_sha256": hashlib.sha256(encoded).hexdigest()}
    except Exception:
        return unavailable


class RealtimeTrace:
    def __init__(self, *, provider: str, audio_session_id: str, jarvis_session_id: UUID) -> None:
        self.context = safe_fields(dict(provider=provider, audio_session_id=audio_session_id,
                                       jarvis_session_id=jarvis_session_id))
        # Old retired IDs may be logged again after eviction. Reports deduplicate
        # by identity too; memory use never grows with session duration.
        self._audio_items: OrderedDict[tuple, None] = OrderedDict()
        self._latency_records = 0

    def event(self, name: str, **fields) -> None:
        try:
            if not isinstance(name, str) or not _TOKEN.fullmatch(name):
                return
            record = {**safe_fields(fields), **self.context, "schema": 1, "event": name,
                      "process_id": PROCESS_ID, "at": datetime.now(UTC).isoformat(),
                      "monotonic_s": time.monotonic()}
            LOGGER.info("realtime_trace %s", json.dumps(record, separators=(",", ":")))
        except Exception:
            # Observation cannot change cancellation, admission or write results.
            pass

    def latency_event(self, name: str, **fields) -> None:
        """Bounded local-stage evidence, not provider acceptance or audible output."""
        try:
            if name not in _LATENCY_EVENTS or self._latency_records > MAX_LATENCY_RECORDS:
                return
            self._latency_records += 1
            if self._latency_records > MAX_LATENCY_RECORDS:
                self.event("latency_diagnostics_truncated", status="truncated", reason="record_limit")
                return
            self.event(name, **fields)
        except Exception:
            pass

    def tool_followup_write(self, *, request_id, anchor, instructions) -> None:
        """At-factory evidence, NOT a transport ACK or provider adoption.

        Hash the final instructions' UTF-8 bytes, not the whole serialized event.
        Inspect only the already-rendered bounded anchor so omitted receipts do
        not appear in diagnostics. Nothing is persisted beyond safe metadata.
        """
        try:
            if (not isinstance(anchor, str) or len(anchor.encode("utf-8")) > 4096
                    or not isinstance(instructions, str)):
                return
            fields, status = [], "absent"
            if anchor:
                data = json.loads(anchor[anchor.index('{"request":'):])
                status = data["request"]["quoted_user_content"]["status"]
                fields = sorted({key for row in data["receipts"] for key in row["owner_outcome"]})
            self.latency_event("tool_followup_write", request_id=request_id,
                boundary="guarded_response_factory_not_transport_ack", anchor_present=bool(anchor),
                anchor_status=status, outcome_fields=fields,
                instructions_sha256=hashlib.sha256(instructions.encode("utf-8")).hexdigest())
        except Exception:
            pass

    def client_event(self, payload: dict, **correlation) -> None:
        kind = payload.get("type")
        name = _CLIENT_EVENTS.get(kind) if isinstance(kind, str) else None
        if name:
            fields = safe_fields(payload)
            fields.update(correlation)
            self.event(name, **fields)

    @staticmethod
    def control_clock() -> float:
        """One process-local clock; never subtract phone/provider wall clocks."""
        return time.monotonic()

    @contextmanager
    def control_handler(self, payload: dict, *, origin: str, dequeued_at: float,
                        receive_started_at: float | None = None):
        """Measure serial dispatch, not network arrival or hidden socket backlog.

        High-volume media deltas only emit a record if their handler is slow.
        Receive waiting includes ordinary idle time and is not network latency.
        """
        started = time.monotonic()
        kind = payload.get("type")
        selected = isinstance(kind, str) and kind in _CONTROL_EVENTS
        dispatch_id = uuid4().hex if selected else None

        def fields():
            response = payload.get("response")
            response_id = payload.get("response_id")
            if not response_id and isinstance(response, dict):
                response_id = response.get("id")
            return dict(
                origin=origin, event_type=kind, dispatch_id=dispatch_id,
                event_id=payload.get("event_id"), client_event_id=payload.get("client_event_id"),
                item_id=payload.get("item_id"), response_id=response_id,
                boundary="application_dequeue_not_wire_arrival",
                dequeued_monotonic_s=dequeued_at,
                decode_dispatch_ms=max(0.0, (started - dequeued_at) * 1000),
                receive_wait_ms=(None if receive_started_at is None else
                                 max(0.0, (dequeued_at - receive_started_at) * 1000)),
            )

        if selected:
            self.event("control_handler_started", **fields())
        status, error_type = "completed", None
        try:
            yield
        except BaseException as error:
            status = "cancelled" if isinstance(error, CancelledError) else "failed"
            error_type = type(error).__name__
            raise
        finally:
            duration_ms = (time.monotonic() - started) * 1000
            if selected or duration_ms >= _SLOW_CONTROL_HANDLER_MS:
                if dispatch_id is None:
                    dispatch_id = uuid4().hex
                self.event("control_handler_finished", **fields(), duration_ms=duration_ms,
                           status=status, error_type=error_type)

    def forwarded(self, payload: dict) -> None:
        kind = payload.get("type")
        if not isinstance(kind, str):
            return
        if kind == "audio.output.delta":
            if not isinstance(payload.get("audio_base64"), str) or not payload["audio_base64"]:
                return
            fields = safe_fields(payload)
            key = (fields.get("response_id"), fields.get("item_id"))
            if key == (None, None):
                # Without identity, do not manufacture a measurable first audio.
                return
            if key in self._audio_items:
                self._audio_items.move_to_end(key)
                return
            self._audio_items[key] = None
            if len(self._audio_items) > MAX_RECENT_AUDIO_ITEMS:
                self._audio_items.popitem(last=False)
            self.event("first_audio_forwarded", boundary="backend_socket_write", **fields)
        elif kind in _WIRE_EVENTS:
            self.event(_WIRE_EVENTS[kind], **safe_fields(payload))


def latency_event(trace, name: str, **fields) -> None:
    """Also tolerate absent traces and injected diagnostic failures in fixtures."""
    try:
        if trace is not None:
            trace.latency_event(name, **fields)
    except Exception:
        pass


def _latency_clock():
    try:
        return time.monotonic()
    except Exception:
        return None


@contextmanager
def latency_phase(trace, name: str, *, phase: str, fields=None):
    """Measure the original operation; never suppress its exception or cancellation.

    Shared fields are sampled at each boundary. In particular, entering a send
    is only possible-write evidence, and a local return is not a provider ACK.
    """
    fields = fields if fields is not None else {}
    started = _latency_clock()
    latency_event(trace, name, phase=phase, status="started", **fields)
    status = "completed"
    try:
        yield
    except BaseException as error:
        status = "cancelled" if isinstance(error, CancelledError) else "failed"
        raise
    finally:
        ended = _latency_clock()
        duration = None if started is None or ended is None else max(0.0, (ended - started) * 1000)
        latency_event(trace, name, phase=phase, status=status, duration_ms=duration, **fields)


class latency_context:
    """Timing proxy for the SAME context manager, preserving its exit/suppression.

    No extra await, task, transaction, guard or timeout is introduced.
    """
    def __init__(self, trace, name, context, *, enter, exit, fields=None):
        self.trace, self.name, self.context = trace, name, context
        self.enter, self.exit, self.fields = enter, exit, fields

    async def __aenter__(self):
        if self.enter is None:
            return await self.context.__aenter__()
        with latency_phase(self.trace, self.name, phase=self.enter, fields=self.fields):
            return await self.context.__aenter__()

    async def __aexit__(self, *error):
        if self.exit is None:
            return await self.context.__aexit__(*error)
        with latency_phase(self.trace, self.name, phase=self.exit, fields=self.fields):
            return await self.context.__aexit__(*error)
