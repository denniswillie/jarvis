"""Bounded identity-only observations. Never request or delivery authority."""
from datetime import UTC, datetime
import json
import logging
import re
import time
from uuid import UUID

from app.realtime.telemetry import PROCESS_ID


MAX_RECORDS = 64
LOGGER = logging.getLogger(__name__)
_IDENTITY = re.compile(r"[A-Za-z0-9_:./-]{1,192}\Z")
_CODES = frozenset({
    "tool_request_source_required", "tool_request_source_invalid",
    "tool_request_source_unavailable", "tool_request_source_schema_conflict",
    "invalid_tool_arguments", "response_dismissed", "tool_unavailable_for_response",
    "tool_result_source_unavailable", "live_context_changed", "live_context_revoked",
    "tool_execution_failed", "tool_receipt_unavailable_after_commit",
})
_STAGES = frozenset({"payload_factory", "writer_finished", "selection", "executor_return"})


def _emit(event, fields):
    # RealtimeTrace intentionally filters unknown fields. This dedicated sink
    # accepts only this module's explicitly projected, content-free records.
    LOGGER.info("request_source_diagnostic %s", json.dumps({
        **fields, "event": event, "process_id": PROCESS_ID,
        "at": datetime.now(UTC).isoformat(), "monotonic_s": time.monotonic(),
    }, separators=(",", ":")))


def _identity(value):
    if isinstance(value, UUID):
        return str(value)
    return value if type(value) is str and _IDENTITY.fullmatch(value) else None


def _integer(value):
    return value if type(value) is int and 0 <= value <= 2**63 - 1 else None


def _request(request):
    record = request.record
    return {
        "source_ref": _identity(request.reference),
        "role": request.kind if request.kind in {"current", "retained_unwritten", "retained_cancelled_empty", "tool_request"} else "unknown",
        "input_id": _identity(request.input_id),
        "source_event_id": _identity(request.source_event_id),
        "provider_item_id": _identity(request.provider_item_id),
        "canonical_event_id": _identity(record.id) if record is not None else None,
        "canonical_revision": _integer(record.revision) if record is not None else None,
        "finality": record.finality if record is not None and record.finality in {"final", "revised"} else "unavailable",
        "canonical_record_available": record is not None,
        "text_offered": request.presentation()["text_status"] == "available",
    }


def _schema(response, offer):
    # Inspect ONLY the payload being returned at the guarded writer. A registry
    # definition or inherited session configuration is not remote ACK evidence.
    tools = response.get("tools")
    if not isinstance(tools, list):
        return {"scope": "inherited_unconfirmed"}
    cancel = [tool for tool in tools if tool.get("name") == "cancel_background_task"]
    tool = cancel[0] if len(cancel) == 1 else {}
    parameters = tool.get("parameters", {})
    properties = parameters.get("properties", {})
    suppression = properties.get("suppress_result_announcement", {})
    selector = properties.get("jarvis_request_source", {})
    expected = [request.reference for request in offer.requests] if offer is not None else None
    return {
        "scope": "response_override", "tool_count": len(tools),
        "cancel_tool_count": len(cancel),
        "suppression_advertised": "suppress_result_announcement" in properties,
        "suppression_boolean": suppression.get("type") == "boolean",
        "suppression_default_false": suppression.get("default") is False,
        "suppression_required": "suppress_result_announcement" in parameters.get("required", ()),
        "selector_advertised": "jarvis_request_source" in properties,
        "selector_enum_matches_offer": expected is not None and selector.get("enum") == expected,
        "selector_schema_required": "jarvis_request_source" in parameters.get("required", ()),
    }


def record_request_source(session, stage, *, reservation=None, state=None, offer=None,
                          response=None, selected=None, call_id=None, name=None,
                          selection_present=False, code=None, outcome=None, result=None):
    """Best effort, no awaits/queries; never serialize caller payloads or errors."""
    try:
        if stage not in _STAGES:
            return
        counters = getattr(session, "_request_source_diagnostic_counts", None)
        if counters is None:
            counters = {"attempted": 0, "emitted": 0, "dropped": 0, "reporting_failed": 0, "closed": False}
            session._request_source_diagnostic_counts = counters
        if counters["closed"]:
            return
        counters["attempted"] += 1
        if counters["attempted"] > MAX_RECORDS:
            counters["dropped"] += 1
            return
        metadata = getattr(state, "metadata", {}) or {}
        fields = {
            "schema_version": 1, "authority": "diagnostic_only",
            "ordinal": counters["attempted"], "stage": stage,
            "request_id": _identity(reservation.request_id if reservation is not None else metadata.get("jarvis_request_id")),
            "response_id": _identity(getattr(state, "response_id", None)),
            "turn_generation": _integer(reservation.turn_generation if reservation is not None else getattr(offer, "generation", None)),
            "owner_id": _identity(session.context.owner_id),
            "session_id": _identity(session.context.jarvis_session_id),
            "audio_session_id": _identity(session.context.audio_session_id),
            "connection_epoch": _identity(session._visual_context.connection.connection_epoch),
        }
        if stage == "payload_factory":
            fields.update(
                offered_count=len(offer.requests) if offer is not None else 0,
                requests=[_request(request) for request in offer.requests[:5]] if offer is not None else [],
                instruction_contract="source_bound_user_v1" if offer is not None else "no_source_offer",
                schema=_schema(response, offer),
                write_evidence="factory_only_not_send_or_ack",
            )
        elif stage == "writer_finished":
            fields.update(local_reservation_submitted=reservation.submitted is True,
                          outcome=outcome if outcome in {"local_send_returned", "not_sent", "no_return"} else "unknown",
                          provider_ack="unmeasured")
        else:
            if result is not None:
                outcome = result.get("status") if isinstance(result, dict) else None
                error = result.get("error") if isinstance(result, dict) else None
                code = error.get("code") if isinstance(error, dict) else "unclassified_error" if error else None
            fields.update(
                call_id=_identity(call_id),
                tool=name if name in {"start_background_task", "get_background_task", "cancel_background_task",
                                     "update_background_task", "resume_background_task"} else "other",
                selection_present=selection_present is True,
                selected=_request(selected) if selected is not None else None,
                outcome=outcome if outcome in {"selected_hot_cache", "single_request_path", "control_path",
                                              "rejected_before_executor", "success", "failed", "cancelled", "exception"} else "unknown",
                error_code=code if code in _CODES else "redacted" if code is not None else None,
            )
        _emit("request_source_boundary", fields)
        counters["emitted"] += 1
    except Exception:
        if isinstance(locals().get("counters"), dict):
            counters["reporting_failed"] += 1


def close_request_source_diagnostics(session):
    """One bounded final counter receipt after owned session work has joined."""
    try:
        counters = getattr(session, "_request_source_diagnostic_counts", None)
        if counters is None or counters["closed"]:
            return
        counters["closed"] = True
        _emit("request_source_diagnostics_closed", {
            "schema_version": 1, "authority": "diagnostic_only", "limit": MAX_RECORDS,
            "owner_id": _identity(session.context.owner_id),
            "session_id": _identity(session.context.jarvis_session_id),
            "audio_session_id": _identity(session.context.audio_session_id),
            **{key: counters[key] for key in ("attempted", "emitted", "dropped", "reporting_failed")},
        })
    except Exception:
        pass
