"""Foreground terminal envelopes on the existing private socket (stdlib only)."""
from dataclasses import asdict


CONTEXT_FIELDS = {"native_session_id", "input_id", "admission_id", "call_id", "turn_id"}
_FIELDS = {"version", "channel", "operation", "context", "attachment", "arguments"}
WORKSPACE = "/home/jarvis/workspace"
SANDBOX_ACTIONS = frozenset({"status", "pause", "resume", "keepalive_enable", "keepalive_disable"})


def _identity(value):
    if not isinstance(value, str) or not value.strip() or len(value) > 256 or "\0" in value:
        raise ValueError("Missing private terminal identity")


def validate_terminal_request(request):
    if (not isinstance(request, dict) or set(request) != _FIELDS
            or type(request["version"]) is not int or request["version"] != 1
            or request["channel"] != "terminal"
            or request["operation"] not in {"attach", "execute", "detach"}
            or not isinstance(request["arguments"], dict)):
        raise ValueError("Invalid terminal envelope")
    operation, context = request["operation"], request["context"]
    if operation == "detach":
        if context is not None or request["arguments"]:
            raise ValueError("Detach accepts only its private attachment")
    else:
        if not isinstance(context, dict) or set(context) != CONTEXT_FIELDS:
            raise ValueError("Missing current native input context")
        for value in context.values():
            _identity(value)
    if operation == "attach":
        if request["attachment"] is not None or request["arguments"]:
            raise ValueError("Attach cannot select a resource")
    else:
        _identity(request["attachment"])
    if operation == "execute" and set(request["arguments"]) != {"command", "cwd", "timeout"}:
        raise ValueError("Invalid foreground arguments")


def invoke_terminal(profile_home, *, operation, context=None, attachment=None, arguments=None, timeout=125.0):
    """One exchange only. Context comes from native ContextVars, never tool args."""
    from .domain_transport import _invoke_private
    request = {"version": 1, "channel": "terminal", "operation": operation,
               "context": asdict(context) if context is not None else None,
               "attachment": attachment, "arguments": {} if arguments is None else arguments}
    result = _invoke_private(profile_home, request, timeout=timeout)
    if result.get("status") == "failed":
        # A timeout/disconnect is not cancellation or evidence of non-execution.
        return {"status": "unknown", "error": {"code": "terminal_transport_uncertain",
            "message": "No complete terminal result. Effects may have occurred; do not resubmit blindly."}}
    return result


def validate_sandbox_control_request(request):
    if (not isinstance(request, dict)
            or set(request) != {"version", "channel", "context", "arguments"}
            or type(request["version"]) is not int or request["version"] != 1
            or request["channel"] != "sandbox_control"):
        raise ValueError("Invalid sandbox control envelope")
    context, arguments = request["context"], request["arguments"]
    if not isinstance(context, dict) or set(context) != CONTEXT_FIELDS:
        raise ValueError("Missing current native input context")
    for value in context.values():
        _identity(value)
    if (not isinstance(arguments, dict) or set(arguments) - {"action", "target_session_id"}
            or not isinstance(arguments.get("action"), str)
            or arguments["action"] not in SANDBOX_ACTIONS):
        raise ValueError("Invalid sandbox action")
    if "target_session_id" in arguments:
        _identity(arguments["target_session_id"])


def invoke_sandbox_control(profile_home, *, context, arguments, timeout=65.0):
    """One private control exchange; never retry an uncertain lifecycle effect."""
    from .domain_transport import _invoke_private
    result = _invoke_private(profile_home, {"version": 1, "channel": "sandbox_control",
        "context": asdict(context), "arguments": arguments}, timeout=timeout)
    if result.get("error", {}).get("code") in {
        "domain_transport_unavailable", "domain_request_timeout", "domain_request_unavailable",
    }:
        return {"status": "unknown", "error": {"code": "sandbox_control_uncertain",
            "message": "Lifecycle outcome unconfirmed. Inspect the exact resource before retrying; no automatic retry ran."}}
    return result
