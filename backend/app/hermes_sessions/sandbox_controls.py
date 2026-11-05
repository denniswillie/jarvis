"""Opt-in sandbox controls and native master notices, with a light import boundary.

Backend composition supplies existing services; native handlers use only the
private RPC. Importing this module neither opens a store nor imports the E2B SDK,
backend DB packages, or a resource owner. No timer or gateway is started here.
"""
from __future__ import annotations

import asyncio
import json
import math
from pathlib import Path

from .bindings import bounded_id, identity
from .domain_plugin import capture_native_context
from .domain_scope import DomainInputScope
from .terminal_transport import invoke_sandbox_control, validate_sandbox_control_request


SCHEMA = {
    "name": "session_sandbox",
    "description": (
        "Inspect, pause or resume an existing retained sandbox. A child may control only its own sandbox; "
        "the master may select existing same-profile sandboxes and alone enable/disable keepalive. "
        "Keepalive performs bounded micro-pause/resume maintenance and sends system events to the master; "
        "it does not schedule scripts or notify Voice. No action creates or recreates a sandbox. "
        "Pause/resume is not a guarantee that processes survived. Background process management is not enabled."
    ),
    "parameters": {"type": "object", "properties": {
        "action": {"type": "string", "enum": ["status", "pause", "resume", "keepalive_enable", "keepalive_disable"]},
        "target_session_id": {"type": "string", "minLength": 1, "maxLength": 256,
            "description": "Exact retained logical session ID from work_session; omit for your own sandbox. "
                           "Only the master may select another session."},
    }, "required": ["action"], "additionalProperties": False},
}


class SandboxControls:
    """Use on the existing resource owner's event loop, not native executor workers."""
    def __init__(self, *, sessions, sandboxes, keepalive):
        if sessions.bindings.owner_id != sandboxes.bindings.owner_id:
            raise ValueError("Session and sandbox owners differ")
        if keepalive.sandboxes is not sandboxes:
            raise ValueError("Keepalive must use the same sandbox service")
        self.profile_home = Path(sessions.bindings.profile_home).resolve()
        self.sessions, self.sandboxes, self.keepalive = sessions, sandboxes, keepalive

    @staticmethod
    def _failure(code, *, uncertain=False):
        return {"status": "unknown" if uncertain else "failed", "error": {"code": code,
            "message": ("Lifecycle outcome unconfirmed; inspect the exact resource before retrying."
                        if uncertain else "Sandbox control refused; no lifecycle operation was requested.")}}

    def _policy(self, target):
        row = self.keepalive.journal.policy(target)
        return {"enabled": bool(row and row["enabled"]), "due_at": row["due_at"] if row else None}

    async def invoke(self, *, context, arguments):
        effect_started = False
        try:
            validate_sandbox_control_request(dict(version=1, channel="sandbox_control",
                                                  context=context, arguments=arguments))
            actor = await self.sessions.caller(context["native_session_id"])
            action = arguments["action"]
            target = arguments.get("target_session_id", actor.logical_id)
            if action.startswith("keepalive_") and not actor.is_master:
                return self._failure("sandbox_keepalive_master_required")
            if actor.is_master:
                if actor.logical_id != self.keepalive.master:
                    return self._failure("sandbox_control_master_mismatch")
            elif target != actor.logical_id:
                return self._failure("sandbox_control_own_session_required")
            # A selector must name this owner's actual conversation, not a raw
            # E2B ID or an invented logical session accepted by a resource store.
            binding = await asyncio.to_thread(self.sessions.bindings.get, target)
            if binding.native_id is None:
                return self._failure("sandbox_control_session_unavailable")
            allocated = self.sandboxes.bindings.get(target)
            if allocated is None or allocated.sandbox_id is None:
                return self._failure("session_sandbox_not_allocated")
            scope = allocated.scope
            if (scope.owner_id != self.sessions.bindings.owner_id
                    or scope.profile_id != self.sandboxes.bindings.profile_id
                    or scope.logical_session_id != target):
                return self._failure("sandbox_control_scope_mismatch")
            result = {"status": "success", "action": action, "target_session_id": target}
            if action.startswith("keepalive_"):
                effect_started = True
                method = self.keepalive.enable if action == "keepalive_enable" else self.keepalive.disable
                await method(actor_session_id=actor.logical_id, target_session_id=target)
            else:
                effect_started = action != "status"
                method = {"status": self.sandboxes.inspect, "pause": self.sandboxes.pause,
                          "resume": self.sandboxes.resume}[action]
                info = await method(target)
                result["sandbox"] = {"sandbox_id": info.sandbox_id, "state": info.state,
                    "expires_at": info.expires_at.isoformat() if info.expires_at is not None else None,
                    "lifetime_seconds": info.policy.lifetime_seconds}
            result["keepalive"] = self._policy(target)
            return result
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            code = getattr(exc, "code", "sandbox_control_unavailable")
            return self._failure(code, uncertain=effect_started)


class NativeMasterNoticeDelivery:
    """Keepalive delivery callback: native FIFO admission ACK, not model success.

    For a domain-scoped service, prepare_scope is mandatory and must be async:
      prepare_scope(*, master_session_id, native_input_id, notice) -> DomainInputScope
    Supply the existing trusted Jarvis domain session/read allowlist explicitly.
    The returned scope must carry NO original request reference. This adapter
    validates and durably binds it before send; it never selects a latest request.
    Constructing this callback need not materialize the master. Delivery waits
    for an existing, bound master and never creates a native conversation.
    """
    def __init__(self, *, sessions, master_session_id, prepare_scope=None):
        bounded_id(master_session_id)
        self.scoped = hasattr(sessions, "scopes")
        if self.scoped and prepare_scope is None:
            raise ValueError("Domain-scoped notices require explicit authority-stripped scope preparation")
        if prepare_scope is not None and not callable(prepare_scope):
            raise ValueError("Scope preparation must be callable")
        self.sessions, self.master, self.prepare_scope = sessions, master_session_id, prepare_scope

    def _notice(self, notice):
        required = {"event_id", "target", "master", "sandbox_id", "started_at", "kind", "outcome", "observed_state"}
        if (not isinstance(notice, dict) or not required <= set(notice) or set(notice) - required - {"error"}
                or notice["master"] != self.master or notice["kind"] != "sandbox_maintenance"):
            raise ValueError("Invalid maintenance notice")
        for key in ("event_id", "target", "master", "sandbox_id"):
            bounded_id(notice[key])
        if (type(notice["started_at"]) not in (int, float) or not math.isfinite(notice["started_at"])
                or notice["observed_state"] not in (None, "running", "paused")
                or notice["outcome"] not in {"micro_pause_resumed", "recovered_running_cycle_unconfirmed",
                                             "paused_requires_master", "unconfirmed"}
                or ("error" in notice and (not isinstance(notice["error"], str) or len(notice["error"]) > 200))):
            raise ValueError("Invalid maintenance result")
        encoded = json.dumps(notice, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
        text = ("[SYSTEM EVENT: sandbox maintenance; not a new user request]\n"
                "Resource-controller data follows. It carries no original-user write authority. "
                "A running sandbox does not prove that its programs survived pause/resume. "
                "Decide whether a user-facing notice is needed; no Voice notification was sent.\n" + encoded)
        if len(text.encode()) > 4096:
            raise ValueError("Maintenance notice exceeds its bound")
        return json.loads(encoded), text

    async def deliver(self, notice) -> bool:
        try:
            notice, text = self._notice(notice)
            master = await asyncio.to_thread(self.sessions.bindings.get, self.master)
            if not master.is_master or master.native_id is None:
                return False
            # Validate the event's selector against the same private profile too.
            await asyncio.to_thread(self.sessions.bindings.get, notice["target"])
            event_id = notice["event_id"]
            native_input_id = identity(self.master, "input", event_id)
            if self.scoped:
                scope = await self.prepare_scope(master_session_id=self.master,
                    native_input_id=native_input_id, notice=notice)
                if (not isinstance(scope, DomainInputScope) or scope.owner_id != self.sessions.bindings.owner_id
                        or scope.logical_session_id != self.master or scope.input_id != native_input_id
                        or scope.request_event_id is not None or scope.request_revision is not None):
                    return False
                await self.sessions.bind_input_scope(self.master, input_id=event_id,
                    jarvis_session_id=scope.jarvis_session_id, allowed_tools=scope.allowed_tools,
                    request_event_id=None, request_revision=None)
            receipt = await self.sessions.send(self.master, input_id=event_id, text=text)
            return (receipt.session_id == self.master and receipt.input_id == event_id
                    and receipt.native_input_id == native_input_id)
        except asyncio.CancelledError:
            raise
        except Exception:
            # The journal retains this exact event/text identity for retry. No
            # second admission or model call is launched here after uncertainty.
            return False


def register_sandbox_controls(ctx):
    from hermes_constants import get_hermes_home
    home = Path(get_hermes_home()).resolve()
    try:
        config = json.loads((home / "jarvis-adapter.json").read_text())
    except FileNotFoundError:
        return
    if config.get("sandbox_controls") is not True:
        return

    def handle(arguments, **kwargs):
        try:
            context = capture_native_context(kwargs.get("session_id"))
            result = invoke_sandbox_control(home, context=context, arguments=arguments)
        except Exception:
            result = SandboxControls._failure("sandbox_control_context_unavailable")
        return json.dumps(result, ensure_ascii=False)

    ctx.register_tool(name="session_sandbox", toolset="jarvis_sandbox", schema=SCHEMA,
                      handler=handle, is_async=False)
