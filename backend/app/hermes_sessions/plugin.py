"""Pinned native plugin adapter; no Hermes imports enter the general service.

The private approval-context variables below are a deliberate, version-pinned
compatibility seam: the registry handler does not receive tool-call/turn IDs.
Native execution supplies them, never the model. Qualification covers this seam.
"""

import json
from pathlib import Path

from .bindings import BindingError, SessionBindings, identity
from .client import HermesGatewayClient, HermesGatewayError
from .service import RetainedSessionService
from .work import WorkDeliveries, WorkSessionBridge


SCHEMA = {
    "name": "work_session",
    "description": (
        "Manage reusable child work conversations without blocking your conversation. "
        "create needs a label and a self-contained message with the relevant context; it returns before work finishes. "
        "send continues the SAME child, including after it finished a turn. steer sends guidance without cancelling it. "
        "list shows your children; status reads their actual native result; stop cancels the selected admission, not history. "
        "Completion will wake you separately. A completed child turn may be a question, not a finished task. "
        "A child can use message_parent to ask its master for help. No delete/archive operation exists."
    ),
    "parameters": {"type": "object", "properties": {
        "action": {"type": "string", "enum": ["create", "send", "steer", "list", "status", "stop", "message_parent"]},
        "session_id": {"type": "string", "description": "Retained child ID returned by create/list; required except create/list/message_parent."},
        "label": {"type": "string", "description": "Short purpose label for create."},
        "message": {"type": "string", "description": "Original instruction/context or follow-up for create/send/steer/message_parent."},
        "admission_id": {"type": "string", "description": "Optional exact input receipt for status/stop; otherwise the latest input."},
        "after": {"type": "string", "description": "Optional list pagination cursor."},
    }, "required": ["action"], "additionalProperties": False},
}


def compose(home: Path, client: HermesGatewayClient) -> WorkSessionBridge:
    # This file is written by trusted profile provisioning, never a model tool.
    config = json.loads((home / "jarvis-adapter.json").read_text())
    if config.get("domain_tools") is True:
        from .domain_service import compose_domain_work
        return compose_domain_work(home, client, config=config)
    directory = home / "jarvis"
    bindings = SessionBindings(directory / "bindings.sqlite", owner_id=config["owner_id"], profile_home=home)
    service = RetainedSessionService(client, bindings, launch=config["launch"])
    deliveries = WorkDeliveries(directory / "deliveries.sqlite", bindings, max_pending=config.get("max_pending", 8))
    return WorkSessionBridge(service, deliveries)


async def handle(args: dict, **kwargs) -> str:
    try:
        from hermes_constants import get_hermes_home
        from tools.approval_context import _approval_tool_call_id, _approval_turn_id
        call_id, turn_id = _approval_tool_call_id.get(), _approval_turn_id.get()
        if not call_id or not turn_id or not kwargs.get("session_id"):
            raise BindingError("Missing trusted native tool-call identity")
        home = Path(get_hermes_home()).resolve()
        async with HermesGatewayClient(home) as client:
            bridge = compose(home, client)
            if getattr(bridge, "domain_scoped", False):
                from .domain_plugin import capture_native_context
                context = capture_native_context(kwargs["session_id"])
                result = await bridge.tool(context.native_session_id, context.call_id, args,
                                           input_id=context.input_id,
                                           admission_id=context.admission_id)
            else:
                result = await bridge.tool(kwargs["session_id"], identity(turn_id, call_id), args)
        return json.dumps(result, ensure_ascii=False)
    except HermesGatewayError as exc:
        return json.dumps({"error": exc.reason, "request_id": getattr(exc, "request_id", None),
                           "instruction": "Check existing work/status; do not create duplicate work to retry an uncertain outcome."})
    except BindingError as exc:
        return json.dumps({"error": str(exc)})
    except Exception:
        return json.dumps({"error": "Retained work service unavailable; do not claim the operation succeeded."})


def register(ctx):
    from hermes_constants import get_hermes_home
    from .native_guidance import enable_live_guidance
    from .domain_plugin import register_domain_tools
    from .terminal_plugin import register_terminal_provider
    from .sandbox_controls import register_sandbox_controls
    enable_live_guidance(Path(get_hermes_home()).resolve())
    ctx.register_tool(name="work_session", toolset="jarvis_work", schema=SCHEMA, handler=handle, is_async=True)
    register_domain_tools(ctx)
    register_terminal_provider(ctx)
    register_sandbox_controls(ctx)
