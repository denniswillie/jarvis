"""Native catalog wiring with private, executing-input context (no backend DB imports)."""

from dataclasses import dataclass
import json
from pathlib import Path

from .bindings import BindingError, bounded_id


@dataclass(frozen=True)
class NativeInputContext:
    native_session_id: str
    input_id: str
    admission_id: str
    call_id: str
    turn_id: str


def capture_native_context(native_session_id: str) -> NativeInputContext:
    """Called inside the registered handler, not inferred from turn IDs or receipts."""
    from agent.runtime_session_store import is_worker_process
    from gateway.session_display import _submission_turn
    from tools.approval_context import _approval_tool_call_id, _approval_turn_id

    if is_worker_process():
        raise BindingError("Domain tools are not qualified in managed subprocesses")
    submission = _submission_turn.get()
    if not isinstance(submission, tuple) or len(submission) != 2:
        raise BindingError("Missing native originating submission context")
    admission_id, input_id = submission
    call_id, turn_id = _approval_tool_call_id.get(), _approval_turn_id.get()
    for value in (native_session_id, admission_id, input_id, call_id, turn_id):
        bounded_id(value)
    return NativeInputContext(native_session_id, input_id, admission_id, call_id, turn_id)


def _handler(home: Path, name: str):
    def invoke(arguments, **kwargs):
        try:
            context = capture_native_context(kwargs.get("session_id"))
            if not isinstance(arguments, dict):
                raise BindingError("Domain tool arguments must be an object")
            from .domain_transport import invoke_domain_tool
            result = invoke_domain_tool(
                home, native_session_id=context.native_session_id, input_id=context.input_id,
                admission_id=context.admission_id, call_id=context.call_id,
                name=name, arguments=arguments,
            )
            if not isinstance(result, dict):
                raise BindingError("Invalid domain tool response")
            if result.get("_multimodal") is True:
                return result  # Native registry unwraps image blocks only from dictionaries.
            return json.dumps(result, ensure_ascii=False)
        except BindingError as exc:
            return json.dumps({"error": str(exc)})
        except Exception:
            return json.dumps({"error": "Domain tool unavailable; no scope authority inferred"})
    return invoke


def register_domain_tools(ctx):
    """Schemas are discovery only. The transport/adapter authorizes every invocation."""
    from hermes_constants import get_hermes_home
    home = Path(get_hermes_home()).resolve()
    try:
        config = json.loads((home / "jarvis-adapter.json").read_text())
    except FileNotFoundError:
        return
    if config.get("domain_tools") is not True:
        return
    catalog_path = home / "jarvis" / "domain-tools.json"
    if catalog_path.stat().st_size > 512_000:
        raise BindingError("Domain catalog exceeds its bound")
    catalog = json.loads(catalog_path.read_text())
    if not isinstance(catalog, dict) or type(catalog.get("version")) is not int or catalog["version"] != 1:
        raise BindingError("Unsupported domain catalog version")
    tools = catalog.get("tools")
    if not isinstance(tools, list) or len(tools) > 64:
        raise BindingError("Invalid domain catalog tools")
    names = set()
    for schema in tools:
        if not isinstance(schema, dict) or set(schema) != {"name", "description", "parameters"}:
            raise BindingError("Invalid domain tool schema")
        name = bounded_id(schema["name"])
        if name == "work_session" or name in names:
            raise BindingError("Duplicate/reserved domain tool name")
        if not isinstance(schema["description"], str) or len(schema["description"]) > 16_000:
            raise BindingError("Invalid domain tool description")
        if not isinstance(schema["parameters"], dict) or schema["parameters"].get("type") != "object":
            raise BindingError("Invalid domain tool parameters")
        names.add(name)
    for schema in tools:
        ctx.register_tool(name=schema["name"], toolset="jarvis_domain", schema=schema,
                          handler=_handler(home, schema["name"]), is_async=False)
