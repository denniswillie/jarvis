"""Isolated Hermes entry point. Keep Jarvis/voice dependencies out of this process."""
from __future__ import annotations

import contextvars
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import select
import traceback
import sys
import threading
import urllib.error
import urllib.request

MAX_INPUT = 32000000
EXPECTED_VERSION = "0.19.0"
_call_id = contextvars.ContextVar("jarvis_native_tool_call", default="")
_run_id = None


def run_conversation(agent, config):
    """Pass canonical dialogue through the maintained native history interface."""
    return agent.run_conversation(
        json.dumps(config["manifest"]), system_message=config["system_prompt"],
        conversation_history=config.get("conversation_history") or None,
        task_id=config["task_id"],
    )


def handle_control(agent, message):
    """Forward ordinary input to the existing agent; do not interpret it here."""
    if type(message) is not dict:
        raise ValueError("Invalid native control")
    if message == {"operation": "cancel"}:
        agent.interrupt("The owning Jarvis worker stopped this attempt.")
        return False
    if (set(message) != {"operation", "text"} or message["operation"] != "steer"
            or type(message["text"]) is not str or not message["text"].strip()
            or len(message["text"].encode("utf-8")) > 64000):
        raise ValueError("Invalid native follow-up")
    agent.steer(message["text"])
    return True


def main():
    wire = sys.stdout
    sys.stdout = sys.stderr  # Native diagnostics cannot corrupt the JSON protocol.
    wire_lock = threading.Lock()

    def emit(event):
        with wire_lock:
            wire.write(json.dumps({**event, "run_id": _run_id}) + "\n")
            wire.flush()

    def progress(text, **_):
        # Ordinary assistant commentary only, never a reasoning callback.
        if isinstance(text, str) and text.strip() and len(text.encode("utf-8")) <= 16384:
            emit({"type": "progress", "text": text})

    pending = bytearray()
    def read_line(limit, stopped=None):
        while b"\n" not in pending:
            if stopped is not None and stopped.is_set():
                return None
            readable, _, _ = select.select([sys.stdin.fileno()], [], [], .1)
            if not readable:
                continue
            chunk = os.read(sys.stdin.fileno(),65536)
            if not chunk:
                return b""
            pending.extend(chunk)
            if len(pending)>limit:
                raise ValueError("Oversized native input")
        end = pending.index(b"\n")+1
        value = bytes(pending[:end])
        del pending[:end]
        return value
    first = read_line(MAX_INPUT)
    if not first or len(first) > MAX_INPUT:
        raise ValueError("Missing or oversized native configuration")
    config = json.loads(first)
    global _run_id
    _run_id = config.get("run_id")
    home = Path(os.environ["HERMES_HOME"])
    marker = json.loads((home.parent / "run.json").read_text())
    if (home.is_symlink() or marker["run_id"] != config["run_id"]
            or marker["attempt_id"] != config["attempt_id"] or marker["pid"] != os.getpid()):
        raise ValueError("Native home does not belong to this attempt")
    if importlib.metadata.version("hermes-agent") != EXPECTED_VERSION:
        raise ValueError("The installed Hermes version does not match the pinned adapter")

    # Imports stay inside the pinned child, never in the voice/API interpreter.
    from run_agent import AIAgent
    from hermes_cli.plugins import PluginContext, PluginManifest, get_plugin_manager
    from tools.registry import registry

    def correlate(*, next_call, args, tool_call_id="", **_):
        token = _call_id.set(tool_call_id)
        try:
            return next_call(args)
        finally:
            _call_id.reset(token)

    plugin = PluginContext(PluginManifest(name="jarvis-task-bridge"), get_plugin_manager())
    plugin.register_middleware("tool_execution", correlate)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def handler(name):
        def execute(arguments, **_):
            identity = _call_id.get()
            if not identity:
                return json.dumps({"status":"failed","code":"native_tool_identity_missing"})
            request = urllib.request.Request(
                config["base_url"] + "/task/tools/" + name,
                data=json.dumps({"arguments":arguments,"call_id":identity}).encode(),
                headers={"Authorization":"Bearer "+config["token"],"Content-Type":"application/json"},
                method="POST",
            )
            try:
                with opener.open(request, timeout=config["tool_timeout_seconds"]) as response:
                    raw = response.read(MAX_INPUT + 1)
                    if len(raw) > MAX_INPUT:
                        raise ValueError("Oversized scoped tool result")
                result = json.loads(raw)
                return result if result.get("_multimodal") is True else json.dumps(result)
            except Exception:
                # No raw HTTP body, credential or exception path is model-visible.
                return json.dumps({"status":"failed","code":"task_tool_transport_failed",
                                   "meaning":"The operation is not confirmed. Do not repeat effects."})
        return execute

    for schema in config["tools"]:
        registry.register(schema["name"], "jarvis_task", schema, handler(schema["name"]),
                          max_result_size_chars=1000000)
    agent = AIAgent(
        model=config["model"], provider="openai", api_mode="codex_responses",
        base_url=config["base_url"], api_key=config["token"], enabled_toolsets=["jarvis_task"],
        skip_context_files=True, skip_memory=True, load_soul_identity=False, session_db=None,
        save_trajectories=False, quiet_mode=True, verbose_logging=False, checkpoints_enabled=False,
        max_iterations=config["max_iterations"], tool_delay=0, max_tokens=config["max_output_tokens"],
        reasoning_config={"effort":config["reasoning_effort"]}, session_id=config["attempt_id"],
        platform="jarvis-task", request_overrides={"store":False},
        interim_assistant_callback=progress if config.get("progress_enabled") is True else None,
    )
    if agent.valid_tool_names != {item["name"] for item in config["tools"]}:
        raise ValueError("Native catalog differs from the exact task allowlist")
    done = threading.Event()

    def control():
        while not done.is_set():
            line = read_line(128000,done)
            if line is None:
                return
            if not line:
                agent.interrupt("The owning Jarvis worker disconnected.")
                # No gateway credential can become an upstream credential. A
                # hung native thread must still die after the parent disappears.
                if not done.wait(5):
                    shutil.rmtree(home, ignore_errors=True)
                    os._exit(70)
                return
            try:
                message = json.loads(line)
                keep_reading = handle_control(agent, message)
            except (ValueError, TypeError):
                agent.interrupt("Invalid worker control message.")
                return
            if not keep_reading:
                return

    controller = threading.Thread(target=control, name="jarvis-parent-control", daemon=True)
    controller.start()
    emit({"type":"ready", "version":EXPECTED_VERSION})
    try:
        result = run_conversation(agent, config)
        output = {"type":"result","run_id":config["run_id"],
                  "context_compressions":agent.context_compressor.compression_count,
                  **{key:result.get(key) for key in
                     ("completed","failed","interrupted","partial","final_response","api_calls","turn_exit_reason")}}
        emit(output)
    finally:
        done.set()
        controller.join(timeout=1)
        # This home contains only the admitted run's ephemeral native copies.
        shutil.rmtree(home, ignore_errors=True)


def probe():
    wire = sys.stdout
    sys.stdout = sys.stderr
    if importlib.metadata.version("hermes-agent") != EXPECTED_VERSION:
        raise ValueError("Unexpected Hermes version")
    from run_agent import AIAgent
    from hermes_cli.plugins import PluginContext
    from tools.registry import registry
    ready = (callable(getattr(AIAgent,"run_conversation",None))
             and callable(getattr(AIAgent,"interrupt",None))
             and callable(getattr(AIAgent,"steer",None)))
    ready = ready and callable(getattr(PluginContext,"register_middleware",None)) and callable(registry.register)
    wire.write(json.dumps({"ready":ready,"version":EXPECTED_VERSION})+"\n")
    wire.flush()
    if not ready:
        raise ValueError("Native interfaces are unavailable")


if __name__ == "__main__":
    try:
        if sys.argv[1:] == ["--probe"]:
            probe()
        else:
            main()
    except Exception as error:
        # Keep actionable startup diagnostics, but not exception messages,
        # provider bodies, task text, paths or credentials.
        sys.__stdout__.write(json.dumps({
            "type":"error","run_id":_run_id,"error_type":type(error).__name__,
            "frames":[{"file":Path(frame.filename).name,"function":frame.name,"line":frame.lineno}
                      for frame in traceback.extract_tb(error.__traceback__)[-8:]],
        })+"\n")
        sys.__stdout__.flush()
        raise SystemExit(1)
