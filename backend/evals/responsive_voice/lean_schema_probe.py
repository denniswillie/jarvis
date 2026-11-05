"""One bounded provider schema/setup probe, not a conversation or app test.

No response is requested and no tool/domain action is executed. Workspace
producer placeholders are only used to select the actual note/open schemas.
"""
import argparse
import asyncio
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace
from uuid import uuid4

from app.live_state.briefing import LiveBriefingReader
from app.live_state.workspace_dispatch import LeanWorkspaceDispatcher
from app.realtime.lean_session import LEAN_INSTRUCTIONS
from app.realtime.openai_client import OpenAIRealtimeConnection, build_session_update_event
from app.settings import JarvisSettings
from app.tools.registry import JarvisToolRegistry
from app.workspace.commands import WorkspaceCommands
from evals.responsive_voice.model_wire_trace import TracedSocket, WireTrace, render


async def probe(directory):
    settings = JarvisSettings()
    owner, session = "lean-schema-synthetic", uuid4()
    registry = JarvisToolRegistry()
    registry.register(LiveBriefingReader(owner_id=owner, session_id=session, snapshot=lambda: None,
        clock=lambda: datetime.now(UTC), monotonic=time.monotonic).tool())
    dispatcher = LeanWorkspaceDispatcher(owner_id=owner, session_id=session,
        provider_session_key=lambda: "probe_not_an_admission", is_active=lambda: False,
        commands=WorkspaceCommands(object(), SimpleNamespace(exporter=None)), journal=object())
    registry.register_many(dispatcher.definitions())
    tools = registry.openai_realtime_tools()
    trace = WireTrace(directory)
    connection = OpenAIRealtimeConnection(settings=settings, owner_id=owner)
    result = {"boundary": "provider_schema_acceptance_only", "model": settings.OPENAI_REALTIME_MODEL,
              "tool_names": [tool["name"] for tool in tools], "response_requested": False,
              "domain_effects": False, "passed": False}
    try:
        async with asyncio.timeout(25):
            await connection.connect()
            connection._websocket = TracedSocket(connection._websocket, trace, str(session))
            await connection.send_event(build_session_update_event(settings=settings,
                instructions=LEAN_INSTRUCTIONS, tools=tools))
            for _ in range(10):
                event = await connection.receive_event()
                if event.get("type") == "error":
                    result["error"] = {key: (event.get("error") or {}).get(key)
                                       for key in ("type", "code", "param", "message")}
                    break
                if event.get("type") == "session.updated":
                    observed = event.get("session", {}).get("tools", [])
                    result["observed_tool_names"] = [tool.get("name") for tool in observed]
                    result["passed"] = sorted(result["observed_tool_names"]) == sorted(result["tool_names"])
                    result["requested_tools_sha256"] = hashlib.sha256(json.dumps(tools, sort_keys=True).encode()).hexdigest()
                    result["observed_tools_sha256"] = hashlib.sha256(json.dumps(observed, sort_keys=True).encode()).hexdigest()
                    break
    except Exception as error:
        result["error_type"] = type(error).__name__
    finally:
        try:
            async with asyncio.timeout(5):
                await connection.close()
        except Exception as error:
            result["close_error_type"] = type(error).__name__
        trace.close()
        result["trace_complete"] = trace.incomplete is None
        (directory / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        render(directory, directory / "timeline.md")
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] and result["trace_complete"] else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    arguments = parser.parse_args()
    raise SystemExit(asyncio.run(probe(arguments.run_dir)))
