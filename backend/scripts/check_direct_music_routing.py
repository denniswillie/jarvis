"""Bounded real-Luna routing check; every service/action is an inert fixture.

Run with the backend environment:
    python -B scripts/check_direct_music_routing.py --run
    python -B scripts/check_direct_music_routing.py --run --case resume

Uses freshly imported production LunaLiveRouter and CompanionTools schemas.
At most eight provider calls, no retries, production eight-second call timeout.
Never invokes Hermes, song search, playback, a database, or a running server.
Receipts are created exclusively: the resume-only run uses a separate receipt.
"""

import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import sys
from time import monotonic
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, Mock
from uuid import uuid4


sys.dont_write_bytecode = True
BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
RECEIPT = BACKEND.parent / ".jarvis-dev/soloist/direct-routing-receipt.json"
CASES = (
    ("song", "Play Lost in Japan by Shawn Mendes", "media.start"),
    ("resume", "Resume music.", "media.start"),
    ("stop", "Stop the music.", "media.stop"),
    ("volume", "Set music volume to 35 percent.", "media.volume"),
)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def compact_arguments(arguments):
    # Only schema fields, never complete model output or free-form exception text.
    allowed = {"action", "song_title", "artist", "spotify_uri", "media_id",
               "volume", "section", "purpose", "fresh_view", "reason",
               "resume", "recording_mode", "include_jarvis_audio", "recording_id",
               "allocation_id", "generation", "view_id"}
    return {key: value[:200] if isinstance(value, str) else value
            for key, value in arguments.items()
            if key in allowed and value is not None}


async def run_case(case, client, production, voice_tools, receipt):
    name, utterance, action = case
    media_id = str(uuid4())
    needs_id = name in {"stop", "volume"}
    row = {"case": name, "request": utterance, "passed": False,
           "route": [], "calls": [], "failures": []}
    receipt["cases"].append(row)
    if needs_id:
        row["fixture_current_media_id"] = media_id

    def check(condition, failure):
        if not condition and failure not in row["failures"]:
            row["failures"].append(failure)

    async def create(**request):
        if (len(row["calls"]) >= 2
                or receipt["provider_calls"] >= receipt["limits"]["provider_calls"]):
            raise AssertionError("provider_call_budget")
        outputs = [item for item in request["input"]
                   if item.get("type") == "function_call_output"]
        other_input = [item for item in request["input"]
                       if item.get("type") != "function_call_output"]
        check(media_id not in json.dumps(other_input), "id_leaked_outside_tool_output")
        check(request["instructions"] == production.INSTRUCTIONS, "instructions_changed")
        check(not any(item.get("role") in {"system", "developer"}
                      for item in request["input"]), "injected_instruction_message")
        if outputs:
            check(len(outputs) == 1, "unexpected_tool_output_count")
            state = json.loads(outputs[0]["output"])
            check(state.get("system", {}).get("app", {}).get("media", {}).get("media_id")
                  == media_id, "tool_output_missing_current_id")
        call = {"model": request["model"], "max_output_tokens": request["max_output_tokens"],
                "reasoning": request["reasoning"], "store": request["store"],
                "tool_result_supplied": bool(outputs), "returned_tools": []}
        row["calls"].append(call)
        receipt["provider_calls"] += 1
        started = monotonic()
        try:
            response = await client.responses.create(**request)
            call["status"] = response.status
            call["returned_tools"] = [item.name for item in response.output
                                      if item.type == "function_call"]
            check("send_to_hermes" not in call["returned_tools"], "send_to_hermes_selected")
            if response.usage is not None:
                call["usage"] = {field: getattr(response.usage, field)
                                 for field in ("input_tokens", "output_tokens", "total_tokens")}
            return response
        except BaseException as error:
            call["error_type"] = type(error).__name__
            status = getattr(error, "status_code", None)
            if isinstance(status, int):
                call["http_status"] = status
            raise
        finally:
            call["elapsed_seconds"] = round(monotonic() - started, 3)

    def observe(event):
        if event["stage"] == "router_selected":
            row["route"].append({"tool": event["tool"],
                                 "arguments": compact_arguments(event["arguments"])})
        elif event["stage"] == "router_failed":
            check(False, "router_failed:" + event["error_type"])

    # These objects cannot reach any production service. Only the model client
    # above has I/O. Include the normal optional memory tool as a distractor.
    work = AsyncMock()
    work.route_followups = Mock()
    invoke = AsyncMock(return_value={"status": "probe_only", "effect_committed": False})
    memory = NS(scope=NS(owner_id="routing-probe"),
                recall=AsyncMock(return_value={"results": []}))
    read_state = AsyncMock(return_value=({"human": None, "system": {"app": {
        "phone_connected": True, "media": {
            "media_id": media_id, "source_kind": "spotify", "state": "playing",
            "base_volume": 0.7, "state_basis": "phone_receipt_not_audibility",
            "capabilities": {"volume": True, "stop": True, "record_pcm": True}}}}}, ()))
    route = NS(scope=NS(owner_id="routing-probe"))
    session = NS(work=work, request=NS(route=route), ready=True,
                 context=NS(jarvis_session_id=uuid4()), read_live_state=read_state,
                 services=NS(shared_memory=memory, companion_tools=NS(
                     owner_id="routing-probe", voice_tools=lambda: voice_tools, invoke=invoke)))
    packet = NS(route=route, window=NS(request=NS(delegation_id="routing-probe-" + name)),
                as_context=lambda: {"dialogue": [{"speaker": "user", "delta": utterance}]})
    publish = AsyncMock(return_value=NS(provider_received=True, reason=None))
    try:
        router = production.LunaLiveRouter(session, NS(responses=NS(create=create)), observe)
        await router(packet, publish=publish, is_current=lambda: True)
    except Exception as error:
        check(False, "execution_error:" + type(error).__name__)

    expected = ["read_live_state", "control_app"] if needs_id else ["control_app"]
    check([item["tool"] for item in row["route"]] == expected, "unexpected_route")
    check(len(row["calls"]) == len(expected), "unexpected_provider_call_count")
    check(work.await_count == 0, "hermes_handoff_attempted")
    check(memory.recall.await_count == 0, "memory_recall_attempted")
    check(read_state.await_count == int(needs_id), "unexpected_state_read_count")
    if needs_id and read_state.await_count:
        check(read_state.await_args.kwargs == {"purpose": "act"}, "state_not_for_action")
    check(invoke.await_count == 1, "expected_one_stubbed_action")
    if invoke.await_count == 1:
        tool, arguments = invoke.await_args.args
        check(tool == "control_app", "wrong_action_tool")
        from pydantic import ValidationError
        try:
            from app.app_controls import AppCommand
            command = AppCommand.model_validate(arguments)
            check(command.action == action, "wrong_media_action")
            if name == "song":
                check(command.song_title == "Lost in Japan", "wrong_song_title")
                check(command.artist == "Shawn Mendes", "wrong_artist")
                check(command.spotify_uri is None, "invented_spotify_uri")
            if name == "resume":
                check(command.song_title is None and command.artist is None
                      and command.spotify_uri is None, "resume_has_song_selector")
            if needs_id:
                check(str(command.media_id) == media_id, "wrong_current_media_id")
            if name == "volume":
                check(command.volume == 0.35, "wrong_volume")
        except ValidationError as error:
            row["validation_errors"] = [
                {"loc": list(item["loc"]), "type": item["type"]}
                for item in error.errors(include_url=False, include_context=False, include_input=False)
            ]
            check(False, "invalid_app_command:ValidationError")
        except Exception as error:
            check(False, "invalid_app_command:" + type(error).__name__)
    row["passed"] = not row["failures"]
    print(json.dumps({"case": name, "passed": row["passed"], "route": row["route"],
                      "failures": row["failures"],
                      "validation_errors": row.get("validation_errors", [])}), flush=True)


async def run(receipt, cases):
    # Import here in this fresh process, never from a running server or a frozen
    # copy of its prompt/schema. Environment loading does not contact providers.
    from openai import AsyncOpenAI
    from app.settings import JarvisSettings
    from app.realtime import live_router
    from app.live_state.companion_tools import CompanionTools

    schemas = CompanionTools(NS(owner_id="routing-probe"), app_controls=object()).voice_tools()
    control = next(tool for tool in schemas if tool["name"] == "control_app")
    if not {"song_title", "artist"} <= control["parameters"]["properties"].keys():
        raise AssertionError("current_song_schema_missing")
    if "Basic music playback, stop and volume never" not in live_router.INSTRUCTIONS:
        raise AssertionError("current_direct_music_instructions_missing")
    receipt.update(instructions_sha256=digest(live_router.INSTRUCTIONS),
                   offered_tools_sha256=digest(live_router.TOOLS + [live_router.MEMORY_TOOL] + schemas),
                   source="fresh production LunaLiveRouter and CompanionTools.voice_tools",
                   limits={"provider_calls": 2 * len(cases), "output_tokens_per_call": 256,
                           "seconds_per_call": 8, "total_seconds": 16 * len(cases) + 11,
                           "sdk_retries": 0})
    settings = JarvisSettings(_env_file=BACKEND / ".env")
    key = settings.OPENAI_API_KEY
    if not key:
        raise RuntimeError("configured_openai_key_missing")
    async with asyncio.timeout(receipt["limits"]["total_seconds"]):
        async with AsyncOpenAI(api_key=key, max_retries=0, timeout=8) as client:
            for case in cases:
                await run_case(case, client, live_router, schemas, receipt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="authorize this bounded paid routing probe")
    parser.add_argument("--case", choices=("all", "resume"), default="all",
                        help="resume writes a separate receipt and does not rerun other cases")
    args = parser.parse_args()
    if not args.run:
        parser.error("--run is required; otherwise no provider call is made")
    # Never emit SDK/HTTP logs, settings values, or raw provider error bodies.
    logging.disable(logging.CRITICAL)
    cases = CASES if args.case == "all" else tuple(case for case in CASES if case[0] == args.case)
    receipt_path = RECEIPT if args.case == "all" else RECEIPT.with_name("direct-routing-resume-receipt.json")
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        evidence = receipt_path.open("x", encoding="utf-8")
    except FileExistsError:
        print("Receipt already exists; preserved without running the probe.", file=sys.stderr)
        return 2
    receipt = {"started_at": datetime.now(timezone.utc).isoformat(), "passed": False,
               "provider_calls": 0, "cases": [], "case_selection": args.case,
               "effects": "all stubbed; no search or playback"}
    with evidence:
        try:
            asyncio.run(run(receipt, cases))
            receipt["passed"] = len(receipt["cases"]) == len(cases) and all(
                case["passed"] for case in receipt["cases"])
        except (Exception, KeyboardInterrupt) as error:
            receipt["error_type"] = type(error).__name__
        finally:
            receipt["finished_at"] = datetime.now(timezone.utc).isoformat()
            json.dump(receipt, evidence, indent=2, sort_keys=True)
            evidence.write("\n")
    print(json.dumps({"passed": receipt["passed"], "provider_calls": receipt["provider_calls"],
                      "error_type": receipt.get("error_type"), "receipt": str(receipt_path)}))
    return 0 if receipt["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
