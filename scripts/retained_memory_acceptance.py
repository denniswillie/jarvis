#!/usr/bin/env python3
"""One native read-only recall, or reconcile an existing receipt without resubmitting.

Live execution requires Main's coordinated reload and explicit --reload-confirmed.
Only the final response, tool names and receipt status leave native result history.
This script never creates/resets a session, changes policy, or restarts a service.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from pathlib import Path
import sys
import time
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
MASTER = "jarvis-8c0a9d419068115a5b2060aafa35acbe7ba481c90ee5bf4b798638489d2a0bf8"
NATIVE = "local-e4f2b997e9853186d3da83d14b0fef6ee8dda8aa0d84e8b4419c123e12da9c78"
PROMPT = (
    "Read-only native Hindsight qualification. Make exactly ONE hindsight_recall "
    "call to find the number saved in voice-smoke-03.txt. You may use tool_search "
    "to discover hindsight_recall if needed. Answer from that tool's returned "
    "source evidence, naming the file and number. Do not answer from conversation "
    "history or guess. Do not use hindsight_retain, hindsight_reflect, terminal, "
    "file tools, other domain tools, or any write operation. If hindsight_recall "
    "is unavailable or fails, report that and stop; do not retry."
)


def save(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def tool_names(value, depth=0) -> list[str]:
    """Inspect structured calls, never serialize arbitrary message/reasoning text."""
    if depth > 12:
        return []
    if isinstance(value, list):
        return [name for item in value for name in tool_names(item, depth + 1)]
    if not isinstance(value, dict):
        return []
    names = []
    function = value.get("function")
    if isinstance(function, dict) and isinstance(function.get("name"), str):
        names.append(function["name"])
        if function["name"] == "tool_call":
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except ValueError:
                    arguments = None
            if isinstance(arguments, dict):
                for call in arguments.get("calls", []):
                    if isinstance(call, dict) and isinstance(call.get("name"), str):
                        names.append(call["name"])
    if value.get("type") in {"tool_use", "function_call"} and isinstance(value.get("name"), str):
        names.append(value["name"])
    for key in ("messages", "tool_calls", "content", "output"):
        names.extend(tool_names(value.get(key), depth + 1))
    return names


def text_content(value) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(part.get("text", "") for part in value
                       if isinstance(part, dict) and isinstance(part.get("text"), str))
    return ""


def project(receipt: dict, *, current_prompt: str | None = None) -> dict:
    result = receipt.get("result") or {}
    final = result.get("final_response") if isinstance(result, dict) else None
    messages = result.get("messages") if isinstance(result, dict) else None
    exact_turn = False
    selected = messages
    if current_prompt is not None and isinstance(messages, list):
        indices = [i for i, message in enumerate(messages) if isinstance(message, dict)
                   and message.get("role") == "user"
                   and text_content(message.get("content")) == current_prompt]
        if indices:
            selected = messages[indices[-1] + 1:]
            exact_turn = True
        else:
            selected = []  # Do not count a prior conversation's tool use as this test.
    names = tool_names(selected)
    terminal = receipt.get("status") == "terminal"
    allowed = {"hindsight_recall", "tool_search", "tool_call"}
    qualified = (terminal and receipt.get("outcome") == "completed" and exact_turn
                 and names.count("hindsight_recall") == 1 and set(names) <= allowed
                 and isinstance(final, str) and "391" in final and "voice-smoke-03.txt" in final)
    return {
        "status": receipt.get("status"), "outcome": receipt.get("outcome"),
        "terminal": terminal, "admission_id": receipt.get("admission_id"),
        "final_response": final, "tool_names": names,
        "tool_scope": "exact_current_turn" if exact_turn else "receipt_history_unqualified",
        "recall_tool_called_once_and_expected_answer": qualified,
        "note": "Call names plus final answer do not independently prove source accuracy.",
    }


async def live(args, report: dict) -> None:
    from app.hermes_sessions.client import HermesGatewayClient

    async with HermesGatewayClient(args.profile, timeout=10, max_message_bytes=16 * 1024 * 1024) as client:
        await client.snapshot(args.native)
        admission_id = args.admission_id
        if admission_id is None:
            input_id = "native-memory-" + uuid4().hex
            save(args.output / "request.json", {
                "master": args.master, "native": args.native, "input_id": input_id,
                "text": PROMPT, "reload_confirmed_by_operator": True,
            })
            report["input_id"] = input_id
            report["status"] = "submission_outcome_unknown"
            admission = await client.submit(args.native, input_id, PROMPT)
            admission_id = admission.get("admission_id")
            save(args.output / "admission.json", {key: admission.get(key) for key in
                 ("admission_id", "status", "outcome", "sequence", "execution_generation")})
            if not isinstance(admission_id, str) or not admission_id:
                raise ValueError("missing_admission_identity")
        report["admission_id"] = admission_id
        started = time.monotonic()
        try:
            async with asyncio.timeout(args.seconds):
                while True:
                    receipt = await client.request("prompt.receipt", session_id=args.native,
                                                   admission_id=admission_id, include_result=True)
                    report.update(project(receipt, current_prompt=None if args.admission_id else PROMPT))
                    if receipt.get("status") == "terminal":
                        break
                    await asyncio.sleep(1)
        except TimeoutError:
            report.update(status="local_deadline", outcome="unknown", terminal=False)
            # A local deadline is not evidence that the model or memory call timed out.
            # Preserve the admission for explicit reconciliation; never retry the prompt.
        finally:
            report["elapsed_seconds"] = round(time.monotonic() - started, 3)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, default=ROOT / ".jarvis-dev/integration-demo/profile")
    parser.add_argument("--master", default=MASTER)
    parser.add_argument("--native", default=NATIVE)
    parser.add_argument("--output", type=Path, required=True, help="New evidence directory; never overwrite a prior run")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--receipt-file", type=Path, help="Project an already completed saved receipt, no network")
    mode.add_argument("--admission-id", help="Poll an existing native admission; never resubmit")
    mode.add_argument("--execute-paid", action="store_true", help="Submit exactly one read-only prompt")
    parser.add_argument("--reload-confirmed", action="store_true")
    parser.add_argument("--seconds", type=int, default=120)
    args = parser.parse_args()
    if not 1 <= args.seconds <= 120:
        parser.error("seconds must be between 1 and 120")
    if args.execute_paid and not args.reload_confirmed:
        parser.error("Main must coordinate reload before --reload-confirmed is supplied")
    args.output.mkdir(parents=True, exist_ok=False)
    logging.disable(logging.CRITICAL)
    report = {"master": args.master, "native": args.native, "status": "not_started"}
    try:
        if args.receipt_file:
            with args.receipt_file.open(encoding="utf-8") as stream:
                saved = json.load(stream)
            report.update(project(saved.get("receipt", saved)))
        else:
            asyncio.run(live(args, report))
    except Exception as exc:
        report["error_type"] = type(exc).__name__  # No credential-bearing exception text.
    save(args.output / "report.json", report)
    print(json.dumps(report, indent=2))
    if args.execute_paid:
        return 0 if report.get("recall_tool_called_once_and_expected_answer") else 1
    return 0 if report.get("terminal") else 1


if __name__ == "__main__":
    raise SystemExit(main())
