"""Native Hermes composition and crash probes with inert, run-owned effects."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import signal
import sqlite3
import subprocess
import sys
import time
import uuid

from contracts import assert_tool_receipts, fixture_effect, tool_result
from environment import credentials, isolated_env, root_path, save


def setup(root):
    os.chdir(root)
    os.environ.update(isolated_env(root))
    os.environ["HERMES_HOME"] = str(root / "hermes-acceptance-home")


def agent(root, toolset, allowed):
    from run_agent import AIAgent
    instance = AIAgent(
        provider="openai", api_mode="codex_responses", base_url="https://api.openai.com/v1",
        api_key=credentials(), model="gpt-5.6", max_iterations=5, max_tokens=1800,
        enabled_toolsets=[toolset], skip_context_files=True, skip_memory=True,
        load_soul_identity=False, quiet_mode=True, save_trajectories=False,
        reasoning_config={"effort": "low"}, session_id="final-" + uuid.uuid4().hex,
        ephemeral_system_prompt="Use only provided evidence and authorized fixture tools. Never invent missing context or claim an action was delivered without its receipt.")
    names = {t.get("function", t).get("name") for t in instance.tools}
    if names != allowed:
        instance.close()
        raise RuntimeError("Unexpected tool allowlist")
    return instance


def child(root, phase, case):
    setup(root)
    from tools.registry import registry
    database = root / (case + "-effect.sqlite")
    marker = root / (case + "-entered.json")

    def effect(arguments, **kwargs):
        if arguments:
            raise ValueError("Fixture effect takes no model-selected scope or payload")
        if phase == "before":
            save(marker, {"phase": phase, "effect_committed": False})
            time.sleep(300)
        result = fixture_effect(database, "fixed-authorized-operation", {"artifact": "synthetic-note", "owner": "fixture-a"})
        if phase == "after":
            save(marker, {"phase": phase, "effect_committed": True, "receipt": result})
            time.sleep(300)
        return tool_result(result)

    registry.register(name="fixture_effect", toolset="final_effect", handler=effect, schema={
        "name": "fixture_effect", "description": "Perform or recover one already authorized synthetic operation; no external effect.",
        "parameters": {"type": "object", "properties": {}, "additionalProperties": False}})
    instance = agent(root, "final_effect", {"fixture_effect"})
    try:
        result = instance.run_conversation(user_message="Call fixture_effect once. Report its receipt verbatim and whether it was newly created or already recorded. Do not claim email or notification delivery.")
        assert_tool_receipts(result)
        save(root / "reports" / (case + "-recovery.json"), result)
    finally:
        instance.close()


def run(root):
    setup(root)
    from tools.registry import registry
    from comparison import Candidate
    report = {"model": "gpt-5.6", "api_mode": "codex_responses", "errors": [], "crashes": [],
              "scope": "native Hermes plus explicit fixture ledger; not production Jarvis or autonomous native restart",
              "sandbox": "No host shell enabled. Fixed-path artifact callback is not native sandbox containment acceptance."}
    output = root / "reports/worker-acceptance.json"

    def persist():
        save(output, report)

    def spawn(phase, case):
        log = open(root / (case + "-" + phase + ".log"), "w")
        process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--root", str(root),
                                    "--child", phase, "--case", case], cwd=root, env=isolated_env(root),
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        log.close()
        path = root / "resources.json"
        manifest = json.loads(path.read_text())
        manifest["processes"].append({"pid": process.pid, "executable": str(Path(__file__).resolve())})
        save(path, manifest)
        return process

    def count(database):
        if not database.exists():
            return 0
        with sqlite3.connect(database) as connection:
            return connection.execute("SELECT count(*) FROM effects").fetchone()[0]

    try:
        for case in ("before", "after"):
            state = {"phase": case}
            report["crashes"].append(state)
            process = spawn(case, case)
            marker = root / (case + "-entered.json")
            deadline = time.monotonic() + 90
            try:
                while not marker.exists() and process.poll() is None and time.monotonic() < deadline:
                    time.sleep(0.2)
                if not marker.exists():
                    raise RuntimeError("Native child did not reach requested crash boundary")
                state["boundary"] = json.loads(marker.read_text())
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
            database = root / (case + "-effect.sqlite")
            state["effects_before_recovery"] = count(database)
            recovery = spawn("recover", case)
            try:
                state["recovery_exit_code"] = recovery.wait(timeout=180)
            finally:
                if recovery.poll() is None:
                    os.killpg(recovery.pid, signal.SIGKILL)
                    recovery.wait(timeout=10)
            if state["recovery_exit_code"] != 0:
                raise RuntimeError("Recovery child failed; inspect private child log")
            state["effects_after_recovery"] = count(database)
            result = json.loads((root / "reports" / (case + "-recovery.json")).read_text())
            assert_tool_receipts(result)
            receipt = fixture_effect(database, "fixed-authorized-operation", {"artifact": "synthetic-note", "owner": "fixture-a"})
            state["receipt_consumed"] = receipt["receipt"] in str(result.get("final_response", ""))
            state["one_effect"] = state["effects_after_recovery"] == 1
            persist()

        config = json.loads((root / "private-config.json").read_text())
        comparison = json.loads((root / "reports/comparison.json").read_text())
        bank = next(r["bank"] for r in comparison["runs"] if r["provider"] == "hindsight" and r.get("ready"))
        candidate = Candidate("hindsight", config)
        nonce = uuid.uuid4().hex
        artifact = root / "delegated-report.md"
        calls = []
        report["composition"] = {"bank": bank, "tools": calls, "artifact": str(artifact)}

        def recall(arguments, **kwargs):
            if set(arguments) != {"query"}:
                raise ValueError("Scope is trusted and cannot be selected by the model")
            payload = candidate.recall(bank, arguments["query"])
            calls.append({"tool": "recall_fixture_memory", "bank": bank})
            return tool_result({"evidence": candidate.evidence(payload), "receipt_nonce": nonce})

        def write(arguments, **kwargs):
            if set(arguments) != {"content"} or not isinstance(arguments["content"], str):
                raise ValueError("Only fixture content is accepted")
            content = arguments["content"]
            if len(content) > 20000:
                raise ValueError("Fixture artifact exceeds its bound")
            artifact.write_text(content)
            digest = hashlib.sha256(content.encode()).hexdigest()
            calls.append({"tool": "write_fixture_report", "sha256": digest})
            return tool_result({"path": str(artifact), "sha256": digest, "created": True, "email_sent": False})

        registry.register(name="recall_fixture_memory", toolset="final_composition", handler=recall, schema={
            "name": "recall_fixture_memory", "description": "Retrieve evidence from this task's trusted synthetic memory bank.",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"], "additionalProperties": False}})
        registry.register(name="write_fixture_report", toolset="final_composition", handler=write, schema={
            "name": "write_fixture_report", "description": "Write one synthetic markdown report at a fixed isolated path. Does not send email.",
            "parameters": {"type": "object", "properties": {"content": {"type": "string"}}, "required": ["content"], "additionalProperties": False}})
        instance = agent(root, "final_composition", {"recall_fixture_memory", "write_fixture_report"})
        try:
            image = base64.b64encode((root / "coding-fixture.png").read_bytes()).decode()
            prompt = [{"type": "text", "text": (
                "Retrieve the coding progress, corrected gym set, and stove evidence. Write a short markdown report using "
                "that evidence and this image. Include original source IDs where available and the memory receipt_nonce "
                "verbatim in the artifact. Do not assume the missing earlier audio was understood or interrupted speech "
                "was heard. The current request is a report, not a full function rewrite. After writing, report the "
                "artifact path and sha256 from the tool receipt. No email is requested.")},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64," + image}}]
            started = time.monotonic()
            result = instance.run_conversation(user_message=prompt, conversation_history=[
                {"role": "user", "content": "Earlier I wanted a full rewrite. I now only want a grounded report. An earlier utterance has no finalized transcript."}])
            assert_tool_receipts(result)
            report["composition"]["seconds"] = time.monotonic() - started
            report["composition"]["output"] = result
            if artifact.exists():
                content = artifact.read_text()
                digest = hashlib.sha256(content.encode()).hexdigest()
                report["composition"]["artifact_content"] = content
                report["composition"]["memory_receipt_consumed"] = nonce in content
                report["composition"]["artifact_receipt_consumed"] = digest in str(result.get("final_response", ""))
            else:
                report["composition"]["missing_artifact"] = True
        finally:
            instance.close()
            candidate.client.close()
    except Exception as error:
        report["errors"].append({"type": type(error).__name__, "message": str(error)})
    finally:
        persist()
    print(json.dumps({"report": str(output), "errors": report["errors"],
                      "crashes": report["crashes"],
                      "composition_checks": {k: v for k, v in report.get("composition", {}).items() if k.endswith("consumed")}}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--child", choices=("before", "after", "recover"))
    parser.add_argument("--case", choices=("before", "after"))
    args = parser.parse_args()
    root = root_path(args.root)
    if args.child:
        if not args.case:
            parser.error("--child requires --case")
        child(root, args.child, args.case)
    else:
        run(root)
