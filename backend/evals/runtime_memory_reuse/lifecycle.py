"""Positive-control deletion, namespace and replay checks through native REST."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import uuid

from comparison import Candidate
from environment import root_path, save

QUERY = "What is the fictional vault phrase? If no evidence remains, say you do not know."


def run(root, provider=None):
    config = json.loads((root / "private-config.json").read_text())
    corpus = json.loads(Path(__file__).with_name("fixtures.json").read_text())
    comparison = json.loads((root / "reports/comparison.json").read_text())
    report = {"providers": {}, "scope": "Native source/summary deletion and namespace controls; not complete authorization or queued-write tombstone acceptance"}
    output = root / "reports" / (f"lifecycle-{provider}.json" if provider else "lifecycle.json")

    def persist():
        save(output, report)

    for name in ((provider,) if provider else ("hindsight", "honcho")):
        candidate = Candidate(name, config)
        state = {"errors": [], "polls": []}
        report["providers"][name] = state
        bank = "final-delete-" + uuid.uuid4().hex[:12]
        other = "final-other-" + uuid.uuid4().hex[:12]
        primary = next(r["bank"] for r in comparison["runs"] if r["provider"] == name and r.get("ready"))
        state["resources"] = [bank, other]
        persist()
        try:
            candidate.create(other)
            canary = {"id": "private-b", "time": "2026-09-12T12:00:00Z", "kind": "user_report", "text": corpus["other_owner_canary"]}
            candidate.ingest(other, [canary])
            candidate.wait(other)
            query = "What is owner B's private project code?"
            state["canary_positive"] = candidate.recall(other, query)
            state["canary_other_owner_query"] = candidate.recall(primary, query)
            state["canary_control_present"] = "QUARTZ-MOTH-729" in json.dumps(state["canary_positive"])
            state["canary_leaked"] = "QUARTZ-MOTH-729" in json.dumps(state["canary_other_owner_query"])

            candidate.create(bank)
            secret = {"id": "forget-me", "time": "2026-09-12T13:00:00Z", "kind": "user_report", "text": corpus["deletion_fixture"]}
            state["seed"] = candidate.ingest(bank, [secret])
            state["ready"] = candidate.wait(bank)
            state["secret_before"] = candidate.answer(bank, QUERY)
            state["delete_precondition"] = "COPPER-FERN-418" in json.dumps(state["secret_before"])
            if not state["delete_precondition"]:
                raise AssertionError("Deletion positive control failed")
            if name == "hindsight":
                state["replay_before"] = candidate.wait(bank)
                state["identical_replay"] = candidate.ingest(bank, [secret])
                state["replay_after"] = candidate.wait(bank)
                operation = candidate.call("POST", f"/v1/default/banks/{bank}/mental-models", {
                    "id": "vault", "name": "Synthetic vault", "source_query": QUERY, "max_tokens": 400})
                deadline = time.monotonic() + 180
                while True:
                    status = candidate.call("GET", f"/v1/default/banks/{bank}/operations/{operation['operation_id']}")
                    if status["status"] == "completed":
                        break
                    if status["status"] in ("failed", "cancelled") or time.monotonic() > deadline:
                        raise RuntimeError("Saved summary did not complete")
                    time.sleep(1)
                state["summary_before"] = candidate.call("GET", f"/v1/default/banks/{bank}/mental-models/vault")
                state["summary_positive"] = "COPPER-FERN-418" in json.dumps(state["summary_before"])
                state["delete"] = candidate.call("DELETE", f"/v1/default/banks/{bank}/documents/forget-me")
                candidate.wait(bank)
                state["raw_after_source_delete"] = candidate.recall(bank, QUERY)
                state["answer_after_source_delete"] = candidate.answer(bank, QUERY)
                state["summary_after_source_delete"] = candidate.call("GET", f"/v1/default/banks/{bank}/mental-models/vault")
                state["delete_summary"] = candidate.call("DELETE", f"/v1/default/banks/{bank}/mental-models/vault")
                state["summary_after_cascade"] = candidate.call("GET", f"/v1/default/banks/{bank}/mental-models/vault", allow_missing=True)
                state["history_after_cascade"] = candidate.call("GET", f"/v1/default/banks/{bank}/mental-models/vault/history", allow_missing=True)
                state["answer_after_cascade"] = candidate.answer(bank, QUERY)
                state["raw_after_cascade"] = candidate.recall(bank, QUERY)
                state["cascade_forgotten"] = (
                    state["summary_after_cascade"].get("http_status") == 404
                    and state["history_after_cascade"].get("http_status") == 404
                    and "COPPER-FERN-418" not in json.dumps([state["answer_after_cascade"], state["raw_after_cascade"]]))
            else:
                started = time.monotonic()
                state["delete"] = candidate.call("DELETE", f"/v3/workspaces/{bank}/sessions/session-1")
                for attempt in range(7):
                    session = candidate.call("POST", f"/v3/workspaces/{bank}/sessions/list", {})
                    if session["pages"] > 1:
                        raise RuntimeError("Unexpected pagination in two-session fixture")
                    raw = candidate.recall(bank, QUERY)
                    answer = candidate.answer(bank, QUERY)
                    row = {"elapsed_seconds": time.monotonic() - started, "session_listing": session, "recall": raw, "answer": answer}
                    row["phrase_returned"] = "COPPER-FERN-418" in json.dumps([raw, answer])
                    row["session_absent"] = not any(item["id"] == "session-1" for item in session["items"])
                    state["polls"].append(row)
                    persist()
                    if row["session_absent"] and not row["phrase_returned"]:
                        state["access_barrier_observed_seconds"] = row["elapsed_seconds"]
                        break
                    time.sleep(5)
                state["forgotten_without_restart"] = "access_barrier_observed_seconds" in state
            persist()
        except Exception as error:
            state["errors"].append({"type": type(error).__name__, "message": str(error)})
        finally:
            candidate.client.close()
            persist()
    print(json.dumps({"report": str(output), "providers": {name: {
        "errors": state["errors"], "canary_leaked": state.get("canary_leaked"),
        "cascade_forgotten": state.get("cascade_forgotten"), "forgotten_without_restart": state.get("forgotten_without_restart")}
        for name, state in report["providers"].items()}}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--provider", choices=("hindsight", "honcho"))
    args = parser.parse_args()
    run(root_path(args.root), args.provider)
