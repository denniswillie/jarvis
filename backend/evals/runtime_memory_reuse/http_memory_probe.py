"""Bounded diagnostic variant: native REST, completed ingestion, no SDK null defaults."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import httpx
from environment import root_path, save


def run(root):
    config = json.loads((root / "private-config.json").read_text())
    first = json.loads((root / "reports/memory.json").read_text())
    corpus = json.loads(Path(__file__).with_name("fixtures.json").read_text())
    report = {"variant": "REST optional fields omitted; Honcho flush-enabled after migration readiness",
              "same_corpus": first["corpus_sha256"], "providers": {}, "semantic_review_required": True}
    out = root / "reports/memory-rest.json"

    def persist():
        save(out, report)

    for provider in ("hindsight", "honcho"):
        state = {"queries": [], "errors": []}
        report["providers"][provider] = state
        bank, other, deletion = first["providers"][provider]["resources"]
        headers = {"Authorization": "Bearer " + config["hindsight_token"]} if provider == "hindsight" else {}
        with httpx.Client(base_url=config[provider + "_url"], headers=headers, timeout=180) as client:
            def call(method, path, body=None):
                response = client.request(method, path, json=body) if body is not None else client.request(method, path)
                response.raise_for_status()
                return response.json() if response.content else {"status_code": response.status_code}

            try:
                if provider == "honcho":
                    deadline = time.monotonic() + 120
                    while True:
                        status = call("GET", f"/v3/workspaces/{bank}/queue/status")
                        state["queue"] = status
                        if status.get("pending_work_units") == 0 and status.get("in_progress_work_units") == 0:
                            break
                        if time.monotonic() >= deadline:
                            raise TimeoutError("Flush-enabled Honcho queue still pending")
                        time.sleep(2)
                for question in corpus["questions"]:
                    item = {"id": question["id"], "query": question["query"]}
                    state["queries"].append(item)
                    t = time.monotonic()
                    try:
                        if provider == "hindsight":
                            item["retrieval"] = call("POST", f"/v1/default/banks/{bank}/memories/recall",
                                                     {"query": question["query"], "budget": "low", "max_tokens": 2200})
                            item["retrieval_seconds"] = time.monotonic() - t
                            t = time.monotonic()
                            item["answer"] = call("POST", f"/v1/default/banks/{bank}/reflect",
                                                  {"query": question["query"], "budget": "low", "max_tokens": 1200})
                        else:
                            item["retrieval"] = call("POST", f"/v3/workspaces/{bank}/peers/alex/search",
                                                     {"query": question["query"], "limit": 12})
                            item["retrieval_seconds"] = time.monotonic() - t
                            t = time.monotonic()
                            item["answer"] = call("POST", f"/v3/workspaces/{bank}/peers/alex/chat",
                                                  {"query": question["query"], "reasoning_level": "low"})
                        item["answer_seconds"] = time.monotonic() - t
                    except Exception as error:
                        item["error"] = {"type": type(error).__name__, "message": str(error)}
                    persist()

                if provider == "hindsight":
                    # Re-seed the previously deleted source and create a new
                    # summary using REST defaults. The first failed precondition
                    # is retained in memory.json rather than overwritten.
                    state["secret_seed"] = call("POST", f"/v1/default/banks/{deletion}/memories", {
                        "items": [{"content": corpus["deletion_fixture"], "document_id": "forget-me"}]})
                    state["secret_before"] = call("POST", f"/v1/default/banks/{deletion}/reflect",
                                                   {"query": "What is the fictional vault phrase?", "budget": "low", "max_tokens": 400})
                    op = call("POST", f"/v1/default/banks/{deletion}/mental-models", {
                        "id": "vault-rest", "name": "Vault phrase REST", "source_query": "What is the fictional vault phrase?", "max_tokens": 500})
                    deadline = time.monotonic() + 120
                    while True:
                        status = call("GET", f"/v1/default/banks/{deletion}/operations/{op['operation_id']}")
                        if status.get("status") == "completed":
                            break
                        if status.get("status") in ("failed", "cancelled") or time.monotonic() >= deadline:
                            raise RuntimeError("Mental-model operation failed or exceeded deadline")
                        time.sleep(1)
                    state["summary_before"] = call("GET", f"/v1/default/banks/{deletion}/mental-models/vault-rest")
                    state["delete_response"] = call("DELETE", f"/v1/default/banks/{deletion}/documents/forget-me")
                    state["summary_after"] = call("GET", f"/v1/default/banks/{deletion}/mental-models/vault-rest")
                    state["secret_after"] = call("POST", f"/v1/default/banks/{deletion}/reflect",
                                                  {"query": "What is the fictional vault phrase?", "budget": "low", "max_tokens": 400})
                else:
                    state["secret_before"] = call("POST", f"/v3/workspaces/{deletion}/peers/alex/chat",
                                                   {"query": "What is the fictional vault phrase?", "reasoning_level": "low"})
                    state["delete_response"] = call("DELETE", f"/v3/workspaces/{deletion}/sessions/session-1")
                    deadline = time.monotonic() + 120
                    while True:
                        status = call("GET", f"/v3/workspaces/{deletion}/queue/status")
                        if status.get("pending_work_units") == 0 and status.get("in_progress_work_units") == 0:
                            break
                        if time.monotonic() >= deadline:
                            raise TimeoutError("Honcho deletion queue did not drain")
                        time.sleep(2)
                    state["deletion_queue"] = status
                    state["secret_after"] = call("POST", f"/v3/workspaces/{deletion}/peers/alex/chat",
                                                  {"query": "What is the fictional vault phrase?", "reasoning_level": "low"})
                    state["isolation_recall"] = call("POST", f"/v3/workspaces/{bank}/peers/alex/search",
                                                      {"query": "What is owner B's private project code?", "limit": 10})
                state["delete_precondition"] = "COPPER-FERN-418" in json.dumps(state.get("secret_before", {}))
                state["deleted_phrase_in_answer"] = "COPPER-FERN-418" in json.dumps(state.get("secret_after", {}))
            except Exception as error:
                state["errors"].append({"type": type(error).__name__, "message": str(error)})
            persist()
    print(json.dumps({p: {"queries": len(v["queries"]), "errors": v["errors"]} for p, v in report["providers"].items()}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    run(root_path(parser.parse_args().root))
