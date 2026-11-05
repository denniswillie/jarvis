"""Native local Hindsight/Honcho trials; identical synthetic attributed corpus."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import time
import uuid

import httpx
from environment import root_path, save


def value(obj):
    return obj.model_dump(mode="json") if hasattr(obj, "model_dump") else obj


def run(root):
    from hindsight_client import Hindsight
    corpus_path = Path(__file__).with_name("fixtures.json")
    raw = corpus_path.read_bytes()
    corpus = json.loads(raw)
    config = json.loads((root / "private-config.json").read_text())
    suffix = uuid.uuid4().hex[:12]
    report = {"synthetic_only": True, "corpus_sha256": hashlib.sha256(raw).hexdigest(),
              "track": "provider-native answers plus raw retrieval; no common-answer-model comparison",
              "existing_jarvis_baseline": "not run", "providers": {}, "run_id": suffix,
              "authenticated_cross_owner_isolation": "not proven: Hindsight shared service key and Honcho local auth disabled"}
    out = root / "reports/memory.json"

    def persist():
        save(out, report)

    def request(client, method, path, body=None):
        response = client.request(method, path, json=body) if body is not None else client.request(method, path)
        response.raise_for_status()
        return response.json() if response.content else {"status_code": response.status_code}

    for provider in ("hindsight", "honcho"):
        state = {"queries": [], "operations": [], "resources": [], "errors": [], "cleanup": []}
        report["providers"][provider] = state
        persist()
        started = time.monotonic()
        base = config[provider + "_url"]
        headers = {"Authorization": "Bearer " + config["hindsight_token"]} if provider == "hindsight" else {}
        client = httpx.Client(base_url=base, headers=headers, timeout=180)
        bank = "reuse-" + suffix + "-a"
        other = "reuse-" + suffix + "-b"
        deletion = "reuse-" + suffix + "-delete"
        sdk = Hindsight(base_url=base, api_key=config["hindsight_token"], timeout=180) if provider == "hindsight" else None

        def honcho_wait(workspace, seconds=240):
            deadline = time.monotonic() + seconds
            last = None
            while time.monotonic() < deadline:
                last = request(client, "GET", f"/v3/workspaces/{workspace}/queue/status")
                if last.get("pending_work_units") == 0 and last.get("in_progress_work_units") == 0:
                    return last
                time.sleep(2)
            raise TimeoutError("Honcho ingestion did not finish: " + json.dumps(last))

        def hindsight_wait(bank_id, operation, seconds=180):
            operation = value(operation)
            operation_id = operation.get("operation_id") if isinstance(operation, dict) else None
            if not operation_id:
                raise RuntimeError("Missing asynchronous operation identity")
            deadline = time.monotonic() + seconds
            while time.monotonic() < deadline:
                status = request(client, "GET", f"/v1/default/banks/{bank_id}/operations/{operation_id}")
                if status.get("status") in ("completed", "failed", "cancelled"):
                    if status["status"] != "completed":
                        raise RuntimeError("Hindsight operation did not complete: " + json.dumps(status))
                    return status
                time.sleep(2)
            raise TimeoutError("Hindsight operation deadline exceeded")

        def attributed(source):
            return f"Source ID: {source['id']}\nEvent time: {source['time']}\nEvidence type: {source['kind']}\n{source['text']}"

        try:
            if provider == "hindsight":
                for name in (bank, other, deletion):
                    state["resources"].append(name)
                    persist()
                    sdk.create_bank(bank_id=name, name="Synthetic Jarvis reuse evaluation", enable_reranking=False)
                t = time.monotonic()
                state["ingestion"] = value(sdk.retain_batch(bank_id=bank, items=[
                    {"content": attributed(s), "timestamp": s["time"], "document_id": s["id"],
                     "metadata": {"source_id": s["id"], "evidence_type": s["kind"]}}
                    for s in corpus["sources"]]))
                state["ingestion_seconds"] = time.monotonic() - t
                sdk.retain(bank_id=other, content=corpus["other_owner_canary"], document_id="private-b")
                sdk.retain(bank_id=deletion, content=corpus["deletion_fixture"], document_id="forget-me")
            else:
                for name in (bank, other, deletion):
                    state["resources"].append(name)
                    persist()
                    request(client, "POST", "/v3/workspaces", {"id": name})
                    request(client, "POST", f"/v3/workspaces/{name}/peers", {"id": "alex"})
                    request(client, "POST", f"/v3/workspaces/{name}/sessions", {"id": "session-1", "peers": {"alex": {}}})
                # Split across two sessions to exercise actual cross-session recall.
                request(client, "POST", f"/v3/workspaces/{bank}/sessions", {"id": "session-2", "peers": {"alex": {}}})
                t = time.monotonic()
                ingested = []
                for session, sources in (("session-1", corpus["sources"][:4]), ("session-2", corpus["sources"][4:])):
                    ingested.append(request(client, "POST", f"/v3/workspaces/{bank}/sessions/{session}/messages", {
                        "messages": [{"peer_id": "alex", "content": attributed(s), "created_at": s["time"],
                                      "metadata": {"source_id": s["id"], "evidence_type": s["kind"]}} for s in sources]}))
                state["ingestion"] = ingested
                for name, content in ((other, corpus["other_owner_canary"]), (deletion, corpus["deletion_fixture"])):
                    request(client, "POST", f"/v3/workspaces/{name}/sessions/session-1/messages", {
                        "messages": [{"peer_id": "alex", "content": content}]})
                state["ingestion_queue"] = honcho_wait(bank)
                state["ingestion_seconds"] = time.monotonic() - t

            persist()
            for question in corpus["questions"]:
                entry = {"id": question["id"], "query": question["query"]}
                state["queries"].append(entry)
                t = time.monotonic()
                try:
                    if provider == "hindsight":
                        entry["retrieval"] = value(sdk.recall(bank_id=bank, query=question["query"], budget="low",
                                                              max_tokens=2200, include_chunks=True, include_source_facts=True))
                        entry["retrieval_seconds"] = time.monotonic() - t
                        a = time.monotonic()
                        entry["answer"] = value(sdk.reflect(bank_id=bank, query=question["query"], budget="low",
                                                            max_tokens=1200, include_facts=True, include_tool_calls=True))
                    else:
                        entry["retrieval"] = request(client, "POST", f"/v3/workspaces/{bank}/peers/alex/search",
                                                     {"query": question["query"], "limit": 12})
                        entry["retrieval_seconds"] = time.monotonic() - t
                        a = time.monotonic()
                        entry["answer"] = request(client, "POST", f"/v3/workspaces/{bank}/peers/alex/chat",
                                                  {"query": question["query"], "reasoning_level": "low"})
                    entry["answer_seconds"] = time.monotonic() - a
                except Exception as error:
                    entry["error"] = {"type": type(error).__name__, "message": str(error)}
                persist()

            if provider == "hindsight":
                state["isolation_recall"] = value(sdk.recall(bank_id=bank, query="What is owner B's private project code?", budget="low"))
                state["stats_before_replay"] = request(client, "GET", f"/v1/default/banks/{bank}/stats")
                source = corpus["sources"][0]
                state["replay"] = value(sdk.retain(bank_id=bank, content=attributed(source), document_id=source["id"],
                                                 timestamp=datetime.fromisoformat(source["time"].replace("Z", "+00:00"))))
                state["stats_after_replay"] = request(client, "GET", f"/v1/default/banks/{bank}/stats")
                operation = sdk.create_mental_model(bank_id=deletion, name="Vault phrase", source_query="What is the fictional vault phrase?", id="vault-phrase", max_tokens=500)
                state["mental_model_operation"] = hindsight_wait(deletion, operation)
                state["mental_model_before_delete"] = value(sdk.get_mental_model(bank_id=deletion, mental_model_id="vault-phrase"))
                state["delete_response"] = request(client, "DELETE", f"/v1/default/banks/{deletion}/documents/forget-me")
                state["mental_model_after_delete"] = value(sdk.get_mental_model(bank_id=deletion, mental_model_id="vault-phrase"))
                state["recall_after_delete"] = value(sdk.recall(bank_id=deletion, query="What is the fictional vault phrase?", budget="low", include_chunks=True))
                state["reflect_after_delete"] = value(sdk.reflect(bank_id=deletion, query="What is the fictional vault phrase?", budget="low", max_tokens=600))
            else:
                state["isolation_recall"] = request(client, "POST", f"/v3/workspaces/{bank}/peers/alex/search",
                                                    {"query": "What is owner B's private project code?", "limit": 10})
                source = corpus["sources"][0]
                state["replay"] = request(client, "POST", f"/v3/workspaces/{bank}/sessions/session-1/messages", {
                    "messages": [{"peer_id": "alex", "content": attributed(source), "created_at": source["time"], "metadata": {"source_id": source["id"]}}]})
                honcho_wait(deletion)
                state["answer_before_delete"] = request(client, "POST", f"/v3/workspaces/{deletion}/peers/alex/chat",
                                                        {"query": "What is the fictional vault phrase?", "reasoning_level": "low"})
                state["delete_response"] = request(client, "DELETE", f"/v3/workspaces/{deletion}/sessions/session-1")
                state["deletion_queue"] = honcho_wait(deletion)
                state["recall_after_delete"] = request(client, "POST", f"/v3/workspaces/{deletion}/peers/alex/search",
                                                      {"query": "What is the fictional vault phrase?", "limit": 10})
                state["answer_after_delete"] = request(client, "POST", f"/v3/workspaces/{deletion}/peers/alex/chat",
                                                      {"query": "What is the fictional vault phrase?", "reasoning_level": "low"})
            state["forbidden_canary_in_owner_a_recall"] = "QUARTZ-MOTH-729" in json.dumps(state["isolation_recall"])
            state["semantic_review_required"] = True
        except Exception as error:
            state["errors"].append({"type": type(error).__name__, "message": str(error)})
        finally:
            state["elapsed_seconds"] = time.monotonic() - started
            # Retain fixture stores until explicit service teardown so restart and
            # independent source/derived-data inspection remain possible.
            state["cleanup"] = "Pending run-owned environment teardown; resources recorded above"
            persist()
            client.close()
            if sdk is not None:
                close = getattr(sdk, "close", None)
                if callable(close):
                    close()
    print(json.dumps({"report": str(out), "providers": {p: {"queries": len(s["queries"]), "errors": s["errors"]}
                                                         for p, s in report["providers"].items()}}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    run(root_path(parser.parse_args().root))
