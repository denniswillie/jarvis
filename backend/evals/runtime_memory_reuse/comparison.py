"""Repeated native memory comparison and common-answer control, synthetic only."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import time
import uuid

import httpx
from environment import credentials, root_path, save

ANSWER_RULES = (
    "\nAnswer in at most 180 words. Use only the stored evidence. Distinguish observed, "
    "reported, corrected, and unknown facts. Cite original Source ID values only if you "
    "can retrieve them; never invent identifiers. A request is not a completed action."
)


def attributed(source):
    return (f"Source ID: {source['id']}\nEvent time: {source['time']}\n"
            f"Evidence type: {source['kind']}\n{source['text']}")


class Candidate:
    def __init__(self, name, config):
        self.name = name
        headers = {"Authorization": "Bearer " + config["hindsight_token"]} if name == "hindsight" else {}
        self.client = httpx.Client(base_url=config[name + "_url"], headers=headers, timeout=180)
        self.calls = 0

    def call(self, method, path, body=None, allow_missing=False):
        self.calls += 1
        response = self.client.request(method, path, json=body) if body is not None else self.client.request(method, path)
        if allow_missing and response.status_code == 404:
            return {"http_status": 404}
        response.raise_for_status()
        return response.json() if response.content else {"http_status": response.status_code}

    def wait(self, bank, seconds=600):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if self.name == "hindsight":
                state = self.call("GET", f"/v1/default/banks/{bank}/stats")
                if state["failed_operations"] or state["failed_consolidation"]:
                    raise RuntimeError("Hindsight processing failed")
                done = state["pending_operations"] == 0 and state["pending_consolidation"] == 0
            else:
                state = self.call("GET", f"/v3/workspaces/{bank}/queue/status")
                done = state["pending_work_units"] == 0 and state["in_progress_work_units"] == 0
            if done:
                return state
            time.sleep(2)
        raise TimeoutError(self.name + " ingestion deadline exceeded")

    def create(self, bank):
        if not bank.startswith("final-"):
            raise ValueError("Only final-evaluation namespaces are permitted")
        if self.name == "hindsight":
            return self.call("PUT", f"/v1/default/banks/{bank}", {"name": "Synthetic repeated comparison", "enable_reranking": False})
        self.call("POST", "/v3/workspaces", {"id": bank})
        self.call("POST", f"/v3/workspaces/{bank}/peers", {"id": "alex"})
        for session in ("session-1", "session-2"):
            self.call("POST", f"/v3/workspaces/{bank}/sessions", {"id": session, "peers": {"alex": {}}})
        return {"created": bank}

    def ingest(self, bank, sources):
        if self.name == "hindsight":
            return self.call("POST", f"/v1/default/banks/{bank}/memories", {"items": [
                {"content": attributed(s), "timestamp": s["time"], "document_id": s["id"],
                 "metadata": {"source_id": s["id"], "evidence_type": s["kind"], "event_time": s["time"]}}
                for s in sources]})
        receipts = []
        for session, selected in (("session-1", sources[:4]), ("session-2", sources[4:])):
            if selected:
                receipts.append(self.call("POST", f"/v3/workspaces/{bank}/sessions/{session}/messages", {"messages": [
                    {"peer_id": "alex", "content": attributed(s), "created_at": s["time"],
                     "metadata": {"source_id": s["id"], "evidence_type": s["kind"], "event_time": s["time"]}}
                    for s in selected]}))
        return receipts

    def recall(self, bank, query):
        if self.name == "hindsight":
            return self.call("POST", f"/v1/default/banks/{bank}/memories/recall",
                             {"query": query, "budget": "low", "max_tokens": 2400})
        return self.call("POST", f"/v3/workspaces/{bank}/peers/alex/search", {"query": query, "limit": 12})

    def answer(self, bank, query, **extra):
        if self.name == "hindsight":
            return self.call("POST", f"/v1/default/banks/{bank}/reflect",
                             {"query": query, "budget": "low", "max_tokens": 900, **extra})
        return self.call("POST", f"/v3/workspaces/{bank}/peers/alex/chat", {"query": query, "reasoning_level": "low", **extra})

    def evidence(self, retrieved):
        """Normalize shape only; do not add fixture facts missing from retrieval."""
        records = retrieved.get("results", []) if self.name == "hindsight" else retrieved
        return [{
            "retrieved_id": item.get("id"),
            "source_id": item.get("document_id") or item.get("metadata", {}).get("source_id"),
            "evidence_type": item.get("metadata", {}).get("evidence_type"),
            "original_event_time": item.get("metadata", {}).get("event_time"),
            "text": item.get("text", item.get("content", "")),
        } for item in records]


def run(root, repeats):
    corpus_path = Path(__file__).with_name("fixtures.json")
    raw = corpus_path.read_bytes()
    corpus = json.loads(raw)
    config = json.loads((root / "private-config.json").read_text())
    output = root / "reports/comparison.json"
    report = {"version": 2, "synthetic_only": True, "corpus_sha256": hashlib.sha256(raw).hexdigest(),
              "repeats": repeats, "runs": [], "errors": [],
              "limits": {"native_answers": 12 * repeats, "common_answers": 12 * repeats,
                         "request_timeout_seconds": 180, "ingestion_deadline_seconds": 600},
              "common_model": "gpt-5.4-mini", "answer_rules": ANSWER_RULES,
              "grading": "Independent manual review against frozen fixture oracles, not model self-grading"}
    save(output, report)
    with ExitStack() as stack:
        candidates = {name: Candidate(name, config) for name in ("hindsight", "honcho")}
        for candidate in candidates.values():
            stack.callback(candidate.client.close)
        common = stack.enter_context(httpx.Client(base_url="https://api.openai.com/v1",
                                                  headers={"Authorization": "Bearer " + credentials()}, timeout=180))
        for repeat in range(repeats):
            order = ("hindsight", "honcho") if repeat % 2 == 0 else ("honcho", "hindsight")
            for name in order:
                candidate = candidates[name]
                bank = f"final-{uuid.uuid4().hex[:12]}-r{repeat}"
                state = {"provider": name, "repeat": repeat + 1, "bank": bank, "queries": [], "errors": []}
                report["runs"].append(state)
                save(output, report)
                try:
                    started = time.monotonic()
                    candidate.create(bank)
                    state["ingestion"] = candidate.ingest(bank, corpus["sources"])
                    state["ready"] = candidate.wait(bank)
                    state["ingestion_seconds"] = time.monotonic() - started
                    print(json.dumps({"provider": name, "repeat": repeat + 1, "ingested": True}), flush=True)
                    for question in corpus["questions"]:
                        item = {"id": question["id"], "query": question["query"]}
                        state["queries"].append(item)
                        try:
                            started = time.monotonic()
                            item["retrieval"] = candidate.recall(bank, question["query"])
                            item["retrieval_seconds"] = time.monotonic() - started
                            item["evidence"] = candidate.evidence(item["retrieval"])
                            item["source_ids"] = sorted({e["source_id"] for e in item["evidence"] if e["source_id"]})
                            started = time.monotonic()
                            item["native"] = candidate.answer(bank, question["query"] + ANSWER_RULES)
                            item["native_seconds"] = time.monotonic() - started
                            started = time.monotonic()
                            response = common.post("/responses", json={
                                "model": report["common_model"], "reasoning": {"effort": "low"},
                                "max_output_tokens": 1100,
                                "instructions": "Answer using only the supplied retrieved evidence. " + ANSWER_RULES,
                                "input": json.dumps({"question": question["query"], "evidence": item["evidence"]})})
                            response.raise_for_status()
                            item["common"] = response.json()
                            item["common_seconds"] = time.monotonic() - started
                        except Exception as error:
                            item["error"] = {"type": type(error).__name__, "message": str(error)}
                        save(output, report)
                    print(json.dumps({"provider": name, "repeat": repeat + 1, "questions": len(state["queries"]),
                                      "errors": sum("error" in q for q in state["queries"])}), flush=True)
                except Exception as error:
                    state["errors"].append({"type": type(error).__name__, "message": str(error)})
                save(output, report)
        report["request_counts"] = {name: candidate.calls for name, candidate in candidates.items()}
    save(output, report)
    print(json.dumps({"report": str(output), "runs": len(report["runs"])}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--repeats", type=int, choices=(1, 2, 3), default=3)
    args = parser.parse_args()
    run(root_path(args.root), args.repeats)
