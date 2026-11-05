"""Test whether original-source expansion preserves facts lost in summaries."""
from __future__ import annotations

import argparse
import json
from urllib.parse import quote

import httpx
from comparison import ANSWER_RULES, Candidate
from environment import credentials, root_path, save


def run(root):
    config = json.loads((root / "private-config.json").read_text())
    prior = json.loads((root / "reports/comparison.json").read_text())
    candidate = Candidate("hindsight", config)
    report = {"track": "same common model/prompt, Hindsight retained original documents attached",
              "honcho_control": "Original-source common answers already recorded in comparison.json; not counted as new calls",
              "queries": [], "errors": []}
    output = root / "reports/source-grounding.json"
    try:
        with httpx.Client(base_url="https://api.openai.com/v1",
                          headers={"Authorization": "Bearer " + credentials()}, timeout=180) as client:
            for run in prior["runs"]:
                if run["provider"] != "hindsight" or not run.get("ready"):
                    continue
                cache = {}
                for previous in run["queries"]:
                    if previous.get("error"):
                        continue
                    item = {"repeat": run["repeat"], "bank": run["bank"], "id": previous["id"],
                            "query": previous["query"], "sources": []}
                    report["queries"].append(item)
                    try:
                        for source_id in previous["source_ids"]:
                            if source_id not in cache:
                                cache[source_id] = candidate.call("GET", f"/v1/default/banks/{run['bank']}/documents/{quote(source_id, safe='')}")
                            document = cache[source_id]
                            metadata = document.get("document_metadata") or {}
                            item["sources"].append({
                                "source_id": document["id"], "text": document["original_text"],
                                "evidence_type": metadata.get("evidence_type"),
                                "original_event_time": metadata.get("event_time"),
                            })
                        response = client.post("/responses", json={
                            "model": "gpt-5.4-mini", "reasoning": {"effort": "low"}, "max_output_tokens": 1100,
                            "instructions": "Answer using only the supplied retrieved evidence. " + ANSWER_RULES,
                            "input": json.dumps({"question": item["query"], "evidence": item["sources"]})})
                        response.raise_for_status()
                        item["answer"] = response.json()
                    except Exception as error:
                        item["error"] = {"type": type(error).__name__, "message": str(error)}
                    save(output, report)
    except Exception as error:
        report["errors"].append({"type": type(error).__name__, "message": str(error)})
    finally:
        candidate.client.close()
        save(output, report)
    print(json.dumps({"report": str(output), "questions": len(report["queries"]), "errors": report["errors"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    run(root_path(parser.parse_args().root))
