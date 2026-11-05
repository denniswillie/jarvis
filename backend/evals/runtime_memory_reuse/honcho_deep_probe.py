"""Bounded higher-reasoning sensitivity check, not a new memory corpus."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from comparison import ANSWER_RULES, Candidate
from environment import root_path, save


def run(root):
    config = json.loads((root / "private-config.json").read_text())
    corpus = json.loads(Path(__file__).with_name("fixtures.json").read_text())
    prior = json.loads((root / "reports/comparison.json").read_text())
    bank = next(r["bank"] for r in prior["runs"] if r["provider"] == "honcho" and r["repeat"] == 1 and r.get("ready"))
    report = {"bank": bank, "reasoning_level": "high", "queries": [], "repeats": 1,
              "scope": "Six native answers on the first matched corpus; low/high sensitivity, not repeated acceptance"}
    output = root / "reports/honcho-high.json"
    candidate = Candidate("honcho", config)
    try:
        for question in corpus["questions"]:
            item = {"id": question["id"], "query": question["query"]}
            report["queries"].append(item)
            started = time.monotonic()
            try:
                item["answer"] = candidate.answer(bank, question["query"] + ANSWER_RULES, reasoning_level="high")
            except Exception as error:
                item["error"] = {"type": type(error).__name__, "message": str(error)}
            item["seconds"] = time.monotonic() - started
            save(output, report)
            print(json.dumps({"id": item["id"], "seconds": item["seconds"], "error": item.get("error")}), flush=True)
    finally:
        candidate.client.close()
        save(output, report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    run(root_path(parser.parse_args().root))
