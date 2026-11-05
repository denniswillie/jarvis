"""Bounded, opt-in dated-memory check in a new synthetic Hindsight bank."""
import argparse
import asyncio
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

from app.shared_memory.client import BankScope, MemoryDocument, NativeMemory
from isolated_memory_lifecycle_acceptance import wait_operations


async def run(root, key_file):
    root.mkdir(mode=0o700, parents=True, exist_ok=False)
    scope = BankScope.for_owner("temporal-test:" + root.name, generation="temporal-v1")
    memory = NativeMemory(scope=scope, base_url="http://127.0.0.1:8888",
                          api_key=key_file.read_text().strip(), timeout=40)
    documents = [
        MemoryDocument("january-session", "Fictional test person Rowan: On January 5, 2026, "
                       "I put my spare house key in the cobalt drawer. This is a test, not Dennis's life.",
                       datetime(2026, 1, 5, 12, tzinfo=timezone.utc)),
        MemoryDocument("march-session", "Fictional test person Rowan: On March 5, 2026, "
                       "I moved the same spare house key out of the cobalt drawer into the amber box. "
                       "The amber box is now its location. This is a test, not Dennis's life.",
                       datetime(2026, 3, 5, 12, tzinfo=timezone.utc)),
    ]
    report = {"bank_id": scope.bank_id, "synthetic": True,
              "scope": "native_memory_adapter_across_dated_sessions_not_voice", "steps": []}

    def save():
        (root / "report.json").write_text(json.dumps(report, indent=2, default=str) + "\n")

    async def record(name, operation):
        started = time.monotonic()
        result = await operation
        report["steps"].append({"name": name, "elapsed_seconds": time.monotonic() - started,
                                "result": result})
        save()
        return result

    try:
        await record("create_isolated_bank", memory.create_fresh_bank())
        for document in documents:
            receipt = await record("retain_" + document.document_id, memory.retain([document]))
            await wait_operations(memory, [receipt], timeout=120)
        # A new adapter has no in-process dialogue/history from the earlier writes.
        await memory.aclose()
        memory = NativeMemory(scope=scope, base_url="http://127.0.0.1:8888",
                              api_key=key_file.read_text().strip(), timeout=40)
        await record("march_recall", memory.recall(
            "Where did Rowan put the spare house key, and when did its location change?",
            at=datetime(2026, 3, 6, 12, tzinfo=timezone.utc), max_tokens=1500))
        await record("dated_answer", memory.reflect(
            "For fictional Rowan, where was the spare house key in January 2026, "
            "and where was it most recently placed as of March 6, 2026? "
            "Distinguish the earlier location from the later one. Use only retained evidence.",
            max_tokens=600, budget="low"))
        report["expected"] = {"january": "cobalt drawer", "march": "amber box"}
        report["semantic_acceptance"] = "requires_review_of_returned_evidence_and_answer"
        for document in documents:
            await record("cleanup_" + document.document_id,
                         memory._request("DELETE", "/documents/" + document.document_id, missing_ok=True))
        report["source_cleanup"] = all(
            [await memory.get_document(document.document_id) is None for document in documents])
        save()
    except Exception as error:
        report["failure"] = {"type": type(error).__name__, "code": getattr(error, "code", None)}
        save()
        raise
    finally:
        await memory.aclose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--key-file", type=Path, required=True)
    parser.add_argument("--execute-paid", action="store_true", required=True)
    args = parser.parse_args()
    os.umask(0o077)
    asyncio.run(run(args.output, args.key_file))
