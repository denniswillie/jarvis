"""Explicit paid-provider qualification; never discovered by unittest.

Run with HINDSIGHT_API_KEY in the environment and the isolated pinned plugin /
client on PYTHONPATH. --root is a private evidence directory, not a Hermes demo
profile. This script deletes only FIXTURE, never the bank or other documents.
"""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import time

import httpx

from app.shared_memory import BankScope, MemoryDocument, MemoryRequestError, NativeMemory
from app.shared_memory.client import MENTAL_MODELS


FIXTURE = "jarvis-shared-memory-qualification-v1"
OLD = "cobalt-orchid-731"
NEW = "amber-willow-942"
QUERY = "For the synthetic Jarvis qualification fixture, what is the fictional lighthouse's test token?"


async def qualify(args, record):
    scope = BankScope.for_owner(args.owner, generation="integration-demo-v1")
    headers = {"Authorization": "Bearer " + os.environ["HINDSIGHT_API_KEY"]}
    base = args.url.rstrip("/") + "/v1/default/banks/" + scope.bank_id
    async with httpx.AsyncClient(headers=headers, timeout=65, follow_redirects=False) as http:
        memory = NativeMemory(scope=scope, base_url=args.url, api_key=os.environ["HINDSIGHT_API_KEY"],
                              http_client=http, timeout=65)
        provider = None
        wrote_fixture = False
        started = time.monotonic()

        async def operations(receipt):
            if hasattr(receipt, "to_dict"):
                receipt = receipt.to_dict()
            if hasattr(receipt, "model_dump"):
                receipt = receipt.model_dump(by_alias=True)
            ids = receipt.get("operation_ids") or ([receipt["operation_id"]] if receipt.get("operation_id") else [])
            for operation_id in ids:
                deadline = min(started + 540, time.monotonic() + 180)
                while time.monotonic() < deadline:
                    response = await http.get(base + "/operations/" + operation_id)
                    response.raise_for_status()
                    status = response.json().get("status")
                    if status == "completed":
                        break
                    if status in {"failed", "cancelled"}:
                        raise RuntimeError("native_operation_" + status)
                    await asyncio.sleep(2)
                else:
                    raise TimeoutError("native_operation_deadline")

        def text_of(result):
            return "\n".join(item.get("text", "") for item in result.get("results", []))

        async def probe(label, expected, absent=None):
            before = time.monotonic()
            result = await memory.recall(QUERY, max_tokens=1200, budget="low")
            text = text_of(result)
            good = expected in text and (absent is None or absent not in text)
            record(label, ok=good, result_count=len(result.get("results", [])),
                   milliseconds=round(1000 * (time.monotonic() - before), 1))
            if not good:
                raise AssertionError(label)

        try:
            try:
                await memory.create_fresh_bank()
                record("bank", created=True, bank_id=scope.bank_id)
            except MemoryRequestError as exc:
                if exc.code != "hindsight_bank_already_exists":
                    raise
                record("bank", created=False, bank_id=scope.bank_id)
            receipts = await memory.seed_mental_models()
            for receipt in receipts.values():
                await operations(receipt)
            record("mental_models", count=len(receipts), initial_operations_completed=True)

            import plugin_hindsight
            provider = plugin_hindsight.HindsightMemoryProvider()
            await asyncio.to_thread(provider.initialize, "isolated-shared-memory-qualification", agent_context="primary")
            tool_names = sorted(item["name"] for item in provider.get_tool_schemas())
            record("plugin", version="1.2.3", client=importlib.metadata.version("hindsight-client"),
                   embed=importlib.metadata.version("hindsight-embed"), tools=tool_names,
                   bank_matches=provider._bank_id == scope.bank_id,
                   auto_retain=provider._auto_retain, auto_recall=provider._auto_recall,
                   memory_mode=provider._memory_mode)
            assert provider._bank_id == scope.bank_id and not provider._auto_retain and not provider._auto_recall

            content = ("SYNTHETIC QUALIFICATION FIXTURE, not about a real user. "
                       "The fictional lighthouse's test token is " + OLD + ".")
            wrote_fixture = True  # A lost acceptance response still requires cleanup.
            before = time.monotonic()
            receipt = await memory.retain([MemoryDocument(FIXTURE, content, datetime.now(timezone.utc),
                                           context="Synthetic integration test only",
                                           metadata={"fixture": FIXTURE})])
            await operations(receipt)
            record("direct_retain", completed=True, milliseconds=round(1000 * (time.monotonic() - before), 1))
            await probe("direct_recall", OLD)

            before = time.monotonic()
            recalled = await asyncio.to_thread(provider._tool_recall, {"query": QUERY})
            record("plugin_recall", ok=OLD in recalled, milliseconds=round(1000 * (time.monotonic() - before), 1))
            assert OLD in recalled
            reflected = await asyncio.to_thread(provider._tool_reflect, {"query": QUERY})
            record("plugin_reflect", ok=OLD in reflected)
            assert OLD in reflected

            correction = ("SYNTHETIC QUALIFICATION FIXTURE, not about a real user. "
                          "This replaces the previous fixture. The fictional lighthouse's test token is now " + NEW + ".")
            item = provider._build_retain_kwargs(correction, context="Synthetic integration test only",
                                                 metadata={"fixture": FIXTURE}, update_mode="replace")
            # Use the plugin's real client dispatch, but supply the one known test
            # document ID so its ad-hoc retain tool cannot create an untracked copy.
            receipt = await asyncio.to_thread(provider._retain_batch, item, bank_id=scope.bank_id,
                                             document_id=FIXTURE, retain_async=True)
            await operations(receipt)
            record("plugin_retain_correction", completed=True, single_document=True)
            await probe("corrected_direct_recall", NEW, OLD)
        finally:
            if provider is not None:
                await asyncio.to_thread(provider.shutdown)
            if wrote_fixture:
                async def invalidate_fixture_only(forget_scope):
                    assert forget_scope.document_ids == (FIXTURE,) and forget_scope.owner_id == args.owner
                result = await memory.forget([FIXTURE], invalidate=invalidate_fixture_only)
                document_absent = await memory.get_document(FIXTURE) is None
                recalled = await memory.recall(QUERY, max_tokens=1200)
                remaining = text_of(recalled)
                record("cleanup", requested_deletions_applied=result.requested_deletions_applied,
                       document_absent=document_absent, old_token_in_recall=OLD in remaining,
                       new_token_in_recall=NEW in remaining, derived_cleanup=result.derived_cleanup,
                       invalidation_adapter="fixture_only_not_live_state")
                for model_id in MENTAL_MODELS:
                    response = await http.get(base + "/mental-models/" + model_id)
                    if response.status_code == 200:
                        encoded = json.dumps(response.json())
                        record("derived_cleanup_probe", model_id=model_id,
                               old_token_present=OLD in encoded, new_token_present=NEW in encoded)
                assert result.requested_deletions_applied and document_absent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--url", default="http://127.0.0.1:8888")
    parser.add_argument("--owner", default="default")
    args = parser.parse_args()
    logging.disable(logging.CRITICAL)
    path = args.root / f"receipt-{time.time_ns()}.jsonl"
    with path.open("x", encoding="utf-8") as evidence:
        def record(stage, **details):
            line = json.dumps({"stage": stage, **details})
            evidence.write(line + "\n")
            evidence.flush()
            print(line, flush=True)
        try:
            asyncio.run(qualify(args, record))
        except Exception as exc:
            record("failure", error_type=type(exc).__name__,
                   code=exc.code if isinstance(exc, MemoryRequestError) else None,
                   http_status=exc.status if isinstance(exc, MemoryRequestError) else None)
            return 1
        record("finished", ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
