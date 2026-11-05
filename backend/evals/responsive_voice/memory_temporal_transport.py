"""File-backed, controlled memory HTTP fixture. Never opens a network socket.

Documents exist only after normal retain requests. Recall is deterministic
lexical candidate ordering, not semantic/model quality or a seeded answer.
One parent and its sequential children own this file; concurrent processes are
not supported. Each HTTP operation commits atomically before returning.
"""
import json
from pathlib import Path
import re
from urllib.parse import unquote

import httpx

from app.knowledge.hindsight import HindsightClient
from evals.responsive_voice.memory_lifecycle_qualification import require, save


class PersistentControlledMemory:
    def __init__(self, path, bank, *, create=False):
        self.path, self.bank = Path(path), bank
        require(self.path.is_absolute() and bank.startswith("jarvis_temporal_eval_"), "controlled_transport_scope")
        if create:
            require(not self.path.exists(), "controlled_transport_already_exists")
            save(self.path, {"bank": bank, "banks": {}, "main_http": 0, "cleanup_http": 0,
                             "recalls": 0, "cleaning": False})
        self.read()

    def read(self):
        require(self.path.stat().st_size <= 8_000_000, "controlled_transport_size")
        state = json.loads(self.path.read_text())
        require(state["bank"] == self.bank and set(state["banks"]).issubset({self.bank}), "controlled_transport_scope")
        return state

    def begin_cleanup(self):
        state = self.read()
        state["cleaning"] = True
        save(self.path, state)

    def __call__(self, request):
        require(str(request.url).startswith("http://127.0.0.1:19999/"), "controlled_transport_origin")
        state = self.read()
        counter, limit = ("cleanup_http", 400) if state["cleaning"] else ("main_http", 1200)
        require(state[counter] < limit, "controlled_transport_call_budget")
        state[counter] += 1
        parts = [unquote(value) for value in request.url.path.split("/") if value]
        body = json.loads(request.content) if request.content else {}

        def reply(value, status=200):
            save(self.path, state)
            return httpx.Response(status, json=value)

        if parts == ["v1", "default", "banks"]:
            return reply({"banks": [{"bank_id": identity} for identity in state["banks"]]})
        require(parts[:3] == ["v1", "default", "banks"] and len(parts) >= 4
                and parts[3] == self.bank, "controlled_transport_foreign_bank")
        tail = parts[4:]
        if not tail and request.method == "PUT":
            state["banks"].setdefault(self.bank, {"documents": {}, "operations": {}})
            return reply({"bank_id": self.bank})
        if self.bank not in state["banks"]:
            return reply({"detail": "not found"}, 404)
        bank = state["banks"][self.bank]
        if not tail and request.method == "DELETE":
            del state["banks"][self.bank]
            return reply({"success": True})
        if tail == ["config"]:
            return reply({"bank_id": self.bank, "config": {
                "audit_log_enabled": False, "store_document_text": True}})
        if tail == ["mental-models"]:
            return reply({"total": 0, "items": []})
        if tail == ["memories"] and request.method == "POST":
            operation = body["operation_id"]
            bank["operations"][operation] = {"operation_id": operation, "status": "completed",
                                             "child_operations": [], "next_retry_at": None}
            for item in body["items"]:
                bank["documents"][item["document_id"]] = {"id": item["document_id"], "bank_id": self.bank,
                    "original_text": item["content"], "document_metadata": item["metadata"]}
            return reply({"async": True, "operation_id": operation})
        if tail == ["memories", "recall"]:
            require(state["recalls"] < 12, "controlled_recall_budget")
            state["recalls"] += 1
            terms = set(re.findall(r"[a-z0-9]+", body["query"].casefold()))
            documents = list(bank["documents"].values())
            documents.sort(key=lambda item: -len(terms.intersection(
                re.findall(r"[a-z0-9]+", item["original_text"].casefold()))))
            return reply({"results": [{"document_id": item["id"]} for item in documents]})
        if tail and tail[0] == "documents" and len(tail) == 2:
            if request.method == "DELETE":
                bank["documents"].pop(tail[1], None)
                return reply({"success": True})
            item = bank["documents"].get(tail[1])
            return reply(item or {"detail": "not found"}, 200 if item else 404)
        if tail and tail[0] == "operations":
            if len(tail) == 3 and tail[2] == "delete" and request.method == "DELETE":
                bank["operations"].pop(tail[1], None)
                return reply({"success": True, "operation_id": tail[1]})
            require(len(tail) == 2, "controlled_operation_path")
            item = bank["operations"].get(tail[1])
            return reply(item or {"detail": "not found"}, 200 if item else 404)
        raise AssertionError("Unsupported controlled memory operation")

    def client(self, *args, **kwargs):
        return HindsightClient(*args, **kwargs, transport=httpx.MockTransport(self))
