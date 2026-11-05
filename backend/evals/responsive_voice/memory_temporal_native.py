"""Opt-in temporal qualification transport using the existing Hindsight client.

One parent and six sequential children share a private budget, not a memory
store. Counting precedes dispatch, including failed/ambiguous requests. No
whole-run retry, service startup, model substitution or voice client exists.
"""
import json
import os
from pathlib import Path
import time
from urllib.parse import unquote, urlsplit

from app.knowledge.hindsight import HindsightClient
from evals.responsive_voice.memory_lifecycle_qualification import configured_inputs, require, save


RELEASE_ENV = "JARVIS_MEMORY_TEMPORAL_ACTUAL_RELEASE"
MAIN_HTTP_LIMIT = 1200
CLEANUP_HTTP_LIMIT = 400
RECALL_LIMIT = 12
RETAIN_LIMIT = 32
RETAIN_CHARACTER_LIMIT = 128000


def require_actual_release():
    require(os.getenv(RELEASE_ENV) == "1", "actual_temporal_release_required")


def actual_inputs():
    # Check release BEFORE loading configuration or reading a credential file.
    require_actual_release()
    dsn, resolution = configured_inputs()
    require(not os.getenv("SECOND_BRAIN_HINDSIGHT_API_KEY"), "actual_temporal_inline_key_forbidden")
    path = Path(os.getenv("SECOND_BRAIN_HINDSIGHT_API_KEY_FILE", ""))
    require(path.is_absolute() and path.is_file(), "actual_temporal_key_file_required")
    require(bool(dsn) and resolution.configured, "actual_temporal_configuration_unavailable")
    return dsn, resolution.configuration


class PersistentActualMemory:
    """Existing real client plus a small process-persistent accounting seam.

    All clients borrow the same bank and budget. Sequential process ownership
    is mandatory; each child's stage is claimed once before any database work.
    HTTP attempts are NOT counts of the service's internal model calls/tokens.
    """

    def __init__(self, path, bank, origin, *, create=False):
        self.path, self.bank, self.origin = Path(path), bank, origin.rstrip("/")
        require(self.path.is_absolute() and bank.startswith("jarvis_temporal_eval_"), "native_transport_scope")
        if create:
            require(not self.path.exists(), "native_budget_already_exists")
            save(self.path, {"format": "jarvis.temporal_native_budget.v1", "bank": bank, "origin": self.origin,
                "main_http": 0, "cleanup_http": 0, "recalls": 0, "retain_requests": 0,
                "retain_characters": 0, "failed_http": 0, "http_elapsed_ms": 0,
                "cleaning": False, "bank_admitted": False, "children": [],
                "main_deadline_monotonic": time.monotonic() + 480,
                "provider_internal_model_calls": None, "provider_tokens": None, "dollar_cost": None})
        self.read()

    def read(self):
        require(self.path.stat().st_size <= 65536, "native_budget_size")
        state = json.loads(self.path.read_text())
        require(state["bank"] == self.bank and state["origin"] == self.origin, "native_budget_scope")
        return state

    def admit_bank(self):
        state = self.read()
        require(not state["cleaning"] and not state["bank_admitted"], "native_bank_already_admitted")
        state["bank_admitted"] = True
        save(self.path, state)

    def claim_child(self, stage, position):
        state = self.read()
        require(state["bank_admitted"] and not state["cleaning"], "native_child_not_admitted")
        require(type(position) is int and 0 <= position < 6 and len(state["children"]) == position
                and stage not in {child["stage"] for child in state["children"]}, "native_child_replay_or_order")
        state["children"].append({"stage": stage, "pid": os.getpid()})
        save(self.path, state)

    def begin_cleanup(self):
        state = self.read()
        require(not state["cleaning"], "native_cleanup_already_started")
        state.update(cleaning=True, cleanup_deadline_monotonic=time.monotonic() + 180)
        save(self.path, state)

    def _reserve(self, method, path, body):
        parsed = urlsplit(path)
        require(not parsed.scheme and not parsed.netloc and not parsed.fragment,
                "native_request_not_relative")
        parts = [unquote(part) for part in parsed.path.split("/") if part]
        inventory = parts == ["v1", "default", "banks"] and method == "GET" and not parsed.query
        state = self.read()
        recall = retain = False
        characters = 0
        if not inventory:
            require(parts[:3] == ["v1", "default", "banks"] and len(parts) >= 4
                    and parts[3] == self.bank, "native_foreign_bank")
            require(state["bank_admitted"], "native_bank_not_admitted")
            tail = parts[4:]
            allowed = ((not tail and method in {"PUT", "DELETE"})
                or (tail in (["config"], ["mental-models"]) and method == "GET")
                or (tail in (["memories"], ["memories", "recall"]) and method == "POST")
                or (len(tail) == 2 and tail[0] == "documents" and method in {"GET", "DELETE"})
                or (len(tail) == 2 and tail[0] == "operations" and method == "GET")
                or (len(tail) == 3 and tail[0] == "operations" and tail[2] == "delete" and method == "DELETE"))
            require(allowed, "native_operation_not_qualified")
            recall, retain = tail == ["memories", "recall"], tail == ["memories"]
            if recall:
                require(not state["cleaning"] and state["recalls"] < RECALL_LIMIT, "native_recall_budget")
            if retain:
                items = body.get("items") if isinstance(body, dict) else None
                require(isinstance(items, list) and len(items) == 1
                        and isinstance(items[0], dict) and isinstance(items[0].get("content"), str),
                        "native_retain_shape")
                characters = len(items[0]["content"])
                require(state["retain_requests"] < RETAIN_LIMIT
                        and state["retain_characters"] + characters <= RETAIN_CHARACTER_LIMIT,
                        "native_retain_budget")
        field, limit, deadline = (("cleanup_http", CLEANUP_HTTP_LIMIT, "cleanup_deadline_monotonic")
            if state["cleaning"] else ("main_http", MAIN_HTTP_LIMIT, "main_deadline_monotonic"))
        require(time.monotonic() < state[deadline], "native_request_deadline")
        require(state[field] < limit, "native_http_budget")
        state[field] += 1
        state["recalls"] += int(recall)
        state["retain_requests"] += int(retain)
        state["retain_characters"] += characters
        save(self.path, state)

    def client(self, base_url, api_key, **kwargs):
        require(base_url.rstrip("/") == self.origin, "native_client_origin_changed")
        client = HindsightClient(base_url, api_key, **kwargs)
        call = client._call

        async def counted(method, path, *, body=None, missing=False):
            self._reserve(method, path, body)
            started, failed = time.monotonic(), True
            try:
                result = await call(method, path, body=body, missing=missing)
                failed = False
                return result
            finally:
                state = self.read()
                state["failed_http"] += int(failed)
                state["http_elapsed_ms"] = round(state["http_elapsed_ms"]
                    + (time.monotonic() - started) * 1000, 3)
                save(self.path, state)
        client._call = counted
        return client

    async def preflight(self, configuration):
        client = self.client(configuration.base_url, configuration.api_key.get_secret_value(),
            timeout=min(6, configuration.http_timeout_seconds), private_http_origin=configuration.private_http_origin)
        try:
            inventory = await client._call("GET", "/v1/default/banks")
            require(isinstance(inventory.get("banks"), list), "native_inventory_invalid")
            require(self.bank not in {item["bank_id"] for item in inventory["banks"]}, "native_bank_already_exists")
            self.admit_bank()
        finally:
            await client.close()

    def report(self):
        state = self.read()
        return {key: value for key, value in state.items() if key not in {
            "origin", "main_deadline_monotonic", "cleanup_deadline_monotonic"}} | {
            "usage_limitation": "HTTP attempts and submitted characters are not provider-internal model calls, tokens or dollar cost."}
