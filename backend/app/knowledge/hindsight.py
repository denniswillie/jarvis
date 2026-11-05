"""Bounded adapter to Hindsight's maintained REST API, tested against 0.9.2.

No reflection/answer generation, saved mental models or provider-native tools
are exposed. Bank and operation identities must come from trusted repositories.
"""
from __future__ import annotations

import json
import math
import time
from urllib.parse import quote
from uuid import UUID

import httpx

from app.knowledge.connection import hindsight_origin
from app.knowledge.models import SourceError, SourceVersion
from app.knowledge.service_policy import ServicePolicyError, validate_bank_configuration


class HindsightError(SourceError):
    def __init__(self, code, *, retryable=False):
        super().__init__(code, "The configured memory service could not complete this operation.", 503)
        self.retryable = retryable


def original_document(source: SourceVersion) -> str:
    if source.source_type == "recording_file":
        # The immutable technical original remains in Files/source storage.
        # Semantic memory only needs to know that the saved resource exists.
        content = source.content
        record = content.get("record", {})
        if (content.get("format") != "jarvis.recording_file_original.v1"
                or not isinstance(record, dict) or record.get("content_type") != "video/mp4"):
            raise SourceError("recording_memory_reference_invalid", "The saved recording reference is unavailable.")
        try:
            file_id = str(UUID(record["id"]))
        except (ValueError, TypeError, KeyError):
            raise SourceError("recording_memory_reference_invalid", "The saved recording reference is unavailable.") from None
        return json.dumps({"event": "recording_saved", "saved_at": source.recorded_at.isoformat(),
            "resource": {"file_id": file_id, "version_id": source.document_id, "media_type": "video/mp4"}},
            sort_keys=True, separators=(",", ":"))
    keys = ("source_id", "revision", "content_digest", "source_type", "trust",
            "source_time", "time_basis", "recorded_at", "content")
    payload = source.model_dump(mode="json", include=set(keys))
    return json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":"), allow_nan=False)


def document_metadata(source: SourceVersion) -> dict:
    return {"jarvis_source_id": str(source.source_id), "jarvis_revision": str(source.revision),
            "jarvis_digest": source.content_digest, "jarvis_area_id": str(source.area_id)}


class HindsightClient:
    def __init__(self, base_url, api_key, *, timeout=20.0, max_response_bytes=2_000_000,
                 transport=None, private_http_origin=None):
        base_url = hindsight_origin(base_url, approved_http_origin=private_http_origin)
        if (not isinstance(api_key, str) or not api_key.strip()
                or not math.isfinite(timeout) or timeout <= 0
                or type(max_response_bytes) is not int or max_response_bytes < 1024):
            raise ValueError("Invalid private Hindsight connection configuration")
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"), headers={"Authorization": "Bearer " + api_key},
            timeout=timeout, follow_redirects=False, trust_env=False, transport=transport,
        )
        self.max_response_bytes = max_response_bytes
        self.requests = self.errors = 0
        self.elapsed_seconds = 0.0

    def diagnostics(self):
        return {"requests": self.requests, "errors": self.errors,
                "request_seconds": self.elapsed_seconds, "provider_token_usage": None}

    async def close(self):
        await self._client.aclose()

    async def _call(self, method, path, *, body=None, missing=False):
        started = time.monotonic()
        self.requests += 1
        try:
            async with self._client.stream(method, path, **({"json": body} if body is not None else {})) as response:
                if missing and response.status_code == 404:
                    return None
                if not 200 <= response.status_code < 300:
                    raise HindsightError("hindsight_http_error",
                                         retryable=response.status_code in {408, 429} or response.status_code >= 500)
                raw = bytearray()
                async for chunk in response.aiter_bytes():
                    raw.extend(chunk)
                    if len(raw) > self.max_response_bytes:
                        raise HindsightError("hindsight_response_too_large")
                payload = json.loads(raw) if raw else {}
                if not isinstance(payload, dict):
                    raise HindsightError("hindsight_invalid_response")
                return payload
        except (httpx.HTTPError, TimeoutError):
            self.errors += 1
            raise HindsightError("hindsight_transport_unavailable", retryable=True) from None
        except (ValueError, UnicodeError):
            self.errors += 1
            raise HindsightError("hindsight_invalid_response") from None
        except HindsightError:
            self.errors += 1
            raise
        finally:
            self.elapsed_seconds += time.monotonic() - started

    @staticmethod
    def _bank(bank):
        if not isinstance(bank, str) or not bank or len(bank) > 200:
            raise ValueError("Invalid trusted bank identity")
        return "/v1/default/banks/" + quote(bank, safe="")

    async def ensure_bank(self, bank):
        await self._call("PUT", self._bank(bank), body={
            "name": "Jarvis source-backed knowledge", "enable_reranking": False,
            "retain_custom_instructions": (
                "Preserve the supplied source origin and uncertainty. User reports, camera observations, "
                "assistant output and tool receipts are different evidence. An intention, absence of "
                "evidence, or unconfirmed notification is not a completed action. "
                "source_time=null means the original event time is unknown."
            ),
        })
        await self.check_bank_policy(bank)

    async def check_bank_policy(self, bank):
        models = await self._call("GET", self._bank(bank) + "/mental-models?limit=1")
        if type(models.get("total")) is not int or not isinstance(models.get("items"), list):
            raise HindsightError("hindsight_invalid_cache_inventory")
        if models["total"] or models["items"]:
            raise HindsightError("hindsight_saved_models_not_supported")
        # This endpoint intentionally omits server-wide privacy fields. The
        # checked deployment launcher enforces those before starting the API.
        configuration = await self._call("GET", self._bank(bank) + "/config")
        try:
            validate_bank_configuration(configuration, bank)
        except ServicePolicyError:
            raise HindsightError("hindsight_bank_policy_not_satisfied") from None

    async def retain(self, source: SourceVersion, operation_id: UUID):
        result = await self._call("POST", self._bank(source.bank_name) + "/memories", body={
            "async": True, "operation_id": str(operation_id), "items": [{
                "content": original_document(source), "document_id": source.document_id,
                "timestamp": source.source_time.isoformat() if source.source_time else "unset",
                "metadata": document_metadata(source),
            }],
        })
        if result.get("operation_id") != str(operation_id) or result.get("async") is not True:
            raise HindsightError("hindsight_operation_receipt_mismatch")
        return result

    async def operation(self, bank, operation_id):
        identity = str(UUID(str(operation_id)))
        result = await self._call("GET", self._bank(bank) + "/operations/" + identity, missing=True)
        if result is not None:
            if result.get("operation_id") != identity or result.get("status") not in {
                "pending", "processing", "completed", "failed", "cancelled", "not_found",
            }:
                raise HindsightError("hindsight_operation_receipt_mismatch")
            # Native 0.9.2 also represents absence as HTTP 200. Validate the
            # exact requested identity before treating that receipt as missing.
            if result["status"] == "not_found":
                return None
        return result

    @staticmethod
    def child_operation_ids(operation):
        children = operation.get("child_operations")
        if children is None:
            return ()
        if not isinstance(children, list) or len(children) > 64:
            raise HindsightError("hindsight_invalid_child_operations")
        identities = []
        for child in children:
            if not isinstance(child, dict) or child.get("status") not in {
                "pending", "processing", "completed", "failed", "cancelled",
            }:
                raise HindsightError("hindsight_invalid_child_operations")
            try:
                identities.append(UUID(str(child.get("operation_id"))))
            except (ValueError, TypeError, AttributeError):
                raise HindsightError("hindsight_invalid_child_operations") from None
        return tuple(identities)

    async def terminal_operations(self, bank, operation_id):
        """Collect a bounded terminal inventory; absence is not proof of cleanup."""
        pending, collected = [UUID(str(operation_id))], []
        while pending:
            identity = pending.pop(0)
            if identity in collected:
                continue
            if len(collected) >= 64:
                raise HindsightError("hindsight_operation_inventory_too_large")
            operation = await self.operation(bank, identity)
            if operation is None:
                raise HindsightError("hindsight_retention_outcome_unknown", retryable=True)
            if operation["status"] in {"pending", "processing"} or operation.get("next_retry_at"):
                return None
            collected.append(identity)
            pending.extend(self.child_operation_ids(operation))
        return tuple(collected)

    async def delete_terminal_operation(self, bank, operation_id):
        identity = str(UUID(str(operation_id)))
        result = await self._call(
            "DELETE", self._bank(bank) + "/operations/" + identity + "/delete", missing=True,
        )
        if result is None:
            return False
        if result.get("success") is not True or result.get("operation_id") != identity:
            raise HindsightError("hindsight_operation_delete_receipt_mismatch")
        return True

    async def erase_operations(self, bank, operation_ids):
        """Execute only a durable, previously terminal inventory supplied by Jarvis."""
        identities = tuple(UUID(str(item)) for item in operation_ids)
        if len(identities) > 64 or len(set(identities)) != len(identities):
            raise ValueError("Invalid prepared memory cleanup inventory")
        states = {}
        for identity in identities:
            operation = await self.operation(bank, identity)
            states[identity] = operation
            # Absence is safe here only because the caller persisted this ID
            # before deletion. Never use this method to discover an unknown send.
            if operation is None:
                continue
            if operation["status"] in {"pending", "processing"} or operation.get("next_retry_at"):
                return False
            if not set(self.child_operation_ids(operation)).issubset(identities):
                raise HindsightError("hindsight_cleanup_inventory_changed", retryable=True)
        # Preserve the root until its discovered children have been deleted.
        for identity in reversed(identities):
            if states[identity] is not None:
                await self.delete_terminal_operation(bank, identity)
        for identity in identities:
            if await self.operation(bank, identity) is not None:
                return False
        return True

    async def document(self, bank, document_id):
        result = await self._call("GET", self._bank(bank) + "/documents/" + quote(document_id, safe=""), missing=True)
        if result is not None and (result.get("id") != document_id or result.get("bank_id") != bank):
            raise HindsightError("hindsight_document_scope_mismatch")
        return result

    async def delete_document(self, bank, document_id):
        await self._call("DELETE", self._bank(bank) + "/documents/" + quote(document_id, safe=""), missing=True)
        if await self.document(bank, document_id) is not None:
            raise HindsightError("hindsight_document_cleanup_pending", retryable=True)

    async def recall(self, bank, query, *, max_tokens=2400):
        if not isinstance(query, str) or not query.strip() or len(query) > 8000:
            raise ValueError("Invalid memory query")
        if type(max_tokens) is not int or not 128 <= max_tokens <= 8000:
            raise ValueError("Invalid memory retrieval budget")
        await self.check_bank_policy(bank)
        # Similarity is not a calibrated relevance probability. The native 0.3
        # default can discard a relevant original after small extraction changes.
        # Keep native ranking/budgets; consumers still verify source and relevance.
        result = await self._call("POST", self._bank(bank) + "/memories/recall",
                                  body={"query": query, "budget": "low", "max_tokens": max_tokens,
                                        "min_scores": {"semantic": 0},
                                        "include": {"source_facts": {
                                            "max_tokens": max_tokens,
                                            "max_tokens_per_observation": min(max_tokens, 1200),
                                        }}})
        if not isinstance(result.get("results"), list):
            raise HindsightError("hindsight_invalid_recall")
        return result

    @staticmethod
    def verify_document(source, document):
        if (document is None or document.get("id") != source.document_id
                or document.get("bank_id") != source.bank_name
                or document.get("original_text") != original_document(source)
                or any((document.get("document_metadata") or {}).get(k) != v
                       for k, v in document_metadata(source).items())):
            raise HindsightError("hindsight_original_evidence_mismatch")
