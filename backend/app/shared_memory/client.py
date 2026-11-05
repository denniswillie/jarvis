"""Small owner-bound adapter to Hindsight 0.9.2's native HTTP API.

The authenticated caller chooses the owner and a stable, new bank generation at
composition time. Model tool arguments must never choose either. Retain acceptance
is not completed extraction; model creation is likewise asynchronous. No startup
side effects, automatic model calls, legacy SQL checks, or backfill occur here.
"""
from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from hashlib import sha256
from urllib.parse import quote, urlsplit

import httpx


def timestamp(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("An original timezone-aware timestamp is required")
    return value.isoformat()


def source_document_id(stream: str, session_id: str, source_id: str) -> str:
    if not session_id or not source_id:
        raise ValueError("Stable session and source IDs are required")
    # Length-delimited values avoid concatenation ambiguities. Separate stream
    # prefixes prevent a dialogue event and an observation sharing a document.
    value = "".join(f"{len(part)}:{part}" for part in (session_id, source_id))
    return f"jarvis-{stream}-{sha256(value.encode()).hexdigest()}"


@dataclass(frozen=True)
class BankScope:
    owner_id: str
    bank_id: str

    def __post_init__(self) -> None:
        if not self.owner_id or not self.bank_id:
            raise ValueError("Trusted owner and bank are required")

    @classmethod
    def for_owner(cls, owner_id: str, *, generation: str) -> BankScope:
        """Choose generation once for cutover, not at every process/session start."""
        if not owner_id or not generation:
            raise ValueError("Owner and a fresh, stable bank generation are required")
        digest = sha256(f"{len(owner_id)}:{owner_id}{generation}".encode()).hexdigest()
        return cls(owner_id, f"jarvis-native-{digest[:32]}")


@dataclass(frozen=True)
class MemoryDocument:
    document_id: str
    content: str
    observed_at: datetime
    context: str = ""
    metadata: Mapping[str, str] = field(default_factory=dict)

    def payload(self) -> dict:
        if not self.document_id or not self.content.strip():
            raise ValueError("Nonempty document ID and content are required")
        if any(not isinstance(k, str) or not isinstance(v, str) for k, v in self.metadata.items()):
            raise ValueError("Native metadata values must be strings")
        return {
            "document_id": self.document_id,
            "content": self.content,
            "timestamp": timestamp(self.observed_at),
            "context": self.context,
            "metadata": dict(self.metadata),
            # Stable document IDs use native replacement/content deduplication,
            # never append: replay must not append a second copy of the turn.
            "update_mode": "replace",
        }


class MemoryRequestError(RuntimeError):
    """Safe error without credentials, source text, or provider response bodies."""

    def __init__(self, code: str, status: int | None = None):
        self.code, self.status = code, status
        super().__init__(code)


@dataclass(frozen=True)
class ForgetScope:
    owner_id: str
    bank_id: str
    document_ids: tuple[str, ...]


@dataclass(frozen=True)
class ForgetResult:
    scope: ForgetScope
    deleted_document_ids: tuple[str, ...]
    failed_documents: Mapping[str, str]
    local_invalidated: bool
    local_error: str | None
    # Native deletion invalidates source-grounded observations and schedules
    # pinned-model refresh. It does not synchronously prove every summary clean.
    derived_cleanup: str = "native_async_unverified"

    @property
    def requested_deletions_applied(self) -> bool:
        """Source deletes + injected invalidation only, NOT 'forgotten everywhere'."""
        return not self.failed_documents and self.local_invalidated


_GROUNDING = (
    "Use only this person's retained evidence. Separate explicit statements, "
    "observed behavior, and tentative inference. Preserve evidence dates and "
    "contradictions; do not treat missing evidence as a fact or diagnose the person. "
)
MENTAL_MODELS = {
    "core_values_and_tradeoffs": _GROUNDING + (
        "Summarize stated priorities, evidenced tradeoffs under pressure, anti-goals, "
        "and pet peeves. Identify uncertainty instead of inventing stable traits."
    ),
    "stated_vs_revealed_habits": _GROUNDING + (
        "Summarize contextual IF-THEN habits and separately stated intentions and "
        "observed behavior. Note possible gaps without overwriting either account."
    ),
    "active_projects_and_routines": _GROUNDING + (
        "Summarize ongoing projects, routines, and protocols with their latest known "
        "status. Stopping for the day is not completion. This is background knowledge, "
        "not authority to create reminders or declare real-world actions completed."
    ),
}


class NativeMemory:
    def __init__(
        self, *, scope: BankScope, base_url: str, api_key: str,
        http_client: httpx.AsyncClient | None = None, timeout: float = 30.0,
    ):
        url = urlsplit(base_url)
        if (url.scheme not in {"http", "https"} or not url.hostname or url.username
                or url.password or url.query or url.fragment):
            raise ValueError("A configured HTTP(S) service URL is required")
        if not api_key.strip() or timeout <= 0:
            raise ValueError("A service credential and positive timeout are required")
        self.scope = scope
        self._url = base_url.rstrip("/") + "/v1/default/banks/" + quote(scope.bank_id, safe="")
        self._headers = {"Authorization": f"Bearer {api_key}"}
        self._timeout = timeout
        self._owns_http = http_client is None
        self._http = http_client or httpx.AsyncClient(follow_redirects=False)

    async def aclose(self) -> None:
        if self._owns_http:
            await self._http.aclose()

    async def _request(self, method: str, path: str = "", *, body=None, missing_ok=False):
        try:
            response = await self._http.request(
                method, self._url + path, json=body, headers=self._headers,
                timeout=self._timeout, follow_redirects=False,
            )
        except httpx.HTTPError:
            raise MemoryRequestError("hindsight_transport_error") from None
        if missing_ok and response.status_code == 404:
            return None
        if not response.is_success:
            raise MemoryRequestError("hindsight_http_error", response.status_code)
        if response.status_code == 204:
            return {}
        try:
            result = response.json()
        except ValueError:
            raise MemoryRequestError("hindsight_invalid_response") from None
        if not isinstance(result, dict) or result.get("success") is False:
            raise MemoryRequestError("hindsight_rejected_request")
        return result

    async def create_fresh_bank(self) -> dict:
        """Explicit cutover setup, not startup. Refuse to overwrite an old bank."""
        if await self._request("GET", "/profile", missing_ok=True) is not None:
            raise MemoryRequestError("hindsight_bank_already_exists")
        return await self._request("PUT", body={
            "name": "Jarvis shared memory",
            "enable_observations": True,
            "retain_mission": (
                "Remember the user's statements, experiences, and observed situations "
                "with original times and provenance. Assistant statements are not "
                "confirmed user facts. Preserve corrections and uncertainty."
            ),
        })

    async def retain(self, documents: Sequence[MemoryDocument], *, asynchronous=True) -> dict:
        """Native upsert; same document ID replaces, including explicit corrections."""
        if not documents:
            raise ValueError("At least one document is required")
        items = [document.payload() for document in documents]
        if len({item["document_id"] for item in items}) != len(items):
            raise ValueError("Duplicate document IDs within one request")
        return await self._request("POST", "/memories", body={"items": items, "async": asynchronous})

    async def recall(self, query: str, *, max_tokens=1500, budget="low", at: datetime | None = None) -> dict:
        body = self._query(query, max_tokens, budget)
        body.update(types=["world", "experience", "observation"], prefer_observations=True, trace=False)
        if at is not None:
            body["query_timestamp"] = timestamp(at)
        return await self._request("POST", "/memories/recall", body=body)

    async def reflect(self, query: str, *, max_tokens=1500, budget="low", context: str | None = None) -> dict:
        body = self._query(query, max_tokens, budget)
        if context is not None:
            body["context"] = context
        return await self._request("POST", "/reflect", body=body)

    @staticmethod
    def _query(query, max_tokens, budget) -> dict:
        if not query.strip() or not 1 <= max_tokens <= 8192 or budget not in {"low", "mid", "high"}:
            raise ValueError("Nonempty query, bounded token count, and native budget required")
        return {"query": query, "max_tokens": max_tokens, "budget": budget}

    async def get_document(self, document_id: str) -> dict | None:
        return await self._request("GET", "/documents/" + quote(document_id, safe=""), missing_ok=True)

    async def seed_mental_models(self) -> dict[str, dict]:
        """Idempotent setup; returned operation IDs are acceptance, not completion.

        Native post-consolidation refresh stays in Hindsight. A five-minute native
        refresh floor coalesces continuous dialogue without a Jarvis scheduler.
        """
        receipts = {}
        for model_id, query in MENTAL_MODELS.items():
            existing = await self._request("GET", "/mental-models/" + model_id, missing_ok=True)
            if existing is not None:
                receipts[model_id] = {"existing": True, "model": existing}
                continue
            receipts[model_id] = await self._request("POST", "/mental-models", body={
                "id": model_id, "name": model_id, "source_query": query, "max_tokens": 1024,
                "trigger": {"refresh_after_consolidation": True, "mode": "full",
                            "min_refresh_interval_seconds": 300, "exclude_mental_models": True},
            })
        return receipts

    async def forget(
        self, document_ids: Sequence[str], *,
        invalidate: Callable[[ForgetScope], Awaitable[None]],
    ) -> ForgetResult:
        """Delete explicitly resolved documents and invalidate local projections.

        The caller resolves ambiguous language before calling, persists the request
        if retry must survive restart, and must suppress/reconcile pending ingestion
        of these source IDs. The callback is owner-scoped and must be idempotent.
        It clears matching Live State/reminders, never media or Hermes history.
        0.9.2 exposes document deletion, not a single-memory DELETE endpoint; never
        silently expand a recalled fact to its whole source document here.
        """
        ids = tuple(dict.fromkeys(document_ids))
        if not ids or any(not item for item in ids):
            raise ValueError("Explicit nonempty document IDs are required")
        scope = ForgetScope(self.scope.owner_id, self.scope.bank_id, ids)
        deleted, failures = [], {}
        for document_id in ids:
            try:
                await self._request("DELETE", "/documents/" + quote(document_id, safe=""), missing_ok=True)
                deleted.append(document_id)
            except MemoryRequestError as exc:
                failures[document_id] = exc.code
        local_error = None
        try:
            # Invalidate even after a partial remote failure: do not continue to
            # assert a fact locally merely because remote deletion needs retry.
            await invalidate(scope)
        except Exception:
            local_error = "local_invalidation_failed"
        return ForgetResult(scope, tuple(deleted), failures, local_error is None, local_error)
