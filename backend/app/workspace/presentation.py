"""Bounded durable phone delivery requests, separate from content and task state.

The injected resolver/exporter own source authorization and retained bytes. This
module never substitutes a newer version, executes phone effects, or holds a DB
transaction while preparing files. Core wires only qualified producers.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
import hashlib
import json
from typing import Annotated, Literal, Protocol
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_serializer, model_validator

from app.errors import JarvisHTTPError
from app.files.transfer import CAPTURE_MAX_BYTES, FileTransferPolicy


State = Literal["pending", "claimed", "displayed", "saved_local", "dismissed", "failed", "expired"]
Action = Literal["open", "download"]
Kind = Literal["note", "file", "sandbox"]
TERMINAL = frozenset({"displayed", "saved_local", "dismissed", "failed", "expired"})
# Source availability is a domain outcome; some owners correctly use HTTP 409.
# Busy/conflicting/corrupt sources are not silently treated as withdrawn offers.
UNAVAILABLE_SOURCE_CODES = frozenset({
    "knowledge_source_unavailable", "knowledge_payload_unavailable",
    "note_source_unavailable", "note_original_unavailable", "file_source_unavailable",
    "file_original_unavailable", "file_publication_revoked",
    "sandbox_not_running", "sandbox_not_ready", "sandbox_view_replaced",
    "sandbox_view_credential_unavailable",
})


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class LocalReceipt(StrictModel):
    file_id: UUID
    version_id: str = Field(min_length=1, max_length=256)
    byte_count: Annotated[int, Field(strict=True, ge=0)]
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    destination: Literal["app_storage", "files"]


class PresentationAck(StrictModel):
    session_id: UUID
    state: Literal["claimed", "displayed", "saved_local", "dismissed", "failed"]
    local_receipt: LocalReceipt | None = None
    error_code: str | None = Field(default=None, min_length=1, max_length=100, pattern=r"^[a-z0-9_]+$")

    @model_validator(mode="after")
    def check_outcome(self):
        if (self.state == "saved_local") != (self.local_receipt is not None):
            raise ValueError("Only saved_local requires and accepts a local receipt")
        if self.error_code is not None and self.state != "failed":
            raise ValueError("Only a failed outcome accepts an error code")
        return self


class PreparedContent(StrictModel):
    file_id: UUID
    byte_count: Annotated[int, Field(strict=True, ge=0, le=CAPTURE_MAX_BYTES)]
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    transfer_policy: Literal["mp4_capture_v1", "mp4_output_v1"] | None = None
    transfer_limit_bytes: Annotated[int, Field(strict=True, ge=1, le=CAPTURE_MAX_BYTES)] | None = None

    @model_validator(mode="after")
    def bounded_transfer(self):
        _check_transfer_bounds(self.transfer_policy, self.transfer_limit_bytes, self.byte_count)
        return self

    @model_serializer(mode="wrap")
    def unchanged_document_shape(self, handler):
        result = handler(self)
        if self.transfer_policy is None:
            result.pop("transfer_policy", None)
            result.pop("transfer_limit_bytes", None)
        return result


def _check_transfer_bounds(policy, limit, count):
    if policy is None:
        if limit is not None or (count is not None and count > 8 * 1024 * 1024):
            raise ValueError("Ordinary file/note delivery retains its 8 MiB ceiling")
    elif limit is None or count is None:
        raise ValueError("Media delivery needs an exact byte count and effective limit")
    else:
        selected = FileTransferPolicy(policy, limit)
        if selected.max_bytes != limit or not 0 < count <= limit:
            raise ValueError("Media delivery exceeds its trusted effective policy")


class PresentationRequest(StrictModel):
    request_id: UUID
    action: Action
    artifact_kind: Kind
    artifact_id: UUID
    version_id: str = Field(min_length=1, max_length=256)
    format: Literal["text", "json"] | None = None
    content_ready: bool = False
    file_id: UUID | None = None
    byte_count: Annotated[int, Field(strict=True, ge=0, le=CAPTURE_MAX_BYTES)] | None = None
    content_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    transfer_policy: Literal["mp4_capture_v1", "mp4_output_v1"] | None = None
    transfer_limit_bytes: Annotated[int, Field(strict=True, ge=1, le=CAPTURE_MAX_BYTES)] | None = None
    state: State = "pending"
    created_at: datetime
    expires_at: datetime
    local_receipt: LocalReceipt | None = None
    error_code: str | None = None

    @model_validator(mode="after")
    def valid_request(self):
        if self.artifact_kind == "sandbox" and (self.action != "open" or self.file_id is not None
                or self.byte_count is not None or self.content_sha256 is not None):
            raise ValueError("Sandbox presentation is open-only and has no retained file")
        _check_transfer_bounds(self.transfer_policy, self.transfer_limit_bytes, self.byte_count)
        if self.transfer_policy is not None and (self.artifact_kind != "file" or self.action != "download"
                                                or not self.content_ready):
            raise ValueError("Only an exact prepared file download carries media policy")
        if self.created_at.tzinfo is None or self.expires_at.tzinfo is None:
            raise ValueError("Request times must be timezone aware")
        if self.expires_at <= self.created_at:
            raise ValueError("Expiry must follow reservation")
        if self.format is not None and (self.action != "download" or self.artifact_kind != "note"):
            raise ValueError("Only a note download has an export format")
        if self.action == "download" and self.content_ready:
            if any(value is None for value in (self.file_id, self.byte_count, self.content_sha256)):
                raise ValueError("Ready download needs exact retained bytes")
        if self.state == "saved_local" and (self.action != "download" or self.local_receipt is None):
            raise ValueError("Saved outcome requires a download receipt")
        if self.state == "displayed" and self.action != "open":
            raise ValueError("Only an open request can be displayed")
        return self

    def wire(self):
        return self.model_dump(mode="json", exclude_none=True)


def fail(code: str, message: str, status: int = 409):
    return JarvisHTTPError(status, code, message)


def _replace(request, **changes):
    return PresentationRequest.model_validate(request.model_dump() | changes)


def expire(request: PresentationRequest, now: datetime) -> PresentationRequest:
    if request.state == "pending" and now >= request.expires_at:
        return _replace(request, state="expired")
    return request


def acknowledge(request: PresentationRequest, ack: PresentationAck, now: datetime) -> PresentationRequest:
    """Pure transition rules. The repository binds owner/session under its lock."""
    request = expire(request, now)
    if ack.state == "claimed" and request.state != "pending":
        # A retry of a claim observes the actual current state, never rewinds it.
        return request
    if request.state in TERMINAL:
        if (ack.state == request.state and ack.local_receipt == request.local_receipt
                and ack.error_code == request.error_code):
            return request
        raise fail("presentation_terminal_conflict", "This phone request already has a different final outcome.")
    if ack.state == "claimed":
        if not request.content_ready:
            raise fail("content_not_ready", "The exact content is not ready for the phone.")
        return _replace(request, state="claimed")
    if ack.state in {"displayed", "saved_local"}:
        if request.state != "claimed":
            raise fail("presentation_not_claimed", "Claim the exact request before reporting its local effect.")
        expected = "open" if ack.state == "displayed" else "download"
        if request.action != expected:
            raise fail("presentation_action_mismatch", "The local outcome does not match the requested action.")
    if ack.state == "saved_local":
        receipt = ack.local_receipt
        if (receipt.file_id, receipt.version_id, receipt.byte_count, receipt.content_sha256) != (
                request.file_id, request.version_id, request.byte_count, request.content_sha256):
            raise fail("presentation_receipt_mismatch", "The saved bytes do not match this exact prepared version.")
    return _replace(request, state=ack.state, local_receipt=ack.local_receipt, error_code=ack.error_code)


class RequestContext(Protocol):
    owner_id: str
    session_id: UUID
    operation_id: UUID | None


class PresentationRepository:
    """Small delivery journal; no duplicate artifact contents or task engine."""
    def __init__(self, pool):
        self.pool = pool

    @asynccontextmanager
    async def _transaction(self, ctx, *, write=True):
        if not isinstance(ctx.session_id, UUID):
            raise fail("workspace_session_required", "An owned session is required.", 403)
        async with self.pool.acquire() as conn:
            async with conn.transaction(readonly=not write):
                session_query = "select true from jarvis_sessions where owner_id=$1 and id=$2"
                if write:
                    session_query += " for share"
                if not await conn.fetchval(session_query,
                                           ctx.owner_id, ctx.session_id):
                    raise fail("workspace_session_unavailable", "The session is outside this owner scope.", 404)
                # Per-session capacity and transitions serialize only this small journal.
                if write:
                    await conn.execute("select pg_advisory_xact_lock(hashtextextended($1, 0))",
                                       "jarvis.workspace.requests:" + json.dumps([ctx.owner_id, str(ctx.session_id)]))
                yield conn

    @staticmethod
    def _decode(row):
        if row is None:
            return None
        value = row["payload"]
        return PresentationRequest.model_validate_json(value) if isinstance(value, str) else PresentationRequest.model_validate(value)

    async def _row(self, conn, ctx, request_id, *, lock=True):
        query = """select * from jarvis_workspace_requests
            where owner_id=$1 and request_id=$2"""
        row = await conn.fetchrow(query + (" for update" if lock else ""), ctx.owner_id, request_id)
        if row is not None and row["session_id"] != ctx.session_id:
            raise fail("workspace_operation_scope", "This phone request belongs to another session.", 403)
        return row

    async def _save(self, conn, ctx, request):
        await conn.execute("""update jarvis_workspace_requests set payload=$4::jsonb
            where owner_id=$1 and session_id=$2 and request_id=$3""",
            ctx.owner_id, ctx.session_id, request.request_id, json.dumps(request.wire(), separators=(",", ":")))

    async def get(self, ctx, request_id, *, digest=None):
        async with self._transaction(ctx, write=False) as conn:
            row = await self._row(conn, ctx, request_id, lock=False)
            if row is None:
                return None
            if digest is not None and row["parameters_digest"] != digest:
                raise fail("operation_conflict", "This operation was already used for a different phone request.")
            # Expiry is observable without a write; the next mutation persists it.
            return expire(self._decode(row), await conn.fetchval("select clock_timestamp()"))

    async def reserve(self, ctx, *, digest, target):
        async with self._transaction(ctx) as conn:
            row = await self._row(conn, ctx, ctx.operation_id)
            if row is not None:
                if row["parameters_digest"] != digest:
                    raise fail("operation_conflict", "This operation was already used for a different phone request.")
                original = self._decode(row)
                current = expire(original, await conn.fetchval("select clock_timestamp()"))
                if current != original:
                    await self._save(conn, ctx, current)
                return current
            now = await conn.fetchval("select clock_timestamp()")
            rows = await conn.fetch("""select * from jarvis_workspace_requests
                where owner_id=$1 and session_id=$2 and payload->>'state' in ('pending','claimed') for update""",
                ctx.owner_id, ctx.session_id)
            count = 0
            for row in rows:
                original = self._decode(row)
                current = expire(original, now)
                if current != original:
                    await self._save(conn, ctx, current)
                count += current.state in {"pending", "claimed"}
            if count >= 8:
                raise fail("presentation_capacity", "The phone has eight outstanding requests; resolve one first.")
            request = PresentationRequest(request_id=ctx.operation_id, created_at=now,
                expires_at=now + timedelta(minutes=5), **target)
            await conn.execute("""insert into jarvis_workspace_requests
                (owner_id,session_id,request_id,parameters_digest,payload,created_at)
                values ($1,$2,$3,$4,$5::jsonb,$6)""", ctx.owner_id, ctx.session_id, ctx.operation_id,
                digest, json.dumps(request.wire(), separators=(",", ":")), now)
            return request

    async def mark_ready(self, ctx, request_id, prepared):
        async with self._transaction(ctx) as conn:
            row = await self._row(conn, ctx, request_id)
            if row is None:
                raise fail("presentation_not_found", "The phone request is unavailable.", 404)
            original = self._decode(row)
            request = expire(original, await conn.fetchval("select clock_timestamp()"))
            if request != original:
                await self._save(conn, ctx, request)
            if request.state != "pending":
                return request  # Late preparation cannot reactivate expiry or final state.
            changes = prepared.model_dump()
            if request.content_ready:
                # Wire omission preserves legacy documents; it cannot erase a
                # media policy from the exact preparation identity on replay.
                if any(getattr(request, key) != getattr(prepared, key) for key in PreparedContent.model_fields):
                    raise fail("presentation_content_conflict", "Prepared bytes changed for this phone request.")
                return request
            request = _replace(request, content_ready=True, **changes)
            await self._save(conn, ctx, request)
            return request

    async def ack(self, ctx, request_id, ack):
        if ack.session_id != ctx.session_id:
            raise fail("workspace_session_mismatch", "The acknowledgement belongs to a different session.", 403)
        async with self._transaction(ctx) as conn:
            row = await self._row(conn, ctx, request_id)
            if row is None:
                raise fail("presentation_not_found", "The phone request is unavailable.", 404)
            request = acknowledge(self._decode(row), ack, await conn.fetchval("select clock_timestamp()"))
            await self._save(conn, ctx, request)
            return request

    async def list(self, ctx):
        async with self._transaction(ctx, write=False) as conn:
            # Active claims/offers cannot be displaced by many completed requests.
            # Expired-but-unwritten offers must not occupy the active priority.
            now = await conn.fetchval("select clock_timestamp()")
            rows = await conn.fetch("""select * from jarvis_workspace_requests
                where owner_id=$1 and session_id=$2
                order by (payload->>'state' = 'claimed' or
                    (payload->>'state' = 'pending' and (payload->>'expires_at')::timestamptz > $3)) desc,
                    created_at desc,request_id desc
                limit 8""", ctx.owner_id, ctx.session_id, now)
            return [expire(self._decode(row), now) for row in rows]


class PresentationService:
    """Resolver(ctx, kind, id, version) -> authorized exact version_id string.

    Exporter(ctx, request) -> PreparedContent must reconcile its same operation
    key, preserve source dependencies and never create a duplicate on response
    loss. It operates after reservation, outside journal/source transactions.
    """
    def __init__(self, repository, *, resolver, exporter=None):
        self.repository, self.resolver, self.exporter = repository, resolver, exporter

    async def _authorize(self, ctx, request):
        if request.artifact_kind == "note":
            raise fail("presentation_retired", "Note presentation is no longer supported.", 410)
        version = await self.resolver(ctx, request.artifact_kind, request.artifact_id, request.version_id)
        if version != request.version_id:
            raise fail("presentation_version_unavailable", "The exact saved version is unavailable.", 404)

    async def present(self, ctx, *, action: Action, artifact_id: UUID, artifact_kind: Kind = "file",
                      version_id: str | None = None, format: str | None = None):
        if not isinstance(ctx.operation_id, UUID):
            raise fail("workspace_admission_required", "A trusted admitted operation is required.", 403)
        if (action not in {"open", "download"} or artifact_kind not in {"file", "sandbox"}
                or (artifact_kind == "sandbox" and action != "open")):
            raise fail("presentation_invalid", "Unsupported phone request.", 422)
        if format is not None:
            raise fail("presentation_invalid_format", "This request does not accept an export format.", 422)
        parameters = [action, artifact_kind, str(artifact_id), version_id, format]
        digest = hashlib.sha256(json.dumps(parameters, separators=(",", ":"), ensure_ascii=True).encode()).hexdigest()
        request = await self.repository.get(ctx, ctx.operation_id, digest=digest)
        if request is not None and request.state != "pending":
            # Known completed/claimed/expired work is a receipt, not a new
            # presentation effect. Source withdrawal cannot erase that outcome.
            return self.outcome(request)
        if request is None:
            if action == "download" and self.exporter is None:
                raise fail("workspace_export_unavailable", "The exact-version export producer is unavailable.", 503)
            exact = await self.resolver(ctx, artifact_kind, artifact_id, version_id)
            request = await self.repository.reserve(ctx, digest=digest, target={
                "action": action, "artifact_kind": artifact_kind, "artifact_id": artifact_id,
                "version_id": exact, "format": format, "content_ready": action == "open"})
        await self._authorize(ctx, request)
        if request.action == "download" and not request.content_ready and request.state == "pending":
            # Retried preparation is permitted only via the same-key producer.
            if self.exporter is None:
                return self.outcome(request, uncertain=True)
            try:
                prepared = PreparedContent.model_validate(await self.exporter(ctx, request))
                request = await self.repository.mark_ready(ctx, request.request_id, prepared)
            except JarvisHTTPError as error:
                if error.details.get("effect_committed") is False:
                    request = await self.repository.ack(ctx, request.request_id,
                        PresentationAck(session_id=ctx.session_id, state="failed", error_code=error.code))
                else:
                    return self.outcome(request, uncertain=True)
            except Exception:
                # Unknown upload/adoption outcome is not evidence of no effect.
                return self.outcome(request, uncertain=True)
        return self.outcome(request)

    async def operation_receipt(self, ctx, operation_id):
        """Core reconciles its executor with a committed domain reservation.

        Frozen identities/outcomes only; the receipt never grants content access.
        The repository still binds authenticated owner and persisted session.
        """
        request = await self.repository.get(ctx, operation_id)
        return None if request is None else self.outcome(request)

    async def list_requests(self, ctx):
        items = []
        for request in await self.repository.list(ctx):
            if request.artifact_kind == "note":
                continue  # Preserve retired offers without resurfacing the removed feature.
            try:
                await self._authorize(ctx, request)
            except JarvisHTTPError as error:
                if error.status_code not in {403, 404, 410} and error.code not in UNAVAILABLE_SOURCE_CODES:
                    raise
                continue  # Never disclose now-withdrawn content through a saved offer.
            items.append(request.wire())
        return {"items": items}

    async def acknowledge(self, ctx, request_id, ack):
        request = await self.repository.get(ctx, request_id)
        if request is None:
            raise fail("presentation_not_found", "The phone request is unavailable.", 404)
        if ack.state == "claimed":
            await self._authorize(ctx, request)
        # Terminal ACKs reconcile a prior local attempt against frozen identity,
        # even after source withdrawal. Repository scope/transition/byte checks
        # still apply; this grants no content read, new effect or source revival.
        result = await self.repository.ack(ctx, request_id, ack)
        return {"request": result.wire()}

    @staticmethod
    def outcome(request, *, uncertain=False):
        completed = request.state in {"displayed", "saved_local"}
        outcome = "outcome_unknown" if uncertain or request.state == "claimed" else (
            "completed" if completed else "failed" if request.state in {"failed", "expired", "dismissed"} else "accepted")
        summary = {"displayed": "The phone reported displaying the saved version.",
                   "saved_local": "The phone reported saving the exact version locally.",
                   "failed": "The phone request failed.", "expired": "The phone offer expired.",
                   "dismissed": "The phone request was dismissed.",
                   "claimed": "The phone claimed the request; its local outcome is not yet known.",
                   "pending": "The phone request was saved; its local effect is not yet confirmed."}[request.state]
        if request.artifact_kind == "sandbox" and request.state == "displayed":
            summary = "The phone reported displaying the live sandbox screen."
        return {"status": "success", "outcome": outcome, "effect_committed": True,
                "operation_id": str(request.request_id), "summary": summary,
                "artifact": {"artifact_id": str(request.artifact_id), "version_id": request.version_id,
                             "kind": request.artifact_kind},
                "presentation": {"request_id": str(request.request_id), "state": request.state}}
