"""Source-backed response admission, not a replacement conversation or memory engine.

Track only trusted source identities actually supplied to this provider session.
A response gets an immutable dependency payload before its provider write; later
native output can reuse that payload as its original. No source text is copied
here. This does not by itself purge provider history or stop already-sent audio.
"""
from __future__ import annotations

import asyncio

from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import logging
from uuid import UUID

from app.tasks.models import TaskError
from app.guard_diagnostics import guard_context, guard_wait_stage
from app.realtime.telemetry import latency_context, latency_event, latency_phase


_LOCK_LOGGER = logging.getLogger(__name__)
_LOCK_LOG_LIMIT = 8
# Default 128 original roots plus a response snapshot's materialized payload.
# Larger configured source budgets are not changed; diagnostics explicitly omit
# their excess roots rather than allowing an unbounded log record.
_LOCK_LOG_ROOT_LIMIT = 129


def _diagnostic_uuid(value):
    try:
        if isinstance(value, UUID):
            return str(UUID(int=value.int))
        if type(value) is str and len(value) <= 45:
            return str(UUID(value))
    except (ValueError, TypeError, AttributeError):
        pass
    return None


class LiveContextError(TaskError):
    def __init__(self, code="live_context_revoked"):
        super().__init__(code, "The audio connection's source context must be rebuilt.", 409)


@dataclass(frozen=True)
class ResponseSourceSnapshot:
    payload_id: UUID
    sources: tuple[UUID, ...]
    version: int
    prepared_at: datetime | None = None


class LiveResponseSources:
    def __init__(self, pool, *, owner_id: str, session_id: UUID, audio_session_id: str,
                 max_sources=128, max_responses=512):
        self.pool = pool
        self.owner_id, self.session_id, self.audio_session_id = owner_id, session_id, audio_session_id
        self.max_sources, self.max_responses = max_sources, max_responses
        self._sources: set[UUID] = set()
        self._version = 0
        self._responses: dict[str, ResponseSourceSnapshot] = {}
        self._changed = asyncio.Event()
        self._borrowed_transaction = ContextVar("live_response_source_transaction", default=None)
        self._lock_rejections_reported = 0
        self._latency_trace = None

    def bind_trace(self, trace) -> None:
        self._latency_trace = trace

    @asynccontextmanager
    async def _timed_transaction(self, fields):
        # Wrap the original contexts; no second acquisition or new transaction.
        async with latency_context(
            self._latency_trace, "response_source_latency", self.pool.acquire(),
            enter="pool_acquire", exit="pool_release", fields=fields,
        ) as connection, latency_context(
            self._latency_trace, "response_source_latency", connection.transaction(),
            enter="transaction_enter", exit="transaction_exit", fields=fields,
        ):
            yield connection

    @asynccontextmanager
    async def use_transaction(self, connection):
        """Share canonical writer locks only within the exact owning task."""
        if not connection.is_in_transaction():
            raise LiveContextError("live_context_transaction_required")
        token = self._borrowed_transaction.set((asyncio.current_task(), connection))
        try:
            yield
        finally:
            self._borrowed_transaction.reset(token)

    def _current_connection(self):
        borrowed = self._borrowed_transaction.get()
        return borrowed[1] if borrowed is not None and borrowed[0] is asyncio.current_task() else None

    @property
    def has_sources(self):
        return bool(self._sources)

    def observe(self, sources):
        additions = set(sources)
        if any(not isinstance(value, UUID) for value in additions):
            raise LiveContextError("live_context_source_invalid")
        combined = self._sources | additions
        if len(combined) > self.max_sources:
            raise LiveContextError("live_context_source_limit")
        if combined != self._sources:
            self._sources = combined
            self._version += 1
            self._changed.set()

    async def wait_for_invalidation(self, *, interval=1.0, timeout=5.0):
        while True:
            if not self.has_sources:
                await self._changed.wait()
            self._changed.clear()
            try:
                async with asyncio.timeout(timeout), self.context_guard():
                    pass
            except LiveContextError as error:
                if error.code == "live_context_changed":
                    continue
                raise
            try:
                async with asyncio.timeout(interval):
                    await self._changed.wait()
            except TimeoutError:
                pass

    def source_for(self, request_id):
        snapshot = self._responses.get(request_id)
        return snapshot.payload_id if snapshot is not None else None

    def snapshot_for(self, request_id):
        """Immutable prepared evidence; only actual response admission may pin it."""
        return self._responses.get(request_id)

    @asynccontextmanager
    async def output_guard(self, snapshot: ResponseSourceSnapshot):
        """Authorize delivery of the accepted as-of evidence, not a new action.

        Original roots and their materialized copy remain subject to canonical
        withdrawal/correction/permission checks. Adding unrelated context does
        not rewrite this snapshot or make its observations newly current.
        """
        if not isinstance(snapshot, ResponseSourceSnapshot) or not any(
                snapshot is prepared for prepared in self._responses.values()):
            raise LiveContextError("live_context_source_invalid")
        roots = (*snapshot.sources, snapshot.payload_id)
        borrowed = self._current_connection()
        if borrowed is not None:
            if not borrowed.is_in_transaction():
                raise LiveContextError("live_context_transaction_required")
            await self._lock(borrowed, roots)
            yield
            return
        if self.pool is None:
            raise LiveContextError("live_context_storage_unavailable")
        async with self.pool.acquire() as connection, connection.transaction():
            await self._lock(connection, roots)
            yield

    async def _lock(self, connection, sources):
        attempted = list(sources)
        with guard_wait_stage("response_source_lock"):
            available = await connection.fetchval(
                "select lock_jarvis_copy_sources($1,$2::uuid[])", self.owner_id, attempted)
        if not available:
            if available is False:
                self._report_lock_rejection(attempted)
            raise LiveContextError()

    def _report_lock_rejection(self, attempted):
        """Report only the attempted roots, not the SQL predicate's unknown cause."""
        try:
            if self._lock_rejections_reported > _LOCK_LOG_LIMIT:
                return
            identity = {"session_id": _diagnostic_uuid(self.session_id),
                        "audio_session_id": _diagnostic_uuid(self.audio_session_id)}
            if self._lock_rejections_reported == _LOCK_LOG_LIMIT:
                self._lock_rejections_reported += 1
                record = {**identity, "stage": "truncated", "record_limit": _LOCK_LOG_LIMIT,
                          "absence_after_limit_is_inconclusive": True}
            else:
                self._lock_rejections_reported += 1
                limit = (min(self.max_sources + 1, _LOCK_LOG_ROOT_LIMIT)
                         if type(self.max_sources) is int and self.max_sources >= 0 else 0)
                roots = [root for value in attempted[:limit]
                         if (root := _diagnostic_uuid(value)) is not None]
                omitted = len(attempted) - len(roots)
                record = {**identity, "stage": "source_lock_rejected",
                          "code": "live_context_revoked", "reason": "lock_predicate_false",
                          "attempted_root_ids": roots, "attempted_root_count": len(attempted),
                          "retained_root_count": len(roots), "root_limit": limit,
                          "omitted_root_count": omitted, "roots_truncated": omitted > 0}
            _LOCK_LOGGER.warning("source_context_lock_rejection_diagnostic %s",
                                 json.dumps(record, sort_keys=True, separators=(",", ":")))
        except Exception:
            # Diagnostics must never bypass a rejection or replace its error.
            pass

    async def prepare(self, request_id: str, *, refresh_stale: bool = False, extra_sources=()):
        """Refresh lineage only when the caller proves this request is unwritten.

        Default callers retain their original immutable response identity. A
        prewrite retry creates a new dependency payload, never edits the old
        payload or drops accumulated sources. Failed refreshes keep the cache.
        """
        fields = {"request_id": request_id, "kind": "prepare"}
        trace = self._latency_trace
        with latency_phase(trace, "response_source_latency", phase="prepare", fields=fields):
            additions = set(extra_sources)
            if any(not isinstance(value, UUID) for value in additions):
                raise LiveContextError("live_context_source_invalid")
            combined = self._sources | additions
            if len(combined) > self.max_sources:
                raise LiveContextError("live_context_source_limit")
            if not combined:
                fields["reason"] = "no_sources"
                return None
            cached = self._responses.get(request_id)
            if (cached is not None and additions.issubset(cached.sources)
                    and (not refresh_stale or cached.version == self._version)):
                fields.update(reason="cache_reused", source_snapshot_id=cached.payload_id)
                return cached
            if self.pool is None:
                raise LiveContextError("live_context_storage_unavailable")
            if cached is None and len(self._responses) >= self.max_responses:
                raise LiveContextError("live_context_response_limit")
            sources, version = tuple(sorted(combined)), self._version
            prepared_at = datetime.now(timezone.utc)
            async with self._timed_transaction(fields) as connection:
                with latency_phase(trace, "response_source_latency", phase="source_lock", fields=fields):
                    await self._lock(connection, sources)
                with latency_phase(trace, "response_source_latency", phase="payload_insert", fields=fields):
                    payload_id = await connection.fetchval(
                        """insert into jarvis_context_payloads(owner_id,availability,content)
                           values($1,'available',$2::jsonb) returning id""",
                        self.owner_id, json.dumps({"kind":"live_response_dependencies",
                            "session_id":str(self.session_id), "audio_session_id":self.audio_session_id,
                            "request_id":request_id, "prepared_at":prepared_at.isoformat()}))
                    fields["source_snapshot_id"] = payload_id
                with latency_phase(trace, "response_source_latency", phase="dependency_insert", fields=fields):
                    await connection.executemany("""insert into jarvis_context_payload_dependencies
                        (owner_id,source_payload_id,dependent_payload_id) values($1,$2,$3)""",
                        [(self.owner_id, source, payload_id) for source in sources])
            # Commit lineage before a provider can possibly accept the response.
            snapshot = ResponseSourceSnapshot(payload_id, sources, version, prepared_at)
            self._responses[request_id] = snapshot
            return snapshot

    @asynccontextmanager
    async def context_guard(self, *, connection=None, extra_sources=()):
        connection = connection if connection is not None else self._current_connection()
        additions = set(extra_sources)
        if any(not isinstance(value, UUID) for value in additions):
            raise LiveContextError("live_context_source_invalid")
        combined = self._sources | additions
        if len(combined) > self.max_sources:
            raise LiveContextError("live_context_source_limit")
        # Validate possible new inputs without enrolling them into remote
        # history until the guarded writer actually prepares their payload.
        sources, version = tuple(sorted(combined)), self._version
        if not sources:
            yield
            return
        if connection is not None:
            # The canonical scope/source guard already owns its transaction.
            # Acquiring a second owner lock after locking copied roots can
            # deadlock against withdrawal (owner lock -> roots).
            if not connection.is_in_transaction():
                raise LiveContextError("live_context_transaction_required")
            await self._lock(connection, sources)
            if version != self._version:
                raise LiveContextError("live_context_changed")
            yield
            return
        if self.pool is None:
            raise LiveContextError("live_context_storage_unavailable")
        async with guard_context(self.pool.acquire(), "pool_acquire") as owned, \
                guard_context(owned.transaction(), "transaction_enter"):
            await self._lock(owned, sources)
            if version != self._version:
                raise LiveContextError("live_context_changed")
            yield

    @asynccontextmanager
    async def guard(self, snapshot: ResponseSourceSnapshot):
        fields = {"source_snapshot_id": snapshot.payload_id, "kind": "guard"}
        trace = self._latency_trace
        with latency_phase(trace, "response_source_latency", phase="guard", fields=fields):
            if snapshot.version != self._version:
                raise LiveContextError("live_context_changed")
            borrowed = self._current_connection()
            if borrowed is not None:
                if not borrowed.is_in_transaction():
                    raise LiveContextError("live_context_transaction_required")
                latency_event(trace, "response_source_latency", phase="transaction_reuse", status="reused", **fields)
                with latency_phase(trace, "response_source_latency", phase="source_lock", fields=fields):
                    await self._lock(borrowed, (*snapshot.sources, snapshot.payload_id))
                if snapshot.version != self._version:
                    raise LiveContextError("live_context_changed")
                yield
                return
            async with self._timed_transaction(fields) as connection:
                # Oldest originals first, then their newer materializations. This
                # ordering also applies when history contains both ancestor and copy.
                with latency_phase(trace, "response_source_latency", phase="source_lock", fields=fields):
                    await self._lock(connection, (*snapshot.sources, snapshot.payload_id))
                if snapshot.version != self._version:
                    raise LiveContextError("live_context_changed")
                yield
