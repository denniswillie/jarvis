"""Coalesced background rereads of canonical state; bus delivery is only a hint."""
from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
import math
from typing import Any

from app.knowledge.models import SourceError
from app.live_state.models import SessionScope
from app.runtime.events import DomainEvent, DomainEventBus, DomainEventName


REFRESH_SOURCE_ERROR_CODES = frozenset({
    "live_state_runtime_unavailable", "live_state_scope_unavailable",
    "live_state_scope_changed", "live_state_session_unavailable",
    "live_state_context_revoked", "live_state_context_expired",
    "live_state_source_scope", "live_state_source_version", "live_state_source_unsupported",
    "live_state_source_changed", "live_state_event_changed", "live_state_event_scope",
    "live_state_memory_scope", "live_state_projection_unavailable",
    "live_state_canonical_owners_changed", "live_state_canonical_owners_unavailable",
    "live_state_reconstruction_required", "knowledge_area_unavailable",
    "knowledge_source_not_found", "knowledge_source_unavailable", "knowledge_source_changed",
    "knowledge_session_unavailable", "knowledge_payload_unavailable",
    "knowledge_source_deleted", "knowledge_source_forgotten", "knowledge_source_revision_conflict",
    "live_task_scope_changed", "live_task_context_unavailable", "live_task_request_changed",
    "live_task_time_unavailable", "live_task_context_oversized", "live_task_source_version",
    "live_task_snapshot_unavailable", "live_task_snapshot_changed", "live_task_context_expired",
    "live_episode_owner_unavailable", "live_episode_reference_invalid",
    "live_episode_source_changed", "live_episode_projection_unavailable",
    "live_episode_context_expired", "live_episode_cue_inapplicable",
})


def _failure_source_code(error):
    """Diagnostic-only metadata; never retain arbitrary exception attributes/text."""
    if not isinstance(error, SourceError):
        return None
    try:
        code = error.code
        if type(code) is str and code in REFRESH_SOURCE_ERROR_CODES:
            return code
    except Exception:
        pass
    return "unrecognized_source_error"


class CanonicalRefreshWorker:
    def __init__(self, *, scope: SessionScope, bus: DomainEventBus,
                 event_names: tuple[DomainEventName, ...],
                 read: Callable[[], Awaitable[Any]],
                 apply: Callable[[Any], Awaitable[None]],
                 invalidate: Callable[[str], Awaitable[None]],
                 interval_seconds: float = 1.0, timeout_seconds: float = 2.0,
                 apply_timeout_seconds: float | None = None):
        if (not event_names or len(set(event_names)) != len(event_names)
                or not all(math.isfinite(value) and value > 0
                           for value in (interval_seconds, timeout_seconds))):
            raise ValueError("Refresh requires unique event names and finite positive deadlines")
        if apply_timeout_seconds is not None and (
                not math.isfinite(apply_timeout_seconds) or apply_timeout_seconds <= 0):
            raise ValueError("A separate apply deadline must be finite and positive")
        self.scope, self.bus, self.event_names = scope, bus, event_names
        self.read, self.apply, self.invalidate = read, apply, invalidate
        self.interval_seconds, self.timeout_seconds = interval_seconds, timeout_seconds
        # Opt-in for projections that perform multiple independently guarded
        # operations. Other producers retain their original combined deadline.
        self.apply_timeout_seconds = apply_timeout_seconds
        self._wake = asyncio.Event()
        self._task: asyncio.Task | None = None
        self._closed = False
        self._loop = None
        self.refresh_count = 0
        self.failure_count = 0
        self.last_error: str | None = None
        self.last_error_stage: str | None = None
        self.last_error_code: str | None = None

    async def hint(self, event: DomainEvent):
        # Never await a repository/model while DomainEventBus publishes. It
        # invokes subscribers serially on the canonical producer's own path.
        if self._closed or event.name not in self.event_names:
            return
        payload = event.payload
        if payload.owner_id != self.scope.owner_id:
            return
        if getattr(payload, "session_id", None) not in (None, self.scope.session_id):
            return
        if asyncio.get_running_loop() is not self._loop:
            raise RuntimeError("Refresh hints must run on the state owner's loop")
        self._wake.set()

    def start(self):
        if self._task is not None or self._closed:
            raise RuntimeError("A refresh worker can be started only once")
        self._loop = asyncio.get_running_loop()
        for name in self.event_names:
            self.bus.subscribe(name, self.hint)
        self._wake.set()
        self._task = asyncio.create_task(self._run(), name="live-state-canonical-refresh")

    async def _run(self):
        while not self._closed:
            try:
                async with asyncio.timeout(self.interval_seconds):
                    await self._wake.wait()
            except TimeoutError:
                pass
            self._wake.clear()
            stage = "read"
            try:
                async with asyncio.timeout(self.timeout_seconds) as deadline:
                    value = await self.read()
                    if self._closed:
                        return
                    stage = "apply"
                    if self.apply_timeout_seconds is not None:
                        deadline.reschedule(asyncio.get_running_loop().time() + self.apply_timeout_seconds)
                    await self.apply(value)
                self.refresh_count += 1
                self.last_error = None
                self.last_error_stage = None
                self.last_error_code = None
            except asyncio.CancelledError:
                raise
            except Exception as error:
                self.failure_count += 1
                self.last_error = type(error).__name__
                self.last_error_stage = stage
                self.last_error_code = _failure_source_code(error)
                # Failure is not evidence of absence. Withdraw this producer's
                # view; recovery rereads the authority instead of replaying hints.
                await self.invalidate("canonical_refresh_unavailable")
                await asyncio.sleep(self.interval_seconds)

    async def close(self):
        if self._closed:
            return
        self._closed = True
        for name in self.event_names:
            self.bus.unsubscribe(name, self.hint)
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
